import socket
import pickle
import struct
import numpy as np
from threading import Thread, Lock
import cv2
import time
import logging
from flask import Flask, Response, jsonify, render_template, request, redirect, url_for
from ast import literal_eval
from collections import defaultdict
from ultralytics import YOLO  # YOLOv8
# Violence detection model
import tensorflow as tf
from collections import deque

app = Flask(__name__, template_folder='../templates', static_folder='../static')

CONFIDENCE_THRESHOLD = 0.85  # High threshold for precision
CONSECUTIVE_DETECTIONS_REQUIRED = 5  # Multiple consecutive detections before alerting
ALERT_COOLDOWN = 10  # Seconds between alerts
violence_frames_buffers = defaultdict(list)
prediction_histories = defaultdict(lambda: deque(maxlen=12))
motion_histories = defaultdict(lambda: deque(maxlen=12))
face_size_histories = defaultdict(lambda: deque(maxlen=8))
alert_counters = defaultdict(int)
alert_statuses = defaultdict(bool)
last_alert_times = defaultdict(float)


try:
    violence_model = tf.keras.models.load_model('models/cnn_lstm_model.keras')
    print("Violence detection model loaded successfully")
except Exception as e:
    print(f"Failed to load violence detection model: {e}")
    violence_model = None

# Dictionary to hold client streams with thread-safe access
client_streams = {}
streams_lock = Lock()
detection_settings = defaultdict(lambda: {'active': False, 'target_gender': None, 'last_alert': None, 'mode': None})
face_trackers = defaultdict(dict)
face_id_counter = defaultdict(lambda: 0)

# Initialize Gender Detection models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)
genderList = ['Male', 'Female']

MAX_HITS = 15
GENDER_LOCK_TIMEOUT = 3  # seconds
# Initialize Motion Detection models
yolo_model = YOLO('yolov8n-pose.pt')  # YOLOv8 small pose model

# Using OpenCV's DNN-based hand detector instead of MediaPipe
handProto = "hand_detector.prototxt"  # You'll need to provide these files
handModel = "hand_detector.caffemodel"  # You'll need to provide these files
try:
    handNet = cv2.dnn.readNet(handModel, handProto)
    use_hand_dnn = True
except:
    print("Hand detection model files not found. Using pose detection for hand tracking.")
    use_hand_dnn = False

# Motion detection parameters
min_area = 600
threshold_value = 25

# Unattended object detection parameters
MIN_CONTOUR_AREA = 400
PERSISTENCE_FRAMES = 100
LEARNING_RATE = 0.010
MATCH_DISTANCE = 100  # Maximum distance for object matching

# Store previous keypoints for fight detection (per client)
previous_keypoints = {}

# Store tracked objects for unattended object detection (per client)
unattended_objects = defaultdict(dict)
object_id_counters = defaultdict(int)
background_models = {}

def preprocess_frame(frame, target_size=224):
    """Preprocess frame for violence detection model input"""
    resized = cv2.resize(frame, (target_size, target_size))
    normalized = resized.astype(np.float32) / 255.0
    return normalized

def distance_correction(prediction, face_size, addr):
    """Apply distance-based correction to prediction"""
    if len(face_size_histories[addr]) < 3:
        return prediction
    
    # Calculate average face size
    avg_face_size = sum(face_size_histories[addr]) / len(face_size_histories[addr])
    
    if avg_face_size > 0:
        # Apply distance correction
        # Large face = close to camera = higher chance of false positive
        # So we reduce the prediction confidence when face is large/close
        distance_factor = min(1.0, 10000 / max(avg_face_size, 1))
        
        # Apply stronger correction when face is very close
        if avg_face_size > 15000:  # Very close face
            prediction *= (distance_factor * 0.7)
        elif avg_face_size > 8000:  # Moderately close face
            prediction *= (distance_factor * 0.85)
    
    return prediction

def complex_temporal_filtering(new_prediction, motion_level, addr):
    """Apply complex temporal filtering with motion context"""
    prediction_histories[addr].append(new_prediction)
    motion_histories[addr].append(motion_level)
    
    if len(prediction_histories[addr]) < 5:
        return new_prediction
    
    # Calculate motion pattern (increasing, stable, decreasing)
    recent_motion = list(motion_histories[addr])[-5:]
    motion_increasing = sum(recent_motion[-2:]) > sum(recent_motion[:2])
    
    # Apply different filtering strategies based on motion pattern
    if motion_increasing:
        # Rapid motion increase - be more sensitive to potential fights
        weights = np.linspace(0.5, 1.0, len(prediction_histories[addr]))
        weights = weights / np.sum(weights)  # Normalize weights
        smoothed = sum(w * p for w, p in zip(weights, prediction_histories[addr]))
    else:
        # Stable or decreasing motion - require more consistent evidence
        smoothed = sum(prediction_histories[addr]) / len(prediction_histories[addr])
        
        # Further reduce if motion is very low
        if motion_level < 0.005:
            smoothed = smoothed * 0.5  # Strong reduction for static scenes
    
    return smoothed

def check_alert_status(prediction, current_time, addr):
    """Manage alert state to prevent false alarms"""
    # Check if prediction exceeds threshold
    if prediction > CONFIDENCE_THRESHOLD:
        alert_counters[addr] += 1
        # Require consecutive detections before triggering alert
        if alert_counters[addr] >= CONSECUTIVE_DETECTIONS_REQUIRED:
            # Only trigger new alert if cooldown period has passed
            if not alert_statuses[addr] and (current_time - last_alert_times.get(addr, 0)) > ALERT_COOLDOWN:
                alert_statuses[addr] = True
                last_alert_times[addr] = current_time
                print(f"\n⚠️ ALERT: Fight detected with high confidence for client {addr}!")
    else:
        # Reset counter if prediction drops below threshold
        alert_counters[addr] = max(0, alert_counters[addr] - 1)
        
        # Turn off alert if counter drops significantly
        if alert_counters[addr] < 2:
            alert_statuses[addr] = False
    
    return alert_statuses[addr]

def process_violence_detection(frame, addr):
    """Apply violence detection model to the frame"""
    # If model isn't loaded, return frame with error message
    if violence_model is None:
        cv2.putText(frame, "Violence model not loaded!", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame, False, 0.0
    
    current_time = time.time()
    
    # Calculate motion level
    with streams_lock:
        if addr in client_streams and 'prev_frame' in client_streams[addr]:
            prev_frame = client_streams[addr]['prev_frame']
        else:
            prev_frame = None
    
    if prev_frame is not None:
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of pixels that changed
        motion_level = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)
    else:
        motion_level = 0.0
    
    # Store current frame for next comparison
    with streams_lock:
        if addr in client_streams:
            client_streams[addr]['prev_frame'] = frame.copy()
    
    # Detect faces for distance estimation
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    largest_face_size = 0
    for (x, y, w, h) in faces:
        # Draw rectangle around faces for visualization
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_size = w * h
        largest_face_size = max(largest_face_size, face_size)
    
    # Update face size history
    face_size_histories[addr].append(largest_face_size)
    
    # Process frame for model input
    processed_frame = preprocess_frame(frame)
    
    # Update frames buffer
    violence_frames_buffers[addr].append(processed_frame)
    if len(violence_frames_buffers[addr]) > 16:  # Keep only last 16 frames
        violence_frames_buffers[addr].pop(0)
    
    current_prediction = 0.0
    
    # Make prediction if we have enough frames
    if len(violence_frames_buffers[addr]) == 16:
        frames_input = np.expand_dims(np.array(violence_frames_buffers[addr]), axis=0)
        raw_prediction = violence_model.predict(frames_input, verbose=0)[0][0]
        
        # Apply temporal filtering with motion context
        filtered_prediction = complex_temporal_filtering(raw_prediction, motion_level, addr)
        
        # Apply distance-based correction
        current_prediction = distance_correction(filtered_prediction, largest_face_size, addr)
        
        # Check alert status
        alert_status = check_alert_status(current_prediction, current_time, addr)
    else:
        alert_status = False
    
    # Display prediction and information
    result_frame = display_prediction(frame, current_prediction, motion_level, largest_face_size, alert_status)
    
    return result_frame, alert_status, current_prediction

def display_prediction(frame, prediction, motion_level, face_size, alert_active):
    """Display prediction and relevant information on the frame"""
    display_frame = frame.copy()
    
    # Choose color and text based on prediction and alert status
    if alert_active:
        color = (0, 0, 255)  # Red for active alert
        text = f"⚠️ UNAUTHORIZED ENTRY DETECTED ⚠️"
        # Add alert animation (flashing border)
        border_thickness = int(time.time() * 4) % 10 + 2
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), 
                    color, border_thickness)
    elif prediction > CONFIDENCE_THRESHOLD:
        color = (0, 140, 255)  # Orange for high confidence but not yet alerting
        text = f"Potential Fight ({prediction:.2f})"
    else:
        color = (0, 255, 0)  # Green for no fight
        text = f"No Fight ({prediction:.2f})"
    
    # Add semi-transparent overlay
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
    
    # Add text with all relevant information
    cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(display_frame, f"Motion: {motion_level:.3f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(display_frame, f"Face Size: {face_size}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Add threshold visualization bar
    bar_width = 200
    bar_height = 20
    bar_x = display_frame.shape[1] - bar_width - 10
    bar_y = 30
    
    # Draw background bar
    cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
    
    # Draw prediction level
    pred_width = int(prediction * bar_width)
    cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + pred_width, bar_y + bar_height), color, -1)
    
    # Draw threshold line
    threshold_x = bar_x + int(CONFIDENCE_THRESHOLD * bar_width)
    cv2.line(display_frame, (threshold_x, bar_y - 5), (threshold_x, bar_y + bar_height + 5), (255, 255, 255), 2)
    
    return display_frame

def safe_client_id_to_tuple(client_id):
    """Safely convert client_id string to tuple format"""
    try:
        if isinstance(client_id, str) and client_id.startswith('('):
            return literal_eval(client_id)  # Safer than eval
        elif ':' in client_id:  # Handle "ip:port" format
            ip, port = client_id.split(':')
            return ip, int(port)
    except:
        pass
    return None


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def match_face_to_tracker(addr, new_box):
    trackers = face_trackers[addr]
    for face_id, info in trackers.items():
        existing_box = info['bbox']
        if iou(existing_box, new_box) > 0.4:
            return face_id
    return None


def apply_gender_detection(frame, addr):
    upscale_frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
    cpy_input_image = upscale_frame.copy()
    frameWidth, frameHeight = upscale_frame.shape[1], upscale_frame.shape[0]

    blob = cv2.dnn.blobFromImage(upscale_frame, scalefactor=1, size=(300, 300),
                                 mean=(104, 117, 123), swapRB=False, crop=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()

    detected_genders = set()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            box = (max(0, x1), max(0, y1), min(frameWidth - 1, x2), min(frameHeight - 1, y2))
            x1, y1, x2, y2 = box

            face_id = match_face_to_tracker(addr, box)
            if face_id is None:
                face_id = face_id_counter[addr]
                face_id_counter[addr] += 1
                face_trackers[addr][face_id] = {'bbox': box, 'hits': 0, 'genders': [], 'gender': None}

            tracker = face_trackers[addr][face_id]
            tracker['bbox'] = box
            tracker['last_seen'] = time.time()

            if tracker['gender'] is None and tracker['hits'] < MAX_HITS:
                face_img = upscale_frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                face_blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227, 227),
                                                  mean=[78.4263377603, 87.7689143744, 114.895847746], crop=False)
                genderNet.setInput(face_blob)
                gender_preds = genderNet.forward()
                gender = genderList[gender_preds[0].argmax()]
                tracker['genders'].append(gender)
                tracker['hits'] += 1

                if tracker['hits'] == MAX_HITS:
                    tracker['gender'] = max(set(tracker['genders']), key=tracker['genders'].count)

            label = tracker['gender'] or f"Detecting {tracker['hits']}/{MAX_HITS}"
            if tracker['gender']:
                detected_genders.add(tracker['gender'])

            cv2.rectangle(cpy_input_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(cpy_input_image, f"ID {face_id}: {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    now = time.time()
    face_trackers[addr] = {fid: data for fid, data in face_trackers[addr].items()
                           if now - data.get('last_seen', 0) <= GENDER_LOCK_TIMEOUT}

    # Convert set to list for JSON serialization
    return cpy_input_image, list(detected_genders)


def detect_hands_opencv(frame):
    """Detect hands using OpenCV DNN instead of MediaPipe"""
    if not use_hand_dnn:
        return frame, False
    
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [127.5, 127.5, 127.5], swapRB=True, crop=False)
    handNet.setInput(blob)
    detections = handNet.forward()
    
    hand_detected = False
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            hand_detected = True
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Hand: {confidence:.2f}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, hand_detected


def process_motion_detection(frame, addr):
    """Apply ML models to detect motion, pose, and fights"""
    # Convert to grayscale for motion detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    
    # Get or initialize average frame for this client
    with streams_lock:
        if addr in client_streams and 'avg_frame' in client_streams[addr]:
            avg_frame = client_streams[addr]['avg_frame']
        else:
            avg_frame = gray_frame.copy().astype("float")
            if addr in client_streams:
                client_streams[addr]['avg_frame'] = avg_frame
    
    # Update background for motion detection
    cv2.accumulateWeighted(gray_frame, avg_frame, 0.05)
    frame_delta = cv2.absdiff(cv2.convertScaleAbs(avg_frame), gray_frame)
    
    thresh = cv2.threshold(frame_delta, threshold_value, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check for motion
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Pose detection
    results = yolo_model.predict(frame, imgsz=640, conf=0.5, verbose=False)
    
    # Skeleton connections
    skeleton = [
        (1, 2), (1, 3), (2, 4),
        (5, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 11), (6, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    current_keypoints = None
    
    # Process each detected pose
    for idx, pose_result in enumerate(results):
        if hasattr(pose_result, 'keypoints') and pose_result.keypoints is not None:
            for idx, pose in enumerate(pose_result.keypoints.xy):
                color = colors[idx % len(colors)]
                keypoints = pose.cpu().numpy()
                
                current_keypoints = keypoints
                
                for connection in skeleton:
                    part_a, part_b = connection
                    if keypoints.shape[0] > max(part_a, part_b):
                        xa, ya = keypoints[part_a]
                        xb, yb = keypoints[part_b]
                        if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2)
    
    # Hand detection using OpenCV DNN
    frame, hand_detected = detect_hands_opencv(frame)
    
    # Fight detection logic
    fight_detected = False
    
    if addr in previous_keypoints and current_keypoints is not None:
        prev_keypoints = previous_keypoints[addr]
        if prev_keypoints is not None and prev_keypoints.shape[0] >= 17 and current_keypoints.shape[0] >= 17:
            # Check how many keypoints are valid (non-zero)
            valid_keypoints = 0
            speeds = []
            
            important_parts = [5, 6, 7, 8, 9, 10]  # Shoulders, elbows, wrists
            
            for part in important_parts:
                prev_pt = prev_keypoints[part]
                curr_pt = current_keypoints[part]
                
                if np.all(prev_pt > 0) and np.all(curr_pt > 0):  # Both points exist
                    speed = np.linalg.norm(curr_pt - prev_pt)
                    speeds.append(speed)
                    valid_keypoints += 1
            
            # Only detect fight if at least 4 points are valid (out of 6)
            if valid_keypoints >= 4:
                avg_speed = np.mean(speeds)
                if avg_speed > 20:
                    fight_detected = True
    
    # Update previous keypoints
    previous_keypoints[addr] = current_keypoints
    
    # Display motion and fight detection status
    if fight_detected:
        status_text = ""
    else:
        status_text = " Unauthorized Access Detected "

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX  # SIMPLEX is very clean
    font_scale = 0.6  # Not too big
    thickness = 1     # Thin but visible
    color = (0, 0, 0)  # Bright red

    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(status_text, font, font_scale, thickness)

    # Center the text
    frame_height, frame_width = frame.shape[:2]
    x = int((frame_width - text_width) / 2)
    y = int(frame_height * 0.08)  # a little lower from the top

    # Optional: add soft black shadow for elegance
    #cv2.putText(frame, status_text, (x+2, y+2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Draw the hot red text
    cv2.putText(frame, status_text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    # Draw actual text (red)

    
    #cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    #motion_status_text = "Motion: DETECTED" if motion_detected else "Motion: NOT DETECTED"
    #motion_status_color = (0, 0, 255) if motion_detected else (0, 255, 0)
    #cv2.putText(frame, motion_status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, motion_status_color, 2)
    
    # Update client stream data with detection results
    with streams_lock:
        if addr in client_streams:
            client_streams[addr]['avg_frame'] = avg_frame
            client_streams[addr]['motion_detected'] = motion_detected
            client_streams[addr]['fight_detected'] = fight_detected
    
    return frame, motion_detected, fight_detected


def process_unattended_object_detection(frame, addr):
    """Detect unattended objects in the frame"""
    # Convert to grayscale and preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Initialize or get background model for this client
    if addr not in background_models:
        background_models[addr] = {
            'model': cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False),
            'static_bg': None
        }
    
    # Apply background subtraction
    if background_models[addr]['static_bg'] is not None:
        diff = cv2.absdiff(gray, background_models[addr]['static_bg'])
        _, fg_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    else:
        fg_mask = background_models[addr]['model'].apply(gray, learningRate=LEARNING_RATE)
    
    # Noise reduction
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_objects = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2
        current_objects.append((cx, cy, x, y, w, h))
    
    # Get tracked objects for this client
    tracked_objects = unattended_objects[addr]
    updated_objects = {}
    used_ids = set()
    
    # Try to match existing objects
    for obj in current_objects:
        cx, cy, x, y, w, h = obj
        best_match = None
        min_distance = float('inf')
        
        for obj_id, data in tracked_objects.items():
            if obj_id in used_ids:
                continue
            
            last_pos = data['positions'][-1]
            distance = np.sqrt((cx - last_pos[0])**2 + (cy - last_pos[1])**2)
            
            if distance < MATCH_DISTANCE and distance < min_distance:
                min_distance = distance
                best_match = obj_id
        
        if best_match is not None:
            # Update existing object
            tracked_objects[best_match]['positions'].append((cx, cy))
            tracked_objects[best_match]['bbox'] = (x, y, w, h)
            if len(tracked_objects[best_match]['positions']) > PERSISTENCE_FRAMES:
                tracked_objects[best_match]['positions'].pop(0)
            updated_objects[best_match] = tracked_objects[best_match]
            used_ids.add(best_match)
        else:
            # Create new object
            updated_objects[object_id_counters[addr]] = {
                'positions': [(cx, cy)],
                'bbox': (x, y, w, h),
                'age': 0
            }
            object_id_counters[addr] += 1
    
    # Handle unmatched existing objects
    for obj_id, data in tracked_objects.items():
        if obj_id not in used_ids:
            data['age'] += 1
            if data['age'] < 10:  # Keep objects for 10 frames after disappearance
                updated_objects[obj_id] = data
    
    unattended_objects[addr] = updated_objects
    
    # Draw persistent objects and count them
    unattended_count = 0
    for obj_id, data in unattended_objects[addr].items():
        if len(data['positions']) >= PERSISTENCE_FRAMES:
            unattended_count += 1
            x, y, w, h = data['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f'Unattended Object {obj_id}', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display unattended object count
    cv2.putText(frame, f"Unattended Objects: {unattended_count}", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if unattended_count > 0 else (0, 255, 0), 2)
    
    # Update client stream data
    with streams_lock:
        if addr in client_streams:
            client_streams[addr]['unattended_objects'] = unattended_count
            client_streams[addr]['fg_mask'] = fg_mask
    
    return frame, unattended_count


def set_background_for_client(addr, frame):
    """Set static background for unattended object detection"""
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed_bg = clahe.apply(gray)
        processed_bg = cv2.GaussianBlur(processed_bg, (9, 9), 2)
        
        if addr not in background_models:
            background_models[addr] = {
                'model': cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False),
                'static_bg': None
            }
        
        background_models[addr]['static_bg'] = processed_bg
        return True
    return False


def handle_client(conn, addr):
    print(f"Connected: {addr}")
    data = b""
    payload_size = struct.calcsize("Q")

    try:
        while True:
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet:
                    raise ConnectionError("Client disconnected")
                data += packet

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            while len(data) < msg_size:
                packet = conn.recv(4096)
                if not packet:
                    raise ConnectionError("Client disconnected")
                data += packet

            frame_data = data[:msg_size]
            data = data[msg_size:]
            frame = pickle.loads(frame_data)
            
            # Store original frame for detection
            original_frame = frame.copy()
            
            # Process frame based on detection mode set for this client
            with streams_lock:
                settings = detection_settings[addr] if addr in detection_settings else {'active': False, 'mode': None}
                
            if settings['active']:
                if settings['mode'] == 'gender':
                    processed_frame, detected_gender = apply_gender_detection(original_frame, addr)
                    
                    # Update gender detection alert status
                    if detected_gender and settings.get('target_gender') in detected_gender:
                        if settings.get('last_alert') != settings.get('target_gender'):
                            print(f"Alert: {settings.get('target_gender')} detected for {addr}")
                            settings['last_alert'] = settings.get('target_gender')
                    else:
                        settings['last_alert'] = None
                    
                    motion_detected = False
                    fight_detected = False
                    unattended_count = 0
                    violence_prediction = 0.0
                elif settings['mode'] == 'motion':
                    processed_frame, motion_detected, fight_detected = process_motion_detection(original_frame, addr)
                    detected_gender = []
                    unattended_count = 0
                    violence_prediction = 0.0
                elif settings['mode'] == 'unattended':
                    processed_frame, unattended_count = process_unattended_object_detection(original_frame, addr)
                    motion_detected = False
                    fight_detected = False
                    detected_gender = []
                    violence_prediction = 0.0
                elif settings['mode'] == 'violence':
                    # New mode for violence detection
                    processed_frame, fight_detected, violence_prediction = process_violence_detection(original_frame, addr)
                    motion_detected = False
                    detected_gender = []
                    unattended_count = 0
                else:
                    # Default to original frame if unknown mode
                    processed_frame = original_frame
                    motion_detected = False
                    fight_detected = False
                    detected_gender = []
                    unattended_count = 0
                    violence_prediction = 0.0
            else:
                # If detection is not active, just use the original frame
                processed_frame = original_frame
                motion_detected = False
                fight_detected = False
                detected_gender = []
                unattended_count = 0
                violence_prediction = 0.0
            
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                raise ValueError("Could not encode frame")

            with streams_lock:
                client_streams[addr] = {
                    'frame': buffer.tobytes(),
                    'timestamp': time.time(),
                    'motion_detected': motion_detected,
                    'fight_detected': fight_detected,
                    'detected_gender': detected_gender,
                    'unattended_objects': unattended_count,
                    'violence_prediction': violence_prediction,
                    'prev_frame': original_frame.copy()  # Store for motion calculation
                }
    except Exception as e:
        print(f"Client {addr} error: {str(e)}")
    finally:
        with streams_lock:
            client_streams.pop(addr, None)
            if addr in previous_keypoints:
                del previous_keypoints[addr]
            if addr in unattended_objects:
                del unattended_objects[addr]
            if addr in background_models:
                del background_models[addr]
            # Clean up violence detection data
            if addr in violence_frames_buffers:
                del violence_frames_buffers[addr]
            if addr in prediction_histories:
                del prediction_histories[addr]
            if addr in motion_histories:
                del motion_histories[addr]
            if addr in face_size_histories:
                del face_size_histories[addr]
        conn.close()
        print(f"Connection closed: {addr}")


def start_socket_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', 9999))
    server_socket.listen(5)
    print("Socket server listening on port 9999")

    while True:
        conn, addr = server_socket.accept()
        Thread(target=handle_client, args=(conn, addr), daemon=True).start()


@app.route('/')
def index():
    return render_template('login.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Here you would typically verify username and password
        # For now, just redirect to the dashboard
        return redirect(url_for('dashboard'))
    else:
        # If it's a GET request, show the login page
        return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    return render_template('index.html')


@app.route('/active_streams')
def active_streams():
    current_time = time.time()
    
    with streams_lock:
        # Remove stale clients
        stale_clients = [addr for addr, data in client_streams.items() 
                        if current_time - data['timestamp'] > 5]
        for client in stale_clients:
            del client_streams[client]
            if client in previous_keypoints:
                del previous_keypoints[client]
            if client in unattended_objects:
                del unattended_objects[client]
                
        # Get client ids as strings and convert to list for JSON serialization
        client_ids = [f"{ip}:{port}" for ip, port in client_streams.keys()]
        
        # Get detection status for each client
        client_statuses = {}
        for addr in client_streams.keys():
            client_id = f"{addr[0]}:{addr[1]}"
            
            # Ensure detected_gender is JSON serializable
            detected_gender = client_streams[addr].get('detected_gender', [])
            if isinstance(detected_gender, set):
                detected_gender = list(detected_gender)
            
            client_statuses[client_id] = {
                'motion_detected': client_streams[addr].get('motion_detected', False),
                'fight_detected': client_streams[addr].get('fight_detected', False),
                'detected_gender': detected_gender[0] if detected_gender else None,
                'unattended_objects': client_streams[addr].get('unattended_objects', 0),
                'violence_prediction': client_streams[addr].get('violence_prediction', 0.0),
                'mode': detection_settings[addr].get('mode', None),
                'active': detection_settings[addr].get('active', False),
                'target_gender': detection_settings[addr].get('target_gender', None),
                'has_background': background_models.get(addr, {}).get('static_bg') is not None
            }
            
    return jsonify({
        'clients': client_ids,
        'statuses': client_statuses
    })


@app.route('/toggle_detection/<client_id>', methods=['POST'])
def toggle_detection(client_id):
    addr = safe_client_id_to_tuple(client_id)
    if not addr:
        return jsonify({'error': 'Invalid client ID'}), 400

    data = request.get_json()
    mode = data.get('mode')
    gender = data.get('gender', None)
    active = data.get('active', False)

    if not mode:
        return jsonify({'error': 'Detection mode not specified'}), 400

    detection_settings[addr]['active'] = active
    detection_settings[addr]['mode'] = mode

    if mode == 'gender':
        if gender in genderList or gender == 'None':
            detection_settings[addr]['target_gender'] = None if gender == 'None' else gender
            detection_settings[addr]['last_alert'] = None
            return jsonify({
                'status': 'Gender detection ' + ('activated' if active else 'deactivated'), 
                'target': gender if gender != 'None' else 'None'
            })
        else:
            return jsonify({'error': 'Invalid gender specified'}), 400
    elif mode == 'motion':
        return jsonify({'status': 'Motion detection ' + ('activated' if active else 'deactivated')})
    elif mode == 'unattended':
        return jsonify({'status': 'Unattended object detection ' + ('activated' if active else 'deactivated')})
    elif mode == 'violence':
        # Reset violence detection buffers when toggling
        if addr in violence_frames_buffers:
            violence_frames_buffers[addr] = []
        alert_counters[addr] = 0
        alert_statuses[addr] = False
        return jsonify({'status': 'Violence detection ' + ('activated' if active else 'deactivated')})
    else:
        return jsonify({'error': 'Invalid detection mode'}), 400


@app.route('/set_background/<client_id>', methods=['POST'])
def set_background(client_id):
    addr = safe_client_id_to_tuple(client_id)
    if not addr:
        return jsonify({'error': 'Invalid client ID'}), 400
    
    with streams_lock:
        if addr in client_streams:
            # Get the last frame from the client
            frame_data = client_streams[addr].get('frame')
            if frame_data:
                try:
                    # Convert JPEG bytes back to numpy array
                    nparr = np.frombuffer(frame_data, np.uint8)
                    last_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if last_frame is not None and last_frame.size > 0:
                        success = set_background_for_client(addr, last_frame)
                        if success:
                            # Reset tracking for this client
                            if addr in unattended_objects:
                                unattended_objects[addr] = {}
                            object_id_counters[addr] = 0
                            return jsonify({'status': 'Background set successfully'})
                except Exception as e:
                    print(f"Error setting background: {str(e)}")
    
    return jsonify({'error': 'Could not set background, no frame available'}), 400
                    
@app.route('/video_feed/<client_id>')
def video_feed(client_id):
    def generate():
        no_signal_img = np.zeros((480, 640, 3), dtype=np.uint8)
        no_signal_img = cv2.putText(no_signal_img, 'No Signal', (100, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', no_signal_img)
        no_signal_frame = buffer.tobytes()

        while True:
            addr = safe_client_id_to_tuple(client_id)
            with streams_lock:
                stream_data = client_streams.get(addr) if addr else None
                settings = detection_settings.get(addr, {'active': False, 'target_gender': None, 'last_alert': None})

            if stream_data:
                frame = stream_data['frame']
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + no_signal_frame + b'\r\n')
            
            time.sleep(0.033)  # ~30fps

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reset_trackers/<client_id>', methods=['POST'])
def reset_trackers(client_id):
    """Reset face trackers and object trackers for a client"""
    addr = safe_client_id_to_tuple(client_id)
    if not addr:
        return jsonify({'error': 'Invalid client ID'}), 400

    with streams_lock:
        if addr in face_trackers:
            face_trackers[addr] = {}
            face_id_counter[addr] = 0
        
        if addr in unattended_objects:
            unattended_objects[addr] = {}
            object_id_counters[addr] = 0
            
    return jsonify({'status': 'Trackers reset successfully'})


@app.route('/detection_stats')
def detection_stats():
    """Get statistics for all active detection streams"""
    with streams_lock:
        stats = {}
        for addr, data in client_streams.items():
            client_id = f"{addr[0]}:{addr[1]}"
            stats[client_id] = {
                'motion_detected': data.get('motion_detected', False),
                'fight_detected': data.get('fight_detected', False),
                'detected_gender': data.get('detected_gender', []),
                'unattended_objects': data.get('unattended_objects', 0),
                'mode': detection_settings[addr].get('mode', None),
                'active': detection_settings[addr].get('active', False),
                'target_gender': detection_settings[addr].get('target_gender', None)
            }
            
    return jsonify(stats)


@app.route('/system_info')
def system_info():
    """Get system information"""
    models_loaded = {
        'face_detection': faceNet is not None,
        'gender_detection': genderNet is not None,
        'pose_detection': yolo_model is not None,
        'hand_detection': use_hand_dnn,
        'violence_detection': violence_model is not None
    }
    
    active_clients = len(client_streams)
    
    return jsonify({
        'models_loaded': models_loaded,
        'active_clients': active_clients,
        'detection_modes': ['gender', 'motion', 'unattended', 'violence']
    })


@app.route('/view_background/<client_id>')
def view_background(client_id):
    """View the stored background for a client"""
    addr = safe_client_id_to_tuple(client_id)
    if not addr or addr not in background_models or background_models[addr].get('static_bg') is None:
        return jsonify({'error': 'No background available for this client'}), 404
    
    bg = background_models[addr]['static_bg']
    # Convert single channel to 3 channels for display
    bg_display = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    _, buffer = cv2.imencode('.jpg', bg_display)
    
    return Response(buffer.tobytes(), mimetype='image/jpeg')


@app.route('/api/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'time': time.time(),
        'active_clients': len(client_streams)
    })


@app.route('/api/snapshot/<client_id>', methods=['GET'])
def take_snapshot(client_id):
    """Take a snapshot from the client's video feed"""
    addr = safe_client_id_to_tuple(client_id)
    if not addr or addr not in client_streams:
        return jsonify({'error': 'Client not found or not connected'}), 404
    
    with streams_lock:
        frame_data = client_streams.get(addr, {}).get('frame')
        if not frame_data:
            return jsonify({'error': 'No frame available'}), 404
    
    # Create a timestamp for the filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"snapshot_{addr[0]}_{addr[1]}_{timestamp}.jpg"
    
    # Save the snapshot to disk
    try:
        # Convert the JPEG binary data back to numpy array
        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(f"snapshots/{filename}", img)
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'timestamp': timestamp
        })
    except Exception as e:
        return jsonify({'error': f'Failed to save snapshot: {str(e)}'}), 500


if __name__ == '__main__':
    # Create snapshots directory if it doesn't exist
    import os
    if not os.path.exists("snapshots"):
        os.makedirs("snapshots")
    
    # Suppress Flask request logs for cleaner console output
    #log = logging.getLogger('werkzeug')
    #log.setLevel(logging.ERROR)

    # Start the socket server in a background thread
    Thread(target=start_socket_server, daemon=True).start()
    
    # Start the Flask application
    app.run(host='0.0.0.0', port=5000, threaded=True)