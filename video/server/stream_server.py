# server.py
import socket
import pickle
import struct
import numpy as np
from threading import Thread, Lock
import cv2
import time
from flask import Flask, Response, jsonify, render_template, request, redirect, url_for
from ast import literal_eval
from collections import defaultdict
from ultralytics import YOLO  # YOLOv8
import mediapipe as mp

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Dictionary to hold client streams with thread-safe access
client_streams = {}
streams_lock = Lock()
detection_settings = defaultdict(lambda: {'active': False, 'mode': None, 'target_gender': None, 'last_alert': None})

# Initialize Gender Detection models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)
genderList = ['Male', 'Female']

# Initialize Motion Detection models
yolo_model = YOLO('yolov8n-pose.pt')  # YOLOv8 small pose model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Motion detection parameters
min_area = 600
threshold_value = 25

# Store previous keypoints for fight detection (per client)
previous_keypoints = {}


def safe_client_id_to_tuple(client_id):
    """Safely convert client_id string to tuple format"""
    try:
        if isinstance(client_id, str) and client_id.startswith('('):
            return literal_eval(client_id)  # Safer than eval
        elif ':' in client_id:  # Handle "ip:port" format
            ip, port = client_id.split(':')
            return (ip, int(port))
    except:
        pass
    return None


def apply_gender_detection(frame):
    upscale_frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
    cpy_input_image = upscale_frame.copy()
    frameWidth = cpy_input_image.shape[1]
    frameHeight = cpy_input_image.shape[0]

    blob = cv2.dnn.blobFromImage(cpy_input_image, scalefactor=1, size=(227, 227),
                                 mean=(104, 117, 123), crop=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()

    detected_gender = None

    for i in range(detections.shape[2]):
        confidence_score = detections[0, 0, i, 2]
        if confidence_score > 0.7:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            h, w = upscale_frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)

            if x2 - x1 > 0 and y2 - y1 > 0:
                face_region = upscale_frame[y1:y2, x1:x2]
                if face_region.size != 0:
                    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    equalized = clahe.apply(gray)
                    face_region = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

                    blob = cv2.dnn.blobFromImage(face_region, scalefactor=1, size=(227, 227),
                                                 mean=[78.4263377603, 87.7689143744, 114.895847746], crop=False)
                    genderNet.setInput(blob)
                    prediction = genderNet.forward()

                    top_gender = genderList[prediction[0].argmax()]
                    confidence = prediction[0].max()

                    if confidence > 0.6:
                        detected_gender = top_gender
                        cv2.rectangle(cpy_input_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        cv2.putText(cpy_input_image, detected_gender, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)

    return cpy_input_image, detected_gender


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
    
    # Hand detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(rgb_frame)
    
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
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
        status_text = "FIGHT DETECTED!"
        status_color = (0, 0, 255)
    else:
        status_text = "No Fight"
        status_color = (0, 255, 0)
    
    cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    motion_status_text = "Motion: DETECTED" if motion_detected else "Motion: NOT DETECTED"
    motion_status_color = (0, 0, 255) if motion_detected else (0, 255, 0)
    cv2.putText(frame, motion_status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, motion_status_color, 2)
    
    # Update client stream data with detection results
    with streams_lock:
        if addr in client_streams:
            client_streams[addr]['avg_frame'] = avg_frame
            client_streams[addr]['motion_detected'] = motion_detected
            client_streams[addr]['fight_detected'] = fight_detected
    
    return frame, motion_detected, fight_detected


def handle_client(conn, addr):
    print(f"Connected: {addr}")
    data = b""
    payload_size = struct.calcsize("Q")

    try:
        while True:
            while len(data) < payload_size:
                packet = conn.recv(4 * 1024)
                if not packet:
                    raise ConnectionError("Client disconnected")
                data += packet

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            while len(data) < msg_size:
                packet = conn.recv(4 * 1024)
                if not packet:
                    raise ConnectionError("Client disconnected")
                data += packet

            frame_data = data[:msg_size]
            data = data[msg_size:]
            frame = pickle.loads(frame_data)
            
            # Store original frame for gender detection
            original_frame = frame.copy()
            
            # Process frame based on detection mode set for this client
            with streams_lock:
                settings = detection_settings[addr] if addr in detection_settings else {'active': False, 'mode': None}
                
            if settings['active']:
                if settings['mode'] == 'gender':
                    processed_frame, detected_gender = apply_gender_detection(original_frame)
                    
                    # Update gender detection alert status
                    if detected_gender == settings.get('target_gender'):
                        if settings.get('last_alert') != detected_gender:
                            print(f"Alert: {detected_gender} detected for {addr}")
                            settings['last_alert'] = detected_gender
                    else:
                        settings['last_alert'] = None
                        
                    motion_detected = False
                    fight_detected = False
                else:  # Mode is motion
                    processed_frame, motion_detected, fight_detected = process_motion_detection(original_frame, addr)
                    detected_gender = None
            else:
                # If detection is not active, just use the original frame
                processed_frame = original_frame
                motion_detected = False
                fight_detected = False
                detected_gender = None
            
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                raise ValueError("Could not encode frame")

            with streams_lock:
                client_streams[addr] = {
                    'frame': buffer.tobytes(),
                    'timestamp': time.time(),
                    'motion_detected': motion_detected,
                    'fight_detected': fight_detected,
                    'detected_gender': detected_gender
                }
    except Exception as e:
        print(f"Client {addr} error: {str(e)}")
    finally:
        with streams_lock:
            client_streams.pop(addr, None)
            if addr in previous_keypoints:
                del previous_keypoints[addr]
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
                
        # Convert all keys to string representation for JSON
        client_ids = [f"{ip}:{port}" for ip, port in client_streams.keys()]
        
        # Get detection status for each client
        client_statuses = {}
        for addr in client_streams.keys():
            client_id = f"{addr[0]}:{addr[1]}"
            client_statuses[client_id] = {
                'motion_detected': client_streams[addr].get('motion_detected', False),
                'fight_detected': client_streams[addr].get('fight_detected', False),
                'detected_gender': client_streams[addr].get('detected_gender', None),
                'mode': detection_settings[addr].get('mode', None),
                'active': detection_settings[addr].get('active', False)
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
    else:
        return jsonify({'error': 'Invalid detection mode'}), 400


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

            if stream_data:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       stream_data['frame'] + b'\r\n')
            else:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       no_signal_frame + b'\r\n')

            time.sleep(0.033)  # ~30fps

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    Thread(target=start_socket_server, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)