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

app = Flask(__name__, template_folder='../templates', static_folder='../static')

client_streams = {}
streams_lock = Lock()
detection_settings = defaultdict(lambda: {'active': False, 'target_gender': None, 'last_alert': None})
face_trackers = defaultdict(dict)
face_id_counter = defaultdict(lambda: 0)

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)
genderList = ['Male', 'Female']

MAX_HITS = 20
GENDER_LOCK_TIMEOUT = 3  # seconds

def safe_client_id_to_tuple(client_id):
    try:
        if isinstance(client_id, str) and client_id.startswith('('):
            return literal_eval(client_id)
        elif ':' in client_id:
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

    return cpy_input_image, detected_genders

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
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                raise ValueError("Could not encode frame")

            with streams_lock:
                client_streams[addr] = {'frame': buffer.tobytes(), 'timestamp': time.time()}
    except Exception as e:
        print(f"Client {addr} error: {str(e)}")
    finally:
        with streams_lock:
            client_streams.pop(addr, None)
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
    with streams_lock:
        clients = [f"{ip}:{port}" for ip, port in client_streams.keys()]
    return jsonify({'clients': clients})

@app.route('/toggle_detection/<client_id>', methods=['POST'])
def toggle_detection(client_id):
    addr = safe_client_id_to_tuple(client_id)
    if not addr:
        return jsonify({'error': 'Invalid client ID'}), 400

    data = request.get_json()
    gender = data.get('gender')

    if gender == 'None':
        detection_settings[addr]['active'] = False
        detection_settings[addr]['target_gender'] = None
        detection_settings[addr]['last_alert'] = None
        return jsonify({'status': 'Detection deactivated'})

    if gender not in genderList:
        return jsonify({'error': 'Invalid gender'}), 400

    detection_settings[addr]['active'] = True
    detection_settings[addr]['target_gender'] = gender
    detection_settings[addr]['last_alert'] = None
    return jsonify({'status': f'Detection ON for {gender}'})

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
                settings = detection_settings[addr] if addr else {'active': False}

            if stream_data:
                frame = cv2.imdecode(np.frombuffer(stream_data['frame'], np.uint8), cv2.IMREAD_COLOR)
                if settings['active']:
                    frame, detected_genders = apply_gender_detection(frame, addr)
                    if settings['target_gender'] in detected_genders:
                        if settings.get('last_alert') != settings['target_gender']:
                            print(f"🔔 ALERT: {settings['target_gender']} detected for {addr}")
                            settings['last_alert'] = settings['target_gender']
                    else:
                        settings['last_alert'] = None

                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
            else:
                frame_bytes = no_signal_frame

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Suppress Flask request logs
    #log = logging.getLogger('werkzeug')
    #log.setLevel(logging.ERROR)

    Thread(target=start_socket_server, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
