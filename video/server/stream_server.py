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

app = Flask(__name__, template_folder='../templates', static_folder='../static')

client_streams = {}
streams_lock = Lock()
detection_settings = defaultdict(lambda: {'active': False, 'target_gender': None, 'last_alert': None})

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)
genderList = ['Male', 'Female']

def safe_client_id_to_tuple(client_id):
    try:
        if isinstance(client_id, str) and client_id.startswith('('):
            return literal_eval(client_id)
        elif ':' in client_id:
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

    if addr not in detection_settings:
        detection_settings[addr] = {}

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
                    frame, detected_gender = apply_gender_detection(frame)
                    if detected_gender == settings.get('target_gender'):
                        if settings.get('last_alert') != detected_gender:
                            print(f"Alert: {detected_gender} detected for {addr}")
                            settings['last_alert'] = detected_gender
                    else:
                        settings['last_alert'] = None
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
            else:
                frame_bytes = no_signal_frame

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')

            time.sleep(0.033)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    Thread(target=start_socket_server, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
