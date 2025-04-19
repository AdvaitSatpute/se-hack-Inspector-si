import socket
import pickle
import struct
import numpy as np
from threading import Thread, Lock
import cv2
import time
from flask import Flask, Response, jsonify, render_template
import os
from ast import literal_eval

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Dictionary to hold client streams with thread-safe access
client_streams = {}
streams_lock = Lock()

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

@app.route('/active_streams')
def active_streams():
    current_time = time.time()
    
    with streams_lock:
        # Remove stale clients
        stale_clients = [addr for addr, data in client_streams.items() 
                        if current_time - data['timestamp'] > 5]
        for client in stale_clients:
            del client_streams[client]
        
        # Convert all keys to string representation for JSON
        client_ids = [str(addr) for addr in client_streams.keys()]
        return jsonify({
            'clients': client_ids
        })

def handle_client(conn, addr):
    print(f"Connected: {addr}")
    data = b""
    payload_size = struct.calcsize("Q")
    
    try:
        while True:
            # Receive message length
            while len(data) < payload_size:
                packet = conn.recv(4*1024)
                if not packet: 
                    raise ConnectionError("Client disconnected")
                data += packet
            
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]
            
            # Receive frame data
            while len(data) < msg_size:
                packet = conn.recv(4*1024)
                if not packet:
                    raise ConnectionError("Client disconnected")
                data += packet
            
            frame_data = data[:msg_size]
            data = data[msg_size:]
            
            # Unpickle and store frame
            frame = pickle.loads(frame_data)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                raise ValueError("Could not encode frame")
            
            with streams_lock:
                client_streams[addr] = {
                    'frame': buffer.tobytes(),
                    'timestamp': time.time()
                }
                #print(f"Stored frame for {addr}")  # Debug
                
    except (ConnectionError, struct.error, ValueError) as e:
        print(f"Client {addr} error: {str(e)}")
    finally:
        with streams_lock:
            if addr in client_streams:
                del client_streams[addr]
        conn.close()
        print(f"Connection closed: {addr}")

def start_socket_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', 9999))
    server_socket.listen(5)
    print("Socket server listening on port 9999")
    
    try:
        while True:
            conn, addr = server_socket.accept()
            Thread(target=handle_client, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("Shutting down socket server")
    finally:
        server_socket.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<client_id>')
def video_feed(client_id):
    def generate():
        # Initialize default no_signal frame
        no_signal_img = np.zeros((480, 640, 3), dtype=np.uint8)
        no_signal_img = cv2.putText(
            no_signal_img, 'No Signal', (100, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        _, buffer = cv2.imencode('.jpg', no_signal_img)
        no_signal_frame = buffer.tobytes()
        
        while True:
            # Convert client_id to tuple format
            addr = safe_client_id_to_tuple(client_id)
            
            with streams_lock:
                stream_data = client_streams.get(addr) if addr else None
                
                if stream_data:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + 
                          stream_data['frame'] + b'\r\n')
                else:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + 
                          no_signal_frame + b'\r\n')
            
            time.sleep(0.033)  # ~30fps

    return Response(generate(),
                  mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start socket server in a separate thread
    Thread(target=start_socket_server, daemon=True).start()
    
    # Start Flask web server
    app.run(host='0.0.0.0', port=5000, threaded=True)