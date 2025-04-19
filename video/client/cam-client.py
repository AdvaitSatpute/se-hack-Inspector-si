import cv2
import socket
import pickle
import struct
import time

def send_video():
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(('10.10.41.52', 9999))  # Replace with server IP
            print("Connected to server successfully")
            cam = cv2.VideoCapture(0)
            
            while cam.isOpened():
                ret, frame = cam.read()
                #print(f"Frame captured: {ret}, Size: {frame.shape if ret else 'N/A'}")
                if not ret:
                    break
                
                # Resize frame for better performance
                frame = cv2.resize(frame, (640, 480))
                data = pickle.dumps(frame)
                message = struct.pack("Q", len(data)) + data
                
                try:
                    client_socket.sendall(message)
                    #print(f"Sent frame ({len(data)} bytes)")
                except (ConnectionResetError, BrokenPipeError):
                    print("Connection lost, reconnecting...")
                    time.sleep(2)
                    break
                
            cam.release()
            client_socket.close()
            
        except ConnectionRefusedError:
            print("Server unavailable, retrying in 5 seconds...")
            time.sleep(5)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    send_video()