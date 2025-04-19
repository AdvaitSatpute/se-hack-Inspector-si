import socket
import struct
import pickle
import cv2
import threading
import queue
import math
from tkinter import Tk, Label, Frame, Canvas
from PIL import Image, ImageTk, ImageOps

class CCTVServer:
    def __init__(self):
        self.root = Tk()
        self.root.title("CCTV Master Console")
        self.root.geometry("1200x800")
        self.root.minsize(640, 480)
        
        # Main container
        self.main_frame = Frame(self.root, bg='white')
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.labels = {}
        self.canvases = {}  # To maintain aspect ratio with white padding
        self.aspect_ratios = {}
        self.frame_queue = queue.Queue()
        
        # Server setup
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', 9999))
        self.server_socket.listen(5)
        
        # Start threads
        threading.Thread(target=self.accept_connections, daemon=True).start()
        self.root.after(100, self.update_gui)
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)
        self.root.bind('<Configure>', self.on_window_resize)

    def on_window_resize(self, event=None):
        self.arrange_grid()

    def calculate_grid_dimensions(self):
        num_cams = max(1, len(self.labels))
        if num_cams == 0:
            return 0, 0
        
        # Calculate nearest perfect square (for N×N grid)
        grid_size = math.ceil(math.sqrt(num_cams))
        # Ensure even number of rows/columns
        grid_size = grid_size + 1 if grid_size % 2 != 0 else grid_size
        return grid_size, grid_size

    def arrange_grid(self):
        rows, cols = self.calculate_grid_dimensions()
        
        # Clear current grid
        for widget in self.main_frame.winfo_children():
            widget.grid_forget()
        
        # Recreate grid
        for idx, (client_id, canvas) in enumerate(self.canvases.items()):
            row = idx // cols
            col = idx % cols
            canvas.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
        
        # Configure grid weights
        for r in range(rows):
            self.main_frame.grid_rowconfigure(r, weight=1)
        for c in range(cols):
            self.main_frame.grid_columnconfigure(c, weight=1)

    def update_gui(self):
        while not self.frame_queue.empty():
            client_id, frame = self.frame_queue.get()
            
            # Store aspect ratio if new camera
            if client_id not in self.aspect_ratios:
                height, width = frame.shape[:2]
                self.aspect_ratios[client_id] = width / height
                if client_id not in self.canvases:
                    self.add_new_camera(client_id)
            
            # Skip if frame is invalid
            if frame.size == 0:
                continue
                
            canvas = self.canvases[client_id]
            try:
                # Get available space
                canvas.update_idletasks()
                canvas_width = canvas.winfo_width()
                canvas_height = canvas.winfo_height()
                
                if canvas_width <= 1 or canvas_height <= 1:  # Skip if not yet visible
                    continue
                
                # Calculate target size with aspect ratio
                aspect = self.aspect_ratios[client_id]
                target_width = min(canvas_width, int(canvas_height * aspect))
                target_height = min(canvas_height, int(canvas_width / aspect))
                
                # Ensure minimum dimensions
                target_width = max(10, target_width)
                target_height = max(10, target_height)
                
                if target_width > 0 and target_height > 0:
                    # Resize frame
                    resized_frame = cv2.resize(frame, (target_width, target_height))
                    
                    # Convert to PIL image
                    img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
                    
                    # Create white background image
                    bg = Image.new('RGB', (canvas_width, canvas_height), color='white')
                    
                    # Calculate centered position
                    x = (canvas_width - target_width) // 2
                    y = (canvas_height - target_height) // 2
                    
                    # Paste the image centered on white background
                    bg.paste(img, (x, y))
                    
                    # Convert to Tkinter format
                    imgtk = ImageTk.PhotoImage(image=bg)
                    
                    # Update canvas
                    canvas.delete("all")
                    canvas.create_image(0, 0, anchor='nw', image=imgtk)
                    canvas.image = imgtk
                    
            except Exception as e:
                print(f"Display error for {client_id}: {str(e)}")
                continue
            
        self.root.after(50, self.update_gui)

    def add_new_camera(self, client_id):
        # Use Canvas instead of Label for better control
        canvas = Canvas(self.main_frame, bg='white', highlightthickness=0)
        self.canvases[client_id] = canvas
        self.arrange_grid()

    def handle_client(self, client_socket, addr):
        data = b""
        payload_size = struct.calcsize("Q")
        try:
            while True:
                while len(data) < payload_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        raise ConnectionError
                    data += packet
                
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]
                
                while len(data) < msg_size:
                    data += client_socket.recv(4096)
                
                frame_data = data[:msg_size]
                data = data[msg_size:]
                frame = pickle.loads(frame_data)
                self.frame_queue.put((addr[0], frame))
                
        except (ConnectionResetError, ConnectionAbortedError):
            print(f"{addr} Disconnected")
        finally:
            client_socket.close()

    def accept_connections(self):
        while True:
            try:
                client_socket, addr = self.server_socket.accept()
                print("Connected to:", addr)
                threading.Thread(target=self.handle_client, 
                               args=(client_socket, addr), daemon=True).start()
            except OSError:
                break

    def cleanup(self):
        self.server_socket.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    server = CCTVServer()
    server.run()