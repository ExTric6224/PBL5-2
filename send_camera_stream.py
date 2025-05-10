# send_camera_stream.py (chạy trên Raspberry Pi)
import socket
import cv2
import struct
import time
from picamera2 import Picamera2

# 📌 Thay địa chỉ IP bên dưới bằng IP của máy LAPTOP bạn (máy chạy EmotionDetector)
SERVER_IP = '192.168.137.1'  # ← thay bằng địa chỉ IP thật của laptop bạn
PORT = 9999

# Khởi tạo camera
picam = Picamera2()
picam.preview_configuration.main.size = (320, 240)
picam.preview_configuration.main.format = "RGB888"
picam.configure("preview")
picam.start()

# Kết nối socket tới laptop
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(f"Kết nối tới máy chủ tại {SERVER_IP}:{PORT}...")
client_socket.connect((SERVER_IP, PORT))
print("✅ Kết nối thành công.")
conn = client_socket.makefile('wb')

try:
    while True:
        frame = picam.capture_array()
        _, encoded = cv2.imencode('.jpg', frame)
        data = encoded.tobytes()
        # Gửi kích thước + dữ liệu
        client_socket.sendall(struct.pack('>L', len(data)) + data)
        time.sleep(0.03)  # Gửi khoảng ~30fps (tùy tốc độ mạng)
except KeyboardInterrupt:
    print("\nĐã dừng gửi ảnh.")
finally:
    picam.close()
    conn.close()
    client_socket.close()
