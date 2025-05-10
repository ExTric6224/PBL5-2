# emotion_detector.py
import cv2
import numpy as np
import argparse
import os
import time
import threading
from datetime import datetime
import socket
import struct

# --- Nhập các lớp Analyzer ---
from face_analyzer import FaceAnalyzer # Giả sử bạn có file này
from emotion_history_item import EmotionHistoryItem # Giả sử bạn có file này
from voice_analyzer import VoiceAnalyzer # Giả sử bạn có file này

# db
from db_utils import load_emotion_history_from_db # Giả sử bạn có file này

# --- Thư viện âm thanh (cho xử lý local) ---
SOUND_DEVICE_AVAILABLE = False # Sẽ được cập nhật dựa trên VoiceAnalyzer nếu dùng mic local
try:
    # Ví dụ, nếu VoiceAnalyzer kiểm tra sounddevice
    # Hoặc bạn có thể bỏ qua nếu VoiceAnalyzer không dùng mic local nữa
    if hasattr(VoiceAnalyzer, 'check_sound_device_availability'): # Giả sử có hàm này
        SOUND_DEVICE_AVAILABLE = VoiceAnalyzer.check_sound_device_availability()
    else: # Hoặc nếu VoiceAnalyzer luôn dùng mic local và không có check
        import sounddevice as sd # Thử import để xem có lỗi không
        SOUND_DEVICE_AVAILABLE = True
except ImportError:
    print("Cảnh báo: Thư viện sounddevice không được tìm thấy cho audio local.")
    SOUND_DEVICE_AVAILABLE = False
except Exception as e:
    print(f"Lỗi khi kiểm tra sounddevice: {e}")
    SOUND_DEVICE_AVAILABLE = False


class EmotionDetector:
    def __init__(self, face_analyzer, voice_analyzer, cascade_path='models/haarcascade_frontalface_default.xml',
                 enable_analysis_face=True, enable_analysis_voice=True,
                 camera_host='0.0.0.0', camera_port=9999,
                 # Các tham số cho audio server (nếu muốn nhận audio từ mạng)
                 # audio_host='0.0.0.0', audio_port=9998,
                 # receive_audio_from_network=False 
                 ):
        if not isinstance(face_analyzer, FaceAnalyzer):
            raise TypeError("'face_analyzer' phải là một instance của FaceAnalyzer.")
        self.face_analyzer = face_analyzer
        self.voice_analyzer = voice_analyzer
        self.face_cascade = self._load_cascade(cascade_path)

        if self.face_cascade is None:
            # Đã có exit trong __main__, có thể chỉ cần print ở đây nếu muốn class linh hoạt hơn
            print(f"LỖI NGHIÊM TRỌNG: Không thể tải Haar cascade từ: {cascade_path}")
            raise RuntimeError(f"Không thể tải Haar cascade từ: {cascade_path}")


        self.latest_frame = None
        self.last_face_emotion = "N/A"
        self.last_face_emotion_probabilities = None
        
        self.last_voice_emotion = "N/A (Local Mic)" # Thay đổi nếu nhận từ mạng
        self.last_voice_probabilities = None
        self.frame_lock = threading.Lock()
        self.emotion_lock = threading.Lock()

        self.stop_event = threading.Event()
        self.face_thread = None
        self.voice_thread = None
        
        self.enable_analysis_face = enable_analysis_face
        self.enable_analysis_voice = enable_analysis_voice
        self.emotion_history = load_emotion_history_from_db("emotion_log.db")
        self.can_send_to_UI = True

        # --- Network Server cho Camera ---
        self.camera_host = camera_host
        self.camera_port = camera_port
        self.camera_server_socket = None
        self.camera_client_conn = None
        self.camera_source_initialized = False

        # --- Cấu hình cho Audio (hiện tại là local, có thể mở rộng) ---
        # self.receive_audio_from_network = receive_audio_from_network
        # self.audio_host = audio_host
        # self.audio_port = audio_port
        # self.audio_server_socket = None
        # self.audio_client_conn = None
        # self.audio_source_initialized = False # Cần nếu nhận audio từ mạng


    def _load_cascade(self, cascade_path):
        if not os.path.exists(cascade_path):
            print(f"Lỗi: File cascade không tồn tại tại {cascade_path}")
            return None
        try:
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                print(f"Lỗi: File cascade tại {cascade_path} trống hoặc không hợp lệ.")
                return None
            print(f"Cascade đã tải thành công từ {cascade_path}")
            return cascade
        except Exception as e:
            print(f"Lỗi khi tải cascade từ {cascade_path}: {e}")
            return None

    def _init_camera_server(self):
        """Khởi tạo server socket để nhận dữ liệu hình ảnh từ Raspberry Pi."""
        try:
            self.camera_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.camera_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.camera_server_socket.bind((self.camera_host, self.camera_port))
            self.camera_server_socket.listen(1)
            print(f"📷 Camera Server: Đang lắng nghe trên {self.camera_host}:{self.camera_port} chờ Raspberry Pi kết nối...")
            self.camera_client_conn, client_addr = self.camera_server_socket.accept()
            print(f"✅ Camera Server: Raspberry Pi từ {client_addr} đã kết nối.")
            self.camera_source_initialized = True
            return True
        except Exception as e:
            print(f"❌ Camera Server: Lỗi khởi tạo socket: {e}")
            self.camera_source_initialized = False
            if self.camera_server_socket:
                self.camera_server_socket.close()
            return False

    def _face_processing_loop(self):
        print("🙂 Luồng xử lý khuôn mặt (nhận từ mạng): Bắt đầu.")
        if not self.camera_source_initialized or not self.camera_client_conn:
            print("❌ Luồng khuôn mặt: Nguồn camera (socket) chưa được khởi tạo hoặc chưa có kết nối.")
            return

        active_connection = True
        while not self.stop_event.is_set() and active_connection:
            if not self.can_send_to_UI:
                time.sleep(1) # Giảm tải CPU nếu UI không sẵn sàng
                continue
            
            try:
                # 1. Đọc 4 byte đầu tiên để lấy độ dài dữ liệu ảnh
                packed_image_len = self.camera_client_conn.recv(4)
                if not packed_image_len:
                    print("⚠️ Luồng khuôn mặt: Client (Pi) đã ngắt kết nối (không gửi độ dài).")
                    active_connection = False
                    break
                
                image_len = struct.unpack('>L', packed_image_len)[0]
                if image_len == 0: # Có thể là tín hiệu đặc biệt hoặc lỗi
                    print("⚠️ Luồng khuôn mặt: Nhận được độ dài ảnh bằng 0.")
                    time.sleep(0.1)
                    continue

                # 2. Đọc dữ liệu ảnh (JPEG bytes)
                image_data = b''
                while len(image_data) < image_len:
                    packet = self.camera_client_conn.recv(image_len - len(image_data))
                    if not packet:
                        print("⚠️ Luồng khuôn mặt: Client (Pi) ngắt kết nối khi đang gửi dữ liệu ảnh.")
                        active_connection = False
                        break
                    image_data += packet
                
                if not active_connection: break # Thoát nếu kết nối đã mất

                # 3. Giải mã ảnh JPEG thành frame OpenCV
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if frame is None:
                    print(" Lỗi Luồng khuôn mặt: Không thể giải mã frame nhận được từ Pi.")
                    time.sleep(0.1)
                    continue

                # --- Phần xử lý cảm xúc giữ nguyên ---
                if self.enable_analysis_face:   
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # (Giữ lại x, y, w, h ở scope rộng hơn nếu cần cho EmotionHistoryItem khi không có face)
                    detected_faces_in_frame = self.face_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(40, 40))
                    processed_frame = frame.copy() # Luôn tạo processed_frame
                    current_face_emotion_in_frame = "N/A"
                    current_emotion_prob = 0.0
                    face_coords = "N/A" # Khởi tạo tọa độ
                    predicted_distribution = None # Khởi tạo phân phối

                    for (x, y, w, h) in detected_faces_in_frame:
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_gray = gray_frame[y:y + h, x:x + w]
                        predicted = self.face_analyzer.analyzeFace(roi_gray)
                        if predicted:
                            current_face_emotion_in_frame, current_emotion_prob = max(predicted.items(), key=lambda item: item[1])
                            face_coords = f"{x}x{y}" # Cập nhật tọa độ
                            predicted_distribution = predicted # Lưu phân phối
                            # Hiển thị cảm xúc lên frame (ví dụ)
                            cv2.putText(processed_frame, f"{current_face_emotion_in_frame} ({current_emotion_prob:.2f})",
                                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            break # Chỉ xử lý khuôn mặt đầu tiên

                    with self.frame_lock:
                        self.latest_frame = processed_frame

                    # Chỉ cập nhật và ghi lịch sử nếu có sự thay đổi hoặc là N/A
                    # Hoặc nếu bạn muốn ghi lại mọi frame có cảm xúc (ngay cả khi giống frame trước)
                    # if current_face_emotion_in_frame != "N/A": # Điều kiện này có thể hơi chặt
                    with self.emotion_lock:
                        if self.last_face_emotion != current_face_emotion_in_frame or current_face_emotion_in_frame == "N/A":
                            if current_face_emotion_in_frame != "N/A" and predicted_distribution:
                                emotion_item = EmotionHistoryItem(
                                    timestamp=datetime.now(),
                                    face_location=face_coords,
                                    duration=None,
                                    result=current_face_emotion_in_frame,
                                    source="NetworkCamera", # Nguồn từ Camera Pi gửi qua
                                    emotion_distribution=predicted_distribution
                                )
                                self.emotion_history.append(emotion_item)
                        
                        self.last_face_emotion = current_face_emotion_in_frame
                        self.last_face_emotion_probabilities = current_emotion_prob
                
                else: # self.enable_analysis_face is False
                    with self.frame_lock:
                        self.latest_frame = frame.copy() # Gửi frame gốc nếu không phân tích
                    with self.emotion_lock:
                        self.last_face_emotion = "N/A"
                        self.last_face_emotion_probabilities = 0.0
            
            except socket.error as se:
                print(f"❌ Luồng khuôn mặt: Lỗi socket khi nhận dữ liệu: {se}")
                active_connection = False # Dừng vòng lặp nếu có lỗi socket nghiêm trọng
            except struct.error as ste:
                print(f"❌ Luồng khuôn mặt: Lỗi giải nén dữ liệu (struct): {ste}")
                # Có thể client gửi dữ liệu sai định dạng, cân nhắc đóng kết nối
                active_connection = False
            except Exception as e:
                print(f"❌ Luồng khuôn mặt: Lỗi không xác định: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5) # Đợi một chút nếu có lỗi lạ

        print("🙂 Luồng xử lý khuôn mặt (nhận từ mạng): Đã dừng.")
        # Dọn dẹp socket ở hàm stop() hoặc cleanup()
        if self.camera_client_conn:
            try:
                self.camera_client_conn.close()
            except: pass # Bỏ qua lỗi nếu đã đóng
            self.camera_client_conn = None
        # Không đóng server_socket ở đây để có thể chấp nhận kết nối mới nếu cần,
        # trừ khi chương trình dừng hẳn. Việc đóng server_socket nên ở stop() hoặc cleanup().

    def get_latest_data(self): # Giữ nguyên
        frame = None
        face_emo = "N/A"
        face_emo_prob = 0.0
        voice_emo = self.last_voice_emotion # Lấy từ self
        voice_emo_pros = self.last_voice_probabilities # Lấy từ self

        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()

        with self.emotion_lock:
            face_emo = self.last_face_emotion
            face_emo_prob = self.last_face_emotion_probabilities if self.last_face_emotion_probabilities is not None else 0.0
        
        if not isinstance(face_emo_prob, float): # Đảm bảo là float
            try: face_emo_prob = float(face_emo_prob)
            except: face_emo_prob = 0.0

        return frame, face_emo, face_emo_prob, voice_emo, voice_emo_pros

    def _voice_processing_loop(self):
        """
        Vòng lặp xử lý âm thanh. Hiện tại dùng microphone local.
        Để nhận từ mạng:
        1. Trong __init__, khởi tạo self.audio_server_socket, self.audio_client_conn tương tự camera.
        2. Trong start(), gọi một hàm _init_audio_server() để bind, listen, accept.
        3. Trong vòng lặp này, thay vì self.voice_analyzer.record_audio(),
           hãy conn.recv() dữ liệu audio (độ dài + mảng float32) từ Pi.
        4. Chuyển đổi bytes nhận được thành mảng NumPy float32.
        5. Gọi self.voice_analyzer.predict_emotion(audio_array_from_network).
        """
        print("🎤 Luồng âm thanh (local mic): Bắt đầu.")
        if not SOUND_DEVICE_AVAILABLE and self.enable_analysis_voice:
            print("Cảnh báo: Sounddevice không sẵn sàng, không thể xử lý âm thanh local.")
            # Vô hiệu hóa phân tích giọng nói nếu không có thiết bị
            # Hoặc nếu sau này bạn chuyển sang nhận audio từ mạng, phần này sẽ khác.
            with self.emotion_lock:
                self.last_voice_emotion = "N/A (No Mic)"
                self.last_voice_probabilities = None
            self.enable_analysis_voice = False # Tạm vô hiệu hóa
            # return # Thoát nếu không có mic và đang bật phân tích

        while not self.stop_event.is_set():
            if not self.enable_analysis_voice:
                # Nếu không phân tích giọng nói, cập nhật trạng thái N/A và ngủ
                if self.voice_analyzer and hasattr(self.voice_analyzer, 'emotion_labels'):
                     probabilities = {label: 0.0 for label in self.voice_analyzer.emotion_labels}
                else: # Fallback nếu voice_analyzer không có emotion_labels
                    probabilities = {"neutral":0.0, "happy":0.0, "sad":0.0, "angry":0.0, "fearful":0.0, "disgusted":0.0, "surprised":0.0} # Ví dụ
                with self.emotion_lock:
                    self.last_voice_emotion = "N/A"
                    self.last_voice_probabilities = probabilities
                time.sleep(1)
                continue

            # === Logic xử lý âm thanh local (giữ nguyên từ code gốc của bạn) ===
            try:
                # Giả sử voice_analyzer.record_audio() trả về mảng numpy
                # và voice_analyzer.predict_emotion() nhận mảng numpy
                if not self.voice_analyzer: # Kiểm tra xem voice_analyzer có tồn tại không
                    print("Lỗi: VoiceAnalyzer chưa được khởi tạo.")
                    time.sleep(1)
                    continue

                audio_array = self.voice_analyzer.record_audio() # Ghi âm local
                if audio_array is None: # Xử lý trường hợp record_audio trả về None (ví dụ: lỗi)
                    print("Không nhận được dữ liệu audio từ record_audio.")
                    time.sleep(1)
                    continue

                probabilities = self.voice_analyzer.predict_emotion(audio_array)

                if probabilities is None or "error" in probabilities: # Xử lý trường hợp predict_emotion lỗi
                    print("Lỗi khi dự đoán cảm xúc giọng nói hoặc không có kết quả.")
                    # Đặt giá trị mặc định an toàn
                    if hasattr(self.voice_analyzer, 'emotion_labels'):
                        safe_probabilities = {label: 0.0 for label in self.voice_analyzer.emotion_labels}
                    else:
                        safe_probabilities = {"neutral":1.0} # Ví dụ
                    with self.emotion_lock:
                        self.last_voice_emotion = "Error/None"
                        self.last_voice_probabilities = safe_probabilities
                    time.sleep(1)
                    continue
                
                with self.emotion_lock:
                    self.last_voice_emotion = max(probabilities, key=probabilities.get)
                    self.last_voice_probabilities = probabilities
                    
                    duration_ms = 0
                    if hasattr(self.voice_analyzer, 'duration_sec'):
                        duration_ms = int(self.voice_analyzer.duration_sec * 1000)

                    emotion_item = EmotionHistoryItem(
                        timestamp=datetime.now(),
                        face_location=None,
                        duration=duration_ms,
                        result=self.last_voice_emotion,
                        source="Microphone (Local)",
                        emotion_distribution=probabilities
                    )
                    self.emotion_history.append(emotion_item)

            except AttributeError as ae: # Bắt lỗi nếu voice_analyzer thiếu phương thức
                print(f"Lỗi thuộc tính trong VoiceAnalyzer: {ae}. Đảm bảo VoiceAnalyzer được triển khai đúng.")
                # Tạm dừng phân tích giọng nói nếu có lỗi nghiêm trọng với analyzer
                self.enable_analysis_voice = False 
                with self.emotion_lock:
                    self.last_voice_emotion = "Analyzer Error"
                    self.last_voice_probabilities = None
            except Exception as e:
                print(f"Lỗi trong luồng xử lý âm thanh: {e}")
                # Có thể đặt last_voice_emotion về trạng thái lỗi ở đây
                with self.emotion_lock:
                    self.last_voice_emotion = "Error"
                    self.last_voice_probabilities = None # Hoặc một dict xác suất lỗi
            
            # Thêm sleep nhỏ để tránh vòng lặp quá nhanh nếu record/predict nhanh
            time.sleep(0.1) # Hoặc dựa trên duration của voice_analyzer
        print("🎤 Luồng âm thanh (local mic): Đã dừng.")


    def start(self):
        if self.face_thread or self.voice_thread:
            print("Detector đã chạy rồi.")
            return

        # Khởi tạo server camera
        if not self._init_camera_server():
            # Không raise RuntimeError ở đây nữa để cho phép audio (nếu có) vẫn chạy
            # Hoặc nếu camera là bắt buộc thì raise
            print("LỖI NGHIÊM TRỌNG: Không thể khởi tạo Camera Server. Luồng khuôn mặt sẽ không hoạt động.")
            # self.camera_source_initialized sẽ là False
        else:
            print("Camera server đã sẵn sàng.")


        # Khởi tạo audio (hiện tại là local, nếu là server thì tương tự camera)
        # if self.receive_audio_from_network:
        #     if not self._init_audio_server(): # Cần tạo hàm này
        #         print("LỖI NGHIÊM TRỌNG: Không thể khởi tạo Audio Server.")
        # else:
        #     # Kiểm tra mic local nếu không nhận từ mạng và voice analysis được bật
        if self.enable_analysis_voice and not SOUND_DEVICE_AVAILABLE:
             print("Cảnh báo: Phân tích giọng nói được bật nhưng không tìm thấy thiết bị sounddevice local.")
             # Có thể quyết định dừng ở đây hoặc chỉ vô hiệu hóa voice_thread


        self.stop_event.clear()
        print("Bắt đầu các luồng xử lý nền...")
        
        if self.camera_source_initialized: # Chỉ start face_thread nếu server camera OK
            self.face_thread = threading.Thread(target=self._face_processing_loop, name="FaceProcessingThread", daemon=True)
            self.face_thread.start()
        else:
            print("Luồng khuôn mặt không được khởi động do Camera Server lỗi.")

        # Luồng voice vẫn có thể chạy với mic local (hoặc server audio nếu bạn triển khai)
        if self.enable_analysis_voice: # Chỉ start nếu được enable
            self.voice_thread = threading.Thread(target=self._voice_processing_loop, name="VoiceProcessingThread", daemon=True)
            self.voice_thread.start()
        else:
            print("Luồng giọng nói không được khởi động (đã bị vô hiệu hóa).")


    def stop(self):
        if self.stop_event.is_set(): return
        print("Bắt đầu quá trình dừng EmotionDetector...")
        self.stop_event.set()

        threads_to_join = []
        if self.face_thread and self.face_thread.is_alive():
            threads_to_join.append(self.face_thread)
        if self.voice_thread and self.voice_thread.is_alive():
            threads_to_join.append(self.voice_thread)

        for t in threads_to_join:
            print(f"Đang đợi luồng {t.name} dừng...")
            t.join(timeout=3.0) # Tăng timeout một chút
            if t.is_alive(): print(f"Cảnh báo: Luồng {t.name} không dừng kịp thời.")
        
        print("Các luồng xử lý nền đã dừng (hoặc đã được yêu cầu dừng).")
        self.cleanup() # Gọi cleanup sau khi các luồng đã join

    def cleanup(self):
        print("EmotionDetector: Thực hiện cleanup cuối cùng.")
        # Đóng socket camera
        if self.camera_client_conn:
            try: self.camera_client_conn.close()
            except Exception as e: print(f"Lỗi khi đóng camera_client_conn: {e}")
            self.camera_client_conn = None
        if self.camera_server_socket:
            try: self.camera_server_socket.close()
            except Exception as e: print(f"Lỗi khi đóng camera_server_socket: {e}")
            self.camera_server_socket = None
        print("Các socket camera đã được đóng (nếu có).")
        
        # Tương tự cho audio socket nếu bạn triển khai server audio
        # if self.audio_client_conn:
        #     try: self.audio_client_conn.close()
        #     except: pass
        # if self.audio_server_socket:
        #     try: self.audio_server_socket.close()
        #     except: pass
        # print("Các socket audio đã được đóng (nếu có).")
        pass


# --- Điểm khởi chạy chính ---
if __name__ == "__main__": # Sửa thành "__main__"
    # --- Import UIController ở đây ---
    # Đảm bảo UIController của bạn có thể xử lý việc frame đến trễ hoặc không có
    try:
        from ui_controller import EmotionGUI 
    except ImportError:
        print("LỖI: Không tìm thấy ui_controller.py. Giao diện sẽ không hoạt động.")
        # Có thể thoát ở đây hoặc chạy không có UI nếu detector có thể hoạt động độc lập
        EmotionGUI = None 


    ap = argparse.ArgumentParser(description="Chạy nhận diện cảm xúc đa luồng với UI Controller (Server Mode).")
    ap.add_argument("--face_model", default="models/emotion_model.h5", help="Đường dẫn face model")
    ap.add_argument("--cascade", default="models/haarcascade_frontalface_default.xml", help="Đường dẫn Haar cascade")
    # Thêm các argument cho host/port nếu muốn tùy chỉnh từ dòng lệnh
    ap.add_argument("--camera_host", default="0.0.0.0", help="Địa chỉ IP để Camera Server lắng nghe")
    ap.add_argument("--camera_port", default=9999, type=int, help="Cổng để Camera Server lắng nghe")
    # ap.add_argument("--receive_audio_network", action='store_true', help="Nhận audio từ mạng thay vì mic local")

    args = ap.parse_args()

    # --- Kiểm tra file ---
    if not os.path.exists(args.cascade):
        exit(f"Lỗi: Cascade không tìm thấy tại đường dẫn được cung cấp: {args.cascade}. Vui lòng kiểm tra lại.")
    if not os.path.exists(args.face_model):
        exit(f"Lỗi: Face model không tìm thấy tại: {args.face_model}. Vui lòng kiểm tra lại.")

    main_detector = None
    root = None # Khai báo để có thể truy cập trong finally nếu cần

    try:
        face_analyzer_inst = FaceAnalyzer(model_path=args.face_model)
        voi_analyzer_inst = VoiceAnalyzer() # Giả sử VoiceAnalyzer có thể khởi tạo không tham số
                                            # hoặc bạn cần cấu hình nó cho mic local/network
        
        main_detector = EmotionDetector(
            face_analyzer=face_analyzer_inst,
            voice_analyzer=voi_analyzer_inst,
            cascade_path=args.cascade,
            camera_host=args.camera_host,
            camera_port=args.camera_port,
            # receive_audio_from_network=args.receive_audio_network # Nếu có arg này
        )

        main_detector.start() # Khởi tạo server và các luồng
        print("EmotionDetector (Server Mode) đã khởi động. Đang chờ client (Raspberry Pi) kết nối...")

        if EmotionGUI: # Chỉ chạy UI nếu import thành công
            from ttkbootstrap import Style
            try:
                style = Style(theme="superhero") 
                root = style.master
            except Exception as e: # Bắt lỗi nếu ttkbootstrap có vấn đề
                print(f"Lỗi khởi tạo style/root cho ttkbootstrap: {e}")
                print("Sẽ thử tạo Tk root cơ bản.")
                import tkinter as tk
                root = tk.Tk()
                root.title("Emotion Detector (Basic Fallback UI)")


            gui = EmotionGUI(root) # Truyền root (master_window) vào EmotionGUI
            gui.detector = main_detector 

            def update_gui():
                # Lấy dữ liệu an toàn, có thể là None nếu chưa có gì
                frame, face_emo, face_emo_pro, voice_emo, voice_emo_pros = main_detector.get_latest_data()
                
                # update_video_frame cần có khả năng xử lý frame=None
                gui.update_video_frame(
                    frame, 
                    face_emotion=face_emo,
                    emotion_probability=face_emo_pro,
                    voice_emotion=voice_emo,
                    voice_probabilities=voice_emo_pros
                )
                if root: # Chỉ gọi after nếu root tồn tại
                    root.after(50, update_gui) # Tăng delay một chút cho server mode
            
            if root: # Chỉ chạy mainloop nếu root tồn tại
                update_gui()
                root.mainloop()
            else:
                print("Không thể khởi tạo root UI. Chạy ở chế độ không UI.")
                # Giữ luồng chính chạy để các luồng con (detector) tiếp tục
                while not main_detector.stop_event.is_set():
                    try:
                        time.sleep(1)
                    except KeyboardInterrupt:
                        print("\nPhát hiện Ctrl+C! Đang dừng detector...")
                        break # Thoát vòng lặp này, finally sẽ được gọi
        else:
            print("Chạy ở chế độ không UI (UIController không được tải).")
            # Giữ luồng chính chạy
            while not main_detector.stop_event.is_set():
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    print("\nPhát hiện Ctrl+C! Đang dừng detector...")
                    break


    except RuntimeError as re: # Bắt cụ thể lỗi Runtime từ init của EmotionDetector
        print(f"Lỗi RuntimeError khi khởi tạo EmotionDetector: {re}")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn trong __main__: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Bắt đầu quá trình dọn dẹp cuối cùng trong __main__...")
        if main_detector:
            print("Yêu cầu EmotionDetector dừng các luồng và dọn dẹp...")
            main_detector.stop() # stop() sẽ gọi cleanup() bên trong nó
            
            # Lưu lịch sử cảm xúc (nếu cần thiết và logic vẫn giữ)
            if hasattr(main_detector, 'emotion_history') and main_detector.emotion_history:
                 from db_utils import save_all_emotions_to_db
                 save_all_emotions_to_db("emotion_log.db", main_detector.emotion_history)
                 print("✅ Đã lưu toàn bộ lịch sử cảm xúc vào cơ sở dữ liệu.")
            else:
                print("Không có lịch sử cảm xúc để lưu hoặc thuộc tính không tồn tại.")
        
        # if EmotionGUI and root and hasattr(root, 'destroy'): # Đóng cửa sổ giao diện nếu có
        #     print("Đang đóng cửa sổ giao diện (nếu có)...")
        #     # root.destroy() # mainloop() sẽ tự xử lý việc này khi thoát

        print("Chương trình đã kết thúc hoàn toàn.")