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
# import queue # Có thể vẫn cần sau này cho giao tiếp phức tạp hơn

# --- Nhập các lớp Analyzer ---
from face_analyzer import FaceAnalyzer
from emotion_history_item import EmotionHistoryItem
from voice_analyzer import VoiceAnalyzer

# db
from db_utils import load_emotion_history_from_db

# --- Thư viện âm thanh (vẫn giữ) ---
try:
    # import sounddevice as sd
    SOUND_DEVICE_AVAILABLE = True
except ImportError:
    SOUND_DEVICE_AVAILABLE = False
except Exception:
    SOUND_DEVICE_AVAILABLE = False

class EmotionDetector:
    """
    Quản lý các luồng xử lý nền (khuôn mặt, giọng nói) và cung cấp
    dữ liệu (frame, cảm xúc) cho bên ngoài (ví dụ: UIController thông qua luồng chính).
    """
    def __init__(self, face_analyzer,voice_analyzer,cascade_path='src/haarcascade_frontalface_default.xml',
             enable_analysis_face=True, enable_analysis_voice=True):
        # --- Components ---
        if not isinstance(face_analyzer, FaceAnalyzer):
             raise TypeError("'face_analyzer' phải là một instance của FaceAnalyzer.")
        self.face_analyzer = face_analyzer
        self.voice_analyzer = voice_analyzer  # Initialize as an instance of VoiceAnalyzer
        self.face_cascade = self._load_cascade(cascade_path)

        if self.face_cascade is None:
            raise RuntimeError("Không thể tải Haar cascade.")

        # --- Shared Data ---
        self.latest_frame = None
        self.last_face_emotion = "N/A"
        self.last_face_emotion_probabilities = None # Có thể dùng sau này
        
        self.last_voice_emotion = "N/A (Placeholder)"
        self.last_voice_probabilities = None  # Lưu probabilities để gửi cho UI
        self.frame_lock = threading.Lock()
        self.emotion_lock = threading.Lock()

        # --- Thread Control ---
        self.stop_event = threading.Event()
        self.face_thread = None
        self.voice_thread = None
        self.cap = None
        
        self.enable_analysis_face = enable_analysis_face
        self.enable_analysis_voice = enable_analysis_voice
        self.emotion_history = []  # Danh sách lưu trữ các EmotionHistoryItem
        self.emotion_history = load_emotion_history_from_db("emotion_log.db")
        self.can_send_to_UI = True;
        
        ##socket
        self.socket = None
        self.connection = None


    def _load_cascade(self, cascade_path):
        # (Giữ nguyên)  
        if not os.path.exists(cascade_path): return None
        try:
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty(): raise IOError("Cascade trống")
            return cascade
        except Exception as e: print(f"Lỗi tải cascade: {e}"); return None

    def _init_webcam(self):
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind(('0.0.0.0', 9999))  # Lắng nghe trên tất cả IP máy
            self.socket.listen(1)
            print("🕓 Đang chờ Raspberry Pi kết nối...")
            self.conn, _ = self.socket.accept()
            self.connection = self.conn.makefile('rb')
            print("✅ Raspberry Pi đã kết nối.")
            return True
        except Exception as e:
            print(f"Lỗi socket: {e}")
            return False


    # --- Luồng xử lý khuôn mặt ---
    def _face_processing_loop(self):
        print("Luồng webcam: Bắt đầu.")
        if not self.cap:
            print("Lỗi: Webcam chưa được khởi tạo.")
            return

        while not self.stop_event.is_set():
            if self.can_send_to_UI:
                image_len_data = self.connection.read(4)
                if not image_len_data:
                    time.sleep(0.1)
                    continue

                image_len = struct.unpack('>L', image_len_data)[0]
                image_data = self.connection.read(image_len)

                image_array = np.frombuffer(image_data, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                ret = frame is not None

                if not ret:
                    print("Lỗi đọc frame từ webcam.")
                    time.sleep(0.5)
                    continue

                if self.enable_analysis_face:   
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(40, 40))
                    processed_frame = frame.copy()
                    current_face_emotion_in_frame = "N/A"

                    for (x, y, w, h) in faces:
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_gray = gray_frame[y:y + h, x:x + w]
                        predicted = self.face_analyzer.analyzeFace(roi_gray)
                        if predicted:
                            # Lấy nhãn cảm xúc có xác suất cao nhất
                            current_face_emotion_in_frame, prob = max(predicted.items(), key=lambda item: item[1])
                            break

                    with self.frame_lock:
                        self.latest_frame = processed_frame

                    if current_face_emotion_in_frame != "N/A":
                        with self.emotion_lock:
                            if self.last_face_emotion != current_face_emotion_in_frame:
                                # Thêm vào danh sách emotion_history
                                emotion_item = EmotionHistoryItem(
                                    timestamp=datetime.now(),
                                    face_location=f"{x}x{y}",
                                    duration=None,  # Hoặc duration tạm thời nếu bạn đo được thời gian hiện diện
                                    result=current_face_emotion_in_frame,
                                    source="Webcam",
                                    emotion_distribution=predicted
                                )
                                self.emotion_history.append(emotion_item)
    
                            self.last_face_emotion = current_face_emotion_in_frame
                            self.last_face_emotion_probabilities = prob

                else:
                    # Không phân tích: chỉ hiển thị frame gốc
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                    with self.emotion_lock:
                        self.last_face_emotion = "N/A"
                        self.last_face_emotion_probabilities = 0.0
            else:
                # Nếu không gửi được đến UI, chỉ cần chờ một chút
                time.sleep(2)
        print("Luồng webcam: Đã dừng.")
        if self.cap:
            self.cap.release()
            print("Webcam đã được giải phóng.")

    def get_latest_data(self):
        """Lấy dữ liệu mới nhất từ các luồng (thread-safe)."""
        frame = None
        face_emo = "N/A"
        face_emo_prob = 0.0
        voice_emo = "N/A (Placeholder)"
        voice_emo_pros = None

        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy() # Trả về bản sao

        with self.emotion_lock:
            face_emo = self.last_face_emotion
            face_emo_prob = self.last_face_emotion_probabilities if self.last_face_emotion_probabilities else 0.0
            voice_emo = self.last_voice_emotion
            voice_emo_pros = self.last_voice_probabilities if self.last_voice_probabilities else None

        return frame, face_emo,face_emo_prob , voice_emo, voice_emo_pros

    # --- Luồng xử lý giọng nói (Placeholder) ---
    def _voice_processing_loop(self):
        """Vòng lặp xử lý âm thanh bằng VoiceAnalyzer mới."""
        print("Luồng âm thanh: Bắt đầu.")
        while not self.stop_event.is_set():
            if not self.enable_analysis_voice:
                probabilities = {label: 0.0 for label in self.voice_analyzer.emotion_labels}
                with self.emotion_lock:
                    self.last_voice_emotion = "N/A"
                    self.last_voice_probabilities = probabilities
                time.sleep(1)
                continue

            try:
                # Ghi âm và nhận mảng âm thanh
                audio_array = self.voice_analyzer.record_audio()

                # Dự đoán cảm xúc
                probabilities = self.voice_analyzer.predict_emotion(audio_array)

                if "error" in probabilities:
                    print("Lỗi khi trích xuất đặc trưng. Bỏ qua lần này.")
                    continue

                with self.emotion_lock:
                    self.last_voice_emotion = max(probabilities, key=probabilities.get)
                    self.last_voice_probabilities = probabilities

                    # Ghi lịch sử vào RAM (hoặc DB nếu bạn muốn)
                    emotion_item = EmotionHistoryItem(
                        timestamp=datetime.now(),
                        face_location=None,
                        duration=int(self.voice_analyzer.duration_sec * 1000),
                        result=self.last_voice_emotion,
                        source="Microphone",
                        emotion_distribution=probabilities
                    )
                    self.emotion_history.append(emotion_item)

            except Exception as e:
                print(f"Lỗi trong luồng âm thanh: {e}")

            time.sleep(1)
        print("Luồng âm thanh: Đã dừng.")


    # --- Điều khiển chính ---
    def start(self):
        """Khởi tạo webcam và bắt đầu các luồng xử lý."""
        if self.face_thread or self.voice_thread: return # Đã chạy

        self.cap = self._init_webcam()
        if not self.cap: raise RuntimeError("Không thể khởi tạo webcam.")

        self.stop_event.clear()
        print("Bắt đầu các luồng xử lý nền...")
        self.face_thread = threading.Thread(target=self._face_processing_loop, daemon=True)
        self.voice_thread = threading.Thread(target=self._voice_processing_loop, daemon=True)
        self.face_thread.start()
        self.voice_thread.start()

    def stop(self):
        """Báo hiệu dừng cho các luồng và đợi chúng kết thúc."""
        if self.stop_event.is_set(): return # Đã yêu cầu dừng
        print("Bắt đầu quá trình dừng EmotionDetector...")
        self.stop_event.set()

        threads_to_join = [self.face_thread, self.voice_thread]
        for t in threads_to_join:
             if t and t.is_alive():
                 print(f"Đang đợi luồng {t.name} dừng...")
                 t.join(timeout=2.0)
                 if t.is_alive(): print(f"Cảnh báo: Luồng {t.name} không dừng kịp thời.")

        print("Các luồng xử lý nền đã dừng.")
        # Không gọi cleanup() ở đây nữa, cleanup chỉ chứa logic không liên quan đến thread join

    def cleanup(self):
        """Dọn dẹp tài nguyên không được quản lý bởi luồng con."""
        print("EmotionDetector: Thực hiện cleanup cuối cùng (nếu có).")
        # Ví dụ: đóng file log, ngắt kết nối DB,...
        # Webcam và audio (sau này) được release trong luồng của chúng.
        # Cửa sổ OpenCV do UIController quản lý.
        pass


# --- Điểm khởi chạy chính ---
if __name__ == "__main__":
    # --- Import UIController ở đây ---
    from ui_controller import EmotionGUI

    ap = argparse.ArgumentParser(description="Chạy nhận diện cảm xúc đa luồng với UI Controller.")
    ap.add_argument("--face_model", default="emotion_model.h5", help="Đường dẫn face model")
    ap.add_argument("--cascade", default="src/haarcascade_frontalface_default.xml", help="Đường dẫn Haar cascade")
    args = ap.parse_args()

    # --- Kiểm tra file ---
    if not os.path.exists(args.cascade): exit(f"Lỗi: Cascade không tìm thấy: {args.cascade}")
    if not os.path.exists(args.face_model): exit(f"Lỗi: Face model không tìm thấy: {args.face_model}")

    # --- Khởi tạo ---
    main_detector = None
    ui_controller = None
    keep_running = True # Cờ để điều khiển vòng lặp chính

    try:
        # 1. Khởi tạo FaceAnalyzer
        face_analyzer_inst = FaceAnalyzer(model_path=args.face_model)
        
        # khởi tạo VoiceAnalyzer 
        voi_analyzer_inst = VoiceAnalyzer();
        
        # 2. Khởi tạo EmotionDetector
        main_detector = EmotionDetector(
            face_analyzer=face_analyzer_inst,
            voice_analyzer=voi_analyzer_inst,
            cascade_path=args.cascade
        )

        # 3. Khởi tạo giao diện EmotionGUI
        from ttkbootstrap import Style
        style = Style("superhero")  # hoặc "litera", "flatly", "darkly"
        root = style.master
        gui = EmotionGUI(root)
        gui.detector = main_detector  # GÁN emotion detector cho GUI để có thể điều khiển enable_analysis
        # 4. Bắt đầu các luồng xử lý nền của detector
        main_detector.start()
        print("Các luồng nền đã bắt đầu. Bắt đầu vòng lặp hiển thị chính...")
        time.sleep(1.0) # Đợi chút

        # 5. Vòng lặp hiển thị và xử lý input (trong luồng chính)
        def update_gui():
            frame, face_emo,face_emo_pro, voice_emo, voice_emo_pros = main_detector.get_latest_data()
            gui.update_video_frame(
                frame,
                face_emotion=face_emo,
                emotion_probability=face_emo_pro,
                voice_emotion=voice_emo,
                voice_probabilities=voice_emo_pros
            )

            root.after(30, update_gui)

        update_gui()
        root.mainloop()

    except (ValueError, TypeError, RuntimeError, IOError) as e:
         print(f"Lỗi nghiêm trọng khi khởi tạo hoặc chạy: {e}")
         keep_running = False # Dừng vòng lặp nếu có lỗi
    except KeyboardInterrupt:
         print("\nPhát hiện Ctrl+C!")
         keep_running = False # Dừng vòng lặp
    except Exception as e:
        print(f"\nĐã xảy ra lỗi không mong muốn: {e}")
        import traceback
        traceback.print_exc()
        keep_running = False # Dừng vòng lặp
    finally:
        print("Bắt đầu quá trình dọn dẹp cuối cùng...")
        if main_detector:
            from db_utils import save_all_emotions_to_db
            save_all_emotions_to_db("emotion_log.db", main_detector.emotion_history)
            print("✅ Đã lưu toàn bộ lịch sử cảm xúc vào cơ sở dữ liệu.")
            print("Yêu cầu EmotionDetector dừng các luồng...")
            main_detector.stop() # Đợi các luồng kết thúc
            main_detector.cleanup() # Chạy cleanup của detector (nếu có)

        # 2. Đóng cửa sổ giao diện
        if ui_controller:
            print("Yêu cầu UIController đóng cửa sổ...")
            ui_controller.destroy_windows()

        print("Chương trình đã kết thúc hoàn toàn.")
