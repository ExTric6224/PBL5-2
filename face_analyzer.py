# face_analyzer.py
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Tắt bớt các thông báo log của TensorFlow (tùy chọn, có thể đặt ở file chính)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FaceAnalyzer:
    """
    Chịu trách nhiệm tải mô hình nhận diện cảm xúc và phân tích
    một hình ảnh khuôn mặt (ROI) để dự đoán cảm xúc.
    """
    def __init__(self, model_path='emotion_model.h5'):
        """
        Khởi tạo FaceAnalyzer bằng cách tải mô hình Keras.

        Args:
            model_path (str): Đường dẫn đến tệp mô hình .h5 đã huấn luyện.
        """
        self.img_rows, self.img_cols = 48, 48 # Kích thước ảnh đầu vào cho mô hình
        # QUAN TRỌNG: Thứ tự phải khớp với thứ tự thư mục con trong data/train
        self.emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
        self.model = self._load_model(model_path)
        if self.model is None:
             # Có thể raise Exception ở đây nếu muốn việc khởi tạo thất bại rõ ràng hơn
             raise ValueError(f"Không thể tải model từ {model_path}")

    def _load_model(self, model_path):
        """Tải mô hình Keras từ đường dẫn được cung cấp."""
        if not os.path.exists(model_path):
            print(f"Lỗi: Không tìm thấy tệp mô hình tại '{model_path}'")
            return None
        try:
            print(f"FaceAnalyzer: Đang tải mô hình từ {model_path}...")
            model = load_model(model_path)
            print("FaceAnalyzer: Tải mô hình thành công.")
            # In cấu trúc nếu muốn kiểm tra
            # print(model.summary())
            return model
        except Exception as e:
            print(f"FaceAnalyzer: Lỗi nghiêm trọng khi tải mô hình từ {model_path}: {e}")
            return None # Trả về None và để __init__ xử lý

    def analyzeFace(self, roi_gray):
        """
        Phân tích vùng khuôn mặt (ROI) ảnh xám để dự đoán cảm xúc.

        Args:
            roi_gray (numpy.ndarray): Hình ảnh khuôn mặt thang xám đã được cắt ra.

        Returns:
            dict: Một dictionary chứa 7 nhãn cảm xúc và xác suất tương ứng của chúng,
                  hoặc None nếu có lỗi trong quá trình phân tích.
        """
        try:
            # Thay đổi kích thước ROI về kích thước mô hình yêu cầu
            roi_gray_resized = cv2.resize(roi_gray, (self.img_rows, self.img_cols), interpolation=cv2.INTER_AREA)

            # Chuẩn hóa và định dạng lại ảnh để đưa vào mô hình
            roi = roi_gray_resized.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0) # Thêm chiều batch (1, 48, 48, 1)

            # Dự đoán cảm xúc
            prediction = self.model.predict(roi, verbose=0) # verbose=0 để không in log

            # Tạo dictionary chứa nhãn cảm xúc và xác suất tương ứng
            emotion_probabilities = {self.emotion_labels[i]: float(prediction[0][i]) for i in range(len(self.emotion_labels))}

            return emotion_probabilities

        except Exception as e:
            # Log lỗi chi tiết hơn có thể hữu ích
            print(f"FaceAnalyzer: Lỗi trong quá trình phân tích khuôn mặt: {e}")
            return None

# --- Khối kiểm thử (tùy chọn) ---
if __name__ == "__main__":
    print("Đang chạy kiểm thử cục bộ cho FaceAnalyzer...")
    # Đường dẫn mặc định, bạn có thể thay đổi nếu cần
    default_model_path = 'emotion_model.h5'

    # Kiểm tra xem model có tồn tại không
    if not os.path.exists(default_model_path):
        print(f"Lỗi: Không tìm thấy tệp model '{default_model_path}'. Không thể chạy kiểm thử.")
    else:
        try:
            # Khởi tạo analyzer
            analyzer = FaceAnalyzer(model_path=default_model_path)
            print("FaceAnalyzer đã được khởi tạo thành công.")

            # Tạo một ảnh xám giả 48x48 để kiểm tra
            # (Trong thực tế, bạn nên tải một ảnh khuôn mặt thật)
            dummy_face_roi = np.random.randint(0, 256, (100, 100), dtype=np.uint8) # Giả lập ROI lớn hơn
            print(f"Đang phân tích một ROI giả kích thước: {dummy_face_roi.shape}")

            # Phân tích ảnh giả
            predicted = analyzer.analyzeFace(dummy_face_roi)

            if predicted:
                print(f"Kết quả dự đoán cho ROI giả: {predicted}")
            else:
                print("Không thể dự đoán cảm xúc cho ROI giả (có thể có lỗi bên trong analyzeFace).")

        except ValueError as ve:
             print(f"Lỗi khi khởi tạo FaceAnalyzer: {ve}")
        except Exception as e:
            print(f"Đã xảy ra lỗi không mong muốn trong quá trình kiểm thử: {e}")

    print("Kết thúc kiểm thử FaceAnalyzer.")