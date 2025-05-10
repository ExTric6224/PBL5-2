import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os


class FaceAnalyzer:
    def __init__(self, model_path='./models/efficientnetb0_emotion_final.keras'):
        self.img_size = 48  # Kích thước ảnh đầu vào cho model
        self.emotion_labels = ['anger', 'happy', 'neutral', 'sad', 'surprise']
        self.model = self._load_model(model_path)
        if self.model is None:
            raise ValueError(f"❌ Không thể tải model từ {model_path}")

    def _load_model(self, path):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Không tìm thấy model tại: {path}")
            model = load_model(path)
            print(f"✅ Đã tải model từ {path}")
            return model
        except Exception as e:
            print(f"❌ Lỗi khi tải model: {e}")
            return None

    def analyzeFace(self, roi_gray):
        """
        Dự đoán cảm xúc từ một vùng khuôn mặt ảnh xám.
        Args:
            roi_gray (numpy.ndarray): ảnh mặt (gray, 2D)

        Returns:
            dict: {emotion: probability} hoặc None nếu lỗi
        """
        try:
            # Resize và chuẩn hóa ảnh
            face_resized = cv2.resize(roi_gray, (self.img_size, self.img_size))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=(0, -1))  # (1, 48, 48, 1)

            # Dự đoán
            predictions = self.model.predict(face_input, verbose=0)[0]
            result = {
                self.emotion_labels[i]: float(predictions[i])
                for i in range(len(self.emotion_labels))
            }
            return result

        except Exception as e:
            print(f"❌ Lỗi khi phân tích khuôn mặt: {e}")
            return None
