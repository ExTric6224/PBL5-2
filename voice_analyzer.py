import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import joblib
import json
import os

class VoiceAnalyzer:
    def __init__(self,
                 model_path="./models/best_crnn_model.keras",
                 emotion_classes_path="emotion_classes.json",
                 feature_params_path="feature_extraction_params.json"):
        
        # --- Load mô hình ---
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model tại {model_path}")
        self.model = tf.keras.models.load_model(model_path)

        # --- Load danh sách nhãn cảm xúc ---
        if not os.path.exists(emotion_classes_path):
            raise FileNotFoundError(f"Không tìm thấy emotion_classes tại {emotion_classes_path}")
        with open(emotion_classes_path, 'r') as f:
            self.emotion_labels = json.load(f)

        # --- Load tham số trích đặc trưng ---
        if not os.path.exists(feature_params_path):
            raise FileNotFoundError(f"Không tìm thấy tham số đặc trưng tại {feature_params_path}")
        with open(feature_params_path, 'r') as f:
            self.params = json.load(f)

        # --- Gán các tham số ---
        self.sr = self.params["SR"]
        self.n_fft = self.params["N_FFT"]
        self.hop_length = self.params["HOP_LENGTH"]
        self.n_mels = self.params["N_MELS"]
        self.max_frames = self.params["MAX_FRAMES"]
        self.duration_sec = self.max_frames * self.hop_length / self.sr  # tính thời lượng từ số frame

    def record_audio(self, duration=None, filename=None):
        """Ghi âm và trả về dữ liệu audio dưới dạng numpy array."""
        duration = duration or self.duration_sec
        print(f"🎙️ Ghi âm trong {duration:.2f} giây...")
        recording = sd.rec(int(duration * self.sr), samplerate=self.sr, channels=1, dtype='float32')
        sd.wait()
        print("✅ Ghi âm xong.")
        audio = recording.flatten()

        if filename:
            from scipy.io.wavfile import write
            write(filename, self.sr, recording)

        return audio

    def _extract_mel_spectrogram(self, audio):
        """Trích xuất log Mel Spectrogram khớp với input khi train."""
        mel_spec = librosa.feature.melspectrogram(y=audio,
                                                  sr=self.sr,
                                                  n_fft=self.n_fft,
                                                  hop_length=self.hop_length,
                                                  n_mels=self.n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Cắt/pad chiều thời gian
        current_frames = log_mel_spec.shape[1]
        if current_frames > self.max_frames:
            log_mel_spec = log_mel_spec[:, :self.max_frames]
        elif current_frames < self.max_frames:
            pad_width = self.max_frames - current_frames
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=np.min(log_mel_spec))

        return log_mel_spec

    def predict_emotion(self, audio_array):
        """Dự đoán cảm xúc từ dữ liệu audio."""
        try:
            features = self._extract_mel_spectrogram(audio_array)
            input_tensor = features[np.newaxis, ..., np.newaxis]  # shape: (1, n_mels, max_frames, 1)

            probabilities = self.model.predict(input_tensor)[0]  # vector xác suất
            result = {self.emotion_labels[i]: float(probabilities[i]) for i in range(len(self.emotion_labels))}
            return result
        except Exception as e:
            print(f"Lỗi khi dự đoán cảm xúc: {e}")
            return {"error": str(e)}
