import os
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class VoiceAnalyzer:
    def __init__(self,
                 model_path="best_crnn_model.keras",
                 scaler_mean_path="crnn_scaler_mean.npy",
                 scaler_scale_path="crnn_scaler_scale.npy",
                 label_classes_path="crnn_label_encoder_classes.npy",
                 target_sr=16000,
                 duration_sec=3.0):
        
        # --- Thông số ---
        self.target_sr = target_sr
        self.duration_sec = duration_sec
        self.fixed_samples = int(target_sr * duration_sec)

        # --- Cấu hình đặc trưng ---
        self.n_mfcc = 21
        self.fixed_frames = 49
        self.frame_length_fft = 2048
        self.hop_length_fft = 512

        # --- Tải model và scaler ---
        self.model = tf.keras.models.load_model(model_path)
        mean = np.load(scaler_mean_path)
        scale = np.load(scaler_scale_path)
        self.scaler = StandardScaler()
        self.scaler.mean_ = mean
        self.scaler.scale_ = scale
        self.scaler.n_features_in_ = len(mean)

        # --- Tải nhãn ---
        self.emotion_labels = np.load(label_classes_path, allow_pickle=True)

    def _extract_features(self, audio):
        try:
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            rmse = np.mean(librosa.feature.rms(y=audio, frame_length=self.frame_length_fft, hop_length=self.hop_length_fft))
            mfccs = librosa.feature.mfcc(y=audio, sr=self.target_sr, n_mfcc=self.n_mfcc,
                                         n_fft=self.frame_length_fft, hop_length=self.hop_length_fft)
            # Pad MFCC
            mfccs = self._pad_truncate_mfcc_time_axis(mfccs, self.fixed_frames)
            features = np.hstack([zcr, rmse, mfccs.flatten()])

            if features.shape[0] != 1031:
                print("⚠️ Số đặc trưng không khớp 1031.")
                return None
            return features
        except Exception as e:
            print(f"Lỗi khi trích xuất đặc trưng: {e}")
            return None

    def _pad_truncate_mfcc_time_axis(self, mfccs, fixed_frames):
        current_frames = mfccs.shape[1]
        if current_frames > fixed_frames:
            return mfccs[:, :fixed_frames]
        elif current_frames < fixed_frames:
            pad = fixed_frames - current_frames
            return np.pad(mfccs, ((0, 0), (0, pad)), mode='constant')
        return mfccs

    def record_audio(self, filename="temp_recording.wav", duration=None):
        if duration is None:
            duration = self.duration_sec
        print(f"🎙️ Đang ghi âm trong {duration} giây...")
        recording = sd.rec(int(duration * self.target_sr), samplerate=self.target_sr, channels=1, dtype='float32')
        sd.wait()
        from scipy.io.wavfile import write
        write(filename, self.target_sr, recording)
        print(f"✅ Ghi xong: {filename}")
        return recording.flatten()

    def extract_mel_spectrogram(self, wav_path):
        # Không còn dùng trong model này → có thể bỏ
        pass

    def predict_emotion(self, audio):
        features = self._extract_features(audio)
        if features is None:
            return {"error": "feature_extraction_failed"}
        scaled = self.scaler.transform(features.reshape(1, -1))
        model_input = scaled.reshape(1, 1031, 1)
        prediction = self.model.predict(model_input)[0]
        result = {self.emotion_labels[i]: float(prediction[i]) for i in range(len(prediction))}
        return result
