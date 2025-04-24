import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# === Custom layers đã khai báo như trong mô hình gốc ===

from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention

class PositionalEncoding(Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = int(position)
        self.d_model = int(d_model)
        self.pos_encoding = self.positional_encoding(self.position, self.d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class TransformerEncoderBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))


class VoiceAnalyzer:
    def __init__(self, model_path="best_cnn_transformer_model.keras"):
        self.model = load_model(model_path, custom_objects={
            "PositionalEncoding": PositionalEncoding,
            "TransformerEncoderBlock": TransformerEncoderBlock
        })
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        print("✅ Đã load mô hình.")

    def extract_mel_spectrogram(self, file_path, sr=16000, n_mels=128):
        y, sr = librosa.load(file_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = log_mel[..., np.newaxis]  # (n_mels, time, 1)
        return log_mel

    def predict_emotion(self, mel):
        mel = tf.image.resize(mel, (128, 126)).numpy()  # Adjust width to 126 as expected by the model
        mel = np.expand_dims(mel, axis=0)  # (1, 128, 126, 1)
        preds = self.model.predict(mel)[0]  # Get the first (and only) batch of predictions
        probabilities = {self.emotion_labels[i]: float(preds[i]) for i in range(len(self.emotion_labels))}
        return probabilities

    def record_audio(self, filename="temp_recording.wav", duration=3, fs=16000):
        """Ghi âm từ microphone và lưu vào tệp."""
        print(f"🎙️ Đang ghi âm trong {duration} giây...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        wav.write(filename, fs, recording)
        print(f"✅ Ghi âm xong, lưu vào {filename}")
