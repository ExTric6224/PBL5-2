import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization # Thêm BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Tắt bớt các thông báo log của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Xử lý tham số dòng lệnh ---
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, help="Chế độ hoạt động: train/display")
args = ap.parse_args()
mode = args.mode

# --- Hàm vẽ đồ thị lịch sử huấn luyện ---
def plot_model_history(model_history):
    """ Vẽ đồ thị độ chính xác và mất mát từ đối tượng history """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # Đồ thị Accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')
    # Đồ thị Loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')
    fig.tight_layout()
    plt.show()

# --- Định nghĩa các tham số và đường dẫn ---
# Giả sử chạy script từ thư mục gốc DuanMoi, hoặc điều chỉnh đường dẫn tương đối nếu chạy từ src/
train_dir = 'data/train'
val_dir = 'data/test'

# QUAN TRỌNG: Xác định số lớp cảm xúc dựa trên số thư mục con trong train_dir
# Ví dụ: num_classes = len(os.listdir(train_dir))
# Ở đây tạm giả định là 7 lớp phổ biến (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
# *** Bạn cần kiểm tra lại thư mục data/train của mình có bao nhiêu thư mục con ***
try:
    num_classes = len(os.listdir(train_dir))
    print(f"Detected {num_classes} classes based on folders in {train_dir}")
except FileNotFoundError:
    print(f"Error: Training directory not found at {train_dir}")
    print("Please make sure you have moved the 'train' folder to 'DuanMoi/data/train'")
    exit() # Thoát nếu không tìm thấy thư mục train

img_rows, img_cols = 48, 48 # Kích thước ảnh đầu vào mô hình
batch_size = 64           # Số lượng ảnh xử lý trong một lô (có thể chỉnh 32, 64, 128)
epochs = 50               # Số lượt huấn luyện trên toàn bộ dữ liệu (có thể chỉnh 50, 100, ...)

# --- Bộ tạo dữ liệu ảnh (Data Generators) ---
# Sử dụng tăng cường dữ liệu (augmentation) cho tập huấn luyện để mô hình tổng quát tốt hơn
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Chuẩn hóa pixel về khoảng [0, 1]
    rotation_range=30,        # Xoay ảnh ngẫu nhiên
    shear_range=0.3,          # Biến dạng cắt ảnh
    zoom_range=0.3,           # Phóng to/thu nhỏ ảnh
    horizontal_flip=True,     # Lật ảnh ngang ngẫu nhiên
    fill_mode='nearest'       # Cách lấp đầy pixel mới khi biến đổi ảnh
)

# Không tăng cường dữ liệu cho tập validation/test
validation_datagen = ImageDataGenerator(rescale=1./255)

# Tạo generator cho tập huấn luyện
train_generator = train_datagen.flow_from_directory(
    train_dir,
    color_mode='grayscale',       # Sử dụng ảnh thang xám (1 kênh màu)
    target_size=(img_rows, img_cols), # Đưa ảnh về kích thước 48x48
    batch_size=batch_size,
    class_mode='categorical',     # Phân loại đa lớp (các cảm xúc)
    shuffle=True                  # Xáo trộn dữ liệu huấn luyện
)

# Tạo generator cho tập kiểm thử (validation)
validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    color_mode='grayscale',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False                 # Không cần xáo trộn dữ liệu kiểm thử
)

# --- Xây dựng Mô hình CNN ---
model = Sequential(name="Emotion_CNN") # Đặt tên cho mô hình

# Input shape: (chiều cao, chiều rộng, số kênh màu) - ảnh xám là 1 kênh
input_shape = (img_rows, img_cols, 1)

# Block 1
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu')) # Có thể thêm lớp conv nữa nếu muốn
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 3
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
# model.add(Conv2D(256, kernel_size=(3, 3), activation='relu')) # Có thể thêm lớp conv nữa
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Lớp Flatten và Fully Connected (Dense)
model.add(Flatten())
model.add(Dense(256, activation='relu')) # Lớp ẩn
model.add(BatchNormalization())
model.add(Dropout(0.5)) # Dropout mạnh hơn trước lớp output
model.add(Dense(num_classes, activation='softmax')) # Lớp Output với softmax cho phân loại đa lớp

# In cấu trúc mô hình ra màn hình
print(model.summary())

# --- Biên dịch Mô hình ---
model.compile(loss='categorical_crossentropy', # Hàm mất mát cho phân loại đa lớp
              optimizer=Adam(learning_rate=0.0001), # Trình tối ưu hóa Adam với learning rate nhỏ
              metrics=['accuracy']) # Theo dõi độ chính xác

# --- Huấn luyện Mô hình (Chỉ khi mode là 'train') ---
if mode == "train":
    print(f"Starting training for {epochs} epochs...")

    # Tính toán số bước cho mỗi epoch
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size

    if steps_per_epoch == 0:
        print("Warning: steps_per_epoch is zero. Check batch_size and number of training images.")
    if validation_steps == 0:
         print("Warning: validation_steps is zero. Check batch_size and number of validation images.")

    # Bắt đầu huấn luyện
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch if steps_per_epoch > 0 else 1, # Đảm bảo không bị lỗi chia cho 0
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps if validation_steps > 0 else 1 # Đảm bảo không bị lỗi chia cho 0
    )

    # Lưu mô hình đã huấn luyện
   # model_save_path = ' model.h5'
    model_save_path = 'emotion_model.h5' # Lưu ở thư mục gốc dự án (DuanMoi)
    model.save(model_save_path)
    print(f"Training finished. Model saved as {model_save_path}")

    # Vẽ đồ thị kết quả huấn luyện
    plot_model_history(history)

# --- Chế độ Hiển thị/Dự đoán (Cần được cài đặt thêm) ---
# (Các phần import, argparse, plot_model_history, định nghĩa model, compile ... giữ nguyên như trước)

# --- Chế độ Hiển thị/Dự đoán ---
elif mode == "display":
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    import numpy as np
    import cv2

    # --- Tải mô hình và bộ phát hiện khuôn mặt ---
    try:
        # Đường dẫn đến mô hình đã lưu (lưu ở thư mục gốc DuanMoi)
        model_path = 'emotion_model.h5'
        #model_path = 'model.h5'
        emotion_model = load_model(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Please ensure the model file exists and training was successful.")
        exit()

    try:
        # Đường dẫn đến tệp Haar Cascade (giả sử nằm trong thư mục src/ hoặc gốc DuanMoi)
        # Hãy đảm bảo bạn có tệp này và đường dẫn là đúng!
        face_cascade_path = 'src/haarcascade_frontalface_default.xml' # Hoặc 'haarcascade_frontalface_default.xml' nếu ở gốc
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if face_cascade.empty():
            raise IOError(f"Could not load Haar cascade classifier from {face_cascade_path}")
        print(f"Loaded face cascade from {face_cascade_path}")
    except Exception as e:
        print(f"Error loading Haar cascade: {e}")
        exit()

    # --- Định nghĩa nhãn cảm xúc ---
    # QUAN TRỌNG: Thứ tự phải khớp với thứ tự thư mục con trong data/train (thường theo alphabet)
    # Hãy kiểm tra lại tên thư mục con của bạn!
    emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
    num_classes = len(emotion_labels) # Nên là 7 nếu khớp với training

    # Kích thước ảnh đầu vào cho mô hình
    img_rows, img_cols = 48, 48

    # --- Khởi động Webcam ---
    cap = cv2.VideoCapture(0) # 0 là webcam mặc định

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Starting real-time emotion detection... Press 'q' to quit.")

    # --- Vòng lặp xử lý video ---
    while True:
        ret, frame = cap.read() # Đọc một khung hình
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Lật ảnh ngang (tùy chọn, cho giống soi gương)
        # frame = cv2.flip(frame, 1)

        # Chuyển sang ảnh xám để phát hiện khuôn mặt
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1, # Giảm kích thước ảnh ở mỗi lần quét
            minNeighbors=5,  # Số lượng hàng xóm tối thiểu để xác nhận là mặt
            minSize=(30, 30), # Kích thước khuôn mặt tối thiểu
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Xử lý từng khuôn mặt phát hiện được
        for (x, y, w, h) in faces:
            # Vẽ khung bao quanh khuôn mặt trên ảnh màu gốc
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Màu xanh lá, độ dày 2

            # Cắt vùng khuôn mặt ảnh xám (ROI)
            roi_gray = gray_frame[y:y + h, x:x + w]

            # Thay đổi kích thước ROI về 48x48
            roi_gray_resized = cv2.resize(roi_gray, (img_rows, img_cols), interpolation=cv2.INTER_AREA)

            # Chuẩn hóa và định dạng lại ảnh để đưa vào mô hình
            roi = roi_gray_resized.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0) # Thêm chiều batch (1, 48, 48, 1)

            # Dự đoán cảm xúc
            prediction = emotion_model.predict(roi, verbose=0) # verbose=   0 để không in log predict

            # Lấy nhãn cảm xúc có xác suất cao nhất
            emotion_probability = np.max(prediction)
            emotion_label_arg = np.argmax(prediction)
            predicted_emotion = emotion_labels[emotion_label_arg]

            # Hiển thị nhãn cảm xúc lên khung hình
            label_position = (x, y - 10) # Vị trí hiển thị text (phía trên khung)
            cv2.putText(frame, predicted_emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Hiển thị khung hình kết quả
        cv2.imshow('Emotion Detection', frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Giải phóng tài nguyên ---
    print("Exiting...")
    cap.release()
    cv2.destroyAllWindows()

# --- Các trường hợp mode khác ---
# (else và phần báo lỗi mode không hợp lệ giữ nguyên)
else:
    print("Lỗi: Chế độ không hợp lệ.")
    print("Vui lòng sử dụng: --mode train hoặc --mode display")