import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Veri seti yolları
train_dir = 'dataset_new/train'
test_dir = 'dataset_new/test'
classes = ['Closed', 'Open', 'no_yawn', 'yawn']

# Veri ve etiket listeleri
X = []
y = []

# Eğitim verisini yükleme
for label, class_name in enumerate(classes):
    class_path = os.path.join(train_dir, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (24, 24))
        img = img / 255.0  # Normalizasyon
        X.append(img)
        y.append(label)

# Test verisini yükleme
for label, class_name in enumerate(classes):
    class_path = os.path.join(test_dir, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (24, 24))
        img = img / 255.0  # Normalizasyon
        X.append(img)
        y.append(label)

# Veriyi numpy array'e çevirme
X = np.array(X)
y = np.array(y)

# Veriyi yeniden şekillendirme (model için uygun hale getirme)
X = X.reshape(-1, 24, 24, 1)

# Etiketleri kategorik hale getirme
y = to_categorical(y, num_classes=4)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN Modeli oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 sınıf: Closed, Open, no_yawn, yawn
])

# Modeli derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Modeli kaydetme
model.save('models/drowsiness_model.h5')

print("Model başarıyla eğitildi ve 'models/drowsiness_model.h5' olarak kaydedildi.")