import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Veri seti yolları
train_dir = 'dataset_new/train'
test_dir = 'dataset_new/test'

# Görüntü boyutları ve batch boyutu
IMG_HEIGHT = 24
IMG_WIDTH = 24
BATCH_SIZE = 32

# Veri artırma ve yükleme (train ve test için)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim veri setini yükleme (sadece Open ve Closed sınıflarını al)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['Closed', 'Open']  # Sadece Closed ve Open sınıflarını al
)

# Test veri setini yükleme (sadece Open ve Closed sınıflarını al)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['Closed', 'Open']  # Sadece Closed ve Open sınıflarını al
)

# CNN modelini oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Modeli derleme
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model özetini yazdırma
model.summary()

# Modeli eğitme
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Modeli kaydetme
model.save('models/cnncat2.h5')

# Eğitim sonuçlarını yazdırma
print("Eğitim tamamlandı. Model 'models/cnncat2.h5' olarak kaydedildi.")