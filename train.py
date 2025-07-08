import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# ----------------------
# Set parameters
# ----------------------
img_size = 224
batch_size = 32
base_dir = "D:/CAT_VS_DOG"

# ----------------------
# Data Preprocessing
# ----------------------
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                                   zoom_range=0.2, shear_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, "train"),
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(base_dir, "val"),
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary"
)

# ----------------------
# Build Model
# ----------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze weights

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# ----------------------
# Compile Model
# ----------------------
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

from PIL import Image, UnidentifiedImageError
import os

def clean_broken_images(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()  # Check image integrity
            except (IOError, SyntaxError, UnidentifiedImageError):
                print(f"❌ Deleted corrupted file: {path}")
                os.remove(path)
                count += 1
    print(f"\n✅ Cleaning completed. Total corrupted images deleted: {count}")

# ✅ Replace with your exact folder paths
train_dir = "D:/CAT_VS_DOG/train"
val_dir   = "D:/CAT_VS_DOG/val"

# Run the cleaner
clean_broken_images(train_dir)
clean_broken_images(val_dir)
import os

def remove_tiff_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".tiff"):
                path = os.path.join(root, file)
                os.remove(path)
                print(f"Deleted TIFF image: {path}")

remove_tiff_images("D:/CAT_VS_DOG/train")
remove_tiff_images("D:/CAT_VS_DOG/val")




# ----------------------
# Train Model
# ----------------------
history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=5)

# ----------------------
# Save Model
# ----------------------
model.save("dog_cat_mobilenet_model.h5")
