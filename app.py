import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

model = tf.keras.models.load_model(r"D:\CAT_VS_DOG\dog_cat_mobilenet_model.h5")


st.title("🐶🐱 Cat vs Dog Classifier")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).resize((224, 224))
    st.image(image, caption="Uploaded Image")

    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success(f"🐶 It's a Dog ({prediction:.2f})")
    else:
        st.success(f"🐱 It's a Cat ({1 - prediction:.2f})")
