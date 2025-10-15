import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
from PIL import Image

# === Load model once ===
@st.cache_resource
def load_model_and_classes():
    model = load_model("models/DenseNet201_model80ft.h5")
    class_names = sorted(os.listdir('new_dataset/80-20/train'))
    return model, class_names

model, class_names = load_model_and_classes()

# === Image preprocessing ===
def preprocess_image(img, target_size=(250, 250)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# === Streamlit UI ===
st.title("🐬 Sea Animal Classifier - DenseNet201")

uploaded_file = st.file_uploader("Upload an image of a sea animal", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(img)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_class = class_names[predicted_index]

    st.success(f"✅ Predicted Class: **{predicted_class}**")
