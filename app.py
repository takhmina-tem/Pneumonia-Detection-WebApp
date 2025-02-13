import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import requests
import os

# Set model path
MODEL_PATH = "pneumonia_detection.h5"

# ✅ Cache model loading to improve performance
@st.cache_resource
def load_model():
    # Check if model already exists locally
    if not os.path.exists(MODEL_PATH):
        # ✅ Download the model from Google Drive
        model_url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
        response = requests.get(model_url)
        
        # Save the downloaded model
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

    # ✅ Load the model from local file
    return tf.keras.models.load_model(MODEL_PATH)

# Load the model
model = load_model()

# Function to preprocess the uploaded X-ray image
def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (150, 150))  # Resize to match CNN input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model
    return img

# Streamlit Web App UI
st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload a Chest X-ray image to check for pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded X-ray
    st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess and Predict
    img = preprocess_image(uploaded_file)
    prediction = model.predict(img)[0][0]

    # Display results
    st.subheader("Prediction Result:")
    if prediction > 0.5:
        st.error("⚠️ Pneumonia Detected!")
        st.write(f"Confidence Score: {prediction * 100:.2f}%")
    else:
        st.success("✅ Normal X-ray")
        st.write(f"Confidence Score: {(1 - prediction) * 100:.2f}%")
