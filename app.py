import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import requests
import os

# ✅ Model Settings
MODEL_PATH = "pneumonia_detection.h5"  # You can rename this if needed
MODEL_URL = "https://drive.google.com/uc?export=download&id=1tPhsj5zb-lnOJX0Vk9kM4VMHg9rJF903"  # Your Google Drive link

# ✅ Function to Load the Model
@st.cache_resource
def load_model():
    try:
        # Check if model exists locally
        if not os.path.exists(MODEL_PATH):
            st.write("📥 Downloading model from Google Drive...")
            response = requests.get(MODEL_URL)

            # ✅ Check if download was successful
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    f.write(response.content)
                st.write("✅ Model downloaded successfully.")
            else:
                st.error("❌ Failed to download model. Check your Google Drive link.")
                return None

        # ✅ Load the model
        st.write("📂 Loading model...")
        return tf.keras.models.load_model(MODEL_PATH)

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load model
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

    # Ensure the model loaded correctly before making predictions
    if model:
        prediction = model.predict(img)[0][0]
        st.subheader("Prediction Result:")
        if prediction > 0.5:
            st.error("⚠️ Pneumonia Detected!")
            st.write(f"Confidence Score: {prediction * 100:.2f}%")
        else:
            st.success("✅ Normal X-ray")
            st.write(f"Confidence Score: {(1 - prediction) * 100:.2f}%")
    else:
        st.error("❌ Model could not be loaded. Please check logs.")
