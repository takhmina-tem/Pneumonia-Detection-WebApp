import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import requests
import os

# ✅ Model Path & Google Drive Link
MODEL_PATH = "pneumonia_detection.keras"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1-4x5GUclvR_viuTh0x8QBegtgTvMjQh7"  # Replace with actual File ID

@st.cache_resource
def load_model():
    try:
        # ✅ Check if the model exists and has a valid size
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:  # Less than 1MB means corrupt
            st.write("📥 Downloading model from Google Drive...")
            response = requests.get(MODEL_URL, stream=True)

            # ✅ Save the model file
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.write("✅ Model downloaded successfully.")
            else:
                st.error("❌ Failed to download model. Check Google Drive link.")
                return None

        # ✅ Check file size after download
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Convert bytes to MB
        st.write(f"📂 Model file size: {file_size:.2f} MB")

        # ✅ If file size is too small, delete and retry
        if file_size < 10:  # If less than 10MB, assume corrupt
            os.remove(MODEL_PATH)
            st.error("❌ Model file is too small and possibly corrupted. Please check your Google Drive file.")
            return None

        # ✅ Load the model if size is correct
        st.write("📂 Loading model...")
        return tf.keras.models.load_model(MODEL_PATH)

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ✅ Load model
model = load_model()
