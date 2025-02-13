import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import requests
import os

# ✅ Model Path & Google Drive Link
MODEL_PATH = "pneumonia_detection.keras"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1-4x5GUclvR_viuTh0x8QBegtgTvMjQh7"  # Replace with actual Google Drive File ID

@st.cache_resource
def load_model():
    try:
        # ✅ Check if the model exists locally
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:  # Less than 1MB means corrupt
            st.write("📥 Downloading model from Google Drive...")
            response = requests.get(MODEL_URL, stream=True)

            # ✅ If response is OK, save the file
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.write("✅ Model downloaded successfully.")
            else:
                st.error("❌ Failed to download model. Check Google Drive link.")
                return None

        # ✅ Verify if the model file exists after download
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
            st.write("📂 Loading model...")
            return tf.keras.models.load_model(MODEL_PATH)
        else:
            st.error("❌ Model file is missing or corrupted after download.")
            return None

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ✅ Load model
model = load_model()
