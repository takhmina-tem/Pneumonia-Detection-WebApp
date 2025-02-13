import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import requests
import os
import time  # Import time for delay

# ✅ Force TensorFlow to Use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ✅ Model Path & Hugging Face Link
MODEL_PATH = "pneumonia_detection.keras"
MODEL_URL = "https://huggingface.co/takhminatem/pneumonia-detection/resolve/main/pneumonia_detection.keras"

@st.cache_resource
def load_model():
    try:
        # ✅ Check if model exists
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
            st.write("📥 Downloading model from Hugging Face...")
            response = requests.get(MODEL_URL, stream=True)

            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.write("✅ Model downloaded successfully.")
            else:
                st.error("❌ Failed to download model. Check Hugging Face link.")
                return None

        # ✅ Delay before loading (helps Streamlit handle large files)
        time.sleep(5)

        # ✅ Load the model WITHOUT the optimizer
        st.write("📂 Loading model... (This may take a minute)")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # Fix optimizer issue
        st.write("✅ Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ✅ Load model
model = load_model()
