import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import requests
import os

st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
    <style>
        body { background-color: #f8f9fa; }
        .main-title { font-size: 36px; font-weight: bold; text-align: center; color: #343a40; margin-bottom: 10px; }
        .sub-text { text-align: center; font-size: 18px; color: #6c757d; margin-bottom: 40px; }
        .upload-box { background-color: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
        .result-box { padding: 20px; border-radius: 8px; text-align: center; font-size: 24px; font-weight: bold; }
        .positive { background-color: #f8d7da; color: #721c24; }
        .negative { background-color: #d4edda; color: #155724; }
        .sidebar .sidebar-content { background-color: #f8f9fa; }
    </style>
""", unsafe_allow_html=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL_PATH = "pneumonia_detection.keras"
MODEL_URL = "https://huggingface.co/takhminatem/pneumonia-detection/resolve/main/pneumonia_detection.keras"

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
            st.info("📥 Downloading model from cloud storage...")
            response = requests.get(MODEL_URL, stream=True)

            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ Model downloaded successfully.")
            else:
                st.error("❌ Model download failed. Please try again later.")
                return None

        st.info("📂 Loading AI model... Please wait.")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("✅ AI model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

st.markdown('<h1 class="main-title">Pneumonia Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Upload a Chest X-ray image for automated analysis.</p>', unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload X-ray image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # 🖼️ Read and Decode Image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        with col2:
            st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)

        image_resized = cv2.resize(image, (150, 150))
        image_array = np.expand_dims(image_resized, axis=0) / 255.0

        if model:
            prediction = model.predict(image_array)[0][0]
            result = "Pneumonia Detected" if prediction > 0.5 else "Normal X-ray"
            confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

            result_style = "positive" if prediction > 0.5 else "negative"
            st.markdown(f'<div class="result-box {result_style}">{result}<br>Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
        else:
            st.error("❌ AI model could not process this image. Try again.")
