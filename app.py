import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import requests
import os

# ✅ Force TensorFlow to Use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ✅ Model Path & Hugging Face Link
MODEL_PATH = "pneumonia_detection.keras"
MODEL_URL = "https://huggingface.co/takhminatem/pneumonia-detection/resolve/main/pneumonia_detection.keras"

@st.cache_resource
def load_model():
    try:
        # ✅ Check if model exists locally
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

        # ✅ Load the model WITHOUT the optimizer (fixes optimizer issue)
        st.write("📂 Loading model... (This may take a minute)")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.write("✅ Model loaded successfully!")
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ✅ Load model
model = load_model()

# 🎨 Streamlit UI
st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload a Chest X-ray image to check for pneumonia.")

# 📤 File Upload Section
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 🖼️ Read and Decode Image
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # 📸 Display Uploaded Image
    st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)  # Fixed deprecated warning

    # 🔄 Preprocess Image for Model
    image_resized = cv2.resize(image, (150, 150))  # Resize to model input size
    image_array = np.expand_dims(image_resized, axis=0) / 255.0  # Normalize

    # 🤖 Make Prediction
    if model:
        prediction = model.predict(image_array)[0][0]
        result = "🚨 Pneumonia Detected!" if prediction > 0.5 else "✅ Normal X-ray"
        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

        # 📊 Display Result
        st.subheader(result)
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.error("❌ Model could not be loaded. Check logs.")

