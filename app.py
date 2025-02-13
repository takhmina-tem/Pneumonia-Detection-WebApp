import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

# 🎨 Streamlit UI
st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload a Chest X-ray image to check for pneumonia.")

# 📤 File Upload Section
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 🖼️ Display Uploaded Image
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    # 🔄 Preprocess Image for Model
    image_resized = cv2.resize(image, (150, 150))
    image_array = np.expand_dims(image_resized, axis=0) / 255.0

    # 🤖 Make Prediction
    model = tf.keras.models.load_model("pneumonia_detection.keras")  # Load model
    prediction = model.predict(image_array)[0][0]
    result = "🚨 Pneumonia Detected!" if prediction > 0.5 else "✅ Normal X-ray"
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

    # 📊 Display Result
    st.subheader(result)
    st.write(f"Confidence: {confidence:.2f}%")
