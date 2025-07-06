import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load model
model = load_model("mask_detector.model")

st.title("Face Mask Detection App")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Predict
    prediction = model.predict(img)[0]
    label = "Mask" if prediction[0] > prediction[1] else "No Mask"
    confidence = max(prediction) * 100

    st.markdown(f"### Prediction: **{label}** ({confidence:.2f}% confidence)")
