import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load Haar cascade
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Streamlit title
st.title("🚘 Number Plate Detection App")

# File uploader
uploaded_file = st.file_uploader("Upload a car image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image as a PIL Image
    img = Image.open(uploaded_file).convert("RGB")  # Ensure it's RGB
    img_np = np.array(img)

    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Detect number plates
    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.01,
        minNeighbors=2,
        minSize=(20, 20)
    )

    # Draw rectangles
    for (x, y, w, h) in plates:
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Show result
    st.image(img_np, caption="Detected Number Plates", use_container_width=True)

    # Detection message
    if len(plates) == 0:
        st.warning("⚠️ No number plate detected in this image.")
else:
    st.info("Please upload an image to start.")
