import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# Load Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model(
    os.path.join(BASE_DIR, "cnn_model.keras")
)

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="✍️"
)

st.title("✍️ Handwritten Digit Recognition using CNN")

uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=250
    )

    # Convert to grayscale
    image = image.convert("L")

    # Resize
    image = image.resize((28,28))

    # Convert to array
    img_array = np.array(image)

    # Invert colors
    img_array = 255 - img_array

    # Normalize
    img_array = img_array / 255.0

    # Reshape
    img_array = img_array.reshape(
        1,28,28,1
    )

    prediction = model.predict(img_array)

    digit = np.argmax(prediction)

    confidence = np.max(prediction)

    st.success(
        f"Predicted Digit: {digit}"
    )

    st.write(
        f"Confidence: {confidence:.2%}"
    )