import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model (IMPORTANT: compile=False for compatibility)
model = tf.keras.models.load_model("animal_species_model.keras")


# Class names (same order as training)
class_names = [
    'cane', 'cavallo', 'elefante', 'farfalla', 'gallina',
    'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo'
]

st.title("🐾 Animal Species Classification")
st.write("Upload an animal image and the model will predict the species.")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img)
    cls = class_names[np.argmax(pred)]
    conf = np.max(pred) * 100

    st.success(f"Predicted Species: {cls}")
    st.info(f"Confidence: {conf:.2f}%")

