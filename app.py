import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image, ImageOps

# Load the model
model = pickle.load(open("model.pkl", "rb"))

st.title("🖌️ Draw the Digit - Digit Recognizer")

# Upload or draw
canvas_result = st.canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'), 'L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img_array = np.array(img).reshape(1, -1) / 255.0
    prediction = model.predict(img_array)
    st.subheader(f"Prediction: {prediction[0]}")
