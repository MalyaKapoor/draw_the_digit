# app.py
import streamlit as st
import numpy as np
import pickle
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("✍️ Draw a Digit - Digit Recognizer")
st.markdown("Draw a digit (0–9) in the box below and click **Predict** to see the result.")

# Create a canvas for drawing digits
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Prediction button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data

        # Preprocess the image
        img = Image.fromarray((255 - img[:, :, 0]).astype(np.uint8))  # Convert to grayscale
        img = ImageOps.fit(img, (8, 8), method=0, bleed=0.0, centering=(0.5, 0.5))
        img = np.array(img).astype(np.float64)
        img = img.reshape(1, -1)

        # Scale to match digits dataset scale (0–16)
        img = (img / 255.0) * 16

        # Predict
        prediction = model.predict(img)[0]
        st.success(f"🔢 Predicted Digit: **{prediction}**")
    else:
        st.warning("Please draw a digit first.")

