import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Page config
st.set_page_config(page_title="Draw a Digit (0–9)", layout="centered")
st.title("🎨 Draw a Digit (0–9)")
st.markdown("Draw a digit, and the model will predict what you drew!")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Process and predict
if canvas_result.image_data is not None:
    img = canvas_result.image_data
    img = Image.fromarray(np.uint8(img)).convert("L")  # grayscale
    img = ImageOps.invert(img)  # white digit on black background
    img = img.resize((28, 28))
    img = np.array(img).reshape(1, 28, 28, 1) / 255.0  # normalize

    if np.sum(img) > 10:  # make sure something was drawn
        pred = model.predict(img)
        predicted_class = np.argmax(pred)
        confidence = np.max(pred) * 100

        st.subheader("📊 Prediction")
        st.write(f"**Digit:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")
    else:
        st.warning("🖌 Draw a digit to get prediction!")
