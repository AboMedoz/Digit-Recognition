import os

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

from utilities.helpers import preprocess_img

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_PATH = os.path.join(ROOT, 'models')

st.set_page_config(page_title="Digit Recognition")

model = load_model(os.path.join(MODELS_PATH, 'digit_recognition.keras'))

st.title("MNIST Digit Recognition")
st.write("Draw a digit below and click **Predict**.")

canvas = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas.image_data is None:
        st.warning("Draw a digit first.")
        st.stop()

    img = preprocess_img(canvas.image_data)

    if img is None:
        st.warning("Draw a digit first.")
        st.stop()

    prediction = model.predict(
        np.expand_dims(img, 0),
        verbose=0
    )[0]

    digit = np.argmax(prediction)
    confidence = prediction[digit]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Processed Image")
        st.image(img.squeeze(), clamp=True)

    with col2:
        st.subheader("Prediction")
        st.metric("Digit", digit)
        st.metric("Confidence", f"{confidence:.2%}")

    st.subheader("Probabilities")

    probs = {
        str(i): float(prediction[i])
        for i in range(10)
    }

    st.bar_chart(probs)