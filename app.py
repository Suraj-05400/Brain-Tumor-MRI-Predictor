"""Streamlit app: upload a brain MRI image -> predicted tumor type + confidence scores.
Run:  streamlit run app.py      (needs models/*.h5 and models/class_names.json from src/train.py)"""
import glob
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras  # type: ignore[import-not-found]

import src.models  # noqa: F401  (registers the custom augmentation layer needed to load the .h5 files)

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")
CONF_THRESHOLD = 0.60


@st.cache_resource
def load_model(path):
    return keras.models.load_model(path, compile=False)


st.title("🧠 Brain Tumor MRI Classifier")
st.caption("Educational decision-support demo. NOT a medical device and not for diagnosis.")

files = sorted(glob.glob("models/*.h5") + glob.glob("models/*.keras"))
if not files or not os.path.exists("models/class_names.json"):
    st.error("No trained model found. Run `python -m src.train --data_dir <dataset>` first.")
    st.stop()
classes = json.load(open("models/class_names.json"))

default = next((i for i, f in enumerate(files) if os.path.basename(f).startswith("best_model")), 0)
choice = st.sidebar.selectbox("Model", files, index=default, format_func=os.path.basename)
model = load_model(choice)
size = model.input_shape[1]
st.sidebar.write(f"Input size: {size}x{size}  \nClasses: {', '.join(classes)}")

uploads = st.file_uploader("Upload MRI image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
for f in uploads or []:
    img = Image.open(f).convert("RGB")
    x = np.expand_dims(np.asarray(img.resize((size, size)), dtype="float32"), 0)   # raw 0-255; scaling is inside the model
    probs = model.predict(x, verbose=0)[0]
    top = int(np.argmax(probs))
    col1, col2 = st.columns(2)
    col1.image(img, caption=f.name, use_container_width=True)
    col2.metric("Predicted tumor type", classes[top], f"{probs[top]:.1%} confidence")
    col2.bar_chart(pd.DataFrame({"probability": probs}, index=classes))
    if probs[top] < CONF_THRESHOLD:
        st.warning("Low confidence - please have a specialist review this scan.")
    st.divider()
