# ----------------------------------------
# Force CPU mode to avoid silent GPU crash
# ----------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------------------
# Page configuration
# ----------------------------------------
st.set_page_config(
    page_title="Aircraft Classifier",
    page_icon="✈️",
    layout="centered"
)

# ----------------------------------------
# Simple clean styling
# ----------------------------------------
st.markdown("""
<style>
/* Overall spacing */
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 900px;}

/* Title */
h1, h2, h3 { letter-spacing: 0.3px; }

/* "Card" feel */
.card {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 18px 18px;
    background: rgba(255,255,255,0.03);
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}

/* Metric styling */
.metric {
    border-radius: 14px;
    padding: 12px 14px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
}

/* Smaller caption */
.small {
    opacity: 0.75;
    font-size: 0.92rem;
}

/* Upload box nicer spacing */
[data-testid="stFileUploader"] section {
    padding: 12px !important;
    border-radius: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# Header (hero)
# ----------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("## ✈️ Aircraft Image Classifier")
st.markdown(
    "<div class='small'>Upload an aircraft image. The model will predict the class index and confidence.</div>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# ----------------------------------------
# Sidebar
# ----------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    target_size = st.selectbox("Input image size", [(64, 64)], index=0)  # keep your original setting
    show_top5 = st.toggle("Show Top-5 probabilities", value=True)
    st.markdown("---")
    st.markdown("**Model file:** `best_aircraft_model.tflite`")
    st.markdown("**Mode:** CPU-only")

# ----------------------------------------
# Load TFLite model (cached)
# ----------------------------------------
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="best_aircraft_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_tflite_model()
except Exception as e:
    st.error(f"❌ Failed to load TFLite model: {e}")
    st.stop()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.success("✅ TFLite model loaded successfully")

# ----------------------------------------
# Image preprocessing
# ----------------------------------------
def preprocess_image(image, target_size=(64, 64)):
    image = image.resize(target_size)
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ----------------------------------------
# Upload section
# ----------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📤 Upload an aircraft image")
uploaded_file = st.file_uploader(
    "Choose a JPG/PNG file",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)
st.write("")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Layout: image left, results right
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🖼️ Preview")
        st.image(image, use_container_width=True)
        st.markdown(
            f"<div class='small'>Converted to RGB • Resized to {target_size[0]}×{target_size[1]}</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Prediction")

        with st.spinner("Running inference..."):
            input_data = preprocess_image(image, target_size=target_size)

            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]["index"])[0]  # shape: (num_classes,)

            predicted_class = int(np.argmax(output_data))
            confidence = float(np.max(output_data))

        st.markdown('<div class="metric">', unsafe_allow_html=True)
        st.markdown(f"**Predicted Class Index:** `{predicted_class}`")
        st.markdown(f"**Confidence:** `{confidence:.4f}`")
        st.progress(min(max(confidence, 0.0), 1.0))
        st.markdown("</div>", unsafe_allow_html=True)

        if show_top5:
            st.markdown("#### Top-5 probabilities")
            topk = np.argsort(output_data)[::-1][:5]
            for rank, idx in enumerate(topk, start=1):
                st.write(f"{rank}. Class `{int(idx)}` — `{float(output_data[idx]):.4f}`")

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.info("Tip: If your Part 1 model was trained with a different input size (e.g. 224×224), update the `target_size` to match.")
