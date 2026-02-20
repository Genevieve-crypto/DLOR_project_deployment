# ----------------------------------------
# Streamlit Aircraft Classifier (TFLite)
# Aesthetic UI + Background + No TensorFlow
# ----------------------------------------
import os
import base64
import streamlit as st
import numpy as np
from PIL import Image

# Use tflite-runtime (recommended for Streamlit Cloud)
from tflite_runtime.interpreter import Interpreter

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Aircraft Classifier",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Background (image optional)
# Put bg.jpg in same folder as app.py, or it will fallback to gradient
# -----------------------------
def set_background():
    if os.path.exists("background.jpg"):
        with open("background.jpg", "rb") as f:
            data = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
              background:
                linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                url("data:image/jpg;base64,{data}") no-repeat center center fixed;
              background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp{
              background: linear-gradient(135deg,#0b1220 0%, #050814 55%, #0b1220 100%);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

set_background()

# -----------------------------
# Extra CSS (cards/buttons/text)
# -----------------------------
st.markdown(
    """
    <style>
    /* Make content area a centered card */
    .block-container{
        max-width: 820px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }
    .glass {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 18px;
        padding: 18px 18px 10px 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    .title{
        font-size: 2.0rem;
        font-weight: 800;
        color: rgba(255,255,255,0.96);
        margin-bottom: 0.2rem;
        letter-spacing: 0.2px;
    }
    .subtitle{
        color: rgba(255,255,255,0.75);
        font-size: 1.02rem;
        margin-bottom: 1rem;
    }
    .smallhint{
        color: rgba(255,255,255,0.65);
        font-size: 0.95rem;
    }
    /* Buttons */
    .stButton>button{
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.10);
        color: white;
        padding: 0.6rem 1rem;
        font-weight: 650;
    }
    .stButton>button:hover{
        background: rgba(255,255,255,0.16);
        border-color: rgba(255,255,255,0.26);
    }
    /* File uploader */
    section[data-testid="stFileUploader"]{
        background: rgba(255,255,255,0.06);
        border: 1px dashed rgba(255,255,255,0.18);
        padding: 14px;
        border-radius: 14px;
    }
    /* Metric */
    div[data-testid="stMetricValue"]{
        color: rgba(255,255,255,0.96) !important;
    }
    div[data-testid="stMetricLabel"]{
        color: rgba(255,255,255,0.70) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load labels (optional)
# labels.txt: one label per line
# -----------------------------
@st.cache_data
def load_labels(path="labels.txt"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        return labels
    return None

labels = load_labels()

# -----------------------------
# Load TFLite model (cached)
# -----------------------------
@st.cache_resource
def load_tflite_interpreter(model_path="best_aircraft_model.tflite"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Make sure it is in the same folder as app.py in your GitHub repo."
        )
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_tflite_interpreter()
except Exception as e:
    st.error(f"❌ Failed to load TFLite model.\n\n**Error:** {e}")
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# Helpers
# -----------------------------
def get_model_input_size():
    # Typical: (1, H, W, C)
    shape = input_details[0]["shape"]
    h, w = int(shape[1]), int(shape[2])
    return (w, h)

def preprocess_image(img: Image.Image):
    target_w, target_h = get_model_input_size()
    img = img.convert("RGB").resize((target_w, target_h))
    x = np.array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)  # (1,H,W,3)
    return x

def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-9)

def predict(img: Image.Image):
    x = preprocess_image(img)
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details[0]["index"])[0]  # (num_classes,)

    # Some models already output probabilities; some output logits.
    # We'll safely normalize into probs.
    if np.min(y) < 0 or np.max(y) > 1.0:
        probs = softmax(y)
    else:
        s = float(np.sum(y))
        probs = y / s if s > 0 else softmax(y)

    pred_idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    return pred_idx, conf, probs

# -----------------------------
# UI
# -----------------------------
st.markdown(
    """
    <div class="glass">
      <div class="title">✈️ Aircraft Image Classifier</div>
      <div class="subtitle">
        Upload an aircraft image and the model will predict the class.
      </div>
      <div class="smallhint">
        Tip: Use a clear side view / high-resolution image for better accuracy.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")  # spacing

uploaded_file = st.file_uploader("📤 Upload an aircraft image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([1.2, 1])

with col2:
    st.markdown(
        """
        <div class="glass">
          <div style="color:rgba(255,255,255,0.92); font-weight:700; font-size:1.05rem;">Model Status</div>
          <div style="color:rgba(255,255,255,0.70); margin-top:6px;">
            ✅ TFLite model loaded<br/>
            📐 Input size auto-detected
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col1:
    st.markdown(
        """
        <div class="glass">
          <div style="color:rgba(255,255,255,0.92); font-weight:700; font-size:1.05rem;">Prediction</div>
          <div style="color:rgba(255,255,255,0.70); margin-top:6px;">
            Upload an image to see results.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running inference..."):
        pred_idx, conf, probs = predict(image)

    # Label name if available
    pred_name = labels[pred_idx] if labels and pred_idx < len(labels) else f"Class {pred_idx}"

    st.subheader("✅ Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Class", pred_name)
    c2.metric("Class Index", pred_idx)
    c3.metric("Confidence", f"{conf*100:.2f}%")

    # Top-5 table (nice for demo)
    st.write("")
    st.markdown("**Top-5 Predictions**")
    topk = min(5, len(probs))
    top_idxs = np.argsort(probs)[::-1][:topk]

    rows = []
    for i in top_idxs:
        name = labels[i] if labels and i < len(labels) else f"Class {int(i)}"
        rows.append((name, int(i), float(probs[i])))

    st.dataframe(
        {
            "Label": [r[0] for r in rows],
            "Index": [r[1] for r in rows],
            "Confidence": [f"{r[2]*100:.2f}%" for r in rows],
        },
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.markdown(
    "<div style='text-align:center; color:rgba(255,255,255,0.55); font-size:0.9rem;'>"
    "DLOR Project • Streamlit Deployment • TFLite Inference"
    "</div>",
    unsafe_allow_html=True,
)
