import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import plotly.express as px
from PIL import Image
from huggingface_hub import hf_hub_download

# -----------------------------
# Page & Theme
# -----------------------------
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐶🐱",
    layout="wide",
)

st.title("🐶🐱 Dog vs Cat Image Classifier")
st.markdown("Upload gambar, model akan mengklasifikasikan apakah gambar berisi **Dog** atau **Cat**.")

# -----------------------------
# Hugging Face Model Setup
# -----------------------------
# Ganti repo_id & filename sesuai model kamu di HF
REPO_ID = "Syahhh01/CatVSDog"       # contoh repo Hugging Face
MODEL_FILE = "Dog_vs_cat_model.h5"       # nama file di repo HF

cache_dir = os.path.join(os.getcwd(), "models")
with st.spinner("Downloading model from Hugging Face..."):
    MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE,cache_dir = cache_dir)

IMG_SIZE = (224, 224)
PROB_DECIMALS = 3

LABEL_MAP = {1: "Cat", 0: "Dog"}
CLASS_COLORS = {"Cat": "#636EFA", "Dog": "#EF553B"}

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_my_model(path):
    model = load_model(path)
    return model

try:
    model = load_my_model(MODEL_PATH)
except Exception as e:
    st.error("Gagal memuat model dari Hugging Face.")
    st.exception(e)
    st.stop()

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader("Upload gambar (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img_resized = img.resize(IMG_SIZE)
    x = image.img_to_array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)

    # Prediksi
    probs = model.predict(x)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = LABEL_MAP[pred_idx]
    top_prob = float(probs[pred_idx])

    # Badge
    badge_color = CLASS_COLORS.get(pred_label, "#1f77b4")
    st.markdown(
        f"<div style='padding:10px;border-radius:12px;background:{badge_color};color:white;display:inline-block;'>"
        f"Prediksi: <b>{pred_label}</b> • Prob: {top_prob:.{PROB_DECIMALS}f}"
        f"</div>", unsafe_allow_html=True
    )

    # Bar chart probabilitas
    df_probs = {
        "Class": [LABEL_MAP[i] for i in range(len(probs))],
        "Probability": probs
    }
    fig = px.bar(
        df_probs, 
        x="Class", 
        y="Probability", 
        text=[f"{p:.{PROB_DECIMALS}f}" for p in probs],
        title="Class Probabilities", 
        range_y=[0, 1]
    )
    fig.update_traces(
        marker_color=[CLASS_COLORS.get(c, None) for c in df_probs["Class"]],
        textposition="outside"
    )
    st.plotly_chart(fig, use_container_width=True)





