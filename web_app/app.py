import base64
import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

MODEL_DIR = "./web_app/model"
SAMPLES_DIR = "./web_app/samples"

# Folder name in SAMPLES_DIR -> label as written in labels.txt
SAMPLE_CLASSES = {
    "monkeypox": "Monkeypox",
    "chickenpox_measles": "Chickenpox+Measles",
    "normal": "Normal",
}

CLASS_STYLES = {
    "Monkeypox": {"color": "#e5484d", "display": "Monkeypox"},
    "Chickenpox+Measles": {"color": "#f5a524", "display": "Chickenpox / Measles"},
    "Normal": {"color": "#30a46c", "display": "Normal skin"},
}

st.set_page_config(
    page_title="Enhancing Monkeypox Diagnosis",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; padding-bottom: 4rem; max-width: 820px; }

    .hero {
        background: linear-gradient(135deg, #0f3d3e 0%, #146c6d 55%, #1f9e8f 100%);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        color: #fff;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 32px rgba(15, 61, 62, 0.22);
    }
    .hero .eyebrow {
        text-transform: uppercase; letter-spacing: .14em; font-size: .72rem;
        font-weight: 600; opacity: .8; margin-bottom: .5rem;
    }
    .hero h1 { color: #fff; font-size: 1.95rem; line-height: 1.2; margin: 0 0 .6rem 0; padding: 0; }
    .hero p { margin: 0; opacity: .88; font-size: .98rem; line-height: 1.55; }
    .hero .chips { margin-top: 1.1rem; display: flex; gap: .5rem; flex-wrap: wrap; }
    .hero .chip {
        background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.22);
        border-radius: 999px; padding: .25rem .75rem; font-size: .78rem;
    }

    .section-hint { color: #6b7280; font-size: .9rem; margin: -.2rem 0 .9rem 0; }

    .gallery { display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem; }
    @media (max-width: 640px) { .gallery { grid-template-columns: repeat(2, 1fr); } }
    .gallery a {
        position: relative; display: block; border-radius: 14px; overflow: hidden;
        aspect-ratio: 1 / 1; border: 3px solid transparent;
        box-shadow: 0 2px 8px rgba(0,0,0,.08);
        transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
    }
    .gallery a:hover { transform: translateY(-3px); box-shadow: 0 10px 22px rgba(0,0,0,.16); }
    .gallery a.selected { border-color: #146c6d; }
    .gallery img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .gallery .tag {
        position: absolute; left: .45rem; bottom: .45rem;
        background: rgba(17,24,39,.72); color: #fff; font-size: .68rem; font-weight: 600;
        padding: .18rem .5rem; border-radius: 999px; backdrop-filter: blur(4px);
    }
    .gallery .tag i {
        display: inline-block; width: .5rem; height: .5rem; border-radius: 50%;
        margin-right: .3rem; vertical-align: middle;
    }

    .result-title { font-size: .78rem; text-transform: uppercase; letter-spacing: .12em; color: #6b7280; font-weight: 600; }
    .result-label { font-size: 1.9rem; font-weight: 700; line-height: 1.15; margin: .2rem 0 .1rem 0; }
    .result-conf { color: #4b5563; font-size: .95rem; margin-bottom: 1.1rem; }
    .truth { font-size: .85rem; margin-bottom: 1.1rem; padding: .45rem .7rem; border-radius: 10px; }
    .truth.match { background: #e9f7ef; color: #1e6b45; }
    .truth.miss { background: #fdecec; color: #9b2c2f; }

    .bar-row { margin-bottom: .7rem; }
    .bar-head { display: flex; justify-content: space-between; font-size: .85rem; margin-bottom: .25rem; }
    .bar-head span:last-child { font-variant-numeric: tabular-nums; color: #4b5563; }
    .bar-track { height: 9px; background: #eef0f3; border-radius: 999px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 999px; }

    .disclaimer {
        margin-top: 2.2rem; font-size: .8rem; color: #6b7280; text-align: center; line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource()
def get_model():
    loaded_model = tf.keras.models.load_model("{}/keras_model.h5".format(MODEL_DIR), compile=False)
    return loaded_model


@st.cache_data()
def get_samples():
    """Return {sample_id: {"path", "label", "data_uri"}} for the bundled example images."""
    samples = {}
    for folder, label in SAMPLE_CLASSES.items():
        folder_path = os.path.join(SAMPLES_DIR, folder)
        for file in sorted(os.listdir(folder_path)):
            path = os.path.join(folder_path, file)
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
            samples["{}-{}".format(folder, os.path.splitext(file)[0])] = {
                "path": path,
                "label": label,
                "data_uri": "data:image/jpeg;base64," + encoded,
            }
    return samples


def classify(image):
    # Resize to at least 224x224 and crop from the center, then scale to [-1, 1]
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = (np.asarray(image).astype(np.float32) / 127.5) - 1
    return image, model.predict(data, verbose=0)[0]


def render_result(image, prediction, true_label=None):
    index = int(np.argmax(prediction))
    predicted = class_names[index]
    style = CLASS_STYLES.get(predicted, {"color": "#146c6d", "display": predicted})

    bars = ""
    for i in np.argsort(prediction)[::-1]:
        name = class_names[i]
        s = CLASS_STYLES.get(name, {"color": "#146c6d", "display": name})
        pct = float(prediction[i]) * 100
        bars += (
            '<div class="bar-row"><div class="bar-head"><span>{}</span><span>{:.2f}%</span></div>'
            '<div class="bar-track"><div class="bar-fill" style="width:{:.2f}%;background:{}"></div></div></div>'
        ).format(s["display"], pct, pct, s["color"])

    truth = ""
    if true_label is not None:
        truth_display = CLASS_STYLES.get(true_label, {"display": true_label})["display"]
        if true_label == predicted:
            truth = '<div class="truth match">✓ Matches the dataset label: <b>{}</b></div>'.format(truth_display)
        else:
            truth = '<div class="truth miss">✗ Dataset label is <b>{}</b></div>'.format(truth_display)

    with st.container(border=True):
        left, right = st.columns([1, 1.15], gap="large")
        with left:
            st.image(image)
        with right:
            st.markdown(
                '<div class="result-title">Prediction</div>'
                '<div class="result-label" style="color:{}">{}</div>'
                '<div class="result-conf">{:.2f}% confidence</div>'
                "{}{}".format(style["color"], style["display"], float(prediction[index]) * 100, truth, bars),
                unsafe_allow_html=True,
            )


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
class_names = [line[2:].strip() for line in open("{}/labels.txt".format(MODEL_DIR), "r").readlines() if line.strip()]

with st.sidebar:
    st.markdown("### 🎓 Thesis")
    st.caption("Enhancing Monkeypox Diagnosis: A Machine Learning Approach for Skin Lesion Classification")
    st.markdown("#### Submitted by")
    df = pd.DataFrame(
        {
            "ID": ["19101320", "19101315", "19101313", "19101137"],
            "Name": ["Sadia Mahmud", "Namirah Nazmee", "Mashyat Samiha Ali", "Khusbo Alam"],
        }
    )
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.markdown("#### Model")
    st.caption("Keras CNN with a MobileNetV2 backbone, 224×224 RGB input.")
    st.markdown("#### Classes")
    for name in class_names:
        s = CLASS_STYLES.get(name, {"color": "#146c6d", "display": name})
        st.markdown(
            '<span style="color:{}">●</span> {}'.format(s["color"], s["display"]),
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Thesis project · Skin lesion classification</div>
        <h1>Enhancing Monkeypox Diagnosis</h1>
        <p>A machine learning approach that classifies skin lesion photos as monkeypox,
        chickenpox / measles, or normal skin. Pick an example below or upload your own photo.</p>
        <div class="chips">
            <span class="chip">🧠 Deep learning</span>
            <span class="chip">🖼️ 224×224 input</span>
            <span class="chip">🏷️ 3 classes</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Loading model..."):
    model = get_model()

samples = get_samples()
selected_id = st.query_params.get("sample")
if selected_id not in samples:
    selected_id = None

tab_samples, tab_upload = st.tabs(["🖼️  Try an example", "📤  Upload your own"])

with tab_samples:
    st.markdown(
        '<p class="section-hint">Click any image to classify it. These come from the held-out test split of the dataset.</p>',
        unsafe_allow_html=True,
    )
    tiles = ""
    for sample_id, sample in samples.items():
        s = CLASS_STYLES[sample["label"]]
        tiles += (
            '<a href="?sample={id}" target="_self" class="{cls}" title="Classify this image">'
            '<img src="{src}" alt="{label} example"/>'
            '<span class="tag"><i style="background:{color}"></i>{label}</span></a>'
        ).format(
            id=sample_id,
            cls="selected" if sample_id == selected_id else "",
            src=sample["data_uri"],
            label=s["display"],
            color=s["color"],
        )
    st.markdown('<div class="gallery">{}</div>'.format(tiles), unsafe_allow_html=True)

with tab_upload:
    st.markdown(
        '<p class="section-hint">Upload a clear, close-up photo of the skin area. JPG and PNG are supported.</p>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

result = None
if uploaded_file is not None:
    result = (Image.open(uploaded_file).convert("RGB"), None)
elif selected_id is not None:
    sample = samples[selected_id]
    result = (Image.open(sample["path"]).convert("RGB"), sample["label"])

if result is not None:
    st.markdown("### Result")
    with st.spinner("Identifying image..."):
        image, prediction = classify(result[0])
    render_result(image, prediction, true_label=result[1])
    print("Class:", class_names[int(np.argmax(prediction))], "Confidence Score:", float(np.max(prediction)))

st.markdown(
    '<div class="disclaimer">⚠️ This is a research prototype built for a thesis, not a medical device. '
    "Its predictions are not a diagnosis. Please consult a healthcare professional about any skin concerns.</div>",
    unsafe_allow_html=True,
)
