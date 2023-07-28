import pandas as pd
import os
import streamlit as st 
from PIL import Image, ImageOps
import numpy as np
from io import StringIO
import tensorflow as tf

st.set_page_config(
    page_title="Monkeypox Classification",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title('Thesis')
st.subheader('Monkeypox skin disease classification based on machine learning models: A viability study and analysis')
st.divider()

st.sidebar.title("Submitted by")
df = pd.DataFrame(
    {
        "ID": ["19101315", "19101320", "19101313", "19101137"],
        "Name": ["Namirah Nazmee", "Sadia Mahmud", "Mashyat Samiha Ali", "Khusbo Alam"],
    }
)
st.sidebar.dataframe(df)

@st.cache_resource()
def get_model():
    loaded_model = tf.keras.models.load_model("./model/{}.h5".format('keras_model'), compile=False)
    return loaded_model


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
class_names = open("./model/labels.txt", "r").readlines()
# Get list of saved h5 models, which will be displayed in option to load.  
h5_file_list = [file for file in os.listdir("./model") if file.endswith(".h5")]
h5_file_names = [os.path.splitext(file)[0] for file in h5_file_list]
model_type = st.radio("Selected model: ", h5_file_names)
# with st.spinner("Loading model..."):
model = get_model()

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    st.image(image, use_column_width=False)
    placeholder = st.empty()
    placeholder.text("Identifying Image...")
    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    placeholder.empty()
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    confidence_score = confidence_score * 100
    score_percentage = "{:.2f}".format(confidence_score)
    predicted_class = class_name[2:]
    st.markdown(f'**Class:** :green[{predicted_class}]')
    st.markdown(f'**Confidence Score:** :green[{score_percentage}%]')

