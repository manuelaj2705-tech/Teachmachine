import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

st.title("Detector de Persona en Cámara")

@st.cache_resource
def cargar_modelo():
    return load_model("keras_model.h5", compile=False)

model = cargar_modelo()
class_names = open("labels.txt", "r").readlines()

img_file_buffer = st.camera_input("Toma una foto")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert("RGB")
    st.image(image, caption="Captura", use_column_width=True)

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)[0]

    prob_persona = 0
    prob_no = 0

    for i, name in enumerate(class_names):
        if "Persona" in name:
            prob_persona = prediction[i]
        else:
            prob_no = prediction[i]

    if prob_persona > prob_no and prob_persona > 0.75:
        st.success(f"🟢 Estás en cámara ({prob_persona * 100:.2f}%)")
    else:
        st.error(f"🔴 No estás en cámara ({prob_no * 100:.2f}%)")
