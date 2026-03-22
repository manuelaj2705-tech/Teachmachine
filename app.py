import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

st.title("Detector de Persona en Cámara")

# Cargar modelo UNA sola vez
@st.cache_resource
def cargar_modelo():
    return load_model("keras_model.h5", compile=False)

model = cargar_modelo()

# Cargar etiquetas
class_names = open("labels.txt", "r").readlines()

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Preparar imagen
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predicción
    prediction = model.predict(data)
    index = np.argmax(prediction)

    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # 🔥 UMBRAL
    THRESHOLD = 0.90

    # 🔥 RESULTADO
    if class_name == "Persona" and confidence_score > THRESHOLD:
        st.success(f"🟢 Estás en cámara ({confidence_score * 100:.2f}%)")
    else:
        st.error(f"🔴 No estás en cámara ({confidence_score * 100:.2f}%)")
