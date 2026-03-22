import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import time

# ─── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="DETECTOR DE PERSONA EN CAMARA ",
    page_icon="📷",
    layout="centered"
)

st.markdown("""
    <style>
        .main { background-color: #0e0e0e; }
        .stApp { background-color: #0e0e0e; color: white; }
        .title { text-align: center; font-size: 10rem; font-weight: 800; color: #ffffff; margin-bottom: 0.5rem; }
        .subtitle { text-align: center; font-size: 5 rem; color: #888; margin-bottom: 1.5rem; }

        .card-verde {
            background: linear-gradient(135deg, #0f3d1f, #1a6b35);
            border: 2px solid #2ecc71;
            border-radius: 16px;
            padding: 1.5rem 2rem;
            text-align: center;
            margin-top: 1rem;
            box-shadow: 0 0 20px rgba(46, 204, 113, 0.3);
        }
        .card-rojo {
            background: linear-gradient(135deg, #3d0f0f, #6b1a1a);
            border: 2px solid #e74c3c;
            border-radius: 16px;
            padding: 1.5rem 2rem;
            text-align: center;
            margin-top: 1rem;
            box-shadow: 0 0 20px rgba(231, 76, 60, 0.3);
        }
        .estado-texto {
            font-size: 1.6rem;
            font-weight: 700;
            margin: 0;
        }
        .prob-texto {
            font-size: 1.1rem;
            color: #ddd;
            margin-top: 0.4rem;
        }
        .barra-contenedor {
            background: #222;
            border-radius: 12px;
            height: 22px;
            width: 100%;
            margin-top: 1rem;
            overflow: hidden;
        }
        .barra-verde {
            background: linear-gradient(90deg, #27ae60, #2ecc71);
            height: 100%;
            border-radius: 12px;
            transition: width 0.3s ease;
        }
        .barra-roja {
            background: linear-gradient(90deg, #c0392b, #e74c3c);
            height: 100%;
            border-radius: 12px;
            transition: width 0.3s ease;
        }
        div[data-testid="stCameraInput"] label { color: #ccc !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">📷 Detector de Persona en Cámara</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Usando un modelo entrenado en Teachable Machine, esta aplicación identifica si una persona está presente frente a la cámara y muestra la probabilidad de detección en tiempo real.</p>', unsafe_allow_html=True)

# ─── Cargar modelo y etiquetas ─────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    return load_model("keras_model.h5", compile=False)

@st.cache_data
def cargar_labels():
    with open("labels.txt", "r") as f:
        return f.readlines()

model = cargar_modelo()
class_names = cargar_labels()

# Mapear índice → nombre legible
# labels.txt: "0 Manu" / "1 Sin manu"
def nombre_clase(raw: str) -> str:
    """Devuelve solo el nombre (sin el número) en minúsculas."""
    partes = raw.strip().split(" ", 1)
    return partes[1] if len(partes) > 1 else partes[0]

THRESHOLD = 0.99  # Umbral de confianza

# ─── Predicción ────────────────────────────────────────────────────────────────
def predecir(pil_image: Image.Image):
    size = (224, 224)
    img = ImageOps.fit(pil_image.convert("RGB"), size, Image.Resampling.LANCZOS)
    arr = (np.asarray(img).astype(np.float32) / 127.5) - 1
    data = arr[np.newaxis, ...]          # shape (1,224,224,3)
    pred = model.predict(data, verbose=0)
    idx = int(np.argmax(pred))
    confianza = float(pred[0][idx])
    nombre = nombre_clase(class_names[idx])
    return nombre, confianza, idx


# ─── Cámara ────────────────────────────────────────────────────────────────────
placeholder_resultado = st.empty()

img_buffer = st.camera_input(
    "Captura en vivo — cada foto se analiza automáticamente",
    label_visibility="visible"
)

if img_buffer is not None:
    image = Image.open(img_buffer)

    nombre, confianza, idx = predecir(image)
    porcentaje = confianza * 100

    # Clase 0 → persona presente  |  Clase 1 → sin persona
    esta_presente = (idx == 0 and confianza >= THRESHOLD)

    with placeholder_resultado.container():
        if esta_presente:
            st.markdown(f"""
                <div class="card-verde">
                    <p class="estado-texto">🟢 ¡Estás en cámara!</p>
                    <p class="prob-texto">Probabilidad: <strong>{porcentaje:.2f}%</strong></p>
                    <div class="barra-contenedor">
                        <div class="barra-verde" style="width:{porcentaje:.1f}%"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Si la confianza es alta pero en "Sin manu", mostramos esa prob
            # Si la confianza es baja en "Manu", mostramos la probabilidad de Sin manu
            if idx == 0:
                prob_display = (1 - confianza) * 100   # prob de estar fuera
                label_display = nombre_clase(class_names[1])
            else:
                prob_display = porcentaje
                label_display = nombre_clase(class_names[1])

            st.markdown(f"""
                <div class="card-rojo">
                    <p class="estado-texto">🔴 No estás en cámara</p>
                    <p class="prob-texto">Probabilidad de ausencia: <strong>{prob_display:.2f}%</strong></p>
                    <div class="barra-contenedor">
                        <div class="barra-roja" style="width:{prob_display:.1f}%"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # Modo continuo: fuerza re-run automático cada 1.5 s
    if modo_continuo:
        time.sleep(1.5)
        st.rerun()

else:
    placeholder_resultado.markdown("""
        <div style="text-align:center; color:#555; margin-top:2rem; font-size:1rem;">
            👆 Activa la cámara para comenzar la detección
        </div>
    """, unsafe_allow_html=True)
