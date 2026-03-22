from PIL import Image
import io
import base64
import streamlit as st
import numpy as np
import pandas as pd
import requests

st.set_page_config(
    page_title="Detección de Objetos - Manuela",
    page_icon="🔍",
    layout="wide"
)

# ─── Configuración Roboflow ────────────────────────────────────────────────────
API_KEY    = "LfNFfrw2rgzqLmzkrWkg"
MODEL_URL  = "manuelas-workspace-jeiss/manuela-instant-1"

# ─── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .stApp { background-color: #0e0e0e; color: white; }
        .title { text-align:center; font-size:2.5rem; font-weight:800; color:#fff; margin-bottom:0.3rem; }
        .subtitle { text-align:center; font-size:1rem; color:#888; margin-bottom:1.5rem; }
        .card-det {
            background: linear-gradient(135deg, #0f3d1f, #1a6b35);
            border: 2px solid #2ecc71;
            border-radius: 16px;
            padding: 1.2rem 1.5rem;
            margin-top: 1rem;
            box-shadow: 0 0 20px rgba(46,204,113,0.25);
        }
        .card-none {
            background: linear-gradient(135deg, #3d0f0f, #6b1a1a);
            border: 2px solid #e74c3c;
            border-radius: 16px;
            padding: 1.2rem 1.5rem;
            margin-top: 1rem;
            box-shadow: 0 0 20px rgba(231,76,60,0.25);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🔍 Detector de Objetos — Manuela</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Modelo Roboflow Instant con detección en tiempo real vía cámara</p>', unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Parámetros")
    conf_threshold = st.slider("Confianza mínima", 0, 100, 50, 1)
    overlap        = st.slider("Overlap (IoU)", 0, 100, 50, 1)

# ─── Función de inferencia vía API ────────────────────────────────────────────
def detectar(pil_img: Image.Image, confidence: int, overlap: int):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    url = f"https://detect.roboflow.com/{MODEL_URL}"
    params = {
        "api_key":    API_KEY,
        "confidence": confidence,
        "overlap":    overlap,
    }
    response = requests.post(
        url,
        params=params,
        data=img_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    return response.json()

# ─── Dibujar cajas sobre la imagen ────────────────────────────────────────────
def dibujar_cajas(pil_img: Image.Image, predictions: list) -> Image.Image:
    from PIL import ImageDraw
    img  = pil_img.copy()
    draw = ImageDraw.Draw(img)

    for pred in predictions:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2

        label = pred.get("class", "?")
        conf  = pred.get("confidence", 0) * 100

        draw.rectangle([x1, y1, x2, y2], outline="#2ecc71", width=3)
        draw.rectangle([x1, y1 - 22, x1 + len(label) * 10 + 60, y1], fill="#2ecc71")
        draw.text((x1 + 4, y1 - 20), f"{label} {conf:.1f}%", fill="black")

    return img

# ─── Cámara ───────────────────────────────────────────────────────────────────
picture = st.camera_input("📷 Capturar imagen para analizar")

if picture:
    pil_img = Image.open(io.BytesIO(picture.getvalue())).convert("RGB")

    with st.spinner("🔍 Enviando al modelo Roboflow..."):
        try:
            resultado = detectar(pil_img, conf_threshold, overlap)
        except Exception as e:
            st.error(f"❌ Error al conectar con Roboflow: {str(e)}")
            st.stop()

    predictions = resultado.get("predictions", [])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Imagen con detecciones")
        if predictions:
            img_anotada = dibujar_cajas(pil_img, predictions)
            st.image(img_anotada, use_container_width=True)
        else:
            st.image(pil_img, use_container_width=True)

    with col2:
        st.subheader("📋 Resultados")
        if predictions:
            conteo = {}
            confs  = {}
            for p in predictions:
                cls = p.get("class", "?")
                conteo[cls] = conteo.get(cls, 0) + 1
                confs.setdefault(cls, []).append(p.get("confidence", 0) * 100)

            data = [
                {
                    "Clase":              cls,
                    "Cantidad":           count,
                    "Confianza promedio": f"{np.mean(confs[cls]):.1f}%"
                }
                for cls, count in conteo.items()
            ]
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df.set_index("Clase")["Cantidad"])

            total = len(predictions)
            st.markdown(f"""
                <div class="card-det">
                    <p style="font-size:1.3rem; font-weight:700; margin:0;">
                        🟢 {total} objeto{"s" if total > 1 else ""} detectado{"s" if total > 1 else ""}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="card-none">
                    <p style="font-size:1.3rem; font-weight:700; margin:0;">
                        🔴 No se detectaron objetos
                    </p>
                    <p style="color:#ddd; margin-top:0.4rem;">
                        Prueba bajando el umbral de confianza en el panel izquierdo.
                    </p>
                </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Modelo: Roboflow Instant · manuelas-workspace-jeiss/manuela-instant-1")
