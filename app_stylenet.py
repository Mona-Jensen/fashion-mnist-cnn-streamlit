import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import os

# 1. Configuración de estilo
st.set_page_config(page_title="StyleNet Pro", page_icon="👔", layout="wide")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.title("👔 StyleNet: Análisis de Inteligencia Artificial")
st.markdown("---")

# 2. Carga del Modelo
@st.cache_resource
def cargar_modelo():
    modelo_path = 'modelo_stylenet_cnn.h5'
    if os.path.exists(modelo_path):
        return tf.keras.models.load_model(modelo_path)
    return None

model = cargar_modelo()
clases = ['Polera/Top', 'Pantalón', 'Pullover', 'Vestido', 'Abrigo', 
          'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']

# 3. Interfaz de Usuario
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Carga de Prenda")
    archivo = st.file_uploader("Sube una imagen para analizar", type=["jpg", "png", "jpeg"])
    
    if archivo:
        img_original = Image.open(archivo)
        st.image(img_original, caption="Prenda capturada", use_container_width=True)

with col2:
    st.header("📊 Resultado del Análisis")
    if archivo and model:
        with st.spinner('Procesando redes neuronales...'):
            # Preprocesamiento exacto (Fashion-MNIST style)
            img_processed = img_original.convert('L').resize((28, 28))
            img_array = np.array(img_processed)
            img_array = (255 - img_array) / 255.0  # Inversión y normalización
            img_array = img_array.reshape(1, 28, 28, 1)

            # Predicción
            preds = model.predict(img_array)[0]
            idx = np.argmax(preds)
            
            # Métricas principales
            st.metric("Categoría Detectada", clases[idx], f"{preds[idx]*100:.2f}% Confianza")
            
            # Gráfico interactivo de probabilidades
            df_prob = pd.DataFrame({'Prenda': clases, 'Confianza (%)': preds * 100})
            df_prob = df_prob.sort_values('Confianza (%)', ascending=True)
            
            fig = px.bar(df_prob, x='Confianza (%)', y='Prenda', orientation='h',
                         color='Confianza (%)', color_continuous_scale='Viridis')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    elif not model:
        st.error("Error: No se encontró 'modelo_stylenet_cnn.h5'")
    else:
        st.info("Esperando carga de imagen...")