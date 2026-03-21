# Fashion-MNIST CNN Classifier & Streamlit Deployment 👕👟

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

## 📋 Resumen del Proyecto
Implementación de un sistema de visión computacional basado en **Redes Neuronales Convolucionales (CNN)** para la clasificación automatizada de 10 categorías de prendas de vestir. El proyecto abarca desde el entrenamiento del modelo con el dataset **Fashion-MNIST** hasta el despliegue de una interfaz interactiva funcional.

## 🚀 Arquitectura del Modelo (StyleNet)
Se diseñó una arquitectura CNN optimizada para la extracción de características jerárquicas:
- **Capas Convolucionales:** Detección de patrones espaciales y texturas.
- **Max-Pooling:** Reducción de dimensionalidad y optimización de cómputo.
- **Dropout (0.3):** Regularización para mitigar el sobreajuste (*overfitting*).
- **Capa de Salida:** Activación Softmax para la distribución de probabilidades multiclase.

### Métricas de Desempeño
| Modelo | Accuracy (Exactitud) | Loss (Pérdida) |
| :--- | :--- | :--- |
| Red Neuronal Densa (Baseline) | 87.72% | 0.35 |
| **Red Convolutiva (CNN)** | **90.45%** | **0.26** |

## 🛠️ Tecnologías Utilizadas
- **Lenguaje:** Python 3.12
- **Deep Learning:** TensorFlow / Keras
- **Web App:** Streamlit
- **Visualización:** Plotly & Matplotlib
- **Procesamiento de Imágenes:** Pillow & NumPy

## 🔧 Instalación y Uso
1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/Mona-Jensen/fashion-mnist-cnn-streamlit.git](https://github.com/Mona-Jensen/fashion-mnist-cnn-streamlit.git)