# StyleNet — Clasificación de prendas con redes neuronales

Proyecto académico de Ciencia de Datos de **Gianina Jensen Guzmán**. Clasifica imágenes de prendas en las diez categorías de Fashion-MNIST y permite explorar las salidas del modelo en una interfaz Streamlit.

[Ver portafolio profesional](https://github.com/Mona-Jensen/m-dulos)

## Qué desarrollé

- Preparación de imágenes y normalización de valores.
- Una red neuronal densa de referencia y una red convolucional con TensorFlow/Keras.
- Visualización de resultados con Matplotlib y de salidas por categoría con Plotly.
- Interfaz para cargar imágenes y consultar la clasificación del modelo guardado.

## Arquitectura CNN

Dos capas convolucionales de 32 y 64 filtros con max-pooling, una capa densa de 64 unidades, dropout de 0,3 y salida softmax para diez clases. El script configura 15 épocas y reserva el 20 % del conjunto de entrenamiento para validación.

## Archivos principales

| Archivo | Función |
| --- | --- |
| [clasificador.py](clasificador.py) | Modelo de referencia. |
| [clasificador_cnn.py](clasificador_cnn.py) | Entrenamiento de la CNN. |
| [app_stylenet.py](app_stylenet.py) | Interfaz Streamlit. |
| [requirements.txt](requirements.txt) | Dependencias. |
| [INFORME_TÉCNICO.pdf](INFORME_TÉCNICO.pdf) | Informe del trabajo. |

## Ejecución

Desde la carpeta del repositorio, instalar las dependencias en un entorno virtual compatible con las versiones de requirements.txt:

```bash
python -m venv .venv
# Activar el entorno virtual según el sistema operativo.
python -m pip install -r requirements.txt
python -m streamlit run app_stylenet.py
```

Para volver a entrenar, ejecutar `python clasificador_cnn.py`. El script descarga Fashion-MNIST si no está en la caché y guarda `modelo_stylenet_cnn.h5`. La descarga requiere conexión y el entrenamiento puede consumir recursos considerables.

## Resultados registrados en la documentación original

| Modelo | Exactitud reportada | Pérdida reportada |
| --- | ---: | ---: |
| Red densa | 87,72 % | 0,35 |
| CNN | 90,45 % | 0,26 |

Son valores históricos del proyecto, no métricas reproducidas en la revisión del portafolio. Para compararlos formalmente hay que documentar el conjunto de evaluación, la semilla, las versiones y el protocolo de entrenamiento. El script CNN no evalúa actualmente el conjunto test después de entrenar.

## Alcance y verificación

Se revisó el código y se corrigió un carácter accidental al final del script CNN; la sintaxis Python quedó validada. No se reentrenó el modelo ni se ejecutó la interfaz en esta revisión.

Fashion-MNIST contiene imágenes de 28 × 28 en escala de grises. Fotografías reales pueden diferir de ese dominio. Las salidas softmax no equivalen a una confianza calibrada ni garantizan clasificación correcta.
