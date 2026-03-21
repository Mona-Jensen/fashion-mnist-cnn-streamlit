import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# 1. Configuración de nombres de clases de Fashion-MNIST
clases = ['Polera/Top', 'Pantalón', 'Pullover', 'Vestido', 'Abrigo', 
          'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botín']

# 2. Cargar el modelo entrenado
modelo_path = 'modelo_stylenet_cnn.h5'

if os.path.exists(modelo_path):
    model = tf.keras.models.load_model(modelo_path)
    print("Modelo StyleNet cargado con éxito.")
else:
    print(f"ERROR: No se encontró el archivo {modelo_path}")

def clasificar_prenda(ruta_archivo):
    if not os.path.exists(ruta_archivo):
        print(f"ERROR: No existe el archivo '{ruta_archivo}' en la carpeta.")
        return

    # Preprocesamiento exacto (28x28, escala de grises e inversión)
    img = image.load_img(ruta_archivo, target_size=(28, 28), color_mode="grayscale")
    img_array = image.img_to_array(img)
    
    # Invertir colores: Fashion-MNIST espera fondo negro (0) y ropa blanca (255)
    img_array = (255 - img_array) / 255.0 
    img_array = img_array.reshape(1, 28, 28, 1)

    # Realizar la predicción
    predicciones = model.predict(img_array)
    indice_final = np.argmax(predicciones)
    porcentaje = predicciones[0][indice_final] * 100

    print("-" * 30)
    print(f"RESULTADO: {clases[indice_final]}")
    print(f"CONFIANZA: {porcentaje:.2f}%")
    print("-" * 30)

# EJECUCIÓN: Asegúrate de que el nombre coincida con tu foto
clasificar_prenda('prenda_test.jpg')