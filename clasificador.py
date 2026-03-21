import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
import os

# 1. Cargar el dataset de StyleNet (Fashion-MNIST)
# Según el requerimiento de usar datasets reales [cite: 18, 47]
print("Cargando datos de Fashion-MNIST...")
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# Normalización: Escalar píxeles de 0-255 a un rango de 0-1
train_images, test_images = train_images / 255.0, test_images / 255.0

# 2. Definir Arquitectura de Red Densa (Lección 1)
# Se identifican capas, pesos y funciones de activación [cite: 33, 35]
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),          # Capa de entrada (aplana la imagen)
    layers.Dense(128, activation='relu'),            # Capa oculta con activación ReLU
    layers.Dropout(0.2),                            # Técnica de regularización [cite: 20, 49]
    layers.Dense(10, activation='softmax')          # Capa de salida para 10 categorías
])

# 3. Configuración y Compilación
# Uso de optimizadores y funciones de pérdida adecuados [cite: 58]
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Entrenamiento del Modelo (Lección 3)
# CORRECCIÓN: Asignamos el resultado a 'history' para evitar el error anterior
print("\nIniciando entrenamiento...")
history = model.fit(train_images, train_labels, 
                    epochs=10, 
                    validation_split=0.2) # Validación del rendimiento [cite: 19, 48]

# 5. Evaluación y Guardado (Entregables)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nPrecisión final en el set de prueba: {test_acc:.4f}')

# Guardar el modelo físico en la carpeta PROYECTO_ALKEMY_MLLIB
model.save('modelo_stylenet_v1.h5')
print(f"\nArchivo guardado exitosamente en: {os.path.abspath('modelo_stylenet_v1.h5')}")

# 6. Generación de Gráficos de Entrenamiento (Requerimiento Lección 3)
# Estos gráficos son necesarios para la documentación técnica [cite: 63, 85]
plt.figure(figsize=(12, 4))

# Gráfico de Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo (Accuracy)')
plt.xlabel('Época')
plt.ylabel('Exactitud')
plt.legend()

# Gráfico de Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo (Loss)')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()