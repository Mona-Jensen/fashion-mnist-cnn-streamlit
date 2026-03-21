import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
import os

# 1. Carga y normalización de datos
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0

# 2. Arquitectura CNN mejorada (Lección 4)
model_cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),  # Regularización para evitar sobreajuste
    layers.Dense(10, activation='softmax')
])

# 3. Compilación
model_cnn.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# 4. Entrenamiento (Subimos a 15 épocas para mayor precisión)
print("Iniciando entrenamiento de la CNN...")
history_cnn = model_cnn.fit(train_images, train_labels, epochs=15, validation_split=0.2)

# 5. Guardar modelo
model_cnn.save('modelo_stylenet_cnn.h5')
print("Modelo guardado como 'modelo_stylenet_cnn.h5'")

# 6. Visualización de resultados
plt.plot(history_cnn.history['accuracy'], label='Precisión')
plt.title('Rendimiento de la Red Convolutiva (15 épocas)')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.show()¿