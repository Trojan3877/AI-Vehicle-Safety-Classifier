# AI-Vehicle-Safety-Classifier
AI/ML model to classify safe vs. unsafe driving conditions for autonomous vehicles
!pip install kaggle
!kaggle datasets download -d <dataset-name>
!unzip <dataset-name>.zip -d data/
train_model.py
# train_model.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# Load and preprocess data
def load_data(data_dir):
    # Placeholder: Replace with your dataset loading logic
    images = []  # Load images from data_dir
    labels = []  # 0 = safe, 1 = unsafe
    # Preprocess: Resize to 224x224, normalize to [0,1]
    images = np.array(images) / 255.0
    labels = np.array(labels)
    return images, labels

# Build CNN model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary output
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    data_dir = "data/"
    X, y = load_data(data_dir)
    model = build_model()
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    model.save("safety_classifier.h5")

![Classifier Logic](./images/classifier_logic.png)
