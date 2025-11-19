"""
AI Vehicle Safety Classifier — Inference (Prediction) Module
Author: Corey Leath (Trojan3877)
L5/L6 Production-Ready Architecture

Handles:
✔ Loading best saved model
✔ Preprocessing input image
✔ Predicting safety class
✔ JSON-style output
✔ Reusable for FastAPI / Streamlit / Flask
"""

import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


# ---------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------
def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# Load best model
# ---------------------------------------------------------------------
def load_best_model(config_path="config/config.yaml"):
    config = load_config(config_path)
    model_path = os.path.join(config["paths"]["model_dir"], "best_model.keras")
    return tf.keras.models.load_model(model_path)


# ---------------------------------------------------------------------
# Preprocess image to model-ready tensor
# ---------------------------------------------------------------------
def preprocess_image(img_path, config_path="config/config.yaml"):

    config = load_config(config_path)
    target_size = tuple(config["model"]["input_shape"][:2])

    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ---------------------------------------------------------------------
# Predict class from image path
# ---------------------------------------------------------------------
def predict_image(img_path, config_path="config/config.yaml"):

    # Load model and config
    model = load_best_model(config_path)
    config = load_config(config_path)

    # Preprocess input
    img_array = preprocess_image(img_path, config_path)

    # Predict probabilities
    preds = model.predict(img_array)
    pred_index = np.argmax(preds, axis=1)[0]
    confidence = preds[0][pred_index]

    # Load label map
    label_map_path = os.path.join(config["paths"]["model_dir"], "label_mapping.txt")
    label_map = {}

    with open(label_map_path, "r") as f:
        for line in f:
            idx, label = line.strip().split(": ")
            label_map[int(idx)] = label

    predicted_label = label_map[pred_index]

    # JSON-style output
    result = {
        "predicted_class": predicted_label,
        "confidence": float(confidence)
    }

    return result


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict image using Vehicle Safety Classifier")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    output = predict_image(args.image_path, args.config)
    
    print("\nPrediction Result:")
    print(output)
