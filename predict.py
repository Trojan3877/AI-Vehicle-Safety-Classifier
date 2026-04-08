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
# Rule-based driving condition classifier
# ---------------------------------------------------------------------
def classify_driving_conditions(weather: str, visibility: str, traffic: str, driver_state: str):
    """
    Classify driving conditions based on environmental and driver inputs.

    Args:
        weather: One of "clear", "rain", "snow", "fog", or similar.
        visibility: One of "high", "medium", "low".
        traffic: One of "light", "moderate", "heavy".
        driver_state: One of "alert", "distracted", "drowsy".

    Returns:
        Tuple of (safety_score: int, risk_level: str, explanation: str)
        where safety_score is 0–100, risk_level is "low"/"medium"/"high".
    """
    weather_penalty = {"clear": 0, "sunny": 0, "rain": 20, "snow": 25, "fog": 30}
    visibility_penalty = {"high": 0, "medium": 15, "low": 30}
    traffic_penalty = {"light": 0, "moderate": 10, "heavy": 20}
    driver_penalty = {"alert": 0, "distracted": 20, "drowsy": 30}

    penalty = (
        weather_penalty.get(str(weather).lower(), 15)
        + visibility_penalty.get(str(visibility).lower(), 15)
        + traffic_penalty.get(str(traffic).lower(), 10)
        + driver_penalty.get(str(driver_state).lower(), 15)
    )

    score = max(0, 100 - penalty)

    if score >= 70:
        risk_level = "low"
        explanation = "Driving conditions are safe. No major risk factors detected."
    elif score >= 40:
        risk_level = "medium"
        explanation = "Moderate risk detected. Exercise caution while driving."
    else:
        risk_level = "high"
        explanation = "High risk conditions detected. Driving is not recommended."

    return score, risk_level, explanation


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
