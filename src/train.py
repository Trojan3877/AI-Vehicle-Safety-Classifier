"""
AI Vehicle Safety Classifier - Training Script
Author: Corey Leath (Trojan3877)
L5/L6 Production-Ready Version

Usage:
    python src/train.py --config config/config.yaml

Description:
    Loads dataset, preprocesses images, trains a CNN classifier,
    saves the trained model, and outputs metrics + visualizations.
"""

import argparse
import logging
import yaml
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from utils import (
    load_dataset,
    build_cnn_model,
    preprocess_batch,
    save_model_with_versioning,
    plot_training_metrics,
    plot_confusion_matrix,
)

# -------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Vehicle Safety Classifier Model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


# -------------------------------------------------------------
# Main Training Function
# -------------------------------------------------------------
def main(config_path):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Logging Config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("Loading dataset from: %s", cfg["data"]["path"])
    X, y, class_names = load_dataset(cfg)

    # Train/Validation Split
    logging.info("Splitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg["training"]["validation_split"],
        random_state=42,
        stratify=y,
    )

    logging.info("Building CNN model...")
    model = build_cnn_model(
        input_shape=cfg["model"]["input_shape"],
        num_classes=len(class_names),
        dropout_rate=cfg["model"]["dropout"],
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["training"]["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    logging.info("Starting training...")
    history = model.fit(
        preprocess_batch(X_train),
        y_train,
        validation_data=(preprocess_batch(X_val), y_val),
        epochs=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
    )

    # Save model with automatic versioning
    model_path = save_model_with_versioning(model, cfg)
    logging.info("Model saved at: %s", model_path)

    # Generate predictions for metrics
    y_pred = np.argmax(model.predict(preprocess_batch(X_val)), axis=1)

    logging.info("Classification Report:\n%s", classification_report(y_val, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plot_confusion_matrix(cm, class_names, output_dir=cfg["paths"]["metrics"])

    # Training Curves
    plot_training_metrics(history, output_dir=cfg["paths"]["metrics"])

    logging.info("Training complete! All metrics exported successfully.")


# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    main(args.config)
