# predict.py
# Author: Corey Leath (coreyleath10@gmail.com)
# Purpose: Inference script for the AI-Driven Autonomous Vehicle Safety Classifier.
# This script loads a trained TensorFlow model and predicts if a driving condition
# image is "Safe" or "Unsafe," designed for autonomous vehicle safety applications.

import tensorflow as tf  # TensorFlow for model loading and prediction
import numpy as np      # NumPy for array manipulation during preprocessing

def load_model(model_path):
    """
    Load the pre-trained safety classifier model from a file.
    
    Args:
        model_path (str): Path to the saved model file (e.g., 'safety_classifier.h5').
    
    Returns:
        tf.keras.Model: Loaded TensorFlow model ready for inference.
    """
    # Use TensorFlow's load_model to restore the trained CNN from disk
    return tf.keras.models.load_model(model_path)

def predict_image(image_path, model):
    """
    Preprocess an input image and predict its safety classification.
    
    Args:
        image_path (str): Path to the test image file (e.g., 'test_image.jpg').
        model (tf.keras.Model): Loaded model for inference.
    
    Returns:
        str: Prediction result - "Safe" or "Unsafe" driving condition.
    """
    # Load and resize image to 224x224 (matching training input size)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    
    # Convert image to array and normalize pixel values to [0,1] for model consistency
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    
    # Add batch dimension (model expects shape: (1, 224, 224, 3))
    img = np.expand_dims(img, axis=0)
    
    # Run inference: predict probability (0 to 1) of unsafe condition
    pred = model.predict(img)
    
    # Threshold at 0.5: <0.5 = Safe, >=0.5 = Unsafe (binary classification)
    return "Safe" if pred[0][0] < 0.5 else "Unsafe"

if __name__ == "__main__":
    # Entry point for standalone execution or testing
    # Load the pre-trained model from the saved file
    model = load_model("safety_classifier.h5")
    
    # Predict safety for a sample image (replace with your test image path)
    result = predict_image("test_image.jpg", model)
    
    # Output the result to console for user feedback
    print(f"Driving Condition: {result}")
