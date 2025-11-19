# 🚗 AI Vehicle Safety Classifier
# 🚗 AI Vehicle Safety Classifier  
### **A Production-Ready ML System for Classifying Safe vs. Unsafe Driving Conditions**  
**Author:** Corey Leath (GitHub: [Trojan3877](https://github.com/Trojan3877))  
**Level:** L5/L6 Machine Learning Engineer Project  
---

## 📊 Badges (Auto-Updated After Training/Evaluation)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)
![Model](https://img.shields.io/badge/Model-CNN%20%2B%20MobileNetV2-important)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📘 Overview

The **AI Vehicle Safety Classifier** is a production-grade machine learning system designed to classify vehicle images as **Safe** or **Unsafe** based on driving conditions.  
This system follows real-world ML engineering practices:

✔ Modular source code  
✔ Config-driven pipeline  
✔ Transfer learning support  
✔ Full evaluation suite  
✔ Deployment-ready inference module  
✔ Artifact tracking + reproducible experiments  

This project is structured exactly like a system used by:  
**Tesla Autopilot, Waymo Safety, Cruise AV, and major ML Ops teams.**

---

## 🧠 Key Features

### **Modeling**
- Custom CNN or MobileNetV2 Transfer Learning
- Dropout regularization
- Adam optimizer with configurable LR
- Full model summary saved to artifacts

### **Data Pipeline**
- Directory-based dataset loader
- Automatic train/val/test generator creation
- Augmentation: rotation, zoom, shift, horizontal flip
- Fully controlled by `config/config.yaml`

### **Training**
- Early stopping  
- Model checkpointing  
- Training history export  
- Metric logging  
- Clean separation of concerns (`data.py`, `model.py`, `train.py`)  

### **Evaluation**
- Confusion matrix (PNG)
- Classification report (TXT)
- ROC-AUC score
- Test accuracy & loss
- Label mapping file (critical for deployment)

### **Inference**
- Production-ready `predict.py`
- JSON-style output
- Identical preprocessing to training
- CLI usage and API-friendly structure

---

## 🏗 Project Architecture (L6 Diagram)

AI-Vehicle-Safety-Classifier/
│
├── config/
│ └── config.yaml # Global hyperparameters (L6 standard)
│
├── src/
│ ├── data.py # Data loaders & augmentation
│ ├── model.py # Model builder (CNN or MobileNet)
│ ├── train.py # Training pipeline w/ callbacks
│ ├── evaluate.py # Metrics, confusion matrix, AUC
│ └── predict.py # Deployment inference module
│
├── artifacts/
│ ├── model/ # Saved model + metrics
│ └── logs/ # Training logs
│
├── data/ # (Excluded from GitHub)
│ ├── train/
│ ├── val/
│ └── test/
│
└── README.md # You are here

---

## License

MIT License

