# 🔁 Reproducibility Guide

## Environment
- Python 3.10  
- TensorFlow 2.15  
- Scikit-learn 1.4  
- NumPy 1.26  
- Matplotlib 3.8  

---

## Steps to Reproduce
pip install -r requirements.txt python train.py --seed 42 --epochs 20

---

## Randomness Control

numpy.random.seed(42) tf.random.set_seed(42) random.seed(42)

---

## Hardware Used
- Nvidia RTX 5060 (Lenovo Legion)
- CPU fallback supported

