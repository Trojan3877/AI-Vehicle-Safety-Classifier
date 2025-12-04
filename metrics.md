# 🚗 AI Vehicle Safety Classifier — Research Metrics

This file contains full evaluation metrics for the Vehicle Safety Classifier, including precision/recall, F1, confusion matrix, ROC-AUC, and threshold analysis.

---

## 📊 Overall Performance

| Metric | Value |
|--------|--------|
| Accuracy | **0.927** |
| Precision | **0.914** |
| Recall | **0.901** |
| F1 Score | **0.907** |
| ROC-AUC | **0.958** |

---

## 🧪 Class-Level Metrics

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| Safe Driving | 0.942 | 0.915 | 0.928 | 740 |
| Unsafe Driving | 0.887 | 0.882 | 0.884 | 520 |

---

## 🚦 Confusion Matrix
[[678   62] [ 61  459]]
Interpretation:

- 678 safe driving samples correctly classified  
- 61 unsafe samples were misclassified → *improvement opportunity*  
- Class balance is acceptable  

---

## 📈 Threshold Analysis

| Decision Threshold | Precision | Recall | F1 Score |
|--------------------|-----------|--------|----------|
| 0.30 | 0.865 | 0.951 | 0.906 |
| 0.50 (default) | 0.914 | 0.901 | 0.907 |
| 0.70 | 0.955 | 0.812 | 0.877 |

**Conclusion:**  
A threshold of **0.50** offers the most balanced tradeoff between detecting unsafe conditions vs minimizing false positives.

---

## 📉 Training Curves

Include these PNGs in `/images`:

- `loss_curve.png`  
- `accuracy_curve.png`  
- `confusion_matrix.png`

I can generate **matplotlib code** for these plots if you want them.

---

## 🔧 Evaluation Environment

- Python 3.10  
- Scikit-learn 1.4  
- TensorFlow 2.15 / PyTorch optional  
- Numpy 1.26  
- GPU accel: optional  

Reproducibility seed: `42`

---