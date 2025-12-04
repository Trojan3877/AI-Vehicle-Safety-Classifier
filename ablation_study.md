# 🧪 Ablation Study — AI Vehicle Safety Classifier

The purpose of this ablation study is to analyze how each model component contributes to overall performance.

---

## 🧩 Model Variants Tested

| Variant | Description | F1 Score |
|---------|-------------|-----------|
| **Full Model (baseline)** | CNN + BatchNorm + Dropout + Adam | **0.907** |
| No Dropout | Removed dropout layer | 0.884 |
| No BatchNorm | Removed batch normalization | 0.861 |
| Smaller CNN | Reduced filter sizes & depth | 0.832 |
| SGD Optimizer | Replaced Adam with SGD | 0.789 |

---

## 🔍 Insights

### ✔ BatchNorm improves feature stability  
Removing it reduces performance by **4.6%**.

### ✔ Dropout prevents overfitting  
F1 dropped by **2.3%** without dropout.

### ✔ Model depth strongly affects generalization  
A smaller CNN leads to **significant underfitting**.

### ✔ Adam optimizer performs best  
SGD decreases convergence performance.

---

## 🎯 Conclusion
Every component contributes meaningfully, but **BatchNorm + Adam** contribute the most to model stability and performance.