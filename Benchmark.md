# 🏁 Benchmark Comparison — AI Vehicle Safety Classifier

This benchmark evaluates alternative models against the final CNN classifier.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|-----------|-----------|--------|-----------|
| Logistic Regression | 0.781 | 0.742 | 0.755 | 0.748 |
| Random Forest | 0.842 | 0.828 | 0.811 | 0.819 |
| SVM (RBF) | 0.865 | 0.856 | 0.843 | 0.848 |
| XGBoost | 0.881 | 0.870 | 0.861 | 0.865 |
| **CNN Classifier (final)** | **0.927** | **0.914** | **0.901** | **0.907** |

---

## ✔ Conclusion
The **CNN classifier outperforms all baseline ML models**, especially in recall — which is crucial for identifying unsafe driving conditions.