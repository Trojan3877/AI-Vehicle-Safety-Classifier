# 📊 Dataset Statistics — AI Vehicle Safety Classifier

## Class Distribution
- Safe: 740 images  
- Unsafe: 520 images  

**Conclusion:**  
Slight class imbalance → handled with class weights.

---

## Feature Characteristics
- RGB images  
- Resolution normalized to 128×128  
- Motion blur present in ~11% of unsafe samples  
- Night conditions present in <5% of data

---

## Correlations
- Unsafe conditions correlate strongly with blur + sharp steering angles  
- Safe conditions correlate with centered lane position

---

## Train/Val/Test Split
- Train: 70%  
- Validation: 15%  
- Test: 15%  

Random seed: 42