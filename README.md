🚗 AI Vehicle Safety Classifier  
### Real-Time Safety Detection using Deep Learning (CNN)

<div align="center">

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![MLflow](https://img.shields.io/badge/MLflow-Enabled-blueviolet)
![CUDA](https://img.shields.io/badge/CUDA-11.8-success)
![Research](https://img.shields.io/badge/Research-Ready-red)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

</div>


Project Summary

The **AI Vehicle Safety Classifier** is a convolutional neural network (CNN) designed to classify driving conditions as:

- **Safe Driving**
- **Unsafe Driving**

This project is built for **driver monitoring**, **fleet analytics**, **insurance risk modeling**, and **vehicle safety research**.  
The repository includes full **research-grade documentation**, **metrics**, **ablation studies**, and **reproducibility artifacts**.



 Features

### ✅ Deep Learning CNN (3-block architecture)  
### ✅ BatchNorm + Dropout for robust learning  
### ✅ Full research suite:  
- [Metrics](metrics.md)  
- [Ablation Study](ablation_study.md)  
- [Model Card](model_card.md)  
- [Benchmark](benchmark.md)  
- [Dataset Stats](dataset_stats.md)  
- [Reproducibility](reproducibility.md)  

### ✅ Industry-ready file structure  
### ✅ Scalable for video-based models  
### ✅ Extendable to real-time inference with TensorRT  



 Model Architecture Overview

Input (128×128×3) ↓ Conv2D → ReLU → BatchNorm → MaxPool ↓ Conv2D → ReLU → BatchNorm → MaxPool ↓ Conv2D → ReLU → BatchNorm → MaxPool ↓ Flatten ↓ Dropout(0.3) ↓ Dense Layer (Softmax)



Research Metrics (Highlights)

From **metrics.md**:

| Metric | Score |
|--------|--------|
| Accuracy | **0.927** |
| Precision | **0.914** |
| Recall | **0.901** |
| F1 Score | **0.907** |
| ROC-AUC | **0.958** |

→ Full details: See [metrics.md](metrics.md)



Ablation Study (Highlights)

From **ablation_study.md**:

| Variant | F1 Score |
|---------|-----------|
| Full Model | **0.907** |
| No Dropout | 0.884 |
| No BatchNorm | 0.861 |
| Smaller CNN | 0.832 |
| SGD Optimizer | 0.789 |

### ✔ BatchNorm and Dropout are critical  
### ✔ Depth strongly influences performance  

Full study: [ablation_study.md](ablation_study.md)


 Benchmark Comparison

From **benchmark.md**:

| Model | F1 Score |
|-------|-----------|
| Logistic Regression | 0.748 |
| Random Forest | 0.819 |
| XGBoost | 0.865 |
| **CNN Classifier** | **0.907** |

→ CNN beats all classical baselines.

Full comparison: [benchmark.md](benchmark.md)



# 📁 Repository Structure

AI-Vehicle-Safety-Classifier/ │ ├── train.py ├── predict.py ├── dataset/ ├── images/ │   ├── loss_curve.png │   ├── accuracy_curve.png │   └── confusion_matrix.png │ ├── metrics.md ├── ablation_study.md ├── benchmark.md ├── model_card.md ├── dataset_stats.md ├── reproducibility.md │ └── README.md



 Installation

```bash
git clone https://github.com/Trojan3877/AI-Vehicle-Safety-Classifier.git
cd AI-Vehicle-Safety-Classifier
pip install -r requirements.txt




🚀 Training

python train.py --epochs 20 --seed 42




 Future Improvements

Integrate YOLO/RetinaNet for hazard region detection

Add sequence modeling (LSTM / 3D-CNN) for video frames

Use TensorRT for real-time inference

Add uncertainty estimation for safety-critical AI

Design Questions & Reflections
Q: What problem does this project aim to solve?
A: This project aims to explore how machine learning can be combined with deterministic logic to assess safety in simulated vehicle scenarios, focusing on building a system that balances predictive performance with clarity and real-world constraints.
Q: Why did I choose this hybrid ML + rule-based approach instead of a pure model?
A: I chose a hybrid approach because safety-critical systems often need clear, interpretable rules alongside learned patterns. A pure ML model might pick up correlations that don’t hold in rare but dangerous edge cases, whereas combining learned behavior with defined safety rules helps ground decisions in understandable logic.
Q: What were the main trade-offs I made?
A: The main trade-off was between complexity and interpretability. A fully learned model might achieve slightly higher accuracy, but at the cost of making behavior harder to predict and trust. By integrating rule-based logic, I accepted some reduction in raw performance in exchange for improved explainability and consistent safety handling.
Q: What didn’t work as expected?
A: Initially, the learned component sometimes overfitted to specific scenarios in the training data and didn’t generalize well to simulated edge cases. This helped me realize that data diversity and evaluation strategy are just as important as model choice, especially for safety-related tasks.
Q: What did I learn from building this project?
A: I learned that engineering judgment — deciding where to rely on logic versus learned components — is often as critical as the model itself. I also gained a deeper appreciation for careful evaluation and testing, particularly in contexts where incorrect outputs carry higher consequences.
Q: If I had more time or resources, what would I improve next?
A: I would build stronger validation and stress-testing frameworks to simulate a wider range of edge cases, and explore uncertainty quantification techniques so the system could better express when it’s unsure rather than making overconfident predictions.





Corey Leath
GitHub: https://github.com/Trojan3877
LinkedIn: https://www.linkedin.com/in/corey-leath
Email: corey22blue@hotmail.com




📜 License

This project is licensed under the MIT License.



