# AI Vehicle Safety Classifier 🚗🔍

This project implements a vehicle safety classification system using both **rule-based logic** and a **simulated machine learning approach** (logistic-style scoring). Written in **C++**, it demonstrates modular design, metric evaluation, and real-world data interpretation.

---

## 🧠 Overview

The system classifies vehicles as **SAFE** or **UNSAFE** based on input features like:
- Speed
- Weight
- Braking Distance
- Crash Test Rating

We use two classifiers:
- **Rule-Based**: Simple condition checks
- **Score-Based**: A hand-tuned scoring function acting like logistic regression

---

## 📁 Project Structure

```
ai-vehicle-safety-classifier/
├── include/
│   └── classifier.h
├── src/
│   ├── classifier.cpp
│   ├── main.cpp
├── data/
│   └── vehicles.csv
├── results/
│   └── evaluation.txt
├── classifier_logic.png
├── README.md
```

---

## ⚙️ How It Works

```cpp
// Rule-Based Classifier
IF crash_test_rating >= 4 AND braking_distance <= 40
    --> SAFE
ELSE
    --> UNSAFE

// Score-Based Classifier (Logistic-Style)
score = (2 * crash_rating) - (0.1 * braking_distance) - (0.05 * speed)
IF score >= 5.0 --> SAFE
ELSE --> UNSAFE
```

---

## 📊 Evaluation Results

From a sample of 5 vehicle entries:

```
Rule-Based Accuracy:   80.00%
Score-Based Accuracy:  80.00%
```

Metrics will vary based on dataset size and distribution. The evaluation can be extended to include Precision, Recall, and F1-Score using `metrics.cpp`.

---

## 🧪 Python vs. C++ Comparison

To demonstrate language versatility, the same logic is implemented in both:
- 🐍 `predict.py` – Python version of rule/score classifier
- 💻 `main.cpp` – C++ implementation with evaluation

This shows your ability to write clean, performant ML logic in both interpreted and compiled languages — a skill valued in both research and production teams.

---

## 🖼️ Visual Classifier Logic

![Classifier Logic](classifier_logic.png)

---

## 📜 License

This project is open-sourced under the MIT License.

---

## ✍️ Author

**Corey Leath**  
Aspiring AI/ML Engineer | Dual B.S. Candidate (AI + Computer Science)  
Future MSSE Student @ UC Berkeley | Ph.D. Path in Artificial Intelligence  
📫 [GitHub.com/Trojan3877](https://github.com/Trojan3877)  
🔗 [linkedin.com/in/corey-leath](https://linkedin.com/in/corey-leath)
