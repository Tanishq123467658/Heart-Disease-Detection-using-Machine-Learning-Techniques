
# ❤️ Heart Disease Detection using Machine Learning Techniques

A comprehensive machine learning pipeline designed to detect heart disease from patient data using classical algorithms—complete with preprocessing, feature engineering, model training, evaluation, and analytic visualizations.

---

## 📌 Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Dataset & Preprocessing](#dataset--preprocessing)  
7. [Models & Evaluation](#models--evaluation)  
8. [Results & Visualization](#results--visualization)  
9. [Contributing](#contributing)  
10. [Support](#support)

---

## Overview

Heart disease is a leading cause of death globally. This project provides a machine learning solution to predict potential heart disease in patients using crucial medical indicators such as cholesterol, blood pressure, and age—offering both accuracy and interpretability.

---

## Features

- 🔹 Data cleaning & preprocessing (handling missing values, normalization)  
- 🔹 Feature selection and engineering pipelines  
- 🔹 Multiple classification models: Logistic Regression, Decision Trees, Random Forests, SVM  
- 🔹 Hyperparameter tuning via cross-validation  
- 🔹 Comprehensive evaluation: Accuracy, Precision, Recall, F1-score, ROC–AUC  
- 🔹 Clear visualizations: confusion matrices, ROC curves, feature importances

---

## Project Structure

```text
├── data/                          
│   └── heart.csv                 # UCI Heart Disease dataset or equivalent
├── src/
│   ├── preprocess.py            # Data cleaning & feature pipelines
│   ├── features.py              # Feature selection & transformation
│   ├── train_models.py          # Model training and CV tuning
│   ├── evaluate.py              # Evaluation metrics and ROC curves
│   └── visualize.py             # Plots for metrics and feature importances
├── notebooks/                   # Optional Jupyter notebooks for EDA
├── models/                      # Serialized model files (.pkl)
├── results/                     # Saved figures and metrics
├── requirements.txt             # Project dependencies
└── main.py                      # End-to-end pipeline runner
└── README.md                    # This file
```

---

## Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Tanishq123467658/Heart-Disease-Detection-using-Machine-Learning-Techniques.git
   cd Heart-Disease-Detection-using-Machine-Learning-Techniques
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

- **Run the full ML pipeline:**
  ```bash
  python main.py
  ```

- **Or run individual components:**
  ```bash
  python src/preprocess.py
  python src/train_models.py
  python src/evaluate.py
  python src/visualize.py
  ```

---

## Dataset & Preprocessing

- **Dataset**: UCI Heart Disease dataset (or similar CSV) containing medical features like age, sex, trestbps, chol, etc.
- **Preprocessing** steps:
  - Missing value imputation  
  - One-hot encoding for categorical variables  
  - Standard scaling / normalization

---

## Models & Evaluation

Trained models include:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine

Evaluation metrics:
- Accuracy, Precision, Recall, F1-score, ROC–AUC  
- Visual tools: confusion matrix and ROC curves

---

## Results & Visualization

Once executed, you’ll find:
- **`results/metrics.json`** with all metric scores  
- **ROC curve plots** for each classifier  
- **Confusion matrices** displayed as heatmaps  
- **Feature importance charts** to understand key predictors (e.g., cholesterol, max heart rate)

---

## Contributing

Feel free to improve the project by:
- Adding additional algorithms (e.g., Gradient Boosting, XGBoost)  
- Enhancing data preprocessing (outlier detection, balancing techniques)  
- Building a web or mobile interface for prediction  
- Including deployment scripts or Docker support

To contribute:
1. Fork the repository  
2. Create a new branch  
3. Develop and test your feature  
4. Submit a pull request!

---

## Support

For bug reports, feature requests, or general questions, please open an issue or reach out to me on GitHub.

---
