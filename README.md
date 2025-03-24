# 🌧️ Rainfall Prediction Pipeline

A complete machine learning pipeline for rainfall prediction using real-world weather data. The project handles everything from exploratory analysis to model optimization, and is organized into modular Python scripts for clarity and reuse.

---

##  Project Structure

```
rainfall-prediction/
├── data_analysis.py           # Exploratory Data Analysis (EDA)
├── data_modification.py       # Data cleaning, feature engineering, scaling
├── training.py                # Training multiple classifiers & evaluation
├── optimization.py            # Grid search + test set prediction
├── pipeline.py                # Run the full pipeline end-to-end
├── requirements.txt           # Package Requirements
└── train-val.csv              # Train-Val dataset
└── test.csv                   # Test dataset
```

---

##  How to Run

1. **Install dependencies and check Python version(at least 3.8)**

```bash
python -V
pip install -r requirements.txt
```

2. **Place your datasets**

- `train-val.csv` in the project root (used for training and validation)
- `test.csv` in the project root (used for final prediction)

3. **Run the pipeline**

```bash
python pipeline.py
```

This will execute:

- EDA and visualization
- Data preprocessing (imputation, encoding, scaling)
- Training of 7 classifiers with evaluation
- Optimization with GridSearchCV
- Prediction on test set and export to `predictions.csv`

---

## 📊 Models Trained

The pipeline evaluates the following classifiers:

- Gaussian Naive Bayes           (F1:0.70, Acc:0.67)
- K-Nearest Neighbors            (F1:0.71, Acc:0.75)
- Logistic Regression            (F1:0.83, Acc:0.84)
- Multi-layer Perceptron (MLP)   (F1:0.82, Acc:0.84)
- Support Vector Machine (SVC)   (F1:0.68, Acc:0.78)
- Decision Tree                  (F1:0.77, Acc:0.77)
- Random Forest                  (F1:0.82, Acc:0.84)

Each model is scored using **F1 Score** and **Accuracy**, with comparison plots for validation performance.

---

## Model Optimization

Grid search is performed for all classifiers with cross-validation, tuning parameters like:

- Number of neighbors (KNN)
- Regularization (LogReg)
- Hidden layers and learning rates (MLP)
- Tree depth and splits (Tree & Forest)

The best model is selected based on F1 score and used to predict on the test set.

Naive Bayes:          0.8103725873166571
KNeighborsClassifier: 0.7278787845763692
LogisticRegression:   0.8266027051135619
MLPClassifier:        0.8293428870644503
SVC:                  0.6800675935481126
DecisionTree:         0.786803708667071
RandomForest:         0.8278891397805659

---

## Output Files

- `data_splits.joblib`: Preprocessed training/validation data
- `classifiers.joblib`: Dictionary of trained classifiers
- `best_model.joblib`: The best-performing model
- `predictions.csv`: Test set predictions (submission-ready)

---


##  Notes

- This project is intended for educational purposes.
- Data preprocessing includes thoughtful handling of cyclic features, log transforms, and missing data imputation. Modification may introduce better results.
- The pipeline is modular and easily extensible.
- Access to a GPU is recommended.

---



