# Chronic Kidney Disease (CKD) Model Benchmarking & Empirical Comparison

**Experiment Date:** 2026-10-26  
**Dataset:** UCI Chronic Kidney Disease ($N=400$, 24 clinical features)  
**Evaluation Strategy:** Stratified 5-Fold Cross-Validation on Training Set ($N=320$) + Held-Out Test Set ($N=80$)  
**Random Seed:** 42 (strictly reproducible)

---

## 1. Held-Out Test Cohort Performance ($N=80$: 50 CKD, 30 Non-CKD)

| Model | Accuracy | Recall (Sensitivity) | Specificity | Precision | F1-Score | ROC-AUC | Confusion Matrix (TN, FP, FN, TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Baseline)** | 96.25% | 94.00% | 100.00% | 100.00% | 0.9691 | 0.9890 | `TN=30, FP=0, FN=3, TP=47` |
| **Logistic Regression** | 97.50% | 96.00% | 100.00% | 100.00% | 0.9796 | 0.9993 | `TN=30, FP=0, FN=2, TP=48` |
| **Random Forest** | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 | 1.0000 | `TN=30, FP=0, FN=0, TP=50` |
| **XGBoost** | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 | 1.0000 | `TN=30, FP=0, FN=0, TP=50` |

---

## 2. Stratified 5-Fold Cross-Validation Performance (Mean $\pm$ Std on Training Set $N=320$)

| Model | CV Accuracy | CV Recall (Sensitivity) | CV Specificity | CV Precision | CV F1-Score | CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Baseline)** | 95.00% $\pm$ 2.30% | 96.00% $\pm$ 3.00% | 93.33% $\pm$ 3.33% | 96.03% $\pm$ 1.88% | 0.9599 $\pm$ 0.0187 | 0.9525 $\pm$ 0.0215 |
| **Logistic Regression** | 99.38% $\pm$ 0.76% | 99.00% $\pm$ 1.22% | 100.00% $\pm$ 0.00% | 100.00% $\pm$ 0.00% | 0.9949 $\pm$ 0.0062 | 1.0000 $\pm$ 0.0000 |
| **Random Forest** | 99.06% $\pm$ 1.25% | 99.50% $\pm$ 1.00% | 98.33% $\pm$ 2.04% | 99.01% $\pm$ 1.21% | 0.9925 $\pm$ 0.0100 | 0.9994 $\pm$ 0.0008 |
| **XGBoost** | 97.81% $\pm$ 1.59% | 98.00% $\pm$ 1.00% | 97.50% $\pm$ 3.33% | 98.52% $\pm$ 1.96% | 0.9825 $\pm$ 0.0126 | 0.9990 $\pm$ 0.0011 |

---

## 3. Tuned Hyperparameters

- **Decision Tree (Baseline)**: `{"criterion": "gini", "max_depth": 5, "min_samples_split": 4, "min_samples_leaf": 2, "class_weight": "balanced", "random_state": 42}`
- **Logistic Regression**: `{"C": 10.0, "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced"}`
- **Random Forest**: `{"max_depth": 3, "min_samples_leaf": 1, "min_samples_split": 2, "n_estimators": 100, "class_weight": "balanced"}`
- **XGBoost**: `{"colsample_bytree": 1.0, "learning_rate": 0.05, "max_depth": 2, "n_estimators": 150, "subsample": 1.0, "scale_pos_weight": 0.6}`