# Model Performance Evaluation Report

**Model**: Decision Tree Classifier  
**Dataset**: UCI Chronic Kidney Disease (Stratified 80/20 Split)  
**Sample Sizes**: Train = 320 patients | Held-Out Test = 80 patients  

---

## 1. Executive Summary & Test Performance

| Metric | Score | Clinical Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **96.25%** | Proportion of all test predictions that were correct. |
| **Recall / Sensitivity (CKD)** | **94.00%** | Percentage of actual CKD patients correctly flagged (47 of 50). |
| **Specificity (Healthy)** | **100.00%** | Percentage of healthy Non-CKD patients correctly ruled out (30 of 30). |
| **Precision (PPV)** | **100.00%** | When the model predicts CKD, probability that the patient truly has CKD. |
| **Negative Predictive Value (NPV)** | **90.91%** | When the model predicts Non-CKD, probability that the patient is truly Non-CKD. |
| **F1-Score** | **96.91%** | Harmonic balance between Precision and Sensitivity. |
| **ROC-AUC** | **0.9890** | Area under the Receiver Operating Characteristic curve. |

---

## 2. Per-Class Performance Breakdown

| Class | Precision | Recall / Sensitivity | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Non-CKD (Class 0)** | 90.91% | 100.00% | 95.24% | 30.0 |
| **CKD (Class 1)** | 100.00% | 94.00% | 96.91% | 50.0 |
| **Macro Average** | 95.45% | 97.00% | 96.07% | 80.0 |
| **Weighted Average** | 96.59% | 96.25% | 96.28% | 80.0 |

---

## 3. Confusion Matrix Analysis & False Negative Risk

```
                         Actual Non-CKD (0)    Actual CKD (1)
Predicted Non-CKD (0):         30 (TN)               3 (FN)
Predicted CKD (1):              0 (FP)              47 (TP)
```

### Clinical Risk of False Negatives (3 Cases Observed)
- **Clinical Consequence**: In nephrology, a **False Negative** represents a patient who actively has kidney damage but is incorrectly told they are risk-free. This can delay critical early nephroprotective interventions (blood pressure control, SGLT2 inhibitors, dietary protein regulation), allowing renal damage to progress silently towards irreversible End-Stage Renal Disease (ESRD).
- **False Positives (0 Cases Observed)**: Zero healthy individuals were flagged incorrectly. While a false positive triggers follow-up lab tests (e.g. repeat serum creatinine or ultrasound), it carries far lower physical risk than a missed positive.

---

## 4. Stratified 5-Fold Cross-Validation (Training Set Only)

Cross-validation was performed on the 320 training samples to confirm generalization stability without touching the test set:
- **CV Mean Accuracy**: 95.00% (± 2.30%)
- **CV Mean Recall**: 96.00% (± 3.00%)
- **CV Mean F1-Score**: 95.99% (± 1.86%)
- **CV Mean ROC-AUC**: 0.9525 (± 0.0215)

---

## 5. Baseline Comparison Benchmarking

| Model Architecture | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Majority)** | 62.50% | 62.50% | 100.00% | 76.92% | 0.5000 |
| **Logistic Regression** | 97.50% | 100.00% | 96.00% | 97.96% | 1.0000 |
| **Decision Tree (Primary)** | 96.25% | 100.00% | 94.00% | 96.91% | 0.9890 |
| **Random Forest** | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |

---

## 6. Dataset Limitations & Ethical Boundaries

1. **Sample Size**: The dataset contains 400 patient records from Apollo Hospitals (India). While valuable as an academic benchmark, clinical deployment requires validation across diverse global populations and demographic cohorts.
2. **Missing Clinical History**: Key longitudinal indicators, such as serial eGFR trajectory over months, family history, and renal biopsy data, are absent from this single-timepoint dataset.
3. **Non-Diagnostic Scope**: High statistical accuracy does not equate to clinical diagnosis. This tool is designed strictly as a clinical decision support system, not an autonomous diagnostic agent.
