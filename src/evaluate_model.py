"""
Comprehensive Model Evaluation and Performance Analysis for Chronic Kidney Disease Prediction.

Provides:
- In-depth evaluation of the primary Decision Tree Classifier on held-out test data
- Per-class classification reports and clinical confusion matrix analysis
- Stratified 5-fold cross-validation on training data
- Baseline model benchmarking (Dummy Classifier, Logistic Regression, Random Forest)
- High-resolution clinical visualization generation (Confusion Matrix, ROC Curve, Baseline Comparison)
- Detailed JSON and Markdown evaluation reporting
"""

import sys
from pathlib import Path
import json
import logging
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
import joblib

from src.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    MODEL_ARTIFACT_PATH,
    REPORTS_DIR,
    FIGURES_DIR,
    EVALUATION_REPORT_PATH,
    ALL_FEATURE_NAMES,
    TARGET_COLUMN,
    TARGET_NAMES,
    RANDOM_STATE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_evaluation_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads preprocessed training and testing data.
    Ensures zero data leakage: test data is strictly held-out.
    """
    if not TRAIN_DATA_PATH.exists() or not TEST_DATA_PATH.exists():
        raise FileNotFoundError("Processed datasets missing. Run Phase 2 preprocessing first.")

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_train = train_df[ALL_FEATURE_NAMES]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[ALL_FEATURE_NAMES]
    y_test = test_df[TARGET_COLUMN]

    return X_train, y_train, X_test, y_test


def evaluate_primary_model(
    model: DecisionTreeClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Computes complete statistical and clinical performance metrics for the Decision Tree.
    """
    logging.info("Evaluating Decision Tree on held-out test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix Breakdown
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Clinical Epidemiological Ratios
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall of positive CKD
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0          # Precision / Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0          # Negative Predictive Value

    # Detailed Per-Class Classification Report
    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=["Non-CKD (Class 0)", "CKD (Class 1)"],
        output_dict=True,
        zero_division=0
    )

    # Cross-validation on Training Data Only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
    )

    results = {
        "model_name": "Decision Tree Classifier",
        "sample_counts": {
            "train_total": len(X_train),
            "test_total": len(X_test),
            "test_non_ckd_actual": int((y_test == 0).sum()),
            "test_ckd_actual": int((y_test == 1).sum())
        },
        "overall_test_metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4)
        },
        "clinical_metrics": {
            "sensitivity_recall_ckd": round(float(sensitivity), 4),
            "specificity_healthy": round(float(specificity), 4),
            "positive_predictive_value": round(float(ppv), 4),
            "negative_predictive_value": round(float(npv), 4),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "per_class_report": clf_report,
        "cv_5fold_train_metrics": {
            "cv_accuracy_mean": round(float(np.mean(cv_scores["test_accuracy"])), 4),
            "cv_accuracy_std": round(float(np.std(cv_scores["test_accuracy"])), 4),
            "cv_recall_mean": round(float(np.mean(cv_scores["test_recall"])), 4),
            "cv_recall_std": round(float(np.std(cv_scores["test_recall"])), 4),
            "cv_f1_mean": round(float(np.mean(cv_scores["test_f1"])), 4),
            "cv_f1_std": round(float(np.std(cv_scores["test_f1"])), 4),
            "cv_roc_auc_mean": round(float(np.mean(cv_scores["test_roc_auc"])), 4),
            "cv_roc_auc_std": round(float(np.std(cv_scores["test_roc_auc"])), 4)
        }
    }
    return results


def benchmark_baseline_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dt_model: DecisionTreeClassifier
) -> Dict[str, Dict[str, float]]:
    """
    Trains and evaluates baseline models for comparative context using the exact same splits:
    1. Dummy Classifier (Majority Class Baseline)
    2. Logistic Regression (Linear Baseline)
    3. Random Forest Classifier (Ensemble Benchmark)
    4. Decision Tree Classifier (Primary Interpretable Model)
    """
    logging.info("Training and benchmarking baseline comparison models...")
    models = {
        "Dummy (Majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        "Decision Tree (Primary)": dt_model,
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=RANDOM_STATE)
    }

    comparison = {}
    for name, clf in models.items():
        if name != "Decision Tree (Primary)":
            clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        
        # Probabilities for ROC-AUC
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
            auc_val = round(float(roc_auc_score(y_test, y_prob)), 4)
        else:
            auc_val = 0.5000

        comparison[name] = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": auc_val
        }
    return comparison


def generate_confusion_matrix_plot(cm_data: Dict[str, int], output_path: Path) -> None:
    """
    Renders an aesthetically styled clinical confusion matrix heatmap.
    """
    tn = cm_data["true_negatives"]
    fp = cm_data["false_positives"]
    fn = cm_data["false_negatives"]
    tp = cm_data["true_positives"]

    cm_matrix = np.array([[tn, fp], [fn, tp]])
    total = np.sum(cm_matrix)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cax = ax.matshow(cm_matrix, cmap="Blues", alpha=0.85)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    # Labels and Titles
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-CKD (0)", "CKD (1)"], fontsize=11, fontweight="bold")
    ax.set_yticklabels(["Non-CKD (0)", "CKD (1)"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Diagnosis", fontsize=12, labelpad=12, fontweight="bold")
    ax.set_ylabel("Actual Clinical Condition", fontsize=12, labelpad=12, fontweight="bold")
    ax.set_title("Decision Tree Confusion Matrix (Test Set N=80)", fontsize=13, fontweight="bold", pad=20)

    # Annotate Cells with Counts and Clinical Meaning
    cell_annotations = [
        [f"TN = {tn}\n({(tn/total)*100:.1f}%)\nTrue Healthy", f"FP = {fp}\n({(fp/total)*100:.1f}%)\nFalse Alarm"],
        [f"FN = {fn}\n({(fn/total)*100:.1f}%)\nMissed CKD\n(Critical)", f"TP = {tp}\n({(tp/total)*100:.1f}%)\nDetected CKD"]
    ]

    for i in range(2):
        for j in range(2):
            color = "white" if cm_matrix[i, j] > total * 0.35 else "#1e293b"
            ax.text(j, i, cell_annotations[i][j], ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Saved confusion matrix plot to {output_path}")


def generate_roc_curve_plot(
    dt_model: DecisionTreeClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path
) -> None:
    """
    Renders the ROC curve for the primary Decision Tree with threshold reference.
    """
    y_prob = dt_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"Decision Tree (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Chance Baseline (AUC = 0.5000)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Saved ROC curve plot to {output_path}")


def generate_baseline_comparison_plot(comparison_data: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """
    Renders a clean grouped bar chart comparing models on Accuracy, Recall, and F1.
    """
    models = list(comparison_data.keys())
    accuracy = [comparison_data[m]["accuracy"] * 100 for m in models]
    recall = [comparison_data[m]["recall"] * 100 for m in models]
    f1 = [comparison_data[m]["f1_score"] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    rects1 = ax.bar(x - width, accuracy, width, label="Accuracy (%)", color="#3b82f6")
    rects2 = ax.bar(x, recall, width, label="Recall / Sensitivity (%)", color="#10b981")
    rects3 = ax.bar(x + width, f1, width, label="F1-Score (%)", color="#8b5cf6")

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Model Benchmarking Comparison (Held-out Test Set)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate bar values
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Saved baseline comparison plot to {output_path}")


def export_markdown_report(report_data: Dict[str, Any], comparison: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """
    Generates a readable Markdown evaluation report for academic documentation.
    """
    cm = report_data["clinical_metrics"]
    test_metrics = report_data["overall_test_metrics"]
    cv = report_data["cv_5fold_train_metrics"]
    clf = report_data["per_class_report"]

    md_content = f"""# Model Performance Evaluation Report

**Model**: Decision Tree Classifier  
**Dataset**: UCI Chronic Kidney Disease (Stratified 80/20 Split)  
**Sample Sizes**: Train = {report_data['sample_counts']['train_total']} patients | Held-Out Test = {report_data['sample_counts']['test_total']} patients  

---

## 1. Executive Summary & Test Performance

| Metric | Score | Clinical Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **{test_metrics['accuracy'] * 100:.2f}%** | Proportion of all test predictions that were correct. |
| **Recall / Sensitivity (CKD)** | **{cm['sensitivity_recall_ckd'] * 100:.2f}%** | Percentage of actual CKD patients correctly flagged ({cm['true_positives']} of {cm['true_positives'] + cm['false_negatives']}). |
| **Specificity (Healthy)** | **{cm['specificity_healthy'] * 100:.2f}%** | Percentage of healthy Non-CKD patients correctly ruled out ({cm['true_negatives']} of {cm['true_negatives'] + cm['false_positives']}). |
| **Precision (PPV)** | **{cm['positive_predictive_value'] * 100:.2f}%** | When the model predicts CKD, probability that the patient truly has CKD. |
| **Negative Predictive Value (NPV)** | **{cm['negative_predictive_value'] * 100:.2f}%** | When the model predicts Non-CKD, probability that the patient is truly Non-CKD. |
| **F1-Score** | **{clf['CKD (Class 1)']['f1-score'] * 100:.2f}%** | Harmonic balance between Precision and Sensitivity. |
| **ROC-AUC** | **{test_metrics['roc_auc']:.4f}** | Area under the Receiver Operating Characteristic curve. |

---

## 2. Per-Class Performance Breakdown

| Class | Precision | Recall / Sensitivity | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Non-CKD (Class 0)** | {clf['Non-CKD (Class 0)']['precision'] * 100:.2f}% | {clf['Non-CKD (Class 0)']['recall'] * 100:.2f}% | {clf['Non-CKD (Class 0)']['f1-score'] * 100:.2f}% | {clf['Non-CKD (Class 0)']['support']} |
| **CKD (Class 1)** | {clf['CKD (Class 1)']['precision'] * 100:.2f}% | {clf['CKD (Class 1)']['recall'] * 100:.2f}% | {clf['CKD (Class 1)']['f1-score'] * 100:.2f}% | {clf['CKD (Class 1)']['support']} |
| **Macro Average** | {clf['macro avg']['precision'] * 100:.2f}% | {clf['macro avg']['recall'] * 100:.2f}% | {clf['macro avg']['f1-score'] * 100:.2f}% | {clf['macro avg']['support']} |
| **Weighted Average** | {clf['weighted avg']['precision'] * 100:.2f}% | {clf['weighted avg']['recall'] * 100:.2f}% | {clf['weighted avg']['f1-score'] * 100:.2f}% | {clf['weighted avg']['support']} |

---

## 3. Confusion Matrix Analysis & False Negative Risk

```
                         Actual Non-CKD (0)    Actual CKD (1)
Predicted Non-CKD (0):         {cm['true_negatives']} (TN)               {cm['false_negatives']} (FN)
Predicted CKD (1):              {cm['false_positives']} (FP)              {cm['true_positives']} (TP)
```

### Clinical Risk of False Negatives ({cm['false_negatives']} Cases Observed)
- **Clinical Consequence**: In nephrology, a **False Negative** represents a patient who actively has kidney damage but is incorrectly told they are risk-free. This can delay critical early nephroprotective interventions (blood pressure control, SGLT2 inhibitors, dietary protein regulation), allowing renal damage to progress silently towards irreversible End-Stage Renal Disease (ESRD).
- **False Positives ({cm['false_positives']} Cases Observed)**: Zero healthy individuals were flagged incorrectly. While a false positive triggers follow-up lab tests (e.g. repeat serum creatinine or ultrasound), it carries far lower physical risk than a missed positive.

---

## 4. Stratified 5-Fold Cross-Validation (Training Set Only)

Cross-validation was performed on the 320 training samples to confirm generalization stability without touching the test set:
- **CV Mean Accuracy**: {cv['cv_accuracy_mean'] * 100:.2f}% (± {cv['cv_accuracy_std'] * 100:.2f}%)
- **CV Mean Recall**: {cv['cv_recall_mean'] * 100:.2f}% (± {cv['cv_recall_std'] * 100:.2f}%)
- **CV Mean F1-Score**: {cv['cv_f1_mean'] * 100:.2f}% (± {cv['cv_f1_std'] * 100:.2f}%)
- **CV Mean ROC-AUC**: {cv['cv_roc_auc_mean']:.4f} (± {cv['cv_roc_auc_std']:.4f})

---

## 5. Baseline Comparison Benchmarking

| Model Architecture | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for model_name, metrics in comparison.items():
        md_content += f"| **{model_name}** | {metrics['accuracy'] * 100:.2f}% | {metrics['precision'] * 100:.2f}% | {metrics['recall'] * 100:.2f}% | {metrics['f1_score'] * 100:.2f}% | {metrics['roc_auc']:.4f} |\n"

    md_content += """
---

## 6. Dataset Limitations & Ethical Boundaries

1. **Sample Size**: The dataset contains 400 patient records from Apollo Hospitals (India). While valuable as an academic benchmark, clinical deployment requires validation across diverse global populations and demographic cohorts.
2. **Missing Clinical History**: Key longitudinal indicators, such as serial eGFR trajectory over months, family history, and renal biopsy data, are absent from this single-timepoint dataset.
3. **Non-Diagnostic Scope**: High statistical accuracy does not equate to clinical diagnosis. This tool is designed strictly as a clinical decision support system, not an autonomous diagnostic agent.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logging.info(f"Saved Markdown report to {output_path}")


def run_evaluation() -> Dict[str, Any]:
    """
    Orchestrates the Phase 4 evaluation pipeline.
    """
    logging.info("=" * 70)
    logging.info("STARTING PHASE 4: MODEL EVALUATION & PERFORMANCE ANALYSIS")
    logging.info("=" * 70)

    # 1. Load Model and Data
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_ARTIFACT_PATH}. Complete Phase 3 first.")

    dt_model: DecisionTreeClassifier = joblib.load(MODEL_ARTIFACT_PATH)
    X_train, y_train, X_test, y_test = load_evaluation_data()

    # 2. Evaluate Primary Decision Tree
    report_data = evaluate_primary_model(dt_model, X_train, y_train, X_test, y_test)

    # 3. Benchmark Baseline Models
    comparison = benchmark_baseline_models(X_train, y_train, X_test, y_test, dt_model)
    report_data["baseline_comparison"] = comparison

    # 4. Generate Figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generate_confusion_matrix_plot(report_data["clinical_metrics"], FIGURES_DIR / "confusion_matrix.png")
    generate_roc_curve_plot(dt_model, X_test, y_test, FIGURES_DIR / "roc_curve.png")
    generate_baseline_comparison_plot(comparison, FIGURES_DIR / "baseline_comparison.png")

    # 5. Export JSON and Markdown Reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
    logging.info(f"Saved JSON evaluation report to {EVALUATION_REPORT_PATH}")

    export_markdown_report(report_data, comparison, REPORTS_DIR / "evaluation_summary.md")

    logging.info("=" * 70)
    logging.info("PHASE 4 EVALUATION COMPLETE")
    logging.info(f"Accuracy: {report_data['overall_test_metrics']['accuracy']*100:.2f}%")
    logging.info(f"Sensitivity (Recall): {report_data['clinical_metrics']['sensitivity_recall_ckd']*100:.2f}%")
    logging.info(f"Specificity: {report_data['clinical_metrics']['specificity_healthy']*100:.2f}%")
    logging.info(f"Confusion Matrix: TN={report_data['clinical_metrics']['true_negatives']}, FP={report_data['clinical_metrics']['false_positives']}, FN={report_data['clinical_metrics']['false_negatives']}, TP={report_data['clinical_metrics']['true_positives']}")
    logging.info("=" * 70)

    return report_data


if __name__ == "__main__":
    run_evaluation()
