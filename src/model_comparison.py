"""
Experimental Model Comparison Module for Chronic Kidney Disease (CKD) Prediction.

Phase 2: Rigorous empirical evaluation comparing:
1. Decision Tree (Current Production Baseline)
2. Logistic Regression (Linear L2 Regularized with StandardScaler)
3. Random Forest Classifier (Bagging Ensemble)
4. XGBoost Classifier (Gradient Boosted Decision Trees)

Methodological Guarantees:
- Strict train-test separation: 80% train (N=320), 20% held-out test (N=80), stratified by class.
- Zero data leakage: CKDPreprocessor and StandardScaler are fit strictly on training splits.
- In-fold cross-validation: Preprocessing is fit strictly within each training fold during 5-fold CV.
- Inner cross-validation hyperparameter tuning: No test data is touched during parameter selection.
- Comprehensive metrics: Accuracy, Precision, Recall (Sensitivity), Specificity, F1-score, ROC-AUC,
  and Confusion Matrix (TN, FP, FN, TP).
- Reproducibility: Deterministic random seeds (seed=42) and recorded environment metadata.
- Production isolation: The existing production Decision Tree artifact (models/ckd_model.joblib)
  is preserved and remains untouched.
"""

import sys
from pathlib import Path
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import joblib
import xgboost as xgb

from src.config import (
    RAW_DATA_PATH,
    ALL_FEATURE_NAMES,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
    RANDOM_STATE,
    REPORTS_DIR,
    MODELS_DIR
)
from src.preprocessing import CKDPreprocessor, load_raw_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ckd.model_comparison")

CANDIDATES_DIR = MODELS_DIR / "candidates"


def calculate_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Computes all standard and clinical diagnostic performance metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision), 4),
        "recall_sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }


def run_leak_free_cross_validation(
    model_factory,
    X_raw: pd.DataFrame,
    y: pd.Series,
    is_linear: bool = False,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE
) -> Dict[str, Any]:
    """
    Runs 5-fold Stratified Cross-Validation with strict leak prevention:
    In each fold, the CKDPreprocessor (and StandardScaler for linear models)
    is fit ONLY on the training fold, then used to transform the validation fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_raw, y), start=1):
        X_train_fold_raw = X_raw.iloc[train_idx].copy()
        y_train_fold = y.iloc[train_idx].values
        X_val_fold_raw = X_raw.iloc[val_idx].copy()
        y_val_fold = y.iloc[val_idx].values

        # 1. Fit preprocessor strictly on this fold's training instances
        fold_preprocessor = CKDPreprocessor()
        fold_preprocessor.fit(X_train_fold_raw)

        X_train_proc = fold_preprocessor.transform(X_train_fold_raw)
        X_val_proc = fold_preprocessor.transform(X_val_fold_raw)

        # 2. If linear model, scale features within fold
        if is_linear:
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train_proc)
            X_val_proc = scaler.transform(X_val_proc)

        # 3. Instantiate and fit model
        model = model_factory()
        model.fit(X_train_proc, y_train_fold)

        # 4. Predict on held-out validation fold
        y_pred = model.predict(X_val_proc)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val_proc)[:, 1]
        else:
            y_prob = y_pred.astype(float)

        metrics = calculate_clinical_metrics(y_val_fold, y_pred, y_prob)
        fold_metrics.append(metrics)

    # Compute mean and standard deviation across folds
    metric_keys = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc"]
    summary = {}
    for k in metric_keys:
        vals = [fm[k] for fm in fold_metrics]
        summary[f"{k}_mean"] = round(float(np.mean(vals)), 4)
        summary[f"{k}_std"] = round(float(np.std(vals)), 4)

    summary["per_fold_details"] = fold_metrics
    return summary


def tune_and_evaluate_model(
    model_name: str,
    base_estimator,
    param_grid: Dict[str, List[Any]],
    X_train_processed: pd.DataFrame,
    y_train: pd.Series,
    X_test_processed: pd.DataFrame,
    y_test: pd.Series,
    is_linear: bool = False,
    cv_n_splits: int = 5
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Performs grid-search hyperparameter tuning using inner 5-fold Stratified CV on training data,
    then evaluates the best refitted estimator on the held-out test set.
    """
    logger.info(f"Tuning hyperparameters for {model_name}...")

    inner_cv = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=RANDOM_STATE)

    if is_linear:
        # Pipeline with StandardScaler
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", base_estimator)
        ])
        grid_params = {f"clf__{k}": v for k, v in param_grid.items()}
        grid = GridSearchCV(
            pipe,
            grid_params,
            cv=inner_cv,
            scoring="f1",
            n_jobs=-1,
            refit=True
        )
        grid.fit(X_train_processed, y_train)
        best_model = grid.best_estimator_
        best_params = {k.replace("clf__", ""): v for k, v in grid.best_params_.items()}
    else:
        grid = GridSearchCV(
            base_estimator,
            param_grid,
            cv=inner_cv,
            scoring="f1",
            n_jobs=-1,
            refit=True
        )
        grid.fit(X_train_processed, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_

    logger.info(f"{model_name} Best Params: {best_params} (Inner CV F1: {grid.best_score_:.4f})")

    # Evaluate best model on the held-out test set
    y_pred = best_model.predict(X_test_processed)
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test_processed)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    test_metrics = calculate_clinical_metrics(y_test.values, y_pred, y_prob)

    return best_model, best_params, test_metrics


def run_experiment_pipeline() -> Dict[str, Any]:
    """
    Executes the comprehensive Phase 2 experimental comparison across all 4 algorithms.
    """
    logger.info("=" * 80)
    logger.info("PHASE 2: EMPIRICAL ALGORITHM COMPARISON & BENCHMARKING")
    logger.info("=" * 80)

    # 1. Load authentic raw data
    raw_df = load_raw_data(RAW_DATA_PATH)
    X_raw = raw_df[ALL_FEATURE_NAMES].copy()
    y_raw = raw_df[TARGET_COLUMN].map(TARGET_MAPPING)

    # 2. Strict Stratified Train / Test Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_raw
    )
    logger.info(f"Dataset Split: Train={len(X_train_raw)} (CKD={sum(y_train==1)}, Non-CKD={sum(y_train==0)}) | "
                f"Test={len(X_test_raw)} (CKD={sum(y_test==1)}, Non-CKD={sum(y_test==0)})")

    # 3. Fit Preprocessor EXCLUSIVELY on training features for the held-out test evaluation
    train_preprocessor = CKDPreprocessor()
    train_preprocessor.fit(X_train_raw)
    X_train_proc = train_preprocessor.transform(X_train_raw)
    X_test_proc = train_preprocessor.transform(X_test_raw)

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "metadata": {
            "experiment_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": RANDOM_STATE,
            "test_split_ratio": TEST_SIZE,
            "train_samples": len(X_train_raw),
            "test_samples": len(X_test_raw),
            "class_distribution_overall": {"ckd": int((y_raw == 1).sum()), "notckd": int((y_raw == 0).sum())},
            "class_distribution_train": {"ckd": int((y_train == 1).sum()), "notckd": int((y_train == 0).sum())},
            "class_distribution_test": {"ckd": int((y_test == 1).sum()), "notckd": int((y_test == 0).sum())},
            "dependencies": {
                "python": sys.version.split()[0],
                "scikit-learn": joblib.__version__,
                "xgboost": xgb.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__
            }
        },
        "models": {}
    }

    # ------------------------------------------------------------------------
    # Algorithm 1: Decision Tree (Baseline)
    # ------------------------------------------------------------------------
    logger.info("\n--- Evaluating Model 1: Decision Tree (Baseline) ---")
    dt_baseline_params = {
        "criterion": "gini",
        "max_depth": 5,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE
    }

    def dt_factory():
        return DecisionTreeClassifier(**dt_baseline_params)

    # Leak-free cross-validation on training partition
    dt_cv = run_leak_free_cross_validation(dt_factory, X_train_raw, y_train, is_linear=False)

    # Train on full X_train_proc and evaluate on X_test_proc
    dt_model = dt_factory()
    dt_model.fit(X_train_proc, y_train)
    y_pred_dt = dt_model.predict(X_test_proc)
    y_prob_dt = dt_model.predict_proba(X_test_proc)[:, 1]
    dt_test = calculate_clinical_metrics(y_test.values, y_pred_dt, y_prob_dt)

    # Save candidate artifact
    joblib.dump(dt_model, CANDIDATES_DIR / "decision_tree_baseline.joblib")

    results["models"]["Decision Tree (Baseline)"] = {
        "model_type": "DecisionTreeClassifier",
        "is_baseline": True,
        "hyperparameters": dt_baseline_params,
        "cross_validation_5fold": dt_cv,
        "test_metrics": dt_test
    }

    # ------------------------------------------------------------------------
    # Algorithm 2: Logistic Regression (L2 Regularized + StandardScaler)
    # ------------------------------------------------------------------------
    logger.info("\n--- Evaluating Model 2: Logistic Regression ---")
    lr_param_grid = {
        "C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    }
    lr_base = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    # Tune hyperparameters on X_train_proc
    lr_best_model, lr_best_params, lr_test = tune_and_evaluate_model(
        "Logistic Regression",
        lr_base,
        lr_param_grid,
        X_train_proc,
        y_train,
        X_test_proc,
        y_test,
        is_linear=True
    )

    # Leak-free CV with the tuned parameter
    def lr_factory():
        return LogisticRegression(
            C=lr_best_params.get("C", 1.0),
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE
        )

    lr_cv = run_leak_free_cross_validation(lr_factory, X_train_raw, y_train, is_linear=True)

    joblib.dump(lr_best_model, CANDIDATES_DIR / "logistic_regression.joblib")

    results["models"]["Logistic Regression"] = {
        "model_type": "LogisticRegression (with StandardScaler)",
        "is_baseline": False,
        "hyperparameters": {**lr_best_params, "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced"},
        "cross_validation_5fold": lr_cv,
        "test_metrics": lr_test
    }

    # ------------------------------------------------------------------------
    # Algorithm 3: Random Forest Classifier
    # ------------------------------------------------------------------------
    logger.info("\n--- Evaluating Model 3: Random Forest ---")
    rf_param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 8],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2]
    }
    rf_base = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    rf_best_model, rf_best_params, rf_test = tune_and_evaluate_model(
        "Random Forest",
        rf_base,
        rf_param_grid,
        X_train_proc,
        y_train,
        X_test_proc,
        y_test,
        is_linear=False
    )

    def rf_factory():
        return RandomForestClassifier(
            **rf_best_params,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

    rf_cv = run_leak_free_cross_validation(rf_factory, X_train_raw, y_train, is_linear=False)

    joblib.dump(rf_best_model, CANDIDATES_DIR / "random_forest.joblib")

    results["models"]["Random Forest"] = {
        "model_type": "RandomForestClassifier",
        "is_baseline": False,
        "hyperparameters": {**rf_best_params, "class_weight": "balanced"},
        "cross_validation_5fold": rf_cv,
        "test_metrics": rf_test
    }

    # ------------------------------------------------------------------------
    # Algorithm 4: XGBoost Classifier
    # ------------------------------------------------------------------------
    logger.info("\n--- Evaluating Model 4: XGBoost ---")
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)  # 120 / 200 = 0.60

    xgb_param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    xgb_base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE
    )

    xgb_best_model, xgb_best_params, xgb_test = tune_and_evaluate_model(
        "XGBoost",
        xgb_base,
        xgb_param_grid,
        X_train_proc,
        y_train,
        X_test_proc,
        y_test,
        is_linear=False
    )

    def xgb_factory():
        return xgb.XGBClassifier(
            **xgb_best_params,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE
        )

    xgb_cv = run_leak_free_cross_validation(xgb_factory, X_train_raw, y_train, is_linear=False)

    joblib.dump(xgb_best_model, CANDIDATES_DIR / "xgboost.joblib")

    results["models"]["XGBoost"] = {
        "model_type": "XGBClassifier",
        "is_baseline": False,
        "hyperparameters": {**xgb_best_params, "scale_pos_weight": round(scale_pos_weight, 4)},
        "cross_validation_5fold": xgb_cv,
        "test_metrics": xgb_test
    }

    # 4. Save JSON results report
    json_path = REPORTS_DIR / "model_comparison_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info(f"\nDetailed comparison metrics saved to {json_path}")

    # 5. Generate formatted Markdown report
    md_path = REPORTS_DIR / "model_comparison.md"
    generate_markdown_summary(results, md_path)
    logger.info(f"Markdown comparative analysis saved to {md_path}")

    return results


def generate_markdown_summary(results: Dict[str, Any], output_path: Path) -> None:
    """
    Creates an academic comparison table and clinical analysis document.
    """
    models_data = results["models"]

    md = []
    md.append("# Chronic Kidney Disease (CKD) Model Benchmarking & Empirical Comparison\n")
    md.append(f"**Experiment Date:** {results['metadata']['experiment_timestamp_utc'][:10]}  ")
    md.append(f"**Dataset:** UCI Chronic Kidney Disease ($N=400$, 24 clinical features)  ")
    md.append(f"**Evaluation Strategy:** Stratified 5-Fold Cross-Validation on Training Set ($N=320$) + Held-Out Test Set ($N=80$)  ")
    md.append(f"**Random Seed:** {results['metadata']['random_seed']} (strictly reproducible)\n")
    md.append("---\n")

    md.append("## 1. Held-Out Test Cohort Performance ($N=80$: 50 CKD, 30 Non-CKD)\n")
    md.append("| Model | Accuracy | Recall (Sensitivity) | Specificity | Precision | F1-Score | ROC-AUC | Confusion Matrix (TN, FP, FN, TP) |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for name, data in models_data.items():
        tm = data["test_metrics"]
        cm = tm["confusion_matrix"]
        cm_str = f"TN={cm['true_negatives']}, FP={cm['false_positives']}, FN={cm['false_negatives']}, TP={cm['true_positives']}"
        md.append(
            f"| **{name}** | {tm['accuracy']*100:.2f}% | {tm['recall_sensitivity']*100:.2f}% | "
            f"{tm['specificity']*100:.2f}% | {tm['precision']*100:.2f}% | {tm['f1_score']:.4f} | "
            f"{tm['roc_auc']:.4f} | `{cm_str}` |"
        )

    md.append("\n---\n")
    md.append("## 2. Stratified 5-Fold Cross-Validation Performance (Mean $\\pm$ Std on Training Set $N=320$)\n")
    md.append("| Model | CV Accuracy | CV Recall (Sensitivity) | CV Specificity | CV Precision | CV F1-Score | CV ROC-AUC |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for name, data in models_data.items():
        cv = data["cross_validation_5fold"]
        md.append(
            f"| **{name}** | {cv['accuracy_mean']*100:.2f}% $\\pm$ {cv['accuracy_std']*100:.2f}% | "
            f"{cv['recall_sensitivity_mean']*100:.2f}% $\\pm$ {cv['recall_sensitivity_std']*100:.2f}% | "
            f"{cv['specificity_mean']*100:.2f}% $\\pm$ {cv['specificity_std']*100:.2f}% | "
            f"{cv['precision_mean']*100:.2f}% $\\pm$ {cv['precision_std']*100:.2f}% | "
            f"{cv['f1_score_mean']:.4f} $\\pm$ {cv['f1_score_std']:.4f} | "
            f"{cv['roc_auc_mean']:.4f} $\\pm$ {cv['roc_auc_std']:.4f} |"
        )

    md.append("\n---\n")
    md.append("## 3. Tuned Hyperparameters\n")
    for name, data in models_data.items():
        md.append(f"- **{name}**: `{json.dumps(data['hyperparameters'])}`")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    run_experiment_pipeline()
