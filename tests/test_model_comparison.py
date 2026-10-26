"""
Unit and regression tests for Phase 2 Model Comparison & Benchmarking.
Validates:
1. All 4 candidate models exist in models/candidates/ and load cleanly.
2. Candidate models implement predict() and predict_proba().
3. Evaluation metrics artifact (model_comparison_results.json) contains all required metrics.
4. Evaluation report artifact (model_comparison.md) is generated.
5. Production model (models/ckd_model.joblib) is completely untouched.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pytest

from src.config import MODELS_DIR, REPORTS_DIR, ALL_FEATURE_NAMES

CANDIDATES_DIR = MODELS_DIR / "candidates"


def test_production_model_and_backup():
    """Verify active model is RandomForest and backup DecisionTree is fully intact."""
    prod_model_path = MODELS_DIR / "ckd_model.joblib"
    backup_model_path = MODELS_DIR / "ckd_model_decision_tree_backup.joblib"

    assert prod_model_path.exists(), "Production model ckd_model.joblib must exist."
    prod_model = joblib.load(prod_model_path)
    assert hasattr(prod_model, "predict")
    # Verify active production model is RandomForestClassifier
    assert prod_model.__class__.__name__ == "RandomForestClassifier"

    # Verify backup DecisionTree is preserved for instantaneous rollback
    assert backup_model_path.exists(), "Backup DecisionTree must exist."
    backup_model = joblib.load(backup_model_path)
    assert backup_model.__class__.__name__ == "DecisionTreeClassifier"
    assert hasattr(backup_model, "predict")


def test_candidate_models_exist_and_load():
    """Verify all 4 candidate models were serialized correctly."""
    expected_models = [
        ("decision_tree_baseline.joblib", "DecisionTreeClassifier"),
        ("logistic_regression.joblib", "Pipeline"),
        ("random_forest.joblib", "RandomForestClassifier"),
        ("xgboost.joblib", "XGBClassifier"),
    ]
    for filename, expected_cls in expected_models:
        path = CANDIDATES_DIR / filename
        assert path.exists(), f"Candidate artifact {filename} missing in {CANDIDATES_DIR}"
        model = joblib.load(path)
        assert model.__class__.__name__ == expected_cls
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")


def test_candidate_models_predict_synthetic_sample():
    """Verify each candidate model can make predictions on transformed sample vectors."""
    n_features = len(ALL_FEATURE_NAMES)
    sample_vector = np.zeros((1, n_features))

    candidate_files = [
        "decision_tree_baseline.joblib",
        "logistic_regression.joblib",
        "random_forest.joblib",
        "xgboost.joblib",
    ]
    for filename in candidate_files:
        model = joblib.load(CANDIDATES_DIR / filename)
        pred = model.predict(sample_vector)
        probs = model.predict_proba(sample_vector)

        assert pred.shape == (1,)
        assert probs.shape == (1, 2)
        assert 0.0 <= probs[0, 0] <= 1.0
        assert 0.0 <= probs[0, 1] <= 1.0
        assert np.isclose(probs[0, 0] + probs[0, 1], 1.0, atol=1e-4)


def test_comparison_report_artifacts_complete():
    """Verify reports/model_comparison_results.json has all required algorithms and metrics."""
    json_path = REPORTS_DIR / "model_comparison_results.json"
    assert json_path.exists(), "reports/model_comparison_results.json must exist."

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_models = [
        "Decision Tree (Baseline)",
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    ]
    for m in required_models:
        assert m in data["models"], f"Model {m} missing from comparison results"

        model_entry = data["models"][m]
        # Check test metrics
        tm = model_entry["test_metrics"]
        for metric in ["accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc"]:
            assert metric in tm, f"Metric {metric} missing from test metrics of {m}"
            assert 0.0 <= tm[metric] <= 1.0

        # Check confusion matrix
        cm = tm["confusion_matrix"]
        for cm_key in ["true_negatives", "false_positives", "false_negatives", "true_positives"]:
            assert cm_key in cm
            assert isinstance(cm[cm_key], int)

        # Check cross-validation metrics
        cv = model_entry["cross_validation_5fold"]
        for metric in ["accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc"]:
            assert f"{metric}_mean" in cv
            assert f"{metric}_std" in cv
            assert 0.0 <= cv[f"{metric}_mean"] <= 1.0

    # Markdown report existence
    md_path = REPORTS_DIR / "model_comparison.md"
    assert md_path.exists(), "reports/model_comparison.md must exist."
