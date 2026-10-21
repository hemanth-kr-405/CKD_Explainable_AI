"""
Unit tests verifying the evaluation artifacts and mathematical consistency of the reported metrics.
"""

import json
import pytest
from src.config import (
    REPORTS_DIR,
    FIGURES_DIR,
    EVALUATION_REPORT_PATH
)


def test_evaluation_report_file_exists():
    """Verify that the JSON evaluation report was generated and is valid."""
    assert EVALUATION_REPORT_PATH.exists(), f"Evaluation report missing at {EVALUATION_REPORT_PATH}"
    with open(EVALUATION_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["model_name"] == "Decision Tree Classifier"
    assert "overall_test_metrics" in data
    assert "clinical_metrics" in data
    assert "baseline_comparison" in data


def test_evaluation_metric_mathematical_consistency():
    """Verify that reported rates precisely match the confusion matrix counts."""
    with open(EVALUATION_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = data["clinical_metrics"]
    tn, fp, fn, tp = cm["true_negatives"], cm["false_positives"], cm["false_negatives"], cm["true_positives"]
    total = tn + fp + fn + tp

    expected_acc = round((tp + tn) / total, 4)
    expected_sens = round(tp / (tp + fn), 4)
    expected_spec = round(tn / (tn + fp), 4)
    expected_ppv = round(tp / (tp + fp), 4)
    expected_npv = round(tn / (tn + fn), 4)

    assert data["overall_test_metrics"]["accuracy"] == expected_acc
    assert cm["sensitivity_recall_ckd"] == expected_sens
    assert cm["specificity_healthy"] == expected_spec
    assert cm["positive_predictive_value"] == expected_ppv
    assert cm["negative_predictive_value"] == expected_npv


def test_visualization_figures_generated():
    """Verify all required high-resolution visualization plots were created."""
    cm_plot = FIGURES_DIR / "confusion_matrix.png"
    roc_plot = FIGURES_DIR / "roc_curve.png"
    comparison_plot = FIGURES_DIR / "baseline_comparison.png"

    assert cm_plot.exists() and cm_plot.stat().st_size > 10000, "Confusion matrix plot missing or empty"
    assert roc_plot.exists() and roc_plot.stat().st_size > 10000, "ROC curve plot missing or empty"
    assert comparison_plot.exists() and comparison_plot.stat().st_size > 10000, "Baseline comparison plot missing or empty"
