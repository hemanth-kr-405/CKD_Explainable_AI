"""
Unit and integration tests for the SHAP explainability engine (CKDExplainer).

Verifies:
- Proper initialization and positive-class index resolution
- Global SHAP values calculation and feature dimension alignment
- Individual patient explanation schema and directional attribution
- Mathematical additivity: Base Value + Sum(SHAP values) == Model Output
- Waterfall plot generation and base64 encoding
- Resilience to partial/missing clinical inputs
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    MODEL_ARTIFACT_PATH,
    PREPROCESSOR_ARTIFACT_PATH,
    TEST_DATA_PATH,
    ALL_FEATURE_NAMES,
    FIGURES_DIR
)
from src.explain_predictions import CKDExplainer


@pytest.fixture(scope="module")
def explainer():
    """Module-level fixture to instantiate CKDExplainer once."""
    return CKDExplainer(MODEL_ARTIFACT_PATH, PREPROCESSOR_ARTIFACT_PATH)


def test_explainer_initialization(explainer):
    """Verify explainer loads correctly and resolves positive class index."""
    assert explainer.model is not None
    assert explainer.preprocessor is not None
    assert explainer.explainer is not None
    assert explainer.ckd_class_idx in [0, 1]
    assert explainer.model.classes_[explainer.ckd_class_idx] == 1
    assert len(explainer.feature_names) == 24
    assert len(explainer.clinical_labels) == 24


def test_global_shap_calculation(explainer):
    """Verify global explanation generation and dimension matching."""
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df[ALL_FEATURE_NAMES].iloc[:10]  # Subset for fast test

    exp_ckd = explainer._prepare_explanation_object(X_test)
    assert exp_ckd.values.shape == (10, 24)
    assert len(exp_ckd.feature_names) == 24
    assert not np.isnan(exp_ckd.values).any()


def test_patient_explanation_schema(explainer):
    """Verify explain_patient returns complete schema with top risk and protective factors."""
    sample_patient = {
        "age": 55.0,
        "bp": 90.0,
        "sg": 1.010,
        "al": 3.0,
        "su": 1.0,
        "bgr": 180.0,
        "bu": 65.0,
        "sc": 2.8,
        "sod": 132.0,
        "pot": 4.8,
        "hemo": 9.5,
        "pcv": 30.0,
        "wbcc": 9500.0,
        "rbcc": 3.6,
        "rbc": "abnormal",
        "pc": "normal",
        "pcc": "present",
        "ba": "notpresent",
        "htn": "yes",
        "dm": "yes",
        "cad": "no",
        "appet": "poor",
        "pe": "yes",
        "ane": "yes"
    }

    result = explainer.explain_patient(sample_patient, return_base64_plot=True)

    assert "prediction" in result
    assert "shap_explanation" in result
    assert "disclaimer" in result
    assert "waterfall_plot_base64" in result

    pred = result["prediction"]
    assert pred["class_code"] in [0, 1]
    assert 0.0 <= pred["ckd_probability"] <= 1.0
    assert 0.0 <= pred["non_ckd_probability"] <= 1.0

    shap_exp = result["shap_explanation"]
    assert "base_value" in shap_exp
    assert "top_risk_drivers" in shap_exp
    assert "top_protective_factors" in shap_exp
    assert len(shap_exp["all_feature_contributions"]) == 24


def test_shap_mathematical_additivity(explainer):
    """Verify SHAP additivity property: Base Value + Sum(SHAP values) == Model Output."""
    sample_patient = {
        "age": 40.0,
        "bp": 70.0,
        "sg": 1.025,
        "al": 0.0,
        "hemo": 15.5,
        "sc": 0.8,
        "htn": "no"
    }

    result = explainer.explain_patient(sample_patient)
    shap_exp = result["shap_explanation"]

    base_val = shap_exp["base_value"]
    shap_sum = sum(f["shap_value"] for f in shap_exp["all_feature_contributions"])
    reconstructed_output = base_val + shap_sum

    assert pytest.approx(reconstructed_output, abs=1e-3) == shap_exp["patient_model_output"]


def test_waterfall_plot_file_saving(explainer, tmp_path):
    """Verify patient waterfall plot can be saved to a specific filesystem path."""
    sample_patient = {"age": 50.0, "hemo": 11.0, "sc": 1.5, "htn": "yes"}
    out_file = tmp_path / "custom_waterfall.png"

    result = explainer.explain_patient(sample_patient, save_plot_path=out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 5000
    assert "waterfall_plot_path" in result


def test_partial_patient_with_missing_inputs(explainer):
    """Verify explainer handles partial clinical inputs gracefully using learned imputation."""
    partial_patient = {
        "hemo": 8.0,
        "sc": 4.5
    }

    result = explainer.explain_patient(partial_patient)
    assert result["prediction"]["class_code"] in [0, 1]
    assert len(result["shap_explanation"]["all_feature_contributions"]) == 24
