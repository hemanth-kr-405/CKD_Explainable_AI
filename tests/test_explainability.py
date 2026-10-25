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


# ============================================================================
# Phase 7: Enhanced SHAP Verification & Governance Tests
# ============================================================================

def test_shap_output_shape_and_feature_order(explainer):
    """Verify SHAP output shape matches exactly (N, 24) and feature order matches ALL_FEATURE_NAMES."""
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_sample = test_df[ALL_FEATURE_NAMES].iloc[:5]

    exp = explainer._prepare_explanation_object(X_sample)
    assert exp.values.shape == (5, 24)
    assert list(exp.feature_names) == explainer.clinical_labels

    # Check individual patient explanation feature list order and completeness
    sample_patient = {"age": 55, "sc": 2.0, "hemo": 10.0}
    patient_res = explainer.explain_patient(sample_patient)
    all_attribs = patient_res["shap_explanation"]["all_feature_contributions"]
    assert len(all_attribs) == 24
    assert [item["feature"] for item in all_attribs] == ALL_FEATURE_NAMES


def test_shap_class_handling_and_index(explainer):
    """Verify class indexing resolves positive class 1 (CKD) correctly and directional convention."""
    assert explainer.ckd_class_idx == 1
    assert explainer.model.classes_[explainer.ckd_class_idx] == 1

    sample_patient = {"sc": 4.5, "hemo": 7.5, "htn": "yes"}
    result = explainer.explain_patient(sample_patient)
    shap_exp = result["shap_explanation"]

    # Positive SHAP values must have effect == "increases_risk"
    for item in shap_exp["top_risk_drivers"]:
        assert item["shap_value"] > 0
        assert item["effect"] == "increases_risk"
        assert "Increases" in item["direction"]

    # Negative SHAP values must have effect == "decreases_risk"
    for item in shap_exp["top_protective_factors"]:
        assert item["shap_value"] < 0
        assert item["effect"] == "decreases_risk"
        assert "Decreases" in item["direction"]


def test_shap_most_influential_ranking(explainer):
    """Verify most_influential_features are properly sorted descending by absolute SHAP attribution."""
    sample_patient = {
        "age": 60, "bp": 90, "sg": 1.010, "al": 3, "sc": 3.2, "hemo": 8.5, "htn": "yes"
    }
    result = explainer.explain_patient(sample_patient)
    shap_exp = result["shap_explanation"]

    assert "most_influential_features" in shap_exp
    influential = shap_exp["most_influential_features"]
    assert len(influential) > 0

    # Ensure sorted by |SHAP| descending
    abs_shaps = [abs(item["shap_value"]) for item in influential]
    assert abs_shaps == sorted(abs_shaps, reverse=True)


def test_shap_scale_and_consistency_with_prediction(explainer):
    """Verify that SHAP explanation operates on Probability Scale [0.0 to 1.0] and matches model probabilities."""
    sample_patient = {
        "age": 65, "sc": 4.0, "hemo": 8.0, "al": 2, "htn": "yes"
    }
    result = explainer.explain_patient(sample_patient)
    shap_exp = result["shap_explanation"]
    pred = result["prediction"]

    assert shap_exp["output_scale"] == "Probability Scale [0.0 to 1.0]"
    assert "scale_interpretation" in shap_exp
    assert "directionality_guide" in shap_exp
    assert "causality_caveat" in shap_exp

    # Terminal leaf model output matches sum of baseline and attributions
    base_val = shap_exp["base_value"]
    shap_sum = sum(item["shap_value"] for item in shap_exp["all_feature_contributions"])
    reconstructed = base_val + shap_sum
    assert pytest.approx(reconstructed, abs=1e-3) == shap_exp["patient_model_output"]
    assert pytest.approx(shap_exp["patient_model_output"], abs=1e-3) == pred["ckd_probability"]


def test_shap_input_correspondence(explainer):
    """Verify that feature attribution records correspond to the exact input and transformed values."""
    patient_data = {
        "age": 52.0,
        "sc": 2.1,
        "hemo": 11.2,
        "htn": "yes"
    }
    result = explainer.explain_patient(patient_data)
    attrib_map = {item["feature"]: item for item in result["shap_explanation"]["all_feature_contributions"]}

    assert attrib_map["age"]["patient_value"] == 52.0
    assert attrib_map["sc"]["patient_value"] == 2.1
    assert attrib_map["hemo"]["patient_value"] == 11.2
    assert attrib_map["htn"]["patient_value"] == "yes"

    # Transformed values must be numeric floats
    for feat in ["age", "sc", "hemo", "htn"]:
        assert isinstance(attrib_map[feat]["transformed_value"], (int, float))


def test_shap_incompatible_model_handling(tmp_path):
    """Verify CKDExplainer raises TypeError when given an incompatible non-tree model."""
    from sklearn.linear_model import LogisticRegression
    import joblib

    bad_model_path = tmp_path / "bad_model.joblib"
    joblib.dump(LogisticRegression(), bad_model_path)

    with pytest.raises(TypeError, match="Incompatible model"):
        CKDExplainer(bad_model_path, PREPROCESSOR_ARTIFACT_PATH)


def test_shap_incompatible_input_type(explainer):
    """Verify explainer rejects non-dict and invalid input types gracefully."""
    with pytest.raises(TypeError, match="Unsupported patient_data type"):
        explainer.explain_patient(["invalid", "list"])


def test_shap_non_causality_and_disclaimer(explainer):
    """Verify explanations include explicit non-causality statement and medical disclaimer."""
    sample_patient = {"sc": 2.0, "hemo": 10.0}
    result = explainer.explain_patient(sample_patient)

    assert "causality_caveat" in result["shap_explanation"]
    assert "causation" in result["shap_explanation"]["causality_caveat"].lower()
    assert "disclaimer" in result
    assert "clinical" in result["disclaimer"].lower() or "research" in result["disclaimer"].lower()

