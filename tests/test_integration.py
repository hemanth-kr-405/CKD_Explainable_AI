"""
End-to-End Integration Test Suite for the CKD Explainable AI Project.

Validates the complete workflow across:
A. Backend service health and lifecycle
B. Frontend dashboard HTML and static asset delivery
C. End-to-end prediction with real held-out test set cases (CKD and Healthy)
D. SHAP local explanation calculation and mathematical additivity
E. Base64 waterfall plot generation and PNG header integrity
F. Rejection of invalid inputs, out-of-range values, and parameter pollution (422)
G. Dynamic, non-hardcoded explainability across varying clinical inputs
H. Schema synchronization between frontend form and backend models
"""

import base64
import json
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from src.config import (
    TEST_DATA_PATH,
    ALL_FEATURE_NAMES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_MAPPINGS,
    TARGET_NAMES
)

client = TestClient(app)

INVERSE_MAPPINGS = {
    feat: {v: k for k, v in mapping.items()}
    for feat, mapping in CATEGORICAL_MAPPINGS.items()
}


def _extract_patient_payload(row) -> dict:
    """Extracts a valid clinical payload from a preprocessed dataframe row."""
    payload = {}
    for feat in ALL_FEATURE_NAMES:
        val = row[feat]
        if pd.notna(val):
            if feat in CATEGORICAL_FEATURES:
                payload[feat] = INVERSE_MAPPINGS[feat].get(int(val))
            else:
                payload[feat] = float(val)
    return payload


# ============================================================================
# A & B: Service Readiness and Frontend Dashboard Serving
# ============================================================================

def test_service_health():
    """Verify backend health endpoint reports operational readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True
    assert data["explainer_ready"] is True
    assert data["features_count"] == 24


def test_dashboard_serves_complete_html():
    """Verify dashboard HTML is served correctly with disclaimer and required components."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text

    # Verify title and branding
    assert "Explainable AI for Chronic Kidney Disease Prediction" in html
    assert "ckd-form" in html
    assert "btn-submit" in html
    assert "btn-reset" in html

    # Verify medical disclaimer verbatim
    required_disclaimer = (
        "This tool provides an experimental Machine Learning prediction for educational and "
        "research purposes only. It is not a medical diagnosis and must not replace advice, "
        "testing, or evaluation by a qualified healthcare professional."
    )
    assert required_disclaimer in html

    # Verify static asset linkages
    assert 'href="style.css"' in html or 'href="/frontend/style.css"' in html
    assert 'src="script.js"' in html or 'src="/frontend/script.js"' in html


def test_static_assets_served():
    """Verify CSS and JS static assets are accessible via FastAPI mount."""
    css_res = client.get("/frontend/style.css")
    assert css_res.status_code == 200
    assert "text/css" in css_res.headers["content-type"]

    js_res = client.get("/frontend/script.js")
    assert js_res.status_code == 200
    assert "javascript" in js_res.headers["content-type"]


# ============================================================================
# C, D, E, F, G: End-to-End Prediction & SHAP Explainability
# ============================================================================

def test_end_to_end_prediction_ckd_case():
    """
    Test end-to-end inference and SHAP attribution on a held-out CKD test case.
    Verifies prediction, probability scores, SHAP values, additivity, and plot rendering.
    """
    test_df = pd.read_csv(TEST_DATA_PATH)
    ckd_row = test_df[test_df["class"] == 1].iloc[0]
    payload = _extract_patient_payload(ckd_row)
    payload["include_plot"] = True

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()

    # 1. Prediction Label & Code
    assert result["prediction_label"] == "Chronic Kidney Disease (CKD)"
    assert result["prediction_code"] == 1

    # 2. Probability Scores
    probs = result["probabilities"]
    assert 0.0 <= probs["ckd_probability"] <= 1.0
    assert 0.0 <= probs["non_ckd_probability"] <= 1.0
    assert pytest.approx(probs["ckd_probability"] + probs["non_ckd_probability"]) == 1.0

    # 3. SHAP Explanation Structure
    exp = result["explanation"]
    assert exp["explanation_target"] == "Probability of Chronic Kidney Disease (CKD)"
    assert len(exp["all_feature_contributions"]) == 24
    assert len(exp["top_risk_drivers"]) > 0

    # 4. Mathematical Additivity Property
    base_val = exp["base_value"]
    shap_sum = sum(f["shap_value"] for f in exp["all_feature_contributions"])
    reconstructed_score = base_val + shap_sum
    assert pytest.approx(reconstructed_score, abs=1e-3) == exp["patient_model_output"]

    # 5. Waterfall Plot PNG Integrity
    plot_b64 = result.get("waterfall_plot_base64")
    assert plot_b64 is not None
    plot_bytes = base64.b64decode(plot_b64)
    # Check valid PNG magic header (\x89PNG\r\n\x1a\n)
    assert plot_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(plot_bytes) > 5000

    # 6. Clinical Narrative & Disclaimer
    narrative = result["clinical_summary"]
    assert "Chronic Kidney Disease" in narrative
    assert "model" in narrative or "path" in narrative


def test_end_to_end_prediction_healthy_case():
    """
    Test end-to-end inference and SHAP attribution on a held-out Healthy test case.
    Verifies Non-CKD classification, negative SHAP contributions, and additivity.
    """
    test_df = pd.read_csv(TEST_DATA_PATH)
    healthy_row = test_df[test_df["class"] == 0].iloc[0]
    payload = _extract_patient_payload(healthy_row)
    payload["include_plot"] = True

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()

    assert result["prediction_label"] == "Non-CKD"
    assert result["prediction_code"] == 0
    assert result["probabilities"]["non_ckd_probability"] >= 0.5
    assert len(result["explanation"]["top_protective_factors"]) > 0

    # Mathematical additivity
    exp = result["explanation"]
    reconstructed = exp["base_value"] + sum(f["shap_value"] for f in exp["all_feature_contributions"])
    assert pytest.approx(reconstructed, abs=1e-3) == exp["patient_model_output"]


def test_dynamic_shap_non_hardcoded():
    """
    Verify that SHAP explanations are truly dynamic and computed from the model,
    not hardcoded strings or static arrays.
    """
    # Create two patients with different hemoglobin and creatinine levels
    patient_a = {"hemo": 8.0, "sc": 4.5, "sg": 1.010, "htn": "yes"}
    patient_b = {"hemo": 16.0, "sc": 0.6, "sg": 1.025, "htn": "no"}

    res_a = client.post("/predict", json=patient_a).json()
    res_b = client.post("/predict", json=patient_b).json()

    assert res_a["prediction_code"] != res_b["prediction_code"]
    assert res_a["explanation"]["patient_model_output"] != res_b["explanation"]["patient_model_output"]

    # Compare specific feature attributions
    hemo_a = next(f for f in res_a["explanation"]["all_feature_contributions"] if f["feature"] == "hemo")
    hemo_b = next(f for f in res_b["explanation"]["all_feature_contributions"] if f["feature"] == "hemo")

    # In patient A (low hemo 8.0), hemoglobin increases risk; in patient B (high hemo 16.0), it lowers risk
    assert hemo_a["shap_value"] != hemo_b["shap_value"]
    assert hemo_a["shap_value"] > hemo_b["shap_value"]


# ============================================================================
# H: Input Validation & Error Handling
# ============================================================================

def test_validation_rejects_out_of_bounds_numerical():
    """Verify physiological bound checking rejects implausible values with HTTP 422."""
    # Serum creatinine 999.0 mg/dL exceeds max 100.0
    payload = {"sc": 999.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    err_body = response.json()
    assert "sc" in str(err_body)


def test_validation_rejects_invalid_categorical_option():
    """Verify categorical validation rejects unverified medical categories with HTTP 422."""
    payload = {"htn": "unknown_option"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    err_body = response.json()
    assert "htn" in str(err_body)


def test_validation_rejects_extra_fields():
    """Verify parameter pollution rejection: Unknown extra fields trigger HTTP 422."""
    payload = {"hemo": 12.0, "unrecognized_biomarker": 42.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    err_body = response.json()
    assert "extra" in str(err_body).lower() or "unrecognized_biomarker" in str(err_body)


# ============================================================================
# Schema Synchronization & Metrics Verification
# ============================================================================

def test_features_metadata_schema_sync():
    """Verify /features metadata completely matches the 24 clinical features."""
    response = client.get("/features")
    assert response.status_code == 200
    data = response.json()
    assert set(data["feature_order"]) == set(ALL_FEATURE_NAMES)
    assert set(data["features"].keys()) == set(ALL_FEATURE_NAMES)


def test_metrics_endpoint_integration():
    """Verify /metrics returns verified evaluation results."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "test_metrics" in data
    assert data["test_metrics"]["accuracy"] == 0.9625
    assert data["test_metrics"]["precision"] == 1.0
    assert data["test_metrics"]["recall_sensitivity"] == 0.94
