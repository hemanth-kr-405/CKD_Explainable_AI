"""
Comprehensive test suite for the FastAPI backend.

Verifies:
- GET / (Root endpoint)
- GET /health (Health check and model readiness)
- GET /features (Schema metadata for 24 clinical features)
- POST /predict (Valid CKD patient payload, valid healthy payload, partial payload)
- Input validation: Rejecting out-of-bound numbers, invalid categories, and extra fields (422)
- Output validation: Correct prediction, probabilities, SHAP attribution list, base64 waterfall plot
- GET /metrics (Evaluation metrics retrieval)
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root endpoint returns status 200 and navigation links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "/docs" in data["documentation"]
    assert "/health" in data["health_check"]


def test_health_endpoint():
    """Verify health endpoint reports healthy status and model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True
    assert data["explainer_ready"] is True
    assert data["features_count"] == 24


def test_features_endpoint():
    """Verify /features endpoint returns all 24 clinical features with metadata."""
    response = client.get("/features")
    assert response.status_code == 200
    data = response.json()
    assert data["total_features"] == 24
    assert len(data["numerical_features"]) == 14
    assert len(data["categorical_features"]) == 10
    assert len(data["features"]) == 24
    # Check specific feature structure
    assert "hemo" in data["features"]
    assert data["features"]["hemo"]["type"] == "numerical"
    assert data["features"]["hemo"]["min"] is not None
    assert "htn" in data["features"]
    assert data["features"]["htn"]["type"] == "categorical"
    assert "yes" in data["features"]["htn"]["options"]


def test_predict_ckd_patient_returns_valid_shap():
    """Verify prediction on a high-risk CKD profile returns CKD class and SHAP risk drivers."""
    payload = {
        "age": 60.0,
        "bp": 90.0,
        "sg": 1.010,
        "al": 3.0,
        "su": 1.0,
        "bgr": 180.0,
        "bu": 65.0,
        "sc": 3.5,
        "sod": 130.0,
        "pot": 5.2,
        "hemo": 8.5,
        "pcv": 28.0,
        "wbcc": 11000.0,
        "rbcc": 3.2,
        "rbc": "abnormal",
        "pc": "abnormal",
        "pcc": "present",
        "ba": "notpresent",
        "htn": "yes",
        "dm": "yes",
        "cad": "no",
        "appet": "poor",
        "pe": "yes",
        "ane": "yes",
        "include_plot": True
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction_code"] == 1
    assert data["prediction_label"] == "Chronic Kidney Disease (CKD)"
    assert data["probabilities"]["ckd_probability"] > 0.70
    assert data["explanation_method"] == "SHAP (TreeExplainer)"
    assert data["explanation"]["base_value"] > 0
    assert len(data["explanation"]["top_risk_drivers"]) > 0
    assert len(data["explanation"]["all_feature_contributions"]) == 24
    assert data["waterfall_plot_base64"] is not None
    assert len(data["waterfall_plot_base64"]) > 500
    assert "medical_disclaimer" in data


def test_predict_healthy_patient():
    """Verify prediction on a healthy profile returns Non-CKD class and protective factors."""
    payload = {
        "age": 35.0,
        "bp": 70.0,
        "sg": 1.025,
        "al": 0.0,
        "su": 0.0,
        "bgr": 90.0,
        "bu": 25.0,
        "sc": 0.8,
        "sod": 140.0,
        "pot": 4.2,
        "hemo": 15.8,
        "pcv": 48.0,
        "wbcc": 6500.0,
        "rbcc": 5.4,
        "rbc": "normal",
        "pc": "normal",
        "pcc": "notpresent",
        "ba": "notpresent",
        "htn": "no",
        "dm": "no",
        "cad": "no",
        "appet": "good",
        "pe": "no",
        "ane": "no",
        "include_plot": False
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction_code"] == 0
    assert data["prediction_label"] == "Non-CKD"
    assert data["probabilities"]["non_ckd_probability"] > 0.70
    assert len(data["explanation"]["top_protective_factors"]) > 0
    assert data["waterfall_plot_base64"] is None  # Requested false


def test_predict_partial_inputs():
    """Verify API handles partial inputs gracefully using learned training medians/modes."""
    payload = {
        "hemo": 14.5,
        "sc": 0.9,
        "htn": "no"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_code"] in [0, 1]
    assert len(data["explanation"]["all_feature_contributions"]) == 24


def test_reject_invalid_numerical_bounds():
    """Verify 422 error is returned for physiologically invalid numerical values."""
    # Negative blood pressure
    res1 = client.post("/predict", json={"bp": -50.0})
    assert res1.status_code == 422
    assert "bp" in res1.text

    # Impossible age (> 120)
    res2 = client.post("/predict", json={"age": 250.0})
    assert res2.status_code == 422
    assert "age" in res2.text

    # Negative serum creatinine
    res3 = client.post("/predict", json={"sc": -1.5})
    assert res3.status_code == 422
    assert "sc" in res3.text


def test_reject_invalid_categorical_options():
    """Verify 422 error is returned for unsupported categorical strings."""
    res = client.post("/predict", json={"htn": "sometimes"})
    assert res.status_code == 422
    assert "htn" in res.text


def test_reject_unrecognized_extra_fields():
    """Verify 422 error is returned when extra unvalidated fields are sent (extra='forbid')."""
    res = client.post("/predict", json={"hemo": 12.0, "unknown_medical_marker": 999.0})
    assert res.status_code == 422
    assert "unknown_medical_marker" in res.text


def test_metrics_endpoint():
    """Verify /metrics endpoint returns model performance metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["algorithm"] == "DecisionTreeClassifier"
    assert "test_metrics" in data
    assert data["test_metrics"]["accuracy"] > 0.85


def test_dashboard_endpoint():
    """Verify /dashboard serves the frontend HTML interface."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "CKD Explainable AI" in response.text

