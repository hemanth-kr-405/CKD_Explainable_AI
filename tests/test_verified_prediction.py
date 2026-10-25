"""
Automated Test Suite for Phase 6: Verified Patient Data Integration with the Existing CKD Model.

Validates:
1. Exact output parity between POST /predict and POST /predict-verified when provided identical inputs:
   - Full CKD patient profile
   - Full Non-CKD (healthy) patient profile
   - Sparse patient profile with missing values (verifying training median/mode imputation parity)
2. Invariance check:
   - Document upload and extraction features do not alter existing model predictions or state.
3. Decoupled prediction execution:
   - Manually entered and verified values can be evaluated without requiring document upload.
4. Clinical distinction and probability caveats:
   - Model probability is explicitly caveated as uncalibrated leaf proportion, NOT patient clinical risk.
   - Explicitly distinguishes model prediction, model probability, and clinical diagnosis (which ML cannot establish).
   - Medical disclaimer asserts no clinical safety or efficacy based solely on software tests.
5. Provenance accounting:
   - Correctly tallies extracted, manual, verified, and imputed missing counts.
6. Validation gating:
   - Rejects invalid numbers, out-of-range bounds, unknown categories, and unresolved conflicts with HTTP 422.
"""

import io
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.config import ALL_FEATURE_NAMES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def ckd_patient_dict():
    """A realistic clinical profile with elevated indicators of Chronic Kidney Disease."""
    return {
        "age": 62.0,
        "bp": 90.0,
        "sg": 1.010,
        "al": 3.0,
        "su": 1.0,
        "bgr": 180.0,
        "bu": 85.0,
        "sc": 3.2,
        "sod": 130.0,
        "pot": 5.4,
        "hemo": 9.4,
        "pcv": 28.0,
        "wbcc": 9800.0,
        "rbcc": 3.4,
        "rbc": "abnormal",
        "pc": "abnormal",
        "pcc": "present",
        "ba": "notpresent",
        "htn": "yes",
        "dm": "yes",
        "cad": "yes",
        "appet": "poor",
        "pe": "yes",
        "ane": "yes",
    }


@pytest.fixture
def healthy_patient_dict():
    """A realistic clinical profile of a patient with healthy kidney function."""
    return {
        "age": 34.0,
        "bp": 70.0,
        "sg": 1.025,
        "al": 0.0,
        "su": 0.0,
        "bgr": 92.0,
        "bu": 24.0,
        "sc": 0.8,
        "sod": 142.0,
        "pot": 4.1,
        "hemo": 15.6,
        "pcv": 46.0,
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
    }


# ============================================================================
# 1. Output Parity Tests: /predict vs /predict-verified (Requirement 12)
# ============================================================================

def test_prediction_parity_full_ckd_patient(client, ckd_patient_dict):
    """
    Submitting identical CKD patient data to /predict and /predict-verified
    must yield identical predictions, probabilities, and top risk attributions.
    """
    # 1. Standard prediction endpoint
    pred_payload = dict(ckd_patient_dict)
    pred_payload["include_plot"] = False
    resp_standard = client.post("/predict", json=pred_payload)
    assert resp_standard.status_code == 200
    data_standard = resp_standard.json()

    # 2. Verified prediction endpoint
    verified_payload = {
        "parameters": {
            k: {"value": v, "origin": "verified", "status": "verified"}
            for k, v in ckd_patient_dict.items()
        },
        "allow_imputation": True,
        "include_plot": False
    }
    resp_verified = client.post("/predict-verified", json=verified_payload)
    assert resp_verified.status_code == 200
    data_verified = resp_verified.json()

    # Parity assertions
    assert data_verified["prediction_label"] == data_standard["prediction_label"]
    assert data_verified["prediction_code"] == data_standard["prediction_code"]
    assert data_verified["prediction_code"] == 1
    assert data_verified["probabilities"]["ckd_probability"] == data_standard["probabilities"]["ckd_probability"]
    assert data_verified["probabilities"]["non_ckd_probability"] == data_standard["probabilities"]["non_ckd_probability"]

    # SHAP scores parity
    assert data_verified["explanation"]["base_value"] == data_standard["explanation"]["base_value"]
    assert data_verified["explanation"]["patient_model_output"] == data_standard["explanation"]["patient_model_output"]

    # Top risk drivers parity
    std_risk_feats = [f["feature"] for f in data_standard["explanation"]["top_risk_drivers"]]
    ver_risk_feats = [f["feature"] for f in data_verified["explanation"]["top_risk_drivers"]]
    assert std_risk_feats == ver_risk_feats


def test_prediction_parity_full_healthy_patient(client, healthy_patient_dict):
    """
    Submitting identical healthy patient data to /predict and /predict-verified
    must yield identical predictions and probabilities.
    """
    pred_payload = dict(healthy_patient_dict)
    pred_payload["include_plot"] = False
    resp_standard = client.post("/predict", json=pred_payload)
    assert resp_standard.status_code == 200
    data_standard = resp_standard.json()

    verified_payload = {
        "parameters": {
            k: {"value": v, "origin": "verified", "status": "verified"}
            for k, v in healthy_patient_dict.items()
        },
        "allow_imputation": True,
        "include_plot": False
    }
    resp_verified = client.post("/predict-verified", json=verified_payload)
    assert resp_verified.status_code == 200
    data_verified = resp_verified.json()

    assert data_verified["prediction_label"] == data_standard["prediction_label"]
    assert data_verified["prediction_code"] == data_standard["prediction_code"]
    assert data_verified["prediction_code"] == 0
    assert data_verified["probabilities"]["ckd_probability"] == data_standard["probabilities"]["ckd_probability"]
    assert data_verified["probabilities"]["non_ckd_probability"] == data_standard["probabilities"]["non_ckd_probability"]


def test_prediction_parity_partial_input_with_imputation(client):
    """
    Sparse input with missing parameters must yield identical results across both endpoints,
    confirming that the preprocessor's training median/mode imputation is applied identically.
    """
    partial_dict = {
        "sc": 2.1,
        "bu": 60.0,
        "hemo": 10.5,
        "htn": "yes",
        "age": 55.0
        # 19 features missing
    }

    # 1. Standard prediction
    pred_payload = dict(partial_dict)
    pred_payload["include_plot"] = False
    resp_std = client.post("/predict", json=pred_payload)
    assert resp_std.status_code == 200
    data_std = resp_std.json()

    # 2. Verified prediction
    ver_payload = {
        "parameters": {
            k: {"value": v, "origin": "verified", "status": "verified"}
            for k, v in partial_dict.items()
        },
        "allow_imputation": True,
        "include_plot": False
    }
    resp_ver = client.post("/predict-verified", json=ver_payload)
    assert resp_ver.status_code == 200
    data_ver = resp_ver.json()

    assert data_ver["prediction_label"] == data_std["prediction_label"]
    assert data_ver["prediction_code"] == data_std["prediction_code"]
    assert data_ver["probabilities"]["ckd_probability"] == data_std["probabilities"]["ckd_probability"]
    assert data_ver["probabilities"]["non_ckd_probability"] == data_std["probabilities"]["non_ckd_probability"]


# ============================================================================
# 2. Upload Invariance Test (Requirement 13)
# ============================================================================

def test_upload_feature_does_not_mutate_model_predictions(client, ckd_patient_dict):
    """
    Executing document uploads and extractions must NOT change the predictions
    returned by the existing prediction model for the same valid input.
    """
    pred_payload = dict(ckd_patient_dict)
    pred_payload["include_plot"] = False

    # 1. Predict baseline before upload
    resp_before = client.post("/predict", json=pred_payload)
    assert resp_before.status_code == 200
    data_before = resp_before.json()

    # 2. Perform a document upload flow
    fake_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    upload_resp = client.post(
        "/upload-document",
        files={"file": ("lab_report.png", io.BytesIO(fake_png), "image/png")}
    )
    assert upload_resp.status_code == 200
    upload_id = upload_resp.json()["upload_id"]

    # 3. Clean up the uploaded session
    remove_resp = client.delete(f"/upload-document/{upload_id}")
    assert remove_resp.status_code == 200

    # 4. Predict again after upload
    resp_after = client.post("/predict", json=pred_payload)
    assert resp_after.status_code == 200
    data_after = resp_after.json()

    # Ensure absolute invariance
    assert data_before["prediction_label"] == data_after["prediction_label"]
    assert data_before["prediction_code"] == data_after["prediction_code"]
    assert data_before["probabilities"]["ckd_probability"] == data_after["probabilities"]["ckd_probability"]
    assert data_before["explanation"]["patient_model_output"] == data_after["explanation"]["patient_model_output"]


# ============================================================================
# 3. Clinical Distinction, Governance & Caveats (Requirement 7, 8, 9, 14)
# ============================================================================

def test_clinical_distinction_and_caveats(client, ckd_patient_dict):
    """
    Verifies that the response explicitly distinguishes model prediction,
    uncalibrated probability, and clinical diagnosis, and does not claim clinical safety.
    """
    payload = {
        "parameters": {
            k: {"value": v, "origin": "verified", "status": "verified"}
            for k, v in ckd_patient_dict.items()
        },
        "allow_imputation": True,
        "include_plot": False
    }
    resp = client.post("/predict-verified", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # 1. Probability caveat: explicitly denies that leaf purity is patient clinical risk
    caveat = data["probability_caveat"]
    assert "empirical sample proportions" in caveat
    assert "MUST NOT be interpreted as an individual patient's clinical risk" in caveat

    # 2. Clinical distinction: 3 explicit categories
    distinction = data["clinical_distinction"]
    assert "DecisionTree algorithmic classification" in distinction["model_prediction"]
    assert "Terminal leaf proportion" in distinction["model_probability"]
    assert "CANNOT be established by machine learning" in distinction["clinical_diagnosis"]
    assert "formal clinical evaluation by a licensed physician" in distinction["clinical_diagnosis"]

    # 3. Medical disclaimer: no claim of clinical safety on software tests alone
    disclaimer = data["medical_disclaimer"]
    assert "do not constitute a medical diagnosis" in disclaimer
    assert "Clinical safety and diagnostic accuracy cannot be claimed on the basis of software tests alone" in disclaimer


# ============================================================================
# 4. Provenance Accounting (Requirement 9 & 11)
# ============================================================================

def test_provenance_accounting_in_prediction(client):
    """
    Verifies that the prediction response correctly audits data provenance
    across extracted, manual, verified, and imputed missing counts.
    """
    mixed_parameters = {
        "sc": {"value": 2.2, "origin": "extracted", "status": "verified"},
        "bu": {"value": 55.0, "origin": "extracted", "status": "verified"},
        "hemo": {"value": 11.0, "origin": "manual", "status": "verified"},
        "age": {"value": 60.0, "origin": "manual", "status": "verified"},
        "bp": {"value": 80.0, "origin": "verified", "status": "verified"},
        # Remaining 19 features missing
    }

    payload = {
        "parameters": mixed_parameters,
        "allow_imputation": True,
        "include_plot": False
    }
    resp = client.post("/predict-verified", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    prov = data["provenance_summary"]
    assert prov["total_features"] == 24
    assert prov["extracted_count"] == 2
    assert prov["manual_count"] == 2
    assert prov["verified_count"] == 5
    assert prov["missing_count"] == 19
    assert prov["imputation_applied"] is True
    assert "median for numerics" in prov["imputation_strategy"]


def test_decoupled_manual_entry_testing(client, healthy_patient_dict):
    """
    Requirement 11: Manually entered verified values must be testable without any document upload.
    """
    manual_params = {
        k: {"value": v, "origin": "manual", "status": "verified"}
        for k, v in healthy_patient_dict.items()
    }
    payload = {
        "upload_id": None,  # Decoupled from document upload
        "parameters": manual_params,
        "allow_imputation": True,
        "include_plot": False
    }
    resp = client.post("/predict-verified", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction_code"] == 0
    assert data["provenance_summary"]["manual_count"] == 24
    assert data["provenance_summary"]["extracted_count"] == 0


# ============================================================================
# 5. Validation Gating and Error Handling (Requirement 2 & 10)
# ============================================================================

def test_predict_verified_blocks_out_of_bounds_number(client, ckd_patient_dict):
    """Out-of-range numerical values must return HTTP 422 and block prediction."""
    params = {
        k: {"value": v, "origin": "manual", "status": "verified"}
        for k, v in ckd_patient_dict.items()
    }
    params["sc"]["value"] = 300.0  # Well above 100.0 mg/dL max

    resp = client.post("/predict-verified", json={"parameters": params})
    assert resp.status_code == 422
    data = resp.json()
    assert "Validation failed" in data["detail"]
    assert "sc" in data["detail"]


def test_predict_verified_blocks_invalid_categorical(client, ckd_patient_dict):
    """Unsupported categorical values must return HTTP 422 and block prediction."""
    params = {
        k: {"value": v, "origin": "manual", "status": "verified"}
        for k, v in ckd_patient_dict.items()
    }
    params["htn"]["value"] = "unknown"  # Only 'yes' or 'no' allowed

    resp = client.post("/predict-verified", json={"parameters": params})
    assert resp.status_code == 422
    assert "Validation failed" in resp.json()["detail"]


def test_predict_verified_blocks_conflicting_status(client, ckd_patient_dict):
    """Unresolved conflicting parameters must return HTTP 422 and block prediction."""
    params = {
        k: {"value": v, "origin": "manual", "status": "verified"}
        for k, v in ckd_patient_dict.items()
    }
    params["sc"]["status"] = "conflicting"  # Conflicting readings across report

    resp = client.post("/predict-verified", json={"parameters": params})
    assert resp.status_code == 422
    assert "unresolved conflicts" in resp.json()["detail"]
