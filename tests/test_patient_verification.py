"""
Automated Test Suite for Phase 5: Patient Data Verification Interface.

Validates:
1. Missing fields handling:
   - Missing fields are kept missing (None) and never filled with arbitrary zeros or fabricated defaults.
   - Acknowledged for model preprocessor median/mode imputation.
2. Invalid categories:
   - Rejects unsupported categorical options (e.g. 'yellow' for rbc, 'maybe' for htn).
3. Invalid numbers:
   - Physiological range enforcement (below min or above max).
   - Rejection of NaN, infinite, and non-numeric values.
   - Specific gravity bounds checking (1.000 to 1.040).
4. Unit integrity:
   - Correct units matched to canonical model definitions.
5. User corrections and provenance tracking:
   - Preserves distinction between 'extracted', 'manual', and 'verified'.
   - Tracks status transitions ('requires_review' -> 'verified').
6. Conflicting values:
   - Parameters with unresolved cross-page conflicts block prediction.
7. Prediction gating:
   - Prediction is blocked when invalid or conflicting values exist.
   - Allowed when all present values are valid and missing values are acknowledged.
8. Independent backend validation:
   - POST /verify-parameters endpoint returns structured validation results and sanitized input.
   - Sanitized input passes directly into POST /predict for inference and SHAP attribution.
"""

import math
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas import VerificationSubmissionRequest, VerificationValidationResult
from src.patient_verifier import (
    validate_single_parameter,
    verify_patient_parameters_payload,
    CLINICAL_BOUNDS,
    VerifiedParameterRecord,
    VerificationResult
)
from src.config import ALL_FEATURE_NAMES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_full_patient_dict():
    """A realistic, clinically valid set of all 24 CKD features."""
    return {
        "age": 56.0,
        "bp": 80.0,
        "sg": 1.020,
        "al": 1.0,
        "su": 0.0,
        "bgr": 130.0,
        "bu": 42.0,
        "sc": 1.4,
        "sod": 137.0,
        "pot": 4.5,
        "hemo": 13.8,
        "pcv": 41.0,
        "wbcc": 7600.0,
        "rbcc": 4.9,
        "rbc": "normal",
        "pc": "normal",
        "pcc": "notpresent",
        "ba": "notpresent",
        "htn": "yes",
        "dm": "no",
        "cad": "no",
        "appet": "good",
        "pe": "no",
        "ane": "no",
    }


# ============================================================================
# 1. Missing Fields Tests (Requirement 7 & 8)
# ============================================================================

def test_missing_field_preserves_none():
    """Missing fields must strictly remain None and never be filled with 0.0 or defaults."""
    # Test None
    record = validate_single_parameter("sc", None)
    assert record.value is None, "Missing numerical parameter must be None, NOT zero or fabricated default."
    assert record.status == "missing"
    assert record.is_valid is True

    # Test empty string
    record_empty = validate_single_parameter("hemo", "")
    assert record_empty.value is None
    assert record_empty.status == "missing"

    # Test explicit "missing" / "n/a" string
    record_na = validate_single_parameter("age", "N/A")
    assert record_na.value is None
    assert record_na.status == "missing"

    # Test categorical missing
    record_cat = validate_single_parameter("htn", None)
    assert record_cat.value is None
    assert record_cat.status == "missing"


def test_payload_with_missing_fields_retains_none_and_allows_imputation():
    """Payload with missing values retains None in sanitized_input for preprocessor median/mode imputation."""
    partial_payload = {
        "sc": {"value": 1.5, "unit": "mg/dL", "origin": "extracted", "status": "verified"},
        "bu": {"value": 45.0, "unit": "mg/dL", "origin": "extracted", "status": "verified"},
        "age": {"value": 60.0, "unit": "years", "origin": "manual", "status": "verified"},
        # All other 21 features omitted
    }
    result = verify_patient_parameters_payload(partial_payload, allow_imputation=True)
    assert result.is_valid_for_prediction is True
    assert result.missing_count == 21
    assert result.sanitized_input is not None
    # Crucial: missing fields must be None, NOT 0.0
    assert result.sanitized_input["hemo"] is None
    assert result.sanitized_input["bp"] is None
    assert result.sanitized_input["htn"] is None
    assert result.sanitized_input["sc"] == 1.5
    assert "training median/mode" in result.message


# ============================================================================
# 2. Invalid Categories Tests (Requirement 5 & 12)
# ============================================================================

def test_invalid_categorical_rejected():
    """Unrecognized categorical options must be rejected with errors and status='invalid'."""
    # Invalid rbc value
    record_rbc = validate_single_parameter("rbc", "yellow")
    assert record_rbc.is_valid is False
    assert record_rbc.status == "invalid"
    assert record_rbc.value is None
    assert any("Supported options are" in err for err in record_rbc.errors)

    # Invalid htn value
    record_htn = validate_single_parameter("htn", "maybe")
    assert record_htn.is_valid is False
    assert record_htn.status == "invalid"

    # Numeric value in categorical feature
    record_pe = validate_single_parameter("pe", 123)
    assert record_pe.is_valid is False
    assert record_pe.status == "invalid"


def test_valid_categorical_accepted():
    """Recognized categorical options must be properly verified and normalized."""
    record_rbc = validate_single_parameter("rbc", "NORMAL", origin="manual")
    assert record_rbc.is_valid is True
    assert record_rbc.value == "normal"
    assert record_rbc.status == "verified"

    record_pcc = validate_single_parameter("pcc", "present", origin="extracted")
    assert record_pcc.is_valid is True
    assert record_pcc.value == "present"
    assert record_pcc.status == "requires_review"


# ============================================================================
# 3. Invalid Numbers Tests (Requirement 5 & 12)
# ============================================================================

def test_numerical_out_of_physiological_bounds():
    """Values outside defined physiological limits must be flagged as invalid."""
    # Age below 1
    rec_age_low = validate_single_parameter("age", 0.0)
    assert rec_age_low.is_valid is False
    assert rec_age_low.status == "invalid"
    assert any("below the physiological minimum" in err for err in rec_age_low.errors)

    # Age above 120
    rec_age_high = validate_single_parameter("age", 145.0)
    assert rec_age_high.is_valid is False
    assert rec_age_high.status == "invalid"
    assert any("exceeds the physiological maximum" in err for err in rec_age_high.errors)

    # Serum Creatinine above 100 mg/dL
    rec_sc_high = validate_single_parameter("sc", 150.0)
    assert rec_sc_high.is_valid is False
    assert rec_sc_high.status == "invalid"

    # Blood Pressure below 40 mm/Hg
    rec_bp_low = validate_single_parameter("bp", 25.0)
    assert rec_bp_low.is_valid is False
    assert rec_bp_low.status == "invalid"


def test_numerical_nan_and_inf_rejected():
    """NaN, infinite, and non-numeric strings must be rejected cleanly without exceptions."""
    rec_nan = validate_single_parameter("sc", float("nan"))
    assert rec_nan.is_valid is False
    assert rec_nan.status == "invalid"
    assert any("NaN or Infinite" in err for err in rec_nan.errors)

    rec_inf = validate_single_parameter("bp", float("inf"))
    assert rec_inf.is_valid is False
    assert rec_inf.status == "invalid"

    rec_str = validate_single_parameter("bgr", "unreadable_text")
    assert rec_str.is_valid is False
    assert rec_str.status == "invalid"
    assert any("not a valid decimal number" in err for err in rec_str.errors)


def test_specific_gravity_validation():
    """Urine specific gravity must fall within 1.000 to 1.040."""
    rec_valid = validate_single_parameter("sg", 1.025)
    assert rec_valid.is_valid is True
    assert rec_valid.value == 1.025

    rec_invalid = validate_single_parameter("sg", 1.080)
    assert rec_invalid.is_valid is False
    assert rec_invalid.status == "invalid"


def test_dipstick_grades_warning():
    """Albumin and Sugar dipstick non-integer values produce warnings but remain usable if in [0, 5]."""
    rec_warn = validate_single_parameter("al", 2.5)
    assert rec_warn.is_valid is True
    assert len(rec_warn.warnings) > 0
    assert any("non-standard for dipstick scale" in w for w in rec_warn.warnings)

    rec_out = validate_single_parameter("al", 7.0)
    assert rec_out.is_valid is False
    assert rec_out.status == "invalid"


# ============================================================================
# 4. Provenance and User Corrections Tests (Requirement 3 & 9)
# ============================================================================

def test_provenance_tracking():
    """Preserves distinction between 'extracted', 'manual', and 'verified'."""
    # Extracted value defaults to requires_review until confirmed
    rec_ext = validate_single_parameter("sc", 1.4, origin="extracted")
    assert rec_ext.origin == "extracted"
    assert rec_ext.status == "requires_review"

    # Manual entry is verified by default
    rec_man = validate_single_parameter("sc", 1.4, origin="manual")
    assert rec_man.origin == "manual"
    assert rec_man.status == "verified"

    # Explicitly verified
    rec_ver = validate_single_parameter("sc", 1.4, origin="verified")
    assert rec_ver.origin == "verified"
    assert rec_ver.status == "verified"


def test_user_correction_workflow():
    """Simulates a user correcting an erroneous extracted value."""
    # Extracted value was out of range (e.g. OCR misread 140 instead of 1.4)
    ocr_bad = validate_single_parameter("sc", 140.0, origin="extracted")
    assert ocr_bad.status == "invalid"
    assert ocr_bad.is_valid is False

    # User corrects the value in the interface to 1.4 mg/dL
    user_corrected = validate_single_parameter(
        canonical_name="sc",
        raw_value=1.4,
        unit="mg/dL",
        origin="verified",
        source_text="S. Creatinine: 1.4 mg/dL (user corrected from 140)",
        source_page=1
    )
    assert user_corrected.is_valid is True
    assert user_corrected.value == 1.4
    assert user_corrected.origin == "verified"
    assert user_corrected.status == "verified"


# ============================================================================
# 5. Conflicting Values Tests (Requirement 4 & 8)
# ============================================================================

def test_conflicting_parameter_blocks_prediction():
    """Unresolved conflicting values must set status='conflicting' and block prediction."""
    rec_conflict = validate_single_parameter("sc", 1.5, declared_status="conflicting")
    assert rec_conflict.status == "conflicting"
    assert rec_conflict.is_valid is False

    payload = {
        "sc": {"value": 1.5, "status": "conflicting", "origin": "extracted"}
    }
    result = verify_patient_parameters_payload(payload)
    assert result.is_valid_for_prediction is False
    assert result.conflicting_count == 1
    assert "unresolved conflicts" in result.message


# ============================================================================
# 6. Complete Verification Payload Validation (Requirement 8 & 11)
# ============================================================================

def test_verify_payload_all_valid(valid_full_patient_dict):
    """Payload with all 24 valid features produces clean sanitized input and is valid for prediction."""
    payload = {
        k: {"value": v, "origin": "verified", "status": "verified"}
        for k, v in valid_full_patient_dict.items()
    }
    result = verify_patient_parameters_payload(payload)
    assert result.is_valid_for_prediction is True
    assert result.invalid_count == 0
    assert result.conflicting_count == 0
    assert result.verified_count == 24
    assert result.sanitized_input is not None
    assert len(result.sanitized_input) == 24
    assert result.sanitized_input["sc"] == 1.4
    assert result.sanitized_input["htn"] == "yes"


def test_verify_payload_invalid_value_blocks_prediction(valid_full_patient_dict):
    """If even a single value is invalid, prediction must be strictly blocked."""
    valid_full_patient_dict["bp"] = 350.0  # Excessive blood pressure
    payload = {
        k: {"value": v, "origin": "manual"}
        for k, v in valid_full_patient_dict.items()
    }
    result = verify_patient_parameters_payload(payload)
    assert result.is_valid_for_prediction is False
    assert result.invalid_count == 1
    assert "bp" in result.validation_errors
    assert result.sanitized_input is None


# ============================================================================
# 7. FastAPI Endpoint Integration Tests (Requirement 10 & 11)
# ============================================================================

def test_verify_parameters_endpoint_success(client, valid_full_patient_dict):
    """POST /verify-parameters successfully validates a complete clean payload."""
    payload = {
        "parameters": {
            k: {"value": v, "origin": "verified", "status": "verified"}
            for k, v in valid_full_patient_dict.items()
        },
        "allow_imputation": True
    }
    response = client.post("/verify-parameters", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid_for_prediction"] is True
    assert data["invalid_count"] == 0
    assert data["total_features"] == 24
    assert data["sanitized_input"]["sc"] == 1.4


def test_verify_parameters_endpoint_invalid_block(client, valid_full_patient_dict):
    """POST /verify-parameters returns is_valid_for_prediction=False when invalid bounds exist."""
    invalid_params = {
        k: {"value": v, "origin": "manual"}
        for k, v in valid_full_patient_dict.items()
    }
    invalid_params["age"] = {"value": -5.0, "origin": "manual"}  # Negative age
    invalid_params["rbc"] = {"value": "blue", "origin": "manual"}  # Invalid category

    payload = {"parameters": invalid_params, "allow_imputation": True}
    response = client.post("/verify-parameters", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid_for_prediction"] is False
    assert data["invalid_count"] == 2
    assert "age" in data["validation_errors"]
    assert "rbc" in data["validation_errors"]
    assert data["sanitized_input"] is None


def test_verify_and_execute_prediction_pipeline(client, valid_full_patient_dict):
    """
    End-to-end integration:
    1. Submit parameters to /verify-parameters.
    2. Ensure validation passes and extracts sanitized_input.
    3. Submit sanitized_input directly to /predict.
    4. Confirm successful prediction and SHAP explanation.
    """
    verify_payload = {
        "parameters": {
            k: {"value": v, "origin": "verified", "status": "verified"}
            for k, v in valid_full_patient_dict.items()
        },
        "allow_imputation": True
    }
    verify_resp = client.post("/verify-parameters", json=verify_payload)
    assert verify_resp.status_code == 200
    verify_data = verify_resp.json()
    assert verify_data["is_valid_for_prediction"] is True

    sanitized = verify_data["sanitized_input"]
    sanitized["include_plot"] = True

    predict_resp = client.post("/predict", json=sanitized)
    assert predict_resp.status_code == 200
    predict_data = predict_resp.json()
    assert "prediction_label" in predict_data
    assert "probabilities" in predict_data
    assert "explanation" in predict_data
    assert predict_data["explanation_method"] == "SHAP (TreeExplainer)"
    assert predict_data["waterfall_plot_base64"] is not None
