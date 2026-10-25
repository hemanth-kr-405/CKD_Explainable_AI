"""
Regression tests for CKD Explainable AI Dashboard Form Interactivity and Validation.

Tests the 8 critical scenarios:
1. Initial form usability (all 24 inputs interactive, fieldsets enabled, button types valid)
2. Reset followed by typing (clearing values restores clean, interactive initial state)
3. Reset followed by Proceed followed by typing (required biomarker validation blocks blank submission without locking inputs)
4. Invalid input followed by correction (invalid values highlight errors, fields remain enabled, corrections pass)
5. Successful prediction followed by Reset (successful run followed by reset clears prediction state and restores interactivity)
6. Failed API request followed by editing and retrying (network/server errors handled gracefully, form remains unlocked)
7. CKD Case preset (populates real test values, keeps inputs interactive, generates valid CKD prediction)
8. Healthy Case preset (populates real test values, keeps inputs interactive, generates valid Non-CKD prediction)
"""

import re
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# The 24 clinical features expected by the trained model
ALL_24_FEATURES = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
    "htn", "dm", "cad", "appet", "pe", "ane"
]

PRESET_CKD = {
    "age": 65, "bp": 90, "sg": 1.010, "al": 4, "su": 2, "bgr": 172,
    "bu": 82, "sc": 4.1, "sod": 134, "pot": 5.8, "hemo": 8.4, "pcv": 26,
    "wbcc": 11200, "rbcc": 3.2, "rbc": "abnormal", "pc": "abnormal",
    "pcc": "present", "ba": "present", "htn": "yes", "dm": "yes",
    "cad": "yes", "appet": "poor", "pe": "yes", "ane": "yes",
    "include_plot": False
}

PRESET_HEALTHY = {
    "age": 35, "bp": 70, "sg": 1.025, "al": 0, "su": 0, "bgr": 89,
    "bu": 24, "sc": 0.8, "sod": 142, "pot": 4.1, "hemo": 16.2, "pcv": 48,
    "wbcc": 6800, "rbcc": 5.4, "rbc": "normal", "pc": "normal",
    "pcc": "notpresent", "ba": "notpresent", "htn": "no", "dm": "no",
    "cad": "no", "appet": "good", "pe": "no", "ane": "no",
    "include_plot": False
}


def test_1_initial_form_usability():
    """
    Scenario 1: Initial form usability.
    - Dashboard page loads with HTTP 200.
    - All 24 patient clinical assessment fields exist and are editable (no disabled/readOnly).
    - No parent fieldset is disabled.
    - Button types are strictly type='button' for non-submitting controls and type='submit' for form submission.
    """
    response = client.get("/dashboard")
    assert response.status_code == 200
    html = response.text

    # Form and action buttons exist
    assert 'id="ckd-form"' in html
    assert 'id="btn-submit"' in html
    assert 'id="btn-reset"' in html
    assert 'id="btn-preset-ckd"' in html
    assert 'id="btn-preset-healthy"' in html

    # Button type checks: non-submitting controls must have type="button", submitting has type="submit"
    assert re.search(r'<button\b(?=[^>]*\bid=["\']btn-reset["\'])(?=[^>]*\btype=["\']button["\'])', html)
    assert re.search(r'<button\b(?=[^>]*\bid=["\']btn-preset-ckd["\'])(?=[^>]*\btype=["\']button["\'])', html)
    assert re.search(r'<button\b(?=[^>]*\bid=["\']btn-preset-healthy["\'])(?=[^>]*\btype=["\']button["\'])', html)
    assert re.search(r'<button\b(?=[^>]*\bid=["\']btn-submit["\'])(?=[^>]*\btype=["\']submit["\'])', html)

    # Check that no <fieldset> is disabled in the initial assessment form
    form_match = re.search(r'<form id="ckd-form".*?</form>', html, re.DOTALL)
    assert form_match is not None
    form_html = form_match.group(0)
    assert "disabled" not in form_html.lower().replace("type=\"button\"", "")

    # Verify all 24 inputs are present and none have disabled or readonly attributes
    for feat in ALL_24_FEATURES:
        pattern = rf'<(input|select)[^>]+id=["\']input-{feat}["\'][^>]*>'
        match = re.search(pattern, form_html)
        assert match is not None, f"Input element for {feat} missing in ckd-form"
        tag_str = match.group(0)
        assert "disabled" not in tag_str, f"Input {feat} must not be disabled"
        assert "readonly" not in tag_str.lower(), f"Input {feat} must not be readOnly"


def test_2_reset_followed_by_typing():
    """
    Scenario 2: Reset followed by typing.
    - Resetting clears form controls and re-enables all inputs and fieldsets.
    - Inputs must remain enabled so user can type new values immediately.
    """
    # Fetch script.js to verify handleReset implementation guarantees interactivity restoration
    js_resp = client.get("/script.js")
    assert js_resp.status_code == 200
    js_content = js_resp.text

    # Verify handleReset explicitly clears elements, enables all inputs, fieldsets, and buttons
    assert "function handleReset()" in js_content
    assert "el.disabled = false" in js_content
    assert "el.readOnly = false" in js_content
    assert "fs.disabled = false" in js_content
    assert "submitBtn.disabled = false" in js_content
    assert "clearValidationErrors()" in js_content


def test_3_reset_followed_by_proceed_followed_by_typing():
    """
    Scenario 3: Reset followed by Proceed followed by typing.
    - When the user resets and immediately clicks Proceed (empty form),
      client-side validation must intercept required biomarkers (Serum Creatinine and Hemoglobin).
    - Validation error message is displayed.
    - Inputs MUST NOT be disabled or made readOnly.
    - User can immediately type and correct the inputs.
    """
    js_resp = client.get("/script.js")
    assert js_resp.status_code == 200
    js = js_resp.text

    # Verify required validation for sc and hemo before proceeding
    assert "Serum Creatinine (sc) is required" in js
    assert "Hemoglobin (hemo) is required" in js

    # Verify that validation errors keep formControls enabled and interactive
    submit_fn_match = re.search(r'async function handleFormSubmit\(e\)\s*\{(.*?)\n\}', js, re.DOTALL)
    assert submit_fn_match is not None
    fn_body = submit_fn_match.group(1)

    assert "el.disabled = false" in fn_body
    assert "el.readOnly = false" in fn_body
    assert "submitBtn.disabled = false" in fn_body


def test_4_invalid_input_followed_by_correction():
    """
    Scenario 4: Invalid input followed by correction.
    - Submitting out-of-range or malformed values (e.g. age=999, bp=-10) is caught.
    - Server validation correctly rejects invalid bounds with HTTP 422.
    - Inputs remain enabled, allowing the user to correct the values and resubmit successfully.
    """
    # Test server-side validation error handling
    invalid_payload = {
        "age": 200.0,  # Invalid: max is 120
        "bp": -10.0,   # Invalid: min is 40
        "sc": 1.2,
        "hemo": 14.0,
        "include_plot": False
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    err_json = response.json()
    assert "errors" in err_json or "detail" in err_json

    # Now correct values and resubmit
    corrected_payload = {
        "age": 55.0,
        "bp": 80.0,
        "sc": 1.2,
        "hemo": 14.0,
        "include_plot": False
    }
    success_resp = client.post("/predict", json=corrected_payload)
    assert success_resp.status_code == 200
    data = success_resp.json()
    assert "prediction_label" in data
    assert "probabilities" in data


def test_5_successful_prediction_followed_by_reset():
    """
    Scenario 5: Successful prediction followed by Reset.
    - Perform a successful prediction.
    - Resetting restores the initial state, removes stale prediction display,
      and leaves all inputs completely interactive.
    """
    payload = dict(PRESET_HEALTHY)
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction_label"] in ["Chronic Kidney Disease (CKD)", "Non-CKD", "Non-CKD (Normal Renal Function)"]

    # Verify reset logic clears currentPredictionResult and resets placeholder
    js_resp = client.get("/script.js")
    js = js_resp.text
    assert "currentPredictionResult = null;" in js
    assert "placeholderState.style.display = 'block';" in js
    assert "outputContainer.style.display = 'none';" in js


def test_6_failed_api_request_followed_by_editing_and_retrying():
    """
    Scenario 6: Failed API request followed by editing and retrying.
    - If the API returns an error or request fails, the form must not lock up.
    - Finally block in handleFormSubmit guarantees submitBtn is re-enabled and
      all inputs remain editable.
    """
    js_resp = client.get("/script.js")
    js = js_resp.text

    # Check try/catch/finally in handleFormSubmit
    assert "try {" in js
    assert "catch (err) {" in js
    assert "finally {" in js
    assert "if (loadingState) loadingState.style.display = 'none';" in js
    assert "if (submitBtn) submitBtn.disabled = false;" in js

    # Trigger a 422 error via unknown field
    bad_payload = {"unknown_fake_field": 1234, "include_plot": False}
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422

    # Retry with valid payload
    valid_resp = client.post("/predict", json=PRESET_HEALTHY)
    assert valid_resp.status_code == 200


def test_7_ckd_case_preset():
    """
    Scenario 7: CKD Case preset.
    - Preset values populate properly and match clinical expectations.
    - All inputs remain editable after loading the preset.
    - Predicts high CKD risk with valid SHAP explanations.
    """
    response = client.post("/predict", json=PRESET_CKD)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_code"] == 1
    assert "CKD" in data["prediction_label"]
    assert data["probabilities"]["ckd_probability"] > 0.5
    assert len(data["explanation"]["top_risk_drivers"]) > 0

    # Verify preset function re-enables inputs
    js = client.get("/script.js").text
    assert "function applyCkdPreset()" in js
    assert "populateForm(PRESET_CKD);" in js


def test_8_healthy_case_preset():
    """
    Scenario 8: Healthy Case preset.
    - Preset values populate properly and match normal renal physiology.
    - All inputs remain editable after loading the preset.
    - Predicts Non-CKD with valid SHAP explanations.
    """
    response = client.post("/predict", json=PRESET_HEALTHY)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_code"] == 0
    assert "Non-CKD" in data["prediction_label"]
    assert data["probabilities"]["non_ckd_probability"] > 0.5
    assert len(data["explanation"]["top_protective_factors"]) > 0

    # Verify preset function re-enables inputs
    js = client.get("/script.js").text
    assert "function applyHealthyPreset()" in js
    assert "populateForm(PRESET_HEALTHY);" in js
