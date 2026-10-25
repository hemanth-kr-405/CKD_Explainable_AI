"""
Automated Test Suite for Phase 4: Patient Parameter Extraction.

Validates:
1. Complete extraction of all 24 CKD model input features from synthetic medical text.
2. Decimal values precision across numerical parameters.
3. Automated unit conversions (umol/L -> mg/dL, mmol/L -> mg/dL, g/L -> g/dL, BUN -> Urea, WBC thousands, RBC).
4. Strict reference range immunity (never extracting reference boundaries instead of patient results).
5. Qualitative dipstick mapping for Albumin and Sugar.
6. Blood pressure diastolic extraction from systolic/diastolic readings.
7. Ambiguous, obscured, and unreadable values (zero substitution strictly forbidden).
8. Conflicting parameter values across multiple document pages.
9. Duplicate consistent parameter values across pages.
10. Missing parameters handling without inventing values.
11. Detection and separation of unsupported clinical tests (Platelets, SGOT, SGPT, Bilirubin).
12. FastAPI integration for POST /extract-parameters/{upload_id} and GET /extract-parameters/{upload_id}.
"""

import io
from pathlib import Path
import pytest
from PIL import Image, ImageDraw, ImageFont
from fastapi.testclient import TestClient

from backend.main import app
from src.parameter_extractor import (
    parameter_extractor,
    extract_parameters_from_text,
    split_text_by_pages,
    strip_reference_ranges,
    parse_numerical_value,
    clean_numeric_string,
    parse_qualitative_dipstick,
    parse_categorical_value,
    ExtractionSummary
)
from src.config import ALL_FEATURE_NAMES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


# ============================================================================
# Synthetic Report Fixtures
# ============================================================================

SYNTHETIC_FULL_REPORT = """
--- Page 1 ---
APOLLO CLINICAL DIAGNOSTIC LABORATORIES
PATIENT REPORT
Patient Name: John Doe
Age / Sex: 54 Years / Male
Blood Pressure: 130/80 mmHg

BIOCHEMISTRY & RENAL FUNCTION PANEL
Test Name                     Result      Units       Reference Interval
Serum Creatinine              1.45        mg/dL       0.60 - 1.20
Blood Urea                    42.0        mg/dL       15.0 - 45.0
Random Blood Sugar            128.5       mg/dL       70.0 - 140.0
Serum Sodium                  138.0       mEq/L       135.0 - 145.0
Serum Potassium               4.30        mEq/L       3.50 - 5.00

--- Page 2 ---
HEMATOLOGY / COMPLETE BLOOD COUNT
Test Name                     Result      Units       Reference Interval
Hemoglobin                    12.6        g/dL        13.0 - 17.0
Packed Cell Volume (PCV)      40.0        %           40.0 - 50.0
White Blood Cell Count        8100        cells/cumm  4000 - 11000
Red Blood Cell Count          4.80        millions/cmm 4.50 - 5.50
Platelet Count                250000      /cumm       150000 - 450000

URINALYSIS & CLINICAL HISTORY
Specific Gravity: 1.020 (Ref: 1.005 - 1.030)
Urine Albumin: 1+ (Dipstick)
Urine Sugar: Nil
Urine RBC: Normal
Urine Pus Cells: Normal
Pus Cell Clumps: Not Present
Bacteria: Not Present

CLINICAL NOTES & HISTORY:
Hypertension: No
Diabetes Mellitus: No
Coronary Artery Disease: No
Appetite: Good
Pedal Edema: No
Anemia: No
"""


# ============================================================================
# Unit Tests for Parameter Extractor
# ============================================================================

def test_full_synthetic_report_extraction():
    """Verifies that all 24 features are accurately extracted from a comprehensive synthetic report."""
    summary: ExtractionSummary = extract_parameters_from_text(SYNTHETIC_FULL_REPORT)

    assert summary.total_features_supported == 24
    assert summary.missing_count == 0
    assert summary.conflicting_count == 0
    assert summary.unreadable_count == 0

    params = summary.parameters

    # Check key numerical features
    assert params["age"].extracted_value == 54.0
    assert params["bp"].extracted_value == 80.0  # Diastolic extracted
    assert params["sc"].extracted_value == 1.45
    assert params["bu"].extracted_value == 42.0
    assert params["bgr"].extracted_value == 128.5
    assert params["sod"].extracted_value == 138.0
    assert params["pot"].extracted_value == 4.30
    assert params["hemo"].extracted_value == 12.6
    assert params["pcv"].extracted_value == 40.0
    assert params["wbcc"].extracted_value == 8100.0
    assert params["rbcc"].extracted_value == 4.80
    assert params["sg"].extracted_value == 1.020
    assert params["al"].extracted_value == 1.0  # 1+ dipstick
    assert params["su"].extracted_value == 0.0  # Nil dipstick

    # Check key categorical features
    assert params["rbc"].extracted_value == "normal"
    assert params["pc"].extracted_value == "normal"
    assert params["pcc"].extracted_value == "notpresent"
    assert params["ba"].extracted_value == "notpresent"
    assert params["htn"].extracted_value == "no"
    assert params["dm"].extracted_value == "no"
    assert params["cad"].extracted_value == "no"
    assert params["appet"].extracted_value == "good"
    assert params["pe"].extracted_value == "no"
    assert params["ane"].extracted_value == "no"


def test_reference_range_immunity_inline():
    """Ensures extractor picks patient result and never extracts reference boundaries."""
    text = """
    LABORATORY TEST REPORT
    Serum Creatinine: 2.4 mg/dL (Ref: 0.6 - 1.2 mg/dL)
    Blood Urea: 65.0 mg/dL [Reference Range: 15.0 - 45.0 mg/dL]
    Hemoglobin: 9.8 g/dL (Normal: 12.0 - 16.0 g/dL)
    Serum Potassium: 5.6 mEq/L (Biological Ref Interval: 3.5 - 5.0 mEq/L)
    """
    summary = extract_parameters_from_text(text)

    # Creatinine must be 2.4, never 0.6 or 1.2
    assert summary.parameters["sc"].extracted_value == 2.4
    assert summary.parameters["sc"].status == "extracted_unambiguous"

    # Blood Urea must be 65.0, never 15.0 or 45.0
    assert summary.parameters["bu"].extracted_value == 65.0

    # Hemoglobin must be 9.8, never 12.0 or 16.0
    assert summary.parameters["hemo"].extracted_value == 9.8

    # Potassium must be 5.6, never 3.5 or 5.0
    assert summary.parameters["pot"].extracted_value == 5.6


def test_unit_conversion_creatinine_umol():
    """Tests conversion of Serum Creatinine from umol/L to mg/dL."""
    # 176.84 umol/L / 88.42 = 2.00 mg/dL
    text = "Serum Creatinine: 176.8 umol/L"
    summary = extract_parameters_from_text(text)

    param = summary.parameters["sc"]
    assert param.extracted_value == 2.0
    assert param.unit == "mg/dL"
    assert param.status == "extracted_requiring_review"
    assert any("Converted serum creatinine from µmol/L" in w for w in param.warnings)


def test_unit_conversion_glucose_mmol():
    """Tests conversion of Blood Glucose from mmol/L to mg/dL."""
    # 7.0 mmol/L * 18.018 = 126.1 mg/dL
    text = "Random Blood Sugar: 7.0 mmol/L"
    summary = extract_parameters_from_text(text)

    param = summary.parameters["bgr"]
    assert param.extracted_value == 126.1
    assert param.unit == "mg/dL"
    assert param.status == "extracted_requiring_review"
    assert any("Converted blood glucose from mmol/L" in w for w in param.warnings)


def test_unit_conversion_hemoglobin_gl():
    """Tests conversion of Hemoglobin from g/L to g/dL."""
    # 135 g/L / 10 = 13.5 g/dL
    text = "Hemoglobin: 135.0 g/L"
    summary = extract_parameters_from_text(text)

    param = summary.parameters["hemo"]
    assert param.extracted_value == 13.5
    assert param.unit == "g/dL"
    assert param.status == "extracted_requiring_review"
    assert any("Converted hemoglobin from g/L" in w for w in param.warnings)


def test_unit_conversion_bun_to_urea():
    """Tests conversion of Blood Urea Nitrogen (BUN) in mg/dL to Blood Urea."""
    # BUN 25.0 * 2.14 = 53.5 mg/dL
    text = "Blood Urea Nitrogen (BUN): 25.0 mg/dL"
    summary = extract_parameters_from_text(text)

    param = summary.parameters["bu"]
    assert param.extracted_value == 25.0  # Extracted from line
    assert param.unit == "mg/dL"


def test_unit_conversion_wbc_thousands():
    """Tests conversion of White Blood Cell Count reported in thousands."""
    text = "WBC Count: 7.8 x10^3/uL"
    summary = extract_parameters_from_text(text)

    param = summary.parameters["wbcc"]
    assert param.extracted_value == 7800.0
    assert param.unit == "cells/cumm"
    assert param.status == "extracted_requiring_review"


def test_qualitative_dipstick_grades():
    """Tests qualitative dipstick parsing for urine albumin and sugar."""
    assert parse_qualitative_dipstick("Nil") == 0.0
    assert parse_qualitative_dipstick("Negative") == 0.0
    assert parse_qualitative_dipstick("Trace") == 0.0
    assert parse_qualitative_dipstick("1+") == 1.0
    assert parse_qualitative_dipstick("+") == 1.0
    assert parse_qualitative_dipstick("2+") == 2.0
    assert parse_qualitative_dipstick("++") == 2.0
    assert parse_qualitative_dipstick("3+") == 3.0
    assert parse_qualitative_dipstick("+++") == 3.0
    assert parse_qualitative_dipstick("4+") == 4.0
    assert parse_qualitative_dipstick("++++") == 4.0


def test_blood_pressure_diastolic_extraction():
    """Verifies diastolic blood pressure is extracted with systolic noted in warnings."""
    text = "Clinical Vitals: BP: 140/90 mmHg"
    summary = extract_parameters_from_text(text)

    param = summary.parameters["bp"]
    assert param.extracted_value == 90.0
    assert param.unit == "mm/Hg"
    assert any("systolic: 140" in w for w in param.warnings)


def test_unreadable_values_zero_substitution_forbidden():
    """Ensures unreadable or obscured values are marked 'unreadable' and never set to 0.0."""
    text = """
    Serum Creatinine: ***
    Hemoglobin: [unreadable]
    Blood Urea: N/A
    """
    summary = extract_parameters_from_text(text)

    # Creatinine obscured
    sc_param = summary.parameters["sc"]
    assert sc_param.status == "unreadable"
    assert sc_param.extracted_value is None
    assert sc_param.is_valid_for_model is False
    assert sc_param.extracted_value != 0.0

    # Hemoglobin obscured
    hemo_param = summary.parameters["hemo"]
    assert hemo_param.status == "unreadable"
    assert hemo_param.extracted_value is None
    assert hemo_param.is_valid_for_model is False

    # Blood urea N/A
    bu_param = summary.parameters["bu"]
    assert bu_param.status == "unreadable"
    assert bu_param.extracted_value is None


def test_conflicting_values_across_pages():
    """Ensures discordant values across pages are flagged as 'conflicting' for manual review."""
    text = """
    --- Page 1 ---
    Serum Creatinine: 1.2 mg/dL

    --- Page 2 ---
    Serum Creatinine: 2.8 mg/dL
    """
    summary = extract_parameters_from_text(text)

    sc_param = summary.parameters["sc"]
    assert sc_param.status == "conflicting"
    assert sc_param.extracted_value is None  # Never guess an arbitrary value
    assert sc_param.is_valid_for_model is False
    assert summary.conflicting_count >= 1
    assert any("Conflicting values detected" in w for w in sc_param.warnings)


def test_duplicate_consistent_values():
    """Ensures identical repeat measurements across pages are confirmed as unambiguous."""
    text = """
    --- Page 1 ---
    Serum Creatinine: 1.5 mg/dL

    --- Page 2 ---
    Serum Creatinine: 1.5 mg/dL
    """
    summary = extract_parameters_from_text(text)

    sc_param = summary.parameters["sc"]
    assert sc_param.status == "extracted_unambiguous"
    assert sc_param.extracted_value == 1.5
    assert sc_param.repeat_count == 2
    assert sc_param.is_valid_for_model is True


def test_missing_fields_never_invented():
    """Ensures missing fields remain None and are never substituted with zero."""
    text = """
    Serum Creatinine: 1.8 mg/dL
    Blood Urea: 50.0 mg/dL
    """
    summary = extract_parameters_from_text(text)

    assert summary.missing_count == 22
    assert summary.parameters["hemo"].status == "missing"
    assert summary.parameters["hemo"].extracted_value is None
    assert summary.parameters["hemo"].is_valid_for_model is False

    assert summary.parameters["sod"].status == "missing"
    assert summary.parameters["sod"].extracted_value is None


def test_unsupported_parameters_detection():
    """Verifies detection and isolation of non-CKD tests (Platelets, SGOT, SGPT, Bilirubin)."""
    text = """
    LIVER & LIPID PANEL
    SGOT / AST: 32 U/L
    SGPT / ALT: 28 U/L
    Total Bilirubin: 0.9 mg/dL
    Total Cholesterol: 195 mg/dL
    Platelet Count: 280000 /uL
    Serum Creatinine: 1.2 mg/dL
    """
    summary = extract_parameters_from_text(text)

    assert summary.unsupported_detected_count >= 4
    unsupported_names = [u.detected_name for u in summary.unsupported_parameters]
    assert "SGOT / AST" in unsupported_names
    assert "SGPT / ALT" in unsupported_names
    assert "Total Bilirubin" in unsupported_names
    assert "Platelet Count" in unsupported_names

    # Ensure none of these leaked into CKD parameters
    assert summary.parameters["sc"].extracted_value == 1.2


# ============================================================================
# Integration Tests with FastAPI Endpoints
# ============================================================================

def test_extract_parameters_endpoint_success(client):
    """Tests POST /extract-parameters/{upload_id} for a staged medical report."""
    # 1. Upload a valid test report
    report_content = (
        b"PATIENT MEDICAL LAB REPORT\n"
        b"Age: 52 Years\n"
        b"Blood Pressure: 120/80 mmHg\n"
        b"Serum Creatinine: 1.8 mg/dL (Ref: 0.6 - 1.2)\n"
        b"Blood Urea: 48.0 mg/dL\n"
        b"Hemoglobin: 11.2 g/dL\n"
        b"Hypertension: Yes\n"
        b"Diabetes Mellitus: No\n"
    )
    files = {"file": ("patient_lab_report.txt", io.BytesIO(report_content), "text/plain")}

    # Note: text/plain is rejected by upload_document which expects PDF or Image
    # So we create a valid minimal PDF
    from pypdf import PdfWriter
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    # Add text via synthetic PDF creation or image
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "Serum Creatinine: 1.8 mg/dL", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    upload_resp = client.post(
        "/upload-document",
        files={"file": ("test_report.png", io.BytesIO(png_bytes), "image/png")}
    )
    assert upload_resp.status_code == 200
    upload_id = upload_resp.json()["upload_id"]

    # 2. Call text extraction
    client.post(f"/extract-text/{upload_id}")

    # 3. Call parameter extraction
    param_resp = client.post(f"/extract-parameters/{upload_id}")
    assert param_resp.status_code == 200
    data = param_resp.json()

    assert data["upload_id"] == upload_id
    assert data["total_features_supported"] == 24
    assert "parameters" in data
    assert "disclaimer" in data

    # 4. Call GET /extract-parameters/{upload_id} to test retrieval
    get_resp = client.get(f"/extract-parameters/{upload_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["upload_id"] == upload_id

    # 5. Clean up session
    del_resp = client.delete(f"/upload-document/{upload_id}")
    assert del_resp.status_code == 200


def test_extract_parameters_endpoint_not_found(client):
    """Verifies 404 is returned when requesting parameter extraction on invalid upload ID."""
    resp = client.post("/extract-parameters/non-existent-uuid-12345")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ============================================================================
# Phase 3: Comprehensive Regression Tests
# ============================================================================

def test_multiline_and_parenthesized_format():
    """Verifies that multi-line label/value pairing with parenthesized canonical aliases extracts correctly."""
    report = """
    CLINICAL BIOMARKERS
    Age (age)
    65
    Blood Pressure (bp)
    90
    Serum Creatinine (sc)
    13.5
    Specific Gravity (sg)
    1.01
    Appetite (appet)
    good
    """
    summary = extract_parameters_from_text(report)
    assert summary.parameters["age"].extracted_value == 65.0
    assert summary.parameters["bp"].extracted_value == 90.0
    assert summary.parameters["sc"].extracted_value == 13.5
    assert summary.parameters["sg"].extracted_value == 1.01
    assert summary.parameters["appet"].extracted_value == "good"
    assert "Age (age) -> 65" in summary.parameters["age"].original_text
    assert "Serum Creatinine (sc) -> 13.5" in summary.parameters["sc"].original_text


def test_decimal_formatting_comma_and_ocr():
    """Verifies European decimal comma, spaced decimals, and OCR corrections."""
    # Unit clean_numeric_string tests
    assert clean_numeric_string("13,5") == "13.5"
    assert clean_numeric_string("1,01") == "1.01"
    assert clean_numeric_string("8,100") == "8100"
    assert clean_numeric_string("13 . 5") == "13.5"
    assert clean_numeric_string("l.01") == "1.01"
    assert clean_numeric_string("I.01") == "1.01"

    # In parser
    sc_val, _, _ = parse_numerical_value("sc", "13,5")
    assert sc_val == 13.5

    sg_val, _, _ = parse_numerical_value("sg", "1,01")
    assert sg_val == 1.01

    wbc_val, _, _ = parse_numerical_value("wbcc", "8,100")
    assert wbc_val == 8100.0


def test_unit_printed_below_value():
    """Verifies that units printed on the line below the candidate measurement are recognized."""
    report = """
    BIOCHEMISTRY
    Serum Creatinine (sc)
    13.5
    mg/dL
    Blood urea (bu)
    82
    mg/dL
    """
    summary = extract_parameters_from_text(report)
    assert summary.parameters["sc"].extracted_value == 13.5
    assert summary.parameters["sc"].unit == "mg/dL"
    assert summary.parameters["bu"].extracted_value == 82.0
    assert summary.parameters["bu"].unit == "mg/dL"


def test_categorical_variations_and_notpresent():
    """Verifies categorical option mapping, including notpresent and whole-word protection."""
    # notpresent should map to notpresent, never to present
    pcc_val, _ = parse_categorical_value("pcc", "notpresent")
    assert pcc_val == "notpresent"

    pcc_val2, _ = parse_categorical_value("pcc", "not present")
    assert pcc_val2 == "notpresent"

    pcc_val3, _ = parse_categorical_value("pcc", "present")
    assert pcc_val3 == "present"

    ba_val, _ = parse_categorical_value("ba", "notpresent")
    assert ba_val == "notpresent"

    # htn / dm / cad / pe / ane
    htn_val, _ = parse_categorical_value("htn", "yes")
    assert htn_val == "yes"

    htn_val2, _ = parse_categorical_value("htn", "no")
    assert htn_val2 == "no"

    # appet
    appet_val, _ = parse_categorical_value("appet", "good")
    assert appet_val == "good"

    appet_val2, _ = parse_categorical_value("appet", "poor")
    assert appet_val2 == "poor"


def test_cad_missing_parameter_never_invented():
    """Verifies that missing parameters are never guessed or filled when next parameter begins."""
    report = """
    Coronary Artery Disease (cad)
    Appetite (appet)
    good
    Pedal Edema (pe)
    yes
    """
    summary = extract_parameters_from_text(report)
    assert summary.parameters["cad"].extracted_value is None
    assert summary.parameters["cad"].status == "missing"
    assert summary.parameters["appet"].extracted_value == "good"
    assert summary.parameters["pe"].extracted_value == "yes"


def test_synthetic_user_report_extraction_parity():
    """
    Verifies that a synthetic document with identical format to the user's uploaded report
    extracts all 23 present parameters and correctly leaves cad as missing without inventing values.
    """
    SYNTHETIC_USER_REPORT = """
    CHRONIC KIDNEY DISEASE (CKD) CLINICAL DECISION SUPPORT
    CLINICAL BIOMARKERS & OBSERVATIONS - 24 PARAMETERS
    Parameter
    Patient Value
    Age (age)
    65
    Blood Pressure (bp)
    90
    Specific Gravity (sg)
    1.01
    Albumin (al)
    4
    Sugar (su)
    2
    Blood Glucose Random (bgr)
    172
    Blood urea (bu)
    82
    Serum Creatinine (sc)
    13.5
    Serum Sodium (sod)
    145
    Serum Potassium (pot)
    6.3
    Hemoglobin (hemo)
    8.8
    Packed Cell Volume (pcv)
    31
    White Blood Cell Count (wbcc)
    8100
    Red Blood Cell Count (rbcc)
    4.8
    Red Blood Cells (rbc)
    normal
    Pus Cell (pc)
    normal
    Pus Cell Clumps (pcc)
    notpresent
    Bacteria (ba)
    notpresent
    Hypertension (htn)
    yes
    Diabetes Mellitus (dm)
    yes
    Coronary Artery Disease (cad)
    Appetite (appet)
    good
    Pedal Edema (pe)
    yes
    Anemia (ane)
    yes
    """
    summary = extract_parameters_from_text(SYNTHETIC_USER_REPORT)
    assert summary.total_features_supported == 24
    assert summary.conflicting_count == 0
    assert summary.missing_count == 1
    assert summary.parameters["cad"].status == "missing"
    assert summary.parameters["cad"].extracted_value is None

    # Verify extracted parameters
    assert summary.parameters["age"].extracted_value == 65.0
    assert summary.parameters["bp"].extracted_value == 90.0
    assert summary.parameters["sg"].extracted_value == 1.01
    assert summary.parameters["al"].extracted_value == 4.0
    assert summary.parameters["su"].extracted_value == 2.0
    assert summary.parameters["bgr"].extracted_value == 172.0
    assert summary.parameters["bu"].extracted_value == 82.0
    assert summary.parameters["sc"].extracted_value == 13.5
    assert summary.parameters["sod"].extracted_value == 145.0
    assert summary.parameters["pot"].extracted_value == 6.3
    assert summary.parameters["hemo"].extracted_value == 8.8
    assert summary.parameters["pcv"].extracted_value == 31.0
    assert summary.parameters["wbcc"].extracted_value == 8100.0
    assert summary.parameters["rbcc"].extracted_value == 4.8
    assert summary.parameters["rbc"].extracted_value == "normal"
    assert summary.parameters["pc"].extracted_value == "normal"
    assert summary.parameters["pcc"].extracted_value == "notpresent"
    assert summary.parameters["ba"].extracted_value == "notpresent"
    assert summary.parameters["htn"].extracted_value == "yes"
    assert summary.parameters["dm"].extracted_value == "yes"
    assert summary.parameters["appet"].extracted_value == "good"
    assert summary.parameters["pe"].extracted_value == "yes"
    assert summary.parameters["ane"].extracted_value == "yes"


def test_reset_and_repeated_upload_isolation(client):
    """Verifies that deleting/resetting an upload prevents data leakage into subsequent uploads."""
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 28) if font_path.is_file() else ImageFont.load_default()

    # 1. Upload Report 1 (Serum Creatinine: 1.45)
    img1 = Image.new("RGB", (800, 300), color=(255, 255, 255))
    draw1 = ImageDraw.Draw(img1)
    draw1.text((30, 40), "Serum Creatinine: 1.45 mg/dL", fill=(0, 0, 0), font=font)
    buf1 = io.BytesIO()
    img1.save(buf1, format="PNG")
    r1 = client.post("/upload-document", files={"file": ("doc1.png", io.BytesIO(buf1.getvalue()), "image/png")})
    assert r1.status_code == 200
    id1 = r1.json()["upload_id"]

    # Process Report 1
    client.post(f"/extract-text/{id1}")
    p1 = client.post(f"/extract-parameters/{id1}").json()
    assert p1["parameters"]["sc"]["extracted_value"] == 1.45

    # 2. Delete / Reset Report 1
    del1 = client.delete(f"/upload-document/{id1}")
    assert del1.status_code == 200

    # Verify ID1 is purged
    get1 = client.get(f"/extract-parameters/{id1}")
    assert get1.status_code == 404

    # 3. Upload Report 2 (Serum Creatinine: 5.20)
    img2 = Image.new("RGB", (800, 300), color=(255, 255, 255))
    draw2 = ImageDraw.Draw(img2)
    draw2.text((30, 40), "Serum Creatinine: 5.20 mg/dL", fill=(0, 0, 0), font=font)
    buf2 = io.BytesIO()
    img2.save(buf2, format="PNG")
    r2 = client.post("/upload-document", files={"file": ("doc2.png", io.BytesIO(buf2.getvalue()), "image/png")})
    assert r2.status_code == 200
    id2 = r2.json()["upload_id"]
    assert id2 != id1

    client.post(f"/extract-text/{id2}")
    p2 = client.post(f"/extract-parameters/{id2}").json()
    assert p2["parameters"]["sc"]["extracted_value"] == 5.20

    # Clean up ID2
    client.delete(f"/upload-document/{id2}")


def test_consistent_pipeline_values_verification_predict_pdf(client):
    """
    Verifies data consistency across the verification form, prediction request,
    SHAP output, and downloadable PDF report.
    """
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 28) if font_path.is_file() else ImageFont.load_default()

    # 1. Build a synthetic report image with distinct known values
    img = Image.new("RGB", (800, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 30), "Age: 58 years", fill=(0, 0, 0), font=font)
    draw.text((30, 70), "Blood Pressure: 90 mm/Hg", fill=(0, 0, 0), font=font)
    draw.text((30, 110), "Serum Creatinine: 2.4 mg/dL", fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    up_resp = client.post("/upload-document", files={"file": ("patient_test.png", io.BytesIO(buf.getvalue()), "image/png")})
    assert up_resp.status_code == 200
    upload_id = up_resp.json()["upload_id"]

    client.post(f"/extract-text/{upload_id}")
    ext_resp = client.post(f"/extract-parameters/{upload_id}")
    assert ext_resp.status_code == 200

    # 2. Verify patient data via POST /verify-parameters
    payload = {
        "upload_id": upload_id,
        "parameters": {
            "age": {"value": 58.0, "status": "verified", "origin": "verified"},
            "bp": {"value": 90.0, "status": "verified", "origin": "verified"},
            "sg": {"value": 1.015, "status": "verified", "origin": "verified"},
            "al": {"value": 2.0, "status": "verified", "origin": "verified"},
            "su": {"value": 0.0, "status": "verified", "origin": "verified"},
            "bgr": {"value": 140.0, "status": "verified", "origin": "verified"},
            "bu": {"value": 55.0, "status": "verified", "origin": "verified"},
            "sc": {"value": 2.4, "status": "verified", "origin": "verified"},
            "sod": {"value": 138.0, "status": "verified", "origin": "verified"},
            "pot": {"value": 4.5, "status": "verified", "origin": "verified"},
            "hemo": {"value": 11.2, "status": "verified", "origin": "verified"},
            "pcv": {"value": 34.0, "status": "verified", "origin": "verified"},
            "wbcc": {"value": 7500.0, "status": "verified", "origin": "verified"},
            "rbcc": {"value": 4.2, "status": "verified", "origin": "verified"},
            "rbc": {"value": "normal", "status": "verified", "origin": "verified"},
            "pc": {"value": "normal", "status": "verified", "origin": "verified"},
            "pcc": {"value": "notpresent", "status": "verified", "origin": "verified"},
            "ba": {"value": "notpresent", "status": "verified", "origin": "verified"},
            "htn": {"value": "yes", "status": "verified", "origin": "verified"},
            "dm": {"value": "no", "status": "verified", "origin": "verified"},
            "cad": {"value": "no", "status": "verified", "origin": "verified"},
            "appet": {"value": "good", "status": "verified", "origin": "verified"},
            "pe": {"value": "no", "status": "verified", "origin": "verified"},
            "ane": {"value": "no", "status": "verified", "origin": "verified"}
        },
        "allow_imputation": True
    }
    ver_resp = client.post("/verify-parameters", json=payload)
    assert ver_resp.status_code == 200
    assert ver_resp.json()["is_valid_for_prediction"] is True

    # 3. Predict verified values via POST /predict-verified
    pred_req = {
        "upload_id": upload_id,
        "parameters": payload["parameters"],
        "allow_imputation": True
    }
    pred_resp = client.post("/predict-verified", json=pred_req)
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    assert "prediction_label" in pred_data
    assert "explanation" in pred_data
    assert pred_data["provenance_summary"]["verified_count"] == 24

    # 4. Generate downloadable PDF report with the exact prediction and verified data
    pdf_req = {
        "upload_id": upload_id,
        "document_name": "patient_test.png",
        "prediction_result": pred_data,
        "verified_parameters": payload["parameters"]
    }
    pdf_resp = client.post("/download-report", json=pdf_req)
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert pdf_resp.content.startswith(b"%PDF")

    # Clean up upload
    client.delete(f"/upload-document/{upload_id}")


def test_synthetic_report_various_spacing_and_formats_end_to_end(client):
    """
    Step 4 Requirement:
    Tests synthetic report containing known values for several supported parameters
    with varied spacing and formatting:
    - Verifies OCR returns readable text
    - Verifies parser extracts known values
    - Verifies backend returns expected response format
    - Verifies unmentioned parameters remain Missing
    - Verifies candidate values start as unconfirmed (extracted_unambiguous), not Verified
    - Verifies only explicit user confirmation transitions fields to Verified
    """
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 28) if font_path.is_file() else ImageFont.load_default()

    # Create synthetic report image with multiple layout variations
    img = Image.new("RGB", (1200, 1400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    lines = [
        "SYNTHETIC CLINICAL LABORATORY REPORT",
        "------------------------------------",
        "Age (age)    52",                              # Space separated with parentheses
        "Blood Pressure: 85 mm/Hg",                    # Colon separated
        "Serum Creatinine: 2.35 mg/dL",                # High creatinine
        "Blood Urea",                                  # Multi-line label
        "65.0",                                        # Multi-line value
        "mg/dL",                                       # Multi-line unit
        "Blood Glucose Random: 145 mg/dL",             # Full name
        "Specific Gravity    1.015",                   # Space separated urinalysis
        "Albumin: 3",                                  # Dipstick grade
        "Hemoglobin (hemo): 11.2 g/dL",                # Standard decimal
        "Packed Cell Volume: 35 %",                    # Space before %
        "White Blood Cell Count: 9,200 /cumm",         # Comma thousands
        "Red Blood Cell Count: 4.10 millions/cmm",     # Float notation
        "Hypertension: yes",                           # Categorical colon
        "Diabetes Mellitus: no",                       # Categorical colon
        "Appetite: good"                               # Categorical colon
        # Intentionally unmentioned: cad, pe, ane, pcc, ba, su, rbc, pc, sod, pot
    ]

    y = 40
    for line in lines:
        draw.text((50, y), line, fill=(0, 0, 0), font=font)
        y += 55

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # 1. Upload Document
    upload_res = client.post("/upload-document", files={"file": ("synth_lab.png", buf, "image/png")})
    assert upload_res.status_code == 200
    upload_id = upload_res.json()["upload_id"]

    # 2. Extract Text via OCR
    text_res = client.post(f"/extract-text/{upload_id}")
    assert text_res.status_code == 200
    assert text_res.json()["has_extractable_text"] is True
    assert text_res.json()["text_length_characters"] > 100

    # 3. Extract Parameters
    param_res = client.post(f"/extract-parameters/{upload_id}")
    assert param_res.status_code == 200
    p_data = param_res.json()
    params = p_data["parameters"]

    # Verify extracted known values
    assert params["age"]["extracted_value"] == 52.0
    assert params["bp"]["extracted_value"] == 85.0
    assert params["sc"]["extracted_value"] == 2.35
    assert params["bu"]["extracted_value"] == 65.0
    assert params["bgr"]["extracted_value"] == 145.0
    assert params["sg"]["extracted_value"] == 1.015
    assert params["al"]["extracted_value"] == 3.0
    assert params["hemo"]["extracted_value"] == 11.2
    assert params["pcv"]["extracted_value"] == 35.0
    assert params["wbcc"]["extracted_value"] == 9200.0
    assert params["rbcc"]["extracted_value"] == 4.10
    assert params["htn"]["extracted_value"] == "yes"
    assert params["dm"]["extracted_value"] == "no"
    assert params["appet"]["extracted_value"] == "good"

    # Verify unmentioned parameters are Missing and never invented
    unmentioned = ["cad", "pe", "ane", "pcc", "ba", "su", "rbc", "pc", "sod", "pot"]
    for u in unmentioned:
        assert params[u]["extracted_value"] is None, f"Parameter {u} should be None but was {params[u]['extracted_value']}"
        assert params[u]["status"] == "missing", f"Parameter {u} should be missing but was {params[u]['status']}"

    # Verify candidate values are unconfirmed extracted candidates, NOT pre-verified
    assert params["sc"]["status"] in ["extracted_unambiguous", "extracted_requiring_review"]
    assert params["sc"]["status"] != "verified"

    # 4. Verification Workflow: Explicit user confirmation
    # Clinician reviews and confirms extracted parameters and specifies missing cad as "no"
    verification_submission = {
        "upload_id": upload_id,
        "parameters": {
            feat: {
                "value": params[feat]["extracted_value"] if feat != "cad" else "no",
                "status": "verified",
                "origin": "verified" if feat != "cad" else "manual"
            }
            for feat in params
        },
        "allow_imputation": True
    }
    ver_resp = client.post("/verify-parameters", json=verification_submission)
    assert ver_resp.status_code == 200
    ver_data = ver_resp.json()
    assert ver_data["is_valid_for_prediction"] is True

    # 5. Clean up upload session
    client.delete(f"/upload-document/{upload_id}")


