"""
Unit and integration tests for Clinical PDF Report Generation and Workflow Endpoints.

Verifies:
1. Generation of multi-page vector PDF reports with correct %PDF header and non-empty content
2. Accurate reflection of all 11 required report sections:
   - Section 1: Title and generation timestamp
   - Section 2: Unique non-identifying analysis reference
   - Section 3: Document processing status
   - Section 4: Extracted and user-verified parameters, with units
   - Section 5: Missing or unresolved information and imputation audit
   - Section 6: Model prediction
   - Section 7: Model probability, correctly interpreted
   - Section 8: SHAP explanation, feature contributions, and directionality guide
   - Section 9: Data extraction limitations
   - Section 10: Model limitations
   - Section 11: Medical disclaimer
3. Handling of missing fields (partial or minimal prediction payloads)
4. Handling of special characters (TeX math $, HTML <script>, ampersands, unicode µmol/L, control chars)
5. Handling of long content (long document names, lengthy feature values, extended strings)
6. Handling of failed predictions (prediction failure codes, error messages)
7. On-demand in-memory streaming with zero disk retention
8. POST /download-report HTTP endpoint response, streaming headers, and schema validation
9. End-to-end integration across upload, extraction, verification, prediction, and report download
"""

import io
import os
import pytest
import pypdf
from fastapi.testclient import TestClient
from pathlib import Path

from backend.main import app
from src.report_generator import generate_clinical_pdf_report, sanitize_pdf_text
from src.config import ALL_FEATURE_NAMES


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def sample_prediction_data():
    """Complete synthetic prediction and explanation fixture."""
    return {
        "prediction_label": "Chronic Kidney Disease (CKD)",
        "prediction_code": 1,
        "probabilities": {
            "ckd_probability": 1.0,
            "non_ckd_probability": 0.0
        },
        "probability_caveat": "Decision Tree leaf probabilities reflect empirical sample proportions.",
        "clinical_distinction": {
            "model_prediction": "DecisionTree classification: CKD",
            "model_probability": "100.0% leaf sample purity",
            "clinical_diagnosis": "CANNOT be established by machine learning."
        },
        "provenance_summary": {
            "verified_count": 16,
            "extracted_count": 16,
            "manual_count": 2,
            "missing_count": 6,
            "imputation_applied": True,
            "imputation_strategy": "Training cohort median/mode"
        },
        "explanation_method": "SHAP (TreeExplainer)",
        "explanation": {
            "base_value": 0.625,
            "patient_model_output": 1.0,
            "output_scale": "Probability Scale [0.0 to 1.0]",
            "scale_interpretation": "TreeExplainer attributions reflect the probability scale of the leaf nodes.",
            "directionality_guide": {
                "positive_shap": "Pushes toward Class 1 (CKD)",
                "negative_shap": "Pushes toward Class 0 (Non-CKD)"
            },
            "causality_caveat": "Feature attributions reflect statistical associations, not biological causation.",
            "top_risk_drivers": [
                {
                    "feature": "hemo",
                    "clinical_label": "Hemoglobin (g/dL)",
                    "patient_value": 8.8,
                    "transformed_value": 8.8,
                    "shap_value": 0.215,
                    "direction": "Increases Predicted CKD Probability",
                    "effect": "increases_risk",
                    "unit": "g/dL"
                },
                {
                    "feature": "sc",
                    "clinical_label": "Serum Creatinine (mg/dL)",
                    "patient_value": 3.4,
                    "transformed_value": 3.4,
                    "shap_value": 0.142,
                    "direction": "Increases Predicted CKD Probability",
                    "effect": "increases_risk",
                    "unit": "mg/dL"
                }
            ],
            "top_protective_factors": [],
            "most_influential_features": [
                {
                    "feature": "hemo",
                    "clinical_label": "Hemoglobin (g/dL)",
                    "patient_value": 8.8,
                    "transformed_value": 8.8,
                    "shap_value": 0.215,
                    "direction": "Increases Predicted CKD Probability",
                    "effect": "increases_risk",
                    "unit": "g/dL"
                }
            ],
            "all_feature_contributions": [
                {
                    "feature": feat,
                    "clinical_label": feat.upper(),
                    "patient_value": 1.0 if idx < 12 else None,
                    "transformed_value": 1.0,
                    "shap_value": 0.05 if idx == 0 else 0.0,
                    "direction": "Neutral",
                    "effect": "neutral",
                    "unit": ""
                }
                for idx, feat in enumerate(ALL_FEATURE_NAMES)
            ]
        },
        "clinical_summary": "Decision Tree leaf classified patient as CKD based on renal markers.",
        "waterfall_plot_base64": None,
        "medical_disclaimer": "RESEARCH & CLINICAL DECISION SUPPORT ONLY. NOT A DIAGNOSIS."
    }


def test_generate_clinical_pdf_report_validity(sample_prediction_data):
    """Verify that report generator produces a well-formed vector PDF byte stream."""
    pdf_bytes = generate_clinical_pdf_report(
        prediction_data=sample_prediction_data,
        document_name="synthetic_patient_report.pdf",
        session_id="test-session-123"
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 10000, "Generated PDF is unexpectedly small"
    assert pdf_bytes.startswith(b"%PDF"), "Missing standard PDF header signature"
    assert b"%%EOF" in pdf_bytes, "Missing standard PDF EOF marker"


def test_generate_clinical_pdf_report_all_11_sections(sample_prediction_data):
    """
    Verify that the generated PDF contains all 11 required report sections across 3 pages:
    1. Title and timestamp
    2. Unique non-identifying analysis reference
    3. Document processing status
    4. Extracted and user-verified parameters
    5. Missing or unresolved information
    6. Model prediction
    7. Model probability
    8. SHAP explanation and feature contributions
    9. Data extraction limitations
    10. Model limitations
    11. Medical disclaimer
    """
    pdf_bytes = generate_clinical_pdf_report(
        prediction_data=sample_prediction_data,
        document_name="synthetic_renal_panel.pdf",
        session_id="sess-alpha-99"
    )

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) == 3, f"Expected 3 pages, found {len(reader.pages)}"

    extracted_pages = [page.extract_text() for page in reader.pages]
    all_text = "\n".join(extracted_pages)

    # 1. Title & Timestamp
    assert "CHRONIC KIDNEY DISEASE (CKD) AI PREDICTION REPORT" in all_text
    assert "Generated:" in all_text

    # 2. Unique Non-Identifying Analysis Reference
    assert "ANALYSIS REFERENCE:" in all_text
    assert "CKD-REF-" in all_text

    # 3. Document Processing Status
    assert "DOCUMENT PROCESSING STATUS:" in all_text
    assert "Privacy: Ephemeral Session" in all_text

    # 4. Extracted & User-Verified Parameters
    assert "SECTION 4: EXTRACTED & USER-VERIFIED PARAMETERS" in all_text
    assert "PARAMETER NAME" in all_text

    # 5. Missing or Unresolved Information
    assert "SECTION 5: MISSING OR UNRESOLVED INFORMATION" in all_text
    assert "Imputation" in all_text

    # 6. Model Prediction
    assert "SECTION 6: MODEL PREDICTION OUTCOME" in all_text
    assert "CHRONIC KIDNEY DISEASE" in all_text

    # 7. Model Probability
    assert "SECTION 7: MODEL PROBABILITY & LEAF PURITY" in all_text
    assert "Terminal Leaf Purity:" in all_text

    # 8. SHAP Explanation & Contributions
    assert "SECTION 8: SHAP EXPLANATION & FEATURE CONTRIBUTIONS" in all_text
    assert "DIRECTIONALITY GUIDE" in all_text

    # 9. Data Extraction Limitations
    assert "SECTION 9: DATA EXTRACTION LIMITATIONS" in all_text
    assert "OCR Sensitivity" in all_text

    # 10. Model Limitations
    assert "SECTION 10: MODEL LIMITATIONS" in all_text
    assert "Cross-Sectional Only" in all_text
    assert "DecisionTreeClassifier" in all_text

    # 11. Medical Disclaimer
    assert "SECTION 11: FORMAL MEDICAL & REGULATORY DISCLAIMER" in all_text
    assert "DOES NOT CONSTITUTE A FORMAL CLINICAL DIAGNOSIS" in all_text


def test_generate_clinical_pdf_report_non_ckd_case(sample_prediction_data):
    """Verify report generation handles Non-CKD cases with protective attributions cleanly."""
    non_ckd_data = dict(sample_prediction_data)
    non_ckd_data["prediction_label"] = "Non-CKD"
    non_ckd_data["prediction_code"] = 0
    non_ckd_data["probabilities"] = {"ckd_probability": 0.0, "non_ckd_probability": 1.0}
    non_ckd_data["explanation"]["patient_model_output"] = 0.0

    pdf_bytes = generate_clinical_pdf_report(
        prediction_data=non_ckd_data,
        document_name="healthy_scan.png",
        session_id="healthy-session-456"
    )

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    all_text = "\n".join(p.extract_text() for p in reader.pages)
    assert "NON-CKD" in all_text
    assert "Terminal Leaf Purity: 100.0%" in all_text


def test_report_generation_missing_fields():
    """
    Verify report generator handles missing or incomplete input fields safely:
    - Missing probabilities
    - Missing explanation
    - Missing provenance summary
    - Missing verified_parameters
    - Empty dictionaries
    """
    partial_prediction = {
        "prediction_label": "Chronic Kidney Disease (CKD)",
        "prediction_code": 1
        # Omitted: probabilities, explanation, provenance_summary, etc.
    }

    pdf_bytes = generate_clinical_pdf_report(
        prediction_data=partial_prediction,
        verified_parameters=None,
        document_name=None,
        session_id=None
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) == 3

    all_text = "\n".join(p.extract_text() for p in reader.pages)
    assert "CHRONIC KIDNEY DISEASE (CKD)" in all_text
    assert "Model Probability: Not Applicable" in all_text or "Probability Uncalibrated" in all_text
    assert "Manual Clinical Entry / Structured Form" in all_text


def test_report_generation_special_characters():
    """
    Verify text sanitization and rendering of untrusted strings containing:
    - TeX math delimiters ($100, $x^2$)
    - HTML script tags (<script>alert(1)</script>)
    - Special symbols (&, ", ', %)
    - Unicode characters (µmol/L, €, é, ñ)
    - Control and newline characters (\r\n, \x00, \x07)
    """
    special_name = "report_$100_<script>alert('xss')</script>_&_100%_µmol.pdf"
    special_params = {
        "sc": {
            "value": "1.5 $USD µmol/L\r\n\x00",
            "unit": "mg/dL & µmol",
            "status": "verified"
        },
        "bu": {
            "value": "45.0 <script>",
            "unit": "mg/dL",
            "status": "extracted"
        }
    }

    prediction_data = {
        "prediction_label": "Chronic Kidney Disease (CKD) & Stage 3$",
        "prediction_code": 1,
        "probabilities": {"ckd_probability": 0.88, "non_ckd_probability": 0.12},
        "explanation": {
            "base_value": 0.625,
            "patient_model_output": 0.88,
            "all_feature_contributions": [
                {
                    "feature": "sc",
                    "clinical_label": "Creatinine $µmol/L",
                    "patient_value": "1.5$ & special",
                    "shap_value": 0.12
                }
            ]
        }
    }

    pdf_bytes = generate_clinical_pdf_report(
        prediction_data=prediction_data,
        verified_parameters=special_params,
        document_name=special_name,
        session_id="special$123"
    )

    assert pdf_bytes.startswith(b"%PDF")
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) == 3
    # Ensure sanitization worked and file is readable
    text = reader.pages[0].extract_text()
    assert len(text) > 200


def test_sanitize_pdf_text_utility():
    """Verify specific behavior of the sanitize_pdf_text helper."""
    # Control chars removed
    assert sanitize_pdf_text("Hello\x00World\x07") == "Hello World"
    # Dollars escaped
    assert r"\$" in sanitize_pdf_text("$100 dollars")
    # Truncation
    assert sanitize_pdf_text("A" * 100, max_len=10) == "AAAAAAA..."
    # None handling
    assert sanitize_pdf_text(None) == ""


def test_report_generation_long_content():
    """
    Verify report generator handles extremely long content without crashing:
    - 500-char document name
    - 300-char parameter values
    - Lengthy missing parameters list
    """
    long_doc_name = "A" * 600 + ".pdf"
    long_params = {
        feat: {
            "value": "EXTREMELY_LONG_VALUE_" + "9" * 200,
            "unit": "UNUSUALLY_LONG_UNIT_" + "X" * 100,
            "status": "verified"
        }
        for feat in ALL_FEATURE_NAMES
    }

    prediction_data = {
        "prediction_label": "Chronic Kidney Disease (CKD) " + "X" * 100,
        "prediction_code": 1,
        "probabilities": {"ckd_probability": 0.99, "non_ckd_probability": 0.01},
        "explanation": {
            "base_value": 0.625,
            "patient_model_output": 0.99,
            "all_feature_contributions": [
                {
                    "feature": feat,
                    "clinical_label": "LABEL_" + feat + "_" + "Z" * 50,
                    "patient_value": "VALUE_" + "Y" * 50,
                    "shap_value": 0.01
                }
                for feat in ALL_FEATURE_NAMES
            ]
        }
    }

    pdf_bytes = generate_clinical_pdf_report(
        prediction_data=prediction_data,
        verified_parameters=long_params,
        document_name=long_doc_name,
        session_id="long-session-id-" + "7" * 50
    )

    assert pdf_bytes.startswith(b"%PDF")
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) == 3


def test_report_generation_failed_predictions():
    """
    Verify report generator handles failed or incomplete predictions gracefully:
    - prediction_label: 'Prediction Failed'
    - prediction_code: -1
    - error_message provided
    - Probabilities None
    - Explanation None
    """
    failed_prediction = {
        "prediction_label": "Prediction Failed",
        "prediction_code": -1,
        "error_message": "Feature validation failed: Serum Creatinine 95.0 exceeds physiological limits (0.1 to 30.0 mg/dL).",
        "probabilities": None,
        "explanation": None
    }

    pdf_bytes = generate_clinical_pdf_report(
        prediction_data=failed_prediction,
        document_name="abnormal_patient_chart.pdf",
        session_id="failed-sess-001"
    )

    assert pdf_bytes.startswith(b"%PDF")
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) == 3

    all_text = "\n".join(p.extract_text() for p in reader.pages)
    assert "PREDICTION INCOMPLETE / FAILED" in all_text
    assert "Status: Failed / Rejected" in all_text
    assert "Feature validation failed:" in all_text
    assert "Model Probability: Not Applicable" in all_text or "Inapplicable" in all_text
    assert "SECTION 11: FORMAL MEDICAL & REGULATORY DISCLAIMER" in all_text


def test_report_generation_on_demand_no_disk_retention(tmp_path):
    """Verify report generation is on-demand in-memory and does not leave temporary PDF files on disk."""
    initial_files = set(os.listdir("."))

    prediction_data = {
        "prediction_label": "Non-CKD",
        "prediction_code": 0,
        "probabilities": {"ckd_probability": 0.05, "non_ckd_probability": 0.95}
    }

    pdf_bytes = generate_clinical_pdf_report(
        prediction_data=prediction_data,
        document_name="memory_test.pdf",
        session_id="in-memory-session"
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0

    current_files = set(os.listdir("."))
    # No new files created in root directory
    assert current_files == initial_files


def test_download_report_endpoint_success(client, sample_prediction_data):
    """Verify POST /download-report returns 200 OK with application/pdf and attachment headers."""
    payload = {
        "upload_id": "session-test-abc",
        "document_name": "lab_report_patient_1.pdf",
        "prediction_result": sample_prediction_data,
        "verified_parameters": {
            "hemo": {"value": 8.8, "unit": "g/dL", "status": "verified"},
            "sc": {"value": 3.4, "unit": "mg/dL", "status": "verified"}
        }
    }

    response = client.post("/download-report", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "attachment; filename=" in response.headers["content-disposition"]
    assert "no-cache" in response.headers["cache-control"]
    assert response.content.startswith(b"%PDF")
    assert b"%%EOF" in response.content


def test_download_report_endpoint_rejects_missing_prediction(client):
    """Verify POST /download-report validates required payload schema and rejects invalid input with 422."""
    payload = {
        "upload_id": "test-session",
        "document_name": "test.pdf"
        # missing prediction_result
    }

    response = client.post("/download-report", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    # Ensure internal paths are not leaked
    assert "C:" not in str(data) and "/PROJECT" not in str(data)


def test_complete_document_workflow_integration(client):
    """
    Test complete multi-step document workflow through the backend:
    Step 1: Upload synthetic PDF document
    Step 2: Check upload status and file record
    Step 3: Extract parameters
    Step 4: Verify & Predict via /predict-verified
    Step 5: Download PDF clinical report
    Step 6: Purge / restart workflow via DELETE /upload-document
    """
    from tests.test_document_upload import VALID_PDF_BYTES

    # 1. Upload
    upload_res = client.post(
        "/upload-document",
        files={"file": ("synthetic_patient_chart.pdf", VALID_PDF_BYTES, "application/pdf")}
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    upload_id = upload_data["upload_id"]

    # 2. Extract parameters
    extract_res = client.post(f"/extract-parameters/{upload_id}")
    assert extract_res.status_code == 200
    extract_data = extract_res.json()
    assert "parameters" in extract_data

    # 3. Construct verified parameter payload
    verified_payload = {
        feat: {
            "canonical_name": feat,
            "clinical_label": feat,
            "value": 1.2 if feat == "sc" else 13.0 if feat == "hemo" else None,
            "unit": "mg/dL" if feat == "sc" else "g/dL" if feat == "hemo" else "",
            "origin": "verified" if feat in ["sc", "hemo"] else "extracted",
            "status": "verified" if feat in ["sc", "hemo"] else "missing",
            "source_text": None,
            "source_page": None,
            "warnings": []
        }
        for feat in ALL_FEATURE_NAMES
    }

    # 4. Predict verified
    predict_res = client.post(
        "/predict-verified",
        json={
            "upload_id": upload_id,
            "parameters": verified_payload,
            "allow_imputation": True,
            "include_plot": False
        }
    )
    assert predict_res.status_code == 200
    predict_data = predict_res.json()
    assert predict_data["prediction_label"] in ["Chronic Kidney Disease (CKD)", "Non-CKD"]
    assert "explanation" in predict_data
    assert predict_data["explanation"]["output_scale"] == "Probability Scale [0.0 to 1.0]"

    # 5. Download Clinical Report
    report_res = client.post(
        "/download-report",
        json={
            "upload_id": upload_id,
            "document_name": "synthetic_patient_chart.pdf",
            "prediction_result": predict_data,
            "verified_parameters": verified_payload
        }
    )
    assert report_res.status_code == 200
    assert report_res.headers["content-type"] == "application/pdf"
    assert report_res.content.startswith(b"%PDF")

    # 6. Purge session / restart
    delete_res = client.delete(f"/upload-document/{upload_id}")
    assert delete_res.status_code == 200
    delete_data = delete_res.json()
    assert delete_data["upload_id"] == upload_id
    assert delete_data["status"] == "removed"
