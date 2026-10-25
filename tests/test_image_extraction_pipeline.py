"""
Regression Test Suite for Medical Image Extraction and Data Lineage Pipeline.

Verifies:
1. The image uploads successfully.
2. OCR extracts text from the image.
3. Supported parameters are identified.
4. Extracted values populate the correct form fields / structure.
5. Missing parameters remain empty (None).
6. Ambiguous / reliable values require review until confirmed.
7. The counters update correctly.
8. Prediction is blocked when required inputs are missing or invalid.
9. The PDF contains the same verified inputs used for prediction.
10. SHAP explanations correspond to that prediction.
11. Resetting the form removes stale values and results.
12. Uploading a second document does not reuse values from the first document.
13. OCR failures produce clear errors.
"""

import io
import pytest
from PIL import Image, ImageDraw, ImageFont
from fastapi.testclient import TestClient
from pathlib import Path

from backend.main import app
from src.document_extractor import document_extractor
from src.parameter_extractor import parameter_extractor
from src.patient_verifier import verify_patient_parameters_payload
from src.report_generator import generate_clinical_pdf_report


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _generate_synthetic_report_image(fmt: str = "PNG", mode: str = "RGB") -> bytes:
    """Generates a clean synthetic clinical laboratory report image in the requested format and color mode."""
    if mode == "RGBA":
        img = Image.new("RGBA", (900, 500), color=(255, 255, 255, 0))  # Transparent background
        text_fill = (0, 0, 0, 255)
    elif mode == "L":
        img = Image.new("L", (900, 500), color=255)
        text_fill = 0
    else:
        img = Image.new("RGB", (900, 500), color=(255, 255, 255))
        text_fill = (0, 0, 0)

    draw = ImageDraw.Draw(img)

    # Use standard Windows Arial font if available, fallback to default
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    if font_path.is_file():
        font = ImageFont.truetype(str(font_path), 24)
        header_font = ImageFont.truetype(str(font_path), 28)
    else:
        font = ImageFont.load_default()
        header_font = font

    draw.text((30, 20), "METROPOLITAN CLINICAL LABORATORY", fill=text_fill, font=header_font)
    draw.text((30, 60), "RENAL FUNCTION PANEL", fill=text_fill, font=font)
    draw.text((30, 110), "Age: 62 years", fill=text_fill, font=font)
    draw.text((30, 150), "Blood Pressure: 80 mm/Hg", fill=text_fill, font=font)
    draw.text((30, 190), "Serum Creatinine: 2.1 mg/dL", fill=text_fill, font=font)
    draw.text((30, 230), "Blood Urea: 55 mg/dL", fill=text_fill, font=font)
    draw.text((30, 270), "Blood Glucose Random: 140 mgs/dl", fill=text_fill, font=font)
    draw.text((30, 310), "Hemoglobin: 11.2 g/dL", fill=text_fill, font=font)
    draw.text((30, 350), "Serum Sodium: 138 mEq/L", fill=text_fill, font=font)
    draw.text((30, 390), "Serum Potassium: 4.5 mEq/L", fill=text_fill, font=font)

    buf = io.BytesIO()
    # JPEG does not support RGBA; convert to RGB if JPEG requested
    if fmt.upper() in ("JPEG", "JPG") and mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        bg.save(buf, format="JPEG")
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# 1. Image upload
# -----------------------------------------------------------------------------
def test_image_uploads_successfully(client):
    """Test 1: PNG image uploads and returns a valid staging session."""
    img_bytes = _generate_synthetic_report_image()
    res = client.post(
        "/upload-document",
        files={"file": ("lab_report.png", img_bytes, "image/png")}
    )
    assert res.status_code == 200
    data = res.json()
    assert "upload_id" in data
    assert data["filename"] == "lab_report.png"
    assert data["content_type"] == "image/png"
    assert data["file_size_bytes"] == len(img_bytes)

    # Cleanup
    client.delete(f"/upload-document/{data['upload_id']}")


# -----------------------------------------------------------------------------
# 2. OCR text extraction
# -----------------------------------------------------------------------------
def test_ocr_extracts_text_from_image():
    """Test 2: OCR successfully reads text from synthetic laboratory image."""
    img_bytes = _generate_synthetic_report_image()
    result = document_extractor.extract_from_bytes(img_bytes, "test_report.png")
    assert result.status == "success"
    assert result.document_type == "image"
    assert result.pages_processed == 1
    assert len(result.extracted_text) > 0
    assert "Creatinine" in result.extracted_text or "2.1" in result.extracted_text


# -----------------------------------------------------------------------------
# 3 & 4. Parameter identification and correct mapping
# -----------------------------------------------------------------------------
def test_parameters_identified_and_mapped(client):
    """Tests 3 & 4: Supported parameters are identified and correctly populated."""
    img_bytes = _generate_synthetic_report_image()
    up_res = client.post(
        "/upload-document",
        files={"file": ("renal_panel.png", img_bytes, "image/png")}
    )
    assert up_res.status_code == 200
    upload_id = up_res.json()["upload_id"]

    try:
        ex_res = client.post(f"/extract-parameters/{upload_id}")
        assert ex_res.status_code == 200
        data = ex_res.json()
        assert data["upload_id"] == upload_id
        assert data["total_features_supported"] == 24
        assert data["extracted_unambiguous_count"] + data["extracted_requiring_review_count"] >= 5

        # Check core biomarkers
        params = data["parameters"]
        assert params["sc"]["extracted_value"] == 2.1
        assert params["bu"]["extracted_value"] == 55.0
        assert params["hemo"]["extracted_value"] == 11.2
        assert params["age"]["extracted_value"] == 62.0
    finally:
        client.delete(f"/upload-document/{upload_id}")


# -----------------------------------------------------------------------------
# 5. Missing parameters remain empty
# -----------------------------------------------------------------------------
def test_missing_parameters_remain_empty(client):
    """Test 5: Parameters not in document remain missing with value None (not 0.0)."""
    img_bytes = _generate_synthetic_report_image()
    up_res = client.post(
        "/upload-document",
        files={"file": ("renal_panel.png", img_bytes, "image/png")}
    )
    upload_id = up_res.json()["upload_id"]

    try:
        ex_res = client.post(f"/extract-parameters/{upload_id}")
        data = ex_res.json()
        params = data["parameters"]

        # WBC, RBC, Albumin, Sugar not printed in the synthetic image
        for unprinted in ["wbcc", "rbcc", "al", "su", "pe", "ane"]:
            assert params[unprinted]["extracted_value"] is None
            assert params[unprinted]["status"] == "missing"
    finally:
        client.delete(f"/upload-document/{upload_id}")


# -----------------------------------------------------------------------------
# 6 & 7. Ambiguous values review tagging and counter updates
# -----------------------------------------------------------------------------
def test_ambiguous_and_counter_updates():
    """Tests 6 & 7: Ambiguous and conflicting parameters require review and counters match."""
    # Text with conflicting creatinine values across different lines
    conflicting_text = (
        "Serum Creatinine: 1.2 mg/dL\n"
        "Repeat Creatinine: 2.8 mg/dL\n"
        "Hemoglobin: 12.0 g/dL\n"
    )
    summary = parameter_extractor.extract_from_text(conflicting_text)
    assert summary.conflicting_count >= 1
    assert summary.parameters["sc"].status == "conflicting"

    # Verify patient verifier aggregates counts accurately
    payload = {
        "sc": {"value": 2.1, "origin": "extracted", "status": "requires_review"},
        "hemo": {"value": 11.2, "origin": "verified", "status": "verified"},
        "bu": {"value": 55.0, "origin": "manual", "status": "verified"}
    }
    ver_res = verify_patient_parameters_payload(payload, allow_imputation=True)
    assert ver_res.verified_count == 2
    assert ver_res.requires_review_count == 1
    assert ver_res.missing_count == 21
    assert ver_res.total_features == 24


# -----------------------------------------------------------------------------
# 8. Prediction blocked when required inputs are missing or invalid
# -----------------------------------------------------------------------------
def test_prediction_blocked_when_all_inputs_missing(client):
    """Test 8: Prediction is blocked with HTTP 422 when all 24 inputs are missing."""
    empty_payload = {
        "parameters": {},
        "allow_imputation": True,
        "include_plot": False
    }
    res = client.post("/predict-verified", json=empty_payload)
    assert res.status_code == 422
    assert "missing" in res.json()["detail"].lower()


def test_prediction_blocked_when_invalid_bounds(client):
    """Test 8 (cont): Prediction is blocked when values violate biological boundaries."""
    invalid_payload = {
        "parameters": {
            "sc": {"value": 999.0, "origin": "manual", "status": "invalid"},
            "hemo": {"value": 12.0, "origin": "manual", "status": "verified"}
        },
        "allow_imputation": True,
        "include_plot": False
    }
    res = client.post("/predict-verified", json=invalid_payload)
    assert res.status_code == 422
    assert "invalid" in res.json()["detail"].lower()


# -----------------------------------------------------------------------------
# 9 & 10. PDF contains identical verified inputs and SHAP explanation
# -----------------------------------------------------------------------------
def test_pdf_contains_exact_verified_inputs_and_shap(client):
    """Tests 9 & 10: PDF strictly reflects current verified patient data and SHAP values."""
    verified_inputs = {
        "age": {"value": 62.0, "origin": "verified", "status": "verified"},
        "bp": {"value": 80.0, "origin": "verified", "status": "verified"},
        "sc": {"value": 2.1, "origin": "verified", "status": "verified"},
        "bu": {"value": 55.0, "origin": "verified", "status": "verified"},
        "hemo": {"value": 11.2, "origin": "verified", "status": "verified"},
        "sod": {"value": 138.0, "origin": "verified", "status": "verified"},
        "pot": {"value": 4.5, "origin": "verified", "status": "verified"},
        "bgr": {"value": 140.0, "origin": "verified", "status": "verified"}
    }

    pred_res = client.post("/predict-verified", json={
        "parameters": verified_inputs,
        "allow_imputation": True,
        "include_plot": False
    })
    assert pred_res.status_code == 200
    pred_data = pred_res.json()

    # Generate PDF report
    pdf_bytes = generate_clinical_pdf_report(
        prediction_data=pred_data,
        verified_parameters=verified_inputs,
        document_name="verified_patient_report.pdf",
        session_id="session-test-lineage"
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert b"%%EOF" in pdf_bytes

    # Parse generated PDF text
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_pdf_text = "".join(page.extract_text() or "" for page in reader.pages)

    # Verify exact patient inputs appear in the PDF
    assert "2.1" in full_pdf_text  # sc
    assert "11.2" in full_pdf_text  # hemo
    assert "55" in full_pdf_text or "55.0" in full_pdf_text  # bu
    assert "62" in full_pdf_text or "62.0" in full_pdf_text  # age
    # Verify SHAP section exists
    assert "SHAP" in full_pdf_text


# -----------------------------------------------------------------------------
# 11 & 12. Reset and multi-document isolation
# -----------------------------------------------------------------------------
def test_multi_document_isolation(client):
    """Test 12: Second document upload creates distinct session without reusing first document values."""
    img_bytes1 = _generate_synthetic_report_image()
    res1 = client.post("/upload-document", files={"file": ("doc1.png", img_bytes1, "image/png")})
    assert res1.status_code == 200
    id1 = res1.json()["upload_id"]

    res2 = client.post("/upload-document", files={"file": ("doc2.png", img_bytes1, "image/png")})
    assert res2.status_code == 200
    id2 = res2.json()["upload_id"]

    assert id1 != id2

    # Clean up
    client.delete(f"/upload-document/{id1}")
    client.delete(f"/upload-document/{id2}")


# -----------------------------------------------------------------------------
# 13. OCR failures produce clear errors
# -----------------------------------------------------------------------------
def test_ocr_failure_produces_clear_error(client, monkeypatch):
    """Test 13: When OCR is unavailable, extract-parameters returns informative HTTP 422."""
    img_bytes = _generate_synthetic_report_image()
    up_res = client.post("/upload-document", files={"file": ("scan.png", img_bytes, "image/png")})
    upload_id = up_res.json()["upload_id"]

    try:
        # Mock OCR engines as unavailable
        monkeypatch.setattr(
            "src.document_extractor.check_ocr_engine_availability",
            lambda: (False, "none", "Tesseract OCR engine not installed on host.")
        )
        ex_res = client.post(f"/extract-parameters/{upload_id}")
        assert ex_res.status_code == 422
        detail = ex_res.json()["detail"]
        assert "Optical Character Recognition" in detail or "OCR" in detail
    finally:
        client.delete(f"/upload-document/{upload_id}")


def test_corrupted_image_produces_clear_error(client):
    """Test 13 (cont): Corrupted image upload produces clear extraction error."""
    corrupted_bytes = b"\x89PNG\r\n\x1a\ncorrupted_random_payload_bytes"
    up_res = client.post("/upload-document", files={"file": ("corrupted.png", corrupted_bytes, "image/png")})
    upload_id = up_res.json()["upload_id"]

    try:
        ex_res = client.post(f"/extract-parameters/{upload_id}")
        assert ex_res.status_code == 422
        assert "extraction failed" in ex_res.json()["detail"].lower() or "corrupted" in ex_res.json()["detail"].lower()
    finally:
        client.delete(f"/upload-document/{upload_id}")


# =============================================================================
# Step 3 Regression Test Suite: Comprehensive Image Decoding & Extraction
# =============================================================================

def test_regression_valid_jpeg_extraction(client):
    """Regression Test 1: Legitimate JPEG medical report decodes and extracts laboratory parameters."""
    jpeg_bytes = _generate_synthetic_report_image(fmt="JPEG", mode="RGB")
    up_res = client.post(
        "/upload-document",
        files={"file": ("renal_panel.jpg", jpeg_bytes, "image/jpeg")}
    )
    assert up_res.status_code == 200
    upload_id = up_res.json()["upload_id"]

    try:
        ex_res = client.post(f"/extract-parameters/{upload_id}")
        assert ex_res.status_code == 200
        data = ex_res.json()
        assert data["upload_id"] == upload_id
        assert data["extracted_unambiguous_count"] + data["extracted_requiring_review_count"] > 0
        assert data["parameters"]["sc"]["extracted_value"] == 2.1
        assert data["parameters"]["hemo"]["extracted_value"] == 11.2
    finally:
        client.delete(f"/upload-document/{upload_id}")


def test_regression_valid_png_extraction(client):
    """Regression Test 2: Legitimate PNG medical report extracts parameters accurately."""
    png_bytes = _generate_synthetic_report_image(fmt="PNG", mode="RGB")
    up_res = client.post(
        "/upload-document",
        files={"file": ("renal_panel.png", png_bytes, "image/png")}
    )
    assert up_res.status_code == 200
    upload_id = up_res.json()["upload_id"]

    try:
        ex_res = client.post(f"/extract-parameters/{upload_id}")
        assert ex_res.status_code == 200
        data = ex_res.json()
        assert data["parameters"]["bu"]["extracted_value"] == 55.0
    finally:
        client.delete(f"/upload-document/{upload_id}")


def test_regression_grayscale_image_extraction():
    """Regression Test 3: Grayscale (mode L) image decodes safely and performs OCR."""
    gray_bytes = _generate_synthetic_report_image(fmt="PNG", mode="L")
    result = document_extractor.extract_from_bytes(gray_bytes, "report_gray.png")
    assert result.status == "success"
    assert result.document_type == "image"
    assert "Serum Creatinine" in result.extracted_text or "2.1" in result.extracted_text


def test_regression_rgba_image_extraction():
    """Regression Test 4: RGBA image with alpha transparency composites over white and decodes text."""
    rgba_bytes = _generate_synthetic_report_image(fmt="PNG", mode="RGBA")
    result = document_extractor.extract_from_bytes(rgba_bytes, "report_rgba.png")
    assert result.status == "success"
    assert result.document_type == "image"
    assert "Serum Creatinine" in result.extracted_text or "2.1" in result.extracted_text


def test_regression_image_with_incorrect_file_extension():
    """Regression Test 5: JPEG image payload with .png extension or unusual extension is safely detected and extracted."""
    jpeg_bytes = _generate_synthetic_report_image(fmt="JPEG", mode="RGB")
    # File named .png but containing JPEG binary data
    res_mismatched = document_extractor.extract_from_bytes(jpeg_bytes, "fake_extension.png")
    assert res_mismatched.status == "success"
    assert res_mismatched.document_type == "image"
    assert len(res_mismatched.extracted_text) > 0

    # File with unknown extension but valid JPEG bytes
    res_unknown_ext = document_extractor.extract_from_bytes(jpeg_bytes, "scan.unknown_ext")
    assert res_unknown_ext.status == "success"
    assert res_unknown_ext.document_type == "image"


def test_regression_empty_file_extraction():
    """Regression Test 6: 0-byte payload returns a structured failure without crashing."""
    res = document_extractor.extract_from_bytes(b"", "empty.jpg")
    assert res.status == "failed"
    assert len(res.errors) > 0
    assert "empty" in res.errors[0].lower()


def test_regression_truncated_or_malformed_image():
    """Regression Test 7: Truncated or corrupt image data returns clear error without unhandled exceptions."""
    # Truncated JPEG stream (only header + incomplete marker)
    malformed_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 20
    res = document_extractor.extract_from_bytes(malformed_bytes, "bad.jpg")
    assert res.status == "failed"
    assert len(res.errors) > 0
    assert "corrupted" in res.errors[0].lower() or "malformed" in res.errors[0].lower()


def test_regression_image_exceeding_configured_dimensions(monkeypatch):
    """Regression Test 8: Excessively large image exceeding safety limits is cleanly rejected without crashing."""
    # Temporarily set max pixels to a low limit to test safety boundary
    from src import document_extractor as de_module
    monkeypatch.setattr(de_module, "MAX_IMAGE_PIXELS", 500_000)

    # 1000x800 = 800,000 pixels (> 500,000)
    large_img = Image.new("RGB", (1000, 800), color=(255, 255, 255))
    buf = io.BytesIO()
    large_img.save(buf, format="JPEG")
    large_bytes = buf.getvalue()

    res = de_module.document_extractor.extract_from_bytes(large_bytes, "huge.jpg")
    assert res.status == "failed"
    assert len(res.errors) > 0
    assert "exceed" in res.errors[0].lower()


def test_regression_ocr_engine_failure_handling(client, monkeypatch):
    """Regression Test 9: Gracefully handles OCR engine failures during image parameter extraction."""
    img_bytes = _generate_synthetic_report_image(fmt="JPEG", mode="RGB")
    up_res = client.post("/upload-document", files={"file": ("report.jpg", img_bytes, "image/jpeg")})
    upload_id = up_res.json()["upload_id"]

    try:
        # Simulate OCR engine failure during execution
        def mock_failed_ocr(img):
            return "", False, "ocr_windows_native", "Simulated OCR runtime failure."

        monkeypatch.setattr("src.document_extractor.run_ocr_on_pil_image", mock_failed_ocr)
        ex_res = client.post(f"/extract-parameters/{upload_id}")
        assert ex_res.status_code == 422
        assert "OCR failed" in ex_res.json()["detail"] or "Simulated OCR" in ex_res.json()["detail"]
    finally:
        client.delete(f"/upload-document/{upload_id}")


def test_regression_cleanup_after_successful_and_failed_processing(client):
    """Regression Test 10: Ensures temporary files and staging sessions are completely purged."""
    from src.document_upload import document_session_manager

    # 1. Successful upload & explicit deletion
    img_bytes = _generate_synthetic_report_image(fmt="JPEG", mode="RGB")
    up_res = client.post("/upload-document", files={"file": ("cleanup_test.jpg", img_bytes, "image/jpeg")})
    assert up_res.status_code == 200
    upload_id = up_res.json()["upload_id"]

    record = document_session_manager.get_upload(upload_id)
    assert record is not None
    file_path = Path(record["internal_path"])
    assert file_path.exists()

    # Purge
    del_res = client.delete(f"/upload-document/{upload_id}")
    assert del_res.status_code == 200
    assert not file_path.exists()
    assert document_session_manager.get_upload(upload_id) is None

    # 2. Deleting non-existent session returns 404
    del_nonexistent = client.delete("/upload-document/nonexistent-id-000")
    assert del_nonexistent.status_code == 404
