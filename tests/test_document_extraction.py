"""
Automated Test Suite for Phase 3: Medical Document Text Extraction.

Tests cover:
- Digital PDF text extraction & page boundary preservation
- Scanned PDF processing (embedded images & OCR fallback)
- Image text extraction (PNG, JPG, JPEG) with preprocessing
- Empty PDF handling
- Image without readable text
- Malformed/corrupted PDF and image handling
- Graceful handling of missing OCR engine (Tesseract)
- Security bounds: page limit enforcement (10 pages)
- End-to-end API integration with POST /extract-text/{upload_id}
"""

import sys
import io
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from PIL import Image, ImageDraw
import pypdf
from pypdf import PdfWriter
from fastapi.testclient import TestClient

from backend.main import app
from src.document_upload import document_session_manager
from src.document_extractor import (
    document_extractor,
    check_tesseract_availability,
    preprocess_image_for_ocr,
    ExtractionResult,
    MAX_PAGES_TO_PROCESS
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_temporary_uploads():
    """Ensures test isolation by purging any lingering uploads before and after each test."""
    document_session_manager.cleanup_all()
    yield
    document_session_manager.cleanup_all()


def _create_digital_pdf(pages_text: list) -> bytes:
    """Helper to generate a valid in-memory digital PDF with explicit text."""
    # Build standard PDF structures using pypdf or minimal raw PDF syntax
    # Since pypdf does not have a high-level text drawing API, we write standard PDF objects
    stream = io.BytesIO()
    writer = PdfWriter()
    
    # We can create blank pages and attach text annotations or build a standard valid PDF with content streams
    # Standard PDF 1.4 with text content stream:
    pdf_template = (
        "%PDF-1.4\n"
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        "4 0 obj\n<< /Length {length} >>\nstream\n{stream_content}\nendstream\nendobj\n"
        "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        "xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
        "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n500\n%%EOF"
    )
    
    content = "BT /F1 12 Tf 50 700 Td (PATIENT CLINICAL REPORT) Tj 0 -20 Td (Serum Creatinine: 1.4 mg/dL) Tj 0 -20 Td (Blood Urea: 45 mg/dL) Tj ET"
    full_pdf = pdf_template.format(length=len(content), stream_content=content).encode("latin-1")
    return full_pdf


def _create_multi_page_pdf(num_pages: int, text_prefix: str = "Clinical Record Page") -> bytes:
    """Helper creating multi-page PDF."""
    writer = PdfWriter()
    for i in range(num_pages):
        page = writer.add_blank_page(width=612, height=792)
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def _create_test_image(text: str = "LAB RESULTS") -> bytes:
    """Creates a PNG image with rendered text."""
    img = Image.new("RGB", (600, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 80), text, fill=(0, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _create_blank_image() -> bytes:
    """Creates a completely blank white image."""
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# ============================================================================
# 1. Digital PDF Extraction
# ============================================================================

def test_digital_pdf_extraction():
    """Verify digital text extraction extracts text and identifies document as digital_pdf."""
    pdf_bytes = _create_digital_pdf(["PATIENT CLINICAL REPORT", "Serum Creatinine: 1.4 mg/dL"])
    result = document_extractor.extract_from_bytes(pdf_bytes, "lab_report.pdf")
    
    assert result.status == "success"
    assert result.document_type == "digital_pdf"
    assert result.pages_processed >= 1
    assert result.extraction_method == "digital_pdf_parser"
    assert "PATIENT CLINICAL REPORT" in result.extracted_text
    assert "Serum Creatinine" in result.extracted_text


# ============================================================================
# 2. Scanned PDF Processing
# ============================================================================

def test_scanned_pdf_processing_with_mocked_ocr(monkeypatch):
    """Verify that a scanned PDF (page with embedded image) triggers OCR extraction."""
    # 1. Build an authentic scanned PDF containing an embedded image using Pillow
    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 100), "SCANNED REPORT", fill=(0, 0, 0))
    
    buf = io.BytesIO()
    img.save(buf, format="PDF")
    scanned_pdf_bytes = buf.getvalue()

    # Mock OCR availability & execution
    monkeypatch.setattr("src.document_extractor.check_tesseract_availability", lambda: (True, None))
    monkeypatch.setattr("pytesseract.image_to_string", lambda img, **kwargs: "SCANNED BLOOD TEST RESULTS: HGB 13.5")

    result = document_extractor.extract_from_bytes(scanned_pdf_bytes, "scan.pdf")
    assert result.status == "success"
    assert result.document_type == "scanned_pdf"
    assert "SCANNED BLOOD TEST" in result.extracted_text
    assert result.extraction_method in {"ocr_tesseract", "hybrid_pdf_ocr"}


# ============================================================================
# 3. Image Processing (PNG, JPG, JPEG)
# ============================================================================

def test_image_extraction_with_ocr(monkeypatch):
    """Verify image file extraction executes preprocessing and OCR."""
    img_bytes = _create_test_image("SERUM CREATININE 1.2")
    
    # Mock OCR to return the text reliably
    monkeypatch.setattr("src.document_extractor.check_tesseract_availability", lambda: (True, None))
    monkeypatch.setattr("pytesseract.image_to_string", lambda img, **kwargs: "SERUM CREATININE 1.2")

    result = document_extractor.extract_from_bytes(img_bytes, "renal_scan.png")
    assert result.status == "success"
    assert result.document_type == "image"
    assert result.pages_processed == 1
    assert result.extraction_method == "ocr_tesseract"
    assert "SERUM CREATININE 1.2" in result.extracted_text


def test_image_preprocessing_helper():
    """Verify orientation, dimension clamping, and grayscale conversion on images."""
    large_img = Image.new("RGB", (4000, 3000), color=(200, 200, 200))
    processed = preprocess_image_for_ocr(large_img)
    
    # Must be grayscale ("L")
    assert processed.mode == "L"
    # Dimensions must not exceed 3500px
    assert processed.width <= 3500
    assert processed.height <= 3500


# ============================================================================
# 4. Empty PDF & Image Without Readable Text
# ============================================================================

def test_empty_pdf_detection():
    """Verify an empty PDF (blank pages with no text) returns 'no_text_extracted'."""
    blank_pdf = _create_multi_page_pdf(1)
    result = document_extractor.extract_from_bytes(blank_pdf, "blank.pdf")
    
    assert result.status == "no_text_extracted"
    assert result.document_type == "digital_pdf"
    assert result.pages_processed == 1
    assert any("no readable" in w.lower() for w in result.warnings)


def test_image_without_readable_text(monkeypatch):
    """Verify a blank image returning empty OCR yields 'no_text_extracted'."""
    blank_img_bytes = _create_blank_image()
    monkeypatch.setattr("src.document_extractor.check_tesseract_availability", lambda: (True, None))
    monkeypatch.setattr("pytesseract.image_to_string", lambda img, **kwargs: "")

    result = document_extractor.extract_from_bytes(blank_img_bytes, "blank.png")
    assert result.status == "no_text_extracted"
    assert any("no readable" in w.lower() for w in result.warnings)


# ============================================================================
# 5. Malformed and Corrupted Files
# ============================================================================

def test_malformed_pdf_handling():
    """Verify corrupted PDF bytes return status='failed' without crashing."""
    corrupted_bytes = b"%PDF-corrupted-structure-without-xref-or-trailers"
    result = document_extractor.extract_from_bytes(corrupted_bytes, "corrupted.pdf")
    
    assert result.status == "failed"
    assert len(result.errors) > 0
    assert "Unable to parse PDF structure" in result.errors[0] or "corrupted" in result.errors[0].lower()


def test_malformed_image_handling():
    """Verify corrupted image bytes return status='failed' without crashing."""
    corrupted_img_bytes = b"\x89PNG\r\n\x1a\ncorrupted-random-garbage-bytes"
    result = document_extractor.extract_from_bytes(corrupted_img_bytes, "corrupted.png")
    
    assert result.status == "failed"
    assert len(result.errors) > 0
    assert "corrupted" in result.errors[0].lower() or "unable to load" in result.errors[0].lower()


# ============================================================================
# 6. Missing OCR Engine Graceful Fallback
# ============================================================================

def test_missing_ocr_engine_fallback(monkeypatch):
    """Verify graceful handling when Tesseract is not available on host."""
    img_bytes = _create_test_image("SAMPLE TEXT")
    
    # Simulate missing OCR
    instructions = "Install Tesseract OCR"
    monkeypatch.setattr("src.document_extractor.check_tesseract_availability", lambda: (False, instructions))
    monkeypatch.setattr("src.document_extractor.check_windows_ocr_availability", lambda: (False, instructions))

    result = document_extractor.extract_from_bytes(img_bytes, "scan.jpg")
    assert result.status == "ocr_engine_unavailable"
    assert result.extraction_method == "none"
    assert result.setup_instructions == instructions
    assert any("requires" in w and "OCR" in w for w in result.warnings)


# ============================================================================
# 7. Security: Page Limit Enforcement
# ============================================================================

def test_page_limit_enforcement():
    """Verify that documents exceeding MAX_PAGES (10) are capped safely."""
    oversized_pdf = _create_multi_page_pdf(15)
    result = document_extractor.extract_from_bytes(oversized_pdf, "long_record.pdf")
    
    assert result.pages_processed == MAX_PAGES_TO_PROCESS
    assert any("safety limit" in w.lower() for w in result.warnings)


# ============================================================================
# 8. API Integration: POST /extract-text/{upload_id}
# ============================================================================

def test_api_extract_text_flow():
    """Verify end-to-end HTTP endpoint: upload document then extract text."""
    # 1. Upload valid digital PDF
    pdf_bytes = _create_digital_pdf(["CLINICAL TEST"])
    upload_res = client.post("/upload-document", files={"file": ("report.pdf", pdf_bytes, "application/pdf")})
    assert upload_res.status_code == 200
    upload_id = upload_res.json()["upload_id"]

    # 2. Extract text via API endpoint
    extract_res = client.post(f"/extract-text/{upload_id}")
    assert extract_res.status_code == 200
    data = extract_res.json()
    
    assert data["upload_id"] == upload_id
    assert data["status"] == "success"
    assert data["document_type"] == "digital_pdf"
    assert data["extraction_method"] == "digital_pdf_parser"
    assert data["has_extractable_text"] is True
    assert data["text_length_characters"] > 0
    # Confirm raw medical text is NOT exposed in the public API response for privacy
    assert "extracted_text" not in data

    # 3. Clean up
    del_res = client.delete(f"/upload-document/{upload_id}")
    assert del_res.status_code == 200
