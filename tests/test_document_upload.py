"""
Automated Test Suite for Phase 2: Secure Medical Document Upload & Validation.

Tests cover:
- Valid PDF document upload
- Valid PNG and JPEG image upload
- Unsupported file extension rejection
- Oversized file rejection (> 10 MB limit)
- Empty file rejection (0 bytes)
- Malformed/spoofed file rejection (magic byte signature mismatch)
- Path traversal defense & filename sanitization
- Safe temporary-file deletion and lifecycle cleanup
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.document_upload import (
    document_session_manager,
    MAX_UPLOAD_SIZE_BYTES,
    sanitize_filename,
    verify_magic_bytes,
    DocumentValidationError,
    FileTooLargeError,
    EmptyFileError,
    UnsupportedExtensionError,
    MalformedFileError
)

client = TestClient(app)

# Minimal authentic binary samples
VALID_PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\ntrailer\n<<\n>>\n%%EOF"
VALID_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
VALID_JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\xff\xd9"


@pytest.fixture(autouse=True)
def cleanup_temporary_uploads():
    """Ensures test isolation by purging any lingering uploads before and after each test."""
    document_session_manager.cleanup_all()
    yield
    document_session_manager.cleanup_all()


# ============================================================================
# 1. Valid Document Uploads
# ============================================================================

def test_valid_pdf_upload():
    """Verify an authentic PDF file is accepted, validated, and staged in temporary storage."""
    files = {"file": ("patient_lab_report.pdf", VALID_PDF_BYTES, "application/pdf")}
    response = client.post("/upload-document", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "upload_id" in data
    assert data["filename"] == "patient_lab_report.pdf"
    assert data["content_type"] == "application/pdf"
    assert data["file_size_bytes"] == len(VALID_PDF_BYTES)
    assert data["status"] == "uploaded_and_verified"
    
    # Verify temporary file exists on disk
    record = document_session_manager.get_upload(data["upload_id"])
    assert record is not None
    assert Path(record["internal_path"]).exists()


def test_valid_png_upload():
    """Verify an authentic PNG lab scan is accepted and staged with correct MIME type."""
    files = {"file": ("biomarker_scan.png", VALID_PNG_BYTES, "image/png")}
    response = client.post("/upload-document", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["content_type"] == "image/png"
    assert data["filename"] == "biomarker_scan.png"
    assert data["status"] == "uploaded_and_verified"


def test_valid_jpeg_upload():
    """Verify JPEG images (.jpg and .jpeg) are accepted with image/jpeg MIME type."""
    for filename in ["ultrasound.jpg", "renal_panel.jpeg"]:
        files = {"file": (filename, VALID_JPEG_BYTES, "image/jpeg")}
        response = client.post("/upload-document", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["content_type"] == "image/jpeg"
        assert data["filename"] == filename


# ============================================================================
# 2. Unsupported Extensions & Format Rejections
# ============================================================================

@pytest.mark.parametrize("unsupported_filename", [
    "malicious_script.exe",
    "notes.txt",
    "dataset.csv",
    "report.docx",
    "archive.zip",
    "payload.py",
    "no_extension_file"
])
def test_unsupported_file_extension(unsupported_filename):
    """Verify rejection of files with extensions outside the whitelist."""
    files = {"file": (unsupported_filename, b"some arbitrary text content", "application/octet-stream")}
    response = client.post("/upload-document", files=files)
    
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Unsupported file extension" in detail or "Only PDF" in detail


# ============================================================================
# 3. Oversized and Empty File Rejections
# ============================================================================

def test_oversized_file():
    """Verify that files exceeding the maximum size limit (10 MB) are rejected with HTTP 413."""
    oversized_bytes = b"%PDF-" + b"0" * (MAX_UPLOAD_SIZE_BYTES + 1024)
    files = {"file": ("huge_scan.pdf", oversized_bytes, "application/pdf")}
    response = client.post("/upload-document", files=files)
    
    assert response.status_code == 413
    assert "exceeds maximum permitted limit" in response.json()["detail"]


def test_empty_file():
    """Verify that 0-byte empty files are rejected with HTTP 400."""
    files = {"file": ("empty_report.pdf", b"", "application/pdf")}
    response = client.post("/upload-document", files=files)
    
    assert response.status_code == 400
    assert "empty (0 bytes)" in response.json()["detail"]


# ============================================================================
# 4. Malformed and Spoofed File Rejections (Magic Byte Checks)
# ============================================================================

def test_malformed_pdf_magic_bytes_mismatch():
    """Verify rejection when a file named .pdf contains plaintext rather than %PDF- magic bytes."""
    fake_pdf = b"This is just a text file masquerading as a PDF document."
    files = {"file": ("spoofed.pdf", fake_pdf, "application/pdf")}
    response = client.post("/upload-document", files=files)
    
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Malformed or spoofed file" in detail
    assert "PDF binary signature" in detail


def test_malformed_png_magic_bytes_mismatch():
    """Verify rejection when an image named .png lacks PNG header bytes."""
    fake_png = b"NOT_A_PNG_HEADER_12345678"
    files = {"file": ("fake_image.png", fake_png, "image/png")}
    response = client.post("/upload-document", files=files)
    
    assert response.status_code == 400
    assert "PNG binary signature" in response.json()["detail"]


def test_mismatched_extension_and_content():
    """Verify rejection when JPEG content is uploaded with a .pdf extension."""
    files = {"file": ("mismatched.pdf", VALID_JPEG_BYTES, "application/pdf")}
    response = client.post("/upload-document", files=files)
    
    assert response.status_code == 400
    assert "Malformed or spoofed file" in response.json()["detail"]


# ============================================================================
# 5. Path Traversal & Filename Sanitization
# ============================================================================

def test_path_traversal_sanitization():
    """Verify filenames with directory traversal attempts are sanitized safely."""
    malicious_name = "../../../../etc/passwd.pdf"
    files = {"file": (malicious_name, VALID_PDF_BYTES, "application/pdf")}
    response = client.post("/upload-document", files=files)
    
    assert response.status_code == 200
    data = response.json()
    # Check that directory traversal components were stripped
    assert "/" not in data["filename"]
    assert ".." not in data["filename"]
    assert data["filename"] == "passwd.pdf"
    
    # Verify the internal storage path does not escape the controlled temporary directory
    record = document_session_manager.get_upload(data["upload_id"])
    internal_path = Path(record["internal_path"])
    assert internal_path.parent == document_session_manager.storage_dir
    assert internal_path.exists()


def test_filename_sanitization_helper():
    """Verify unit-level sanitization rules."""
    assert sanitize_filename("normal_report.pdf") == "normal_report.pdf"
    assert sanitize_filename("../../../secret.pdf") == "secret.pdf"
    assert sanitize_filename("..\\..\\windows\\win.ini.pdf") == "win.ini.pdf"
    assert sanitize_filename("report\x00with_null.pdf") == "reportwith_null.pdf"
    assert sanitize_filename("report with spaces & special #$.pdf") == "report_with_spaces___special___.pdf"


# ============================================================================
# 6. Safe Temporary-File Deletion & Lifecycle Cleanup
# ============================================================================

def test_safe_temporary_file_cleanup_on_delete():
    """Verify client can cancel/remove an upload and the temporary file is purged immediately."""
    # 1. Upload valid document
    files = {"file": ("test_report.pdf", VALID_PDF_BYTES, "application/pdf")}
    upload_res = client.post("/upload-document", files=files)
    assert upload_res.status_code == 200
    upload_id = upload_res.json()["upload_id"]
    
    record = document_session_manager.get_upload(upload_id)
    temp_file = Path(record["internal_path"])
    assert temp_file.exists()
    
    # 2. Issue DELETE request to purge
    del_res = client.delete(f"/upload-document/{upload_id}")
    assert del_res.status_code == 200
    del_data = del_res.json()
    assert del_data["status"] == "removed"
    assert del_data["upload_id"] == upload_id
    
    # 3. Assert temporary file is purged from disk
    assert not temp_file.exists()
    assert document_session_manager.get_upload(upload_id) is None


def test_delete_nonexistent_upload():
    """Verify DELETE returns 404 for unknown or already purged upload IDs."""
    response = client.delete("/upload-document/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert "not found or already purged" in response.json()["detail"]


def test_session_manager_cleanup_all():
    """Verify document_session_manager.cleanup_all() purges all active temporary files."""
    # Stage multiple temporary documents
    u1 = document_session_manager.save_upload(VALID_PDF_BYTES, "doc1.pdf")
    u2 = document_session_manager.save_upload(VALID_PNG_BYTES, "doc2.png")
    
    p1 = Path(u1["internal_path"])
    p2 = Path(u2["internal_path"])
    assert p1.exists() and p2.exists()
    
    # Purge all
    purged_count = document_session_manager.cleanup_all()
    assert purged_count == 2
    assert not p1.exists()
    assert not p2.exists()
