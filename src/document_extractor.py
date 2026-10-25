"""
Secure Medical Document Text Extraction Module for CKD Explainable AI.

Provides:
- Digital PDF extraction with page preservation and character decoding
- Scanned PDF processing via embedded image inspection and local OCR
- Image processing (PNG, JPG, JPEG) with EXIF orientation correction & contrast enhancement
- Graceful detection and handling of missing local OCR engines (Tesseract)
- Strict security limits: page count cap (10 pages), image resolution constraints
- Structured extraction metadata (status, document_type, pages_processed, method, warnings)
- Absolute privacy: zero external network calls, zero logging of extracted medical text,
  immediate ephemeral cleanup
"""

import os
import io
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from PIL import Image, ImageOps, ImageEnhance, ImageFile
import pypdf
from pypdf import PdfReader
from pypdf.errors import PdfReadError
import pytesseract

logger = logging.getLogger("ckd.document_extractor")

# Security Constraints
MAX_PAGES_TO_PROCESS = 10
MAX_IMAGE_DIMENSION = 3500
MAX_IMAGE_PIXELS = 200_000_000  # 200 Megapixels safety limit (supports high-res scans/camera exports)
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# Enable loading of truncated images to tolerate minor trailing byte variations
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Known Tesseract installation paths on Windows / Linux
COMMON_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\AppData\Local\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
]


@dataclass
class ExtractionResult:
    """Structured container for document text extraction output."""
    status: str  # "success", "no_text_extracted", "ocr_engine_unavailable", "failed"
    document_type: str  # "digital_pdf", "scanned_pdf", "image", "unknown"
    pages_processed: int
    extracted_text: str  # Kept in-memory for downstream parsing only
    extraction_method: str  # "digital_pdf_parser", "ocr_tesseract", "hybrid_pdf_ocr", "none"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    setup_instructions: Optional[str] = None

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Returns safe metadata dictionary excluding sensitive medical text for logging/API."""
        return {
            "status": self.status,
            "document_type": self.document_type,
            "pages_processed": self.pages_processed,
            "extraction_method": self.extraction_method,
            "text_length_characters": len(self.extracted_text),
            "warnings": self.warnings,
            "errors": self.errors,
            "setup_instructions": self.setup_instructions
        }


import sys
import subprocess
import tempfile

WIN_OCR_HELPER_PATH = Path(__file__).resolve().parent / "win_ocr_helper.ps1"


def check_tesseract_availability() -> Tuple[bool, Optional[str]]:
    """
    Checks if the local Tesseract OCR engine is installed and operational.
    Auto-discovers Tesseract from system PATH and standard Windows installation paths.
    Returns: (is_available, setup_instructions_if_missing)
    """
    # 1. Check if user configured explicit TESSERACT_CMD env variable
    custom_cmd = os.environ.get("TESSERACT_CMD")
    if custom_cmd and Path(custom_cmd).is_file():
        pytesseract.pytesseract.tesseract_cmd = custom_cmd
        return True, None

    # 2. Check if pytesseract already finds it in PATH
    try:
        pytesseract.get_tesseract_version()
        return True, None
    except Exception:
        pass

    # 3. Check system PATH via shutil.which
    which_path = shutil.which("tesseract")
    if which_path:
        pytesseract.pytesseract.tesseract_cmd = which_path
        return True, None

    # 4. Check known common installation paths
    for p in COMMON_TESSERACT_PATHS:
        candidate = Path(p)
        if candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            try:
                pytesseract.get_tesseract_version()
                return True, None
            except Exception:
                continue

    # Not found
    instructions = (
        "Tesseract OCR engine is not installed or not in system PATH. "
        "To enable optical text recognition for scanned documents and images:\n"
        "- Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki "
        "and install to 'C:\\Program Files\\Tesseract-OCR'.\n"
        "- Linux: Run 'sudo apt-get install tesseract-ocr'.\n"
        "- macOS: Run 'brew install tesseract'.\n"
        "Digital text PDFs do not require Tesseract and can be parsed directly."
    )
    return False, instructions


def check_windows_ocr_availability() -> Tuple[bool, Optional[str]]:
    """
    Checks whether the native Windows Media OCR runtime is accessible via PowerShell.
    """
    if sys.platform != "win32":
        return False, "Windows native OCR is only available on Windows platforms."
    if not WIN_OCR_HELPER_PATH.is_file():
        return False, f"Windows OCR helper script not found at {WIN_OCR_HELPER_PATH}"
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "[void][Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType = WindowsRuntime]; Write-Output 'OK'"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if res.returncode == 0 and "OK" in res.stdout:
            return True, None
    except Exception as e:
        return False, f"Windows Media OCR probe failed: {e}"
    return False, "Windows Media OCR engine unavailable."


def check_ocr_engine_availability() -> Tuple[bool, str, Optional[str]]:
    """
    Checks for available local OCR engines (Tesseract or Windows Media OCR).
    Returns (available, engine_name, instructions_if_unavailable).
    """
    tess_ok, tess_instr = check_tesseract_availability()
    if tess_ok:
        return True, "ocr_tesseract", None

    win_ok, win_instr = check_windows_ocr_availability()
    if win_ok:
        return True, "ocr_windows_native", None

    instructions = tess_instr or win_instr or (
        "Optical Character Recognition (OCR) engine is not available locally. "
        "To enable text extraction for scanned documents and images:\n"
        "- Windows: Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki "
        "or enable Windows Media OCR.\n"
        "- Linux: Run 'sudo apt-get install tesseract-ocr'.\n"
        "- macOS: Run 'brew install tesseract'."
    )
    return False, "none", instructions


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Applies image preprocessing to improve OCR accuracy on medical scans:
    - Normalizes EXIF orientation
    - Safely composites RGBA/LA transparency over white background
    - Converts CMYK and 1-bit formats to standard color spaces
    - Clamps dimensions to avoid excessive processing/memory consumption
    - Converts to grayscale
    - Enhances contrast
    """
    # 1. Orientation correction from EXIF metadata
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # 2. Alpha channel and color mode normalization
    try:
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(bg, rgba).convert("RGB")
        elif image.mode == "CMYK":
            image = image.convert("RGB")
        elif image.mode == "1":
            image = image.convert("L")
    except Exception as e:
        logger.warning(f"Color mode normalization warning: {e}")

    # 3. Dimension constraints (downscale high-res images to safe bounds for OCR)
    w, h = image.size
    if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
        image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)

    # 4. Grayscale conversion
    if image.mode != "L":
        image = image.convert("L")

    # 5. Contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.8)

    return image


def run_windows_native_ocr(image: Image.Image) -> Tuple[str, bool, Optional[str]]:
    """
    Executes local Windows Media OCR via helper script on a PIL image object.
    """
    if not WIN_OCR_HELPER_PATH.is_file():
        return "", False, "Windows OCR helper script missing."

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_name = tmp.name

    try:
        image.save(tmp_name, format="PNG")
        cmd = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", str(WIN_OCR_HELPER_PATH),
            "-ImagePath", tmp_name
        ]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30
        )
        if res.returncode == 0:
            return res.stdout.strip(), True, None
        else:
            err = res.stderr.strip() or "Windows OCR returned non-zero exit code."
            return "", False, err
    except Exception as e:
        return "", False, f"Windows OCR execution error: {str(e)}"
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass


def run_ocr_on_pil_image(image: Image.Image) -> Tuple[str, bool, str, Optional[str]]:
    """
    Executes local OCR on a PIL image object using the best available engine.
    Returns: (extracted_text, success_bool, method_used, warning_or_error)
    """
    available, engine, setup_msg = check_ocr_engine_availability()
    if not available:
        return "", False, "none", setup_msg

    processed_img = preprocess_image_for_ocr(image)

    if engine == "ocr_tesseract":
        try:
            text = pytesseract.image_to_string(processed_img)
            return text.strip(), True, "ocr_tesseract", None
        except Exception as e:
            logger.warning(f"Tesseract execution error: {e}. Attempting fallback if available.")
            win_ok, _ = check_windows_ocr_availability()
            if win_ok:
                w_text, w_ok, w_err = run_windows_native_ocr(processed_img)
                if w_ok:
                    return w_text, True, "ocr_windows_native", None
            return "", False, "ocr_tesseract", f"Tesseract error: {str(e)}"
    elif engine == "ocr_windows_native":
        text, success, err = run_windows_native_ocr(processed_img)
        if success:
            return text, True, "ocr_windows_native", None
        else:
            return "", False, "ocr_windows_native", err

    return "", False, "none", "No supported OCR engine found."


class MedicalDocumentExtractor:
    """
    Comprehensive document text extractor for clinical lab reports and records.
    Supports digital PDFs, scanned PDFs with fallback OCR, and medical scan images.
    """

    def __init__(self, max_pages: int = MAX_PAGES_TO_PROCESS):
        self.max_pages = max_pages

    def extract_from_bytes(self, file_bytes: bytes, filename: str) -> ExtractionResult:
        """
        Extracts text from raw bytes based on file format.
        """
        if not file_bytes:
            return ExtractionResult(
                status="failed",
                document_type="unknown",
                pages_processed=0,
                extracted_text="",
                extraction_method="none",
                errors=["The supplied file payload is empty (0 bytes)."]
            )

        ext = Path(filename).suffix.lower()

        # Check by declared extension or content magic bytes
        if ext == ".pdf" or file_bytes.startswith(b"%PDF-"):
            return self._extract_from_pdf_bytes(file_bytes)
        elif ext in {".png", ".jpg", ".jpeg"} or file_bytes.startswith(b"\x89PNG\r\n\x1a\n") or file_bytes.startswith(b"\xff\xd8\xff"):
            return self._extract_from_image_bytes(file_bytes)
        else:
            # Check if bytes represent a valid readable image format (e.g. incorrect or unusual extension)
            try:
                with Image.open(io.BytesIO(file_bytes)) as probe:
                    if probe.format in ("PNG", "JPEG"):
                        return self._extract_from_image_bytes(file_bytes)
            except Exception:
                pass

            return ExtractionResult(
                status="failed",
                document_type="unknown",
                pages_processed=0,
                extracted_text="",
                extraction_method="none",
                errors=[f"Unsupported document format '{ext}' for text extraction."]
            )

    def extract_from_path(self, file_path: Path) -> ExtractionResult:
        """
        Extracts text from a file located at a filesystem path.
        """
        if not file_path.is_file():
            return ExtractionResult(
                status="failed",
                document_type="unknown",
                pages_processed=0,
                extracted_text="",
                extraction_method="none",
                errors=[f"File not found at path: {file_path}"]
            )

        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            return self.extract_from_bytes(file_bytes, file_path.name)
        except Exception as e:
            return ExtractionResult(
                status="failed",
                document_type="unknown",
                pages_processed=0,
                extracted_text="",
                extraction_method="none",
                errors=[f"Failed to read file from disk: {str(e)}"]
            )

    def _extract_from_pdf_bytes(self, pdf_bytes: bytes) -> ExtractionResult:
        """
        Parses PDF document:
        1. Attempts native digital text extraction per page.
        2. Detects pages with no text but embedded images (scanned PDF).
        3. Applies OCR if needed and available.
        4. Preserves page boundaries.
        """
        warnings = []
        errors = []

        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
        except PdfReadError as e:
            logger.warning(f"Malformed PDF encountered: {e}")
            return ExtractionResult(
                status="failed",
                document_type="digital_pdf",
                pages_processed=0,
                extracted_text="",
                extraction_method="none",
                errors=["Malformed or corrupted PDF document. Unable to parse PDF structure."]
            )
        except Exception as e:
            return ExtractionResult(
                status="failed",
                document_type="digital_pdf",
                pages_processed=0,
                extracted_text="",
                extraction_method="none",
                errors=[f"Unexpected error opening PDF: {str(e)}"]
            )

        total_pages = len(reader.pages)
        if total_pages == 0:
            return ExtractionResult(
                status="no_text_extracted",
                document_type="digital_pdf",
                pages_processed=0,
                extracted_text="",
                extraction_method="digital_pdf_parser",
                warnings=["The PDF document contains 0 pages."]
            )

        pages_to_process = min(total_pages, self.max_pages)
        if total_pages > self.max_pages:
            warnings.append(
                f"Document exceeds the {self.max_pages}-page safety limit ({total_pages} total pages). "
                f"Only the first {self.max_pages} pages were processed."
            )

        page_texts: List[str] = []
        scanned_pages_detected = 0
        ocr_performed_pages = 0
        ocr_unavailable_warned = False
        setup_instructions = None

        for page_idx in range(pages_to_process):
            page = reader.pages[page_idx]
            page_num = page_idx + 1

            # 1. Attempt digital text extraction
            digital_text = ""
            try:
                extracted = page.extract_text()
                if extracted:
                    digital_text = extracted.strip()
            except Exception as e:
                warnings.append(f"Page {page_num}: Digital text extraction encountered warning: {e}")

            if digital_text:
                page_texts.append(f"--- Page {page_num} ---\n{digital_text}")
                continue

            # 2. No digital text found: inspect for embedded images (scanned PDF)
            page_images = []
            try:
                page_images = list(page.images)
            except Exception:
                pass

            if page_images:
                scanned_pages_detected += 1
                ocr_available, ocr_engine, ocr_setup = check_ocr_engine_availability()
                if not ocr_available:
                    if not ocr_unavailable_warned:
                        warnings.append(f"Page {page_num} appears to be a scanned image, but no local OCR engine is available.")
                        setup_instructions = ocr_setup
                        ocr_unavailable_warned = True
                else:
                    # Run OCR on the largest embedded image
                    try:
                        best_img_data = max(page_images, key=lambda img: len(img.data))
                        pil_img = Image.open(io.BytesIO(best_img_data.data))
                        ocr_text, ocr_success, method_used, ocr_err = run_ocr_on_pil_image(pil_img)
                        if ocr_success and ocr_text:
                            page_texts.append(f"--- Page {page_num} [OCR] ---\n{ocr_text}")
                            ocr_performed_pages += 1
                        elif ocr_err:
                            warnings.append(f"Page {page_num} OCR warning: {ocr_err}")
                    except Exception as img_err:
                        warnings.append(f"Page {page_num} image processing warning: {img_err}")
            else:
                # Blank page with no text or images
                pass

        # Aggregate combined text
        full_text = "\n\n".join(page_texts).strip()

        # Determine final status and classification
        is_scanned = scanned_pages_detected > 0
        doc_type = "scanned_pdf" if is_scanned and ocr_performed_pages > 0 else "digital_pdf"

        if ocr_performed_pages > 0 and len(page_texts) > ocr_performed_pages:
            method = "hybrid_pdf_ocr"
        elif ocr_performed_pages > 0:
            method = "ocr_tesseract"
        else:
            method = "digital_pdf_parser"

        # Check whether any meaningful alphanumeric characters exist in the extracted text
        # Filter out boundary header markers like '--- Page 1 ---'
        import re
        text_without_headers = re.sub(r"--- Page \d+(?: \[OCR\])? ---", "", full_text).strip()
        has_content = any(c.isalnum() for c in text_without_headers)

        if not has_content:
            if ocr_unavailable_warned:
                status = "ocr_engine_unavailable"
            else:
                status = "no_text_extracted"
                warnings.append("Document processed successfully but contains no readable alphanumeric text.")
        else:
            status = "success"

        logger.info(
            f"PDF Extraction completed [Pages={pages_to_process}, Method={method}, "
            f"Status={status}, TextLength={len(full_text)} chars]"
        )

        return ExtractionResult(
            status=status,
            document_type=doc_type,
            pages_processed=pages_to_process,
            extracted_text=full_text,
            extraction_method=method,
            warnings=warnings,
            errors=errors,
            setup_instructions=setup_instructions
        )

    def _extract_from_image_bytes(self, image_bytes: bytes) -> ExtractionResult:
        """
        Processes standalone image files (PNG, JPG, JPEG):
        1. Opens and validates image stream safely.
        2. Normalizes orientation and prepares for OCR.
        3. Applies local OCR (Tesseract or Windows Media OCR).
        """
        try:
            stream = io.BytesIO(image_bytes)
            pil_image = Image.open(stream)
            w, h = pil_image.size
            if (w * h) > MAX_IMAGE_PIXELS:
                logger.warning(f"Image exceeds configured dimension limit: {w}x{h} ({w*h} px > {MAX_IMAGE_PIXELS} px)")
                return ExtractionResult(
                    status="failed",
                    document_type="image",
                    pages_processed=0,
                    extracted_text="",
                    extraction_method="none",
                    errors=[f"Image dimensions ({w}x{h}, {w*h} pixels) exceed maximum supported safety limit of {MAX_IMAGE_PIXELS} pixels."]
                )
            # Force actual pixel decoding to ensure file integrity
            pil_image.load()
        except Image.DecompressionBombError as dbe:
            logger.warning(f"Decompression bomb limit exceeded: {dbe}")
            return ExtractionResult(
                status="failed",
                document_type="image",
                pages_processed=0,
                extracted_text="",
                extraction_method="none",
                errors=["Image dimensions exceed maximum supported safety limit. Please resize or crop the image."]
            )
        except Exception as e:
            logger.warning(f"Malformed or unreadable image: {e}")
            return ExtractionResult(
                status="failed",
                document_type="image",
                pages_processed=0,
                extracted_text="",
                extraction_method="none",
                errors=["Malformed or corrupted image format. Unable to load image data."]
            )

        ocr_available, ocr_engine, setup_msg = check_ocr_engine_availability()
        if not ocr_available:
            return ExtractionResult(
                status="ocr_engine_unavailable",
                document_type="image",
                pages_processed=1,
                extracted_text="",
                extraction_method="none",
                warnings=["Image uploaded successfully, but optical text recognition requires Tesseract OCR or Windows Media OCR."],
                setup_instructions=setup_msg
            )

        try:
            text, success, method_used, ocr_err = run_ocr_on_pil_image(pil_image)
        except Exception as e:
            logger.warning(f"OCR execution failure on image: {e}")
            return ExtractionResult(
                status="failed",
                document_type="image",
                pages_processed=1,
                extracted_text="",
                extraction_method="none",
                errors=[f"Image OCR failed: {str(e)}"]
            )

        if not success:
            return ExtractionResult(
                status="failed",
                document_type="image",
                pages_processed=1,
                extracted_text="",
                extraction_method=method_used,
                errors=[f"Image OCR failed: {ocr_err}"]
            )

        has_content = any(c.isalnum() for c in text)
        status = "success" if has_content else "no_text_extracted"
        warnings = [] if has_content else ["Image processed successfully but contains no readable alphanumeric text."]

        logger.info(f"Image OCR completed [Status={status}, Method={method_used}, TextLength={len(text)} chars]")

        return ExtractionResult(
            status=status,
            document_type="image",
            pages_processed=1,
            extracted_text=text,
            extraction_method=method_used,
            warnings=warnings,
            errors=[]
        )


# Global singleton extractor
document_extractor = MedicalDocumentExtractor()
