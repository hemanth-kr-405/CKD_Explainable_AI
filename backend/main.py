"""
FastAPI application for Explainable AI Chronic Kidney Disease Prediction.

Provides:
- GET /health: Health status and model readiness
- GET /features: Clinical feature names, metadata, types, and ranges
- POST /predict: Validates patient vitals, predicts CKD probability, and generates SHAP explanations
- GET /metrics: Transparent model evaluation and cross-validation performance metrics
"""

import sys
from pathlib import Path
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.config import (
    MODEL_ARTIFACT_PATH,
    PREPROCESSOR_ARTIFACT_PATH,
    METRICS_ARTIFACT_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURE_NAMES,
    FEATURE_METADATA,
    TARGET_NAMES
)
from src.explain_predictions import CKDExplainer
from src.document_upload import (
    document_session_manager,
    DocumentValidationError
)
from src.document_extractor import document_extractor
from src.parameter_extractor import parameter_extractor
from src.patient_verifier import verify_patient_parameters_payload
from src.report_generator import generate_clinical_pdf_report
from backend.schemas import (
    PatientInput,
    PredictionResponse,
    ProbabilityScores,
    SHAPExplanationData,
    FeatureContribution,
    FeaturesResponse,
    FeatureMetadataDetail,
    HealthResponse,
    DocumentUploadResponse,
    DocumentRemoveResponse,
    DocumentExtractionResponse,
    ParameterExtractionResponse,
    ExtractedParameterDetail,
    UnsupportedParameterDetail,
    VerifiedParameterItem,
    VerificationSubmissionRequest,
    VerificationValidationResult,
    ClinicalDistinction,
    ProvenanceSummary,
    VerifiedPredictionResponse,
    ReportDownloadRequest
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ============================================================================
# Singleton Explainer & Metrics Caching
# ============================================================================

_explainer_instance: Optional[CKDExplainer] = None
_metrics_cache: Optional[Dict[str, Any]] = None


def get_explainer() -> CKDExplainer:
    """
    Returns the singleton CKDExplainer instance.
    Initializes once and caches in memory; fails fast if artifacts are missing.
    """
    global _explainer_instance
    if _explainer_instance is None:
        if not MODEL_ARTIFACT_PATH.exists():
            err = f"Critical error: Trained model artifact not found at {MODEL_ARTIFACT_PATH}. Complete Phase 3 first."
            logging.error(err)
            raise RuntimeError(err)

        if not PREPROCESSOR_ARTIFACT_PATH.exists():
            err = f"Critical error: Preprocessor artifact not found at {PREPROCESSOR_ARTIFACT_PATH}. Complete Phase 2 first."
            logging.error(err)
            raise RuntimeError(err)

        logging.info("Initializing singleton CKDExplainer...")
        _explainer_instance = CKDExplainer(MODEL_ARTIFACT_PATH, PREPROCESSOR_ARTIFACT_PATH)
        logging.info("CKDExplainer singleton ready.")
    return _explainer_instance


def get_metrics_data() -> Dict[str, Any]:
    """
    Returns cached model evaluation metrics loaded from models/metrics.json.
    """
    global _metrics_cache
    if _metrics_cache is None:
        if METRICS_ARTIFACT_PATH.exists():
            try:
                with open(METRICS_ARTIFACT_PATH, "r", encoding="utf-8") as f:
                    _metrics_cache = json.load(f)
                logging.info("Model metrics loaded into cache.")
            except Exception as e:
                logging.warning(f"Could not load metrics: {e}")
                _metrics_cache = {}
        else:
            _metrics_cache = {}
    return _metrics_cache


# ============================================================================
# Lifespan Management (Startup and Shutdown)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-warms models, preprocessors, and explainer during application startup.
    """
    logging.info("Starting CKD Explainable AI FastAPI Service...")
    # Pre-warm explainer and metrics
    try:
        get_explainer()
        get_metrics_data()
    except Exception as e:
        logging.error(f"Failed to initialize during lifespan startup: {e}")
        raise

    yield

    logging.info("Shutting down CKD Explainable AI FastAPI Service...")
    cleaned = document_session_manager.cleanup_all()
    if cleaned > 0:
        logging.info(f"Purged {cleaned} temporary document upload(s) during shutdown.")


# ============================================================================
# FastAPI App Initialization
# ============================================================================

app = FastAPI(
    title="Explainable AI for Chronic Kidney Disease Prediction API",
    description=(
        "Clinical decision-support API predicting Chronic Kidney Disease (CKD) risk "
        "and explaining predictions via SHAP (SHapley Additive exPlanations)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable Cross-Origin Resource Sharing (CORS) with optional environment override
import os
cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Mount static frontend assets
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get("/style.css", include_in_schema=False)
async def serve_root_css():
    """Serves stylesheet directly for root-relative dashboard requests."""
    css_file = FRONTEND_DIR / "style.css"
    if css_file.exists():
        return FileResponse(css_file, media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found.")


@app.get("/script.js", include_in_schema=False)
async def serve_root_js():
    """Serves frontend JavaScript directly for root-relative dashboard requests."""
    js_file = FRONTEND_DIR / "script.js"
    if js_file.exists():
        return FileResponse(js_file, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="script.js not found.")


@app.get("/dashboard", tags=["Frontend Dashboard"])
async def dashboard():
    """Serves the interactive frontend web dashboard."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Dashboard index.html not found.")



# ============================================================================
# Custom Exception Handlers
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Returns clean, field-specific validation errors without exposing internal stack traces.
    """
    errors = []
    for err in exc.errors():
        field_path = " -> ".join(str(loc) for loc in err["loc"] if loc != "body")
        errors.append({
            "field": field_path,
            "message": err["msg"],
            "type": err["type"]
        })
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Patient input validation failed. Please review clinical ranges and types.",
            "errors": errors
        }
    )


@app.exception_handler(DocumentValidationError)
async def document_validation_exception_handler(request: Request, exc: DocumentValidationError):
    """
    Returns structured error response for document validation failures without stack traces.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.message,
            "error_type": exc.__class__.__name__
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Catches unhandled errors, logs them internally, and returns a safe error message.
    """
    logging.error(f"Internal server error handling request {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred while processing the clinical prediction."}
    )


# ============================================================================
# Helper Functions
# ============================================================================

def formulate_clinical_summary(explanation_data: Dict[str, Any]) -> str:
    """
    Synthesizes a plain-English, medically phrased summary explaining the primary
    drivers of the model's decision for the patient without asserting biological
    causation or a confirmed diagnosis.
    """
    pred_label = explanation_data["prediction"]["class_label"]
    ckd_prob = explanation_data["prediction"]["ckd_probability"]
    risk_drivers = explanation_data["shap_explanation"]["top_risk_drivers"]
    protective_factors = explanation_data["shap_explanation"]["top_protective_factors"]

    if explanation_data["prediction"]["class_code"] == 1:
        # High Risk / CKD Case
        risk_strs = [
            f"{f['clinical_label']} (value: {f['patient_value']})"
            for f in risk_drivers[:3]
        ]
        drivers_text = ", ".join(risk_strs) if risk_strs else "multivariate clinical markers"
        summary = (
            f"The model's decision path predicts {pred_label} with an uncalibrated leaf "
            f"probability of {ckd_prob * 100:.1f}%. The primary features increasing the predicted "
            f"probability in this tree path are: {drivers_text}. These statistical contributions shifted "
            "the model score away from the cohort baseline. This output represents algorithmic attribution, "
            "not biological causation or a confirmed clinical diagnosis."
        )
    else:
        # Low Risk / Non-CKD Case
        protective_strs = [
            f"{f['clinical_label']} (value: {f['patient_value']})"
            for f in protective_factors[:3]
        ]
        factors_text = ", ".join(protective_strs) if protective_strs else "multivariate clinical markers"
        summary = (
            f"The model's decision path predicts {pred_label} with an uncalibrated leaf "
            f"probability of {(1.0 - ckd_prob) * 100:.1f}%. The primary features decreasing the predicted "
            f"probability in this tree path are: {factors_text}. These values pulled the score below the "
            "cohort baseline. These associations reflect learned model patterns rather than clinical proof "
            "of renal health."
        )

    return summary


# ============================================================================
# API Routes
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Service status and navigation links."""
    return {
        "service": "Explainable AI for Chronic Kidney Disease Prediction API",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "feature_schema": "/features",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Verifies service health, model readiness, and explainer availability.
    """
    try:
        explainer = get_explainer()
        explainer_ready = explainer is not None
    except Exception:
        explainer_ready = False

    return HealthResponse(
        status="healthy" if explainer_ready else "unhealthy",
        model_loaded=MODEL_ARTIFACT_PATH.exists(),
        preprocessor_loaded=PREPROCESSOR_ARTIFACT_PATH.exists(),
        explainer_ready=explainer_ready,
        algorithm="DecisionTreeClassifier",
        features_count=len(ALL_FEATURE_NAMES),
        version="1.0.0"
    )


@app.get("/features", response_model=FeaturesResponse, tags=["Metadata"])
async def get_features_schema():
    """
    Returns the required 24 clinical features, data types, valid ranges,
    options, and baseline imputation defaults learned from training.
    """
    explainer = get_explainer()
    preprocessor = explainer.preprocessor

    features_dict = {}
    for feat in ALL_FEATURE_NAMES:
        meta = FEATURE_METADATA.get(feat, {})
        is_num = feat in NUMERICAL_FEATURES
        default_impute = (
            preprocessor.numerical_medians_.get(feat)
            if is_num
            else preprocessor.categorical_modes_.get(feat)
        )

        features_dict[feat] = FeatureMetadataDetail(
            key=feat,
            label=meta.get("label", feat),
            unit=meta.get("unit", ""),
            type="numerical" if is_num else "categorical",
            min=meta.get("min"),
            max=meta.get("max"),
            options=meta.get("options"),
            training_imputation_value=default_impute
        )

    return FeaturesResponse(
        total_features=len(ALL_FEATURE_NAMES),
        numerical_features=list(NUMERICAL_FEATURES),
        categorical_features=list(CATEGORICAL_FEATURES),
        feature_order=list(ALL_FEATURE_NAMES),
        features=features_dict
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference & XAI"])
async def predict_and_explain(patient_input: PatientInput):
    """
    Accepts patient clinical attributes, validates bounds, predicts CKD or Non-CKD,
    and returns a structured SHAP-based local explanation.
    """
    # Extract patient attributes dictionary excluding control flag
    patient_data = patient_input.model_dump(exclude={"include_plot"})
    include_plot = patient_input.include_plot

    explainer = get_explainer()

    try:
        # Run inference and SHAP explanation
        raw_result = explainer.explain_patient(
            patient_data,
            return_base64_plot=include_plot
        )

        # Formulate clinical summary
        clinical_summary = formulate_clinical_summary(raw_result)

        # Assemble structured response
        response = PredictionResponse(
            prediction_label=raw_result["prediction"]["class_label"],
            prediction_code=raw_result["prediction"]["class_code"],
            probabilities=ProbabilityScores(
                ckd_probability=raw_result["prediction"]["ckd_probability"],
                non_ckd_probability=raw_result["prediction"]["non_ckd_probability"]
            ),
            explanation_method="SHAP (TreeExplainer)",
            explanation=SHAPExplanationData(
                base_value=raw_result["shap_explanation"]["base_value"],
                patient_model_output=raw_result["shap_explanation"]["patient_model_output"],
                explanation_target=raw_result["shap_explanation"]["explanation_target"],
                output_scale=raw_result["shap_explanation"].get("output_scale", "Probability Scale [0.0 to 1.0]"),
                scale_interpretation=raw_result["shap_explanation"].get("scale_interpretation"),
                directionality_guide=raw_result["shap_explanation"].get("directionality_guide"),
                causality_caveat=raw_result["shap_explanation"].get("causality_caveat"),
                top_risk_drivers=[
                    FeatureContribution(**f) for f in raw_result["shap_explanation"]["top_risk_drivers"]
                ],
                top_protective_factors=[
                    FeatureContribution(**f) for f in raw_result["shap_explanation"]["top_protective_factors"]
                ],
                most_influential_features=[
                    FeatureContribution(**f) for f in raw_result["shap_explanation"].get("most_influential_features", [])
                ],
                all_feature_contributions=[
                    FeatureContribution(**f) for f in raw_result["shap_explanation"]["all_feature_contributions"]
                ]
            ),
            clinical_summary=clinical_summary,
            waterfall_plot_base64=raw_result.get("waterfall_plot_base64"),
            medical_disclaimer=raw_result["disclaimer"]
        )

        return response

    except Exception as e:
        logging.error(f"Prediction execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating prediction and explanation. Please verify input data."
        ) from e


# ============================================================================
# Document Processing Routes (Phase 2)
# ============================================================================

@app.post("/upload-document", response_model=DocumentUploadResponse, tags=["Document Processing"])
async def upload_medical_document(file: UploadFile = File(...)):
    """
    Accepts and securely validates an uploaded medical document (PDF, PNG, JPG/JPEG).
    Validates binary magic bytes, enforces a 10MB size limit, generates safe internal storage,
    and isolates in a temporary controlled location without persistent storage or external transmission.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided. Please upload a valid medical report."
        )

    try:
        file_bytes = await file.read()
    except Exception as e:
        logging.error(f"Failed to read uploaded file stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded file content."
        )

    try:
        record = document_session_manager.save_upload(file_bytes, file.filename)
        return DocumentUploadResponse(
            upload_id=record["upload_id"],
            filename=record["filename"],
            content_type=record["content_type"],
            file_size_bytes=record["file_size_bytes"],
            file_size_formatted=record["file_size_formatted"],
            status="uploaded_and_verified",
            message="Document uploaded and verified successfully. Ready for review and extraction.",
            uploaded_at=record["uploaded_at"]
        )
    except DocumentValidationError:
        raise
    except Exception as e:
        logging.error(f"Unexpected error during document upload processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error securely validating and staging the document."
        )


@app.delete("/upload-document/{upload_id}", response_model=DocumentRemoveResponse, tags=["Document Processing"])
async def remove_uploaded_document(upload_id: str):
    """
    Cancels and immediately purges a temporarily uploaded medical document.
    """
    purged = document_session_manager.delete_upload(upload_id)
    if not purged:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload session '{upload_id}' not found or already purged."
        )
    return DocumentRemoveResponse(
        upload_id=upload_id,
        status="removed",
        message="Temporary medical document purged securely from temporary storage."
    )


@app.post("/extract-text/{upload_id}", response_model=DocumentExtractionResponse, tags=["Document Processing"])
async def extract_document_text(upload_id: str):
    """
    Extracts text content from a previously uploaded and validated document session.
    Supports digital PDFs, scanned PDFs (with OCR), and image formats (PNG, JPG, JPEG).
    Returns structured extraction metadata while keeping extracted text strictly in-memory
    for downstream verification without external transmission.
    """
    record = document_session_manager.get_upload(upload_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload session '{upload_id}' not found or already purged. Please upload the document again."
        )

    file_path = Path(record["internal_path"])
    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Underlying temporary document file was purged or is inaccessible."
        )

    try:
        result = document_extractor.extract_from_path(file_path)

        # Cache extracted text in-memory within the active session for Phase 4
        record["extracted_text"] = result.extracted_text
        record["extraction_result"] = result

        has_content = len(result.extracted_text.strip()) > 0
        return DocumentExtractionResponse(
            upload_id=upload_id,
            status=result.status,
            document_type=result.document_type,
            pages_processed=result.pages_processed,
            extraction_method=result.extraction_method,
            text_length_characters=len(result.extracted_text),
            has_extractable_text=has_content,
            warnings=result.warnings,
            errors=result.errors,
            setup_instructions=result.setup_instructions
        )
    except Exception as e:
        logging.error(f"Error during document text extraction for upload {upload_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during document text extraction."
        )


@app.post("/extract-parameters/{upload_id}", response_model=ParameterExtractionResponse, tags=["Document Processing"])
async def extract_patient_parameters(upload_id: str):
    """
    Extracts structured laboratory and clinical parameters strictly mapped to the 24 CKD model features.
    Uses deterministic parsing with reference-range immunity, unit normalization, and conflict detection.
    Does NOT connect to the ML model or trigger predictions in this phase (Phase 4).
    """
    record = document_session_manager.get_upload(upload_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload session '{upload_id}' not found or already purged. Please upload the report again."
        )

    # If text extraction has not been performed yet, perform it now automatically
    extracted_text = record.get("extracted_text")
    if extracted_text is None:
        file_path = Path(record["internal_path"])
        if not file_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Underlying temporary document file was purged or is inaccessible."
            )
        try:
            extraction_result = document_extractor.extract_from_path(file_path)
            record["extracted_text"] = extraction_result.extracted_text
            record["extraction_result"] = extraction_result
            extracted_text = extraction_result.extracted_text
        except Exception as e:
            logging.error(f"Error executing text extraction prior to parameter extraction for {upload_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error reading text from document before parameter extraction."
            )

    extraction_result = record.get("extraction_result")
    if extraction_result:
        if extraction_result.status == "ocr_engine_unavailable":
            instructions = extraction_result.setup_instructions or "No supported OCR engine is available."
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Optical Character Recognition (OCR) is unavailable: {instructions}"
            )
        elif extraction_result.status == "failed" and extraction_result.document_type == "image":
            err_msg = "; ".join(extraction_result.errors) if extraction_result.errors else "Document text extraction failed."
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Document text extraction failed: {err_msg}"
            )

    try:
        summary = parameter_extractor.extract_from_text(extracted_text)

        # Cache structured parameters within the session for Phase 5 verification & prediction
        record["extracted_parameters"] = summary

        return ParameterExtractionResponse(
            upload_id=upload_id,
            total_features_supported=summary.total_features_supported,
            extracted_unambiguous_count=summary.extracted_unambiguous_count,
            extracted_requiring_review_count=summary.extracted_requiring_review_count,
            conflicting_count=summary.conflicting_count,
            unreadable_count=summary.unreadable_count,
            missing_count=summary.missing_count,
            unsupported_detected_count=summary.unsupported_detected_count,
            parameters={
                k: ExtractedParameterDetail(
                    canonical_name=v.canonical_name,
                    clinical_label=v.clinical_label,
                    extracted_value=v.extracted_value,
                    raw_value_str=v.raw_value_str,
                    unit=v.unit,
                    original_text=v.original_text,
                    source_page=v.source_page,
                    status=v.status,
                    warnings=v.warnings,
                    is_valid_for_model=v.is_valid_for_model,
                    repeat_count=v.repeat_count
                )
                for k, v in summary.parameters.items()
            },
            unsupported_parameters=[
                UnsupportedParameterDetail(
                    detected_name=u.detected_name,
                    raw_snippet=u.raw_snippet,
                    source_page=u.source_page,
                    status=u.status,
                    note=u.note
                )
                for u in summary.unsupported_parameters
            ],
            disclaimer=summary.disclaimer
        )
    except Exception as e:
        logging.error(f"Error extracting clinical parameters for upload {upload_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error extracting clinical parameters from document text."
        )


@app.get("/extract-parameters/{upload_id}", response_model=ParameterExtractionResponse, tags=["Document Processing"])
async def get_extracted_parameters(upload_id: str):
    """
    Retrieves previously extracted structured parameters for an active document session.
    """
    record = document_session_manager.get_upload(upload_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload session '{upload_id}' not found or already purged."
        )

    cached_summary = record.get("extracted_parameters")
    if not cached_summary:
        # If not cached yet, run extraction on the staged text
        return await extract_patient_parameters(upload_id)

    return ParameterExtractionResponse(
        upload_id=upload_id,
        total_features_supported=cached_summary.total_features_supported,
        extracted_unambiguous_count=cached_summary.extracted_unambiguous_count,
        extracted_requiring_review_count=cached_summary.extracted_requiring_review_count,
        conflicting_count=cached_summary.conflicting_count,
        unreadable_count=cached_summary.unreadable_count,
        missing_count=cached_summary.missing_count,
        unsupported_detected_count=cached_summary.unsupported_detected_count,
        parameters={
            k: ExtractedParameterDetail(
                canonical_name=v.canonical_name,
                clinical_label=v.clinical_label,
                extracted_value=v.extracted_value,
                raw_value_str=v.raw_value_str,
                unit=v.unit,
                original_text=v.original_text,
                source_page=v.source_page,
                status=v.status,
                warnings=v.warnings,
                is_valid_for_model=v.is_valid_for_model,
                repeat_count=v.repeat_count
            )
            for k, v in cached_summary.parameters.items()
        },
        unsupported_parameters=[
            UnsupportedParameterDetail(
                detected_name=u.detected_name,
                raw_snippet=u.raw_snippet,
                source_page=u.source_page,
                status=u.status,
                note=u.note
            )
            for u in cached_summary.unsupported_parameters
        ],
        disclaimer=cached_summary.disclaimer
    )


@app.post("/verify-parameters", response_model=VerificationValidationResult, tags=["Document Processing"])
async def verify_patient_parameters(request: VerificationSubmissionRequest):
    """
    Independently validates patient parameters submitted via the verification interface.
    Enforces physiological ranges and categorical domains, tracks provenance (extracted, manual, verified),
    blocks invalid or unresolved conflicting values, and sanitizes input for prediction.
    """
    try:
        validation_result = verify_patient_parameters_payload(
            submitted_parameters=request.parameters,
            allow_imputation=request.allow_imputation
        )

        # If upload session UUID provided, cache verification in ephemeral storage
        if request.upload_id:
            record = document_session_manager.get_upload(request.upload_id)
            if record:
                record["verified_parameters"] = validation_result
                if validation_result.sanitized_input:
                    record["sanitized_patient_input"] = validation_result.sanitized_input

        # Convert records to Pydantic items
        param_items = {
            k: VerifiedParameterItem(
                canonical_name=v.canonical_name,
                clinical_label=v.clinical_label,
                value=v.value,
                unit=v.unit,
                origin=v.origin,
                status=v.status,
                source_text=v.source_text,
                source_page=v.source_page,
                warnings=v.warnings,
                errors=v.errors,
                is_valid=v.is_valid
            )
            for k, v in validation_result.parameters.items()
        }

        return VerificationValidationResult(
            is_valid_for_prediction=validation_result.is_valid_for_prediction,
            total_features=validation_result.total_features,
            verified_count=validation_result.verified_count,
            requires_review_count=validation_result.requires_review_count,
            missing_count=validation_result.missing_count,
            conflicting_count=validation_result.conflicting_count,
            invalid_count=validation_result.invalid_count,
            extracted_count=validation_result.extracted_count,
            manual_count=validation_result.manual_count,
            parameters=param_items,
            validation_errors=validation_result.validation_errors,
            sanitized_input=validation_result.sanitized_input,
            message=validation_result.message
        )
    except Exception as e:
        logging.error(f"Error during parameter verification: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error validating patient parameters."
        )


@app.post("/predict-verified", response_model=VerifiedPredictionResponse, tags=["Inference & XAI"])
async def predict_verified_parameters(request: VerificationSubmissionRequest):
    """
    Accepts verified patient parameters, executes independent backend verification,
    constructs model input matching the exact 24-feature schema, preserves training-time
    preprocessing and feature ordering, runs inference via the existing trained Decision Tree,
    computes SHAP local explanations, and returns predictions with explicit probability caveats
    and clear distinctions from clinical diagnoses.
    """
    # 1. Independent backend validation
    try:
        validation_result = verify_patient_parameters_payload(
            submitted_parameters=request.parameters,
            allow_imputation=request.allow_imputation
        )
    except Exception as e:
        logging.error(f"Error during backend verification in predict-verified: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error validating patient parameters before prediction."
        )

    # 2. Gate prediction: Reject if invalid bounds or unresolved conflicts exist
    if not validation_result.is_valid_for_prediction or not validation_result.sanitized_input:
        error_details = []
        for feat, errs in validation_result.validation_errors.items():
            error_details.append(f"{feat}: {'; '.join(errs)}")
        error_msg = validation_result.message
        if error_details:
            error_msg += f" Details: {' | '.join(error_details)}"
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_msg
        )

    if validation_result.missing_count == 24:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prediction is blocked: All 24 patient clinical parameters are missing. Please enter or verify patient lab values."
        )

    sanitized_input = validation_result.sanitized_input

    # 3. Cache verified parameters in session if upload_id was provided
    if request.upload_id:
        record = document_session_manager.get_upload(request.upload_id)
        if record:
            record["verified_parameters"] = validation_result
            record["sanitized_patient_input"] = sanitized_input

    # 4. Generate prediction and SHAP explanation using existing CKDExplainer singleton
    try:
        explainer = get_explainer()
        raw_result = explainer.explain_patient(
            sanitized_input,
            return_base64_plot=request.include_plot
        )
    except Exception as e:
        logging.error(f"Prediction execution failed on verified parameters: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error generating model prediction and explanation from verified parameters."
        )

    # 5. Formulate clinical narrative, distinction, and structured response
    clinical_summary = formulate_clinical_summary(raw_result)

    pred_label = raw_result["prediction"]["class_label"]
    pred_code = raw_result["prediction"]["class_code"]
    ckd_prob = raw_result["prediction"]["ckd_probability"]
    non_ckd_prob = raw_result["prediction"]["non_ckd_probability"]

    clinical_distinction = ClinicalDistinction(
        model_prediction=f"DecisionTree algorithmic classification: {pred_label} (Class {pred_code})",
        model_probability=(
            f"Terminal leaf proportion: {ckd_prob * 100:.1f}% for CKD, {non_ckd_prob * 100:.1f}% for Non-CKD. "
            f"This represents training sample leaf purity, NOT an individual patient's clinical risk of disease."
        ),
        clinical_diagnosis=(
            "CANNOT be established by machine learning. A definitive medical diagnosis requires "
            "formal clinical evaluation by a licensed physician, comprehensive medical history, and confirmatory laboratory workup."
        )
    )

    provenance_summary = ProvenanceSummary(
        total_features=validation_result.total_features,
        verified_count=validation_result.verified_count,
        extracted_count=validation_result.extracted_count,
        manual_count=validation_result.manual_count,
        missing_count=validation_result.missing_count,
        imputation_applied=(validation_result.missing_count > 0),
        imputation_strategy="Documented training-cohort median for numerics and mode for categoricals"
    )

    response = VerifiedPredictionResponse(
        prediction_label=pred_label,
        prediction_code=pred_code,
        probabilities=ProbabilityScores(
            ckd_probability=ckd_prob,
            non_ckd_probability=non_ckd_prob
        ),
        probability_caveat=(
            "Decision Tree leaf probabilities reflect empirical sample proportions of the terminal "
            "leaf node from the training cohort. They are uncalibrated frequency estimates and MUST NOT "
            "be interpreted as an individual patient's clinical risk of disease."
        ),
        clinical_distinction=clinical_distinction,
        provenance_summary=provenance_summary,
        explanation_method="SHAP (TreeExplainer)",
        explanation=SHAPExplanationData(
            base_value=raw_result["shap_explanation"]["base_value"],
            patient_model_output=raw_result["shap_explanation"]["patient_model_output"],
            explanation_target=raw_result["shap_explanation"]["explanation_target"],
            output_scale=raw_result["shap_explanation"].get("output_scale", "Probability Scale [0.0 to 1.0]"),
            scale_interpretation=raw_result["shap_explanation"].get("scale_interpretation"),
            directionality_guide=raw_result["shap_explanation"].get("directionality_guide"),
            causality_caveat=raw_result["shap_explanation"].get("causality_caveat"),
            top_risk_drivers=[
                FeatureContribution(**f) for f in raw_result["shap_explanation"]["top_risk_drivers"]
            ],
            top_protective_factors=[
                FeatureContribution(**f) for f in raw_result["shap_explanation"]["top_protective_factors"]
            ],
            most_influential_features=[
                FeatureContribution(**f) for f in raw_result["shap_explanation"].get("most_influential_features", [])
            ],
            all_feature_contributions=[
                FeatureContribution(**f) for f in raw_result["shap_explanation"]["all_feature_contributions"]
            ]
        ),
        clinical_summary=clinical_summary,
        waterfall_plot_base64=raw_result.get("waterfall_plot_base64"),
        medical_disclaimer=(
            "Educational and Clinical Decision Support: Model outputs are statistical patterns and "
            "do not constitute a medical diagnosis or treatment directive. Clinical safety and diagnostic "
            "accuracy cannot be claimed on the basis of software tests alone. Final medical determinations "
            "must be made by qualified healthcare professionals."
        )
    )

    if request.upload_id:
        record = document_session_manager.get_upload(request.upload_id)
        if record:
            record["verified_prediction_response"] = response.model_dump()

    return response


@app.post("/download-report", tags=["Clinical Decision Support"])
async def download_clinical_report(request: ReportDownloadRequest):
    """
    Generates and downloads a vector PDF clinical decision support report containing:
    - Patient and document metadata
    - Model prediction and uncalibrated leaf purity
    - Clinical vs model boundary distinction
    - Complete 24-parameter laboratory provenance record
    - Quantitative SHAP attribution analysis and directionality guide
    - Medical and legal non-diagnostic disclaimers
    """
    try:
        pdf_bytes = generate_clinical_pdf_report(
            prediction_data=request.prediction_result,
            verified_parameters=request.verified_parameters,
            document_name=request.document_name,
            session_id=request.upload_id
        )

        filename = f"CKD_Clinical_Report_{request.upload_id[:8] if request.upload_id else 'Session'}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=\"{filename}\"",
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache"
            }
        )
    except Exception as exc:
        logging.error(f"Report generation error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate clinical PDF report. Please verify prediction results."
        )


@app.get("/metrics", tags=["Metadata"])
async def get_model_metrics():
    """
    Returns model performance metrics and cross-validation results.
    """
    metrics = get_metrics_data()
    if metrics:
        return metrics
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Model metrics file not found."
    )


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    logging.info(f"Starting server on {host}:{port}...")
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
