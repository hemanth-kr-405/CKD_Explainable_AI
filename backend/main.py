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

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
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
from backend.schemas import (
    PatientInput,
    PredictionResponse,
    ProbabilityScores,
    SHAPExplanationData,
    FeatureContribution,
    FeaturesResponse,
    FeatureMetadataDetail,
    HealthResponse
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
    allow_methods=["GET", "POST", "OPTIONS"],
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
                interpretation_guide=raw_result["shap_explanation"]["interpretation_guide"],
                top_risk_drivers=[
                    FeatureContribution(**f) for f in raw_result["shap_explanation"]["top_risk_drivers"]
                ],
                top_protective_factors=[
                    FeatureContribution(**f) for f in raw_result["shap_explanation"]["top_protective_factors"]
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
