"""
Pydantic schemas for the CKD Explainable AI FastAPI backend.

Provides strict data validation, clinical range enforcement, and consistent
response structures for predictions, SHAP explanations, health checks, and feature metadata.
"""

from typing import Optional, List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Input Clinical Patient Schema
# ============================================================================

class PatientInput(BaseModel):
    """
    Clinical measurements and patient attributes for Chronic Kidney Disease screening.
    Enforces physiologically realistic clinical bounds while allowing optional fields
    (missing values are safely handled by the trained median/mode preprocessor).
    """
    model_config = ConfigDict(
        extra="forbid",  # Reject unknown / unverified clinical fields
        json_schema_extra={
            "example": {
                "age": 48.0,
                "bp": 80.0,
                "sg": 1.020,
                "al": 1.0,
                "su": 0.0,
                "bgr": 121.0,
                "bu": 36.0,
                "sc": 1.2,
                "sod": 138.0,
                "pot": 4.3,
                "hemo": 15.4,
                "pcv": 44.0,
                "wbcc": 7800.0,
                "rbcc": 5.2,
                "rbc": "normal",
                "pc": "normal",
                "pcc": "notpresent",
                "ba": "notpresent",
                "htn": "yes",
                "dm": "yes",
                "cad": "no",
                "appet": "good",
                "pe": "no",
                "ane": "no",
                "include_plot": True
            }
        }
    )

    # --- 14 Numerical Features ---
    age: Optional[float] = Field(
        default=None, ge=1.0, le=120.0,
        description="Patient age in years (1 to 120)"
    )
    bp: Optional[float] = Field(
        default=None, ge=40.0, le=250.0,
        description="Blood pressure in mm/Hg (40 to 250)"
    )
    sg: Optional[float] = Field(
        default=None, ge=1.000, le=1.040,
        description="Specific gravity of urine (1.000 to 1.040, e.g. 1.010, 1.015, 1.020, 1.025)"
    )
    al: Optional[float] = Field(
        default=None, ge=0.0, le=5.0,
        description="Urine albumin grade (0 to 5, where 0=normal and 5=severe proteinuria)"
    )
    su: Optional[float] = Field(
        default=None, ge=0.0, le=5.0,
        description="Urine sugar grade (0 to 5)"
    )
    bgr: Optional[float] = Field(
        default=None, ge=20.0, le=600.0,
        description="Blood glucose random in mg/dL (20 to 600)"
    )
    bu: Optional[float] = Field(
        default=None, ge=1.0, le=500.0,
        description="Blood urea in mg/dL (1 to 500)"
    )
    sc: Optional[float] = Field(
        default=None, ge=0.1, le=100.0,
        description="Serum creatinine in mg/dL (0.1 to 100.0)"
    )
    sod: Optional[float] = Field(
        default=None, ge=50.0, le=200.0,
        description="Serum sodium in mEq/L (50 to 200)"
    )
    pot: Optional[float] = Field(
        default=None, ge=1.0, le=60.0,
        description="Serum potassium in mEq/L (1.0 to 60.0)"
    )
    hemo: Optional[float] = Field(
        default=None, ge=2.0, le=25.0,
        description="Hemoglobin in g/dL (2.0 to 25.0)"
    )
    pcv: Optional[float] = Field(
        default=None, ge=5.0, le=70.0,
        description="Packed cell volume / hematocrit in % (5 to 70)"
    )
    wbcc: Optional[float] = Field(
        default=None, ge=500.0, le=50000.0,
        description="White blood cell count in cells/cumm (500 to 50,000)"
    )
    rbcc: Optional[float] = Field(
        default=None, ge=0.5, le=12.0,
        description="Red blood cell count in millions/cmm (0.5 to 12.0)"
    )

    # --- 10 Categorical Features ---
    rbc: Optional[Literal["normal", "abnormal"]] = Field(
        default=None, description="Red blood cells in urine ('normal' or 'abnormal')"
    )
    pc: Optional[Literal["normal", "abnormal"]] = Field(
        default=None, description="Pus cells in urine ('normal' or 'abnormal')"
    )
    pcc: Optional[Literal["notpresent", "present"]] = Field(
        default=None, description="Pus cell clumps ('notpresent' or 'present')"
    )
    ba: Optional[Literal["notpresent", "present"]] = Field(
        default=None, description="Bacteria in urine ('notpresent' or 'present')"
    )
    htn: Optional[Literal["no", "yes"]] = Field(
        default=None, description="Hypertension diagnosis ('no' or 'yes')"
    )
    dm: Optional[Literal["no", "yes"]] = Field(
        default=None, description="Diabetes mellitus diagnosis ('no' or 'yes')"
    )
    cad: Optional[Literal["no", "yes"]] = Field(
        default=None, description="Coronary artery disease ('no' or 'yes')"
    )
    appet: Optional[Literal["good", "poor"]] = Field(
        default=None, description="Appetite quality ('good' or 'poor')"
    )
    pe: Optional[Literal["no", "yes"]] = Field(
        default=None, description="Pedal edema / swelling in feet ('no' or 'yes')"
    )
    ane: Optional[Literal["no", "yes"]] = Field(
        default=None, description="Anemia diagnosis ('no' or 'yes')"
    )

    # Optional control flag
    include_plot: bool = Field(
        default=True,
        description="Whether to generate and return a base64-encoded SHAP waterfall plot"
    )


# ============================================================================
# Response Schemas
# ============================================================================

class FeatureContribution(BaseModel):
    """Attribution details for an individual clinical parameter."""
    feature: str = Field(..., description="Internal feature key (e.g. 'hemo')")
    clinical_label: str = Field(..., description="Human-readable medical label with unit")
    patient_value: Any = Field(..., description="Input or imputed value for this patient")
    transformed_value: float = Field(..., description="Numeric value after preprocessing")
    shap_value: float = Field(..., description="Shapley contribution value towards CKD risk")
    direction: str = Field(..., description="Clinical directional impact description")
    effect: str = Field(..., description="'increases_risk', 'decreases_risk', or 'neutral'")
    unit: str = Field(default="", description="Clinical unit of measurement")


class SHAPExplanationData(BaseModel):
    """Structured SHAP explanation output."""
    base_value: float = Field(..., description="Background training cohort average risk baseline")
    patient_model_output: float = Field(..., description="Cumulative model risk score (base_value + sum of SHAP)")
    explanation_target: str = Field(default="Probability of Chronic Kidney Disease (CKD)")
    output_scale: str = Field(default="Probability Scale [0.0 to 1.0]", description="Scale of model output and SHAP attributions")
    scale_interpretation: Optional[str] = Field(
        default="TreeExplainer attributions reflect the probability scale of the leaf nodes for DecisionTreeClassifier.",
        description="Explanation of the attribution scale"
    )
    directionality_guide: Optional[Dict[str, str]] = Field(
        default=None,
        description="Directional interpretation guide (positive toward CKD, negative toward Non-CKD)"
    )
    causality_caveat: Optional[str] = Field(
        default="Feature attributions reflect statistical associations in the trained model, not biological causation or patient clinical risk factors.",
        description="Explicit non-causality warning"
    )
    top_risk_drivers: List[FeatureContribution] = Field(..., description="Top factors elevating CKD probability")
    top_protective_factors: List[FeatureContribution] = Field(..., description="Top factors lowering CKD probability")
    most_influential_features: List[FeatureContribution] = Field(
        default_factory=list,
        description="Features ranked by absolute attribution magnitude (|SHAP|)"
    )
    all_feature_contributions: List[FeatureContribution] = Field(..., description="Attributions across all 24 features")



class ProbabilityScores(BaseModel):
    """Model confidence probability scores."""
    ckd_probability: float = Field(..., ge=0.0, le=1.0, description="Estimated probability of CKD (0.0 to 1.0)")
    non_ckd_probability: float = Field(..., ge=0.0, le=1.0, description="Estimated probability of Non-CKD (0.0 to 1.0)")


class PredictionResponse(BaseModel):
    """Consistent JSON response format for clinical prediction & explanation."""
    prediction_label: str = Field(..., description="'Chronic Kidney Disease (CKD)' or 'Non-CKD'")
    prediction_code: int = Field(..., description="1 for CKD, 0 for Non-CKD")
    probabilities: ProbabilityScores
    explanation_method: str = Field(default="SHAP (TreeExplainer)")
    explanation: SHAPExplanationData
    clinical_summary: str = Field(..., description="Carefully worded plain-English clinical explanation")
    waterfall_plot_base64: Optional[str] = Field(default=None, description="Base64 PNG image of SHAP waterfall plot")
    medical_disclaimer: str = Field(..., description="Non-diagnostic research disclaimer")


class FeatureMetadataDetail(BaseModel):
    """Metadata describing a single clinical feature."""
    key: str
    label: str
    unit: str
    type: str  # 'numerical' or 'categorical'
    min: Optional[float] = None
    max: Optional[float] = None
    options: Optional[List[str]] = None
    training_imputation_value: Any = None


class FeaturesResponse(BaseModel):
    """Comprehensive schema metadata for the frontend form."""
    total_features: int
    numerical_features: List[str]
    categorical_features: List[str]
    feature_order: List[str]
    features: Dict[str, FeatureMetadataDetail]


class HealthResponse(BaseModel):
    """Application health and status summary."""
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    explainer_ready: bool
    algorithm: str
    features_count: int
    version: str


# ============================================================================
# Document Upload Schemas
# ============================================================================

class DocumentUploadResponse(BaseModel):
    """Metadata response returned after securely validating and staging a document."""
    upload_id: str = Field(..., description="Unique ephemeral session reference UUID")
    filename: str = Field(..., description="Sanitized original filename")
    content_type: str = Field(..., description="Verified MIME type")
    file_size_bytes: int = Field(..., description="File size in bytes")
    file_size_formatted: str = Field(..., description="Human-readable file size")
    status: str = Field(default="uploaded_and_verified")
    message: str = Field(..., description="Status confirmation message")
    uploaded_at: str = Field(..., description="ISO 8601 UTC timestamp of upload")


class DocumentRemoveResponse(BaseModel):
    """Response returned upon cancelling/purging a temporary uploaded document."""
    upload_id: str = Field(..., description="Identifier of the purged document session")
    status: str = Field(default="removed")
    message: str = Field(..., description="Status message")


class DocumentExtractionResponse(BaseModel):
    """Structured metadata response for document text extraction."""
    upload_id: str = Field(..., description="Unique ephemeral session reference UUID")
    status: str = Field(..., description="Extraction outcome: 'success', 'no_text_extracted', 'ocr_engine_unavailable', 'failed'")
    document_type: str = Field(..., description="Detected document type: digital_pdf, scanned_pdf, image")
    pages_processed: int = Field(..., description="Number of document pages parsed")
    extraction_method: str = Field(..., description="Method used: digital_pdf_parser, ocr_tesseract, hybrid_pdf_ocr, none")
    text_length_characters: int = Field(..., description="Total count of extracted characters")
    has_extractable_text: bool = Field(..., description="Whether meaningful alphanumeric text was retrieved")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal extraction warnings")
    errors: List[str] = Field(default_factory=list, description="Fatal extraction error descriptions")
    setup_instructions: Optional[str] = Field(default=None, description="Setup instructions if external software like Tesseract is needed")


# ============================================================================
# Parameter Extraction Schemas (Phase 4)
# ============================================================================

class ExtractedParameterDetail(BaseModel):
    """Structured detail for an individual extracted clinical parameter."""
    canonical_name: str = Field(..., description="Canonical model feature key, e.g. 'sc', 'hemo', 'bp'")
    clinical_label: str = Field(..., description="Human-readable clinical parameter name")
    extracted_value: Optional[Union[float, str]] = Field(default=None, description="Extracted numerical or categorical value")
    raw_value_str: Optional[str] = Field(default=None, description="Raw literal string matched from source")
    unit: Optional[str] = Field(default=None, description="Canonical or extracted clinical unit")
    original_text: Optional[str] = Field(default=None, description="Source line/fragment where parameter was located")
    source_page: Optional[int] = Field(default=None, description="Document page number where parameter was found")
    status: str = Field(..., description="'extracted_unambiguous', 'extracted_requiring_review', 'conflicting', 'unreadable', 'missing'")
    warnings: List[str] = Field(default_factory=list, description="Clinical warnings, ambiguity details, or conversion notes")
    is_valid_for_model: bool = Field(default=False, description="Whether this value is clean and verified for model input")
    repeat_count: int = Field(default=0, description="Number of times this parameter was observed across the document")


class UnsupportedParameterDetail(BaseModel):
    """Detail for a recognized medical test outside the 24 CKD features."""
    detected_name: str = Field(..., description="Recognized test name, e.g. 'Platelet Count', 'SGPT / ALT'")
    raw_snippet: str = Field(..., description="Source text snippet containing the unsupported test")
    source_page: Optional[int] = Field(default=None, description="Page where the test was mentioned")
    status: str = Field(default="unsupported")
    note: str = Field(default="Parameter detected in document but not part of the 24-feature CKD model schema.")


class ParameterExtractionResponse(BaseModel):
    """Complete response returned for patient parameter extraction."""
    upload_id: str = Field(..., description="Unique ephemeral session reference UUID")
    total_features_supported: int = Field(default=24, description="Total features in the CKD model schema")
    extracted_unambiguous_count: int = Field(..., description="Count of clean, unambiguous parameter extractions")
    extracted_requiring_review_count: int = Field(..., description="Count of extractions requiring clinician review")
    conflicting_count: int = Field(..., description="Count of parameters with discordant values across pages")
    unreadable_count: int = Field(..., description="Count of parameters with damaged/obscured readings")
    missing_count: int = Field(..., description="Count of supported features absent from the document")
    unsupported_detected_count: int = Field(..., description="Count of recognized non-CKD tests detected")
    parameters: Dict[str, ExtractedParameterDetail] = Field(..., description="Map of canonical feature keys to extraction details")
    unsupported_parameters: List[UnsupportedParameterDetail] = Field(default_factory=list, description="Non-CKD tests observed in report")
    disclaimer: str = Field(..., description="Clinical non-diagnostic observation disclaimer")


# ============================================================================
# Verification Interface Schemas (Phase 5)
# ============================================================================

class VerifiedParameterItem(BaseModel):
    """Represents an individual parameter after clinical verification."""
    canonical_name: str = Field(..., description="Canonical model key, e.g. 'sc', 'hemo', 'bp'")
    clinical_label: Optional[str] = Field(default=None, description="Human readable label")
    value: Optional[Union[float, str]] = Field(default=None, description="Validated numerical or categorical value")
    unit: Optional[str] = Field(default=None, description="Canonical unit of measurement")
    origin: str = Field(default="manual", description="'extracted', 'manual', or 'verified'")
    status: str = Field(default="verified", description="'verified', 'requires_review', 'missing', 'conflicting', 'invalid'")
    source_text: Optional[str] = Field(default=None, description="Original report snippet if extracted")
    source_page: Optional[int] = Field(default=None, description="Source page if extracted")
    warnings: List[str] = Field(default_factory=list, description="Verification warnings or conversion notes")
    errors: List[str] = Field(default_factory=list, description="Validation errors if any")
    is_valid: bool = Field(default=True, description="Whether parameter passed range/category checks")


class VerificationSubmissionRequest(BaseModel):
    """Payload sent by the verification interface when confirming/submitting parameters."""
    upload_id: Optional[str] = Field(default=None, description="Optional active session upload UUID")
    parameters: Dict[str, Any] = Field(..., description="Dictionary of parameter items or primitive values")
    allow_imputation: bool = Field(default=True, description="Whether preprocessor training median/mode imputation is acknowledged for missing fields")
    include_plot: bool = Field(default=True, description="Whether to generate and return a base64-encoded SHAP waterfall plot")


class VerificationValidationResult(BaseModel):
    """Result returned by the backend after independent validation of verified parameters."""
    is_valid_for_prediction: bool = Field(..., description="True if input passes all validation checks and can proceed to prediction")
    total_features: int = Field(default=24, description="Total features in the CKD model schema")
    verified_count: int = Field(..., description="Count of verified parameters")
    requires_review_count: int = Field(..., description="Count of parameters flagged for review")
    missing_count: int = Field(..., description="Count of missing/unpopulated parameters")
    conflicting_count: int = Field(..., description="Count of parameters with unresolved conflicts")
    invalid_count: int = Field(..., description="Count of out-of-range or invalid parameters")
    extracted_count: int = Field(..., description="Count of parameters originating from extraction")
    manual_count: int = Field(..., description="Count of parameters manually entered or edited")
    parameters: Dict[str, VerifiedParameterItem] = Field(..., description="Validated parameter items with provenance and status")
    validation_errors: Dict[str, List[str]] = Field(default_factory=dict, description="Field-specific validation error messages")
    sanitized_input: Optional[Dict[str, Any]] = Field(default=None, description="Sanitized dictionary ready for PatientInput / prediction")
    message: str = Field(..., description="Validation summary message")


# ============================================================================
# Phase 6: Verified Prediction & Model Integration Schemas
# ============================================================================

class ClinicalDistinction(BaseModel):
    """
    Explicitly separates algorithmic outputs from clinical diagnoses to prevent
    misleading medical interpretation.
    """
    model_prediction: str = Field(
        ...,
        description="Machine learning classification from the trained Decision Tree model."
    )
    model_probability: str = Field(
        ...,
        description="Uncalibrated empirical class proportion observed at the terminating tree leaf."
    )
    clinical_diagnosis: str = Field(
        default=(
            "CANNOT be established by machine learning. A definitive medical diagnosis requires "
            "formal clinical evaluation by a licensed physician, clinical history, and laboratory workup."
        ),
        description="Clear statement that the model cannot establish a medical diagnosis."
    )


class ProvenanceSummary(BaseModel):
    """Audit summary of data sources and missing value handling for this prediction."""
    total_features: int = Field(default=24, description="Total features in the CKD model schema")
    verified_count: int = Field(..., description="Number of parameters verified by user or review")
    extracted_count: int = Field(..., description="Number of parameters directly extracted from medical document")
    manual_count: int = Field(..., description="Number of parameters manually entered or modified")
    missing_count: int = Field(..., description="Number of parameters missing from the report")
    imputation_applied: bool = Field(..., description="Whether training-set median/mode imputation was applied")
    imputation_strategy: str = Field(
        default="Documented training-cohort median for numerics and mode for categoricals",
        description="Description of the validated imputation methodology"
    )


class VerifiedPredictionResponse(BaseModel):
    """
    Structured response for predictions generated from verified patient data.
    Provides prediction, leaf probabilities with appropriate caveats, explicit clinical distinction,
    provenance accounting, and SHAP explainability.
    """
    prediction_label: str = Field(..., description="'Chronic Kidney Disease (CKD)' or 'Non-CKD'")
    prediction_code: int = Field(..., description="1 for CKD, 0 for Non-CKD")
    probabilities: ProbabilityScores
    probability_caveat: str = Field(
        default=(
            "Decision Tree leaf probabilities reflect empirical sample proportions of the terminal "
            "leaf node from the training cohort. They are uncalibrated frequency estimates and MUST NOT "
            "be interpreted as an individual patient's clinical risk of disease."
        ),
        description="Explicit caveat clarifying that uncalibrated probabilities are not patient risk"
    )
    clinical_distinction: ClinicalDistinction
    provenance_summary: ProvenanceSummary
    explanation_method: str = Field(default="SHAP (TreeExplainer)")
    explanation: SHAPExplanationData
    clinical_summary: str = Field(..., description="Carefully worded algorithmic reasoning narrative")
    waterfall_plot_base64: Optional[str] = Field(default=None, description="Base64 PNG image of SHAP waterfall plot")
    medical_disclaimer: str = Field(
        default=(
            "Educational and Clinical Decision Support: Model outputs are statistical patterns and "
            "do not constitute a medical diagnosis or treatment directive. Clinical safety and diagnostic "
            "accuracy cannot be claimed on the basis of software tests alone. Final medical determinations "
            "must be made by qualified healthcare professionals."
        ),
        description="Mandatory clinical safety and diagnostic non-assertion disclaimer"
    )


class ReportDownloadRequest(BaseModel):
    """Payload to generate and download a comprehensive clinical PDF report."""
    upload_id: Optional[str] = Field(default=None, description="Optional upload session id")
    document_name: Optional[str] = Field(default=None, description="Original uploaded document name")
    prediction_result: Dict[str, Any] = Field(..., description="Full prediction result object")
    verified_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Verified parameters state")




