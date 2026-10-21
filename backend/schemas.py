"""
Pydantic schemas for the CKD Explainable AI FastAPI backend.

Provides strict data validation, clinical range enforcement, and consistent
response structures for predictions, SHAP explanations, health checks, and feature metadata.
"""

from typing import Optional, List, Dict, Any, Literal
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
    interpretation_guide: str = Field(..., description="Guidance on reading positive vs negative contributions")
    top_risk_drivers: List[FeatureContribution] = Field(..., description="Top factors elevating CKD probability")
    top_protective_factors: List[FeatureContribution] = Field(..., description="Top factors lowering CKD probability")
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
