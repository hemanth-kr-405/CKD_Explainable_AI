"""
Deterministic Patient Parameter Extraction Module for CKD Explainable AI.

Objective:
Extract structured clinical and laboratory parameters from unstructured medical document text
produced by Phase 3 (text extraction), mapped strictly to the real model's 24 input features.

Rules & Guarantees:
1. Strict schema alignment: Only extracts features belonging to the 24 UCI CKD model inputs.
2. Reference range immunity: Strips or bypasses laboratory reference intervals (e.g., 'Ref: 0.6 - 1.2')
   so reference range boundaries are never mistaken for patient results.
3. Zero invention: Never invents missing values, substitutes zero, or guesses unreadable data.
4. Unit-aware normalization: Safely converts standard clinical laboratory units:
   - Serum Creatinine: umol/L -> mg/dL (divide by 88.42)
   - Blood Glucose: mmol/L -> mg/dL (multiply by 18.018)
   - Hemoglobin: g/L -> g/dL (divide by 10)
   - Blood Urea: BUN mg/dL -> Blood Urea mg/dL (multiply by 2.14)
   - White Blood Cell Count: 10^3/uL -> cells/cumm (multiply by 1000)
   - Red Blood Cell Count: 10^6/uL -> millions/cmm (1:1 standard float)
5. Status stratification:
   - 'extracted_unambiguous': clean, directly matched result within physiological bounds.
   - 'extracted_requiring_review': unit converted, qualitative dipstick mapped, or outlier value.
   - 'conflicting': multiple discordant values found across pages or sections.
   - 'unreadable': parameter name found but value obscured, hemolyzed, or damaged.
   - 'missing': parameter not present in the document.
   - 'unsupported': detected non-CKD tests (e.g. Platelets, SGPT, Bilirubin) recorded separately.
6. Non-diagnostic boundary: Extraction is purely structural observation; never infers a diagnosis.
7. Absolute privacy: 100% local deterministic regex and text parsing. Zero cloud or external API calls.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURE_NAMES,
    FEATURE_METADATA,
    CATEGORICAL_MAPPINGS
)

logger = logging.getLogger("ckd.parameter_extractor")

# ============================================================================
# Data Models for Parameter Extraction
# ============================================================================

@dataclass
class ExtractedParameter:
    """Represents the extraction result for a single canonical CKD model feature."""
    canonical_name: str
    clinical_label: str
    extracted_value: Optional[Union[float, str]]
    raw_value_str: Optional[str]
    unit: Optional[str]
    original_text: Optional[str]
    source_page: Optional[int]
    status: str  # "extracted_unambiguous", "extracted_requiring_review", "conflicting", "unreadable", "missing"
    warnings: List[str] = field(default_factory=list)
    is_valid_for_model: bool = False
    repeat_count: int = 0
    all_observed_values: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary suitable for JSON serialization."""
        return asdict(self)


@dataclass
class UnsupportedParameter:
    """Represents a recognized medical/lab test outside the 24-feature CKD schema."""
    detected_name: str
    raw_snippet: str
    source_page: Optional[int]
    status: str = "unsupported"
    note: str = "Parameter detected in document but not part of the 24-feature CKD model schema."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionSummary:
    """Complete summary of structured parameter extraction for a medical document."""
    total_features_supported: int = len(ALL_FEATURE_NAMES)
    extracted_unambiguous_count: int = 0
    extracted_requiring_review_count: int = 0
    conflicting_count: int = 0
    unreadable_count: int = 0
    missing_count: int = 0
    unsupported_detected_count: int = 0
    parameters: Dict[str, ExtractedParameter] = field(default_factory=dict)
    unsupported_parameters: List[UnsupportedParameter] = field(default_factory=list)
    disclaimer: str = (
        "Parameter extraction is observational and experimental. It does not constitute a clinical diagnosis "
        "or medical interpretation. Extracted values must be verified by a qualified healthcare professional."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_features_supported": self.total_features_supported,
            "extracted_unambiguous_count": self.extracted_unambiguous_count,
            "extracted_requiring_review_count": self.extracted_requiring_review_count,
            "conflicting_count": self.conflicting_count,
            "unreadable_count": self.unreadable_count,
            "missing_count": self.missing_count,
            "unsupported_detected_count": self.unsupported_detected_count,
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
            "unsupported_parameters": [u.to_dict() for u in self.unsupported_parameters],
            "disclaimer": self.disclaimer
        }


# ============================================================================
# Reference Range & Unreadable Patterns
# ============================================================================

# Patterns that represent laboratory reference ranges / normal intervals
REF_RANGE_PATTERNS = [
    # (Ref: 0.6 - 1.2 mg/dL) or [Ref: 0.6-1.2]
    re.compile(r'(?:\(|\[)\s*(?:Bio(?:logical)?\.?\s*)?Ref(?:erence)?\.?(?:\s*(?:Range|Interval|Values?))?[\s:]*([0-9]+(?:\.[0-9]+)?)\s*(?:-|–|—|to)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:[a-zA-Z/%^0-9]+)?\s*(?:\)|\])', re.IGNORECASE),
    # Ref: 0.6 - 1.2 or Reference Range 15.0 - 45.0
    re.compile(r'\b(?:Bio(?:logical)?\.?\s*)?Ref(?:erence)?\.?(?:\s*(?:Range|Interval|Values?))?[\s:]+([0-9]+(?:\.[0-9]+)?)\s*(?:-|–|—|to)\s*([0-9]+(?:\.[0-9]+)?)\b', re.IGNORECASE),
    # (Normal: 135 - 145) or [Normal: 3.5 - 5.0]
    re.compile(r'(?:\(|\[)\s*Normal[\s:]*([0-9]+(?:\.[0-9]+)?)\s*(?:-|–|—|to)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:[a-zA-Z/%^0-9]+)?\s*(?:\)|\])', re.IGNORECASE),
    # Unilateral bounds: (Ref: < 140) or (Ref: > 60)
    re.compile(r'(?:\(|\[)\s*(?:Bio(?:logical)?\.?\s*)?Ref(?:erence)?\.?(?:\s*(?:Range|Interval|Values?))?[\s:]*(?:<|>|<=|>=)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:[a-zA-Z/%^0-9]+)?\s*(?:\)|\])', re.IGNORECASE),
    # Scale ranges: (grade (0-5)) or (scale 0-5) or (0-5)
    re.compile(r'(?:\(|\[)\s*(?:grade|scale)?\s*\(?[0-9]+(?:\.[0-9]+)?\s*(?:-|–|—|to)\s*[0-9]+(?:\.[0-9]+)?\)?\s*(?:\)|\])', re.IGNORECASE),
]

# Patterns representing illegible, obscured, or cancelled laboratory results
UNREADABLE_PATTERNS = [
    re.compile(r'(?:\*{2,}|\-{2,}|#{2,}|\?{2,}|\bN/A\b|\bNA\b|\bNULL\b|\bERROR\b|\bUNREADABLE\b|\bILLEGIBLE\b|\bHEMOLYZED\b|\bCLOTTED\b|\bINSUFFICIENT\b)', re.IGNORECASE),
    re.compile(r'\[(?:unreadable|obscured|illegible|error)\]', re.IGNORECASE)
]

# Common non-CKD tests in routine panels
UNSUPPORTED_TEST_DEFINITIONS = {
    "Platelet Count": re.compile(r'\b(?:Platelets?(?:\s*Count)?|PLT)\b', re.IGNORECASE),
    "Total Bilirubin": re.compile(r'\b(?:Total\s*Bilirubin|T\.?\s*Bilirubin|T\.?\s*Bili)\b', re.IGNORECASE),
    "Direct Bilirubin": re.compile(r'\b(?:Direct\s*Bilirubin|D\.?\s*Bilirubin|D\.?\s*Bili)\b', re.IGNORECASE),
    "SGOT / AST": re.compile(r'\b(?:SGOT|AST|Aspartate\s*Aminotransferase)\b', re.IGNORECASE),
    "SGPT / ALT": re.compile(r'\b(?:SGPT|ALT|Alanine\s*Aminotransferase)\b', re.IGNORECASE),
    "Alkaline Phosphatase": re.compile(r'\b(?:Alkaline\s*Phosphatase|ALP)\b', re.IGNORECASE),
    "Total Cholesterol": re.compile(r'\b(?:Total\s*Cholesterol|S\.?\s*Cholesterol)\b', re.IGNORECASE),
    "Triglycerides": re.compile(r'\b(?:Triglycerides|TGL)\b', re.IGNORECASE),
    "HDL Cholesterol": re.compile(r'\b(?:HDL(?:\s*Cholesterol)?)\b', re.IGNORECASE),
    "LDL Cholesterol": re.compile(r'\b(?:LDL(?:\s*Cholesterol)?)\b', re.IGNORECASE),
    "Serum Uric Acid": re.compile(r'\b(?:Uric\s*Acid|S\.?\s*Uric\s*Acid)\b', re.IGNORECASE),
    "Thyroid Stimulating Hormone (TSH)": re.compile(r'\b(?:TSH|Thyroid\s*Stimulating\s*Hormone)\b', re.IGNORECASE),
    "Total Calcium": re.compile(r'\b(?:Total\s*Calcium|Serum\s*Calcium|S\.?\s*Calcium)\b', re.IGNORECASE),
    "Serum Phosphorus": re.compile(r'\b(?:Serum\s*Phosphorus|S\.?\s*Phosphorus|Inorganic\s*Phosphorus)\b', re.IGNORECASE),
    "Erythrocyte Sedimentation Rate (ESR)": re.compile(r'\b(?:ESR|Erythrocyte\s*Sedimentation\s*Rate)\b', re.IGNORECASE),
}


# ============================================================================
# Clinical Parameter Feature Definitions
# ============================================================================

FEATURE_DEFINITIONS = {
    # ------------------ 14 Numerical Features ------------------
    "age": {
        "label": "Age",
        "type": "numerical",
        "canonical_unit": "years",
        "aliases": [
            re.compile(r'\bAge\s*\((?:age|years?)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Patient\s*)?Age\s*(?:/|\s+and\s+)?(?:Sex|Gender)?\b', re.IGNORECASE),
            re.compile(r'\bAge\b', re.IGNORECASE),
            re.compile(r'\bY(?:ears?)?\s*Old\b', re.IGNORECASE),
            re.compile(r'\bY/?O\b', re.IGNORECASE),
        ],
        "min": 1.0,
        "max": 120.0
    },
    "bp": {
        "label": "Blood Pressure",
        "type": "numerical",
        "canonical_unit": "mm/Hg",
        "aliases": [
            re.compile(r'\bBlood\s*Pressure\s*\((?:bp|mm/?hg)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Blood\s*Pressure|B\.?P\.?|NIBP)\b', re.IGNORECASE),
        ],
        "min": 40.0,
        "max": 250.0
    },
    "sg": {
        "label": "Specific Gravity",
        "type": "numerical",
        "canonical_unit": "density",
        "aliases": [
            re.compile(r'\bSpecific\s*Gravity\s*\((?:sg|density)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Sp(?:ecific)?\.?\s*Gravity|S\.?G\.?|Urine\s*Specific\s*Gravity|U_SG)\b', re.IGNORECASE),
        ],
        "min": 1.000,
        "max": 1.040
    },
    "al": {
        "label": "Albumin",
        "type": "numerical",
        "canonical_unit": "grade (0-5)",
        "aliases": [
            re.compile(r'\bAlbumin\s*\((?:al|urine|dipstick)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Urine\s*Albumin|Albumin\s*\(Urine\)|Urine\s*Protein|Protein\s*\(Urine\)|U_Albumin|Albumin\s*\(Dipstick\)|Protein\s*\(Dipstick\))\b', re.IGNORECASE),
            re.compile(r'\bAlbumin\b', re.IGNORECASE),
            re.compile(r'\bProtein\b', re.IGNORECASE),
        ],
        "min": 0.0,
        "max": 5.0
    },
    "su": {
        "label": "Sugar",
        "type": "numerical",
        "canonical_unit": "grade (0-5)",
        "aliases": [
            re.compile(r'\bSugar\s*\((?:su|urine|dipstick)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Urine\s*Sugar|Sugar\s*\(Urine\)|Urine\s*Glucose|Glucose\s*\(Urine\)|U_Sugar|Sugar\s*\(Dipstick\))\b', re.IGNORECASE),
            re.compile(r'\bUrine\s*Sugar\b', re.IGNORECASE),
            re.compile(r'\bSugar\b', re.IGNORECASE),
        ],
        "min": 0.0,
        "max": 5.0
    },
    "bgr": {
        "label": "Blood Glucose Random",
        "type": "numerical",
        "canonical_unit": "mg/dL",
        "aliases": [
            re.compile(r'\bBlood\s*Glucose\s*Random\s*\((?:bgr|mg/?dl)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Random\s*Blood\s*Sugar|Blood\s*Glucose\s*Random|RBS|BGR|Blood\s*Sugar\s*\(Random\)|Glucose,\s*Random|Random\s*Plasma\s*Glucose)\b', re.IGNORECASE),
            re.compile(r'\b(?:Fasting\s*Blood\s*Sugar|FBS|Fasting\s*Plasma\s*Glucose)\b', re.IGNORECASE),
            re.compile(r'\b(?:Post\s*Prandial\s*Blood\s*Sugar|PPBS)\b', re.IGNORECASE),
            re.compile(r'\b(?:Blood\s*Glucose|Plasma\s*Glucose|Blood\s*Sugar)\b', re.IGNORECASE),
        ],
        "min": 20.0,
        "max": 600.0
    },
    "bu": {
        "label": "Blood Urea",
        "type": "numerical",
        "canonical_unit": "mg/dL",
        "aliases": [
            re.compile(r'\bBlood\s*Urea\s*\((?:bu|mg/?dl)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Blood\s*Urea\s*Nitrogen|BUN)\b', re.IGNORECASE),
            re.compile(r'\b(?:Blood\s*Urea|Serum\s*Urea|B\.?\s*Urea|Urea)\b', re.IGNORECASE),
        ],
        "min": 1.0,
        "max": 500.0
    },
    "sc": {
        "label": "Serum Creatinine",
        "type": "numerical",
        "canonical_unit": "mg/dL",
        "aliases": [
            re.compile(r'\bSerum\s*Creatinine\s*\((?:sc|mg/?dl)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Serum\s*Creatinine|S\.?\s*Creatinine|Creatinine,\s*Serum|S\.?\s*Creat)\b', re.IGNORECASE),
            re.compile(r'\bCreatinine\b', re.IGNORECASE),
        ],
        "min": 0.1,
        "max": 100.0
    },
    "sod": {
        "label": "Serum Sodium",
        "type": "numerical",
        "canonical_unit": "mEq/L",
        "aliases": [
            re.compile(r'\bSerum\s*Sodium\s*\((?:sod|meq/?l)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Serum\s*Sodium|S\.?\s*Sodium|Sodium\s*\(Na\+?\)|Serum\s*Na\+?)\b', re.IGNORECASE),
            re.compile(r'\bSodium\b', re.IGNORECASE),
            re.compile(r'\bNa\+\b', re.IGNORECASE),
        ],
        "min": 50.0,
        "max": 200.0
    },
    "pot": {
        "label": "Serum Potassium",
        "type": "numerical",
        "canonical_unit": "mEq/L",
        "aliases": [
            re.compile(r'\bSerum\s*Potassium\s*\((?:pot|meq/?l)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Serum\s*Potassium|S\.?\s*Potassium|Potassium\s*\(K\+?\)|Serum\s*K\+?)\b', re.IGNORECASE),
            re.compile(r'\bPotassium\b', re.IGNORECASE),
            re.compile(r'\bK\+\b', re.IGNORECASE),
        ],
        "min": 1.0,
        "max": 60.0
    },
    "hemo": {
        "label": "Hemoglobin",
        "type": "numerical",
        "canonical_unit": "g/dL",
        "aliases": [
            re.compile(r'\bHa?emoglobin\s*\((?:hemo|hb|g/?dl)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Ha?emoglobin|Hb%?|Hgb)\b', re.IGNORECASE),
        ],
        "min": 2.0,
        "max": 25.0
    },
    "pcv": {
        "label": "Packed Cell Volume",
        "type": "numerical",
        "canonical_unit": "%",
        "aliases": [
            re.compile(r'\bPacked\s*Cell\s*Volume\s*\((?:pcv|%)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Packed\s*Cell\s*Volume|PCV|Ha?ematocrit|HCT)\b', re.IGNORECASE),
        ],
        "min": 5.0,
        "max": 70.0
    },
    "wbcc": {
        "label": "White Blood Cell Count",
        "type": "numerical",
        "canonical_unit": "cells/cumm",
        "aliases": [
            re.compile(r'\bWhite\s*Blood\s*Cell\s*Count\s*\((?:wbcc|cells/?cumm)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:White\s*Blood\s*Cells?(?:\s*Count)?|WBC(?:\s*Count)?|Total\s*Leuko?cyte\s*Count|TLC|W\.?B\.?C\.?)\b', re.IGNORECASE),
            re.compile(r'\bLeuko?cytes\b', re.IGNORECASE),
        ],
        "min": 500.0,
        "max": 50000.0
    },
    "rbcc": {
        "label": "Red Blood Cell Count",
        "type": "numerical",
        "canonical_unit": "millions/cmm",
        "aliases": [
            re.compile(r'\bRed\s*Blood\s*Cell\s*Count\s*\((?:rbcc|millions/?cmm)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Red\s*Blood\s*Cells?(?:\s*Count)?|RBC(?:\s*Count)?|Total\s*RBC(?:\s*Count)?|Total\s*Erythrocyte\s*Count|R\.?B\.?C\.?)\b', re.IGNORECASE),
            re.compile(r'\bErythrocytes\b', re.IGNORECASE),
        ],
        "min": 0.5,
        "max": 12.0
    },

    # ------------------ 10 Categorical Features ------------------
    "rbc": {
        "label": "Red Blood Cells",
        "type": "categorical",
        "canonical_unit": "categorical",
        "options": ["normal", "abnormal"],
        "aliases": [
            re.compile(r'\bRed\s*Blood\s*Cells\s*\((?:rbc)\)\b', re.IGNORECASE),
            re.compile(r'\bRed\s*Blood\s*Cells\b(?!\s*Count)', re.IGNORECASE),
            re.compile(r'\b(?:Urine\s*RBC|RBC\s*\(Urine\)|RBCs\s*in\s*Urine|Microscopic\s*RBC|Urine\s*Microscopy[:\s]*RBC)\b', re.IGNORECASE),
        ]
    },
    "pc": {
        "label": "Pus Cell",
        "type": "categorical",
        "canonical_unit": "categorical",
        "options": ["normal", "abnormal"],
        "aliases": [
            re.compile(r'\bPus\s*Cells?\s*\((?:pc)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Pus\s*Cells?(?!\s*Clumps)(?:\s*\(Urine\))?|Urine\s*Pus\s*Cells?|WBC\s*\(Urine\)|Urine\s*WBC|Leuko?cytes\s*\(Urine\))\b', re.IGNORECASE),
        ]
    },
    "pcc": {
        "label": "Pus Cell Clumps",
        "type": "categorical",
        "canonical_unit": "categorical",
        "options": ["notpresent", "present"],
        "aliases": [
            re.compile(r'\bPus\s*(?:Cell\s*)?Clumps\s*\((?:pcc)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Pus\s*(?:Cell\s*)?Clumps|Leuko?cyte\s*Clumps|WBC\s*Clumps)\b', re.IGNORECASE),
        ]
    },
    "ba": {
        "label": "Bacteria",
        "type": "categorical",
        "canonical_unit": "categorical",
        "options": ["notpresent", "present"],
        "aliases": [
            re.compile(r'\bBacteria\s*\((?:ba)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Bacteria\s*\(Urine\)|Urine\s*Bacteria|Bacteriuria)\b', re.IGNORECASE),
            re.compile(r'\bBacteria\b', re.IGNORECASE),
        ]
    },
    "htn": {
        "label": "Hypertension",
        "type": "categorical",
        "canonical_unit": "categorical",
        "options": ["no", "yes"],
        "aliases": [
            re.compile(r'\bHypertension\s*\((?:htn)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Hypertension|HTN|High\s*Blood\s*Pressure\s*History|History\s*of\s*Hypertension|Hypertensive\s*Status)\b', re.IGNORECASE),
        ]
    },
    "dm": {
        "label": "Diabetes Mellitus",
        "type": "categorical",
        "canonical_unit": "categorical",
        "options": ["no", "yes"],
        "aliases": [
            re.compile(r'\bDiabetes\s*Mellitus\s*\((?:dm)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Diabetes\s*Mellitus|Diabetes|DM|T2DM|Type\s*2\s*Diabetes|Diabetic\s*Status|History\s*of\s*Diabetes)\b', re.IGNORECASE),
        ]
    },
    "cad": {
        "label": "Coronary Artery Disease",
        "type": "categorical",
        "canonical_unit": "categorical",
        "options": ["no", "yes"],
        "aliases": [
            re.compile(r'\bCoronary\s*Artery\s*Disease\s*\((?:cad)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Coronary\s*Artery\s*Disease|CAD|Ische?mic\s*Heart\s*Disease|IHD|History\s*of\s*CAD)\b', re.IGNORECASE),
        ]
    },
    "appet": {
        "label": "Appetite",
        "type": "categorical",
        "canonical_unit": "categorical",
        "options": ["good", "poor"],
        "aliases": [
            re.compile(r'\bAppetite\s*\((?:appet)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Appetite|Patient\s*Appetite|Dietary\s*Intake)\b', re.IGNORECASE),
        ]
    },
    "pe": {
        "label": "Pedal Edema",
        "type": "categorical",
        "canonical_unit": "categorical",
        "options": ["no", "yes"],
        "aliases": [
            re.compile(r'\bPedal\s*O?edema\s*\((?:pe)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Pedal\s*O?edema|Ankle\s*Swelling|Foot\s*Swelling|Leg\s*Swelling|Peripheral\s*O?edema|O?edema\s*\(Pedal\))\b', re.IGNORECASE),
        ]
    },
    "ane": {
        "label": "Anemia",
        "type": "categorical",
        "canonical_unit": "categorical",
        "options": ["no", "yes"],
        "aliases": [
            re.compile(r'\bAna?emia\s*\((?:ane)\)\b', re.IGNORECASE),
            re.compile(r'\b(?:Ana?emia|Clinical\s*Ana?emia|History\s*of\s*Ana?emia|Pallor)\b', re.IGNORECASE),
        ]
    }
}


# ============================================================================
# Text Preprocessing & Line Sanitization
# ============================================================================

def split_text_by_pages(text: str) -> List[Tuple[int, str]]:
    """
    Splits extracted document text into individual pages based on '--- Page X ---' markers.
    If no markers exist, returns [(1, full_text)].
    """
    page_marker = re.compile(r'---\s*Page\s*(\d+)\s*---', re.IGNORECASE)
    splits = page_marker.split(text)

    # If no markers were found, splits has 1 element
    if len(splits) == 1:
        return [(1, text)]

    # If markers found, splits = ['', '1', 'page 1 text', '2', 'page 2 text', ...]
    pages: List[Tuple[int, str]] = []
    idx = 1
    while idx < len(splits):
        try:
            page_num = int(splits[idx])
            page_text = splits[idx + 1] if idx + 1 < len(splits) else ""
            pages.append((page_num, page_text))
            idx += 2
        except (ValueError, IndexError):
            idx += 1

    return pages if pages else [(1, text)]


def strip_reference_ranges(line: str) -> Tuple[str, List[str]]:
    """
    Identifies and strips reference ranges from a text line.
    Returns: (cleaned_line, list_of_detected_reference_ranges)
    """
    detected_ranges = []
    cleaned_line = line

    for pattern in REF_RANGE_PATTERNS:
        matches = list(pattern.finditer(cleaned_line))
        for m in reversed(matches):
            detected_ranges.append(m.group(0))
            # Replace the reference range clause with a neutral delimiter
            start, end = m.span()
            cleaned_line = cleaned_line[:start] + " " + cleaned_line[end:]

    return cleaned_line, detected_ranges


def check_for_unreadable(text_snippet: str) -> bool:
    """Detects whether text contains unreadable/obscured indicator symbols."""
    for pattern in UNREADABLE_PATTERNS:
        if pattern.search(text_snippet):
            return True
    return False


# ============================================================================
# Value Parsers for Individual Parameters
# ============================================================================

def clean_numeric_string(raw_str: str) -> str:
    """
    Cleans and normalizes numerical strings from medical OCR text:
    - Normalizes spaced digits from OCR kerning (e.g. '1 1.2' -> '11.2', '8 100' -> '8100')
    - Normalizes spaced decimals (e.g. '13 . 5' -> '13.5')
    - Corrects OCR character confusions (e.g. 'l.01' or 'I.01' -> '1.01')
    - Distinguishes decimal comma ('13,5' -> '13.5') from thousands separator ('8,100' -> '8100')
    """
    s = raw_str.strip()
    s = re.sub(r'(?<=\b[0-9])\s+(?=[0-9]+(?:\.[0-9]+)?\b)', '', s)
    s = re.sub(r'(\d+)\s*[\.]\s*(\d+)', r'\1.\2', s)
    s = re.sub(r'\b[lI](\.\d+)', r'1\1', s)
    s = re.sub(r'(\d+\.)[oO](\d*)', r'\g<1>0\2', s)
    if re.search(r'^\s*[-+]?\d+,\d{1,2}(?:[^\d]|$)', s):
        s = re.sub(r'(\d+),(\d{1,2})', r'\1.\2', s)
    s = re.sub(r'(\d+),(\d{3})', r'\1\2', s)
    s = s.replace(",", "")
    return s


def parse_numerical_value(
    feature: str,
    raw_str: str,
    unit_str: Optional[str] = None
) -> Tuple[Optional[float], Optional[str], List[str]]:
    """
    Parses a raw string into a float, applying unit conversion and plausibility checks.
    Returns: (parsed_value, canonical_unit, warnings)
    """
    warnings: List[str] = []
    clean_val = clean_numeric_string(raw_str)

    # Try extracting floating point number
    match = re.search(r'([0-9]+(?:\.[0-9]+)?)', clean_val)
    if not match:
        return None, None, ["No valid numerical value found."]

    val = float(match.group(1))
    unit = (unit_str or "").strip().lower()

    # 1. Age
    if feature == "age":
        return val, "years", warnings

    # 2. Blood Pressure
    if feature == "bp":
        # Check if raw_str was a pair like 120/80
        pair_match = re.search(r'([0-9]{2,3})\s*/\s*([0-9]{2,3})', clean_val)
        if pair_match:
            systolic = float(pair_match.group(1))
            diastolic = float(pair_match.group(2))
            warnings.append(
                f"Extracted diastolic blood pressure {diastolic:.1f} mm/Hg from reading "
                f"'{systolic:.0f}/{diastolic:.0f}' (systolic: {systolic:.0f} mm/Hg)."
            )
            return diastolic, "mm/Hg", warnings
        return val, "mm/Hg", warnings

    # 3. Specific Gravity
    if feature == "sg":
        # Handle integer shorthand e.g. 1020 -> 1.020
        if val >= 1000.0:
            converted = val / 1000.0
            warnings.append(f"Converted integer specific gravity {val:.0f} to ratio {converted:.3f}.")
            return converted, "density", warnings
        return val, "density", warnings

    # 4. Urine Albumin (Dipstick)
    if feature == "al":
        # Numerical grade 0-5
        if 0.0 <= val <= 5.0:
            return val, "grade (0-5)", warnings
        warnings.append(f"Albumin value {val} is outside standard dipstick scale (0-5).")
        return val, "grade (0-5)", warnings

    # 5. Urine Sugar (Dipstick)
    if feature == "su":
        if 0.0 <= val <= 5.0:
            return val, "grade (0-5)", warnings
        warnings.append(f"Sugar value {val} is outside standard dipstick scale (0-5).")
        return val, "grade (0-5)", warnings

    # 6. Blood Glucose Random (bgr)
    if feature == "bgr":
        if "mmol" in unit:
            converted = round(val * 18.018, 1)
            warnings.append(f"Converted blood glucose from mmol/L ({val}) to mg/dL ({converted}) using factor 18.018.")
            return converted, "mg/dL", warnings
        # Plausibility check: if value < 20 and no unit, could be mmol/L
        if val < 25.0 and ("mmol" in unit or unit == ""):
            converted = round(val * 18.018, 1)
            warnings.append(f"Small glucose value ({val}) assumed to be in mmol/L; converted to {converted} mg/dL. Review advised.")
            return converted, "mg/dL", warnings
        return val, "mg/dL", warnings

    # 7. Blood Urea (bu)
    if feature == "bu":
        if "mmol" in unit:
            converted = round(val * 6.006, 1)
            warnings.append(f"Converted blood urea from mmol/L ({val}) to mg/dL ({converted}) using factor 6.006.")
            return converted, "mg/dL", warnings
        return val, "mg/dL", warnings

    # 8. Serum Creatinine (sc)
    if feature == "sc":
        if "umol" in unit or "µmol" in unit or "micromol" in unit:
            converted = round(val / 88.42, 2)
            warnings.append(f"Converted serum creatinine from µmol/L ({val}) to mg/dL ({converted}) by dividing by 88.42.")
            return converted, "mg/dL", warnings
        # Plausibility check: creatinine > 40 with no unit is likely umol/L
        if val > 40.0 and unit == "":
            converted = round(val / 88.42, 2)
            warnings.append(f"Large creatinine value ({val}) assumed to be in µmol/L; converted to {converted} mg/dL. Review advised.")
            return converted, "mg/dL", warnings
        return val, "mg/dL", warnings

    # 9. Serum Sodium (sod)
    if feature == "sod":
        # mEq/L and mmol/L are identical for sodium
        return val, "mEq/L", warnings

    # 10. Serum Potassium (pot)
    if feature == "pot":
        # mEq/L and mmol/L are identical for potassium
        return val, "mEq/L", warnings

    # 11. Hemoglobin (hemo)
    if feature == "hemo":
        if "g/l" in unit or unit == "g/l":
            converted = round(val / 10.0, 1)
            warnings.append(f"Converted hemoglobin from g/L ({val}) to g/dL ({converted}) by dividing by 10.")
            return converted, "g/dL", warnings
        if "mmol" in unit:
            converted = round(val * 1.611, 1)
            warnings.append(f"Converted hemoglobin from mmol/L ({val}) to g/dL ({converted}) using factor 1.611.")
            return converted, "g/dL", warnings
        # Plausibility: if > 30 and no unit, likely g/L
        if val > 30.0 and unit == "":
            converted = round(val / 10.0, 1)
            warnings.append(f"Hemoglobin value ({val}) assumed to be in g/L; converted to {converted} g/dL. Review advised.")
            return converted, "g/dL", warnings
        return val, "g/dL", warnings

    # 12. Packed Cell Volume (pcv)
    if feature == "pcv":
        # If expressed as decimal ratio e.g. 0.42 -> 42.0%
        if 0.0 < val < 1.0:
            converted = round(val * 100.0, 1)
            warnings.append(f"Converted decimal hematocrit ratio ({val}) to percentage ({converted}%).")
            return converted, "%", warnings
        return val, "%", warnings

    # 13. White Blood Cell Count (wbcc)
    if feature == "wbcc":
        if "10^3" in unit or "k/" in unit or "thou" in unit or "x10" in unit:
            converted = round(val * 1000.0)
            warnings.append(f"Converted WBC count from thousands ({val} x10^3/µL) to cells/cumm ({converted}).")
            return float(converted), "cells/cumm", warnings
        # Plausibility: if value < 50, likely in thousands
        if val < 50.0:
            converted = round(val * 1000.0)
            warnings.append(f"WBC count ({val}) inferred to be in thousands; converted to {converted} cells/cumm. Review advised.")
            return float(converted), "cells/cumm", warnings
        return val, "cells/cumm", warnings

    # 14. Red Blood Cell Count (rbcc)
    if feature == "rbcc":
        # If absolute count given e.g. 5,200,000 /uL
        if val > 500000.0:
            converted = round(val / 1000000.0, 2)
            warnings.append(f"Converted RBC count from absolute cells/µL ({val}) to millions/cmm ({converted}).")
            return converted, "millions/cmm", warnings
        return val, "millions/cmm", warnings

    return val, unit_str, warnings


def parse_qualitative_dipstick(raw_str: str) -> Optional[float]:
    """
    Parses qualitative dipstick strings for Albumin ('al') or Sugar ('su').
    Returns float grade (0.0 - 5.0) or None.
    """
    s = raw_str.strip().lstrip(":=- ").strip().lower()

    # Negative / Nil
    if s in ["nil", "negative", "neg", "normal", "absent", "none", "0", "zero", "-", "not detected"]:
        return 0.0

    # Trace
    if s in ["trace", "tr", "traces"]:
        return 0.0

    # Plus signs or grades
    if s in ["+", "1+", "+1", "positive (+)", "present (+)", "positive", "present", "1"]:
        return 1.0
    if s in ["++", "2+", "+2", "positive (++)", "present (++)", "2"]:
        return 2.0
    if s in ["+++", "3+", "+3", "positive (+++)", "present (+++)", "3"]:
        return 3.0
    if s in ["++++", "4+", "+4", "positive (++++)", "present (++++)", "4"]:
        return 4.0
    if s in ["+++++", "5+", "+5", "5"]:
        return 5.0

    # Quantitative dipstick protein approx
    if "30" in s and "mg" in s:
        return 1.0
    if "100" in s and "mg" in s:
        return 2.0
    if "300" in s and "mg" in s:
        return 3.0
    if "1000" in s and "mg" in s:
        return 4.0

    return None


def parse_categorical_value(feature: str, raw_str: str) -> Tuple[Optional[str], List[str]]:
    """
    Maps raw string tokens to canonical categorical options.
    Returns: (mapped_option, warnings)
    """
    warnings: List[str] = []
    s = raw_str.strip().lstrip(":=- ").strip().lower()
    # Strip common prefixes from candidate line (e.g. "Patient: 4.8", "Result: yes")
    s = re.sub(r'^(?:patient|result|value|reading|observed|status)\s*[:=-]?\s*', '', s, flags=re.IGNORECASE).strip()

    # 1. rbc (urine red blood cells: normal / abnormal)
    if feature == "rbc":
        if any(term in s for term in ["plenty", "numerous", "loaded", "gross", "many", "abnormal", "packed", "positive"]):
            return "abnormal", warnings
        if any(term in s for term in ["nil", "normal", "occasional", "rare", "negative", "absent", "none"]):
            return "normal", warnings
        # HPF ranges e.g. 0-2 /HPF is normal, >3 is abnormal
        range_match = re.search(r'([0-9]+)\s*-\s*([0-9]+)', s)
        if range_match:
            upper = int(range_match.group(2))
            if upper <= 3:
                return "normal", warnings
            else:
                warnings.append(f"Urine RBC microscopic count '{s}' mapped to abnormal (>3 /HPF).")
                return "abnormal", warnings

    # 2. pc (pus cells in urine: normal / abnormal)
    if feature == "pc":
        if any(term in s for term in ["plenty", "numerous", "loaded", "clumps", "many", "abnormal", "positive"]):
            return "abnormal", warnings
        if any(term in s for term in ["nil", "normal", "occasional", "few", "rare", "negative", "absent", "none"]):
            return "normal", warnings
        range_match = re.search(r'([0-9]+)\s*-\s*([0-9]+)', s)
        if range_match:
            upper = int(range_match.group(2))
            if upper <= 5:
                return "normal", warnings
            else:
                warnings.append(f"Urine pus cell microscopic count '{s}' mapped to abnormal (>5 /HPF).")
                return "abnormal", warnings

    # 3. pcc (pus cell clumps: notpresent / present)
    if feature == "pcc":
        if any(term in s for term in ["notpresent", "not present", "not-present", "absent", "nil", "none", "negative", "not seen"]) or re.search(r'\bno\b', s):
            return "notpresent", warnings
        if any(term in s for term in ["present", "seen", "positive", "detected"]) or re.search(r'\byes\b', s):
            return "present", warnings

    # 4. ba (bacteria: notpresent / present)
    if feature == "ba":
        if any(term in s for term in ["notpresent", "not present", "not-present", "absent", "nil", "none", "negative", "not seen"]) or re.search(r'\bno\b', s):
            return "notpresent", warnings
        if any(term in s for term in ["present", "seen", "plenty", "positive", "detected", "abundant"]) or re.search(r'\byes\b', s):
            return "present", warnings

    # 5. htn (hypertension: no / yes)
    if feature == "htn":
        if any(term in s for term in ["non-hypertensive", "absent", "negative", "nil", "false"]) or re.search(r'\bno\b', s):
            return "no", warnings
        if any(term in s for term in ["hypertensive", "present", "positive", "known", "diagnosed", "true"]) or re.search(r'\byes\b', s):
            return "yes", warnings

    # 6. dm (diabetes mellitus: no / yes)
    if feature == "dm":
        if any(term in s for term in ["non-diabetic", "absent", "negative", "nil", "false"]) or re.search(r'\bno\b', s):
            return "no", warnings
        if any(term in s for term in ["diabetic", "present", "positive", "known", "diagnosed", "true"]) or re.search(r'\byes\b', s):
            return "yes", warnings

    # 7. cad (coronary artery disease: no / yes)
    if feature == "cad":
        if any(term in s for term in ["absent", "negative", "nil", "false"]) or re.search(r'\bno\b', s):
            return "no", warnings
        if any(term in s for term in ["present", "positive", "known", "diagnosed", "ihd", "true"]) or re.search(r'\byes\b', s):
            return "yes", warnings

    # 8. appet (appetite: good / poor)
    if feature == "appet":
        if any(term in s for term in ["poor", "reduced", "loss", "anorexia", "decreased", "low"]):
            return "poor", warnings
        if any(term in s for term in ["good", "normal", "adequate", "fair", "well", "fine"]):
            return "good", warnings

    # 9. pe (pedal edema: no / yes)
    if feature == "pe":
        if any(term in s for term in ["absent", "nil", "none", "negative"]) or re.search(r'\bno\b', s):
            return "no", warnings
        if any(term in s for term in ["present", "positive", "bilateral", "pitting", "seen", "swelling"]) or re.search(r'\byes\b', s):
            return "yes", warnings

    # 10. ane (anemia: no / yes)
    if feature == "ane":
        if any(term in s for term in ["absent", "negative", "nil", "non-anemic", "false"]) or re.search(r'\bno\b', s):
            return "no", warnings
        if any(term in s for term in ["present", "positive", "diagnosed", "anemic", "pallor", "true"]) or re.search(r'\byes\b', s):
            return "yes", warnings

    return None, ["Could not map raw value to standard categorical options."]


# ============================================================================
# Main Extraction Engine
# ============================================================================

class ParameterExtractor:
    """
    Deterministic rule-based patient parameter extractor.
    Iterates through document text, identifies supported and unsupported laboratory
    and clinical parameters, validates against reference ranges, and structures observations.
    """

    def __init__(self):
        self.feature_defs = FEATURE_DEFINITIONS
        self.unsupported_defs = UNSUPPORTED_TEST_DEFINITIONS

    def extract_from_text(self, text: str) -> ExtractionSummary:
        """
        Executes parameter extraction on full document text with page awareness.
        """
        summary = ExtractionSummary()

        if not text or not text.strip():
            # If document has no text, all 24 parameters are missing
            for feat_name, feat_meta in self.feature_defs.items():
                summary.parameters[feat_name] = ExtractedParameter(
                    canonical_name=feat_name,
                    clinical_label=feat_meta["label"],
                    extracted_value=None,
                    raw_value_str=None,
                    unit=feat_meta["canonical_unit"],
                    original_text=None,
                    source_page=None,
                    status="missing",
                    warnings=["No extractable text found in document."],
                    is_valid_for_model=False
                )
            summary.missing_count = len(ALL_FEATURE_NAMES)
            return summary

        pages = split_text_by_pages(text)

        # Dictionary to track all candidate observations per feature
        # key: feature_name -> list of observation dicts
        observations: Dict[str, List[Dict[str, Any]]] = {feat: [] for feat in self.feature_defs}
        unsupported_found: List[UnsupportedParameter] = []

        # Process each page line-by-line
        for page_num, page_content in pages:
            lines = page_content.splitlines()

            for line_idx, raw_line in enumerate(lines):
                line = raw_line.strip()
                if not line:
                    continue

                # 1. Check for unsupported tests first to record context
                self._detect_unsupported_parameters(line, page_num, unsupported_found)

                # 2. Check for each of the 24 supported features
                for feat_name, feat_meta in self.feature_defs.items():
                    obs = self._try_extract_feature_from_line(
                        feat_name,
                        feat_meta,
                        line,
                        lines,
                        line_idx,
                        page_num
                    )
                    if obs:
                        observations[feat_name].append(obs)

        # 3. Consolidate and resolve observations for each of the 24 features
        self._consolidate_parameters(observations, summary)

        # 4. Attach unsupported parameters
        # Deduplicate unsupported by detected_name
        seen_unsupported = set()
        deduped_unsupported = []
        for u in unsupported_found:
            if u.detected_name not in seen_unsupported:
                seen_unsupported.add(u.detected_name)
                deduped_unsupported.append(u)

        summary.unsupported_parameters = deduped_unsupported
        summary.unsupported_detected_count = len(deduped_unsupported)

        return summary

    def _detect_unsupported_parameters(
        self,
        line: str,
        page_num: int,
        unsupported_found: List[UnsupportedParameter]
    ):
        """Identifies recognized laboratory tests not included in the CKD schema."""
        for test_name, pattern in self.unsupported_defs.items():
            if pattern.search(line):
                unsupported_found.append(
                    UnsupportedParameter(
                        detected_name=test_name,
                        raw_snippet=line[:100],
                        source_page=page_num
                    )
                )

    def _try_extract_feature_from_line(
        self,
        feat_name: str,
        feat_meta: Dict[str, Any],
        line: str,
        all_lines: List[str],
        line_idx: int,
        page_num: int
    ) -> Optional[Dict[str, Any]]:
        """
        Attempts to match a feature alias on the current line and extract its value.
        """
        # Match alias pattern
        matched_alias = None
        match_end = -1
        for alias_pattern in feat_meta["aliases"]:
            m = alias_pattern.search(line)
            if m:
                # Disambiguation: avoid matching "Albumin" when "Urine Albumin" exists,
                # or "Urea" when "Blood Urea Nitrogen" was matched
                # Check for negative lookaheads / context
                if feat_name == "al" and re.search(r'\b(?:Serum|S\.?|Blood)\s*Albumin\b', line, re.IGNORECASE):
                    # Serum albumin is liver/nutrition, not urine dipstick albumin
                    continue
                if feat_name == "su" and re.search(r'\b(?:Blood|Serum|Random|Fasting|PP|Post\s*Prandial)\s*(?:Blood\s*)?Sugar\b', line, re.IGNORECASE):
                    # Blood sugar is bgr, not urine dipstick sugar
                    continue
                if feat_name == "rbc" and re.search(r'\bRed\s*Blood\s*Cells?\s*Count\b', line, re.IGNORECASE):
                    # Red blood cell count is rbcc, not urine microscopic rbc
                    continue
                if feat_name == "bu" and "nitrogen" in line.lower():
                    # Handle BUN specifically under 'bu' with conversion
                    pass
                matched_alias = m.group(0)
                match_end = m.end()
                break

        if not matched_alias:
            return None

        # Look at the text following the matched alias
        remainder = line[match_end:].strip()
        matched_line_text = line
        detected_unit_override = None

        # Check for unreadable / obscured indicators immediately on the line
        if check_for_unreadable(remainder):
            return {
                "status": "unreadable",
                "value": None,
                "raw_str": remainder,
                "unit": feat_meta["canonical_unit"],
                "original_text": matched_line_text,
                "page": page_num,
                "warnings": ["Parameter value appears obscured, hemolyzed, or unreadable in source text. No value substituted."],
                "is_valid": False
            }

        # Determine if the current line actually contains a candidate value.
        # Laboratory reports frequently include abbreviations or units on the label line, e.g.:
        # "Age (age)", "Blood Pressure (bp)", "Serum Creatinine (sc)", "Hemoglobin (g/dL)"
        # where the actual measurement is on the subsequent line.
        clean_sub = re.sub(r'\s*\([^\)]*\)', '', remainder).strip().lstrip(":=- ").strip()
        has_val_on_line = False
        if feat_meta["type"] == "numerical":
            cleaned_num_probe = clean_numeric_string(clean_sub)
            if re.search(r'[0-9]', cleaned_num_probe):
                has_val_on_line = True
            elif feat_name in ["al", "su"] and parse_qualitative_dipstick(clean_sub) is not None:
                has_val_on_line = True
        else:
            cat_probe, _ = parse_categorical_value(feat_name, clean_sub)
            if cat_probe is not None:
                has_val_on_line = True

        # If current line lacks a candidate value, inspect subsequent lines (multi-line layout)
        if not has_val_on_line:
            candidate_next = None
            for offset in range(1, 4):
                if line_idx + offset >= len(all_lines):
                    break
                next_raw = all_lines[line_idx + offset].strip()
                if not next_raw:
                    continue

                # Stop if next line is another feature label
                is_other_feat = False
                for other_feat, other_meta in self.feature_defs.items():
                    for p in other_meta["aliases"]:
                        if p.search(next_raw):
                            if other_feat == "al" and re.search(r'\b(?:Serum|S\.?|Blood)\s*Albumin\b', next_raw, re.IGNORECASE):
                                continue
                            if other_feat == "rbc" and re.search(r'\bRed\s*Blood\s*Cells?\s*Count\b', next_raw, re.IGNORECASE):
                                continue
                            is_other_feat = True
                            break
                    if is_other_feat:
                        break
                if is_other_feat:
                    break

                # Stop if next line is an unsupported test
                if any(p.search(next_raw) for p in self.unsupported_defs.values()):
                    break

                # Stop if next line is a document section header
                if re.search(r'^(?:---|===|CLINICAL|PATIENT|EXPLAINABLE|AVAILABLE|MOST|BIOCHEMISTRY|HEMATOLOGY|URINALYSIS|TEST\s*NAME|PARAMETER)', next_raw, re.IGNORECASE):
                    break

                candidate_next = next_raw
                # Check if the line immediately following candidate_next contains a standalone unit
                if line_idx + offset + 1 < len(all_lines):
                    cand_unit_line = all_lines[line_idx + offset + 1].strip()
                    if re.match(r'^(?:mg/dl|g/dl|g/l|mmol/l|umol/l|µmol/l|meq/l|cells/cumm|millions/cmm|%|mm/hg|density|years)$', cand_unit_line, re.IGNORECASE):
                        detected_unit_override = cand_unit_line
                break

            if candidate_next:
                remainder = candidate_next
                matched_line_text = f"{line} -> {candidate_next}"
            else:
                return None

        # Check for unreadable / obscured indicators
        if check_for_unreadable(remainder):
            return {
                "status": "unreadable",
                "value": None,
                "raw_str": remainder,
                "unit": feat_meta["canonical_unit"],
                "original_text": matched_line_text,
                "page": page_num,
                "warnings": ["Parameter value appears obscured, hemolyzed, or unreadable in source text. No value substituted."],
                "is_valid": False
            }

        # Clean reference ranges from remainder to prevent capturing range boundaries
        cleaned_remainder, ref_ranges = strip_reference_ranges(remainder)

        # Detect unit string from remainder, label line, or detected_unit_override
        unit_match = re.search(r'\b(mg/dl|g/dl|g/l|mmol/l|umol/l|µmol/l|meq/l|cells/cumm|cells/ul|k/ul|thou/ul|millions/cmm|mil/cumm|%|mm/hg|mmhg)\b', remainder, re.IGNORECASE)
        if not unit_match:
            unit_match = re.search(r'\b(mg/dl|g/dl|g/l|mmol/l|umol/l|µmol/l|meq/l|cells/cumm|cells/ul|k/ul|thou/ul|millions/cmm|mil/cumm|%|mm/hg|mmhg)\b', line, re.IGNORECASE)
        detected_unit = detected_unit_override or (unit_match.group(1) if unit_match else None)

        if feat_meta["type"] == "numerical":
            # Qualitative dipstick check for al or su
            if feat_name in ["al", "su"]:
                qual_val = parse_qualitative_dipstick(cleaned_remainder)
                if qual_val is not None:
                    warnings = [f"Mapped qualitative dipstick result '{cleaned_remainder.strip()}' to grade {qual_val}."]
                    return {
                        "status": "extracted_requiring_review",
                        "value": qual_val,
                        "raw_str": cleaned_remainder.strip(),
                        "unit": "grade (0-5)",
                        "original_text": matched_line_text,
                        "page": page_num,
                        "warnings": warnings,
                        "is_valid": True
                    }

            # Numerical parsing
            # If blood pressure, look for pair like 120/80
            if feat_name == "bp":
                bp_match = re.search(r'([0-9]{2,3})\s*/\s*([0-9]{2,3})', cleaned_remainder)
                if bp_match:
                    parsed_val, canon_unit, warnings = parse_numerical_value("bp", bp_match.group(0), detected_unit)
                    return {
                        "status": "extracted_requiring_review" if warnings else "extracted_unambiguous",
                        "value": parsed_val,
                        "raw_str": bp_match.group(0),
                        "unit": canon_unit,
                        "original_text": matched_line_text,
                        "page": page_num,
                        "warnings": warnings,
                        "is_valid": parsed_val is not None
                    }

            # Standard numerical match: prioritize the first number immediately following the test label or candidate
            cleaned_rem_stripped = re.sub(r'^(?:patient|result|value|reading|observed|status)\s*[:=-]?\s*', '', cleaned_remainder, flags=re.IGNORECASE).strip()
            cleaned_rem_num = clean_numeric_string(cleaned_rem_stripped)
            num_match = re.search(r'^\s*[:=\s\-]?\s*([0-9]+(?:\.[0-9]+)?)', cleaned_rem_num)
            if not num_match:
                num_match = re.search(r'[:=\s]\s*([0-9]+(?:\.[0-9]+)?)', cleaned_rem_num)
            if not num_match:
                num_match = re.search(r'([0-9]+(?:\.[0-9]+)?)', cleaned_rem_num)

            if num_match:
                raw_num_str = num_match.group(1)
                parsed_val, canon_unit, warnings = parse_numerical_value(feat_name, raw_num_str, detected_unit)

                # Track whether unit conversion or other non-standard parsing occurred
                requires_review = bool(warnings)

                # Check if this parsed number accidentally matches reference range boundary
                # while another number was available
                if ref_ranges:
                    warnings.append(f"Laboratory reference range detected and bypassed: '{', '.join(ref_ranges)}'.")

                # Plausibility bounds check
                min_val = feat_meta.get("min", float("-inf"))
                max_val = feat_meta.get("max", float("inf"))

                if parsed_val is not None:
                    if parsed_val < min_val or parsed_val > max_val:
                        requires_review = True
                        warnings.append(
                            f"Value {parsed_val} {canon_unit} is outside standard clinical reference interval "
                            f"({min_val} to {max_val}). Flagged for clinician verification."
                        )

                status = "extracted_requiring_review" if requires_review else "extracted_unambiguous"

                return {
                    "status": status,
                    "value": parsed_val,
                    "raw_str": raw_num_str,
                    "unit": canon_unit or feat_meta["canonical_unit"],
                    "original_text": matched_line_text,
                    "page": page_num,
                    "warnings": warnings,
                    "is_valid": parsed_val is not None
                }

        elif feat_meta["type"] == "categorical":
            # Categorical matching
            cat_val, warnings = parse_categorical_value(feat_name, cleaned_remainder)
            if cat_val:
                return {
                    "status": "extracted_unambiguous" if not warnings else "extracted_requiring_review",
                    "value": cat_val,
                    "raw_str": cleaned_remainder.strip(),
                    "unit": "categorical",
                    "original_text": matched_line_text,
                    "page": page_num,
                    "warnings": warnings,
                    "is_valid": True
                }

        return None

    def _consolidate_parameters(
        self,
        observations: Dict[str, List[Dict[str, Any]]],
        summary: ExtractionSummary
    ):
        """
        Consolidates observations across pages, detects conflicts or duplicates,
        and assigns final parameter statuses.
        """
        for feat_name, feat_meta in self.feature_defs.items():
            obs_list = observations.get(feat_name, [])

            # Case A: Missing parameter
            if not obs_list:
                param = ExtractedParameter(
                    canonical_name=feat_name,
                    clinical_label=feat_meta["label"],
                    extracted_value=None,
                    raw_value_str=None,
                    unit=feat_meta.get("canonical_unit"),
                    original_text=None,
                    source_page=None,
                    status="missing",
                    warnings=[],
                    is_valid_for_model=False,
                    repeat_count=0
                )
                summary.parameters[feat_name] = param
                summary.missing_count += 1
                continue

            # Case B: Exactly one observation
            if len(obs_list) == 1:
                obs = obs_list[0]
                param = ExtractedParameter(
                    canonical_name=feat_name,
                    clinical_label=feat_meta["label"],
                    extracted_value=obs["value"],
                    raw_value_str=obs["raw_str"],
                    unit=obs["unit"],
                    original_text=obs["original_text"],
                    source_page=obs["page"],
                    status=obs["status"],
                    warnings=obs["warnings"],
                    is_valid_for_model=obs["is_valid"],
                    repeat_count=1,
                    all_observed_values=[obs]
                )
                summary.parameters[feat_name] = param
                self._increment_status_count(obs["status"], summary)
                continue

            # Case C: Multiple observations (check for duplicate vs conflicting)
            # Filter valid vs unreadable
            valid_obs = [o for o in obs_list if o.get("is_valid") and o.get("value") is not None]

            if not valid_obs:
                # All occurrences were unreadable
                first_obs = obs_list[0]
                param = ExtractedParameter(
                    canonical_name=feat_name,
                    clinical_label=feat_meta["label"],
                    extracted_value=None,
                    raw_value_str=first_obs["raw_str"],
                    unit=first_obs["unit"],
                    original_text=first_obs["original_text"],
                    source_page=first_obs["page"],
                    status="unreadable",
                    warnings=["All observed occurrences in the document were illegible or unreadable."],
                    is_valid_for_model=False,
                    repeat_count=len(obs_list),
                    all_observed_values=obs_list
                )
                summary.parameters[feat_name] = param
                summary.unreadable_count += 1
                continue

            # Compare observed values
            first_val = valid_obs[0]["value"]
            is_conflicting = False
            for other_obs in valid_obs[1:]:
                other_val = other_obs["value"]
                if feat_meta["type"] == "numerical":
                    # Numerical conflict: difference > 5% or > 0.2 absolute
                    if abs(first_val - other_val) > max(0.05 * abs(first_val), 0.2):
                        is_conflicting = True
                        break
                else:
                    # Categorical conflict: strings differ
                    if str(first_val).lower() != str(other_val).lower():
                        is_conflicting = True
                        break

            if is_conflicting:
                all_vals_str = ", ".join(
                    f"{o['value']} (Page {o['page']})" for o in valid_obs
                )
                param = ExtractedParameter(
                    canonical_name=feat_name,
                    clinical_label=feat_meta["label"],
                    extracted_value=None,  # Do not choose arbitrarily when conflicting
                    raw_value_str=str(first_val),
                    unit=valid_obs[0]["unit"],
                    original_text=valid_obs[0]["original_text"],
                    source_page=valid_obs[0]["page"],
                    status="conflicting",
                    warnings=[
                        f"Conflicting values detected across document occurrences: {all_vals_str}. "
                        f"Manual clinical review required."
                    ],
                    is_valid_for_model=False,
                    repeat_count=len(obs_list),
                    all_observed_values=obs_list
                )
                summary.parameters[feat_name] = param
                summary.conflicting_count += 1
            else:
                # Identical repeated values
                base_obs = valid_obs[0]
                merged_warnings = list(base_obs["warnings"])
                merged_warnings.append(
                    f"Parameter observed {len(valid_obs)} times across document with consistent values."
                )
                status = "extracted_unambiguous" if not base_obs["warnings"] else "extracted_requiring_review"
                param = ExtractedParameter(
                    canonical_name=feat_name,
                    clinical_label=feat_meta["label"],
                    extracted_value=base_obs["value"],
                    raw_value_str=base_obs["raw_str"],
                    unit=base_obs["unit"],
                    original_text=base_obs["original_text"],
                    source_page=base_obs["page"],
                    status=status,
                    warnings=merged_warnings,
                    is_valid_for_model=True,
                    repeat_count=len(obs_list),
                    all_observed_values=obs_list
                )
                summary.parameters[feat_name] = param
                self._increment_status_count(status, summary)

    def _increment_status_count(self, status: str, summary: ExtractionSummary):
        """Updates aggregate extraction counts."""
        if status == "extracted_unambiguous":
            summary.extracted_unambiguous_count += 1
        elif status == "extracted_requiring_review":
            summary.extracted_requiring_review_count += 1
        elif status == "conflicting":
            summary.conflicting_count += 1
        elif status == "unreadable":
            summary.unreadable_count += 1
        elif status == "missing":
            summary.missing_count += 1


# Global module singleton instance
parameter_extractor = ParameterExtractor()


def extract_parameters_from_text(text: str) -> ExtractionSummary:
    """Convenience functional wrapper for parameter extraction."""
    return parameter_extractor.extract_from_text(text)
