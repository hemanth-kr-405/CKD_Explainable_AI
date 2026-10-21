"""
Configuration file containing paths, feature definitions, data types,
and clinical metadata for the Chronic Kidney Disease (CKD) project.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "chronic_kidney_disease.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
PREPROCESSING_METADATA_PATH = PROCESSED_DATA_DIR / "preprocessing_metadata.json"

MODELS_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_ARTIFACT_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_ARTIFACT_PATH = MODELS_DIR / "ckd_model.joblib"
METRICS_ARTIFACT_PATH = MODELS_DIR / "metrics.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.json"

# UCI Dataset URL
UCI_CKD_URL = "https://archive.ics.uci.edu/static/public/336/data.csv"

# Target Definition
TARGET_COLUMN = "class"
TARGET_MAPPING = {
    "ckd": 1,
    "notckd": 0
}
TARGET_NAMES = {
    1: "Chronic Kidney Disease (CKD)",
    0: "Non-CKD"
}

# Numerical Clinical Features (14 features)
NUMERICAL_FEATURES = [
    "age",    # Age in years
    "bp",     # Blood Pressure in mm/Hg
    "sg",     # Specific Gravity (1.005, 1.010, 1.015, 1.020, 1.025)
    "al",     # Albumin (0 to 5)
    "su",     # Sugar (0 to 5)
    "bgr",    # Blood Glucose Random (mg/dL)
    "bu",     # Blood Urea (mg/dL)
    "sc",     # Serum Creatinine (mg/dL)
    "sod",    # Sodium (mEq/L)
    "pot",    # Potassium (mEq/L)
    "hemo",   # Hemoglobin (g/dL)
    "pcv",    # Packed Cell Volume (%)
    "wbcc",   # White Blood Cell Count (cells/cumm)
    "rbcc"    # Red Blood Cell Count (millions/cmm)
]

# Categorical Clinical Features (10 features)
CATEGORICAL_FEATURES = [
    "rbc",    # Red Blood Cells (normal, abnormal)
    "pc",     # Pus Cell (normal, abnormal)
    "pcc",    # Pus Cell Clumps (notpresent, present)
    "ba",     # Bacteria (notpresent, present)
    "htn",    # Hypertension (yes, no)
    "dm",     # Diabetes Mellitus (yes, no)
    "cad",    # Coronary Artery Disease (yes, no)
    "appet",  # Appetite (good, poor)
    "pe",     # Pedal Edema (yes, no)
    "ane"     # Anemia (yes, no)
]

ALL_FEATURE_NAMES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Categorical Value Normalization & Encoding Maps
# In clinical modeling, 1 typically represents abnormal/present/risk, 0 represents normal/absent/healthy
CATEGORICAL_MAPPINGS = {
    "rbc": {"normal": 0, "abnormal": 1},
    "pc": {"normal": 0, "abnormal": 1},
    "pcc": {"notpresent": 0, "present": 1},
    "ba": {"notpresent": 0, "present": 1},
    "htn": {"no": 0, "yes": 1},
    "dm": {"no": 0, "yes": 1},
    "cad": {"no": 0, "yes": 1},
    "appet": {"good": 0, "poor": 1},
    "pe": {"no": 0, "yes": 1},
    "ane": {"no": 0, "yes": 1}
}

# Clinical Descriptions and Units for Human-Understandable Explanations
FEATURE_METADATA = {
    "age": {"label": "Age", "unit": "years", "type": "numerical", "min": 1, "max": 100},
    "bp": {"label": "Blood Pressure", "unit": "mm/Hg", "type": "numerical", "min": 50, "max": 200},
    "sg": {"label": "Specific Gravity", "unit": "density", "type": "numerical", "min": 1.000, "max": 1.035},
    "al": {"label": "Albumin", "unit": "grade (0-5)", "type": "numerical", "min": 0, "max": 5},
    "su": {"label": "Sugar", "unit": "grade (0-5)", "type": "numerical", "min": 0, "max": 5},
    "bgr": {"label": "Blood Glucose Random", "unit": "mg/dL", "type": "numerical", "min": 20, "max": 500},
    "bu": {"label": "Blood Urea", "unit": "mg/dL", "type": "numerical", "min": 1, "max": 400},
    "sc": {"label": "Serum Creatinine", "unit": "mg/dL", "type": "numerical", "min": 0.1, "max": 80.0},
    "sod": {"label": "Serum Sodium", "unit": "mEq/L", "type": "numerical", "min": 50, "max": 180},
    "pot": {"label": "Serum Potassium", "unit": "mEq/L", "type": "numerical", "min": 1.0, "max": 50.0},
    "hemo": {"label": "Hemoglobin", "unit": "g/dL", "type": "numerical", "min": 2.0, "max": 25.0},
    "pcv": {"label": "Packed Cell Volume", "unit": "%", "type": "numerical", "min": 5, "max": 65},
    "wbcc": {"label": "White Blood Cell Count", "unit": "cells/cumm", "type": "numerical", "min": 1000, "max": 35000},
    "rbcc": {"label": "Red Blood Cell Count", "unit": "millions/cmm", "type": "numerical", "min": 1.0, "max": 10.0},
    "rbc": {"label": "Red Blood Cells", "unit": "categorical", "type": "categorical", "options": ["normal", "abnormal"]},
    "pc": {"label": "Pus Cell", "unit": "categorical", "type": "categorical", "options": ["normal", "abnormal"]},
    "pcc": {"label": "Pus Cell Clumps", "unit": "categorical", "type": "categorical", "options": ["notpresent", "present"]},
    "ba": {"label": "Bacteria", "unit": "categorical", "type": "categorical", "options": ["notpresent", "present"]},
    "htn": {"label": "Hypertension", "unit": "categorical", "type": "categorical", "options": ["no", "yes"]},
    "dm": {"label": "Diabetes Mellitus", "unit": "categorical", "type": "categorical", "options": ["no", "yes"]},
    "cad": {"label": "Coronary Artery Disease", "unit": "categorical", "type": "categorical", "options": ["no", "yes"]},
    "appet": {"label": "Appetite", "unit": "categorical", "type": "categorical", "options": ["good", "poor"]},
    "pe": {"label": "Pedal Edema", "unit": "categorical", "type": "categorical", "options": ["no", "yes"]},
    "ane": {"label": "Anemia", "unit": "categorical", "type": "categorical", "options": ["no", "yes"]}
}

# Human-Understandable Feature Labels for SHAP Explanations
CLINICAL_FEATURE_LABELS = {
    feat: f"{meta['label']} ({meta['unit']})" if meta.get("unit") and meta["unit"] != "categorical" else meta["label"]
    for feat, meta in FEATURE_METADATA.items()
}

# Data Splitting Parameters
TEST_SIZE = 0.20
RANDOM_STATE = 42
