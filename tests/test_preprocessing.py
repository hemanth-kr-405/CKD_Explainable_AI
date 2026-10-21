"""
Unit tests for the CKD Preprocessing pipeline, verifying data integrity,
stratified train/test split, absence of data leakage, and reusable inference transformation.
"""

import os
import pytest
import pandas as pd
import numpy as np
import joblib

from src.config import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    PREPROCESSOR_ARTIFACT_PATH,
    PREPROCESSING_METADATA_PATH,
    ALL_FEATURE_NAMES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN
)
from src.preprocessing import CKDPreprocessor, load_raw_data, prepare_and_split_data


def test_raw_data_integrity():
    """Verify raw dataset exists and has authentic 400 rows and 25 columns."""
    assert os.path.exists(RAW_DATA_PATH), f"Raw dataset not found at {RAW_DATA_PATH}"
    df_raw = pd.read_csv(RAW_DATA_PATH)
    assert df_raw.shape == (400, 25), f"Unexpected shape {df_raw.shape}, expected (400, 25)"
    assert TARGET_COLUMN in df_raw.columns, f"Target column '{TARGET_COLUMN}' missing from raw data"


def test_preprocessor_fit_learns_valid_statistics():
    """Verify preprocessor calculates medians and modes on training set without NaNs."""
    df_raw = load_raw_data()
    X = df_raw[ALL_FEATURE_NAMES]
    
    preprocessor = CKDPreprocessor()
    preprocessor.fit(X.iloc[:320])  # Simulate training subset
    
    assert preprocessor.is_fitted_ is True
    # Check numerical medians
    for feat in NUMERICAL_FEATURES:
        assert feat in preprocessor.numerical_medians_
        assert not np.isnan(preprocessor.numerical_medians_[feat])
        assert isinstance(preprocessor.numerical_medians_[feat], float)
        
    # Check categorical modes
    for feat in CATEGORICAL_FEATURES:
        assert feat in preprocessor.categorical_modes_
        assert isinstance(preprocessor.categorical_modes_[feat], str)
        assert preprocessor.categorical_modes_[feat] in ["normal", "abnormal", "present", "notpresent", "yes", "no", "good", "poor"]


def test_transform_produces_no_nans_and_preserves_order():
    """Verify transform eliminates all NaNs and strictly enforces feature order."""
    df_raw = load_raw_data()
    X = df_raw[ALL_FEATURE_NAMES]
    
    preprocessor = CKDPreprocessor()
    preprocessor.fit(X.iloc[:320])
    
    transformed_train = preprocessor.transform(X.iloc[:320])
    transformed_test = preprocessor.transform(X.iloc[320:])
    
    # Assert column ordering
    assert list(transformed_train.columns) == ALL_FEATURE_NAMES
    assert list(transformed_test.columns) == ALL_FEATURE_NAMES
    
    # Assert 0 missing values
    assert transformed_train.isna().sum().sum() == 0
    assert transformed_test.isna().sum().sum() == 0
    
    # Assert correct row counts
    assert len(transformed_train) == 320
    assert len(transformed_test) == 80


def test_stratified_train_test_split():
    """Verify that train and test splits maintain identical class ratios (stratification)."""
    assert os.path.exists(TRAIN_DATA_PATH)
    assert os.path.exists(TEST_DATA_PATH)
    
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    assert len(train_df) == 320
    assert len(test_df) == 80
    assert TARGET_COLUMN in train_df.columns
    assert TARGET_COLUMN in test_df.columns
    
    # Expected: 62.5% CKD (1) and 37.5% Non-CKD (0)
    train_ckd_ratio = (train_df[TARGET_COLUMN] == 1).mean()
    test_ckd_ratio = (test_df[TARGET_COLUMN] == 1).mean()
    
    assert pytest.approx(train_ckd_ratio, rel=1e-2) == 0.625
    assert pytest.approx(test_ckd_ratio, rel=1e-2) == 0.625


def test_preprocessor_artifact_serialization():
    """Verify preprocessor joblib artifact loads and transforms single records accurately."""
    assert os.path.exists(PREPROCESSOR_ARTIFACT_PATH)
    loaded_preprocessor: CKDPreprocessor = joblib.load(PREPROCESSOR_ARTIFACT_PATH)
    
    assert loaded_preprocessor.is_fitted_ is True
    assert len(loaded_preprocessor.numerical_medians_) == len(NUMERICAL_FEATURES)
    assert len(loaded_preprocessor.categorical_modes_) == len(CATEGORICAL_FEATURES)
    
    # Test single clinical payload (such as from FastAPI /predict endpoint)
    sample_patient = {
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
        "ane": "no"
    }
    
    transformed_single = loaded_preprocessor.transform_single(sample_patient)
    assert isinstance(transformed_single, pd.DataFrame)
    assert transformed_single.shape == (1, 24)
    assert list(transformed_single.columns) == ALL_FEATURE_NAMES
    assert transformed_single.isna().sum().sum() == 0
    # Categorical verification: 'htn' yes -> 1, 'cad' no -> 0
    assert transformed_single.loc[0, "htn"] == 1
    assert transformed_single.loc[0, "cad"] == 0


def test_single_record_with_missing_values():
    """Verify preprocessor handles incomplete patient inputs gracefully using learned medians/modes."""
    loaded_preprocessor: CKDPreprocessor = joblib.load(PREPROCESSOR_ARTIFACT_PATH)
    
    # Patient provides only a subset of measurements
    partial_patient = {
        "age": 60.0,
        "bp": 90.0,
        "sc": 2.5,
        "htn": "yes"
    }
    
    transformed = loaded_preprocessor.transform_single(partial_patient)
    assert transformed.shape == (1, 24)
    assert transformed.isna().sum().sum() == 0
    # Imputed features should match learned training statistics
    assert transformed.loc[0, "hemo"] == loaded_preprocessor.numerical_medians_["hemo"]
    assert transformed.loc[0, "dm"] == loaded_preprocessor.categorical_mappings["dm"][loaded_preprocessor.categorical_modes_["dm"]]
