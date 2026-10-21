"""
Preprocessing pipeline for the Chronic Kidney Disease (CKD) dataset.

Provides:
- Data cleaning and standardization (whitespace trimming, tab stripping)
- Clinically sound missing value imputation (median for skewed numerics, mode for categoricals)
- Binary ordinal encoding preserving medical interpretability
- Prevention of data leakage by fitting exclusively on training splits
- Standalone transformer class usable for both batch training and single-record inference
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import joblib

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    PREPROCESSING_METADATA_PATH,
    MODELS_DIR,
    PREPROCESSOR_ARTIFACT_PATH,
    TARGET_COLUMN,
    TARGET_MAPPING,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURE_NAMES,
    CATEGORICAL_MAPPINGS,
    TEST_SIZE,
    RANDOM_STATE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CKDPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible preprocessor for Chronic Kidney Disease clinical data.
    Fits missing value statistics exclusively on training data to strictly prevent data leakage.
    """

    def __init__(self):
        self.numerical_features = list(NUMERICAL_FEATURES)
        self.categorical_features = list(CATEGORICAL_FEATURES)
        self.all_feature_names = list(ALL_FEATURE_NAMES)
        self.categorical_mappings = dict(CATEGORICAL_MAPPINGS)
        
        # Learned statistics (populated during fit)
        self.numerical_medians_: Dict[str, float] = {}
        self.categorical_modes_: Dict[str, str] = {}
        self.is_fitted_: bool = False

    def _clean_raw_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans tabs, trailing whitespace, and special characters from string columns.
        """
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype == object or pd.api.types.is_string_dtype(df_clean[col]):
                # Strip leading/trailing whitespace and tabs
                df_clean[col] = df_clean[col].astype(str).str.strip().replace({
                    "?": np.nan,
                    "nan": np.nan,
                    "NaN": np.nan,
                    "None": np.nan,
                    "": np.nan
                })
        return df_clean

    def fit(self, X: pd.DataFrame, y: Any = None):
        """
        Calculates and stores medians for numerical columns and modes for categorical columns
        using ONLY the provided training data X.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.all_feature_names)

        X_clean = self._clean_raw_strings(X)

        # 1. Learn Numerical Medians
        self.numerical_medians_ = {}
        for col in self.numerical_features:
            if col in X_clean.columns:
                series_num = pd.to_numeric(X_clean[col], errors="coerce")
                median_val = float(series_num.median())
                if np.isnan(median_val):
                    median_val = 0.0
                self.numerical_medians_[col] = round(median_val, 4)

        # 2. Learn Categorical Modes
        self.categorical_modes_ = {}
        for col in self.categorical_features:
            if col in X_clean.columns:
                mode_series = X_clean[col].dropna().mode()
                mode_val = mode_series.iloc[0] if not mode_series.empty else list(self.categorical_mappings[col].keys())[0]
                self.categorical_modes_[col] = str(mode_val)

        self.is_fitted_ = True
        logging.info("CKDPreprocessor successfully fitted on training data.")
        return self

    def transform(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Transforms input clinical records using the statistics learned during fit.
        Guarantees exact feature ordering, no NaNs, and numerical types.
        """
        if not self.is_fitted_:
            raise RuntimeError("The preprocessor must be fitted before calling transform().")

        if isinstance(X, dict):
            X_df = pd.DataFrame([X])
        elif isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.all_feature_names)

        X_clean = self._clean_raw_strings(X_df)

        # Ensure all expected columns exist (fill missing columns with NaN so imputer handles them)
        for col in self.all_feature_names:
            if col not in X_clean.columns:
                X_clean[col] = np.nan

        transformed = pd.DataFrame(index=X_clean.index)

        # 1. Transform Numerical Features
        for col in self.numerical_features:
            series_num = pd.to_numeric(X_clean[col], errors="coerce")
            # Impute missing values with learned training median
            imputed = series_num.fillna(self.numerical_medians_[col])
            transformed[col] = imputed.astype(float)

        # 2. Transform Categorical Features
        for col in self.categorical_features:
            cat_series = X_clean[col].astype(str).str.strip().replace({"nan": np.nan, "NaN": np.nan, "": np.nan})
            # Impute missing values with learned training mode
            imputed_cat = cat_series.fillna(self.categorical_modes_[col])
            
            # Map categories to binary values (0 or 1)
            mapping = self.categorical_mappings[col]
            mapped = imputed_cat.map(mapping)
            
            # Fallback for unexpected or unknown string values
            unmapped_mask = mapped.isna()
            if unmapped_mask.any():
                default_val = mapping[self.categorical_modes_[col]]
                mapped = mapped.fillna(default_val)
                
            transformed[col] = mapped.astype(int)

        # Guarantee strict feature ordering
        transformed = transformed[self.all_feature_names]
        return transformed

    def transform_single(self, record: Dict[str, Any]) -> pd.DataFrame:
        """
        Transforms a single patient's dictionary payload into a 1-row DataFrame
        ready for ML prediction and SHAP TreeExplainer attribution.
        """
        return self.transform(record)


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads the authentic raw dataset and performs non-destructive initial string cleanup on the target column.
    """
    df = pd.read_csv(filepath)
    # Target column cleaning (e.g., 'ckd\t' -> 'ckd')
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
    return df


def prepare_and_split_data(
    raw_path: str = RAW_DATA_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, CKDPreprocessor]:
    """
    Executes the complete Phase 2 data preparation workflow:
    1. Loads raw dataset without overwriting.
    2. Separates features and target.
    3. Performs stratified train/test split.
    4. Fits CKDPreprocessor strictly on X_train.
    5. Transforms both X_train and X_test.
    6. Saves processed datasets, metadata, and the fitted preprocessor artifact.
    """
    logging.info(f"Loading raw dataset from {raw_path}...")
    df = load_raw_data(raw_path)

    # Separate features and target
    X_raw = df[ALL_FEATURE_NAMES].copy()
    y_raw = df[TARGET_COLUMN].map(TARGET_MAPPING)

    # Check for invalid targets
    if y_raw.isna().any():
        raise ValueError("Target column contains unmapped or missing values.")

    logging.info(f"Target distribution in full dataset:\n{y_raw.value_counts()}")

    # Stratified Train/Test Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=test_size,
        random_state=random_state,
        stratify=y_raw
    )
    logging.info(f"Split completed: Train instances = {len(X_train_raw)}, Test instances = {len(X_test_raw)}")

    # Fit preprocessor EXCLUSIVELY on training features
    preprocessor = CKDPreprocessor()
    preprocessor.fit(X_train_raw)

    # Transform both splits
    X_train_processed = preprocessor.transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    # Prepare directories
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save processed CSVs with target included
    train_export = X_train_processed.copy()
    train_export[TARGET_COLUMN] = y_train.values
    train_export.to_csv(TRAIN_DATA_PATH, index=False)

    test_export = X_test_processed.copy()
    test_export[TARGET_COLUMN] = y_test.values
    test_export.to_csv(TEST_DATA_PATH, index=False)

    # Save preprocessor artifact (.joblib)
    joblib.dump(preprocessor, PREPROCESSOR_ARTIFACT_PATH)
    logging.info(f"Fitted preprocessor serialized to {PREPROCESSOR_ARTIFACT_PATH}")

    # Save preprocessing metadata JSON for documentation and validation
    metadata = {
        "dataset_name": "UCI Chronic Kidney Disease",
        "dataset_source": "https://archive.ics.uci.edu/static/public/336/data.csv",
        "total_instances": len(df),
        "total_features": len(ALL_FEATURE_NAMES),
        "train_instances": len(X_train_processed),
        "test_instances": len(X_test_processed),
        "test_size": test_size,
        "random_state": random_state,
        "class_distribution_overall": {
            "ckd": int((y_raw == 1).sum()),
            "notckd": int((y_raw == 0).sum())
        },
        "class_distribution_train": {
            "ckd": int((y_train == 1).sum()),
            "notckd": int((y_train == 0).sum())
        },
        "class_distribution_test": {
            "ckd": int((y_test == 1).sum()),
            "notckd": int((y_test == 0).sum())
        },
        "learned_numerical_medians": preprocessor.numerical_medians_,
        "learned_categorical_modes": preprocessor.categorical_modes_,
        "feature_order": ALL_FEATURE_NAMES
    }
    with open(PREPROCESSING_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Preprocessing metadata saved to {PREPROCESSING_METADATA_PATH}")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


if __name__ == "__main__":
    import src.preprocessing as _sp
    X_train, X_test, y_train, y_test, prep = _sp.prepare_and_split_data()
    print("\nPreprocessing complete!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train distribution:\n{y_train.value_counts()}")
    print(f"y_test distribution:\n{y_test.value_counts()}")
    print(f"Any NaNs in X_train: {X_train.isna().sum().sum()}")
    print(f"Any NaNs in X_test: {X_test.isna().sum().sum()}")
