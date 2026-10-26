"""
Unit and integration tests for the trained Decision Tree model.

Verifies:
- Model artifact existence and reloadability via joblib
- Consistent input feature shapes and order (24 features)
- Valid prediction classes (0 or 1) and probability ranges [0, 1]
- End-to-end inference using raw clinical dictionary -> preprocessor -> model
"""

import json
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    MODEL_ARTIFACT_PATH,
    METRICS_ARTIFACT_PATH,
    PREPROCESSOR_ARTIFACT_PATH,
    ALL_FEATURE_NAMES,
    TEST_DATA_PATH,
    TARGET_COLUMN,
    TARGET_NAMES
)
from src.preprocessing import CKDPreprocessor


def test_saved_model_artifact_exists():
    """Verify that the model file was serialized to models/ckd_model.joblib."""
    assert MODEL_ARTIFACT_PATH.exists(), f"Model artifact not found at {MODEL_ARTIFACT_PATH}"
    assert MODEL_ARTIFACT_PATH.stat().st_size > 0, "Model artifact file is empty"


from sklearn.ensemble import RandomForestClassifier

def test_saved_metrics_exist_and_valid():
    """Verify that the evaluation metrics file exists and contains valid metrics."""
    assert METRICS_ARTIFACT_PATH.exists(), f"Metrics file not found at {METRICS_ARTIFACT_PATH}"
    with open(METRICS_ARTIFACT_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["algorithm"] in ["RandomForestClassifier", "DecisionTreeClassifier"]
    assert "test_metrics" in metrics
    assert "accuracy" in metrics["test_metrics"]
    assert "precision" in metrics["test_metrics"]
    assert "recall_sensitivity" in metrics["test_metrics"]
    assert "f1_score" in metrics["test_metrics"]
    assert "roc_auc" in metrics["test_metrics"]
    assert "confusion_matrix" in metrics["test_metrics"]
    assert metrics["test_metrics"]["accuracy"] > 0.85
    assert metrics["test_metrics"]["recall_sensitivity"] > 0.85


def test_model_reload_and_properties():
    """Verify model can be reloaded and maintains expected tree properties and feature count."""
    model = joblib.load(MODEL_ARTIFACT_PATH)
    assert isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)), f"Expected tree classifier, got {type(model)}"
    assert model.n_features_in_ == len(ALL_FEATURE_NAMES), (
        f"Model feature count {model.n_features_in_} does not match expected {len(ALL_FEATURE_NAMES)}"
    )
    assert hasattr(model, "classes_")
    assert list(model.classes_) == [0, 1]


def test_model_predictions_on_test_dataset():
    """Verify predictions on test dataset output valid classes and probability distributions."""
    model: DecisionTreeClassifier = joblib.load(MODEL_ARTIFACT_PATH)
    assert TEST_DATA_PATH.exists()
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    X_test = test_df[ALL_FEATURE_NAMES]
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    assert len(predictions) == len(test_df)
    assert set(np.unique(predictions)).issubset({0, 1})
    
    # Probabilities shape and range
    assert probabilities.shape == (len(test_df), 2)
    assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)
    # Rows should sum to 1.0
    row_sums = probabilities.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_end_to_end_inference_flow():
    """Verify complete inference flow from raw dictionary payload to final class prediction."""
    preprocessor: CKDPreprocessor = joblib.load(PREPROCESSOR_ARTIFACT_PATH)
    model: DecisionTreeClassifier = joblib.load(MODEL_ARTIFACT_PATH)

    # Sample clinical patient payload
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

    # Transform through preprocessor
    transformed = preprocessor.transform_single(sample_patient)
    assert transformed.shape == (1, 24)

    # Predict with model
    pred_class = int(model.predict(transformed)[0])
    pred_prob = model.predict_proba(transformed)[0]

    assert pred_class in [0, 1]
    assert 0.0 <= pred_prob[1] <= 1.0
    assert pred_class in TARGET_NAMES
    assert isinstance(TARGET_NAMES[pred_class], str)
