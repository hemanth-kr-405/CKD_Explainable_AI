"""
Module for training and evaluating the Decision Tree Classifier for Chronic Kidney Disease (CKD) prediction.

Adheres to:
- Strict train-test separation (trained purely on X_train, evaluated on X_test)
- Fixed random seed for reproducibility
- Class weight balancing for clinical sensitivity
- Full serialization of fitted model and evaluation metrics
"""

import sys
from pathlib import Path
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import joblib

from src.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    MODEL_ARTIFACT_PATH,
    METRICS_ARTIFACT_PATH,
    MODELS_DIR,
    ALL_FEATURE_NAMES,
    TARGET_COLUMN,
    RANDOM_STATE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_processed_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads the preprocessed training and testing datasets created in Phase 2.
    Ensures input features and target are strictly separated.
    """
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found at {TRAIN_DATA_PATH}. Run Phase 2 preprocessing first.")
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found at {TEST_DATA_PATH}. Run Phase 2 preprocessing first.")

    logging.info(f"Loading training data from {TRAIN_DATA_PATH}...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    logging.info(f"Loading testing data from {TEST_DATA_PATH}...")
    test_df = pd.read_csv(TEST_DATA_PATH)

    # Validate feature consistency
    for col in ALL_FEATURE_NAMES:
        if col not in train_df.columns:
            raise ValueError(f"Expected feature '{col}' missing from train.csv")
        if col not in test_df.columns:
            raise ValueError(f"Expected feature '{col}' missing from test.csv")

    X_train = train_df[ALL_FEATURE_NAMES].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_test = test_df[ALL_FEATURE_NAMES].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    logging.info(f"Features loaded: {len(ALL_FEATURE_NAMES)} columns.")
    logging.info(f"Training set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows.")
    return X_train, y_train, X_test, y_test


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int = 5,
    min_samples_split: int = 4,
    min_samples_leaf: int = 2,
    random_state: int = RANDOM_STATE
) -> DecisionTreeClassifier:
    """
    Initializes and fits a Decision Tree Classifier with balanced class weights.
    
    Hyperparameters chosen for healthcare decision support:
    - max_depth=5: Restricts tree complexity to prevent overfitting and ensure clinical interpretability.
    - min_samples_split=4, min_samples_leaf=2: Prevents isolated single-instance leaf decisions.
    - class_weight='balanced': Compensates for class distribution disparity.
    """
    logging.info("Initializing Decision Tree Classifier...")
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state
    )

    logging.info("Fitting Decision Tree on training dataset (X_train only)...")
    model.fit(X_train, y_train)
    logging.info("Model fitting complete.")
    return model


def evaluate_model(
    model: DecisionTreeClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluates the model using:
    1. Stratified 5-Fold Cross-Validation on the training set (to check generalization stability).
    2. Hold-out Test Set evaluation (for unbiased final performance metrics).
    """
    logging.info("Running Stratified 5-Fold Cross-Validation on X_train...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        return_train_score=False
    )

    cv_summary = {
        "cv_accuracy_mean": round(float(np.mean(cv_results["test_accuracy"])), 4),
        "cv_accuracy_std": round(float(np.std(cv_results["test_accuracy"])), 4),
        "cv_recall_mean": round(float(np.mean(cv_results["test_recall"])), 4),
        "cv_recall_std": round(float(np.std(cv_results["test_recall"])), 4),
        "cv_f1_mean": round(float(np.mean(cv_results["test_f1"])), 4),
        "cv_f1_std": round(float(np.std(cv_results["test_f1"])), 4),
        "cv_roc_auc_mean": round(float(np.mean(cv_results["test_roc_auc"])), 4),
        "cv_roc_auc_std": round(float(np.std(cv_results["test_roc_auc"])), 4)
    }

    logging.info("Evaluating on held-out test set (X_test)...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity / Recall = TP / (TP + FN)
    # Specificity = TN / (TN + FP)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall_sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }

    # Extract Gini Feature Importances
    feature_importances = dict(zip(ALL_FEATURE_NAMES, model.feature_importances_))
    sorted_importances = {
        k: round(float(v), 4)
        for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
        if v > 0
    }

    evaluation_report = {
        "algorithm": "DecisionTreeClassifier",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "hyperparameters": {
            "criterion": model.criterion,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
            "class_weight": model.class_weight
        },
        "dataset_splits": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": len(ALL_FEATURE_NAMES)
        },
        "cross_validation_5fold": cv_summary,
        "test_metrics": test_metrics,
        "tree_depth": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves()),
        "gini_feature_importances": sorted_importances,
        "feature_names_order": ALL_FEATURE_NAMES
    }

    return evaluation_report


def save_model_artifacts(model: DecisionTreeClassifier, evaluation_report: Dict[str, Any]) -> None:
    """
    Serializes the trained Decision Tree model via joblib and exports evaluation metrics to JSON.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save Model Artifact (.joblib)
    logging.info(f"Saving trained model to {MODEL_ARTIFACT_PATH}...")
    joblib.dump(model, MODEL_ARTIFACT_PATH)

    # 2. Save Metrics & Metadata (.json)
    logging.info(f"Saving evaluation report to {METRICS_ARTIFACT_PATH}...")
    with open(METRICS_ARTIFACT_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=4)

    logging.info("Artifacts saved successfully.")


def run_training_pipeline() -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    """
    Orchestrates the entire Phase 3 training and evaluation pipeline.
    """
    logging.info("=" * 70)
    logging.info("STARTING PHASE 3: DECISION TREE TRAINING & EVALUATION")
    logging.info("=" * 70)

    # Step 1: Load preprocessed train and test data
    X_train, y_train, X_test, y_test = load_processed_data()

    # Step 2: Train the model
    model = train_decision_tree(X_train, y_train)

    # Step 3: Evaluate model (CV and Test)
    report = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Step 4: Save model and metrics artifacts
    save_model_artifacts(model, report)

    logging.info("=" * 70)
    logging.info("PHASE 3 TRAINING PIPELINE COMPLETE")
    logging.info(f"Test Accuracy: {report['test_metrics']['accuracy'] * 100:.2f}%")
    logging.info(f"Test Recall / Sensitivity: {report['test_metrics']['recall_sensitivity'] * 100:.2f}%")
    logging.info(f"Test F1-Score: {report['test_metrics']['f1_score'] * 100:.2f}%")
    logging.info(f"Test ROC-AUC: {report['test_metrics']['roc_auc']:.4f}")
    logging.info(f"Top 5 Gini Features: {list(report['gini_feature_importances'].keys())[:5]}")
    logging.info("=" * 70)

    return model, report


if __name__ == "__main__":
    run_training_pipeline()
