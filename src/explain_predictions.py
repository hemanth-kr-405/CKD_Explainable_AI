"""
Explainable AI (XAI) module using SHAP (SHapley Additive exPlanations)
for Chronic Kidney Disease (CKD) prediction.

Provides:
- TreeExplainer integration for the trained Decision Tree Classifier
- Dynamic class resolution ensuring correct positive-class attribution (CKD = Class 1)
- Global feature importance (mean absolute SHAP) and beeswarm distribution plots
- Local patient-level waterfall plots and feature contribution breakdowns
- Seamless mapping of clinical markers to human-understandable medical labels
- End-to-end patient explanation pipeline: raw clinical payload -> preprocessing -> prediction -> SHAP explanation
"""

import sys
from pathlib import Path
import json
import io
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend for plot generation
import matplotlib.pyplot as plt
import shap
from sklearn.tree import DecisionTreeClassifier
import joblib

from src.config import (
    MODEL_ARTIFACT_PATH,
    PREPROCESSOR_ARTIFACT_PATH,
    TEST_DATA_PATH,
    FIGURES_DIR,
    REPORTS_DIR,
    ALL_FEATURE_NAMES,
    CLINICAL_FEATURE_LABELS,
    FEATURE_METADATA,
    TARGET_NAMES
)
from src.preprocessing import CKDPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CKDExplainer:
    """
    Explainable AI engine for Chronic Kidney Disease predictions using SHAP TreeExplainer.
    Loads the trained model and preprocessing pipeline without retraining.
    """

    def __init__(
        self,
        model_path: Path = MODEL_ARTIFACT_PATH,
        preprocessor_path: Path = PREPROCESSOR_ARTIFACT_PATH
    ):
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}. Train model in Phase 3 first.")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {preprocessor_path}. Complete Phase 2 first.")

        logging.info(f"Loading trained Decision Tree from {model_path}...")
        self.model: DecisionTreeClassifier = joblib.load(model_path)

        logging.info(f"Loading fitted preprocessor from {preprocessor_path}...")
        self.preprocessor: CKDPreprocessor = joblib.load(preprocessor_path)

        # Inspect classes to verify positive class index dynamically
        if 1 not in self.model.classes_:
            raise ValueError(f"Expected class label '1' (CKD) not found in model classes: {self.model.classes_}")
        
        self.classes = list(self.model.classes_)
        self.ckd_class_idx = int(np.where(self.model.classes_ == 1)[0][0])
        logging.info(f"Model classes: {self.classes}. CKD positive class index: {self.ckd_class_idx}")

        # Initialize SHAP TreeExplainer (exact polynomial-time algorithm for decision trees)
        logging.info("Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_names = list(ALL_FEATURE_NAMES)
        self.clinical_labels = [CLINICAL_FEATURE_LABELS.get(f, f) for f in self.feature_names]

    def _prepare_explanation_object(self, X_df: pd.DataFrame) -> shap.Explanation:
        """
        Computes SHAP Explanation object and slices to the positive CKD class (Class 1).
        Assigns human-readable clinical feature labels.
        """
        raw_exp = self.explainer(X_df)
        
        # Binary classification in TreeExplainer yields shape (n_samples, n_features, n_classes)
        if len(raw_exp.shape) == 3 and raw_exp.shape[2] > 1:
            exp_ckd = raw_exp[:, :, self.ckd_class_idx]
        else:
            exp_ckd = raw_exp

        # Set human-readable feature names for clean plot labels
        exp_ckd.feature_names = self.clinical_labels
        return exp_ckd

    def generate_global_explanations(
        self,
        X_test: pd.DataFrame,
        output_dir: Path = FIGURES_DIR,
        reports_dir: Path = REPORTS_DIR
    ) -> Dict[str, Any]:
        """
        Computes global feature attributions across the held-out test cohort.
        Generates and saves:
        1. Global feature-importance bar plot (mean absolute SHAP)
        2. Beeswarm summary plot (feature value distribution vs SHAP impact)
        3. Global importance JSON summary
        """
        logging.info(f"Generating global SHAP explanations across {len(X_test)} test instances...")
        exp_ckd = self._prepare_explanation_object(X_test)
        output_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # 1. SHAP Global Feature Importance Bar Plot
        bar_plot_path = output_dir / "shap_summary_bar.png"
        plt.figure(figsize=(9, 7), dpi=300)
        shap.plots.bar(exp_ckd, max_display=12, show=False)
        plt.title("SHAP Global Feature Importance (Mean |SHAP Value| on Test Cohort)", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved SHAP bar plot to {bar_plot_path}")

        # 2. SHAP Beeswarm Summary Plot
        beeswarm_plot_path = output_dir / "shap_beeswarm.png"
        plt.figure(figsize=(10, 7), dpi=300)
        shap.plots.beeswarm(exp_ckd, max_display=12, show=False)
        plt.title("SHAP Summary Beeswarm Plot (Impact of Clinical Values on CKD Risk)", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(beeswarm_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved SHAP beeswarm plot to {beeswarm_plot_path}")

        # Calculate Mean Absolute SHAP per feature for reporting
        mean_abs_shap = np.mean(np.abs(exp_ckd.values), axis=0)
        feature_importance_ranking = []
        for feat_name, clin_label, importance in zip(self.feature_names, self.clinical_labels, mean_abs_shap):
            feature_importance_ranking.append({
                "feature": feat_name,
                "clinical_label": clin_label,
                "mean_abs_shap": round(float(importance), 4)
            })

        # Sort descending by importance
        feature_importance_ranking.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

        summary_data = {
            "title": "Global SHAP Feature Importance Summary",
            "cohort_size": len(X_test),
            "target_class_explained": "Chronic Kidney Disease (Class 1)",
            "top_features": feature_importance_ranking,
            "plots": {
                "bar_plot": str(bar_plot_path),
                "beeswarm_plot": str(beeswarm_plot_path)
            }
        }

        json_out_path = reports_dir / "shap_global_importance.json"
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)
        logging.info(f"Saved global SHAP metadata to {json_out_path}")

        return summary_data

    def explain_patient(
        self,
        patient_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        save_plot_path: Optional[Path] = None,
        return_base64_plot: bool = False
    ) -> Dict[str, Any]:
        """
        Accepts raw/unprocessed patient clinical attributes, applies the saved preprocessing
        pipeline, computes the model prediction and probabilities, and returns a detailed
        local SHAP explanation with top risk-elevating and risk-lowering factors.
        """
        # 1. Preprocess raw patient inputs using the fitted pipeline
        if isinstance(patient_data, pd.Series):
            patient_dict = patient_data.to_dict()
        elif isinstance(patient_data, pd.DataFrame):
            patient_dict = patient_data.iloc[0].to_dict()
        else:
            patient_dict = dict(patient_data)

        X_transformed = self.preprocessor.transform_single(patient_dict)

        # 2. Predict with Decision Tree
        pred_class = int(self.model.predict(X_transformed)[0])
        pred_probs = self.model.predict_proba(X_transformed)[0]
        ckd_probability = round(float(pred_probs[self.ckd_class_idx]), 4)
        non_ckd_probability = round(float(pred_probs[1 - self.ckd_class_idx]), 4)

        # 3. Compute SHAP Values for Patient
        patient_exp = self._prepare_explanation_object(X_transformed)[0]
        base_value = float(patient_exp.base_values)
        shap_values = patient_exp.values.astype(float)

        # Sum of SHAP values + base value reconstructs model output
        model_output_score = base_value + float(np.sum(shap_values))

        # 4. Deconstruct Individual Feature Attributions
        feature_contributions = []
        for feat, label, shap_val in zip(self.feature_names, self.clinical_labels, shap_values):
            raw_val = patient_dict.get(feat, X_transformed[feat].iloc[0])
            transformed_val = float(X_transformed[feat].iloc[0])

            # Determine clinical directional influence
            if shap_val > 0.0001:
                direction = "Increases Predicted CKD Probability"
                effect = "increases_risk"
            elif shap_val < -0.0001:
                direction = "Decreases Predicted CKD Probability"
                effect = "decreases_risk"
            else:
                direction = "Neutral / Negligible Impact"
                effect = "neutral"

            feature_contributions.append({
                "feature": feat,
                "clinical_label": label,
                "patient_value": raw_val,
                "transformed_value": transformed_val,
                "shap_value": round(float(shap_val), 4),
                "direction": direction,
                "effect": effect,
                "unit": FEATURE_METADATA.get(feat, {}).get("unit", "")
            })

        # Rank factors: Top risk factors (positive SHAP) and top protective factors (negative SHAP)
        risk_factors = sorted(
            [f for f in feature_contributions if f["shap_value"] > 0],
            key=lambda x: x["shap_value"],
            reverse=True
        )
        protective_factors = sorted(
            [f for f in feature_contributions if f["shap_value"] < 0],
            key=lambda x: x["shap_value"]
        )

        # 5. Generate Waterfall Plot
        plot_base64 = None
        if save_plot_path is not None or return_base64_plot:
            fig = plt.figure(figsize=(9, 6), dpi=300)
            shap.plots.waterfall(patient_exp, max_display=10, show=False)
            plt.title(
                f"SHAP Waterfall: Patient Risk Breakdown (Prediction: {TARGET_NAMES[pred_class]})",
                fontsize=11, fontweight="bold", pad=15
            )
            plt.tight_layout()

            if save_plot_path is not None:
                save_plot_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
                logging.info(f"Saved patient waterfall plot to {save_plot_path}")

            if return_base64_plot:
                buffer = io.BytesIO()
                plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                buffer.seek(0)
                plot_base64 = base64.b64encode(buffer.read()).decode("utf-8")

            plt.close()

        explanation_result = {
            "prediction": {
                "class_code": pred_class,
                "class_label": TARGET_NAMES[pred_class],
                "ckd_probability": ckd_probability,
                "non_ckd_probability": non_ckd_probability
            },
            "shap_explanation": {
                "base_value": round(base_value, 4),
                "patient_model_output": round(model_output_score, 4),
                "explanation_target": "Probability of Chronic Kidney Disease (CKD)",
                "interpretation_guide": (
                    "Positive SHAP values indicate features pushing the prediction higher toward CKD. "
                    "Negative SHAP values indicate features pushing the prediction lower toward Non-CKD. "
                    "Base value represents the training cohort baseline expectation. "
                    "Attributions reflect statistical model influence, not biological causation."
                ),
                "top_risk_drivers": risk_factors[:5],
                "top_protective_factors": protective_factors[:5],
                "all_feature_contributions": feature_contributions
            },
            "disclaimer": (
                "Educational Decision Support: SHAP explanations explain algorithmic reasoning. "
                "Feature attributions are statistical associations learned by the model and do NOT constitute biological causation or a confirmed medical diagnosis."
            )
        }

        if plot_base64 is not None:
            explanation_result["waterfall_plot_base64"] = plot_base64
        if save_plot_path is not None:
            explanation_result["waterfall_plot_path"] = str(save_plot_path)

        return explanation_result


def run_explainability_pipeline() -> None:
    """
    Executes Phase 5 XAI generation:
    1. Instantiates CKDExplainer.
    2. Loads held-out test data and generates global summary plots.
    3. Runs individual patient explanations on real test cases (one CKD, one Non-CKD).
    4. Saves all figures in reports/figures/.
    """
    logging.info("=" * 70)
    logging.info("STARTING PHASE 5: EXPLAINABLE AI USING SHAP")
    logging.info("=" * 70)

    explainer = CKDExplainer()

    # Load test dataset
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df[ALL_FEATURE_NAMES]
    y_test = test_df["class"]

    # 1. Global Explanations
    global_summary = explainer.generate_global_explanations(X_test)
    logging.info(f"Top 5 globally influential features: {[f['feature'] for f in global_summary['top_features'][:5]]}")

    # 2. Local Patient Explanations (Pick representative examples from test set)
    # Case A: A True Positive CKD Patient
    ckd_test_idx = y_test[y_test == 1].index[0]
    patient_ckd_row = test_df.loc[ckd_test_idx].to_dict()
    ckd_plot_path = FIGURES_DIR / "shap_waterfall_patient_ckd.png"

    logging.info(f"Explaining True CKD Patient (Index {ckd_test_idx})...")
    ckd_explanation = explainer.explain_patient(
        patient_ckd_row,
        save_plot_path=ckd_plot_path
    )
    logging.info(f"Patient CKD Prediction: {ckd_explanation['prediction']['class_label']} (Probability: {ckd_explanation['prediction']['ckd_probability']*100:.1f}%)")
    logging.info(f"Top risk drivers: {[f['clinical_label'] for f in ckd_explanation['shap_explanation']['top_risk_drivers']]}")

    # Case B: A True Negative Healthy (Non-CKD) Patient
    healthy_test_idx = y_test[y_test == 0].index[0]
    patient_healthy_row = test_df.loc[healthy_test_idx].to_dict()
    healthy_plot_path = FIGURES_DIR / "shap_waterfall_patient_healthy.png"

    logging.info(f"Explaining Healthy Non-CKD Patient (Index {healthy_test_idx})...")
    healthy_explanation = explainer.explain_patient(
        patient_healthy_row,
        save_plot_path=healthy_plot_path
    )
    logging.info(f"Patient Healthy Prediction: {healthy_explanation['prediction']['class_label']} (Probability: {healthy_explanation['prediction']['ckd_probability']*100:.1f}%)")
    logging.info(f"Top protective factors: {[f['clinical_label'] for f in healthy_explanation['shap_explanation']['top_protective_factors']]}")

    logging.info("=" * 70)
    logging.info("PHASE 5 SHAP EXPLAINABILITY PIPELINE COMPLETE")
    logging.info("Generated Plots:")
    logging.info(f"  - {FIGURES_DIR / 'shap_summary_bar.png'}")
    logging.info(f"  - {FIGURES_DIR / 'shap_beeswarm.png'}")
    logging.info(f"  - {ckd_plot_path}")
    logging.info(f"  - {healthy_plot_path}")
    logging.info("=" * 70)


if __name__ == "__main__":
    run_explainability_pipeline()
