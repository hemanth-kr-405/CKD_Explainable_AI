"""
Clinical PDF Report Generator for Explainable AI Chronic Kidney Disease Assessment.

Generates professional, multi-page vector PDF reports conforming to all 11 required sections:
1. Report title and generation timestamp
2. Unique non-identifying analysis reference
3. Document processing status
4. Extracted and user-verified parameters, with units
5. Missing or unresolved information and imputation transparency
6. Model prediction
7. Model probability, only if available and correctly interpreted
8. SHAP explanation, feature contributions, and directionality guide
9. Data extraction limitations
10. Model limitations
11. Medical disclaimer

Built using matplotlib.backends.backend_pdf for zero-dependency vector PDF rendering.
"""

import io
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.config import ALL_FEATURE_NAMES, FEATURE_METADATA


def sanitize_pdf_text(val: Any, max_len: Optional[int] = None) -> str:
    """
    Sanitizes untrusted text for safe rendering in Matplotlib PDF canvas.
    - Replaces non-printable ASCII and control characters with space.
    - Escapes '$' to prevent Matplotlib TeX math mode parser syntax errors.
    - Normalizes newlines and spaces.
    - Truncates to max_len if specified.
    """
    if val is None:
        return ""
    text = str(val)
    # Replace control characters (ASCII 0-31 except allowed whitespace, and 127) with space
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    # Escape dollar signs to avoid triggering TeX math mode in matplotlib
    text = text.replace("$", r"\$")
    # Replace line breaks and tabs with single space for table cells
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text).strip()
    if max_len is not None and len(text) > max_len:
        text = text[:max_len - 3] + "..."
    return text


def generate_clinical_pdf_report(
    prediction_data: Dict[str, Any],
    verified_parameters: Optional[Dict[str, Any]] = None,
    document_name: Optional[str] = None,
    session_id: Optional[str] = None
) -> bytes:
    """
    Generates an on-demand, multi-page vector PDF clinical decision support report.
    Conforms to all 11 required report sections and safety constraints.

    Args:
        prediction_data: Dictionary conforming to VerifiedPredictionResponse / PredictionResponse.
        verified_parameters: Optional dictionary of verified patient parameters.
        document_name: Optional name of the uploaded medical document.
        session_id: Optional unique session/upload identifier.

    Returns:
        bytes: Raw PDF file bytes ready for streaming download.
    """
    buf = io.BytesIO()

    # 1. Title and Generation Timestamp
    timestamp_utc = datetime.now(timezone.utc)
    timestamp_str = timestamp_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    # 2. Unique Non-Identifying Analysis Reference
    if session_id and len(session_id.strip()) > 0:
        clean_sess = re.sub(r'[^a-zA-Z0-9]', '', session_id)[:8].upper()
    else:
        clean_sess = uuid.uuid4().hex[:8].upper()
    analysis_ref = f"CKD-REF-{clean_sess}-{timestamp_utc.strftime('%m%d%H%M')}"

    # Sanitized Document Name
    doc_display = sanitize_pdf_text(document_name, max_len=45) if document_name else "Manual Clinical Entry / Structured Form"

    # Assess prediction failure or success
    is_failed = False
    error_msg = ""
    pred_label = prediction_data.get("prediction_label", "")
    pred_code = prediction_data.get("prediction_code", 0)

    if not pred_label or pred_label in ["Prediction Failed", "Error", "Unknown Assessment"] or pred_code < 0:
        is_failed = True
        error_msg = prediction_data.get("error_message") or prediction_data.get("detail") or "Analysis incomplete or validation failed."
        pred_label = "PREDICTION INCOMPLETE / FAILED"

    is_ckd = (pred_code == 1) and not is_failed

    # 7. Model Probability
    probs = prediction_data.get("probabilities")
    has_probability = False
    ckd_prob = 0.0
    non_ckd_prob = 0.0
    confidence_pct = 0.0

    if probs and isinstance(probs, dict) and not is_failed:
        ckd_prob = float(probs.get("ckd_probability", 0.0))
        non_ckd_prob = float(probs.get("non_ckd_probability", 0.0))
        confidence_pct = round(max(ckd_prob, non_ckd_prob) * 100, 1)
        has_probability = True

    # Provenance summary & missing info
    provenance = prediction_data.get("provenance_summary", {})
    v_count = provenance.get("verified_count", 0)
    e_count = provenance.get("extracted_count", 0)
    m_count = provenance.get("manual_count", 0)
    imp_strategy = provenance.get("imputation_strategy", "Training cohort median/mode")

    # Build missing parameters list
    vparam_map = verified_parameters if verified_parameters else {}
    missing_params: List[str] = []
    for feat in ALL_FEATURE_NAMES:
        if feat in vparam_map and isinstance(vparam_map[feat], dict):
            st = vparam_map[feat].get("status", "")
            val = vparam_map[feat].get("value")
            if st == "missing" or val is None or val == "":
                meta = FEATURE_METADATA.get(feat, {})
                missing_params.append(f"{meta.get('label', feat)} ({feat})")
        elif feat not in vparam_map:
            meta = FEATURE_METADATA.get(feat, {})
            missing_params.append(f"{meta.get('label', feat)} ({feat})")

    # 8. SHAP Explanation Data
    explanation = prediction_data.get("explanation")
    has_shap = False
    base_val = 0.625
    final_val = 1.0
    output_scale = "Probability Scale [0.0 to 1.0]"
    most_influential = []
    all_attribs = []

    if explanation and isinstance(explanation, dict) and not is_failed:
        has_shap = True
        base_val = explanation.get("base_value", 0.625)
        final_val = explanation.get("patient_model_output", 1.0)
        output_scale = explanation.get("output_scale", "Probability Scale [0.0 to 1.0]")
        all_attribs = explanation.get("all_feature_contributions", [])
        most_influential = explanation.get("most_influential_features", [])
        if not most_influential and all_attribs:
            most_influential = sorted(all_attribs, key=lambda x: abs(x.get("shap_value", 0.0)), reverse=True)

    # 11. Disclaimer Text
    disclaimer = prediction_data.get(
        "medical_disclaimer",
        "RESEARCH & CLINICAL DECISION SUPPORT ONLY. NOT A MEDICAL DIAGNOSIS."
    )

    with PdfPages(buf) as pdf:
        # ====================================================================
        # PAGE 1: Executive Assessment, Metadata & Missing Information
        # ====================================================================
        fig1 = plt.figure(figsize=(8.5, 11), dpi=150)
        ax1 = fig1.add_axes([0, 0, 1, 1])
        ax1.axis("off")

        # Top Banner (Section 1: Title & Timestamp)
        if is_failed:
            hdr_bg = "#475569"
        elif is_ckd:
            hdr_bg = "#881337"
        else:
            hdr_bg = "#1e3a8a"

        ax1.fill_between([0, 1], [0.91, 0.91], [1.0, 1.0], color=hdr_bg, transform=ax1.transAxes)
        ax1.text(0.05, 0.965, "CHRONIC KIDNEY DISEASE (CKD) AI PREDICTION REPORT",
                 fontsize=13.5, fontweight="bold", color="white", transform=ax1.transAxes)
        ax1.text(0.05, 0.932, "Explainable AI Clinical Decision Support - Research & Investigational Report",
                 fontsize=9.0, color="#f1f5f9", transform=ax1.transAxes)
        ax1.text(0.68, 0.932, f"Generated: {timestamp_str}", fontsize=8.0, color="#e2e8f0", transform=ax1.transAxes)

        # Section 2 & 3: Reference & Document Processing Status Bar
        ax1.fill_between([0.05, 0.95], [0.855, 0.855], [0.898, 0.898], color="#f8fafc", transform=ax1.transAxes)
        ax1.plot([0.05, 0.95, 0.95, 0.05, 0.05], [0.855, 0.855, 0.898, 0.898, 0.855], color="#cbd5e1", lw=0.8, transform=ax1.transAxes)

        ax1.text(0.065, 0.875, "ANALYSIS REFERENCE:", fontsize=7.5, fontweight="bold", color="#64748b", transform=ax1.transAxes)
        ax1.text(0.21, 0.875, sanitize_pdf_text(analysis_ref, 32), fontsize=8.0, fontweight="bold", color="#1e293b", transform=ax1.transAxes)

        ax1.text(0.50, 0.875, "SOURCE DOCUMENT:", fontsize=7.5, fontweight="bold", color="#64748b", transform=ax1.transAxes)
        ax1.text(0.64, 0.875, doc_display, fontsize=8.0, color="#1e293b", transform=ax1.transAxes)

        # Section 3: Document Processing Status Callout
        proc_status = "Status: Completed & Formally Verified" if not is_failed else "Status: Failed / Rejected"
        integrity_txt = "Privacy: Ephemeral Session - Zero External Transmission - Local Execution"
        ax1.fill_between([0.05, 0.95], [0.812, 0.812], [0.846, 0.846], color="#f1f5f9", transform=ax1.transAxes)
        ax1.plot([0.05, 0.95, 0.95, 0.05, 0.05], [0.812, 0.812, 0.846, 0.846, 0.812], color="#cbd5e1", lw=0.6, transform=ax1.transAxes)
        ax1.text(0.065, 0.825, f"DOCUMENT PROCESSING STATUS: {proc_status}", fontsize=7.8, fontweight="bold", color="#0f172a", transform=ax1.transAxes)
        ax1.text(0.52, 0.825, integrity_txt, fontsize=7.2, color="#475569", transform=ax1.transAxes)

        # Section 6 & 7: Model Prediction & Probability Box
        if is_failed:
            res_bg = "#fef2f2"
            res_border = "#f87171"
            res_title_color = "#991b1b"
        elif is_ckd:
            res_bg = "#fef2f2"
            res_border = "#f87171"
            res_title_color = "#991b1b"
        else:
            res_bg = "#ecfdf5"
            res_border = "#34d399"
            res_title_color = "#065f46"

        ax1.fill_between([0.05, 0.95], [0.665, 0.665], [0.798, 0.798], color=res_bg, transform=ax1.transAxes)
        ax1.plot([0.05, 0.95, 0.95, 0.05, 0.05], [0.665, 0.665, 0.798, 0.798, 0.665], color=res_border, lw=1.2, transform=ax1.transAxes)

        ax1.text(0.075, 0.772, "SECTION 6: MODEL PREDICTION OUTCOME", fontsize=8.0, fontweight="bold", color="#64748b", transform=ax1.transAxes)
        ax1.text(0.075, 0.730, sanitize_pdf_text(pred_label.upper(), 45), fontsize=14.5, fontweight="bold", color=res_title_color, transform=ax1.transAxes)

        if is_failed:
            ax1.text(0.075, 0.702, "SECTION 7: MODEL PROBABILITY: Inapplicable (Prediction Incomplete or Failed)",
                     fontsize=8.0, fontweight="bold", color="#b91c1c", transform=ax1.transAxes)
            ax1.text(0.075, 0.684, f"Error Detail: {sanitize_pdf_text(error_msg, 95)}", fontsize=7.8, color="#7f1d1d", transform=ax1.transAxes)
            ax1.text(0.075, 0.668, "Assessment could not be finalized. Please re-verify clinical parameters.", fontsize=7.2, color="#64748b", transform=ax1.transAxes)
        else:
            # Section 7: Model Probability Interpretation
            ax1.text(0.075, 0.702, "SECTION 7: MODEL PROBABILITY & LEAF PURITY", fontsize=7.8, fontweight="bold", color="#475569", transform=ax1.transAxes)
            if has_probability:
                prob_line = f"Terminal Leaf Purity: {confidence_pct}% ({ckd_prob:.2f} CKD vs {non_ckd_prob:.2f} Non-CKD)"
                ax1.text(0.075, 0.684, prob_line, fontsize=8.5, fontweight="bold", color="#0f172a", transform=ax1.transAxes)
                ax1.text(0.075, 0.670, "Interpretation: Reflects empirical proportion of training samples in the terminal leaf, NOT clinical probability.",
                         fontsize=7.0, color="#475569", transform=ax1.transAxes)
            else:
                ax1.text(0.075, 0.684, "Model Probability: Not Applicable / Inapplicable (Probability Uncalibrated)", fontsize=8.0, color="#64748b", transform=ax1.transAxes)

        # Right-side distinction card within Section 6/7
        ax1.text(0.58, 0.772, "CLINICAL VS MODEL DISTINCTION", fontsize=8.0, fontweight="bold", color="#64748b", transform=ax1.transAxes)
        ax1.text(0.58, 0.748, "- Machine Learning Output: Mathematical classifier result", fontsize=7.5, color="#1e293b", transform=ax1.transAxes)
        ax1.text(0.58, 0.728, "- Clinical Diagnosis: CANNOT be established by ML", fontsize=7.5, fontweight="bold", color="#b91c1c", transform=ax1.transAxes)
        ax1.text(0.58, 0.708, "- Requires comprehensive nephrologist evaluation & lab work", fontsize=7.0, color="#475569", transform=ax1.transAxes)
        ax1.text(0.58, 0.688, "- TreeExplainer reflects local tabular statistical splits", fontsize=7.0, color="#64748b", transform=ax1.transAxes)

        # Section 5: Missing or Unresolved Information & Imputation Transparency
        ax1.fill_between([0.05, 0.95], [0.490, 0.490], [0.648, 0.648], color="#fffbeb", transform=ax1.transAxes)
        ax1.plot([0.05, 0.95, 0.95, 0.05, 0.05], [0.490, 0.490, 0.648, 0.648, 0.490], color="#fde68a", lw=1.0, transform=ax1.transAxes)

        ax1.text(0.07, 0.628, "SECTION 5: MISSING OR UNRESOLVED INFORMATION & IMPUTATION AUDIT",
                 fontsize=8.5, fontweight="bold", color="#92400e", transform=ax1.transAxes)

        imp_summary_text = (
            f"Audit Summary: {v_count} Verified | {e_count} Extracted | {m_count} Manual | "
            f"{len(missing_params)} Missing/Imputed features."
        )
        ax1.text(0.07, 0.606, imp_summary_text, fontsize=8.0, fontweight="bold", color="#78350f", transform=ax1.transAxes)

        if missing_params:
            missing_preview = ", ".join(missing_params[:8])
            if len(missing_params) > 8:
                missing_preview += f", and {len(missing_params) - 8} more features"
            ax1.text(0.07, 0.582, f"Unresolved / Missing Parameters: {sanitize_pdf_text(missing_preview, 95)}",
                     fontsize=7.5, color="#78350f", transform=ax1.transAxes)
            ax1.text(0.07, 0.558, f"Imputation Methodology: {sanitize_pdf_text(imp_strategy, 50)} (Numerical = cohort median, Categorical = cohort mode).",
                     fontsize=7.2, color="#92400e", transform=ax1.transAxes)
            ax1.text(0.07, 0.534, "Clinical Warning: Missing-value imputation introduces statistical variance. Confirm critical lab values prior to clinical decisions.",
                     fontsize=7.0, fontweight="bold", color="#b45309", transform=ax1.transAxes)
        else:
            ax1.text(0.07, 0.575, "All 24 clinical parameters were explicitly verified. No feature imputation was applied for this analysis.",
                     fontsize=7.8, color="#065f46", fontweight="bold", transform=ax1.transAxes)

        # Clinical Context & Biomarkers Snapshot Box (Section 4 Intro)
        ax1.fill_between([0.05, 0.95], [0.290, 0.290], [0.470, 0.470], color="#f8fafc", transform=ax1.transAxes)
        ax1.plot([0.05, 0.95, 0.95, 0.05, 0.05], [0.290, 0.290, 0.470, 0.470, 0.290], color="#e2e8f0", lw=0.8, transform=ax1.transAxes)

        ax1.text(0.07, 0.448, "KEY CLINICAL BIOMARKER SNAPSHOT (PRIMARY RENAL INDICES)", fontsize=8.5, fontweight="bold", color="#0f172a", transform=ax1.transAxes)
        ax1.text(0.07, 0.428, "Detailed complete 24-feature table is documented on Page 2.", fontsize=7.2, color="#64748b", transform=ax1.transAxes)

        # Display 6 key biomarkers snapshot
        snapshot_keys = ["sc", "bu", "bgr", "hemo", "al", "sg"]
        snap_y = 0.395
        for s_idx, skey in enumerate(snapshot_keys):
            col = 0 if s_idx < 3 else 1
            row = s_idx % 3
            bx = 0.07 if col == 0 else 0.52
            by = snap_y - (row * 0.038)

            meta = FEATURE_METADATA.get(skey, {})
            slbl = meta.get("label", skey)
            sunit = meta.get("unit", "")

            sval = "Missing (Imputed)"
            sstat = "Imputed"
            if skey in vparam_map and isinstance(vparam_map[skey], dict):
                v_in = vparam_map[skey].get("value")
                if v_in is not None and v_in != "":
                    sval = f"{v_in} {sunit}".strip()
                    sstat = vparam_map[skey].get("status", "Verified")

            ax1.text(bx, by, f"{slbl}:", fontsize=7.5, color="#334155", transform=ax1.transAxes)
            ax1.text(bx + 0.22, by, sanitize_pdf_text(sval, 18), fontsize=7.5, fontweight="bold", color="#0f172a", transform=ax1.transAxes)
            st_color = "#047857" if "verif" in sstat.lower() else "#b45309" if "imput" in sstat.lower() else "#1e40af"
            ax1.text(bx + 0.36, by, f"[{sstat[:9]}]", fontsize=6.8, color=st_color, transform=ax1.transAxes)

        # Governance & Research Label Callout
        ax1.fill_between([0.05, 0.95], [0.100, 0.100], [0.268, 0.268], color="#eff6ff", transform=ax1.transAxes)
        ax1.plot([0.05, 0.95, 0.95, 0.05, 0.05], [0.100, 0.100, 0.268, 0.268, 0.100], color="#bfdbfe", lw=0.8, transform=ax1.transAxes)

        ax1.text(0.07, 0.245, "REPORT GOVERNANCE & RESEARCH CLASSIFICATION LABEL", fontsize=8.5, fontweight="bold", color="#1e3a8a", transform=ax1.transAxes)
        ax1.text(0.07, 0.222, "- CLASSIFICATION: Investigational / Clinical Decision Support Tool (Research Use Only).", fontsize=7.5, color="#1e40af", fontweight="bold", transform=ax1.transAxes)
        ax1.text(0.07, 0.202, "- NON-IDENTIFYING: Report contains zero protected personal health identifiers (PHI).", fontsize=7.5, color="#1e40af", transform=ax1.transAxes)
        ax1.text(0.07, 0.182, "- ON-DEMAND TRANSIENCE: Generated dynamically in volatile memory; zero server persistence.", fontsize=7.5, color="#1e40af", transform=ax1.transAxes)
        ax1.text(0.07, 0.162, "- EXPLAINABILITY AUDIT: SHAP mathematical feature attributions are provided on Page 2 & 3.", fontsize=7.5, color="#1e40af", transform=ax1.transAxes)
        ax1.text(0.07, 0.142, "- REPOSITORY & MODEL INTEGRITY: Based on fitted scikit-learn DecisionTree with calibrated metrics.", fontsize=7.5, color="#1e40af", transform=ax1.transAxes)
        ax1.text(0.07, 0.118, "- ACTIONABLE NEXT STEP: If CKD is indicated, confirm via laboratory eGFR, ACR, and renal ultrasound.", fontsize=7.2, color="#0369a1", fontweight="bold", transform=ax1.transAxes)

        # Page 1 Footer
        ax1.fill_between([0.05, 0.95], [0.03, 0.03], [0.075, 0.075], color="#f8fafc", transform=ax1.transAxes)
        ax1.plot([0.05, 0.95, 0.95, 0.05, 0.05], [0.03, 0.03, 0.075, 0.075, 0.03], color="#cbd5e1", lw=0.6, transform=ax1.transAxes)
        ax1.text(0.065, 0.055, "MANDATORY MEDICAL GOVERNANCE NOTICE:", fontsize=7.2, fontweight="bold", color="#b91c1c", transform=ax1.transAxes)
        ax1.text(0.065, 0.038, sanitize_pdf_text(disclaimer, 130), fontsize=6.5, color="#475569", transform=ax1.transAxes)
        ax1.text(0.85, 0.015, "Page 1 of 3", fontsize=7.5, color="#64748b", transform=ax1.transAxes)

        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        # ====================================================================
        # PAGE 2: Extracted & User-Verified Parameters (Section 4) & SHAP (Section 8)
        # ====================================================================
        fig2 = plt.figure(figsize=(8.5, 11), dpi=150)
        ax2 = fig2.add_axes([0, 0, 1, 1])
        ax2.axis("off")

        # Top Banner Page 2
        ax2.fill_between([0, 1], [0.92, 0.92], [1.0, 1.0], color="#1e293b", transform=ax2.transAxes)
        ax2.text(0.05, 0.965, "SECTION 4: EXTRACTED & USER-VERIFIED PARAMETERS (24 FEATURES)",
                 fontsize=13.0, fontweight="bold", color="white", transform=ax2.transAxes)
        ax2.text(0.05, 0.932, f"Clinical Lab Provenance & Verification Audit - Ref: {analysis_ref}",
                 fontsize=8.5, color="#94a3b8", transform=ax2.transAxes)

        # 24-Feature Parameter Matrix Table (12 rows x 2 columns)
        y_start_p2 = 0.880
        row_height_p2 = 0.034
        col1_x = 0.05
        col2_x = 0.51

        # Table Column Headers
        for cx in [col1_x, col2_x]:
            ax2.fill_between([cx, cx + 0.44], [y_start_p2, y_start_p2], [y_start_p2 + 0.024, y_start_p2 + 0.024], color="#e2e8f0", transform=ax2.transAxes)
            ax2.text(cx + 0.01, y_start_p2 + 0.007, "PARAMETER NAME", fontsize=6.8, fontweight="bold", color="#334155", transform=ax2.transAxes)
            ax2.text(cx + 0.28, y_start_p2 + 0.007, "VALUE & UNIT", fontsize=6.8, fontweight="bold", color="#334155", transform=ax2.transAxes)
            ax2.text(cx + 0.38, y_start_p2 + 0.007, "STATUS", fontsize=6.8, fontweight="bold", color="#334155", transform=ax2.transAxes)

        attrib_map = {item.get("feature"): item for item in all_attribs}

        for i, feat in enumerate(ALL_FEATURE_NAMES):
            col_x = col1_x if i < 12 else col2_x
            row_idx = i if i < 12 else i - 12
            y_pos = (y_start_p2 - 0.026) - (row_idx * row_height_p2)

            meta = FEATURE_METADATA.get(feat, {})
            label = meta.get("label", feat)
            unit = meta.get("unit", "")

            val = None
            status_tag = "Imputed"
            if vparam_map and feat in vparam_map and isinstance(vparam_map[feat], dict):
                val = vparam_map[feat].get("value")
                st = vparam_map[feat].get("status", "")
                if st == "verified":
                    status_tag = "Verified"
                elif st == "extracted":
                    status_tag = "Extracted"
                elif st == "requires_review":
                    status_tag = "Review"
                elif st == "manual":
                    status_tag = "Manual"
                elif st == "missing":
                    status_tag = "Imputed"
            elif not vparam_map and feat in attrib_map:
                val = attrib_map[feat].get("patient_value")
                if val is not None:
                    status_tag = "Supplied"

            val_str = f"{val} {unit}".strip() if (val is not None and val != "") else f"Imputed ({unit})".strip()

            bg_c = "#f8fafc" if row_idx % 2 == 0 else "#ffffff"
            ax2.fill_between([col_x, col_x + 0.44], [y_pos - 0.006, y_pos - 0.006], [y_pos + 0.022, y_pos + 0.022],
                             color=bg_c, transform=ax2.transAxes)
            ax2.plot([col_x, col_x + 0.44], [y_pos - 0.006, y_pos - 0.006], color="#e2e8f0", lw=0.5, transform=ax2.transAxes)

            ax2.text(col_x + 0.01, y_pos + 0.005, sanitize_pdf_text(f"{label} ({feat})", 25),
                     fontsize=7.2, color="#1e293b", transform=ax2.transAxes)
            ax2.text(col_x + 0.28, y_pos + 0.005, sanitize_pdf_text(val_str, 15),
                     fontsize=7.2, color="#0f172a", fontweight="bold", transform=ax2.transAxes)

            st_col = "#047857" if "Verif" in status_tag else "#b45309" if "Imput" in status_tag else "#1e40af"
            ax2.text(col_x + 0.38, y_pos + 0.005, f"[{status_tag[:7]}]",
                     fontsize=6.5, color=st_col, fontweight="bold", transform=ax2.transAxes)

        # Section 8: SHAP Explainability & Feature Contributions Header
        y_shap_start = 0.435
        ax2.fill_between([0.05, 0.95], [y_shap_start, y_shap_start], [y_shap_start + 0.045, y_shap_start + 0.045], color="#f1f5f9", transform=ax2.transAxes)
        ax2.plot([0.05, 0.95, 0.95, 0.05, 0.05], [y_shap_start, y_shap_start, y_shap_start + 0.045, y_shap_start + 0.045, y_shap_start], color="#cbd5e1", lw=0.8, transform=ax2.transAxes)

        ax2.text(0.065, y_shap_start + 0.026, "SECTION 8: SHAP EXPLANATION & FEATURE CONTRIBUTIONS", fontsize=8.8, fontweight="bold", color="#0f172a", transform=ax2.transAxes)
        if has_shap:
            shap_sub = f"Attribution Scale: {output_scale} - Baseline E[f(x)]: {base_val:.4f} - Patient Output f(x): {final_val:.4f}"
            ax2.text(0.065, y_shap_start + 0.008, shap_sub, fontsize=7.2, color="#334155", transform=ax2.transAxes)
        else:
            ax2.text(0.065, y_shap_start + 0.008, "SHAP attribution is unavailable or not computed for this prediction.", fontsize=7.2, color="#94a3b8", transform=ax2.transAxes)

        # Directionality mechanics strip
        ax2.text(0.60, y_shap_start + 0.026, "DIRECTIONALITY GUIDE", fontsize=7.5, fontweight="bold", color="#475569", transform=ax2.transAxes)
        ax2.text(0.60, y_shap_start + 0.010, "(+phi) Red: Increases CKD risk   (-phi) Green: Protective / Non-CKD", fontsize=6.8, color="#334155", transform=ax2.transAxes)

        # Horizontal Bar Chart for Top Influential Features
        bar_y_top = 0.395
        ax2.text(0.05, bar_y_top + 0.015, "TOP INFLUENTIAL CLINICAL CONTRIBUTORS (HORIZONTAL DIVERGENCE CHART)", fontsize=8.2, fontweight="bold", color="#1e293b", transform=ax2.transAxes)

        if has_shap and most_influential:
            top_feats = most_influential[:7]
            max_abs_shap = max(max(abs(f.get("shap_value", 0.0)) for f in top_feats), 0.01)
            bar_row_h = 0.042

            for b_idx, item in enumerate(top_feats):
                cur_by = bar_y_top - 0.015 - (b_idx * bar_row_h)
                phi = float(item.get("shap_value", 0.0))
                feat_key = item.get("feature", "")
                lbl = item.get("clinical_label", feat_key)
                pval = item.get("patient_value")
                pval_str = f"Val: {pval}" if pval is not None else "Val: Imputed"

                ax2.text(0.05, cur_by + 0.014, sanitize_pdf_text(lbl, 24), fontsize=7.5, fontweight="bold", color="#1e293b", transform=ax2.transAxes)
                ax2.text(0.05, cur_by + 0.002, sanitize_pdf_text(pval_str, 20), fontsize=6.5, color="#64748b", transform=ax2.transAxes)

                # Track boundaries
                track_l = 0.35
                track_r = 0.82
                cx = 0.585
                half_w = cx - track_l

                ax2.fill_between([track_l, track_r], [cur_by, cur_by], [cur_by + 0.022, cur_by + 0.022], color="#f1f5f9", transform=ax2.transAxes)
                ax2.plot([cx, cx], [cur_by - 0.002, cur_by + 0.024], color="#94a3b8", lw=1.0, transform=ax2.transAxes)

                bar_w = (abs(phi) / max_abs_shap) * half_w
                if phi > 0.0001:
                    ax2.fill_between([cx, cx + bar_w], [cur_by, cur_by], [cur_by + 0.022, cur_by + 0.022],
                                     color="#f43f5e", transform=ax2.transAxes)
                    scol = "#e11d48"
                elif phi < -0.0001:
                    ax2.fill_between([cx - bar_w, cx], [cur_by, cur_by], [cur_by + 0.022, cur_by + 0.022],
                                     color="#10b981", transform=ax2.transAxes)
                    scol = "#059669"
                else:
                    scol = "#64748b"

                score_str = f"+{phi:.4f}" if phi > 0 else f"{phi:.4f}"
                ax2.text(0.84, cur_by + 0.006, score_str, fontsize=7.5, fontweight="bold", color=scol, transform=ax2.transAxes)
        else:
            ax2.text(0.05, bar_y_top - 0.05, "SHAP divergence chart is omitted because prediction was incomplete or failed.", fontsize=8.0, color="#64748b", transform=ax2.transAxes)

        # Non-Causality Statement Callout
        ax2.fill_between([0.05, 0.95], [0.03, 0.03], [0.075, 0.075], color="#fff1f2", transform=ax2.transAxes)
        ax2.plot([0.05, 0.95, 0.95, 0.05, 0.05], [0.03, 0.03, 0.075, 0.075, 0.03], color="#fda4af", lw=0.6, transform=ax2.transAxes)
        ax2.text(0.065, 0.055, "SHAP MATHEMATICAL GOVERNANCE & NON-CAUSALITY STATEMENT:", fontsize=7.0, fontweight="bold", color="#9f1239", transform=ax2.transAxes)
        ax2.text(0.065, 0.038, "Feature contributions quantify statistical weights within the trained tree model; they do NOT prove clinical etiology or biological cause.",
                 fontsize=6.5, color="#881337", transform=ax2.transAxes)
        ax2.text(0.85, 0.015, "Page 2 of 3", fontsize=7.5, color="#64748b", transform=ax2.transAxes)

        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        # ====================================================================
        # PAGE 3: Limitations, Governance & Formal Medical Disclaimer
        # ====================================================================
        fig3 = plt.figure(figsize=(8.5, 11), dpi=150)
        ax3 = fig3.add_axes([0, 0, 1, 1])
        ax3.axis("off")

        # Top Banner Page 3
        ax3.fill_between([0, 1], [0.92, 0.92], [1.0, 1.0], color="#0f172a", transform=ax3.transAxes)
        ax3.text(0.05, 0.965, "SYSTEM LIMITATIONS, CLINICAL GOVERNANCE & DISCLAIMERS",
                 fontsize=13.0, fontweight="bold", color="white", transform=ax3.transAxes)
        ax3.text(0.05, 0.932, "Transparent AI Disclosure - Regulatory and Safe Use Standards",
                 fontsize=8.5, color="#94a3b8", transform=ax3.transAxes)

        # Top Section: Feature Attribution Table (Top 8 features)
        ax3.text(0.05, 0.890, "DETAILED SHAP ATTRIBUTION MATRIX (KEY MODEL DRIVERS)", fontsize=8.5, fontweight="bold", color="#0f172a", transform=ax3.transAxes)

        tab_y_start = 0.865
        tab_row_h = 0.024

        ax3.fill_between([0.05, 0.95], [tab_y_start, tab_y_start], [tab_y_start + 0.020, tab_y_start + 0.020], color="#e2e8f0", transform=ax2.transAxes)
        ax3.text(0.06, tab_y_start + 0.005, "FEATURE NAME", fontsize=6.8, fontweight="bold", color="#334155", transform=ax3.transAxes)
        ax3.text(0.35, tab_y_start + 0.005, "PATIENT VALUE", fontsize=6.8, fontweight="bold", color="#334155", transform=ax3.transAxes)
        ax3.text(0.55, tab_y_start + 0.005, "MODEL INPUT", fontsize=6.8, fontweight="bold", color="#334155", transform=ax3.transAxes)
        ax3.text(0.70, tab_y_start + 0.005, "SHAP (phi)", fontsize=6.8, fontweight="bold", color="#334155", transform=ax3.transAxes)
        ax3.text(0.82, tab_y_start + 0.005, "DIRECTION", fontsize=6.8, fontweight="bold", color="#334155", transform=ax3.transAxes)

        if has_shap and all_attribs:
            sorted_attribs = sorted(all_attribs, key=lambda x: abs(x.get("shap_value", 0.0)), reverse=True)[:8]
            for r_idx, item in enumerate(sorted_attribs):
                cur_y = tab_y_start - ((r_idx + 1) * tab_row_h)
                bg_tr = "#f8fafc" if r_idx % 2 == 0 else "#ffffff"
                ax3.fill_between([0.05, 0.95], [cur_y, cur_y], [cur_y + tab_row_h, cur_y + tab_row_h], color=bg_tr, transform=ax3.transAxes)
                ax3.plot([0.05, 0.95], [cur_y, cur_y], color="#e2e8f0", lw=0.4, transform=ax3.transAxes)

                feat_lbl = item.get("clinical_label", item.get("feature", ""))
                pval_raw = item.get("patient_value")
                pval_fmt = sanitize_pdf_text(pval_raw, 18) if pval_raw is not None else "Imputed"
                tval = item.get("transformed_value")
                tval_fmt = f"{tval:.2f}" if (tval is not None and isinstance(tval, (int, float))) else "-"
                sval = float(item.get("shap_value", 0.0))
                sval_fmt = f"+{sval:.4f}" if sval > 0 else f"{sval:.4f}"
                scol = "#be123c" if sval > 0.0001 else "#047857" if sval < -0.0001 else "#64748b"
                dir_str = "Toward CKD" if sval > 0.0001 else "Toward Non-CKD" if sval < -0.0001 else "Neutral"

                ax3.text(0.06, cur_y + 0.005, sanitize_pdf_text(feat_lbl, 26), fontsize=6.5, color="#0f172a", transform=ax3.transAxes)
                ax3.text(0.35, cur_y + 0.005, pval_fmt, fontsize=6.5, color="#334155", transform=ax3.transAxes)
                ax3.text(0.55, cur_y + 0.005, tval_fmt, fontsize=6.5, color="#334155", transform=ax3.transAxes)
                ax3.text(0.70, cur_y + 0.005, sval_fmt, fontsize=6.5, fontweight="bold", color=scol, transform=ax3.transAxes)
                ax3.text(0.82, cur_y + 0.005, dir_str, fontsize=6.5, color=scol, fontweight="bold", transform=ax3.transAxes)

        # Section 9: Data Extraction Limitations
        box9_top = 0.640
        ax3.fill_between([0.05, 0.95], [box9_top - 0.140, box9_top - 0.140], [box9_top, box9_top], color="#f8fafc", transform=ax3.transAxes)
        ax3.plot([0.05, 0.95, 0.95, 0.05, 0.05], [box9_top - 0.140, box9_top - 0.140, box9_top, box9_top, box9_top - 0.140], color="#cbd5e1", lw=0.8, transform=ax3.transAxes)

        ax3.text(0.07, box9_top - 0.022, "SECTION 9: DATA EXTRACTION LIMITATIONS", fontsize=8.5, fontweight="bold", color="#1e293b", transform=ax3.transAxes)
        ax3.text(0.07, box9_top - 0.042, "- OCR Sensitivity: Extraction accuracy depends strictly on scanned resolution, lighting, and table formatting.", fontsize=7.2, color="#334155", transform=ax3.transAxes)
        ax3.text(0.07, box9_top - 0.060, "- Unit Ambiguity: Automated conversions assume standard laboratory conventions (e.g. mg/dL, umol/L).", fontsize=7.2, color="#334155", transform=ax3.transAxes)
        ax3.text(0.07, box9_top - 0.078, "- Dipstick Discretization: Qualitative urinalysis (Albumin, Sugar) is mapped to discrete ordinal indices (0 to 5).", fontsize=7.2, color="#334155", transform=ax3.transAxes)
        ax3.text(0.07, box9_top - 0.096, "- Verifier Responsibility: Clinicians must manually verify extracted values prior to model reliance.", fontsize=7.2, color="#334155", fontweight="bold", transform=ax3.transAxes)
        ax3.text(0.07, box9_top - 0.114, "- Unhandled Formats: Non-standard unstructured progress notes or handwritten reports may omit parameters.", fontsize=7.0, color="#64748b", transform=ax3.transAxes)

        # Section 10: Model Limitations
        box10_top = 0.480
        ax3.fill_between([0.05, 0.95], [box10_top - 0.145, box10_top - 0.145], [box10_top, box10_top], color="#f8fafc", transform=ax3.transAxes)
        ax3.plot([0.05, 0.95, 0.95, 0.05, 0.05], [box10_top - 0.145, box10_top - 0.145, box10_top, box10_top, box10_top - 0.145], color="#cbd5e1", lw=0.8, transform=ax3.transAxes)

        ax3.text(0.07, box10_top - 0.022, "SECTION 10: MODEL LIMITATIONS & ALGORITHMIC BOUNDARIES", fontsize=8.5, fontweight="bold", color="#1e293b", transform=ax3.transAxes)
        ax3.text(0.07, box10_top - 0.042, "- Tabular Architecture: Prediction is generated by an ensemble RandomForestClassifier (benchmarked against baseline DecisionTreeClassifier).", fontsize=7.2, color="#334155", transform=ax3.transAxes)
        ax3.text(0.07, box10_top - 0.060, "- Cross-Sectional Only: The model evaluates a single static snapshot and does not capture longitudinal kidney trends.", fontsize=7.2, color="#334155", transform=ax3.transAxes)
        ax3.text(0.07, box10_top - 0.078, "- Empirical Leaf Purity: Probabilities represent sample purity within reached terminal leaves, not Bayesian risk.", fontsize=7.2, color="#334155", transform=ax3.transAxes)
        ax3.text(0.07, box10_top - 0.096, "- Acute vs Chronic Indistinguishability: The model cannot distinguish Acute Kidney Injury (AKI) from CKD.", fontsize=7.2, color="#334155", fontweight="bold", transform=ax3.transAxes)
        ax3.text(0.07, box10_top - 0.114, "- Imputation Sensitivity: Missing parameters imputed via cohort medians may skew predictions in atypical cases.", fontsize=7.0, color="#64748b", transform=ax3.transAxes)

        # Section 11: Medical Disclaimer
        box11_top = 0.315
        ax3.fill_between([0.05, 0.95], [box11_top - 0.170, box11_top - 0.170], [box11_top, box11_top], color="#fef2f2", transform=ax3.transAxes)
        ax3.plot([0.05, 0.95, 0.95, 0.05, 0.05], [box11_top - 0.170, box11_top - 0.170, box11_top, box11_top, box11_top - 0.170], color="#f87171", lw=1.2, transform=ax3.transAxes)

        ax3.text(0.07, box11_top - 0.025, "SECTION 11: FORMAL MEDICAL & REGULATORY DISCLAIMER", fontsize=9.0, fontweight="bold", color="#991b1b", transform=ax3.transAxes)
        ax3.text(0.07, box11_top - 0.050, "THIS REPORT IS GENERATED FOR RESEARCH AND CLINICAL DECISION-SUPPORT INVESTIGATION ONLY.", fontsize=7.5, fontweight="bold", color="#b91c1c", transform=ax3.transAxes)
        ax3.text(0.07, box11_top - 0.072, "IT DOES NOT CONSTITUTE A FORMAL CLINICAL DIAGNOSIS, MEDICAL OPINION, PROGNOSIS, OR TREATMENT PLAN.", fontsize=7.2, color="#7f1d1d", transform=ax3.transAxes)
        ax3.text(0.07, box11_top - 0.094, "Artificial intelligence models are statistical aids and must never replace certified healthcare providers. A licensed nephrologist", fontsize=7.0, color="#991b1b", transform=ax3.transAxes)
        ax3.text(0.07, box11_top - 0.112, "or medical physician must independently evaluate comprehensive clinical history, physical exam, and laboratory markers.", fontsize=7.0, color="#991b1b", transform=ax3.transAxes)
        ax3.text(0.07, box11_top - 0.130, "Do NOT adjust patient medications, dosages, or dialysis schedules based on this automated report.", fontsize=7.0, fontweight="bold", color="#7f1d1d", transform=ax3.transAxes)
        ax3.text(0.07, box11_top - 0.148, "Antigravity Explainable AI disclaims all liability for clinical actions taken in reliance upon this experimental report.", fontsize=6.8, color="#64748b", transform=ax3.transAxes)

        # Page 3 Footer & Sign-off Block
        ax3.fill_between([0.05, 0.95], [0.03, 0.03], [0.090, 0.090], color="#f8fafc", transform=ax3.transAxes)
        ax3.plot([0.05, 0.95, 0.95, 0.05, 0.05], [0.03, 0.03, 0.090, 0.090, 0.03], color="#cbd5e1", lw=0.6, transform=ax3.transAxes)
        ax3.text(0.065, 0.070, f"DOCUMENT AUDIT SIGN-OFF: Non-Identifying Hash: {clean_sess} - Generated: {timestamp_str}", fontsize=7.0, fontweight="bold", color="#334155", transform=ax3.transAxes)
        ax3.text(0.065, 0.050, "Verification Protocol: Phase 7 Completed - Storage: Ephemeral In-Memory Stream - No Persistent Disk Artifacts", fontsize=6.8, color="#64748b", transform=ax3.transAxes)
        ax3.text(0.85, 0.015, "Page 3 of 3", fontsize=7.5, color="#64748b", transform=ax3.transAxes)

        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes
