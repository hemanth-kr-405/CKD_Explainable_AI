/**
 * Explainable AI for Chronic Kidney Disease Prediction
 * Frontend Application Logic (Vanilla JavaScript ES6+)
 */

// Determine API Base URL dynamically (supports cloud domains, local ports, and filesystem file:/// testing)
const API_BASE = window.location.protocol.startsWith('http')
    ? window.location.origin
    : 'http://127.0.0.1:8000';

let featuresSchema = null;

// ============================================================================
// Initialization on DOM Ready
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    checkBackendHealth();
    loadFeatureMetadata();
    loadModelMetrics();
    setupEventListeners();
});

// ============================================================================
// Health Check & Schema Loading
// ============================================================================

async function checkBackendHealth() {
    const statusEl = document.getElementById('system-status');
    const dotEl = statusEl.querySelector('.status-dot');

    try {
        const response = await fetch(`${API_BASE}/health`, { method: 'GET' });
        if (response.ok) {
            const data = await response.json();
            dotEl.className = 'status-dot online';
            statusEl.innerHTML = `<span class="status-dot online"></span> API Ready (${data.algorithm})`;
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (err) {
        dotEl.className = 'status-dot offline';
        statusEl.innerHTML = `<span class="status-dot offline"></span> API Disconnected`;
        console.warn('Backend service offline or unreachable:', err);
    }
}

async function loadFeatureMetadata() {
    try {
        const response = await fetch(`${API_BASE}/features`);
        if (response.ok) {
            featuresSchema = await response.json();
            console.log('Feature schema loaded:', featuresSchema.total_features, 'features');
        }
    } catch (err) {
        console.warn('Failed to load features metadata:', err);
    }
}

async function loadModelMetrics() {
    try {
        const response = await fetch(`${API_BASE}/metrics`);
        if (response.ok) {
            const data = await response.json();
            if (data.test_metrics) {
                const tm = data.test_metrics;
                document.getElementById('metric-accuracy').textContent = (tm.accuracy * 100).toFixed(1) + '%';
                document.getElementById('metric-precision').textContent = (tm.precision * 100).toFixed(1) + '%';
                document.getElementById('metric-recall').textContent = (tm.recall_sensitivity * 100).toFixed(1) + '%';
                document.getElementById('metric-specificity').textContent = (tm.specificity * 100).toFixed(1) + '%';
                document.getElementById('metric-roc-auc').textContent = tm.roc_auc.toFixed(4);
            }
            if (data.gini_feature_importances) {
                renderGlobalImportance(data.gini_feature_importances);
            }
        }
    } catch (err) {
        console.warn('Failed to load metrics:', err);
    }
}

function renderGlobalImportance(importances) {
    const listEl = document.getElementById('global-ranking-list');
    listEl.innerHTML = '';

    const entries = Object.entries(importances).slice(0, 6);
    const maxVal = Math.max(...entries.map(e => e[1])) || 1.0;

    const labelMap = {
        'hemo': 'Hemoglobin (g/dL)',
        'sg': 'Specific Gravity',
        'rbcc': 'Red Blood Cell Count',
        'htn': 'Hypertension',
        'pcv': 'Packed Cell Volume (%)',
        'sc': 'Serum Creatinine (mg/dL)'
    };

    entries.forEach(([feat, val]) => {
        const pct = (val / maxVal) * 100;
        const item = document.createElement('div');
        item.className = 'ranking-item';
        item.innerHTML = `
            <div class="ranking-meta">
                <span class="ranking-label">${labelMap[feat] || feat}</span>
                <span class="ranking-val">${(val * 100).toFixed(1)}% influence</span>
            </div>
            <div class="ranking-bar-track">
                <div class="ranking-bar-fill" style="width: ${pct}%;"></div>
            </div>
        `;
        listEl.appendChild(item);
    });
}

// ============================================================================
// Event Listeners & Presets
// ============================================================================

function setupEventListeners() {
    const form = document.getElementById('ckd-form');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);

        // Real-time error dismissal when user corrects invalid inputs
        form.querySelectorAll('input, select').forEach(input => {
            input.addEventListener('input', () => {
                const group = input.closest('.form-group');
                if (group && group.classList.contains('has-error')) {
                    group.classList.remove('has-error');
                }
            });
            input.addEventListener('change', () => {
                const group = input.closest('.form-group');
                if (group && group.classList.contains('has-error')) {
                    group.classList.remove('has-error');
                }
            });
        });
    }

    const btnReset = document.getElementById('btn-reset');
    if (btnReset) btnReset.addEventListener('click', handleReset);

    const btnSubmit = document.getElementById('btn-submit');
    const btnProceed = document.getElementById('btn-proceed');
    if (btnProceed && btnProceed !== btnSubmit) {
        btnProceed.addEventListener('click', handleFormSubmit);
    }

    const btnPresetCkd = document.getElementById('btn-preset-ckd');
    if (btnPresetCkd) btnPresetCkd.addEventListener('click', applyCkdPreset);

    const btnPresetHealthy = document.getElementById('btn-preset-healthy');
    if (btnPresetHealthy) btnPresetHealthy.addEventListener('click', applyHealthyPreset);

    // Phase 2: Document upload listeners
    setupUploadListeners();

    // Phase 7: SHAP 24-Feature Matrix toggle
    const btnToggleShapTable = document.getElementById('btn-toggle-shap-table');
    const shapTableWrapper = document.getElementById('shap-table-wrapper');
    if (btnToggleShapTable && shapTableWrapper) {
        btnToggleShapTable.addEventListener('click', () => {
            const isHidden = shapTableWrapper.style.display === 'none' || shapTableWrapper.style.display === '';
            shapTableWrapper.style.display = isHidden ? 'block' : 'none';
            btnToggleShapTable.innerHTML = isHidden 
                ? '<span>Hide 24-Feature Matrix</span>' 
                : '<span>Show All 24 Features</span>';
        });
    }
}

const PRESET_CKD = {
    age: 65,
    bp: 90,
    sg: "1.010",
    al: "4",
    su: "2",
    bgr: 172,
    bu: 82,
    sc: 13.5,
    sod: 145,
    pot: 6.3,
    hemo: 8.8,
    pcv: 31,
    wbcc: 8100,
    rbcc: 4.8,
    rbc: "normal",
    pc: "normal",
    pcc: "notpresent",
    ba: "notpresent",
    htn: "yes",
    dm: "yes",
    cad: "no",
    appet: "good",
    pe: "yes",
    ane: "yes"
};

const PRESET_HEALTHY = {
    age: 44,
    bp: 60,
    sg: "1.020",
    al: "0",
    su: "0",
    bgr: 95,
    bu: 46,
    sc: 0.5,
    sod: 138,
    pot: 4.2,
    hemo: 15.0,
    pcv: 50,
    wbcc: 7700,
    rbcc: 6.3,
    rbc: "normal",
    pc: "normal",
    pcc: "notpresent",
    ba: "notpresent",
    htn: "no",
    dm: "no",
    cad: "no",
    appet: "good",
    pe: "no",
    ane: "no"
};

function populateForm(data) {
    clearValidationErrors();
    const form = document.getElementById('ckd-form');
    if (!form) return;

    // Ensure all elements are enabled and editable
    const formControls = form.querySelectorAll('input, select, textarea');
    formControls.forEach(el => {
        el.disabled = false;
        el.readOnly = false;
    });

    Object.keys(data).forEach(key => {
        const input = form.elements[key];
        if (input) {
            input.value = data[key];
        }
    });
}

function applyCkdPreset() {
    populateForm(PRESET_CKD);
}

function applyHealthyPreset() {
    populateForm(PRESET_HEALTHY);
}

function handleReset() {
    const form = document.getElementById('ckd-form');
    if (form) {
        form.reset();

        // Explicitly clear all patient-entered values and restore input interactivity
        const formControls = form.querySelectorAll('input, select, textarea');
        formControls.forEach(el => {
            el.value = '';
            el.disabled = false;
            el.readOnly = false;
        });

        // Ensure all parent fieldsets are interactive
        const fieldsets = form.querySelectorAll('fieldset');
        fieldsets.forEach(fs => {
            fs.disabled = false;
        });
    }

    // Clear stale validation errors
    clearValidationErrors();

    // Reset button states
    const submitBtn = document.getElementById('btn-submit');
    if (submitBtn) submitBtn.disabled = false;
    const proceedBtn = document.getElementById('btn-proceed');
    if (proceedBtn) proceedBtn.disabled = false;

    // Reset loading and display states; remove stale prediction results
    const placeholderState = document.getElementById('placeholder-state');
    const loadingState = document.getElementById('loading-state');
    const outputContainer = document.getElementById('output-container');
    if (placeholderState) placeholderState.style.display = 'block';
    if (loadingState) loadingState.style.display = 'none';
    if (outputContainer) outputContainer.style.display = 'none';

    currentPredictionResult = null;
    currentVerificationParameters = {};
    activeUpload = null;

    const verifyCard = document.getElementById('verification-card');
    const reviewCard = document.getElementById('review-card');
    const fileCard = document.getElementById('uploaded-file-card');
    const extractAction = document.getElementById('upload-extract-action');
    const fileInput = document.getElementById('medical-file-input');
    if (verifyCard) verifyCard.style.display = 'none';
    if (reviewCard) reviewCard.style.display = 'none';
    if (fileCard) fileCard.style.display = 'none';
    if (extractAction) extractAction.style.display = 'none';
    if (fileInput) fileInput.value = '';

    clearUploadAlerts();
    clearVerificationAlert();
    setWorkflowStep(1);
}

function clearValidationErrors() {
    const errorAlert = document.getElementById('form-error-alert');
    if (errorAlert) {
        errorAlert.style.display = 'none';
        errorAlert.textContent = '';
    }
    document.querySelectorAll('.form-group.has-error').forEach(el => {
        el.classList.remove('has-error');
    });
}

// ============================================================================
// Form Validation & Prediction Workflow
// ============================================================================

async function handleFormSubmit(e) {
    if (e && e.preventDefault) {
        e.preventDefault();
    }
    clearValidationErrors();

    const form = document.getElementById('ckd-form');
    if (!form) return;

    const formData = new FormData(form);
    const payload = {};
    const validationErrors = [];

    // Ensure all inputs remain enabled and editable
    const formControls = form.querySelectorAll('input, select, textarea');
    formControls.forEach(el => {
        el.disabled = false;
        el.readOnly = false;
    });

    // 1. Required Clinical Fields Validation
    // Serum Creatinine (sc) and Hemoglobin (hemo) are mandatory core clinical biomarkers
    const scVal = formData.get('sc');
    if (scVal === null || scVal === undefined || scVal.trim() === '') {
        validationErrors.push('Serum Creatinine (sc) is required to evaluate renal filtration.');
        document.getElementById('group-sc')?.classList.add('has-error');
    }

    const hemoVal = formData.get('hemo');
    if (hemoVal === null || hemoVal === undefined || hemoVal.trim() === '') {
        validationErrors.push('Hemoglobin (hemo) is required to evaluate renal anemia.');
        document.getElementById('group-hemo')?.classList.add('has-error');
    }

    // Numerical feature bounds definitions
    const numBounds = {
        age: [1, 120],
        bp: [40, 250],
        bgr: [20, 600],
        bu: [1, 500],
        sc: [0.1, 100.0],
        sod: [50, 200],
        pot: [1.0, 60.0],
        hemo: [2.0, 25.0],
        pcv: [5, 70],
        wbcc: [500, 50000],
        rbcc: [0.5, 12.0]
    };

    // Extract and validate numerical inputs
    for (const [key, [min, max]] of Object.entries(numBounds)) {
        const rawVal = formData.get(key);
        if (rawVal !== null && rawVal.trim() !== '') {
            const num = parseFloat(rawVal);
            if (isNaN(num)) {
                validationErrors.push(`${key}: must be a valid number`);
                document.getElementById(`group-${key}`)?.classList.add('has-error');
            } else if (num < min || num > max) {
                validationErrors.push(`${key}: must be between ${min} and ${max}`);
                document.getElementById(`group-${key}`)?.classList.add('has-error');
            } else {
                payload[key] = num;
            }
        }
    }

    // Extract dropdowns
    const dropdownKeys = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'];
    dropdownKeys.forEach(key => {
        const val = formData.get(key);
        if (val && val.trim() !== '') {
            if (key === 'sg') {
                payload[key] = parseFloat(val);
            } else if (key === 'al' || key === 'su') {
                payload[key] = parseFloat(val);
            } else {
                payload[key] = val;
            }
        }
    });

    // Client-side validation display
    if (validationErrors.length > 0) {
        const errorAlert = document.getElementById('form-error-alert');
        if (errorAlert) {
            errorAlert.innerHTML = `<strong>Please fix input errors before proceeding:</strong><br>${validationErrors.join('<br>')}`;
            errorAlert.style.display = 'block';
        }
        // Ensure all inputs and buttons remain enabled and interactive
        formControls.forEach(el => {
            el.disabled = false;
            el.readOnly = false;
        });
        const submitBtn = document.getElementById('btn-submit');
        if (submitBtn) submitBtn.disabled = false;
        return;
    }

    payload.include_plot = true;

    // Show Loading State
    const placeholderState = document.getElementById('placeholder-state');
    const loadingState = document.getElementById('loading-state');
    const outputContainer = document.getElementById('output-container');
    const submitBtn = document.getElementById('btn-submit');

    if (placeholderState) placeholderState.style.display = 'none';
    if (outputContainer) outputContainer.style.display = 'none';
    if (loadingState) loadingState.style.display = 'block';
    if (submitBtn) submitBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Prediction failed with HTTP status ${response.status}.`);
        }

        const result = await response.json();
        renderPredictionResults(result);

    } catch (err) {
        console.error('API Error:', err);
        const errorAlert = document.getElementById('form-error-alert');
        if (errorAlert) {
            errorAlert.innerHTML = `<strong>Prediction Error:</strong> ${err.message || 'Could not connect to backend server. Make sure FastAPI is running.'}`;
            errorAlert.style.display = 'block';
        }
        if (placeholderState) placeholderState.style.display = 'block';
    } finally {
        if (loadingState) loadingState.style.display = 'none';
        if (submitBtn) submitBtn.disabled = false;
        // Guarantee input interactivity is fully restored
        formControls.forEach(el => {
            el.disabled = false;
            el.readOnly = false;
        });
        const fieldsets = form.querySelectorAll('fieldset');
        fieldsets.forEach(fs => {
            fs.disabled = false;
        });
    }
}

// ============================================================================
// Render Results & SHAP Explanations
// ============================================================================

function renderPredictionResults(data) {
    currentPredictionResult = data;
    setWorkflowStep(4);

    const outputContainer = document.getElementById('output-container');
    const outcomeCard = document.getElementById('outcome-card');
    const outcomeLabel = document.getElementById('outcome-label');
    const confidenceBadge = document.getElementById('confidence-badge');
    const outcomeNarrative = document.getElementById('outcome-narrative');
    const probBarFill = document.getElementById('prob-bar-fill');
    const ckdProbText = document.getElementById('ckd-prob-text');
    const nonCkdProbText = document.getElementById('non-ckd-prob-text');

    const isCkd = data.prediction_code === 1;
    const ckdProb = data.probabilities.ckd_probability;
    const nonCkdProb = data.probabilities.non_ckd_probability;

    // Outcome Card Styling
    outcomeCard.className = `outcome-card ${isCkd ? 'outcome-ckd' : 'outcome-healthy'}`;
    outcomeLabel.textContent = data.prediction_label;
    confidenceBadge.textContent = `${(Math.max(ckdProb, nonCkdProb) * 100).toFixed(1)}% Leaf Probability (Uncalibrated)`;
    outcomeNarrative.textContent = data.clinical_summary;

    // Probability Bar
    const ckdPct = (ckdProb * 100).toFixed(1);
    const nonCkdPct = (nonCkdProb * 100).toFixed(1);
    ckdProbText.textContent = `${ckdPct}%`;
    nonCkdProbText.textContent = `${nonCkdPct}%`;
    probBarFill.style.width = isCkd ? `${ckdPct}%` : `${nonCkdPct}%`;

    // Baseline Pill & Guidance Box (Phase 7)
    const baseValStr = (data.explanation.base_value !== undefined && data.explanation.base_value !== null) ? data.explanation.base_value.toFixed(4) : '0.6250';
    const finalValStr = (data.explanation.patient_model_output !== undefined && data.explanation.patient_model_output !== null) ? data.explanation.patient_model_output.toFixed(4) : '1.0000';

    const shapBaseEl = document.getElementById('shap-base-val');
    const shapFinalEl = document.getElementById('shap-final-val');
    const guidanceBaseEl = document.getElementById('guidance-base-val');
    const guidanceFinalEl = document.getElementById('guidance-final-val');

    if (shapBaseEl) shapBaseEl.textContent = baseValStr;
    if (shapFinalEl) shapFinalEl.textContent = finalValStr;
    if (guidanceBaseEl) guidanceBaseEl.textContent = baseValStr;
    if (guidanceFinalEl) guidanceFinalEl.textContent = finalValStr;

    // Top Risk Drivers
    const riskContainer = document.getElementById('risk-factors-list');
    riskContainer.innerHTML = '';
    const riskDrivers = data.explanation.top_risk_drivers;

    if (riskDrivers && riskDrivers.length > 0) {
        riskDrivers.forEach(factor => {
            const card = document.createElement('div');
            card.className = 'factor-card factor-risk';
            card.innerHTML = `
                <div class="factor-info">
                    <span class="factor-label">${factor.clinical_label}</span>
                    <span class="factor-val">Patient value: <strong>${factor.patient_value}</strong></span>
                </div>
                <span class="factor-badge">+${factor.shap_value.toFixed(4)}</span>
            `;
            riskContainer.appendChild(card);
        });
    } else {
        riskContainer.innerHTML = '<div class="subtext" style="padding: 0.5rem 0;">No significant features increasing predicted probability observed.</div>';
    }

    // Top Protective Factors
    const protectiveContainer = document.getElementById('protective-factors-list');
    protectiveContainer.innerHTML = '';
    const protectiveFactors = data.explanation.top_protective_factors;

    if (protectiveFactors && protectiveFactors.length > 0) {
        protectiveFactors.forEach(factor => {
            const card = document.createElement('div');
            card.className = 'factor-card factor-protective';
            card.innerHTML = `
                <div class="factor-info">
                    <span class="factor-label">${factor.clinical_label}</span>
                    <span class="factor-val">Patient value: <strong>${factor.patient_value}</strong></span>
                </div>
                <span class="factor-badge">${factor.shap_value.toFixed(4)}</span>
            `;
            protectiveContainer.appendChild(card);
        });
    } else {
        protectiveContainer.innerHTML = '<div class="subtext" style="padding: 0.5rem 0;">No significant features decreasing predicted probability observed.</div>';
    }

    // ========================================================================
    // Phase 7: Interactive Horizontal Feature-Contribution Divergence Chart
    // ========================================================================
    const chartBody = document.getElementById('shap-horizontal-chart');
    if (chartBody) {
        chartBody.innerHTML = '';
        const influentialFeatures = (data.explanation.most_influential_features && data.explanation.most_influential_features.length > 0)
            ? data.explanation.most_influential_features.slice(0, 8)
            : [...(data.explanation.all_feature_contributions || [])].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)).slice(0, 8);

        if (influentialFeatures.length > 0) {
            const maxShap = Math.max(...influentialFeatures.map(f => Math.abs(f.shap_value)), 0.01);

            influentialFeatures.forEach(feat => {
                const phi = feat.shap_value;
                const absPhi = Math.abs(phi);
                const pct = Math.min(100, Math.max(2, Math.round((absPhi / maxShap) * 100)));
                const isPos = phi > 0.0001;
                const isNeg = phi < -0.0001;

                const row = document.createElement('div');
                row.className = 'hbar-row';

                const patValDisplay = (feat.patient_value !== null && feat.patient_value !== undefined)
                    ? `${feat.patient_value}${feat.unit ? ' ' + feat.unit : ''}`
                    : 'Imputed/Missing';

                let barContainerHtml = '';
                if (isNeg) {
                    barContainerHtml = `
                        <div class="hbar-bar-container-left">
                            <div class="hbar-fill-neg" style="width: ${pct}%;"></div>
                        </div>
                    `;
                } else if (isPos) {
                    barContainerHtml = `
                        <div class="hbar-bar-container-right">
                            <div class="hbar-fill-pos" style="width: ${pct}%;"></div>
                        </div>
                    `;
                }

                const scoreClass = isPos ? 'score-pos' : isNeg ? 'score-neg' : 'score-zero';
                const scoreText = (phi > 0 ? '+' : '') + phi.toFixed(4);

                row.innerHTML = `
                    <div class="hbar-label">
                        <span class="hbar-name" title="${escapeHtml(feat.clinical_label || feat.feature)}">${escapeHtml(feat.clinical_label || feat.feature)}</span>
                        <span class="hbar-val">Patient: ${escapeHtml(patValDisplay)}</span>
                    </div>
                    <div class="hbar-track">
                        <div class="hbar-center-line"></div>
                        ${barContainerHtml}
                    </div>
                    <span class="hbar-score ${scoreClass}">${scoreText}</span>
                `;
                chartBody.appendChild(row);
            });
        } else {
            chartBody.innerHTML = '<div class="subtext">No significant feature contributions calculated.</div>';
        }
    }

    // ========================================================================
    // Phase 7: Complete 24-Feature Attribution Matrix Table
    // ========================================================================
    const tableBody = document.getElementById('shap-table-body');
    if (tableBody) {
        tableBody.innerHTML = '';
        const allFeatures = [...(data.explanation.all_feature_contributions || [])].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));

        if (allFeatures.length > 0) {
            allFeatures.forEach(feat => {
                const tr = document.createElement('tr');
                const isPos = feat.shap_value > 0.0001;
                const isNeg = feat.shap_value < -0.0001;
                const scoreClass = isPos ? 'score-pos' : isNeg ? 'score-neg' : 'score-zero';
                const pillClass = isPos ? 'pill-pos' : isNeg ? 'pill-neg' : 'pill-neutral';
                const directionLabel = isPos ? '🔺 Towards CKD' : isNeg ? '🔻 Towards Non-CKD' : '— Neutral';

                const patValDisplay = (feat.patient_value !== null && feat.patient_value !== undefined)
                    ? `${escapeHtml(String(feat.patient_value))}${feat.unit ? ' ' + escapeHtml(feat.unit) : ''}`
                    : '<span style="color:#94a3b8;font-style:italic;">Imputed/Missing</span>';

                const transformedValDisplay = (feat.transformed_value !== null && feat.transformed_value !== undefined)
                    ? Number(feat.transformed_value).toFixed(2)
                    : '—';

                tr.innerHTML = `
                    <td>
                        <strong>${escapeHtml(feat.clinical_label || feat.feature)}</strong>
                        <small style="display:block;color:#64748b;font-family:var(--font-mono);font-size:0.7rem;">${escapeHtml(feat.feature)}</small>
                    </td>
                    <td>${patValDisplay}</td>
                    <td class="col-mono">${transformedValDisplay}</td>
                    <td class="col-mono ${scoreClass}">${feat.shap_value > 0 ? '+' : ''}${feat.shap_value.toFixed(4)}</td>
                    <td>
                        <span class="pill-direction ${pillClass}">${directionLabel}</span>
                    </td>
                `;
                tableBody.appendChild(tr);
            });
        } else {
            tableBody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#64748b;">No feature attribution records found.</td></tr>';
        }
    }

    // Waterfall Plot Image
    const waterfallCard = document.getElementById('waterfall-container');
    const waterfallImg = document.getElementById('waterfall-img');

    if (data.waterfall_plot_base64) {
        waterfallImg.src = `data:image/png;base64,${data.waterfall_plot_base64}`;
        waterfallCard.style.display = 'block';
    } else {
        waterfallCard.style.display = 'none';
    }

    // Phase 6: Clinical Distinction & Medical Governance Card
    const distinctionCard = document.getElementById('clinical-distinction-card');
    if (distinctionCard) {
        if (data.clinical_distinction) {
            document.getElementById('distinction-prediction').textContent = data.clinical_distinction.model_prediction;
            document.getElementById('distinction-probability').textContent = data.clinical_distinction.model_probability;
            document.getElementById('distinction-diagnosis').textContent = data.clinical_distinction.clinical_diagnosis;
            distinctionCard.style.display = 'block';
        } else {
            // Standard fallback when invoking direct /predict
            document.getElementById('distinction-prediction').textContent = `DecisionTree classification: ${data.prediction_label}`;
            document.getElementById('distinction-probability').textContent = `Terminal leaf proportion: ${(Math.max(ckdProb, nonCkdProb) * 100).toFixed(1)}% (Uncalibrated sample purity)`;
            document.getElementById('distinction-diagnosis').textContent = 'CANNOT be established by machine learning. A definitive medical diagnosis requires formal clinical evaluation by a licensed physician.';
            distinctionCard.style.display = 'block';
        }
    }

    // Phase 6: Provenance & Imputation Summary Card
    const provenanceCard = document.getElementById('provenance-summary-card');
    const provenanceBadges = document.getElementById('provenance-badges');
    const provenanceNote = document.getElementById('provenance-note');
    if (provenanceCard && provenanceBadges && provenanceNote) {
        if (data.provenance_summary) {
            const p = data.provenance_summary;
            provenanceBadges.innerHTML = `
                <span class="provenance-badge pbadge-verified">✓ ${p.verified_count} Verified</span>
                <span class="provenance-badge pbadge-extracted">📄 ${p.extracted_count} Extracted</span>
                <span class="provenance-badge pbadge-manual">✍️ ${p.manual_count} Manual</span>
                <span class="provenance-badge pbadge-imputed">⚙️ ${p.missing_count} Imputed (${p.imputation_applied ? 'Active' : 'None'})</span>
            `;
            provenanceNote.textContent = `Imputation Methodology: ${p.imputation_strategy}. Missing features remain unpopulated and are never fabricated with arbitrary zeros.`;
            provenanceCard.style.display = 'block';
        } else {
            provenanceCard.style.display = 'none';
        }
    }

    // Hide Placeholder & Loading; Show Container with smooth entrance
    const placeholderState = document.getElementById('placeholder-state');
    const loadingState = document.getElementById('loading-state');
    if (placeholderState) placeholderState.style.display = 'none';
    if (loadingState) loadingState.style.display = 'none';
    outputContainer.style.display = 'flex';
}

// ============================================================================
// Phase 2: Secure Medical Document Upload Logic
// ============================================================================

let activeUpload = null;
let currentPredictionResult = null;
let currentWorkflowStep = 1;

function setWorkflowStep(stepNumber) {
    currentWorkflowStep = stepNumber;
    for (let i = 1; i <= 5; i++) {
        const node = document.getElementById(`step-node-${i}`);
        if (!node) continue;

        node.classList.remove('active', 'completed', 'pending');
        const badge = node.querySelector('.step-badge');

        if (i < stepNumber) {
            node.classList.add('completed');
            if (badge) badge.innerHTML = '✓';
        } else if (i === stepNumber) {
            node.classList.add('active');
            if (badge) badge.textContent = String(i);
        } else {
            node.classList.add('pending');
            if (badge) badge.textContent = String(i);
        }

        // Connector lines
        if (i < 5) {
            const conn = document.getElementById(`step-conn-${i}`);
            if (conn) {
                conn.classList.remove('completed', 'active');
                if (i < stepNumber) {
                    conn.classList.add('completed');
                } else if (i === stepNumber) {
                    conn.classList.add('active');
                }
            }
        }
    }
}

function setupUploadListeners() {
    const fileInput = document.getElementById('medical-file-input');
    const dropzone = document.getElementById('upload-dropzone');
    const removeBtn = document.getElementById('btn-remove-file');

    if (!fileInput || !dropzone) return;

    // Click on dropzone triggers file picker
    dropzone.addEventListener('click', (e) => {
        // Avoid double triggering if click was directly on the file input
        if (e.target !== fileInput) {
            fileInput.click();
        }
    });

    // Keyboard accessibility for dropzone
    dropzone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });

    // File selected via standard file picker
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            uploadDocument(e.target.files[0]);
        }
    });

    // Drag & Drop events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-active');
        });
    });

    ['dragleave', 'dragend', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-active');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files.length > 0) {
            uploadDocument(dt.files[0]);
        }
    });

    // Cancel / Remove uploaded file button
    if (removeBtn) {
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            handleRemoveUpload();
        });
    }

    // Phase 5: Extract & Verify Button
    const extractVerifyBtn = document.getElementById('btn-extract-verify');
    if (extractVerifyBtn) {
        extractVerifyBtn.addEventListener('click', handleExtractAndVerify);
    }

    // Phase 5: Verification Action Buttons
    const btnCancel = document.getElementById('btn-verify-cancel');
    if (btnCancel) btnCancel.addEventListener('click', handleVerificationCancel);

    const btnEdit = document.getElementById('btn-verify-edit');
    if (btnEdit) btnEdit.addEventListener('click', handleVerificationEditAll);

    const btnConfirm = document.getElementById('btn-verify-confirm');
    if (btnConfirm) btnConfirm.addEventListener('click', handleVerificationConfirmAll);

    const btnProceed = document.getElementById('btn-verify-proceed');
    if (btnProceed) btnProceed.addEventListener('click', handleVerificationProceed);

    // Phase 8: Review & Report Action Handlers
    const btnReviewBack = document.getElementById('btn-review-back');
    if (btnReviewBack) btnReviewBack.addEventListener('click', handleReviewBack);

    const btnReviewConfirm = document.getElementById('btn-review-confirm-predict');
    if (btnReviewConfirm) btnReviewConfirm.addEventListener('click', handleConfirmPrediction);

    const btnDownloadReport = document.getElementById('btn-download-pdf-report');
    if (btnDownloadReport) btnDownloadReport.addEventListener('click', handleDownloadReport);

    const btnPrintSummary = document.getElementById('btn-print-summary');
    if (btnPrintSummary) btnPrintSummary.addEventListener('click', () => window.print());

    const btnRestartWorkflow = document.getElementById('btn-restart-workflow');
    if (btnRestartWorkflow) btnRestartWorkflow.addEventListener('click', handleRestartWorkflow);
}

function clearUploadAlerts() {
    const successAlert = document.getElementById('upload-success-alert');
    const errorAlert = document.getElementById('upload-error-alert');
    if (successAlert) {
        successAlert.style.display = 'none';
        successAlert.innerHTML = '';
    }
    if (errorAlert) {
        errorAlert.style.display = 'none';
        errorAlert.innerHTML = '';
    }
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

async function uploadDocument(file) {
    clearUploadAlerts();

    const successAlert = document.getElementById('upload-success-alert');
    const errorAlert = document.getElementById('upload-error-alert');
    const progressContainer = document.getElementById('upload-progress-container');
    const progressBar = document.getElementById('upload-progress-bar');
    const statusText = document.getElementById('upload-status-text');
    const fileCard = document.getElementById('uploaded-file-card');
    const dropzone = document.getElementById('upload-dropzone');
    const fileNameDisplay = document.getElementById('file-name-display');
    const fileSizeDisplay = document.getElementById('file-size-display');
    const fileTypeIcon = document.getElementById('file-type-icon');
    const extractAction = document.getElementById('upload-extract-action');

    // 1. Client-Side Preliminary Checks
    if (!file) return;

    const allowedExtensions = ['.pdf', '.png', '.jpg', '.jpeg'];
    const lowerName = file.name.toLowerCase();
    const hasValidExt = allowedExtensions.some(ext => lowerName.endsWith(ext));

    if (!hasValidExt) {
        errorAlert.innerHTML = '<strong>Unsupported Format:</strong> Please upload a PDF (.pdf) or image (.png, .jpg, .jpeg) file.';
        errorAlert.style.display = 'block';
        return;
    }

    if (file.size === 0) {
        errorAlert.innerHTML = '<strong>Empty File:</strong> The selected file contains 0 bytes. Please select a valid document.';
        errorAlert.style.display = 'block';
        return;
    }

    const MAX_BYTES = 10 * 1024 * 1024; // 10 MB
    if (file.size > MAX_BYTES) {
        errorAlert.innerHTML = `<strong>File Too Large:</strong> File size (${formatBytes(file.size)}) exceeds the 10 MB limit.`;
        errorAlert.style.display = 'block';
        return;
    }

    // 2. Display Progress State
    progressContainer.style.display = 'block';
    progressBar.style.width = '30%';
    progressBar.classList.add('indeterminate');
    statusText.textContent = `Validating ${file.name} format & security...`;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/upload-document`, {
            method: 'POST',
            body: formData
        });

        progressBar.style.width = '85%';

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Upload failed validation.');
        }

        // 3. Successful Upload
        activeUpload = result;
        currentPredictionResult = null;
        currentVerificationParameters = {};

        // Invalidate previous prediction and verification views
        const outputContainer = document.getElementById('output-container');
        const placeholderState = document.getElementById('placeholder-state');
        if (outputContainer) outputContainer.style.display = 'none';
        if (placeholderState) placeholderState.style.display = 'block';

        const verifyCard = document.getElementById('verification-card');
        const reviewCard = document.getElementById('review-card');
        if (verifyCard) verifyCard.style.display = 'none';
        if (reviewCard) reviewCard.style.display = 'none';

        setWorkflowStep(1);
        progressBar.style.width = '100%';
        progressBar.classList.remove('indeterminate');
        statusText.textContent = 'Upload verified successfully!';

        setTimeout(() => {
            progressContainer.style.display = 'none';
            progressBar.style.width = '0%';
        }, 500);

        // Populate File Card
        const isPdf = result.filename.toLowerCase().endsWith('.pdf');
        fileTypeIcon.textContent = isPdf ? '📄' : '🖼️';
        fileNameDisplay.textContent = result.filename;
        fileSizeDisplay.textContent = result.file_size_formatted;
        fileCard.style.display = 'flex';

        // Show Extract Action Button
        if (extractAction) extractAction.style.display = 'block';

        // Display Success Alert
        successAlert.innerHTML = `
            <strong>✓ Document Staged:</strong> <em>${result.filename}</em> (${result.file_size_formatted}) 
            verified. Temporary secure session active (ID: <code>${result.upload_id.substring(0, 8)}...</code>).
        `;
        successAlert.style.display = 'block';

    } catch (err) {
        console.error('Upload Error:', err);
        progressContainer.style.display = 'none';
        progressBar.style.width = '0%';
        errorAlert.innerHTML = `<strong>Upload Rejected:</strong> ${err.message || 'Could not upload file to backend.'}`;
        errorAlert.style.display = 'block';
        document.getElementById('medical-file-input').value = '';
    }
}

async function handleRemoveUpload() {
    clearUploadAlerts();

    const fileCard = document.getElementById('uploaded-file-card');
    const fileInput = document.getElementById('medical-file-input');
    const successAlert = document.getElementById('upload-success-alert');
    const extractAction = document.getElementById('upload-extract-action');
    const verifyCard = document.getElementById('verification-card');

    if (activeUpload && activeUpload.upload_id) {
        try {
            await fetch(`${API_BASE}/upload-document/${activeUpload.upload_id}`, {
                method: 'DELETE'
            });
        } catch (err) {
            console.warn('Could not notify server of file deletion:', err);
        }
    }

    activeUpload = null;
    currentVerificationParameters = {};
    currentPredictionResult = null;
    if (fileInput) fileInput.value = '';
    if (fileCard) fileCard.style.display = 'none';
    if (extractAction) extractAction.style.display = 'none';
    if (verifyCard) verifyCard.style.display = 'none';
    const reviewCard = document.getElementById('review-card');
    if (reviewCard) reviewCard.style.display = 'none';
    setWorkflowStep(1);

    successAlert.innerHTML = 'Uploaded medical report removed from temporary storage.';
    successAlert.style.display = 'block';
    setTimeout(() => {
        successAlert.style.display = 'none';
    }, 3000);
}


// ============================================================================
// Phase 5: Verification Interface Implementation
// ============================================================================

let currentVerificationParameters = {};

const CLINICAL_GROUPS = [
    {
        id: "group-hemodynamics",
        title: "1. Demographics & Hemodynamics",
        icon: "👤",
        features: ["age", "bp"]
    },
    {
        id: "group-renal",
        title: "2. Renal Biomarkers & Chemistry",
        icon: "🧪",
        features: ["sc", "bu", "bgr", "sod", "pot"]
    },
    {
        id: "group-hematology",
        title: "3. Hematology & Blood Counts",
        icon: "🩸",
        features: ["hemo", "pcv", "rbcc", "wbcc"]
    },
    {
        id: "group-urinalysis",
        title: "4. Urinalysis Parameters",
        icon: "🔬",
        features: ["sg", "al", "su", "rbc", "pc", "pcc", "ba"]
    },
    {
        id: "group-comorbidities",
        title: "5. Clinical History & Comorbidities",
        icon: "📋",
        features: ["htn", "dm", "cad", "appet", "pe", "ane"]
    }
];

const PARAMETER_METADATA = {
    age: { label: "Age", unit: "years", type: "numerical", min: 1, max: 120, step: 1, placeholder: "e.g. 55" },
    bp: { label: "Blood Pressure (Diastolic)", unit: "mm/Hg", type: "numerical", min: 40, max: 250, step: 5, placeholder: "e.g. 80" },
    sg: { label: "Specific Gravity", unit: "density", type: "numerical", min: 1.000, max: 1.040, step: 0.005, options: ["", "1.005", "1.010", "1.015", "1.020", "1.025"] },
    al: { label: "Albumin", unit: "grade (0-5)", type: "numerical", min: 0, max: 5, step: 1, options: ["", "0", "1", "2", "3", "4", "5"] },
    su: { label: "Sugar", unit: "grade (0-5)", type: "numerical", min: 0, max: 5, step: 1, options: ["", "0", "1", "2", "3", "4", "5"] },
    bgr: { label: "Blood Glucose Random", unit: "mg/dL", type: "numerical", min: 20, max: 600, step: 1, placeholder: "e.g. 120" },
    bu: { label: "Blood Urea", unit: "mg/dL", type: "numerical", min: 1, max: 500, step: 1, placeholder: "e.g. 42" },
    sc: { label: "Serum Creatinine", unit: "mg/dL", type: "numerical", min: 0.1, max: 100, step: 0.1, placeholder: "e.g. 1.2" },
    sod: { label: "Serum Sodium", unit: "mEq/L", type: "numerical", min: 50, max: 200, step: 0.5, placeholder: "e.g. 138" },
    pot: { label: "Serum Potassium", unit: "mEq/L", type: "numerical", min: 1.0, max: 60.0, step: 0.1, placeholder: "e.g. 4.3" },
    hemo: { label: "Hemoglobin", unit: "g/dL", type: "numerical", min: 2.0, max: 25.0, step: 0.1, placeholder: "e.g. 12.6" },
    pcv: { label: "Packed Cell Volume", unit: "%", type: "numerical", min: 5, max: 70, step: 1, placeholder: "e.g. 40" },
    wbcc: { label: "White Blood Cell Count", unit: "cells/cumm", type: "numerical", min: 500, max: 50000, step: 100, placeholder: "e.g. 8100" },
    rbcc: { label: "Red Blood Cell Count", unit: "millions/cmm", type: "numerical", min: 0.5, max: 12.0, step: 0.1, placeholder: "e.g. 4.8" },
    rbc: { label: "Red Blood Cells in Urine", unit: "categorical", type: "categorical", options: ["", "normal", "abnormal"] },
    pc: { label: "Pus Cells in Urine", unit: "categorical", type: "categorical", options: ["", "normal", "abnormal"] },
    pcc: { label: "Pus Cell Clumps", unit: "categorical", type: "categorical", options: ["", "notpresent", "present"] },
    ba: { label: "Bacteria in Urine", unit: "categorical", type: "categorical", options: ["", "notpresent", "present"] },
    htn: { label: "Hypertension", unit: "categorical", type: "categorical", options: ["", "no", "yes"] },
    dm: { label: "Diabetes Mellitus", unit: "categorical", type: "categorical", options: ["", "no", "yes"] },
    cad: { label: "Coronary Artery Disease", unit: "categorical", type: "categorical", options: ["", "no", "yes"] },
    appet: { label: "Appetite", unit: "categorical", type: "categorical", options: ["", "good", "poor"] },
    pe: { label: "Pedal Edema", unit: "categorical", type: "categorical", options: ["", "no", "yes"] },
    ane: { label: "Anemia History", unit: "categorical", type: "categorical", options: ["", "no", "yes"] }
};

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

async function handleExtractAndVerify() {
    if (!activeUpload || !activeUpload.upload_id) {
        alert('No active uploaded report found.');
        return;
    }

    const extractBtn = document.getElementById('btn-extract-verify');
    extractBtn.disabled = true;
    extractBtn.innerHTML = '<span class="spinner" style="width:16px;height:16px;margin-right:6px;"></span> Extracting Lab Parameters...';

    // Show extraction animated progress steps
    const progressContainer = document.getElementById('extraction-progress-container');
    const step1 = document.getElementById('ex-step-1');
    const step2 = document.getElementById('ex-step-2');
    const step3 = document.getElementById('ex-step-3');
    if (progressContainer) progressContainer.style.display = 'block';

    setWorkflowStep(2); // Step 2: Extract & Verify

    if (step1) step1.className = 'ex-step-item in-progress';
    if (step2) step2.className = 'ex-step-item pending';
    if (step3) step3.className = 'ex-step-item pending';

    setTimeout(() => {
        if (step1) step1.className = 'ex-step-item completed';
        if (step2) step2.className = 'ex-step-item in-progress';
    }, 400);

    setTimeout(() => {
        if (step2) step2.className = 'ex-step-item completed';
        if (step3) step3.className = 'ex-step-item in-progress';
    }, 800);

    const verifyCard = document.getElementById('verification-card');
    const reviewCard = document.getElementById('review-card');
    if (reviewCard) reviewCard.style.display = 'none';
    clearVerificationAlert();

    try {
        const response = await fetch(`${API_BASE}/extract-parameters/${activeUpload.upload_id}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Parameter extraction failed.');
        }

        if (step3) step3.className = 'ex-step-item completed';

        // Populate verification state
        currentVerificationParameters = {};
        for (const [feat, meta] of Object.entries(PARAMETER_METADATA)) {
            const extracted = data.parameters[feat];
            if (extracted && extracted.extracted_value !== null && extracted.extracted_value !== undefined) {
                // Reliable extracted values require review until confirmed by user
                const origin = 'extracted';
                const status = 'requires_review';
                currentVerificationParameters[feat] = {
                    canonical_name: feat,
                    clinical_label: meta.label,
                    value: extracted.extracted_value,
                    unit: extracted.unit || meta.unit,
                    origin: origin,
                    status: status,
                    source_text: extracted.original_text,
                    source_page: extracted.source_page,
                    warnings: extracted.warnings || [],
                    errors: [],
                    is_valid: true
                };
            } else if (extracted && extracted.status === 'conflicting') {
                currentVerificationParameters[feat] = {
                    canonical_name: feat,
                    clinical_label: meta.label,
                    value: null,
                    unit: meta.unit,
                    origin: 'extracted',
                    status: 'conflicting',
                    source_text: extracted.original_text,
                    source_page: extracted.source_page,
                    warnings: extracted.warnings || ['Conflicting values detected across pages.'],
                    errors: ['Conflict requires manual clinician resolution.'],
                    is_valid: false
                };
            } else {
                currentVerificationParameters[feat] = {
                    canonical_name: feat,
                    clinical_label: meta.label,
                    value: null,
                    unit: meta.unit,
                    origin: 'extracted',
                    status: 'missing',
                    source_text: null,
                    source_page: null,
                    warnings: [],
                    errors: [],
                    is_valid: true
                };
            }
        }

        setTimeout(() => {
            if (progressContainer) progressContainer.style.display = 'none';
            renderVerificationInterface();
            verifyCard.style.display = 'block';
            verifyCard.scrollIntoView({ behavior: 'smooth' });

            const vAlert = document.getElementById('verification-alert');
            if (data.extracted_unambiguous_count === 0 && data.extracted_requiring_review_count === 0) {
                if (vAlert) {
                    vAlert.className = 'alert alert-warning';
                    vAlert.innerHTML = '<strong>⚠️ No Supported Lab Biomarkers Detected:</strong> No recognizable renal laboratory values were detected in the document. Please verify the document or enter clinical values manually.';
                    vAlert.style.display = 'block';
                }
            }
        }, 500);

    } catch (err) {
        console.error('Extraction error:', err);
        if (progressContainer) progressContainer.style.display = 'none';
        setWorkflowStep(1);
        const errorAlert = document.getElementById('upload-error-alert');
        if (errorAlert) {
            errorAlert.innerHTML = `<strong>Extraction Error:</strong> ${escapeHtml(err.message || 'Error communicating with backend.')}`;
            errorAlert.style.display = 'block';
            errorAlert.scrollIntoView({ behavior: 'smooth' });
        } else {
            alert(`Extraction failed: ${err.message || 'Error communicating with backend.'}`);
        }
    } finally {
        extractBtn.disabled = false;
        extractBtn.innerHTML = '<span class="btn-icon">🔬</span><span>Extract &amp; Verify Lab Parameters</span>';
    }
}

function validateLocalParameter(feat, rawVal) {
    const meta = PARAMETER_METADATA[feat];
    if (!meta) return { is_valid: false, error: 'Unknown feature' };

    if (rawVal === null || rawVal === undefined || rawVal === '') {
        return { is_valid: true, is_missing: true };
    }

    if (meta.type === 'numerical') {
        const num = parseFloat(rawVal);
        if (isNaN(num)) {
            return { is_valid: false, error: 'Must be a valid decimal number.' };
        }
        if (num < meta.min) {
            return { is_valid: false, error: `Value ${num} is below clinical minimum (${meta.min} ${meta.unit}).` };
        }
        if (num > meta.max) {
            return { is_valid: false, error: `Value ${num} exceeds clinical maximum (${meta.max} ${meta.unit}).` };
        }
        return { is_valid: true, value: num };
    } else {
        const s = String(rawVal).trim().toLowerCase();
        if (!meta.options.includes(s) || s === '') {
            return { is_valid: false, error: `Invalid option '${rawVal}'.` };
        }
        return { is_valid: true, value: s };
    }
}

function renderVerificationInterface() {
    const container = document.getElementById('verification-groups-container');
    container.innerHTML = '';

    CLINICAL_GROUPS.forEach(group => {
        const groupEl = document.createElement('div');
        groupEl.className = 'verification-group';
        groupEl.id = group.id;

        const titleEl = document.createElement('div');
        titleEl.className = 'verification-group-title';
        titleEl.innerHTML = `<span>${group.icon}</span> <span>${group.title}</span>`;
        groupEl.appendChild(titleEl);

        const gridEl = document.createElement('div');
        gridEl.className = 'verification-grid';

        group.features.forEach(feat => {
            const param = currentVerificationParameters[feat] || {
                canonical_name: feat,
                clinical_label: PARAMETER_METADATA[feat].label,
                value: null,
                unit: PARAMETER_METADATA[feat].unit,
                origin: 'extracted',
                status: 'missing',
                warnings: [],
                errors: []
            };

            const card = document.createElement('div');
            card.className = `param-item-card status-${param.status}`;
            card.id = `param-card-${feat}`;

            // Origin Tag
            let originLabel = '[Extracted]';
            if (param.origin === 'manual') originLabel = '[Manual Edit]';
            if (param.origin === 'verified') originLabel = '[Verified ✓]';
            if (param.source_page) originLabel = `[Extracted: Page ${param.source_page}]`;

            // Status Tag
            let statusTagClass = `tag-${param.status}`;
            let statusText = param.status.replace('_', ' ').toUpperCase();
            if (param.status === 'requires_review') statusText = 'REVIEW NEEDED';

            const meta = PARAMETER_METADATA[feat];
            let inputHtml = '';

            if (meta.type === 'numerical' && !meta.options) {
                const valStr = (param.value !== null && param.value !== undefined) ? param.value : '';
                inputHtml = `
                    <input type="number" class="param-input-field" id="vinput-${feat}"
                        step="${meta.step}" min="${meta.min}" max="${meta.max}"
                        placeholder="${meta.placeholder}" value="${escapeHtml(valStr)}">
                    <span class="param-unit-label">${meta.unit}</span>
                `;
            } else if (meta.options) {
                const valStr = (param.value !== null && param.value !== undefined) ? String(param.value).trim() : '';
                const optionsHtml = meta.options.map(opt => {
                    let sel = '';
                    if (valStr !== '' && opt !== '') {
                        if (meta.type === 'numerical') {
                            sel = (parseFloat(opt) === parseFloat(valStr)) ? 'selected' : '';
                        } else {
                            sel = (opt.toLowerCase() === valStr.toLowerCase()) ? 'selected' : '';
                        }
                    } else if (opt === '' && (valStr === '' || param.value === null)) {
                        sel = 'selected';
                    }
                    const optLabel = opt === '' ? 'Unpopulated / Missing' : opt;
                    return `<option value="${escapeHtml(opt)}" ${sel}>${escapeHtml(optLabel)}</option>`;
                }).join('');
                inputHtml = `
                    <select class="param-input-field" id="vinput-${feat}">
                        ${optionsHtml}
                    </select>
                    <span class="param-unit-label">${meta.unit}</span>
                `;
            }

            let sourceSnippetHtml = '';
            if (param.source_text) {
                sourceSnippetHtml = `
                    <div class="source-snippet">
                        📄 Report snippet: "${escapeHtml(param.source_text)}"
                    </div>
                `;
            }

            let warningHtml = '';
            if (param.warnings && param.warnings.length > 0) {
                warningHtml = `
                    <div class="param-warning-box">
                        ⚠️ ${escapeHtml(param.warnings.join(' '))}
                    </div>
                `;
            }

            let errorHtml = '';
            if (param.errors && param.errors.length > 0) {
                errorHtml = `
                    <div class="param-error-box">
                        ❌ ${escapeHtml(param.errors.join(' '))}
                    </div>
                `;
            }

            card.innerHTML = `
                <div class="param-header">
                    <div class="param-title-group">
                        <span class="param-name">${meta.label}</span>
                        <span class="param-key">${feat}</span>
                    </div>
                    <div style="display:flex; flex-direction:column; align-items:flex-end; gap:0.2rem;">
                        <span class="param-status-tag ${statusTagClass}" id="vtag-${feat}">${statusText}</span>
                        <span class="param-origin-meta">${originLabel}</span>
                    </div>
                </div>
                <div class="param-body">
                    <div class="param-input-row">
                        ${inputHtml}
                    </div>
                    ${sourceSnippetHtml}
                    ${warningHtml}
                    ${errorHtml}
                    <div class="param-actions">
                        <button type="button" class="btn-param-clear" data-feat="${feat}" title="Leave value unpopulated">Clear / Missing</button>
                        <button type="button" class="btn-param-confirm" data-feat="${feat}" title="Confirm this value">✓ Confirm</button>
                    </div>
                </div>
            `;

            gridEl.appendChild(card);
        });

        groupEl.appendChild(gridEl);
        container.appendChild(groupEl);
    });

    // Attach listeners for all dynamic fields
    Object.keys(PARAMETER_METADATA).forEach(feat => {
        const inputEl = document.getElementById(`vinput-${feat}`);
        if (inputEl) {
            inputEl.addEventListener('input', (e) => handleFieldChange(feat, e.target.value));
            inputEl.addEventListener('change', (e) => handleFieldChange(feat, e.target.value));
        }
    });

    // Attach listeners for per-field buttons
    container.querySelectorAll('.btn-param-confirm').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const feat = e.target.getAttribute('data-feat');
            handleFieldConfirm(feat);
        });
    });

    container.querySelectorAll('.btn-param-clear').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const feat = e.target.getAttribute('data-feat');
            handleFieldClear(feat);
        });
    });

    updateVerificationStatsBadges();
}

function handleFieldChange(feat, newVal) {
    const param = currentVerificationParameters[feat];
    if (!param) return;

    param.origin = 'manual';
    param.warnings = [];
    param.errors = [];

    const valResult = validateLocalParameter(feat, newVal);

    if (valResult.is_missing) {
        param.value = null;
        param.status = 'missing';
        param.is_valid = true;
    } else if (!valResult.is_valid) {
        param.value = newVal;
        param.status = 'invalid';
        param.errors = [valResult.error];
        param.is_valid = false;
    } else {
        param.value = valResult.value;
        param.status = 'verified';
        param.is_valid = true;
    }

    refreshSingleParamCard(feat);
    updateVerificationStatsBadges();
}

function handleFieldConfirm(feat) {
    const param = currentVerificationParameters[feat];
    if (!param) return;

    if (param.status === 'invalid' || !param.is_valid) {
        alert(`Cannot confirm ${param.clinical_label}: value is invalid.`);
        return;
    }

    param.status = 'verified';
    param.origin = 'verified';
    param.errors = [];

    refreshSingleParamCard(feat);
    updateVerificationStatsBadges();
}

function handleFieldClear(feat) {
    const param = currentVerificationParameters[feat];
    if (!param) return;

    param.value = null;
    param.status = 'missing';
    param.origin = 'manual';
    param.errors = [];
    param.warnings = ['Marked unpopulated by clinician.'];

    const inputEl = document.getElementById(`vinput-${feat}`);
    if (inputEl) inputEl.value = '';

    refreshSingleParamCard(feat);
    updateVerificationStatsBadges();
}

function refreshSingleParamCard(feat) {
    const param = currentVerificationParameters[feat];
    const card = document.getElementById(`param-card-${feat}`);
    const tag = document.getElementById(`vtag-${feat}`);
    if (!card || !tag) return;

    card.className = `param-item-card status-${param.status}`;

    let statusText = param.status.replace('_', ' ').toUpperCase();
    if (param.status === 'requires_review') statusText = 'REVIEW NEEDED';
    tag.className = `param-status-tag tag-${param.status}`;
    tag.textContent = statusText;

    // Refresh error box
    let errorBox = card.querySelector('.param-error-box');
    if (param.errors && param.errors.length > 0) {
        if (!errorBox) {
            errorBox = document.createElement('div');
            errorBox.className = 'param-error-box';
            card.querySelector('.param-body').appendChild(errorBox);
        }
        errorBox.textContent = `❌ ${param.errors.join(' ')}`;
        errorBox.style.display = 'block';
    } else if (errorBox) {
        errorBox.style.display = 'none';
    }
}

function updateVerificationStatsBadges() {
    let verified = 0;
    let review = 0;
    let missing = 0;
    let conflict = 0;
    let invalid = 0;

    Object.values(currentVerificationParameters).forEach(p => {
        if (p.status === 'verified') verified++;
        else if (p.status === 'requires_review') review++;
        else if (p.status === 'missing') missing++;
        else if (p.status === 'conflicting') conflict++;
        else if (p.status === 'invalid') invalid++;
    });

    document.getElementById('badge-count-verified').textContent = `${verified} Verified`;
    document.getElementById('badge-count-review').textContent = `${review} Review`;
    document.getElementById('badge-count-missing').textContent = `${missing} Missing`;

    const conflictBadge = document.getElementById('badge-count-conflict');
    if (conflict > 0) {
        conflictBadge.textContent = `${conflict} Conflicts`;
        conflictBadge.style.display = 'inline-flex';
    } else {
        conflictBadge.style.display = 'none';
    }

    const invalidBadge = document.getElementById('badge-count-invalid');
    if (invalid > 0) {
        invalidBadge.textContent = `${invalid} Invalid`;
        invalidBadge.style.display = 'inline-flex';
    } else {
        invalidBadge.style.display = 'none';
    }

    // Toggle proceed button availability
    const proceedBtn = document.getElementById('btn-verify-proceed');
    if (proceedBtn) {
        proceedBtn.disabled = (invalid > 0 || conflict > 0);
    }
}

function handleVerificationConfirmAll() {
    let confirmedCount = 0;
    Object.keys(currentVerificationParameters).forEach(feat => {
        const param = currentVerificationParameters[feat];
        if (param.is_valid && param.status !== 'invalid' && param.status !== 'conflicting' && param.value !== null && param.value !== undefined && param.value !== '') {
            param.status = 'verified';
            param.origin = 'verified';
            refreshSingleParamCard(feat);
            confirmedCount++;
        }
    });

    updateVerificationStatsBadges();
    const alertEl = document.getElementById('verification-alert');
    alertEl.className = 'alert alert-success';
    alertEl.innerHTML = `<strong>✓ Confirmed:</strong> ${confirmedCount} populated parameter(s) marked as verified and locked for prediction. Missing fields remain unpopulated.`;
    alertEl.style.display = 'block';
    setTimeout(() => { alertEl.style.display = 'none'; }, 4000);
}

function handleVerificationEditAll() {
    // Focus first input field
    const firstInput = document.getElementById('vinput-age');
    if (firstInput) {
        firstInput.focus();
        firstInput.select();
    }
}

function handleVerificationCancel() {
    const verifyCard = document.getElementById('verification-card');
    if (verifyCard) verifyCard.style.display = 'none';
}

function clearVerificationAlert() {
    const alertEl = document.getElementById('verification-alert');
    if (alertEl) {
        alertEl.style.display = 'none';
        alertEl.innerHTML = '';
    }
}

function handleVerificationProceed() {
    clearVerificationAlert();
    const alertEl = document.getElementById('verification-alert');

    // 1. Client-Side Preliminary Validation across all 24 features
    const errors = [];
    const payloadParams = {};

    for (const feat of Object.keys(PARAMETER_METADATA)) {
        const item = currentVerificationParameters[feat] || {
            canonical_name: feat,
            value: null,
            unit: PARAMETER_METADATA[feat].unit,
            origin: "manual",
            status: "missing"
        };

        const validation = validateLocalParameter(feat, item.value);
        if (!validation.is_valid) {
            errors.push(`${PARAMETER_METADATA[feat].label}: ${validation.error}`);
            item.status = "invalid";
            refreshSingleParamCard(feat);
        } else if (item.status === "conflicting") {
            errors.push(`${PARAMETER_METADATA[feat].label}: Has unresolved conflicting values across report sections.`);
        }

        payloadParams[feat] = {
            canonical_name: feat,
            clinical_label: PARAMETER_METADATA[feat].label,
            value: item.value,
            unit: item.unit || PARAMETER_METADATA[feat].unit,
            origin: item.origin || "manual",
            status: item.status || "verified",
            source_text: item.source_text || null,
            source_page: item.source_page || null,
            warnings: item.warnings || []
        };
    }

    // Required Clinical Biomarkers Validation
    let populatedCount = 0;
    for (const item of Object.values(payloadParams)) {
        if (item.value !== null && item.value !== undefined && item.value !== '') {
            populatedCount++;
        }
    }

    if (populatedCount === 0) {
        errors.push('All 24 patient clinical parameters are missing. Please enter and verify patient values before proceeding.');
    } else {
        const scVal = payloadParams['sc'] ? payloadParams['sc'].value : null;
        const hemoVal = payloadParams['hemo'] ? payloadParams['hemo'].value : null;
        if (scVal === null || scVal === undefined || scVal === '') {
            errors.push('Serum Creatinine (sc) is required to evaluate renal filtration.');
        }
        if (hemoVal === null || hemoVal === undefined || hemoVal === '') {
            errors.push('Hemoglobin (hemo) is required to evaluate renal anemia.');
        }
    }

    if (errors.length > 0) {
        alertEl.className = 'alert alert-danger';
        alertEl.innerHTML = `<strong>Verification Incomplete:</strong> Please resolve the following before proceeding to review:<ul>${errors.map(e => `<li>${escapeHtml(e)}</li>`).join('')}</ul>`;
        alertEl.style.display = 'block';
        updateVerificationStatsBadges();
        return;
    }

    // 2. Open Pre-Prediction Review Screen (Step 3)
    renderReviewScreen(payloadParams);
}

function renderReviewScreen(payloadParams) {
    const verifyCard = document.getElementById('verification-card');
    const reviewCard = document.getElementById('review-card');
    if (verifyCard) verifyCard.style.display = 'none';
    if (!reviewCard) return;

    let vCount = 0;
    let eCount = 0;
    let mCount = 0;
    let impCount = 0;
    const missingList = [];

    for (const [feat, item] of Object.entries(payloadParams)) {
        if (item.value === null || item.value === undefined || item.value === '') {
            impCount++;
            missingList.push(item.clinical_label);
        } else if (item.origin === 'verified') {
            vCount++;
        } else if (item.origin === 'extracted') {
            eCount++;
        } else {
            mCount++;
        }
    }

    // Update Summary Stats Badges
    document.getElementById('rstat-verified').textContent = vCount;
    document.getElementById('rstat-extracted').textContent = eCount;
    document.getElementById('rstat-manual').textContent = mCount;
    document.getElementById('rstat-imputed').textContent = impCount;

    // Critical Biomarkers Table
    const criticalFeats = [
        { feat: 'sc', role: 'Primary GFR Proxy (Serum Creatinine)' },
        { feat: 'bu', role: 'Nitrogen Waste Retention (Blood Urea)' },
        { feat: 'bgr', role: 'Diabetic Nephropathy Risk (Blood Glucose)' },
        { feat: 'hemo', role: 'Renal Anemia Biomarker (Hemoglobin)' },
        { feat: 'pcv', role: 'Hematocrit Fraction (Packed Cell Volume)' },
        { feat: 'bp', role: 'Cardiorenal Load (Blood Pressure)' },
        { feat: 'sg', role: 'Tubular Concentrating Ability (Specific Gravity)' },
        { feat: 'al', role: 'Proteinuria / Glomerular Damage (Albumin)' }
    ];

    const tbody = document.getElementById('review-critical-tbody');
    if (tbody) {
        tbody.innerHTML = criticalFeats.map(item => {
            const p = payloadParams[item.feat];
            const meta = PARAMETER_METADATA[item.feat];
            const hasVal = p && p.value !== null && p.value !== undefined && p.value !== '';
            const valDisplay = hasVal ? `<strong>${p.value}</strong> ${p.unit || meta.unit}` : '<span style="color:#7c3aed;font-style:italic;">Imputed (Training Median/Mode)</span>';
            const statusClass = hasVal ? (p.status === 'verified' ? 'vbadge-verified' : 'vbadge-review') : 'vbadge-missing';
            const statusTxt = hasVal ? (p.status === 'verified' ? 'Verified ✓' : 'Extracted') : 'Imputed';
            return `
                <tr>
                    <td><strong>${escapeHtml(meta.label)}</strong></td>
                    <td style="color:#64748b;font-size:0.75rem;">${escapeHtml(item.role)}</td>
                    <td>${valDisplay}</td>
                    <td><span class="vbadge ${statusClass}">${statusTxt}</span></td>
                </tr>
            `;
        }).join('');
    }

    // Missing List Description
    const missingEl = document.getElementById('review-missing-list');
    if (missingEl) {
        if (missingList.length > 0) {
            missingEl.textContent = missingList.join(', ');
        } else {
            missingEl.textContent = 'None — all 24 parameters verified/populated.';
        }
    }

    setWorkflowStep(3); // Step 3: Review Inputs
    reviewCard.style.display = 'block';
    reviewCard.scrollIntoView({ behavior: 'smooth' });
}

function handleReviewBack() {
    const reviewCard = document.getElementById('review-card');
    const verifyCard = document.getElementById('verification-card');
    if (reviewCard) reviewCard.style.display = 'none';
    if (verifyCard) {
        verifyCard.style.display = 'block';
        verifyCard.scrollIntoView({ behavior: 'smooth' });
    }
    setWorkflowStep(2); // Step 2: Extract & Verify
}

async function handleConfirmPrediction() {
    const alertEl = document.getElementById('review-alert');
    if (alertEl) alertEl.style.display = 'none';

    // Compile payload
    const payloadParams = {};
    for (const feat of Object.keys(PARAMETER_METADATA)) {
        const item = currentVerificationParameters[feat] || {
            canonical_name: feat,
            value: null,
            unit: PARAMETER_METADATA[feat].unit,
            origin: "manual",
            status: "missing"
        };
        payloadParams[feat] = {
            canonical_name: feat,
            clinical_label: PARAMETER_METADATA[feat].label,
            value: item.value,
            unit: item.unit || PARAMETER_METADATA[feat].unit,
            origin: item.origin || "manual",
            status: item.status || "verified",
            source_text: item.source_text || null,
            source_page: item.source_page || null,
            warnings: item.warnings || []
        };
    }

    const confirmBtn = document.getElementById('btn-review-confirm-predict');
    confirmBtn.disabled = true;
    confirmBtn.innerHTML = '<span class="spinner" style="width:16px;height:16px;margin-right:6px;"></span> Validating &amp; Predicting...';

    const placeholderState = document.getElementById('placeholder-state');
    const loadingState = document.getElementById('loading-state');
    const outputContainer = document.getElementById('output-container');
    if (placeholderState) placeholderState.style.display = 'none';
    if (outputContainer) outputContainer.style.display = 'none';
    if (loadingState) loadingState.style.display = 'block';

    try {
        const response = await fetch(`${API_BASE}/predict-verified`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                upload_id: activeUpload ? activeUpload.upload_id : null,
                parameters: payloadParams,
                allow_imputation: true,
                include_plot: true
            })
        });

        const predictResult = await response.json();

        if (!response.ok) {
            throw new Error(predictResult.detail || 'Server validation rejected patient parameters for prediction.');
        }

        // Synchronize verified inputs into main assessment form
        for (const [feat, item] of Object.entries(payloadParams)) {
            const inputEl = document.getElementById(`input-${feat}`);
            if (inputEl) {
                inputEl.value = (item.value !== null && item.value !== undefined) ? item.value : '';
            }
        }

        currentPredictionResult = predictResult;

        // Render results & SHAP
        renderPredictionResults(predictResult);
        setWorkflowStep(4); // Step 4: Prediction & SHAP

        // Hide review card
        const reviewCard = document.getElementById('review-card');
        if (reviewCard) reviewCard.style.display = 'none';

        // Scroll to results panel
        const resultsPanel = document.getElementById('results-panel');
        if (resultsPanel) resultsPanel.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        console.error('Prediction Error:', err);
        if (alertEl) {
            alertEl.innerHTML = `<strong>Prediction Blocked:</strong> ${escapeHtml(err.message || 'Error validating parameters.')}`;
            alertEl.style.display = 'block';
        }
        if (loadingState) loadingState.style.display = 'none';
        if (placeholderState && (!outputContainer || outputContainer.style.display === 'none')) {
            placeholderState.style.display = 'block';
        }
    } finally {
        confirmBtn.disabled = false;
        confirmBtn.innerHTML = '<span class="btn-icon">⚡</span><span>Confirm &amp; Run CKD Prediction</span>';
    }
}

async function handleDownloadReport() {
    if (!currentPredictionResult) {
        alert('Please complete a prediction before downloading the clinical report.');
        return;
    }

    const downloadBtn = document.getElementById('btn-download-pdf-report');
    const statusEl = document.getElementById('report-download-status');

    downloadBtn.disabled = true;
    downloadBtn.innerHTML = '<span class="spinner" style="width:14px;height:14px;margin-right:6px;"></span> Generating PDF...';
    if (statusEl) {
        statusEl.textContent = 'Assembling clinical biomarkers, decision tree metrics & SHAP attributions into vector PDF...';
        statusEl.style.display = 'block';
    }

    try {
        const response = await fetch(`${API_BASE}/download-report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                upload_id: activeUpload ? activeUpload.upload_id : null,
                document_name: activeUpload ? activeUpload.filename : 'Patient_Assessment.pdf',
                prediction_result: currentPredictionResult,
                verified_parameters: currentVerificationParameters
            })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server returned HTTP ${response.status}`);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const uploadSuffix = (activeUpload && activeUpload.upload_id) ? activeUpload.upload_id.substring(0, 8) : 'Assessment';
        a.download = `CKD_Clinical_Report_${uploadSuffix}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();

        setWorkflowStep(5); // Step 5: Clinical Report Complete

        if (statusEl) {
            statusEl.textContent = '✓ Formal clinical decision support PDF report downloaded successfully.';
            setTimeout(() => { statusEl.style.display = 'none'; }, 6000);
        }
    } catch (err) {
        console.error('Report Download Error:', err);
        if (statusEl) {
            statusEl.textContent = `❌ Error generating report: ${err.message}`;
        } else {
            alert(`Could not download report: ${err.message}`);
        }
    } finally {
        downloadBtn.disabled = false;
        downloadBtn.innerHTML = '<span class="btn-icon">📥</span><span>Download Clinical Report (PDF)</span>';
    }
}

async function handleRestartWorkflow() {
    if (activeUpload && activeUpload.upload_id) {
        try {
            await fetch(`${API_BASE}/upload-document/${activeUpload.upload_id}`, { method: 'DELETE' });
        } catch (e) {
            console.warn('Session cleanup on restart:', e);
        }
    }

    // Reset internal state
    activeUpload = null;
    currentVerificationParameters = {};
    currentPredictionResult = null;

    // Reset upload UI elements
    const fileInput = document.getElementById('medical-file-input');
    if (fileInput) fileInput.value = '';
    const fileCard = document.getElementById('uploaded-file-card');
    if (fileCard) fileCard.style.display = 'none';
    const dropzonePrompt = document.getElementById('dropzone-prompt');
    if (dropzonePrompt) dropzonePrompt.style.display = 'flex';
    const extractAction = document.getElementById('upload-extract-action');
    if (extractAction) extractAction.style.display = 'none';
    const progressContainer = document.getElementById('upload-progress-container');
    if (progressContainer) progressContainer.style.display = 'none';
    const exProgress = document.getElementById('extraction-progress-container');
    if (exProgress) exProgress.style.display = 'none';

    // Reset cards
    const verifyCard = document.getElementById('verification-card');
    if (verifyCard) verifyCard.style.display = 'none';
    const reviewCard = document.getElementById('review-card');
    if (reviewCard) reviewCard.style.display = 'none';
    const outputContainer = document.getElementById('output-container');
    if (outputContainer) outputContainer.style.display = 'none';
    const placeholderState = document.getElementById('placeholder-state');
    if (placeholderState) placeholderState.style.display = 'block';

    // Clear alerts
    clearUploadAlerts();
    clearVerificationAlert();
    const reviewAlert = document.getElementById('review-alert');
    if (reviewAlert) reviewAlert.style.display = 'none';

    // Reset assessment form
    const form = document.getElementById('ckd-form');
    if (form) form.reset();

    // Reset Stepper
    setWorkflowStep(1);

    // Notification toast
    const successAlert = document.getElementById('upload-success-alert');
    if (successAlert) {
        successAlert.innerHTML = '<strong>Workflow Restarted:</strong> Previous session purged. Ready for a new medical document or manual assessment.';
        successAlert.style.display = 'block';
        setTimeout(() => { successAlert.style.display = 'none'; }, 4000);
    }

    const uploadCard = document.getElementById('upload-card');
    if (uploadCard) uploadCard.scrollIntoView({ behavior: 'smooth' });
}

