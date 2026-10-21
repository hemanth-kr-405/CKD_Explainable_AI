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
    form.addEventListener('submit', handleFormSubmit);

    document.getElementById('btn-reset').addEventListener('click', handleReset);
    document.getElementById('btn-preset-ckd').addEventListener('click', applyCkdPreset);
    document.getElementById('btn-preset-healthy').addEventListener('click', applyHealthyPreset);
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
    document.getElementById('ckd-form').reset();
    clearValidationErrors();
    document.getElementById('placeholder-state').style.display = 'block';
    document.getElementById('loading-state').style.display = 'none';
    document.getElementById('output-container').style.display = 'none';
}

function clearValidationErrors() {
    const errorAlert = document.getElementById('form-error-alert');
    errorAlert.style.display = 'none';
    errorAlert.textContent = '';
    document.querySelectorAll('.form-group.has-error').forEach(el => {
        el.classList.remove('has-error');
    });
}

// ============================================================================
// Form Validation & Prediction Workflow
// ============================================================================

async function handleFormSubmit(e) {
    e.preventDefault();
    clearValidationErrors();

    const form = document.getElementById('ckd-form');
    const formData = new FormData(form);
    const payload = {};
    const validationErrors = [];

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
        errorAlert.innerHTML = `<strong>Please fix input errors:</strong><br>${validationErrors.join('<br>')}`;
        errorAlert.style.display = 'block';
        return;
    }

    payload.include_plot = true;

    // Show Loading State
    const placeholderState = document.getElementById('placeholder-state');
    const loadingState = document.getElementById('loading-state');
    const outputContainer = document.getElementById('output-container');
    const submitBtn = document.getElementById('btn-submit');

    placeholderState.style.display = 'none';
    outputContainer.style.display = 'none';
    loadingState.style.display = 'block';
    submitBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Prediction failed.');
        }

        const result = await response.json();
        renderPredictionResults(result);

    } catch (err) {
        console.error('API Error:', err);
        const errorAlert = document.getElementById('form-error-alert');
        errorAlert.innerHTML = `<strong>Error:</strong> ${err.message || 'Could not connect to backend server. Make sure FastAPI is running.'}`;
        errorAlert.style.display = 'block';
        placeholderState.style.display = 'block';
    } finally {
        loadingState.style.display = 'none';
        submitBtn.disabled = false;
    }
}

// ============================================================================
// Render Results & SHAP Explanations
// ============================================================================

function renderPredictionResults(data) {
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

    // Baseline Pill
    document.getElementById('shap-base-val').textContent = data.explanation.base_value.toFixed(4);
    document.getElementById('shap-final-val').textContent = data.explanation.patient_model_output.toFixed(4);

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

    // Waterfall Plot Image
    const waterfallCard = document.getElementById('waterfall-container');
    const waterfallImg = document.getElementById('waterfall-img');

    if (data.waterfall_plot_base64) {
        waterfallImg.src = `data:image/png;base64,${data.waterfall_plot_base64}`;
        waterfallCard.style.display = 'block';
    } else {
        waterfallCard.style.display = 'none';
    }

    // Show Container with smooth entrance
    outputContainer.style.display = 'flex';
}
