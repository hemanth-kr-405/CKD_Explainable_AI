# Academic Project Report
# Explainable AI for Chronic Kidney Disease Prediction using Machine Learning and SHAP

---

## 1. Abstract

Chronic Kidney Disease (CKD) represents a significant global healthcare burden characterized by the gradual, often asymptomatic loss of renal function. While supervised machine learning models demonstrate high diagnostic accuracy on complex multivariate clinical datasets, their adoption in clinical practice is severely hindered by the "black-box" dilemma—the inability of clinicians to interrogate the rationale behind an individual patient's prediction. 

This project develops an interpretable, end-to-end clinical decision-support system that couples an optimized **Decision Tree Classifier** with **SHAP (SHapley Additive exPlanations)**. Utilizing the authentic UCI Chronic Kidney Disease dataset (400 patient instances, 24 clinical features), a leak-free preprocessing pipeline was implemented with stratified 80/20 train-test partitioning. The trained model achieves **96.25% test accuracy**, **100.00% precision (PPV)**, **94.00% sensitivity / recall**, **100.00% specificity (TNR)**, an **F1-score of 0.9691**, and an **ROC-AUC of 0.9890** on the held-out test cohort ($N=80$), with 5-fold cross-validation confirming robust generalization (mean accuracy $95.00\% \pm 3.06\%$). 

Through `shap.TreeExplainer`, the system derives exact, polynomial-time Shapley attributions in probability space, validating the mathematical additivity property ($E[f(x)] + \sum \phi_i = f(x)$). Global feature importance identifies Hemoglobin (59.3% Gini influence), Specific Gravity (20.5%), and Red Blood Cell Count (9.6%) as dominant cohort-level predictors, while case-level waterfall plots elucidate multivariate risk and protective factors for individual patient encounters. The system is delivered via an asynchronous **FastAPI** backend and a responsive, zero-dependency **HTML5/Vanilla CSS/JavaScript** web dashboard with explicit probability calibration notices and non-diagnostic research disclaimers.

---

## 2. Introduction

Chronic Kidney Disease (CKD) describes progressive, permanent kidney damage and diminished glomerular filtration rate (GFR). According to global epidemiological data, CKD affects over 10% of the world's population, with millions succumbing annually due to lack of timely interventions. Because early renal impairment frequently produces no outward clinical symptoms ("silent progression"), patient diagnosis is often delayed until advanced stages (Stage 4 or 5 / End-Stage Renal Disease), necessitating chronic hemodialysis or renal transplantation.

Biochemical screening—encompassing serum creatinine, blood urea nitrogen, urine specific gravity, albuminuria, hematocrit, and blood pressure—provides diagnostic signals capable of identifying early impairment. Machine Learning (ML) excels at synthesizing these multidimensional parameters into predictive risk stratifications. However, algorithmic deployment in nephrology requires both verifiable statistical performance and explainable, case-specific reasoning. This project addresses this mandate by integrating machine learning with game-theoretic Explainable AI (XAI).

---

## 3. Problem Statement

1. **Early Detection Imperative**: Early-stage CKD is asymptomatic, leading to under-diagnosis in primary care settings before irreversible parenchymal damage occurs.
2. **The "Black-Box" Barrier in Healthcare**: Complex predictive models (e.g., deep neural networks, ensemble forests) fail to provide interpretable explanations for their outputs. Ethically, medically, and legally, clinicians cannot base interventions on opaque risk scores without understanding the contributing factors.
3. **Data Leakage in Medical Informatics**: Many existing literature models leak information from test cohorts into training transformers (e.g., computing global dataset means or scaling across combined sets), yielding overly optimistic performance that collapses in clinical deployment.
4. **Lack of Transparent Patient Interaction**: Existing prototypes rarely bridge the gap between academic code notebooks and user-friendly, privacy-preserving clinical decision-support tools.

---

## 4. Objectives

1. **Leak-Free Preprocessing**: Construct a scikit-learn compatible preprocessing pipeline that fits missing value imputation (median for skewed continuous variables, mode for discrete categoricals) and binary encoding strictly on training splits.
2. **Interpretable Model Training**: Train a Decision Tree Classifier with balanced class weights (`class_weight='balanced'`) and constrained depth (`max_depth=5`) to balance high sensitivity with inherent interpretability.
3. **Empirical Performance Evaluation**: Conduct rigorous evaluation on a strictly held-out test partition ($N=80$), reporting accuracy, precision, sensitivity, specificity, F1-score, confusion matrix, ROC-AUC, and 5-fold cross-validation.
4. **Game-Theoretic XAI via SHAP**: Integrate `shap.TreeExplainer` to compute exact, additive Shapley feature attributions, delivering cohort-level global rankings and patient-level waterfall visualizations.
5. **Asynchronous Microservice Architecture**: Develop a high-performance FastAPI backend with strict Pydantic schema validation, physiological bounds checking, and error isolation.
6. **Accessible Clinical Web Dashboard**: Deliver a responsive, modern HTML5/CSS/JavaScript web dashboard featuring one-click clinical presets, probability calibration disclosures, and dynamic XAI visualizations without external framework dependencies.

---

## 5. Literature Survey

The application of machine learning to Chronic Kidney Disease prediction has gained substantial attention in biomedical informatics:

- **Rubini and Eswaran (2015)** evaluated baseline classification algorithms on the UCI CKD dataset, observing that tree-based algorithms and radial basis neural networks achieved high predictive accuracy, though the lack of model interpretability was noted as an operational drawback.
- **Soundarapandian et al. (2015)** curated and donated the 400-instance Chronic Kidney Disease dataset from Apollo Hospitals (India) to the UCI Machine Learning Repository, establishing a recognized benchmark for multivariate nephrology modeling.
- **Lundberg and Lee (2017)** introduced **SHAP (SHapley Additive exPlanations)**, establishing a unified game-theoretic framework connecting cooperative game theory (Lloyd Shapley, 1953) with local model-agnostic feature attributions, guaranteeing mathematical properties of local accuracy, missingness, and consistency.
- **Lundberg et al. (2020)** published **TreeSHAP** in *Nature Machine Intelligence*, formulating an exact polynomial-time algorithm ($O(TLD^2)$) for computing Shapley values in tree-based models, eliminating the need for heuristic sampling.
- **Vellido (2020)** emphasized the clinical necessity of interpretable machine learning in healthcare, highlighting that decision support tools must communicate risk factors in familiar physiological units to foster clinical trust and safety.

---

## 6. Existing Systems and Limitations

| Existing Approach | Working Mechanism | Critical Limitations |
| :--- | :--- | :--- |
| **Traditional Clinical Staging (KDIGO)** | Tabular lookup based on estimated GFR (eGFR) and urine albumin-to-creatinine ratio (ACR). | Univariate focus; does not synthesize subtle multi-organ interactions (e.g., electrolyte imbalances, early normocytic anemia). |
| **Black-Box Ensembles (GBM / Deep ANN)** | Multi-layer nonlinear feature mapping optimizing accuracy metrics. | Zero transparency; unable to explain why an individual patient was categorized as high risk; prone to catastrophic failure on out-of-distribution cases. |
| **Post-Hoc LIME Models** | Local linear surrogate approximations fitted on perturbed synthetic samples. | Sampling variance produces unstable, non-deterministic explanations; violates mathematical additivity; computationally expensive. |
| **Uncalibrated Risk Tools** | Raw decision tree or uncalibrated softmax scores presented as absolute risks. | Misleads users into treating terminal leaf purity percentages (e.g., 100%) as certified diagnostic certainty. |

---

## 7. Proposed System

The proposed system addresses these shortcomings by unifying leak-free preprocessing, an interpretable Decision Tree estimator, exact TreeSHAP explanations, and a modern microservice architecture:

1. **Leakage-Resistant Architecture**: Implements `CKDPreprocessor`, calculating data transforms exclusively from the 320-instance training split.
2. **Clinical Sensitivity Optimization**: Employs balanced class weighting to minimize False Negatives ($FN=3$), ensuring patients with early renal impairment are detected.
3. **Exact Shapley Deconstruction**: Computes local feature attributions that additively reconstruct the patient's leaf probability score from the cohort expectation ($0.5000$).
4. **Transparent Narrative Synthesis**: Automatically generates a natural-language summary separating statistical attribution from biological causation.
5. **Privacy by Design**: Operates entirely in memory with zero persistent logging of patient vitals.

---

## 8. System Architecture

```
                                  [ Patient / Clinician ]
                                             │
                                             ▼
                 ┌────────────────────────────────────────────────────────┐
                 │       Interactive Web Dashboard (frontend/)            │
                 │  - Clinical Vitals & Lab Form (24 Parameters)          │
                 │  - Client-Side Physiological Bounds Validation         │
                 │  - One-Click Clinical Presets (UCI #204 & #342)        │
                 │  - Calibration Disclosures & Medical Disclaimer        │
                 └───────────────────────────┬────────────────────────────┘
                                             │ (HTTP JSON POST /predict)
                                             ▼
                 ┌────────────────────────────────────────────────────────┐
                 │           FastAPI Microservice (backend/)              │
                 │  - Lifespan Singleton Artifact Loading                 │
                 │  - Pydantic v2 Schema Validation (extra="forbid")      │
                 │  - Field-Specific Error Isolation (HTTP 422)           │
                 └───────────────────────────┬────────────────────────────┘
                                             │
                                             ▼
                 ┌────────────────────────────────────────────────────────┐
                 │        Inference Pipeline (models/ & src/)             │
                 │                                                        │
                 │   [ CKDPreprocessor ] (models/preprocessor.joblib)     │
                 │   - Median Imputation (14 Numerics)                    │
                 │   - Mode Imputation & Binary Mapping (10 Categoricals) │
                 │                            │                           │
                 │                            ▼                           │
                 │   [ Decision Tree Classifier ] (models/ckd_model.joblib│
                 │   - Balanced Class Weights, Depth 5                    │
                 │   - Prediction & Leaf Probability Output               │
                 │                            │                           │
                 │                            ▼                           │
                 │   [ SHAP TreeExplainer Engine ] (src/explain_predictions)
                 │   - Positive Class Dynamic Slicing (Class 1)           │
                 │   - Additivity Check: E[f(x)] + Σφ_i = f(x)            │
                 │   - Matplotlib Headless Waterfall Plot (Base64 PNG)    │
                 └───────────────────────────┬────────────────────────────┘
                                             │
                                             ▼
                 ┌────────────────────────────────────────────────────────┐
                 │               Structured Response Output               │
                 │  - Prediction Label & Uncalibrated Leaf Probability    │
                 │  - Ranked Factors Increasing / Decreasing Risk         │
                 │  - Base64 Waterfall Vector Plot                        │
                 │  - Plain-English Clinical Summary                      │
                 │  - Mandatory Medical Disclaimer                        │
                 └────────────────────────────────────────────────────────┘
```

---

## 9. Methodology

### 9.1 Dataset Acquisition and Characteristics
The system uses the authentic UCI Chronic Kidney Disease dataset:
- **Source**: UCI Machine Learning Repository (Dataset ID: 336).
- **Instances**: 400 patient records (250 CKD, 150 Non-CKD).
- **Features**: 24 clinical parameters (14 numerical, 10 categorical) plus the binary target column `class`.
- **Target Distribution**: 62.5% CKD ($y=1$), 37.5% Non-CKD ($y=0$).

### 9.2 Stratified Partitioning and Preprocessing
To guarantee zero data contamination:
1. Data was partitioned into 80% training ($N=320$) and 20% testing ($N=80$) using stratified sampling (`random_state=42`).
2. Imputation parameters were fitted strictly on `X_train`:
   - Continuous numerical features (e.g., Serum Creatinine, Hemoglobin, Blood Urea) were imputed using median values to mitigate skewness from extreme pathological outliers.
   - Categorical features (e.g., Hypertension, Diabetes Mellitus, Red Blood Cells) were imputed using mode values and mapped into consistent binary indicators ($0 = \text{normal/absent}$, $1 = \text{abnormal/present}$).
3. The fitted transformer was serialized to `models/preprocessor.joblib` for identical inference deployment.

### 9.3 Model Selection and Training
A **Decision Tree Classifier** was trained using Scikit-learn with hyperparameters optimized for clinical interpretability:
- `criterion='gini'`
- `max_depth=5` (prevents deep, uninterpretable split chains while capturing multi-organ interactions)
- `min_samples_split=4`, `min_samples_leaf=2` (eliminates single-instance outlier leaves)
- `class_weight='balanced'` (assigns inverse-frequency weights to ensure high clinical sensitivity)
- `random_state=42` (ensures exact deterministic reproducibility)

### 9.4 SHAP Explainability Mathematical Framework
Using cooperative game theory, the Shapley value $\phi_i$ allocates the payout among players (features) according to their marginal contributions:
$$\phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

For tree models, `shap.TreeExplainer` computes exact Shapley attributions in $O(TLD^2)$ time. In our binary classification setup:
1. The explainer evaluates against the positive CKD class (Class 1).
2. The cohort baseline $E[f(x)]$ is $0.5000$ (reflecting balanced class weighting).
3. Efficiency / Additivity holds exactly:
   $$f(x) = E[f(x)] + \sum_{i=1}^{24} \phi_i(x)$$
   where positive $\phi_i > 0$ increases predicted CKD risk, and negative $\phi_i < 0$ decreases predicted CKD risk.

---

## 10. Implementation

The implementation is structured across decoupled modules:

1. [src/config.py](file:///c:/PROJECT/MINI%20PROJECT/CKD_Explainable_AI/src/config.py): Centralized repository configuration specifying absolute directory paths, target mappings, numerical bounds, categorical mappings, and human-readable feature labels with units.
2. [src/preprocessing.py](file:///c:/PROJECT/MINI%20PROJECT/CKD_Explainable_AI/src/preprocessing.py): Contains `CKDPreprocessor`, implementing string sanitation, median/mode fitting, binary encoding, and stratified data partitioning.
3. [src/train_model.py](file:///c:/PROJECT/MINI%20PROJECT/CKD_Explainable_AI/src/train_model.py): Orchestrates model instantiation, stratified 5-fold cross-validation, training on $X_{\text{train}}$, and serialization to `models/ckd_model.joblib`.
4. [src/evaluate_model.py](file:///c:/PROJECT/MINI%20PROJECT/CKD_Explainable_AI/src/evaluate_model.py): Generates confusion matrices, ROC curves, baseline model comparisons, and writes comprehensive JSON and Markdown reports.
5. [src/explain_predictions.py](file:///c:/PROJECT/MINI%20PROJECT/CKD_Explainable_AI/src/explain_predictions.py): Houses `CKDExplainer`, generating global summary bar plots, beeswarm distribution charts, and patient-level waterfall plots.
6. [backend/main.py](file:///c:/PROJECT/MINI%20PROJECT/CKD_Explainable_AI/backend/main.py): High-performance FastAPI application hosting `/health`, `/features`, `/predict`, `/metrics`, and `/dashboard` endpoints with lifespan caching.
7. [backend/schemas.py](file:///c:/PROJECT/MINI%20PROJECT/CKD_Explainable_AI/backend/schemas.py): Pydantic models enforcing strict data validation with `extra="forbid"`, physiological ranges, and structured XAI responses.
8. [frontend/index.html](file:///c:/PROJECT/MINI%20PROJECT/CKD_Explainable_AI/frontend/index.html), [frontend/style.css](file:///c:/PROJECT/MINI%20PROJECT/CKD_Explainable_AI/frontend/style.css), [frontend/script.js](file:///c:/PROJECT/MINI%20PROJECT/CKD_Explainable_AI/frontend/script.js): Accessible clinical UI featuring real-time client validation, one-click presets, probability calibration disclosures, and dynamic SHAP visualizations.

---

## 11. Results and Discussion

### 11.1 Test Set Classification Performance ($N=80$)

```
                  precision    recall  f1-score   support

     Non-CKD (0)       0.91      1.00      0.95        30
         CKD (1)       1.00      0.94      0.97        50

        accuracy                           0.96        80
       macro avg       0.95      0.97      0.96        80
    weighted avg       0.97      0.96      0.96        80
```

- **Accuracy**: $96.25\%$ ($77/80$ cases correct)
- **Precision (PPV)**: $100.00\%$ ($47/47$ positive calls correct, $FP=0$)
- **Sensitivity / Recall**: $94.00\%$ ($47/50$ CKD cases detected, $FN=3$)
- **Specificity (TNR)**: $100.00\%$ ($30/30$ healthy cases ruled out)
- **F1-Score**: $0.9691$
- **ROC-AUC**: $0.9890$

### 11.2 Baseline Model Comparison

| Model | Test Accuracy | Precision (CKD) | Recall (CKD) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Most Frequent)** | 62.50% | 62.50% | 100.00% | 0.7692 | 0.5000 |
| **Logistic Regression (Standardized)** | 97.50% | 96.15% | 100.00% | 0.9804 | 0.9980 |
| **Random Forest (100 Trees)** | 98.75% | 98.04% | 100.00% | 0.9901 | 0.9993 |
| **Primary Decision Tree (Interpretable)** | **96.25%** | **100.00%** | **94.00%** | **0.9691** | **0.9890** |

*Discussion*: While the complex 100-tree Random Forest achieves $98.75\%$ accuracy, the single Decision Tree achieves competitive performance ($96.25\%$) with $100\%$ precision, while providing direct, human-verifiable decision paths.

### 11.3 Cross-Validation Robustness
Stratified 5-fold cross-validation on the training set confirmed strong stability:
- Fold Accuracies: `[0.9531, 0.9844, 0.9062, 0.9375, 0.9688]`
- **Mean Accuracy**: `95.00% ± 3.06%`
- Mean ROC-AUC: `0.9427 ± 0.0350`

### 11.4 Global Feature Importance
Both Gini impurity reduction and mean absolute SHAP values across the test cohort identified consistent primary clinical drivers:
1. **Hemoglobin (hemo)**: 59.3% Gini influence (mean $|SHAP| = 0.1742$). Anemia strongly correlates with advanced erythropoietin deficiency in failing renal parenchyma.
2. **Specific Gravity (sg)**: 20.5% Gini influence (mean $|SHAP| = 0.1085$). Reflects the kidney's tubular concentrating capacity.
3. **Red Blood Cell Count (rbcc)**: 9.6% Gini influence (mean $|SHAP| = 0.0412$).
4. **Hypertension (htn)**: 5.2% Gini influence (mean $|SHAP| = 0.0398$).

### 11.5 Probability Calibration Analysis
On the test set, the Brier score loss is $0.0302$. Because the decision tree was grown until terminal leaf purity was reached, test inputs landing in pure leaves receive raw probabilities of $1.0$ or $0.0$. The user interface explicitly informs users that these values indicate leaf sample purity rather than true calibrated clinical probabilities.

### 11.6 Study Limitations
1. **Uncalibrated Leaf Purity**: Terminal leaf node probabilities ($0.0$ or $1.0$) represent training sample class purity rather than calibrated clinical risk estimates or Bayesian posterior probabilities.
2. **Single-Center Cohort**: The model was developed using 400 historical records from Apollo Hospitals (Tamil Nadu, India). Multi-center external validation across diverse demographics and clinical settings remains unperformed.
3. **Absence of Direct KDIGO Staging Markers**: Standard KDIGO clinical staging criteria require directly measured or calculated estimated Glomerular Filtration Rate (eGFR) and Urine Albumin-to-Creatinine Ratio (uACR), which are absent from this historical benchmark. Consequently, the model operates exclusively as a binary classifier (`CKD` vs `Non-CKD`) and **does not determine clinical CKD stages (Stage 1 to 5)**.
4. **Statistical Attribution vs. Biological Causation**: Feature attributions generated via SHAP reflect mathematical influence along learned decision paths; they do not assert physiological cause-and-effect.

---

## 12. Conclusion

This academic mini-project successfully demonstrates that healthcare machine learning models need not compromise explainability to achieve high diagnostic performance. By combining a leak-free preprocessing pipeline, a balanced Decision Tree Classifier, and exact TreeSHAP attributions, the system delivers:
1. High discriminatory accuracy ($96.25\%$) and zero false positives ($FP=0$).
2. Mathematically grounded local and global feature explanations.
3. A robust, containerized microservice and responsive web dashboard designed with privacy and safety safeguards.

---

## 13. Future Scope

1. **Probability Calibration**: Implement Platt scaling or isotonic regression to map terminal leaf frequencies into continuous, calibrated epidemiological risk estimates.
2. **Ensemble TreeSHAP Integration**: Extend the XAI pipeline to ensemble models (e.g., Random Forest, XGBoost) using GPU-accelerated TreeSHAP.
3. **Multi-Center External Validation**: Validate the trained preprocessor and estimator on modern electronic health record (EHR) cohorts.
4. **Clinical Interoperability**: Implement HL7 FHIR (Fast Healthcare Interoperability Resources) data ingestion connectors.

---

## 14. References

1. **Lundberg, S. M., & Lee, S.-I. (2017).** A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 4765–4774.
2. **Lundberg, S. M., Erion, G., Chen, H., et al. (2020).** From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67.
3. **Rubini, P. E., & Eswaran, P. (2015).** Generating comparative analysis of various classifiers on Chronic Kidney Disease dataset. *International Journal of Computer Applications*, 123(11), 38–43.
4. **Soundarapandian, P., et al. (2015).** *Chronic Kidney Disease Dataset*. UCI Machine Learning Repository. DOI: `https://doi.org/10.24432/C5G016`.
5. **KDIGO (2012).** Kidney Disease: Improving Global Outcomes (KDIGO) CKD Work Group. Clinical practice guideline for the evaluation and management of chronic kidney disease. *Kidney International Supplements*, 3(1), 1–150.
6. **Vellido, A. (2020).** The importance of interpretability and visualization in machine learning for healthcare. *Neural Computing and Applications*, 32(24), 18069–18083.
7. **Shapley, L. S. (1953).** A value for n-person games. *Contributions to the Theory of Games*, 2(28), 307–317.
