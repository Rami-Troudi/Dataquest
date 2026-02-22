# Technical Report — Insurance Bundle Recommender

> **Team:** DataQuest  
> **Date:** February 2026  
> **Phase:** II — Productization & Deployment

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Feature Engineering](#2-feature-engineering)
3. [Model Selection & Justification](#3-model-selection--justification)
4. [Explainability](#4-explainability)
5. [API & Frontend](#5-api--frontend)
6. [Deployment](#6-deployment)
7. [EDA Highlights](#7-eda-highlights)
8. [Failed Attempts & Lessons Learned](#8-failed-attempts--lessons-learned)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                     │
│                    frontend/index.html                       │
│         Form Input ──► POST /predict ──► Display Result     │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP (JSON)
┌────────────────────────────▼────────────────────────────────┐
│                   FastAPI Backend (api/)                     │
│                                                             │
│  GET  /health         → Model & system status               │
│  POST /predict        → Bundle prediction + reason codes    │
│  GET  /explain/global → Ranked feature importances          │
│  GET  /               → Serves frontend                     │
│                                                             │
│  ┌───────────────────────────────────────────────────┐      │
│  │              Inference Pipeline                    │      │
│  │  1. Validate input (Pydantic schemas)             │      │
│  │  2. Build DataFrame from JSON                     │      │
│  │  3. preprocess() — feature engineering            │      │
│  │  4. _transform_with_preprocessor() — encode       │      │
│  │  5. rf_model.predict_proba() — probabilities      │      │
│  │  6. Local reason codes — explainability            │      │
│  │  7. Return JSON response                          │      │
│  └───────────────────────────────────────────────────┘      │
│                                                             │
│  Model Bundle (model.pkl)                                   │
│  ├── rf_model          (RandomForestClassifier)             │
│  ├── preprocessor      (encodings, medians, col order)      │
│  └── class_order       (label mapping)                      │
└─────────────────────────────────────────────────────────────┘
                             │
                    Docker Container
                  (Dockerfile + compose)
```

### Key Design Decisions

- **Single-process serving:** Uvicorn serves both API and static frontend from one container — simple, resource-efficient, no CORS issues.
- **Model cached on startup:** `model.pkl` is loaded once during FastAPI lifespan and kept in memory. Cold start ≈ 2 seconds.
- **Preprocessing reuse:** The same `solution.py` functions are imported by both the competition judge and the API, ensuring zero drift between evaluation and production.

---

## 2. Feature Engineering

We engineered **22 new features** from the original 27 columns. All transformations are in `solution.py::preprocess()`.

### Feature Groups

| # | Feature | Formula / Logic | Rationale |
|---|---------|-----------------|-----------|
| 1 | `Total_Dependents` | Adult + Child + Infant | Family size is a strong proxy for bundle type |
| 2 | `Child_Ratio` | Child / (Total + 1) | Families with many children lean toward Family bundles |
| 3 | `Adult_Ratio` | Adult / (Total + 1) | Adult-only households prefer Auto or Health |
| 4 | `Infant_Ratio` | Infant / (Total + 1) | Infants signal comprehensive coverage needs |
| 5 | `Total_Policy_Duration_Years` | Prior months / 12 | Normalize tenure to years |
| 6 | `Claims_Frequency` | Claims / (Prior months + 1) | Incident rate per unit time |
| 7 | `Claims_to_YearsWithout_Ratio` | Claims / (Claim-free years + 1) | Risk behavior signal |
| 8 | `Income_per_Dependent` | Income / (Total deps + 1) | Spending power per covered person |
| 9 | `Total_Processing_Days` | Quote days + UW days | Total pipeline duration |
| 10 | `Processing_Efficiency` | UW days / (Quote days + 1) | How fast underwriting moved |
| 11 | `Amendments_per_Day` | Amendments / (Quote days + 1) | Policy modification intensity |
| 12 | `Vehicles_per_Adult` | Vehicles / (Adults + 1) | Vehicle density per adult |
| 13 | `High_Risk_Customer` | Claims > 2 AND cancelled | Binary risk flag |
| 14-15 | `Policy_Start_Month_Sin/Cos` | Cyclical encoding of month | Captures seasonality without ordinal artifacts |
| 16-17 | `Policy_Start_Day_Sin/Cos` | Cyclical encoding of day | Day-of-month patterns |
| 18-19 | `Policy_Start_Week_Sin/Cos` | Cyclical encoding of week | Week-of-year patterns |
| 20 | `Loyalty_Score` | 0.3×ClaimFreeYrs + 0.02×Duration − 0.5×Claims | Composite loyalty metric |
| 21 | `Payment_Flexibility` | Grace period extensions | Alias for readability |
| 22 | `Policy_Start_Month_Num` | Month string → integer | Numeric month for model consumption |

### Top 10 Features by Model Importance

| Rank | Feature | Importance (%) |
|------|---------|---------------|
| 1 | Estimated_Annual_Income | 14.76% |
| 2 | Total_Dependents | 14.26% |
| 3 | Broker_Agency_Type | 9.16% |
| 4 | Broker_ID | 8.65% |
| 5 | Adult_Ratio | 5.93% |
| 6 | Deductible_Tier | 4.84% |
| 7 | Income_per_Dependent | 4.38% |
| 8 | Adult_Dependents | 3.46% |
| 9 | Acquisition_Channel | 2.72% |
| 10 | Policy_Start_Year | 2.60% |

**Insight:** Income and family composition together account for ~29% of predictive power, confirming that bundle selection is strongly driven by household economics.

---

## 3. Model Selection & Justification

### Why Random Forest?

We evaluated multiple approaches and selected **Random Forest** as the final model for these reasons:

| Criterion | Random Forest | XGBoost | Neural Network |
|-----------|:------------:|:-------:|:--------------:|
| Macro F1 (validation) | **0.60** | 0.58 | 0.52 |
| Model size (compressed) | **3.15 MB** | 8+ MB | 50+ MB |
| Inference latency | **~30ms** | ~40ms | ~200ms |
| Interpretability | High | Medium | Low |
| Overfitting risk | Low | Medium | High |

### Training Strategy

1. **Stratified 80/20 split** — preserves class distribution.
2. **Randomized hyperparameter search** (20 iterations) across:
   - `n_estimators`: 50–180
   - `max_depth`: 8–24
   - `min_samples_split`, `min_samples_leaf`, `max_features`, `class_weight`, `criterion`
3. **Neighborhood refinement** — 40 additional candidates around the best configuration.
4. **Minority oversampling** — tested plain, 300-sample floor, and 500-sample floor to handle class imbalance (class 8 and 9 are extremely rare).
5. **Size-aware selection** — candidates exceeding 35 MB were discarded automatically.

### Final Configuration

```python
{
    'n_estimators': 50,
    'max_depth': 16,
    'min_samples_split': 10,
    'min_samples_leaf': 9,
    'max_features': 0.7,
    'class_weight': None,
    'criterion': 'entropy',
    'oversampling': 'min_count=300'
}
```

### Final Metrics

| Metric | Value |
|--------|-------|
| Validation Macro F1 | 0.6009 |
| Model size (compressed) | 3.15 MB |
| Inference latency | ~29 ms |
| Score proxy (F1 × size × latency) | 0.5394 |

---

## 4. Explainability

We implemented two complementary explainability methods in `explainability.py`:

### Global Explainability
- Uses Random Forest's Gini / entropy-based `feature_importances_`.
- Outputs a ranked table with absolute and percentage importance.
- Exported to `explainability_global_importance.csv`.

### Local Explainability (Per-Prediction Reason Codes)
- For each prediction, computes: `local_score[i] = |feature_value[i] − baseline_median[i]| × global_importance[i]`
- Selects the top-K features with highest local score as "reason codes."
- Decodes categorical values back to human-readable labels.
- Each API response includes 3 reason codes per prediction.
- Exported to `explainability_reason_codes.csv` for batch analysis.

### Why This Approach

- **No extra dependencies** — works with sklearn's built-in importances.
- **Fast** — no SHAP overhead; runs in <1ms per row.
- **Business-readable** — outputs like `Estimated_Annual_Income=65000` are immediately understandable.

---

## 5. API & Frontend

### API Endpoints

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `GET` | `/health` | Model health check | `{status, model_loaded, model_type, num_features, num_classes}` |
| `POST` | `/predict` | Predict bundles | `{predictions: [{User_ID, predicted_bundle_id, predicted_bundle_name, confidence, reason_codes}]}` |
| `GET` | `/explain/global` | Feature importance | `{features: [{rank, feature, importance, importance_pct}]}` |
| `GET` | `/` | Frontend UI | HTML page |

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| Schemas | Pydantic v2 |
| Frontend | Vanilla HTML/CSS/JS (single page) |
| Model serving | joblib + scikit-learn |
| Container | Docker + docker-compose |

### Frontend Features

- Grouped input form (demographics, history, policy, sales, timeline)
- Real-time prediction with confidence bar
- Top-3 reason code display per prediction
- Global feature importance bar chart
- Live health status badge

---

## 6. Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements-api.txt

# Start server
cd Dataquest
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Open browser
http://localhost:8000
```

### Docker Deployment

```bash
# Build and run
docker-compose up --build -d

# Access
http://localhost:8000
```

### Container Specs

| Parameter | Value |
|-----------|-------|
| Base image | `python:3.10-slim` |
| Memory limit | 1 GB |
| CPU limit | 1 core |
| Port | 8000 |

---

## 7. EDA Highlights

Key observations from exploratory data analysis:

1. **Severe class imbalance:** Class 2 (Basic_Health) dominates at ~60% of training data. Classes 8 and 9 have <10 samples in validation.
2. **Income is bimodal:** Clear separation between low-income (health/renter bundles) and high-income (auto/home/premium bundles).
3. **Family composition matters:** Households with children almost exclusively choose Family_Comprehensive or Health_Dental_Vision.
4. **Broker channel effect:** Urban_Boutique brokers sell different product mixes than National_Corporate.
5. **Temporal stability:** No significant concept drift across policy start years (2014–2024).

---

## 8. Failed Attempts & Lessons Learned

| Attempt | Outcome | Lesson |
|---------|---------|--------|
| XGBoost with 500 estimators | F1 ~0.58 but model was 12 MB | Diminishing returns at scale; size penalty hurt |
| MLP ensemble (RF + neural net) | F1 ~0.57, slower inference | Neural net added complexity without improving F1 |
| SMOTE oversampling | F1 decreased vs. simple random oversampling | SMOTE generated noisy synthetic samples for rare classes |
| Target encoding for Broker_ID | Slight improvement but risk of leakage | Kept ordinal encoding for safety |
| Deep trees (max_depth=30+) | Overfitting, validation F1 dropped | Depth 16 was the sweet spot |

---

## Appendix: Repository Structure

```
Dataquest/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI backend
│   └── schemas.py            # Pydantic models
├── frontend/
│   └── index.html            # Prediction UI
├── solution.py               # Competition interface (preprocess, load_model, predict)
├── explainability.py          # Global + local explainability
├── v4_rf_only.py             # Training pipeline
├── test_solution.py          # Validation script
├── model.pkl                 # Trained model bundle (compressed)
├── requirements-api.txt      # API dependencies
├── Dockerfile                # Container build
├── docker-compose.yml        # Orchestration
├── .gitignore                # Git exclusions
├── TECHNICAL_REPORT.md       # This document
└── README.md                 # Project overview & quickstart
```
