# DataQuest Phase II — Implementation Action Plan

**Timebox**: 3 hours  
**Team**: 4 members (parallelization plan included)  
**Model**: LightGBM (primary, `submission/model.joblib`) + RandomForest (backup, `random forest/model.joblib`)  
**Phase I Final Score**: ~0.45–0.57 (Macro-F1 0.55–0.61 × size/latency penalties)

---

## 1) Phase II Goal & Success Criteria

### Definition of Done
- **API**: Running FastAPI server with ≥2 endpoints (`POST /predict`, `GET /health`), accepts JSON input, returns predicted bundle + probabilities.
- **UI**: Working Next.js + shadcn/ui frontend with an input form, prediction results with top-k probabilities, and bundle name mapping.
- **Report**: PDF/Markdown technical report with architecture diagram, feature engineering explanation, model justification, EDA notes, failed attempts.
- **Pitch**: 5–7 slide deck covering business value, architecture, demo screenshots, results.
- **Repo**: Clean structure, README with setup instructions, `.gitignore`, no credentials.

### Must-Have (Minimum Viable)
- [ ] `POST /predict` endpoint — accepts customer JSON, returns bundle prediction + probabilities
- [ ] `GET /health` endpoint — returns model version, uptime, status
- [ ] UI form with all input fields → submit → display prediction + bundle name + top-3 probabilities
- [ ] Technical report (architecture diagram, features, model choice)
- [ ] Clean repo with `README.md`

### Nice-to-Have (Bonus — +25 pts)
- [ ] `GET /schema` endpoint (returns input schema)
- [ ] `POST /explain` endpoint (SHAP/feature importance for a prediction)
- [ ] GitHub Actions CI/CD (lint + test)
- [ ] `train.py` retraining stub with config
- [ ] In-memory LRU cache on `/predict`
- [ ] Live deployment (Vercel for UI + Render for API)
- [ ] Pre-filled demo examples in UI
- [ ] Batch prediction endpoint

---

## 2) Repo Audit Summary

### Current Structure
```
Dataquest/
├── Context/                  # Hackathon brief, plans, template
│   ├── Context.md            # Full competition rules (transcription)
│   ├── plan-phaseI.md        # Phase I action plan
│   ├── solution.py           # Original template (empty)
│   └── DataQuest-Brief-Document (1).pdf
├── Data/
│   ├── train (1).csv         # 60,868 rows × 29 cols
│   └── test.csv              # 15,218 rows × 28 cols
├── Talel/                    # Main model development
│   ├── train_final.py        # Final LightGBM training pipeline
│   ├── train_fast.py         # Fast iteration training
│   ├── ablation.py           # Feature ablation experiments
│   ├── grid_search.py        # Hyperparameter search
│   ├── validate_submission.py
│   ├── train_fast_report.json
│   ├── tuning_stage1.csv
│   └── tuning_stage2.csv
├── random forest/            # RF alternative model
│   ├── train_rf.py
│   ├── solution.py
│   ├── model.joblib
│   └── predictions.csv
├── submission/               # Final submission artifacts
│   ├── solution.py           # Judge submission code (EMPTY — rewritten each run)
│   ├── model.joblib           # LightGBM trained model (~1 MB)
│   ├── requirements.txt      # Empty
│   └── predictions.csv
├── Submission Template/
│   ├── README.md
│   └── solution.py           # Original template
├── current_progress.md
├── submission_ready.zip
└── train.log
```

### What Exists (Reusable)
- **LightGBM model**: `submission/model.joblib` — 1 MB, contains booster + feature maps + label encoders + thresholds
- **Feature engineering**: `Talel/train_final.py::all_features()` — 49 features, well-documented
- **Inference logic**: `submission/solution.py` (but currently EMPTY — needs rewrite) and `random forest/solution.py` (has full pipeline)
- **Training pipelines**: `train_final.py`, `train_rf.py` — both functional
- **Ablation results**: `ablation.py`, tuning CSVs
- **Class mapping**: available in Context.md and README.md

### Gaps (Must Fix)
- [ ] **No API** — no `api/` folder, no FastAPI/Flask code
- [ ] **No UI** — no frontend code at all
- [ ] **No technical report** — no `docs/report.md` or PDF
- [ ] **No architecture diagram** — needs to be created
- [ ] **No tests** — zero test files
- [ ] **No CI/CD** — no `.github/workflows/`
- [ ] **No requirements.txt for API/UI** — only the empty submission one
- [ ] **No Docker/deployment** — no Dockerfile, no deploy config
- [ ] **`submission/solution.py` is EMPTY** — inference wrapper must be restored
- [ ] **Messy folder names** — spaces in "random forest", "train (1).csv"
- [ ] **No `.gitignore`** — `.venv/`, `__pycache__/`, data files exposed

---

## 3) Target Architecture

### Architecture Diagram Description
```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                              │
│   Next.js + shadcn/ui (or Streamlit fallback)                │
│   ┌──────────┐  ┌───────────┐  ┌──────────────────┐        │
│   │ Input    │→ │ POST      │→ │ Result Display    │        │
│   │ Form     │  │ /predict  │  │ - Bundle name     │        │
│   │ (fields) │  │ via fetch │  │ - Probabilities   │        │
│   └──────────┘  └───────────┘  │ - Confidence bar  │        │
│                                 └──────────────────┘        │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP (JSON)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      API (FastAPI)                            │
│                                                               │
│  GET /health  → {status, model_version, uptime}              │
│  POST /predict → {bundle_id, bundle_name, probabilities}     │
│  GET /schema   → {input_fields: [...]}  (bonus)              │
│  POST /explain → {feature_importances}  (bonus)              │
│                                                               │
│  ┌────────────┐  ┌────────────────┐  ┌──────────────┐       │
│  │ Pydantic   │→ │ Feature        │→ │ LightGBM     │       │
│  │ Validation │  │ Engineering    │  │ Booster      │       │
│  │            │  │ (all_features) │  │ .predict()   │       │
│  └────────────┘  └────────────────┘  └──────┬───────┘       │
│                                              │               │
│                                    ┌─────────▼─────────┐    │
│                                    │ Threshold Adjust   │    │
│                                    │ (proba × weights)  │    │
│                                    │ → argmax → class   │    │
│                                    └───────────────────┘    │
│                                                               │
│  Cache: functools.lru_cache on serialized input (bonus)      │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODEL ARTIFACT                             │
│  model.joblib (~1 MB)                                        │
│  Contains: booster, fe_maps, label_enc, feature_list,        │
│            cat_cols, thresholds, predict_cfg                  │
└─────────────────────────────────────────────────────────────┘
```

### Inference Pipeline (exact flow)
1. **Request arrives** at `POST /predict` → JSON body with customer fields
2. **Pydantic validation** → `CustomerInput` model validates types, ranges, enums
3. **Convert to DataFrame** → single-row `pd.DataFrame`
4. **Feature engineering** → `all_features(df)` from `core/features.py` (same as training)
5. **Frequency encoding** → map `Broker_ID`, `Employer_ID`, `Region_Code` using `fe_maps` from artifact
6. **Label encoding** → map `CAT_COLS` using `label_enc` from artifact
7. **Matrix construction** → reindex to `feature_list`, convert to `float32` numpy array
8. **Prediction** → `booster.predict(X)` → 10-class probability vector
9. **Threshold adjustment** → `(proba * thresholds).argmax()` → class ID
10. **Response** → `{bundle_id: 2, bundle_name: "Basic_Health", probabilities: {...}}`

### Target Folder Layout
```
Dataquest/
├── api/
│   ├── main.py               # FastAPI app, endpoints
│   ├── models.py             # Pydantic input/output schemas
│   ├── inference.py          # Model loading + prediction logic
│   ├── requirements.txt      # fastapi, uvicorn, joblib, etc.
│   └── Dockerfile            # (bonus) containerized API
├── ui/
│   ├── package.json
│   ├── next.config.js
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx          # Main prediction form
│   │   └── globals.css
│   ├── components/
│   │   ├── prediction-form.tsx
│   │   ├── result-card.tsx
│   │   └── ui/               # shadcn components
│   └── lib/
│       └── constants.ts      # Bundle mapping, field definitions
├── core/
│   ├── features.py           # all_features() — shared between train & API
│   ├── config.py             # Paths, constants
│   └── train.py              # Retraining stub (bonus)
├── models/
│   └── model.joblib          # Trained model artifact
├── docs/
│   ├── report.md             # Technical report
│   ├── architecture.png      # Architecture diagram
│   └── slides.md             # Pitch deck outline
├── tests/
│   ├── test_api.py
│   └── test_features.py
├── .github/
│   └── workflows/
│       └── ci.yml            # Lint + test (bonus)
├── .gitignore
├── README.md                 # Setup instructions
├── docker-compose.yml        # (bonus) orchestration
│
│ # Phase I artifacts (kept for reference)
├── Data/                     # Training data (gitignored)
├── Talel/                    # Training scripts
├── submission/               # Judge submission
└── Context/                  # Competition docs
```

---

## 4) API Implementation Plan (≥2 endpoints)

### Technology Choice: **FastAPI**
- Async, auto-docs (Swagger), Pydantic-native, minimal boilerplate
- Runs with `uvicorn api.main:app`

### Endpoint 1: `POST /predict`

**Input Schema** (`CustomerInput` — Pydantic BaseModel):
```python
class CustomerInput(BaseModel):
    User_ID: str = "USR_000001"
    Adult_Dependents: int = Field(ge=0, le=20, default=0)
    Child_Dependents: int = Field(ge=0, le=20, default=0)
    Infant_Dependents: int = Field(ge=0, le=10, default=0)
    Estimated_Annual_Income: float = Field(ge=0, default=50000)
    Employment_Status: str = "Employed"
    Region_Code: Optional[str] = None
    Existing_Policyholder: int = Field(ge=0, le=1, default=0)
    Previous_Claims_Filed: int = Field(ge=0, default=0)
    Years_Without_Claims: int = Field(ge=0, default=0)
    Previous_Policy_Duration_Months: int = Field(ge=0, default=12)
    Policy_Cancelled_Post_Purchase: int = Field(ge=0, le=1, default=0)
    Deductible_Tier: str = "Medium"
    Payment_Schedule: str = "Monthly"
    Vehicles_on_Policy: int = Field(ge=0, default=1)
    Custom_Riders_Requested: int = Field(ge=0, default=0)
    Grace_Period_Extensions: int = Field(ge=0, default=0)
    Days_Since_Quote: int = Field(ge=0, default=30)
    Underwriting_Processing_Days: int = Field(ge=0, default=7)
    Policy_Amendments_Count: int = Field(ge=0, default=0)
    Acquisition_Channel: str = "Online"
    Broker_Agency_Type: str = "Independent"
    Broker_ID: Optional[str] = None
    Employer_ID: Optional[str] = None
    Policy_Start_Year: int = 2026
    Policy_Start_Month: str = "February"
    Policy_Start_Week: int = 8
    Policy_Start_Day: int = 22
```

**Output Schema**:
```python
class PredictionOutput(BaseModel):
    user_id: str
    predicted_bundle_id: int
    predicted_bundle_name: str
    confidence: float
    probabilities: dict[str, float]   # {bundle_name: probability}
    latency_ms: float
```

**Payload Example — Request**:
```json
{
  "User_ID": "USR_099999",
  "Adult_Dependents": 2,
  "Child_Dependents": 1,
  "Infant_Dependents": 0,
  "Estimated_Annual_Income": 75000,
  "Employment_Status": "Employed",
  "Region_Code": "R_01",
  "Existing_Policyholder": 1,
  "Previous_Claims_Filed": 0,
  "Years_Without_Claims": 5,
  "Previous_Policy_Duration_Months": 36,
  "Policy_Cancelled_Post_Purchase": 0,
  "Deductible_Tier": "Medium",
  "Payment_Schedule": "Monthly",
  "Vehicles_on_Policy": 2,
  "Custom_Riders_Requested": 1,
  "Grace_Period_Extensions": 0,
  "Days_Since_Quote": 14,
  "Underwriting_Processing_Days": 5,
  "Policy_Amendments_Count": 0,
  "Acquisition_Channel": "Online",
  "Broker_Agency_Type": "Independent",
  "Broker_ID": null,
  "Employer_ID": null,
  "Policy_Start_Year": 2026,
  "Policy_Start_Month": "February",
  "Policy_Start_Week": 8,
  "Policy_Start_Day": 22
}
```

**Payload Example — Response**:
```json
{
  "user_id": "USR_099999",
  "predicted_bundle_id": 4,
  "predicted_bundle_name": "Health_Dental_Vision",
  "confidence": 0.482,
  "probabilities": {
    "Auto_Comprehensive": 0.031,
    "Auto_Liability_Basic": 0.052,
    "Basic_Health": 0.287,
    "Family_Comprehensive": 0.089,
    "Health_Dental_Vision": 0.482,
    "Home_Premium": 0.012,
    "Home_Standard": 0.018,
    "Premium_Health_Life": 0.024,
    "Renter_Basic": 0.003,
    "Renter_Premium": 0.002
  },
  "latency_ms": 12.3
}
```

**Error Handling**:
- 422: Pydantic validation error (auto from FastAPI)
- 500: Model inference failure → return `{"error": "Inference failed", "detail": str(e)}`

### Endpoint 2: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model_version": "lgbm_v2_2026_02_22",
  "model_size_mb": 1.01,
  "uptime_seconds": 3412.5,
  "features_count": 49,
  "num_classes": 10
}
```

### Bonus Endpoints

**`GET /schema`** → Returns the input field definitions with types and valid ranges.  
**`POST /explain`** → Returns LightGBM feature importances (gain) for the top-10 most important features for a given input.  
**`POST /predict/batch`** → Accepts a list of customers, returns list of predictions.

### Caching Strategy
- `functools.lru_cache(maxsize=256)` on a hashable version of input (tuple of sorted items)
- Invalidated on model reload
- Expected cache hit rate: low for real traffic, useful for demo repeated clicks

### Latency Considerations
- Model loading: once at startup (`@app.on_event("startup")`)
- Feature engineering: ~5ms for single row
- LightGBM predict: ~1ms for single row (vs ~500ms for 15K batch)
- Total expected: <20ms per request

---

## 5) UI Implementation Plan

### Technology: **Next.js 14 + shadcn/ui + Tailwind CSS**
- Modern, fast, beautiful UI out of the box
- Fallback: Streamlit (if time is too short — 30min implementation)

### UI Flow
```
┌─────────────────────────────────────────────┐
│         Insurance Bundle Recommender         │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │  CUSTOMER INFORMATION FORM             │  │
│  │                                        │  │
│  │  [Demographics]  [History]  [Policy]   │  │
│  │    (tabs or accordion)                 │  │
│  │                                        │  │
│  │  Adult Dependents: [___]               │  │
│  │  Child Dependents: [___]               │  │
│  │  Income: [___]                         │  │
│  │  Employment: [dropdown ▼]              │  │
│  │  ... (all 27 fields)                   │  │
│  │                                        │  │
│  │  [🔮 Predict Bundle]  [📋 Load Demo]  │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │  PREDICTION RESULT                     │  │
│  │                                        │  │
│  │  Recommended: Health_Dental_Vision     │  │
│  │  Confidence: 48.2%                     │  │
│  │                                        │  │
│  │  [═══════════] Basic_Health     28.7%  │  │
│  │  [════════]    Family_Comp      8.9%   │  │
│  │  [════]        Auto_Liability   5.2%   │  │
│  │  ...                                   │  │
│  └────────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### Bundle Name Mapping (display in UI)
```typescript
const BUNDLE_MAP: Record<number, { name: string; icon: string; color: string }> = {
  0: { name: "Auto Comprehensive",    icon: "🚗", color: "#3B82F6" },
  1: { name: "Auto Liability Basic",  icon: "🚙", color: "#60A5FA" },
  2: { name: "Basic Health",          icon: "🏥", color: "#10B981" },
  3: { name: "Family Comprehensive",  icon: "👨‍👩‍👧‍👦", color: "#8B5CF6" },
  4: { name: "Health Dental Vision",  icon: "🦷", color: "#06B6D4" },
  5: { name: "Home Premium",          icon: "🏠", color: "#F59E0B" },
  6: { name: "Home Standard",         icon: "🏡", color: "#FBBF24" },
  7: { name: "Premium Health Life",   icon: "💎", color: "#EC4899" },
  8: { name: "Renter Basic",          icon: "🔑", color: "#6366F1" },
  9: { name: "Renter Premium",        icon: "🏢", color: "#8B5CF6" },
};
```

### Demo Examples (Pre-filled)
Prepare 3 pre-filled profiles:
1. **Young Renter**: single, no dependents, low income, renter → expects class 8/9
2. **Family with Health**: 2 adults, 1 child, mid income, existing policyholder → expects class 3/4
3. **Homeowner**: high income, 0 claims, long policy history → expects class 5/6

### Streamlit Fallback (if Next.js too slow)
- Single `ui/app.py` file, ~100 lines
- `st.form()` with all fields → `requests.post(API_URL + "/predict")` → `st.bar_chart()` for probabilities
- Command: `streamlit run ui/app.py`
- Time to implement: ~30 minutes

---

## 6) Technical Report Plan (10 points)

### Outline
```markdown
# DataQuest: Insurance Bundle Recommender — Technical Report

## 1. Problem Statement & Objective (0.5 page)
- Business context: personalized insurance recommendations
- Technical framing: 10-class classification, Macro-F1 metric
- Success criteria: score = F1 × size_penalty × latency_penalty

## 2. Data Overview (0.5 page)
- 60,868 training samples, 15,218 test samples
- 25 raw features across 5 categories
- Class imbalance: class 2 (Basic_Health) dominates at 59%, classes 8/9 have <10 samples
- Table: class distribution with percentages
- Note on missing values (Broker_ID, Employer_ID nullable)

## 3. Exploratory Data Analysis (1 page — bonus)
- Class distribution bar chart
- Correlation heatmap of numeric features
- Income distribution by bundle (box plot)
- Key insight: temporal features (Year/Week/Day) are unexpectedly predictive (+3pp)

## 4. Feature Engineering (1.5 pages)
- 49 total features from 25 raw columns
- Table of all engineered features with formula and rationale:
  | Feature | Formula | Signal |
  |---------|---------|--------|
  | Total_Dependents | Adult + Child + Infant | Family size → bundle type |
  | Claims_Rate | Claims / (Duration + 1) | Risk profile |
  | Risk_Score | Claims / (ClaimFreeYears + 1) | Normalized risk |
  | Loyalty_Score | 0.3×YearsNoClaim + 0.02×Duration - 0.5×Claims | Retention |
  | Policy_Complexity | Riders + Vehicles | Product affinity |
  | Tenure_Bucket | np.digitize(duration, [1,2,3,4]) | Experience tier |
  | Month_Sin/Cos | Cyclical encoding | Seasonality |
  | Frequency encoding | Broker/Employer/Region count ratios | Entity popularity |
  | ... | ... | ... |
- Ablation study summary: which features moved the needle

## 5. Model Choice & Justification (1 page)
- LightGBM chosen over: RandomForest (slower, larger), XGBoost (comparable but slower), Neural nets (too large for 1MB budget)
- Key hyperparameters: 31 leaves, depth 5, lr 0.1, 100 rounds
- Class balancing: compute_sample_weight("balanced") with max clip at 10.0
- Threshold optimization: greedy per-class with constrained candidates [0.5–2.0], skip classes <50 samples
- Model artifact: 1 MB joblib (booster + encoders + thresholds)
- Comparison table:
  | Model | CV F1 | Size | Latency |
  |-------|-------|------|---------|
  | LightGBM (final) | 0.608 | 1.0 MB | ~0.8s |
  | RandomForest | 0.52 | ~5 MB | ~1.5s |
  | LightGBM (aggressive) | 0.640 | 1.1 MB | ~0.8s |

## 6. System Architecture (0.5 page)
- Diagram: User → Next.js UI → FastAPI → Feature Engineering → LightGBM → Response
- Deployment topology: local Docker Compose or Vercel + Render

## 7. Experiments & Failed Attempts (1 page — bonus)
- pd.qcut for Tenure_Bucket → failed on judge (dynamic bins)
- Unconstrained threshold optimization → massive overfitting on rare classes
- UTF-8 BOM and non-ASCII characters → judge rejection
- Aggressive sample weights → memorized 5-sample classes
- Feature ablation: some features (Income_Decile, Income_Per_Dependent) had no impact

## 8. Limitations & Next Steps (0.5 page)
- Class 8 (6 samples) and Class 9 (5 samples) are essentially unlearnable
- Threshold optimization overfits on OOF — need held-out threshold validation
- No online learning or feedback loop
- Future: ensemble LightGBM + CatBoost, target encoding with proper regularization
```

### Artifacts to Generate
- [ ] Class distribution bar chart (matplotlib, save as PNG)
- [ ] Architecture diagram (draw.io or Mermaid → PNG)
- [ ] Feature importance plot (LightGBM gain)
- [ ] Confusion matrix heatmap
- [ ] Model comparison table
- [ ] Ablation results table (from `Talel/ablation.py` output)
- [ ] Feature engineering summary table

---

## 7) 3-Hour Timeline & Team Parallelization

### Team Roles (4 members)

| Person | Role | Focus Area |
|--------|------|------------|
| **P1** | API Engineer | FastAPI, endpoints, inference.py |
| **P2** | Frontend Engineer | Next.js + shadcn/ui (or Streamlit fallback) |
| **P3** | Report & Docs | Technical report, diagrams, EDA plots |
| **P4** | MLOps & Integration | CI/CD, Dockerfile, testing, repo cleanup |

### Hour 1 (0:00–1:00) — Foundation

**P1 (API)**:
- [ ] Create `api/` folder structure
- [ ] Implement `api/models.py` (Pydantic schemas)
- [ ] Implement `api/inference.py` (load model + predict logic — extract from `train_final.py`)
- [ ] Implement `api/main.py` with `POST /predict` and `GET /health`
- [ ] Test with `curl` or Swagger UI

**P2 (Frontend)**:
- [ ] `npx create-next-app@latest ui --typescript --tailwind`
- [ ] `npx shadcn@latest init` + add components (button, input, card, select, badge)
- [ ] Build `components/prediction-form.tsx` with all 27 input fields
- [ ] Build `components/result-card.tsx` with bundle name + probability bars
- [ ] Wire up API call in `app/page.tsx`

**P3 (Report)**:
- [ ] Create `docs/report.md` with full outline
- [ ] Write sections 1–3 (Problem, Data, EDA)
- [ ] Generate EDA plots (class distribution, correlation heatmap) — save to `docs/`
- [ ] Draft feature engineering table (section 4)

**P4 (MLOps + Repo)**:
- [ ] Clean repo structure: move model to `models/`, create `core/features.py`
- [ ] Create `.gitignore` (Data/, .venv/, __pycache__/, *.pyc, .env)
- [ ] Create root `README.md` with project overview + setup instructions
- [ ] Extract `core/features.py` from `Talel/train_final.py::all_features()`
- [ ] Create `core/train.py` retraining stub
- [ ] Set up `tests/test_api.py` (basic smoke test)

### Hour 2 (1:00–2:00) — Integration & Polish

**P1 (API)**:
- [ ] Add error handling, CORS middleware
- [ ] Add `/schema` endpoint (bonus)
- [ ] Implement in-memory LRU caching on `/predict`
- [ ] Write `api/Dockerfile` (bonus)
- [ ] Integration test: UI → API → Response flow

**P2 (Frontend)**:
- [ ] Add 3 pre-filled demo examples (load on button click)
- [ ] Style results: color-coded probability bars, bundle icons
- [ ] Add loading spinner, error handling
- [ ] Test full flow with running API
- [ ] Responsive layout adjustments

**P3 (Report)**:
- [ ] Write sections 5–6 (Model Choice, Architecture)
- [ ] Create architecture diagram (Mermaid or draw.io)
- [ ] Generate feature importance plot + confusion matrix
- [ ] Write section 7 (failed attempts)

**P4 (MLOps + Integration)**:
- [ ] Create `.github/workflows/ci.yml` (lint + test)
- [ ] Create `docker-compose.yml` (API + UI)
- [ ] Run full end-to-end test locally
- [ ] Fix any integration issues

### Hour 3 (2:00–3:00) — Finalize & Demo

**P1 + P2 (Demo prep)**:
- [ ] Record demo walkthrough or prepare live demo script
- [ ] Fix any last-minute UI/API bugs
- [ ] (Bonus) Deploy to Render (API) + Vercel (UI)

**P3 (Report finalize)**:
- [ ] Write section 8 (Limitations + Next Steps)
- [ ] Proofread, format, export to PDF if needed
- [ ] Add all plots/diagrams to report

**P4 (Pitch + Final packaging)**:
- [ ] Create 5–7 slide pitch deck:
  - Slide 1: Problem & business value
  - Slide 2: Data & class distribution
  - Slide 3: Feature engineering highlights
  - Slide 4: Model architecture diagram
  - Slide 5: Results (F1 score, comparison table)
  - Slide 6: Live demo / screenshots
  - Slide 7: Next steps & business impact
- [ ] Final `git push`, verify repo is clean
- [ ] Verify README has setup instructions

---

## 8) Mandatory & Bonus Implementation Details

### MLOps: CI/CD (GitHub Actions)

**`.github/workflows/ci.yml`**:
```yaml
name: CI
on: [push, pull_request]
jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -r api/requirements.txt
      - run: pip install ruff pytest
      - run: ruff check api/ core/
      - run: pytest tests/ -v
```

### Retraining Stub

**`core/train.py`**:
```python
"""
Retraining pipeline.
Usage: python core/train.py --data Data/train.csv --output models/model.joblib
"""
# Steps:
# 1. Load CSV
# 2. Feature engineering (core.features.all_features)
# 3. Compute label encoders + freq maps from training data
# 4. 5-fold stratified CV with LightGBM
# 5. Threshold optimization
# 6. Retrain on full data
# 7. Save artifact to --output
```

### Caching

```python
from functools import lru_cache
import hashlib, json

def _hash_input(data: dict) -> str:
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()

# In predict endpoint:
@lru_cache(maxsize=256)
def cached_predict(input_hash: str, data_tuple):
    # reconstruct dict from tuple, run inference
    ...
```

### Deployment

**Primary Path (local demo)**:
```bash
# Terminal 1: API
cd api && pip install -r requirements.txt && uvicorn main:app --port 8000

# Terminal 2: UI
cd ui && npm install && npm run dev
# Open http://localhost:3000
```

**Bonus Path (cloud)**:
- **API → Render**: Free tier, add `render.yaml` with build command `pip install -r api/requirements.txt`, start command `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- **UI → Vercel**: Connect repo, set root directory to `ui/`, auto-detects Next.js
- **Environment variable**: `NEXT_PUBLIC_API_URL` pointing to Render URL

**Fallback (simplest possible)**:
- If Next.js is too slow: replace with Streamlit (`ui/app.py`, 100 lines)
- `streamlit run ui/app.py` — single command, no build step

---

## 9) Key Commands Reference

```bash
# Setup
git clone <repo>
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r api/requirements.txt

# Run API
cd api && uvicorn main:app --reload --port 8000

# Run UI (Next.js)
cd ui && npm install && npm run dev

# Run UI (Streamlit fallback)
pip install streamlit && streamlit run ui/app.py

# Run tests
pytest tests/ -v

# Lint
ruff check api/ core/

# Retrain model
python core/train.py --data Data/train.csv --output models/model.joblib

# Build Docker (bonus)
docker-compose up --build
```

---

## 10) Risk Register

| # | Risk | Symptom | Impact | Mitigation | Quick Test |
|---|------|---------|--------|------------|------------|
| 1 | **Schema mismatch (API vs model)** | 422 or KeyError on predict | API returns 500, demo fails | Pydantic schema mirrors exact training columns; unit test with known input | `curl -X POST localhost:8000/predict -d @test_payload.json` |
| 2 | **Feature engineering parity** | Different features at train vs serve time | Silent wrong predictions | Extract `all_features()` into shared `core/features.py` used by both `train.py` and `api/inference.py` | Compare `feature_list` from artifact vs API output columns |
| 3 | **Model file path** | `FileNotFoundError` on startup | API won't start | Use `os.environ.get("MODEL_PATH", "models/model.joblib")` with fallback | `python -c "import joblib; joblib.load('models/model.joblib')"` |
| 4 | **Missing columns in input** | `NaN` propagation → wrong predictions | Wrong class predicted silently | Pydantic defaults for all fields; fill missing with artifact defaults | Test with minimal payload (only required fields) |
| 5 | **CORS blocking frontend** | Browser console `Access-Control-Allow-Origin` error | UI can't reach API | Add `CORSMiddleware(allow_origins=["*"])` in FastAPI | Open browser console, check network tab |
| 6 | **Dependency version conflict** | Import error or wrong behavior | API crashes on start | Pin versions in `api/requirements.txt`; test in clean venv | `pip install -r requirements.txt && python -c "import fastapi, lightgbm, joblib"` |
| 7 | **LightGBM version mismatch** | Model fails to load | API crashes | Save model as text string (`.model_to_string()`) or ensure same LightGBM version | `python -c "import lightgbm; print(lightgbm.__version__)"` |
| 8 | **Next.js build failure** | `npm run build` fails | No UI for demo | Have Streamlit fallback ready (`ui/app.py`); test build early | `cd ui && npm run build` |
| 9 | **Demo breakage under pressure** | Any component fails during live demo | Lost presentation points | Pre-record 30s video of working demo as backup; test full flow 15min before pitch | Run full demo script once before presentation |
| 10 | **Git push with large files** | Push rejected or slow | Can't submit repo | `.gitignore` Data/, verify `du -sh` before push | `git status`, check repo size < 10 MB without data |

---

## Appendix: Critical File List (Must Create)

| File | Owner | Priority | Est. Time |
|------|-------|----------|-----------|
| `api/main.py` | P1 | MUST | 30 min |
| `api/models.py` | P1 | MUST | 15 min |
| `api/inference.py` | P1 | MUST | 20 min |
| `api/requirements.txt` | P1 | MUST | 5 min |
| `ui/app/page.tsx` (or `ui/app.py` for Streamlit) | P2 | MUST | 45 min |
| `ui/components/prediction-form.tsx` | P2 | MUST | 30 min |
| `ui/components/result-card.tsx` | P2 | MUST | 20 min |
| `core/features.py` | P4 | MUST | 15 min |
| `docs/report.md` | P3 | MUST | 90 min |
| `README.md` | P4 | MUST | 15 min |
| `.gitignore` | P4 | MUST | 5 min |
| `.github/workflows/ci.yml` | P4 | BONUS | 15 min |
| `core/train.py` | P4 | BONUS | 20 min |
| `api/Dockerfile` | P4 | BONUS | 10 min |
| `docker-compose.yml` | P4 | BONUS | 10 min |
| `tests/test_api.py` | P4 | BONUS | 15 min |
| `docs/architecture.png` | P3 | MUST | 20 min |

---

## Appendix: `api/requirements.txt` Content

```
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.3
joblib==1.3.2
numpy==1.26.4
pandas==2.1.4
lightgbm==4.6.0
scikit-learn==1.3.2
python-multipart==0.0.6
```

---

**END OF PLAN — Total estimated effort: ~12 person-hours across 4 people in 3 clock-hours.**
