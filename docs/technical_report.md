# Technical Report — DataQuest Insurance Recommender (Phase II)

## 1. Problem and Objective
- Task: multi-class classification (10 classes) for `Purchased_Coverage_Bundle`.
- Objective: provide production-style inference through API + UI while preserving model parity.

## 2. Data Overview
- Training set: 60,868 rows.
- Test set: 15,218 rows.
- Input schema: 28 feature columns + `User_ID`.

## 3. Feature Engineering
Feature engineering is applied in `core/inference_core.py` and mirrors the Phase I model logic.
Main engineered families:
- Household composition: `Total_Dependents`, ratios.
- Risk and claim behavior: `Claims_Frequency`, `Claims_to_YearsWithout_Ratio`.
- Policy/process behavior: `Total_Processing_Days`, `Processing_Efficiency`, `Amendments_per_Day`.
- Temporal cyclic encoding: month/day/week sin-cos transforms.
- Composite behavior signals: `High_Risk_Customer`, `Loyalty_Score`, `Payment_Flexibility`.

## 4. Model Choice Justification (RandomForest)
- Chosen model: RandomForest (`rf_model` in `model.pkl`).
- Rationale:
  - Good tabular robustness with mixed nonlinear patterns.
  - Stable inference behavior for API productization.
  - Native feature importance support for explainability extensions.

## 5. Architecture
- Diagram file: `docs/architecture_diagram.svg`.
- Runtime flow:
  - UI -> FastAPI `/predict` -> validation -> feature engineering -> RF `predict_proba` -> top-k response.

## 6. API and Web Application
- API endpoints:
  - `GET /health`
  - `POST /predict`
  - `GET /schema`
  - `GET /metrics`
- UI:
  - Next.js form submission for full record inference.
  - Returns predicted bundle + top-3 probabilities.

## 7. Experiments and Lessons Learned
- Initial iterations emphasized offline score optimization and packaging constraints.
- Productization phase focused on strict inference parity and deterministic API behavior.
- Eliminated fallback inference paths to maintain predictable runtime behavior.

## 8. Limitations and Next Steps
- Add authenticated API access for production environments.
- Add model/version registry and artifact lineage tracking.
- Add deployment automation (Render/Vercel) and monitoring dashboards.
