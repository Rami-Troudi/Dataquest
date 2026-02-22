# Explainability Features

## Status
Explainability is fully implemented and integrated into the API and UI.

## Implemented Components

### Core module
- `core/explainability.py`
  - `build_global_feature_importance(rf_model, feature_columns)`
  - `build_local_reason_codes(X_encoded, user_ids, rf_model, preprocessor, top_k=3)`

### API integration
- `GET /model/feature_importance`
  - Global ranked feature importance for deployed model
- `POST /explain`
  - Local explanation for one customer record
- `POST /explain_csv`
  - Local explanations for JSON batch (`records: []`)
- `POST /explain_csv_upload`
  - Local explanations for uploaded CSV (`multipart/form-data`)

## Endpoint Contracts

### `GET /model/feature_importance`
Response:
```json
{
  "model": "rf_only",
  "total_features": 49,
  "features": [
    {
      "rank": 1,
      "feature": "Policy_Start_Year",
      "importance": 0.183,
      "importance_pct": 18.3
    }
  ]
}
```

### `POST /explain`
Request:
```json
{
  "record": {
    "User_ID": "USR_060868",
    "Policy_Cancelled_Post_Purchase": 0
  },
  "top_k_reasons": 3
}
```

Response:
```json
{
  "User_ID": "USR_060868",
  "prediction": 2,
  "confidence": 0.81,
  "class_probabilities": [0.01, 0.02, 0.81],
  "reason_codes": [
    {"rank": 1, "feature": "Policy_Start_Year", "value": "2015", "contribution": "high"},
    {"rank": 2, "feature": "Estimated_Annual_Income", "value": "24493.8500", "contribution": "medium"},
    {"rank": 3, "feature": "Broker_Agency_Type", "value": "Urban_Boutique", "contribution": "medium"}
  ]
}
```

### `POST /explain_csv` (JSON batch)
Request:
```json
{
  "records": [
    {"User_ID": "USR_001", "Policy_Cancelled_Post_Purchase": 0}
  ],
  "top_k_reasons": 3
}
```

Response:
```json
{
  "count": 1,
  "predictions": [
    {
      "User_ID": "USR_001",
      "prediction": 2,
      "confidence": 0.81,
      "reason_1": "Policy_Start_Year=2015",
      "reason_2": "Estimated_Annual_Income=24493.8500",
      "reason_3": "Broker_Agency_Type=Urban_Boutique"
    }
  ]
}
```

### `POST /explain_csv_upload` (CSV upload)
Form data:
- `file`: CSV file with the model input columns
- `top_k_reasons`: integer (optional, default `3`)

Response shape is the same as `POST /explain_csv`.

## UI Coverage
Next.js UI (`ui/src/app/page.tsx`) exposes:
- Global feature-importance visualization
- Local reason-code generation for current profile
- What-if scenario analysis
- Batch prediction view

## Technical Notes
- Local explanations are deviation-based (baseline vs current encoded values) weighted by RF feature importance.
- Categorical values are decoded back to human-readable labels using preprocessor mappings.
- If RF importances are unavailable/invalid, safe uniform fallback is used.

## Validation
Recommended checks:
```bash
pytest -q tests/test_api.py
```

And manual endpoint checks:
- `/model/feature_importance`
- `/explain`
- `/explain_csv`
- `/explain_csv_upload`
