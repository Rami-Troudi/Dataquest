# MLOps Integration Guide

Complete MLOps layer for your Insurance Model with easy model and code switching.

## Overview

This MLOps setup uses:
- **Existing `model.pkl`** - No need to retrain
- **v4_rf_only.py preprocessing** - Same feature engineering pipeline
- **test.csv format** - Consistent data format for predictions
- **Easy model switching** - Load different models without code changes

## Quick Start

### 1. Run Integration Tests

Verify everything is working:

```bash
cd /path/to/Dataquest
python mlops/integration_test.py
```

This will test:
- ✓ Model loading from model.pkl
- ✓ Single predictions
- ✓ Batch predictions  
- ✓ Feature extraction

### 2. Start the API

```bash
python -m uvicorn mlops.app_v3:app --reload --host 0.0.0.0 --port 8000
```

Visit: http://localhost:8000/docs (Swagger UI)

### 3. Make Predictions

#### Single Prediction (via API)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"User_ID":"TEST_001","Policy_Cancelled_Post_Purchase":0, ...}'
```

#### Batch from CSV File

```bash
curl -X POST "http://localhost:8000/predict_csv" \
  -F "file=@test.csv"
```

#### Using CLI

```bash
python mlops/manage_models_v2.py predict --input test.csv --output predictions.csv
```

## API Endpoints

### Health Check
```
GET /health
```
Returns model status and readiness

### Model Information
```
GET /model/info
```
Returns current model details and tuning config

### Load Model
```
POST /model/load?model_path=model.pkl&model_name=insurance_rf&model_version=1.0.0
```
Load a new model and make it active

### Single Prediction
```
POST /predict
Content-Type: application/json

{
  "User_ID": "USR_001",
  "Policy_Cancelled_Post_Purchase": 0,
  "Policy_Start_Year": 2015,
  "Policy_Start_Week": 43,
  ...all fields from test.csv format...
}
```

### Batch Predictions
```
POST /predict_batch
{
  "predictions": [
    {...record1...},
    {...record2...}
  ]
}
```

### CSV Upload Predictions
```
POST /predict_csv
(upload test.csv file)
```

### Required Features
```
GET /features
```
Get list of features needed for predictions

## CLI Usage

### Load a Model

```bash
python mlops/manage_models_v2.py load \
  --path model.pkl \
  --name insurance_rf \
  --version 1.0.0
```

### Show Model Info

```bash
python mlops/manage_models_v2.py info
```

Output:
```
📋 Current Model: insurance_rf:1.0.0
Type:          rf_sklearn
Features:      48
Classes:       [0, 1]

📈 Tuning Configuration:
   best_validation_macro_f1: 0.8500
   best_validation_model_size_mb: 28.50
   ...
```

### List Required Features

```bash
python mlops/manage_models_v2.py features
```

### Make Predictions from CSV

```bash
python mlops/manage_models_v2.py predict \
  --input test.csv \
  --output predictions.csv
```

## Architecture

### Component: `model_loader.py`

**Purpose**: Load and manage models

**Key Classes**:
- `ModelLoader` - Load pkl files into memory
- `RFPreprocessor` - Apply v4_rf_only preprocessing
- `UnifiedModelServer` - Easy model switching and predictions

**Example**:
```python
from mlops.model_loader import get_model_server

server = get_model_server()
server.load_model("model.pkl", "insurance_rf", "1.0.0")

# Predict
predictions = server.predict(df)
result = server.predict_with_confidence(df)
```

### Component: `app_v3.py`

**Purpose**: FastAPI REST service

**Key Endpoints**:
- `/health` - Health check
- `/predict` - Single prediction
- `/predict_batch` - Batch predictions
- `/predict_csv` - CSV file upload
- `/model/info` - Model information
- `/model/load` - Load new model
- `/features` - Get required features

### Component: `integration_test.py`

**Purpose**: Verify MLOps integration

Tests:
1. Model loading
2. Single predictions
3. Batch predictions
4. Feature extraction

### Component: `manage_models_v2.py`

**Purpose**: CLI for model management

Commands:
- `load` - Load model from file
- `info` - Show model details
- `features` - List required features
- `predict` - Make predictions on CSV

## Easy Model Switching

### Scenario: You have multiple models

```
models/
├── model.pkl (current v4_rf_only)
├── model_v2.pkl (new XGBoost)
└── model_lgb.pkl (LightGBM alternative)
```

### Switch models without code changes:

#### Option 1: CLI
```bash
python mlops/manage_models_v2.py load --path models/model_v2.pkl --name insurance_xgb
```

#### Option 2: API
```bash
curl -X POST "http://localhost:8000/model/load?model_path=models/model_v2.pkl&model_name=insurance_xgb"
```

#### Option 3: Python
```python
server = get_model_server()
server.load_model("models/model_v2.pkl", "insurance_xgb", "1.0.0")

# All predictions now use insurance_xgb
predictions = server.predict(df)
```

## Data Format

Input data must be in **test.csv format**:

Required columns:
```
User_ID,
Policy_Cancelled_Post_Purchase,
Policy_Start_Year,
Policy_Start_Week,
Policy_Start_Day,
Grace_Period_Extensions,
Previous_Policy_Duration_Months,
Adult_Dependents,
Child_Dependents,
Infant_Dependents,
Region_Code,
Existing_Policyholder,
Previous_Claims_Filed,
Years_Without_Claims,
Policy_Amendments_Count,
Broker_ID,
Employer_ID,
Underwriting_Processing_Days,
Vehicles_on_Policy,
Custom_Riders_Requested,
Broker_Agency_Type,
Deductible_Tier,
Acquisition_Channel,
Payment_Schedule,
Employment_Status,
Estimated_Annual_Income,
Days_Since_Quote,
Policy_Start_Month
```

## Preprocessing Pipeline

The MLOps layer uses v4_rf_only.py preprocessing:

1. **Load model bundle** from pickle file
2. **Extract preprocessor** configuration (categorical mappings, medians, etc)
3. **Apply categorical encoding** using stored mappings
4. **Fill missing values** with stored numeric medians
5. **Select required features** in correct order
6. **Pass to RandomForest model** for prediction

All preprocessing is **consistent** with training pipeline.

## Adding New Models

### Step 1: Train your model

```python
# Use v4_rf_only.py or train new model
# Save with same bundle format:

bundle = {
    'rf_model': trained_model,
    'preprocessor': {
        'categorical_columns': [...],
        'cat_mappings': {...},
        'numeric_medians': {...},
        'feature_columns': [...]
    },
    'class_order': [...],
    'model_type': 'rf_sklearn'
}

joblib.dump(bundle, 'models/model_v2.pkl')
```

### Step 2: Load in MLOps

```bash
python mlops/manage_models_v2.py load \
  --path models/model_v2.pkl \
  --name insurance_rf_v2 \
  --version 2.0.0
```

### Step 3: Make predictions

```bash
python mlops/manage_models_v2.py predict \
  --input test.csv \
  --output predictions_v2.csv
```

## Troubleshooting

### Problem: "No model loaded"
```
❌ Model server not initialized

Solution:
1. Ensure model.pkl exists
2. Run integration test: python mlops/integration_test.py
3. Check model.pkl path is correct
```

### Problem: "Feature not found"
```
❌ Prediction failed: 'some_feature' not in index

Solution:
1. Check input CSV has all required columns
2. List required features:
   python mlops/manage_models_v2.py features
3. Compare with your CSV columns
```

### Problem: "Model file corrupted"
```
❌ Failed to load model: EOFError

Solution:
1. Verify model.pkl is not corrupted
2. Try: python -c "import joblib; joblib.load('model.pkl')"
3. Retrain if necessary
```

## Performance

Typical timings:

- **Model loading**: ~2-5 seconds
- **Single prediction**: ~10-50 ms
- **Batch prediction (1000 records)**: ~500-1000 ms
- **Memory usage**: ~200-500 MB

## Files Overview

```
mlops/
├── app_v3.py                  # FastAPI REST server
├── model_loader.py            # Model loading & preprocessing
├── manage_models_v2.py        # CLI for model management
├── integration_test.py        # Integration tests
├── models/
│   └── registry.json          # Model registry
├── ssl/                       # SSL certificates (for HTTPS)
├── requirements-docker.txt    # Dependencies
└── README.md                  # This file
```

## Next Steps

1. ✅ Run `python mlops/integration_test.py` to verify setup
2. ✅ Start API: `python -m uvicorn mlops.app_v3:app --reload`
3. ✅ Try predictions via `/docs` UI or CLI
4. ✅ Add new models when ready
5. ✅ Deploy with Docker for production

## Production Deployment

For production, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

```bash
# Docker deployment
docker-compose -f docker-compose.prod.yml up -d

# API will be available at:
# http://localhost/ (app_v3)
# Metrics at /metrics
```

---

**Version**: 3.0.0  
**Last Updated**: 2026-02-22  
**Uses**: model.pkl + v4_rf_only preprocessing
