# Model Management & Easy Switching - Implementation Guide

## Overview

A comprehensive system for **easy model switching** and **flexible feature management** in your MLOps pipeline.

### Key Components

1. **Model Registry** - Centralized model management
2. **Feature Manager** - Unified feature configuration
3. **Enhanced API** - Updated REST endpoints
4. **CLI Tool** - Command-line model management
5. **Setup Helper** - Quick initialization

---

## 📚 Architecture

```
┌─────────────────────────────────────────────────────┐
│         Your Application / API                       │
├─────────────────────────────────────────────────────┤
│                   Enhanced App (app_v2.py)          │
│  - /models                                          │
│  - /models/switch (change active model)             │
│  - /predict (uses current model)                    │
│  - /features (list all features)                    │
├─────────────────────────────────────────────────────┤
│  ┌──────────────────┐    ┌──────────────────┐      │
│  │  Model Registry  │    │ Feature Manager  │      │
│  │  - Load models   │    │ - Validate data  │      │
│  │  - Switch models │    │ - preprocess     │      │
│  │  - Track versions│    │ - configuration  │      │
│  └──────────────────┘    └──────────────────┘      │
├─────────────────────────────────────────────────────┤
│              Model Files Storage                    │
│  models/                                            │
│  ├── insurance_lgb.joblib                          │
│  ├── insurance_xgb.joblib                          │
│  └── registry.json (model metadata)                │
│  features_config.json (feature configuration)      │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Setup

### 1. Initialize Features & Models

```bash
cd mlops
python quickstart.py
```

This will:
- Create and configure features
- Register your existing model
- Display all models and features

### 2. View All Models

```bash
python manage_models.py list --verbose
```

Output:
```
📚 Available Models (1 total)
────────────────────────────────────────────────────────────────────────────────────────
→ insurance_lgb:1.0.0  | Status: active     | Features: 42
  Type: lightgbm
  Accuracy: 0.85
  Description: LightGBM insurance policy prediction model

Current model: insurance_lgb:1.0.0
```

### 3. Register New Model

```bash
python manage_models.py register \
  --name insurance_xgb \
  --version 1.0.0 \
  --type xgboost \
  --file models/insurance_xgb.joblib \
  --description "XGBoost insurance model" \
  --accuracy 0.87
```

### 4. Switch Models

```bash
# Switch to different model
python manage_models.py switch --name insurance_xgb --version 1.0.0

# Verify
python manage_models.py list
```

### 5. Start Enhanced API

```bash
# Requires model to be set
python -m uvicorn app_v2:app --reload --host 0.0.0.0 --port 8000
```

---

## 🎯 Features

### Model Registry

**Register a model:**
```python
from model_registry import get_registry

registry = get_registry()

registry.register_model(
    name="insurance_lgb",
    version="2.0.0",
    model_type="lightgbm",
    file_path="models/insurance_lgb_v2.joblib",
    features=["feature1", "feature2", ...],
    accuracy=0.88,
    description="Improved LightGBM model"
)
```

**Load and use a model:**
```python
registry.set_current_model("insurance_lgb", "2.0.0")
model = registry.get_current_model()

# Make predictions
predictions = model.predict(X)
```

**List all models:**
```python
models = registry.list_models()
for m in models:
    print(f"{m['key']} - {m['status']}: {m['accuracy']}")
```

### Feature Manager

**Configure features:**
```python
from feature_manager import get_feature_manager, FeatureConfig

fm = get_feature_manager()

# Add individual feature
fm.add_feature(
    name="Adult_Dependents",
    feature_type="numeric",
    group="household",
    required=True,
    validation_rules={"min": 0, "max": 10},
    description="Number of adult dependents"
)

# Save to file
fm.save_config()
```

**Validate data:**
```python
import pandas as pd

df = pd.read_csv("data.csv")

is_valid, errors = fm.validate_data(df)
if is_valid:
    print("✓ Data is valid")
else:
    for error in errors:
        print(f"✗ {error}")
```

**Preprocess data:**
```python
# Automatic preprocessing
df_clean = fm.preprocess_data(df)

# Custom transformations
custom_transforms = {
    "feature1": lambda x: np.log1p(x)
}
df_clean = fm.preprocess_data(df, transformations=custom_transforms)
```

### Enhanced API (app_v2.py)

**List available models:**
```bash
curl http://localhost:8000/models
```

Response:
```json
{
  "total": 2,
  "current": "insurance_lgb:1.0.0",
  "models": [
    {
      "key": "insurance_lgb:1.0.0",
      "name": "insurance_lgb",
      "version": "1.0.0",
      "type": "lightgbm",
      "status": "active",
      "accuracy": 0.85,
      "features_count": 42
    }
  ]
}
```

**Switch models:**
```bash
curl -X POST "http://localhost:8000/models/switch?model_name=insurance_xgb&model_version=1.0.0"
```

**Make prediction (flexible input):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "User_ID": "USER_001",
    "features": {
      "Adult_Dependents": 1,
      "Child_Dependents": 2,
      ...
    },
    "model_name": "insurance_lgb",
    "model_version": "1.0.0"
  }'
```

**List features:**
```bash
curl http://localhost:8000/features
```

---

## 💻 Command-Line Interface

### Model Management

```bash
# Register a model
python manage_models.py register \
  --name model_name \
  --version 1.0.0 \
  --type lightgbm \
  --file path/to/model.joblib \
  --accuracy 0.85

# Switch to a model
python manage_models.py switch --name model_name

# List all models
python manage_models.py list --verbose

# Get model details
python manage_models.py info --name model_name

# Delete a model
python manage_models.py delete --name model_name --version 1.0.0

# Compare two models
python manage_models.py compare --model1 model_a --model2 model_b
```

### Feature Management

```bash
# Add a feature
python manage_models.py add-feature \
  --name Adult_Dependents \
  --type numeric \
  --group household \
  --description "Number of adult dependents"

# List all features
python manage_models.py list-features

# Validate data against schema
python manage_models.py validate --file data.csv
```

---

## 📋 Configuration Files

### Registry File (models/registry.json)

```json
{
  "insurance_lgb:1.0.0": {
    "name": "insurance_lgb",
    "version": "1.0.0",
    "model_type": "lightgbm",
    "file_path": "model.joblib",
    "features": ["feature1", "feature2", ...],
    "description": "LightGBM model",
    "accuracy": 0.85,
    "status": "active"
  },
  "insurance_xgb:1.0.0": {
    "name": "insurance_xgb",
    "version": "1.0.0",
    "model_type": "xgboost",
    "file_path": "models/insurance_xgb.joblib",
    "features": ["feature1", "feature2", ...],
    "description": "XGBoost model",
    "accuracy": 0.87,
    "status": "active"
  }
}
```

### Features Config (features_config.json)

```json
{
  "household": {
    "description": "Household composition features",
    "features": [
      {
        "name": "Adult_Dependents",
        "type": "numeric",
        "required": true,
        "description": "Number of adult dependents",
        "validation_rules": {"min": 0, "max": 10},
        "transformation": null
      }
    ]
  },
  "claims": {
    "description": "Claims history features",
    "features": [...]
  }
}
```

---

## 🔄 Workflow Examples

### Example 1: Train New Model & Register

```bash
# After training a new XGBoost model
python manage_models.py register \
  --name insurance_xgb \
  --version 2.0.0 \
  --type xgboost \
  --file models/insurance_xgb_v2.joblib \
  --accuracy 0.88

# Compare with current model
python manage_models.py compare --model1 insurance_lgb --model2 insurance_xgb

# If better, switch to new model
python manage_models.py switch --name insurance_xgb --version 2.0.0
```

### Example 2: A/B Testing Different Models

```python
from model_registry import get_registry

registry = get_registry()

# Make 50% of predictions with model A, 50% with model B
import random

def get_active_model():
    choice = random.random()
    if choice < 0.5:
        return registry.load_model("insurance_lgb", "1.0.0")
    else:
        return registry.load_model("insurance_xgb", "2.0.0")

model = get_active_model()
predictions = model.predict(X)
```

### Example 3: Deprecate Old Models

```bash
# Mark old model as deprecated
python manage_models.py info --name insurance_lgb --version 1.0.0

# To actually deprecate, edit registry.json and set status to "deprecated"

# List only active models
python manage_models.py list --status active
```

---

## 📊 Performance Monitoring

Track model performance over time:

```python
from model_registry import get_registry
from datetime import datetime

registry = get_registry()

# After getting predictions
predictions = model.predict(X)
actual = y_true

# Calculate metrics
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(actual, predictions.round())

# Update model metadata (would need to modify registry)
# This allows tracking performance over time
```

---

## 🔒 Best Practices

### 1. Version Your Models
- Always use semantic versioning (1.0.0, 1.1.0, 2.0.0)
- Keep old versions for rollback capability

### 2. Document Features
```bash
python manage_models.py add-feature \
  --name My_Feature \
  --type numeric \
  --group my_group \
  --description "Clear description of what this feature represents"
```

### 3. Validate Data Before Prediction
```python
fm = get_feature_manager()
is_valid, errors = fm.validate_data(df)
if not is_valid:
    for error in errors:
        logger.warning(f"Data validation warning: {error}")
```

### 4. Test Model Switching
```bash
# Ensure both models work before production switch
python manage_models.py info --name model_a
python manage_models.py info --name model_b

# Then switch
python manage_models.py switch --name model_b
```

### 5. Regular Backups
```bash
# Backup registry and configs
cp models/registry.json models/registry.json.backup
cp features_config.json features_config.json.backup
```

---

## 🚨 Troubleshooting

### Model Not Loading
```bash
# Check if file exists
python manage_models.py info --name model_name

# Verify file path in registry
cat models/registry.json
```

### Feature Validation Failures
```bash
# Check feature schema
python manage_models.py list-features --verbose

# Validate your data
python manage_models.py validate --file your_data.csv
```

### API Errors
```bash
# Check current model
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/models

# Switch to valid model
curl -X POST "http://localhost:8000/models/switch?model_name=insurance_lgb"
```

---

## 📖 Next Steps

1. **Run quickstart**: `python quickstart.py`
2. **Try CLI**: `python manage_models.py list`
3. **Start API**: `python -m uvicorn app_v2:app --reload`
4. **Make predictions**: See API docs at `http://localhost:8000/docs`
5. **Add more models**: Register additional trained models
6. **Create features**: Add all your features to configuration

---

## 🎓 Advanced Usage

### Custom Preprocessing
```python
fm = get_feature_manager()

def custom_log_transform(x):
    return np.log1p(x.clip(lower=0))

fm.register_transformation("custom_log", custom_log_transform)

fm.add_feature(
    "My_Feature",
    "numeric",
    transformation="custom_log"
)
```

### Model Ensembles
```python
registry = get_registry()

model1 = registry.load_model("insurance_lgb")
model2 = registry.load_model("insurance_xgb")

# Average predictions
pred1 = model1.predict(X)
pred2 = model2.predict(X)
ensemble_pred = (pred1 + pred2) / 2
```

### Dynamic Model Selection
```python
# Load model based on user type or region
user_region = data.get("region")

if user_region == "NA":
    model = registry.load_model("insurance_lgb_na")
elif user_region == "EU":
    model = registry.load_model("insurance_lgb_eu")
else:
    model = registry.load_model("insurance_lgb")

predictions = model.predict(X)
```

---

**Version**: 2.0.0  
**Status**: Production Ready ✅
