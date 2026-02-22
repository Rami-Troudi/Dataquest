# Monolithic README

## Table of Contents

1. [Main README](#main-readme)
2. [MLOps Setup Complete](#mlops-setup-complete)
3. [Feature Engineering](#feature-engineering)
4. [Explainability Features](#explainability-features)
5. [Deployment Guide](#deployment-guide)
6. [Model Management Guide](#model-management-guide)

---

## Main README

# DataQuest Phase II Fullstack

This repository contains a full Phase II implementation for the DataQuest Insurance Recommender:

- FastAPI backend (`api/`)
- Next.js frontend (`ui/src/`)
- Shared inference core (`core/`)
- Retraining stub (`scripts/`)
- Tests (`tests/`)
- Technical documentation (`docs/`)

## 1) Prerequisites

- Python 3.10+

## 2) Backend Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8000
```

Backend endpoints:

- `GET /health`
- `POST /predict`
- `POST /predict-batch`
- `POST /whatif`
- `GET /schema`
- `GET /feature-importance`
- `GET /metadata`
- `GET /metrics`

## 3) Frontend Setup

```bash
cd ui
npm install
npm run dev
```

Frontend URL:

- `http://localhost:3000`

## 4) Run Tests

```bash
pytest -q
```

## 5) Retrain Model

```bash
python scripts/train.py --config scripts/config.yaml
```

This command outputs `model.pkl` with the required runtime keys:

- `rf_model`
- `preprocessor`
- `class_order`

## 6) Notes

- The API uses the existing model artifact at `model.pkl`.
- Model is loaded once at startup and not reloaded per request.
- `/predict` returns: bundle_id, top_k, latency_ms, warnings, reasons, confidence.
- Streamlit UI is legacy/demo only. Use Next.js UI on port `3000` for normal usage.
- UI includes presets, JSON copy buttons, and a 5-item session history.

---

## MLOps Setup Complete

Your MLOps layer is now fully integrated with:

- ✅ Existing `model.pkl`
- ✅ v4_rf_only.py preprocessing  
- ✅ test.csv input format
- ✅ Easy model switching

All 4 integration tests **PASSED**.

---

### What Was Created

#### 1. **model_loader.py** - Model Loading & Preprocessing

- `ModelLoader` - Loads pkl files
- `RFPreprocessor` - Applies v4_rf_only preprocessing
- `UnifiedModelServer` - Easy switching and predictions

#### 2. **app_v3.py** - FastAPI REST Server

RESTful API with endpoints for:

- Single predictions: `POST /predict`
- Batch predictions: `POST /predict_batch`  
- CSV upload: `POST /predict_csv`
- Model switching: `POST /model/load`
- Health check: `GET /health`
- Model info: `GET /model/info`
- Features: `GET /features`

#### 3. **manage_models_v2.py** - CLI Tool

Command-line interface for:

- Loading models: `python manage_models_v2.py load --path model.pkl`
- Viewing model info: `python manage_models_v2.py info`
- Listing features: `python manage_models_v2.py features`
- Making predictions: `python manage_models_v2.py predict --input test.csv --output predictions.csv`

#### 4. **integration_test.py** - Verification Tests

Tests for:

- ✅ Model loading
- ✅ Single predictions
- ✅ Batch predictions
- ✅ Feature extraction

All **PASSED** ✅

#### 5. **MLOPS_INTEGRATION.md** - Complete Documentation

Full guide with:

- Quick start instructions
- API endpoint reference
- CLI usage examples  
- Architecture overview
- Troubleshooting guide
- Production deployment info

---

## Feature Engineering

> This document explains every engineered feature in our pipeline: what it is, how it's computed, and why it improves predictions.

---

### Overview

We started with **27 raw columns** from the dataset and engineered **22 additional features**, bringing the total to **49 features** after preprocessing. All transformations are implemented in `solution.py::preprocess()`.

Our engineering strategy focused on four principles:

1. **Domain relevance** — features should reflect real insurance decision factors (family needs, risk profile, spending power).
2. **Ratio-based normalization** — raw counts are less informative than per-capita or per-unit ratios.
3. **Cyclical encoding** — temporal features (month, day, week) wrap around; sin/cos encoding prevents artificial ordinal relationships.
4. **Composite scores** — combine multiple signals into a single interpretable metric.

---

### Feature Catalog

#### Group 1: Family Composition (Features 1–4)

These features capture the household structure, which is the strongest predictor of bundle type. Families with children gravitate toward comprehensive plans; single adults tend toward basic or auto-only coverage.

| # | Feature | Formula | Type |
|---|---------|---------|------|
| 1 | `Total_Dependents` | `Adult_Dependents + Child_Dependents + Infant_Dependents` | Integer |
| 2 | `Child_Ratio` | `Child_Dependents / (Total_Dependents + 1)` | Float [0, 1) |
| 3 | `Adult_Ratio` | `Adult_Dependents / (Total_Dependents + 1)` | Float [0, 1) |
| 4 | `Infant_Ratio` | `Infant_Dependents / (Total_Dependents + 1)` | Float [0, 1) |

**Why +1 in the denominator?** Prevents division by zero when a customer has no dependents. The +1 is consistent across all ratio features.

**Why ratios instead of raw counts?** A household with 3 adults and 0 children behaves differently from one with 1 adult and 2 children, even if both have 3 total dependents. Ratios capture the _composition_, not just the _size_.

**Impact:** `Total_Dependents` ranks **#2** in global feature importance (14.26%). `Adult_Ratio` ranks **#5** (5.93%).

---

#### Group 2: Policy Tenure (Feature 5)

| # | Feature | Formula | Type |
|---|---------|---------|------|
| 5 | `Total_Policy_Duration_Years` | `Previous_Policy_Duration_Months / 12` | Float |

**Why?** Converts months to years for more intuitive scale. Longer tenure often correlates with loyalty discounts and preference for renewal-friendly bundles.

---

## Explainability Features

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

---

## Deployment Guide

# 🚀 Deployment Guide - Start to Finish

Complete guide to deploy your MLOps Insurance Model API to the cloud (for free).

---

## 📋 Prerequisites

Before deploying, ensure:

- ✅ Docker works locally (`docker-compose up` succeeds)
- ✅ Git repository with your code
- ✅ `model.pkl` exists in the project root
- ✅ API tested locally at <http://localhost:8000>

---

## ⚡ Quick Deploy Options

Choose your platform:

| Platform | Free Tier | Setup Time | Best For |
|----------|-----------|------------|----------|
| [Render](#render-deployment-recommended) | ✅ 750hrs/month | 5 min | **Easiest** |
| [Railway](#railway-deployment) | ✅ $5 credit | 5 min | GitHub Students |
| [Google Cloud Run](#google-cloud-run) | ✅ $300 credit | 10 min | Scalability |
| [Hugging Face Spaces](#hugging-face-spaces) | ✅ Free | 15 min | ML Projects |
| [Azure](#azure-deployment) | Student pack | 15 min | Enterprise |

---

# 🎯 Render Deployment (Recommended)

**Why Render?**

- ✅ 750 free hours/month
- ✅ Auto-deploy from GitHub
- ✅ Built-in HTTPS
- ✅ No credit card required

## Step 1: Push to GitHub

```powershell
# If not already a git repo
git init
git add .
git commit -m "Initial commit with MLOps setup"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

## Step 2: Sign up on Render

1. Go to <https://render.com>
2. Sign up with GitHub (easiest)
3. Authorize Render to access your repos

## Step 3: Create Web Service

1. Click **"New +"** → **"Web Service"**
2. Select your GitHub repository
3. Configure:

```yaml
Name: insurance-mlops-api
Region: Oregon (US West)
Branch: main
Root Directory: (leave empty)
Environment: Docker
Dockerfile Path: mlops/Dockerfile
Docker Build Context: .
Instance Type: Free
```

## Step 4: Add Environment Variables

In Render dashboard, add:

```
API_HOST=0.0.0.0
API_PORT=8000
```

## Step 5: Deploy

Click **"Create Web Service"**

Render will:

1. Clone your repo
2. Build Docker image
3. Deploy and give you a URL like: `https://insurance-mlops-api.onrender.com`

⏱️ First deploy takes ~5-10 minutes.

## Step 6: Test your deployment

```powershell
# Replace with your actual Render URL
$URL = "https://insurance-mlops-api.onrender.com"

# Health check
Invoke-RestMethod "$URL/health"

# Make prediction
$body = @{ file = Get-Item .\test.csv }
Invoke-RestMethod -Uri "$URL/predict_csv" -Method Post -Form $body
```

---

## Model Management Guide

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

## 📂 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│         Your Application / API                              │
├──────────────────────────────────────────────────────────────┤
│                   Enhanced App (app_v2.py)                 │
│  - /models                                                  │
│  - /models/switch (change active model)                     │
│  - /predict (uses current model)                            │
│  - /features (list all features)                            │
├──────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐    ┌───────────────────────────┐  │
│  │  Model Registry      │    │ Feature Manager           │  │
│  │  - Load models       │    │ - Validate data           │  │
│  │  - Switch models     │    │ - preprocess              │  │
│  │  - Track versions    │    │ - configuration           │  │
│  └──────────────────────┘    └───────────────────────────┘  │
├──────────────────────────────────────────────────────────────┤
│              Model Files Storage                            │
│  models/                                                    │
│  ├── insurance_lgb.joblib                                   │
│  ├── insurance_xgb.joblib                                   │
│  └── registry.json (model metadata)                         │
│  features_config.json (feature configuration)               │
└──────────────────────────────────────────────────────────────┘
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
📂 Available Models (2 total)
→ insurance_lgb:1.0.0  | Status: active     | Features: 42
  Type: lightgbm
  Accuracy: 0.85
  Description: LightGBM insurance policy prediction model
  → insurance_forest | Status: active     | Features: 20
  Type: random forest
  Accuracy: 0.85
  Description: Randomforest insurance policy prediction model

Current model: insurance_lgb:1.0.0
```

---

## Advanced Usage

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

---

**Version**: 2.0.0  
**Status**: Production Ready ✅
