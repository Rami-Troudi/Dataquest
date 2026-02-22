# DataQuest Phase II Fullstack

This repository contains a full Phase II implementation for the DataQuest Insurance Recommender:
- FastAPI backend (`api/`)
- Streamlit Broker Assistant UI (`ui/app.py`)
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
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

Frontend URL:
- `http://localhost:8501`

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
- UI includes presets, JSON copy buttons, and a 5-item session history.






# Intelligent Insurance Bundle Recommendation

## Overview

This project implements a lightweight, explainable, and efficient insurance bundle recommendation system for the DataQuest Hackathon.

The objective is to predict the `Purchased_Coverage_Bundle` (10 classes) using customer profile and behavioral features while optimizing Macro F1 under strict size and latency constraints.

---

## Architecture

The system is structured into three layers:

1. Feature Engineering
2. Prediction Model (Random Forest)
3. Business Optimization Layer

---

## Feature Engineering

Engineered features include:

- Total dependents
- Income per dependent
- Claims ratio
- Policy duration in years
- Aggregated financial indicators

Data augmentation using Gaussian noise was applied during training to improve generalization.

---

## Model

- Algorithm: RandomForestClassifier
- Reason:
  - Strong Macro F1 performance
  - Low inference latency
  - Small model size
  - CPU-friendly
  - Robust to tabular feature interactions

---

## Explainability

Explainability is provided using SHAP:

- Global feature importance
- Local prediction explanations
- Visual interpretability via web interface

---

## Deployment

- Local prediction model (secure and replaceable)
- API service for inference
- Optional cloud optimization layer
- Fully offline compatible

---

## Future Extension

The architecture supports economic optimization:

Expected Gain = P(accept) × Margin

Future extension:

Score = P × Margin + α × RetentionScore

This enables long-term customer value optimization.

---

## Constraints Compliance

- < 50 MB model size
- CPU-only inference
- < 120 seconds execution
- Macro F1 optimized
- No internet dependency