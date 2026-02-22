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
