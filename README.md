# DataQuest Insurance Recommender (Phase II)

Full-stack inference product for insurance bundle recommendation:

- FastAPI backend (`api/`)
- Next.js frontend (`ui/`)
- Shared inference core (`core/`)
- Training script (`scripts/train.py`)
- Explainability module (`core/explainability.py`)

## 1) Prerequisites

- Python 3.10+
- Node.js 20+

## 2) Run Backend

```bash
pip install -r api/requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## 3) Run Frontend (Next.js only)

```bash
cd ui
npm install
npm run dev
```

Frontend URL:

- `http://127.0.0.1:3000`

API base URL is read from:

- `NEXT_PUBLIC_API_BASE_URL` (optional)
- fallback: `http://127.0.0.1:8000`

## 4) Run Tests

```bash
pytest -q tests
```

## 5) Train/Rebuild Model Artifact

```bash
set PYTHONPATH=.
python scripts/train.py --config scripts/config.yaml
```

Outputs `model.pkl` with required keys:

- `rf_model`
- `preprocessor`
- `class_order`

## 6) API Endpoints

- `GET /health`
- `POST /predict`
- `POST /predict-batch`
- `POST /whatif`
- `GET /schema`
- `GET /feature-importance`
- `GET /model/feature_importance`
- `POST /explain`
- `POST /explain_csv` (JSON batch)
- `POST /explain_csv_upload` (multipart CSV upload)
- `GET /metadata`
- `GET /metrics`

## 7) Explainability

Implemented in `core/explainability.py`:

- Global feature importance
- Local reason codes

Detailed documentation:

- `EXPLAINABILITY_FEATURES.md`
