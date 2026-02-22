"""
Insurance Bundle Recommender — FastAPI Backend
================================================
Endpoints:
  GET  /health          → system & model health check
  POST /predict         → predict bundle(s) with explainability
  GET  /explain/global  → global feature importance ranking
  GET  /                → serves the frontend UI
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Make sure the project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from solution import preprocess, load_model, _transform_with_preprocessor, _align_proba  # noqa: E402
from explainability import build_global_feature_importance  # noqa: E402
from api.schemas import (  # noqa: E402
    BUNDLE_NAMES,
    CustomerInput,
    ExplainGlobalResponse,
    FeatureImportance,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictionResult,
    ReasonCode,
)

# ---------------------------------------------------------------------------
# Global model cache
# ---------------------------------------------------------------------------
_MODEL_BUNDLE: dict | None = None
_GLOBAL_IMPORTANCE: pd.DataFrame | None = None


def _get_model():
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        os.chdir(str(PROJECT_ROOT))
        _MODEL_BUNDLE = load_model()
    return _MODEL_BUNDLE


def _get_global_importance():
    global _GLOBAL_IMPORTANCE
    if _GLOBAL_IMPORTANCE is None:
        bundle = _get_model()
        rf = bundle.get("rf_model", bundle.get("xgb_model"))
        feature_cols = bundle["preprocessor"]["feature_columns"]
        _GLOBAL_IMPORTANCE = build_global_feature_importance(rf, feature_cols)
    return _GLOBAL_IMPORTANCE


# ---------------------------------------------------------------------------
# Lifespan — warm up the model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_model()
    _get_global_importance()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Insurance Bundle Recommender",
    version="1.0.0",
    description="Predict the most suitable insurance coverage bundle for a customer.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return {"message": "Frontend not found. Visit /docs for the API."}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check API and model readiness."""
    bundle = _get_model()
    rf = bundle.get("rf_model", bundle.get("xgb_model"))
    preprocessor = bundle["preprocessor"]
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_type=bundle.get("model_type", "unknown"),
        num_features=len(preprocessor["feature_columns"]),
        num_classes=len(rf.classes_),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_endpoint(request: PredictRequest):
    """Predict insurance bundle for one or more customers with explainability."""
    if not request.customers:
        raise HTTPException(status_code=422, detail="At least one customer required.")

    bundle = _get_model()
    rf = bundle.get("rf_model", bundle.get("xgb_model"))
    preprocessor = bundle["preprocessor"]
    class_order = np.array(bundle.get("class_order", rf.classes_))

    # Build DataFrame from request
    records = []
    for i, c in enumerate(request.customers):
        d = c.model_dump()
        if not d.get("User_ID"):
            d["User_ID"] = f"USR_{uuid.uuid4().hex[:8].upper()}"
        records.append(d)

    input_df = pd.DataFrame(records)

    # Feature engineering
    engineered = preprocess(input_df)

    # Transform with saved preprocessor
    X = _transform_with_preprocessor(engineered, preprocessor)

    # Predict probabilities
    proba = rf.predict_proba(X)
    proba_aligned = _align_proba(proba, rf.classes_, class_order)
    preds = class_order[np.argmax(proba_aligned, axis=1)]
    confidences = np.max(proba_aligned, axis=1)

    # Compute per-row reason codes (top 3)
    importances = rf.feature_importances_
    feature_cols = preprocessor["feature_columns"]
    medians = preprocessor.get("numeric_medians", {})
    cat_cols = set(preprocessor.get("categorical_columns", []))
    cat_mappings = preprocessor.get("cat_mappings", {})

    baseline = np.array(
        [float(medians.get(f, 0.0)) if f not in cat_cols else 0.0 for f in feature_cols],
        dtype=float,
    )

    X_values = X.to_numpy(dtype=float)
    deviations = np.abs(X_values - baseline)
    local_scores = deviations * (importances + 1e-12)

    results: list[PredictionResult] = []
    for i in range(len(records)):
        top_idx = np.argsort(-local_scores[i])[:3]
        reasons = []
        for rank, idx in enumerate(top_idx, 1):
            feat = feature_cols[idx]
            raw_val = X_values[i, idx]
            # Decode categorical values
            if feat in cat_mappings:
                inv = {int(v): str(k) for k, v in cat_mappings[feat].items()}
                val_str = inv.get(int(round(raw_val)), str(raw_val))
            elif np.isfinite(raw_val):
                val_str = str(int(round(raw_val))) if abs(raw_val - round(raw_val)) < 1e-9 else f"{raw_val:.4f}"
            else:
                val_str = "N/A"
            reasons.append(ReasonCode(rank=rank, feature=feat, value=val_str))

        results.append(
            PredictionResult(
                User_ID=records[i]["User_ID"],
                predicted_bundle_id=int(preds[i]),
                predicted_bundle_name=BUNDLE_NAMES.get(int(preds[i]), "Unknown"),
                confidence=round(float(confidences[i]), 4),
                reason_codes=reasons,
            )
        )

    return PredictResponse(predictions=results)


@app.get("/explain/global", response_model=ExplainGlobalResponse, tags=["Explainability"])
async def explain_global():
    """Return ranked global feature importances."""
    df = _get_global_importance()
    features = [
        FeatureImportance(
            rank=int(row["rank"]),
            feature=row["feature"],
            importance=round(float(row["importance"]), 6),
            importance_pct=round(float(row["importance_pct"]), 2),
        )
        for _, row in df.iterrows()
    ]
    return ExplainGlobalResponse(features=features)
