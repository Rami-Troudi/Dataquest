from __future__ import annotations

import os
from typing import Any

import joblib

from core.constants import MODEL_REQUIRED_KEYS


def load_model_bundle(model_path: str) -> dict[str, Any]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise TypeError("Model artifact must be a dict bundle.")

    missing = sorted(MODEL_REQUIRED_KEYS - set(bundle.keys()))
    if missing:
        raise KeyError(f"Model bundle missing keys: {missing}")

    rf_model = bundle["rf_model"]
    preprocessor = bundle["preprocessor"]
    class_order = bundle["class_order"]

    if not hasattr(rf_model, "predict_proba"):
        raise TypeError("rf_model must expose predict_proba.")
    if not isinstance(preprocessor, dict):
        raise TypeError("preprocessor must be a dict.")
    if not isinstance(class_order, (list, tuple)):
        raise TypeError("class_order must be a list or tuple.")

    return bundle
