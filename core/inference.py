from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.constants import (
    BUNDLE_NAME_BY_ID,
    DEFAULT_VALUES,
    ID_COL,
    INPUT_COLUMNS,
    SUGGESTED_VERIFY_FIELDS,
)
from core.inference_core import preprocess as engineered_preprocess
from core.model_loader import load_model_bundle


@dataclass
class InferenceArtifacts:
    model_bundle: dict[str, Any]
    model: Any
    preprocessor: dict[str, Any]
    feature_list: list[str]
    class_order: np.ndarray
    model_version: str


def load_artifacts(model_path: str) -> InferenceArtifacts:
    bundle = load_model_bundle(model_path)
    model = bundle["rf_model"]
    preprocessor = bundle["preprocessor"]
    feature_list = list(preprocessor["feature_columns"])
    class_order = np.array(bundle["class_order"], dtype=int)
    version = str(bundle.get("model_type", "rf_only"))
    return InferenceArtifacts(
        model_bundle=bundle,
        model=model,
        preprocessor=preprocessor,
        feature_list=feature_list,
        class_order=class_order,
        model_version=version,
    )


def preprocess_input(payload: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    row: dict[str, Any] = {}
    warnings: list[str] = []

    for col in INPUT_COLUMNS:
        if col in payload:
            row[col] = payload[col]
        else:
            row[col] = DEFAULT_VALUES[col]
            warnings.append(f"default used for missing field: {col}")

    df = pd.DataFrame([row])
    return df, warnings


def _transform_with_preprocessor(
    engineered_df: pd.DataFrame, preprocessor: dict[str, Any]
) -> pd.DataFrame:
    d = engineered_df.copy()
    if ID_COL in d.columns:
        d = d.drop(columns=[ID_COL])

    categorical_columns = preprocessor["categorical_columns"]
    cat_mappings = preprocessor["cat_mappings"]
    medians = preprocessor["numeric_medians"]
    feature_columns = preprocessor["feature_columns"]

    for col in categorical_columns:
        values = d[col].astype(str).fillna("__MISSING__")
        d[col] = values.map(cat_mappings[col]).fillna(-1).astype(int)

    for col in feature_columns:
        if col not in d.columns:
            d[col] = np.nan

    d = d.reindex(columns=feature_columns)
    d = d.fillna(pd.Series(medians))
    return d


def predict_proba(df: pd.DataFrame, artifacts: InferenceArtifacts) -> np.ndarray:
    engineered = engineered_preprocess(df)
    transformed = _transform_with_preprocessor(engineered, artifacts.preprocessor)
    transformed = transformed.reindex(columns=artifacts.feature_list)
    return artifacts.model.predict_proba(transformed)


def transform_features(df: pd.DataFrame, artifacts: InferenceArtifacts) -> pd.DataFrame:
    engineered = engineered_preprocess(df)
    transformed = _transform_with_preprocessor(engineered, artifacts.preprocessor)
    return transformed.reindex(columns=artifacts.feature_list)


def predict_topk(
    proba: np.ndarray, class_order: np.ndarray, k: int = 3
) -> list[dict[str, float | int | str]]:
    items = [
        {
            "bundle_id": int(class_order[idx]),
            "bundle_name": BUNDLE_NAME_BY_ID.get(int(class_order[idx]), str(class_order[idx])),
            "proba": float(proba[0, idx]),
        }
        for idx in range(proba.shape[1])
    ]
    items.sort(key=lambda x: float(x["proba"]), reverse=True)
    return items[:k]


def confidence_payload(proba: np.ndarray, threshold: float = 0.40) -> dict[str, Any]:
    max_proba = float(np.max(proba))
    if max_proba < threshold:
        return {
            "confidence": "low",
            "suggested_fields_to_verify": SUGGESTED_VERIFY_FIELDS[:2],
        }
    return {"confidence": "normal", "suggested_fields_to_verify": []}
