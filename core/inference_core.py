from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.bundle_map import bundle_name
from core.constants import ID_COL, MONTH_MAP, TARGET_COL


@dataclass
class InferenceResult:
    user_id: str
    bundle_id: int
    bundle_name: str
    probabilities: dict[int, float]
    top_k: list[dict[str, float | int | str]]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["Total_Dependents"] = (
        d["Adult_Dependents"] + d["Child_Dependents"] + d["Infant_Dependents"]
    )

    d["Child_Ratio"] = d["Child_Dependents"] / (d["Total_Dependents"] + 1)
    d["Adult_Ratio"] = d["Adult_Dependents"] / (d["Total_Dependents"] + 1)
    d["Infant_Ratio"] = d["Infant_Dependents"] / (d["Total_Dependents"] + 1)

    d["Total_Policy_Duration_Years"] = d["Previous_Policy_Duration_Months"] / 12

    d["Claims_Frequency"] = d["Previous_Claims_Filed"] / (
        d["Previous_Policy_Duration_Months"] + 1
    )
    d["Claims_to_YearsWithout_Ratio"] = d["Previous_Claims_Filed"] / (
        d["Years_Without_Claims"] + 1
    )

    d["Income_per_Dependent"] = d["Estimated_Annual_Income"] / (d["Total_Dependents"] + 1)

    d["Total_Processing_Days"] = d["Days_Since_Quote"] + d["Underwriting_Processing_Days"]
    d["Processing_Efficiency"] = d["Underwriting_Processing_Days"] / (d["Days_Since_Quote"] + 1)

    d["Amendments_per_Day"] = d["Policy_Amendments_Count"] / (d["Days_Since_Quote"] + 1)

    d["Vehicles_per_Adult"] = d["Vehicles_on_Policy"] / (d["Adult_Dependents"] + 1)

    d["High_Risk_Customer"] = (
        (d["Previous_Claims_Filed"] > 2) & (d["Policy_Cancelled_Post_Purchase"] == 1)
    ).astype(int)

    month_col = d["Policy_Start_Month"].astype(str).str.strip().map(MONTH_MAP)
    d["Policy_Start_Month_Num"] = pd.to_numeric(month_col, errors="coerce").fillna(6)

    d["Policy_Start_Month_Sin"] = np.sin(2 * np.pi * d["Policy_Start_Month_Num"] / 12)
    d["Policy_Start_Month_Cos"] = np.cos(2 * np.pi * d["Policy_Start_Month_Num"] / 12)
    d["Policy_Start_Day_Sin"] = np.sin(
        2 * np.pi * pd.to_numeric(d["Policy_Start_Day"], errors="coerce") / 31
    )
    d["Policy_Start_Day_Cos"] = np.cos(
        2 * np.pi * pd.to_numeric(d["Policy_Start_Day"], errors="coerce") / 31
    )
    d["Policy_Start_Week_Sin"] = np.sin(
        2 * np.pi * pd.to_numeric(d["Policy_Start_Week"], errors="coerce") / 52
    )
    d["Policy_Start_Week_Cos"] = np.cos(
        2 * np.pi * pd.to_numeric(d["Policy_Start_Week"], errors="coerce") / 52
    )

    d["Loyalty_Score"] = (
        d["Years_Without_Claims"] * 0.3
        + d["Previous_Policy_Duration_Months"] * 0.02
        - d["Previous_Claims_Filed"] * 0.5
    )

    d["Payment_Flexibility"] = d["Grace_Period_Extensions"]

    return d


def transform_with_preprocessor(df: pd.DataFrame, preprocessor: dict[str, Any]) -> pd.DataFrame:
    d = df.copy()

    if ID_COL in d.columns:
        d = d.drop([ID_COL], axis=1)
    if TARGET_COL in d.columns:
        d = d.drop([TARGET_COL], axis=1)

    for col in preprocessor["categorical_columns"]:
        values = d[col].astype(str).fillna("__MISSING__")
        mapping = preprocessor["cat_mappings"][col]
        d[col] = values.map(mapping).fillna(-1).astype(int)

    for col in preprocessor["feature_columns"]:
        if col not in d.columns:
            d[col] = np.nan

    d = d[preprocessor["feature_columns"]]
    d = d.fillna(pd.Series(preprocessor["numeric_medians"]))
    return d


def predict_one(record_df: pd.DataFrame, model_bundle: dict[str, Any], top_k: int = 3) -> InferenceResult:
    if "rf_model" not in model_bundle:
        raise KeyError("Model bundle must contain 'rf_model'.")
    if "preprocessor" not in model_bundle:
        raise KeyError("Model bundle must contain 'preprocessor'.")
    if "class_order" not in model_bundle:
        raise KeyError("Model bundle must contain 'class_order'.")

    rf_model = model_bundle["rf_model"]
    preprocessor = model_bundle["preprocessor"]
    class_order = np.array(model_bundle["class_order"], dtype=int)

    if ID_COL not in record_df.columns:
        raise KeyError("Input DataFrame must include User_ID.")
    if len(record_df) != 1:
        raise ValueError("predict_one expects exactly one record.")

    processed = preprocess(record_df)
    X = transform_with_preprocessor(processed, preprocessor)

    proba = rf_model.predict_proba(X)
    pred_idx = int(np.argmax(proba, axis=1)[0])
    bundle_id = int(class_order[pred_idx])

    prob_map = {
        int(class_order[idx]): float(proba[0, idx]) for idx in range(proba.shape[1])
    }

    ranked = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_items = [
        {
            "bundle_id": int(cls_id),
            "bundle_name": bundle_name(int(cls_id)),
            "probability": float(prob),
        }
        for cls_id, prob in ranked
    ]

    user_id = str(record_df.iloc[0][ID_COL])
    return InferenceResult(
        user_id=user_id,
        bundle_id=bundle_id,
        bundle_name=bundle_name(bundle_id),
        probabilities=prob_map,
        top_k=top_items,
    )
