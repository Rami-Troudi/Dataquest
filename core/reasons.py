from __future__ import annotations

from typing import Any

import pandas as pd


def top_feature_importances(model: Any, feature_names: list[str], top_n: int = 20) -> list[dict[str, Any]]:
    importances = model.feature_importances_
    ranking = sorted(
        zip(feature_names, importances),
        key=lambda x: float(x[1]),
        reverse=True,
    )[:top_n]
    return [
        {"feature": feature, "importance": float(score)}
        for feature, score in ranking
    ]


def generate_reasons(
    raw_record: dict[str, Any],
    top_features: list[dict[str, Any]],
    engineered_row: pd.Series,
) -> list[str]:
    reasons: list[str] = []
    top_names = [item["feature"] for item in top_features]

    if "Claims_Frequency" in top_names:
        claims_freq = float(engineered_row.get("Claims_Frequency", 0.0))
        if claims_freq > 0.2:
            reasons.append("Claims_Frequency is high, suggesting stronger coverage bundles.")
        else:
            reasons.append("Claims_Frequency is low, supporting balanced coverage options.")

    if "Income_per_Dependent" in top_names:
        income = float(engineered_row.get("Income_per_Dependent", 0.0))
        if income > 25000:
            reasons.append("Income_per_Dependent is high, enabling broader bundle affordability.")
        else:
            reasons.append("Income_per_Dependent is moderate, favoring cost-efficient bundles.")

    if "Loyalty_Score" in top_names:
        loyalty = float(engineered_row.get("Loyalty_Score", 0.0))
        if loyalty > 0:
            reasons.append("Positive Loyalty_Score indicates stable policy behavior.")
        else:
            reasons.append("Loyalty_Score is low, indicating a volatile prior policy profile.")

    if not reasons and raw_record.get("Vehicles_on_Policy", 0) > 1:
        reasons.append("Multiple vehicles increase preference for broader automotive coverage.")
    if not reasons:
        reasons.append("Historical and demographic signals align with the recommended bundle.")

    return reasons[:3]
