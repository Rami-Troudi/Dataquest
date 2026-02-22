from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from core.constants import ID_COL, TARGET_COL
from core.inference_core import preprocess


def fit_preprocessor(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.Series]:
    d = df.copy()
    y = d[TARGET_COL]
    d = d.drop([ID_COL, TARGET_COL], axis=1)

    categorical = d.select_dtypes(include=["object", "string"]).columns.tolist()
    cat_mappings: dict[str, dict[str, int]] = {}

    for col in categorical:
        values = d[col].astype(str).fillna("__MISSING__")
        mapping = {value: idx for idx, value in enumerate(sorted(values.unique()))}
        cat_mappings[col] = mapping
        d[col] = values.map(mapping).astype(int)

    numeric_medians = d.median(numeric_only=True)
    d = d.fillna(numeric_medians)

    preprocessor = {
        "categorical_columns": categorical,
        "cat_mappings": cat_mappings,
        "numeric_medians": numeric_medians.to_dict(),
        "feature_columns": d.columns.tolist(),
    }
    return preprocessor, d, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    train_csv = Path(cfg["train_csv"])
    model_out = Path(cfg["model_out"])
    seed = int(cfg["seed"])

    raw = pd.read_csv(train_csv)
    engineered = preprocess(raw)
    preprocessor, X, y = fit_preprocessor(engineered)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    params = cfg["rf_params"]
    model = RandomForestClassifier(random_state=seed, **params)
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    macro_f1 = f1_score(y_val, pred, average="macro", zero_division=0)

    final_model = RandomForestClassifier(random_state=seed, **params)
    final_model.fit(X, y)

    bundle = {
        "rf_model": final_model,
        "preprocessor": preprocessor,
        "class_order": final_model.classes_.tolist(),
        "model_type": "rf_only",
        "tuning": {
            "validation_macro_f1": float(macro_f1),
            "rf_params": params,
        },
    }

    joblib.dump(bundle, model_out, compress=("xz", 3))
    size_mb = model_out.stat().st_size / (1024 ** 2)
    print(f"Saved model to {model_out} ({size_mb:.2f} MB)")
    print(f"Validation Macro-F1: {macro_f1:.5f}")


if __name__ == "__main__":
    main()
