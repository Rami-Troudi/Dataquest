from __future__ import annotations

import importlib.util

import joblib
import pandas as pd

from core.inference import load_artifacts, predict_proba, preprocess_input


def load_solution_module():
    spec = importlib.util.spec_from_file_location("submission_solution", "solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parity_with_solution_predict() -> None:
    sol = load_solution_module()

    model = joblib.load("model.pkl")
    row = pd.read_csv("test.csv").head(1)

    expected = sol.predict(sol.preprocess(row.copy()), model)
    expected_class = int(expected.iloc[0]["Purchased_Coverage_Bundle"])

    artifacts = load_artifacts("model.pkl")
    df, _ = preprocess_input(row.iloc[0].to_dict())
    proba = predict_proba(df, artifacts)
    actual_class = int(artifacts.class_order[proba.argmax(axis=1)[0]])
    assert actual_class == expected_class
