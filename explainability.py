import numpy as np
import pandas as pd


def _safe_feature_importance(rf_model, feature_count: int) -> np.ndarray:
    importances = getattr(rf_model, 'feature_importances_', None)
    if importances is None:
        return np.full(feature_count, 1.0 / max(feature_count, 1), dtype=float)

    importances = np.asarray(importances, dtype=float)
    if importances.shape[0] != feature_count:
        if importances.shape[0] > feature_count:
            importances = importances[:feature_count]
        else:
            padded = np.zeros(feature_count, dtype=float)
            padded[: importances.shape[0]] = importances
            importances = padded

    total = float(importances.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(feature_count, 1.0 / max(feature_count, 1), dtype=float)
    return importances


def build_global_feature_importance(rf_model, feature_columns):
    feature_columns = list(feature_columns)
    importances = _safe_feature_importance(rf_model, len(feature_columns))

    df = pd.DataFrame(
        {
            'feature': feature_columns,
            'importance': importances,
        }
    )
    df = df.sort_values('importance', ascending=False, ignore_index=True)
    total = float(df['importance'].sum())
    if total > 0:
        df['importance_pct'] = (df['importance'] / total) * 100.0
    else:
        df['importance_pct'] = 0.0
    df['rank'] = np.arange(1, len(df) + 1)
    return df[['rank', 'feature', 'importance', 'importance_pct']]


def _build_category_decoders(preprocessor: dict) -> dict:
    decoders = {}
    cat_mappings = preprocessor.get('cat_mappings', {})
    for col, mapping in cat_mappings.items():
        inv = {int(v): str(k) for k, v in mapping.items()}
        inv[-1] = 'UNKNOWN'
        decoders[col] = inv
    return decoders


def _baseline_vector(feature_columns, preprocessor: dict) -> np.ndarray:
    medians = preprocessor.get('numeric_medians', {}) or {}
    cat_cols = set(preprocessor.get('categorical_columns', []) or [])
    baseline = []
    for col in feature_columns:
        if col in cat_cols:
            baseline.append(0.0)
        else:
            value = medians.get(col, 0.0)
            try:
                baseline.append(float(value))
            except Exception:
                baseline.append(0.0)
    return np.asarray(baseline, dtype=float)


def _format_feature_value(feature: str, value: float, decoders: dict) -> str:
    if feature in decoders:
        key = int(round(float(value)))
        raw = decoders[feature].get(key, 'UNKNOWN')
        return f'{feature}={raw}'
    if np.isfinite(value):
        if abs(value - round(value)) < 1e-9:
            return f'{feature}={int(round(value))}'
        return f'{feature}={value:.4f}'
    return f'{feature}=MISSING'


def build_local_reason_codes(X_encoded, user_ids, rf_model, preprocessor, top_k=3):
    if isinstance(X_encoded, pd.DataFrame):
        X_df = X_encoded.copy()
    else:
        X_df = pd.DataFrame(X_encoded, columns=preprocessor.get('feature_columns', None))

    feature_columns = preprocessor.get('feature_columns', X_df.columns.tolist())
    X_df = X_df[feature_columns]

    X_values = X_df.to_numpy(dtype=float, copy=False)
    n_rows, n_features = X_values.shape

    importances = _safe_feature_importance(rf_model, n_features)
    baseline = _baseline_vector(feature_columns, preprocessor)

    deviations = np.abs(X_values - baseline)
    local_scores = deviations * (importances + 1e-12)

    k = int(max(1, min(top_k, n_features)))
    top_idx = np.argpartition(-local_scores, kth=k - 1, axis=1)[:, :k]

    row_selector = np.arange(n_rows)[:, None]
    top_scores = local_scores[row_selector, top_idx]
    ordering = np.argsort(-top_scores, axis=1)
    top_idx = top_idx[row_selector, ordering]

    pred = rf_model.predict(X_df)
    proba = rf_model.predict_proba(X_df)

    class_to_col = {c: i for i, c in enumerate(rf_model.classes_)}
    pred_prob = np.array([proba[i, class_to_col[pred[i]]] for i in range(n_rows)], dtype=float)

    decoders = _build_category_decoders(preprocessor)

    out = pd.DataFrame(
        {
            'User_ID': pd.Series(user_ids).astype(str).values,
            'Purchased_Coverage_Bundle': pred.astype(int),
            'Predicted_Probability': np.round(pred_prob, 6),
        }
    )

    for rank in range(k):
        col_name = f'Reason_{rank + 1}'
        feat_idx = top_idx[:, rank]
        reasons = [
            _format_feature_value(feature_columns[j], X_values[i, j], decoders)
            for i, j in enumerate(feat_idx)
        ]
        out[col_name] = reasons

    return out
