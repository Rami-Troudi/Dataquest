import numpy as np
import pandas as pd


def build_global_feature_importance(rf_model, feature_columns):
    if not hasattr(rf_model, 'feature_importances_'):
        raise ValueError('Model does not expose feature_importances_.')

    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance['importance_rank'] = np.arange(1, len(importance) + 1)
    return importance[['importance_rank', 'feature', 'importance']]


def build_local_reason_codes(
    X_encoded,
    user_ids,
    rf_model,
    preprocessor,
    top_k=3
):
    feature_columns = preprocessor['feature_columns']
    medians = pd.Series(preprocessor['numeric_medians']).reindex(feature_columns).fillna(0.0)
    importances = pd.Series(rf_model.feature_importances_, index=feature_columns)

    centered = X_encoded[feature_columns].subtract(medians, axis=1)
    scores = centered.abs().multiply(importances, axis=1)
    scores_np = scores.to_numpy()
    cols = np.array(feature_columns)

    top_idx = np.argpartition(scores_np, -top_k, axis=1)[:, -top_k:]

    reason_rows = []
    for row_idx in range(scores_np.shape[0]):
        row_top = top_idx[row_idx]
        ranked = row_top[np.argsort(scores_np[row_idx, row_top])[::-1]]
        features = cols[ranked]
        contributions = scores_np[row_idx, ranked]
        raw_values = X_encoded.iloc[row_idx][features].to_numpy()

        reason_rows.append({
            'User_ID': user_ids.iloc[row_idx],
            'reason_1_feature': features[0],
            'reason_1_score': float(contributions[0]),
            'reason_1_value': float(raw_values[0]),
            'reason_2_feature': features[1] if top_k > 1 else '',
            'reason_2_score': float(contributions[1]) if top_k > 1 else 0.0,
            'reason_2_value': float(raw_values[1]) if top_k > 1 else 0.0,
            'reason_3_feature': features[2] if top_k > 2 else '',
            'reason_3_score': float(contributions[2]) if top_k > 2 else 0.0,
            'reason_3_value': float(raw_values[2]) if top_k > 2 else 0.0,
        })

    return pd.DataFrame(reason_rows)
