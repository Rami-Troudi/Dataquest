import numpy as np
import pandas as pd
import joblib
import os
import tempfile
import time
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split, ParameterSampler

from solution import preprocess
from explainability import build_global_feature_importance, build_local_reason_codes

JUDGE_SLOWDOWN_FACTOR = 4.0
MODEL_SIZE_CAP_MB = 35.0
QUICK_MODE = os.getenv('QUICK_MODE', '0') == '1'


def fit_preprocessor(df):
    df = df.copy()
    target = df['Purchased_Coverage_Bundle']
    df = df.drop(['User_ID', 'Purchased_Coverage_Bundle'], axis=1)

    categorical = df.select_dtypes(include=['object', 'str']).columns.tolist()
    cat_mappings = {}

    for col in categorical:
        values = df[col].astype(str).fillna('__MISSING__')
        mapping = {value: index for index, value in enumerate(sorted(values.unique()))}
        cat_mappings[col] = mapping
        df[col] = values.map(mapping).astype(int)

    numeric_medians = df.median(numeric_only=True)
    df = df.fillna(numeric_medians)

    preprocessor = {
        'categorical_columns': categorical,
        'cat_mappings': cat_mappings,
        'numeric_medians': numeric_medians.to_dict(),
        'feature_columns': df.columns.tolist()
    }
    return preprocessor, df, target


def transform_with_preprocessor(df, preprocessor):
    df = df.copy()
    if 'User_ID' in df.columns:
        df = df.drop(['User_ID'], axis=1)
    if 'Purchased_Coverage_Bundle' in df.columns:
        df = df.drop(['Purchased_Coverage_Bundle'], axis=1)

    for col in preprocessor['categorical_columns']:
        values = df[col].astype(str).fillna('__MISSING__')
        mapping = preprocessor['cat_mappings'][col]
        df[col] = values.map(mapping).fillna(-1).astype(int)

    for col in preprocessor['feature_columns']:
        if col not in df.columns:
            df[col] = np.nan

    df = df[preprocessor['feature_columns']]
    df = df.fillna(pd.Series(preprocessor['numeric_medians']))
    return df


def oversample_minority(X, y, min_count=300, seed=42):
    rng = np.random.RandomState(seed)
    counts = y.value_counts()
    X_parts = [X]
    y_parts = [y]

    for cls, cnt in counts.items():
        if cnt < min_count:
            needed = min_count - cnt
            sampled_index = y[y == cls].sample(needed, replace=True, random_state=rng).index
            X_parts.append(X.loc[sampled_index])
            y_parts.append(y.loc[sampled_index])

    X_out = pd.concat(X_parts, axis=0).reset_index(drop=True)
    y_out = pd.concat(y_parts, axis=0).reset_index(drop=True)
    return X_out, y_out


def evaluate_final_score_proxy(model, X_val, y_val):
    start = time.perf_counter()
    val_pred = model.predict(X_val)
    predict_seconds = time.perf_counter() - start

    macro_f1 = f1_score(y_val, val_pred, average='macro')

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        path = tmp.name
    try:
        joblib.dump(model, path)
        size_mb = os.path.getsize(path) / (1024 ** 2)
    finally:
        if os.path.exists(path):
            os.remove(path)

    effective_predict_seconds = predict_seconds * JUDGE_SLOWDOWN_FACTOR
    size_penalty = max(0.5, 1 - size_mb / 200)
    duration_penalty = max(0.5, 1 - effective_predict_seconds / 10)
    proxy = macro_f1 * size_penalty * duration_penalty

    return {
        'macro_f1': macro_f1,
        'predict_seconds': predict_seconds,
        'effective_predict_seconds': effective_predict_seconds,
        'model_size_mb': size_mb,
        'size_penalty': size_penalty,
        'duration_penalty': duration_penalty,
        'final_score_proxy': proxy
    }


print('=' * 80)
print('RF-ONLY TRAINING STARTED')
print('=' * 80)

train_df = pd.read_csv('train (1).csv')
test_df = pd.read_csv('test.csv')

train_engineered = preprocess(train_df)
test_engineered = preprocess(test_df)

preprocessor, X_full, y_full = fit_preprocessor(train_engineered)
X_test = transform_with_preprocessor(test_engineered, preprocessor)

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

print(f'Train shape: {X_train.shape}, Validation shape: {X_val.shape}')

param_distributions = {
    'n_estimators': [50, 80, 110, 140, 180],
    'max_depth': [8, 12, 16, 20, 24],
    'min_samples_split': [2, 4, 8, 12],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.5, 0.7],
    'class_weight': ['balanced', 'balanced_subsample', None],
    'criterion': ['gini', 'entropy']
}

search_iter = 6 if QUICK_MODE else 20
sampled_params = list(ParameterSampler(param_distributions, n_iter=search_iter, random_state=42))

best_score = -1.0
best_metrics = None
best_params = None
best_mode = None
best_min_count = None

search_spaces = [
    ('plain', None),
    ('oversample_300', 300),
    ('oversample_500', 500)
]

for params in sampled_params:
    for mode_name, min_count in search_spaces:
        if min_count is None:
            X_fit, y_fit = X_train, y_train
        else:
            X_fit, y_fit = oversample_minority(X_train, y_train, min_count=min_count, seed=42)

        candidate = RandomForestClassifier(
            **params,
            random_state=42,
            n_jobs=-1
        )
        candidate.fit(X_fit, y_fit)
        metrics = evaluate_final_score_proxy(candidate, X_val, y_val)

        if metrics['model_size_mb'] > MODEL_SIZE_CAP_MB:
            continue

        print(
            f"mode={mode_name:14s} score={metrics['final_score_proxy']:.4f} "
            f"f1={metrics['macro_f1']:.4f} size={metrics['model_size_mb']:.2f}MB "
            f"lat={metrics['predict_seconds']:.3f}s"
        )

        if metrics['final_score_proxy'] > best_score:
            best_score = metrics['final_score_proxy']
            best_metrics = metrics
            best_params = params
            best_mode = mode_name
            best_min_count = min_count

if best_params is None:
    raise RuntimeError('No candidate satisfied the robust size objective. Expand search space.')


def unique_preserve(values):
    seen = set()
    output = []
    for value in values:
        marker = str(value)
        if marker not in seen:
            seen.add(marker)
            output.append(value)
    return output


def neighborhood_for_best(params):
    best_max_features = params['max_features']
    if isinstance(best_max_features, float):
        feature_options = unique_preserve([
            max(0.3, round(best_max_features - 0.1, 2)),
            best_max_features,
            min(0.9, round(best_max_features + 0.1, 2)),
            0.5,
            0.7,
            'sqrt',
            'log2'
        ])
    else:
        feature_options = unique_preserve([best_max_features, 'sqrt', 'log2', 0.5, 0.7])

    depth = params['max_depth']
    depth_options = unique_preserve([
        max(6, depth - 4),
        depth,
        min(30, depth + 4)
    ])

    split = params['min_samples_split']
    split_options = unique_preserve([
        max(2, split - 2),
        split,
        split + 2,
    ])

    leaf = params['min_samples_leaf']
    leaf_options = unique_preserve([
        max(1, leaf - 1),
        leaf,
        leaf + 1,
    ])

    n_est = params['n_estimators']
    est_options = unique_preserve([
        max(40, n_est - 20),
        n_est,
        n_est + 20,
        n_est + 40,
    ])

    class_weight_options = unique_preserve([
        params['class_weight'],
        None,
        'balanced',
        'balanced_subsample',
    ])

    criterion_options = unique_preserve([params['criterion'], 'gini', 'entropy'])

    combos = []
    for e, d, s, l, mf, cw, cr in product(
        est_options,
        depth_options,
        split_options,
        leaf_options,
        feature_options,
        class_weight_options,
        criterion_options,
    ):
        combos.append({
            'n_estimators': e,
            'max_depth': d,
            'min_samples_split': s,
            'min_samples_leaf': l,
            'max_features': mf,
            'class_weight': cw,
            'criterion': cr,
        })

    return combos


print('\n' + '=' * 80)
print('REFINEMENT SEARCH')
print('=' * 80)

refinement_candidates = neighborhood_for_best(best_params)
refinement_iter = 8 if QUICK_MODE else 40
refinement_candidates = list(ParameterSampler(
    {
        'candidate': refinement_candidates
    },
    n_iter=min(refinement_iter, len(refinement_candidates)),
    random_state=7
))

if best_mode == 'plain':
    refinement_spaces = [('plain', None), ('oversample_300', 300)]
elif best_mode == 'oversample_300':
    refinement_spaces = [('oversample_300', 300), ('oversample_500', 500), ('plain', None)]
else:
    refinement_spaces = [('oversample_500', 500), ('oversample_300', 300), ('plain', None)]

for item in refinement_candidates:
    params = item['candidate']
    for mode_name, min_count in refinement_spaces:
        if min_count is None:
            X_fit, y_fit = X_train, y_train
        else:
            X_fit, y_fit = oversample_minority(X_train, y_train, min_count=min_count, seed=42)

        candidate = RandomForestClassifier(
            **params,
            random_state=42,
            n_jobs=-1
        )
        candidate.fit(X_fit, y_fit)
        metrics = evaluate_final_score_proxy(candidate, X_val, y_val)

        if metrics['model_size_mb'] > MODEL_SIZE_CAP_MB:
            continue

        if metrics['final_score_proxy'] > best_score:
            best_score = metrics['final_score_proxy']
            best_metrics = metrics
            best_params = params
            best_mode = mode_name
            best_min_count = min_count
            print(
                f"refine_improve mode={mode_name:14s} score={metrics['final_score_proxy']:.4f} "
                f"f1={metrics['macro_f1']:.4f} size={metrics['model_size_mb']:.2f}MB "
                f"lat={metrics['predict_seconds']:.3f}s"
            )

print('\n' + '=' * 80)
print('BEST RF CONFIGURATION')
print('=' * 80)
print(f'Best mode: {best_mode}')
print(f'Best params: {best_params}')
print(
    f"Best proxy score: {best_metrics['final_score_proxy']:.4f} "
    f"(f1={best_metrics['macro_f1']:.4f}, "
    f"size={best_metrics['model_size_mb']:.2f}MB, "
    f"lat={best_metrics['predict_seconds']:.3f}s, "
    f"lat_x{JUDGE_SLOWDOWN_FACTOR:.1f}={best_metrics['effective_predict_seconds']:.3f}s)"
)

if best_min_count is None:
    X_train_best, y_train_best = X_train, y_train
else:
    X_train_best, y_train_best = oversample_minority(X_train, y_train, min_count=best_min_count, seed=42)

rf_model = RandomForestClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_best, y_train_best)

val_pred = rf_model.predict(X_val)
val_f1 = f1_score(y_val, val_pred, average='macro')
final_metrics = evaluate_final_score_proxy(rf_model, X_val, y_val)
print(f'\nValidation Macro F1 (RF-only tuned): {val_f1:.4f}')
print(
    f"Validation score proxy: {final_metrics['final_score_proxy']:.4f} "
    f"| size={final_metrics['model_size_mb']:.2f}MB "
    f"| latency={final_metrics['predict_seconds']:.3f}s "
    f"| latency_x{JUDGE_SLOWDOWN_FACTOR:.1f}={final_metrics['effective_predict_seconds']:.3f}s"
)
print('\nClassification report:\n')
print(classification_report(y_val, val_pred, zero_division=0))

if best_min_count is None:
    X_fit_final, y_fit_final = X_full, y_full
else:
    X_fit_final, y_fit_final = oversample_minority(X_full, y_full, min_count=best_min_count, seed=42)

rf_final = RandomForestClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1
)
rf_final.fit(X_fit_final, y_fit_final)

bundle = {
    'rf_model': rf_final,
    'preprocessor': preprocessor,
    'class_order': rf_final.classes_.tolist(),
    'model_type': 'rf_only',
    'tuning': {
        'best_mode': best_mode,
        'best_params': best_params,
        'best_validation_macro_f1': float(val_f1),
        'best_validation_score_proxy': float(final_metrics['final_score_proxy']),
        'best_validation_model_size_mb': float(final_metrics['model_size_mb']),
        'best_validation_predict_seconds': float(final_metrics['predict_seconds']),
        'best_validation_predict_seconds_adjusted': float(final_metrics['effective_predict_seconds']),
        'judge_slowdown_factor': float(JUDGE_SLOWDOWN_FACTOR),
        'model_size_cap_mb': float(MODEL_SIZE_CAP_MB)
    }
}
joblib.dump(bundle, 'model.pkl', compress=('xz', 3))

preds = rf_final.predict(X_test)
submission = pd.DataFrame({
    'User_ID': test_engineered['User_ID'],
    'Purchased_Coverage_Bundle': preds.astype(int)
})
submission.to_csv('submission_rf_only.csv', index=False)

print(f"Saved model to model.pkl (compressed, size={os.path.getsize('model.pkl') / (1024 ** 2):.2f}MB)")
print('Saved predictions to submission_rf_only.csv')

global_importance = build_global_feature_importance(rf_final, preprocessor['feature_columns'])
global_importance.to_csv('explainability_global_importance.csv', index=False)

local_reasons = build_local_reason_codes(
    X_encoded=X_test,
    user_ids=test_engineered['User_ID'],
    rf_model=rf_final,
    preprocessor=preprocessor,
    top_k=3
)
local_reasons.to_csv('explainability_reason_codes.csv', index=False)

print('Saved explainability_global_importance.csv')
print('Saved explainability_reason_codes.csv')