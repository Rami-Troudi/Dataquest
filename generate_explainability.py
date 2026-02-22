import joblib
import pandas as pd

from solution import preprocess
from explainability import build_global_feature_importance, build_local_reason_codes


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
            df[col] = 0

    df = df[preprocessor['feature_columns']]
    df = df.fillna(pd.Series(preprocessor['numeric_medians']))
    return df


model_bundle = joblib.load('model.pkl')
rf_model = model_bundle['rf_model']
preprocessor = model_bundle['preprocessor']

test_df = pd.read_csv('test.csv')
test_engineered = preprocess(test_df)
X_test = transform_with_preprocessor(test_engineered, preprocessor)

global_importance = build_global_feature_importance(rf_model, preprocessor['feature_columns'])
global_importance.to_csv('explainability_global_importance.csv', index=False)

local_reasons = build_local_reason_codes(
    X_encoded=X_test,
    user_ids=test_engineered['User_ID'],
    rf_model=rf_model,
    preprocessor=preprocessor,
    top_k=3
)
local_reasons.to_csv('explainability_reason_codes.csv', index=False)

print('Saved explainability_global_importance.csv')
print('Saved explainability_reason_codes.csv')
