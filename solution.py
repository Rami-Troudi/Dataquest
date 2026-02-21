# ----------------------------------------------------------------
# IMPORTANT: This template will be used to evaluate your solution.
#
# Do NOT change the function signatures.
# And ensure that your code runs within the time limits.
# The time calculation will be computed for the predict function only.
#
# Good luck!
# ----------------------------------------------------------------


# Import necessary libraries here
import pandas as pd
import numpy as np
import joblib


def preprocess(df):
    # Implement any preprocessing steps required for your model here.
    # Return a Pandas DataFrame of the data
    #
    # Note: Don't drop the 'User_ID' column here.
    # It will be used in the predict function to return the final predictions.
    
    df = df.copy()
    
    # =============================================================================
    # Feature Engineering
    # =============================================================================
    
    # 1. Total Family Members
    df['Total_Dependents'] = (df['Adult_Dependents'] + 
                              df['Child_Dependents'] + 
                              df['Infant_Dependents'])
    
    # 2. Family composition ratios
    df['Child_Ratio'] = df['Child_Dependents'] / (df['Total_Dependents'] + 1)
    df['Adult_Ratio'] = df['Adult_Dependents'] / (df['Total_Dependents'] + 1)
    df['Infant_Ratio'] = df['Infant_Dependents'] / (df['Total_Dependents'] + 1)
    
    # 3. Contract/Policy Length Features
    df['Total_Policy_Duration_Years'] = df['Previous_Policy_Duration_Months'] / 12
    
    # 4. Claims behavior metrics
    df['Claims_Frequency'] = df['Previous_Claims_Filed'] / (df['Previous_Policy_Duration_Months'] + 1)
    df['Claims_to_YearsWithout_Ratio'] = df['Previous_Claims_Filed'] / (df['Years_Without_Claims'] + 1)
    
    # 5. Average money spent estimate (based on income and deductible tier)
    df['Income_per_Dependent'] = df['Estimated_Annual_Income'] / (df['Total_Dependents'] + 1)
    
    # 6. Processing efficiency
    df['Total_Processing_Days'] = df['Days_Since_Quote'] + df['Underwriting_Processing_Days']
    df['Processing_Efficiency'] = df['Underwriting_Processing_Days'] / (df['Days_Since_Quote'] + 1)
    
    # 7. Policy modification intensity
    df['Amendments_per_Day'] = df['Policy_Amendments_Count'] / (df['Days_Since_Quote'] + 1)
    
    # 8. Vehicle coverage ratio
    df['Vehicles_per_Adult'] = df['Vehicles_on_Policy'] / (df['Adult_Dependents'] + 1)
    
    # 9. Risk indicators
    df['High_Risk_Customer'] = ((df['Previous_Claims_Filed'] > 2) & 
                                (df['Policy_Cancelled_Post_Purchase'] == 1)).astype(int)
    
    # 10. Time features - cyclical encoding
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    month_col = df['Policy_Start_Month'].astype(str).str.strip().map(month_mapping)
    df['Policy_Start_Month_Num'] = pd.to_numeric(month_col, errors='coerce').fillna(6)
    
    df['Policy_Start_Month_Sin'] = np.sin(2 * np.pi * df['Policy_Start_Month_Num'] / 12)
    df['Policy_Start_Month_Cos'] = np.cos(2 * np.pi * df['Policy_Start_Month_Num'] / 12)
    df['Policy_Start_Day_Sin'] = np.sin(2 * np.pi * pd.to_numeric(df['Policy_Start_Day'], errors='coerce') / 31)
    df['Policy_Start_Day_Cos'] = np.cos(2 * np.pi * pd.to_numeric(df['Policy_Start_Day'], errors='coerce') / 31)
    df['Policy_Start_Week_Sin'] = np.sin(2 * np.pi * pd.to_numeric(df['Policy_Start_Week'], errors='coerce') / 52)
    df['Policy_Start_Week_Cos'] = np.cos(2 * np.pi * pd.to_numeric(df['Policy_Start_Week'], errors='coerce') / 52)
    
    # 11. Customer loyalty indicator
    df['Loyalty_Score'] = (df['Years_Without_Claims'] * 0.3 + 
                           df['Previous_Policy_Duration_Months'] * 0.02 - 
                           df['Previous_Claims_Filed'] * 0.5)
    
    # 12. Premium payment behavior
    df['Payment_Flexibility'] = df['Grace_Period_Extensions']
    
    return df


def _transform_with_preprocessor(df, preprocessor):
    """Apply categorical mappings and numeric medians from training."""
    df = df.copy()
    if 'User_ID' in df.columns:
        df = df.drop(['User_ID'], axis=1)
    if 'Purchased_Coverage_Bundle' in df.columns:
        df = df.drop(['Purchased_Coverage_Bundle'], axis=1)

    for col in preprocessor['categorical_columns']:
        values = df[col].astype(str).fillna("__MISSING__")
        mapping = preprocessor['cat_mappings'][col]
        df[col] = values.map(mapping).fillna(-1).astype(int)

    for col in preprocessor['feature_columns']:
        if col not in df.columns:
            df[col] = np.nan

    df = df[preprocessor['feature_columns']]
    df = df.fillna(pd.Series(preprocessor['numeric_medians']))
    return df


def _align_proba(proba, classes, target_classes):
    """Align probability columns to a shared class order."""
    class_index = {cls: idx for idx, cls in enumerate(classes)}
    aligned = np.zeros((proba.shape[0], len(target_classes)))
    for j, cls in enumerate(target_classes):
        aligned[:, j] = proba[:, class_index[cls]]
    return aligned


def load_model():
    model = None
    # ------------------ MODEL LOADING LOGIC ------------------

    # Inside this block, load your trained model bundle.
    model = joblib.load('model.pkl')

    # ------------------ END MODEL LOADING LOGIC ------------------
    return model


def predict(df, model):
    predictions = None
    # ------------------ PREDICTION LOGIC ------------------

    # Inside this block, generate predictions using your model.
    # This function should only contain prediction logic.
    # It must be efficient and run within the time limits.
    #
    # You must return a Pandas DataFrame with exactly two columns:
    #
    #   User_ID,Purchased_Coverage_Bundle
    #   USR_060868,7
    #   USR_060869,2
    #   USR_060870,4
    #   ...
    #
    # --- Example ---
    # import pandas as pd
    # preds = model.predict(df.drop(columns=['User_ID']))
    # predictions = pd.DataFrame({
    #     'User_ID': df['User_ID'],
    #     'Purchased_Coverage_Bundle': preds
    # })
    
    # Store User_ID
    user_ids = df['User_ID'].copy()

    # Ensure feature engineering ran
    if 'Total_Dependents' not in df.columns:
        df = preprocess(df)

    # Load model bundle
    model_bundle = model
    xgb_model = model_bundle['xgb_model']
    mlp_model = model_bundle['mlp_model']
    scaler = model_bundle['scaler']
    preprocessor = model_bundle['preprocessor']
    weight = float(model_bundle.get('ensemble_weight', 0.5))
    class_order = np.array(model_bundle.get('class_order', xgb_model.classes_))

    # Prepare features using saved preprocessing
    X_pred = _transform_with_preprocessor(df, preprocessor)
    X_pred_scaled = scaler.transform(X_pred)

    # Probability predictions
    xgb_proba = xgb_model.predict_proba(X_pred)
    mlp_proba = mlp_model.predict_proba(X_pred_scaled)

    xgb_proba = _align_proba(xgb_proba, xgb_model.classes_, class_order)
    mlp_proba = _align_proba(mlp_proba, mlp_model.classes_, class_order)

    combined = weight * xgb_proba + (1.0 - weight) * mlp_proba
    preds = class_order[np.argmax(combined, axis=1)]

    predictions = pd.DataFrame({
        'User_ID': user_ids,
        'Purchased_Coverage_Bundle': preds.astype(int)
    })

    # ------------------ END PREDICTION LOGIC ------------------
    return predictions


# ----------------------------------------------------------------
# Your code will be called in the following way:
# Note that we will not be using the function defined below.
# ----------------------------------------------------------------


def run(df) -> tuple[float, float, float]:
    from time import time

    # Load the processed data:
    df_processed = preprocess(df)

    # Load the model:
    model = load_model()
    size = get_model_size(model)

    # Get the predictions and time taken:
    start = time.perf_counter()
    predictions = predict(
        df_processed, model
    )  # NOTE: Don't call the `preprocess` function here.

    duration = time.perf_counter() - start
    accuracy = get_model_accuracy(predictions)

    return size, accuracy, duration


# ----------------------------------------------------------------
# Helper functions you should not disturb yourself with.
# ----------------------------------------------------------------


def get_model_size(model):
    pass


def get_model_accuracy(predictions):
    pass
