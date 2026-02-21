"""
DataQuest: Comprehensive Feature Analysis, Correlation Study & Model Building
(Pre-ensemble version)
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import spearmanr, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# =============================================================================
# Step 1: Load and Initial Analysis
# =============================================================================

print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

# Load data
train_df = pd.read_csv('train (1).csv')
test_df = pd.read_csv('test.csv')

print(f"\nTrain shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# =============================================================================
# Step 2: Feature Analysis - Non-null, Data Types, Basic Stats
# =============================================================================

print("\n" + "=" * 80)
print("STEP 2: FEATURE ANALYSIS")
print("=" * 80)

print("\n--- Data Types ---")
print(train_df.dtypes)

print("\n--- Non-Null Count & Missing Values ---")
missing_info = pd.DataFrame({
    'Column': train_df.columns,
    'Non_Null': train_df.count(),
    'Null_Count': train_df.isnull().sum(),
    'Null_Percentage': (train_df.isnull().sum() / len(train_df) * 100).round(2),
    'Data_Type': train_df.dtypes
})
print(missing_info)

print("\n--- Basic Statistics ---")
print(train_df.describe())

print("\n--- Target Distribution ---")
target_dist = train_df['Purchased_Coverage_Bundle'].value_counts().sort_index()
print(target_dist)
print(f"\nTarget balance: \n{(target_dist / len(train_df) * 100).round(2)}")

# Identify categorical and numerical columns
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove identifiers and target
if 'User_ID' in categorical_cols:
    categorical_cols.remove('User_ID')
if 'Purchased_Coverage_Bundle' in numerical_cols:
    numerical_cols.remove('Purchased_Coverage_Bundle')

print(f"\n--- Column Categories ---")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# =============================================================================
# Step 3: Feature Engineering
# =============================================================================

print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 80)


def engineer_features(df):
    """Create new features based on domain knowledge"""
    df = df.copy()

    # 1. Total Family Members
    df['Total_Dependents'] = (
        df['Adult_Dependents'] + df['Child_Dependents'] + df['Infant_Dependents']
    )

    # 2. Family composition ratios
    df['Child_Ratio'] = df['Child_Dependents'] / (df['Total_Dependents'] + 1)
    df['Adult_Ratio'] = df['Adult_Dependents'] / (df['Total_Dependents'] + 1)
    df['Infant_Ratio'] = df['Infant_Dependents'] / (df['Total_Dependents'] + 1)

    # 3. Contract/Policy Length Features
    df['Total_Policy_Duration_Years'] = df['Previous_Policy_Duration_Months'] / 12

    # 4. Claims behavior metrics
    df['Claims_Frequency'] = df['Previous_Claims_Filed'] / (
        df['Previous_Policy_Duration_Months'] + 1
    )
    df['Claims_to_YearsWithout_Ratio'] = df['Previous_Claims_Filed'] / (
        df['Years_Without_Claims'] + 1
    )

    # 5. Average money spent estimate (based on income and deductible tier)
    df['Income_per_Dependent'] = df['Estimated_Annual_Income'] / (
        df['Total_Dependents'] + 1
    )

    # 6. Processing efficiency
    df['Total_Processing_Days'] = df['Days_Since_Quote'] + df['Underwriting_Processing_Days']
    df['Processing_Efficiency'] = df['Underwriting_Processing_Days'] / (
        df['Days_Since_Quote'] + 1
    )

    # 7. Policy modification intensity
    df['Amendments_per_Day'] = df['Policy_Amendments_Count'] / (df['Days_Since_Quote'] + 1)

    # 8. Vehicle coverage ratio
    df['Vehicles_per_Adult'] = df['Vehicles_on_Policy'] / (df['Adult_Dependents'] + 1)

    # 9. Risk indicators
    df['High_Risk_Customer'] = (
        (df['Previous_Claims_Filed'] > 2)
        & (df['Policy_Cancelled_Post_Purchase'] == 1)
    ).astype(int)

    # 10. Time features - cyclical encoding
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }

    month_col = df['Policy_Start_Month'].astype(str).str.strip().map(month_mapping)
    df['Policy_Start_Month_Num'] = pd.to_numeric(month_col, errors='coerce').fillna(6)

    df['Policy_Start_Month_Sin'] = np.sin(2 * np.pi * df['Policy_Start_Month_Num'] / 12)
    df['Policy_Start_Month_Cos'] = np.cos(2 * np.pi * df['Policy_Start_Month_Num'] / 12)
    df['Policy_Start_Day_Sin'] = np.sin(
        2 * np.pi * pd.to_numeric(df['Policy_Start_Day'], errors='coerce') / 31
    )
    df['Policy_Start_Day_Cos'] = np.cos(
        2 * np.pi * pd.to_numeric(df['Policy_Start_Day'], errors='coerce') / 31
    )
    df['Policy_Start_Week_Sin'] = np.sin(
        2 * np.pi * pd.to_numeric(df['Policy_Start_Week'], errors='coerce') / 52
    )
    df['Policy_Start_Week_Cos'] = np.cos(
        2 * np.pi * pd.to_numeric(df['Policy_Start_Week'], errors='coerce') / 52
    )

    # 11. Customer loyalty indicator
    df['Loyalty_Score'] = (
        df['Years_Without_Claims'] * 0.3
        + df['Previous_Policy_Duration_Months'] * 0.02
        - df['Previous_Claims_Filed'] * 0.5
    )

    # 12. Premium payment behavior
    df['Payment_Flexibility'] = df['Grace_Period_Extensions']

    return df


# Apply feature engineering
train_engineered = engineer_features(train_df)
test_engineered = engineer_features(test_df)

print(f"Original features: {train_df.shape[1]}")
print(f"After engineering: {train_engineered.shape[1]}")
print(f"New features added: {train_engineered.shape[1] - train_df.shape[1]}")

# =============================================================================
# Step 4: Correlation Analysis
# =============================================================================

print("\n" + "=" * 80)
print("STEP 4: CORRELATION ANALYSIS")
print("=" * 80)

# Prepare data for correlation
df_for_corr = train_engineered.copy()

# Encode categorical variables for correlation analysis
label_encoders = {}
for col in categorical_cols:
    if col in df_for_corr.columns:
        le = LabelEncoder()
        df_for_corr[col + '_encoded'] = le.fit_transform(df_for_corr[col].astype(str))
        label_encoders[col] = le

# Get all numerical columns (including encoded and engineered)
all_numeric_cols = df_for_corr.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'Purchased_Coverage_Bundle' in all_numeric_cols:
    all_numeric_cols.remove('Purchased_Coverage_Bundle')

# --- PEARSON CORRELATION (Linear Relationships) ---
print("\n--- Pearson Correlation with Target (Top 20) ---")
pearson_corr = df_for_corr[all_numeric_cols + ['Purchased_Coverage_Bundle']].corr()['Purchased_Coverage_Bundle']
pearson_corr = pearson_corr.drop('Purchased_Coverage_Bundle').sort_values(ascending=False, key=abs)
print(pearson_corr.head(20))

# --- SPEARMAN CORRELATION (Monotonic Relationships) ---
print("\n--- Spearman Correlation with Target (Top 20) ---")
spearman_corr = {}
for col in all_numeric_cols:
    corr, _ = spearmanr(df_for_corr[col], df_for_corr['Purchased_Coverage_Bundle'])
    spearman_corr[col] = corr

spearman_corr_series = pd.Series(spearman_corr).sort_values(ascending=False, key=abs)
print(spearman_corr_series.head(20))

# --- CRAMER'S V (Categorical Relationships) ---
print("\n--- Cramer's V for Categorical Variables ---")


def cramers_v(x, y):
    """Calculate Cramer's V statistic for categorical-categorical association"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))


cramers_results = {}
for col in categorical_cols:
    if col in train_engineered.columns:
        v = cramers_v(train_engineered[col], train_engineered['Purchased_Coverage_Bundle'])
        cramers_results[col] = v

cramers_series = pd.Series(cramers_results).sort_values(ascending=False)
print(cramers_series)

# =============================================================================
# Step 5: Feature Selection Based on Correlation
# =============================================================================

print("\n" + "=" * 80)
print("STEP 5: FEATURE SELECTION")
print("=" * 80)

# Select top features from each correlation method
top_pearson = pearson_corr.head(15).index.tolist()
top_spearman = spearman_corr_series.head(15).index.tolist()
top_cramers = cramers_series.head(10).index.tolist()

# Combine all important features
important_features = list(set(top_pearson + top_spearman + top_cramers))
print(f"\nTotal important features selected: {len(important_features)}")
print(f"Features: {important_features}")

# =============================================================================
# Step 6: Prepare Data for Modeling
# =============================================================================

print("\n" + "=" * 80)
print("STEP 6: DATA PREPARATION")
print("=" * 80)


def prepare_data_for_modeling(df, is_train=True):
    """Prepare data with encoding and feature selection"""
    df = df.copy()

    # Store User_ID
    user_ids = df['User_ID']

    # Store target if training
    if is_train:
        target = df['Purchased_Coverage_Bundle']

    # Drop User_ID and target
    df = df.drop(['User_ID'], axis=1)
    if is_train and 'Purchased_Coverage_Bundle' in df.columns:
        df = df.drop(['Purchased_Coverage_Bundle'], axis=1)

    # Encode categorical variables
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Fill any remaining NaN values
    df = df.fillna(df.median())

    if is_train:
        return df, target, user_ids
    return df, user_ids


# Prepare training data
X, y, train_ids = prepare_data_for_modeling(train_engineered, is_train=True)
X_test_final, test_ids = prepare_data_for_modeling(test_engineered, is_train=False)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_test shape: {X_test_final.shape}")

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")

# =============================================================================
# Step 7: Model Training & Evaluation
# =============================================================================

print("\n" + "=" * 80)
print("STEP 7: MODEL TRAINING")
print("=" * 80)

# Try multiple models
models = {}
scores = {}

# --- LightGBM ---
print("\n--- Training LightGBM ---")
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_val)
f1_lgb = f1_score(y_val, y_pred_lgb, average='macro')
print(f"LightGBM Macro F1: {f1_lgb:.4f}")
models['lightgbm'] = lgb_model
scores['lightgbm'] = f1_lgb

# --- XGBoost ---
print("\n--- Training XGBoost ---")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)
f1_xgb = f1_score(y_val, y_pred_xgb, average='macro')
print(f"XGBoost Macro F1: {f1_xgb:.4f}")
models['xgboost'] = xgb_model
scores['xgboost'] = f1_xgb

# --- CatBoost ---
print("\n--- Training CatBoost ---")
cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=7,
    random_state=42,
    verbose=False
)
cat_model.fit(X_train, y_train)
y_pred_cat = cat_model.predict(X_val)
f1_cat = f1_score(y_val, y_pred_cat, average='macro')
print(f"CatBoost Macro F1: {f1_cat:.4f}")
models['catboost'] = cat_model
scores['catboost'] = f1_cat

# --- Random Forest ---
print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)
f1_rf = f1_score(y_val, y_pred_rf, average='macro')
print(f"Random Forest Macro F1: {f1_rf:.4f}")
models['random_forest'] = rf_model
scores['random_forest'] = f1_rf

# Select best model
best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]
best_f1 = scores[best_model_name]

print(f"\n{'=' * 80}")
print(f"BEST MODEL: {best_model_name.upper()} with Macro F1: {best_f1:.4f}")
print(f"{'=' * 80}")

# Detailed evaluation
print("\n--- Classification Report (Best Model) ---")
y_pred_best = best_model.predict(X_val)
print(classification_report(y_val, y_pred_best))

# =============================================================================
# Step 8: Feature Importance
# =============================================================================

print("\n" + "=" * 80)
print("STEP 8: FEATURE IMPORTANCE")
print("=" * 80)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n--- Top 20 Most Important Features ---")
    print(feature_importance.head(20))

# =============================================================================
# Step 9: Train Final Model on Full Data
# =============================================================================

print("\n" + "=" * 80)
print("STEP 9: TRAINING FINAL MODEL ON FULL DATA")
print("=" * 80)

final_model = models[best_model_name].__class__(**models[best_model_name].get_params())
final_model.fit(X, y)

print(f"Final model trained: {best_model_name}")

# =============================================================================
# Step 10: Generate Predictions
# =============================================================================

print("\n" + "=" * 80)
print("STEP 10: GENERATING PREDICTIONS")
print("=" * 80)

# Make predictions
test_predictions = final_model.predict(X_test_final)

# Create submission dataframe
submission = pd.DataFrame({
    'User_ID': test_ids,
    'Purchased_Coverage_Bundle': test_predictions
})

print(f"\nSubmission shape: {submission.shape}")
print(f"\nPrediction distribution:")
print(submission['Purchased_Coverage_Bundle'].value_counts().sort_index())

# =============================================================================
# Step 11: Save Model
# =============================================================================

print("\n" + "=" * 80)
print("STEP 11: SAVING MODEL")
print("=" * 80)

import joblib

# Save the model
joblib.dump(final_model, 'model.pkl')
print("Model saved as 'model.pkl'")

# Save feature engineering and preprocessing info
preprocessing_info = {
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'best_model_name': best_model_name,
    'f1_score': best_f1
}
joblib.dump(preprocessing_info, 'preprocessing_info.pkl')
print("Preprocessing info saved")

print("\n" + "=" * 80)
print("ANALYSIS AND MODEL BUILDING COMPLETE!")
print("=" * 80)
print(f"\nBest Model: {best_model_name}")
print(f"Validation Macro F1: {best_f1:.4f}")
print(f"Model file size: {np.round(os.path.getsize('model.pkl') / (1024 ** 2), 2)} MB")
