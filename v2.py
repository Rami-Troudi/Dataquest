"
DataQuest: Comprehensive Feature Analysis, Correlation Study & Model Building
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, chi2_contingency
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Step 1: Load and Initial Analysis
# =============================================================================

print("="*80)
print("STEP 1: LOADING DATA")
print("="*80)

# Load data
train_df = pd.read_csv('train (1).csv')
test_df = pd.read_csv('test.csv')

print(f"\nTrain shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# =============================================================================
# Step 2: Feature Analysis - Non-null, Data Types, Basic Stats
# =============================================================================

print("\n" + "="*80)
print("STEP 2: FEATURE ANALYSIS")
print("="*80)

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

print("\n" + "="*80)
print("STEP 3: FEATURE ENGINEERING")
print("="*80)

def engineer_features(df):
    """Create new features based on domain knowledge"""
    df = df.copy()
    
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
    # Convert month name to number if it's a string
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October':10, 'November': 11, 'December': 12
    }
    
    # First convert to string, strip, map, and convert to numeric
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

# Apply feature engineering
train_engineered = engineer_features(train_df)
test_engineered = engineer_features(test_df)

print(f"Original features: {train_df.shape[1]}")
print(f"After engineering: {train_engineered.shape[1]}")
print(f"New features added: {train_engineered.shape[1] - train_df.shape[1]}")

# =============================================================================
# Step 4: Correlation Analysis
# =============================================================================

print("\n" + "="*80)
print("STEP 4: CORRELATION ANALYSIS")
print("="*80)

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

# --- CRAMÉR'S V (Categorical Relationships) ---
print("\n--- Cramér's V for Categorical Variables ---")

def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorical-categorical association"""
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

print("\n" + "="*80)
print("STEP 5: FEATURE SELECTION")
print("="*80)

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

print("\n" + "="*80)
print("STEP 6: DATA PREPARATION")
print("="*80)

def fit_preprocessor(df):
    """Fit categorical mappings and numeric medians using training data."""
    df = df.copy()
    target = df['Purchased_Coverage_Bundle']
    df = df.drop(['User_ID', 'Purchased_Coverage_Bundle'], axis=1)

    categorical = df.select_dtypes(include=['object']).columns.tolist()
    cat_mappings = {}

    for col in categorical:
        values = df[col].astype(str).fillna("__MISSING__")
        mapping = {v: i for i, v in enumerate(sorted(values.unique()))}
        cat_mappings[col] = mapping
        df[col] = values.map(mapping).astype(int)

    numeric_medians = df.median(numeric_only=True)
    df = df.fillna(numeric_medians)
    feature_columns = df.columns.tolist()

    preprocessor = {
        'categorical_columns': categorical,
        'cat_mappings': cat_mappings,
        'numeric_medians': numeric_medians.to_dict(),
        'feature_columns': feature_columns
    }

    return preprocessor, df, target


def transform_with_preprocessor(df, preprocessor):
    """Apply categorical mappings and numeric medians to new data."""
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


def oversample_minority(X, y, min_count=500, seed=42):
    """Simple oversampling to boost rare classes."""
    rng = np.random.RandomState(seed)
    counts = y.value_counts()
    X_parts = [X]
    y_parts = [y]

    for cls, cnt in counts.items():
        if cnt < min_count:
            needed = min_count - cnt
            sample_idx = y[y == cls].sample(needed, replace=True, random_state=rng).index
            X_parts.append(X.loc[sample_idx])
            y_parts.append(y.loc[sample_idx])

    X_out = pd.concat(X_parts, axis=0).reset_index(drop=True)
    y_out = pd.concat(y_parts, axis=0).reset_index(drop=True)
    return X_out, y_out


def align_proba(proba, classes, target_classes):
    """Align probability columns to target class order."""
    class_index = {cls: idx for idx, cls in enumerate(classes)}
    aligned = np.zeros((proba.shape[0], len(target_classes)))
    for j, cls in enumerate(target_classes):
        aligned[:, j] = proba[:, class_index[cls]]
    return aligned


# Prepare training data
preprocessor, X_full, y_full = fit_preprocessor(train_engineered)
X_test_final = transform_with_preprocessor(test_engineered, preprocessor)

print(f"X shape: {X_full.shape}")
print(f"y shape: {y_full.shape}")
print(f"X_test shape: {X_test_final.shape}")

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# Oversample rare classes in training only
X_train_os, y_train_os = oversample_minority(X_train, y_train, min_count=500)

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")
print(f"Oversampled Train: {X_train_os.shape}")

# =============================================================================
# Step 7: Model Training & Evaluation
# =============================================================================

print("\n" + "="*80)
print("STEP 7: MODEL TRAINING")
print("="*80)

print("\n--- Training XGBoost ---")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    objective='multi:softprob',
    eval_metric='mlogloss'
)
xgb_model.fit(X_train_os, y_train_os)
xgb_pred = xgb_model.predict(X_val)
f1_xgb = f1_score(y_val, xgb_pred, average='macro')
print(f"XGBoost Macro F1: {f1_xgb:.4f}")

print("\n--- Training MLP (Neural Net) ---")
scaler = StandardScaler()
X_train_os_scaled = scaler.fit_transform(X_train_os)
X_val_scaled = scaler.transform(X_val)

mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    learning_rate_init=0.001,
    max_iter=250,
    early_stopping=True,
    random_state=42
)
mlp_model.fit(X_train_os_scaled, y_train_os)
mlp_pred = mlp_model.predict(X_val_scaled)
f1_mlp = f1_score(y_val, mlp_pred, average='macro')
print(f"MLP Macro F1: {f1_mlp:.4f}")

print("\n--- Ensemble Weight Search (XGBoost + MLP) ---")
xgb_proba = xgb_model.predict_proba(X_val)
mlp_proba = mlp_model.predict_proba(X_val_scaled)

target_classes = xgb_model.classes_
mlp_proba = align_proba(mlp_proba, mlp_model.classes_, target_classes)

best_weight = 0.5
best_f1 = -1.0
for w in np.linspace(0.0, 1.0, 11):
    combined = w * xgb_proba + (1.0 - w) * mlp_proba
    pred = target_classes[np.argmax(combined, axis=1)]
    f1 = f1_score(y_val, pred, average='macro')
    if f1 > best_f1:
        best_f1 = f1
        best_weight = w

print(f"Best ensemble weight (XGB): {best_weight:.2f}")
print(f"Best ensemble Macro F1: {best_f1:.4f}")

print("\n--- Classification Report (Ensemble) ---")
combined = best_weight * xgb_proba + (1.0 - best_weight) * mlp_proba
ensemble_pred = target_classes[np.argmax(combined, axis=1)]
print(classification_report(y_val, ensemble_pred))

# =============================================================================
# Step 8: Train Final Models on Full Data
# =============================================================================

print("\n" + "="*80)
print("STEP 8: TRAINING FINAL MODELS ON FULL DATA")
print("="*80)

X_full_os, y_full_os = oversample_minority(X_full, y_full, min_count=500)

xgb_final = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    objective='multi:softprob',
    eval_metric='mlogloss'
)
xgb_final.fit(X_full_os, y_full_os)

scaler_final = StandardScaler()
X_full_os_scaled = scaler_final.fit_transform(X_full_os)

mlp_final = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    learning_rate_init=0.001,
    max_iter=250,
    early_stopping=True,
    random_state=42
)
mlp_final.fit(X_full_os_scaled, y_full_os)

print("Final models trained")

# =============================================================================
# Step 9: Generate Predictions
# =============================================================================

print("\n" + "="*80)
print("STEP 9: GENERATING PREDICTIONS")
print("="*80)

X_test_scaled = scaler_final.transform(X_test_final)
xgb_test_proba = xgb_final.predict_proba(X_test_final)
mlp_test_proba = mlp_final.predict_proba(X_test_scaled)
mlp_test_proba = align_proba(mlp_test_proba, mlp_final.classes_, xgb_final.classes_)

combined_test = best_weight * xgb_test_proba + (1.0 - best_weight) * mlp_test_proba
test_predictions = xgb_final.classes_[np.argmax(combined_test, axis=1)]

submission = pd.DataFrame({
    'User_ID': test_engineered['User_ID'],
    'Purchased_Coverage_Bundle': test_predictions
})

print(f"\nSubmission shape: {submission.shape}")
print(f"\nPrediction distribution:")
print(submission['Purchased_Coverage_Bundle'].value_counts().sort_index())

# =============================================================================
# Step 10: Save Model
# =============================================================================

print("\n" + "="*80)
print("STEP 10: SAVING MODEL")
print("="*80)

import joblib

model_bundle = {
    'xgb_model': xgb_final,
    'mlp_model': mlp_final,
    'scaler': scaler_final,
    'preprocessor': preprocessor,
    'ensemble_weight': best_weight,
    'class_order': xgb_final.classes_.tolist()
}

joblib.dump(model_bundle, 'model.pkl')
print("Model bundle saved as 'model.pkl'")

print("\n" + "="*80)
print("ANALYSIS AND MODEL BUILDING COMPLETE!")
print("="*80)
print(f"\nEnsemble Macro F1 (val): {best_f1:.4f}")
print(f"Model file size: {np.round(os.path.getsize('model.pkl') / (1024**2), 2)} MB")
