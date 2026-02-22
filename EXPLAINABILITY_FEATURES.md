# 🔍 Explainability Features Documentation

**Model Transparency & Interpretability for Insurance Coverage Prediction**

---

## 📋 Overview

This document describes the explainability features available in the MLOps Insurance Model API. Explainability helps stakeholders understand **WHY** the model makes specific predictions.

### Why Explainability Matters

- **Trust**: Understand model decisions
- **Compliance**: Meet regulatory requirements (e.g., GDPR, FCRA)
- **Debugging**: Identify model biases or errors
- **Business Insights**: Learn which factors drive customer behavior

---

## 🎯 Available Explainability Features

The codebase includes two main explainability approaches:

### 1. **Global Feature Importance** 
*Model-wide understanding*

Shows which features are most important across ALL predictions.

**Use Cases:**
- Feature selection for model improvement
- Understanding overall model behavior
- Communication with business stakeholders
- Identifying data collection priorities

**Method:** RandomForest's built-in feature importance (Gini impurity reduction)

**Output:**
```json
{
  "features": [
    {
      "rank": 1,
      "feature": "Policy_Start_Year",
      "importance": 0.183,
      "importance_pct": 18.3
    },
    {
      "rank": 2,
      "feature": "Estimated_Annual_Income",
      "importance": 0.145,
      "importance_pct": 14.5
    },
    ...
  ],
  "total_features": 49
}
```

### 2. **Local Reason Codes (Per-Prediction Explanations)**
*Individual prediction understanding*

Explains WHY a specific customer received a particular prediction by showing the top contributing features.

**Use Cases:**
- Customer service explanations
- Personalized marketing insights
- Regulatory compliance (adverse action notices)
- Model debugging for edge cases

**Method:** 
- Calculates deviation from baseline (median values)
- Weights by global feature importance
- Identifies top-K most influential features for each prediction

**Output:**
```json
{
  "User_ID": "TEST_001",
  "prediction": 3,
  "confidence": 0.847,
  "reason_codes": [
    {
      "rank": 1,
      "feature": "Policy_Start_Year=2015",
      "contribution": "high"
    },
    {
      "rank": 2,
      "feature": "Broker_Agency_Type=Urban_Boutique",
      "contribution": "medium"
    },
    {
      "rank": 3,
      "feature": "Estimated_Annual_Income=24493.85",
      "contribution": "medium"
    }
  ]
}
```

---

## 📊 Technical Details

### Implementation (`explainability.py`)

#### Function 1: `build_global_feature_importance(rf_model, feature_columns)`

**Purpose:** Extract and rank feature importance from trained RandomForest model

**Algorithm:**
1. Extract `feature_importances_` from RF model
2. Handle edge cases (missing/invalid importances)
3. Sort features by importance descending
4. Calculate percentage contribution
5. Add ranking

**Robustness:**
- Handles missing `feature_importances_` attribute
- Validates feature count matches
- Normalizes importances to percentages
- Safe handling of edge cases (zero sum, NaN values)

**Returns:** Pandas DataFrame with columns:
- `rank`: 1-based ranking
- `feature`: Feature name
- `importance`: Raw importance score
- `importance_pct`: Percentage contribution

---

#### Function 2: `build_local_reason_codes(X_encoded, user_ids, rf_model, preprocessor, top_k=3)`

**Purpose:** Generate per-prediction reason codes showing top contributing features

**Algorithm:**
1. **Baseline Calculation**: Compute median values for numeric features, mode for categorical
2. **Deviation Measurement**: Calculate absolute deviation from baseline for each feature
3. **Weighted Scoring**: Multiply deviations by global feature importance
4. **Top-K Selection**: Identify top-K features per prediction using `argpartition`
5. **Category Decoding**: Convert encoded categorical values back to human-readable labels
6. **Prediction Enrichment**: Add predictions and confidence scores

**Key Features:**
- **Efficient**: Uses NumPy `argpartition` instead of full sort
- **Human-Readable**: Decodes categorical features (e.g., `0` → `"Urban_Boutique"`)
- **Configurable**: Adjustable `top_k` parameter (default: 3)
- **Safe**: Handles missing values and edge cases

**Parameters:**
- `X_encoded`: Preprocessed feature matrix (DataFrame or array)
- `user_ids`: List of User IDs for tracking
- `rf_model`: Trained RandomForest model
- `preprocessor`: Preprocessor dict with metadata
- `top_k`: Number of top features to return (default: 3)

**Returns:** Pandas DataFrame with columns:
- `User_ID`: Customer identifier
- `Purchased_Coverage_Bundle`: Predicted class
- `Predicted_Probability`: Confidence score
- `Reason_1`, `Reason_2`, `Reason_3`: Top contributing features (formatted)

**Example Reason Code Format:**
- Numeric: `Estimated_Annual_Income=24493.85`
- Categorical: `Broker_Agency_Type=Urban_Boutique`
- Integer: `Policy_Start_Year=2015`
- Missing: `feature_name=MISSING`

---

## 🔌 API Integration Status

### Current Status: ⚠️ **NOT INTEGRATED**

The explainability functions exist in `explainability.py` but are **not yet exposed** through the REST API.

### Required Integration Steps:

1. ✅ **Functions Exist**: `explainability.py` contains working code
2. ❌ **API Endpoints Missing**: No `/explain` or `/feature_importance` endpoints in `app_v3.py`
3. ❌ **Model Server Integration**: `model_loader.py` doesn't call explainability functions

---

## 🎯 Proposed API Endpoints

### Endpoint 1: `/model/feature_importance`

**Method:** `GET`

**Description:** Get global feature importance for current model

**Response:**
```json
{
  "model": "insurance_rf:1.0.0",
  "total_features": 49,
  "features": [
    {
      "rank": 1,
      "feature": "Policy_Start_Year",
      "importance": 0.183,
      "importance_pct": 18.3
    },
    ...
  ]
}
```

**Use Case:** Dashboard visualization, model documentation

---

### Endpoint 2: `/explain` (POST)

**Method:** `POST`

**Description:** Get prediction with reason codes for single record

**Request Body:**
```json
{
  "User_ID": "TEST_001",
  "Policy_Start_Year": 2015,
  "Estimated_Annual_Income": 24493.85,
  ...
}
```

**Response:**
```json
{
  "User_ID": "TEST_001",
  "prediction": 3,
  "confidence": 0.847,
  "class_probabilities": [0.02, 0.05, 0.06, 0.847, 0.023],
  "reason_codes": [
    {
      "rank": 1,
      "feature": "Policy_Start_Year",
      "value": "2015",
      "contribution": "high"
    },
    {
      "rank": 2,
      "feature": "Broker_Agency_Type",
      "value": "Urban_Boutique",
      "contribution": "medium"
    },
    {
      "rank": 3,
      "feature": "Estimated_Annual_Income",
      "value": "24493.85",
      "contribution": "medium"
    }
  ]
}
```

**Use Case:** Customer-facing explanations, adverse action notices

---

### Endpoint 3: `/explain_csv` (POST)

**Method:** `POST`

**Description:** Get predictions with reason codes for batch (CSV upload)

**Request:** Multipart form-data with CSV file

**Response:**
```json
{
  "file": "test.csv",
  "count": 100,
  "predictions": [
    {
      "User_ID": "TEST_001",
      "prediction": 3,
      "confidence": 0.847,
      "reason_1": "Policy_Start_Year=2015",
      "reason_2": "Broker_Agency_Type=Urban_Boutique",
      "reason_3": "Estimated_Annual_Income=24493.85"
    },
    ...
  ]
}
```

**Use Case:** Bulk analysis, compliance documentation

---

## 🧠 Explainability Techniques Comparison

| Technique | Scope | Speed | Precision | Complexity |
|-----------|-------|-------|-----------|------------|
| **Feature Importance** | Global | Fast | Medium | Low |
| **Reason Codes** | Local | Fast | High | Medium |
| SHAP Values | Both | Slow | Very High | High |
| LIME | Local | Slow | High | High |
| Partial Dependence | Global | Medium | High | Medium |

**Current Implementation:** Feature Importance + Reason Codes
- **Pros:** Fast, model-agnostic (for tree models), easy to understand
- **Cons:** Less precise than SHAP, assumes feature independence

---

## 📈 Business Value

### For Customers
- **Transparency**: Understand why they got specific coverage recommendations
- **Trust**: Clear explanations build confidence in automated decisions
- **Actionability**: Know what changes could lead to different coverage options

### For Business
- **Compliance**: Meet regulatory requirements for model transparency
- **Marketing**: Personalize messaging based on key drivers
- **Risk Management**: Identify suspicious patterns or model biases
- **Product Development**: Understand which features drive customer choices

### For Data Science Team
- **Model Validation**: Verify model learns expected patterns
- **Feature Engineering**: Identify important vs. redundant features
- **Debugging**: Diagnose edge cases and errors
- **Stakeholder Communication**: Explain model behavior to non-technical audiences

---

## 🚀 Future Enhancements

### Phase 1: Current Implementation (Ready)
- ✅ Global feature importance
- ✅ Local reason codes (deviation-based)

### Phase 2: Advanced Explainability
- [ ] SHAP (SHapley Additive exPlanations) integration
- [ ] LIME (Local Interpretable Model-agnostic Explanations)
- [ ] Partial Dependence Plots
- [ ] Individual Conditional Expectation (ICE) plots

### Phase 3: Interactive Explanations
- [ ] Web-based dashboard for exploring explanations
- [ ] What-if analysis ("How would prediction change if income increased 10%?")
- [ ] Feature contribution waterfall charts
- [ ] Global surrogate models for interpretability

### Phase 4: Production Monitoring
- [ ] Log explanations for audit trail
- [ ] Track feature importance drift over time
- [ ] Alert on unexpected reason code patterns
- [ ] A/B testing with explanation quality metrics

---

## 🧪 Testing Explainability Features

### Test 1: Global Feature Importance

```python
from explainability import build_global_feature_importance
import joblib

# Load model
bundle = joblib.load('model.pkl')
rf_model = bundle['rf_model']
feature_columns = bundle['preprocessor']['feature_columns']

# Get feature importance
importance_df = build_global_feature_importance(rf_model, feature_columns)
print(importance_df.head(10))
```

**Expected:** Top 10 features ranked by importance

---

### Test 2: Local Reason Codes

```python
from explainability import build_local_reason_codes
import pandas as pd

# Load test data
test_df = pd.read_csv('test.csv')
user_ids = test_df['User_ID']

# Preprocess (assuming preprocessor available)
X_encoded = preprocessor.transform(test_df)

# Get reason codes
reasons_df = build_local_reason_codes(
    X_encoded, 
    user_ids, 
    rf_model, 
    bundle['preprocessor'],
    top_k=3
)
print(reasons_df.head())
```

**Expected:** DataFrame with predictions and top 3 reason codes per user

---

## 📝 Regulatory Compliance

### GDPR (General Data Protection Regulation)
- **Article 22**: Right to explanation for automated decisions
- **Implementation**: Local reason codes provide per-decision explanations

### FCRA (Fair Credit Reporting Act)
- **Requirement**: Adverse action notices must include principal reasons
- **Implementation**: Top reason codes serve as principal reasons

### EU AI Act
- **High-Risk AI Systems**: Must be transparent and explainable
- **Implementation**: Global + local explainability provides full transparency

---

## 🔗 References

### Papers
- Breiman, L. (2001). "Random Forests." Machine Learning.
- Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions" (SHAP)
- Ribeiro et al. (2016). "Why Should I Trust You?" (LIME)

### Libraries
- scikit-learn: `feature_importances_` attribute
- SHAP: https://github.com/slundberg/shap
- LIME: https://github.com/marcotcr/lime

### Best Practices
- Molnar, C. (2022). "Interpretable Machine Learning"
- Google's "Explainable AI Best Practices"

---

## 📞 Contact & Support

For questions about explainability features:
- **Technical**: Check `explainability.py` source code
- **Business**: Contact Data Science team
- **Regulatory**: Consult Legal/Compliance team

---

**Document Version:** 1.0  
**Last Updated:** February 22, 2026  
**Status:** Features implemented but NOT integrated into API (pending integration)
