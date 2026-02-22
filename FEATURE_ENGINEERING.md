# Feature Engineering — Detailed Documentation

> This document explains every engineered feature in our pipeline: what it is, how it's computed, and why it improves predictions.

---

## Overview

We started with **27 raw columns** from the dataset and engineered **22 additional features**, bringing the total to **49 features** after preprocessing. All transformations are implemented in `solution.py::preprocess()`.

Our engineering strategy focused on four principles:

1. **Domain relevance** — features should reflect real insurance decision factors (family needs, risk profile, spending power).
2. **Ratio-based normalization** — raw counts are less informative than per-capita or per-unit ratios.
3. **Cyclical encoding** — temporal features (month, day, week) wrap around; sin/cos encoding prevents artificial ordinal relationships.
4. **Composite scores** — combine multiple signals into a single interpretable metric.

---

## Feature Catalog

### Group 1: Family Composition (Features 1–4)

These features capture the household structure, which is the strongest predictor of bundle type. Families with children gravitate toward comprehensive plans; single adults tend toward basic or auto-only coverage.

| # | Feature | Formula | Type |
|---|---------|---------|------|
| 1 | `Total_Dependents` | `Adult_Dependents + Child_Dependents + Infant_Dependents` | Integer |
| 2 | `Child_Ratio` | `Child_Dependents / (Total_Dependents + 1)` | Float [0, 1) |
| 3 | `Adult_Ratio` | `Adult_Dependents / (Total_Dependents + 1)` | Float [0, 1) |
| 4 | `Infant_Ratio` | `Infant_Dependents / (Total_Dependents + 1)` | Float [0, 1) |

**Why +1 in the denominator?** Prevents division by zero when a customer has no dependents. The +1 is consistent across all ratio features.

**Why ratios instead of raw counts?** A household with 3 adults and 0 children behaves differently from one with 1 adult and 2 children, even if both have 3 total dependents. Ratios capture the _composition_, not just the _size_.

**Impact:** `Total_Dependents` ranks **#2** in global feature importance (14.26%). `Adult_Ratio` ranks **#5** (5.93%).

---

### Group 2: Policy Tenure (Feature 5)

| # | Feature | Formula | Type |
|---|---------|---------|------|
| 5 | `Total_Policy_Duration_Years` | `Previous_Policy_Duration_Months / 12` | Float |

**Why?** Converts months to years for more intuitive scale. Longer tenure often correlates with loyalty discounts and preference for renewal-friendly bundles.

---

### Group 3: Claims Behavior (Features 6–7)

These features quantify how risky a customer is, based on past claim patterns.

| # | Feature | Formula | Type |
|---|---------|---------|------|
| 6 | `Claims_Frequency` | `Previous_Claims_Filed / (Previous_Policy_Duration_Months + 1)` | Float |
| 7 | `Claims_to_YearsWithout_Ratio` | `Previous_Claims_Filed / (Years_Without_Claims + 1)` | Float |

**Why two claim features?**
- `Claims_Frequency` measures the _rate_ of claims per month of coverage — a customer who filed 2 claims in 6 months is very different from one who filed 2 in 60 months.
- `Claims_to_YearsWithout_Ratio` captures the balance between filing and not filing — a high value means frequent claims despite few clean years, signaling high risk.

**Why this matters:** High-risk customers tend to select higher-coverage bundles (Home Premium, Premium Health Life) or are routed there by brokers.

---

### Group 4: Spending Power (Feature 8)

| # | Feature | Formula | Type |
|---|---------|---------|------|
| 8 | `Income_per_Dependent` | `Estimated_Annual_Income / (Total_Dependents + 1)` | Float |

**Why?** $100k income for a single person vs. a family of 5 implies very different purchasing behavior. This feature normalizes income by household size.

**Impact:** `Income_per_Dependent` ranks **#7** in global importance (4.38%), and `Estimated_Annual_Income` (raw) ranks **#1** (14.76%). Together they capture both absolute and relative wealth.

---

### Group 5: Processing Pipeline (Features 9–11)

These features describe how the policy sale progressed through underwriting.

| # | Feature | Formula | Type |
|---|---------|---------|------|
| 9 | `Total_Processing_Days` | `Days_Since_Quote + Underwriting_Processing_Days` | Integer |
| 10 | `Processing_Efficiency` | `Underwriting_Processing_Days / (Days_Since_Quote + 1)` | Float |
| 11 | `Amendments_per_Day` | `Policy_Amendments_Count / (Days_Since_Quote + 1)` | Float |

**Why?**
- `Total_Processing_Days` — longer pipelines may indicate complex bundles requiring more review.
- `Processing_Efficiency` — a ratio near 1.0 means underwriting took as long as the entire quote period, suggesting a complex or unusual policy.
- `Amendments_per_Day` — frequent modifications suggest an indecisive customer or a broker upselling, both of which correlate with specific bundle choices.

---

### Group 6: Vehicle Density (Feature 12)

| # | Feature | Formula | Type |
|---|---------|---------|------|
| 12 | `Vehicles_per_Adult` | `Vehicles_on_Policy / (Adult_Dependents + 1)` | Float |

**Why?** Distinguishes between a single person with 2 cars (likely auto-focused bundle) and a family of 4 with 2 cars (likely family/comprehensive bundle). Raw vehicle count alone doesn't capture this.

---

### Group 7: Risk Flag (Feature 13)

| # | Feature | Formula | Type |
|---|---------|---------|------|
| 13 | `High_Risk_Customer` | `1 if (Previous_Claims_Filed > 2 AND Policy_Cancelled_Post_Purchase == 1) else 0` | Binary |

**Why?** Combines two risk signals into a single binary flag. Customers who both file many claims _and_ cancel policies are a distinct behavioral segment that the model can use as a shortcut.

---

### Group 8: Cyclical Time Encoding (Features 14–20)

Temporal columns (`Policy_Start_Month`, `Policy_Start_Day`, `Policy_Start_Week`) are categorical or ordinal. Encoding them as integers creates artificial ordinality (December=12 would be "far" from January=1). Cyclical sin/cos encoding places them on a circle.

| # | Feature | Formula | Type |
|---|---------|---------|------|
| 14 | `Policy_Start_Month_Num` | Month name → integer (January=1, ..., December=12) | Integer |
| 15 | `Policy_Start_Month_Sin` | `sin(2π × Month_Num / 12)` | Float [-1, 1] |
| 16 | `Policy_Start_Month_Cos` | `cos(2π × Month_Num / 12)` | Float [-1, 1] |
| 17 | `Policy_Start_Day_Sin` | `sin(2π × Day / 31)` | Float [-1, 1] |
| 18 | `Policy_Start_Day_Cos` | `cos(2π × Day / 31)` | Float [-1, 1] |
| 19 | `Policy_Start_Week_Sin` | `sin(2π × Week / 52)` | Float [-1, 1] |
| 20 | `Policy_Start_Week_Cos` | `cos(2π × Week / 52)` | Float [-1, 1] |

**Why sin AND cos?** A single sin or cos function maps two different values to the same number (e.g., sin is the same for month 3 and month 9). Using both sin and cos together gives a unique (x, y) coordinate for every position on the circle.

**Why this matters for insurance:** Policy start timing can reflect open enrollment periods, tax season behavior, or seasonal promotions — all of which influence bundle selection.

---

### Group 9: Loyalty Score (Feature 21)

| # | Feature | Formula | Type |
|---|---------|---------|------|
| 21 | `Loyalty_Score` | `Years_Without_Claims × 0.3 + Previous_Policy_Duration_Months × 0.02 − Previous_Claims_Filed × 0.5` | Float |

**Why a composite score?** Loyalty has three components:
- **Claim-free years** (positive signal, weight 0.3)
- **Policy duration** (positive signal, weight 0.02 per month ≈ 0.24/year)
- **Claims filed** (negative signal, weight −0.5)

The weights were chosen through heuristic seaarch to balance scale differences (years vs. months vs. count). A long-tenured, claim-free customer gets a high score; a short-tenured frequent claimant gets a negative score.

---

### Group 10: Payment Behavior (Feature 22)

| # | Feature | Formula | Type |
|---|---------|---------|------|
| 22 | `Payment_Flexibility` | `Grace_Period_Extensions` (alias) | Integer |

**Why alias?** This is a semantic rename for clarity. Grace period extensions indicate how often a customer needed extra time to pay — a proxy for financial stress or disengagement. We kept it as a named feature so the model and explainability outputs use a business-meaningful label.

---

## Categorical Encoding

After feature engineering, categorical columns are encoded during the `_transform_with_preprocessor()` step:

| Column | Encoding | Values |
|--------|----------|--------|
| `Employment_Status` | Ordinal mapping | Contractor, Employed_FullTime, Self_Employed, Unemployed |
| `Region_Code` | Ordinal mapping | 170+ country codes (ISO alpha-3) |
| `Broker_Agency_Type` | Ordinal mapping | National_Corporate, Urban_Boutique |
| `Deductible_Tier` | Ordinal mapping | Tier_1_High_Ded → Tier_4_Zero_Ded |
| `Acquisition_Channel` | Ordinal mapping | Affiliate_Group, Aggregator_Site, Corporate_Partner, Direct_Website, Local_Broker |
| `Payment_Schedule` | Ordinal mapping | Annual_Upfront, Monthly_EFT, Quarterly_Invoice |
| `Policy_Start_Month` | Ordinal mapping | January–December (also cyclical-encoded above) |

Mappings are built from training data (sorted unique values → integer indices) and stored in the model bundle for consistent application at inference time. Unknown values at prediction time are mapped to −1.

---

## Missing Value Strategy

- **Numeric columns:** filled with training set medians (stored in model bundle).
- **Categorical columns:** filled with `__MISSING__` token before encoding.
- **Derived features:** division by zero is prevented by +1 in all denominators.

---

## Feature Importance Summary

Top 10 features by Random Forest importance:

| Rank | Feature | Importance (%) | Engineered? |
|------|---------|:--------------:|:-----------:|
| 1 | Estimated_Annual_Income | 14.76 | No (raw) |
| 2 | Total_Dependents | 14.26 | **Yes** |
| 3 | Broker_Agency_Type | 9.16 | No (raw) |
| 4 | Broker_ID | 8.65 | No (raw) |
| 5 | Adult_Ratio | 5.93 | **Yes** |
| 6 | Deductible_Tier | 4.84 | No (raw) |
| 7 | Income_per_Dependent | 4.38 | **Yes** |
| 8 | Adult_Dependents | 3.46 | No (raw) |
| 9 | Acquisition_Channel | 2.72 | No (raw) |
| 10 | Policy_Start_Year | 2.60 | No (raw) |

**3 of the top 7 features are engineered**, confirming that feature engineering meaningfully contributes to model performance.
