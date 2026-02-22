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
import lightgbm as lgb


def preprocess(df):
    # Implement any preprocessing steps required for your model here.
    # Return a Pandas DataFrame of the data
    #
    # Note: Don't drop the 'User_ID' column here.
    # It will be used in the predict function to return the final predictions.

    _MONTH_MAP = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }

    d = df.copy()

    # Household features
    d["Total_Dependents"] = (
        d["Adult_Dependents"] + d["Child_Dependents"] + d["Infant_Dependents"]
    )
    d["Has_Dependents"] = (d["Total_Dependents"] > 0).astype(np.int8)
    d["Has_Child"]      = (d["Child_Dependents"]  > 0).astype(np.int8)
    d["Has_Infant"]     = (d["Infant_Dependents"] > 0).astype(np.int8)

    # Risk and loyalty features
    d["Claims_Rate"]         = d["Previous_Claims_Filed"] / (d["Previous_Policy_Duration_Months"] + 1)
    d["Risk_Score"]          = d["Previous_Claims_Filed"] / (d["Years_Without_Claims"] + 1)
    d["Loyalty_Score"]       = (
        d["Years_Without_Claims"] * 0.3
        + d["Previous_Policy_Duration_Months"] * 0.02
        - d["Previous_Claims_Filed"] * 0.5
    )
    d["Claimfree_vs_Claims"] = d["Years_Without_Claims"] / (1 + d["Previous_Claims_Filed"])
    d["Existing_x_Claims"]   = d["Existing_Policyholder"] * d["Previous_Claims_Filed"]

    # Policy features
    d["Post_Purchase_Activity"] = (
        d["Policy_Cancelled_Post_Purchase"] * d["Policy_Amendments_Count"]
    )
    d["Policy_Complexity"] = d["Custom_Riders_Requested"] + d["Vehicles_on_Policy"]
    d["Quote_To_UW_Ratio"] = d["Days_Since_Quote"] / (d["Underwriting_Processing_Days"] + 1)
    d["PostPurchase_v2"]   = d["Grace_Period_Extensions"] + d["Policy_Amendments_Count"]
    d["Grace_per_Month"]   = d["Grace_Period_Extensions"] / (1 + d["Previous_Policy_Duration_Months"])

    # Temporal features (month name -> int -> sin/cos)
    month_num = (
        d["Policy_Start_Month"].astype(str).str.strip().map(_MONTH_MAP).fillna(6).astype(int)
    )
    d["Month_Sin"] = np.sin(2 * np.pi * month_num / 12)
    d["Month_Cos"] = np.cos(2 * np.pi * month_num / 12)

    # Presence flags
    d["Has_Broker"]   = d["Broker_ID"].notna().astype(np.int8)
    d["Has_Employer"] = d["Employer_ID"].notna().astype(np.int8)

    # Tenure bucket using np.digitize (robust, no pandas cut issues)
    # Training bin edges: 0-1, 1-2, 2-3, 3-4, 4+ months
    tenure = d["Previous_Policy_Duration_Months"].fillna(0).values
    d["Tenure_Bucket"] = np.digitize(tenure, bins=[1, 2, 3, 4]).astype(np.int8)

    # Underwriting friction
    d["Underwriting_Friction"] = np.log1p(
        d["Underwriting_Processing_Days"].fillna(0).clip(lower=0)
    )

    # Deductible x risk interaction
    ded_map = {"Low": 0, "Medium": 1, "High": 2}
    ded_ord = d["Deductible_Tier"].astype(str).map(ded_map).fillna(1)
    d["Deductible_x_Risk"] = d["Claims_Rate"] * ded_ord

    return d


def load_model():
    model = None
    # ------------------ MODEL LOADING LOGIC ------------------

    model = joblib.load("model.joblib")

    # ------------------ END MODEL LOADING LOGIC ------------------
    return model


def predict(df, model):
    predictions = None
    # ------------------ PREDICTION LOGIC ------------------

    user_ids     = df["User_ID"].values
    fe_maps      = model["fe_maps"]
    label_enc    = model["label_enc"]
    feature_list = model["feature_list"]
    cat_cols     = model["cat_cols"]
    booster      = model["model"]
    thresholds   = model.get("thresholds", np.ones(10))

    d = df.copy()

    # Frequency encoding for high-cardinality ID columns
    for col in ["Broker_ID", "Employer_ID", "Region_Code"]:
        freq_key = col + "_freq"
        d[col + "_Freq"] = d[col].map(fe_maps[freq_key]).fillna(0).astype(np.float32)

    # Integer label-encoding for categorical columns
    for col in cat_cols:
        if col in d.columns:
            d[col] = d[col].astype(str).map(label_enc[col]).fillna(-1).astype(np.int32)

    # Align to training feature order (fill any missing with 0)
    for c in feature_list:
        if c not in d.columns:
            d[c] = 0.0

    X = d[feature_list].values.astype(np.float32)

    # Run inference
    proba = booster.predict(X)

    # Apply per-class threshold weights
    preds = (proba * thresholds).argmax(axis=1)

    predictions = pd.DataFrame({
        "User_ID": user_ids,
        "Purchased_Coverage_Bundle": preds.astype(int),
    })

    # ------------------ END PREDICTION LOGIC ------------------
    return predictions


# ----------------------------------------------------------------
# Your code will be called in the following way:
# Note that we will not be using the function defined below.
# ----------------------------------------------------------------


def run(df) -> tuple[float, float, float]:
    from time import perf_counter

    # Load the processed data:
    df_processed = preprocess(df)

    # Load the model:
    model = load_model()
    size = get_model_size(model)

    # Get the predictions and time taken:
    start = perf_counter()
    predictions = predict(
        df_processed, model
    )  # NOTE: Don't call the `preprocess` function here.

    duration = perf_counter() - start
    accuracy = get_model_accuracy(predictions)

    return size, accuracy, duration


# ----------------------------------------------------------------
# Helper functions you should not disturb yourself with.
# ----------------------------------------------------------------


def get_model_size(model):
    pass


def get_model_accuracy(predictions):
    pass
