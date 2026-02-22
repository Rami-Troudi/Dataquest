from __future__ import annotations

ID_COL = "User_ID"
TARGET_COL = "Purchased_Coverage_Bundle"

MODEL_REQUIRED_KEYS = {
    "rf_model",
    "preprocessor",
    "class_order",
}

INPUT_COLUMNS = [
    "User_ID",
    "Policy_Cancelled_Post_Purchase",
    "Policy_Start_Year",
    "Policy_Start_Week",
    "Policy_Start_Day",
    "Grace_Period_Extensions",
    "Previous_Policy_Duration_Months",
    "Adult_Dependents",
    "Child_Dependents",
    "Infant_Dependents",
    "Region_Code",
    "Existing_Policyholder",
    "Previous_Claims_Filed",
    "Years_Without_Claims",
    "Policy_Amendments_Count",
    "Broker_ID",
    "Employer_ID",
    "Underwriting_Processing_Days",
    "Vehicles_on_Policy",
    "Custom_Riders_Requested",
    "Broker_Agency_Type",
    "Deductible_Tier",
    "Acquisition_Channel",
    "Payment_Schedule",
    "Employment_Status",
    "Estimated_Annual_Income",
    "Days_Since_Quote",
    "Policy_Start_Month",
]

MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

BUNDLE_NAME_BY_ID = {
    0: "Auto_Comprehensive",
    1: "Auto_Liability_Basic",
    2: "Basic_Health",
    3: "Family_Comprehensive",
    4: "Health_Dental_Vision",
    5: "Home_Premium",
    6: "Home_Standard",
    7: "Premium_Health_Life",
    8: "Renter_Basic",
    9: "Renter_Premium",
}

DEFAULT_VALUES = {
    "User_ID": "USR_DEMO",
    "Policy_Cancelled_Post_Purchase": 0,
    "Policy_Start_Year": 2019,
    "Policy_Start_Week": 26,
    "Policy_Start_Day": 15,
    "Grace_Period_Extensions": 0,
    "Previous_Policy_Duration_Months": 12,
    "Adult_Dependents": 1,
    "Child_Dependents": 0.0,
    "Infant_Dependents": 0,
    "Region_Code": "DEU",
    "Existing_Policyholder": 0,
    "Previous_Claims_Filed": 0,
    "Years_Without_Claims": 1,
    "Policy_Amendments_Count": 0,
    "Broker_ID": None,
    "Employer_ID": None,
    "Underwriting_Processing_Days": 3,
    "Vehicles_on_Policy": 1,
    "Custom_Riders_Requested": 0,
    "Broker_Agency_Type": "Urban_Boutique",
    "Deductible_Tier": "Tier_2_Mid_Ded",
    "Acquisition_Channel": "Local_Broker",
    "Payment_Schedule": "Monthly_EFT",
    "Employment_Status": "Employed_FullTime",
    "Estimated_Annual_Income": 45000.0,
    "Days_Since_Quote": 20,
    "Policy_Start_Month": "June",
}

FIELD_ENUMS = {
    "Policy_Start_Month": list(MONTH_MAP.keys()),
}

FIELD_TYPES = {
    "User_ID": "str",
    "Policy_Cancelled_Post_Purchase": "int",
    "Policy_Start_Year": "int",
    "Policy_Start_Week": "int",
    "Policy_Start_Day": "int",
    "Grace_Period_Extensions": "int",
    "Previous_Policy_Duration_Months": "int",
    "Adult_Dependents": "int",
    "Child_Dependents": "float",
    "Infant_Dependents": "int",
    "Region_Code": "str",
    "Existing_Policyholder": "int",
    "Previous_Claims_Filed": "int",
    "Years_Without_Claims": "int",
    "Policy_Amendments_Count": "int",
    "Broker_ID": "float|null",
    "Employer_ID": "float|null",
    "Underwriting_Processing_Days": "int",
    "Vehicles_on_Policy": "int",
    "Custom_Riders_Requested": "int",
    "Broker_Agency_Type": "str",
    "Deductible_Tier": "str",
    "Acquisition_Channel": "str",
    "Payment_Schedule": "str",
    "Employment_Status": "str",
    "Estimated_Annual_Income": "float",
    "Days_Since_Quote": "int",
    "Policy_Start_Month": "str",
}

SUGGESTED_VERIFY_FIELDS = [
    "Estimated_Annual_Income",
    "Previous_Claims_Filed",
    "Previous_Policy_Duration_Months",
    "Deductible_Tier",
    "Payment_Schedule",
]
