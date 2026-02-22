from __future__ import annotations

from copy import deepcopy

from core.constants import DEFAULT_VALUES


def low_risk_single() -> dict:
    p = deepcopy(DEFAULT_VALUES)
    p.update(
        {
            "User_ID": "USR_PRESET_LOW",
            "Adult_Dependents": 1,
            "Child_Dependents": 0.0,
            "Infant_Dependents": 0,
            "Previous_Claims_Filed": 0,
            "Years_Without_Claims": 8,
            "Vehicles_on_Policy": 1,
            "Estimated_Annual_Income": 52000.0,
        }
    )
    return p


def family_vehicles() -> dict:
    p = deepcopy(DEFAULT_VALUES)
    p.update(
        {
            "User_ID": "USR_PRESET_FAMILY",
            "Adult_Dependents": 2,
            "Child_Dependents": 2.0,
            "Infant_Dependents": 1,
            "Vehicles_on_Policy": 2,
            "Custom_Riders_Requested": 2,
            "Estimated_Annual_Income": 90000.0,
            "Previous_Policy_Duration_Months": 24,
        }
    )
    return p


def high_risk_profile() -> dict:
    p = deepcopy(DEFAULT_VALUES)
    p.update(
        {
            "User_ID": "USR_PRESET_HIGH",
            "Previous_Claims_Filed": 5,
            "Policy_Cancelled_Post_Purchase": 1,
            "Underwriting_Processing_Days": 45,
            "Policy_Amendments_Count": 6,
            "Days_Since_Quote": 180,
            "Vehicles_on_Policy": 3,
            "Estimated_Annual_Income": 28000.0,
        }
    )
    return p
