from __future__ import annotations

from typing import Any


def evaluate_guardrails(record: dict[str, Any], defaults_warnings: list[str]) -> list[str]:
    warnings = list(defaults_warnings)

    numeric_fields = [
        "Policy_Cancelled_Post_Purchase",
        "Policy_Start_Year",
        "Policy_Start_Week",
        "Policy_Start_Day",
        "Grace_Period_Extensions",
        "Previous_Policy_Duration_Months",
        "Adult_Dependents",
        "Child_Dependents",
        "Infant_Dependents",
        "Existing_Policyholder",
        "Previous_Claims_Filed",
        "Years_Without_Claims",
        "Policy_Amendments_Count",
        "Underwriting_Processing_Days",
        "Vehicles_on_Policy",
        "Custom_Riders_Requested",
        "Estimated_Annual_Income",
        "Days_Since_Quote",
    ]

    # 1) Negative values
    for field in numeric_fields:
        value = record.get(field)
        if value is not None and value < 0:
            warnings.append(f"negative value detected: {field}")

    # 2) claims with zero duration
    if (
        record.get("Previous_Policy_Duration_Months", 0) == 0
        and record.get("Previous_Claims_Filed", 0) > 0
    ):
        warnings.append("inconsistent history: claims > 0 while previous duration = 0")

    # 3) high income outlier
    if record.get("Estimated_Annual_Income", 0) > 1_000_000:
        warnings.append("income appears out-of-range")

    # 4) very high quote delay
    if record.get("Days_Since_Quote", 0) > 365:
        warnings.append("days since quote unusually high")

    # 5) very high underwriting delay
    if record.get("Underwriting_Processing_Days", 0) > 120:
        warnings.append("underwriting processing days unusually high")

    # 6) dependents vs adults inconsistency
    if record.get("Adult_Dependents", 0) == 0 and (
        record.get("Child_Dependents", 0) > 0 or record.get("Infant_Dependents", 0) > 0
    ):
        warnings.append("household inconsistency: no adults but child/infant dependents present")

    # 7) vehicle outlier
    if record.get("Vehicles_on_Policy", 0) > 8:
        warnings.append("vehicles_on_policy unusually high")

    # 8) riders outlier
    if record.get("Custom_Riders_Requested", 0) > 10:
        warnings.append("custom_riders_requested unusually high")

    # 9) week/day bounds check
    week = record.get("Policy_Start_Week", 1)
    day = record.get("Policy_Start_Day", 1)
    if week < 1 or week > 53 or day < 1 or day > 31:
        warnings.append("temporal fields out of expected range")

    # 10) weak profile signal
    if (
        record.get("Previous_Claims_Filed", 0) == 0
        and record.get("Years_Without_Claims", 0) == 0
        and record.get("Previous_Policy_Duration_Months", 0) == 0
    ):
        warnings.append("sparse prior-history profile detected")

    return warnings
