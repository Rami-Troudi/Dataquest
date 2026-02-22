from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

SAMPLE_RECORD = {
    "User_ID": "USR_060868",
    "Policy_Cancelled_Post_Purchase": 0,
    "Policy_Start_Year": 2015,
    "Policy_Start_Week": 43,
    "Policy_Start_Day": 20,
    "Grace_Period_Extensions": 1,
    "Previous_Policy_Duration_Months": 5,
    "Adult_Dependents": 2,
    "Child_Dependents": 0.0,
    "Infant_Dependents": 0,
    "Region_Code": "DEU",
    "Existing_Policyholder": 0,
    "Previous_Claims_Filed": 0,
    "Years_Without_Claims": 0,
    "Policy_Amendments_Count": 0,
    "Broker_ID": 16.0,
    "Employer_ID": None,
    "Underwriting_Processing_Days": 0,
    "Vehicles_on_Policy": 0,
    "Custom_Riders_Requested": 0,
    "Broker_Agency_Type": "Urban_Boutique",
    "Deductible_Tier": "Tier_1_High_Ded",
    "Acquisition_Channel": "Local_Broker",
    "Payment_Schedule": "Monthly_EFT",
    "Employment_Status": "Employed_FullTime",
    "Estimated_Annual_Income": 24493.85,
    "Days_Since_Quote": 87,
    "Policy_Start_Month": "February",
}


def test_model_feature_importance_endpoint() -> None:
    with TestClient(app) as client:
        res = client.get("/model/feature_importance")
        assert res.status_code == 200
        body = res.json()
        assert "features" in body
        assert body["total_features"] >= 1


def test_explain_endpoint() -> None:
    with TestClient(app) as client:
        res = client.post(
            "/explain",
            json={"record": SAMPLE_RECORD, "top_k_reasons": 3},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["User_ID"] == SAMPLE_RECORD["User_ID"]
        assert isinstance(body["reason_codes"], list)
        assert len(body["reason_codes"]) == 3


def test_explain_csv_json_batch_endpoint() -> None:
    with TestClient(app) as client:
        res = client.post(
            "/explain_csv",
            json={"records": [SAMPLE_RECORD], "top_k_reasons": 3},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["count"] == 1
        assert "reason_1" in body["predictions"][0]


def test_explain_csv_upload_endpoint() -> None:
    csv_content = (
        "User_ID,Policy_Cancelled_Post_Purchase,Policy_Start_Year,Policy_Start_Week,"
        "Policy_Start_Day,Grace_Period_Extensions,Previous_Policy_Duration_Months,"
        "Adult_Dependents,Child_Dependents,Infant_Dependents,Region_Code,"
        "Existing_Policyholder,Previous_Claims_Filed,Years_Without_Claims,"
        "Policy_Amendments_Count,Broker_ID,Employer_ID,Underwriting_Processing_Days,"
        "Vehicles_on_Policy,Custom_Riders_Requested,Broker_Agency_Type,"
        "Deductible_Tier,Acquisition_Channel,Payment_Schedule,Employment_Status,"
        "Estimated_Annual_Income,Days_Since_Quote,Policy_Start_Month\n"
        "USR_060868,0,2015,43,20,1,5,2,0.0,0,DEU,0,0,0,0,16.0,,0,0,0,"
        "Urban_Boutique,Tier_1_High_Ded,Local_Broker,Monthly_EFT,Employed_FullTime,"
        "24493.85,87,February\n"
    )
    with TestClient(app) as client:
        res = client.post(
            "/explain_csv_upload",
            files={"file": ("input.csv", csv_content, "text/csv")},
            data={"top_k_reasons": "3"},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["count"] == 1
        assert body["file"] == "input.csv"
