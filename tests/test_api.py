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


def test_health_ok() -> None:
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert "build_sha" in body


def test_predict_schema_and_range() -> None:
    with TestClient(app) as client:
        res = client.post("/predict", json={"record": SAMPLE_RECORD, "top_k": 3})
        assert res.status_code == 200
        body = res.json()
        assert 0 <= body["bundle_id"] <= 9
        assert len(body["top_k"]) == 3
        assert "warnings" in body
        assert "model_version" in body


def test_predict_is_deterministic() -> None:
    with TestClient(app) as client:
        p1 = client.post("/predict", json={"record": SAMPLE_RECORD, "top_k": 3}).json()[
            "bundle_id"
        ]
        p2 = client.post("/predict", json={"record": SAMPLE_RECORD, "top_k": 3}).json()[
            "bundle_id"
        ]
        p3 = client.post("/predict", json={"record": SAMPLE_RECORD, "top_k": 3}).json()[
            "bundle_id"
        ]
        assert p1 == p2 == p3


def test_validation_error_shape() -> None:
    invalid = dict(SAMPLE_RECORD)
    invalid["Policy_Start_Week"] = "bad"
    with TestClient(app) as client:
        res = client.post("/predict", json={"record": invalid})
        assert res.status_code == 422
        body = res.json()
        assert body["error_code"] == "VALIDATION_ERROR"
