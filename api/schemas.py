"""Pydantic request / response models for the Insurance Bundle Recommender API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Bundle label map (shared constant)
# ---------------------------------------------------------------------------
BUNDLE_NAMES = {
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


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class CustomerInput(BaseModel):
    """Single customer feature vector."""

    # Identifiers (optional — auto-generated if absent)
    User_ID: Optional[str] = None

    # Demographics & Financials
    Adult_Dependents: int = Field(0, ge=0, description="Number of adults covered")
    Child_Dependents: float = Field(0, ge=0, description="Number of children covered")
    Infant_Dependents: int = Field(0, ge=0, description="Number of infants covered")
    Estimated_Annual_Income: float = Field(30000, ge=0)
    Employment_Status: str = Field("Employed_FullTime")
    Region_Code: str = Field("USA")

    # Customer History & Risk
    Existing_Policyholder: int = Field(0, ge=0, le=1)
    Previous_Claims_Filed: int = Field(0, ge=0)
    Years_Without_Claims: int = Field(0, ge=0)
    Previous_Policy_Duration_Months: int = Field(0, ge=0)
    Policy_Cancelled_Post_Purchase: int = Field(0, ge=0, le=1)

    # Policy Details
    Deductible_Tier: str = Field("Tier_2_Mid_Ded")
    Payment_Schedule: str = Field("Monthly_EFT")
    Vehicles_on_Policy: int = Field(0, ge=0)
    Custom_Riders_Requested: int = Field(0, ge=0)
    Grace_Period_Extensions: int = Field(0, ge=0)

    # Sales & Underwriting
    Days_Since_Quote: int = Field(30, ge=0)
    Underwriting_Processing_Days: int = Field(0, ge=0)
    Policy_Amendments_Count: int = Field(0, ge=0)
    Acquisition_Channel: str = Field("Direct_Website")
    Broker_Agency_Type: str = Field("Urban_Boutique")
    Broker_ID: Optional[float] = None
    Employer_ID: Optional[float] = None

    # Timeline
    Policy_Start_Year: int = Field(2024, ge=2000)
    Policy_Start_Month: str = Field("January")
    Policy_Start_Week: int = Field(1, ge=1, le=53)
    Policy_Start_Day: int = Field(1, ge=1, le=31)


class PredictRequest(BaseModel):
    """One or more customers to predict."""

    customers: List[CustomerInput]


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------
class ReasonCode(BaseModel):
    rank: int
    feature: str
    value: str


class PredictionResult(BaseModel):
    User_ID: str
    predicted_bundle_id: int
    predicted_bundle_name: str
    confidence: float
    reason_codes: List[ReasonCode] = []


class PredictResponse(BaseModel):
    predictions: List[PredictionResult]
    model_type: str = "RandomForest"
    model_version: str = "v4_rf_only"


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    model_type: str
    num_features: int
    num_classes: int


class FeatureImportance(BaseModel):
    rank: int
    feature: str
    importance: float
    importance_pct: float


class ExplainGlobalResponse(BaseModel):
    features: List[FeatureImportance]
