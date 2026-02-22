from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CustomerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    User_ID: str
    Policy_Cancelled_Post_Purchase: int
    Policy_Start_Year: int
    Policy_Start_Week: int
    Policy_Start_Day: int
    Grace_Period_Extensions: int
    Previous_Policy_Duration_Months: int
    Adult_Dependents: int
    Child_Dependents: float
    Infant_Dependents: int
    Region_Code: str
    Existing_Policyholder: int
    Previous_Claims_Filed: int
    Years_Without_Claims: int
    Policy_Amendments_Count: int
    Broker_ID: float | None = None
    Employer_ID: float | None = None
    Underwriting_Processing_Days: int
    Vehicles_on_Policy: int
    Custom_Riders_Requested: int
    Broker_Agency_Type: str
    Deductible_Tier: str
    Acquisition_Channel: str
    Payment_Schedule: str
    Employment_Status: str
    Estimated_Annual_Income: float
    Days_Since_Quote: int
    Policy_Start_Month: str


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record: CustomerRecord
    top_k: int = Field(default=3, ge=1, le=10)


class TopKItem(BaseModel):
    bundle_id: int
    bundle_name: str
    proba: float


class PredictResponse(BaseModel):
    bundle_id: int
    top_k: list[TopKItem]
    latency_ms: float
    warnings: list[str]
    reasons: list[str]
    confidence: str
    suggested_fields_to_verify: list[str]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    build_sha: str
    uptime_seconds: float
    model_loaded: bool


class FeatureImportanceResponse(BaseModel):
    model_version: str
    items: list[dict[str, Any]]


class MetadataResponse(BaseModel):
    bundle_mapping: dict[int, str]


class SchemaResponse(BaseModel):
    fields: dict[str, str]
    required: list[str]
    enums: dict[str, list[str]]
    defaults: dict[str, Any]
    example: dict[str, Any]


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: Any


class WhatIfRequest(BaseModel):
    base_record: CustomerRecord
    modifications: list[dict[str, Any]]


class WhatIfResponse(BaseModel):
    scenarios: list[dict[str, Any]]


class BatchPredictRequest(BaseModel):
    records: list[CustomerRecord]
    top_k: int = Field(default=3, ge=1, le=10)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
