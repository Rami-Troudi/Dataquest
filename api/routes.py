from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    FeatureImportanceResponse,
    HealthResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
    SchemaResponse,
    WhatIfRequest,
    WhatIfResponse,
)
from core.constants import BUNDLE_NAME_BY_ID, DEFAULT_VALUES, FIELD_ENUMS, FIELD_TYPES, INPUT_COLUMNS
from core.guardrails import evaluate_guardrails
from core.inference import confidence_payload, predict_proba, predict_topk, preprocess_input
from core.inference_core import preprocess as engineered_preprocess
from core.reasons import generate_reasons, top_feature_importances

router = APIRouter()


def _predict_record(request: Request, record: dict[str, Any], top_k: int) -> PredictResponse:
    artifacts = request.app.state.artifacts
    t0 = time.perf_counter()
    df, default_warnings = preprocess_input(record)
    proba = predict_proba(df, artifacts)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    top_items = predict_topk(proba, artifacts.class_order, k=top_k)
    bundle_id = int(top_items[0]["bundle_id"])
    warnings = evaluate_guardrails(record, default_warnings)
    confidence = confidence_payload(proba, threshold=0.40)

    engineered_row = engineered_preprocess(df).iloc[0]
    feature_rank = top_feature_importances(
        artifacts.model,
        artifacts.feature_list,
        top_n=20,
    )
    reasons = generate_reasons(record, feature_rank, engineered_row)

    request.app.state.requests_total += 1
    request.app.state.total_latency_ms += latency_ms

    return PredictResponse(
        bundle_id=bundle_id,
        top_k=top_items,
        latency_ms=latency_ms,
        warnings=warnings,
        reasons=reasons,
        confidence=confidence["confidence"],
        suggested_fields_to_verify=confidence["suggested_fields_to_verify"],
        model_version=artifacts.model_version,
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=request.app.state.artifacts.model_version,
        build_sha=request.app.state.build_sha,
        uptime_seconds=time.time() - request.app.state.start_ts,
        model_loaded=True,
    )


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    return _predict_record(
        request=request,
        record=payload.record.model_dump(),
        top_k=payload.top_k,
    )


@router.post("/predict-batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    results = [
        _predict_record(request, record.model_dump(), payload.top_k)
        for record in payload.records
    ]
    return BatchPredictResponse(results=results)


@router.post("/whatif", response_model=WhatIfResponse)
def whatif(payload: WhatIfRequest, request: Request) -> WhatIfResponse:
    base = payload.base_record.model_dump()
    scenarios: list[dict[str, Any]] = []
    for idx, patch in enumerate(payload.modifications, start=1):
        trial = {**base, **patch}
        pred = _predict_record(request, trial, top_k=3)
        scenarios.append(
            {
                "scenario_id": idx,
                "modifications": patch,
                "bundle_id": pred.bundle_id,
                "top_k": [item.model_dump() for item in pred.top_k],
                "warnings": pred.warnings,
                "confidence": pred.confidence,
            }
        )
    return WhatIfResponse(scenarios=scenarios)


@router.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    return SchemaResponse(
        fields=FIELD_TYPES,
        required=INPUT_COLUMNS,
        enums=FIELD_ENUMS,
        defaults=DEFAULT_VALUES,
        example=DEFAULT_VALUES,
    )


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
def feature_importance(request: Request) -> FeatureImportanceResponse:
    artifacts = request.app.state.artifacts
    items = top_feature_importances(
        artifacts.model,
        artifacts.feature_list,
        top_n=20,
    )
    return FeatureImportanceResponse(model_version=artifacts.model_version, items=items)


@router.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    return MetadataResponse(bundle_mapping=BUNDLE_NAME_BY_ID)


@router.get("/metrics")
def metrics(request: Request) -> dict[str, Any]:
    avg = 0.0
    if request.app.state.requests_total > 0:
        avg = request.app.state.total_latency_ms / request.app.state.requests_total
    return {
        "requests_total": request.app.state.requests_total,
        "avg_latency_ms": avg,
        "model_load_count": request.app.state.model_load_count,
    }
