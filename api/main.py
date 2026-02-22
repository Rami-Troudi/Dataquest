from __future__ import annotations

import os
import time
from pathlib import Path




from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from core.inference import load_artifacts

MODEL_PATH = os.getenv("MODEL_PATH", str(Path("model.pkl").resolve()))
BUILD_SHA = os.getenv("BUILD_SHA", "local-dev")

app = FastAPI(title="DataQuest Broker Assistant API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup() -> None:
    app.state.start_ts = time.time()
    app.state.build_sha = BUILD_SHA
    app.state.requests_total = 0
    app.state.total_latency_ms = 0.0
    app.state.model_load_count = 0
    app.state.artifacts = load_artifacts(MODEL_PATH)
    app.state.model_load_count += 1
    print(f"[startup] model loaded once from {MODEL_PATH}")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error_code": "VALIDATION_ERROR",
            "message": "Request payload failed validation.",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": "Unexpected server error.",
            "details": str(exc),
        },
    )
