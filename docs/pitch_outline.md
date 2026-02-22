# Pitch Outline (5–7 slides)

## Slide 1 — Problem
- Insurance bundle recommendation is complex and multi-factor.
- Objective: predict final bundle selection with production-ready access.

## Slide 2 — Solution
- RandomForest-based inference system.
- Fullstack delivery: FastAPI + Next.js + shared feature core.

## Slide 3 — Feature Engineering
- Household, risk, policy-process, and temporal features.
- Strong focus on inference parity between training and runtime.

## Slide 4 — System Architecture
- UI -> API -> feature pipeline -> model -> result.
- Deterministic outputs and strict schema validation.

## Slide 5 — Demo
- Live prediction from input form.
- Top-3 probability display and health status.

## Slide 6 — Results and Quality
- Reproducible runtime stack.
- Tests for API schema and prediction determinism.

## Slide 7 — Roadmap
- Explainability endpoints.
- CI/CD hardening and deployment.
- Continuous retraining pipeline.
