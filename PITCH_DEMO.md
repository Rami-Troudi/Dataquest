# Pitch Demo Script — Insurance Bundle Recommender

> Use this as a guide during the 3-minute pitch presentation.

---

## Slide 1: The Problem (30 sec)

**Title:** *Which insurance bundle should we recommend?*

- Insurance brokers serve thousands of customers with **10 different coverage bundles**.
- Manual recommendation is slow, inconsistent, and subjective.
- **Challenge:** Build a system that predicts the right bundle based on customer profile data.

---

## Slide 2: Our Approach (45 sec)

**Title:** *Data-Driven Recommendation Engine*

- **22 engineered features** from 27 raw columns (family ratios, risk scores, loyalty metrics).
- **Random Forest classifier** — fast, interpretable, small (3.15 MB compressed).
- **Two-phase hyperparameter search:** randomized → neighborhood refinement.
- **Minority oversampling** to handle severe class imbalance (class 2 = 60% of data).

---

## Slide 3: Results (30 sec)

**Title:** *Performance Under Constraints*

| Metric | Value |
|--------|-------|
| Macro F1 | 0.60 |
| Model size | 3.15 MB |
| Inference | ~29 ms |
| Final score proxy | 0.54 |

- Optimized for the **composite score** (F1 × size × latency), not just accuracy.
- Competitive despite using a simple, interpretable model.

---

## Slide 4: Live Demo (60 sec)

**Title:** *Working Inference System*

1. **Open** `http://localhost:8000` → show the UI.
2. **Fill** a sample customer profile (e.g., family with 2 adults, 1 child, income $65k).
3. **Click** "Predict Bundle" → show result: **Family_Comprehensive** at 34% confidence.
4. **Show reason codes:** top features driving the prediction.
5. **Show** `/health` and `/explain/global` endpoints in browser.

> *Demo tip:* Prepare 2-3 pre-filled scenarios that produce different bundles to show variety.

### Demo Scenarios

| Scenario | Key inputs | Expected bundle |
|----------|-----------|-----------------|
| Young renter | Income $22k, 0 deps, 0 vehicles | Basic_Health |
| Family | Income $65k, 2 adults, 1 child | Family_Comprehensive |
| Homeowner | Income $90k, 1 adult, 2 vehicles | Auto_Comprehensive or Home_Premium |

---

## Slide 5: Architecture & Engineering (30 sec)

**Title:** *Production-Ready Stack*

- **FastAPI** backend with validated Pydantic schemas.
- **Single-page frontend** — no framework overhead.
- **Dockerized** deployment with docker-compose.
- **CI pipeline** via GitHub Actions (lint → test → Docker build).
- **Explainability built-in:** global importance + per-prediction reason codes.

---

## Slide 6: Business Value (15 sec)

**Title:** *Why This Matters*

- **Faster onboarding:** instant recommendations vs. manual assessment.
- **Consistent:** same inputs → same outputs, no broker bias.
- **Transparent:** reason codes explain every recommendation.
- **Lightweight:** runs on 1 core, 1 GB RAM — deployable anywhere.

---

## Closing

> "We built a system that not only predicts well, but explains why — and it's ready for production."
