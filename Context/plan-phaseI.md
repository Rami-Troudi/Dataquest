 DataQuest Phase I — Action Plan

**Scope**: 7-hour sprint → 5 submissions max (of 20 lifetime)  
**Goal**: Valid submission + competitive Macro F1 + minimal size/latency penalties  
**Scoring**: `score = macro_f1 × max(0.5, 1 − size_MB/200) × max(0.5, 1 − predict_s/10)`

---

## Constraints Summary

| Constraint | Value | Implication |
|---|---|---|
| RAM | 1 GB | No one-hot on high-cardinality cols |
| CPU | 1 core | Cap tuning at 30 trials |
| Timeout | 120 s total | All heavy work in `preprocess()` |
| ZIP size | 50 MB | Model artifact < 5 MB ideal |
| Submissions | 20 lifetime | Budget 5 for Phase I |
| Timed function | `predict()` only | `preprocess()` and `load_model()` are free |

---

## Pipeline Architecture (from solution.py template)

Three mandatory functions — signatures are **frozen**, do not modify them:

1. **`preprocess(df)`** → DataFrame. All feature engineering here. Never drop `User_ID`.
2. **`load_model()`** → model object. Load serialized artifact from disk.
3. **`predict(df, model)`** → DataFrame with exactly `[User_ID, Purchased_Coverage_Bundle]`. Integers 0–9. All test IDs present. Keep this function minimal — it's the only one timed.

The `run()` function in the template shows the exact call order the judge uses. `preprocess` runs first, then `load_model`, then `predict` is timed.

---

## Model Strategy

- **Primary**: LightGBM (fast, small files, native categorical support, pre-installed)
- **Why not deep learning**: >50 MB models, slow on 1 CPU, no GPU
- **Artifact format**: Single joblib file containing a dict with the fitted model + metadata (feature list, categorical columns) to prevent train/test column mismatches
- **Target size**: 2–5 MB → penalty multiplier ≥ 0.975
- **Target latency**: < 0.3 s → penalty multiplier ≥ 0.97

### Penalty Math (Why Small Wins)

| Size | F1 | Score |
|---|---|---|
| 2 MB, 0.1s | 0.76 | **0.74** |
| 50 MB, 1.0s | 0.81 | 0.55 |

A 2 MB model at F1=0.76 **beats** a 50 MB model at F1=0.81. Optimize for lightweight.

---

## Execution Timeline

### Phase 0 — Setup (H0:00–H0:30)

- Create project folders: `data/`, scripts
- Activate Python 3.10 env
- Load `train.csv` (60,868 × 26) and `test.csv` (15,218 × 25), verify shapes
- Confirm `User_ID` present in both, target in train only
- Create local validation script (ZIP structure, output format, latency, size checks)

**Gate**: Data loads, imports work, validation script ready.

### Phase 1 — EDA + Baseline (H0:30–H1:15) → Submission #1

- Class distribution across 10 bundles (check imbalance severity)
- Feature types: ~8 categorical, ~18 numerical (verify from README column list)
- Null counts per column
- High-cardinality check: `Broker_ID`, `Employer_ID` (use native LGB categorical or frequency encoding — never one-hot)
- Train baseline LightGBM (defaults, `is_unbalance=True`, 100 trees)
- 5-fold stratified CV → record Macro F1 (expect ~0.72)
- Save artifact, create ZIP, validate locally, submit #1

**Gate**: CV F1 ≥ 0.70, model < 3 MB, validation passes.

### Phase 2 — Feature Engineering (H1:15–H3:00) → Submission #2

10 derived features — all use pre-quote information only (zero leakage):

| Feature | Formula | Signal |
|---|---|---|
| Total_Dependents | Adult + Child + Infant | Family size → bundle type |
| Income_Per_Dependent | Income / (1 + Total_Deps) | Purchasing power |
| Claims_Rate | Claims / (1 + Duration_Months) | Risk profile |
| Policy_Complexity | Vehicles × (1 + Riders) | Multi-product affinity |
| Post_Purchase_Activity | Grace_Extensions + Amendments | Engagement level |
| Loyalty_Score | Duration × (1 − Cancelled) | Retention signal |
| Quote_To_UW_Ratio | Days_Since_Quote / (1 + UW_Days) | Decision speed |
| Has_Dependents | Binary: Total_Deps > 0 | Family vs solo |
| Income_Decile | qcut(Income, 10) | Income band |
| Risk_Score | Claims_Rate × Duration / (1 + Vehicles) | Normalized risk |

- Implement in `preprocess()` (applied to both train and test)
- Freeze feature list in saved artifact to guarantee train/test alignment
- CV with features → expect +0.02 F1
- Save, ZIP, validate, submit #2

**Gate**: CV F1 ≥ 0.73, no target leakage, validation passes.

### Phase 3 — Hyperparameter Tuning (H3:00–H4:45) → Submission #3–#4

- RandomizedSearchCV, 30 trials, 5-fold stratified, `scoring='f1_macro'`, `n_jobs=1`
- Search space: `n_estimators` [100–300], `max_depth` [5–9], `learning_rate` [0.01–0.15], `num_leaves` [20–50], `subsample` [0.7–1.0], `colsample_bytree` [0.7–1.0], `reg_alpha/lambda` [0–0.5]
- Optionally test `class_weight='balanced'` vs `is_unbalance=True`
- Record best params + CV score in notes
- Save best model, validate, submit #3 (or #4 if confident)

**Gate**: CV F1 ≥ 0.76, model < 5 MB, latency < 0.3 s.

### Phase 4 — Analysis & Hardening (H4:45–H6:30)

- Per-class F1 breakdown (identify weak bundles, consider class-weight boost)
- Optional: SHAP feature importances for explainability bonus
- Run 10-point hard check:
  1. `User_ID` in preprocess output
  2. All 15,218 test IDs in predictions
  3. `Purchased_Coverage_Bundle` is integer dtype
  4. No nulls anywhere
  5. Values in [0, 9]
  6. No duplicate User_IDs
  7. ZIP is flat, 3 files
  8. Model < 5 MB
  9. Latency < 0.3 s
  10. Artifact keys/structure intact
- Prepare backup model with lighter params (fewer trees, shallower) as safety net

**Gate**: All 10 checks pass. Primary + backup ZIPs ready.

### Phase 5 — Final Submission (H6:30–H7:00) → Submission #4–#5

- Upload primary model (submission #4)
- Hold backup (submission #5) — only use if #4 fails
- Finalize notes: CV scores, params, feature importances, top errors per bundle
- Back up all artifacts locally

**Gate**: Phase I complete. 5 submissions used, 15 reserved for Phase II.

---

## Risk Register (Key Risks Only)

| Risk | Impact | Mitigation |
|---|---|---|
| Drop `User_ID` in preprocess | AUTO-FAIL | Template says "Don't drop User_ID" — respect it |
| Non-integer or out-of-range predictions | AUTO-FAIL | Cast `.astype(int)`, assert range [0,9] |
| Missing test IDs in output | AUTO-FAIL | Return exactly `len(df)` rows from `predict()` |
| OOM on one-hot encoding | Crash | Use native LGB categoricals or frequency encoding |
| Model > 5 MB | Score tanked | Cap trees at 300, depth ≤ 8, use `compress=3` |
| Slow `predict()` | Score tanked | Zero FE in predict — only `.predict()` call |
| Feature mismatch train/test | Crash | Freeze feature list in artifact, reindex strictly |
| Wasted submissions | No recovery | Always validate locally before uploading |
| Target leakage in features | Overfit → bad score | All 10 features use only pre-quote data |
| Stale model in ZIP | Wrong version submitted | Version tag in artifact dict, verify before ZIP |

---

## Deliverables Checklist

- [ ] `submission.zip` — flat, 3 files: `solution.py`, `model.joblib`, `requirements.txt`
- [ ] `solution.py` — implements `preprocess`, `load_model`, `predict` per template
- [ ] `model.joblib` — < 5 MB, contains model + feature list + cat cols
- [ ] `requirements.txt` — exists (can be empty; all deps pre-installed)
- [ ] Local validation passes 100% before every submission
- [ ] `notes.md` — CV scores, best params, feature importances, top errors per class
- [ ] 5 submissions used, 15 reserved

---

## Quick Reference

- **10 classes**: 0=Auto_Comprehensive, 1=Auto_Liability_Basic, 2=Basic_Health, 3=Family_Comprehensive, 4=Health_Dental_Vision, 5=Home_Premium, 6=Home_Standard, 7=Premium_Health_Life, 8=Renter_Basic, 9=Renter_Premium
- **Metric**: Macro F1 (all classes weighted equally)
- **Pre-installed**: numpy, pandas, sklearn, lightgbm, xgboost, catboost, joblib, torch, tensorflow
- **Only `predict()` is timed** — do everything heavy in `preprocess()`
