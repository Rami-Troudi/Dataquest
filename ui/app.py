from __future__ import annotations

import json
from typing import Any

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

from presets import family_vehicles, high_risk_profile, low_risk_single

API_BASE_URL = st.secrets.get("API_BASE_URL", "http://127.0.0.1:8000")

GROUPS = {
    "Household": ["Adult_Dependents", "Child_Dependents", "Infant_Dependents"],
    "Financial / Demographics": [
        "Estimated_Annual_Income",
        "Employment_Status",
        "Region_Code",
    ],
    "Risk": [
        "Existing_Policyholder",
        "Previous_Claims_Filed",
        "Years_Without_Claims",
        "Previous_Policy_Duration_Months",
        "Policy_Cancelled_Post_Purchase",
    ],
    "Policy": [
        "Deductible_Tier",
        "Payment_Schedule",
        "Vehicles_on_Policy",
        "Custom_Riders_Requested",
        "Grace_Period_Extensions",
    ],
    "Temporal": [
        "Policy_Start_Year",
        "Policy_Start_Month",
        "Policy_Start_Week",
        "Policy_Start_Day",
        "Days_Since_Quote",
        "Underwriting_Processing_Days",
        "Policy_Amendments_Count",
    ],
    "Broker": [
        "Broker_ID",
        "Employer_ID",
        "Broker_Agency_Type",
        "Acquisition_Channel",
    ],
}


def api_get(path: str) -> dict[str, Any]:
    resp = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=45)
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except Exception:
            detail = {"message": resp.text}
        raise RuntimeError(json.dumps(detail, ensure_ascii=False))
    return resp.json()


def copy_button(label: str, content: str, key: str) -> None:
    safe = content.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    components.html(
        f"""
        <button onclick="navigator.clipboard.writeText(`{safe}`)"
        style="padding:8px 10px;border-radius:8px;border:1px solid #2563eb;background:#2563eb;color:white;cursor:pointer;">
        {label}
        </button>
        """,
        height=44,
        key=key,
    )


def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "record" not in st.session_state:
        st.session_state.record = {}


def render_record_editor(schema: dict[str, Any]) -> dict[str, Any]:
    defaults = schema["defaults"]
    enums = schema["enums"]
    fields = schema["required"]
    record = dict(st.session_state.record or defaults)

    st.markdown("### Predict")
    for group_name, group_fields in GROUPS.items():
        st.markdown(f"#### {group_name}")
        cols = st.columns(2)
        for idx, field in enumerate(group_fields):
            if field not in fields:
                continue
            with cols[idx % 2]:
                default_val = record.get(field, defaults.get(field))
                if field in enums:
                    options = enums[field]
                    selected = st.selectbox(
                        field,
                        options=options,
                        index=options.index(default_val) if default_val in options else 0,
                        key=f"in_{field}",
                    )
                    record[field] = selected
                elif isinstance(defaults.get(field), float):
                    value = st.number_input(
                        field,
                        value=float(default_val or 0.0),
                        key=f"in_{field}",
                    )
                    record[field] = float(value)
                elif isinstance(defaults.get(field), int):
                    value = st.number_input(
                        field,
                        value=int(default_val or 0),
                        step=1,
                        key=f"in_{field}",
                    )
                    record[field] = int(value)
                else:
                    value = st.text_input(field, value=str(default_val or ""), key=f"in_{field}")
                    record[field] = value

    st.markdown("#### Identity")
    record["User_ID"] = st.text_input("User_ID", value=str(record.get("User_ID", "USR_UI")), key="in_User_ID")
    st.session_state.record = record
    return record


def render_predict_result(result: dict[str, Any], metadata: dict[str, Any]) -> None:
    mapping = metadata["bundle_mapping"]
    bundle_id = int(result["bundle_id"])
    bundle_name = mapping.get(str(bundle_id), mapping.get(bundle_id, str(bundle_id)))

    st.success(f"Recommended Bundle: {bundle_id} - {bundle_name}")
    st.write(f"Latency: {result['latency_ms']:.2f} ms")
    st.write(f"Model Version: {result['model_version']}")
    st.write(f"Confidence: {result['confidence']}")

    st.markdown("##### Top-K")
    top_df = pd.DataFrame(result["top_k"])
    st.dataframe(top_df, use_container_width=True)

    st.markdown("##### Warnings")
    if result["warnings"]:
        for w in result["warnings"]:
            st.warning(w)
    else:
        st.info("No warnings")

    st.markdown("##### Why this recommendation?")
    for reason in result["reasons"]:
        st.markdown(f"- {reason}")

    st.markdown("##### Suggested fields to verify")
    if result["suggested_fields_to_verify"]:
        st.write(", ".join(result["suggested_fields_to_verify"]))
    else:
        st.write("None")


def main() -> None:
    st.set_page_config(page_title="DataQuest Broker Assistant", layout="wide")
    init_state()

    st.title("DataQuest Broker Assistant")

    try:
        health = api_get("/health")
        schema = api_get("/schema")
        metadata = api_get("/metadata")
    except Exception as exc:
        st.error(f"API connection failed: {exc}")
        st.stop()

    st.caption(
        f"Status: {health['status']} | version: {health['model_version']} | "
        f"build: {health['build_sha']} | uptime: {health['uptime_seconds']:.1f}s"
    )

    c1, c2, c3 = st.columns(3)
    if c1.button("Preset: Low risk single"):
        st.session_state.record = low_risk_single()
    if c2.button("Preset: Family + vehicles"):
        st.session_state.record = family_vehicles()
    if c3.button("Preset: High risk"):
        st.session_state.record = high_risk_profile()

    tab_predict, tab_whatif, tab_batch = st.tabs(["Predict", "What-if", "Batch"])

    with tab_predict:
        record = render_record_editor(schema)
        if st.button("Run Prediction", type="primary"):
            response = api_post("/predict", {"record": record, "top_k": 3})
            render_predict_result(response, metadata)

            st.session_state.history.insert(
                0,
                {
                    "user_id": record["User_ID"],
                    "bundle_id": response["bundle_id"],
                    "confidence": response["confidence"],
                },
            )
            st.session_state.history = st.session_state.history[:5]

            request_json = json.dumps({"record": record, "top_k": 3}, indent=2)
            response_json = json.dumps(response, indent=2)
            st.markdown("##### Request JSON")
            st.code(request_json, language="json")
            copy_button("Copy request JSON", request_json, "copy_req")
            st.markdown("##### Response JSON")
            st.code(response_json, language="json")
            copy_button("Copy response JSON", response_json, "copy_res")

        st.markdown("##### Session history (last 5)")
        if st.session_state.history:
            st.table(pd.DataFrame(st.session_state.history))
        else:
            st.info("No predictions yet.")

    with tab_whatif:
        st.markdown("### What-if Simulator")
        base = dict(st.session_state.record or schema["defaults"])
        income_delta = st.slider("Income delta", -20000, 20000, 0, step=1000)
        claims_delta = st.slider("Claims delta", -2, 5, 0, step=1)
        vehicles_delta = st.slider("Vehicles delta", -1, 3, 0, step=1)
        if st.button("Run What-if"):
            modifications = [
                {"Estimated_Annual_Income": max(0.0, float(base["Estimated_Annual_Income"]) + income_delta)},
                {"Previous_Claims_Filed": max(0, int(base["Previous_Claims_Filed"]) + claims_delta)},
                {"Vehicles_on_Policy": max(0, int(base["Vehicles_on_Policy"]) + vehicles_delta)},
            ]
            result = api_post(
                "/whatif",
                {"base_record": base, "modifications": modifications},
            )
            st.dataframe(pd.DataFrame(result["scenarios"]), use_container_width=True)

    with tab_batch:
        st.markdown("### Batch Inference")
        uploaded = st.file_uploader("Upload CSV with model input columns", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            records = df.to_dict(orient="records")
            if st.button("Run Batch"):
                result = api_post("/predict-batch", {"records": records, "top_k": 3})
                out_rows = []
                for idx, row in enumerate(result["results"]):
                    out_rows.append(
                        {
                            "row_id": idx,
                            "bundle_id": row["bundle_id"],
                            "confidence": row["confidence"],
                            "latency_ms": row["latency_ms"],
                        }
                    )
                out_df = pd.DataFrame(out_rows)
                st.dataframe(out_df, use_container_width=True)
                st.download_button(
                    "Download results CSV",
                    out_df.to_csv(index=False),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
