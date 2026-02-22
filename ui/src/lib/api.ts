import {
  BatchPredictResponse,
  ExplainResponse,
  GlobalFeatureImportanceResponse,
  HealthResponse,
  MetadataResponse,
  PredictResponse,
  SchemaResponse,
  WhatIfResponse,
} from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Health request failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchSchema(): Promise<SchemaResponse> {
  const res = await fetch(`${API_BASE}/schema`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Schema request failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchMetadata(): Promise<MetadataResponse> {
  const res = await fetch(`${API_BASE}/metadata`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Metadata request failed: ${res.status}`);
  }
  return res.json();
}

export async function predict(
  record: Record<string, unknown>,
  topK = 3
): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ record, top_k: topK }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Predict request failed (${res.status}): ${text}`);
  }
  return res.json();
}

export async function predictBatch(
  records: Record<string, unknown>[],
  topK = 3
): Promise<BatchPredictResponse> {
  const res = await fetch(`${API_BASE}/predict-batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ records, top_k: topK }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Batch request failed (${res.status}): ${text}`);
  }
  return res.json();
}

export async function runWhatIf(
  baseRecord: Record<string, unknown>,
  modifications: Record<string, unknown>[]
): Promise<WhatIfResponse> {
  const res = await fetch(`${API_BASE}/whatif`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ base_record: baseRecord, modifications }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`What-if request failed (${res.status}): ${text}`);
  }
  return res.json();
}

export async function fetchGlobalFeatureImportance(): Promise<GlobalFeatureImportanceResponse> {
  const res = await fetch(`${API_BASE}/model/feature_importance`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Global feature importance request failed: ${res.status}`);
  }
  return res.json();
}

export async function explainOne(
  record: Record<string, unknown>,
  topKReasons = 3
): Promise<ExplainResponse> {
  const res = await fetch(`${API_BASE}/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ record, top_k_reasons: topKReasons }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Explain request failed (${res.status}): ${text}`);
  }
  return res.json();
}
