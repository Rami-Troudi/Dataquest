import { HealthResponse, PredictResponse, SchemaResponse } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL;

if (!API_BASE) {
  throw new Error("NEXT_PUBLIC_API_BASE_URL is required.");
}

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

export async function predict(
  record: Record<string, unknown>
): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ record }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Predict request failed (${res.status}): ${text}`);
  }
  return res.json();
}
