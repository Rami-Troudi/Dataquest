export type SchemaResponse = {
  fields: Record<string, string>;
  required: string[];
  enums: Record<string, string[]>;
  defaults: Record<string, string | number | null>;
  example: Record<string, string | number | null>;
};

export type HealthResponse = {
  status: string;
  model_version: string;
  build_sha: string;
  uptime_seconds: number;
  model_loaded: boolean;
};

export type TopKItem = {
  bundle_id: number;
  bundle_name: string;
  proba: number;
};

export type PredictResponse = {
  bundle_id: number;
  top_k: TopKItem[];
  latency_ms: number;
  warnings: string[];
  reasons: string[];
  confidence: string;
  suggested_fields_to_verify: string[];
  model_version: string;
};

export type BatchPredictResponse = {
  results: PredictResponse[];
};

export type WhatIfResponse = {
  scenarios: Array<{
    scenario_id: number;
    modifications: Record<string, unknown>;
    bundle_id: number;
    top_k: TopKItem[];
    warnings: string[];
    confidence: string;
  }>;
};

export type MetadataResponse = {
  bundle_mapping: Record<string, string>;
};

export type GlobalFeatureImportanceItem = {
  rank: number;
  feature: string;
  importance: number;
  importance_pct: number;
};

export type GlobalFeatureImportanceResponse = {
  model: string;
  total_features: number;
  features: GlobalFeatureImportanceItem[];
};

export type ExplainReasonCode = {
  rank: number;
  feature: string;
  value: string;
  contribution: string;
};

export type ExplainResponse = {
  User_ID: string;
  prediction: number;
  confidence: number;
  class_probabilities: number[];
  reason_codes: ExplainReasonCode[];
};
