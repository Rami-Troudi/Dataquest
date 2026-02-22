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
