"use client";

import { useEffect, useState } from "react";

import PredictForm from "@/components/predict-form";
import ResultCard from "@/components/result-card";
import { fetchHealth, fetchSchema } from "@/lib/api";
import { HealthResponse, PredictResponse, SchemaResponse } from "@/lib/types";

export default function HomePage() {
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const init = async () => {
      try {
        const [schemaRes, healthRes] = await Promise.all([
          fetchSchema(),
          fetchHealth(),
        ]);
        setSchema(schemaRes);
        setHealth(healthRes);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      }
    };
    void init();
  }, []);

  if (error) {
    return (
      <main className="container py-10">
        <h1 className="text-3xl font-bold tracking-tight">
          DataQuest Insurance Recommender
        </h1>
        <p className="mt-4 text-red-600">{error}</p>
      </main>
    );
  }

  if (!schema || !health) {
    return (
      <main className="container py-10">
        <h1 className="text-3xl font-bold tracking-tight">
          DataQuest Insurance Recommender
        </h1>
        <p className="mt-4 text-muted-foreground">Loading API metadata...</p>
      </main>
    );
  }

  return (
    <main className="container py-10">
      <h1 className="text-3xl font-bold tracking-tight">
        DataQuest Insurance Recommender
      </h1>
      <p className="mt-2 text-sm text-muted-foreground">
        API status: <strong>{health.status}</strong> | model version:{" "}
        <strong>{health.model_version}</strong>
      </p>
      <div
        className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-[1.35fr_1fr]"
      >
        <PredictForm
          requiredFields={schema.required}
          sampleRecord={schema.example || schema.defaults}
          onResult={setResult}
        />
        <ResultCard result={result} />
      </div>
    </main>
  );
}
