"use client";

import { FormEvent, useMemo, useState } from "react";

import { predict } from "@/lib/api";
import { PredictResponse } from "@/lib/types";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type Props = {
  requiredFields: string[];
  sampleRecord: Record<string, string | number | null>;
  onResult: (result: PredictResponse) => void;
};

const NUMBER_FIELDS = new Set([
  "Policy_Cancelled_Post_Purchase",
  "Policy_Start_Year",
  "Policy_Start_Week",
  "Policy_Start_Day",
  "Grace_Period_Extensions",
  "Previous_Policy_Duration_Months",
  "Adult_Dependents",
  "Child_Dependents",
  "Infant_Dependents",
  "Existing_Policyholder",
  "Previous_Claims_Filed",
  "Years_Without_Claims",
  "Policy_Amendments_Count",
  "Broker_ID",
  "Employer_ID",
  "Underwriting_Processing_Days",
  "Vehicles_on_Policy",
  "Custom_Riders_Requested",
  "Estimated_Annual_Income",
  "Days_Since_Quote",
]);

const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

export default function PredictForm({
  requiredFields,
  sampleRecord,
  onResult,
}: Props) {
  const [formData, setFormData] = useState<Record<string, string>>(() => {
    const initial: Record<string, string> = {};
    for (const key of requiredFields) {
      const value = sampleRecord[key];
      initial[key] = value === null || value === undefined ? "" : String(value);
    }
    return initial;
  });

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fields = useMemo(() => requiredFields, [requiredFields]);

  const toPayload = (): Record<string, unknown> => {
    const out: Record<string, unknown> = {};
    for (const key of fields) {
      const value = formData[key] ?? "";
      if (NUMBER_FIELDS.has(key)) {
        out[key] = value === "" ? null : Number(value);
      } else {
        out[key] = value;
      }
    }
    return out;
  };

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const response = await predict(toPayload());
      onResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setSubmitting(false);
    }
  };

  const reloadSample = () => {
    const reset: Record<string, string> = {};
    for (const key of fields) {
      const value = sampleRecord[key];
      reset[key] = value === null || value === undefined ? "" : String(value);
    }
    setFormData(reset);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Customer Input</CardTitle>
        <CardDescription>
          Complete all required fields, then submit to get a bundle recommendation.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={onSubmit} className="space-y-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {fields.map((field) => {
              const value = formData[field] ?? "";
              if (field === "Policy_Start_Month") {
                return (
                  <div key={field} className="space-y-2">
                    <Label htmlFor={field}>{field}</Label>
                    <Select
                      value={value}
                      onValueChange={(val) =>
                        setFormData((prev) => ({ ...prev, [field]: val }))
                      }
                    >
                      <SelectTrigger id={field}>
                        <SelectValue placeholder="Select month" />
                      </SelectTrigger>
                      <SelectContent>
                        {MONTHS.map((month) => (
                          <SelectItem key={month} value={month}>
                            {month}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                );
              }

              return (
                <div key={field} className="space-y-2">
                  <Label htmlFor={field}>{field}</Label>
                  <Input
                    id={field}
                    type={NUMBER_FIELDS.has(field) ? "number" : "text"}
                    step={
                      field === "Estimated_Annual_Income" ||
                      field === "Child_Dependents"
                        ? "any"
                        : "1"
                    }
                    value={value}
                    onChange={(e) =>
                      setFormData((prev) => ({ ...prev, [field]: e.target.value }))
                    }
                    required={field !== "Broker_ID" && field !== "Employer_ID"}
                  />
                </div>
              );
            })}
          </div>

          <div className="flex flex-wrap gap-3">
            <Button type="submit" disabled={submitting}>
              {submitting ? "Predicting..." : "Predict"}
            </Button>
            <Button type="button" variant="outline" onClick={reloadSample}>
              Reload Sample
            </Button>
          </div>

          {error ? <p className="text-sm text-red-600">{error}</p> : null}
        </form>
      </CardContent>
    </Card>
  );
}
