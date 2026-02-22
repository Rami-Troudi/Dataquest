import { PredictResponse } from "@/lib/types";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

type Props = {
  result: PredictResponse | null;
};

export default function ResultCard({ result }: Props) {
  if (!result) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Prediction</CardTitle>
          <CardDescription>
            Submit a customer record to get a recommendation.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Prediction</CardTitle>
        <CardDescription>
          Bundle recommendation for input record.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        <p>
          <strong>Bundle:</strong> {result.bundle_id}
        </p>
        <p>
          <strong>Latency:</strong> {result.latency_ms.toFixed(2)} ms
        </p>
        <p>
          <strong>Model version:</strong> {result.model_version}
        </p>
        <div>
          <h3 className="font-semibold">Top-3 Probabilities</h3>
          <ul className="mt-2 space-y-1 text-sm">
            {result.top_k.map((item) => (
              <li key={item.bundle_id}>
                {item.bundle_id} - {item.bundle_name}:{" "}
                {(item.proba * 100).toFixed(2)}%
              </li>
            ))}
          </ul>
        </div>
        {result.warnings && result.warnings.length > 0 && (
          <div>
            <h3 className="font-semibold">Warnings</h3>
            <ul className="mt-2 space-y-1 text-sm">
              {result.warnings.map((w, idx) => (
                <li key={idx}>{w}</li>
              ))}
            </ul>
          </div>
        )}
        {result.reasons && result.reasons.length > 0 && (
          <div>
            <h3 className="font-semibold">Reasons</h3>
            <ul className="mt-2 space-y-1 text-sm">
              {result.reasons.map((r, idx) => (
                <li key={idx}>{r}</li>
              ))}
            </ul>
          </div>
        )}
        {result.suggested_fields_to_verify && result.suggested_fields_to_verify.length > 0 && (
          <div>
            <h3 className="font-semibold">Suggested fields to verify</h3>
            <ul className="mt-2 space-y-1 text-sm">
              {result.suggested_fields_to_verify.map((f, idx) => (
                <li key={idx}>{f}</li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
