import { useEffect, useRef, useState } from "react";
import { Activity, LoaderCircle, RotateCcw } from "lucide-react";
import { api } from "../api";
import {
  predictResponseSchema,
  type PredictResponse,
  type Sample,
} from "../contracts";

function makeRequestId() {
  return typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `ui-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

export function ModelReadout({ sample }: { sample: Sample | null }) {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const request = useRef<AbortController | null>(null);

  useEffect(() => {
    request.current?.abort();
    setResult(null);
    setError("");
    setLoading(false);
    return () => request.current?.abort();
  }, [sample?.sample_id]);

  async function analyse() {
    if (!sample || loading) return;
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    const requestId = makeRequestId();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const prediction = await api("/predict", predictResponseSchema, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_id: sample.sample_id,
          model_name: "tslm",
          request_id: requestId,
        }),
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      if (
        prediction.request_id !== requestId ||
        prediction.sample_id !== sample.sample_id ||
        prediction.input_sha256 !== sample.input_sha256
      )
        throw new Error(
          "The model response does not match the selected recording.",
        );
      setResult(prediction);
    } catch (reason) {
      if (!controller.signal.aborted)
        setError(
          reason instanceof Error ? reason.message : "Model inference failed.",
        );
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }

  const score = result?.class_scores?.leak;
  return (
    <div className="model-block">
      <dl className="model-readout" aria-live="polite">
        <div>
          <dt>Model</dt>
          <dd>
            {result ? result.model_version : "TSLM · V2 RELIABILITY"}{" "}
            <span className={result ? "live-dot" : "pending-dot"} />
          </dd>
        </div>
        <div>
          <dt>Decision</dt>
          <dd className={result ? "prediction-value" : "pending-value"}>
            {loading
              ? "RUNNING"
              : result
                ? result.prediction === "leak"
                  ? "LEAK-ASSOCIATED"
                  : "NON-LEAK"
                : "NOT RUN"}
          </dd>
        </div>
        <div>
          <dt>Leak score</dt>
          <dd>{score == null ? "—" : score.toFixed(3)}</dd>
        </div>
        <div>
          <dt>Latency</dt>
          <dd>{result ? `${Math.round(result.latency_ms)} ms` : "—"}</dd>
        </div>
      </dl>
      {result && (
        <div className="model-result">
          <p>{result.description ?? "No model description returned."}</p>
          <small>
            Raw relative score · not calibrated · threshold {result.threshold}
          </small>
          {result.fallback_used && (
            <small>DSP template fallback used; raw text was not displayed.</small>
          )}
          {result.warnings.map((warning) => (
            <small key={warning}>{warning}</small>
          ))}
        </div>
      )}
      {error && (
        <p className="model-error" role="alert">
          {error}
        </p>
      )}
      <button
        className="model-run"
        disabled={!sample || loading}
        onClick={() => void analyse()}
      >
        {loading ? (
          <LoaderCircle size={14} className="spin" />
        ) : error ? (
          <RotateCcw size={14} />
        ) : (
          <Activity size={14} />
        )}
        {loading ? "Running TSLM…" : error ? "Retry TSLM V2" : "Run TSLM V2"}
      </button>
      <p className="model-caveat">
        The score is a model output, not a field probability or an inspection
        recommendation.
      </p>
    </div>
  );
}
