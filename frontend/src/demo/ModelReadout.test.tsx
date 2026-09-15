// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "../api";
import type { Sample } from "../contracts";
import { ModelReadout } from "./ModelReadout";

vi.mock("../api", () => ({ api: vi.fn() }));

const sample: Sample = {
  sample_id: "a".repeat(24),
  input_sha256: "b".repeat(64),
  duration_seconds: 1,
  sample_rate_hz: 8000,
  channels: 1,
  source: "test",
  execution_mode: "development_fixture",
  warnings: [],
  label_available: false,
};

const response = {
  schema_version: "0.1",
  request_id: "request-test",
  sample_id: sample.sample_id,
  input_sha256: sample.input_sha256,
  model_input_sha256: "c".repeat(64),
  model_name: "tslm",
  model_version: "icham-v2-test",
  preprocessing_version: "prep-v2-test",
  prediction: "leak",
  class_scores: { leak: 0.625, no_leak: 0.375 },
  score_type: "raw",
  abstained: false,
  abstention_reason: null,
  observations: [],
  description: "LEAK; greatest mean spectral energy: 0-1000 Hz",
  latency_ms: 12.4,
  warnings: ["Score brut non calibré."],
  execution_mode: "live",
  decision_version: "decision-v2-test",
  decision_artifact_sha256: "d".repeat(64),
  threshold: 0.6,
  calibration: "none",
  description_source: "dsp_template_fallback",
  fallback_used: true,
  fallback_reasons: ["invalid_generated_format"],
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("does not request a prediction before the user starts it", () => {
  render(<ModelReadout sample={sample} />);
  expect(screen.getByText("Decision").nextElementSibling).toHaveTextContent(
    "NOT RUN",
  );
  expect(api).not.toHaveBeenCalled();
});

it("renders a validated live response and its reliability provenance", async () => {
  vi.stubGlobal("crypto", { randomUUID: () => "request-test" });
  vi.mocked(api).mockResolvedValue(response);
  render(<ModelReadout sample={sample} />);
  fireEvent.click(screen.getByRole("button", { name: "Run TSLM V2" }));
  expect(await screen.findByText("LEAK-ASSOCIATED")).toBeInTheDocument();
  expect(screen.getByText("0.625")).toBeInTheDocument();
  expect(screen.getByText("icham-v2-test")).toBeInTheDocument();
  expect(screen.getByText(/DSP template fallback used/)).toBeInTheDocument();
  expect(api).toHaveBeenCalledWith(
    "/predict",
    expect.anything(),
    expect.objectContaining({ method: "POST" }),
  );
});

it("rejects a stale or mismatched response", async () => {
  vi.stubGlobal("crypto", { randomUUID: () => "request-test" });
  vi.mocked(api).mockResolvedValue({ ...response, input_sha256: "e".repeat(64) });
  render(<ModelReadout sample={sample} />);
  fireEvent.click(screen.getByRole("button", { name: "Run TSLM V2" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "does not match the selected recording",
  );
  expect(screen.getByRole("button", { name: "Retry TSLM V2" })).toBeInTheDocument();
});
