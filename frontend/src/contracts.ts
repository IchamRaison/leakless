import { z } from "zod";

const mode = z.enum(["live", "replay", "development_fixture"]);
export const sampleSchema = z.object({
  sample_id: z.string().regex(/^[0-9a-f]{24}$/),
  input_sha256: z.string().regex(/^[0-9a-f]{64}$/),
  duration_seconds: z.number().positive(),
  sample_rate_hz: z.number().positive(),
  channels: z.number().int().positive(),
  source: z.string(),
  execution_mode: mode,
  warnings: z.array(z.string()),
  label_available: z.boolean(),
});
export type Sample = z.infer<typeof sampleSchema>;
export const healthSchema = z.object({
  status: z.literal("ok"),
  schema_version: z.literal("0.1"),
  device: z.string(),
  models: z.array(
    z.object({
      name: z.enum(["tslm", "baseline"]),
      available: z.boolean(),
      version: z.string().nullable(),
      reason: z.string(),
    }),
  ),
  capabilities: z.object({
    upload: z.boolean(),
    visualization: z.boolean(),
    perturbation: z.boolean(),
    replay: z.boolean(),
  }),
});
export type Health = z.infer<typeof healthSchema>;
export const visualizationSchema = z.object({
  sample_id: z.string(),
  input_sha256: z.string(),
  visualization_version: z.string(),
  duration_seconds: z.number().positive(),
  waveform: z.object({
    times: z.array(z.number()),
    min: z.array(z.number()),
    max: z.array(z.number()),
  }),
  spectrogram: z.object({
    times: z.array(z.number()),
    frequencies_hz: z.array(z.number()),
    power_db: z.array(z.array(z.number())),
    floor_db: z.number(),
    reference: z.string(),
  }),
  parameters: z.object({
    n_fft: z.number(),
    hop_samples: z.number(),
    window: z.string(),
    channels: z.string(),
    resampling: z.boolean(),
    pooling: z.string(),
  }),
  notice: z.string(),
});
export type Visualization = z.infer<typeof visualizationSchema>;
export const predictionSchema = z
  .object({
    schema_version: z.literal("0.1"),
    sample_id: z.string(),
    input_sha256: z.string().regex(/^[0-9a-f]{64}$/),
    model_name: z.enum(["tslm", "baseline"]),
    model_version: z.string(),
    preprocessing_version: z.string(),
    prediction: z.enum(["leak", "no_leak"]).nullable(),
    class_scores: z.record(z.string(), z.number()).nullable(),
    score_type: z.enum(["raw", "calibrated", "none"]),
    abstained: z.boolean(),
    abstention_reason: z.string().nullable(),
    observations: z.array(
      z.object({
        name: z.string(),
        value: z.union([z.number(), z.string()]),
        unit: z.string().nullable(),
        method: z.string(),
      }),
    ),
    description: z.string().nullable(),
    latency_ms: z.number().nonnegative(),
    warnings: z.array(z.string()),
    execution_mode: mode,
  })
  .strict();
export type Prediction = z.infer<typeof predictionSchema>;

export const predictResponseSchema = predictionSchema.extend({
  request_id: z.string().min(1),
  decision_version: z.string().nullable(),
  decision_artifact_sha256: z.string().regex(/^[0-9a-f]{64}$/).nullable(),
  threshold: z.number().nullable(),
  calibration: z.literal("none"),
  description_source: z
    .enum(["llm_checked_against_dsp", "dsp_template_fallback"])
    .nullable(),
  fallback_used: z.boolean(),
  fallback_reasons: z.array(z.string()),
  model_input_sha256: z.string().regex(/^[0-9a-f]{64}$/).nullable(),
});
export type PredictResponse = z.infer<typeof predictResponseSchema>;
