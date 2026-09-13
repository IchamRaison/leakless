import { api } from "../api";
import {
  sampleSchema,
  visualizationSchema,
  type Sample,
  type Visualization,
} from "../contracts";
import recordings from "./recordings.json";

export type Recording = (typeof recordings.records)[number];
export type LoadedExample = {
  record: Recording;
  sample: Sample;
  visualization: Visualization;
};

/** Fetches a bundled WAV, checks its bytes, decodes it through the API and checks every identity. */
export async function loadExample(
  record: Recording,
  signal: AbortSignal,
): Promise<LoadedExample> {
  const response = await fetch(record.url, { signal });
  if (!response.ok)
    throw new Error("The example recording is unavailable. Retry.");
  const raw = await response.arrayBuffer();
  const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", raw))]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
  if (digest !== record.fileSha256)
    throw new Error(
      "Recording integrity check failed. No measurements displayed.",
    );
  const form = new FormData();
  form.append("file", new Blob([raw], { type: "audio/wav" }), "recording.wav");
  const sample = await api("/samples", sampleSchema, {
    method: "POST",
    body: form,
    signal,
  });
  if (sample.input_sha256 !== record.inputSha256)
    throw new Error("The decoded signal does not match the selected example.");
  const visualization = await api(
    `/samples/${sample.sample_id}/visualization`,
    visualizationSchema,
    { signal },
  );
  if (
    visualization.input_sha256 !== sample.input_sha256 ||
    visualization.sample_id !== sample.sample_id
  )
    throw new Error("The measurements do not match the selected recording.");
  return { record, sample, visualization };
}
