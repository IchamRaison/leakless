import { describe, expect, it } from "vitest";
import type { Visualization } from "../contracts";
import { signalGeometry } from "./signalGeometry";

const visual = (
  powers = [
    [-60, -40],
    [-40, -60],
  ],
): Visualization => ({
  sample_id: "example",
  input_sha256: "a".repeat(64),
  visualization_version: "display-stft-v1",
  duration_seconds: 1,
  waveform: { times: [0], min: [0], max: [0] },
  spectrogram: {
    times: [0.25, 0.75],
    frequencies_hz: [100, 1000],
    power_db: powers,
    floor_db: -120,
    reference: "digital amplitude",
  },
  parameters: {
    n_fft: 1024,
    hop_samples: 256,
    window: "hann",
    channels: "mono",
    resampling: false,
    pooling: "mean power",
  },
  notice: "Display only",
});

describe("measured geometry", () => {
  it("is deterministic and independent of IDs and display context", () => {
    const input = visual();
    const before = JSON.stringify(input);
    expect(signalGeometry(input)).toEqual(
      signalGeometry(structuredClone(input)),
    );
    expect(signalGeometry({ ...input, sample_id: "different-point" })).toEqual(
      signalGeometry(input),
    );
    expect(JSON.stringify(input)).toBe(before);
  });
  it("retains time ordering, measured spectral centroid and fixed energy scales", () => {
    const rings = signalGeometry(visual());
    expect(rings.map((r) => r.time)).toEqual([0.25, 0.75]);
    expect(rings[0].radius).toBeCloseTo(rings[1].radius);
    expect(rings[0].centroidHz).toBeCloseTo(991.0891089);
    expect(rings[0].angle).toBeGreaterThan(rings[1].angle);
    expect(
      signalGeometry(
        visual([
          [-20, -20],
          [-20, -20],
        ]),
      )[0].radius,
    ).toBeGreaterThan(rings[0].radius);
  });
  it("fails closed on missing or non-finite measurements", () => {
    expect(
      signalGeometry(
        visual([
          [NaN, -60],
          [-60, -60],
        ]),
      ),
    ).toEqual([]);
    expect(signalGeometry(visual([[-60], [-60]]))).toEqual([]);
    expect(signalGeometry(visual([]))).toEqual([]);
  });
});
