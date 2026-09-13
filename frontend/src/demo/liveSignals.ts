// Product preview of the live sensor stream. The telemetry is synthetic and deterministic: a pure
// function of the absolute sample index, the sensor and the incidents recorded by the server.
// No recording, no model, no TSLM. The alert decision itself belongs to the backend.
import { SENSORS, type TestIncident } from "./simulation";

export const SAMPLE_MS = 500;
export const WINDOW_SAMPLES = 240; // 120 s on screen
/** A sample is anomalous when RMS exceeds this multiple of the sensor's healthy baseline. */
export const RMS_FACTOR = 1.6;
/** Consecutive anomalous samples before a sensor is reported (filters single transients). */
export const CONFIRM_SAMPLES = 2;

const BASELINE = [
  { rms: 12.1, crest: 5.1 },
  { rms: 13.0, crest: 4.9 },
  { rms: 11.4, crest: 5.3 },
] as const;

export type Reading = { rms: number; crest: number };
/** An incident on the sample clock: active from k0, resolved at k1 (exclusive) if set. */
export type LiveIncident = {
  event: TestIncident;
  k0: number;
  k1?: number | null;
};

/** Absolute sample index of an epoch time: the same instant always maps to the same sample. */
export const sampleAt = (epochMs: number) => Math.floor(epochMs / SAMPLE_MS);

/** Deterministic pseudo-noise in [-1, 1]. */
function noise(sensor: number, k: number, salt: number) {
  const x =
    Math.sin((k + 1) * 12.9898 + sensor * 78.233 + salt * 37.719) * 43758.5453;
  return (x - Math.floor(x)) * 2 - 1;
}

export function baseline(sensor: number) {
  return BASELINE[sensor];
}

const asList = (incidents: LiveIncident | LiveIncident[] | null) =>
  incidents ? (Array.isArray(incidents) ? incidents : [incidents]) : [];

/** One reading of one sensor at sample k. Resolving an incident never changes earlier samples. */
export function reading(
  sensor: number,
  k: number,
  incidents: LiveIncident | LiveIncident[] | null = null,
): Reading {
  const base = BASELINE[sensor];
  // Healthy operation: small fluctuations, slow drift and rare short transients.
  const spike =
    noise(sensor, k, 3) > 0.985 ? 3 + 5 * Math.abs(noise(sensor, k, 4)) : 0;
  let rms =
    base.rms +
    0.35 * noise(sensor, k, 1) +
    0.25 * Math.sin(k / 9 + sensor) +
    spike;
  let crest = base.crest + 0.22 * noise(sensor, k, 2) + spike * 0.18;
  const incident = asList(incidents).find(
    (item) => k >= item.k0 && (item.k1 == null || k < item.k1),
  );
  if (incident) {
    // Injected test incident: a sustained broadband rise, stronger near the incident.
    const strongest = Math.max(
      ...incident.event.responses.map((r) => r.response),
    );
    const level = Math.sqrt(
      incident.event.responses[sensor].response / strongest,
    );
    const ramp = Math.min(1, (k - incident.k0) / 6);
    rms += ramp * level * 22 * (1 + 0.12 * noise(sensor, k, 5));
    crest -= ramp * level * 1.8;
  }
  return { rms, crest: Math.max(1.2, crest) };
}

export function isAnomalous(sensor: number, value: Reading) {
  return value.rms > BASELINE[sensor].rms * RMS_FACTOR;
}

/** Consecutive anomalous samples of a sensor ending at sample k. */
export function anomalyRun(
  sensor: number,
  k: number,
  incidents: LiveIncident | LiveIncident[] | null,
) {
  let run = 0;
  while (isAnomalous(sensor, reading(sensor, k - run, incidents))) {
    run += 1;
    if (run > 480) break;
  }
  return run;
}

export type SensorStatus = "healthy" | "under observation" | "confirmed";

export type SensorState = {
  sensor: (typeof SENSORS)[number];
  reading: Reading;
  status: SensorStatus;
};

/** Per-sensor signal state. "confirmed" only once the server has confirmed persistence. */
export function sensorStates(
  k: number,
  incidents: LiveIncident | LiveIncident[] | null,
  confirmed: boolean,
): SensorState[] {
  return SENSORS.map((sensor, s) => {
    const signalAnomaly = anomalyRun(s, k, incidents) >= CONFIRM_SAMPLES;
    return {
      sensor,
      reading: reading(s, k, incidents),
      status: !signalAnomaly
        ? "healthy"
        : confirmed
          ? "confirmed"
          : "under observation",
    };
  });
}
