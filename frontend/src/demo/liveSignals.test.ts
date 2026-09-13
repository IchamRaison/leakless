import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, it } from "vitest";
import {
  CONFIRM_SAMPLES,
  baseline,
  reading,
  sampleAt,
  sensorStates,
} from "./liveSignals";
import { simulateIncident } from "./simulation";

const event = simulateIncident({ x: 1000, y: 240 });

it("keeps moving in healthy operation without reporting an anomaly", () => {
  const values = Array.from({ length: 240 }, (_, k) => reading(0, k).rms);
  expect(new Set(values.map((v) => v.toFixed(2))).size).toBeGreaterThan(60);
  // Two hours of healthy stream: small fluctuations, transients, never an observed anomaly.
  for (let k = 0; k < 14_400; k += 3)
    for (const [s, sensor] of sensorStates(k, [], false).entries()) {
      expect(sensor.status).toBe("healthy");
      expect(Math.abs(sensor.reading.rms - baseline(s).rms)).toBeLessThan(10);
    }
});

it("maps the same instant to the same sample, so a reload redraws the same history", () => {
  const at = Date.parse("2026-09-13T10:57:50");
  expect(sampleAt(at)).toBe(sampleAt(at + 499));
  expect(sampleAt(at + 500)).toBe(sampleAt(at) + 1);
  expect(reading(2, 1234, { event, k0: 1200 })).toEqual(
    reading(2, 1234, { event, k0: 1200 }),
  );
});

it("shows the incident immediately and reports observation until the server confirms", () => {
  const incident = { event, k0: 100 };
  const [s1, s2, s3] = sensorStates(112, incident, false);
  expect(s3.reading.rms).toBeGreaterThan(s2.reading.rms);
  expect(s2.reading.rms).toBeGreaterThan(s1.reading.rms);
  expect(s3.reading.crest).toBeLessThan(baseline(2).crest - 1);
  expect(s3.status).toBe("under observation");
  expect(s1.status).toBe("healthy");
  expect(sensorStates(100, incident, false)[2].status).toBe("healthy");
  expect(sensorStates(112, incident, true)[2].status).toBe("confirmed");
  expect(CONFIRM_SAMPLES).toBeGreaterThan(1);
});

it("resolving never rewrites history: past samples stay, future samples return to normal", () => {
  const ongoing = { event, k0: 1000 };
  const resolved = { event, k0: 1000, k1: 1200 };
  for (let k = 990; k < 1200; k += 1)
    for (let s = 0; s < 3; s += 1)
      expect(reading(s, k, resolved)).toEqual(reading(s, k, ongoing));
  for (let k = 1200; k < 1300; k += 1)
    for (let s = 0; s < 3; s += 1)
      expect(reading(s, k, resolved)).toEqual(reading(s, k, null));
  // Earlier incidents in the history keep drawing their own window.
  const later = { event, k0: 5000 };
  expect(reading(2, 1100, [later, resolved])).toEqual(
    reading(2, 1100, ongoing),
  );
});

it("live stream code never imports the official result, a model, or Telegram", () => {
  for (const file of [
    "liveSignals.ts",
    "LiveSignalChart.tsx",
    "SensorTable.tsx",
    "SimulationPanel.tsx",
    "incidents.ts",
  ]) {
    const source = readFileSync(join(process.cwd(), "src/demo", file), "utf8");
    expect(source, file).not.toMatch(
      /from "\.\/(official|officialResult|evidence|ModelReadout)|\/predict|LSTM|TELEGRAM_|api\.telegram/,
    );
  }
});
