import { expect, it } from "vitest";
import { formatClock, levelHistory, loopTime, ringAt, toUnit } from "./replay";
import type { SignalRing } from "./signalGeometry";

const ring = (time: number, energyDb: number): SignalRing => ({
  time,
  energyDb,
  radius: 0,
  angle: 0,
  centroidHz: 0,
  bandsDb: [0, 0, 0],
});
const rings = [ring(0.1, -50), ring(0.5, -40), ring(0.9, -60)];

it("replays measured windows in a loop without interpolating", () => {
  expect(loopTime(2.25, 1)).toBeCloseTo(0.25);
  expect(loopTime(-0.25, 1)).toBeCloseTo(0.75);
  expect(ringAt(rings, 1, 0.05).energyDb).toBe(-50);
  expect(ringAt(rings, 1, 0.6).energyDb).toBe(-40);
  expect(ringAt(rings, 1, 3.95).energyDb).toBe(-60);
  expect(levelHistory(rings, 1, 1.6, 3, 0.5)).toEqual([-40, -50, -40]);
  expect(levelHistory(rings, 1, 1.95, 2, 0.5)).toEqual([-50, -60]);
});

it("uses fixed scales and a readable clock", () => {
  expect(toUnit(-50, [-70, -30])).toBe(0.5);
  expect(toUnit(-10, [-70, -30])).toBe(1);
  expect(formatClock(72.34)).toBe("01:12.3");
});
