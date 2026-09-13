import type { SignalRing } from "./signalGeometry";

// Fixed display scales shared by every channel; never normalised per clip.
export const LEVEL_DB: [number, number] = [-70, -30];
export const BAND_DB: [number, number] = [-90, -30];

export const toUnit = (value: number, [low, high]: [number, number]) =>
  Math.max(0, Math.min(1, (value - low) / (high - low)));

/** Position inside a looped clip; the replay clock keeps running across loops. */
export const loopTime = (clock: number, duration: number) =>
  ((clock % duration) + duration) % duration;

/** Latest measured window at or before the looped time. Never interpolates. */
export function ringAt(
  rings: SignalRing[],
  duration: number,
  clock: number,
): SignalRing {
  const t = loopTime(clock, duration);
  let current = rings[0];
  for (const ring of rings) {
    if (ring.time > t) break;
    current = ring;
  }
  return current;
}

/** Level trace for the last `count` steps, oldest first, read from the same measured windows. */
export function levelHistory(
  rings: SignalRing[],
  duration: number,
  clock: number,
  count = 80,
  step = 0.1,
): number[] {
  return Array.from(
    { length: count },
    (_, index) =>
      ringAt(rings, duration, clock - (count - 1 - index) * step).energyDb,
  );
}

export function formatClock(seconds: number) {
  const whole = Math.max(0, seconds);
  const minutes = Math.floor(whole / 60);
  return `${String(minutes).padStart(2, "0")}:${(whole - minutes * 60)
    .toFixed(1)
    .padStart(4, "0")}`;
}
