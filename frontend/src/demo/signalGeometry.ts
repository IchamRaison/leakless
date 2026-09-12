import type { Visualization } from "../contracts";

export type SignalRing = {
  time: number;
  radius: number;
  angle: number;
  energyDb: number;
  centroidHz: number;
  bandsDb: number[];
};
const clamp = (x: number) => Math.max(0, Math.min(1, x));
export const displayEnergy = (db: number) => clamp((db + 120) / 120);

/** Deterministic display transform of the measured, pooled display STFT.
 * Not the ML preprocessor. No labels, predictions, sensor positions or RNG inputs.
 * Fixed scales across recordings; power is numerical, not calibrated sound pressure.
 */
export function signalGeometry(data: Visualization): SignalRing[] {
  const {
    times,
    frequencies_hz: frequencies,
    power_db: matrix,
  } = data.spectrogram;
  if (
    !times.length ||
    !frequencies.length ||
    matrix.length !== frequencies.length ||
    matrix.some(
      (row) =>
        row.length !== times.length || row.some((v) => !Number.isFinite(v)),
    ) ||
    times.some(
      (t) => !Number.isFinite(t) || t < 0 || t > data.duration_seconds,
    ) ||
    frequencies.some((f) => !Number.isFinite(f) || f < 0)
  )
    return [];
  const maxFrequency = Math.max(1, ...frequencies);
  // Keep at most 72 actual time columns; never invent intermediate measurements.
  const stride = Math.max(1, Math.ceil(times.length / 72));
  return times.flatMap((time, t) => {
    if (t % stride !== 0) return [];
    const power = matrix.map((row) => 10 ** (row[t] / 10));
    const total = power.reduce((a, b) => a + b, 0);
    const energyDb = 10 * Math.log10(Math.max(total / power.length, 1e-12));
    const centroidHz =
      energyDb <= -119.99
        ? 0
        : power.reduce((sum, value, f) => sum + value * frequencies[f], 0) /
          total;
    const bandsDb = [0, 1, 2].map((band) => {
      const values = power.filter(
        (_, f) =>
          Math.min(2, Math.floor((frequencies[f] / maxFrequency) * 3)) === band,
      );
      return (
        10 *
        Math.log10(
          Math.max(
            values.reduce((a, b) => a + b, 0) / (values.length || 1),
            1e-12,
          ),
        )
      );
    });
    return [
      {
        time,
        energyDb,
        centroidHz,
        bandsDb,
        radius: 0.18 + 0.95 * displayEnergy(energyDb),
        angle: (centroidHz / maxFrequency) * Math.PI,
      },
    ];
  });
}
