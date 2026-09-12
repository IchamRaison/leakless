import { useEffect, useRef } from "react";
import type { Visualization } from "./contracts";
import { formatTime } from "./api";

export function SignalView({
  data,
  time,
}: {
  data: Visualization;
  time: number;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const waveform = data.waveform;
  const peak = Math.max(
    1,
    ...waveform.max.map(Math.abs),
    ...waveform.min.map(Math.abs),
  );
  const points = waveform.times
    .flatMap((t, i) => {
      const x = (t / data.duration_seconds) * 1000;
      return [
        `M${x},${60 - (waveform.min[i] / peak) * 48}`,
        `L${x},${60 - (waveform.max[i] / peak) * 48}`,
      ];
    })
    .join(" ");
  const position = Math.min(
    100,
    Math.max(0, (time / data.duration_seconds) * 100),
  );

  useEffect(() => {
    const node = canvas.current;
    if (!node) return;
    const matrix = data.spectrogram.power_db;
    const height = matrix.length;
    const width = matrix[0]?.length ?? 0;
    if (!width || !height) return;
    node.width = width;
    node.height = height;
    const ctx = node.getContext("2d");
    if (!ctx) return;
    const pixels = ctx.createImageData(width, height);
    // Échelle fixe -120..0 dB ; aucune normalisation automatique par clip.
    const stops = [
      [18, 25, 35],
      [28, 72, 91],
      [43, 127, 132],
      [172, 193, 112],
      [249, 224, 132],
    ];
    for (let f = 0; f < height; f++)
      for (let t = 0; t < width; t++) {
        const value = Math.max(0, Math.min(1, (matrix[f][t] + 120) / 120)) * 4;
        const a = Math.min(3, Math.floor(value));
        const mix = value - a;
        const offset = ((height - 1 - f) * width + t) * 4;
        for (let c = 0; c < 3; c++)
          pixels.data[offset + c] =
            stops[a][c] * (1 - mix) + stops[a + 1][c] * mix;
        pixels.data[offset + 3] = 255;
      }
    ctx.putImageData(pixels, 0, 0);
  }, [data]);

  return (
    <div className="signal-views">
      <div className="plot-heading">
        <h3>Forme d’onde</h3>
        <span>Amplitude numérique · ±{peak.toFixed(1)}</span>
      </div>
      <div className="plot waveform">
        <svg
          viewBox="0 0 1000 120"
          preserveAspectRatio="none"
          role="img"
          aria-label="Forme d’onde du signal importé, amplitude en fonction du temps"
        >
          <line x1="0" y1="60" x2="1000" y2="60" stroke="#deded8" />
          <path d={points} stroke="#45685c" strokeWidth="1.3" />
        </svg>
        <div className="playhead" style={{ left: `${position}%` }} />
      </div>
      <div className="time-axis">
        <span>0 s</span>
        <span>{formatTime(data.duration_seconds / 2)}</span>
        <span>{formatTime(data.duration_seconds)}</span>
      </div>
      <div className="plot-heading">
        <h3>Spectrogramme</h3>
        <span>Fréquence (kHz) · énergie (dB)</span>
      </div>
      <div className="spectral-layout">
        <div className="frequency-axis">
          <span>
            {Math.round(data.spectrogram.frequencies_hz.at(-1)! / 1000)}
          </span>
          <span>0</span>
        </div>
        <div className="plot spectrogram">
          <canvas
            ref={canvas}
            role="img"
            aria-label="Spectrogramme du même signal ; fréquence verticale et temps horizontal"
          />
          <div className="playhead light" style={{ left: `${position}%` }} />
        </div>
      </div>
      <div className="time-axis">
        <span>0 s</span>
        <span>{formatTime(data.duration_seconds / 2)}</span>
        <span>{formatTime(data.duration_seconds)}</span>
      </div>
      <div className="plot-foot">
        <span>
          Hann · FFT {data.parameters.n_fft} · hop {data.parameters.hop_samples}
        </span>
        <span className="legend">
          −120 <i /> 0 dB
        </span>
      </div>
      <p className="fine-print">
        {data.notice} Amplitude non calibrée en pression acoustique.
      </p>
    </div>
  );
}
