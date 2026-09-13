import { useLayoutEffect, useRef, useState } from "react";
import {
  RMS_FACTOR,
  SAMPLE_MS,
  WINDOW_SAMPLES,
  baseline,
  reading,
  type LiveIncident,
} from "./liveSignals";
import { SENSORS } from "./simulation";

export const SENSOR_COLORS = ["#6cb4ee", "#8fd18a", "#c7a6ff"] as const;

const PANELS = [
  {
    key: "rms",
    title: "RMS",
    unit: "mg",
    min: 0,
    max: 50,
    ticks: [0, 10, 20, 30, 40, 50],
  },
  {
    key: "crest",
    title: "Crest factor",
    unit: "",
    min: 0,
    max: 8,
    ticks: [0, 2, 4, 6, 8],
  },
] as const;
const PAD = { left: 44, right: 14, top: 10, bottom: 22 };
const PANEL_HEIGHT = 170;

const clock = (ms: number) =>
  new Date(ms).toLocaleTimeString("en-GB", { hour12: false });

/** Live time-series chart: RMS and crest factor of the three sensors over the last 120 s. */
export function LiveSignalChart({
  k,
  incidents,
}: {
  /** Current absolute sample index (see sampleAt). */
  k: number;
  /** Server-recorded incidents on the sample clock; resolved ones keep their past samples. */
  incidents: (LiveIncident & { confirmedK?: number | null })[];
}) {
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(960);
  useLayoutEffect(() => {
    const element = box.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) =>
      setWidth(Math.max(280, Math.round(entry.contentRect.width))),
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const first = k - WINDOW_SAMPLES + 1;
  const plotW = width - PAD.left - PAD.right;
  const plotH = PANEL_HEIGHT - PAD.top - PAD.bottom;
  const x = (j: number) =>
    PAD.left + ((j - first) / (WINDOW_SAMPLES - 1)) * plotW;
  const samples = Array.from({ length: WINDOW_SAMPLES }, (_, i) => first + i);
  const series = SENSORS.map((_, s) =>
    samples.map((j) => reading(s, j, incidents)),
  );
  // Time ticks aligned to the sample clock: every 20 s, every 40 s on narrow screens.
  const tickEvery = (width < 600 ? 40_000 : 20_000) / SAMPLE_MS;
  const timeTicks = samples.filter((j) => j % tickEvery === 0);
  const latest = series.map((values) => values[values.length - 1]);

  return (
    <div className="live-chart" ref={box}>
      <ul className="live-legend" aria-label="Sensor series">
        {SENSORS.map((sensor, s) => (
          <li key={sensor.id}>
            <i style={{ background: SENSOR_COLORS[s] }} />
            {sensor.code}
            <b>{latest[s].rms.toFixed(1)} mg</b>
            <span>crest {latest[s].crest.toFixed(1)}</span>
          </li>
        ))}
      </ul>
      {PANELS.map((panel) => {
        const y = (v: number) =>
          PAD.top +
          plotH -
          ((Math.min(panel.max, Math.max(panel.min, v)) - panel.min) /
            (panel.max - panel.min)) *
            plotH;
        return (
          <svg
            key={panel.key}
            width={width}
            height={PANEL_HEIGHT}
            viewBox={`0 0 ${width} ${PANEL_HEIGHT}`}
            role="img"
            aria-label={`${panel.title} of the three sensors, last 120 seconds`}
          >
            <text className="live-axis-title" x={PAD.left} y={PAD.top + 2}>
              {panel.title}
              {panel.unit && ` (${panel.unit})`}
            </text>
            {panel.ticks.map((tick) => (
              <g key={tick}>
                <line
                  className="live-grid"
                  x1={PAD.left}
                  x2={width - PAD.right}
                  y1={y(tick)}
                  y2={y(tick)}
                />
                <text
                  className="live-tick"
                  x={PAD.left - 6}
                  y={y(tick) + 4}
                  textAnchor="end"
                >
                  {tick}
                </text>
              </g>
            ))}
            {timeTicks.map((j) => (
              <g key={j}>
                <line
                  className="live-grid"
                  x1={x(j)}
                  x2={x(j)}
                  y1={PAD.top}
                  y2={PAD.top + plotH}
                />
                {panel.key === "crest" && (
                  <text
                    className="live-tick"
                    x={x(j)}
                    y={PANEL_HEIGHT - 6}
                    textAnchor="middle"
                  >
                    {clock(j * SAMPLE_MS)}
                  </text>
                )}
              </g>
            ))}
            {incidents
              .filter((item) => (item.k1 ?? k) >= first)
              .map((item) => {
                const start = Math.max(first, item.k0);
                const end = Math.min(k, item.k1 ?? k);
                return (
                  <g className="live-event" key={item.k0}>
                    <rect
                      x={x(start)}
                      y={PAD.top}
                      width={Math.max(0, x(end) - x(start))}
                      height={plotH}
                    />
                    {item.k0 >= first && (
                      <line
                        x1={x(item.k0)}
                        x2={x(item.k0)}
                        y1={PAD.top}
                        y2={PAD.top + plotH}
                      />
                    )}
                    {item.confirmedK != null && item.confirmedK >= first && (
                      <line
                        className="live-alert-line"
                        x1={x(item.confirmedK)}
                        x2={x(item.confirmedK)}
                        y1={PAD.top}
                        y2={PAD.top + plotH}
                      />
                    )}
                    {item.k1 != null && item.k1 >= first && (
                      <line
                        className="live-resolved-line"
                        x1={x(item.k1)}
                        x2={x(item.k1)}
                        y1={PAD.top}
                        y2={PAD.top + plotH}
                      />
                    )}
                  </g>
                );
              })}
            {panel.key === "rms" &&
              SENSORS.map((sensor, s) => (
                <line
                  key={sensor.id}
                  className="live-threshold"
                  x1={PAD.left}
                  x2={width - PAD.right}
                  y1={y(baseline(s).rms * RMS_FACTOR)}
                  y2={y(baseline(s).rms * RMS_FACTOR)}
                  style={{ stroke: SENSOR_COLORS[s] }}
                />
              ))}
            {panel.key === "rms" && (
              <text
                className="live-threshold-label"
                x={PAD.left + 8}
                y={
                  y(
                    Math.max(...SENSORS.map((_, s) => baseline(s).rms)) *
                      RMS_FACTOR,
                  ) - 6
                }
                textAnchor="start"
              >
                Alert threshold · {RMS_FACTOR}× baseline
              </text>
            )}
            {series.map((values, s) => (
              <polyline
                key={s}
                className="live-line"
                style={{ stroke: SENSOR_COLORS[s] }}
                points={values
                  .map(
                    (value, i) =>
                      `${x(first + i).toFixed(1)},${y(value[panel.key]).toFixed(1)}`,
                  )
                  .join(" ")}
              />
            ))}
          </svg>
        );
      })}
    </div>
  );
}
