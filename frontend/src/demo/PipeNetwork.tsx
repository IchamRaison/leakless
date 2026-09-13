import { useRef, type KeyboardEvent, type MouseEvent } from "react";
import { LEVEL_DB, toUnit } from "./replay";
import {
  HIT_STROKE,
  PIPE_PATHS,
  pipePolylines,
  type Point,
  type SimulatedIncident,
} from "./simulation";

export type NetworkChannel = { id: string; name: string; level: number | null };

const SEGMENTS = PIPE_PATHS;
const VIEW = { x: 0, y: 30, width: 1240, height: 280 };

/** Client coordinates to SVG units, with a viewBox fallback where no CTM exists (jsdom). */
function toSvgPoint(
  svg: SVGSVGElement,
  clientX: number,
  clientY: number,
): Point {
  const ctm = svg.getScreenCTM?.();
  if (ctm) {
    const point = new DOMPoint(clientX, clientY).matrixTransform(ctm.inverse());
    return { x: point.x, y: point.y };
  }
  const rect = svg.getBoundingClientRect();
  return {
    x: VIEW.x + ((clientX - rect.left) / (rect.width || 1)) * VIEW.width,
    y: VIEW.y + ((clientY - rect.top) / (rect.height || 1)) * VIEW.height,
  };
}
const JOINTS = [
  [600, 240],
  [300, 240],
  [300, 70],
  [880, 70],
  [880, 240],
  [590, 70],
  [440, 240],
  [1080, 240],
];
const POINTS = [
  [200, 240, -40, -44],
  [590, 118, 34, 8],
  [880, 155, 50, 8],
] as const;
// Keyboard fallback for the pipe hit areas: the point of each pipe's first straight run
// closest to its middle while staying clear of every recording marker, so the simulated
// incident never hides under a REC node.
const REC_CLEARANCE = 70;
const KEYBOARD_TARGETS = pipePolylines().map((line) => {
  const middle = {
    x: (line[0].x + line[1].x) / 2,
    y: (line[0].y + line[1].y) / 2,
  };
  let best: Point = middle;
  let bestDistance = Infinity;
  for (let i = 1; i < line.length; i += 1) {
    const [a, b] = [line[i - 1], line[i]];
    const steps = Math.max(
      1,
      Math.round(Math.hypot(b.x - a.x, b.y - a.y) / 10),
    );
    for (let step = 0; step <= steps; step += 1) {
      const point = {
        x: a.x + ((b.x - a.x) * step) / steps,
        y: a.y + ((b.y - a.y) * step) / steps,
      };
      const clear = POINTS.every(
        ([x, y]) => Math.hypot(point.x - x, point.y - y) >= REC_CLEARANCE,
      );
      const distance = Math.hypot(point.x - middle.x, point.y - middle.y);
      if (clear && distance < bestDistance)
        [best, bestDistance] = [point, distance];
    }
  }
  return best;
});

/** Illustrative network: water motion is decorative; only ripple size is measured (replayed level). */
export function PipeNetwork({
  channels,
  playing,
  focused = null,
  onFocus,
  incident = null,
  onIncident,
}: {
  channels: NetworkChannel[];
  playing: boolean;
  focused?: string | null;
  onFocus?: (id: string) => void;
  incident?: SimulatedIncident | null;
  onIncident?: (click: Point) => void;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const hit = (event: MouseEvent<SVGPathElement>) => {
    if (svg.current && onIncident)
      onIncident(toSvgPoint(svg.current, event.clientX, event.clientY));
  };
  const hitKey = (event: KeyboardEvent<SVGPathElement>, pipe: number) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    onIncident?.(KEYBOARD_TARGETS[pipe]);
  };
  return (
    <figure
      className={[
        "pipe-network",
        playing ? "" : "paused",
        incident ? "simulating" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {incident && <span className="sim-chip">SIMULATION MODE</span>}
      <svg
        ref={svg}
        viewBox={`${VIEW.x} ${VIEW.y} ${VIEW.width} ${VIEW.height}`}
        role="group"
        aria-label="Illustrative pipe network with animated water. Each recording marker ripples with the measured level of its replayed recording. No leak is shown."
      >
        <g className="tank">
          <rect x="40" y="150" width="90" height="120" rx="16" />
          <line x1="40" y1="190" x2="130" y2="190" />
          <line x1="58" y1="270" x2="58" y2="292" />
          <line x1="112" y1="270" x2="112" y2="292" />
        </g>
        {SEGMENTS.map((d) => (
          <g key={d}>
            <path className="casing" d={d} />
            <path className="water" d={d} />
            <path className="flow" d={d} />
          </g>
        ))}
        {JOINTS.map(([x, y]) => (
          <rect
            key={`${x}-${y}`}
            className="joint"
            x={x - 14}
            y={y - 14}
            width="28"
            height="28"
            rx="5"
          />
        ))}
        {onIncident && (
          <g className="pipe-hits">
            {SEGMENTS.map((d, pipe) => (
              <path
                key={d}
                className="pipe-hit"
                d={d}
                strokeWidth={HIT_STROKE}
                role="button"
                tabIndex={0}
                aria-label={`Simulate an incident on pipe ${pipe + 1}`}
                onClick={hit}
                onKeyDown={(event) => hitKey(event, pipe)}
              />
            ))}
          </g>
        )}
        {incident && (
          <g className="sim-layer" aria-label="Simulated sensor responses">
            {incident.responses.map(({ sensor, response }) => (
              <g
                key={sensor.id}
                className={
                  sensor.id === incident.nearest.id
                    ? "sim-sensor is-nearest"
                    : "sim-sensor"
                }
              >
                <circle
                  className="sim-ripple"
                  cx={sensor.x}
                  cy={sensor.y}
                  r={16 + response * 40}
                  style={{ opacity: 0.15 + response * 0.65 }}
                />
                <rect
                  className="sim-node"
                  x={sensor.x - 9}
                  y={sensor.y - 9}
                  width="18"
                  height="18"
                  rx="3"
                />
                <text
                  x={sensor.x + sensor.label[0]}
                  y={sensor.y + sensor.label[1]}
                >
                  SIM {sensor.id}
                </text>
              </g>
            ))}
            <circle
              className="sim-incident"
              cx={incident.position.x}
              cy={incident.position.y}
              r="15"
            />
          </g>
        )}
        {channels.map((channel, index) => {
          const [x, y, dx, dy] = POINTS[index];
          const unit =
            channel.level == null ? 0 : toUnit(channel.level, LEVEL_DB);
          return (
            <g
              className={
                focused === channel.id
                  ? "listen-point is-focused"
                  : "listen-point"
              }
              key={channel.id}
              role="button"
              tabIndex={0}
              aria-label={`Show ${channel.name} in front`}
              aria-pressed={focused === channel.id}
              onClick={() => onFocus?.(channel.id)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onFocus?.(channel.id);
                }
              }}
            >
              <circle
                className="ripple"
                cx={x}
                cy={y}
                r={18 + unit * 38}
                style={{ opacity: 0.15 + unit * 0.55 }}
              />
              <circle className="node" cx={x} cy={y} r="11" />
              <text x={x + dx} y={y + dy}>
                {channel.name}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption>
        Three experimental recordings replayed for demonstration. Illustrative
        network: pipes, water motion and recording markers are drawn for the
        demo, not measured, and no leak is shown. Ripple size = measured level
        of each replayed recording.
        {onIncident &&
          !incident &&
          " Click a pipe to run the simulated incident demo."}
        {incident && (
          <strong className="sim-caption">
            {" "}
            Illustrative sensor positions — not dataset channels.
          </strong>
        )}
      </figcaption>
    </figure>
  );
}
