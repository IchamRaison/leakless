// Illustrative product simulation on the drawn pipe network. Deterministic geometry only:
// no recording, no model, no TSLM. Nothing here is measured.

export type Point = { x: number; y: number };

/** Pipe drawing, in SVG units. Flow order matters: the dash animation runs from start to end. */
export const PIPE_PATHS = [
  "M130,240 H600",
  "M600,240 H1080 V300",
  "M300,240 V70",
  "M300,70 H880",
  "M880,70 V240",
  "M590,70 V160 H440 V240",
] as const;

/** Illustrative sensor positions on the drawing. They are not dataset channels. */
export const SIM_SENSORS = [
  {
    id: 1,
    name: "Sim Sensor 1",
    zone: "Zone N1",
    x: 330,
    y: 240,
    label: [16, -16],
  },
  {
    id: 2,
    name: "Sim Sensor 2",
    zone: "Zone N2",
    x: 520,
    y: 160,
    label: [-20, 42],
  },
  {
    id: 3,
    name: "Sim Sensor 3",
    zone: "Zone N3",
    x: 880,
    y: 200,
    label: [-112, 8],
  },
] as const;
export type SimSensor = (typeof SIM_SENSORS)[number];

/** Attenuation length of the demo rule, in SVG units. */
export const RESPONSE_LAMBDA = 260;
/** Invisible hit stroke around each pipe, in SVG units (~33 px wide on a 420 px screen, where the drawing is ~319 px). */
export const HIT_STROKE = 130;

/** Axis-aligned polylines parsed from the M/H/V pipe paths. */
export function pipePolylines(
  paths: readonly string[] = PIPE_PATHS,
): Point[][] {
  return paths.map((d) => {
    const points: Point[] = [];
    let current: Point = { x: 0, y: 0 };
    for (const [, command, args] of d.matchAll(/([MHV])\s*([-\d.,\s]+)/g)) {
      const values = args
        .trim()
        .split(/[\s,]+/)
        .map(Number);
      if (command === "M") current = { x: values[0], y: values[1] };
      if (command === "H") current = { x: values[0], y: current.y };
      if (command === "V") current = { x: current.x, y: values[0] };
      points.push(current);
    }
    return points;
  });
}

const POLYLINES = pipePolylines();

/** Closest point of the pipe network to a click, in SVG units. */
export function projectToPipe(click: Point, polylines = POLYLINES) {
  let best = { x: 0, y: 0, pipe: -1, distance: Infinity };
  polylines.forEach((line, pipe) => {
    for (let i = 1; i < line.length; i += 1) {
      const a = line[i - 1];
      const b = line[i];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const length = dx * dx + dy * dy;
      const t =
        length === 0
          ? 0
          : Math.max(
              0,
              Math.min(
                1,
                ((click.x - a.x) * dx + (click.y - a.y) * dy) / length,
              ),
            );
      const x = a.x + t * dx;
      const y = a.y + t * dy;
      const distance = Math.hypot(click.x - x, click.y - y);
      if (distance < best.distance) best = { x, y, pipe, distance };
    }
  });
  return best;
}

export type SimulatedIncident = {
  position: Point;
  pipe: number;
  responses: { sensor: SimSensor; distance: number; response: number }[];
  nearest: SimSensor;
};

/** Demo rule: response_i = exp(-distance_i / lambda). Straight-line distance on the drawing. */
export function simulateIncident(click: Point): SimulatedIncident {
  const { x, y, pipe } = projectToPipe(click);
  const responses = SIM_SENSORS.map((sensor) => {
    const distance = Math.hypot(sensor.x - x, sensor.y - y);
    return {
      sensor,
      distance,
      response: Math.exp(-distance / RESPONSE_LAMBDA),
    };
  });
  const nearest = responses.reduce((a, b) =>
    b.distance < a.distance ? b : a,
  ).sensor;
  return { position: { x, y }, pipe, responses, nearest };
}
