import { LEVEL_DB, toUnit } from "./replay";

export type NetworkChannel = { id: string; name: string; level: number | null };

// Flow order matters: the dash animation runs from each path's start to its end.
const SEGMENTS = [
  "M130,240 H600",
  "M600,240 H1080 V300",
  "M300,240 V70",
  "M300,70 H880",
  "M880,70 V240",
  "M590,70 V160 H440 V240",
];
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

/** Illustrative network: water motion is decorative; only ripple size is measured (replayed level). */
export function PipeNetwork({
  channels,
  playing,
}: {
  channels: NetworkChannel[];
  playing: boolean;
}) {
  return (
    <figure className={playing ? "pipe-network" : "pipe-network paused"}>
      <svg
        viewBox="0 30 1240 280"
        role="img"
        aria-label="Illustrative pipe network with animated water. Each channel ripple follows the measured level of its replayed recording. No leak is shown."
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
        {channels.map((channel, index) => {
          const [x, y, dx, dy] = POINTS[index];
          const unit =
            channel.level == null ? 0 : toUnit(channel.level, LEVEL_DB);
          return (
            <g className="listen-point" key={channel.id}>
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
        network: pipes, water motion and channel positions are drawn for the
        demo, not measured, and no leak is shown. Ripple size = measured level
        of each replayed recording.
      </figcaption>
    </figure>
  );
}
