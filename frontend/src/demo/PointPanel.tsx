import { ArrowDown, ArrowRight } from "lucide-react";
import type { Measurement } from "./BuildingScene";
import recordings from "./recordings.json";

/** Fullscreen building panel: an illustrative position, pointing to the live monitor. */
export function PointPanel({
  measurement,
  onInspect,
  onListen,
}: {
  measurement: Measurement;
  onInspect: () => void;
  onListen: () => void;
}) {
  return (
    <aside className="point-panel" aria-label="Illustrative measurement point">
      <span className="point-panel-kicker">Illustrative measurement point</span>
      <strong className="point-panel-id">{measurement}</strong>
      <p className="point-panel-state">
        Illustrative position · does not locate a leak.
      </p>
      <div className="point-panel-recording">
        <span className="point-panel-label">
          Live sensors are monitored on the live monitor.
        </span>
        <button className="point-panel-inspect" onClick={onInspect}>
          Open the live monitor <ArrowRight size={13} />
        </button>
        <button className="point-panel-listen" onClick={onListen}>
          Listen to the {recordings.records.length} dataset recordings{" "}
          <ArrowDown size={12} />
        </button>
      </div>
    </aside>
  );
}
