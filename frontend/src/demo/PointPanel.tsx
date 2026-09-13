import { useMemo } from "react";
import { ArrowRight, RotateCcw } from "lucide-react";
import type { Measurement } from "./BuildingScene";
import type { LoadedExample, Recording } from "./loadExample";
import { tslm } from "./official";
import recordings from "./recordings.json";

export const recLabel = (index: number) => `REC 0${index + 1}`;

/** Fullscreen building panel: an illustrative point, a real recording, no physical association. */
export function PointPanel({
  measurement,
  record,
  active,
  loading,
  error,
  onChoose,
  onRetry,
  onInspect,
}: {
  measurement: Measurement;
  record: Recording | null;
  active: LoadedExample | null;
  loading: boolean;
  error: string;
  onChoose: (record: Recording) => void;
  onRetry: () => void;
  onInspect: (index: number) => void;
}) {
  const index = record ? recordings.records.indexOf(record) : -1;
  const waveform = useMemo(() => {
    if (!active) return "";
    const { times, min, max } = active.visualization.waveform;
    const duration = active.visualization.duration_seconds;
    const peak = Math.max(1e-6, ...max.map(Math.abs), ...min.map(Math.abs));
    return times
      .map((t, i) => {
        const x = (t / duration) * 300;
        return `M${x},${28 - (min[i] / peak) * 24}L${x},${28 - (max[i] / peak) * 24}`;
      })
      .join("");
  }, [active]);
  return (
    <aside className="point-panel" aria-label="Illustrative measurement point">
      <span className="point-panel-kicker">Illustrative measurement point</span>
      <strong className="point-panel-id">{measurement}</strong>
      <p className="point-panel-note">
        No physical association between this point and the recording.
      </p>
      <span className="point-panel-kicker">Experimental recording replay</span>
      <div
        className="point-panel-recs"
        role="group"
        aria-label="Experimental recording replay"
      >
        {recordings.records.map((item, i) => (
          <button
            key={item.id}
            aria-pressed={record?.id === item.id}
            className={record?.id === item.id ? "active" : ""}
            onClick={() => onChoose(item)}
          >
            {recLabel(i)}
          </button>
        ))}
      </div>
      {error ? (
        <div className="point-panel-state" role="alert">
          <span>{error}</span>
          <button className="quiet-button" onClick={onRetry}>
            <RotateCcw size={12} /> Retry
          </button>
        </div>
      ) : active && index >= 0 ? (
        <div className="point-panel-recording">
          <span className="point-panel-label">
            {recLabel(index)} · dataset label <b>{active.record.title}</b>
          </span>
          <svg
            viewBox="0 0 300 56"
            preserveAspectRatio="none"
            role="img"
            aria-label={`Waveform of ${recLabel(index)}`}
          >
            <path d={waveform} />
          </svg>
          <span className="point-panel-tslm">{tslm.status}</span>
          <button
            className="point-panel-inspect"
            onClick={() => onInspect(index)}
          >
            Inspect this recording <ArrowRight size={13} />
          </button>
        </div>
      ) : (
        <p className="point-panel-state" role="status">
          {loading ? "Reading the real recording…" : "Choose a recording."}
        </p>
      )}
    </aside>
  );
}
