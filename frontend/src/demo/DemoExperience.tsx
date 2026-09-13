import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  AudioLines,
  Check,
  ChevronRight,
  ExternalLink,
  Pause,
  Play,
  RotateCcw,
  Upload,
} from "lucide-react";
import { SignalView } from "../SignalView";
import {
  BuildingScene,
  measurementPoints,
  type Measurement,
} from "./BuildingScene";
import { FullscreenButton } from "./FullscreenButton";
import { InspectRecording } from "./InspectRecording";
import { loadExample, type LoadedExample, type Recording } from "./loadExample";
import { ModelReadout } from "./ModelReadout";
import { PointPanel } from "./PointPanel";
import { LEVEL_DB, ringAt, toUnit } from "./replay";
import { signalGeometry } from "./signalGeometry";
import { TemporalSignalMap } from "./TemporalSignalMap";
import { evidence } from "./evidence";
import { tslm } from "./official";
import recordings from "./recordings.json";
import "./demo.css";

type Loaded = LoadedExample;

export default function DemoExperience() {
  const [measurement, setMeasurement] = useState<Measurement | null>(null);
  const [record, setRecord] = useState<Recording | null>(null);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const [paused, setPaused] = useState(false);
  const [time, setTime] = useState(0);
  const request = useRef<AbortController | null>(null);
  const cache = useRef(new Map<string, Loaded>());
  const buildingPanel = useRef<HTMLDivElement>(null);
  const pointSelector = useRef<HTMLDivElement>(null);
  // This view loads lazily, after the browser's own jump to #signal, #evidence or #inspect.
  useEffect(() => {
    const id = decodeURIComponent(location.hash.slice(1));
    const target = id && id !== "demo" ? document.getElementById(id) : null;
    target?.scrollIntoView?.();
  }, []);
  useEffect(() => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoaded(null);
    setError("");
    setTime(0);
    setLoading(false);
    if (!record) return () => controller.abort();
    const current = record;
    const cached = cache.current.get(current.id);
    if (cached) {
      setLoaded(cached);
      return () => controller.abort();
    }
    setLoading(true);
    async function load() {
      try {
        const result = await loadExample(current, controller.signal);
        if (controller.signal.aborted) return;
        cache.current.set(current.id, result);
        setLoaded(result);
      } catch (err) {
        if (!controller.signal.aborted)
          setError(
            err instanceof Error
              ? err.message
              : "Could not load this recording.",
          );
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void load();
    return () => controller.abort();
  }, [record, retry]);

  // Points are illustrative positions, not leak locations: choosing one never picks a recording.
  const selectPoint = (id: Measurement) => setMeasurement(id);
  const leaveFullscreenTo = async (hash: string) => {
    if (document.fullscreenElement)
      await document.exitFullscreen().catch(() => {});
    location.hash = hash;
    // Setting the hash it already has does not scroll; sections of this page scroll themselves.
    document.getElementById(hash)?.scrollIntoView?.();
  };
  // The recordings stay locked until a point is chosen: take the viewer to the N1–N4 choice.
  const goToPoints = () => {
    const reduced = window.matchMedia?.(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    pointSelector.current?.scrollIntoView?.({
      block: "center",
      behavior: reduced ? "auto" : "smooth",
    });
    pointSelector.current
      ?.querySelector("button")
      ?.focus({ preventScroll: true });
  };
  const reload = () => {
    cache.current.clear();
    setRetry((value) => value + 1);
  };
  // Never render previous clip measurements under a new selection, even before effects run.
  const active = loaded?.record.id === record?.id ? loaded : null;
  const pulse = useMemo(() => {
    if (!active) return null;
    const rings = signalGeometry(active.visualization);
    const duration = active.visualization.duration_seconds;
    return rings.length
      ? (seconds: number) =>
          toUnit(ringAt(rings, duration, seconds).energyDb, LEVEL_DB)
      : null;
  }, [active]);

  return (
    <div className="demo-app" id="demo">
      <header className="demo-topbar">
        <a className="demo-brand" href="#demo">
          <AudioLines size={21} strokeWidth={1.6} /> LEAKLESS{" "}
          <span>RESEARCH PROTOTYPE</span>
        </a>
        <nav aria-label="Demo navigation">
          <a href="#signal">Explore the signal</a>
          <a href="#evidence">The evidence</a>
          <a href="#monitor">Live monitor</a>
          <a className="load-link" href="#inspect">
            Load a recording <Upload size={14} />
          </a>
        </nav>
      </header>
      <main className="demo-main">
        <section className="demo-hero" aria-labelledby="why-heading">
          <div className="hero-copy">
            <h1 id="why-heading">
              Water damage <br />
              becomes visible late.
              <br />
              <em>Can acoustic signals help decide when to inspect?</em>
            </h1>
            <p className="hero-subtitle">
              LeakLess helps facilities teams inspect unusual acoustic signals
              before deciding where to investigate.
            </p>
            <p className="prototype-note">
              This prototype tests leak-associated acoustic discrimination.
              <br />
              Early-warning lead time is not demonstrated. No field validation.
            </p>
            <div className="hero-who">
              <span>FOR</span>
              <p>
                Facilities and maintenance teams
                <br />
                responsible for buildings.
              </p>
            </div>
            <a className="explore-link" href="#signal">
              Inspect an acoustic signal <ArrowDown size={15} />
            </a>
          </div>
          <div className="building-panel" ref={buildingPanel}>
            <div className="building-topline">
              <span>01 / BUILDING CONTEXT</span>
              <span>ILLUSTRATIVE SCENE</span>
            </div>
            <BuildingScene
              selected={measurement}
              onSelect={selectPoint}
              paused={paused}
              pulse={pulse}
            />
            {measurement && (
              <PointPanel
                measurement={measurement}
                onInspect={() => void leaveFullscreenTo("monitor")}
                onListen={() => void leaveFullscreenTo("signal")}
              />
            )}
            <div className="building-caption">
              <span>
                <i className={measurement ? "selected-dot" : ""} />
                {measurement
                  ? `Selected measurement point · ${measurement}`
                  : "Select a measurement point"}
              </span>
              <span className="caption-actions">
                <button
                  className="quiet-button"
                  aria-label={
                    paused
                      ? "Resume presentation motion"
                      : "Pause presentation motion"
                  }
                  onClick={() => setPaused(!paused)}
                >
                  {paused ? <Play size={12} /> : <Pause size={12} />}
                  {paused ? "Resume" : "Pause"} motion
                </button>
                <FullscreenButton target={buildingPanel} label="building" />
              </span>
            </div>
            <div
              className="point-selector"
              aria-label="Measurement points"
              ref={pointSelector}
            >
              {measurementPoints.map((id) => (
                <button
                  key={id}
                  aria-label={`Measurement ${id}`}
                  aria-pressed={measurement === id}
                  className={measurement === id ? "active" : ""}
                  onClick={() => selectPoint(id)}
                >
                  {id}
                  <ChevronRight size={12} />
                </button>
              ))}
            </div>
            <p className="association-note">
              N1–N4 are illustrative positions. They do not locate leaks.
              <br />
              No sensor coordinates come from the dataset. Water flow is
              illustrative, not measured.
            </p>
          </div>
        </section>

        <section
          className="signal-section"
          id="signal"
          aria-labelledby="signal-title"
        >
          <div className="demo-section-heading">
            <div>
              <span className="demo-kicker">02 / SELECT A SIGNAL</span>
              <h2 id="signal-title" className="point-heading">
                {measurement ? (
                  <>
                    <span className="point-label">
                      Selected measurement point:
                    </span>{" "}
                    {measurement} — illustrative building position
                  </>
                ) : (
                  "Three recordings from the dataset"
                )}
              </h2>
              <p>
                {measurement
                  ? "The point does not locate a leak. Every point offers the same three demo recordings."
                  : "Select a point in the building to inspect a real recording."}
              </p>
            </div>
            <span className="outline-tag">ILLUSTRATIVE POSITION</span>
          </div>
          <span className="example-label" id="demo-recording-label">
            Demo recording:
            {!measurement && (
              <>
                {" "}
                <button className="example-hint" onClick={goToPoints}>
                  select a measurement point first
                  <ArrowUp size={12} aria-hidden />
                </button>
              </>
            )}
          </span>
          <div
            className="example-selector"
            role="group"
            aria-labelledby="demo-recording-label"
          >
            {recordings.records.map((example, index) => (
              <button
                key={example.id}
                disabled={!measurement}
                aria-pressed={record?.id === example.id}
                className={record?.id === example.id ? "active" : ""}
                onClick={() => setRecord(example)}
              >
                <span className="example-number">0{index + 1}</span>
                {example.title}
                {record?.id === example.id && <Check size={14} />}
              </button>
            ))}
          </div>
          <p className="examples-note">
            Three experimental recordings for demonstration, not predictions.
            The model’s output is binary: leak vs non-leak.
          </p>

          <div className="inspection-grid">
            <section className="map-panel" aria-labelledby="map-title">
              <div className="panel-heading">
                <div>
                  <span className="demo-kicker">03 / OBSERVE</span>
                  <h2 id="map-title">Temporal Signal Map</h2>
                </div>
                <span className="status-tag">
                  {active
                    ? "MEASURED SIGNAL"
                    : loading
                      ? "LOADING"
                      : "NO RECORDING"}
                </span>
              </div>
              {error ? (
                <div className="map-pending" role="alert">
                  <span>Signal unavailable</span>
                  <p>{error}</p>
                  <button onClick={reload}>
                    <RotateCcw size={14} />
                    Retry recording
                  </button>
                </div>
              ) : active ? (
                <TemporalSignalMap
                  data={active.visualization}
                  paused={paused}
                />
              ) : (
                <div className="map-pending" role="status">
                  <AudioLines size={42} strokeWidth={1} />
                  <strong>
                    {loading
                      ? "Reading the real signal…"
                      : "The recording gives this map its shape."}
                  </strong>
                  <p>
                    {loading
                      ? "Checking identity and measured spectral properties."
                      : measurement
                        ? "Choose a demo recording above. No signal values are simulated."
                        : "Choose a measurement point above. No signal values are simulated."}
                  </p>
                </div>
              )}
              <div className="map-key">
                <span>
                  <i />
                  GEOMETRY <small>measured signal properties</small>
                </span>
                <span>
                  <i className="model-key" />
                  MODEL COLOR / STATE <small>{tslm.mapKey}</small>
                </span>
              </div>
              <p className="map-disclaimer">
                An abstract view of the acoustic signal. Not a physical water
                simulation.
              </p>
            </section>
            <aside className="decision-panel" aria-labelledby="decision-title">
              <span className="demo-kicker">04 / INVESTIGATE</span>
              <h2 id="decision-title">
                Does this unusual <br />
                signal deserve <br />
                <em>inspection?</em>
              </h2>
              <ModelReadout />
              <p className="tslm-definition">{tslm.definition}</p>
              <div className="decision-note">
                <span className="small-rule" />
                <p>
                  A measured signal, then a human decision.
                  <br />
                  {tslm.decision}
                </p>
              </div>
              <a href="#evidence" className="evidence-link">
                See what we can demonstrate <ArrowDown size={14} />
              </a>
            </aside>
          </div>

          {active && (
            <details className="raw-signal" open>
              <summary>
                <AudioLines size={16} />
                <span>Listen to the recording. Inspect the waveform.</span>
                <span className="raw-meta">
                  {active.sample.duration_seconds.toFixed(2)} s ·{" "}
                  {active.sample.sample_rate_hz / 1000} kHz ·{" "}
                  {active.record.clipId}
                </span>
              </summary>
              <div className="raw-content">
                <div className="audio-and-source">
                  <audio
                    key={active.sample.sample_id}
                    controls
                    src={active.record.url}
                    onTimeUpdate={(event) =>
                      setTime(event.currentTarget.currentTime)
                    }
                    aria-label="Listen to selected experimental recording"
                  />
                  <p>
                    Source:{" "}
                    <a
                      href={recordings.source}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Zenodo 18631450 <ExternalLink size={11} />
                    </a>{" "}
                    · {recordings.license}
                    <br />
                    Development example from train. No continuous monitoring.
                  </p>
                  <details className="clip-provenance">
                    <summary>Clip identity & provenance</summary>
                    <p>
                      Dataset clip: {active.record.clipId}
                      <br />
                      Dependency cluster: {active.record.groupId}
                      <br />
                      Split: {recordings.split} / {active.record.fold}
                    </p>
                    <code>{active.sample.input_sha256}</code>
                  </details>
                </div>
                <SignalView data={active.visualization} time={time} />
              </div>
            </details>
          )}
        </section>

        <section
          className="evidence-section"
          id="evidence"
          aria-labelledby="evidence-title"
        >
          <div className="demo-section-heading">
            <div>
              <span className="demo-kicker">05 / QUESTION THE RESULT</span>
              <h2 id="evidence-title">What could the model be cheating on?</h2>
              <p>
                Find which information carries the decision. Not just the
                highest score.
              </p>
            </div>
            <span className="report-link">
              Protocol frozen before final TSLM evaluation · protocol-freeze-v1
            </span>
          </div>
          <div className="control-ladder">
            {evidence.controls
              .map((control) =>
                control.id === "C4" ? { ...control, ...tslm.c4 } : control,
              )
              .map((control) => (
                <article
                  className={`control-card ${control.id === "C1" ? "reference-control" : control.clipAuc == null ? "pending-control" : ""}`}
                  key={control.id}
                >
                  <span className="control-id">
                    {control.id}
                    {control.id === "C1" && <small>REFERENCE</small>}
                  </span>
                  <h3>{control.name}</h3>
                  <p>{control.detail}</p>
                  <dl>
                    <div>
                      <dt>Clip AUC</dt>
                      <dd>
                        {control.clipAuc?.toFixed(3) ?? "NOT EVALUATED YET"}
                      </dd>
                    </div>
                    <div>
                      <dt>Cluster AUC</dt>
                      <dd>
                        {control.clusterAuc?.toFixed(3) ?? "NOT EVALUATED YET"}
                      </dd>
                    </div>
                  </dl>
                </article>
              ))}
          </div>
          <p className="evidence-caveat">
            TEST · {evidence.testClusters} dependency clusters, including{" "}
            {evidence.testNonLeakClusters} non-leak. Wide uncertainty intervals.
            C1 vs C0 remains inconclusive. {tslm.verdict}
          </p>
          {tslm.summary && (
            <p className="evidence-caveat stress-caveat">{tslm.summary}</p>
          )}
          {tslm.stress && (
            <p className="evidence-caveat stress-caveat">{tslm.stress}</p>
          )}
          <div className="proof-columns">
            <section>
              <span className="demo-kicker">WE DEMONSTRATED</span>
              <ul>
                {evidence.demonstrated.map((claim) => (
                  <li key={claim}>
                    <Check size={13} />
                    {claim}
                  </li>
                ))}
              </ul>
            </section>
            <section>
              <span className="demo-kicker">WE HAVE NOT DEMONSTRATED</span>
              <ul>
                {evidence.notDemonstrated.map((claim) => (
                  <li key={claim}>
                    <span className="limit-dash">—</span>
                    {claim}
                  </li>
                ))}
              </ul>
            </section>
          </div>
        </section>
        <InspectRecording paused={paused} />
        <footer className="demo-footer">
          <span>
            <AudioLines size={15} /> LEAKLESS · TEMPORAL AI
          </span>
          <p>Research prototype · offline benchmark · no field validation</p>
          <a href="#demo">Back to top ↑</a>
        </footer>
      </main>
    </div>
  );
}
