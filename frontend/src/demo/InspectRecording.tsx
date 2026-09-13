import { useEffect, useRef, useState } from "react";
import { AudioLines, LoaderCircle, Trash2, Upload, X } from "lucide-react";
import { api, formatTime } from "../api";
import {
  sampleSchema,
  visualizationSchema,
  type Sample,
  type Visualization,
} from "../contracts";
import { SignalView } from "../SignalView";
import { ModelReadout } from "./ModelReadout";
import { tslm } from "./official";
import { TemporalSignalMap } from "./TemporalSignalMap";

const MAX_BYTES = 8 * 1024 * 1024;
type Loaded = { sample: Sample; visualization: Visualization };

/** Upload path inherited from the acoustic studio: same API, same checks, no model call. */
export function InspectRecording({ paused }: { paused: boolean }) {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [time, setTime] = useState(0);
  const fileInput = useRef<HTMLInputElement>(null);
  const inFlight = useRef(false);
  const request = useRef<AbortController | null>(null);
  // Server copy held for this section; released when replaced or removed.
  const held = useRef<string | null>(null);
  useEffect(() => () => request.current?.abort(), []);

  async function release(sampleId: string) {
    const response = await fetch(`/api/samples/${sampleId}`, {
      method: "DELETE",
      signal: AbortSignal.timeout(10_000),
    });
    if (!response.ok && response.status !== 404)
      throw new Error("Could not remove this recording from the local server.");
  }

  async function importFile(file: File | undefined) {
    if (!file || inFlight.current) return;
    if (!file.name.toLowerCase().endsWith(".wav")) {
      setError("Choose a .wav file.");
      return;
    }
    if (file.size > MAX_BYTES) {
      setError("File limited to 8 MiB.");
      return;
    }
    const controller = new AbortController();
    request.current = controller;
    inFlight.current = true;
    setBusy(true);
    setError("");
    setLoaded(null);
    setTime(0);
    try {
      const body = new FormData();
      // Neutral name: the original filename may carry a label and is not sent.
      body.append("file", file, "recording.wav");
      const sample = await api("/samples", sampleSchema, {
        method: "POST",
        body,
        signal: controller.signal,
      });
      const visualization = await api(
        `/samples/${sample.sample_id}/visualization`,
        visualizationSchema,
        { signal: controller.signal },
      );
      if (
        visualization.sample_id !== sample.sample_id ||
        visualization.input_sha256 !== sample.input_sha256
      )
        throw new Error(
          "The measurements do not match the uploaded recording.",
        );
      if (controller.signal.aborted) return;
      const previous = held.current;
      held.current = sample.sample_id;
      setLoaded({ sample, visualization });
      if (previous && previous !== sample.sample_id)
        void release(previous).catch(() => {});
    } catch (err) {
      if (!controller.signal.aborted)
        setError(
          err instanceof Error ? err.message : "Could not load this recording.",
        );
    } finally {
      if (!controller.signal.aborted) setBusy(false);
      inFlight.current = false;
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function remove() {
    if (!loaded) return;
    setError("");
    try {
      await release(loaded.sample.sample_id);
      held.current = null;
      setLoaded(null);
      setTime(0);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  const choose = () => fileInput.current?.click();
  const sample = loaded?.sample;
  return (
    <section
      className="inspect-section"
      id="inspect"
      aria-labelledby="inspect-title"
    >
      <div className="demo-section-heading">
        <div>
          <span className="demo-kicker">06 / INSPECT YOUR OWN SIGNAL</span>
          <h2 id="inspect-title">Inspect another recording.</h2>
          <p>
            Measured by the same local API as the examples. No model output is
            produced.
          </p>
        </div>
        <button className="load-button" disabled={busy} onClick={choose}>
          {busy ? (
            <LoaderCircle size={14} className="spin" />
          ) : (
            <Upload size={14} />
          )}
          Load a recording
        </button>
      </div>
      <input
        ref={fileInput}
        type="file"
        accept=".wav,audio/wav"
        className="visually-hidden"
        aria-label="Upload a WAV recording"
        onChange={(event) => void importFile(event.target.files?.[0])}
      />
      {error && (
        <div className="upload-error" role="alert">
          <span>{error}</span>
          <button aria-label="Dismiss message" onClick={() => setError("")}>
            <X size={14} />
          </button>
        </div>
      )}
      {!loaded || !sample ? (
        <div
          className="upload-drop"
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            void importFile(event.dataTransfer.files[0]);
          }}
        >
          <AudioLines size={38} strokeWidth={1} />
          <strong>
            {busy
              ? "Reading your recording…"
              : "Drop a short WAV recording here."}
          </strong>
          <p>
            {busy
              ? "Decoding, checking identity and measuring the spectrum."
              : "WAV · 8 MiB max · 30 seconds max · mono or stereo, 8–192 kHz."}
          </p>
          <button disabled={busy} onClick={choose}>
            <Upload size={14} />
            Choose a WAV file
          </button>
        </div>
      ) : (
        <>
          <div className="inspection-grid">
            <section className="map-panel" aria-labelledby="upload-map-title">
              <div className="panel-heading">
                <div>
                  <span className="demo-kicker">YOUR RECORDING</span>
                  <h2 id="upload-map-title">Temporal Signal Map</h2>
                </div>
                <span className="status-tag">MEASURED SIGNAL</span>
              </div>
              <TemporalSignalMap data={loaded.visualization} paused={paused} />
              <p className="map-disclaimer">
                An abstract view of the acoustic signal. Not a physical water
                simulation.
              </p>
            </section>
            <aside
              className="decision-panel"
              aria-label="Recording metadata and model state"
            >
              <span className="demo-kicker">RECORDING METADATA</span>
              <dl className="recording-meta">
                <div>
                  <dt>Duration</dt>
                  <dd>{formatTime(sample.duration_seconds)}</dd>
                </div>
                <div>
                  <dt>Sample rate</dt>
                  <dd>{sample.sample_rate_hz / 1000} kHz</dd>
                </div>
                <div>
                  <dt>Channels</dt>
                  <dd>{sample.channels === 1 ? "Mono" : "Stereo"}</dd>
                </div>
                <div>
                  <dt>Source</dt>
                  <dd>{sample.source}</dd>
                </div>
              </dl>
              {sample.warnings.map((warning) => (
                <p className="upload-warning" key={warning}>
                  {warning}
                </p>
              ))}
              <ModelReadout />
              <div className="decision-note">
                <span className="small-rule" />
                <p>
                  Measured properties only.
                  <br />
                  {tslm.decision}
                </p>
              </div>
              <button className="quiet-button" onClick={() => void remove()}>
                <Trash2 size={12} />
                Remove recording
              </button>
            </aside>
          </div>
          <div className="raw-signal">
            <div className="raw-content">
              <div className="audio-and-source">
                <audio
                  key={sample.sample_id}
                  controls
                  src={`/api/samples/${sample.sample_id}/audio`}
                  onTimeUpdate={(event) =>
                    setTime(event.currentTarget.currentTime)
                  }
                  onError={() =>
                    setError(
                      "Audio playback failed. Upload the recording again or check the connection.",
                    )
                  }
                  aria-label="Listen to the uploaded recording"
                />
                <p>
                  Original filename not sent. No label is known for this upload.
                  <br />
                  Held in local server memory until removed or the server stops.
                </p>
                <details className="clip-provenance">
                  <summary>Signal identity & provenance</summary>
                  <p>
                    SHA-256 of the decoded signal, including sample rate and
                    channels.
                  </p>
                  <code>{sample.input_sha256}</code>
                </details>
              </div>
              <SignalView data={loaded.visualization} time={time} />
            </div>
          </div>
        </>
      )}
    </section>
  );
}
