import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  AudioLines,
  Pause,
  Play,
  RotateCcw,
  Volume2,
  VolumeX,
} from "lucide-react";
import { FullscreenButton } from "./FullscreenButton";
import { loadExample, type LoadedExample } from "./loadExample";
import { PipeNetwork } from "./PipeNetwork";
import recordings from "./recordings.json";
import {
  BAND_DB,
  LEVEL_DB,
  formatClock,
  levelHistory,
  loopTime,
  ringAt,
  toUnit,
} from "./replay";
import { signalGeometry } from "./signalGeometry";
import "./demo.css";

const TICK_MS = 100;
const bandNames = ["Low", "Mid", "High"];

/** Monitoring-style replay of the three real recordings. No alert, score or detection is produced. */
export default function MonitorReplay() {
  const [channels, setChannels] = useState<Record<string, LoadedExample>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [retry, setRetry] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [clock, setClock] = useState(0);
  const [listening, setListening] = useState<string | null>(null);
  const root = useRef<HTMLDivElement>(null);
  const audio = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    setErrors({});
    for (const record of recordings.records)
      loadExample(record, controller.signal)
        .then((loaded) => {
          if (!controller.signal.aborted)
            setChannels((current) => ({ ...current, [record.id]: loaded }));
        })
        .catch((err: Error) => {
          if (!controller.signal.aborted)
            setErrors((current) => ({ ...current, [record.id]: err.message }));
        });
    return () => controller.abort();
  }, [retry]);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const timer = window.setInterval(() => {
      const now = performance.now();
      setClock((value) => value + (now - last) / 1000);
      last = now;
    }, TICK_MS);
    return () => window.clearInterval(timer);
  }, [playing]);

  useEffect(() => {
    const element = audio.current;
    const channel = listening ? channels[listening] : null;
    if (!element) return;
    if (!channel || !playing) {
      element.pause();
      return;
    }
    if (!element.src.endsWith(channel.record.url))
      element.src = channel.record.url;
    element.currentTime = loopTime(
      clock,
      channel.visualization.duration_seconds,
    );
    void element.play().catch(() => setListening(null));
    // Re-sync only when the listened channel or play state changes, not on every tick.
  }, [listening, playing, channels]);

  const loaded = recordings.records.filter((record) => channels[record.id]);
  const rings = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(channels).map(([id, channel]) => [
          id,
          signalGeometry(channel.visualization),
        ]),
      ),
    [channels],
  );
  const network = recordings.records.map((record, index) => {
    const channel = channels[record.id];
    const measured = rings[record.id];
    return {
      id: record.id,
      name: `CH 0${index + 1}`,
      level:
        channel && measured?.length
          ? ringAt(measured, channel.visualization.duration_seconds, clock)
              .energyDb
          : null,
    };
  });
  return (
    <div className="demo-app monitor" ref={root}>
      <header className="demo-topbar monitor-topbar">
        <a className="demo-brand" href="#demo">
          <AudioLines size={21} strokeWidth={1.6} /> LEAKLESS{" "}
          <span>MONITOR REPLAY</span>
        </a>
        <div className="monitor-controls">
          <span className="monitor-clock" aria-live="off">
            REPLAY {formatClock(clock)}
          </span>
          <button
            className="quiet-button"
            onClick={() => setPlaying(!playing)}
            aria-label={playing ? "Pause replay" : "Resume replay"}
          >
            {playing ? <Pause size={12} /> : <Play size={12} />}
            {playing ? "Pause" : "Resume"}
          </button>
          <FullscreenButton
            target={root}
            label="monitor replay"
            className="quiet-button monitor-fullscreen"
          />
          <a className="monitor-back" href="#demo">
            <ArrowLeft size={13} /> Overview
          </a>
        </div>
      </header>
      <main className="demo-main monitor-main">
        <p className="monitor-banner" role="note">
          <strong>Replay, not a live feed.</strong> Three recorded train clips
          from the experimental dataset, each looped. Channels are recordings,
          not building locations. No detection model output: TSLM pending.
        </p>
        <dl className="monitor-summary">
          <div>
            <dt>Channels</dt>
            <dd>
              {loaded.length} / {recordings.records.length}
            </dd>
          </div>
          <div>
            <dt>Refresh</dt>
            <dd>{1000 / TICK_MS} Hz</dd>
          </div>
          <div>
            <dt>Alerts</dt>
            <dd>None generated</dd>
          </div>
          <div>
            <dt>Model output</dt>
            <dd className="pending-value">PENDING</dd>
          </div>
        </dl>
        <PipeNetwork channels={network} playing={playing} />
        <section className="monitor-grid" aria-label="Replayed channels">
          {recordings.records.map((record, index) =>
            channels[record.id] ? (
              <ChannelCard
                key={record.id}
                index={index}
                channel={channels[record.id]}
                clock={clock}
                listening={listening === record.id}
                onListen={() =>
                  setListening(listening === record.id ? null : record.id)
                }
              />
            ) : (
              <article className="monitor-card monitor-waiting" key={record.id}>
                <span className="demo-kicker">CH 0{index + 1}</span>
                {errors[record.id] ? (
                  <div role="alert">
                    <p>{errors[record.id]}</p>
                    <button onClick={() => setRetry((value) => value + 1)}>
                      <RotateCcw size={13} /> Retry
                    </button>
                  </div>
                ) : (
                  <p role="status">Reading the real recording…</p>
                )}
              </article>
            ),
          )}
        </section>
        <p className="monitor-scales">
          Same fixed scales for every channel. Level: mean spectral power,{" "}
          {LEVEL_DB[0]}…{LEVEL_DB[1]} dB, display STFT, not calibrated sound
          pressure. Bands: three equal frequency bands, {BAND_DB[0]}…
          {BAND_DB[1]} dB. Centroid: 0 to maximum frequency. Measured windows
          are shown as recorded; nothing is interpolated or predicted.
        </p>
      </main>
      <audio ref={audio} loop hidden />
    </div>
  );
}

function ChannelCard({
  channel,
  index,
  clock,
  listening,
  onListen,
}: {
  channel: LoadedExample;
  index: number;
  clock: number;
  listening: boolean;
  onListen: () => void;
}) {
  const { record, visualization } = channel;
  const duration = visualization.duration_seconds;
  const rings = useMemo(() => signalGeometry(visualization), [visualization]);
  const scope = useMemo(() => {
    const { times, min, max } = visualization.waveform;
    const peak = Math.max(1e-6, ...max.map(Math.abs), ...min.map(Math.abs));
    return times
      .map((t, i) => {
        const x = (t / duration) * 300;
        return `M${x},${30 - (min[i] / peak) * 26}L${x},${30 - (max[i] / peak) * 26}`;
      })
      .join("");
  }, [visualization, duration]);
  if (!rings.length)
    return (
      <article className="monitor-card monitor-waiting">
        <span className="demo-kicker">CH 0{index + 1}</span>
        <p>PENDING — valid spectral measurements required.</p>
      </article>
    );
  const now = ringAt(rings, duration, clock);
  const history = levelHistory(rings, duration, clock);
  const maxFrequency = Math.max(...visualization.spectrogram.frequencies_hz);
  const trace = history
    .map(
      (level, i) =>
        `${((i / (history.length - 1)) * 300).toFixed(1)},${(
          44 -
          toUnit(level, LEVEL_DB) * 40
        ).toFixed(1)}`,
    )
    .join(" ");
  const position = loopTime(clock, duration) / duration;
  return (
    <article className="monitor-card" aria-labelledby={`channel-${record.id}`}>
      <header>
        <span className="demo-kicker">CH 0{index + 1}</span>
        <h2 id={`channel-${record.id}`}>
          <span className="monitor-label">Dataset label</span> {record.title}
        </h2>
        <span className="monitor-clip">
          {record.fold} · {record.clipId}
        </span>
      </header>
      <svg
        className="monitor-scope"
        viewBox="0 0 300 60"
        preserveAspectRatio="none"
        role="img"
        aria-label={`Waveform of ${record.title} with replay position`}
      >
        <path d={scope} />
        <line x1={position * 300} x2={position * 300} y1="0" y2="60" />
      </svg>
      <dl className="monitor-metrics">
        <div>
          <dt>Level</dt>
          <dd>
            {now.energyDb.toFixed(1)} <small>dB</small>
          </dd>
          <i style={{ width: `${toUnit(now.energyDb, LEVEL_DB) * 100}%` }} />
        </div>
        <div>
          <dt>Centroid</dt>
          <dd>
            {Math.round(now.centroidHz)} <small>Hz</small>
          </dd>
          <i style={{ width: `${(now.centroidHz / maxFrequency) * 100}%` }} />
        </div>
      </dl>
      <div className="monitor-bands" aria-label="Band power">
        {now.bandsDb.map((band, i) => (
          <div key={bandNames[i]}>
            <span style={{ height: `${toUnit(band, BAND_DB) * 100}%` }} />
            <small>{bandNames[i]}</small>
          </div>
        ))}
      </div>
      <div className="monitor-trace">
        <span>Level · last 8 s (looped clip)</span>
        <svg viewBox="0 0 300 46" preserveAspectRatio="none" aria-hidden>
          <polyline points={trace} />
        </svg>
      </div>
      <footer>
        <span>
          TSLM <b className="pending-value">PENDING</b>
        </span>
        <button
          className="quiet-button"
          aria-pressed={listening}
          onClick={onListen}
        >
          {listening ? <VolumeX size={12} /> : <Volume2 size={12} />}
          {listening ? "Stop" : "Listen"}
        </button>
      </footer>
    </article>
  );
}
