import { useEffect, useRef, useState } from "react";
import { ArrowLeft, AudioLines } from "lucide-react";
import { FullscreenButton } from "./FullscreenButton";
import { PipeNetwork } from "./PipeNetwork";
import { LiveSignalChart } from "./LiveSignalChart";
import { useIncidents, type Incident } from "./incidents";
import { SAMPLE_MS, sampleAt, sensorStates } from "./liveSignals";
import { SensorTable } from "./SensorTable";
import { IncidentHistory, SimulationPanel } from "./SimulationPanel";
import { SENSORS, simulateIncident } from "./simulation";
import "./demo.css";

// "#monitor/sensor-02" (or the older "#monitor/rec-02" from the building panel) highlights Sensor 2.
const focusFromHash = () => {
  const match = location.hash.match(/^#monitor\/(?:rec|sensor)-0([1-9])$/);
  const sensor = match ? SENSORS[Number(match[1]) - 1] : undefined;
  return sensor ? `sensor-${sensor.id}` : null;
};
// Live RMS mapped to the drawing's ripple size.
const rippleOf = (rms: number) => Math.max(0, Math.min(1, (rms - 8) / 32));

// Server incident on the absolute sample clock, with its confirmation and resolution samples.
const onClock = (incident: Incident) => ({
  event: simulateIncident(incident.position),
  k0: sampleAt(incident.started_at * 1000),
  k1: incident.resolved_at ? sampleAt(incident.resolved_at * 1000) : null,
  confirmedK: incident.confirmed_at
    ? sampleAt(incident.confirmed_at * 1000)
    : null,
});

/**
 * Product monitor preview: a synthetic live sensor stream, with incident persistence and automatic
 * Telegram alerts decided by the backend. No recording, no model output.
 */
export default function MonitorReplay() {
  const [focused, setFocused] = useState<string | null>(focusFromHash);
  const [fromNetwork, setFromNetwork] = useState(false);
  const incidents = useIncidents();
  const { snapshot, offsetMs } = incidents;
  // Live stream clock on server time: the same instant is the same sample after a reload.
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), SAMPLE_MS);
    return () => window.clearInterval(timer);
  }, []);
  const serverNowMs = nowMs + offsetMs;
  const k = sampleAt(serverNowMs);
  const active = snapshot?.active ?? null;
  const last = snapshot?.history[0] ?? null;
  const recorded = [active, ...(snapshot?.history ?? [])]
    .filter((item): item is Incident => item !== null)
    .map(onClock);
  const confirmed = !!active && active.status !== "ANOMALY_PENDING";
  const sensors = sensorStates(k, recorded, confirmed);
  // Single source of truth: the persistence duration comes from the server.
  const persistenceS = snapshot?.persistence_seconds ?? 0;
  const elapsedS = active ? serverNowMs / 1000 - active.started_at : 0;
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const change = () => setFocused(focusFromHash());
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  const focusSensor = (id: string) => {
    setFocused(id);
    history.replaceState(null, "", `#monitor/sensor-0${id.slice(-1)}`);
  };
  const inject = (position: { x: number; y: number }, network: boolean) => {
    setFromNetwork(network);
    void incidents.inject(simulateIncident(position).nearest.id, position);
  };
  const network = sensors.map(({ sensor, reading }) => ({
    id: `sensor-${sensor.id}`,
    name: sensor.name,
    label: sensor.code,
    level: null,
    intensity: rippleOf(reading.rms),
  }));
  const recentlyResolved =
    !active &&
    !!last?.resolved_at &&
    serverNowMs / 1000 - last.resolved_at < 300;
  const alertState = !snapshot
    ? { tone: "pending", text: "CONNECTING" }
    : !active
      ? recentlyResolved
        ? { tone: "normal", text: "RESOLVED" }
        : { tone: "normal", text: "NORMAL" }
      : !confirmed
        ? {
            tone: "observing",
            text: `ANOMALY UNDER OBSERVATION · ${Math.min(persistenceS, Math.max(0, Math.floor(elapsedS)))} s / ${persistenceS} s`,
          }
        : {
            tone: "triggered",
            text:
              active.delivery_status === "sent"
                ? "ALERT SENT"
                : "PERSISTENT ANOMALY",
          };
  return (
    <div className="demo-app monitor" ref={root}>
      <header className="demo-topbar monitor-topbar">
        <a className="demo-brand" href="#demo">
          <AudioLines size={21} strokeWidth={1.6} /> LEAKLESS{" "}
          <span>LIVE MONITOR</span>
        </a>
        <div className="monitor-controls">
          <FullscreenButton
            target={root}
            label="live monitor"
            className="quiet-button monitor-fullscreen"
          />
          <a className="monitor-back" href="#demo">
            <ArrowLeft size={13} /> Overview
          </a>
        </div>
      </header>
      <main className="demo-main monitor-main">
        <section className="monitor-status" aria-labelledby="monitor-title">
          <div>
            <h1 id="monitor-title">Continuous monitoring</h1>
            <p>3 sensors online · {1000 / SAMPLE_MS} Hz · last 120 s</p>
          </div>
          <p className={`alert-state is-${alertState.tone}`} aria-live="polite">
            {alertState.text}
          </p>
        </section>
        {incidents.error && (
          <p className="monitor-service-error" role="alert">
            Alert service unreachable. Monitoring display continues; alerts are
            decided by the server.
          </p>
        )}
        <section className="live-card" aria-labelledby="live-title">
          <header>
            <div>
              <h2 id="live-title">Water signal · last 120 s</h2>
              <span>
                <i className="live-dot" /> Live
              </span>
            </div>
            <div className="inject-control">
              <button
                className="quiet-button inject-button"
                disabled={!snapshot || !!active || incidents.busy}
                onClick={() => inject({ x: 1000, y: 240 }, false)}
              >
                Inject incident
              </button>
              <small>Scenario injection</small>
            </div>
          </header>
          <LiveSignalChart k={k} incidents={recorded} />
          <SensorTable
            sensors={sensors}
            focused={focused}
            receivedMs={k * SAMPLE_MS}
          />
        </section>
        {active && (
          <SimulationPanel
            incident={active}
            elapsedS={elapsedS}
            persistenceS={persistenceS}
            busy={incidents.busy}
            reveal={fromNetwork}
            onResolve={() => void incidents.resolve(active.incident_id)}
            onRetry={() => void incidents.retry(active.incident_id)}
          />
        )}
        {!active && last && <IncidentHistory incident={last} />}
        <PipeNetwork
          channels={network}
          playing
          focused={focused}
          onFocus={focusSensor}
          incident={active ? simulateIncident(active.position) : null}
          onIncident={
            snapshot && !active && !incidents.busy
              ? (click) => inject(click, true)
              : undefined
          }
        />
      </main>
    </div>
  );
}
