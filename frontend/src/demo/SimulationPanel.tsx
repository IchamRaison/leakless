import { useEffect, useRef } from "react";
import { ArrowDown, Check, RotateCcw } from "lucide-react";
import type { Incident } from "./incidents";

export const clockTime = (epochSeconds: number) =>
  new Date(epochSeconds * 1000).toLocaleTimeString("en-GB", { hour12: false });

const RECIPIENT = "On-call plumber";

function deliveryStatus(incident: Incident, persistenceS: number) {
  if (incident.status === "ANOMALY_PENDING")
    return `Armed · waiting ${persistenceS} s`;
  if (incident.delivery_status === "dispatching") return "Dispatching";
  if (incident.delivery_status === "sent" && incident.alert_sent_at)
    return `Sent · ${clockTime(incident.alert_sent_at)}`;
  if (incident.delivery_status === "failed")
    return "Notification failed · Telegram delivery unavailable";
  if (incident.delivery_status === "unknown")
    return "Delivery unconfirmed · check Telegram before any new alert";
  return "Not dispatched";
}

const sensorLabel = (n: number) => `Sensor ${String(n).padStart(2, "0")}`;

/** Active incident, as decided by the backend: observation, confirmation, automatic alert. */
export function SimulationPanel({
  incident,
  elapsedS,
  persistenceS,
  busy = false,
  reveal = true,
  onResolve,
  onRetry,
}: {
  incident: Incident;
  /** Server-time seconds since the anomaly started. */
  elapsedS: number;
  persistenceS: number;
  busy?: boolean;
  /** Scroll the panel into view (injection from the network drawing, below the panel). */
  reveal?: boolean;
  onResolve: () => void;
  /** Offered only when the notification genuinely failed. */
  onRetry?: () => void;
}) {
  const panel = useRef<HTMLElement>(null);
  const { x, y } = incident.position;
  useEffect(() => {
    if (!reveal) return;
    const reduced = window.matchMedia?.(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    panel.current?.scrollIntoView?.({
      block: "nearest",
      behavior: reduced ? "auto" : "smooth",
    });
  }, [x, y, reveal]);
  const confirmed = incident.status !== "ANOMALY_PENDING";
  const observed = Math.min(persistenceS, Math.max(0, Math.floor(elapsedS)));
  const sensor = sensorLabel(incident.strongest_sensor);
  return (
    <section
      className={confirmed ? "sim-panel is-alert" : "sim-panel"}
      aria-labelledby="sim-title"
      ref={panel}
    >
      <header>
        <h2 id="sim-title">
          {confirmed
            ? "Persistent anomaly confirmed"
            : "Anomaly under observation"}
        </h2>
      </header>
      <div className="sim-body">
        <div>
          {confirmed && <p className="sim-decision">Inspection recommended</p>}
          <p className={confirmed ? "sim-strongest" : "sim-decision"}>
            Strongest response: {sensor}
          </p>
          <div className="persistence" aria-label="Anomaly persistence">
            <span className="demo-kicker">
              {confirmed && incident.confirmed_at
                ? `Persistent for ${persistenceS} s · confirmed ${clockTime(incident.confirmed_at)}`
                : `Anomaly under observation · ${observed} s / ${persistenceS} s`}
            </span>
            <span className="persistence-bar">
              <i
                style={{
                  width: `${(confirmed ? 1 : observed / persistenceS) * 100}%`,
                }}
              />
            </span>
          </div>
        </div>
        <div className="alert-dispatch">
          <span className="demo-kicker">Automatic alert</span>
          <dl>
            <div>
              <dt>Channel</dt>
              <dd>Telegram → {RECIPIENT}</dd>
            </div>
            <div>
              <dt>Status</dt>
              <dd
                className={`alert-status is-${incident.delivery_status}`}
                role="status"
              >
                {deliveryStatus(incident, persistenceS)}
              </dd>
            </div>
            <div>
              <dt>Acknowledgement</dt>
              <dd>Not implemented</dd>
            </div>
          </dl>
          <div className="sim-actions">
            {incident.delivery_status === "failed" && onRetry && (
              <button
                className="quiet-button"
                onClick={onRetry}
                disabled={busy}
              >
                <RotateCcw size={13} /> Retry notification
              </button>
            )}
            <button
              className="quiet-button"
              onClick={onResolve}
              disabled={busy}
            >
              <Check size={13} /> Resolve incident
            </button>
            <a className="sim-evidence" href="#evidence">
              See the evidence <ArrowDown size={13} />
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}

/** Last resolved incident, kept for whoever arrives later. */
export function IncidentHistory({ incident }: { incident: Incident }) {
  const events: [number | null, string][] = [
    [incident.started_at, "anomaly started"],
    [incident.confirmed_at, "persistence threshold reached"],
    [
      incident.alert_sent_at,
      `alert sent · ${incident.transport ?? "telegram"}`,
    ],
    [incident.resolved_at, "resolved"],
  ];
  return (
    <section className="incident-history" aria-labelledby="history-title">
      <h2 id="history-title">Last incident</h2>
      <ol>
        {events
          .filter(([at]) => at !== null)
          .map(([at, label]) => (
            <li key={label}>
              <time>{clockTime(at!)}</time> {label}
            </li>
          ))}
        {incident.delivery_status === "failed" && (
          <li className="is-failed">notification failed</li>
        )}
      </ol>
      <dl>
        <div>
          <dt>Incident</dt>
          <dd>{incident.incident_id}</dd>
        </div>
        <div>
          <dt>Strongest response</dt>
          <dd>{sensorLabel(incident.strongest_sensor)}</dd>
        </div>
        <div>
          <dt>Delivery</dt>
          <dd>
            {incident.transport ?? "—"} ·{" "}
            {incident.delivery_status.replace("_", " ")}
          </dd>
        </div>
        {incident.telegram_message_id && (
          <div>
            <dt>Telegram message</dt>
            <dd>#{incident.telegram_message_id}</dd>
          </div>
        )}
      </dl>
    </section>
  );
}
