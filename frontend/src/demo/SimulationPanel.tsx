import { ArrowDown, MessageCircle, X } from "lucide-react";
import type { SimulatedIncident } from "./simulation";
import { buildAlertMessage, whatsappLink } from "./whatsapp";

/** Product simulation read-out. Synthetic event, distance rule only: no recording, no model. */
export function SimulationPanel({
  incident,
  onClear,
}: {
  incident: SimulatedIncident;
  onClear: () => void;
}) {
  const message = buildAlertMessage(incident.nearest);
  const strongest = Math.max(...incident.responses.map((r) => r.response));
  return (
    <section className="sim-panel" aria-labelledby="sim-title">
      <header>
        <span className="sim-mode">SIMULATION MODE</span>
        <h2 id="sim-title">
          SIMULATED INCIDENT — synthetic event and illustrative sensor
          responses. TSLM not involved.
        </h2>
      </header>
      <div className="sim-body">
        <div>
          <span className="demo-kicker">Simulated sensor responses</span>
          <ul className="sim-bars">
            {incident.responses.map(({ sensor, response }) => (
              <li
                key={sensor.id}
                className={
                  sensor.id === incident.nearest.id ? "is-nearest" : undefined
                }
              >
                <span>{sensor.name}</span>
                <span
                  className="sim-bar"
                  role="img"
                  aria-label={`${sensor.name} simulated response${
                    sensor.id === incident.nearest.id ? ", strongest" : ""
                  }`}
                >
                  <i style={{ width: `${(response / strongest) * 100}%` }} />
                </span>
              </li>
            ))}
          </ul>
          <p className="sim-note">
            Illustrative sensor positions — not dataset channels.
          </p>
        </div>
        <div>
          <p className="sim-decision">
            Inspection recommended near {incident.nearest.name}
          </p>
          <p className="sim-note">
            Demo rule: response strength is simulated from distance. Not a model
            output.
          </p>
          <figure className="sim-alert" aria-label="Alert preview">
            <figcaption className="demo-kicker">Alert preview</figcaption>
            <pre>{message}</pre>
          </figure>
          <div className="sim-actions">
            <a
              className="load-button"
              href={whatsappLink(message)}
              target="_blank"
              rel="noopener noreferrer"
            >
              <MessageCircle size={14} /> Send WhatsApp alert
            </a>
            <button className="quiet-button" onClick={onClear}>
              <X size={13} /> Clear simulation
            </button>
            <a className="sim-evidence" href="#evidence">
              See the evidence <ArrowDown size={13} />
            </a>
          </div>
          <p className="sim-note">
            Opens WhatsApp with the message prefilled. Nothing is sent until the
            operator confirms.
          </p>
        </div>
      </div>
    </section>
  );
}
