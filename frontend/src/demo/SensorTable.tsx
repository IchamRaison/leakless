import type { SensorState } from "./liveSignals";
import { SENSOR_COLORS } from "./LiveSignalChart";

const clock = (ms: number) =>
  new Date(ms).toLocaleTimeString("en-GB", { hour12: false });

/** Latest reading of each sensor, industrial-table style. */
export function SensorTable({
  sensors,
  receivedMs,
  focused = null,
}: {
  sensors: SensorState[];
  receivedMs: number;
  /** Highlighted row, e.g. "sensor-2". */
  focused?: string | null;
}) {
  return (
    <div className="sensor-table-wrap">
      <table className="sensor-table">
        <caption>Sensors · latest reading</caption>
        <thead>
          <tr>
            <th scope="col">Sensor</th>
            <th scope="col">RMS (mg)</th>
            <th scope="col">Crest</th>
            <th scope="col">Status</th>
            <th scope="col">Last reception</th>
          </tr>
        </thead>
        <tbody>
          {sensors.map(({ sensor, reading, status }, s) => (
            <tr
              key={sensor.id}
              className={`is-${status.replace(/ /g, "-")}${
                focused === `sensor-${sensor.id}` ? " is-focused" : ""
              }`}
              aria-current={focused === `sensor-${sensor.id}` || undefined}
            >
              <th scope="row">
                <i style={{ background: SENSOR_COLORS[s] }} />
                {sensor.code}
              </th>
              <td>{reading.rms.toFixed(1)}</td>
              <td>{reading.crest.toFixed(1)}</td>
              <td>
                <span className="sensor-status">{status}</span>
              </td>
              <td>{clock(receivedMs)} (0 s ago)</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
