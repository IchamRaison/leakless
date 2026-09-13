import { tslm } from "./official";

/** Model state only. No per-recording probability is ever rendered here. */
export function ModelReadout() {
  return (
    <dl className="model-readout">
      <div>
        <dt>Model</dt>
        <dd>
          {tslm.status} {!tslm.evaluated && <span className="pending-dot" />}
        </dd>
      </div>
      <div>
        <dt>Probability leak</dt>
        <dd className="pending-value">{tslm.probability}</dd>
      </div>
      <div>
        <dt>Model output</dt>
        <dd>{tslm.modelOutput}</dd>
      </div>
    </dl>
  );
}
