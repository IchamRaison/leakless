/** Model state only. No probability is rendered until an evaluated checkpoint is wired. */
export function ModelReadout() {
  return (
    <dl className="model-readout">
      <div>
        <dt>Model</dt>
        <dd>
          TSLM · NOT EVALUATED YET <span className="pending-dot" />
        </dd>
      </div>
      <div>
        <dt>Probability leak</dt>
        <dd className="pending-value">NOT EVALUATED YET</dd>
      </div>
      <div>
        <dt>Model output</dt>
        <dd>Awaiting evaluated checkpoint</dd>
      </div>
    </dl>
  );
}
