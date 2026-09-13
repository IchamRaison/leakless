"""43s de vraie inférence C1, OpenTSLM/Qwen et aperçus ; aucun message externe."""
import argparse
import json
from pathlib import Path
import time

import httpx
import numpy as np

from replay_endpoint import Receiver,replay


class DuplicateReceiver(Receiver):
    duplicate_verified=False
    def window(self,envelope):
        result=super().window(envelope)
        if envelope["sequence"]==33:
            second=super().window(envelope)
            assert second["duplicate"] and second["probability_leak"]==result["probability_leak"]
            self.duplicate_verified=True
        return result


def main(args):
    original=json.loads(args.source.read_text())
    # Source : scénario train extrêmes explicitement artificiel de la recette C1.
    low,high=original["entries"][0],original["entries"][3]
    order=[low]*3+[high]*35+[low]*5
    args.output.mkdir(parents=True,exist_ok=False)
    scenario=original|{"entries":[entry|{"source_offset_samples":i*8000} for i,entry in enumerate(order)],
        "incidents":[],"consumer_seconds":0,"selection":"deliberate repeated train extrema; artificial 43s language smoke"}
    (args.output/"scenario.json").write_text(json.dumps(scenario,indent=2))
    receiver=DuplicateReceiver(args.endpoint)
    result=replay(args.output/"scenario.json",args.output/"replay",receiver=receiver)
    rows=[json.loads(line) for line in (args.output/"replay/events.jsonl").read_text().splitlines()]
    received=[r["prediction"] for r in rows if r["event"]=="received"]
    assert len(received)==43 and receiver.duplicate_verified
    assert all(r["preview"] is None for r in received[:33])
    pending=received[33]["preview"]
    assert pending["kind"]=="persistent" and pending["facts"]["current_high_seconds"]==31
    assert pending["recipient"]=="Nevil" and not pending["sent"]
    with httpx.Client(base_url=args.endpoint,timeout=10,trust_env=False) as client:
        for _ in range(100):
            state=client.get(f"/temporal/sessions/{result['session_id']}").raise_for_status().json()
            if len(state["previews"])==2 and all(p.get("status")=="ready" for p in state["previews"]): break
            time.sleep(.1)
        assert len(state["previews"])==2 and all(p["status"]=="ready" for p in state["previews"])
        assert {p["kind"] for p in state["previews"]}=={"persistent","ended"}
        assert all(not p["sent"] for p in state["previews"])
        health=client.get("/temporal/health").raise_for_status().json()
        assert health["temporal_language"] and health["sequence_model_status"]=="prior_lstm_disabled"
    report={"status":"passed","n_windows":43,"notification_first_sequence":33,
        "high_windows_at_notification":31,"duplicate_verified":True,"external_messages_sent":0,
        "model_version":health["temporal_language"],"previews":state["previews"],
        "server_latency_p95_ms":float(np.quantile([r["latency_ms"] for r in received],.95)),
        "artificial_chronology":True,"quality_evaluation":False}
    with (args.output/"report.json").open("x") as file: json.dump(report,file,indent=2,ensure_ascii=False)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--endpoint",default="http://127.0.0.1:8020")
    main(parser.parse_args())
