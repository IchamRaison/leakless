"""Adaptateur HTTP du replay existant : pas de classe ni de score dans les entrées."""
import argparse
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_sensor import replay


class Receiver:
    def __init__(self, endpoint):
        parsed = urlparse(endpoint)
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1")):
            raise ValueError("HTTPS requis hors loopback")
        token = os.getenv("PIPE_TEMPORAL_TOKEN")
        self.client = httpx.Client(base_url=endpoint.rstrip("/"), timeout=2,
            headers={"Authorization": "Bearer " + token} if token else {}, trust_env=False)
        self.session = None

    def start(self, mode):
        response = self.client.post("/temporal/sessions", json={"source_mode": mode})
        response.raise_for_status()
        self.session = response.json()["session_id"]
        return self.session

    def window(self, envelope):
        response = self.client.post(f"/temporal/sessions/{self.session}/windows",
            data={"sequence": envelope["source_offset_samples"] // 8000,
                  "source_end_at": envelope["scheduled_at"]},
            files={"file": ("window.wav", envelope["payload"], "audio/wav")})
        response.raise_for_status()
        return response.json()

    def end(self):
        try:
            if self.session:
                response = self.client.post(f"/temporal/sessions/{self.session}/end")
                response.raise_for_status()
        finally:
            self.client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8019")
    args = parser.parse_args()
    print(json.dumps(replay(args.scenario, args.output, receiver=Receiver(args.endpoint))))
