"""Service de démonstration isolé : loopback, un processus, concurrence bornée."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8019)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port invalide")
    os.umask(0o077)
    os.environ.update(PIPE_TEMPORAL_BUNDLE=str(args.bundle.resolve()), PIPE_TEMPORAL_SHA256=args.sha256,
                      PIPE_TEMPORAL_DB=str(args.db.resolve()), PIPE_TEMPORAL_DEVICE="cuda")
    import torch
    import uvicorn
    torch.set_num_threads(2)
    # ponytail: un processus SQLite/modèles ; répartir par appareil si le débit le justifie.
    uvicorn.run("pipe.api.main:app", host="127.0.0.1", port=args.port, workers=1,
                limit_concurrency=8, backlog=8, timeout_keep_alive=5, access_log=False)
