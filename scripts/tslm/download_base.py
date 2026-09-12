"""Télécharge les poids officiels Qwen, sans compte, à une révision immuable."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tslm/v0.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    snapshot_download(
        repo_id=config["base_model"], revision=config["base_revision"],
        local_dir=args.output, token=False,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja", "LICENSE", "README.md"],
    )
    checksums = {}
    for path in sorted(args.output.iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            checksums[path.name] = {"sha256": checksum, "bytes": path.stat().st_size}
    report = {"repo_id": config["base_model"], "revision": config["base_revision"],
              "authenticated": False, "files": checksums}
    (args.output / "download-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
