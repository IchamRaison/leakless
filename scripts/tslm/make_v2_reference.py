"""Copies A/C distinctes à preprocessing corrigé, sans apprentissage ni resélection."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from export_run import ROOT, sha256_file


def make_reference(source, output, revision, *, amplitude_evidence=False):
    from pipe.tslm.model import AMPLITUDE_SCORING_VERSION, CANONICAL_SCORING_VERSION
    from pipe.tslm.preprocessing import (AMPLITUDE_EVIDENCE_VERSION, CANONICAL_VERSION,
                                         VERSION, amplitude_spec)
    if type(amplitude_evidence) is not bool:
        raise ValueError("Opt-in amplitude_evidence booléen explicite requis")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Révision réelle SHA40 requise")
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink() or output.is_relative_to(source):
        raise ValueError("Dossier neuf hors bundle historique requis")
    checksums = json.loads((source / "checksums.json").read_text())
    for name, digest in checksums.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha256_file(path) != digest:
            raise ValueError("Intégrité V1 invalide")
    original = json.loads((source / "metadata.json").read_text())
    if original["preprocessing_version"] != VERSION:
        raise ValueError("Référence source V1 requise")
    variant = "C" if amplitude_evidence else "A"
    scoring_spec = {**original["scoring_spec"],
                    "version": AMPLITUDE_SCORING_VERSION if amplitude_evidence else CANONICAL_SCORING_VERSION,
                    "acoustic_batching": "one_clip_four_channels"}
    if amplitude_evidence:
        scoring_spec["amplitude_evidence"] = amplitude_spec()
    config = {"source_config_hash": original["config_hash"],
              "preprocessing_version": CANONICAL_VERSION,
              "scoring_spec": scoring_spec, "single_clip_acoustic_encoding": True,
              "variant": variant, "amplitude_evidence": amplitude_evidence,
              "purpose": "parity_only_no_retraining",
              "source_checksums_sha256": sha256_file(source / "checksums.json")}
    if amplitude_evidence:
        config["amplitude_evidence_version"] = AMPLITUDE_EVIDENCE_VERSION
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metadata = {key: original[key] for key in (
        "base_model", "base_revision", "architecture", "opentslm_revision", "timenet_revision", "protocol",
        "max_new_tokens", "scoring_spec", "manifest_sha256", "training_clip_ids", "training_groups")}
    metadata.update(config, config_hash=config_hash, code_revision=revision,
                    model_version=f"pipe-qwen3.5-4b-v2-parity-{variant.lower()}-{revision[:8]}",
                    source_model_version=original["model_version"],
                    source_training_commit=original["training_commit"],
                    retrained=False, use_for_cv_initialization=False)
    output.mkdir(parents=True, exist_ok=False)
    # Copie indépendante : aucun lien dur qui permettrait de modifier V1 ensuite.
    shutil.copytree(source / "base", output / "base")
    for name in ("temporal.pt", "QWEN-LICENSE", "QWEN-MODEL-CARD.md",
                 "requirements-ml.lock", "PROVENANCE.json"):
        shutil.copyfile(source / name, output / name)
    shutil.copyfile(source / "metadata.json", output / "source-v1-metadata.json")
    shutil.copyfile(source / "training-report.json", output / "source-v1-training-report.json")
    for name, value in (("metadata.json", metadata), ("config.json", config), ("scoring_spec.json", scoring_spec)):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    hashes = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
    with (output / "checksums.json").open("x") as stream:
        json.dump(hashes, stream, indent=2)
        stream.write("\n")
    if hashes["temporal.pt"] != checksums["temporal.pt"]:
        raise ValueError("Copie des poids divergente")
    return {"model_version": metadata["model_version"], "variant": variant,
            "amplitude_evidence": amplitude_evidence,
            "checksums_sha256": sha256_file(output / "checksums.json"),
            "temporal_sha256": hashes["temporal.pt"], "retrained": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--amplitude-evidence", action="store_true",
                        help="Opt-in variante C : neuf mesures C1 en texte, sans modifier les poids")
    args = parser.parse_args()
    print(json.dumps(make_reference(args.source, args.output, args.code_revision,
                                    amplitude_evidence=args.amplitude_evidence)), flush=True)
