"""Vérification stdlib, sans modèle ni écriture, des 646 assets train de H100-2."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

REFERENCES = {
    "docs/evidence/tslm-v2/campaign-c13fd47/preregistration.json":
        "5c5e0b4bbed6b973853e53b658350815b2d963547b68d0a08a9ade50c28b1359",
    "docs/evidence/tslm-v2/train-diagnostic-001/folds.json":
        "5c2bea733f8f2a2dc525b9738a5aa40ae3ce220cf6d76d712cab984afb4d5efb",
    "manifests/split_v2.csv": "7a8716a35284434292314c10da58663e9f848be60edf18db0f98ef9d63d17896",
    "manifests/split_v2_audit.csv": "1a3bd3c18ad6d886d42ecc85ba3a5cceaa45a9084102fc112b53e66782e29d61"}


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def aggregate(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify(root, code):
    root, code = root.resolve(strict=True), code.resolve(strict=True)
    for name, expected in REFERENCES.items():
        if digest(code / name) != expected:
            raise ValueError(f"Référence gelée modifiée : {name}")
    registration = json.loads((code / next(iter(REFERENCES))).read_text())
    identity, paths = registration["identity"], registration["paths"]
    folds = json.loads((code / "docs/evidence/tslm-v2/train-diagnostic-001/folds.json").read_text())
    relative = lambda path: str(Path(path).relative_to("/home/hicham/pipe-v0"))
    base = relative(paths["base"])
    expected = {f"{base}/{name}": sha for name, sha in identity["base_sha256"].items()}
    expected.update({f"{relative(paths['prepared'])}/train.npz": identity["cache_sha256"]["train"],
        f"{relative(paths['prepared'])}/preparation.json": identity["preparation_sha256"],
        f"{relative(paths['previous_prepared'])}/train.npz": folds["cache_train_sha256"],
        relative(paths["gate_a"]): identity["gate_sha256"]["A"],
        relative(paths["gate_c"]): identity["gate_sha256"]["C"]})
    with (code / "manifests/split_v2.csv").open(newline="") as stream:
        train_ids = {r["clip_id"] for r in csv.DictReader(stream) if r["fold"] == "train"}
    with (code / "manifests/split_v2_audit.csv").open(newline="") as stream:
        wavs = {f"{relative(paths['data_root'])}/{r['path']}": r["md5"]
                for r in csv.DictReader(stream) if r["clip_id"] in train_ids}
    if len(train_ids) != 598 or len(wavs) != 598 or len(expected) != 48 or set(expected) & set(wavs):
        raise ValueError("Inventaire autre que 43 fichiers Qwen, 5 caches/gates et 598 WAV train")
    wanted, actual = set(expected) | set(wavs), set()
    allowed_dirs = {str(parent) for name in wanted for parent in PurePosixPath(name).parents}
    for name in wanted:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Chemin d'asset non sûr")
    for folder in ("artifacts", "quality-v2-001", "data"):
        scope = root / folder
        for path in [scope, *scope.rglob("*")]:
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise ValueError(f"Asset absent, spécial ou lien symbolique : {path}")
            if path.is_dir() and str(path.relative_to(root)) not in allowed_dirs:
                raise ValueError(f"Dossier d'assets supplémentaire : {path}")
            if path.is_file():
                actual.add(str(path.relative_to(root)))
    if actual != wanted:
        raise ValueError(f"Inventaire non exact : {len(wanted-actual)} manquants, {len(actual-wanted)} supplémentaires")
    hashes = {}
    for name in sorted(wanted):
        hashes[name] = digest(root / name)
        if name in expected and hashes[name] != expected[name]:
            raise ValueError(f"SHA256 divergent : {name}")
        if name in wavs and digest(root / name, "md5") != wavs[name]:
            raise ValueError(f"MD5 source WAV divergent : {name}")
    return {"status": "PASS", "asset_files": 646, "base_files": 43, "cache_and_gate_files": 5,
        "train_wavs": 598, "train_wavs_md5_verified": 598,
        "assets_sha256": aggregate(hashes), "train_wavs_sha256": aggregate({k: hashes[k] for k in wavs}),
        "references_sha256": aggregate(REFERENCES), "model_loaded": False,
        "validation_test_external_data_opened": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.root, args.code), sort_keys=True))
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False))
        raise SystemExit(1)
