#!/usr/bin/env python3
"""Prépare le holdout Aghashahi réservé, sans modèle, score ni métrique.

Les seules entrées destinées au modèle sont inputs.csv et arrays/*.npy.
Les labels, noms d'origine et groupes restent dans des fichiers séparés.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import urllib.request
import zipfile

import numpy as np


DATASET_URL = "https://data.mendeley.com/datasets/tbrnp6vrnj/1"
DOI = "10.17632/tbrnp6vrnj.1"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ARCHIVE_URL = (
    "https://data.mendeley.com/public-files/datasets/tbrnp6vrnj/files/"
    "db8d1475-7cb4-4c60-b9e2-7d47a7d95971/file_downloaded"
)
ARCHIVE_BYTES = 63_994_754
ARCHIVE_SHA256 = "d070b62306e1146072bc6929c64203af95e2a1df21a52ce6d93fe1a75395a38d"
CONVERTER_URL = (
    "https://data.mendeley.com/public-files/datasets/tbrnp6vrnj/files/"
    "dc9d459b-ec6d-4a64-ab69-d2bac7396a5c/file_downloaded"
)
CONVERTER_BYTES = 1722
CONVERTER_SHA256 = "3dd30f364804d4bfe0bea70ab057cbe5fba73083dcaf2dab3328208c63c0af02"
SAMPLE_RATE = 8000
CROP_SAMPLES = 240_000  # Le convertisseur des auteurs applique head(240000).
WINDOW_SAMPLES = 8000
MAX_RAW_BYTES = 2_000_000  # Inventaire audité : maximum 1 961 840 octets.
INPUT_FIELDS = ["clip_id", "array_path", "start_sample", "n_samples", "sample_rate"]
TARGET_FIELDS = ["clip_id", "recording_id", "condition_group_id", "label"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected_size: int, expected_hash: str) -> None:
    if path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
        raise ValueError(f"Taille/SHA-256 de la source incorrect : {path.name}")


def download(url: str, destination: Path, expected_size: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "PIPE-dataset-audit/1.0"})
    count = 0
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("xb") as handle:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            count += len(block)
            if count > expected_size:
                raise ValueError("Source plus volumineuse que l'inventaire gelé")
            handle.write(block)
    if count != expected_size:
        raise ValueError("Téléchargement incomplet")


def expected_recordings() -> dict[str, dict]:
    """Grammaire vérifiée dans le répertoire central officiel : 120 + 2 RAW."""
    records = {}
    topologies = {"BR": "Branched", "LO": "Looped"}
    classes = {"CC": "Circumferential Crack", "GL": "Gasket Leak",
               "LC": "Longitudinal Crack", "NL": "No-leak", "OL": "Orifice Leak"}
    conditions = ("0.18 LPS_N", "0.47 LPS_N", "ND_NN", "ND_N",
                  "Transient_NN", "Transient_N")
    for topology, folder in topologies.items():
        for class_code, class_folder in classes.items():
            for condition in conditions:
                group = f"{topology}_{class_code}_{condition}"
                for sensor in ("H1", "H2"):
                    member = f"Hydrophone/{folder}/{class_folder}/{group}_{sensor}.raw"
                    records[member] = {
                        "subset": "primary", "label": "no_leak" if class_code == "NL" else "leak",
                        "condition_key": group, "topology": topology,
                        "leak_type": class_code, "condition": condition, "sensor": sensor,
                    }
    for sensor in ("H1", "H2"):
        records[f"Hydrophone/Background Noise/Background Noise_{sensor}.raw"] = {
            "subset": "background", "label": "environmental_noise",
            "condition_key": "background_noise", "sensor": sensor,
        }
    return records


def validate_members(archive: zipfile.ZipFile, expected: dict[str, dict]) -> None:
    seen = set()
    files = set()
    directories = {str(parent) + "/" for name in expected
                   for parent in PurePosixPath(name).parents if str(parent) != "."}
    for entry in archive.infolist():
        name = entry.filename
        path = PurePosixPath(name)
        if (name in seen or "\\" in name or "\x00" in entry.orig_filename
                or path.is_absolute() or ".." in path.parts or ":" in name):
            raise ValueError(f"Chemin ZIP dangereux ou dupliqué : {name!r}")
        seen.add(name)
        mode = entry.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR) or entry.flag_bits & 1:
            raise ValueError(f"Entrée ZIP spéciale/chiffrée interdite : {name!r}")
        if entry.is_dir():
            if name not in directories or entry.file_size != 0 or kind == stat.S_IFREG:
                raise ValueError(f"Dossier ZIP inattendu : {name!r}")
            continue
        if kind == stat.S_IFDIR or name not in expected:
            raise ValueError(f"Fichier ZIP inattendu : {name!r}")
        if not CROP_SAMPLES * 4 <= entry.file_size <= MAX_RAW_BYTES or entry.file_size % 4:
            raise ValueError(f"Longueur RAW invalide : {name!r}")
        files.add(name)
    if files != set(expected):
        raise ValueError(f"Archive incomplète : {len(files)}/{len(expected)} RAW attendus")


def decode_raw(data: bytes) -> np.ndarray:
    if len(data) % 4 or not CROP_SAMPLES * 4 <= len(data) <= MAX_RAW_BYTES:
        raise ValueError("RAW incompatible : PCM32 LE, au moins 30 secondes requis")
    # Division exacte en float64 : pas de clipping, gain adaptatif ni conversion PCM16.
    signal = np.frombuffer(data, dtype="<i4", count=CROP_SAMPLES).astype("<f8")
    signal /= 2**31
    return signal


def stable_id(kind: str, key: str) -> str:
    return kind + "_" + hashlib.sha256(f"{DOI}\0{key}".encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(output: Path, archive_path: Path | None, converter_path: Path | None,
            code_revision: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", code_revision):
        raise ValueError("--code-revision doit contenir 40 caractères hexadécimaux minuscules")
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Le dossier de sortie doit être neuf, même si l'ancien est incomplet")
    if not output.parent.is_dir():
        raise ValueError("Le parent de la sortie doit déjà exister")
    expected = expected_recordings()
    with tempfile.TemporaryDirectory(prefix=".pipe-external-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "sources").mkdir()
        (stage / "arrays").mkdir()
        pinned_archive = stage / "sources/Hydrophone.zip"
        pinned_converter = stage / "sources/author_converter.py"
        for source, destination, url, size, digest in (
            (archive_path, pinned_archive, ARCHIVE_URL, ARCHIVE_BYTES, ARCHIVE_SHA256),
            (converter_path, pinned_converter, CONVERTER_URL, CONVERTER_BYTES, CONVERTER_SHA256),
        ):
            if source is None:
                download(url, destination, size)
            else:
                if not source.is_file() or source.stat().st_size != size:
                    raise ValueError(f"Source locale absente ou taille incorrecte : {source}")
                shutil.copyfile(source, destination)
            verify_file(destination, size, digest)

        inputs = {"primary": [], "background": []}
        targets = {"primary": [], "background": []}
        recordings = []
        raw_hashes = set()
        signal_hashes = set()
        with zipfile.ZipFile(pinned_archive) as archive:
            validate_members(archive, expected)
            for member, context in sorted(expected.items()):
                raw = archive.read(member)  # CRC vérifié ; au plus 2 Mo par entrée.
                raw_hash = hashlib.sha256(raw).hexdigest()
                signal = decode_raw(raw)
                signal_hash = hashlib.sha256(signal.tobytes()).hexdigest()
                if raw_hash in raw_hashes or signal_hash in signal_hashes:
                    raise ValueError(f"Enregistrement dupliqué avant/après troncature : {member}")
                raw_hashes.add(raw_hash)
                signal_hashes.add(signal_hash)
                recording_id = stable_id("recording", member)
                group_id = stable_id("condition", context["condition_key"])
                array_path = f"arrays/{recording_id}.npy"
                np.save(stage / array_path, signal, allow_pickle=False)
                recordings.append({
                    **context, "recording_id": recording_id, "condition_group_id": group_id,
                    "archive_member": member, "raw_sha256": raw_hash,
                    "raw_bytes": len(raw), "raw_n_samples": len(raw) // 4,
                    "prepared_samples": len(signal), "prepared_signal_sha256": signal_hash,
                    "array_path": array_path, "array_sha256": sha256_file(stage / array_path),
                })
                for start in range(0, CROP_SAMPLES, WINDOW_SAMPLES):
                    clip_id = stable_id("clip", f"{member}\0{start}\0{WINDOW_SAMPLES}")
                    inputs[context["subset"]].append({
                        "clip_id": clip_id, "array_path": array_path, "start_sample": start,
                        "n_samples": WINDOW_SAMPLES, "sample_rate": SAMPLE_RATE,
                    })
                    targets[context["subset"]].append({
                        "clip_id": clip_id, "recording_id": recording_id,
                        "condition_group_id": group_id, "label": context["label"],
                    })

        for subset, prefix in (("primary", ""), ("background", "background_")):
            ids = [row["clip_id"] for row in inputs[subset]]
            if len(ids) != len(set(ids)):
                raise ValueError("Identifiants de fenêtre dupliqués")
            write_csv(stage / f"{prefix}inputs.csv", INPUT_FIELDS, inputs[subset])
            write_csv(stage / f"{prefix}targets.csv", TARGET_FIELDS, targets[subset])
        write_json(stage / "recordings.json", recordings)
        metadata = {
            "schema": "pipe.external-holdout.v1", "dataset_doi": DOI,
            "dataset_url": DATASET_URL, "dataset_version": 1,
            "authors": ["Mohsen Aghashahi", "Lina Sela", "M. Katherine Banks"],
            "license": "CC-BY-4.0", "license_url": LICENSE_URL,
            "role": "external_holdout_only", "training_permitted": False,
            "model_scoring_performed": False, "threshold_fitting_permitted": False,
            "code_revision": code_revision, "script_sha256": sha256_file(Path(__file__)),
            "numpy_version": np.__version__,
            "archive_url": ARCHIVE_URL, "archive_bytes": ARCHIVE_BYTES,
            "archive_sha256": ARCHIVE_SHA256, "converter_url": CONVERTER_URL,
            "converter_sha256": CONVERTER_SHA256,
            "decoder": {"format": "RAW", "subtype": "PCM_32", "endianness": "little",
                        "channels": 1, "sample_rate": SAMPLE_RATE, "dtype": "<f8",
                        "scale": "signed_int32 / 2147483648", "crop_start_sample": 0,
                        "crop_samples": CROP_SAMPLES, "crop_source": "author_converter.py head(240000)",
                        "window_samples": WINDOW_SAMPLES, "stride_samples": WINDOW_SAMPLES,
                        "resampling": "none", "clipping": "none", "pcm16_requantization": False,
                        "model_preprocessing_applied": False},
            "subsets": {key: {
                "recordings": sum(r["subset"] == key for r in recordings),
                "windows": len(inputs[key]),
                "heuristic_condition_groups": len({r["condition_group_id"] for r in targets[key]}),
                "window_labels": {label: sum(r["label"] == label for r in targets[key])
                                  for label in sorted({r["label"] for r in targets[key]})},
            } for key in inputs},
            "grouping": "topology + leak_type + demand/noise condition; H1/H2 and all windows together",
            "real_session_ids_available": False,
            "cross_corpus_duplicate_audit": "pending; do not claim verified independence from development",
            "continuous_leak_onsets_available": False,
            "excluded_duplicate_doi": "10.17632/xw44wv2g88.2",
            "input_contract": "Only inputs.csv and arrays; targets/recordings are evaluator-only",
        }
        write_json(stage / "metadata.json", metadata)
        hashes = {str(path.relative_to(stage)): sha256_file(path)
                  for path in sorted(stage.rglob("*")) if path.is_file()}
        write_json(stage / "checksums.json", hashes)
        receipt = {
            "status": "complete", "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "checksums_sha256": sha256_file(stage / "checksums.json"),
            "source_integrity_verified": True, "source_code_executed": False,
            "raw_hashes_unique": True, "cropped_signal_hashes_unique": True,
            "model_loaded": False, "scores_computed": False,
        }
        write_json(stage / "receipt.json", receipt)
        # Réservation exclusive : aucun remplacement d'un dossier existant, même vide.
        output.mkdir()
        for path in sorted(stage.iterdir(), key=lambda p: (p.name == "receipt.json", p.name)):
            path.rename(output / path.name)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="Archive locale officielle, vérifiée puis copiée")
    source.add_argument("--download", action="store_true", help="Télécharge seulement les hydrophones, 64 Mo")
    parser.add_argument("--converter", type=Path, help="Convertisseur officiel local ; sinon téléchargement 1722 octets")
    parser.add_argument("--output", type=Path, required=True, help="Dossier neuf ; parent existant")
    parser.add_argument("--code-revision", required=True, help="SHA40 réel du code transféré, même sans .git")
    args = parser.parse_args()
    try:
        metadata = prepare(args.output, args.archive, args.converter, args.code_revision)
    except (ValueError, FileExistsError, OSError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Préparation refusée : {error}\n")
    print(json.dumps({"output": str(args.output), "subsets": metadata["subsets"],
                      "scores_computed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
