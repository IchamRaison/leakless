#!/usr/bin/env python3
"""Audit CPU exhaustif de recouvrement audio ; aucun modèle, fit ou score qualité.

Pour chaque ancien WAV de 1 s et chaque nouvel enregistrement de 30 s :
corrélation Pearson normalisée pour TOUS les offsets entiers possibles. Le gain,
le DC et la polarité ne masquent pas une copie. Aucun rééchantillonnage/inversion
temporelle n'est testé ; une corrélation élevée ne prouve pas une provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import time
import wave
import zipfile

import numpy as np
import scipy
from scipy import fft

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tslm"))
from audit_text import load_expected_audio_md5, verify_audio_bytes  # noqa: E402
from export_run import FROZEN_AUDIT_SHA256  # noqa: E402
from harness.split_loader import FROZEN_SPLIT_SHA256  # noqa: E402
import prepare_external_holdout as external  # noqa: E402

SUSPECT_CORRELATION = 0.995  # Préannoncé avant lecture des signaux/qualités.
N_ORIGINAL = 1000
N_EXTERNAL = 122
WINDOW = 8000
RECORD_SAMPLES = 240000


def safe_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or ".." in path.parts or "\\" in relative
            or ":" in relative or "\x00" in relative):
        raise ValueError("Chemin de source non sûr")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError("Source absente ou sortant du dossier autorisé")
    return target


def bounded_json(path: Path, maximum=2_000_000):
    if path.stat().st_size > maximum:
        raise ValueError("Métadonnées trop volumineuses")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def decode_pcm16(data: bytes) -> np.ndarray:
    """Même frontière WAV que preprocessing.decode_wav, sans importer TimeNet."""
    if not 44 <= len(data) <= 128 * 1024:
        raise ValueError("Taille WAV invalide")
    with wave.open(io.BytesIO(data), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(),
                handle.getnframes(), handle.getcomptype()) != (1, 2, WINDOW, WINDOW, "NONE"):
            raise ValueError("WAV mono PCM16 8000 Hz/8000 points requis")
        pcm = handle.readframes(WINDOW)
    if len(pcm) != WINDOW * 2:
        raise ValueError("WAV tronqué")
    return np.frombuffer(pcm, dtype="<i2").astype("<f8") / 32768.0


def load_original(manifests: Path, data_root: Path):
    if external.sha256_file(manifests / "split_v2.csv") != FROZEN_SPLIT_SHA256:
        raise ValueError("Split historique modifié")
    expected_md5 = load_expected_audio_md5(manifests)
    with (manifests / "split_v2.csv").open(newline="") as handle:
        ids = [row["clip_id"] for row in csv.DictReader(handle)]
    if len(ids) != N_ORIGINAL or len(set(ids)) != N_ORIGINAL or set(ids) != set(expected_md5):
        raise ValueError("Couverture des 1000 originaux incorrecte")
    with (manifests / "split_v2_audit.csv").open(newline="") as handle:
        paths = {row["clip_id"]: row["path"] for row in csv.DictReader(handle)}
    signals, provenance = [], []
    for clip_id in sorted(ids):
        path = safe_path(data_root, paths[clip_id])
        if path.stat().st_size > 128 * 1024:
            raise ValueError("WAV trop volumineux")
        raw = path.read_bytes()
        verify_audio_bytes(raw, clip_id, expected_md5)
        signals.append(decode_pcm16(raw))
        provenance.append({"clip_id": clip_id, "wav_sha256": hashlib.sha256(raw).hexdigest(),
                           "decoded_float64_sha256": signal_hash(signals[-1])})
    return sorted(ids), np.stack(signals), provenance


def load_external(directory: Path):
    receipt = bounded_json(safe_path(directory, "receipt.json"))
    sums_path = safe_path(directory, "checksums.json")
    if (receipt.get("status") != "complete" or receipt.get("source_integrity_verified") is not True
            or external.sha256_file(sums_path) != receipt.get("checksums_sha256")):
        raise ValueError("Préparation externe absente/incomplète ou reçu incompatible")
    checksums = bounded_json(sums_path)
    required = {"metadata.json", "inputs.csv", "background_inputs.csv", "sources/Hydrophone.zip",
                "sources/author_converter.py"}
    array_names = {f"arrays/{external.stable_id('recording', member)}.npy"
                   for member in external.expected_recordings()}
    required |= array_names
    if len(array_names) != N_EXTERNAL or not required <= checksums.keys():
        raise ValueError("Inventaire externe incomplet")
    external.verify_file(safe_path(directory, "sources/Hydrophone.zip"), external.ARCHIVE_BYTES, external.ARCHIVE_SHA256)
    external.verify_file(safe_path(directory, "sources/author_converter.py"), external.CONVERTER_BYTES, external.CONVERTER_SHA256)
    for relative in sorted(required):
        path = safe_path(directory, relative)
        bound = RECORD_SAMPLES * 8 + 10000 if relative in array_names else 2_000_000
        if not relative.startswith("sources/") and path.stat().st_size > bound:
            raise ValueError("Fichier préparé trop volumineux")
        if external.sha256_file(path) != checksums[relative]:
            raise ValueError(f"Empreinte externe incorrecte : {relative}")
    metadata = bounded_json(directory / "metadata.json")
    decoder = metadata.get("decoder", {})
    if (metadata.get("dataset_doi") != external.DOI or metadata.get("role") != "external_holdout_only"
            or decoder.get("dtype") != "<f8" or decoder.get("sample_rate") != WINDOW
            or decoder.get("crop_samples") != RECORD_SAMPLES
            or decoder.get("scale") != "signed_int32 / 2147483648"):
        raise ValueError("Décodeur externe incompatible")
    seen_ids, coverage = set(), {name: set() for name in array_names}
    for relative in ("inputs.csv", "background_inputs.csv"):
        if (directory / relative).stat().st_size > 2_000_000:
            raise ValueError("Manifeste d'entrée trop volumineux")
        with (directory / relative).open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != external.INPUT_FIELDS:
                raise ValueError("Contrat d'entrée externe incompatible")
            for row in reader:
                name, start = row["array_path"], int(row["start_sample"])
                if (name not in coverage or row["clip_id"] in seen_ids or start in coverage[name]
                        or int(row["n_samples"]) != WINDOW or int(row["sample_rate"]) != WINDOW):
                    raise ValueError("Couverture/identifiants externes incohérents")
                coverage[name].add(start)
                seen_ids.add(row["clip_id"])
    if any(starts != set(range(0, RECORD_SAMPLES, WINDOW)) for starts in coverage.values()):
        raise ValueError("Couverture des trente secondes incomplète")
    records = []
    # Revalider les tableaux contre les RAW de l'archive épinglée : le reçu
    # autodéclaré ne suffit pas à garantir le contenu numérique de la préparation.
    with zipfile.ZipFile(directory / "sources/Hydrophone.zip") as archive:
        expected = external.expected_recordings()
        external.validate_members(archive, expected)
        for member in sorted(expected):
            recording_id = external.stable_id("recording", member)
            path = directory / f"arrays/{recording_id}.npy"
            signal = np.load(path, allow_pickle=False, mmap_mode="r")
            if signal.shape != (RECORD_SAMPLES,) or signal.dtype != np.dtype("<f8") or not np.isfinite(signal).all():
                raise ValueError("Tableau externe incompatible/non fini")
            if not np.array_equal(signal, external.decode_raw(archive.read(member))):
                raise ValueError("Tableau externe différent du RAW original vérifié")
            records.append((recording_id, path))
    return records, {"checksums_sha256": receipt["checksums_sha256"],
                     "archive_sha256": external.ARCHIVE_SHA256,
                     "metadata_sha256": checksums["metadata.json"]}


def signal_hash(signal):
    values = np.asarray(signal, dtype="<f8").copy()
    values[values == 0] = 0  # Canonicaliser -0.0, sans quantifier.
    return hashlib.sha256(values.tobytes()).hexdigest()


def centered_unit(signal):
    values = np.asarray(signal, dtype=np.float64) - np.mean(signal)
    norm = np.sqrt(np.sum(values * values))
    return None if norm == 0 else values / norm


def direct_correlation(left, right):
    left, right = centered_unit(left), centered_unit(right)
    if left is None or right is None:
        return None
    value = float(np.sum(left * right))
    if not np.isfinite(value) or abs(value) > 1 + 1e-12:
        raise ValueError("Corrélation directe numériquement invalide")
    return min(1.0, max(-1.0, value))  # Bornes mathématiques, arrondi <= 1e-12 seulement.


def query_spectra(queries, record_length, workers=1):
    queries = np.asarray(queries, dtype=np.float64)
    if (queries.ndim != 2 or queries.shape[0] < 1 or queries.shape[1] < 2
            or queries.shape[1] > record_length or not np.isfinite(queries).all()):
        raise ValueError("Requêtes incompatibles/non finies")
    centered = queries - queries.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(centered * centered, axis=1))
    nfft = fft.next_fast_len(record_length + queries.shape[1] - 1)
    spectra = fft.rfft(centered[:, ::-1], n=nfft, axis=1, workers=workers)
    return spectra, norms, nfft


def sliding_correlations(queries, record, prepared=None, workers=1):
    """Maximum absolu par paire sur tous les offsets ; aucun préfiltrage."""
    queries, record = np.asarray(queries, dtype=np.float64), np.asarray(record, dtype=np.float64)
    if record.ndim != 1 or not np.isfinite(record).all():
        raise ValueError("Enregistrement incompatible/non fini")
    spectra, norms, nfft = prepared if prepared is not None else query_spectra(queries, len(record), workers)
    width = queries.shape[1]
    centered = record - record.mean()
    sums = np.r_[0.0, np.cumsum(centered)]
    squares = np.r_[0.0, np.cumsum(centered**2)]
    window_sum = sums[width:] - sums[:-width]
    window_squares = squares[width:] - squares[:-width]
    energy = window_squares - window_sum**2 / width
    # Borne conservative de cancellation des sommes cumulées. Les fenêtres
    # constantes/quasi nulles sont non évaluables, jamais dotées d'une similarité 0.
    tolerance = 64 * np.finfo(np.float64).eps * max(float(squares[-1]), np.finfo(float).tiny)
    changes = np.r_[0, np.cumsum(np.diff(record) != 0)]
    varying = changes[width - 1:] - changes[:len(record) - width + 1] > 0
    valid = varying & (energy > tolerance)
    record_spectrum = fft.rfft(centered, n=nfft, workers=workers)
    convolution = fft.irfft(spectra * record_spectrum[None, :], n=nfft, axis=1,
                           workers=workers)[:, width - 1:len(record)]
    denominator = np.sqrt(np.maximum(energy, 0))
    # Normalisation en place du lot : éviter plusieurs copies de 232001 valeurs
    # par paire. Même Pearson float64 et même recherche exhaustive.
    np.divide(convolution, denominator[None, :], out=convolution, where=valid[None, :])
    np.divide(convolution, np.where(norms > 0, norms, 1)[:, None], out=convolution)
    np.abs(convolution, out=convolution)
    convolution[:, ~valid] = -np.inf
    result = []
    for index, query in enumerate(queries):
        if norms[index] == 0 or not valid.any():
            result.append({"correlation": None, "max_abs_correlation": None, "offset_samples": None,
                           "unassessable_offsets": len(valid) if norms[index] == 0 else int((~valid).sum()),
                           "reason": "constant_query" if norms[index] == 0 else "no_stable_variance_window"})
            continue
        correlation = convolution[index]
        offset = int(np.argmax(correlation))
        checked = direct_correlation(query, record[offset:offset + width])
        if checked is None or abs(abs(checked) - float(correlation[offset])) > 1e-5:
            raise ValueError("Maximum FFT instable : contrôle direct divergent, audit interrompu")
        result.append({"correlation": checked, "max_abs_correlation": abs(checked),
                       "offset_samples": offset, "unassessable_offsets": int((~valid).sum()), "reason": None})
    return result


def exact_aligned_matches(ids, originals, records):
    indexes = {"decoded_float64": {}, "centered_unit_float64": {}}
    for clip_id, signal in zip(ids, originals):
        for kind, values in (("decoded_float64", signal), ("centered_unit_float64", centered_unit(signal))):
            if values is not None:
                indexes[kind].setdefault(signal_hash(values), []).append(clip_id)
    matches = []
    for recording_id, path in records:
        signal = np.load(path, allow_pickle=False, mmap_mode="r")
        for start in range(0, len(signal) - WINDOW + 1, WINDOW):
            segment = signal[start:start + WINDOW]
            for kind, values in (("decoded_float64", segment), ("centered_unit_float64", centered_unit(segment))):
                if values is not None:
                    for clip_id in indexes[kind].get(signal_hash(values), []):
                        matches.append({"clip_id": clip_id, "recording_id": recording_id,
                                        "offset_samples": start, "match_kind": kind})
    return matches


def audit(args):
    if not re.fullmatch(r"[0-9a-f]{40}", args.code_revision):
        raise ValueError("Révision de code complète SHA40 requise")
    if not 1 <= args.batch_size <= 32 or not 1 <= args.workers <= 8:
        raise ValueError("batch-size doit être1..32, workers1..8")
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("La sortie doit être neuve, même après un audit interrompu")
    if not output.parent.is_dir():
        raise ValueError("Le parent de sortie doit exister")
    ids, originals, old_provenance = load_original(args.manifests, args.data_root)
    records, new_provenance = load_external(args.external)
    output.mkdir()
    write_json(output / "provenance.json", {
        "code_revision": args.code_revision, "script_sha256": external.sha256_file(Path(__file__)),
        "split_sha256": FROZEN_SPLIT_SHA256, "audit_manifest_sha256": FROZEN_AUDIT_SHA256,
        "original_files": old_provenance, "external": new_provenance,
        "numpy": np.__version__, "scipy": scipy.__version__, "batch_size": args.batch_size,
        "workers": args.workers, "threshold": SUSPECT_CORRELATION,
        "helper_sha256": {relative: external.sha256_file(ROOT / relative) for relative in (
            "scripts/eval/prepare_external_holdout.py", "scripts/tslm/audit_text.py",
            "scripts/tslm/export_run.py", "scripts/eval/harness/split_loader.py")},
        "labels_used": False, "model_loaded": False, "quality_metrics_computed": False,
        "offset_range_inclusive": [0, RECORD_SAMPLES - WINDOW], "sample_rate": WINDOW,
    })
    exact = exact_aligned_matches(ids, originals, records)
    write_json(output / "exact_matches.json", exact)
    started = time.perf_counter()
    best = {clip_id: None for clip_id in ids}
    suspects, unassessable, pairs, partial_pairs, unavailable_offsets = 0, 0, 0, 0, 0
    with (output / "pairs.jsonl").open("x") as pairs_log, (output / "suspects.jsonl").open("x") as suspect_log:
        for begin in range(0, len(ids), args.batch_size):
            batch = originals[begin:begin + args.batch_size]
            spectra = query_spectra(batch, RECORD_SAMPLES, args.workers)
            for recording_id, path in records:
                signal = np.load(path, allow_pickle=False, mmap_mode="r")
                results = sliding_correlations(batch, signal, spectra, args.workers)
                for clip_id, result in zip(ids[begin:begin + len(batch)], results):
                    row = {"clip_id": clip_id, "recording_id": recording_id, **result}
                    row["suspect"] = result["max_abs_correlation"] is not None and result["max_abs_correlation"] >= SUSPECT_CORRELATION
                    pairs_log.write(json.dumps(row, allow_nan=False) + "\n")
                    pairs += 1
                    partial_pairs += row["unassessable_offsets"] > 0
                    unavailable_offsets += row["unassessable_offsets"]
                    if row["max_abs_correlation"] is None:
                        unassessable += 1
                    elif best[clip_id] is None or row["max_abs_correlation"] > best[clip_id]["max_abs_correlation"]:
                        best[clip_id] = row
                    if row["suspect"]:
                        suspects += 1
                        suspect_log.write(json.dumps(row, allow_nan=False) + "\n")
            pairs_log.flush()
            suspect_log.flush()
            print(json.dumps({"originals_complete": min(begin + len(batch), len(ids)),
                              "originals_expected": len(ids), "pairs_completed": pairs,
                              "pairs_expected": len(ids) * len(records),
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
    write_json(output / "best_per_original.json", best)
    summary = {
        "status": "complete", "pairs_examined": pairs, "pairs_expected": len(ids) * len(records),
        "offsets_per_pair": RECORD_SAMPLES - WINDOW + 1,
        "suspect_pairs": suspects, "unassessable_pairs": unassessable,
        "pairs_with_unassessable_offsets": partial_pairs,
        "unassessable_offsets_total": unavailable_offsets,
        "exact_aligned_matches": len(exact), "threshold": SUSPECT_CORRELATION,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": "Candidats de recouvrement, pas preuve automatique de provenance ou indépendance",
        "automatic_exclusions": False,
        "limitations": ["Pas de rééchantillonnage, inversion temporelle ni filtrage testés",
                        "Offsets entiers uniquement ; changements de vitesse/fraction de sample non couverts",
                        "Fenêtres de variance nulle ou numériquement non résoluble non évaluables",
                        "Absence de candidat au seuil ne prouve pas des sessions indépendantes"],
    }
    summary["files_sha256"] = {p.name: external.sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, default=ROOT / "manifests")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    try:
        audit(args)
    except (ValueError, OSError, wave.Error, EOFError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Audit interrompu sans verdict d'indépendance : {error}\n")


if __name__ == "__main__":
    main()
