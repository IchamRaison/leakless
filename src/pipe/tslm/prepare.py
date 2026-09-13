"""Téléchargement intègre, WAV originaux, TimeNet et cache numérique séparé par fold."""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen

import libarchive
import numpy as np
from leakless_acoustic.connector import CONNECTOR, _normalise
from timenet.reader.reader import TimeFReader
from timenet.registry.version import DatasetVersion
from timenet.writer.writer import TimeFWriter

from pipe.tslm.preprocessing import CANONICAL_VERSION, VERSION, band_series, decode_wav, measured_band, preprocess_audio

ARCHIVES = {
    "leak acoustic data.rar": "2ec7cef54b0cbc09adab0047761fe9b2",
    "no leak acoustic data.rar": "d6b9fcd8a730f33fa6c155191b32f2a2",
    "environmental noise.rar": "6080966b540eeeb1b5e61a2cb1cc2639",
}
MANIFEST_HASHES = {
    "split_v2.csv": "7a8716a35284434292314c10da58663e9f848be60edf18db0f98ef9d63d17896",
    "split_v2_audit.csv": "1a3bd3c18ad6d886d42ecc85ba3a5cceaa45a9084102fc112b53e66782e29d61",
}
PROTOCOL = "split_v2_binary_with_noise_v0"


def safe_member_path(name: str, root: Path) -> Path:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts or ":" in path.parts[0]:
        raise ValueError("Chemin d'archive non sûr")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Chemin sortant du dossier d'extraction")
    return target


def download_archives(data_dir: Path) -> dict:
    raw_dir, extract_dir = data_dir / "raw", data_dir / "extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    receipt = {}
    for name, md5 in ARCHIVES.items():
        dest = raw_dir / name
        if not dest.exists():
            url = f"https://zenodo.org/api/records/18631450/files/{quote(name)}/content"
            with urlopen(url, timeout=60) as response:
                payload = response.read(16 * 1024 * 1024 + 1)
            if len(payload) > 16 * 1024 * 1024 or hashlib.md5(payload).hexdigest() != md5:
                raise ValueError(f"Archive téléchargée invalide : {name}")
            with dest.open("xb") as output:
                output.write(payload)
        payload = dest.read_bytes()
        if hashlib.md5(payload).hexdigest() != md5:
            raise ValueError(f"Archive existante invalide : {name} ; conservée pour inspection")
        total, count, ignored = 0, 0, []
        # Extraction contrôlée ; ne pas confier les chemins de l'archive à extractall.
        with libarchive.file_reader(str(dest)) as entries:
            for entry in entries:
                path = safe_member_path(entry.pathname, extract_dir)
                if entry.isdir:
                    continue
                if entry.isfile and path.name == "desktop.ini":
                    ignored.append(entry.pathname)
                    continue  # Métadonnée Windows, conservée dans l'archive originale.
                if not entry.isfile or path.suffix.lower() != ".wav":
                    raise ValueError("Entrée non WAV ou lien dans l'archive")
                chunks, size = [], 0
                for block in entry.get_blocks():
                    size += len(block)
                    total += len(block)
                    if size > 128 * 1024 or total > 64 * 1024 * 1024:
                        raise ValueError("Archive décompressée trop grande")
                    chunks.append(block)
                wav_bytes = b"".join(chunks)
                decode_wav(wav_bytes)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    if path.read_bytes() != wav_bytes:
                        raise ValueError("WAV existant différent, aucun écrasement")
                else:
                    with path.open("xb") as output:
                        output.write(wav_bytes)
                count += 1
        receipt[name] = {"md5": md5, "sha256": hashlib.sha256(payload).hexdigest(),
                         "wav_count": count, "ignored_metadata": ignored}
        print(json.dumps({"archive": name, **receipt[name]}), flush=True)
    return receipt


def load_manifest(manifest_dir: Path) -> list[dict]:
    for name, expected in MANIFEST_HASHES.items():
        if hashlib.sha256((manifest_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Le manifeste gelé a changé : {name}")
    with (manifest_dir / "split_v2.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (manifest_dir / "split_v2_audit.csv").open() as stream:
        paths = {row["clip_id"]: row["path"] for row in csv.DictReader(stream)}
    if len(rows) != 1000 or len({r["clip_id"] for r in rows}) != 1000:
        raise ValueError("Effectif/IDs du split invalide")
    folds = collections.defaultdict(set)
    for row in rows:
        folds[row["group_id"]].add(row["fold"])
        row["path"] = paths[row["clip_id"]]
        if row["fold"] not in ("train", "val", "test") or row["label"] not in ("leak", "no_leak"):
            raise ValueError("Fold ou classe inconnue")
    if any(len(fold) != 1 for fold in folds.values()):
        raise ValueError("Groupe à cheval sur les folds")
    return rows


def write_timef(refs: list[dict], target: Path) -> Path:
    connector = CONNECTOR()
    metadata = connector.metadata()
    version_dir = target / metadata.dataset_id / str(metadata.dataset_version)
    if (version_dir / "manifest.json").exists():
        return version_dir
    if version_dir.exists() and any(version_dir.iterdir()):
        raise ValueError("Dossier TimeF incomplet existant ; choisir un nouveau chemin")
    dataset = connector.convert(refs)
    writer = TimeFWriter(root=target, dataset=dataset)
    writer.write()
    writer.close()
    return version_dir


def prepare(data_dir: Path, manifest_dir: Path, output: Path) -> dict:
    rows = load_manifest(manifest_dir)
    archives = download_archives(data_dir)
    data_root = data_dir / "extracted"
    os.environ["LEAKLESS_DATA_ROOT"] = str(data_root.resolve())
    os.environ["LEAKLESS_MANIFEST_DIR"] = str(manifest_dir.resolve())
    indexed = {r["clip_id"]: r for r in rows}
    signal_folds = collections.defaultdict(set)
    for row in rows:
        raw = safe_member_path(row["path"], data_root).read_bytes()
        pcm = decode_wav(raw).astype("<i2").tobytes()
        signal_folds[hashlib.sha256(pcm).hexdigest()].add(row["fold"])
    if any(len(folds) > 1 for folds in signal_folds.values()):
        raise ValueError("Doublon de signal décodé entre folds")
    output.mkdir(parents=True, exist_ok=True)
    first = sorted((r for r in rows if r["fold"] == "train"), key=lambda r: r["clip_id"])[0]
    # Prouver un exemple réel avant de convertir le reste.
    single_dir = write_timef([first], output / "timef-one")
    with TimeFReader(DatasetVersion.open_local(single_dir)) as reader:
        reader.verify()
        record = next(reader.iter_records())
        expected = _normalise(decode_wav((data_root / first["path"]).read_bytes()))
        observed = record.time_series[0].to_numpy()
        np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)
        assert record.subject_ids == (first["group_id"],)
        series = band_series(observed)
        assert series.shape == (4, 64) and np.isfinite(series).all()
        first_report = {"clip_id": first["clip_id"], "group_id": first["group_id"],
                        "fold": first["fold"], "shape": list(series.shape),
                        "rms": float(np.sqrt(np.mean(observed ** 2))),
                        "dominant_band_hz": measured_band(series),
                        "max_roundtrip_error": float(np.max(np.abs(observed - expected)))}
        print(json.dumps({"first_example_verified": first_report}), flush=True)
    timef_dir = write_timef(rows, output / "timef")
    cache = {fold: {"ids": [], "series": []} for fold in ("train", "val", "test")}
    with TimeFReader(DatasetVersion.open_local(timef_dir)) as reader:
        reader.verify()
        targets = {task.record_ids[0]: task.target for task in reader.tasks}
        seen = set()
        for record in reader.iter_records():
            row = indexed[record.record_id]
            if record.record_id in seen or record.subject_ids != (row["group_id"],) or targets[record.record_id] != row["label"]:
                raise ValueError("Désaccord manifeste/TimeF")
            seen.add(record.record_id)
            x = record.time_series[0].to_numpy()
            expected = _normalise(decode_wav((data_root / row["path"]).read_bytes()))
            np.testing.assert_allclose(x, expected, rtol=1e-6, atol=1e-6)
            cache[row["fold"]]["ids"].append(record.record_id)
            cache[row["fold"]]["series"].append(band_series(x))
        if seen != set(indexed):
            raise ValueError("Enregistrements TimeF manquants")
    for fold, item in cache.items():
        np.savez_compressed(output / f"{fold}.npz", ids=np.array(item["ids"]),
                            series=np.stack(item["series"]), preprocessing_version=VERSION)
    report = {"protocol": PROTOCOL, "preprocessing_version": VERSION, "archives": archives,
              "manifest_sha256": MANIFEST_HASHES["split_v2.csv"], "records": len(rows),
              "groups": len({r["group_id"] for r in rows}),
              "fold_counts": {fold: len(item["ids"]) for fold, item in cache.items()},
              "decoded_signal_cross_fold_duplicates": 0, "first_example": first_report,
              "timef_manifest_sha256": hashlib.sha256((timef_dir / "manifest.json").read_bytes()).hexdigest()}
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def prepare_development(data_root: Path, manifest_dir: Path, timef_dir: Path, output: Path) -> dict:
    """Nouveau cache V2 : TimeF existant vérifié contre le chemin canonique WAV.

    Aucun test décodé, pas de réécriture des caches historiques. La conversion
    float32 est celle apprise en V1 ; seule son identité explicite est nouvelle.
    Les mesures et textes d'amplitude sont conservés pour le cache commun A/C ;
    leur présence n'active pas la variante C dans un modèle ou une campagne.
    """
    from pipe.tslm.preprocessing import (AMPLITUDE_EVIDENCE_VERSION,
        amplitude_features, amplitude_from_normalized, amplitude_text)

    rows = {r["clip_id"]: r for r in load_manifest(manifest_dir) if r["fold"] in ("train", "val")}
    with (manifest_dir / "split_v2_audit.csv").open() as stream:
        expected_md5 = {r["clip_id"]: r["md5"] for r in csv.DictReader(stream)}
    output.mkdir(parents=True, exist_ok=False)
    cache = {fold: {"ids": [], "series": [], "amplitude_features": [], "amplitude_text": []}
             for fold in ("train", "val")}
    seen = set()
    with TimeFReader(DatasetVersion.open_local(timef_dir)) as reader:
        for record in reader.iter_records(record_ids=sorted(rows), with_annotations=False):
            cid = record.record_id
            if cid not in rows or cid in seen or record.subject_ids != (rows[cid]["group_id"],):
                raise ValueError("Couverture ou groupe TimeF incompatible")
            raw = safe_member_path(rows[cid]["path"], data_root).read_bytes()
            if hashlib.md5(raw).hexdigest() != expected_md5[cid]:
                raise ValueError(f"WAV modifié : {cid}")
            observed = record.time_series[0].to_numpy()
            waveform = decode_wav(raw)
            np.testing.assert_array_equal(observed, _normalise(waveform).astype(np.float32))
            series = preprocess_audio(waveform, 8000, version=CANONICAL_VERSION)
            np.testing.assert_array_equal(series, band_series(observed))
            raw_features = amplitude_features(waveform, 8000)
            timef_features = amplitude_from_normalized(observed)
            np.testing.assert_array_equal(raw_features, timef_features)
            raw_text, timef_text = amplitude_text(raw_features), amplitude_text(timef_features)
            if raw_text != timef_text:
                raise ValueError("Texte des mesures C1 différent entre WAV canonique et TimeF")
            cache[rows[cid]["fold"]]["ids"].append(cid)
            cache[rows[cid]["fold"]]["series"].append(series)
            cache[rows[cid]["fold"]]["amplitude_features"].append(raw_features)
            cache[rows[cid]["fold"]]["amplitude_text"].append(raw_text)
            seen.add(cid)
    if seen != set(rows):
        raise ValueError("Enregistrements de développement manquants")
    for fold, item in cache.items():
        with (output / f"{fold}.npz").open("xb") as stream:
            np.savez_compressed(stream, ids=np.asarray(item["ids"]), series=np.stack(item["series"]),
                                preprocessing_version=CANONICAL_VERSION,
                                amplitude_features=np.stack(item["amplitude_features"]),
                                amplitude_text=np.asarray(item["amplitude_text"]),
                                amplitude_evidence_version=AMPLITUDE_EVIDENCE_VERSION)
    report = {"protocol": PROTOCOL, "preprocessing_version": CANONICAL_VERSION,
              "manifest_sha256": MANIFEST_HASHES["split_v2.csv"], "records": len(rows),
              "fold_counts": {fold: len(item["ids"]) for fold, item in cache.items()},
              "test_audio_or_cache_opened": False, "canonical_vs_timef_exact": True,
              "amplitude_evidence_version": AMPLITUDE_EVIDENCE_VERSION,
              "amplitude_records_verified": len(seen),
              "amplitude_features_raw_vs_timef_exact": True,
              "amplitude_text_raw_vs_timef_exact": True,
              "timef_manifest_sha256": hashlib.sha256((timef_dir / "manifest.json").read_bytes()).hexdigest(),
              "cache_sha256": {fold: hashlib.sha256((output / f"{fold}.npz").read_bytes()).hexdigest() for fold in cache},
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("prepare.py", "preprocessing.py")}}
    feature_source = Path(__file__).resolve().parents[3] / "scripts/eval/harness/features.py"
    report["source_sha256"]["features.py"] = hashlib.sha256(feature_source.read_bytes()).hexdigest()
    with (output / "preparation.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing-timef", type=Path,
                        help="Opt-in V2 : cache développement canonique, sans téléchargement ni test")
    args = parser.parse_args()
    if args.existing_timef:
        prepare_development(args.data_dir / "extracted", args.manifest_dir, args.existing_timef, args.output)
    else:
        prepare(args.data_dir, args.manifest_dir, args.output)
