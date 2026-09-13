"""Gel du manifeste Aghashahi, réservé à l'évaluateur, sans inférence ni fit.

Le scoreur ne lit pas ce manifeste : il reçoit inputs.csv et les tableaux seuls.
Le bruit annexe ne fait jamais partie du Split primaire renvoyé.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import audit_external_overlap as overlap
import prepare_external_holdout as preparer
from harness.split_loader import Clip, Split

NAME = "external_aghashahi_v1.json"
SCHEMA = "pipe.external-evaluation-manifest.v1"
CSV_FILES = ("inputs.csv", "targets.csv", "background_inputs.csv", "background_targets.csv")
OVERLAP_FILES = {"provenance.json", "pairs.jsonl", "suspects.jsonl", "exact_matches.json", "best_per_original.json"}
HELPER_PATHS = {"scripts/eval/prepare_external_holdout.py", "scripts/tslm/audit_text.py",
                "scripts/tslm/export_run.py", "scripts/eval/harness/split_loader.py"}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Champ JSON répété : {key}")
        result[key] = value
    return result


def read_json(path, bound=4_000_000):
    path = Path(path)
    if path.stat().st_size > bound:
        raise ValueError("Artefact JSON trop volumineux")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)


def source_hashes():
    return {"external_manifest.py": preparer.sha256_file(Path(__file__)),
            "prepare_external_holdout.py": preparer.sha256_file(Path(preparer.__file__)),
            "audit_external_overlap.py": preparer.sha256_file(Path(overlap.__file__))}


def expected_tables():
    """Reconstruction indépendante des CSV préparés depuis l'inventaire officiel."""
    tables = {name: [] for name in CSV_FILES}
    for member, context in sorted(preparer.expected_recordings().items()):
        recording = preparer.stable_id("recording", member)
        group = preparer.stable_id("condition", context["condition_key"])
        prefix = "" if context["subset"] == "primary" else "background_"
        for start in range(0, preparer.CROP_SAMPLES, preparer.WINDOW_SAMPLES):
            cid = preparer.stable_id("clip", f"{member}\0{start}\0{preparer.WINDOW_SAMPLES}")
            tables[prefix + "inputs.csv"].append({"clip_id": cid, "array_path": f"arrays/{recording}.npy",
                "start_sample": str(start), "n_samples": str(preparer.WINDOW_SAMPLES),
                "sample_rate": str(preparer.SAMPLE_RATE)})
            tables[prefix + "targets.csv"].append({"clip_id": cid, "recording_id": recording,
                "condition_group_id": group, "label": context["label"]})
    return {name: sorted(rows, key=lambda r: r["clip_id"]) for name, rows in tables.items()}


def read_csv(path, fields):
    if path.stat().st_size > 2_000_000:
        raise ValueError("CSV externe trop volumineux")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError(f"Colonnes incorrectes : {path.name}")
        rows = list(reader)
    if (any(set(row) != set(fields) or any(value is None for value in row.values()) for row in rows)
            or len({row["clip_id"] for row in rows}) != len(rows)):
        raise ValueError(f"Ligne incorrecte ou identifiant répété : {path.name}")
    return sorted(rows, key=lambda r: r["clip_id"])


def validate_preparation(directory):
    directory = Path(directory)
    records, existing_provenance = overlap.load_external(directory)
    expected = expected_tables()
    sums = read_json(overlap.safe_path(directory, "checksums.json"))
    metadata = read_json(overlap.safe_path(directory, "metadata.json"))
    receipt = read_json(overlap.safe_path(directory, "receipt.json"))
    if (metadata.get("script_sha256") != preparer.sha256_file(Path(preparer.__file__))
            or metadata.get("training_permitted") is not False
            or metadata.get("threshold_fitting_permitted") is not False
            or metadata.get("model_scoring_performed") is not False
            or receipt.get("model_loaded") is not False or receipt.get("scores_computed") is not False):
        raise ValueError("Préparation non réservée ou code préparateur différent")
    digests = {}
    for name in CSV_FILES:
        path = overlap.safe_path(directory, name)
        digest = preparer.sha256_file(path)
        if sums.get(name) != digest:
            raise ValueError(f"SHA de CSV incorrect : {name}")
        fields = preparer.INPUT_FIELDS if name.endswith("inputs.csv") else preparer.TARGET_FIELDS
        if read_csv(path, fields) != expected[name]:
            raise ValueError(f"CSV différent de l'inventaire officiel : {name}")
        digests[name] = digest
    for prefix in ("", "background_"):
        if [r["clip_id"] for r in expected[prefix + "inputs.csv"]] != [r["clip_id"] for r in expected[prefix + "targets.csv"]]:
            raise ValueError("Jointure externe input/cible inexacte")
    primary_ids = {r["clip_id"] for r in expected["targets.csv"]}
    if primary_ids & {r["clip_id"] for r in expected["background_targets.csv"]}:
        raise ValueError("Bruit annexe mélangé au primaire")
    return records, expected, {**existing_provenance,
        "receipt_sha256": preparer.sha256_file(directory / "receipt.json"), "csv_sha256": digests}


def validate_overlap(directory, preparation, recording_ids, manifests):
    """Vérifier le journal complet existant, sans recalculer les corrélations."""
    directory, manifests = Path(directory), Path(manifests)
    summary = read_json(overlap.safe_path(directory, "summary.json"))
    offsets = preparer.CROP_SAMPLES - preparer.WINDOW_SAMPLES + 1
    count = overlap.N_ORIGINAL * len(recording_ids)
    if (summary.get("status") != "complete" or summary.get("threshold") != overlap.SUSPECT_CORRELATION
            or summary.get("pairs_examined") != count or summary.get("pairs_expected") != count
            or summary.get("offsets_per_pair") != offsets
            or any(summary.get(key) != 0 for key in ("suspect_pairs", "unassessable_pairs",
                "pairs_with_unassessable_offsets", "unassessable_offsets_total", "exact_aligned_matches"))
            or summary.get("automatic_exclusions") is not False
            or set(summary.get("files_sha256", {})) != OVERLAP_FILES):
        raise ValueError("Audit de recouvrement incomplet, suspect ou non évaluable")
    for name in OVERLAP_FILES:
        if preparer.sha256_file(overlap.safe_path(directory, name)) != summary["files_sha256"][name]:
            raise ValueError(f"Journal overlap modifié : {name}")
    provenance = read_json(directory / "provenance.json")
    old_split = manifests / "split_v2.csv"
    if (preparer.sha256_file(old_split) != overlap.FROZEN_SPLIT_SHA256
            or preparer.sha256_file(manifests / "split_v2_audit.csv") != overlap.FROZEN_AUDIT_SHA256):
        raise ValueError("Ancien split différent du développement gelé")
    with old_split.open(newline="") as stream:
        original_ids = [r["clip_id"] for r in csv.DictReader(stream)]
    if len(original_ids) != overlap.N_ORIGINAL or len(set(original_ids)) != overlap.N_ORIGINAL:
        raise ValueError("Couverture des anciens IDs invalide")
    expected_external = {key: preparation[key] for key in ("checksums_sha256", "archive_sha256", "metadata_sha256")}
    if (provenance.get("external") != expected_external
            or provenance.get("script_sha256") != preparer.sha256_file(Path(overlap.__file__))
            or provenance.get("split_sha256") != overlap.FROZEN_SPLIT_SHA256
            or provenance.get("audit_manifest_sha256") != overlap.FROZEN_AUDIT_SHA256
            or provenance.get("threshold") != overlap.SUSPECT_CORRELATION
            or provenance.get("offset_range_inclusive") != [0, offsets-1]
            or provenance.get("sample_rate") != preparer.SAMPLE_RATE
            or set(provenance.get("helper_sha256", {})) != HELPER_PATHS
            or any(provenance.get(key) is not False for key in ("labels_used", "model_loaded", "quality_metrics_computed"))
            or sorted(r["clip_id"] for r in provenance.get("original_files", [])) != sorted(original_ids)):
        raise ValueError("Provenance overlap différente des sources réservées")
    for relative, digest in provenance.get("helper_sha256", {}).items():
        if preparer.sha256_file(overlap.safe_path(overlap.ROOT, relative)) != digest:
            raise ValueError("Helper de l'audit overlap différent du code courant")
    if (read_json(directory / "exact_matches.json") != []
            or (directory / "suspects.jsonl").stat().st_size != 0):
        raise ValueError("Recouvrements candidats présents dans les journaux")
    originals, recordings = set(original_ids), set(recording_ids)
    seen, best = set(), {}
    with (directory / "pairs.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if len(line) > 10000:
                raise ValueError("Ligne overlap trop volumineuse")
            row = json.loads(line, object_pairs_hook=_unique)
            pair = (row.get("clip_id"), row.get("recording_id"))
            corr, absolute, start = row.get("correlation"), row.get("max_abs_correlation"), row.get("offset_samples")
            if (pair[0] not in originals or pair[1] not in recordings or pair in seen
                    or row.get("suspect") is not False or row.get("reason") is not None
                    or row.get("unassessable_offsets") != 0
                    or type(corr) not in (int, float) or not math.isfinite(corr)
                    or type(absolute) not in (int, float) or not math.isfinite(absolute)
                    or absolute != abs(corr) or not 0 <= absolute < overlap.SUSPECT_CORRELATION
                    or type(start) is not int or not 0 <= start < offsets):
                raise ValueError("Paire overlap invalide, manquante, dupliquée ou suspecte")
            seen.add(pair)
            if pair[0] not in best or absolute > best[pair[0]]["max_abs_correlation"]:
                best[pair[0]] = row
    if len(seen) != count or read_json(directory / "best_per_original.json") != best:
        raise ValueError("Couverture exhaustive ou meilleurs couples non concordants")
    return {"summary_sha256": preparer.sha256_file(directory / "summary.json"),
            "files_sha256": summary["files_sha256"], "pairs_examined": count,
            "offsets_per_pair": offsets, "threshold": overlap.SUSPECT_CORRELATION,
            "suspect_pairs": 0, "unassessable_pairs": 0, "exact_aligned_matches": 0,
            "original_split_sha256": overlap.FROZEN_SPLIT_SHA256,
            "interpretation": "no_candidate_at_fixed_threshold_not_proof_of_independent_sessions"}


def rules():
    return {"source_doi": preparer.DOI, "archive_sha256": preparer.ARCHIVE_SHA256,
        "converter_sha256": preparer.CONVERTER_SHA256, "sample_rate": preparer.SAMPLE_RATE,
        "raw": "signed_PCM32_little_endian", "array_dtype": "<f8", "scale": "signed_int32 / 2147483648",
        "crop_start_sample": 0, "crop_samples": preparer.CROP_SAMPLES,
        "window_samples": preparer.WINDOW_SAMPLES, "stride_samples": preparer.WINDOW_SAMPLES,
        "resampling": "none", "requantization": False,
        "grouping": "topology + leak_type + demand/noise condition; H1/H2 and all windows together",
        "real_session_ids_available": False, "primary_fold": "external", "background_in_primary": False}


def freeze_external_manifest(external, overlap_directory, manifests, output, code_revision):
    output = Path(output)
    if output.name != NAME:
        raise ValueError(f"Nom externe distinct obligatoire : {NAME}")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Manifeste déjà présent : aucun écrasement")
    if not output.parent.is_dir() or re.fullmatch(r"[0-9a-f]{40}", code_revision) is None:
        raise ValueError("Parent existant et code-revision SHA40 requis")
    records, tables, preparation = validate_preparation(Path(external))
    audit = validate_overlap(overlap_directory, preparation, [r[0] for r in records], manifests)
    manifest = {"schema": SCHEMA, "name": NAME, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": code_revision, "sources_sha256": source_hashes(), "rules": rules(),
        "preparation": preparation, "overlap": audit, "primary_targets": tables["targets.csv"],
        "background_annex": {"included_in_primary": False, "targets": tables["background_targets.csv"]},
        "evaluator_only": True, "model_scoring_performed_here": False,
        "training_permitted": False, "threshold_fitting_permitted": False}
    overlap.write_json(output, manifest)
    return load_external_manifest(output)


def load_external_manifest(path):
    """Lire les seules cibles primaires vérifiées ; aucune ouverture des tableaux."""
    path = Path(path)
    if path.name != NAME or not path.is_file():
        raise ValueError("Manifeste externe nommé explicitement requis")
    manifest = read_json(path)
    tables = expected_tables()
    if (manifest.get("schema") != SCHEMA or manifest.get("name") != NAME
            or manifest.get("rules") != rules() or manifest.get("sources_sha256") != source_hashes()
            or manifest.get("evaluator_only") is not True
            or manifest.get("training_permitted") is not False
            or manifest.get("threshold_fitting_permitted") is not False
            or manifest.get("model_scoring_performed_here") is not False
            or manifest.get("primary_targets") != tables["targets.csv"]
            or manifest.get("background_annex") != {"included_in_primary": False, "targets": tables["background_targets.csv"]}):
        raise ValueError("Contenu externe modifié ou incompatible avec l'inventaire gelé")
    # Les CSV ont été joints et vérifiés au gel. Leur identité voyage avec le
    # manifeste portable ; le scoreur vérifie séparément inputs.csv/arrays.
    preparation, audit = manifest.get("preparation", {}), manifest.get("overlap", {})
    hashes = [preparation.get(key) for key in ("checksums_sha256", "archive_sha256", "metadata_sha256", "receipt_sha256")]
    hashes += [preparation.get("csv_sha256", {}).get(name) for name in CSV_FILES]
    hashes += [audit.get("summary_sha256"), audit.get("original_split_sha256")]
    hashes += [audit.get("files_sha256", {}).get(name) for name in OVERLAP_FILES]
    if (any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes)
            or preparation["archive_sha256"] != preparer.ARCHIVE_SHA256
            or audit.get("threshold") != overlap.SUSPECT_CORRELATION
            or audit.get("pairs_examined") != overlap.N_ORIGINAL * len(preparer.expected_recordings())
            or audit.get("offsets_per_pair") != preparer.CROP_SAMPLES-preparer.WINDOW_SAMPLES+1
            or any(audit.get(key) != 0 for key in ("suspect_pairs", "unassessable_pairs", "exact_aligned_matches"))
            or audit.get("original_split_sha256") != overlap.FROZEN_SPLIT_SHA256):
        raise ValueError("Provenance préparation/overlap incomplète ou non admissible")
    clips = tuple(Clip(clip_id=r["clip_id"], label=int(r["label"] == "leak"), label_3c=r["label"],
                       group_id=r["condition_group_id"], fold="external") for r in tables["targets.csv"])
    return Split(clips=clips, sha256=preparer.sha256_file(path), manifest_path=path, _paths={})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("external", "overlap", "manifests", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    split = freeze_external_manifest(args.external, args.overlap, args.manifests, args.output, args.code_revision)
    print(json.dumps({"manifest": str(split.manifest_path), "sha256": split.sha256,
                      "primary_clips": len(split.clips), "fold": "external", "model_loaded": False}))


if __name__ == "__main__":
    main()
