#!/usr/bin/env python3
"""Construit les variantes TimeF T0 à T3 et vérifie leurs invariants.

Le TSLM n'est PAS évalué ici : nous ne possédons pas son checkpoint. Ce script
prépare les jeux et **mesure** ce que chaque transformation préserve ou détruit,
pour que le test de sensibilité temporelle soit exécutable dès que Hicham peut
faire tourner son modèle dessus.

Chaque variante conserve exactement le même `clip_id`, le même fold, la même
étiquette et le même cluster que `split_v2`. Le script le vérifie au lieu de le
supposer : si un mapping bouge, il échoue.

Sortie parquet **hors du dépôt**. Aucun WAV, aucun parquet dans Git.

Usage (depuis le dépôt TimeNet, qui porte l'environnement) :
  cd <TimeNet>
  LEAKLESS_DATA_ROOT=<WAV> LEAKLESS_MANIFEST_DIR=<dépôt>/manifests \\
  uv run python <dépôt>/scripts/temporal/build_stress_timef.py \\
      --out <sortie hors dépôt> [--transforms T0 T1 T2 T3] [--report <json>]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "timenet"))

from stress import TRANSFORMS, clip_rng, invariants  # noqa: E402
from stress_provenance import transform_provenance  # noqa: E402


def _git_state() -> tuple[str | None, bool | None]:
    import subprocess
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_HERE, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=_HERE, capture_output=True, text=True,
                                    check=True).stdout.strip())
        return head, dirty
    except Exception:
        return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="racine de sortie, HORS dépôt")
    ap.add_argument("--transforms", nargs="+", default=list(TRANSFORMS))
    ap.add_argument("--report", help="fichier JSON du rapport d'invariants")
    ap.add_argument("--limit", type=int, default=0, help="0 = tous les clips")
    args = ap.parse_args()

    from leakless_acoustic.connector import (CONNECTOR, ENV_DATA_ROOT, ENV_MANIFEST_DIR,
                                             _normalise, read_wav)
    from timenet.dataset import TimeFDataset, TimeSeries
    from timenet.dataset.axis import RegularAxis
    from timenet.types import ClassificationTask
    from timenet.writer.writer import TimeFWriter
    from leakless_acoustic.connector import _SIGNAL, SAMPLING_RATE_HZ

    data_root = os.environ[ENV_DATA_ROOT]
    manifest_dir = os.environ[ENV_MANIFEST_DIR]
    connector = CONNECTOR()
    refs = sorted(connector.download(Path(args.out) / "_cache"), key=lambda r: r["clip_id"])
    if args.limit:
        refs = refs[: args.limit]

    # Commit générateur capturé avant la génération, pas relevé après coup.
    generator_commit, generator_dirty = _git_state()
    import hashlib
    split_sha256 = hashlib.sha256(
        (Path(manifest_dir) / "split_v2.csv").read_bytes()).hexdigest()
    baseline = {r["clip_id"]: (r["fold"], r["label"], r["group_id"]) for r in refs}
    report: dict = {"manifest_dir": manifest_dir, "n_clips": len(refs), "transforms": {}}

    for name in args.transforms:
        if name not in TRANSFORMS:
            sys.exit(f"transformation inconnue : {name}")
        fn = TRANSFORMS[name]["fn"]
        dataset = TimeFDataset(metadata=connector.metadata())
        stats = collections.defaultdict(list)
        mapping_violations = []

        for ref in refs:
            cid = ref["clip_id"]
            raw = read_wav(Path(data_root) / ref["path"])
            transformed = fn(raw, clip_rng(name, cid))
            inv = invariants(raw, transformed)
            for k in ("rms_relative_deviation", "fft_magnitude_relative_deviation"):
                stats[k].append(inv[k])
            stats["n_samples_unchanged"].append(inv["n_samples_unchanged"])
            stats["amplitude_histogram_identical"].append(inv["amplitude_histogram_identical"])
            stats["waveform_identical"].append(inv["waveform_identical"])

            # Le mapping clip_id -> (fold, label, cluster) doit être intact.
            if (ref["fold"], ref["label"], ref["group_id"]) != baseline[cid]:
                mapping_violations.append(cid)

            series = TimeSeries.from_values(
                _normalise(transformed), spec=_SIGNAL, signal="acoustic",
                time_axis=RegularAxis.from_rate_hz(SAMPLING_RATE_HZ),
                source_id=cid, time_series_id=f"ts-{cid}")
            record = dataset.add_record(time_series=(series,),
                                        subject_ids=(ref["group_id"],), record_id=cid)
            dataset.add_task(record, ClassificationTask(target=ref["label"], id=f"task-{cid}"))

        out_dir = Path(args.out) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        writer = TimeFWriter(root=out_dir, dataset=dataset)
        writer.write()
        writer.close()

        report["transforms"][name] = {
            "description": TRANSFORMS[name]["description"],
            "output": str(out_dir),
            "n_records": len(dataset.records),
            "n_samples_unchanged_all": bool(all(stats["n_samples_unchanged"])),
            "rms_relative_deviation_max": float(np.max(stats["rms_relative_deviation"])),
            "fft_magnitude_relative_deviation_max": float(
                np.max(stats["fft_magnitude_relative_deviation"])),
            "fft_magnitude_relative_deviation_median": float(
                np.median(stats["fft_magnitude_relative_deviation"])),
            "amplitude_histogram_identical_all": bool(all(stats["amplitude_histogram_identical"])),
            "waveform_identical_all": bool(all(stats["waveform_identical"])),
            "clip_fold_label_cluster_mapping_violations": len(mapping_violations),
            "n_clusters": len({r["group_id"] for r in refs}),
            "clips_par_fold": dict(collections.Counter(r["fold"] for r in refs)),
            "provenance": transform_provenance(
                name, out_dir, split_sha256=split_sha256, generator_commit=generator_commit,
                generator_worktree_dirty=generator_dirty,
                n_records_at_generation=len(dataset.records)),
        }
        r = report["transforms"][name]
        print(f"{name}  {r['description']}")
        print(f"   records={r['n_records']}  clusters={r['n_clusters']}  "
              f"mapping_violations={r['clip_fold_label_cluster_mapping_violations']}")
        print(f"   |FFT| déviation relative  max={r['fft_magnitude_relative_deviation_max']:.2e}"
              f"  médiane={r['fft_magnitude_relative_deviation_median']:.2e}")
        print(f"   RMS déviation relative max={r['rms_relative_deviation_max']:.2e}"
              f"  histogramme d'amplitude identique : {r['amplitude_histogram_identical_all']}")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
            fh.write("\n")


if __name__ == "__main__":
    main()
