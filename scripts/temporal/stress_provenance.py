#!/usr/bin/env python3
"""Manifeste de provenance des jeux de stress T0-T3. Ne régénère rien.

Pour chaque transformation : nom, graine de base, schéma et version de dérivation
des graines, SHA256 du split, commit générateur, SHA256 de l'artefact, nombre de
records. L'empreinte de l'artefact couvre chaque fichier du jeu TimeF (chemin
relatif et contenu, dans l'ordre lexicographique) : un octet modifié la change.

Le commit générateur n'est écrit que s'il a été capturé AU MOMENT de la
génération. Sinon le champ vaut null et `generator_commit_note` dit pourquoi :
un commit relevé après coup ne prouverait rien.

Usage :
  python3 scripts/temporal/stress_provenance.py --stress-root <timef-stress> \\
      --invariants <stress_invariants.json> --out <manifeste.json> \\
      --generator-commit-note "<pourquoi il manque>"

Ce CLI décrit des jeux DÉJÀ générés, dont le commit n'a pas été capturé : la
raison est donc obligatoire. Une génération par build_stress_timef.py capture
elle-même le commit et n'a pas besoin de ce CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "eval"))

import stress  # noqa: E402
from harness import split_loader  # noqa: E402


def artifact_sha256(root: Path) -> tuple[str, int]:
    """SHA256 d'un dossier : chemins relatifs et contenus, dans un ordre stable."""
    if not Path(root).is_dir():
        raise FileNotFoundError(f"jeu de stress introuvable : {root}")
    h = hashlib.sha256()
    files = sorted(p for p in Path(root).rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"jeu de stress vide : {root}")
    for f in files:
        rel = f.relative_to(root).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(8, "big") + rel)
        data = f.read_bytes()
        h.update(len(data).to_bytes(8, "big") + data)
    return h.hexdigest(), len(files)


def count_records(root: Path) -> int | None:
    """Nombre de lignes de la table `records`, relu depuis le parquet si possible."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    parts = sorted(Path(root).rglob("records/*.parquet"))
    return sum(pq.ParquetFile(p).metadata.num_rows for p in parts) if parts else None


def transform_provenance(name: str, root: Path, *, split_sha256: str,
                         generator_commit: str | None, generator_worktree_dirty: bool | None,
                         generator_commit_note: str | None = None,
                         n_records_at_generation: int | None = None) -> dict:
    sha, n_files = artifact_sha256(root)
    entry = {
        "transform": name,
        "description": stress.TRANSFORMS[name]["description"],
        "stress_seed": stress.STRESS_SEED,
        "seed_scheme": stress.SEED_SCHEME,
        "seed_scheme_version": stress.SEED_SCHEME_VERSION,
        "block_samples": stress.BLOCK_SAMPLES if name == "T2" else None,
        "split_filename": split_loader.FROZEN_SPLIT_NAME,
        "split_sha256": split_sha256,
        "generator_commit": generator_commit,
        "generator_worktree_dirty": generator_worktree_dirty,
        "artifact_sha256": sha,
        "artifact_n_files": n_files,
        "n_records": count_records(root),
        "n_records_at_generation": n_records_at_generation,
    }
    if generator_commit_note:
        entry["generator_commit_note"] = generator_commit_note
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stress-root", required=True)
    ap.add_argument("--invariants", required=True)
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--out", required=True)
    ap.add_argument("--generator-commit-note", required=True,
                    help="pourquoi le commit générateur n'a pas été capturé")
    args = ap.parse_args()

    split = split_loader.load_split(args.manifests)
    inv = json.load(open(args.invariants))
    out = {"transforms": {}}
    for name in sorted(inv["transforms"]):
        root = Path(args.stress_root) / name
        out["transforms"][name] = transform_provenance(
            name, root, split_sha256=split.sha256, generator_commit=None,
            generator_worktree_dirty=None, generator_commit_note=args.generator_commit_note,
            n_records_at_generation=inv["transforms"][name]["n_records"])
        e = out["transforms"][name]
        print(f"{name}  records={e['n_records']}  sha256={e['artifact_sha256'][:16]}")
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
