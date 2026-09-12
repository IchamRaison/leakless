#!/usr/bin/env python3
"""Pilote le connecteur TimeNet et écrit le jeu TimeF sur disque, HORS du dépôt.

Le moteur TimeNet fait `download -> convert -> write`. Ici on le fait à la main pour
rester hors du registre : le jeu est local, aucun réseau, aucune publication.

Usage (depuis le dépôt TimeNet, qui porte l'environnement) :

  cd <TimeNet>
  LEAKLESS_DATA_ROOT=<dossier des WAV> \\
  LEAKLESS_MANIFEST_DIR=<dépôt>/manifests \\
  uv run python <dépôt>/scripts/timenet/build_timef.py --out <sortie hors dépôt>

Le dossier de sortie contient du parquet : il ne doit jamais entrer dans Git.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from leakless_acoustic.connector import CONNECTOR, ENV_DATA_ROOT, ENV_MANIFEST_DIR  # noqa: E402

from timenet.writer.writer import TimeFWriter  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="dossier de sortie TimeF, HORS du dépôt")
    args = ap.parse_args()

    connector = CONNECTOR()
    refs = connector.download(Path(args.out) / "_cache")
    print(f"clips référencés depuis le split gelé : {len(refs)}")

    dataset = connector.convert(refs)
    print(f"records construits : {len(dataset.records)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # write() remplit un dossier de staging ; close() écrit le manifeste et publie
    # le dossier de façon atomique. Sans close(), il ne reste qu'un `.tmp-...`
    # illisible par TimeFReader.
    writer = TimeFWriter(root=out, dataset=dataset)
    writer.write()
    writer.close()

    # Contrôle : le jeu TimeF doit refléter le split gelé, record par record.
    folds = collections.Counter(r["fold"] for r in refs)
    groups = {r["group_id"] for r in refs}
    print(json.dumps({
        "sortie": str(out),
        "records": len(dataset.records),
        "grappes_de_dependance": len(groups),
        "clips_par_fold": dict(folds),
        "note": "amplitude normalisée par clip ; aucune métadonnée de nom de fichier "
                "n'entre dans le jeu TimeF",
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
