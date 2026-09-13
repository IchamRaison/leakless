#!/usr/bin/env python3
"""Dépose le résultat TSLM officiel dans la démo. Aucun calcul, aucune retouche.

À lancer UNIQUEMENT après l'annonce de CC2 : CONTRACT PASS / PROVENANCE PASS /
FINAL REPORT GENERATED, sur un rapport produit par le générateur gelé
(protocol-freeze-v1, 3e4e73ab…).

  python3 frontend/scripts/ingest-official-tslm.py <dossier final_evaluation> \\
      --contract PASS --provenance PASS

Copie metrics.json et comparison.json octet pour octet, lit le commit de
génération dans FINAL_EVALUATION.md, écrit receipt.json (identité + empreintes,
aucun score). La validation complète est faite par `npm test` côté frontend.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

PROTOCOL_TAG = "protocol-freeze-v1"
PROTOCOL_COMMIT = "3e4e73ab6e1944b42b0223e9bd9b3d500f49bc65"
DEST = Path(__file__).resolve().parents[1] / "src/demo/official"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report_dir", type=Path)
    ap.add_argument("--contract", required=True, choices=["PASS"])
    ap.add_argument("--provenance", required=True, choices=["PASS"])
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--replace", action="store_true",
                    help="remplacer un résultat officiel déjà déposé")
    args = ap.parse_args()

    src = args.report_dir
    md = (src / "FINAL_EVALUATION.md").read_text()
    found = re.search(r"\| Commit de génération du rapport \| `([^`]*)` \|", md)
    if not found or not re.fullmatch(r"[0-9a-f]{40}", found.group(1)):
        sys.exit("commit de génération absent, inconnu ou worktree modifié : refus")
    report_commit = found.group(1)
    # Le rapport doit venir du commit gelé ou d'un descendant (amendement déclaré).
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, report_commit],
                              capture_output=True, text=True, cwd=Path(__file__).resolve().parent)
    if ancestry.returncode == 1:
        sys.exit(f"commit {report_commit[:12]} ne descend pas de {PROTOCOL_TAG} : refus")
    if ancestry.returncode != 0:
        sys.exit(f"commit {report_commit[:12]} inconnu localement : git fetch origin, puis relancer")
    metrics = json.loads((src / "metrics.json").read_text())
    tslm = metrics.get("tslm_run_id")
    if not tslm:
        sys.exit("rapport sans run TSLM (tslm_run_id null) : rien à intégrer")
    raw = json.dumps([metrics, json.loads((src / "comparison.json").read_text())])
    if "SYNTHETIC" in raw:
        sys.exit("fixture synthétique : refus")

    args.dest.mkdir(parents=True, exist_ok=True)
    if list(args.dest.glob("*.json")) and not args.replace:
        sys.exit(f"{args.dest} contient déjà un résultat : --replace pour le remplacer")
    # Écriture dans un dossier temporaire voisin, puis déplacement : jamais de dépôt à moitié copié.
    stage = Path(tempfile.mkdtemp(prefix=".official-", dir=args.dest.parent))
    for name in ("metrics.json", "comparison.json"):
        shutil.copyfile(src / name, stage / name)
    sha = {n: hashlib.sha256((stage / n).read_bytes()).hexdigest()
           for n in ("metrics.json", "comparison.json")}
    receipt = {
        "status": "OFFICIAL",
        "contract_check": args.contract,
        "provenance_check": args.provenance,
        "final_report": "GENERATED",
        "protocol_tag": PROTOCOL_TAG,
        "protocol_commit": PROTOCOL_COMMIT,
        "report_commit": report_commit,
        "tslm_run_id": tslm,
        "metrics_sha256": sha["metrics.json"],
        "comparison_sha256": sha["comparison.json"],
    }
    (stage / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for old in args.dest.glob("*.json"):
        old.unlink()
    for name in ("metrics.json", "comparison.json", "receipt.json"):
        (stage / name).replace(args.dest / name)
    stage.rmdir()
    print(f"déposé dans {args.dest} : run {tslm}, rapport {report_commit[:12]}")
    # Valeurs à reporter dans le deck (docs/pitch), lues telles quelles, arrondies à 3 décimales.
    comps = json.loads((args.dest / "comparison.json").read_text())
    t = metrics["runs"][tslm]["folds"]["test"]
    c1 = comps.get(f"{tslm}_vs_c1", {})
    print(f"deck C4 : clip AUC {t['clip_level']['roc_auc']:.3f} · cluster AUC "
          f"{t['cluster_level']['roc_auc']:.3f} · {t['n_clusters']} clusters "
          f"({t['n_clusters_non_leak']} non-leak)")
    print("deck TSLM vs C1 : clip AUC " + c1.get("clip_roc_auc", {}).get("lecture", "?")
          + " ; cluster AUC " + c1.get("cluster_roc_auc", {}).get("lecture", "?"))
    print("suite : cd frontend && npm test && npm run build")


if __name__ == "__main__":
    main()
