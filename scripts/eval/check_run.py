#!/usr/bin/env python3
"""Vérifie qu'un run est conforme au contrat. **Aucune métrique n'est calculée.**

Pour Hicham : lance ça avant d'envoyer quoi que ce soit. Le script dit oui ou
non, et s'il dit non il dit pourquoi et sur combien de clips il a regardé.

Il ne calcule ni AUC, ni F1, ni rien qui ressemble à un résultat : la conformité
et l'évaluation sont deux choses séparées, et tu n'as pas à voir la seconde pour
livrer.

Trois usages :

  # 1. fabriquer le squelette avec les 402 clip_id attendus, déjà remplis
  python3 scripts/eval/check_run.py --template mon_run/

  # 2. vérifier un run avant envoi
  python3 scripts/eval/check_run.py --run mon_run/

  # 3. diagnostiquer la distribution des probabilités (dégénérescence, ex aequo)
  python3 scripts/eval/check_run.py --run mon_run/ --inspect

Code de sortie 0 si conforme, 1 sinon.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import contract, split_loader  # noqa: E402

FOLDS = ("val", "test")


def write_template(out: Path, split) -> None:
    """Écrit un squelette avec les bons clip_id et une probabilité à remplir."""
    out.mkdir(parents=True, exist_ok=True)
    clips = sorted([c for c in split.clips if c.fold in FOLDS], key=lambda c: c.clip_id)
    with open(out / "predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["clip_id", "probability_leak"])
        for c in clips:
            w.writerow([c.clip_id, ""])
    meta = {
        "run_id": "REMPLIR-moi",
        "model_name": "REMPLIR-moi",
        "checkpoint": "REMPLIR-moi",
        "training_commit": "REMPLIR-moi",
        "config_hash": None,
        "split_filename": split_loader.FROZEN_SPLIT_NAME,
        "split_sha256": split.sha256,
        "timestamp": "REMPLIR-moi (ISO 8601)",
        "threshold_rule": "aucun seuil appliqué — probabilités brutes non calibrées",
        "test_labels_not_used_for_tuning": True,
    }
    with open(out / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"squelette écrit dans {out}")
    print(f"  predictions.csv : {len(clips)} clip_id déjà remplis "
          f"({sum(1 for c in clips if c.fold == 'val')} val + "
          f"{sum(1 for c in clips if c.fold == 'test')} test)")
    print("  il ne reste qu'à remplir la colonne probability_leak et metadata.json")
    print("\n  ⚠️ L'ordre des lignes n'a aucune importance, mais chaque clip_id doit")
    print("     apparaître exactement une fois. Ne réordonne pas par autre chose que")
    print("     le clip_id si tu produis le fichier autrement.")


def inspect(run, split) -> list[str]:
    """Diagnostics de distribution. Ne juge pas la performance, juge l'exploitabilité."""
    warnings: list[str] = []
    by_id = split.by_id()
    vals = [run.probabilities[c.clip_id] for c in split.clips
            if c.fold in FOLDS and c.clip_id in run.probabilities]
    n = len(vals)
    distinct = len(set(vals))
    counts = collections.Counter(vals)
    top_val, top_n = counts.most_common(1)[0]

    print("\n=== distribution des probabilités (val + test) ===")
    print(f"  n={n}  valeurs distinctes={distinct}  min={min(vals):.4f}  max={max(vals):.4f}")
    print(f"  valeur la plus fréquente : {top_val:.4f} sur {top_n} clips "
          f"({100 * top_n / n:.1f} %)")

    if distinct <= 2:
        warnings.append(
            f"SORTIE BINAIRE : seulement {distinct} valeur(s) distincte(s). Mesuré sur "
            f"notre contrôle C1, passer d'un score continu à du 0/1 coûte 0,046 de clip "
            f"AUC et 0,071 de cluster AUC — plus que l'écart qu'on cherche à mesurer. "
            f"Sors une probabilité continue (softmax sur les log-probabilités des deux "
            f"classes), pas une décision.")
    elif distinct < n / 10:
        warnings.append(
            f"PEU DE VALEURS DISTINCTES ({distinct} pour {n} clips) : beaucoup d'ex aequo. "
            f"Chaque ex aequo compte pour 0,5 dans l'AUC et abaisse mécaniquement le score.")
    if top_n > n * 0.25:
        warnings.append(
            f"UNE VALEUR CONCENTRE {100 * top_n / n:.0f} % DES CLIPS. Souvent le signe d'une "
            f"saturation de softmax ou d'un décodage glouton. Vérifie que tu lis bien des "
            f"log-probabilités et non une décision.")
    if min(vals) > 0.05 and max(vals) < 0.95:
        print("  (plage resserrée — sans effet sur l'AUC, mais le score de Brier en pâtira ; "
              "sans importance si tu déclares les probabilités non calibrées)")

    # La couverture par fold, pour qu'il n'y ait pas de surprise.
    per_fold = collections.Counter(by_id[cid].fold for cid in run.probabilities
                                   if cid in by_id)
    print(f"  couverture : " + ", ".join(f"{k}={v}" for k, v in sorted(per_fold.items())))
    return warnings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="dossier du run à vérifier")
    ap.add_argument("--template", help="dossier où écrire un squelette pré-rempli")
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--inspect", action="store_true",
                    help="diagnostics de distribution des probabilités")
    args = ap.parse_args()

    split = split_loader.load_split(args.manifests)

    if args.template:
        write_template(Path(args.template), split)
        return
    if not args.run:
        ap.error("il faut --run ou --template")

    try:
        run = contract.load_run(args.run, split, folds=FOLDS)
    except contract.ContractError as e:
        print(f"NON CONFORME\n\n  {e}\n")
        print("Rien n'est envoyé tant que ce n'est pas corrigé. "
              "Détail des règles : docs/MODEL_EVAL_CONTRACT.md")
        sys.exit(1)

    print(run.report())
    print("\n✅ CONFORME AU CONTRAT — le run peut être envoyé.")
    print("   Aucune métrique n'a été calculée ici : conformité et évaluation sont séparées.")

    if args.inspect:
        for w in inspect(run, split):
            print(f"\n⚠️  {w}")


if __name__ == "__main__":
    main()
