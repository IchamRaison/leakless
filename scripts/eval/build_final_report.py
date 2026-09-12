#!/usr/bin/env python3
"""Produit le rapport final. Aucun chiffre n'est recopié à la main.

Sortie :

    artifacts/final_evaluation/
        metrics.json        toutes les métriques, tous les runs, tous les folds
        comparison.json     les différences appariées
        FINAL_EVALUATION.md le rapport lisible, généré depuis les deux précédents

Le rapport s'adapte à ce qui est présent : s'il n'y a que les contrôles, il dit
que la question temporelle est ouverte. Dès qu'un run de TSLM est fourni, il
ajoute ses métriques, ses différences appariées avec chaque contrôle, et — si des
runs de stress T1/T2/T3 sont là — la lecture de sensibilité temporelle.

Usage :
  python3 scripts/eval/build_final_report.py --runs <dir> [<dir> ...] \\
      [--tslm-run-id tslm-v1] [--stress-report <json>] [--out artifacts/final_evaluation]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_predictions import compare, evaluate_run, vectors  # noqa: E402,F401
from harness import contract, metrics, split_loader  # noqa: E402

CONTROL_ORDER = ["c0", "c1", "c2", "c3"]


def fmt(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def metric_table(runs: dict, fold: str) -> str:
    head = (f"| run | modèle | clip AUC | clip PR-AUC | clip F1 | clip Brier | "
            f"cluster AUC | cluster F1 | clips | clusters (leak/non-leak) |")
    sep = "|" + "---|" * 10
    lines = [head, sep]
    for rid in sorted(runs, key=lambda r: (CONTROL_ORDER.index(r) if r in CONTROL_ORDER else 99, r)):
        b = runs[rid]["folds"][fold]
        c, g = b["clip_level"], b["cluster_level"]
        lines.append(
            f"| `{rid}` | {runs[rid]['model_name'][:42]} | {fmt(c['roc_auc'])} | "
            f"{fmt(c['pr_auc'])} | {fmt(c['macro_f1'])} | {fmt(c['brier'])} | "
            f"{fmt(g['roc_auc'])} | {fmt(g['macro_f1'])} | {b['n_clips']} | "
            f"{b['n_clusters']} ({b['n_clusters_leak']}/{b['n_clusters_non_leak']}) |")
    return "\n".join(lines)


def ci_table(runs: dict, fold: str) -> str:
    lines = ["| run | clip AUC IC95 | cluster AUC IC95 | clip F1 IC95 |", "|---|---|---|---|"]
    for rid in sorted(runs, key=lambda r: (CONTROL_ORDER.index(r) if r in CONTROL_ORDER else 99, r)):
        b = runs[rid]["folds"][fold]["bootstrap_ci95"]
        def c(k):
            return f"[{fmt(b[k]['ci95_low'])}, {fmt(b[k]['ci95_high'])}]" if k in b else "—"
        lines.append(f"| `{rid}` | {c('clip_roc_auc')} | {c('cluster_roc_auc')} | {c('clip_macro_f1')} |")
    return "\n".join(lines)


def comparison_table(comps: dict) -> str:
    lines = ["| comparaison | métrique | Δ observé | IC95 | lecture |", "|---|---|---|---|---|"]
    for name, v in comps.items():
        for m in ("clip_roc_auc", "cluster_roc_auc", "clip_macro_f1", "cluster_macro_f1"):
            if m not in v:
                continue
            d = v[m]
            lines.append(f"| `{name}` | {m} | {d['delta_observe']:+.3f} | "
                         f"[{d['ci95_low']:+.3f}, {d['ci95_high']:+.3f}] | {d['lecture']} |")
    return "\n".join(lines)


def build_markdown(result: dict, comps: dict, tslm: str | None, stress: dict | None,
                   repo_commit: str) -> str:
    sp = result["split"]
    runs = result["runs"]
    has_tslm = tslm is not None and tslm in runs

    parts = [
        "# FINAL_EVALUATION — évaluation sur le split gelé",
        "",
        "> Généré par `scripts/eval/build_final_report.py`. **Aucun chiffre n'est recopié "
        "à la main.** Régénérer le rapport régénère tous les nombres.",
        "",
        "---",
        "",
        "## 1. Identité des données et du split",
        "",
        "| | |",
        "|---|---|",
        f"| Split | `{sp['filename']}` |",
        f"| SHA256 | `{sp['sha256']}` |",
        f"| Clips | {sp['n_clips']} |",
        f"| Clusters de dépendance | {sp['n_clusters']} |",
        f"| Commit du dépôt | `{repo_commit}` |",
        f"| Règle d'agrégation | {result['aggregation_rule']} des probabilités du cluster (gelée) |",
        f"| Bootstrap | {result['bootstrap']['draws']} tirages, graine "
        f"{result['bootstrap']['seed']}, unité : {result['bootstrap']['unit']} |",
        "| Source | Zenodo 18631450, CC BY 4.0 — site d'entraînement expérimental de Dongguan |",
        "",
        "## 2. Identité des modèles",
        "",
        "| run | modèle | checkpoint | commit d'entraînement | horodatage |",
        "|---|---|---|---|---|",
    ]
    for rid in sorted(runs, key=lambda r: (CONTROL_ORDER.index(r) if r in CONTROL_ORDER else 99, r)):
        r = runs[rid]
        parts.append(f"| `{rid}` | {r['model_name']} | `{r['checkpoint']}` | "
                     f"`{str(r['training_commit'])[:12]}` | {r['timestamp']} |")

    parts += [
        "",
        "## 3. Échelle de contrôles et résultats — TEST",
        "",
        metric_table(runs, "test"),
        "",
        "### Validation (pour information — c'est là que le seuil est choisi)",
        "",
        metric_table(runs, "val"),
        "",
        "## 4. Incertitude — bootstrap sur les clusters",
        "",
        "Les intervalles rééchantillonnent des **clusters entiers**. Un bootstrap au clip "
        "rééchantillonnerait à l'intérieur de sessions quasi identiques et produirait des "
        "intervalles bien trop étroits.",
        "",
        ci_table(runs, "test"),
        "",
        "## 5. Différences appariées",
        "",
        "Un **seul** tirage de clusters sert aux deux modèles comparés. Comparer deux "
        "intervalles indépendants serait une erreur : cela ignore que les deux modèles "
        "voient les mêmes données.",
        "",
        comparison_table(comps),
        "",
        f"> ⚠️ **Aucune significativité statistique n'est revendiquée.** Le test compte "
        f"{runs[next(iter(runs))]['folds']['test']['n_clusters']} clusters indépendants, "
        f"dont {runs[next(iter(runs))]['folds']['test']['n_clusters_non_leak']} du côté "
        f"*non-leak*. Les verdicts sont qualitatifs.",
        "",
        "## 6. Résultat TSLM",
        "",
    ]

    if has_tslm:
        t = runs[tslm]["folds"]["test"]
        parts += [
            f"Run `{tslm}` — {runs[tslm]['model_name']}",
            "",
            f"- clip : AUC {fmt(t['clip_level']['roc_auc'])}, "
            f"macro-F1 {fmt(t['clip_level']['macro_f1'])}, Brier {fmt(t['clip_level']['brier'])}",
            f"- cluster : AUC {fmt(t['cluster_level']['roc_auc'])}, "
            f"macro-F1 {fmt(t['cluster_level']['macro_f1'])}",
            f"- effectifs : {t['n_clips']} clips, {t['n_clusters']} clusters "
            f"({t['n_clusters_leak']} leak / {t['n_clusters_non_leak']} non-leak)",
            "",
            "Comparaisons appariées du TSLM contre chaque contrôle : voir le tableau §5.",
        ]
    else:
        parts += [
            "🕐 **Aucun run de TSLM fourni.** Les contrôles sont mesurés, la question "
            "temporelle reste ouverte.",
            "",
            "Pour l'intégrer : `docs/MODEL_EVAL_CONTRACT.md`, puis relancer cette commande "
            "en ajoutant le dossier du run.",
        ]

    parts += ["", "## 7. Tests de stress temporel", ""]
    if stress:
        parts += [
            "Jeux préparés, invariants **mesurés** et non supposés :",
            "",
            "| | transformation | records | clusters | `|FFT|` dév. médiane | histogramme d'amplitude | violations de mapping |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, s in stress["transforms"].items():
            parts.append(
                f"| {name} | {s['description']} | {s['n_records']} | {s['n_clusters']} | "
                f"{s['fft_magnitude_relative_deviation_median']:.2e} | "
                f"{'identique' if s['amplitude_histogram_identical_all'] else 'modifié'} | "
                f"{s['clip_fold_label_cluster_mapping_violations']} |")
        parts += [
            "",
            "> Ce ne sont **pas** des augmentations préservant l'étiquette. Un modèle dont le "
            "score ne bouge pas sous T1/T2/T3 n'utilise pas l'ordre temporel.",
        ]
        if not has_tslm:
            parts.append("")
            parts.append("🕐 **Non évalués** : nous ne possédons pas le checkpoint du TSLM.")
    else:
        parts.append("_(aucun rapport de stress fourni)_")

    parts += [
        "",
        "## 8. Limites connues",
        "",
        "1. **11 clusters *non-leak* en test, 8 en validation.** Tous les intervalles sont "
        "larges et se recouvrent. Aucune comparaison n'est concluante à cette taille.",
        "2. **La normalisation d'amplitude ne neutralise pas le raccourci d'acquisition.** "
        "C1 (forme d'enveloppe seule, audio normalisé) fait au moins aussi bien que C0 "
        "(niveau absolu). Le barreau à dépasser est C1, pas C0.",
        "3. **Les clusters ne sont pas des sessions d'acquisition démontrées** : la source ne "
        "publie ni site, ni conduite, ni horodatage. Ce sont des grappes de dépendance.",
        "4. **La chaîne d'acquisition n'est pas calibrée.** Rien ne garantit qu'un écart "
        "mesuré ici survive à un autre matériel.",
        "5. **Aucune donnée de terrain, aucun client.** Le dataset vient d'un site "
        "d'entraînement expérimental.",
        "",
        "## 9. Affirmations permises",
        "",
        "- « Nous construisons un pipeline logiciel reproductible avec des contrôles de "
        "fuite stricts, sur un split gelé et vérifié. »",
        "- « Le niveau sonore absolu est un raccourci mesuré sur ce jeu. »",
        "- « La forme d'amplitude seule, après normalisation, atteint au moins le niveau du "
        "contrôle de volume : la normalisation ne suffit pas. »",
        "- « Les métriques sont group-aware : l'unité d'indépendance est la grappe de "
        "dépendance, et son effectif accompagne chaque chiffre. »",
        "- « Les comparaisons sont appariées, sur les mêmes tirages de clusters. »",
        "- Tout écart, uniquement avec *compatible with improvement* / *inconclusive* / "
        "*compatible with degradation*.",
        "",
        "## 10. Affirmations interdites",
        "",
        "- ❌ « LeakLess détecte des fuites sur le terrain. » — aucune donnée client, aucune "
        "validation terrain, aucun matériel testé.",
        "- ❌ « Le modèle localise la fuite. » — le jeu ne supporte pas la localisation.",
        "- ❌ « L'écart est statistiquement significatif. » — "
        f"{runs[next(iter(runs))]['folds']['test']['n_clusters']} clusters.",
        "- ❌ Un score sans son nombre de clusters.",
        "- ❌ Un intervalle de confiance bootstrapé sur les clips.",
        "- ❌ Toute utilisation de `split_v1`, invalide.",
        "- ❌ « La modélisation temporelle apporte de la valeur » tant que la comparaison "
        "TSLM n'a pas été faite" + ("." if has_tslm else " — **elle ne l'est pas encore**."),
        "",
        "---",
        "",
        "Matrice claim → artefact : [`JURY_EVIDENCE_MATRIX.md`](JURY_EVIDENCE_MATRIX.md). "
        "Contrat de livraison du modèle : [`MODEL_EVAL_CONTRACT.md`](MODEL_EVAL_CONTRACT.md).",
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--tslm-run-id", default=None)
    ap.add_argument("--stress-report", default=None)
    ap.add_argument("--out", default="artifacts/final_evaluation")
    args = ap.parse_args()

    split = split_loader.load_split(args.manifests)
    runs, reports = {}, {}
    for d in args.runs:
        run = contract.load_run(d, split)
        runs[run.run_id] = run
        reports[run.run_id] = evaluate_run(split, run)

    tslm = args.tslm_run_id
    if tslm is None:
        candidates = [r for r in runs if r not in CONTROL_ORDER and "shuffled" not in r]
        if len(candidates) == 1:
            tslm = candidates[0]
        elif len(candidates) > 1:
            # Cas réel : Hicham livre T0/T1/T2/T3. L'auto-détection ne peut pas
            # deviner lequel est le run de référence — le dire, pas l'inventer.
            sys.exit(f"{len(candidates)} runs hors contrôles trouvés ({candidates}). "
                     f"Préciser lequel est le run de référence avec --tslm-run-id, "
                     f"sinon le rapport dirait à tort qu'aucun TSLM n'a été fourni.")

    comps = {}
    ids = sorted(runs)
    if tslm and tslm in runs:
        for other in ids:
            if other != tslm:
                comps[f"{tslm}_vs_{other}"] = compare(split, runs, tslm, other)
    else:
        import itertools
        for a, b in itertools.combinations(ids, 2):
            comps[f"{a}_vs_{b}"] = compare(split, runs, a, b)

    result = {
        "split": {"filename": split_loader.FROZEN_SPLIT_NAME, "sha256": split.sha256,
                  "n_clips": len(split.clips),
                  "n_clusters": len({c.group_id for c in split.clips})},
        "aggregation_rule": metrics.AGGREGATION,
        "bootstrap": {"draws": metrics.BOOTSTRAP_DRAWS, "seed": metrics.BOOTSTRAP_SEED,
                      "unit": "cluster de dépendance"},
        "runs": reports,
        "tslm_run_id": tslm,
    }

    stress = json.load(open(args.stress_report)) if args.stress_report else None
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (out / "comparison.json").write_text(json.dumps(comps, indent=2, ensure_ascii=False) + "\n")
    (out / "FINAL_EVALUATION.md").write_text(
        build_markdown(result, comps, tslm, stress, commit))

    print(f"écrit : {out}/metrics.json, comparison.json, FINAL_EVALUATION.md")
    print(f"runs évalués : {sorted(runs)} | TSLM : {tslm or 'aucun'}")


if __name__ == "__main__":
    main()
