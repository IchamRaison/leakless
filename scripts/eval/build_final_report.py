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

Restitution V2 (sans recalcul) :
  python3 scripts/eval/build_final_report.py --v2-campaign <campagne> \\
      [--v2-evaluation <json>] [--v2-audits <parent-export> ...] --out <dossier-neuf>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_predictions import compare, evaluate_run, vectors  # noqa: E402,F401
from harness import contract, metrics, split_loader  # noqa: E402

CONTROL_ORDER = ["c0", "c1", "c2", "c2b", "c3"]


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


def decision_table(test: dict, threshold_caption="au seuil du run choisi sur validation") -> str:
    """Lecture des comptes existants au seuil du run ; aucune nouvelle prédiction."""
    def pct(numerator, denominator):
        return "—" if denominator == 0 else f"{100 * numerator / denominator:.1f} %"

    lines = ["| niveau | fuites | non-fuites | FP | FN | précision fuite | F1 fuite | rappel fuite | FPR | taux manquées |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, name in (("clip_level", "clip"), ("cluster_level", "groupe")):
        c = test[key]
        tp, fp, tn, fn = (c[k] for k in ("tp", "fp", "tn", "fn"))
        lines.append(f"| {name} | {tp + fn} | {tn + fp} | {fp} | {fn} | "
                     f"{pct(tp, tp + fp)} | {pct(2 * tp, 2 * tp + fp + fn)} | "
                     f"{pct(tp, tp + fn)} | {pct(fp, fp + tn)} | {pct(fn, tp + fn)} |")
    lines += ["", "F1 fuite ≠ macro-F1. Pourcentages dérivés des comptes TP/FP/TN/FN "
              f"{threshold_caption} ; — indique un dénominateur nul."]
    return "\n".join(lines)


def build_markdown(result: dict, comps: dict, tslm: str | None, stress: dict | None,
                   repo_commit: str) -> str:
    sp = result["split"]
    runs = result["runs"]
    has_tslm = tslm is not None and tslm in runs
    reference = runs[tslm] if has_tslm else runs[next(iter(runs))]

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
        "Les intervalles rééchantillonnent des **clusters entiers**, pour tenir compte "
        "des dépendances supposées entre clips. Ces groupes heuristiques ne sont pas "
        "des sessions d'acquisition indépendantes démontrées.",
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
        f"{reference['folds']['test']['n_clusters']} groupes de dépendance heuristiques, "
        f"dont {reference['folds']['test']['n_clusters_non_leak']} du côté "
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
            "### Décisions et erreurs au seuil retenu — TEST",
            "",
            decision_table(t),
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
            "> Ce ne sont **pas** des augmentations préservant l'étiquette. Une absence de "
            "variation des scores indique une insensibilité aux transformations testées, "
            "pas une preuve générale que le modèle ignore toute organisation temporelle.",
        ]
        if not has_tslm:
            parts.append("")
            parts.append("🕐 **Non évalués ici** : aucun run de prédictions TSLM n'a été fourni.")
    else:
        parts.append("_(aucun rapport d'invariants fourni ; les métriques des runs stressés "
                     "éventuellement présents figurent aux §3–5)_")

    parts += [
        "",
        "## 8. Limites connues",
        "",
        f"1. **{reference['folds']['test']['n_clusters_non_leak']} groupes *non-leak* en test, "
        f"{reference['folds']['val']['n_clusters_non_leak']} en validation.** L'effectif limite "
        "la précision ; lire les intervalles et verdicts effectivement calculés aux §4–5, "
        "sans préjuger de leur largeur ni de leur recouvrement.",
        "2. **C0 et C1 sondent les raccourcis d'acquisition.** La comparaison du niveau "
        "absolu (C0) et de la forme d'amplitude normalisée (C1) doit se lire dans les "
        "résultats fournis ; aucune supériorité n'est présumée par le gabarit.",
        "3. **Les clusters ne sont pas des sessions d'acquisition démontrées** : "
        "les regroupements sont heuristiques, pas une preuve d'indépendance des acquisitions.",
        "4. **La chaîne d'acquisition n'est pas calibrée.** Rien ne garantit qu'un écart "
        "mesuré ici survive à un autre matériel.",
        "5. **Aucune donnée de terrain, aucun client.** Le dataset vient d'un site "
        "d'entraînement expérimental.",
        "",
        "## 9. Affirmations permises",
        "",
        "- « Nous construisons un pipeline logiciel reproductible avec des contrôles de "
        "fuite stricts, sur un split gelé et vérifié. »",
        "- Toute affirmation de performance ou de raccourci d'acquisition doit renvoyer "
        "aux contrôles, métriques et comparaisons effectivement présents dans ce rapport.",
        "- « Les métriques sont group-aware : l'unité de rééchantillonnage est la grappe "
        "de dépendance, et son effectif accompagne chaque chiffre. »",
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
        "- ❌ « La modélisation temporelle apporte de la valeur » sans comparaison "
        "appropriée et preuves suffisantes ; la seule présence d'un run TSLM ne le démontre pas.",
        "",
        "---",
        "",
        "Matrice claim → artefact : [`JURY_EVIDENCE_MATRIX.md`](JURY_EVIDENCE_MATRIX.md). "
        "Contrat de livraison du modèle : [`MODEL_EVAL_CONTRACT.md`](MODEL_EVAL_CONTRACT.md).",
    ]
    return "\n".join(parts) + "\n"


V2_EXTERNAL_NAME = "external_aghashahi_v1.json"
V2_EXTERNAL_SHA256 = "840078010f4023a045a039167f079ef178f6b0cb56f7f280fd2a5760aad1616b"


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _v2_read(path, sources, *, jsonl=False):
    path = Path(path).resolve()
    sources[str(path)] = _sha(path)
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()] if jsonl else json.load(stream)


def _v2_predictions(path, sources):
    path = Path(path)
    if not path.is_file():
        return None
    sources[str(path.resolve())] = _sha(path)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["clip_id", "probability_leak"]:
            raise ValueError("Colonnes du run différentes du contrat")
        values = {}
        for row in reader:
            score = float(row["probability_leak"])
            if row["clip_id"] in values or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Prédiction répétée, non finie ou hors bornes")
            values[row["clip_id"]] = score
        return values


def _v2_cv(campaign, sources):
    registration = _v2_read(campaign / "preregistration.json", sources)
    selection = _v2_read(campaign / "selection/selection.json", sources)
    if selection.get("preregistration_sha256") != _sha(campaign / "preregistration.json"):
        raise ValueError("Sélection et préinscription incompatibles")
    candidates = selection["candidates"]
    if len(candidates) != 2 or {row["variant"] for row in candidates} != {"A", "C"}:
        raise ValueError("Deux candidats A/C requis")
    c1 = [row for row in selection["c1_candidates"] if row["C"] == selection["selected_C1_C"]]
    if len(c1) != 1:
        raise ValueError("Un seul C1 au C global retenu est requis")
    c1 = c1[0]
    rows = [(row["variant"], row, row["fold_metrics"]) for row in candidates]
    rows.append((f"C1 (C={c1['C']})", c1, c1["folds"]))
    means = ["| candidat | moyenne AUC groupe | moyenne AUC clip |", "|---|---|---|"]
    reports = {}
    for name, candidate, folds in rows:
        if len(folds) != 3 or (name.startswith("C1") and [b["fold_id"] for b in folds] != [0, 1, 2]):
            raise ValueError("Trois folds train ordonnés sont requis pour chaque candidat retenu")
        means.append(f"| {name} | {fmt(candidate['mean_group_roc_auc'], 6)} | {fmt(candidate['mean_clip_roc_auc'], 6)} |")
        for i, block in enumerate(folds):
            if block.get("threshold") != 0.5:
                raise ValueError("Seuil diagnostique CV modifié")
            reports[f"{name}-fold{i}"] = {"model_name": name, "folds": {"cv": block}}
    return registration, selection, "\n".join(means), reports


def _v2_evaluation(path, sources, registration):
    if path is None:
        return None
    result = _v2_read(path, sources)
    if (result.get("split", {}).get("filename") != V2_EXTERNAL_NAME
            or result["split"].get("sha256") != V2_EXTERNAL_SHA256
            or not result.get("runs")):
        raise ValueError("Seule l'évaluation externe gelée peut confirmer cette V2")
    for rid, run in result["runs"].items():
        proof = run.get("fixed_threshold_provenance", {})
        threshold = proof.get("threshold")
        if (run.get("run_id") != rid or set(run.get("folds", {})) != {"external"}
                or run.get("transform") not in ("T0", "T1", "T2", "T3")
                or run.get("training_commit") != registration["code_revision"]
                or proof.get("fit_fold") != "val" or type(threshold) not in (int, float)
                or not math.isfinite(threshold) or proof.get("threshold_repr") != repr(float(threshold))
                or proof.get("validation_split_sha256") != split_loader.FROZEN_SPLIT_SHA256
                or run["folds"]["external"]["threshold"] != threshold):
            raise ValueError("Run externe mélangé avec un autre entraînement, fold ou seuil")
    return result


def _v2_audit_partition(parent, partition, report, sources):
    path = parent / "audit" / f"{partition}.jsonl"
    prediction_path = (parent / "run/predictions.csv" if partition == "external"
                       else parent / "audit/background-predictions.csv")
    predictions = _v2_predictions(prediction_path, sources)
    stored = report.get("partitions", {}).get(partition)
    result = {"partition": partition, "summary_exporter": stored,
        "expected": len(predictions) if predictions is not None else (stored or {}).get("expected"),
        "binary_target_used": False, "status": "non vérifié", "reason": "Journal absent"}
    if not path.is_file():
        return result
    rows = _v2_read(path, sources, jsonl=True)
    ids = [row["clip_id"] for row in rows]
    if len(set(ids)) != len(ids) or (predictions is not None and set(ids) != set(predictions)):
        raise ValueError("Couverture/identifiants de l'audit différents des prédictions")
    counts = {"attempted": len(rows), "errors": 0, "display_checked": 0, "invalid_display": 0,
        "displayed_class_contradictions": 0, "displayed_description_contradictions": 0,
        "score_disagreements": 0, "threshold_disagreements": 0, "fallback_used": 0}
    examples = []
    for row in rows:
        if row.get("transform") != "T0":
            raise ValueError("Journal d'une autre transformation que l'audit T0")
        payload = row.get("payload")
        failures = []
        if row.get("error") is not None:
            counts["errors"] += 1
            failures.append("error")
        elif not isinstance(payload, dict):
            counts["invalid_display"] += 1
            failures.append("invalid_display")
        else:
            score = payload.get("probability_leak")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                counts["invalid_display"] += 1
                failures.append("invalid_display")
            else:
                counts["display_checked"] += 1
                saved = predictions[row["clip_id"]] if predictions is not None else score
                checks = {"score_disagreements": saved != score,
                    "threshold_disagreements": payload.get("threshold") != report["threshold"],
                    "displayed_class_contradictions": payload.get("prediction") != ("leak" if saved >= report["threshold"] else "no_leak"),
                    "displayed_description_contradictions": payload.get("dominant_band_hz") not in
                        ("0-1000", "1000-2000", "2000-3000", "3000-4000") or payload.get("description") !=
                        f"Greatest mean spectral energy: {payload.get('dominant_band_hz')} Hz."}
                for key, contradicted in checks.items():
                    counts[key] += int(contradicted)
                    if contradicted:
                        failures.append(key)
                counts["fallback_used"] += int(payload.get("fallback_used") is True)
        if failures and len(examples) < 3:
            examples.append({"clip_id": row["clip_id"], "failures": failures, "record": row})
    failures = sum(counts[key] for key in ("errors", "invalid_display", "displayed_class_contradictions",
        "displayed_description_contradictions", "score_disagreements", "threshold_disagreements"))
    verified = (predictions is not None and len(rows) > 0 and report.get("status") == "complete"
                and report.get("weights_unchanged") is True and failures == 0)
    return {**result, **counts, "examples_first_in_journal_order": examples,
        "status": "cohérence affichée vérifiée" if verified else "non vérifié ou défaut constaté",
        "reason": None if verified else "Audit incomplet, erreurs, contradictions ou prédictions absentes",
        "raw_and_latency_counts_from_exporter": stored}


def _v2_audit(parent, sources, preregistration_sha256, evaluation):
    parent = Path(parent)
    available = [parent / "audit" / name for name in ("summary.json", "failure.json", "started.json")]
    path = next((path for path in available if path.is_file()), None)
    if path is None:
        return {"parent": str(parent), "status": "non vérifié", "reason": "Aucun reçu d'audit"}
    report = _v2_read(path, sources)
    threshold = report.get("threshold")
    if (report.get("schema") != "pipe-v2-external-export-v1" or report.get("model") != "tslm"
            or report.get("transform") != "T0" or report.get("text_audit") is not True
            or report.get("external_manifest_sha256") != V2_EXTERNAL_SHA256
            or report.get("preregistration_sha256") != preregistration_sha256
            or type(threshold) not in (int, float) or not math.isfinite(threshold)
            or report.get("threshold_repr") != repr(float(threshold))):
        raise ValueError("Audit T0 TSLM d'un autre manifeste, modèle, seuil ou campagne")
    metadata_path = parent / "run/metadata.json"
    if metadata_path.is_file():
        metadata = _v2_read(metadata_path, sources)
        if (metadata.get("split_filename") != V2_EXTERNAL_NAME or metadata.get("split_sha256") != V2_EXTERNAL_SHA256
                or any(metadata.get(key) != report.get(key) for key in ("run_id", "model_identity", "transform",
                    "threshold_provenance_sha256", "preregistration_sha256"))):
            raise ValueError("Run et audit mélangés")
        hashes = {name: _sha(parent / "run" / name) for name in ("metadata.json", "predictions.csv")}
        if report.get("run_sha256") is not None and report["run_sha256"] != hashes:
            raise ValueError("Run modifié depuis l'audit")
        if evaluation is not None:
            run = evaluation["runs"].get(report["run_id"], {})
            proof = run.get("fixed_threshold_provenance", {})
            if (proof.get("run_files_sha256") != hashes or proof.get("threshold") != threshold
                    or proof.get("threshold_provenance_sha256") != report["threshold_provenance_sha256"]):
                raise ValueError("Évaluation et audit ne décrivent pas le même run/seuil")
    return {"parent": str(parent), "run_id": report["run_id"], "exporter_status": report.get("status"),
            "model_load_ms": report.get("model_load_ms"),
            "partitions": {name: _v2_audit_partition(parent, name, report, sources)
                           for name in ("external", "background")}}


def build_v2(args):
    """Restituer uniquement les artefacts existants : aucun accès audio/split/moteur."""
    out = Path(args.out)
    if out.exists() or out.is_symlink():
        raise FileExistsError("Le rapport V2 exige un dossier neuf")
    campaign, sources = Path(args.v2_campaign), {}
    registration, selection, means, cv = _v2_cv(campaign, sources)
    evaluation = _v2_evaluation(args.v2_evaluation, sources, registration)
    audits = [_v2_audit(path, sources, _sha(campaign / "preregistration.json"), evaluation)
              for path in args.v2_audits or []]
    if len({audit.get("run_id", audit["parent"]) for audit in audits}) != len(audits):
        raise ValueError("Audit répété")
    parts = ["# Évaluation V2 — sélection et confirmation distinctes", "",
        "Restitution de JSON et journaux existants, sans nouvelle inférence ni recalcul de métriques.", "",
        "## 1. Sélection sur train uniquement", "",
        f"Candidat retenu : **{selection['selected_variant']}** ; C1 global : **C={selection['selected_C1_C']}**.", "",
        f"Règle préinscrite : `{selection['selection_rule']}`. Moyennes des trois folds groupés, "
        "pas une AUC OOF amalgamée. Ces résultats de sélection ne constituent pas une confirmation indépendante.", "", means,
        "", "## 2. Trois folds internes et erreurs au seuil diagnostique", "", metric_table(cv, "cv"), ""]
    for rid, run in cv.items():
        parts += [f"### {rid}", "", decision_table(run["folds"]["cv"],
            "au seuil 0,5 fixé pour le diagnostic CV, non sélectionné et non servi"), ""]
    parts += ["## 3. Confirmation externe T0", ""]
    if evaluation is None:
        parts += ["Non évaluée dans ce rapport : aucun JSON d'évaluation externe fourni.", ""]
    else:
        runs = evaluation["runs"]
        parts += [f"Manifeste `{V2_EXTERNAL_NAME}`, SHA `{V2_EXTERNAL_SHA256}` ; "
            f"{evaluation['split']['n_clips']} fenêtres, {evaluation['split']['n_clusters']} groupes heuristiques.", ""]
        t0 = {rid: run for rid, run in runs.items() if run["transform"] == "T0"}
        if t0:
            parts += [metric_table(t0, "external"), "", ci_table(t0, "external"), ""]
            for rid, run in t0.items():
                parts += [f"### Erreurs {rid}", "", decision_table(run["folds"]["external"]), ""]
        else:
            parts += ["Aucun run T0 fourni : confirmation principale non vérifiée.", ""]
        parts += ["### Seuils validation gelés et provenance", "",
            "| run | transformation | seuil plein | SHA reçu validation |", "|---|---|---|---|"]
        for rid, run in runs.items():
            p = run["fixed_threshold_provenance"]
            parts.append(f"| `{rid}` | {run['transform']} | `{p['threshold_repr']}` | `{p['threshold_provenance_sha256']}` |")
        parts += ["", "Les mêmes reçus sont conservés pour tous les stress d'un checkpoint ; "
            "aucun seuil n'est choisi sur l'externe. Les empreintes complètes restent dans le JSON source.", ""]
    parts += ["## 4. Sensibilité temporelle et incertitude", "",
        "T1 inverse le temps, T2 permute des blocs de 250 échantillons (31,25 ms), T3 randomise la phase. "
        "Les étiquettes ne sont pas présumées invariantes sous ces transformations ; leurs résultats "
        "sondent la sensibilité, pas une amélioration ni une dégradation terrain démontrée.", ""]
    if evaluation is not None:
        stressed = {rid: run for rid, run in evaluation["runs"].items() if run["transform"] != "T0"}
        if stressed:
            parts += [metric_table(stressed, "external"), "", ci_table(stressed, "external"), ""]
        parts += ["### Comparaisons appariées existantes", "",
                  comparison_table(evaluation.get("paired_comparisons", {})), ""]
        parts += [f"Bootstrap existant : {evaluation['bootstrap']['draws']} tirages, graine "
                  f"{evaluation['bootstrap']['seed']}, unité `{evaluation['bootstrap']['unit']}`. "
                  "Ces intervalles ne créent pas de nouvelles sessions indépendantes.", ""]
    else:
        parts += ["Stress et intervalles externes non vérifiés : résultats non fournis.", ""]
    parts += ["## 5. Cohérence du texte affiché — audit T0", "",
        "Le contrôle compare la classe affichée au score sauvegardé et au seuil plein, et la description à "
        "la bande DSP **déclarée** dans la même sortie. Ce n'est pas une preuve acoustique indépendante : "
        "la mesure source relève du gate et de CoherentPredictor. Les drapeaux de texte brut ne décident pas "
        "du compteur de contradictions affichées.", ""]
    if not audits:
        parts += ["Non vérifié : aucun audit textuel fourni. Aucun zéro défaut ou taux de réussite n'est revendiqué.", ""]
    for audit in audits:
        parts += [f"### {audit.get('run_id', audit['parent'])}", ""]
        if "partitions" not in audit:
            parts += [f"Non vérifié : {audit['reason']}.", ""]
            continue
        parts += [f"Chargement du modèle, séparé des requêtes : {fmt(audit['model_load_ms'])} ms "
                  "(— : durée non fournie).", ""]
        for partition, result in audit["partitions"].items():
            parts += [f"#### {'Primaire externe' if partition == 'external' else 'Bruit annexe — sans cible binaire'}", "",
                f"État : **{result['status']}**. Dénominateur attendu : {fmt(result['expected'], 0)}.", ""]
            if "attempted" not in result:
                parts += [f"Non vérifié : {result['reason']}.", ""]
                continue
            parts += ["| tentatives | erreurs | sorties contrôlées | invalides | contradictions classe | "
                      "contradictions description | écarts score | écarts seuil | fallback |",
                      "|---|---|---|---|---|---|---|---|---|",
                      "| " + " | ".join(str(result[k]) for k in ("attempted", "errors", "display_checked", "invalid_display",
                      "displayed_class_contradictions", "displayed_description_contradictions", "score_disagreements",
                      "threshold_disagreements", "fallback_used")) + " |", ""]
            stored = result["raw_and_latency_counts_from_exporter"]
            parts += (["Compteurs bruts et latences du producteur (conservés, distincts de ce contrôle) :", "",
                      "```json", json.dumps(stored, indent=2, ensure_ascii=False), "```", ""] if stored is not None
                      else ["Compteurs bruts et latences non vérifiés : résumé du producteur absent.", ""])
            for example in result["examples_first_in_journal_order"]:
                parts += [f"Premier défaut dans l'ordre du journal : `{example['clip_id']}` — "
                          ", ".join(example["failures"]) + ".", ""]
    parts += ["## 6. Limites de portée", "",
        "Les groupes sont heuristiques, pas des sessions d'acquisition indépendantes démontrées. "
        "L'externe est un montage expérimental de 47 m en PVC ; il ne valide ni tous les matériaux, "
        "ni les capteurs, ni un bâtiment réel. Les fenêtres d'un même enregistrement sont dépendantes. "
        "Le bruit annexe n'est jamais réétiqueté comme une classe non-fuite. Les scores ne sont pas calibrés ; "
        "aucun résultat par clip n'établit le délai de détection ou les fausses alertes par appareil-heure.", "",
        "## 7. Événement, contexte et utilité du TSLM — non démontrés", "",
        "La campagne A/C reste une étude sur extraits. Elle ne démontre ni compréhension de l'évolution "
        "d'un événement, ni comparaison à un historique, ni investigation interactive utile. Ajouter neuf "
        "mesures au prompt ne suffit pas. Le benchmark final doit comparer, sur les mêmes événements "
        "réservés et contextes disponibles : classifieur + DSP + gabarit ; classifieur + mesures/contexte + Qwen ; "
        "TSLM sur séries. Il faut des références temporelles et des critères métier avant d'affirmer "
        "que le langage ou l'accès aux séries apporte une utilité supplémentaire.", ""]
    for path, digest in sources.items():
        if _sha(path) != digest:
            raise ValueError("Une source a changé pendant la génération du rapport")
    out.mkdir(parents=True, exist_ok=False)
    report_path = out / "FINAL_EVALUATION_V2.md"
    report_path.write_text("\n".join(parts), encoding="utf-8")
    provenance = {"schema": "pipe-v2-report-v1", "metrics_recalculated": False, "audio_or_split_loaded": False,
        "code_revision": args.code_revision, "generator_sha256": _sha(__file__), "source_sha256": sources,
        "report_sha256": _sha(report_path), "external_evaluation_provided": evaluation is not None, "audits": audits}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"écrit : {report_path}, provenance.json — aucun recalcul de métrique")
    return provenance


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--tslm-run-id", default=None)
    ap.add_argument("--stress-report", default=None)
    ap.add_argument("--out")
    ap.add_argument("--code-revision", help="SHA Git complet du code transféré, y compris sans .git")
    ap.add_argument("--v2-campaign", type=Path)
    ap.add_argument("--v2-evaluation", type=Path)
    ap.add_argument("--v2-audits", nargs="+", type=Path)
    args = ap.parse_args(argv)
    if args.code_revision is not None and (len(args.code_revision) != 40
            or any(c not in "0123456789abcdef" for c in args.code_revision)):
        ap.error("--code-revision doit être un SHA complet de 40 caractères hexadécimaux")
    if args.v2_campaign is not None:
        if args.out is None:
            ap.error("La restitution V2 exige --out avec un dossier neuf")
        if args.runs or args.stress_report or args.tslm_run_id:
            ap.error("Ne pas mélanger les modes V1 et restitution V2")
        return build_v2(args)
    if args.v2_evaluation or args.v2_audits or not args.runs:
        ap.error("V1 exige --runs ; le mode V2 exige --v2-campaign")
    args.out = args.out or "artifacts/final_evaluation"

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
    commit = args.code_revision
    if commit is None:
        code_root = Path(__file__).resolve().parents[2]
        try:
            if not (code_root / ".git").exists():
                raise FileNotFoundError("Code archivé sans .git")
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, check=True, cwd=code_root).stdout.strip()
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
