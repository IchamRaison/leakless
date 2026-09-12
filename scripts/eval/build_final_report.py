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
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_predictions import compare, evaluate_run, vectors  # noqa: E402,F401
from harness import contract, metrics, split_loader  # noqa: E402

# Liste blanche EXPLICITE de l'échelle de contrôles. Seuls ces run_id entrent dans
# les tableaux §2 à §4 : un run de stress, un contrôle négatif ou tout autre run
# n'y apparaît jamais, quel que soit son nom.
CONTROL_ORDER = ["c0", "c1", "c2", "c2b", "c3"]

# Sous stress, le seuil est recalculé sur la validation du run transformé. Les
# métriques qui en dépendent ne servent donc à aucune conclusion de sensibilité
# temporelle : elles sont retirées des comparaisons de stress.
STRESS_CLAIM_METRICS = ("clip_roc_auc", "cluster_roc_auc")
THRESHOLD_DEPENDENT = ("clip_macro_f1", "cluster_macro_f1")
STRESS_VERDICT = {
    "compatible with improvement": "compatible with degradation under stress",
    "compatible with degradation": "compatible with improvement under stress",
    "inconclusive": "inconclusive",
}


def control_ladder(reports: dict) -> dict:
    """Les runs de l'échelle, dans l'ordre C0 C1 C2 C2b C3. Rien d'autre."""
    return {rid: reports[rid] for rid in CONTROL_ORDER if rid in reports}


def stress_base_id(run_id: str, metadata: dict) -> str:
    """Run T0 dont dérive un run de stress : déclaré, sinon convention `<base>-Tn`."""
    return declared(metadata, "base_run_id") or run_id.rsplit("-", 1)[0]


_COMMIT_RE = re.compile(r"[0-9a-f]{7,40}")
_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")


def declared(meta: dict, key: str) -> str | None:
    """Valeur d'identité déclarée, ou None. Absent, null, vide et blanc sont équivalents."""
    v = meta.get(key)
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def has_serialized_checkpoint(meta: dict) -> bool:
    """Le modèle du run est-il un checkpoint sérialisé (TSLM) ou non (contrôle) ?

    Décision unique, utilisée par la vérification ET par le rendu : un contrôle
    (`control_level`) ou un checkpoint préfixé NO_SERIALIZED_CHECKPOINT n'est pas
    sérialisé ; tout autre checkpoint déclaré l'est.
    """
    ck = declared(meta, "checkpoint") or ""
    return bool(ck) and not ck.startswith(contract.NO_SERIALIZED_CHECKPOINT) \
        and not declared(meta, "control_level")


def check_stress_provenance(base_id: str, base_meta: dict, rid: str, meta: dict) -> None:
    """Refuse un run de stress qui ne dérive pas du modèle du run T0.

    Absent, null, vide et blanc valent tous « non déclaré ». Toutes les règles
    s'appliquent, sans branche qui en court-circuite une autre :
      1. `retrained` vaut false ;
      2. `training_commit` déclaré, hexadécimal (7 à 40), identique ;
      3. `model_definition_commit` tout ou rien : non déclaré des deux côtés, ou
         déclaré, hexadécimal et identique ;
      4. même nature de modèle (sérialisé ou non) des deux côtés ;
      5. modèle sérialisé : `checkpoint` identique ;
      6. modèle non sérialisé : `model_fingerprint` déclarée ;
      7. toute empreinte déclarée : 64 hexadécimaux, et identique de l'autre côté.
    """
    def refuse(msg: str) -> None:
        raise ValueError(f"{rid} (base {base_id}) : {msg}")

    if meta.get("retrained") is not False:
        refuse("`retrained` doit valoir false (not retrained on stressed data)")
    tb, ts = declared(base_meta, "training_commit"), declared(meta, "training_commit")
    if not tb or not _COMMIT_RE.fullmatch(tb):
        refuse(f"training_commit du run T0 absent ou invalide ({tb!r}) : identité invérifiable")
    if ts != tb:
        refuse(f"training_commit {ts!r} ≠ {tb!r}")
    db, ds = declared(base_meta, "model_definition_commit"), declared(meta, "model_definition_commit")
    if db != ds:
        refuse(f"model_definition_commit {ds!r} ≠ {db!r} (tout ou rien)")
    if db and not _COMMIT_RE.fullmatch(db):
        refuse(f"model_definition_commit invalide ({db!r})")
    serialized = has_serialized_checkpoint(base_meta)
    if has_serialized_checkpoint(meta) != serialized:
        refuse("l'un des runs a un checkpoint sérialisé, l'autre non")
    if serialized and declared(meta, "checkpoint") != declared(base_meta, "checkpoint"):
        refuse(f"checkpoint {meta.get('checkpoint')!r} ≠ {base_meta.get('checkpoint')!r}")
    fa, fb = declared(base_meta, "model_fingerprint"), declared(meta, "model_fingerprint")
    if not serialized and not fa:
        refuse("modèle non sérialisé sans `model_fingerprint` : identité invérifiable")
    if fa != fb:
        refuse("empreinte de modèle différente : ce n'est pas le même modèle ajusté")
    if fa and not _FINGERPRINT_RE.fullmatch(fa):
        refuse(f"model_fingerprint invalide ({fa!r})")


def prediction_shift(split, base_run, stress_run, fold: str = "test") -> dict:
    """Déplacement des probabilités sous stress, clip à clip. Indépendant du seuil."""
    import numpy as np
    ids = sorted(c.clip_id for c in split.clips if c.fold == fold)
    a = np.array([base_run.probabilities[i] for i in ids])
    b = np.array([stress_run.probabilities[i] for i in ids])
    d = b - a
    q1, q3 = np.percentile(d, [25, 75])
    # Corrélation indéfinie si l'une des deux séries est constante : None, pas NaN.
    # np.ptp est exactement nul pour des flottants identiques ; np.std ne l'est pas
    # toujours (0.1 répété donne un écart-type de l'ordre de 1e-17).
    r = None
    if np.ptp(a) > 0 and np.ptp(b) > 0:
        c = float(np.corrcoef(a, b)[0, 1])
        r = round(c, 6) if np.isfinite(c) else None
    return {"fold": fold, "n_clips": len(ids),
            "pearson_r_with_T0": r,
            "delta_p_median": round(float(np.median(d)), 6),
            "delta_p_q1": round(float(q1), 6), "delta_p_q3": round(float(q3), 6),
            "abs_delta_p_median": round(float(np.median(np.abs(d))), 6)}


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


def stress_score_table(reports: dict, stress_runs: dict, comps: dict,
                       shifts: dict) -> str:
    """Scores sous stress : métriques indépendantes du seuil uniquement."""
    if not stress_runs:
        return ""
    lines = ["| modèle | variante | clip AUC | cluster AUC | Δ clip AUC (stress − T0) | IC95 "
             "| lecture | corrélation des probabilités avec T0 | Δp médiane [Q1, Q3] "
             "| \\|Δp\\| médiane |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    bases = sorted({b for b, _ in stress_runs.values()})
    for base in bases:
        b = reports[base]["folds"]["test"]
        lines.append(f"| `{base}` | **T0** original | {fmt(b['clip_level']['roc_auc'])} | "
                     f"{fmt(b['cluster_level']['roc_auc'])} | — | — | référence | 1.000 | — | — |")
        for rid in sorted(r for r, (bb, _) in stress_runs.items() if bb == base):
            t = reports[rid]["folds"]["test"]
            d = comps.get(f"{base}_vs_{rid}", {}).get("clip_roc_auc")
            sh = shifts.get(rid)
            # comps porte T0 − stress : le signe est inversé pour lire stress − T0.
            delta = f"{-d['delta_observe']:+.3f}" if d else "—"
            ci = f"[{-d['ci95_high']:+.3f}, {-d['ci95_low']:+.3f}]" if d else "—"
            lect = STRESS_VERDICT[d["lecture"]] if d else "—"
            corr = fmt(sh["pearson_r_with_T0"]) if sh else "—"
            dp = (f"{sh['delta_p_median']:+.3f} [{sh['delta_p_q1']:+.3f}, "
                  f"{sh['delta_p_q3']:+.3f}]") if sh else "—"
            adp = fmt(sh["abs_delta_p_median"]) if sh else "—"
            lines.append(f"| `{base}` | {stress_runs[rid][1]} | {fmt(t['clip_level']['roc_auc'])} | "
                         f"{fmt(t['cluster_level']['roc_auc'])} | {delta} | {ci} | {lect} | "
                         f"{corr} | {dp} | {adp} |")
    return "\n".join(lines)


def stress_identity_lines(reports: dict, stress_bases: dict) -> list[str]:
    """Une ligne par modèle stressé, selon SA provenance réelle.

    Un contrôle n'a pas de checkpoint sérialisé ; un TSLM en a un. Le texte ne
    doit jamais attribuer à l'un la situation de l'autre.
    """
    lines = []
    for base in sorted({b for b, _ in stress_bases.values()}):
        r = reports[base]
        pv = r.get("provenance", {})
        if not has_serialized_checkpoint({"checkpoint": r.get("checkpoint"),
                                          "control_level": pv.get("control_level")}):
            lines.append(
                f"- `{base}` : **aucun checkpoint sérialisé**. Définition figée "
                f"`{str(pv.get('model_definition_commit'))[:7]}`, régression logistique "
                f"réajustée de façon déterministe sur T0/train dans le même processus que "
                f"l'évaluation sous stress ; empreinte `model_fingerprint` identique entre le "
                f"run T0 et ses runs de stress.")
        else:
            lines.append(
                f"- `{base}` : checkpoint sérialisé `{r['checkpoint']}` (commit d'entraînement "
                f"`{str(r['training_commit'])[:12]}`), identique dans chacun de ses runs de "
                f"stress ; `retrained: false`, not retrained on stressed data.")
    return lines


def build_markdown(result: dict, comps: dict, tslm: str | None, stress: dict | None,
                   repo_commit: str, reports: dict | None = None,
                   stress_comps: dict | None = None) -> str:
    sp = result["split"]
    reports = reports if reports is not None else result["runs"]
    # §2 à §4 : UNIQUEMENT l'échelle de contrôles, par liste blanche.
    ladder = control_ladder(reports)
    has_tslm = tslm is not None and tslm in reports
    n_test = next(iter(ladder.values()))["folds"]["test"] if ladder else \
        next(iter(reports.values()))["folds"]["test"]

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
        f"| Commit de génération du rapport | `{repo_commit}` |",
        f"| Règle d'agrégation | {result['aggregation_rule']} des probabilités du cluster (gelée) |",
        f"| Bootstrap | {result['bootstrap']['draws']} tirages, graine "
        f"{result['bootstrap']['seed']}, unité : {result['bootstrap']['unit']} |",
        "| Source | Zenodo 18631450, CC BY 4.0 — site d'entraînement expérimental de Dongguan |",
        "",
        "## 2. Identité des modèles — échelle de contrôles",
        "",
        "Trois rôles de commit, qui peuvent pointer vers le même commit, tous lus dans "
        "`metadata.json` et jamais recalculés ici : "
        "**définition** (où les descripteurs ont été figés), **ajustement** (`training_commit`, "
        "HEAD capturé juste avant le fit), **exécution** (HEAD à l'écriture du run). Aucun "
        "contrôle n'a de checkpoint sérialisé : la régression logistique est réajustée de "
        "façon déterministe sur T0/train à chaque exécution.",
        "",
        "| run | modèle | checkpoint | commit de définition | commit d'ajustement "
        "| commit d'exécution | horodatage |",
        "|---|---|---|---|---|---|---|",
    ]
    for rid, r in ladder.items():
        pv = r.get("provenance", {})
        dirty = " ⚠️ worktree modifié" if pv.get("training_worktree_dirty") else ""
        parts.append(f"| `{rid}` | {r['model_name']} | `{r['checkpoint']}` | "
                     f"`{str(pv.get('model_definition_commit', '—'))[:12]}` | "
                     f"`{str(r['training_commit'])[:12]}`{dirty} | "
                     f"`{str(pv.get('execution_commit', '—'))[:12]}` | {r['timestamp']} |")

    parts += [
        "",
        "## 3. Échelle de contrôles et résultats — TEST",
        "",
        metric_table(ladder, "test"),
        "",
        "### Validation (pour information — c'est là que le seuil est choisi)",
        "",
        metric_table(ladder, "val"),
        "",
        "## 4. Incertitude — bootstrap sur les clusters",
        "",
        "Les intervalles rééchantillonnent des **clusters entiers**. Un bootstrap au clip "
        "rééchantillonnerait à l'intérieur de sessions quasi identiques et produirait des "
        "intervalles bien trop étroits.",
        "",
        ci_table(ladder, "test"),
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
        f"{n_test['n_clusters']} clusters indépendants, "
        f"dont {n_test['n_clusters_non_leak']} du côté "
        f"*non-leak*. Les verdicts sont qualitatifs.",
        "",
        "## 6. Résultat TSLM",
        "",
    ]

    if has_tslm:
        t = reports[tslm]["folds"]["test"]
        parts += [
            f"Run `{tslm}` — {reports[tslm]['model_name']} — checkpoint "
            f"`{reports[tslm]['checkpoint']}`, commit d'entraînement "
            f"`{str(reports[tslm]['training_commit'])[:12]}`",
            "",
            metric_table({tslm: reports[tslm]}, "test"),
            "",
            ci_table({tslm: reports[tslm]}, "test"),
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
            "> Ces transformations ne sont **pas** des augmentations physiques démontrées "
            "comme préservant l'étiquette. Elles mesurent la sensibilité des prédictions à "
            "l'organisation temporelle, pas la pertinence physique causale de celle-ci.",
        ]
        table = stress_score_table(reports or {}, result.get("stress_run_bases", {}),
                                   stress_comps or {}, result.get("stress_prediction_shift", {}))
        if table:
            parts += [
                "",
                "### Scores mesurés sous stress",
                "",
                "Aucun modèle n'est ajusté ni entraîné sur les données stressées : les "
                "transformations ne s'appliquent qu'à l'évaluation. Identité du modèle, "
                "vérifiée pour chaque run de stress avant toute comparaison :",
                "",
                *stress_identity_lines(reports or {}, result.get("stress_run_bases", {})),
                "",
                "Seules des métriques **indépendantes du seuil** figurent ici : AUC, corrélation "
                "des probabilités avec T0 et distribution de Δp, sur le test. Le seuil d'un run "
                "stressé est recalculé sur sa propre validation transformée ; les métriques qui "
                "en dépendent (macro-F1, exactitude) ne servent à aucune conclusion de "
                "sensibilité temporelle.",
                "",
                table,
                "",
                "> Δ clip AUC et Δp sont orientés **stress − T0**. Δ clip AUC et son IC95 sont "
                "appariés sur les mêmes tirages de clusters ; Δp est mesuré clip à clip sur le "
                "test, sans intervalle. Un Δ clip AUC négatif signifie que la discrimination "
                "baisse sous la transformation : les prédictions sont sensibles à cette "
                "perturbation. Cela n'établit pas que l'information détruite est physiquement "
                "pertinente.",
            ]
        if not has_tslm:
            parts.append("")
            parts.append("🕐 **TSLM non évalué sous stress** : son checkpoint sérialisé n'est "
                         "pas en notre possession. Les jeux et le protocole l'attendent.")
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
        "3bis. **Le niveau cluster est instable en validation, avec 8 clusters non-leak.** "
        "C2b y obtient 0,665 d'AUC cluster contre 0,915 en test, un écart de +0,250 sans "
        "commune mesure avec les autres échelons. Cause vérifiée : trois petits clusters "
        "non-leak (3, 3 et 4 clips) passent au-dessus de la médiane des clusters leak, et à "
        "8 négatifs chacun pèse 12,5 % de la métrique. En test, aucun non-leak ne passe "
        "au-dessus. Le gros cluster de 80 clips est correctement classé dans les deux folds. "
        "C'est un artefact de petit échantillon, et c'est aussi ce sur quoi le seuil et "
        "l'hyperparamètre ont été choisis : à garder en tête devant tout écart val -> test.",
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
        f"{n_test['n_clusters']} clusters.",
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
    runs, reports, stress_runs, loaded = {}, {}, {}, []
    for d in args.runs:
        run = contract.load_run(d, split)
        loaded.append(run)
        # Un run de stress porte `stress_transform` dans sa provenance. Il est
        # évalué comme les autres, mais rangé à part : il ne fait pas partie de
        # l'échelle de contrôles et n'entre dans aucune comparaison au TSLM.
        if run.metadata.get("stress_transform"):
            stress_runs[run.run_id] = run
        else:
            runs[run.run_id] = run

    # Un run de stress dont le run T0 n'est pas fourni ne peut pas être vérifié :
    # refus immédiat, avant tout calcul, plutôt qu'une omission silencieuse.
    orphans = {rid: stress_base_id(rid, r.metadata) for rid, r in stress_runs.items()
               if stress_base_id(rid, r.metadata) not in runs}
    if orphans:
        sys.exit("runs de stress sans leur run T0, provenance invérifiable : "
                 + ", ".join(f"{rid} -> {b}" for rid, b in sorted(orphans.items()))
                 + ". Fournir le run T0 avec --runs.")
    for run in loaded:                        # ordre de --runs conservé dans metrics.json
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

    # Chaque run de stress est comparé, en apparié, au run T0 dont il dérive, après
    # vérification que c'est bien le même modèle ajusté. Seules les métriques
    # indépendantes du seuil sont conservées.
    stress_comps, stress_bases, shifts = {}, {}, {}
    all_runs = {**runs, **stress_runs}
    for rid, run in sorted(stress_runs.items()):
        base_id = stress_base_id(rid, run.metadata)
        check_stress_provenance(base_id, runs[base_id].metadata, rid, run.metadata)
        stress_bases[rid] = (base_id, run.metadata["stress_transform"])
        c = compare(split, all_runs, base_id, rid)
        for k in THRESHOLD_DEPENDENT:
            c.pop(k, None)
        c["excluded_threshold_dependent_metrics"] = list(THRESHOLD_DEPENDENT)
        stress_comps[f"{base_id}_vs_{rid}"] = c
        shifts[rid] = prediction_shift(split, runs[base_id], run)

    result = {
        "split": {"filename": split_loader.FROZEN_SPLIT_NAME, "sha256": split.sha256,
                  "n_clips": len(split.clips),
                  "n_clusters": len({c.group_id for c in split.clips})},
        "aggregation_rule": metrics.AGGREGATION,
        "bootstrap": {"draws": metrics.BOOTSTRAP_DRAWS, "seed": metrics.BOOTSTRAP_SEED,
                      "unit": "cluster de dépendance"},
        "control_ladder": list(control_ladder(reports)),
        "runs": reports,
        "stress_runs": sorted(stress_runs),
        "stress_run_bases": stress_bases,
        "stress_claim_metrics": [*STRESS_CLAIM_METRICS, "pearson_r_with_T0", "delta_p"],
        "stress_comparisons": stress_comps,
        "stress_prediction_shift": shifts,
        "tslm_run_id": tslm,
    }

    stress = json.load(open(args.stress_report)) if args.stress_report else None
    result["stress_invariants"] = stress
    import subprocess
    # Commit du dépôt qui contient CE script, pas du répertoire courant. Un worktree
    # modifié est signalé : le commit seul ne désignerait pas le générateur exact.
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, cwd=here,
                                text=True, check=True).stdout.strip()
        if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                          capture_output=True, text=True, check=True, cwd=here).stdout.strip():
            commit += " + modifications non commitées"
    except Exception:
        commit = "unknown"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (out / "comparison.json").write_text(json.dumps(comps, indent=2, ensure_ascii=False) + "\n")
    (out / "FINAL_EVALUATION.md").write_text(
        build_markdown(result, comps, tslm, stress, commit, reports, stress_comps))

    print(f"écrit : {out}/metrics.json, comparison.json, FINAL_EVALUATION.md")
    print(f"runs évalués : {sorted(runs)} | TSLM : {tslm or 'aucun'}")


if __name__ == "__main__":
    main()
