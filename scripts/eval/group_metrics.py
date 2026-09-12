"""Métriques group-aware, partagées par le contrôle RMS et la baseline.

Applique `docs/EVAL_PROTOCOL.md` §7ter :

  B. tout score est rapporté clip-level ET group-level, avec le nombre de groupes
  C. les intervalles de confiance sont bootstrapés sur les GROUPES, jamais sur les clips
  D. le seuil est calibré sur la validation, pondéré par groupe

L'unité d'indépendance est la grappe de dépendance, pas le clip. Une grappe de 104
clips et une de 2 clips comptent chacune pour **une** unité au niveau groupe.

Aucune de ces fonctions ne touche au test pour choisir quoi que ce soit.
"""

from __future__ import annotations

import collections

import numpy as np

POS = "leak"


def _counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 macro : moyenne non pondérée du F1 de chaque classe."""
    c = _counts(y_true, y_pred)
    f1_pos = _f1(c["tp"], c["fp"], c["fn"])
    f1_neg = _f1(c["tn"], c["fn"], c["fp"])
    return (f1_pos + f1_neg) / 2


def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUC par comptage direct des paires (Mann-Whitney). Renvoie nan si une classe manque."""
    pos, neg = score[y_true == 1], score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def clip_level(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """Métriques où chaque clip pèse pareil. Le chiffre conventionnel."""
    y_pred = (score >= threshold).astype(int)
    c = _counts(y_true, y_pred)
    n_neg = c["tn"] + c["fp"]
    return {
        "n_clips": int(len(y_true)),
        "accuracy": float(np.mean(y_pred == y_true)),
        "macro_f1": macro_f1(y_true, y_pred),
        "leak_recall": c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else float("nan"),
        "false_alarm_rate": c["fp"] / n_neg if n_neg else float("nan"),
        "auc": auc(y_true, score),
        **c,
    }


def _aggregate_groups(y_true: np.ndarray, score: np.ndarray, groups: np.ndarray):
    """Une grappe -> une observation : étiquette de la grappe et score médian."""
    order = collections.OrderedDict()
    for i, g in enumerate(groups):
        order.setdefault(g, []).append(i)
    gids, gy, gs = [], [], []
    for g, idx in order.items():
        labels = {int(y_true[i]) for i in idx}
        if len(labels) > 1:
            raise ValueError(f"la grappe {g} mélange deux classes — le split est invalide")
        gids.append(g)
        gy.append(labels.pop())
        gs.append(float(np.median(score[idx])))
    return np.array(gids), np.array(gy), np.array(gs)


def group_level(y_true: np.ndarray, score: np.ndarray, groups: np.ndarray,
                threshold: float) -> dict:
    """Métriques où chaque grappe pèse pareil, quelle que soit sa taille."""
    gids, gy, gs = _aggregate_groups(y_true, score, groups)
    gpred = (gs >= threshold).astype(int)
    c = _counts(gy, gpred)
    n_neg = c["tn"] + c["fp"]
    # Taux de clips corrects par grappe, moyenné SANS pondération par la taille.
    per_group_acc = []
    for g in gids:
        m = groups == g
        per_group_acc.append(float(np.mean((score[m] >= threshold).astype(int) == y_true[m])))
    return {
        "n_groupes": int(len(gids)),
        "n_groupes_leak": int(np.sum(gy == 1)),
        "n_groupes_non_leak": int(np.sum(gy == 0)),
        "accuracy_moyenne_par_groupe": float(np.mean(per_group_acc)),
        "ecart_type_entre_groupes": float(np.std(per_group_acc)),
        "macro_f1_sur_medianes": macro_f1(gy, gpred),
        "auc_sur_medianes": auc(gy, gs),
        "leak_recall": c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else float("nan"),
        "false_alarm_rate": c["fp"] / n_neg if n_neg else float("nan"),
    }


def cluster_bootstrap(y_true: np.ndarray, score: np.ndarray, groups: np.ndarray,
                      threshold: float, n: int = 2000, seed: int = 20260912) -> dict:
    """IC 95 % en rééchantillonnant des GRAPPES entières, jamais des clips.

    Un bootstrap au clip rééchantillonne à l'intérieur de sessions quasi identiques et
    produit des intervalles bien trop étroits. C'est le point §7ter-C du protocole.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    stats = {"macro_f1_clip": [], "auc_clip": [], "auc_groupe": []}
    for _ in range(n):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in drawn])
        yt, sc = y_true[idx], score[idx]
        if len(np.unique(yt)) < 2:
            continue
        gp = np.concatenate([[f"{g}#{k}"] * len(idx_by_group[g]) for k, g in enumerate(drawn)])
        stats["macro_f1_clip"].append(macro_f1(yt, (sc >= threshold).astype(int)))
        stats["auc_clip"].append(auc(yt, sc))
        try:
            _, gy, gs = _aggregate_groups(yt, sc, gp)
            stats["auc_groupe"].append(auc(gy, gs))
        except ValueError:
            continue
    out = {}
    for k, v in stats.items():
        if v:
            lo, hi = np.percentile(v, [2.5, 97.5])
            out[k] = {"ic95_bas": round(float(lo), 4), "ic95_haut": round(float(hi), 4),
                      "n_tirages": len(v)}
    return out


def pick_threshold(y_true: np.ndarray, score: np.ndarray, groups: np.ndarray) -> float:
    """Choisit le seuil sur la VALIDATION, pondéré par groupe (§7ter-D).

    Le critère est le F1 macro calculé sur les médianes de grappe : une grappe de
    80 clips compte pour 1 unité, pas 80. Sans cela le seuil serait dicté par la
    plus grosse session du fold.
    """
    _, gy, gs = _aggregate_groups(y_true, score, groups)
    candidates = np.unique(np.concatenate([gs, [gs.min() - 1e-9, gs.max() + 1e-9]]))
    best, best_t = -1.0, float(np.median(gs))
    for t in candidates:
        f = macro_f1(gy, (gs >= t).astype(int))
        if f > best:
            best, best_t = f, float(t)
    return best_t


def full_report(name: str, y_true, score, groups, threshold: float) -> dict:
    """Le triptyque imposé : clip-level, group-level, et le nombre de grappes."""
    y_true, score, groups = np.asarray(y_true), np.asarray(score), np.asarray(groups)
    return {
        "fold": name,
        "seuil": round(float(threshold), 6),
        "clip_level": {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in clip_level(y_true, score, threshold).items()},
        "group_level": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in group_level(y_true, score, groups, threshold).items()},
        "ic95_bootstrap_sur_groupes": cluster_bootstrap(y_true, score, groups, threshold),
    }
