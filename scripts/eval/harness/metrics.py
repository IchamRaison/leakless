"""Métriques clip-level et cluster-level, et bootstrap sur les clusters.

Deux règles gelées, valables pour **tous** les modèles, contrôles compris :

  AGRÉGATION DE CLUSTER
      probabilité du cluster = **médiane** des probabilités de ses clips.
      Choix : la médiane résiste aux clips aberrants d'un gros cluster et
      empêche la queue d'un cluster de 104 clips de dicter son score.
      Gelée avant toute prédiction de TSLM.

  SEUIL
      argmax, sur la VALIDATION uniquement, du macro-F1 **cluster-level**.
      Le test n'intervient jamais dans le choix du seuil.

L'unité d'indépendance est le cluster de dépendance, jamais le clip. Tout
intervalle de confiance rééchantillonne des clusters entiers.
"""

from __future__ import annotations

import collections

import numpy as np

AGGREGATION = "median"          # gelé
BOOTSTRAP_SEED = 20260912       # gelé
BOOTSTRAP_DRAWS = 2000          # gelé


# --------------------------------------------------------------------------- #
# Briques élémentaires
# --------------------------------------------------------------------------- #
def _confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {"tp": int(np.sum((y == 1) & (pred == 1))), "fp": int(np.sum((y == 0) & (pred == 1))),
            "fn": int(np.sum((y == 1) & (pred == 0))), "tn": int(np.sum((y == 0) & (pred == 0)))}


def _f1(tp: int, fp: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return 0.0 if d == 0 else 2 * tp / d


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    c = _confusion(y, pred)
    return (_f1(c["tp"], c["fp"], c["fn"]) + _f1(c["tn"], c["fn"], c["fp"])) / 2


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC ROC par comptage de paires. nan si une classe est absente."""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = float((pos[:, None] > neg[None, :]).sum())
    eq = float((pos[:, None] == neg[None, :]).sum())
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Précision moyenne (aire sous précision/rappel), somme de Riemann à gauche."""
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)
    prev_r = 0.0
    total = 0.0
    for p, r in zip(precision, recall):
        total += p * (r - prev_r)
        prev_r = r
    return float(total)


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    c = _confusion(y, pred)
    tpr = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else float("nan")
    tnr = c["tn"] / (c["tn"] + c["fp"]) if (c["tn"] + c["fp"]) else float("nan")
    return float(np.nanmean([tpr, tnr]))


def brier(y: np.ndarray, s: np.ndarray) -> float:
    return float(np.mean((s - y) ** 2))


def _metric_block(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    pred = (s >= threshold).astype(int)
    c = _confusion(y, pred)
    n_neg = c["tn"] + c["fp"]
    return {
        "roc_auc": roc_auc(y, s),
        "pr_auc": pr_auc(y, s),
        "macro_f1": macro_f1(y, pred),
        "balanced_accuracy": balanced_accuracy(y, pred),
        "leak_recall": c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else float("nan"),
        "false_positive_rate": c["fp"] / n_neg if n_neg else float("nan"),
        "brier": brier(y, s),
        **c,
    }


# --------------------------------------------------------------------------- #
# Agrégation en clusters
# --------------------------------------------------------------------------- #
def aggregate_clusters(y: np.ndarray, s: np.ndarray, groups: np.ndarray):
    """Un cluster -> une observation. Règle gelée : médiane des probabilités."""
    idx: dict[str, list[int]] = collections.OrderedDict()
    for i, g in enumerate(groups):
        idx.setdefault(str(g), []).append(i)
    gids, gy, gs = [], [], []
    for g, ii in idx.items():
        labels = {int(y[i]) for i in ii}
        if len(labels) > 1:
            raise ValueError(f"le cluster {g} mélange deux classes — split invalide")
        gids.append(g)
        gy.append(labels.pop())
        gs.append(float(np.median(s[ii])))
    return np.array(gids), np.array(gy), np.array(gs)


class ThresholdFittingError(RuntimeError):
    """On a tenté d'ajuster le seuil sur autre chose que la validation."""


def pick_threshold(y: np.ndarray, s: np.ndarray, groups: np.ndarray,
                   *, fold: str = "val") -> float:
    """Seuil = argmax du macro-F1 CLUSTER-LEVEL, sur la VALIDATION uniquement.

    Le garde-fou est dans la signature, pas dans une consigne : appeler cette
    fonction avec `fold="test"` lève une erreur. Une règle qu'on ne peut pas
    enfreindre par distraction vaut mieux qu'une règle écrite dans un README.
    """
    if fold != "val":
        raise ThresholdFittingError(
            f"ajustement de seuil demandé sur le fold « {fold} ». Le seuil se "
            f"choisit sur la validation, jamais sur le test.")
    _, gy, gs = aggregate_clusters(y, s, groups)
    cands = np.unique(np.concatenate([gs, [gs.min() - 1e-9, gs.max() + 1e-9]]))
    best_f1, best_t = -1.0, float(np.median(gs))
    for t in cands:
        f = macro_f1(gy, (gs >= t).astype(int))
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t


def evaluate(y, s, groups, threshold: float) -> dict:
    """Le bloc complet : clip-level, cluster-level, et les effectifs des deux."""
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    gids, gy, gs = aggregate_clusters(y, s, groups)
    return {
        "n_clips": int(len(y)),
        "n_clusters": int(len(gids)),
        "n_clusters_leak": int(np.sum(gy == 1)),
        "n_clusters_non_leak": int(np.sum(gy == 0)),
        "n_clips_leak": int(np.sum(y == 1)),
        "n_clips_non_leak": int(np.sum(y == 0)),
        "threshold": float(threshold),
        "aggregation": AGGREGATION,
        "clip_level": {k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in _metric_block(y, s, threshold).items()},
        "cluster_level": {k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in _metric_block(gy, gs, threshold).items()},
    }


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def _cluster_draws(groups: np.ndarray, draws: int, seed: int) -> list[np.ndarray]:
    """Tirages de clusters partagés. Les MÊMES tirages servent à tous les modèles.

    C'est ce qui rend les comparaisons appariées : deux modèles voient exactement
    le même rééchantillonnage, donc la différence mesurée n'absorbe pas la
    variabilité du tirage.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    return [rng.choice(uniq, size=len(uniq), replace=True) for _ in range(draws)]


def _resample(idx_by_group: dict, drawn: np.ndarray):
    idx = np.concatenate([idx_by_group[g] for g in drawn])
    gp = np.concatenate([[f"{g}#{k}"] * len(idx_by_group[g]) for k, g in enumerate(drawn)])
    return idx, gp


def bootstrap_ci(y, s, groups, threshold: float, *, draws: int = BOOTSTRAP_DRAWS,
                 seed: int = BOOTSTRAP_SEED) -> dict:
    """IC 95 % en rééchantillonnant des CLUSTERS entiers."""
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    acc: dict[str, list[float]] = collections.defaultdict(list)
    for drawn in _cluster_draws(groups, draws, seed):
        idx, gp = _resample(idx_by_group, drawn)
        yy, ss = y[idx], s[idx]
        if len(np.unique(yy)) < 2:
            continue
        pred = (ss >= threshold).astype(int)
        acc["clip_roc_auc"].append(roc_auc(yy, ss))
        acc["clip_macro_f1"].append(macro_f1(yy, pred))
        _, gy, gs = aggregate_clusters(yy, ss, gp)
        if len(np.unique(gy)) >= 2:
            acc["cluster_roc_auc"].append(roc_auc(gy, gs))
            acc["cluster_macro_f1"].append(macro_f1(gy, (gs >= threshold).astype(int)))
    out = {"draws_effectifs": {}, "seed": seed}
    for k, v in acc.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[k] = {"ci95_low": round(float(lo), 6), "ci95_high": round(float(hi), 6)}
        out["draws_effectifs"][k] = len(v)
    return out


def paired_bootstrap_delta(y, groups, score_a, score_b, threshold_a: float,
                           threshold_b: float, *, draws: int = BOOTSTRAP_DRAWS,
                           seed: int = BOOTSTRAP_SEED) -> dict:
    """Différence appariée A − B sur les MÊMES tirages de clusters.

    Comparer deux intervalles indépendants est une erreur : elle ignore que les
    deux modèles voient les mêmes données et gonfle l'incertitude de la
    différence. Ici un seul tirage sert aux deux, clip par clip.

    La lecture est délibérément qualitative. Avec 41 clusters en test, aucun
    écart n'est déclaré significatif.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    a, b = np.asarray(score_a), np.asarray(score_b)
    if not (len(y) == len(a) == len(b) == len(groups)):
        raise ValueError("populations de clips différentes : comparaison appariée impossible")
    idx_by_group = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    deltas: dict[str, list[float]] = collections.defaultdict(list)
    for drawn in _cluster_draws(groups, draws, seed):
        idx, gp = _resample(idx_by_group, drawn)
        yy, aa, bb = y[idx], a[idx], b[idx]
        if len(np.unique(yy)) < 2:
            continue
        deltas["clip_roc_auc"].append(roc_auc(yy, aa) - roc_auc(yy, bb))
        deltas["clip_macro_f1"].append(
            macro_f1(yy, (aa >= threshold_a).astype(int))
            - macro_f1(yy, (bb >= threshold_b).astype(int)))
        _, gy, gsa = aggregate_clusters(yy, aa, gp)
        _, _, gsb = aggregate_clusters(yy, bb, gp)
        if len(np.unique(gy)) >= 2:
            deltas["cluster_roc_auc"].append(roc_auc(gy, gsa) - roc_auc(gy, gsb))
            deltas["cluster_macro_f1"].append(
                macro_f1(gy, (gsa >= threshold_a).astype(int))
                - macro_f1(gy, (gsb >= threshold_b).astype(int)))

    def observed(metric: str) -> float:
        if metric.startswith("clip_roc"):
            return roc_auc(y, a) - roc_auc(y, b)
        if metric.startswith("clip_macro"):
            return (macro_f1(y, (a >= threshold_a).astype(int))
                    - macro_f1(y, (b >= threshold_b).astype(int)))
        _, gy, gsa = aggregate_clusters(y, a, groups)
        _, _, gsb = aggregate_clusters(y, b, groups)
        if metric.startswith("cluster_roc"):
            return roc_auc(gy, gsa) - roc_auc(gy, gsb)
        return (macro_f1(gy, (gsa >= threshold_a).astype(int))
                - macro_f1(gy, (gsb >= threshold_b).astype(int)))

    out = {"seed": seed, "n_clusters": int(len(idx_by_group))}
    for k, v in deltas.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[k] = {
            "delta_observe": round(float(observed(k)), 6),
            "ci95_low": round(float(lo), 6),
            "ci95_high": round(float(hi), 6),
            "lecture": verdict(float(lo), float(hi)),
            "draws_effectifs": len(v),
        }
    return out


def verdict(lo: float, hi: float) -> str:
    """Formulation imposée. Jamais « significatif » sur un jeu de cette taille."""
    if lo > 0:
        return "compatible with improvement"
    if hi < 0:
        return "compatible with degradation"
    return "inconclusive"
