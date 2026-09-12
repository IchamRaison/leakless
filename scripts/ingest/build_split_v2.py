#!/usr/bin/env python3
"""Construit `split_v2` — le split groupé qui remplace `split_v1` (invalide).

Pourquoi une v2 : `split_v1` laissait 30 conditions physiques traverser les folds
(92 clips), parce que son invariant de sécurité exigeait que les QUATRE champs de
métadonnées soient renseignés, alors que la région est absente pour 348 des 500
clips leak. Voir `docs/SPLIT_V2_AUDIT.md` §2 pour la cause racine démontrée.

Trois principes, chacun en réponse à une cause racine mesurée :

1. **Une métadonnée manquante est une incertitude, pas un filtre.** Aucun
   invariant n'est conditionné à la présence d'un champ facultatif. Les liens de
   dépendance reposent sur les champs effectivement observables, et la COUVERTURE
   de chaque invariant est rapportée à côté de son résultat.

2. **L'affectation est un vrai tirage aléatoire.** v1 triait les groupes par
   taille décroissante puis les plaçait par déficit : le premier groupe de chaque
   classe partait de compteurs à zéro et tombait toujours dans le fold de plus
   grande cible. Sur 500 seeds, test ne recevait jamais de groupe de plus de 43
   clips. Ici, chaque groupe reçoit un tirage i.i.d., et l'ensemble est accepté ou
   rejeté en bloc selon des contraintes déclarées (échantillonnage par rejet).

3. **Les groupes sont les composantes connexes d'un graphe de dépendances
   explicite**, pas le résultat d'une clé de chaîne de caractères. Chaque type
   d'arête est nommé, compté, et re-vérifiable indépendamment.

Le dataset vit HORS du dépôt. Aucun WAV n'entre dans Git.

Usage :
  python3 scripts/ingest/build_split_v2.py --data-root <hors dépôt> \\
      [--out manifests] [--seed 20260912] [--near-dup 0.7]
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import statistics
import sys
import wave

FOLDERS = {
    "leak": "leak acoustic data",
    "no_leak": "no leak acoustic data",
    "noise": "environmental noise",
}
NAME_RE = re.compile(r"^(?P<body>.+?)[-_](?P<w0>\d+)-(?P<w1>\d+)(?:_(?P<rep>\d+))?\.wav$")
NA = "NA"
SPLIT_TARGET = {"train": 0.60, "val": 0.20, "test": 0.20}

# Contraintes d'acceptation du tirage. Fixées AVANT tout résultat, et publiées :
# la position d'un groupe doit être explicable par ces deux nombres, jamais par
# un effet de bord de l'ordre de parcours.
BALANCE_TOL = 0.02          # écart max à la cible, par classe et par fold
MIN_GROUPS_PER_FOLD = 8     # groupes minimum par classe et par fold
MAX_DRAWS = 200_000


def parse(cls: str, filename: str) -> dict:
    m = NAME_RE.match(filename)
    if not m:
        return {"parsed": False, "body": filename, "material": NA, "region": NA,
                "pressure": NA, "flow": NA, "device": NA, "noise_category": NA,
                "window": -1, "rep": None}
    body = m.group("body")
    f = [x.strip() for x in body.split("-")]
    d = {"parsed": True, "body": body, "material": NA, "region": NA, "pressure": NA,
         "flow": NA, "device": NA, "noise_category": NA,
         "window": int(m.group("w0")), "rep": m.group("rep")}
    if cls == "noise" and len(f) == 3:
        d["noise_category"], d["region"], d["device"] = f
    elif cls != "noise" and len(f) == 5:
        d["material"], d["region"], d["pressure"], d["flow"], d["device"] = f
    else:
        d["parsed"] = False
    return d


def load(root: str) -> list[dict]:
    clips = []
    for cls, folder in FOLDERS.items():
        path = os.path.join(root, folder)
        if not os.path.isdir(path):
            sys.exit(f"dossier introuvable : {path}")
        for name in sorted(os.listdir(path)):
            if not name.lower().endswith(".wav"):
                continue
            rel = f"{folder}/{name}"
            with open(os.path.join(path, name), "rb") as fh:
                md5 = hashlib.md5(fh.read()).hexdigest()
            c = parse(cls, name)
            c.update(cls=cls, filename=name, path=rel, md5=md5,
                     clip_id="c" + hashlib.sha1(rel.encode()).hexdigest()[:12],
                     label="leak" if cls == "leak" else "no_leak")
            clips.append(c)
    return clips


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[rb] = ra
        return True


def near_duplicate_pairs(root: str, clips: list[dict], threshold: float):
    """Paires acoustiquement quasi identiques.

    Corrélation de log-spectrogrammes centrés sur la moyenne du corpus. Sans
    numpy/scipy on ne peut pas l'établir : on échoue bruyamment plutôt que de
    produire un split dont la reproductibilité dépend de l'environnement.
    """
    try:
        import numpy as np
        from scipy.signal import stft
    except ImportError:
        sys.exit("numpy/scipy requis : sans eux l'invariant de quasi-doublon ne "
                 "peut pas être établi et le split serait non reproductible.")

    X = np.zeros((len(clips), 8000), dtype=np.float32)
    for i, c in enumerate(clips):
        with wave.open(os.path.join(root, c["path"])) as w:
            a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
        X[i, :min(8000, len(a))] = a[:8000]
    _, _, Z = stft(X, fs=8000, nperseg=256, noverlap=128, axis=1)
    F = np.log1p(np.abs(Z)).reshape(len(clips), -1)
    F = F - F.mean(0, keepdims=True)
    F = F - F.mean(1, keepdims=True)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    C = F @ F.T
    np.fill_diagonal(C, 0.0)
    pairs = [(int(i), int(j)) for i, j in np.argwhere(np.triu(C, 1) > threshold)]
    # Sensibilité : combien de paires à d'autres seuils, pour que le choix du
    # seuil soit auditable au lieu d'être posé sans justification.
    sens = {str(t): int((np.triu(C, 1) > t).sum()) for t in (0.9, 0.8, 0.7, 0.6, 0.5)}
    return pairs, sens


def build_groups(clips: list[dict], near_pairs) -> tuple[list[int], dict]:
    """Composantes connexes du graphe de dépendances. Chaque arête est nommée."""
    uf = UnionFind(len(clips))
    stats: dict = {}

    def link_by(name: str, keyfn, coverage_note: str):
        buckets = collections.defaultdict(list)
        for i, c in enumerate(clips):
            k = keyfn(c)
            if k is not None:
                buckets[k].append(i)
        merges = 0
        for mem in buckets.values():
            for j in mem[1:]:
                merges += uf.union(mem[0], j)
        covered = sum(len(v) for v in buckets.values())
        stats[name] = {"cles": len(buckets), "clips_couverts": covered,
                       "clips_non_couverts": len(clips) - covered,
                       "fusions": merges, "couverture": coverage_note}

    # E1 — fichiers octet-identiques.
    link_by("E1_doublon_exact", lambda c: ("md5", c["md5"]), "100 % des clips")

    # E2 — quasi-doublons acoustiques (arêtes explicites, pas une clé).
    merges = sum(uf.union(i, j) for i, j in near_pairs)
    stats["E2_quasi_doublon"] = {"paires": len(near_pairs), "fusions": merges,
                                 "couverture": "100 % des clips comparés deux à deux"}

    # E3 — même session d'acquisition : mêmes métadonnées complètes + device.
    # Couvre fenêtres consécutives et répétitions du même enregistrement parent.
    link_by("E3_session", lambda c: ("body", c["cls"], c["body"]), "100 % des clips")

    # E4 — même condition physique leak, quel que soit le device.
    # Pression et débit sont les champs discriminants (4 et 2 décimales). La
    # région est IGNORÉE : elle est absente 348 fois sur 500 et l'exiger était
    # exactement la cause racine de l'échec de v1.
    link_by("E4_condition_leak_inter_device",
            lambda c: ("pf", c["pressure"], c["flow"])
            if c["cls"] == "leak" and c["pressure"] != NA and c["flow"] != NA else None,
            "clips leak dont pression ET débit sont renseignés")

    # E5 — même condition non-leak, quel que soit le device. Pression et débit
    # sont NA pour 100 % de la classe : les seuls observables sont matériau et
    # région. Contrainte volontairement grossière, son coût est mesuré.
    link_by("E5_condition_non_leak_inter_device",
            lambda c: ("mr", c["material"], c["region"]) if c["cls"] == "no_leak" else None,
            "100 % des clips no leak")

    roots = [uf.find(i) for i in range(len(clips))]
    return roots, stats


def assign_folds(groups: list[tuple[str, str, int]], seed: int) -> tuple[dict, dict]:
    """Échantillonnage par rejet : un tirage i.i.d. par groupe, accepté en bloc.

    Aucun tri, aucun placement glouton, aucun ordre privilégié. Le seul biais est
    celui des contraintes déclarées (BALANCE_TOL, MIN_GROUPS_PER_FOLD), et il est
    mesurable : relâcher les contraintes déplace les gros groupes.
    """
    total = collections.Counter()
    for _, cls, n in groups:
        total[cls] += n
    rng = random.Random(seed)
    folds = list(SPLIT_TARGET)
    weights = [SPLIT_TARGET[f] for f in folds]

    for draw in range(1, MAX_DRAWS + 1):
        a = {g: rng.choices(folds, weights=weights)[0] for g, _, _ in groups}
        clips_in = collections.defaultdict(collections.Counter)
        groups_in = collections.defaultdict(collections.Counter)
        for g, cls, n in groups:
            clips_in[cls][a[g]] += n
            groups_in[cls][a[g]] += 1
        balanced = all(abs(clips_in[c][f] / total[c] - SPLIT_TARGET[f]) <= BALANCE_TOL
                       for c in total for f in SPLIT_TARGET)
        enough = all(groups_in[c][f] >= MIN_GROUPS_PER_FOLD
                     for c in total for f in SPLIT_TARGET)
        if balanced and enough:
            return a, {"tirage_accepte": draw, "tolerance": BALANCE_TOL,
                       "groupes_min_par_fold": MIN_GROUPS_PER_FOLD}
    sys.exit(f"aucun tirage acceptable en {MAX_DRAWS} essais — relâcher les contraintes "
             f"et le documenter, jamais les ajuster après avoir vu un score")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="manifests")
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--near-dup", type=float, default=0.7)
    args = ap.parse_args()

    clips = load(args.data_root)
    unparsed = [c["filename"] for c in clips if not c["parsed"]]
    if unparsed:
        sys.exit(f"{len(unparsed)} noms non parsables : {unparsed[:5]}")

    near_pairs, sensitivity = near_duplicate_pairs(args.data_root, clips, args.near_dup)
    roots, edge_stats = build_groups(clips, near_pairs)

    members = collections.defaultdict(list)
    for i, r in enumerate(roots):
        members[r].append(clips[i]["clip_id"])
    stable = {r: "g" + hashlib.sha1("\n".join(sorted(v)).encode()).hexdigest()[:10]
              for r, v in members.items()}
    for i, c in enumerate(clips):
        c["group_id"] = stable[roots[i]]

    by_group = collections.defaultdict(list)
    for c in clips:
        by_group[c["group_id"]].append(c)
    for g, v in by_group.items():
        if len({x["label"] for x in v}) > 1:
            sys.exit(f"groupe {g} mélange deux classes — le graphe de dépendances est faux")
    groups = [(g, v[0]["label"], len(v)) for g, v in sorted(by_group.items())]

    folds, draw_info = assign_folds(groups, args.seed)
    for c in clips:
        c["fold"] = folds[c["group_id"]]

    os.makedirs(args.out, exist_ok=True)
    rows = sorted(clips, key=lambda c: c["clip_id"])
    split_path = os.path.join(args.out, "split_v2.csv")
    with open(split_path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["clip_id", "label", "label_3c", "group_id", "fold"])
        for c in rows:
            w.writerow([c["clip_id"], c["label"], c["cls"], c["group_id"], c["fold"]])

    audit_path = os.path.join(args.out, "split_v2_audit.csv")
    with open(audit_path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["clip_id", "path", "device", "material", "region", "pressure_mpa",
                    "flow_ms", "noise_category", "window", "rep", "md5", "group_id", "fold"])
        for c in rows:
            w.writerow([c["clip_id"], c["path"], c["device"], c["material"], c["region"],
                        c["pressure"], c["flow"], c["noise_category"], c["window"],
                        c["rep"] if c["rep"] is not None else "", c["md5"],
                        c["group_id"], c["fold"]])

    def sha256(p: str) -> str:
        with open(p, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    sizes = [n for _, _, n in groups]
    meta = {
        "version": "split_v2",
        "supersedes": "split_v1 (INVALIDE — 30 conditions physiques traversaient les folds)",
        "date": "2026-09-12",
        "source": "zenodo.org/records/18631450 (CC BY 4.0)",
        "script": "scripts/ingest/build_split_v2.py",
        "seed": args.seed,
        "near_dup_threshold": args.near_dup,
        "near_dup_sensibilite": sensitivity,
        "procedure": "composantes connexes d'un graphe de dépendances explicite, puis "
                     "échantillonnage par rejet : un tirage i.i.d. par groupe "
                     "(p = 0.6/0.2/0.2), accepté en bloc si chaque classe est à ±2 % "
                     "de sa cible dans chaque fold et si chaque fold porte au moins "
                     "8 groupes par classe. Aucun tri, aucun placement glouton.",
        "contraintes": draw_info,
        "aretes": edge_stats,
        "n_clips": len(clips),
        "n_groupes": len(groups),
        "taille_groupes": {"min": min(sizes), "mediane": statistics.median(sizes),
                           "max": max(sizes)},
        "sha256": {"split_v2.csv": sha256(split_path),
                   "split_v2_audit.csv": sha256(audit_path)},
        "regle": "split_v2.csv ne contient ni chemin, ni device, ni pression, ni débit. "
                 "Aucune colonne de split_v2_audit.csv n'entre dans une entrée de modèle.",
    }
    with open(os.path.join(args.out, "split_v2.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    per_fold = {}
    for f in SPLIT_TARGET:
        mem = [c for c in clips if c["fold"] == f]
        gs = collections.Counter(c["group_id"] for c in mem)
        per_fold[f] = {
            "clips": len(mem),
            "par_classe": dict(collections.Counter(c["label"] for c in mem)),
            "par_classe_3": dict(collections.Counter(c["cls"] for c in mem)),
            "groupes": len(gs),
            "groupes_par_classe": {
                k: len({c["group_id"] for c in mem if c["label"] == k})
                for k in ("leak", "no_leak")},
            "plus_gros_groupe": max(gs.values()),
        }
    print(json.dumps({"manifeste": meta, "folds": per_fold}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
