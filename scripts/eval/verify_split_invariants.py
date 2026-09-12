#!/usr/bin/env python3
"""Vérifie les invariants de fuite d'un split, en les reconstruisant depuis la source.

Ce script **ne fait confiance à aucune colonne de groupage du manifeste**. Il ne
lit que `clip_id` et `fold`. Tout le reste — md5, quasi-doublons, sessions,
conditions physiques — est recalculé depuis les fichiers audio et leurs noms,
puis confronté à l'affectation des folds.

C'est la différence avec l'ancien `verify_manifest.py`, qui recopiait le prédicat
du script qu'il était censé contrôler et ne pouvait donc pas détecter son erreur.

Invariants vérifiés — chacun doit valoir 0 violation :

  I1  deux fichiers octet-identiques dans deux folds différents
  I2  deux clips acoustiquement quasi identiques dans deux folds différents
  I3  deux clips de la même session d'acquisition dans deux folds différents
  I4  même condition physique leak (pression, débit) dans deux folds différents
  I5  même condition non-leak (matériau, région) dans deux folds différents

Chaque invariant rapporte sa COUVERTURE : le nombre de clips sur lesquels il a
pu être évalué. Un invariant à 0 violation sur 18 % des clips ne dit presque
rien, et c'est précisément l'erreur qui a invalidé `split_v1`.

Usage :
  python3 scripts/eval/verify_split_invariants.py --data-root <hors dépôt> \\
      --manifest manifests/split_v2.csv [--near-dup 0.7]

Code de sortie 1 si un invariant est violé. Aucun entraînement, aucun réseau.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import os
import re
import sys
import wave

FOLDERS = {
    "leak": "leak acoustic data",
    "no_leak": "no leak acoustic data",
    "noise": "environmental noise",
}
NAME_RE = re.compile(r"^(?P<body>.+?)[-_](?P<w0>\d+)-(?P<w1>\d+)(?:_(?P<rep>\d+))?\.wav$")
NA = "NA"


def load_source(root: str) -> dict:
    """Reconstruit tout depuis le disque. Ne lit aucun manifeste."""
    clips = {}
    for cls, folder in FOLDERS.items():
        d = os.path.join(root, folder)
        if not os.path.isdir(d):
            sys.exit(f"dossier introuvable : {d}")
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith(".wav"):
                continue
            rel = f"{folder}/{name}"
            cid = "c" + hashlib.sha1(rel.encode()).hexdigest()[:12]
            with open(os.path.join(d, name), "rb") as fh:
                md5 = hashlib.md5(fh.read()).hexdigest()
            m = NAME_RE.match(name)
            if not m:
                sys.exit(f"nom non parsable : {name}")
            fields = [x.strip() for x in m.group("body").split("-")]
            mat = reg = pres = flow = NA
            if cls != "noise" and len(fields) == 5:
                mat, reg, pres, flow, _dev = fields
            elif cls == "noise" and len(fields) == 3:
                _cat, reg, _dev = fields
            clips[cid] = {"path": rel, "cls": cls, "md5": md5, "body": m.group("body"),
                          "material": mat, "region": reg, "pressure": pres, "flow": flow}
    return clips


def near_pairs(root: str, order: list[str], clips: dict, threshold: float):
    try:
        import numpy as np
        from scipy.signal import stft
    except ImportError:
        sys.exit("numpy/scipy requis pour vérifier l'invariant de quasi-doublon.")
    X = np.zeros((len(order), 8000), dtype=np.float32)
    for i, cid in enumerate(order):
        with wave.open(os.path.join(root, clips[cid]["path"])) as w:
            a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
        X[i, :min(8000, len(a))] = a[:8000]
    _, _, Z = stft(X, fs=8000, nperseg=256, noverlap=128, axis=1)
    F = np.log1p(np.abs(Z)).reshape(len(order), -1)
    F = F - F.mean(0, keepdims=True)
    F = F - F.mean(1, keepdims=True)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    C = F @ F.T
    np.fill_diagonal(C, 0.0)
    return [(order[int(i)], order[int(j)]) for i, j in np.argwhere(np.triu(C, 1) > threshold)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--manifest", required=True,
                    help="CSV portant au minimum les colonnes clip_id et fold")
    ap.add_argument("--near-dup", type=float, default=0.7)
    args = ap.parse_args()

    clips = load_source(args.data_root)

    # Seules DEUX colonnes sont lues. group_id est délibérément ignoré.
    fold = {}
    with open(args.manifest) as fh:
        for r in csv.DictReader(fh):
            fold[r["clip_id"]] = r["fold"]
    print(f"manifeste : {args.manifest}")
    print(f"  colonnes utilisées : clip_id, fold  (group_id ignoré volontairement)")

    missing = set(clips) - set(fold)
    extra = set(fold) - set(clips)
    violations: list[str] = []

    def check(name: str, got: int, coverage: int, detail: str = "") -> None:
        tag = "OK  " if got == 0 else "FAIL"
        pct = 100 * coverage / len(clips)
        print(f"{tag} {name}: {got} violation(s)   [couverture {coverage}/{len(clips)} = {pct:.0f}%]"
              + (f"\n       {detail}" if detail and got else ""))
        if got:
            violations.append(name)

    check("clips du disque absents du manifeste", len(missing), len(clips))
    check("clips du manifeste absents du disque", len(extra), len(clips))
    if missing or extra:
        sys.exit(1)

    def by_key(keyfn):
        """Regroupe les clips par clé et compte les clés étalées sur >1 fold."""
        buckets = collections.defaultdict(set)
        covered = set()
        for cid, c in clips.items():
            k = keyfn(c)
            if k is None:
                continue
            covered.add(cid)
            buckets[k].add(fold[cid])
        bad = {k: v for k, v in buckets.items() if len(v) > 1}
        return bad, len(covered)

    bad, cov = by_key(lambda c: ("md5", c["md5"]))
    check("I1 doublons exacts à cheval sur deux folds", len(bad), cov,
          f"exemples : {list(bad)[:2]}")

    order = sorted(clips)
    np_bad = [(a, b) for a, b in near_pairs(args.data_root, order, clips, args.near_dup)
              if fold[a] != fold[b]]
    check(f"I2 quasi-doublons (seuil {args.near_dup}) à cheval", len(np_bad), len(clips),
          f"exemples : {[(clips[a]['path'], clips[b]['path']) for a, b in np_bad[:2]]}")

    bad, cov = by_key(lambda c: ("body", c["cls"], c["body"]))
    check("I3 même session d'acquisition à cheval", len(bad), cov,
          f"exemples : {list(bad)[:2]}")

    bad, cov = by_key(lambda c: ("pf", c["pressure"], c["flow"])
                      if c["cls"] == "leak" and c["pressure"] != NA and c["flow"] != NA
                      else None)
    check("I4 condition leak (pression,débit) à cheval", len(bad), cov,
          f"exemples : {list(bad)[:3]}")

    bad, cov = by_key(lambda c: ("mr", c["material"], c["region"])
                      if c["cls"] == "no_leak" else None)
    check("I5 condition non-leak (matériau,région) à cheval", len(bad), cov,
          f"exemples : {list(bad)[:3]}")

    # Un groupe mélangeant deux classes rendrait tout équilibrage faux.
    lbl = {cid: ("leak" if c["cls"] == "leak" else "no_leak") for cid, c in clips.items()}
    print("\nrépartition recalculée depuis la source :")
    per = collections.defaultdict(collections.Counter)
    for cid in clips:
        per[fold[cid]][lbl[cid]] += 1
    for f in ("train", "val", "test"):
        tot = sum(per[f].values())
        print(f"  {f:<6} {tot:>4} clips  leak={per[f]['leak']:<4} non-leak={per[f]['no_leak']}")

    print("\n" + ("=== TOUS LES INVARIANTS TENUS ===" if not violations
                  else f"=== {len(violations)} INVARIANT(S) VIOLÉ(S) : {violations} ==="))
    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
