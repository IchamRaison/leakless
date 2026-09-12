#!/usr/bin/env python3
"""Vérificateur indépendant du manifeste `split_v1`.

**N'importe rien de `build_groups.py`.** Il repart des fichiers audio extraits et
des CSV publiés, et recalcule tout de zéro. Le but est qu'un tiers puisse
contrôler les chiffres de `docs/SPLIT_AUDIT.md` sans faire confiance au script
qui les a produits — deux implémentations indépendantes qui tombent d'accord
valent mieux qu'une qui se relit.

Usage :
  python3 scripts/eval/verify_manifest.py --data-root <dossier hors dépôt>

Sortie : une ligne OK/FAIL par contrôle, code de sortie 1 si un seul échoue.
Aucun entraînement, aucune écriture, aucun réseau.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import os
import sys
import wave

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Les chiffres publiés dans docs/SPLIT_AUDIT.md. Si un contrôle échoue, c'est
# soit le manifeste soit la documentation qui a dérivé — dans les deux cas il
# faut le savoir avant d'entraîner quoi que ce soit.
EXPECTED = {
    "clips": 1000,
    "folds": {"train": 600, "val": 200, "test": 200},
    "par_classe": {"train": (300, 300), "val": (100, 100), "test": (100, 100)},
    "groupes": {"train": 184, "val": 60, "test": 62},
    "groupes_non_leak": {"val": 9, "test": 11},
    "groupes_total": 306,
    "doublons_groupes": 36,
    "doublons_fichiers": 72,
    "sha256_split": "89f0624a4543fb1832d70938d5893e50ee131eecaa884e2f2d43b9fe05ca2a36",
}

FAILS: list[str] = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"{'OK  ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f"   (attendu {want})"))
    if not ok:
        FAILS.append(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="dossier HORS dépôt contenant les trois dossiers extraits")
    ap.add_argument("--manifests", default=os.path.join(REPO, "manifests"))
    args = ap.parse_args()

    split = list(csv.DictReader(open(os.path.join(args.manifests, "split_v1.csv"))))
    audit = {r["clip_id"]: r for r in
             csv.DictReader(open(os.path.join(args.manifests, "split_v1_audit.csv")))}

    # --- 0. le manifeste publié est-il bien celui qui est documenté ? ---
    with open(os.path.join(args.manifests, "split_v1.csv"), "rb") as fh:
        check("sha256 de split_v1.csv", hashlib.sha256(fh.read()).hexdigest(),
              EXPECTED["sha256_split"])

    # --- 1. le manifeste décrit-il les fichiers réellement présents ? ---
    on_disk = {}
    for folder in sorted(os.listdir(args.data_root)):
        d = os.path.join(args.data_root, folder)
        if not os.path.isdir(d):
            continue
        for n in os.listdir(d):
            if n.lower().endswith(".wav"):
                on_disk[f"{folder}/{n}"] = os.path.join(d, n)
    check("fichiers WAV sur disque", len(on_disk), EXPECTED["clips"])
    check("lignes dans split_v1.csv", len(split), EXPECTED["clips"])
    paths = {audit[r["clip_id"]]["path"] for r in split}
    check("chemins du manifeste absents du disque", len(paths - set(on_disk)), 0)
    check("fichiers du disque absents du manifeste", len(set(on_disk) - paths), 0)

    # --- 2. md5 et clip_id recalculés depuis les octets ---
    md5_of, bad_md5, bad_id = {}, 0, 0
    for r in split:
        a = audit[r["clip_id"]]
        with open(on_disk[a["path"]], "rb") as fh:
            h = hashlib.md5(fh.read()).hexdigest()
        md5_of[r["clip_id"]] = h
        bad_md5 += h != a["md5"]
        bad_id += "c" + hashlib.sha1(a["path"].encode()).hexdigest()[:12] != r["clip_id"]
    check("md5 divergents", bad_md5, 0)
    check("clip_id non reproductibles", bad_id, 0)

    # --- 3. l'étiquette correspond-elle au dossier source ? ---
    check("étiquettes incohérentes avec le dossier source",
          sum(1 for r in split
              if (audit[r["clip_id"]]["path"].split("/")[0] == "leak acoustic data")
              != (r["label"] == "leak")), 0)

    # --- 4. format audio ---
    props = collections.Counter()
    for r in split:
        with wave.open(on_disk[audit[r["clip_id"]]["path"]]) as w:
            props[(w.getnchannels(), w.getframerate(), w.getsampwidth(), w.getnframes())] += 1
    check("formats audio distincts", len(props), 1)
    check("format unique", list(props)[0], (1, 8000, 2, 8000))

    # --- 5. les trois contrôles d'overlap, recalculés ---
    g_folds, m_folds, c_folds = (collections.defaultdict(set) for _ in range(3))
    for r in split:
        a = audit[r["clip_id"]]
        g_folds[r["group_id"]].add(r["fold"])
        m_folds[md5_of[r["clip_id"]]].add(r["fold"])
        if r["label_3c"] == "leak" and "NA" not in (
                a["material"], a["region"], a["pressure_mpa"], a["flow_ms"]):
            c_folds[(a["material"], a["region"],
                     a["pressure_mpa"], a["flow_ms"])].add(r["fold"])
    check("group_overlap", sum(1 for v in g_folds.values() if len(v) > 1), 0)
    check("duplicate_overlap", sum(1 for v in m_folds.values() if len(v) > 1), 0)
    check("condition_overlap", sum(1 for v in c_folds.values() if len(v) > 1), 0)

    by_group = collections.defaultdict(set)
    for r in split:
        by_group[r["group_id"]].add(r["label"])
    check("groupes mélangeant deux classes",
          sum(1 for v in by_group.values() if len(v) > 1), 0)

    # --- 6. les comptes publiés dans docs/SPLIT_AUDIT.md ---
    for f, n in EXPECTED["folds"].items():
        check(f"clips {f}", sum(1 for r in split if r["fold"] == f), n)
        leak, nleak = EXPECTED["par_classe"][f]
        check(f"  {f}/leak",
              sum(1 for r in split if r["fold"] == f and r["label"] == "leak"), leak)
        check(f"  {f}/no_leak",
              sum(1 for r in split if r["fold"] == f and r["label"] == "no_leak"), nleak)
        check(f"  groupes {f}",
              len({r["group_id"] for r in split if r["fold"] == f}),
              EXPECTED["groupes"][f])
    for f, n in EXPECTED["groupes_non_leak"].items():
        check(f"groupes non-leak {f} (le N honnête)",
              len({r["group_id"] for r in split
                   if r["fold"] == f and r["label"] == "no_leak"}), n)
    check("groupes totaux", len(g_folds), EXPECTED["groupes_total"])

    dup = [n for n in collections.Counter(md5_of.values()).values() if n > 1]
    check("groupes de fichiers octet-identiques", len(dup), EXPECTED["doublons_groupes"])
    check("fichiers impliqués dans un doublon", sum(dup), EXPECTED["doublons_fichiers"])

    # --- 7. fuite d'étiquette dans le contrat de split ---
    check("colonnes révélant l'étiquette dans split_v1.csv",
          sorted(set(split[0]) & {"pressure_mpa", "flow_ms", "path", "material", "region"}),
          [])
    pres = collections.Counter(
        (r["label"], audit[r["clip_id"]]["pressure_mpa"] != "NA") for r in split)
    print(f"\ncontrôle L1 — pression renseignée : leak={pres[('leak', True)]}/500, "
          f"non-leak={pres[('no_leak', True)]}/500 "
          f"(asymétrie attendue : c'est la fuite d'étiquette documentée)")

    print("\n" + ("=== TOUT CONCORDE ===" if not FAILS
                  else f"=== {len(FAILS)} DIVERGENCE(S) : {FAILS} ==="))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
