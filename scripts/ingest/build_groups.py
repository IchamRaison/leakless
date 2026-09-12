#!/usr/bin/env python3
"""Reconstruit des clés de groupe pour le dataset acoustique Zenodo 18631450.

Objectif : mesurer si un split groupé sans leakage est possible. Ce script
NE FAIT AUCUN SPLIT. Il parse les noms de fichiers, propose une clé de groupe
fondée uniquement sur des métadonnées observables, fusionne les groupes reliés
par des doublons audio, et rapporte les ambiguïtés au lieu de les résoudre
silencieusement.

Règle de nommage annoncée par la source (Zenodo, champ description) :
  leak / no leak : Matériau-Région-Pression MPa-Débit ms-Device
  environmental noise : Catégorie-Région-Device
  information manquante = "NA"

Structure réellement observée (voir docs/DATASET_AUDIT.md) : le nom porte en
plus un suffixe de fenêtre temporelle `w0-w1` (seconde w0 à w1 d'un
enregistrement parent de ~5 s) et un index de répétition optionnel `_N` :

  <body>[-_]<w0>-<w1>[_<N>].wav

Usage :
  python3 scripts/ingest/build_groups.py --data-root /chemin/hors/repo \\
      [--json rapport.json] [--near-dup 0.8]

Le dataset vit HORS du dépôt (.gitignore bloque data/, *.rar, *.wav).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import statistics
import sys
import wave
from dataclasses import dataclass, asdict, field

# Les trois dossiers tels qu'extraits des .rar, et la classe associée.
FOLDERS = {
    "leak": "leak acoustic data",
    "no_leak": "no leak acoustic data",
    "noise": "environmental noise",
}

# <body> non gourmand, puis fenêtre w0-w1, puis index de répétition optionnel.
NAME_RE = re.compile(
    r"^(?P<body>.+?)[-_](?P<w0>\d+)-(?P<w1>\d+)(?:_(?P<rep>\d+))?\.wav$"
)

NA = "NA"


@dataclass
class Clip:
    cls: str
    filename: str
    body: str
    material: str
    region: str
    pressure: str
    flow: str
    device: str
    noise_category: str
    window: int
    rep: str | None
    md5: str
    parsed: bool = True
    group: str = ""
    clip_id: str = ""
    ambiguities: list[str] = field(default_factory=list)


def parse_clip(cls: str, filename: str, md5: str) -> Clip:
    """Décompose un nom de fichier. Ne devine rien : ce qui manque reste NA."""
    m = NAME_RE.match(filename)
    if not m:
        return Clip(cls, filename, "", NA, NA, NA, NA, NA, NA, -1, None, md5,
                    parsed=False, ambiguities=["nom non parsable"])

    body = m.group("body")
    fields = [f.strip() for f in body.split("-")]

    material = region = pressure = flow = device = noise_category = NA
    if cls == "noise":
        # Catégorie-Région-Device
        if len(fields) == 3:
            noise_category, region, device = fields
        else:
            return Clip(cls, filename, body, NA, NA, NA, NA, NA, NA,
                        int(m.group("w0")), m.group("rep"), md5, parsed=False,
                        ambiguities=[f"{len(fields)} champs au lieu de 3"])
    else:
        # Matériau-Région-Pression-Débit-Device
        if len(fields) == 5:
            material, region, pressure, flow, device = fields
        else:
            return Clip(cls, filename, body, NA, NA, NA, NA, NA, NA,
                        int(m.group("w0")), m.group("rep"), md5, parsed=False,
                        ambiguities=[f"{len(fields)} champs au lieu de 5"])

    return Clip(cls, filename, body, material, region, pressure, flow, device,
                noise_category, int(m.group("w0")), m.group("rep"), md5)


def group_key(c: Clip) -> str:
    """Clé de groupe, uniquement à partir de métadonnées observables.

    Choix délibérément conservateur : la clé SUR-FUSIONNE plutôt que de
    sous-fusionner. Sur-fusionner coûte des groupes ; sous-fusionner coûte la
    validité du score. La fenêtre temporelle w0-w1 et l'index de répétition _N
    sont volontairement EXCLUS de la clé : ce sont des extraits du même
    enregistrement parent, ils doivent rester du même côté du split.

    Justification mesurée (docs/DATASET_AUDIT.md §3) : à l'intérieur d'un même
    `body`, la corrélation spectrale moyenne est 8 à 13 fois la corrélation
    entre `body` différents.
    """
    if not c.parsed:
        return f"{c.cls}|NON_PARSE|{c.filename}"
    if c.cls == "noise":
        # `dog_1` et `dog_2` sont des enregistrements distincts : on garde
        # l'index de catégorie, qui fait partie du nom de la catégorie.
        return f"{c.cls}|{c.noise_category}|{c.region}|{c.device}"
    # Pression et débit ne sont renseignés que pour la classe leak (voir §4) :
    # les inclure rendrait la clé incomparable d'une classe à l'autre. Ils
    # restent dans la clé car ils séparent des sessions réelles côté leak.
    return f"{c.cls}|{c.material}|{c.region}|{c.pressure}|{c.flow}|{c.device}"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_clips(root: str) -> list[Clip]:
    clips: list[Clip] = []
    for cls, folder in FOLDERS.items():
        path = os.path.join(root, folder)
        if not os.path.isdir(path):
            sys.exit(f"dossier introuvable : {path}")
        for name in sorted(os.listdir(path)):
            if not name.lower().endswith(".wav"):
                continue
            with open(os.path.join(path, name), "rb") as fh:
                md5 = hashlib.md5(fh.read()).hexdigest()
            c = parse_clip(cls, name, md5)
            # Identifiant stable, indépendant de l'ordre de lecture et de toute
            # exécution : hash du chemin relatif dans l'archive.
            c.clip_id = "c" + hashlib.sha1(
                f"{folder}/{name}".encode()).hexdigest()[:12]
            clips.append(c)
    return clips


def wav_properties(root: str, clips: list[Clip]) -> dict:
    props = collections.Counter()
    for c in clips:
        p = os.path.join(root, FOLDERS[c.cls], c.filename)
        with wave.open(p) as w:
            props[(w.getnchannels(), w.getframerate(), w.getsampwidth(),
                   round(w.getnframes() / w.getframerate(), 3))] += 1
    return {f"{ch}ch/{sr}Hz/{sw*8}bit/{dur}s": n for (ch, sr, sw, dur), n in props.items()}


def spectral_correlation_pairs(root: str, clips: list[Clip], threshold: float):
    """Paires de clips au-dessus d'un seuil de corrélation spectrale.

    Corrélation de log-spectrogrammes centrés sur la moyenne globale : sans ce
    centrage, tous les clips se ressemblent (médiane cosinus 0.96) et la mesure
    ne discrimine rien.
    """
    try:
        import numpy as np
        from scipy.signal import stft
    except ImportError:
        print("numpy/scipy absents : analyse de quasi-doublons ignorée", file=sys.stderr)
        return [], None

    X = np.zeros((len(clips), 8000), dtype=np.float32)
    for i, c in enumerate(clips):
        with wave.open(os.path.join(root, FOLDERS[c.cls], c.filename)) as w:
            a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
        X[i, :min(8000, len(a))] = a[:8000]

    _, _, Z = stft(X, fs=8000, nperseg=256, noverlap=128, axis=1)
    F = np.log1p(np.abs(Z)).reshape(len(clips), -1)
    F = F - F.mean(0, keepdims=True)
    F = F - F.mean(1, keepdims=True)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    C = F @ F.T
    np.fill_diagonal(C, 0.0)

    pairs = [(int(i), int(j), float(C[i, j]))
             for i, j in np.argwhere(np.triu(C, 1) > threshold)]
    return pairs, C


def signal_descriptors(root: str, clips: list[Clip]) -> dict:
    """Descripteurs physiques simples, par clip puis agrégés par groupe.

    ⚠️ CE N'EST PAS UNE BASELINE. Rien n'est entraîné, rien n'est ajusté, aucun
    split n'est utilisé. On calcule des statistiques descriptives sur la
    totalité du dataset pour répondre à une seule question : les étiquettes
    correspondent-elles à quelque chose d'audible, ou sont-elles vides ?

    Les AUC rapportées sont calculées sur TOUT le dataset et agrégées par
    groupe pour éviter la pseudo-réplication. Elles ne peuvent PAS être citées
    comme une performance : aucune donnée n'est tenue à l'écart.
    """
    try:
        import numpy as np
    except ImportError:
        return {"erreur": "numpy absent"}

    rows = []
    for c in clips:
        with wave.open(os.path.join(root, FOLDERS[c.cls], c.filename)) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
        x = x - x.mean()
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        freqs = np.fft.rfftfreq(len(x), 1 / 8000)
        total = spec.sum() + 1e-12
        rows.append({
            "cls": c.cls,
            "group": c.group,
            "rms_dbfs": 20 * np.log10(np.sqrt((x ** 2).mean()) / 32768 + 1e-12),
            "centroide_hz": float((freqs * spec).sum() / total),
            "ratio_hf_1k": float(spec[freqs >= 1000].sum() / total),
            "taux_passages_zero": float(np.mean(np.abs(np.diff(np.sign(x))) > 0)),
        })

    feats = ["rms_dbfs", "centroide_hz", "ratio_hf_1k", "taux_passages_zero"]
    out: dict = {"avertissement": "descriptif, calculé sur tout le dataset — PAS une performance"}

    for f in feats:
        out.setdefault("par_classe", {})[f] = {}
        for cls in list(FOLDERS) + ["no_leak_binaire"]:
            if cls == "no_leak_binaire":
                v = np.array([r[f] for r in rows if r["cls"] != "leak"])
            else:
                v = np.array([r[f] for r in rows if r["cls"] == cls])
            out["par_classe"][f][cls] = {
                "mediane": round(float(np.median(v)), 3),
                "q1": round(float(np.percentile(v, 25)), 3),
                "q3": round(float(np.percentile(v, 75)), 3),
            }

    # AUC au niveau du GROUPE (médiane du groupe), tâche binaire leak vs reste.
    by_group: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    out["auc_descriptive_par_groupe"] = {}
    for f in feats:
        pos = [np.median([r[f] for r in v]) for v in by_group.values()
               if v[0]["cls"] == "leak"]
        neg = [np.median([r[f] for r in v]) for v in by_group.values()
               if v[0]["cls"] != "leak"]
        # AUC = P(pos > neg), estimée par comptage direct des paires.
        pos_a, neg_a = np.array(pos), np.array(neg)
        wins = (pos_a[:, None] > neg_a[None, :]).sum() + 0.5 * (pos_a[:, None] == neg_a[None, :]).sum()
        out["auc_descriptive_par_groupe"][f] = round(float(wins / (len(pos) * len(neg))), 3)
    out["auc_descriptive_par_groupe"]["n_groupes_leak"] = sum(
        1 for v in by_group.values() if v[0]["cls"] == "leak")
    out["auc_descriptive_par_groupe"]["n_groupes_non_leak"] = sum(
        1 for v in by_group.values() if v[0]["cls"] != "leak")
    return out


SPLIT_TARGET = {"train": 0.60, "val": 0.20, "test": 0.20}


def assign_folds(clips: list[Clip], seed: int) -> dict[str, str]:
    """Assigne chaque GROUPE entier à un fold. Déterministe, sans recherche.

    Procédure, fixée avant d'avoir vu le moindre score :
      1. mélange des groupes avec `seed` (documenté, unique) ;
      2. tri stable par taille décroissante — les gros groupes, les plus durs à
         placer, passent en premier ;
      3. chaque groupe va au fold dont le déficit relatif est le plus grand
         POUR SA CLASSE (tâche binaire leak / non-leak).

    Aucune recherche sur plusieurs seeds : choisir le « meilleur » split parmi
    N tirages reviendrait à optimiser le split, c'est-à-dire à choisir ses
    données d'évaluation. On prend ce que le seed donne.
    """
    # Clé par clip_id, jamais par nom de fichier : deux dossiers de classes
    # pourraient contenir le même nom, et la collision serait silencieuse.
    binary = {c.clip_id: ("leak" if c.cls == "leak" else "no_leak") for c in clips}
    sizes: dict[str, int] = collections.Counter(c.group for c in clips)
    cls_of: dict[str, str] = {}
    for c in clips:
        seen = cls_of.setdefault(c.group, binary[c.clip_id])
        if seen != binary[c.clip_id]:
            # Un groupe à cheval sur deux classes rendrait l'équilibrage faux
            # sans rien signaler. On refuse plutôt que de deviner.
            raise SystemExit(
                f"groupe {c.group} contient deux classes ({seen} et "
                f"{binary[c.clip_id]}) — split impossible, corriger la clé de groupe")

    groups = sorted(sizes)                      # ordre de départ déterministe
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda g: -sizes[g])        # tri stable : départage par le mélange

    placed: dict[str, collections.Counter] = {f: collections.Counter() for f in SPLIT_TARGET}
    assignment: dict[str, str] = {}
    for g in groups:
        cls = cls_of[g]
        done = sum(placed[f][cls] for f in placed) + sizes[g]
        fold = max(SPLIT_TARGET, key=lambda f: SPLIT_TARGET[f] - placed[f][cls] / done)
        placed[fold][cls] += sizes[g]
        assignment[g] = fold
    return assignment


def write_manifest(out_dir: str, clips: list[Clip], seed: int,
                   near_dup: float, merged_cross_device: bool,
                   near_dup_available: bool) -> dict:
    """Écrit le manifeste versionné. Aucun WAV, aucune donnée audio.

    Deux fichiers, séparés volontairement :

    `split_v1.csv` — le contrat de split. clip_id, label, group_id, device, fold.
        **Ne contient aucune métadonnée révélant l'étiquette.** C'est le seul
        fichier que le code d'entraînement a besoin de lire.

    `split_v1_audit.csv` — colonnes d'audit uniquement : chemin, matériau,
        région, pression, débit, fenêtre, répétition, md5. Pression et débit ne
        sont renseignés que pour la classe leak (risque L1) : **aucune colonne
        de ce fichier ne doit atteindre une entrée de modèle.** Il sert à
        rejouer l'audit et à résoudre clip_id → fichier pour lire la forme
        d'onde.
    """
    import csv

    if not near_dup_available:
        # Sans numpy/scipy, la fusion des quasi-doublons est silencieusement
        # sautée et le groupage change. Un manifeste est un contrat : on refuse
        # d'en écrire un dont on ne peut pas garantir la reproductibilité.
        raise SystemExit(
            "numpy/scipy indisponibles : le groupage serait différent (fusion des "
            "quasi-doublons sautée). Refus d'écrire un manifeste non reproductible.")

    os.makedirs(out_dir, exist_ok=True)
    folds = assign_folds(clips, seed)
    rows = sorted(clips, key=lambda c: c.clip_id)

    split_path = os.path.join(out_dir, "split_v1.csv")
    with open(split_path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["clip_id", "label", "label_3c", "group_id", "device", "fold"])
        for c in rows:
            w.writerow([c.clip_id,
                        "leak" if c.cls == "leak" else "no_leak",
                        c.cls, c.group, c.device, folds[c.group]])

    audit_path = os.path.join(out_dir, "split_v1_audit.csv")
    with open(audit_path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["clip_id", "path", "material", "region", "pressure_mpa",
                    "flow_ms", "noise_category", "window", "rep", "md5",
                    "group_id", "fold"])
        for c in rows:
            w.writerow([c.clip_id, f"{FOLDERS[c.cls]}/{c.filename}", c.material,
                        c.region, c.pressure, c.flow, c.noise_category, c.window,
                        c.rep if c.rep is not None else "", c.md5, c.group,
                        folds[c.group]])

    def sha256(path: str) -> str:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    meta = {
        "version": "split_v1",
        "date": "2026-09-12",
        "source": "zenodo.org/records/18631450 (CC BY 4.0)",
        "seed": seed,
        # Tout paramètre qui change le groupage est enregistré ici : sans cela,
        # deux manifestes différents auraient une provenance identique.
        "near_dup_threshold": near_dup,
        "merge_cross_device": merged_cross_device,
        "script": "scripts/ingest/build_groups.py",
        "procedure": "groupes mélangés avec le seed, triés par taille décroissante, "
                     "chacun placé dans le fold au plus grand déficit relatif pour sa "
                     "classe binaire. Un seul tirage, aucune recherche de seed.",
        "cible": SPLIT_TARGET,
        "n_clips": len(clips),
        "n_groupes": len(set(folds)),
        "sha256": {"split_v1.csv": sha256(split_path),
                   "split_v1_audit.csv": sha256(audit_path)},
        "regle": "split_v1.csv ne contient aucune métadonnée révélant l'étiquette. "
                 "Aucune colonne de split_v1_audit.csv n'entre dans une entrée de modèle.",
    }
    with open(os.path.join(out_dir, "split_v1.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return {"folds": folds, "meta": meta,
            "split_path": split_path, "audit_path": audit_path}


def verify_split(clips: list[Clip], folds: dict[str, str]) -> dict:
    """Contrôles automatiques. Chacun DOIT valoir 0."""
    # Compter les non-affectés AVANT de déréférencer folds : sinon un clip sans
    # fold lève KeyError ici et le contrôle ne peut jamais rapporter autre que 0.
    orphans = [c for c in clips if c.group not in folds]

    by_group = collections.defaultdict(set)
    by_md5 = collections.defaultdict(set)
    by_condition = collections.defaultdict(set)
    for c in clips:
        f = folds.get(c.group)
        if f is None:
            continue
        by_group[c.group].add(f)
        by_md5[c.md5].add(f)
        if c.cls == "leak" and NA not in (c.material, c.region, c.pressure, c.flow):
            by_condition[(c.material, c.region, c.pressure, c.flow)].add(f)

    return {
        "group_overlap": sum(1 for v in by_group.values() if len(v) > 1),
        "duplicate_overlap": sum(1 for v in by_md5.values() if len(v) > 1),
        "condition_overlap": sum(1 for v in by_condition.values() if len(v) > 1),
        "clips_sans_fold": len(orphans),
    }


def describe(sizes: list[int]) -> dict:
    return {
        "n_groupes": len(sizes),
        "min": min(sizes),
        "mediane": statistics.median(sizes),
        "max": max(sizes),
        "moyenne": round(statistics.mean(sizes), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="dossier HORS dépôt contenant les trois dossiers extraits")
    ap.add_argument("--json", help="écrit le rapport machine ici (hors dépôt)")
    ap.add_argument("--near-dup", type=float, default=0.8,
                    help="seuil de corrélation spectrale pour les quasi-doublons")
    ap.add_argument("--no-merge-cross-device", action="store_true",
                    help="ne pas fusionner les conditions vues par les deux devices "
                         "(laisse le même événement physique à cheval sur deux groupes)")
    ap.add_argument("--descriptors", action="store_true",
                    help="statistiques descriptives de signal par classe — PAS une baseline")
    ap.add_argument("--write-manifest", metavar="DIR",
                    help="fige le manifeste versionné (CSV, aucun WAV) dans ce dossier")
    ap.add_argument("--seed", type=int, default=20260912,
                    help="seed du split groupé (défaut : 20260912, documenté dans le manifeste)")
    args = ap.parse_args()

    clips = load_clips(args.data_root)
    report: dict = {"source": "zenodo.org/records/18631450", "n_clips": len(clips)}

    # 1. Inventaire ---------------------------------------------------------
    per_class = collections.Counter(c.cls for c in clips)
    report["clips_par_classe"] = dict(per_class)
    report["proprietes_wav"] = wav_properties(args.data_root, clips)
    report["non_parsables"] = [c.filename for c in clips if not c.parsed]

    # 2. Clé de groupe métadonnées ------------------------------------------
    for c in clips:
        c.group = group_key(c)
    meta_groups = collections.Counter(c.group for c in clips)
    report["cle_metadonnees"] = describe(list(meta_groups.values()))
    report["cle_metadonnees"]["par_classe"] = {
        cls: len({c.group for c in clips if c.cls == cls}) for cls in FOLDERS
    }

    # 3. Doublons octet-identiques ------------------------------------------
    by_md5 = collections.defaultdict(list)
    for c in clips:
        by_md5[c.md5].append(c)
    exact = {k: v for k, v in by_md5.items() if len(v) > 1}
    cross_group = [v for v in exact.values() if len({c.group for c in v}) > 1]
    cross_class = [v for v in exact.values() if len({c.cls for c in v}) > 1]
    report["doublons_exacts"] = {
        "groupes": len(exact),
        "fichiers": sum(len(v) for v in exact.values()),
        "reliant_deux_groupes_metadonnees": len(cross_group),
        "reliant_deux_classes": len(cross_class),
        "exemples_inter_groupes": [[c.filename for c in v] for v in cross_group],
    }

    # 4. Quasi-doublons ------------------------------------------------------
    pairs, corr = spectral_correlation_pairs(args.data_root, clips, args.near_dup)
    near_dup_available = corr is not None
    near_cross_group = [(clips[i].filename, clips[j].filename, round(s, 3))
                        for i, j, s in pairs if clips[i].group != clips[j].group]
    report["quasi_doublons"] = {
        "seuil": args.near_dup,
        "paires": len(pairs),
        "paires_inter_groupes": len(near_cross_group),
        "paires_inter_classes": sum(1 for i, j, _ in pairs if clips[i].cls != clips[j].cls),
        "dont_octet_identiques": sum(1 for i, j, _ in pairs if clips[i].md5 == clips[j].md5),
    }

    # 5. Fusion audio : un groupe final ne peut pas être coupé par un doublon -
    uf = UnionFind()
    for c in clips:
        uf.find(c.group)
    for v in exact.values():
        for other in v[1:]:
            uf.union(v[0].group, other.group)
    for i, j, _ in pairs:
        uf.union(clips[i].group, clips[j].group)

    # 5b. Même condition physique captée par deux devices = un seul événement.
    # Le device reste dans la clé (il sépare des sessions réelles et préserve la
    # granularité), mais les conditions vues par les DEUX instruments sont
    # fusionnées : sinon le même événement physique se retrouve de part et
    # d'autre du split. Coût mesuré : 16 conditions, 67 clips (6,7 % du dataset).
    by_condition = collections.defaultdict(list)
    for c in clips:
        if c.cls == "leak" and NA not in (c.material, c.region, c.pressure, c.flow):
            by_condition[(c.material, c.region, c.pressure, c.flow)].append(c)
    cross_device = [m for m in by_condition.values() if len({x.device for x in m}) > 1]

    if not args.no_merge_cross_device:
        for members in cross_device:
            for other in members[1:]:
                uf.union(members[0].group, other.group)

    for c in clips:
        c.group = uf.find(c.group)
    report["fusion_bi_device"] = {
        "active": not args.no_merge_cross_device,
        "conditions_fusionnees": len(cross_device),
        "clips_concernes": sum(len(m) for m in cross_device),
        "clips_a_exclure_du_holdout_device": sorted(
            x.filename for m in cross_device for x in m if x.device == "hydrophone"),
    }
    # Identifiant de groupe stable : hash de la liste triée de ses membres.
    # Indépendant de l'ordre de parcours, donc comparable d'une exécution à
    # l'autre et citable dans un rapport.
    members = collections.defaultdict(list)
    for c in clips:
        members[c.group].append(c.filename)
    stable = {g: "g" + hashlib.sha1("\n".join(sorted(v)).encode()).hexdigest()[:10]
              for g, v in members.items()}
    for c in clips:
        c.group = stable[c.group]

    final = collections.Counter(c.group for c in clips)
    report["cle_finale"] = describe(list(final.values()))
    report["cle_finale"]["par_classe"] = {
        cls: len({c.group for c in clips if c.cls == cls}) for cls in FOLDERS
    }
    mixed = [g for g in final if len({c.cls for c in clips if c.group == g}) > 1]
    report["cle_finale"]["groupes_multi_classes"] = len(mixed)

    # Vue binaire : la source décrit les clips de bruit comme portant des
    # « detailed no-leak labels ». leak vs (no leak + noise).
    def binary(c: Clip) -> str:
        return "leak" if c.cls == "leak" else "no_leak"

    report["tache_binaire"] = {
        "clips": dict(collections.Counter(binary(c) for c in clips)),
        "groupes": {k: len({c.group for c in clips if binary(c) == k})
                    for k in ("leak", "no_leak")},
        "groupes_multi_classes_binaires": sum(
            1 for g in final if len({binary(c) for c in clips if c.group == g}) > 1),
    }

    # 6. Ambiguïtés ----------------------------------------------------------
    # Un clip est « non ambigu » si son nom parse, si sa clé n'est pas reliée à
    # une autre clé par un doublon audio, et si la clé n'est pas dégénérée
    # (matériau ET région inconnus, donc indiscernable d'une autre session).
    linked = {c.filename for v in cross_group for c in v}
    linked |= {clips[i].filename for i, j, _ in pairs if clips[i].group != clips[j].group}
    linked |= {clips[j].filename for i, j, _ in pairs if clips[i].group != clips[j].group}
    degenerate = {c.filename for c in clips
                  if c.cls != "noise" and c.material == NA and c.region == NA}
    for c in clips:
        if not c.parsed:
            c.ambiguities.append("nom non parsable")
        if c.filename in linked:
            c.ambiguities.append("relié à un autre groupe par un doublon audio")
        if c.filename in degenerate:
            c.ambiguities.append("clé dégénérée : matériau et région inconnus")
    ambiguous = [c for c in clips if c.ambiguities]
    report["ambiguite"] = {
        "clips_ambigus": len(ambiguous),
        "clips_non_ambigus": len(clips) - len(ambiguous),
        "pct_non_ambigu": round(100 * (len(clips) - len(ambiguous)) / len(clips), 2),
        "detail": dict(collections.Counter(a for c in ambiguous for a in c.ambiguities)),
    }

    # 7. Même condition vue par plusieurs fichiers / plusieurs devices --------
    cond = collections.defaultdict(set)
    cond_files = collections.Counter()
    for c in clips:
        if c.cls == "leak" and NA not in (c.material, c.region, c.pressure, c.flow):
            key = (c.material, c.region, c.pressure, c.flow)
            cond[key].add(c.device)
            cond_files[key] += 1
    report["conditions_leak_completes"] = {
        "conditions": len(cond),
        "vues_par_les_deux_devices": sum(1 for v in cond.values() if len(v) > 1),
        "conditions_multi_fichiers": sum(1 for v in cond_files.values() if v > 1),
        "max_fichiers_par_condition": max(cond_files.values()) if cond_files else 0,
    }
    report["device_par_classe"] = {
        cls: dict(collections.Counter(c.device for c in clips if c.cls == cls))
        for cls in FOLDERS
    }
    report["champs_renseignes_par_classe"] = {
        cls: {
            "pression": sum(1 for c in clips if c.cls == cls and c.pressure != NA),
            "debit": sum(1 for c in clips if c.cls == cls and c.flow != NA),
            "region": sum(1 for c in clips if c.cls == cls and c.region != NA),
            "materiau": sum(1 for c in clips if c.cls == cls and c.material != NA),
        }
        for cls in FOLDERS
    }

    # 8. Hold-out device : évaluation secondaire, pas le split principal ------
    excl = set(report["fusion_bi_device"]["clips_a_exclure_du_holdout_device"])
    report["holdout_device"] = {
        "commentaire": "évaluation secondaire de généralisation inter-instrument, "
                       "distincte du split groupé principal",
        "hydrophone_en_test": dict(collections.Counter(
            c.cls for c in clips if c.device == "hydrophone")),
        "apres_exclusion_des_conditions_bi_device": dict(collections.Counter(
            c.cls for c in clips
            if c.device == "hydrophone" and c.filename not in excl)),
        "classe_noise_disponible_en_hydrophone": sum(
            1 for c in clips if c.cls == "noise" and c.device == "hydrophone"),
    }

    # 9. Manifeste versionné + contrôles automatiques du split ---------------
    if args.write_manifest:
        written = write_manifest(
            args.write_manifest, clips, args.seed,
            near_dup=args.near_dup,
            merged_cross_device=not args.no_merge_cross_device,
            near_dup_available=near_dup_available)
        folds = written["folds"]
        report["manifeste"] = written["meta"]
        report["verification_split"] = verify_split(clips, folds)

        bin_of = {c.filename: ("leak" if c.cls == "leak" else "no_leak") for c in clips}
        per_fold: dict = {}
        for f in SPLIT_TARGET:
            members = [c for c in clips if folds[c.group] == f]
            gs = collections.Counter(c.group for c in members)
            per_fold[f] = {
                "clips": len(members),
                "clips_par_classe_binaire": dict(collections.Counter(
                    bin_of[c.filename] for c in members)),
                "clips_par_classe_3": dict(collections.Counter(c.cls for c in members)),
                "groupes": len(gs),
                "groupes_par_classe_binaire": {
                    k: len({c.group for c in members if bin_of[c.filename] == k})
                    for k in ("leak", "no_leak")},
                "taille_groupes": describe(list(gs.values())),
                "plus_gros_groupe_pct_du_fold": round(
                    100 * max(gs.values()) / len(members), 1),
            }
        report["folds"] = per_fold

    if args.descriptors:
        report["descripteurs_signal"] = signal_descriptors(args.data_root, clips)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rapport": report,
                       "clips": [asdict(c) for c in clips]}, fh,
                      indent=2, ensure_ascii=False)
        print(f"\nmanifeste écrit : {args.json}", file=sys.stderr)

    # Aucun split n'est produit ici. Voir docs/EVAL_PROTOCOL.md §2.


if __name__ == "__main__":
    main()
