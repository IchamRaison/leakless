"""Point de passage unique vers le split gelé et vers l'audio.

**Tout** accès au manifeste et aux WAV passe par ici. C'est volontaire : un seul
endroit peut donc violer la règle d'isolation des métadonnées, et un test le
surveille (`tests/test_metadata_isolation.py`).

Règles appliquées ici, pas ailleurs :

  - `split_v2.csv` est la seule source des folds, des étiquettes et des clusters.
  - `split_v2_audit.csv` n'est ouvert QUE pour la colonne `path`. Pression, débit,
    device, matériau, région, md5 ne sont jamais renvoyés.
  - Le SHA256 du split est vérifié contre la valeur gelée. Un écart échoue
    bruyamment : un artefact de contrat ne se dégrade pas en silence.
"""

from __future__ import annotations

import csv
import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Le split gelé. Publié dans docs/SPLIT_V2_AUDIT.md, commit 3efa08f.
FROZEN_SPLIT_SHA256 = "7a8716a35284434292314c10da58663e9f848be60edf18db0f98ef9d63d17896"
FROZEN_SPLIT_NAME = "split_v2.csv"
CLIP_SAMPLES = 8000
SAMPLE_RATE = 8000

# Colonnes de split_v2_audit.csv qu'il est interdit de faire remonter.
FORBIDDEN_AUDIT_COLUMNS = frozenset({
    "device", "material", "region", "pressure_mpa", "flow_ms",
    "noise_category", "window", "rep", "md5",
})


class SplitIntegrityError(RuntimeError):
    """Le manifeste n'est pas celui qui est gelé, ou il est incohérent."""


@dataclass(frozen=True)
class Clip:
    clip_id: str
    label: int          # 1 = leak, 0 = non-leak
    label_3c: str
    group_id: str
    fold: str


@dataclass(frozen=True)
class Split:
    clips: tuple[Clip, ...]
    sha256: str
    manifest_path: Path
    _paths: dict[str, str]

    def fold(self, name: str) -> tuple[Clip, ...]:
        return tuple(c for c in self.clips if c.fold == name)

    def by_id(self) -> dict[str, Clip]:
        return {c.clip_id: c for c in self.clips}

    def path_of(self, clip_id: str) -> str:
        """Chemin relatif du WAV. Seule information tirée du fichier d'audit."""
        return self._paths[clip_id]


def sha256_of(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_split(manifest_dir: str | Path, *, expect_sha256: str | None = FROZEN_SPLIT_SHA256,
               name: str = FROZEN_SPLIT_NAME) -> Split:
    """Charge le split gelé et vérifie son identité.

    Args:
        manifest_dir: dossier contenant `split_v2.csv` et `split_v2_audit.csv`.
        expect_sha256: SHA attendu. `None` désactive la vérification — réservé aux
            fixtures de test, jamais à une évaluation publiée.
        name: nom du fichier de split. Toute valeur autre que `split_v2.csv` est
            refusée : `split_v1` est invalide et ne doit jamais être évalué.

    Raises:
        SplitIntegrityError: nom interdit, SHA divergent, doublon de clip_id, fold
            inconnu, ou cluster à cheval sur deux classes.
    """
    manifest_dir = Path(manifest_dir)
    if name != FROZEN_SPLIT_NAME:
        raise SplitIntegrityError(
            f"split refusé : « {name} ». Seul {FROZEN_SPLIT_NAME} est valide. "
            f"split_v1 est INVALIDE (voir docs/SPLIT_V2_AUDIT.md) et ne doit servir "
            f"à aucune évaluation.")

    split_path = manifest_dir / name
    if not split_path.exists():
        raise SplitIntegrityError(f"manifeste introuvable : {split_path}")

    digest = sha256_of(split_path)
    if expect_sha256 is not None and digest != expect_sha256:
        raise SplitIntegrityError(
            f"SHA256 du split divergent.\n  attendu : {expect_sha256}\n  obtenu  : {digest}\n"
            f"Le split gelé a changé, ou ce n'est pas le bon fichier. Refus d'évaluer.")

    with open(split_path) as fh:
        rows = list(csv.DictReader(fh))

    seen: set[str] = set()
    clips: list[Clip] = []
    for r in rows:
        cid = r["clip_id"]
        if cid in seen:
            raise SplitIntegrityError(f"clip_id en double dans le manifeste : {cid}")
        seen.add(cid)
        if r["fold"] not in ("train", "val", "test"):
            raise SplitIntegrityError(f"fold inconnu « {r['fold']} » pour {cid}")
        clips.append(Clip(clip_id=cid, label=1 if r["label"] == "leak" else 0,
                          label_3c=r["label_3c"], group_id=r["group_id"], fold=r["fold"]))

    by_group: dict[str, set[int]] = {}
    for c in clips:
        by_group.setdefault(c.group_id, set()).add(c.label)
    mixed = [g for g, v in by_group.items() if len(v) > 1]
    if mixed:
        raise SplitIntegrityError(f"{len(mixed)} cluster(s) mélangent deux classes : {mixed[:3]}")

    # La SEULE lecture du fichier d'audit, et la seule colonne qui en sort.
    audit_path = manifest_dir / name.replace(".csv", "_audit.csv")
    paths: dict[str, str] = {}
    if audit_path.exists():
        with open(audit_path) as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                paths[r["clip_id"]] = r["path"]

    return Split(clips=tuple(clips), sha256=digest, manifest_path=split_path, _paths=paths)


def read_wav_raw(data_root: str | Path, rel_path: str) -> np.ndarray:
    """Signal BRUT, sans normalisation. Longueur fixée à CLIP_SAMPLES."""
    with wave.open(str(Path(data_root) / rel_path)) as w:
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    out = np.zeros(CLIP_SAMPLES, dtype=np.float64)
    out[: min(CLIP_SAMPLES, len(a))] = a[:CLIP_SAMPLES]
    return out


def normalise_rms(x: np.ndarray) -> np.ndarray:
    """Normalisation d'amplitude : moyenne retirée, RMS ramené à 1.

    Appliquée identiquement à toutes les classes. C'est la définition unique de
    « audio normalisé » du projet ; C1, C2, C3 et le jeu TimeF l'utilisent tous.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x**2)))
    return x if rms == 0.0 else x / rms
