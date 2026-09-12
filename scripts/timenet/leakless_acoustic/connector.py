"""TimeNet connector : clips acoustiques -> TimeF, groupés par grappe de dépendance.

Le split gelé `manifests/split_v2.csv` est la seule source de vérité. Ce connecteur
ne le recalcule pas et ne le modifie pas : il le lit.

Ce qui entre dans TimeF :

  record_id    = clip_id
  subject_ids  = (group_id,)   la grappe de dépendance, unité de regroupement TimeF
  time series  = forme d'onde **normalisée en amplitude**
  task         = ClassificationTask(target = "leak" | "no_leak")

Ce qui n'y entre PAS, et ne doit jamais y entrer :

  pression, débit, device, matériau, région, chemin du fichier, nom du fichier.

  Pression et débit ne sont renseignés que pour la classe leak (469/500 contre 0/500) :
  les exposer donnerait 93,8 % de rappel sans écouter un son. `split_v2_audit.csv` ne
  sert qu'à résoudre clip_id -> chemin du WAV, rien d'autre.

**L'amplitude est normalisée par clip, ici, par construction.** Les clips leak sont
~11 dB plus forts que les no-leak, et c'est un artefact de protocole d'acquisition
(docs/EVAL_PROTOCOL.md §7bis-A). Un jeu TimeF normalisé ne peut donc pas servir au
contrôle RMS : celui-ci lit le signal brut, séparément (`scripts/eval/rms_control.py`).

`subject_ids` porte le group_id : c'est une clé de regroupement pour l'évaluation,
jamais une entrée de modèle.
"""

from __future__ import annotations

import csv
import os
import wave
from pathlib import Path

import numpy as np

from timenet.connectors import BaseConnector
from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import RegularAxis
from timenet.types import ClassificationTask, DataSource, TimeSeriesSpec, ureg

SAMPLING_RATE_HZ = 8000  # entier : RegularAxis.from_rate_hz refuse un flottant
CLIP_SAMPLES = 8000

# Chemins résolus depuis des variables d'environnement : le dataset vit hors du dépôt
# et le manifeste gelé est dans le dépôt. Aucun chemin en dur.
ENV_DATA_ROOT = "LEAKLESS_DATA_ROOT"
ENV_MANIFEST_DIR = "LEAKLESS_MANIFEST_DIR"

_SOURCE = DataSource(
    data_source_type="measurement",
    name="Outdoor leak-detection training facility, Dongguan",
    provider="Zenodo record 18631450 (CC BY 4.0)",
)
_SIGNAL = TimeSeriesSpec(
    spec_type="signal",
    name="Acoustic pressure (uncalibrated, amplitude-normalised)",
    # L'échantillon WAV est un entier sans unité physique : la chaîne d'acquisition
    # n'est pas calibrée. Lui attacher un pascal serait une invention.
    unit_value=ureg.dimensionless,
    data_source=_SOURCE,
)


def _normalise(x: np.ndarray) -> np.ndarray:
    """Normalisation RMS par clip, identique pour toutes les classes.

    Retire la composante continue puis met le RMS à 1. Un clip parfaitement
    silencieux est laissé à zéro plutôt que divisé par epsilon.
    """
    x = x.astype(np.float64)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x**2)))
    return x if rms == 0.0 else x / rms


def read_wav(path: str | Path) -> np.ndarray:
    """Lit un WAV mono 16 bits et renvoie exactement CLIP_SAMPLES échantillons."""
    with wave.open(str(path)) as w:
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    out = np.zeros(CLIP_SAMPLES, dtype=np.float64)
    out[: min(CLIP_SAMPLES, len(a))] = a[:CLIP_SAMPLES]
    return out


def load_split(manifest_dir: str | Path) -> list[dict]:
    """Lit le split gelé. `split_v2_audit.csv` ne sert qu'à résoudre clip_id -> chemin."""
    manifest_dir = Path(manifest_dir)
    with open(manifest_dir / "split_v2.csv") as fh:
        split = list(csv.DictReader(fh))
    with open(manifest_dir / "split_v2_audit.csv") as fh:
        paths = {r["clip_id"]: r["path"] for r in csv.DictReader(fh)}
    for r in split:
        r["path"] = paths[r["clip_id"]]
    return split


class LeaklessAcousticConnector(BaseConnector[dict]):
    """Convertit les clips du split gelé en un jeu TimeF groupé."""

    CARD = Path(__file__).with_name("dataset.yaml")

    def download(self, cache_dir: Path) -> list[dict]:  # noqa: ARG002
        """Ne télécharge rien : les archives Zenodo sont déjà extraites hors du dépôt.

        Renvoie une référence par clip, lue depuis le split gelé.
        """
        manifest_dir = os.environ.get(ENV_MANIFEST_DIR)
        if not manifest_dir:
            raise RuntimeError(f"{ENV_MANIFEST_DIR} non défini (dossier manifests/ du dépôt)")
        return load_split(manifest_dir)

    def convert(self, raw_refs: list[dict]) -> TimeFDataset:
        """Un record par clip : forme d'onde normalisée, grappe en subject_id, étiquette en tâche."""
        data_root = os.environ.get(ENV_DATA_ROOT)
        if not data_root:
            raise RuntimeError(f"{ENV_DATA_ROOT} non défini (dossier des WAV, hors dépôt)")

        dataset = TimeFDataset(metadata=self.metadata())
        for ref in sorted(raw_refs, key=lambda r: r["clip_id"]):
            values = _normalise(read_wav(Path(data_root) / ref["path"]))
            series = TimeSeries.from_values(
                values,
                spec=_SIGNAL,
                signal="acoustic",
                time_axis=RegularAxis.from_rate_hz(SAMPLING_RATE_HZ),
                source_id=ref["clip_id"],
                time_series_id=f"ts-{ref['clip_id']}",
            )
            record = dataset.add_record(
                time_series=(series,),
                # La grappe de dépendance est le "sujet" TimeF : c'est l'unité qui ne
                # doit jamais être coupée par un split.
                subject_ids=(ref["group_id"],),
                record_id=ref["clip_id"],
            )
            dataset.add_task(
                record,
                ClassificationTask(target=ref["label"], id=f"task-{ref['clip_id']}"),
            )
        return dataset


CONNECTOR = LeaklessAcousticConnector
