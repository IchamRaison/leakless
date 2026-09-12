"""Créer des données synthétiques pour tester la mécanique, jamais le benchmark PIPE."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from .donnees import ecrire_json, empreinte


def creer_fixture(dossier):
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=False)
    generateur = np.random.default_rng(42)
    affectations, exemples = [], []
    for indice in range(40):
        identifiant = hashlib.sha256(f"fixture-sample-{indice}".encode()).hexdigest()[:16]
        partition = "train" if indice < 24 else "validation" if indice < 36 else "test"
        affectations.append({"sample_id": identifiant, "event_group_id": f"g{indice // 4:02d}", "split": partition})
        if partition == "test":
            continue
        etiquette = indice % 2
        valeurs = generateur.normal(size=3).astype(np.float32)
        valeurs[0] += etiquette * 2
        exemples.append({"sample_id": identifiant, "input_sha256": hashlib.sha256(valeurs.tobytes()).hexdigest(), "features": valeurs.tolist(), "label": etiquette})
    donnees = {"schema_version": "baseline-features-v0", "execution_mode": "development_fixture", "dataset_version": "synthetic-mechanics-v0",
               "feature_names": ["band_00_mean", "band_00_std", "band_01_mean"], "feature_version": "synthetic-v0", "preprocessing_version": "synthetic-no-dsp-v0", "samples": exemples}
    ecrire_json(dossier / "split.json", {"schema_version": "baseline-split-v0", "assignments": affectations})
    ecrire_json(dossier / "features.json", donnees)
    configuration = {"data_path": "features.json", "split_path": "split.json", "split_sha256": empreinte(dossier / "split.json"), "output_dir": "run",
                     "seed": 42, "random_forest": {"n_estimators": 32, "max_depth": 5, "min_samples_leaf": 2, "class_weight": None},
                     **{cle: donnees[cle] for cle in ("feature_names", "feature_version", "preprocessing_version", "execution_mode")}}
    ecrire_json(dossier / "config.json", configuration)
    return dossier / "config.json"


if __name__ == "__main__":
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--output", required=True)
    options = arguments.parse_args()
    print(creer_fixture(options.output))
