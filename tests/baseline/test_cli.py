import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pipe.baseline.donnees import ecrire_json, empreinte, lire_json
from pipe.baseline.modele import exemple_inference
from pipe.baseline.train import executer


RACINE = Path(__file__).resolve().parents[2]


def lancer(*arguments):
    environnement = {**os.environ, "PYTHONPATH": str(RACINE / "src"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    return subprocess.run([sys.executable, *arguments], env=environnement, cwd=RACINE, text=True, capture_output=True, timeout=60)


def test_cli_train_reload_export_processus_neuf(jeu):
    chemin, _, donnees, _ = jeu
    resultat = lancer("-m", "pipe.baseline.train", "--config", str(chemin))
    assert resultat.returncode == 0, resultat.stderr
    dossier = chemin.parent / "run"
    metadata = lire_json(dossier / "metadata.json")
    rapport = lire_json(dossier / "metrics.json")
    assert rapport["execution_mode"] == "development_fixture"
    assert rapport["benchmark_eligible"] is False
    assert rapport["final_test_evaluated"] is False
    assert rapport["scope"] == "validation_only"
    predictions = [json.loads(ligne) for ligne in (dossier / "predictions.jsonl").read_text().splitlines()]
    with (dossier / "predictions.csv").open() as fichier:
        csv_lignes = list(csv.DictReader(fichier))
    attendus = [ligne["sample_id"] for ligne in donnees["samples"][24:]]
    assert [ligne["sample_id"] for ligne in predictions] == attendus
    assert [ligne["sample_id"] for ligne in csv_lignes] == attendus
    assert metadata["source_sha256"] and metadata["environment"]
    for prediction, ligne in zip(predictions, donnees["samples"][24:]):
        entree = chemin.parent / "inference.json"
        sortie = chemin.parent / f"prediction-{ligne['sample_id']}.json"
        ecrire_json(entree, exemple_inference(donnees, ligne))
        resultat = lancer("-m", "pipe.baseline.predict", "--model", str(dossier / "model.joblib"), "--sha256", metadata["model_sha256"], "--input", str(entree), "--output", str(sortie), "--trusted-artifact")
        assert resultat.returncode == 0, resultat.stderr
        rechargee = lire_json(sortie)
        assert rechargee["prediction"] == prediction["prediction"]
        assert rechargee["class_scores"] == prediction["class_scores"]
        assert rechargee["execution_mode"] == "development_fixture"
        assert not {"label", "ground_truth", "y", "event_group_id"}.intersection(rechargee)
    marqueur = lire_json(dossier / "run.json")
    assert marqueur["status"] == "complete"
    for nom, hachage in marqueur["files"].items():
        assert empreinte(dossier / nom) == hachage


def test_configuration_reelle_echoue_clairement_sans_donnees():
    resultat = lancer("-m", "pipe.baseline.train", "--config", "configs/baseline/default.json")
    assert resultat.returncode == 2
    assert "demander les fichiers de développement à Nevil" in resultat.stderr
    assert not (RACINE / "artifacts/baseline/development-001").exists()


def test_run_existant_jamais_ecrase(jeu):
    dossier = executer(jeu[0])
    avant = empreinte(dossier / "model.joblib")
    with pytest.raises(ValueError, match="existe déjà"):
        executer(jeu[0])
    assert empreinte(dossier / "model.joblib") == avant


def test_erreur_inference_ne_cree_pas_un_run_complet(jeu, monkeypatch):
    from pipe.baseline import train
    original = train.predict_baseline
    identifiant_erreur = jeu[2]["samples"][26]["sample_id"]

    def erreur_controlee(artefact, exemple):
        if exemple["sample_id"] == identifiant_erreur:
            raise ValueError("échantillon en échec")
        return original(artefact, exemple)

    monkeypatch.setattr(train, "predict_baseline", erreur_controlee)
    with pytest.raises(ValueError, match="échantillon en échec"):
        executer(jeu[0])
    assert not (jeu[0].parent / "run/run.json").exists()
