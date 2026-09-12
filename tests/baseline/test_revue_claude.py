"""Tests proposés par Claude via le bus, relus et complétés par Codex."""
import csv
import json

import numpy as np
import pytest

from pipe.baseline.donnees import ecrire_json, lire_json, verifier_matrice
from pipe.baseline.modele import charger_modele, exemple_inference, metriques, predict_baseline
from pipe.baseline.train import charger_configuration, executer


def test_metriques_recalculables_depuis_predictions_exportees(jeu):
    dossier = executer(jeu[0])
    rapport = lire_json(dossier / "metrics.json")
    exportees = [json.loads(ligne) for ligne in (dossier / "predictions.jsonl").read_text().splitlines()]
    labels = {ligne["sample_id"]: ligne["label"] for ligne in jeu[2]["samples"]}
    cibles = np.array([labels[prediction["sample_id"]] for prediction in exportees])
    predites = np.array([{"leak": 1, "no_leak": 0}[prediction["prediction"]] for prediction in exportees])
    assert metriques(cibles, predites) == rapport["baseline"]


def test_scores_csv_identiques_au_jsonl(jeu):
    dossier = executer(jeu[0])
    exportees = [json.loads(ligne) for ligne in (dossier / "predictions.jsonl").read_text().splitlines()]
    with (dossier / "predictions.csv").open() as fichier:
        lignes = list(csv.DictReader(fichier))
    assert len(lignes) == len(exportees)
    for ligne, prediction in zip(lignes, exportees):
        assert ligne["sample_id"] == prediction["sample_id"]
        assert float(ligne["score_leak"]) == prediction["class_scores"]["leak"]
        assert float(ligne["score_no_leak"]) == prediction["class_scores"]["no_leak"]


def cles_profondes(objet):
    if isinstance(objet, dict):
        for cle, valeur in objet.items():
            yield cle
            yield from cles_profondes(valeur)
    elif isinstance(objet, list):
        for valeur in objet:
            yield from cles_profondes(valeur)


def test_aucune_prediction_ne_porte_de_cible_meme_imbriquee(jeu):
    dossier = executer(jeu[0])
    for ligne in (dossier / "predictions.jsonl").read_text().splitlines():
        for cle in cles_profondes(json.loads(ligne)):
            termes = set(cle.lower().split("_"))
            assert not termes & {"label", "labels", "target", "truth", "ground", "y"}


def test_metriques_suivent_meme_une_divergence_du_chemin_inference(jeu, monkeypatch):
    from pipe.baseline import train
    original = train.predict_baseline

    def toujours_fuite(artefact, exemple):
        prediction = original(artefact, exemple)
        prediction.update(prediction="leak", class_scores={"leak": 1.0, "no_leak": 0.0})
        return prediction

    monkeypatch.setattr(train, "predict_baseline", toujours_fuite)
    dossier = executer(jeu[0])
    rapport = lire_json(dossier / "metrics.json")
    assert rapport["baseline"]["confusion_matrix"] == [[0, 6], [0, 6]]


@pytest.mark.parametrize("nom", ["sklearn", "numpy", "scipy", "joblib"])
def test_erreur_version_nomme_la_dependance(jeu, monkeypatch, nom):
    from pipe.baseline import modele
    dossier = executer(jeu[0])
    metadata = lire_json(dossier / "metadata.json")
    environnement = modele.versions()
    environnement[nom] = "0.0.0"
    monkeypatch.setattr(modele, "versions", lambda: environnement)
    with pytest.raises(ValueError, match=nom):
        charger_modele(dossier / "model.joblib", metadata["model_sha256"], artefact_de_confiance=True)


def test_correctif_python_distinct_tolere_sans_promettre_compatibilite_externe(jeu, monkeypatch):
    from pipe.baseline import modele
    dossier = executer(jeu[0])
    metadata = lire_json(dossier / "metadata.json")
    environnement = modele.versions()
    environnement["python"] = ".".join(environnement["python"].split(".")[:2]) + ".999"
    monkeypatch.setattr(modele, "versions", lambda: environnement)
    assert charger_modele(dossier / "model.joblib", metadata["model_sha256"], artefact_de_confiance=True)


def test_sous_depassement_float32_signale():
    with pytest.raises(ValueError, match="Sous-dépassement"):
        verifier_matrice([[1.0, 1e-46]], 2)


@pytest.mark.parametrize("champ", ["data_path", "split_path", "output_dir"])
def test_chemins_config_mal_types_erreur_nommee(jeu, champ):
    jeu[1][champ] = 5
    ecrire_json(jeu[0], jeu[1])
    with pytest.raises(ValueError, match=champ):
        charger_configuration(jeu[0])


def test_inference_ne_simule_pas_un_replay(jeu):
    dossier = executer(jeu[0])
    metadata = lire_json(dossier / "metadata.json")
    artefact = charger_modele(dossier / "model.joblib", metadata["model_sha256"], artefact_de_confiance=True)
    exemple = exemple_inference(jeu[2], jeu[2]["samples"][0])
    exemple["execution_mode"] = "replay"
    with pytest.raises(ValueError, match="Mode d'inférence invalide"):
        predict_baseline(artefact, exemple)


def test_ecriture_exclusive_ne_detruit_pas_sortie_existante(tmp_path):
    chemin = tmp_path / "prediction.json"
    ecrire_json(chemin, {"ancien": True})
    with pytest.raises(FileExistsError):
        ecrire_json(chemin, {"nouveau": True}, exclusif=True)
    assert lire_json(chemin) == {"ancien": True}
