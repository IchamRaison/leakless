import copy
import warnings

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning

from pipe.baseline.donnees import charger_developpement, ecrire_json, lire_json
from pipe.baseline.modele import charger_modele, entrainer, exemple_inference, metriques, predict_baseline
from pipe.baseline.train import executer


def artefact_fixture(jeu):
    dossier = executer(jeu[0])
    metadata = lire_json(dossier / "metadata.json")
    return charger_modele(dossier / "model.joblib", metadata["model_sha256"], artefact_de_confiance=True), metadata, dossier


def test_fit_ne_voit_que_train(jeu, monkeypatch):
    chemin, configuration, _, _ = jeu
    _, matrice, cibles, partitions = charger_developpement(chemin.parent / "features.json", chemin.parent / "split.json", configuration)
    original = RandomForestClassifier.fit
    appels = []

    def espion(modele, entrees, etiquettes):
        appels.append(len(entrees))
        np.testing.assert_array_equal(entrees, matrice[:24])
        np.testing.assert_array_equal(etiquettes, cibles[:24])
        return original(modele, entrees, etiquettes)

    monkeypatch.setattr(RandomForestClassifier, "fit", espion)
    entrainer(matrice, cibles, partitions, configuration)
    assert appels == [24]


def test_matrice_confusion_independante_orientation_et_scores():
    resultat = metriques(np.array([0, 0, 0, 1, 1, 1, 1]), np.array([0, 1, 1, 0, 1, 1, 1]))
    assert resultat["confusion_matrix"] == [[1, 2], [1, 3]]
    assert resultat["leak_precision"] == pytest.approx(3 / 5)
    assert resultat["leak_recall"] == pytest.approx(3 / 4)
    assert resultat["macro_f1"] == pytest.approx(((2 / 5) + (6 / 9)) / 2)
    assert resultat["false_positive_rate"] == pytest.approx(2 / 3)


def test_classe_absente_signalee():
    resultat = metriques(np.array([1, 1]), np.array([1, 0]))
    assert resultat["warnings"]
    assert resultat["false_positive_rate"] is None


def test_mapping_scores_depend_des_classes_reelles(jeu):
    artefact, _, _ = artefact_fixture(jeu)

    class EstimateurInverse:
        classes_ = np.array([1, 0])

        def predict_proba(self, matrice):
            return np.array([[0.8, 0.2]])

        def predict(self, matrice):
            return np.array([1])

    artefact["model"] = EstimateurInverse()
    resultat = predict_baseline(artefact, exemple_inference(jeu[2], jeu[2]["samples"][0]))
    assert resultat["class_scores"] == {"leak": 0.8, "no_leak": 0.2}
    assert resultat["prediction"] == "leak"


@pytest.mark.parametrize("champ", ["label", "ground_truth", "y", "event_group_id"])
def test_entree_inference_refuse_labels_et_groupes(jeu, champ):
    artefact, _, _ = artefact_fixture(jeu)
    exemple = exemple_inference(jeu[2], jeu[2]["samples"][0])
    exemple[champ] = 1
    with pytest.raises(ValueError, match="aucun label"):
        predict_baseline(artefact, exemple)


@pytest.mark.parametrize("champ", ["feature_version", "preprocessing_version", "feature_names"])
def test_inference_refuse_contrat_different(jeu, champ):
    artefact, _, _ = artefact_fixture(jeu)
    exemple = exemple_inference(jeu[2], jeu[2]["samples"][0])
    exemple[champ] = list(reversed(exemple[champ])) if champ == "feature_names" else "autre-version"
    with pytest.raises(ValueError, match="incompatible"):
        predict_baseline(artefact, exemple)


def test_modele_synthetique_reste_etiquete_sur_entree_live(jeu):
    artefact, _, _ = artefact_fixture(jeu)
    exemple = exemple_inference(jeu[2], jeu[2]["samples"][0])
    exemple["execution_mode"] = "live"
    resultat = predict_baseline(artefact, exemple)
    assert resultat["execution_mode"] == "development_fixture"
    assert resultat["warnings"]
    assert resultat["description"] is None
    assert resultat["observations"] == []


def test_checksum_et_confiance_controles_avant_deserialisation(jeu, monkeypatch):
    _, metadata, dossier = artefact_fixture(jeu)
    fichier = dossier / "model.joblib"
    fichier.write_bytes(fichier.read_bytes() + b"corruption")

    def interdit(*arguments, **options):
        pytest.fail("Désérialisation avant contrôle de confiance et d'intégrité")

    monkeypatch.setattr(joblib, "load", interdit)
    with pytest.raises(ValueError, match="confiance"):
        charger_modele(fichier, metadata["model_sha256"])
    with pytest.raises(ValueError, match="Checksum"):
        charger_modele(fichier, metadata["model_sha256"], artefact_de_confiance=True)


def test_version_sklearn_incompatible_est_une_erreur(jeu, monkeypatch):
    _, metadata, dossier = artefact_fixture(jeu)

    def mauvaise_version(*arguments):
        warnings.warn(InconsistentVersionWarning(estimator_name="RandomForestClassifier", current_sklearn_version="1.9.1", original_sklearn_version="0.0"))

    monkeypatch.setattr(joblib, "load", mauvaise_version)
    with pytest.raises(ValueError, match="sklearn incompatible"):
        charger_modele(dossier / "model.joblib", metadata["model_sha256"], artefact_de_confiance=True)


def test_seed_reproduit_predictions(jeu):
    artefact, _, _ = artefact_fixture(jeu)
    configuration = copy.deepcopy(jeu[1])
    configuration["output_dir"] = "second-run"
    ecrire_json(jeu[0], configuration)
    second_dossier = executer(jeu[0])
    metadata = lire_json(second_dossier / "metadata.json")
    second = charger_modele(second_dossier / "model.joblib", metadata["model_sha256"], artefact_de_confiance=True)
    matrice = np.asarray([ligne["features"] for ligne in jeu[2]["samples"]], dtype=np.float32)
    np.testing.assert_array_equal(artefact["model"].predict_proba(matrice), second["model"].predict_proba(matrice))
