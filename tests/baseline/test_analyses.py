import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from pipe.baseline import analyses
from pipe.baseline.analyses import bootstrap_groupes, metriques_selectives, selectionner_seuil, validation_croisee
from pipe.baseline.donnees import ecrire_json, lire_json
from pipe.baseline.modele import charger_modele, exemple_inference, predict_baseline, sauvegarder
from pipe.baseline.proposer_features import proposer
from pipe.baseline.train import executer
from pipe.baseline.valider import valider


RACINE = Path(__file__).resolve().parents[2]
OPTIONS = RACINE / "configs/baseline/analyses.json"


def test_validateur_rapporte_splits_groupes_et_classe_test_inconnue(jeu):
    rapport = valider(jeu[0])
    assert rapport["partitions"]["train"] == {"samples": 24, "groups": 6, "class_counts": {"0": 12, "1": 12}}
    assert rapport["partitions"]["test"] == {"samples": 4, "groups": 1, "class_counts": None}
    assert rapport["feature_names_received"] == rapport["feature_names_expected"]
    assert not (jeu[0].parent / "run").exists()


def test_validateur_erreur_nomme_sample_id_et_sort_avec_un(jeu):
    ligne = jeu[2]["samples"][3]
    ligne["features"] = [0.0]
    ecrire_json(jeu[0].parent / "features.json", jeu[2])
    resultat = subprocess.run([sys.executable, "-m", "pipe.baseline.valider", "--config", str(jeu[0])], env={**os.environ, "PYTHONPATH": str(RACINE / "src")}, capture_output=True, text=True, timeout=30)
    assert resultat.returncode == 1
    assert ligne["sample_id"] in json.loads(resultat.stdout)["error"]
    assert "Traceback" not in resultat.stderr


def test_seuil_selectionne_couverture_et_precision_retenue():
    rapport = selectionner_seuil(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]), np.array([0.95, 0.95, 0.55, 0.9]), [0.5, 0.6, 0.8, 1.0], 0.75)
    assert rapport["threshold"] == 0.6
    assert rapport["curve"][1]["coverage"] == 0.75
    assert rapport["curve"][1]["retained_accuracy"] == 1
    assert rapport["independent_evaluation"] is False
    assert rapport["curve"][-1]["accepted_only"] is None


def test_abstentions_sur_fuites_reduisent_rappel_global():
    resultat = metriques_selectives(np.array([0, 1, 1]), np.array([0, 1, -1]))
    assert resultat["accepted_only"]["leak_recall"] == 1
    assert resultat["leak_recall_all"] == 0.5
    assert resultat["coverage"] == pytest.approx(2 / 3)
    assert resultat["confusion_matrix_with_abstentions"] == [[1, 0, 0], [0, 1, 1]]


def test_seuil_prefere_macro_f1_a_exactitude_sur_classe_majoritaire():
    rapport = selectionner_seuil(np.array([0] * 8 + [1] * 2), np.array([0] * 6 + [1] * 4), np.array([0.9] * 6 + [0.55] * 4), [0.5, 0.8], 0.6)
    assert rapport["threshold"] == 0.5
    assert rapport["objective"] == "max_retained_macro_f1_subject_to_minimum_coverage"
    assert rapport["curve"][1]["retained_accuracy"] == 1.0
    assert rapport["curve"][1]["leak_recall_all"] == 0.0


def test_politique_abstention_survit_au_reload(jeu):
    dossier = executer(jeu[0])
    metadata = lire_json(dossier / "metadata.json")
    artefact = charger_modele(dossier / "model.joblib", metadata["model_sha256"], artefact_de_confiance=True)
    matrice = np.array([ligne["features"] for ligne in jeu[2]["samples"]], dtype=np.float32)
    indice = np.flatnonzero(artefact["model"].predict_proba(matrice).max(axis=1) < 1)[0]
    artefact["metadata"]["abstention_threshold"] = 1.0
    destination = jeu[0].parent / "abstention"
    hachage = sauvegarder(artefact["model"], artefact["metadata"], destination)
    recharge = charger_modele(destination / "model.joblib", hachage, artefact_de_confiance=True)
    resultat = predict_baseline(recharge, exemple_inference(jeu[2], jeu[2]["samples"][int(indice)]))
    assert resultat["prediction"] is None and resultat["abstained"] is True
    assert resultat["abstention_reason"] and resultat["score_type"] == "raw"


def test_cv_groupes_non_croises_et_seulement_developpement():
    class ForetControlee(RandomForestClassifier):
        def fit(self, X, y, **options):
            self.groupes_vus = set((X[:, 0] // 4).astype(int))
            assert X[:, 0].max() < 24
            return super().fit(X, y, **options)

        def predict(self, X):
            assert not self.groupes_vus & set((X[:, 0] // 4).astype(int))
            return super().predict(X)

    matrice = np.arange(24, dtype=float).reshape(-1, 1)
    resultat = validation_croisee(ForetControlee(n_estimators=4, random_state=42, n_jobs=1), matrice, np.arange(24) % 2, np.arange(24) // 4, 3)
    assert resultat["status"] == "ok"
    assert sum(pli["n_validation"] for pli in resultat["folds"]) == 24
    assert resultat["scope"] == "development_only_without_abstention"


def test_cv_incomplete_ne_moyenne_pas_uniquement_les_bons_plis():
    resultat = validation_croisee(RandomForestClassifier(n_estimators=4), np.arange(8).reshape(-1, 1), np.array([0, 0, 0, 0, 1, 1, 1, 1]), np.array([0, 0, 0, 0, 1, 1, 1, 1]), 2)
    assert resultat["status"] == "incomplete_class_support"
    assert resultat["summary"] is None


def test_bootstrap_trop_peu_de_groupes_pas_intervalle():
    resultat = bootstrap_groupes(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]), np.array([0, 0, 1, 1]), 100, 5, 42)
    assert resultat["interval"] is None
    assert resultat["status"] == "insufficient_group_or_class_support"


def test_bootstrap_rechantillonne_groupes_entiers_pas_clips(monkeypatch):
    tailles = [2, 4, 6, 8, 10]
    groupes = np.repeat(np.arange(5), tailles)
    cibles = np.concatenate([np.arange(taille) % 2 for taille in tailles])
    appels = []

    class GenerateurControle:
        def choice(self, nombre, size, replace):
            assert nombre == size == 5 and replace is True
            return np.array([0, 0, 2, 2, 2])

    def score(vraies, predictions, **options):
        appels.append(len(vraies))
        return 0.5

    monkeypatch.setattr(analyses.np.random, "default_rng", lambda seed: GenerateurControle())
    monkeypatch.setattr(analyses, "f1_score", score)
    resultat = bootstrap_groupes(cibles, cibles, groupes, 100, 5, 42)
    assert appels[:-1] == [22] * 100
    assert appels[-1] == 30
    assert resultat["interval"] == [0.5, 0.5]


def test_bootstrap_reproductible():
    cibles = np.tile([0, 1], 10)
    predites = cibles.copy(); predites[0] = 1
    groupes = np.repeat(np.arange(5), 4)
    premier = bootstrap_groupes(cibles, predites, groupes, 100, 5, 42)
    assert premier == bootstrap_groupes(cibles, predites, groupes, 100, 5, 42)
    assert premier["status"] == "ok"


def test_bootstrap_perte_de_classe_pas_filtrage_silencieux():
    cibles = np.repeat([1, 0, 0, 0, 0], 2)
    resultat = bootstrap_groupes(cibles, cibles, np.repeat(np.arange(5), 2), 100, 5, 42)
    assert resultat["status"] == "unstable_class_support" and resultat["interval"] is None
    assert resultat["n_degenerate_resamples"] > 0


def test_analyses_integrees_sans_data_test_ni_amplitude_inventee(jeu):
    dossier = executer(jeu[0], OPTIONS)
    rapport = lire_json(dossier / "metrics.json")
    metadata = lire_json(dossier / "metadata.json")
    assert rapport["analyses"]["group_cv"]["n_groups"] == 9
    assert rapport["analyses"]["amplitude_ablation"]["status"] == "unavailable"
    assert rapport["analyses"]["group_bootstrap"]["interval"] is None
    assert rapport["selective"]["n_total"] == 12
    assert metadata["abstention_threshold"] == rapport["analyses"]["abstention"]["threshold"]
    assert rapport["benchmark_eligible"] is False


def test_ablation_utilise_uniquement_feature_globale_fournie(jeu, monkeypatch):
    jeu[1]["feature_names"][0] = jeu[2]["feature_names"][0] = "clip_log_rms"
    ecrire_json(jeu[0], jeu[1]); ecrire_json(jeu[0].parent / "features.json", jeu[2])
    original = RandomForestClassifier.fit
    appels_amplitude = []

    def verifier_fit(modele, matrice, cibles, **options):
        if matrice.shape[1] == 1:
            attendues = np.asarray([ligne["features"][0] for ligne in jeu[2]["samples"][:24]], dtype=np.float32)
            np.testing.assert_array_equal(matrice[:, 0], attendues)
            appels_amplitude.append(matrice.shape)
        return original(modele, matrice, cibles, **options)

    monkeypatch.setattr(RandomForestClassifier, "fit", verifier_fit)
    dossier = executer(jeu[0], OPTIONS)
    rapport = lire_json(dossier / "metrics.json")
    assert rapport["analyses"]["amplitude_ablation"]["status"] == "ok"
    assert rapport["analyses"]["amplitude_ablation"]["feature"] == "clip_log_rms"
    assert appels_amplitude == [(24, 1)]


def test_exports_abstention_conservent_toutes_les_lignes_et_les_denominators(jeu, monkeypatch):
    from pipe.baseline import train
    original = train.analyser

    def imposer_seuil_haut(*arguments):
        rapport = original(*arguments)
        rapport["abstention"]["threshold"] = 1.0
        return rapport

    monkeypatch.setattr(train, "analyser", imposer_seuil_haut)
    dossier = executer(jeu[0], OPTIONS)
    rapport = lire_json(dossier / "metrics.json")
    lignes = [json.loads(ligne) for ligne in (dossier / "predictions.jsonl").read_text().splitlines()]
    assert len(lignes) == 12
    nombre = sum(ligne["abstained"] for ligne in lignes)
    assert nombre > 0
    assert all((ligne["prediction"] is None) == ligne["abstained"] for ligne in lignes)
    assert rapport["baseline"] is None
    assert rapport["selective"]["n_abstained"] == nombre
    assert rapport["selective"]["coverage"] == pytest.approx((12 - nombre) / 12)


def test_proposition_features_ordonnee_pas_decision_dsp():
    resultat = proposer(2)
    assert resultat["feature_names"] == ["band_00_mean", "band_00_std", "band_00_q25", "band_00_q50", "band_00_q75", "band_01_mean", "band_01_std", "band_01_q25", "band_01_q50", "band_01_q75", "clip_log_rms"]
    assert resultat["status"] == "proposal_not_approved"
