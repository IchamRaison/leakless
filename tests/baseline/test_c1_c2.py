"""Sentinelles C1/C2. Données synthétiques, aucune mesure de performance PIPE."""
import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from pipe.baseline import audio, nevil, train
from pipe.baseline.donnees import ecrire_json, empreinte, lire_json
from pipe.baseline.integrite import hashes_audio, verifier_audio, lire_provenance
from pipe.baseline.modele import predict_baseline, exemple_inference
from test_integration_nevil import wav_synthetique
from test_modele import artefact_fixture

RACINE = Path(__file__).resolve().parents[2]


def test_wav_remplace_refuse_meme_si_meme_clip_id():
    contenu = wav_synthetique(400)
    hashes = {"clip": hashlib.md5(contenu).hexdigest()}
    verifier_audio(contenu, "clip", hashes)
    with pytest.raises(ValueError, match="Audio différent"):
        verifier_audio(wav_synthetique(1200), "clip", hashes)
    change = bytearray(contenu); change[-1] ^= 1
    with pytest.raises(ValueError, match="clip"):
        verifier_audio(bytes(change), "clip", hashes)
    with pytest.raises(ValueError, match="Empreinte audio absente.*inconnu"):
        verifier_audio(contenu, "inconnu", hashes)


def test_provenance_refusee_si_modifiee_ou_d_un_autre_modele(tmp_path):
    chemin = tmp_path / "source_provenance.json"
    ecrire_json(chemin, {"modele": "A", "source": "fixture"})
    hash_a = empreinte(chemin)
    assert lire_provenance(chemin, hash_a)["modele"] == "A"
    ecrire_json(chemin, {"modele": "B", "source": "fixture"})
    hash_b = empreinte(chemin)
    assert lire_provenance(chemin, hash_b)["modele"] == "B"
    with pytest.raises(ValueError, match="Provenance modifiée"):
        lire_provenance(chemin, hash_a)
    chemin.write_bytes(chemin.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Provenance modifiée"):
        lire_provenance(chemin, hash_b)


def test_table_integrite_scellee_ne_retourne_que_ids_et_md5(tmp_path):
    chemin = tmp_path / "split_v2_audit.csv"
    chemin.write_bytes((RACINE / "manifests/split_v2_audit.csv").read_bytes())
    hashes = hashes_audio(tmp_path)
    assert len(hashes) == 1000 and all(len(v) == 32 for v in hashes.values())
    chemin.write_bytes(chemin.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Audit source modifié"):
        hashes_audio(tmp_path)


@pytest.mark.parametrize("classes,scores,attendu", [([0,1],[0.5,0.5],"no_leak"), ([1,0],[0.5,0.5],"leak"), ([1,0],[0.1,0.9],"no_leak")])
def test_une_seule_traversee_et_mapping_des_classes(jeu, classes, scores, attendu):
    artefact, _, _ = artefact_fixture(jeu)
    class Estimateur:
        classes_ = np.array(classes)
        appels = 0
        def predict_proba(self, x):
            self.appels += 1
            return np.array([scores])
        def predict(self, x):
            pytest.fail("Seconde traversée inutile")
    estimateur = Estimateur(); artefact["model"] = estimateur
    resultat = predict_baseline(artefact, exemple_inference(jeu[2], jeu[2]["samples"][0]))
    assert resultat["prediction"] == attendu and estimateur.appels == 1


def test_scores_invalides_restent_refuses_apres_optimisation(jeu):
    artefact, _, _ = artefact_fixture(jeu)
    class Estimateur:
        classes_ = np.array([0,1])
        def predict_proba(self, x):
            return np.array([[np.nan,0.5]])
    artefact["model"] = Estimateur()
    with pytest.raises(ValueError, match="Scores invalides"):
        predict_baseline(artefact, exemple_inference(jeu[2], jeu[2]["samples"][0]))


def _metadonnees_synthetiques_interface():
    # Faux artefact utilisé uniquement pour tester les contrôles de chargement.
    return {"metadata": {"execution_mode":"live", "feature_version":audio.version_features(),
            "source_files":{n:empreinte(Path(audio.__file__).parent/n) for n in ("audio.py","modele.py","donnees.py")}}}


def test_hash_dsp_verifie_au_chargement_pas_a_chaque_inference(monkeypatch):
    artefact = _metadonnees_synthetiques_interface()
    monkeypatch.setattr(audio, "charger_modele", lambda *a, **k: artefact)
    predicteur = audio.Predictor("fixture", "0" * 64, artefact_de_confiance=True)
    def interdit(*a, **k):
        pytest.fail("Lecture de code/DSP pendant chaque inférence")
    monkeypatch.setattr(audio, "empreinte", interdit)
    monkeypatch.setattr(audio, "version_features", interdit)
    captures = []
    class Capture(Exception): pass
    def capturer(_, exemple):
        captures.append(exemple)
        raise Capture
    monkeypatch.setattr(audio, "predict_baseline", capturer)
    for _ in range(2):
        with pytest.raises(Capture): predicteur.predict(wav_synthetique())
    assert len(captures) == 2 and captures[0]["feature_version"] == artefact["metadata"]["feature_version"]


@pytest.mark.parametrize("nom", ["audio.py", "modele.py", "donnees.py"])
def test_code_incompatible_refuse_au_chargement(monkeypatch, nom):
    artefact = _metadonnees_synthetiques_interface()
    artefact["metadata"]["source_files"][nom] = "0" * 64
    monkeypatch.setattr(audio, "charger_modele", lambda *a, **k: artefact)
    with pytest.raises(audio.PredictionError, match="incompatible"):
        audio.Predictor("fixture", "0" * 64, artefact_de_confiance=True)


def test_dsp_incompatible_refuse_au_chargement(monkeypatch):
    artefact = _metadonnees_synthetiques_interface()
    artefact["metadata"]["feature_version"] = "autre-dsp"
    monkeypatch.setattr(audio, "charger_modele", lambda *a, **k: artefact)
    with pytest.raises(audio.PredictionError, match="DSP incompatible"):
        audio.Predictor("fixture", "0" * 64, artefact_de_confiance=True)


def test_projection_et_labels_recoupes_avec_nevil(tmp_path):
    split = nevil.charger_split(RACINE/"manifests")
    donnees = {"dataset_version":nevil.PROTOCOL+":"+split.sha256,
               "samples":[{"sample_id":c.clip_id,"label":c.label} for c in split.clips if c.fold in ("train","val")]}
    affectations = [{"sample_id":c.clip_id,"event_group_id":c.group_id,
                     "split":"validation" if c.fold=="val" else c.fold} for c in split.clips]
    chemin = tmp_path/"split.json"; ecrire_json(chemin,{"assignments":affectations})
    nevil.verifier_developpement(donnees,chemin,RACINE/"manifests")
    altere = copy.deepcopy(donnees); altere["samples"][0]["label"] ^= 1
    with pytest.raises(ValueError,match="Label développement"):
        nevil.verifier_developpement(altere,chemin,RACINE/"manifests")
    affectations[0]["event_group_id"] = "groupe-invente"
    ecrire_json(chemin,{"assignments":affectations})
    with pytest.raises(ValueError,match="projection"):
        nevil.verifier_developpement(donnees,chemin,RACINE/"manifests")


def test_features_modifiees_refusees_avant_fit(jeu, monkeypatch):
    chemin, _, donnees, _ = jeu
    donnees["dataset_version"] = nevil.PROTOCOL+":"+"f"*64
    ecrire_json(chemin.parent/"features.json",donnees)
    ecrire_json(chemin.parent/"source_provenance.json",{"features_sha256":"0"*64})
    monkeypatch.setattr(nevil,"verifier_developpement",lambda *a: None)
    monkeypatch.setattr(train,"entrainer",lambda *a: pytest.fail("Fit avant contrôle d'intégrité"))
    with pytest.raises(ValueError,match="Features modifiées"):
        train.executer(chemin)
