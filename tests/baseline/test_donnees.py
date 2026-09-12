import copy

import numpy as np
import pytest

from pipe.baseline.donnees import charger_developpement, ecrire_json, empreinte, lire_json, verifier_matrice
from pipe.baseline.train import charger_configuration


def charger(jeu):
    chemin, configuration, donnees, manifeste = jeu
    ecrire_json(chemin.parent / "features.json", donnees)
    ecrire_json(chemin.parent / "split.json", manifeste)
    configuration["split_sha256"] = empreinte(chemin.parent / "split.json")
    return charger_developpement(chemin.parent / "features.json", chemin.parent / "split.json", configuration)


def test_fixture_separe_le_test_scelle(jeu):
    donnees, matrice, cibles, partitions = charger(jeu)
    assert matrice.shape == (36, 3)
    assert matrice.dtype == np.float32
    assert set(partitions) == {"train", "validation"}
    assert set(cibles) == {0, 1}
    assert donnees["execution_mode"] == "development_fixture"


@pytest.mark.parametrize("valeur", [float("nan"), float("inf"), -float("inf"), 1e39])
def test_finitude_apres_conversion_float32(valeur):
    with pytest.raises(ValueError, match="float32"):
        verifier_matrice([[1.0, valeur]], 2)


@pytest.mark.parametrize("valeurs", [[[1, True]], [["1", "2"]], [[1, 2], [3]], [], [1, 2], [[1, 2, 3]]])
def test_features_malformees_rejetees(valeurs):
    with pytest.raises(ValueError):
        verifier_matrice(valeurs, 2)


@pytest.mark.parametrize("nom", ["target_label", "sample_id", "acquisition_device", "source_class", "fileName", "event_group_id", "session", "pressure_value"])
def test_labels_et_metadonnees_interdits_meme_dans_allowlist(jeu, nom):
    jeu[1]["feature_names"][0] = nom
    jeu[2]["feature_names"][0] = nom
    with pytest.raises(ValueError, match="Feature interdite"):
        charger(jeu)


def test_ordre_features_est_contractuel(jeu):
    jeu[2]["feature_names"].reverse()
    with pytest.raises(ValueError, match="Ordre"):
        charger(jeu)


@pytest.mark.parametrize("index", [24, 36])
def test_groupes_communs_validation_ou_test_rejetes(jeu, index):
    jeu[3]["assignments"][index]["event_group_id"] = "g00"
    with pytest.raises(ValueError, match="Intersection"):
        charger(jeu)


def test_doublon_signal_entre_train_validation(jeu):
    jeu[2]["samples"][24]["input_sha256"] = jeu[2]["samples"][0]["input_sha256"]
    with pytest.raises(ValueError, match="Même signal"):
        charger(jeu)


def test_hash_modifie_refuse(jeu):
    chemin, configuration, _, _ = jeu
    configuration["split_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash du split"):
        charger_developpement(chemin.parent / "features.json", chemin.parent / "split.json", configuration)


@pytest.mark.parametrize("etiquette", [True, 0.5, "1", 2, -1, None])
def test_labels_binaires_stricts(jeu, etiquette):
    jeu[2]["samples"][0]["label"] = etiquette
    with pytest.raises(ValueError, match="Labels binaires"):
        charger(jeu)


def test_une_seule_classe_train_refusee(jeu):
    for ligne in jeu[2]["samples"][:24]:
        ligne["label"] = 0
    with pytest.raises(ValueError, match="deux classes"):
        charger(jeu)


def test_aucun_signal_test_accepte(jeu):
    ligne = copy.deepcopy(jeu[2]["samples"][0])
    ligne["sample_id"] = jeu[3]["assignments"][36]["sample_id"]
    jeu[2]["samples"].append(ligne)
    with pytest.raises(ValueError, match="test ou quarantine"):
        charger(jeu)


def test_aucune_omission_silencieuse(jeu):
    jeu[2]["samples"].pop()
    with pytest.raises(ValueError, match="manquants"):
        charger(jeu)


def test_identifiants_uniques(jeu):
    jeu[2]["samples"].append(copy.deepcopy(jeu[2]["samples"][0]))
    with pytest.raises(ValueError, match="répété"):
        charger(jeu)


def test_fixture_ne_peut_pas_etre_declaree_reelle_par_config(jeu):
    jeu[1]["execution_mode"] = "live"
    with pytest.raises(ValueError, match="synthétique/réel"):
        charger(jeu)


def test_config_typo_refusee(jeu):
    chemin, configuration, _, _ = jeu
    configuration["seeed"] = 99
    ecrire_json(chemin, configuration)
    with pytest.raises(ValueError, match="configuration"):
        charger_configuration(chemin)


def test_cles_json_dupliquees_refusees(tmp_path):
    chemin = tmp_path / "duplique.json"
    chemin.write_text('{"seed": 1, "seed": 2}')
    with pytest.raises(ValueError, match="répétée"):
        lire_json(chemin)


def test_json_preserve_les_features_float32(jeu):
    _, matrice, _, _ = charger(jeu)
    for indice, ligne in enumerate(jeu[2]["samples"]):
        np.testing.assert_array_equal(matrice[indice], np.asarray(ligne["features"], dtype=np.float32))
