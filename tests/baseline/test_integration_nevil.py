"""Sentinelles synthétiques de l'intégration, aucune performance PIPE ici."""
import io
import shutil
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pipe.baseline import audio, nevil
from pipe.baseline.donnees import ecrire_json, lire_json
from pipe.tslm.preprocessing import preprocess_audio
from scripts.eval.harness.split_loader import Clip, Split

RACINE = Path(__file__).resolve().parents[2]


def wav_synthetique(frequence=900, taux=8000, n=8000, canaux=1):
    signal = (1000 * np.sin(2 * np.pi * frequence * np.arange(n) / taux)).astype("<i2")
    tampon = io.BytesIO()
    with wave.open(tampon, "wb") as fichier:
        fichier.setnchannels(canaux)
        fichier.setsampwidth(2)
        fichier.setframerate(taux)
        fichier.writeframes(signal.tobytes())
    return tampon.getvalue()


def test_agregats_excluent_padding_et_respectent_ordre():
    serie = np.pad(np.tile(np.arange(61, dtype=float), (4, 1)), ((0, 0), (0, 3)))
    resultat = audio.agreger(serie)
    assert len(resultat) == len(audio.FEATURE_NAMES) == 20
    np.testing.assert_allclose(resultat[:5], [30, np.std(np.arange(61)), 15, 30, 45])
    assert audio.FEATURE_NAMES[:5] == ["band_00_mean", "band_00_std", "band_00_q25", "band_00_q50", "band_00_q75"]


@pytest.mark.parametrize("anomalie", ["forme", "nan", "negatif", "padding"])
def test_series_corrompues_refusees(anomalie):
    serie = np.zeros((4, 64))
    if anomalie == "forme":
        serie = serie[:, :61]
    else:
        serie[0, 63 if anomalie == "padding" else 0] = {"nan": np.nan, "negatif": -1, "padding": 1}[anomalie]
    with pytest.raises(ValueError):
        audio.agreger(serie)


def test_agregats_ne_conservent_pas_ordre_temporel():
    serie = np.pad(np.random.default_rng(5).random((4, 61)), ((0, 0), (0, 3)))
    permutee = serie.copy()
    permutee[:, :61] = serie[:, :61][:, ::-1]
    np.testing.assert_allclose(audio.agreger(serie), audio.agreger(permutee), rtol=1e-6)


@pytest.mark.parametrize("options", [{"taux": 16000}, {"n": 7999}, {"canaux": 2}])
def test_format_wav_incompatible_refuse(options):
    with pytest.raises(audio.PredictionError) as erreur:
        audio.exemple_audio(wav_synthetique(**options))
    assert erreur.value.code == "unsupported_audio"


def test_silence_et_troncature_refuses():
    with pytest.raises(audio.PredictionError) as erreur:
        audio.exemple_audio(wav_synthetique(frequence=0))
    assert erreur.value.code == "silent_audio"
    with pytest.raises(audio.PredictionError):
        audio.exemple_audio(wav_synthetique()[:-2])


def test_wav_emploie_exactement_dsp_icham_et_pas_metadonnees():
    contenu = wav_synthetique()
    a = audio.exemple_audio(contenu, "identifiant-a")
    b = audio.exemple_audio(contenu, "identifiant-b")
    assert a["features"] == b["features"]
    assert a["features"] == audio.agreger(preprocess_audio(audio.preprocessing.decode_wav(contenu), 8000))
    assert len(a["features"]) == 20
    assert set(a) == {"sample_id", "input_sha256", "features", "feature_names", "feature_version", "preprocessing_version", "execution_mode"}


def test_split_gel_reel_et_audit_intacts(tmp_path):
    for nom in nevil.MANIFEST_HASHES:
        shutil.copy(RACINE / "manifests" / nom, tmp_path / nom)
    split = nevil.charger_split(tmp_path)
    assert [len(split.fold(fold)) for fold in ("train", "val", "test")] == [598, 208, 194]
    with (tmp_path / "split_v2_audit.csv").open("a") as fichier:
        fichier.write("\n")
    with pytest.raises(ValueError, match="gelé modifié"):
        nevil.charger_split(tmp_path)


def test_groupes_chevauchants_refuses_avant_features(monkeypatch):
    split = nevil.charger_split(RACINE / "manifests")
    train = split.fold("train")[0]
    val = split.fold("val")[0]
    corrompu = replace(split, clips=tuple(replace(c, group_id=train.group_id) if c == val else c for c in split.clips))
    monkeypatch.setattr(nevil, "load_split", lambda _: corrompu)
    with pytest.raises(ValueError, match="Intersection"):
        nevil.charger_split(RACINE / "manifests")


def test_cache_ids_ordonnees_par_identite_et_non_position(tmp_path):
    split = Split((Clip("a", 0, "no_leak", "g1", "train"), Clip("b", 1, "leak", "g2", "train")), "hash", tmp_path, {})
    series = np.zeros((2, 4, 64))
    series[0, :, :61] = 2
    np.savez(tmp_path / "train.npz", ids=np.array(["b", "a"]), series=series, preprocessing_version=nevil.VERSION)
    cache = nevil.lire_cache(tmp_path, "train", split)
    assert cache["b"][0, 0] == 2 and cache["a"][0, 0] == 0
    np.savez(tmp_path / "train.npz", ids=np.array(["a", "a"]), series=series, preprocessing_version=nevil.VERSION)
    with pytest.raises(ValueError, match="IDs/fold"):
        nevil.lire_cache(tmp_path, "train", split)


def test_preparation_ne_lit_ni_audio_ni_cache_test(tmp_path, monkeypatch):
    # Fixture de plomberie uniquement : aucune exécution de train(), aucun score publié.
    clips = (Clip("a", 0, "no_leak", "g1", "train"), Clip("b", 1, "leak", "g2", "val"), Clip("c", 1, "leak", "g3", "test"))
    split = Split(clips, "f" * 64, tmp_path, {c.clip_id: c.clip_id + ".wav" for c in clips})
    monkeypatch.setattr(nevil, "charger_split", lambda _: split)
    import hashlib
    monkeypatch.setattr(nevil, "hashes_audio", lambda _: {
        c.clip_id: hashlib.md5((tmp_path / (c.clip_id + ".wav")).read_bytes()).hexdigest()
        if c.fold != "test" else "0" * 32 for c in clips})
    ecrire_json(tmp_path / "preparation.json", {"protocol": nevil.PROTOCOL, "manifest_sha256": split.sha256,
                 "preprocessing_version": nevil.VERSION, "records": 1000})
    appels = []
    def cache(_, fold, __):
        appels.append(fold)
        assert fold != "test"
        return {c.clip_id: preprocess_audio(audio.preprocessing.decode_wav((tmp_path / (c.clip_id + ".wav")).read_bytes()), 8000)
                for c in split.fold(fold)}
    monkeypatch.setattr(nevil, "lire_cache", cache)
    for i, clip in enumerate(clips[:2]):
        (tmp_path / (clip.clip_id + ".wav")).write_bytes(wav_synthetique(500 + i * 500))
    for fold in ("train", "val"):
        (tmp_path / f"{fold}.npz").write_bytes(b"synthetic fixture marker")
    nevil.preparer(tmp_path, tmp_path, tmp_path, tmp_path / "sortie")
    assert appels == ["train", "val"]
    donnees = lire_json(tmp_path / "sortie/features.json")
    assert {s["sample_id"] for s in donnees["samples"]} == {"a", "b"}
    assert len(lire_json(tmp_path / "sortie/split.json")["assignments"]) == 3
    assert not (tmp_path / "c.wav").exists()
