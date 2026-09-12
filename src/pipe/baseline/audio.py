"""Interface WAV CPU : même DSP qu'Icham, agrégats sans ordre temporel."""
import hashlib
import time
from pathlib import Path

import numpy as np
from leakless_acoustic import connector

from pipe.contracts import Prediction
from pipe.tslm import preprocessing
from .donnees import empreinte, exiger, verifier_matrice
from .modele import charger_modele, predict_baseline

STATISTIQUES = ("mean", "std", "q25", "q50", "q75")
FEATURE_NAMES = [f"band_{bande:02d}_{nom}" for bande in range(4) for nom in STATISTIQUES]
FEATURE_VERSION = "rf-band20-log1p-61-v1"


def sources_partagees():
    return {"preprocessing.py": empreinte(preprocessing.__file__),
            "connector.py": empreinte(connector.__file__)}


def version_features():
    # Un changement du DSP invalide le modèle, même si VERSION n'a pas été incrémentée.
    contenu = FEATURE_VERSION + "".join(sources_partagees().values())
    return FEATURE_VERSION + ":" + hashlib.sha256(contenu.encode()).hexdigest()


def agreger(series):
    series = np.asarray(series)
    exiger(series.shape == (4, 64) and np.isfinite(series).all()
           and (series >= 0).all(), "Séries finies non négatives (4,64) requises")
    exiger(np.all(series[:, 61:] == 0), "Padding des trois derniers pas non nul")
    # Les trois pas artificiels sont exclus ; statistiques sur l'échelle log1p commune.
    reel = series[:, :61].astype(np.float64)
    quantiles = np.quantile(reel, [0.25, 0.5, 0.75], axis=1, method="linear")
    matrice = np.column_stack((reel.mean(axis=1), reel.std(axis=1), quantiles.T))
    return verifier_matrice([matrice.ravel().tolist()], len(FEATURE_NAMES))[0].tolist()


class PredictionError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _exemple_audio(contenu, sample_id, version):
    try:
        signal = preprocessing.decode_wav(contenu)
        if np.std(signal) == 0:
            raise PredictionError("silent_audio", "Signal constant, aucune énergie acoustique exploitable")
        series = preprocessing.preprocess_audio(signal, 8000)
    except ValueError as erreur:
        raise PredictionError("unsupported_audio", str(erreur)) from erreur
    hachage = hashlib.sha256(contenu).hexdigest()
    return {"sample_id": sample_id or hachage[:24], "input_sha256": hachage,
            "features": agreger(series), "feature_names": FEATURE_NAMES,
            "feature_version": version, "preprocessing_version": preprocessing.VERSION,
            "execution_mode": "live"}


def exemple_audio(contenu, sample_id=None):
    return _exemple_audio(contenu, sample_id, version_features())


class Predictor:
    """Charger une fois un artefact connu, puis appeler predict(bytes) sans label."""
    def __init__(self, checkpoint, sha256, *, artefact_de_confiance=False):
        try:
            self.artefact = charger_modele(Path(checkpoint) / "model.joblib", sha256,
                                          artefact_de_confiance=artefact_de_confiance)
            meta = self.artefact["metadata"]
            exiger(meta["execution_mode"] == "live", "Modèle synthétique interdit dans le callable réel")
            exiger(meta["feature_version"] == version_features(), "DSP incompatible avec le modèle")
            for nom in ("audio.py", "modele.py", "donnees.py"):
                exiger(meta["source_files"][nom] == empreinte(Path(__file__).parent / nom),
                       f"Agrégation/inférence incompatible : {nom}")
            self.version_features = meta["feature_version"]
        except (OSError, ValueError, KeyError, TypeError) as erreur:
            raise PredictionError("model_unavailable", str(erreur)) from erreur

    def predict(self, wav_bytes, *, sample_id=None):
        debut = time.perf_counter()
        exemple = _exemple_audio(wav_bytes, sample_id, self.version_features)
        resultat = predict_baseline(self.artefact, exemple)
        resultat["latency_ms"] = (time.perf_counter() - debut) * 1000
        resultat["warnings"] = ["Données expérimentales ; comparaison finale à valider par Nevil.",
                                "Scores Random Forest non calibrés ; aucune localisation ni cause physique."]
        return Prediction.model_validate(resultat)
