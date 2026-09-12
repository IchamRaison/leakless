"""Entraînement déterministe et interface de prédiction sans label en entrée."""
from __future__ import annotations

import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from .donnees import empreinte, ecrire_json, exiger, verifier_features, verifier_hash, verifier_matrice, verifier_texte


def versions():
    return {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__}


def entrainer(matrice, cibles, partitions, configuration):
    modele = RandomForestClassifier(random_state=configuration["seed"], n_jobs=1, **configuration["random_forest"])
    masque = partitions == "train"
    modele.fit(matrice[masque], cibles[masque])
    return modele


def metriques(cibles, predictions):
    exiger(len(cibles) == len(predictions) and len(cibles) > 0, "Prédictions incomplètes")
    exiger(set(cibles) <= {0, 1} and set(predictions) <= {0, 1}, "Classes de scoring invalides")
    matrice = confusion_matrix(cibles, predictions, labels=[0, 1])
    avertissements = []
    if set(cibles) != {0, 1}:
        avertissements.append("Une classe est absente de la validation ; métriques à interpréter avec Nevil.")
    if 1 not in predictions:
        avertissements.append("Aucune fuite prédite ; précision fixée à zéro par convention.")
    return {"label_order": ["no_leak", "leak"], "confusion_matrix": matrice.tolist(),
            "n_samples": len(cibles), "support": {"no_leak": int(sum(cibles == 0)), "leak": int(sum(cibles == 1))},
            "macro_f1": float(f1_score(cibles, predictions, labels=[0, 1], average="macro", zero_division=0)),
            "leak_precision": float(precision_score(cibles, predictions, zero_division=0)),
            "leak_recall": float(recall_score(cibles, predictions, zero_division=0)),
            "false_positive_rate": float(matrice[0, 1] / matrice[0].sum()) if matrice[0].sum() else None,
            "warnings": avertissements}


def sauvegarder(modele, metadonnees, dossier):
    """Un dossier neuf empêche d'écraser un run précédent."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=False)
    fichier = dossier / "model.joblib"
    joblib.dump({"model": modele, "metadata": metadonnees}, fichier)
    hachage = empreinte(fichier)
    ecrire_json(dossier / "metadata.json", {**metadonnees, "model_sha256": hachage})
    return hachage


def charger_modele(chemin, sha256_attendu, *, artefact_de_confiance=False):
    # Un checksum atteste l'intégrité, jamais la confiance dans l'auteur du pickle.
    exiger(artefact_de_confiance, "joblib exige un artefact de confiance explicitement déclaré")
    verifier_hash(sha256_attendu)
    exiger(empreinte(chemin) == sha256_attendu, "Checksum du modèle incorrect, chargement refusé")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            contenu = joblib.load(chemin)
    except InconsistentVersionWarning as erreur:
        raise ValueError(f"Version sklearn incompatible : {erreur}") from erreur
    environnement = contenu["metadata"]["environment"]
    for nom, version in versions().items():
        attendue = environnement.get(nom)
        compatible = attendue == version
        if nom == "python" and isinstance(attendue, str):
            compatible = attendue.split(".")[:2] == version.split(".")[:2]
        exiger(compatible, f"Version incompatible pour {nom} : artefact={attendue}, environnement={version}")
    return contenu


def predict_baseline(artefact, exemple):
    debut = time.perf_counter()
    attendus = {"sample_id", "input_sha256", "features", "feature_names", "feature_version", "preprocessing_version", "execution_mode"}
    exiger(set(exemple) == attendus, "Entrée d'inférence invalide : aucun label, ID de groupe ou champ supplémentaire accepté")
    verifier_texte(exemple["sample_id"], "sample_id")
    verifier_hash(exemple["input_sha256"])
    verifier_features(exemple["feature_names"])
    exiger(exemple["execution_mode"] in {"live", "development_fixture"}, "Mode d'inférence invalide")
    metadonnees, modele = artefact["metadata"], artefact["model"]
    for cle in ("feature_names", "feature_version", "preprocessing_version"):
        exiger(exemple[cle] == metadonnees[cle], f"{cle} incompatible")
    matrice = verifier_matrice([exemple["features"]], len(metadonnees["feature_names"]))
    classes = list(modele.classes_)
    exiger(len(classes) == 2 and set(classes) == {0, 1}, "Classes du modèle incompatibles")
    probabilites = modele.predict_proba(matrice)[0]
    predictions = modele.predict(matrice)
    exiger(len(probabilites) == 2 and np.isfinite(probabilites).all(), "Scores invalides")
    exiger(bool(((probabilites >= 0) & (probabilites <= 1)).all()) and np.isclose(sum(probabilites), 1), "Distribution des scores invalide")
    exiger(len(predictions) == 1 and predictions[0] in (0, 1), "Prédiction invalide")
    noms = {0: "no_leak", 1: "leak"}
    seuil = metadonnees.get("abstention_threshold", 0.5)
    exiger(type(seuil) in (float, int) and np.isfinite(seuil) and 0.5 <= seuil <= 1, "Seuil d'abstention invalide dans l'artefact")
    abstention = bool(max(probabilites) < seuil)
    mode = "development_fixture" if "development_fixture" in (metadonnees["execution_mode"], exemple["execution_mode"]) else "live"
    return {"schema_version": "0.1", "sample_id": exemple["sample_id"], "input_sha256": exemple["input_sha256"],
            "model_name": "baseline", "model_version": metadonnees["model_version"],
            "preprocessing_version": metadonnees["preprocessing_version"], "prediction": None if abstention else noms[int(predictions[0])],
            "class_scores": {noms[int(classe)]: float(probabilites[indice]) for indice, classe in enumerate(classes)},
            "score_type": "raw", "abstained": abstention, "abstention_reason": "max_class_score_below_validation_threshold" if abstention else None, "observations": [], "description": None,
            "latency_ms": (time.perf_counter() - debut) * 1000,
            "warnings": ["Fixture synthétique : aucune performance PIPE mesurée."] if mode == "development_fixture" else [],
            "execution_mode": mode}


def exemple_inference(donnees, ligne):
    return {**{cle: donnees[cle] for cle in ("feature_names", "feature_version", "preprocessing_version", "execution_mode")},
            **{cle: ligne[cle] for cle in ("sample_id", "input_sha256", "features")}}
