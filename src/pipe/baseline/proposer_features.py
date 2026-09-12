"""Proposer l'ordre des agrégats, sans calculer de DSP ni décider du nombre de bandes."""
import argparse
import json

from .donnees import exiger, verifier_features


def proposer(nombre_bandes):
    exiger(type(nombre_bandes) is int and 1 <= nombre_bandes <= 256, "Nombre de bandes attendu entre 1 et 256")
    statistiques = ("mean", "std", "q25", "q50", "q75")
    noms = [f"band_{bande:02d}_{statistique}" for bande in range(nombre_bandes) for statistique in statistiques]
    noms.append("clip_log_rms")
    verifier_features(noms)
    return {"status": "proposal_not_approved", "band_count": nombre_bandes, "feature_names": noms,
            "band_aggregates": "Statistiques sur l'axe temporel des bandes du SignalExample partagé de Nevil.",
            "clip_log_rms": "Log-RMS du signal avant normalisation individuelle ; échelle et epsilon à documenter par Nevil.",
            "feature_version": "A_VALIDER_PAR_NEVIL", "preprocessing_version": "A_VALIDER_PAR_NEVIL"}


if __name__ == "__main__":
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--bands", type=int, required=True)
    options = arguments.parse_args()
    print(json.dumps(proposer(options.bands), ensure_ascii=False, indent=2))
