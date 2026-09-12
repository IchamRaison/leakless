"""Prédire depuis un artefact de confiance, dans un processus indépendant."""
import argparse

from .donnees import ecrire_json, lire_json
from .modele import charger_modele, predict_baseline


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--model", required=True)
    arguments.add_argument("--sha256", required=True)
    arguments.add_argument("--input", required=True)
    arguments.add_argument("--output", required=True)
    arguments.add_argument("--trusted-artifact", action="store_true")
    options = arguments.parse_args()
    try:
        artefact = charger_modele(options.model, options.sha256, artefact_de_confiance=options.trusted_artifact)
        prediction = predict_baseline(artefact, lire_json(options.input))
        ecrire_json(options.output, prediction, exclusif=True)
    except (OSError, ValueError, KeyError, TypeError) as erreur:
        arguments.exit(2, f"Prédiction refusée : {erreur}\n")


if __name__ == "__main__":
    main()
