"""Vérifier la livraison de Nevil sans entraîner de modèle."""
import argparse
import json
from pathlib import Path

import numpy as np

from .donnees import charger_developpement, lire_json
from .train import charger_configuration


def valider(chemin_configuration, chemin_donnees=None, chemin_split=None):
    chemin = Path(chemin_configuration).resolve()
    configuration = charger_configuration(chemin)
    donnees_path = Path(chemin_donnees).resolve() if chemin_donnees else chemin.parent / configuration["data_path"]
    split_path = Path(chemin_split).resolve() if chemin_split else chemin.parent / configuration["split_path"]
    donnees, matrice, cibles, partitions = charger_developpement(donnees_path, split_path, configuration)
    manifeste = lire_json(split_path)
    effectifs = {}
    for partition in ("train", "validation", "test", "quarantine"):
        lignes = [ligne for ligne in manifeste["assignments"] if ligne["split"] == partition]
        masque = partitions == partition
        effectifs[partition] = {"samples": len(lignes), "groups": len({ligne["event_group_id"] for ligne in lignes}),
                               "class_counts": {str(int(classe)): int(sum(cibles[masque] == classe)) for classe in np.unique(cibles[masque])} if partition in {"train", "validation"} else None}
    return {"status": "valid", "execution_mode": donnees["execution_mode"], "benchmark_eligible": False,
            "shape": list(matrice.shape), "feature_names_received": donnees["feature_names"],
            "feature_names_expected": configuration["feature_names"], "feature_version": donnees["feature_version"],
            "split_sha256": configuration["split_sha256"], "partitions": effectifs,
            "limitations": ["Validation du contrat uniquement, aucun audit physique des groupes ni label du test lu."]}


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--config", required=True)
    arguments.add_argument("--data")
    arguments.add_argument("--split")
    options = arguments.parse_args()
    try:
        rapport = valider(options.config, options.data, options.split)
    except (ValueError, OSError, TypeError, KeyError) as erreur:
        print(json.dumps({"status": "invalid", "error": str(erreur)}, ensure_ascii=False))
        raise SystemExit(1)
    print(json.dumps(rapport, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
