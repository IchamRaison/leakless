"""Temporal Evidence Harness — le seul chemin d'évaluation du projet.

    split_v2.csv (gelé, SHA vérifié)
            │
            ▼
    split_loader   ← point de passage unique vers le manifeste et l'audio
            │
            ▼
    features (C0..C3) ──► run_controls ──┐
                                          │   runs/<run_id>/{metadata.json, predictions.csv}
    TSLM de Hicham ───────────────────────┤   (même contrat pour tout le monde)
                                          ▼
                               contract.load_run  (12 contrôles, couverture publiée)
                                          ▼
                            metrics + bootstrap apparié
                                          ▼
                              evaluate_predictions / build_final_report
"""

from . import contract, features, metrics, split_loader  # noqa: F401

__all__ = ["contract", "features", "metrics", "split_loader"]
