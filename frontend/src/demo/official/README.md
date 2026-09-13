# Résultat TSLM officiel — zone de dépôt

Vide tant que CC2 n'a pas annoncé CONTRACT PASS / PROVENANCE PASS / FINAL REPORT GENERATED.
Tant que ce dossier ne contient pas les trois fichiers ci-dessous, valides et cohérents,
la démo affiche `TSLM · NOT EVALUATED YET` et C4 reste sans score.

| fichier           | origine                                                                                                        |
| ----------------- | -------------------------------------------------------------------------------------------------------------- |
| `metrics.json`    | copie **octet pour octet** de `artifacts/final_evaluation/metrics.json` (générateur gelé `protocol-freeze-v1`) |
| `comparison.json` | copie octet pour octet de `artifacts/final_evaluation/comparison.json`                                         |
| `receipt.json`    | reçu d'intégration, sans aucun score (schéma : `frontend/src/demo/official/INTEGRATION.md`)                    |

Ne jamais déposer ici : évaluation qualité d'Icham, scores V1 non officiels, scores V2 de développement,
sorties de modèle en cache, probabilités d'exemple, fixture synthétique.
