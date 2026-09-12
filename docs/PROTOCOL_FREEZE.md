# PROTOCOL_FREEZE — protocole d'évaluation gelé avant tout résultat TSLM final

> **Gel `protocol-freeze-v1`**, posé sur `nevil/temporal-evidence` à
> `5a3054cd5469783048ce90656b82bcd60a81bd98`, relu par `/review` (GREEN).
> Manifeste machine : [`protocol/FREEZE_v1.json`](../protocol/FREEZE_v1.json).
> Vérification : `python3 scripts/eval/verify_protocol_freeze.py --git --runs-dir <runs>`.

## Pourquoi maintenant

L'équipe a continué à entraîner et à pousser pendant la revue du protocole. Le
gel est posé **avant** d'inspecter ou d'intégrer le moindre nouveau résultat
TSLM, pour que l'expérience ne puisse pas être adaptée après avoir vu la
performance du modèle.

Au moment du gel, les branches distantes suivantes existaient. **Leur contenu n'a
pas été inspecté** ; leurs SHA sont consignés dans le manifeste pour dater le gel
par rapport à elles :

| branche | SHA |
|---|---|
| `feat/icham-tslm` | `b4c8059f997d` |
| `feat/icham-quality-eval` | `16a6df043771` |
| `feat/vincent-baseline-v1` | `fc837f7b05c4` |
| `feat/safoan-app` | `4dd7b8868b8e` |
| `feat/sensor-replay` | `81c987f58397` |
| `main` | `916986f245f7` |

> **Ce qui avait été vu avant le gel.** La V0 smoke-test de Hicham : 8 clips
> train, 40 steps, un exemple qualitatif de validation. C'était un test technique
> de la chaîne d'entraînement, sans `predictions.csv` final, sans benchmark
> held-out final et sans comparaison TSLM vs C1/C2b ; rien de ce qu'elle montrait
> n'a servi à adapter le protocole. Aucun résultat TSLM final n'avait été vu, et
> le contenu des HEAD ci-dessus ne l'a pas été. Manifeste :
> `prior_tslm_smoke_test_inspected: true`,
> `new_final_tslm_results_inspected_before_freeze: false`.

## Ce qui est gelé

| élément | où |
|---|---|
| split `split_v2.csv`, `sha256 7a8716a3…7d17896`, son audit, sa méta et son générateur | `manifests/`, `scripts/ingest/build_split_v2.py`, `scripts/eval/verify_split_invariants.py` |
| définitions C0 / C1 / C2 / C2b / C3 et leurs commits de définition | `scripts/eval/harness/features.py` |
| protocole d'ajustement des contrôles, grille de `C`, métadonnées de stress | `scripts/eval/run_controls.py` |
| discipline train / val / test, seuil sur val, refus du test | `scripts/eval/harness/metrics.py`, `evaluate_predictions.py` |
| agrégation par cluster de dépendance (médiane) | `scripts/eval/harness/metrics.py` |
| bootstrap apparié sur clusters (2000 tirages, graine 20260912) | `scripts/eval/harness/metrics.py` |
| T0 / T1 / T2 / T3 et dérivation SHA-256 des graines | `scripts/temporal/stress.py` |
| identité des jeux de stress T0-T3 | `artifacts/final_evaluation/stress_provenance.json`, `build_stress_timef.py`, `stress_provenance.py` |
| entrée du TSLM (normalisation, rattachement aux clusters) | `scripts/timenet/` |
| contrat de run de prédiction, colonnes interdites | `scripts/eval/harness/contract.py`, `schemas/prediction_run.schema.json`, `check_run.py`, `docs/MODEL_EVAL_CONTRACT.md` |
| règles de provenance et contrôles d'identité des runs de stress TSLM | `scripts/eval/build_final_report.py` |
| interprétation des stress et métriques indépendantes du seuil | `scripts/eval/build_final_report.py`, `docs/C2B_HYPOTHESIS.md` |
| procédure d'évaluation finale | `scripts/eval/build_final_report.py`, `docs/EVAL_PROTOCOL.md` |
| sentinelles et tests du protocole | `tests/run_tests.py` |
| prédictions et identité des 8 runs de contrôle publiés (hors dépôt) | `protocol/FREEZE_v1.json`, clé `frozen_control_runs` |

29 fichiers au total, chacun fixé par SHA-256. Les textes jury (`JURY_EVIDENCE_MATRIX.md`,
`README.md`, `CLAIM_LEDGER.md`) et les sorties régénérées du rapport
(`FINAL_EVALUATION.md`, `metrics.json`, `comparison.json`) ne sont **pas** gardés :
ils doivent pouvoir changer quand un résultat est mesuré, et le rapport est
produit par un générateur, lui, gardé.

## Autorisé après le gel

1. **Ingérer un run TSLM conforme** au contrat. Aucun fichier gardé ne change :
   le run vit hors dépôt, `check_run.py` le valide, `build_final_report.py`
   l'évalue avec les runs de contrôle gelés.
2. **Corriger un défaut d'implémentation démontré.** Uniquement par un
   amendement dans [`protocol/AMENDMENTS.json`](../protocol/AMENDMENTS.json) :

   ```json
   {
     "id": "A1",
     "date": "2026-09-13",
     "category": "implementation-bugfix",
     "paths": {"scripts/eval/harness/contract.py": {"from": "<sha256 avant>", "to": "<sha256 après>"}},
     "demonstrated_by": "test ou reproduction qui échoue avant le correctif",
     "justification": "en quoi c'est un défaut d'implémentation, pas un changement de protocole",
     "tslm_results_inspected_before": false
   }
   ```

   Le vérificateur refuse toute autre catégorie, une chaîne d'empreintes rompue et
   un amendement qui ne déclare pas si des résultats TSLM avaient été vus. Un
   amendement déclaré `true` reste possible mais doit être signalé dans le rapport
   jury.
3. **Mettre à jour les textes jury et UI** à partir des résultats nouvellement
   mesurés.

## Interdit après le gel

- modifier les contrôles à cause de la performance du TSLM ;
- modifier le split ;
- modifier les transformations de stress ;
- modifier les règles de bootstrap ;
- régler un seuil sur le test ;
- ajouter une nouvelle baseline après avoir vu les résultats TSLM ;
- affaiblir les exigences de provenance pour accepter un run.

Un run qui ne passe pas le contrat ou les contrôles d'identité est **renvoyé à
son auteur**, pas accommodé.

## Ce que le gel garantit, et ce qu'il ne garantit pas

Le vérificateur détecte toute modification d'un fichier gardé, d'une prédiction
de contrôle ou de l'identité d'un run gelé, et exige que chaque correction soit
déclarée et chaînée. Le tag Git `protocol-freeze-v1` fixe le manifeste lui-même.

Il ne peut pas juger une intention : un amendement classé `implementation-bugfix`
qui changerait en réalité le protocole passerait le script. La protection contre
ce cas est la relecture : chaque amendement est un commit séparé, avec sa preuve,
dans l'historique.
