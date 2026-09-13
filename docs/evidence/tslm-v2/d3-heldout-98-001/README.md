# D3 — observation des 98 réservés de développement

Exécuté à la demande d'Icham, après le diagnostic de mémorisation et son reload neuf. Ce n'est pas le test officiel ni une nouvelle campagne d'entraînement.

- Code `141ff92`, préinscription publiée dans `84c1890` avant observation ; SHA `78e3b455f1f10e94d1faeced386071180da6f6727720b92e3ba4da7e47cd0eca`.
- Checkpoint D3 au pas400, entraîné uniquement sur32clips ; SHA `3e99f7aecad16b1cf531bbec1af9c20cfb61f631cc3f0204713d323499da8148`.
- Reçu complet `17f8af51d2a50b6c7bec6001fd613526d584e1ca803b8a994a23577dfd0d8ef8` dans `evaluation/complete.json`.
- 98 observations /196 forwards ; aucun optimiseur, réglage, génération ou accès audio/cache val/test/externe. Poids avant/après identiques.
- 98 clips de34groupes, disjoints des32groupes du fit :73fuites et25non-fuite (2vrais sans-fuite,23bruits).

## Résultats au seuil diagnostique fixe 0,5

67/98corrects :52fuites détectées,21manquées,15vrais négatifs et10fausses alertes. Rappel fuite71,23%, taux de fausses alertes40%, exactitude équilibrée65,62%.

AUC clip0,741370 ; AUC groupe0,798077. NLL binaire1,419433, Brier clip0,280533 : classement non trivial observé, mais probabilités brutes pénalisées par des erreurs confiantes. Pas de calibration ou sélection de seuil après lecture.

A0 sur les mêmes98 : AUC clip0,709589 /groupe0,759615. Comparaison descriptive seulement : A0 avait appris500clips pendant4époques, D3 seulement32pendant100époques. Aucun effet causal isolé du budget et aucune supériorité statistique établie.

## Vérifications exécutées

Self-test local et H100 du runner : couverture, ordre, fold et disjonction IDs/groupes. Après le run : reçu et toutes empreintes locales vérifiés avec `run_v2_campaign.finished`; 98scores/NLL reconstruits depuis les log-probabilités ; métriques recomposées avec `diagnose_causal.partition_summary` et strictement identiques ; CSV identique aux observations ; poids identiques au checkpoint D3 ; comptes98/196/zéro optimisation vérifiés. Aucun second passage modèle pour cette vérification.

Les98 proviennent de développement déjà consulté, les groupes sont heuristiques, et seuls2clips sont de vrais tuyaux sans fuite. Ce n'est pas une preuve de fiabilité terrain. Les32/32de D3 mesuraient uniquement la mémorisation.

Commande exécutée une fois sur H100-1 depuis `/home/hicham/pipe-v0/code-causal-d3-c08ff78` :

```bash
PYTHONPATH=src ../.venv-repro/bin/python scripts/tslm/evaluate_memorization_holdout.py run --output /home/hicham/pipe-v0/artifacts/d3-heldout-98-141ff92-001
```
