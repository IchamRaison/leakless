# Intégration du harness pour l'évaluation qualité V1

Origine : branche Nevil, commit `6dfdf63580bc15ce6a1cf0817b9da3569553aca1`.
Cette note décrit le code prêt à exécuter ; aucune métrique du dataset réel n'a
été calculée pour réaliser cette intégration. Préenregistrer les choix avant
d'ouvrir les résultats test ; ne pas réentraîner le TSLM ni modifier T0–T3.

## Provenance et exceptions

Repris byte-identiques : `harness/features.py`, `harness/__init__.py`. Les fichiers de conformité
`contract.py`, `split_loader.py`, `check_run.py` et `stress.py` déjà présents
restent inchangés. Le package harness reprend maintenant son `__init__.py`
officiel, les modules auparavant manquants étant disponibles.

Trois familles d'exceptions déclarées, décidées avant le calcul réel :

1. **`harness/metrics.py`, uniquement `pr_auc` : correction des ex aequo.**
   L'ancien tri individuel dépendait de l'ordre des labels entre scores égaux.
   Exemple synthétique `scores=[0.5,0.5]` : labels `[1,0]` donnaient `1.0`, mais
   `[0,1]` donnaient `0.5`. L'oracle `sklearn.average_precision_score` donne `0.5`
   dans les deux cas. La correction regroupe les scores identiques avant
   l'intégration de précision moyenne. Le comportement `nan` pour une classe
   unique est conservé. ROC-AUC, seuil, F1, agrégation et bootstrap sont inchangés.
   SHA256 officiel : `f8a362079cc91554b4edcf7d763d0a81d5e1dcac93eddbe818bd07fa9a3f6efc` ;
   SHA256 corrigé : `7f0da97ddf86fc8bc976e81bae91289e7051f07792b82f4768a3971be8bd8518`.
   Ce changement de moteur qualité ne crée aucune configuration de modèle.
2. **`build_final_report.py` et `evaluate_predictions.py`, rédaction.** Suppression des conclusions
   prédéterminées sur C1/C0, les raccourcis et le caractère non concluant des
   comparaisons ; effectifs lus depuis le résultat. Les groupes restent décrits
   comme heuristiques, et les stress comme tests de sensibilité sans garantie
   de préservation de l'étiquette. L'avertissement des comparaisons dit aussi
   « groupes heuristiques », pas « clusters indépendants ». L'absence de rapport
   d'invariants n'est plus confondue avec l'absence de runs stressés. Aucun calcul
   n'est modifié par ces adaptations de rédaction. Le rapport ajoute une table
   T0 clip/groupe de précision, F1 fuite, rappel, FPR et taux de fuites manquées,
   dérivés uniquement des comptes TP/FP/TN/FN déjà calculés ; aucun nouveau seuil.
3. **`run_controls.py` et `build_final_report.py`, provenance CLI.** Option
   `--code-revision` validée comme SHA complet de 40 caractères hexadécimaux.
   Elle identifie le code effectivement transféré sans `.git`, notamment sur H100.
   Sans option, Git n'est interrogé que si la racine du code possède son propre
   `.git` ; une archive n'hérite jamais du SHA d'un dépôt parent. Aucun changement
   de modèle, hyperparamètre, seuil ou métrique dans ces scripts.

## Dépendances et vérifications synthétiques

- Évaluation/export du rapport : Python et NumPy, aucun TimeF, GPU ou checkpoint.
- Production des contrôles si leurs prédictions manquent : scikit-learn et WAV
  bruts, sans TimeNet/TimeF. Le scaler est ajusté sur train ; `C` est choisi sur
  validation parmi la grille officielle, puis le même modèle prédit les stress.
- Tests : `python3 -m unittest discover -s tests/eval -p test_harness_quality.py -v`.
  Sept tests réussissent avec NumPy `2.4.4` / scikit-learn `1.7.2` ; sans sklearn,
  seul l'oracle facultatif est ignoré, le test exact des ex aequo reste exécuté.
  Aucun exemple réel, aucune étiquette réelle, aucune métrique TSLM n'y intervient.

## Commandes après préenregistrement

Adapter `WAV_ROOT`, `BASELINES` et `QUALITY_OUT` à des chemins préparés, et
`CODE_REVISION` au SHA complet du code réellement transféré. Ne pas
écraser les anciens runs/rapports : ces commandes officielles n'ont pas toutes
une protection contre un dossier de sortie préexistant.

```bash
# Seulement si les prédictions comparatrices authentiques sont absentes.
python scripts/eval/run_controls.py --data-root WAV_ROOT --manifests manifests \
  --runs-dir BASELINES --controls C0 C1 C2 C2b C3 --code-revision CODE_REVISION

python scripts/eval/build_final_report.py --manifests manifests \
  --runs BASELINES/c0 BASELINES/c1 BASELINES/c2 BASELINES/c2b BASELINES/c3 \
    artifacts/tslm_runs/tslm-v1 artifacts/tslm_runs/tslm-v1-T1 \
    artifacts/tslm_runs/tslm-v1-T2 artifacts/tslm_runs/tslm-v1-T3 \
  --tslm-run-id tslm-v1 --out QUALITY_OUT --code-revision CODE_REVISION
```

Le rapport compare T0 à chaque autre run fourni. Sans `--tslm-run-id`, plusieurs
runs hors contrôles font échouer l'auto-détection. `--stress-report` est facultatif
et désigne uniquement un rapport d'invariants traçable, pas des prédictions.

Le moteur recalcule **un seuil sur validation par run**, y compris chaque stress.
Pour une analyse séparée à seuil T0 constant, lire la pleine précision dans
`metrics.json → runs.tslm-v1.folds.val.threshold` ; le champ voisin
`threshold_recomputed_on_val` est arrondi à six décimales. Ne pas présenter les
métriques seuillées officielles comme utilisant toutes le seuil T0.

Les intervalles et différences appariées couvrent ROC-AUC et macro-F1 uniquement
(clip/groupe), 2000 tirages de groupes, graine `20260912`. AP/Brier/rappel/FPR
n'ont pas d'intervalle ajouté par ce moteur. Les seuils ne sont pas réajustés
à chaque tirage ; ce bootstrap ne mesure pas l'incertitude d'entraînement.
