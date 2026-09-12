# Évaluation qualité V1 - exécution

Icham

## État courant

Les six étapes de [[Protocole évaluation]] ont été exécutées : **évaluation CPU des neuf runs et audit texte 402/402 terminés, code retour 0 ; revue indépendante et publication de la synthèse en cours.** Code/protocole publié avant résultats `c27a43fdd4de73ccc09f5f4bc02acd87c89f58f6`, branche isolée `feat/icham-quality-eval`, worktree `/home/animus/ehl-hackathon-zurich-quality-eval`, base `b4c8059`. Les chantiers ML et simulateur existants sont préservés.

## Résultats mesurés — modèle inchangé

T0 : **25 fuites manquées sur 98 (25,5 %) et 43 fausses alertes sur 96 non-fuites (44,8 %)**. Précision fuite 62,9 %, rappel 74,5 %, F1 fuite 68,2 %, macro-F1 0,646, AP 0,637 et Brier brut 0,241. AUC clip 0,665 [0,577 ; 0,889]. Après médiane par groupe : AUC 0,861 [0,706 ; 0,977], 5/30 groupes fuite manqués, 2/11 non-fuite faussement alertés. Les 41 groupes restent heuristiques. Seuil T0 choisi sur validation uniquement : `0.46893701143635425`.

C1 atteint 0,902 d'AUC clip et 0,927 par groupe. Différence T0−C1 : −0,237 [−0,334 ; −0,001] par clip ; −0,067 [−0,180 ; +0,021] par groupe, **pas de gain TSLM démontré** sur le critère principal groupé. Les cinq contrôles et huit comparaisons appariées sont dans `docs/evidence/quality-v1/report/`. IC bootstrap 2 000 tirages, sans revendication de significativité ni preuve terrain.

Stress, AUC clip/groupe : T1 `0,652/0,824`, T2 `0,612/0,700`, T3 `0,541/0,536`. T2/T3 dégradent la séparation ; sensibilité ne signifie pas utilité temporelle démontrée. Chaque run conserve son propre seuil choisi sur sa validation transformée, pas une robustesse au seuil T0 fixe.

Textes : validation **202/208 bandes correctes (97,1 %), 27/208 désaccords classe/score** ; test **184/194 bandes correctes (94,8 %), 36/194 désaccords (18,6 %)**. Format valide partout, aucune erreur d'exécution ni abstention ; dix bandes test erronées conservées. Latence texte test après chauffe : médiane 705 ms, p95 819 ms sur H100, pas un benchmark de flux ni du score seul. Audit brut `text-audit/raw.jsonl`, SHA `eb7c25ee…` ; chargement 9,23 s, une chauffe exclue. Le texte décrit une bande parmi quatre, pas une cause physique.

**Lecture provisoire : prototype fonctionnel, mais détecteur trop faible et texte pas toujours cohérent avec le score pour promettre une surveillance fiable.** Aucun poids/prompt/score modifié après ces résultats. Écart entre l'ancien diagnostic validation de sélection et l'export T0 en cours d'explication en lecture seule ; il ne justifiera pas une resélection.

## Règles et preuves avant résultats

- `docs/evidence/quality-v1/preregistration.json` fige protocole, population, comparateurs, seuil/agrégation/bootstrap, audit des 402 textes et empreintes avant évaluation réelle.
- Les huit SHA locaux des fichiers T0–T3 restent identiques à leur publication. Split `7a8716a3…`, audit mapping `1a3bd3c1…`. Bundle H100 vérifié : `checksums.json` SHA `b95569c5…`, poids temporels `183a1b1c…` inchangés.
- Runtime distant existant : Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.1, PyTorch 2.8.0+cu128. GPU observé à 0 Mio / 0 % au départ, aucune nouvelle machine/dépendance GPU.
- Recherche Entire retrouve les commits du harness dont `6dfdf63`, pas de transcription consultée. Lecture du code réel : seuil pleine précision disponible dans le bloc `folds.val.threshold` ; l'affichage à six décimales ne doit pas piloter l'audit texte.

## Correction prospective de précision moyenne

Avant ouverture des résultats TSLM, un test jouet démontre une dépendance indue de `pr_auc` à l'ordre des lignes en cas de scores égaux : `[0.5,0.5]` donne 1.0 avec labels `[1,0]` et 0.5 avec `[0,1]`, alors que la précision moyenne standard vaut 0.5 dans les deux cas. Correction minimale implémentée/testée : agréger les scores égaux en un même seuil avant intégration. ROC-AUC, seuil, macro-F1 et bootstrap restent inchangés. Aucun choix motivé par une performance réelle ; provenance différente du fichier amont explicitement tracée. Les phrases de conclusion codées en dur du rapport ont été neutralisées, les scripts acceptent le SHA explicite du code archivé sans `.git`, et une table dérive précision/F1 fuite/taux de manque des comptes existants. Détails : `docs/HARNESS_QUALITY_INTEGRATION.md`.

## Vérifications réalisées avant calcul

- Code `c27a43fd` poussé, SHA distant confirmé, puis archivé sur H100 dans `/home/hicham/pipe-v0/code-quality-c27a43fd`.
- Runtime cible : **7 tests harness + 25 tests TSLM/audit = 32 réussis**, aucun ignoré. Logs `docs/evidence/quality-v1/tests-eval.log` et `tests-tslm.log`.
- Les 1 000 WAV passent le contrôle MD5 attendu et le format exact mono/PCM16/8 kHz/8 000 échantillons ; huit fichiers de run et leurs couvertures 402/402 vérifiés. Preuve `docs/evidence/quality-v1/input-checks.json` ; SHA préinscription `3c2dc7d32e5fc8a51cdfc59a0f72ea2aa76d4b521dadbf45a85623ae2d2d717c`.
- Le vérificateur indépendant du split repasse sur les WAV originaux, zéro violation : 598 train (294 fuite/304 non-fuite), 208 val (108/100), 194 test (98/96). Ses groupes restent heuristiques, pas des sessions indépendantes prouvées.
- Revue indépendante du script texte : ajout de la vérification MD5 WAV avant DSP/inférence, cas substitué conservé comme erreur sans génération ; seuil plein recontrôlé sur validation uniquement. Sources du modèle et CSV TSLM inchangés.

## Exécution terminée

Sous `/home/hicham/pipe-v0/quality-v1-001` : contrôles `controls/c0`, `c1`, `c2`, `c2b`, `c3` générés avec `run_controls.py`, grille officielle, processus terminé code 0. C retenus respectivement `0.01`, `0.01`, `0.1`, `0.1`, `0.01` sur validation ; aucune nouvelle recherche TSLM. Log `controls.log`.

`build_final_report.py` terminé sur ces cinq contrôles + les quatre dossiers TSLM publiés, `--tslm-run-id tslm-v1 --code-revision c27a43fdd4de73ccc09f5f4bc02acd87c89f58f6`. Sortie `report/`, log `evaluation.log`, ancien handle root `55175` terminé.

`audit_text.py --checkpoint /home/hicham/pipe-v0/artifacts/qwen-v1-1199789f-001/bundle --data-root /home/hicham/pipe-v0/data/extracted --manifests manifests --run artifacts/tslm_runs/tslm-v1 --threshold-provenance /home/hicham/pipe-v0/quality-v1-001/report/metrics.json --output /home/hicham/pipe-v0/quality-v1-001/text-audit --code-revision c27a43fdd4de73ccc09f5f4bc02acd87c89f58f6` terminé, ancien handle `56401`. Exécution hors ligne avec `PYTHONPATH` pointant le code archivé `code-quality-c27a43fd/src` et `scripts/timenet`. Audit du 12 septembre 2026, 20:54:58–20:59:59 UTC. GPU revenu à 0 Mio / 0 %, instance laissée allumée. Ne relancer aucun de ces jobs pour reprendre la synthèse.

## Comparateurs disponibles

Contrôles C0/C1/C2/C2b/C3 : reproduction du pipeline existant, grille et descripteurs gelés ; pas de nouvelle recherche de modèle. TSLM T0/T1/T2/T3 déjà exportés, pas à régénérer.

RF Vincent : branche `feat/vincent-baseline-v1` à `fc837f7b05c4ab7dd8a9eabd420daf65b49db9d4`, 402 exports et compatibilité v2 documentés mais CSV/JSON absents des dépôts/worktrees accessibles. Artefacts annoncés sur son Mac sous `/Users/patrickjane/Projets/ehl-baseline-artifacts/verification-c1-c2/rf-v1-001/nevil/`, existence distante non vérifiée. Demande non bloquante faite à Icham pour les deux fichiers ; inclure seulement un export original vérifié, ne pas réentraîner sa RF ni contourner un contrôle de provenance.

## Prochaine action

Terminer la revue indépendante des nombres/provenances et rédiger/publier le verdict avec tous les artefacts. Pas de nouveau calcul GPU nécessaire. Toute future V2 demandera un protocole distinct ; ce test désormais consulté n'est plus vierge.
