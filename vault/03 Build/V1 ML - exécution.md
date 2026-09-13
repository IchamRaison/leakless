# V1 ML - exécution

Icham

## État — 2026-09-12

**Étapes 1–5 réalisées : V1 figée/rechargée, T0–T3 conformes et publiés.** Modèle retenu `pipe-qwen3.5-4b-v1-1199789f-e4`, trois candidats réellement comparés, sélection sur validation seulement. Entraînement `1199789f471acf79f7e82604e2a15acd45e0b2f7`, exporteur renforcé `5a29d6eb9ceca39fdce829eb22dbbfd67b25589b`, branche `feat/icham-tslm`. Les 19 tests CPU du code d'export passent ; modèle/preprocessing/scoring/stress identiques entre ces révisions. V0 conservée intacte, [[V0 ML - exécution]]. Guide code `docs/TSLM_V1.md`, liens vers les huit fichiers de livraison. Aucune métrique finale calculée par Icham.

SSH revérifié : H100 80 Go disponible, 0 Mio utilisés au contrôle initial. Données originales, TimeF/cache et bundle V0 présents sous `/home/hicham/pipe-v0`. Environnement reconstruit `.venv-repro` : PyTorch 2.8.0+cu128, NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.1, CUDA disponible. Pas de nouvelle machine ni dépendance à provisionner.

## Campagne annoncée avant résultats

- Qwen 3.5-4B gelé ; encodeur/projecteur initialisés à neuf puis adaptés. Même TimeNet et preprocessing V0, mêmes manifestes v2.
- Un seul parcours d'entraînement sur les 598 clips train, ordre remélangé à chaque époque. Graine `20260912`, batch effectif 8, LR encodeur `0.0002`, projecteur `0.0001`, weight decay `0.01`, clipping gradient `1.0`.
- **Trois seuls candidats** : checkpoints aux fins d'époques 2, 4 et 8. Ce sont trois durées d'une trajectoire commune, pas trois recherches indépendantes ni un réglage adaptatif après chaque résultat.
- Sélection : ROC-AUC de développement sur les 208 clips validation, après médiane par groupe ; égalité → époque la plus précoce. Les 42 groupes, dont 8 non-fuite, restent heuristiques. `n_configs_compared` sera le nombre réellement comparé, trois seulement si la campagne va au bout.
- Descriptions : diagnostic fixe sur huit clips validation (quatre groupes par classe), validité/fidélité à la mesure DSP ; pas de critère supplémentaire de sélection. Temps/mémoire et reproductibilité suivis séparément.
- Aucun accès au cache test pendant entraînement/sélection ; aucun moteur final de métriques lancé. Ni LoRA ni nouvelle représentation dans cette campagne annoncée.

## Étape 1 vérifiée — score continu

- Code `4456c3168f22b17d81bbb0bc6c2a91e34e14bb30`, transféré dans `/home/hicham/pipe-v0/code-v1-score-4456c316`. Huit tests CPU passent dans `.venv-repro`, dont calcul analytique, longueurs inégales, padding, décalage causal et interdiction des métadonnées.
- `scripts/tslm/check_scoring.py` exécuté hors ligne sur le bundle V0 et les quatre premiers clips train triés uniquement. Rapport réel `/home/hicham/pipe-v0/runs/scoring-train-4456c316.json`, copié dans `docs/evidence/tslm-v1/scoring-train.json`.
- Tokenisation réelle : `leak;` → `[273,570,26]` (3 tokens), `no_leak;` → `[2083,11414,570,26]` (4 tokens). Préfixes identiques à ceux des réponses complètes d'entraînement ; sommes brutes, aucune description/EOS inclus.
- Sur ces quatre clips, scores WAV/brut flottant/séries/répétition/softmax manuel exactement identiques, tous finis, quatre valeurs distinctes, aucun 0/1. Écart maximal cache TimeF vs preprocessing direct : `1.1920928955078125e-07`. Ce sont des contrôles mécaniques train, pas une mesure de qualité ou une sélection de modèle.
- WAV invalide, signal constant et champ `clip_id` dans l'entrée modèle refusés. Test complet ~12,5 s, chargement inclus ; pas un benchmark de débit garanti.
- Code de campagne `1199789f` transféré dans `/home/hicham/pipe-v0/code-v1-1199789f`. SHA-256 locaux/distants de model/campaign/config/export/stress/lock comparés identiques ; aucun `.git` distant requis, révision explicitement passée aux commandes.

## Étape 2 vérifiée — entraînement réel

Commande réelle : depuis `/home/hicham/pipe-v0/code-v1-1199789f`, Python `.venv-repro`, `python -u -m pipe.tslm.campaign --config configs/tslm/v1.json --manifest manifests --prepared /home/hicham/pipe-v0/artifacts/prepared --data-root /home/hicham/pipe-v0/data/extracted --base /home/hicham/pipe-v0/artifacts/base-qwen3.5-4b --output /home/hicham/pipe-v0/artifacts/qwen-v1-1199789f-001 --code-revision 1199789f471acf79f7e82604e2a15acd45e0b2f7 --environment-lock requirements-ml.lock`, avec imports du code transféré et HF hors ligne.

Logs `/home/hicham/pipe-v0/runs/train-v1-1199789f-001.log` et dossier de campagne `training.jsonl`. Préinscription écrite avant chargement du modèle ; hash canonique config `41fa12bf21be6463ae11b45ea8b5010409b526bbba366f83c7ba27b7c3122f70`. **Processus terminé avec code 0**, GPU revenu à 0 Mio avant le reload. Ne pas relancer la campagne : les trois candidats et le bundle retenu existent.

600 étapes, huit époques, 2 441 472 paramètres temporels entraînables ; pertes/gradients finis. Le checkpoint époque 4 / étape 300 est sélectionné selon la règle annoncée, pas nécessairement le dernier. Norme de changement des poids sélectionnés : encodeur `3.143120288848877`, projecteur `0.9004685878753662`. SHA intégral Qwen gelé identique avant/après : `c1b469d8d93ec5ffebd6fdd102296f1b2689d5966bc4fe3077e2f699b887267e`. Durée observée ~605 s, pic alloué `53458143744` octets (~49,8 Gio). Pas de garantie de débit ni de résultat final.

Rapports `docs/evidence/tslm-v1/training-report.json`, `selection.json`, `preregistration.json`. Les valeurs de sélection sont des diagnostics de validation, pas des métriques finales sur test. Aucun WAV/cache test utilisé pendant la campagne.

Suite complète : 18 tests avant entraînement, puis **19 tests** sur l'exporteur renforcé dans `.venv-repro`, dont dérivation officielle des graines et processus distincts T2/T3. Preuves `docs/evidence/tslm-v1/cpu-tests.log` / `cpu-export-tests.log`. La revue indépendante a conduit à ajouter avant T0 le SHA du manifeste d'audit qui mappe les IDs aux WAV ; son fichier réel était correct. Test de refus d'un mapping modifié avant chargement GPU ; aucun changement d'entraînement ou de candidat.

## Étape 3 vérifiée — gel et reload

Bundle autonome `/home/hicham/pipe-v0/artifacts/qwen-v1-1199789f-001/bundle`. SHA `checksums.json` : `b95569c50f5e1bbf3533bddc92530b5f285eb25dec999f83999f60ffd912d1ae` ; SHA `temporal.pt` retenu : `183a1b1c29facf0e27f27efc7adab8c1bb13556cede06abc7f735672cbe7eb82`.

Après fin du processus train : `check_v1_reload.py` lancé dans un processus neuf hors ligne avec le code d'export `5a29d6eb`. Intégrité du bundle vérifiée, même scoring, score attendu à tolérance absolue `1e-6`, génération identique, voies WAV/brut/séries concordantes, mauvais WAV refusé. Rapport `/home/hicham/pipe-v0/runs/reload-v1-5a29d6eb.json`, copie `docs/evidence/tslm-v1/reload.json`. Vérification sur l'exemple de développement figé, pas une preuve de qualité sur tout le dataset.

## Scoring et contrat livrés

Continuation `leak;` ou `no_leak;`, log-probabilités conditionnelles sommées puis softmax à deux candidats. Tokenisation exacte vérifiée sur le modèle ; aucune moyenne par longueur, EOS ou explication dans le score, aucun seuil ou calibration. Diagnostics mécaniques initiaux seulement sur train. La fonction `predict()` Safoan reste compatible ; une voie score seul traite WAV ou signal brut flottant, sans génération longue ni requantification des stress.

Bundle final autonome avec modèle, paramètres temporels, configuration, scoring, provenance, empreintes et exemple validation ; reload dans un processus neuf réalisé avant test. Chaque export T0–T3 : seulement `metadata.json` + `predictions.csv`, exactement `clip_id,probability_leak`, 402 IDs val/test. Nevil reste seul propriétaire de l'évaluation finale.

## Étape 4 vérifiée — T0 conforme

Run réel `/home/hicham/pipe-v0/exports/tslm-v1` ; copie locale dans le dépôt ML `artifacts/tslm_runs/tslm-v1/`. Seulement `metadata.json` et `predictions.csv`, exactement `clip_id,probability_leak`, 402 IDs uniques couvrant 208 validation + 194 test. Contrôleur Nevil terminé avec code 0 ; preuve `docs/evidence/tslm-v1/export-T0.log`. Pas de métrique finale, pas de seuil appliqué. Son avertissement de plage de scores resserrée n'a entraîné ni calibration, ni ajustement, ni nouveau choix de modèle.

SHA-256 local/distant identiques : CSV `0c4dfb3f06c379fe48ce4f7456d38c67d165c80d5e083fbb3f350e0b2732d7b8`, métadonnées `0d8ed4919540987235a3fe98eb2d1cdd30fbe3482f9b2786eb50603f03fb795f`. Le checkpoint, la méthode, le split et le mapping des WAV sont identifiés par empreintes. Les WAV test sont lus uniquement pour inférence après gel/reload ; aucune sélection à partir du test.

**Publication vérifiée :** commit `186c45a3a8f3dd28777a9c1c333836044bc1af0b` poussé sur `feat/icham-tslm`, hash distant confirmé par `git ls-remote`. T0 disponible pour Nevil avant lancement de T1–T3 ; pas de message externe envoyé à sa place.

## Étape 5 vérifiée — stress officiels reproductibles et publiés

Référence récupérée : `6dfdf63580bc15ce6a1cf0817b9da3569553aca1` sur `nevil/temporal-evidence`. Nevil a corrigé la graine dans `b23601ab28c5b3949b2b17c46478daa46c847c0f` : SHA-256 stable du triplet graine/nom/clip. Source `scripts/temporal/stress.py` SHA-256 `7c37e001d270655d2e54ba2087e8b8e476da8cc42948f92bd83c669773ac6350`. La réserve antérieure sur le `hash()` Python est donc levée pour cette révision, pas pour les variantes anciennes.

T1/T2/T3 exécutés après publication T0 : transformations officielles au brut, preprocessing et checkpoint identiques, aucun réentraînement. T2 = 250 échantillons ; aucun reste hors permutation, sans garantie que tous les blocs changent de place. Dérivation de graine et sorties interprocessus T2/T3 vérifiées avant génération, NumPy 2.5.2 et empreintes déclarées.

Trois exports terminés avec code 0 depuis `code-v1-export-5a29d6eb`, même bundle/reload, dossiers `/home/hicham/pipe-v0/exports/tslm-v1-T1/`, `tslm-v1-T2/`, `tslm-v1-T3/`. Chacun passe `check_run.py --run … --inspect`, 402/402 clips (208 validation + 194 test). Logs copiés dans `docs/evidence/tslm-v1/export-v1-T1.log`, `export-v1-T2.log`, `export-v1-T3.log` ; copies des deux fichiers par run dans `artifacts/tslm_runs/`.

**Publication vérifiée :** `7c04c9a5636b2872334da17c54beb7016ffa00a3` poussé sur `feat/icham-tslm`, hash distant confirmé. SHA-256 des CSV, identiques local/H100 :

| Run | SHA-256 de predictions.csv |
| --- | --- |
| T1 | `d510328b660e4247372f2b0bb32c08bcde0f675d02d63f90576f3e563f5aae16` |
| T2 | `9ec9af9256e05a4f61a8a9dd8c184ce890565ca88688dd63fea0db1524d5e5df` |
| T3 | `65676758ee5109dcceb5e8828bbde1cd05cd72d72461c99d1082b17fc74c3fd5` |

Audit indépendant en lecture seule des quatre runs : exactement deux fichiers/deux colonnes, 402 IDs uniques attendus, probabilités finies dans [0,1] ; métadonnées identiques sauf run/transform/horodatage, neuf SHA de code concordants. SHA du bundle toujours `b95569c5…` après stress. Aucune comparaison des effets des stress, métrique finale, calibration ou retouche après inspection. GPU à 0 Mio / 0 % après terminaison de tous les processus ; instance laissée allumée, pas de service public déployé.

## Prochaine action concrète

Nevil peut récupérer les quatre dossiers publiés pour son pipeline final ; aucun message externe envoyé à sa place. Ne pas refaire cette campagne ni ces exports : les preuves sont livrées. Icham/Safoan peuvent maintenant convenir du raccordement de `Predictor` et du budget de calcul au flux/simulateur ; application et surveillance terrain non validées par cette livraison. Aucun réentraînement ni réglage fondé sur les scores test.
