# V1 ML - exécution

Icham

## État — 2026-09-12

Icham autorise explicitement l'implémentation des cinq étapes : scoring continu, campagne V1 limitée, gel/reload, export T0, puis T1–T3 reproductibles. Scoring implémenté et vérifié sur train ; campagne/reload/export implémentés et publiés à `1199789f471acf79f7e82604e2a15acd45e0b2f7` sur `feat/icham-tslm`. **Les 18 tests CPU passent et la campagne V1 est en cours sur H100 ; aucun export encore produit.** V0 conservée intacte, [[V0 ML - exécution]].

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

## Scoring et livraison attendus

## Étape 2 en cours — entraînement réel

Commande réelle : depuis `/home/hicham/pipe-v0/code-v1-1199789f`, Python `.venv-repro`, `python -u -m pipe.tslm.campaign --config configs/tslm/v1.json --manifest manifests --prepared /home/hicham/pipe-v0/artifacts/prepared --data-root /home/hicham/pipe-v0/data/extracted --base /home/hicham/pipe-v0/artifacts/base-qwen3.5-4b --output /home/hicham/pipe-v0/artifacts/qwen-v1-1199789f-001 --code-revision 1199789f471acf79f7e82604e2a15acd45e0b2f7 --environment-lock requirements-ml.lock`, avec imports du code transféré et HF hors ligne.

Logs `/home/hicham/pipe-v0/runs/train-v1-1199789f-001.log` et dossier de campagne `training.jsonl`. Préinscription écrite avant chargement du modèle ; hash canonique config `41fa12bf21be6463ae11b45ea8b5010409b526bbba366f83c7ba27b7c3122f70`. Processus Python distant observé PID `32101`, étapes/gradients finis et progression vérifiée ; la présence de ce PID doit être revérifiée à toute reprise, ne jamais relancer sur un simple délai d'observation.

Suite complète : 18 tests CPU réussis dans `.venv-repro`, dont dérivation officielle des graines et deux processus distincts pour T2/T3. Preuve `docs/evidence/tslm-v1/cpu-tests.log`. Une revue indépendante n'a trouvé aucune sélection fondée sur test ; ajout demandé avant export d'un contrôle d'empreinte du manifeste d'audit qui mappe les IDs aux WAV. Son fichier réel est déjà correct ; ce garde-fou d'export ne modifie ni entraînement ni candidats.

## Scoring et livraison attendus

Continuation `leak;` ou `no_leak;`, log-probabilités conditionnelles sommées puis softmax à deux candidats. Tokenisation exacte vérifiée sur le modèle ; aucune moyenne par longueur, EOS ou explication dans le score, aucun seuil ou calibration. Diagnostics mécaniques initiaux seulement sur train. La fonction `predict()` Safoan reste compatible ; une voie score seul traite WAV ou signal brut flottant, sans génération longue ni requantification des stress.

Bundle final autonome avec modèle, paramètres temporels, configuration, scoring, provenance, empreintes et exemple validation ; reload dans un processus neuf obligatoire avant test. Export T0 : seulement `metadata.json` + `predictions.csv`, exactement `clip_id,probability_leak`, 402 IDs val/test. Nevil reste seul propriétaire de l'évaluation finale.

## Stress officiels désormais reproductibles

Référence récupérée : `6dfdf63580bc15ce6a1cf0817b9da3569553aca1` sur `nevil/temporal-evidence`. Nevil a corrigé la graine dans `b23601ab28c5b3949b2b17c46478daa46c847c0f` : SHA-256 stable du triplet graine/nom/clip. Source `scripts/temporal/stress.py` SHA-256 `7c37e001d270655d2e54ba2087e8b8e476da8cc42948f92bd83c669773ac6350`. La réserve antérieure sur le `hash()` Python est donc levée pour cette révision, pas pour les variantes anciennes.

T1/T2/T3 après T0 : transformations officielles au brut, preprocessing et checkpoint identiques, aucun réentraînement. T2 = 250 échantillons ; aucun reste hors permutation, sans garantie que tous les blocs changent de place. Contrôler la graine publiée et des processus distincts avant génération ; noter version NumPy/empreintes. Aucun stress réel exécuté à ce jalon.

## Prochaine action concrète

Surveiller le processus existant jusqu'à fin des huit époques et des trois candidats, sans relance concurrente. Vérifier le bundle sélectionné, puis lancer le reload dans un processus neuf et livrer T0 avant les stress. Tous les exports passeront par le contrôleur officiel ; aucune métrique finale côté Icham. Mettre cette note à jour avec les résultats et chemins réellement produits, sans anticiper une qualité.
