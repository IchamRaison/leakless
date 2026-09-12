# V1 ML - exécution

Icham

## État — 2026-09-12

Icham autorise explicitement l'implémentation des cinq étapes : scoring continu, campagne V1 limitée, gel/reload, export T0, puis T1–T3 reproductibles. Développement en cours sur `feat/icham-tslm`, worktree `/home/animus/ehl-hackathon-zurich-icham-tslm`. **Aucun entraînement V1 ni export encore exécuté à ce jalon.** V0 conservée intacte, [[V0 ML - exécution]].

SSH revérifié : H100 80 Go disponible, 0 Mio utilisés au contrôle initial. Données originales, TimeF/cache et bundle V0 présents sous `/home/hicham/pipe-v0`. Environnement reconstruit `.venv-repro` : PyTorch 2.8.0+cu128, NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.1, CUDA disponible. Pas de nouvelle machine ni dépendance à provisionner.

## Campagne annoncée avant résultats

- Qwen 3.5-4B gelé ; encodeur/projecteur initialisés à neuf puis adaptés. Même TimeNet et preprocessing V0, mêmes manifestes v2.
- Un seul parcours d'entraînement sur les 598 clips train, ordre remélangé à chaque époque. Graine `20260912`, batch effectif 8, LR encodeur `0.0002`, projecteur `0.0001`, weight decay `0.01`, clipping gradient `1.0`.
- **Trois seuls candidats** : checkpoints aux fins d'époques 2, 4 et 8. Ce sont trois durées d'une trajectoire commune, pas trois recherches indépendantes ni un réglage adaptatif après chaque résultat.
- Sélection : ROC-AUC de développement sur les 208 clips validation, après médiane par groupe ; égalité → époque la plus précoce. Les 42 groupes, dont 8 non-fuite, restent heuristiques. `n_configs_compared` sera le nombre réellement comparé, trois seulement si la campagne va au bout.
- Descriptions : diagnostic fixe sur huit clips validation (quatre groupes par classe), validité/fidélité à la mesure DSP ; pas de critère supplémentaire de sélection. Temps/mémoire et reproductibilité suivis séparément.
- Aucun accès au cache test pendant entraînement/sélection ; aucun moteur final de métriques lancé. Ni LoRA ni nouvelle représentation dans cette campagne annoncée.

## Scoring et livraison attendus

Continuation `leak;` ou `no_leak;`, log-probabilités conditionnelles sommées puis softmax à deux candidats. Tokenisation exacte vérifiée sur le modèle ; aucune moyenne par longueur, EOS ou explication dans le score, aucun seuil ou calibration. Diagnostics mécaniques initiaux seulement sur train. La fonction `predict()` Safoan reste compatible ; une voie score seul traite WAV ou signal brut flottant, sans génération longue ni requantification des stress.

Bundle final autonome avec modèle, paramètres temporels, configuration, scoring, provenance, empreintes et exemple validation ; reload dans un processus neuf obligatoire avant test. Export T0 : seulement `metadata.json` + `predictions.csv`, exactement `clip_id,probability_leak`, 402 IDs val/test. Nevil reste seul propriétaire de l'évaluation finale.

## Stress officiels désormais reproductibles

Référence récupérée : `6dfdf63580bc15ce6a1cf0817b9da3569553aca1` sur `nevil/temporal-evidence`. Nevil a corrigé la graine dans `b23601ab28c5b3949b2b17c46478daa46c847c0f` : SHA-256 stable du triplet graine/nom/clip. Source `scripts/temporal/stress.py` SHA-256 `7c37e001d270655d2e54ba2087e8b8e476da8cc42948f92bd83c669773ac6350`. La réserve antérieure sur le `hash()` Python est donc levée pour cette révision, pas pour les variantes anciennes.

T1/T2/T3 après T0 : transformations officielles au brut, preprocessing et checkpoint identiques, aucun réentraînement. T2 = 250 échantillons ; aucun reste hors permutation, sans garantie que tous les blocs changent de place. Contrôler la graine publiée et des processus distincts avant génération ; noter version NumPy/empreintes. Aucun stress réel exécuté à ce jalon.

## Prochaine action concrète

Terminer les scripts/tests, figer le code et la préinscription, transférer une révision identifiée dans un nouveau dossier distant. Exécuter les tests CPU puis le sanity scoring train sur la V0. Seulement ensuite lancer les huit époques, sélectionner sur validation, figer/recharger et livrer les prédictions. Mettre cette note à jour avec les résultats et chemins réellement produits, sans anticiper une qualité.
