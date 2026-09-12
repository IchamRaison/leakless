# Passation

Icham

## Entire — état vérifié sur Linux

CLI 0.10.6 et 12 skills installés ; capture Codex locale constatée. La recherche distante de commits fonctionne désormais (audit/splits de Nevil retrouvés), contrairement au contrôle initial non authentifié. Cela ne prouve pas la publication des transcriptions. Trois hooks Codex restent signalés à approuver via `/hooks`. État initial : [[Entire - installation et vérification]] ; nouvelle preuve dans [[Journal Icham]].

## Dernier échange — 2026-09-12

**Campagne V1 réellement en cours :** [[V1 ML - exécution]] donne commande, préinscription et chemins. Code `1199789f` publié/transféré, empreintes vérifiées, 18 tests CPU réussis, sanity réel sur quatre clips train conforme. Run `/home/hicham/pipe-v0/artifacts/qwen-v1-1199789f-001`, log `runs/train-v1-1199789f-001.log`, PID distant observé `32101` à revérifier avant toute reprise. Trois checkpoints 2/4/8, sélection ROC-AUC validation groupée seulement ; aucun export/test final. Prochaine action : surveiller le run existant, vérifier sélection/gel/reload, livrer T0 puis stress. Nevil a corrigé les graines SHA-256 (`6dfdf63`, correction `b23601a`) ; l'ancienne réserve T2/T3 ci-dessous est levée pour cette révision. Les paragraphes suivants conservent le cadrage avant le feu vert.

**Plan révisé après les précisions Nevil, pas démarré :** [[Plan surveillance continue]] §4 C1–C4 et §5 conserve la priorité score → V1 train/validation → gel/reload → T0 à Nevil. Nouveau contrat et `check_run.py` lus à `913575406b39beec3ae7da77aec322cca926d034`. Proposition : log-probabilités des deux continuations, sommes brutes sans normalisation automatique par longueur ; scores non calibrés ; au plus trois configurations candidates initiales, nombre réel déclaré. Toute adaptation, y compris LoRA, répond uniquement à un diagnostic train/validation. Réutiliser le contrôleur dans un dossier neuf, sans tuning d'après son inspection val/test ; un avertissement binaire n'empêche pas son retour zéro. Nevil conserve toutes les métriques finales. Prochain bloc : C1 à C3 jusqu'à T0 livrable ; aucune implémentation, entraînement ou prédiction nouvelle dans ce tour.

**Réserve T2/T3, pas blocage T0 :** `clip_rng` repose sur le `hash()` Python salé par processus. Il faut une correction coordonnée et versionnée avec Nevil ou ses signaux transformés exacts et empreintes ; ne pas prétendre reproduire les stress en appelant seulement la même fonction. T1 non concerné par cet aléa. Revue, démonstration jouet et nuances de calibration : [[Journal Icham#Revue des précisions Nevil — scores et stress]]. Aucun correctif ni message envoyé à Nevil. Flux/replay, événements et collecte continuent comme chantiers proposés en parallèle.

**Consigne de livraison Nevil transmise par Icham : prédictions seules, pas de métriques finales calculées par l'agent ML.** Contrat courant relu à `9135754` (consigne initiale `08562e3`). T0 prioritaire : `metadata.json` + `predictions.csv`, exactement `clip_id,probability_leak`, couverture des 402 clips validation/test du v2 gelé ; aucun seuil appliqué ni colonne de métadonnée dataset. Après gel du checkpoint, T1/T2/T3 optionnels en inférence seule et sous la réserve ci-dessus, mêmes poids et méthode de score. Nevil possède seuils, agrégations, métriques et comparaisons finales. Les diagnostics de développement restent séparés, sans test. Manque actuel côté V0 : calcul de probabilité de classe vérifié (`score_type=none`). [[Journal Icham#Consigne de livraison T0 et stress — Nevil]].

**Cadrage utilisateur corrigé : surveillance automatique continue.** Icham précise que l'appareil écoute les tuyaux et signale la fuite ; l'humain reçoit l'alerte, il ne fournit pas l'extrait. [[Plan surveillance continue]] remplace le scénario produit manuel et propose le plan complet : flux/replay, V1 sur fenêtres, suivi d'événement, dashboard, acquisition continue et évaluation opérationnelle. Cap produit confirmé ; choix détaillés proposés, pas exécutés. Les clips d'une seconde ne prouvent ni délai de détection ni fausses alertes/jour. Prochaine action après validation du plan : préciser capteur/flux et objectifs, puis boucle replay sur V0 et V1 en parallèle ; recueillir du continu annoté sans attendre. Ce tour modifie uniquement les notes, aucun entraînement/déploiement.

Livraison Nevil `nevil/temporal-evidence` à `08562e3` lue, non fusionnée ; tests/rapports non reproduits par Icham. Détails : [[Journal Icham#Vérification des nouveaux livrables Nevil]]. Le raccordement par fichiers prévu ci-dessus remplace la proposition initiale d'intégrer son moteur dans notre chantier.

**V0 mécanique réalisée jusqu'à l'étape 5**, avec Qwen 3.5-4B demandé en remplacement de Llama : 1 000 WAV via TimeNet, v2 vérifié, vrai batch GPU, 40 étapes sur 8 groupes train, poids temporels modifiés et Qwen gelé vérifié. Checkpoint autonome rechargé hors ligne dans un processus neuf puis dans un environnement reconstruit du lockfile ; même prédiction sur le clip validation fixé, erreurs contrôlées. Modèle `pipe-qwen3.5-4b-v0-a968405f`, fonction `Predictor.predict(wav_bytes)` livrable à Safoan, guide `docs/TSLM_V0.md` sur `feat/icham-tslm`. **Qualité non validée, application non intégrée, aucun score final annoncé.** Preuves, checksum et prochaine action : [[V0 ML - exécution]].

## Chantier parallèle — simulateur de capteur

2026-09-12 : Icham demande un plan technique indépendant du travail ML. [[Plan simulateur de capteur]] détaille une source locale sans GPU : WAV cadencés, enveloppe de fenêtre, coupures/retards/reprise et tests CPU, puis raccordement au backend en concertation avec Safoan. Plan uniquement, aucun script/test/replay lancé par cette conversation. Prochaine action après demande d'implémentation : vérifier le code existant et réaliser la preuve locale sans toucher au modèle, aux données gelées ni au service GPU. L'état du chantier ML ci-dessus reste géré par l'agent principal.

## Cadrage précédent

Icham demande une critique structurelle, la première réponse étant trop méthodologique. [[Challenge du cadrage de Nevil]] précise l'avis : direction hackathon soutenue, produit à resserrer ; le WHY de surveillance permanente ne correspond pas au WHAT d'analyse d'un extrait choisi. Recommandation à discuter : seconde lecture pour un technicien, avec bénéfice de qualification/documentation à confronter à son travail réel. Nouvelle note distante [[PROBLEM STATEMENT - LeakLess Temporal AI]] intégrée (`0a70a95`) : direction déclarée approuvée en équipe sous réserve de l'audit ; ses trois classes restent à aligner avec les contrats binaires. Aucun contrat ni chantier technique modifié. G0 et entretien métier peuvent avancer en parallèle ; preuves dans [[Journal Icham]].

## Reprise Safoan — 2026-09-12

Livraison `feat/safoan-app` à `4dd7b88` : application/API et contrat Prediction présents, imports/audio/visualisation implémentés sur sa branche ; endpoint modèle encore indisponible volontairement. Code de l'API lu, tests application non rejoués par Icham. Suivi personnel : [[Journal Safoan]].

## Nom proposé

Nevil propose le nom LeakLess et confirme le cadrage software-only acoustique, sous réserve de l’audit. Avis favorable avec maintien obligatoire de TimeNet et entraînement TSLM ; validation finale d’Icham non encore enregistrée. Voir [[Proposition Nevil - LeakLess]]. Le plan PIPE reste applicable, sans renommer ses références avant décision.

## État courant

Direction de travail : PIPE, surveillance acoustique continue avec alerte automatique, TimeNet, TSLM réellement entraîné et baseline. [[Plan surveillance continue]] fixe le nouveau parcours proposé. La preuve mécanique V0 est acquise ; V1 évaluée, flux continu, logique d'événement, intégration et validation terrain restent à réaliser.

Un TSLM adapté fonctionne mécaniquement : [[V0 ML - exécution]], qualité non évaluée. Le rapport baseline de Nevil (`1289095`) reste un résultat séparé non reproduit par Icham ; aucune comparaison finale n'est déduite de la V0. `split_v1` invalide, v2 uniquement. Machine fournie par Icham, aucun nouveau GPU provisionné ; l'instance n'a pas été arrêtée.

## Lire pour reprendre sans le chat

[[Plan surveillance continue]] → [[Contrats techniques]] → [[Protocole évaluation]] → [[Équipe et répartition]] → sa fiche dans 06 Agents. [[Plan directeur agents]] conserve l'organisation et l'ancien scénario à titre historique. [[Coordination et passation agents]] régit les branches et mises à jour. Les nouvelles interfaces de flux/événement sont à convenir, pas déjà fonctionnelles.

## Répartition et prochaines actions proposées

- Icham : [[Agent Icham - ML]] ; score TSLM, V1 et budget de calcul continu, règles d'événement avec Nevil. Journal : [[Journal Icham]].
- Nevil : [[Agent Nevil - Data]] ; validation seule, protocole bruit, données continues et métriques événementielles. Journal : [[Journal Nevil]].
- Safoan : [[Agent Safoan - Application]] ; flux/replay sur V0, santé de surveillance, alertes et preuves, interfaces en concertation. Journal : [[Journal Safoan]].
- Vincent : [[Agent Vincent - Baseline]] ; modèle simple et export comparable, sans confondre son livrable avec les contrôles logistiques de Nevil. Journal : [[Journal Vincent]].

V0 mécanique terminée ; ce nouveau travail n'a pas démarré. Responsable matériel/collecte et pitch/règles organisateurs à désigner par Icham ; répartition détaillée dans [[Plan surveillance continue]].

## Preuves déjà disponibles

Source dataset https://zenodo.org/records/18631450 : 500 fuite, 386 sans fuite, 114 bruits. L'audit publié de Nevil confirme 1 000 WAV mono, 8 kHz, PCM 16 bits, une seconde. `split_v2` publié : 598 train, 208 validation, 194 test, 185 groupes heuristiques ; ses invariants ne prouvent pas une indépendance de sessions réelles. Mesures expérimentales, pas preuve chez des clients. Voir [[Journal Icham]] et [[PIPE - proposition ML et démo]].

Dépôts privés créés et poussés : code https://github.com/IchamRaison/ehl-hackathon-zurich ; notes https://github.com/IchamRaison/ehl-hackathon-zurich-vault. Bootstrap Entire au commit 373d13f ; activé manual-commit, télémétrie et push automatique de sessions désactivés. Capture Codex locale désormais constatée ; publication/indexation distante non vérifiées. Voir [[Entire - installation et vérification]].

## Inconnues et risques

- Sessions réelles et données continues annotées manquantes ; groupes v2 heuristiques malgré invariants vérifiés.
- Capteur/format/montage et disponibilité du flux, objectifs de délai/fausses alertes, débit soutenu du TSLM à établir.
- Acceptation explicite du cadrage acoustique, deadline, durée du pitch et accès/crédit Nebius à confirmer.
- TSLM peut ne pas battre la baseline ou ne pas bénéficier de l'ordre temporel ; l'évaluation doit le montrer.
- Capteurs du coéquipier non intégrés, unités/grandeurs non vérifiées. Replay proposé pour le prochain prototype ; matériel réel nécessaire à une validation du produit.

## Dépôts et état historique

Sur cette machine : /home/animus/ehl-hackathon-zurich pour le code, /home/animus/ehl-hackathon-zurich-vault pour les notes. Sur un autre poste, cloner les dépôts avec accès autorisé. Le vault dédié est la référence ; le miroir vault/ du code peut être ancien.

Les notes débit/pression et comparaisons de pistes sont conservées comme historique. Elles ne remplacent pas le plan acoustique courant. Mettre cette note à jour au prochain résultat réel, sans accumuler des décisions anciennes sous « état courant ».
