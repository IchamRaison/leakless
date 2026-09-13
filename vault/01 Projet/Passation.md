# Passation

Icham

## Entire — état vérifié sur Linux

CLI 0.10.6 et 12 skills installés ; capture Codex locale constatée. La recherche distante de commits fonctionne désormais (audit/splits de Nevil retrouvés), contrairement au contrôle initial non authentifié. Cela ne prouve pas la publication des transcriptions. Trois hooks Codex restent signalés à approuver via `/hooks`. État initial : [[Entire - installation et vérification]] ; nouvelle preuve dans [[Journal Icham]].

## Dernier échange — 2026-09-13

**D2 en préparation, aucun fit lancé :** protocole unique d'arbres boostés fixé après D1, revue méthodologique favorable. Runner `scripts/tslm/diagnose_nonlinear.py` présent uniquement dans le clone sain, non commité/non revu ; fichier de tests pas encore créé. Travail arrêté pendant la question d'Icham sur trois H100 supplémentaires : ne pas lancer avant tests, revue et préinscription machine. Critère primaire : AUC groupe réservée face à la sonde linéaire exacte, C1 fixe séparé. [[Diagnostic causal - exécution#D2 — sonde non linéaire, protocole avant fit]]. CPU uniquement pour ce petit contrôle indépendant ; les apprentissages TSLM restent sur H100. Aucun GPU supplémentaire demandé/provisionné, aucun changement distribué adopté.

**Icham élargit le goal aux étapes 6 à 10 et confirme le renvoi unique au vault.** Le prompt `/goal` renvoie au fichier [[Diagnostic causal TSLM vs C1]] : ce document couvre diagnostic, corrections évaluées, diversification globale conditionnelle, mesure de l'apport des données, confirmation indépendante et produit continu avec benchmark gabarit/Qwen/TSLM. [[Diagnostic causal TSLM vs C1#Critère de complétion du goal élargi]] interdit de terminer le goal à la seule fin du diagnostic. Le prompt court est consigné dans cette note, sans dupliquer le plan ni créer un autre goal. Rôles Aghashahi/Hong Kong et critères des futures campagnes restent à fixer avant usage, sans effacer les résultats historiques.

**D1 terminé :** code `306d330`, 203 tests runtime sans skip, préinscription `37cfcae1…` publiée à `1f89933` avant les 3 588 forwards. A `640d29c` / reçu `f9402c5d…`, C `9c163d4` / reçu `00d8de16…` ; empreintes, poids et reproduction D0 vérifiés localement/distamment, aucune politique adoptée ni lecture test/externe. AUC groupe des 98 réservés BF16→FP32 : A 0,760→0,745 ; C 0,500→0,510. Résolution plus fine, pas de correction uniforme du retard sur C1 ni verdict sur la précision pendant l'entraînement. Prochaine action : borner et préinscrire un seul lecteur non linéaire des 256 entrées exactes sur les trois folds train, en réutilisant le harness. [[Diagnostic causal - exécution]]. Logger `66f390b` testé (neuf contrôles de transparence), pas encore utilisé sur un nouveau fit réel.

**Précision d'Icham : diversité globale, pas seulement sans-fuite.** [[Diagnostic causal TSLM vs C1#Dernière étape conditionnelle — diversifier les données acoustiques]] garde l'élargissement après les diagnostics et corrections si le gain reste insuffisant. D1 est terminé ; les contrôles conditionnels non exécutés ne sont pas des échecs ni des tâches achevées. [[Datasets utiles pour PIPE#Revue du 13 septembre — diversité globale]] : sources relues, Hong Kong candidat à auditer ; Aghashahi déjà préparé/réservé, toujours sans score. Icham interroge le coût de cette réserve : réaffectation possible seulement dans un futur protocole explicite avec autre confirmation, non décidée ici. Aucun nouveau téléchargement de données, collecte, intégration ou fit dans cette revue. Le site principal est commun aux deux classes ; les242 vrais sans-fuite/7 groupes ne sont qu'une partie de la diversité à examiner. [[Diagnostic causal - exécution#Audit des populations et des cibles]].

**Intégration application/TSLM V2 terminée par Safoan au commit `bba67ff` sur
`feat/demo-temporal-building` :** fusion de `feat/icham-v2-reliability`,
adaptateur FastAPI chargé une fois, `POST /predict` réel et bouton UI explicite.
23 tests API, 35 tests frontend, build et 12 tests du wrapper cohérent passent.
Les artefacts sélectionnés ne sont pas dans Git : l'inférence sur poids réels
reste non exécutée et l'UI signale l'indisponibilité sans fabriquer de score.
Prochaine action : fournir bundle/décision/reçu cohérents, puis recette réelle.
Voir [[Journal Safoan]] et `docs/APPLICATION.md`.

**D0 terminé, vérifié et publié : [[Diagnostic causal - exécution]].**1516 observations, zéro fit ; A `f64c943` /reçu `4810270a…`, C `9dd5b73` /reçu `c6d2121f…`, revue indépendante des1516 lignes. AUC fit clip/groupe A0,639/0,731 ; C0,878/0,731 ; C réservé0,499/0,500. Les deux modèles apprennent un classement, C réagit aux séries et au texte, mais ne transfère pas correctement dans ce fold. D1 a depuis isolé un effet de précision de tête à poids fixes ; les écarts LP loss/scoring entre forwards et l'optimisation ne sont pas entièrement expliqués. Ne pas relancer D0 ; aucun refit/test/externe automatique.

**Complément Claude intégré à la checklist, sans exécution :** journalisation NLL classe/binaire par clip, vrai contrôle train, mémorisation32, tête BCE sans Qwen, normes/gradients, sonde non linéaire et neuf mesures C1 par voie entraînable. Certaines pistes étaient déjà prévues ; pas de campagne doublonnée. Relecture corrige la provenance époque4 versus dernier batch époque8 et maintient scoring/optimisation parmi les hypothèses ouvertes : une loss proche de ln2 ne prouve pas un classement au hasard. [[Diagnostic causal TSLM vs C1#Checklist enrichie après la revue de Claude]].

**Runtime vérifié à 03:37 Paris :** aucun processus de campagne actif, H100 0Mio/0%, instance allumée. Tous les reçus/poids distants des six fits et les derniers artefacts rapatriés ont été vérifiés ; Qwen inchangé, acoustique modifiée sur chaque fit. Ne pas relancer. Ancienne préinscription `c13fd47` / SHA `5c5e0b4b…` intacte, C1 douze fits complets ; [[V2 ML - exécution]] conserve l'historique. **Checkout sain `/home/animus/ehl-hackathon-zurich-v2-recovery`**, branche `feat/icham-v2-reliability` ; ancien Git endommagé conservé, ne pas y commiter. Cap produit événements/investigation inchangé et non validé.

## Résultats et orientations précédents — historique

**Recherche datasets terminée :** [[Datasets utiles pour PIPE]] ajoute une shortlist vérifiée sur les pages sources, sans téléchargement ni intégration. Le benchmark actuel Zenodo 18631450 reste inchangé et aucun dataset externe ne doit être fusionné au T0/V1 gelé. Priorité proposée : Aghashahi/Mendeley comme hold-out acoustique hydrophone 8 kHz, puis Hong Kong/Mendeley comme test de transfert terrain ; Yorkshire/Wessex pour événements/priorisation, NTNU/slovaque pour replay hydraulique. Licences, formats, sessions et splits restent à auditer avant usage. Prochaine action : choisir une source externe et faire un audit local minimal ; ne pas régler le modèle courant sur ses résultats.

**Évaluation réelle terminée, vérifiée et publiée à `c8dcb9d` sur `feat/icham-quality-eval` :** T0 sur 194 test manque 25/98 fuites et produit 43/96 fausses alertes. AUC clip/groupe `0,665/0,861`, contre `0,902/0,927` pour C1 ; aucun gain TSLM démontré. Audit 402/402 : sur test, 184/194 bandes correctes et 36/194 désaccords classe/score. [[Évaluation qualité V1 - exécution]] lie le rapport, les preuves et la limite de parité entre scores de campagne/cache et livraison/WAV, cause précise non isolée. Code/préinscription `c27a43fd`, 32 tests réussis, 396 valeurs et 68 IC revérifiés indépendamment, poids/exports inchangés. Aucun job restant, GPU à 0 Mio, instance allumée. Prochaine action : discuter d'une V2 et d'un protocole distinct, pas réentraîner sur les résultats test ni relancer l'audit.

## Livraison et planification — historique

**Nouvelle consigne : nous évaluerons la qualité nous-mêmes, mais planification seulement à ce stade.** Icham indique que Nevil n'a pas encore fait l'évaluation et lève sa responsabilité exclusive. [[Protocole évaluation#Plan qualité V1 — proposé, non exécuté]] décrit détection T0 sur 194 test, comparaison appariée aux contrôles, stress déjà exportés et audit texte complet val/test séparés. Prochaine action : feu vert d'exécution, puis réutilisation du harness existant ; aucune métrique, inférence ou modification du checkpoint lancée dans ce tour. La responsabilité Nevil mentionnée dans les jalons précédents est historique ; les contraintes de gel et d'absence de tuning test restent en vigueur.

**Plan ML entièrement exécuté ; T0–T3 conformes et publiés :** branche `feat/icham-tslm`, T0 à `186c45a`, stress à `7c04c9a5636b2872334da17c54beb7016ffa00a3`. Chacun des quatre dossiers `artifacts/tslm_runs/tslm-v1{,-T1,-T2,-T3}/` contient seulement `metadata.json` et `predictions.csv`, 402 clips val/test et deux colonnes exactes. [[V1 ML - exécution]] donne toutes les preuves. Modèle `pipe-qwen3.5-4b-v1-1199789f-e4`, époque 4 retenue parmi trois candidats sur validation ; Qwen gelé inchangé. Bundle `/home/hicham/pipe-v0/artifacts/qwen-v1-1199789f-001/bundle`, checksum `b95569c5…`, reload neuf et 19 tests CPU réussis. Même scoring et checkpoint pour les stress officiels `6dfdf63`, aucun réentraînement. Audit indépendant des quatre exports réussi ; GPU revenu à 0 Mio, instance laissée allumée. Prochaine action : Nevil évalue ces fichiers ; Icham assure le support d'intégration. Aucun calcul final de métriques ni adaptation après inspection du test, aucune intégration applicative revendiquée.

## Repères plus anciens — historique

**Plan révisé après les précisions Nevil, pas démarré :** [[Plan surveillance continue]] §4 C1–C4 et §5 conserve la priorité score → V1 train/validation → gel/reload → T0 à Nevil. Nouveau contrat et `check_run.py` lus à `913575406b39beec3ae7da77aec322cca926d034`. Proposition : log-probabilités des deux continuations, sommes brutes sans normalisation automatique par longueur ; scores non calibrés ; au plus trois configurations candidates initiales, nombre réel déclaré. Toute adaptation, y compris LoRA, répond uniquement à un diagnostic train/validation. Réutiliser le contrôleur dans un dossier neuf, sans tuning d'après son inspection val/test ; un avertissement binaire n'empêche pas son retour zéro. Nevil conserve toutes les métriques finales. Prochain bloc : C1 à C3 jusqu'à T0 livrable ; aucune implémentation, entraînement ou prédiction nouvelle dans ce tour.

**Réserve T2/T3, pas blocage T0 :** `clip_rng` repose sur le `hash()` Python salé par processus. Il faut une correction coordonnée et versionnée avec Nevil ou ses signaux transformés exacts et empreintes ; ne pas prétendre reproduire les stress en appelant seulement la même fonction. T1 non concerné par cet aléa. Revue, démonstration jouet et nuances de calibration : [[Journal Icham#Revue des précisions Nevil — scores et stress]]. Aucun correctif ni message envoyé à Nevil. Flux/replay, événements et collecte continuent comme chantiers proposés en parallèle.

**Consigne de livraison Nevil transmise par Icham : prédictions seules, pas de métriques finales calculées par l'agent ML.** Contrat courant relu à `9135754` (consigne initiale `08562e3`). T0 prioritaire : `metadata.json` + `predictions.csv`, exactement `clip_id,probability_leak`, couverture des 402 clips validation/test du v2 gelé ; aucun seuil appliqué ni colonne de métadonnée dataset. Après gel du checkpoint, T1/T2/T3 optionnels en inférence seule et sous la réserve ci-dessus, mêmes poids et méthode de score. Nevil possède seuils, agrégations, métriques et comparaisons finales. Les diagnostics de développement restent séparés, sans test. Manque actuel côté V0 : calcul de probabilité de classe vérifié (`score_type=none`). [[Journal Icham#Consigne de livraison T0 et stress — Nevil]].

**Cadrage utilisateur corrigé : surveillance automatique continue.** Icham précise que l'appareil écoute les tuyaux et signale la fuite ; l'humain reçoit l'alerte, il ne fournit pas l'extrait. [[Plan surveillance continue]] remplace le scénario produit manuel et propose le plan complet : flux/replay, V1 sur fenêtres, suivi d'événement, dashboard, acquisition continue et évaluation opérationnelle. Cap produit confirmé ; choix détaillés proposés, pas exécutés. Les clips d'une seconde ne prouvent ni délai de détection ni fausses alertes/jour. Prochaine action après validation du plan : préciser capteur/flux et objectifs, puis boucle replay sur V0 et V1 en parallèle ; recueillir du continu annoté sans attendre. Ce tour modifie uniquement les notes, aucun entraînement/déploiement.

Livraison Nevil `nevil/temporal-evidence` à `08562e3` lue, non fusionnée ; tests/rapports non reproduits par Icham. Détails : [[Journal Icham#Vérification des nouveaux livrables Nevil]]. Le raccordement par fichiers prévu ci-dessus remplace la proposition initiale d'intégrer son moteur dans notre chantier.

**V0 mécanique réalisée jusqu'à l'étape 5**, avec Qwen 3.5-4B demandé en remplacement de Llama : 1 000 WAV via TimeNet, v2 vérifié, vrai batch GPU, 40 étapes sur 8 groupes train, poids temporels modifiés et Qwen gelé vérifié. Checkpoint autonome rechargé hors ligne dans un processus neuf puis dans un environnement reconstruit du lockfile ; même prédiction sur le clip validation fixé, erreurs contrôlées. Modèle `pipe-qwen3.5-4b-v0-a968405f`, fonction `Predictor.predict(wav_bytes)` livrable à Safoan, guide `docs/TSLM_V0.md` sur `feat/icham-tslm`. **Qualité non validée, application non intégrée, aucun score final annoncé.** Preuves, checksum et prochaine action : [[V0 ML - exécution]].

## Chantier parallèle — simulateur de capteur

2026-09-12 : **étapes 1–4 implémentées et testées**, dans `/tmp/ehl-sensor-replay`, branche `feat/sensor-replay`. Deux tests CPU passent ; replay réel de huit secondes : cinq réceptions, trois coupures, un doublon ignoré. Script autonome sans dépendance/GPU ; aucune modification du chantier ML. [[Plan simulateur de capteur]] donne les commandes, preuves et limites ; guide code `docs/REPLAY_SENSOR.md`. Étape 5 non intégrée : prochaine action, convenir avec Safoan du transport et du timeout avant vraie inférence. Provenance synthétique et chronologie artificielle, aucune validation terrain.

## Cadrage précédent

Icham demande une critique structurelle, la première réponse étant trop méthodologique. [[Challenge du cadrage de Nevil]] précise l'avis : direction hackathon soutenue, produit à resserrer ; le WHY de surveillance permanente ne correspond pas au WHAT d'analyse d'un extrait choisi. Recommandation à discuter : seconde lecture pour un technicien, avec bénéfice de qualification/documentation à confronter à son travail réel. Nouvelle note distante [[PROBLEM STATEMENT - LeakLess Temporal AI]] intégrée (`0a70a95`) : direction déclarée approuvée en équipe sous réserve de l'audit ; ses trois classes restent à aligner avec les contrats binaires. Aucun contrat ni chantier technique modifié. G0 et entretien métier peuvent avancer en parallèle ; preuves dans [[Journal Icham]].

## Reprise Safoan — 2026-09-12

Premier studio S0–S1 livré sur `feat/safoan-app` : React/Vite + FastAPI,
import WAV, lecture et visualisation. Le commit `504ad14` a passé 19 tests API,
4 tests React et le build ; Icham a ensuite lu le code à `4dd7b88` sans rejouer
ces tests. Aucun modèle n'était intégré à ce jalon. Voir [[Journal Safoan]] et
`docs/APPLICATION.md` dans le code.

## Nom proposé

Nevil propose le nom LeakLess et confirme le cadrage software-only acoustique, sous réserve de l’audit. Avis favorable avec maintien obligatoire de TimeNet et entraînement TSLM ; validation finale d’Icham non encore enregistrée. Voir [[Proposition Nevil - LeakLess]]. Le plan PIPE reste applicable, sans renommer ses références avant décision.

## État courant

Direction de travail : PIPE, surveillance acoustique continue avec alerte automatique, TimeNet, TSLM réellement entraîné et baseline. [[Plan surveillance continue]] distingue cette ambition de la V1 et de son évaluation réellement exécutée par Icham : [[Évaluation qualité V1 - exécution]]. Le détecteur est insuffisant à ce stade ; aucun raccordement au flux/simulateur ni logique d'événement ou fiabilité terrain n'est prouvé par ces clips.

Le TSLM adapté et son score continu sont vérifiés : [[V1 ML - exécution]]. L'évaluation finale réutilise le harness Nevil, cinq contrôles fixes et les exports gelés ; les résultats ne démontrent pas de supériorité TSLM. `split_v1` invalide, v2 uniquement. Machine fournie par Icham, aucun nouveau GPU provisionné ; l'instance n'a pas été arrêtée.

## Lire pour reprendre sans le chat

[[Plan surveillance continue]] → [[Contrats techniques]] → [[Protocole évaluation]] → [[Équipe et répartition]] → sa fiche dans 06 Agents. [[Plan directeur agents]] conserve l'organisation et l'ancien scénario à titre historique. [[Coordination et passation agents]] régit les branches et mises à jour. Les nouvelles interfaces de flux/événement sont à convenir, pas déjà fonctionnelles.

## Répartition et prochaines actions proposées

- Icham : [[Agent Icham - ML]] ; V1/stress livrés, support d'intégration et mesure du budget de calcul continu ; ne pas optimiser sur les résultats test. Journal : [[Journal Icham]].
- Nevil : [[Agent Nevil - Data]] ; évaluation finale des exports, interprétation des stress, données continues et métriques événementielles. Journal : [[Journal Nevil]].
- Safoan : [[Agent Safoan - Application]] ; flux/replay sur V0, santé de surveillance, alertes et preuves, interfaces en concertation. Journal : [[Journal Safoan]].
- Vincent : [[Agent Vincent - Baseline]] ; modèle simple et export comparable, sans confondre son livrable avec les contrôles logistiques de Nevil. Journal : [[Journal Vincent]].

V0 terminée et V1 livrée pour T0–T3. Aucun entraînement/export en cours ni à refaire pour cette livraison. Le chantier continu ne doit pas être confondu avec cette livraison ML. Responsable matériel/collecte et pitch/règles organisateurs à désigner par Icham ; répartition détaillée dans [[Plan surveillance continue]].

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
