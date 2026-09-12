# Plan surveillance continue

Icham

## Statut et décision utilisateur

2026-09-12 : Icham précise que l'entrée vient d'un appareil qui écoute les tuyaux en permanence et signale automatiquement une fuite, **pas d'un technicien qui importe un extrait**. Cette clarification remplace le scénario produit de relecture manuelle. L'humain reçoit et examine les alertes.

Le cap produit est confirmé. Icham a ensuite autorisé l'exécution du volet ML : **C1 à C4 sont exécutés et vérifiés, T0 et les trois stress sont publiés**. Les jalons produit A/B/D/E/F/G restent proposés ou à coordonner avec [[Plan simulateur de capteur]] ; ni intégration applicative complète, ni matériel réel, ni notification externe ne sont démontrés par cette livraison.

Problème : surveiller un signal de canalisation et avertir lorsqu'un événement compatible avec une fuite persiste, avec des éléments vérifiables pour l'examiner. L'objectif final est bien la détection automatique ; tant que la fiabilité n'est pas validée, la sortie du prototype est une suspicion, pas la confirmation physique d'une fuite.

**État au 2026-09-12 :** V1 entraînée, figée et rechargée dans un processus neuf ; T0 publié sur `feat/icham-tslm` au commit [`186c45a`](https://github.com/IchamRaison/ehl-hackathon-zurich/commit/186c45a). Score continu issu des log-probabilités, trois candidats choisis sur validation uniquement, contrôleur existant réutilisé ; 19 tests CPU passent. Nevil conserve toutes les métriques finales. La reproductibilité T2/T3 a été corrigée officiellement dans `b23601a`, version reprise `6dfdf63` ; T1/T2/T3 ont terminé sans réentraînement et sont conformes, chacun sur 402 clips. Stress publiés au commit [`7c04c9a`](https://github.com/IchamRaison/ehl-hackathon-zurich/commit/7c04c9a5636b2872334da17c54beb7016ffa00a3). Preuves et commandes : [[V1 ML - exécution]].

## 1. Deux niveaux de livraison

- **Prochain jalon hackathon proposé** : un canal, flux horodaté rejoué automatiquement, modèle réellement entraîné, suivi d'événement, alerte et historique dans l'application. Inférence calculée pendant le replay, source enregistrée explicitement signalée. Si un flux compatible d'appareil est disponible, le raccorder après vérification.
- **Validation produit** : appareil réel installé, longues acquisitions continues, défauts et événements annotés, évaluation sur sessions/installations réservées. L'embarqué, le fonctionnement hors réseau et la flotte multi-appareils ne sont pas démontrés par une H100.

Un assemblage de WAV isolés reste un scénario artificiel. Il peut tester le logiciel et la démonstration, jamais prouver un délai physique de détection ou un taux de fausses alertes sur le terrain. L'absence de chronologie réelle ne bloque pas le prototype, mais bloque ces revendications opérationnelles.

## 2. Ce que nous conservons

- V0 Qwen 3.5-4B/OpenTSLM-SP, checkpoint autonome, preprocessing unique et interface `Prediction` : [[V0 ML - exécution]]. Elle classe un extrait, pas un événement suivi dans le temps.
- TimeNet/TimeF pour préparation et traçabilité des données d'apprentissage ; même transformation numérique en inférence. Pas d'obligation ajoutée d'écrire un dataset TimeF à chaque fenêtre live.
- Split v2 et groupes existants, sans redistribution pour améliorer un score. Dataset actuel : 1 000 clips d'une seconde, pas une chronologie continue vérifiée.
- Contrat et contrôleur `check_run.py` Nevil à `6dfdf63580bc15ce6a1cf0817b9da3569553aca1` : produire des exports compatibles ; Nevil possède le moteur final. Seul le sous-ensemble de conformité nécessaire est repris dans la branche ML, pas le moteur de métriques finales. Baseline Vincent si livrée. Les tests T1–T3 sont des perturbations de clips, pas une évaluation du cycle de vie des alertes.
- Audio/spectrogramme et affichage Safoan : ils deviennent les pièces justificatives d'une alerte. L'import manuel reste un outil de diagnostic, pas l'interaction principale.

## 3. Architecture proposée

```text
Appareil réel OU replay horodaté déclaré
  → validation du flux + contrôle de qualité
  → fenêtres causales + preprocessing partagé
  → score fuite issu du TSLM
  → historique récent + règle de déclenchement/fin
  → événement horodaté + alerte + compte rendu
  → tableau de surveillance et examen humain
```

La V0 actuelle traite une seconde PCM16 mono à 8 kHz. Proposition de départ : fenêtres d'une seconde, cadence à figer après mesure. Le flux brut et son gain/qualité restent contrôlés avant normalisation ; une fenêtre silencieuse ou manquante n'est pas une preuve d'absence de fuite.

Deux cadences proposées : score de classe régulier, texte lors de l'ouverture d'un événement ou sur demande. Le score provient du TSLM adapté au signal, pas d'une baseline ensuite racontée par Qwen. La durée, les horaires et les comptes de fenêtres viennent du logiciel ; une description d'un extrait par le TSLM ne doit pas être présentée comme une analyse de plusieurs minutes.

Le calcul de score sans génération longue est implémenté et vérifié dans V1. Le débit d'une boucle continue reste à mesurer ; les latences V0 ~0,75–1,35 s sur un exemple ne garantissent pas un flux soutenu. Mesurer latence de bout en bout, p95, débit, file d'attente et mémoire ; un retard non borné ou des fenêtres perdues doivent dégrader explicitement l'état de surveillance. Le texte ne doit pas retarder l'émission de l'alerte.

Première exécution proposée sur le serveur existant, un canal ; aucune promesse de Qwen embarqué ni besoin déduit de H100 par appareil. Décider edge/cloud ensuite à partir des contraintes réseau, énergie, confidentialité et débit réel.

## 4. Étapes et critères de passage

### A — Cadrer le fonctionnement de l'appareil et le succès

Icham + personne du métier, avec Nevil/Safoan : identifier capteur, fixation/contact, grandeur mesurée, format/fréquence/gain, accès au flux, destination de l'alerte et conditions ordinaires de fonctionnement. Ne pas supposer qu'un microphone d'ambiance ou des ultrasons ont le même domaine que nos WAV. Rééchantillonnage éventuel documenté, anti-aliasing, compatibilité physique à vérifier.

Fixer ensemble délai de détection souhaité, charge de fausses alertes acceptable, durée de validation et comportement en perte de réseau. Aucun seuil chiffré de qualité ou délai garanti n'est encore choisi. Prototype : alerte dans le dashboard, pas de SMS ni d'action automatique sur une vanne.

**Sortie :** contrat de flux et critères métier écrits ; responsabilités et distinction replay/appareil réel explicites.

### B — Construire une boucle continue minimale

Safoan + Icham : ingestion/replay automatique, identifiant d'appareil et de session, numéro de fenêtre, heures source/réception, ordre et déduplication. Une fenêtre n'utilise que des observations déjà reçues, pas de contexte futur. Prévoir redémarrage, trous, données invalides et fin de replay. API privée, entrées validées, mémoire/file bornées ; conservation limitée aux données nécessaires et preuves d'événements, accès autorisé.

Ajouter un état de santé distinct de l'état d'alerte : démarrage, flux frais, surveillance dégradée ou indisponible. En cas de panne, ne pas afficher « pas de fuite » ni fermer implicitement l'incident. L'acquittement humain d'une notification ne signifie pas disparition du signal.

**Sortie :** source → traitement réel → timeline, avec arrêt du flux visible et provenance du replay. Commencer avec V0, sans attendre V1 ; pas de validation de qualité déduite.

### C — Entraîner et choisir un détecteur V1 sur fenêtres

**C1 — Score et contrat : réalisé et vérifié.** Pour chaque continuation de classe imposée, le modèle somme ses log-probabilités conditionnelles, puis applique un softmax stable sur les deux sommes. Même prompt et signal pour les deux candidats ; libellés exacts, espaces, tokenisation et terminaison de classe figés dans `scoring_spec.json`. L'explication qui suit n'entre pas dans le score. Sommes brutes, sans normalisation par longueur : une moyenne par token changerait la définition du score. Toute variante future demanderait une déclaration et un choix sur développement avant un nouveau gel. Preuves : [[V1 ML - exécution]].

Déclarer dans `threshold_rule` des probabilités relatives aux deux continuations, brutes et non calibrées, sans seuil ; ce n'est pas une probabilité terrain étalonnée. Aucun verdict converti artificiellement en 0/1, pourcentage généré, ni bruit ajouté pour donner l'apparence d'un score continu. Vérifier calcul/tokenisation, valeurs finies et reload sur développement. Le squelette Nevil fournit les IDs, pas leur signal : conserver un mapping explicite `clip_id` → entrée numérique, distinct du `sample_id` de l'API, sans appariement par ordre supposé ni labels/métadonnées dans le modèle. Conserver le contrat Safoan en concertation, sans faire de son UI une dépendance.

**C2 — V1 sur développement : réalisée.** Un parcours de 600 étapes sur les 598 clips train, huit époques ; trois candidats préannoncés aux époques 2, 4 et 8. Le checkpoint époque 4 / étape 300 est retenu selon la ROC-AUC de validation après médiane par groupe, avec priorité à l'époque la plus précoce en cas d'égalité. `n_configs_compared = 3` ; aucun réglage sur test. Entraînement `1199789f`, modèle `pipe-qwen3.5-4b-v1-1199789f-e4`, Qwen gelé inchangé ; encodeur/projecteur adaptés. Détails : [[V1 ML - exécution]].

Le v2 binaire avec bruit est conservé : 598 train, 208 validation, 194 test. Les 208 clips validation appartiennent à seulement 42 groupes de dépendance heuristiques, dont 8 non-fuite ; ce ne sont pas 208 observations indépendantes ni 42 sessions prouvées. Conserver la règle de recherche limitée et préannoncée ; toute variante ayant influencé un futur choix, y compris le scoring, doit être comptée honnêtement. Les essais mécaniques V0 restent tracés séparément, pas présentés comme une campagne d'évaluation.

Principe conservé pour une éventuelle campagne distincte : train pour apprentissage, validation pour sélection, et suivi séparé de classification, fidélité de description, invalidités, temps/mémoire et reproductibilité. LoRA, représentation différente ou modification du score seulement pour une faiblesse observée sur train/validation, dans un budget annoncé ; jamais à partir du test, de sa distribution de prédictions ou des résultats finaux Nevil. Aucune boucle de recherche prolongée pour gagner sur huit groupes négatifs. Une autre vue de données reste une expérience séparée à convenir, pas un remplacement de T0.

**C3 — Gel et T0 : réalisés, T0 publié.** Poids, preprocessing, prompt, libellés et scoring figés ; reload hors ligne dans un processus neuf vérifié. Bundle `checksums.json` SHA `b95569c50f5e1bbf3533bddc92530b5f285eb25dec999f83999f60ffd912d1ae`, exporteur `5a29d6e`. Livraison publiée dans [`artifacts/tslm_runs/tslm-v1/` à `186c45a`](https://github.com/IchamRaison/ehl-hackathon-zurich/tree/186c45a/artifacts/tslm_runs/tslm-v1) : uniquement `metadata.json` et `predictions.csv`, 402 IDs val/test uniques, exactement `clip_id,probability_leak`, aucun seuil. Contrôleur de conformité réussi ; 19 tests CPU passent. Aucun résultat final calculé côté ML.

Procédure conservée pour les exports reproductibles : réutiliser `python3 scripts/eval/check_run.py --template mon_run/` une seule fois dans un dossier neuf, car la commande écrase les deux fichiers si le dossier existe déjà. L'exporteur refuse un dossier préexistant, vérifie les SHA du split et de son audit de mapping, puis renseigne la provenance réelle. Pas de fold, label, device, pressure, flow, descriptions ou autres colonnes ; pas de lignes train.

Lancer `python3 scripts/eval/check_run.py --run mon_run/ --inspect` pour la conformité, sans moteur de métriques finales. Contrôler aussi les exigences strictes ci-dessus et l'identité du checkpoint : l'outil accepte certaines colonnes/lignes supplémentaires, des métadonnées d'exemple et ne fait qu'avertir sur une sortie binaire. Un code retour nul ne prouve donc pas une livraison scientifiquement exploitable. L'inspection mélange val/test : elle ne sert jamais à adapter le modèle, le score ou la distribution après gel. Ne pas remplacer les échecs par des probabilités fictives ; une erreur technique de livraison doit être tracée, pas devenir un prétexte au tuning. T0 a été publié sans attendre stress/application. Nevil applique seuils/agrégations et toutes les métriques finales selon `docs/MODEL_EVAL_CONTRACT.md` à `6dfdf63` ; aucun `metrics.json` final côté ML.

**C4 — Stress : exécutés, conformes et publiés à `7c04c9a`.** T1 inversion, T2 permutation de blocs de 250 échantillons (31,25 ms à 8 kHz), T3 randomisation de phase ; même bundle `b95569c5…`, exporteur `5a29d6e`, preprocessing et score qu'en T0, aucun réentraînement ni choix de configuration sur ces stress. Les trois runs ont terminé avec code retour nul, chacun validé sur 402/402 clips ; fichiers disponibles sur `feat/icham-tslm` dans `artifacts/tslm_runs/tslm-v1-T1/`, `-T2/` et `-T3/`. Empreinte du bundle identique après exécution. Aucun message n'a été envoyé à Nevil ; la publication des fichiers est vérifiée, pas leur prise en charge par lui.

Transformation officielle appliquée au signal brut puis preprocessing figé, sans aller-retour WAV qui requantifie ou écrête le signal transformé. T2 couvre les 32 blocs sans reste ; une permutation peut néanmoins laisser certains blocs à leur place. La conformité des exports ne constitue pas leur évaluation finale.

**Ancienne réserve T2/T3 résolue :** Nevil a remplacé le `hash()` Python salé par une dérivation SHA-256 canonique dans `b23601a`, reprise byte-identique depuis `6dfdf63`. La reproductibilité entre processus et une graine de référence sont testées ; aucune correction locale divergente ni choix arbitraire de `PYTHONHASHSEED`. Un run par transformation, provenance conservée. Sensibilité temporelle uniquement, pas mesure de surveillance continue. Historique de la découverte : [[Journal Icham#Revue des précisions Nevil — scores et stress]] ; état exécuté : [[V1 ML - exécution]].

**Sortie C réalisée :** checkpoint V1 autonome, exports T0/T1/T2/T3 vérifiés et publiés. Les résultats finaux sont attendus de Nevil ; aucun tuning sur son test déjà exposé ni nouvelle sélection après la remise. Un résultat final faible sera rapporté ; une nouvelle campagne demanderait un protocole distinct, pas une optimisation répétée sur ce test.

### D — Passer des scores aux événements

Icham + Nevil définissent, Safoan intègre : une règle causale versionnée de persistance et deux conditions distinctes d'ouverture/fin pour éviter les oscillations. États proposés : aucun événement actif → suspicion → alerte active → fin observée. Seuils et durées sont des paramètres à régler sur développement continu, pas une règle arbitraire présentée comme validée.

Un incident persistant produit un événement mis à jour, pas une nouvelle notification par fenêtre. Tester bruit bref, signal intermittent, suspicion persistante, retour durable au calme, fuite présente dès le démarrage, silence, coupure et reprise. Ne pas compter les fenêtres voisines comme des preuves indépendantes. Sans séquences annotées, ces essais prouvent uniquement la logique logicielle.

**Sortie :** événements cohérents/reproductibles avec source, horaires, versions, extraits justificatifs, état/raison et défauts de surveillance. La santé du flux reste distincte de la décision de fuite.

### E — Donner une utilité au langage et au dashboard

Safoan + Icham : appareil connecté et fraîcheur du signal, événement en cours, historique, extrait/spectrogramme, description et mesures distinctes. Une notification suffit pour ouvrir l'examen ; la description peut arriver après. Pas de cause physique, localisation précise, débit perdu ou réparation inventés. Le point d'installation du capteur est une métadonnée connue, pas une fuite localisée par le modèle.

Comparer le compte rendu TSLM à « baseline + mesures + gabarit » pour vérifier sa fidélité et sa compréhension par un destinataire. Une concordance texte/mesure ne démontre pas une explication causale de la décision.

**Sortie :** un responsable peut comprendre l'alerte et retrouver sa preuve ; aucune dépendance à un upload manuel pour la déclencher.

### F — Acquérir la preuve continue manquante

En parallèle dès A, Nevil + responsable du matériel à identifier : récupérer ou enregistrer sur un banc autorisé de longues séquences sans fuite, variations d'usage (vannes, pompes, débit), bruits, pertes de contact et fuites contrôlées avec début/fin annotés et incertitude indiquée. Inclure des sessions où la fuite existe déjà au démarrage. Ne pas créer de fuite sur une installation occupée pour le test.

Conserver identifiants de session/installation/capteur, étalonnage et horodatage. Diviser par acquisition/installation avant fenêtrage ; toute adaptation locale emploie uniquement un passé autorisé et une référence normale vérifiée. Garder des acquisitions réservées, sans mélange de fenêtres voisines entre développement/test. Le droit d'accès et la rétention des enregistrements restent à convenir, notamment si un capteur peut capter des conversations.

**Sortie :** corpus temporel avec durée connue et événements annotés. Les clips actuels restent utilisables pour leur question initiale ; ne pas leur inventer une continuité.

### G — Évaluer le système complet et livrer

Nevil, modèles Icham/Vincent : geler modèle, preprocessing et politique d'alerte, puis mesurer sur acquisitions réservées : rappel par événement, faux événements par appareil-heure réellement surveillée, délai depuis apparition annotée, répétitions, disponibilité/couverture et pertes de données. Rapporter durée totale, nombre d'événements, sessions et capteurs, avec limites d'incertitude. Les temps inconnus ne valent pas des heures négatives correctement surveillées.

Évaluer baseline et TSLM avec la même définition d'événement, les mêmes données et un protocole comparable de choix des seuils. Distinguer scores de classification, stress T1–T3, comportement de la règle d'alerte et évaluation sur continu réel. Aucune extrapolation « fausses alertes/jour » depuis un simple FPR par clip.

Équipe : intégrer le checkpoint figé, tester reprise/panne/reload, répéter la démo et publier preuves/licences/configurations. Si seules des séquences artificielles sont disponibles, présenter une démo de surveillance en replay et une évaluation séparée sur clips, pas un dispositif terrain validé. Vérifier deadline/format/acceptation du replay auprès des organisateurs ; les exigences TimeNet et entraînement TSLM restent maintenues.

## 5. Ordre de travail et répartition

**Volet prioritaire Icham — livraison ML autonome terminée :** C1 score vérifié, C2 campagne limitée terminée, C3 gel/reload/T0 publié ; C4 stress exécutés, conformes et publiés avec le RNG officiel corrigé. Ne pas rouvrir la sélection du checkpoint après T0. Capteur réel, frontend fini ou import complet du harness n'étaient pas des prérequis de cette livraison.

**Volet produit en parallèle :** A critères/capteur + F acquisitions, et B replay/appareil→fenêtres→V0 pour Safoan avec interfaces convenues. Construire ensuite D/E (événements, santé et dashboard), mesurer le débit soutenu et remplacer V0 par le checkpoint V1 figé. Une alerte persistante, une disparition et une coupure doivent être traitées sans scénario qui injecte la bonne réponse. Diagnostics de texte sur développement ; évaluation finale de texte/événements à organiser avec Nevil, hors CSV T0.

**Convergence :** Nevil produit les résultats du benchmark T0/T1–T3 à partir de nos fichiers ; l'équipe relie ces preuves à la bonne version de modèle dans la démo. G pour les résultats continus réels seulement si les acquisitions F le permettent. Sinon : démonstration fonctionnelle en replay + benchmark sur clips présentés séparément. Pas de réglage du modèle ou de politique d'alerte sur le test final pour embellir la démo.

Répartition : Icham modèle/score/exports et support d'intégration ; Nevil contrat, validation et métriques finales, données/stress ; Safoan flux, UI et alertes ; Vincent baseline et export comparable. Capteur/collecte et contraintes opérationnelles : Icham avec responsable matériel à identifier. Aucun changement de propriétaire de code sans concertation.

**Prochaine étape : Nevil évalue les quatre exports disponibles sur `feat/icham-tslm` à `7c04c9a`.** Aucun message ne lui a été envoyé par l'agent. Entraînement, reload et inférences C1 à C4 ne sont pas à refaire. Le volet produit reste à coordonner avec [[Plan simulateur de capteur]] ; cette livraison ne prouve pas son intégration à V1. Pas de nouveau boîtier, flotte distribuée, application mobile, achat matériel ni commande de vanne. Objectifs de délai/fausses alertes, responsable matériel, deadline hackathon et budget opérationnel restent à confirmer en parallèle ; aucun délai de réalisation garanti avant mesure.

## Références et limites

État prouvé : [[V0 ML - exécution]], [[V1 ML - exécution]], [[Journal Icham#Vérification des nouveaux livrables Nevil]]. Le nouveau scénario vient explicitement d'Icham ; ce n'est pas une inférence depuis les ressources disponibles.

Référence courante : [contrat `6dfdf63`](https://github.com/IchamRaison/ehl-hackathon-zurich/blob/6dfdf63580bc15ce6a1cf0817b9da3569553aca1/docs/MODEL_EVAL_CONTRACT.md). Score continu et limitation de recherche retenus ; pas de chantier de calibration pour T0. Nuance conservée : l'invariance de rang par transformation strictement croissante au niveau clip ne garantit pas celle de l'agrégation médiane sur des groupes de taille paire ; ne pas affirmer que seul le Brier peut changer. Les pertes rapportées de C1 sont du même ordre que l'effet étudié, pas une pénalité mesurée pour notre TSLM. Justification historique et limites de `check_run.py` dans [[Journal Icham#Revue des précisions Nevil — scores et stress]].

Référence méthodologique consultée : [DCASE 2022 — évaluation des événements sonores](https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments#evaluation) distingue classe et localisation temporelle, avec scénarios de réactivité et contrôle des faux positifs. Cela étaye la séparation événement/classification, pas la faisabilité de détection de fuites ni des valeurs de seuil pour PIPE. Aucun nouveau dataset DCASE adopté.
