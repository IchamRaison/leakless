# Plan C1 — suivi temporel et notifications

## Statut et objectif

**Autorisation reçue ensuite le 13 septembre :** Icham demande l'implémentation et autorise la réaffectation d'Aghashahi. Il exclut tout test WhatsApp et précise une livraison par endpoint sur le GPU. [[C1 temporel - exécution]] porte le nouveau protocole et les résultats réels ; les paragraphes ci-dessous conservent le plan initial. Le repli `195.242.28.46` reste protégé.

13 septembre 2026 : plan demandé par Icham, **non exécuté**. Trois responsabilités : C1 mesure/classe l'acoustique ; un suivi causal consolide les observations en événements ; un gabarit ou Qwen restitue les faits. Aucun entraînement, changement d'API ou envoi WhatsApp autorisé par cette seule planification.

Deux livraisons : **M1**, replay avec vraie inférence C1, suivi simple et notification de test ; **M2**, comparaison du LSTM sur des séquences admissibles et adoption uniquement si son apport est démontré. M1 ne dépend ni du LSTM ni d'un LLM.

Icham précise que `195.242.28.46` héberge un endpoint placeholder où il prévoit de mettre la V1 en repli. Ne pas confondre cette déclaration avec une V1 déjà déployée/testée. URL complète, route, port, contrat et authentification non fournis ; aucun appel ni contrôle SSH dans ce tour. Préserver ce service et ne plus considérer la machine comme libre sur la base d'un ancien relevé GPU. L'autre H100 reste candidate pour le futur essai, après vérification de disponibilité.

## 0 — Protéger le repli et cadrer les interfaces

- Garder checkpoint/service V1 et résultats historiques intacts. Nouveau travail isolé, sans remplacement automatique du placeholder ni arrêt d'un service existant.
- Avant raccordement : relever avec Icham/Safoan le contrat réel de l'endpoint et de l'application, sans deviner une URL depuis l'IP. Secrets par configuration privée, jamais dans les notes ou Git.
- Versionner séparément détecteur C1, suivi temporel et restitution. Une réponse de C1 ne doit pas être étiquetée « TSLM V1 ». Un changement de détecteur ouvre une nouvelle session de suivi ; ne pas mélanger silencieusement les scores/états.
- Premier périmètre : un canal, audio enregistré explicitement déclaré replay, fenêtre non chevauchante d'une seconde à 8 kHz. Le capteur réel est une intégration ultérieure conditionnée au domaine physique et au format.

**Preuve attendue :** contrat d'entrée/sortie et procédure de retour au repli écrits, endpoint existant inchangé.

## 1 — Rendre C1 utilisable sur chaque fenêtre

Réutiliser `scripts/eval/harness/features.py:c1_envelope` et `scripts/tslm/v2_c1.py` : neuf descripteurs d'amplitude/enveloppe, scaler et régression logistique. Ne pas modifier les descripteurs gelés ni emprunter le spectrogramme d'affichage comme entrée ML. La sélection interne existante retient `C=0.01` ; ne pas relancer une recherche de paramètres.

Charger un checkpoint existant seulement s'il est complet et vérifié ; sinon prévoir un unique fit final au C déjà choisi, sur les 598 train autorisés, avec provenance explicite. Les helpers sklearn existants restent sur CPU ; ce petit fit n'est pas un entraînement Qwen sur CPU. Aucun portage CUDA de C1 prévu.

Sortie : les neuf mesures, le score continu `probability_leak`, la version et l'identité de fenêtre. Horodatage, session et qualité restent dans l'orchestrateur, pas des variables prédictives telles que label, nom de fichier, pression ou site. Score brut, pas probabilité terrain calibrée. Silence, NaN, saturation et données manquantes sont des défauts de qualité à traiter explicitement, jamais une réponse « sans fuite » par défaut.

**Preuve attendue :** mêmes mesures/scores par voie offline et replay, reload neuf reproductible, erreurs contrôlées et débit soutenu mesuré à la cadence choisie. Aucun résultat de généralisation déduit de ces tests mécaniques.

## 2 — Qualifier les séquences avant de choisir la supervision LSTM

Ce volet documentaire/data peut avancer pendant la construction de M1, après autorisation d'exécution.

- Zenodo actuel : clips d'une seconde ; concaténation possible pour tester le logiciel, pas chronologie physique reconstituée.
- Aghashahi : candidat séquentiel concret. La page primaire décrit des signaux de 30 secondes, des hydrophones à 8 kHz et certaines variations de débit à la seconde 20. **Ce changement de débit n'est pas une annotation d'apparition de fuite.** Vérifier acquisition, correspondances capteurs, ordre et étiquettes des sources originales. [Source](https://data.mendeley.com/datasets/tbrnp6vrnj/1).
- Aghashahi est actuellement une réserve de confirmation. Recommandation à discuter : un nouveau protocole pourrait en utiliser des acquisitions pour apprendre le LSTM, en renonçant à présenter cette partie comme confirmation indépendante et en désignant une réserve distincte. Aucun changement de rôle automatique ; Hong Kong reste candidat à auditer, pas réserve déjà garantie.
- Couverture visée : fuite et sans fuite, variations normales, bruits/intermittences, plusieurs acquisitions/installations ; fenêtres/modalités de la même expérience toujours du même côté du split, avant fenêtrage. Pas de jeu de confirmation déjà exploité pour sélectionner le modèle.

**Décision de passage :** avec séquences et bornes fiables, apprentissage/évaluation événementiels ; avec seulement une classe globale par enregistrement, essai de classification séquentielle limité à cette cible, sans inventer des bornes ou labels par instant ; sans séquences autorisées, M1 continue mais aucun LSTM métier prétendument validé. Les fichiers synthétiques testent le logiciel et restent séparés des preuves terrain.

## 3 — Construire le suivi simple de référence et M1

Une politique causale de persistance avec conditions distinctes d'ouverture et de fermeture : absence d'événement actif → suspicion → alerte active → fin observée. Seuils et durées documentés comme paramètres de développement, pas réglés sur confirmation. Leur valeur métier reste à convenir ; le seuil 0,5 du benchmark ne devient pas automatiquement un seuil d'alarme.

- Utiliser uniquement les fenêtres disponibles ; garder heure source, heure de réception et compteur de séquence. Dédupliquer par session/séquence, pas seulement par hash audio.
- Séparer santé du flux et état de l'événement. Une interruption rend la surveillance dégradée, ne clôt pas la fuite et ne compte pas comme continuité observée. Reprise/session nouvelle : réinitialiser le contexte causal et signaler l'incertitude.
- Conserver `first_observed_at`, `confirmed_at`, `last_observed_at`, durée effectivement observée et couverture. Ne pas dater l'apparition physique avant le début de l'acquisition. L'ouverture peut confirmer une suspicion antérieure, avec les deux dates distinctes.
- Une alerte persistante conserve son identifiant ; pas de nouvelle notification à chaque seconde. Conserver un minimum d'état durable pour éviter les répétitions après redémarrage.

Réutiliser le simulateur `feat/sensor-replay` pour cadence/pertes/doublons et les composants du Monitor de Safoan pour l'affichage. Attention : son replay actuel boucle des mesures, **pas des prédictions ni un suivi d'événements** ; il ne prouve pas plusieurs minutes d'acquisition et ne fournit aucune alerte modèle à réutiliser telle quelle.

**Preuve attendue :** tests déterministes bruit bref, activité persistante, intermittence, retour au calme, anomalie dès démarrage, silence, coupure, doublon et redémarrage. Puis vraie inférence C1 pendant un replay fini, sans injecter les classes attendues depuis le scénario.

## 4 — Entraîner un petit LSTM causal, après le passage données

Proposition initiale, à figer avant fit : dix entrées par instant (neuf mesures + score C1), une couche unidirectionnelle de 32 unités, tête binaire ; historique de départ jusqu'à 30 secondes si le corpus qualifié le permet. PyTorch déjà installé, précision FP32, une H100 disponible, aucun entraînement distribué ou nouveau LoRA. [API LSTM du runtime 2.8](https://docs.pytorch.org/docs/2.8/generated/torch.nn.LSTM.html).

Le LSTM produit un score, pas une durée ni un message. Il alimente la même gestion d'événement que la référence. Contexte borné, initialisation et traitement des trous identiques entre apprentissage et service ; jamais de futur ni d'état partagé entre acquisitions. Tester notamment que modifier le futur ne change aucune sortie antérieure.

Précautions de supervision : BCE sur les cibles réellement disponibles, suivi NLL/erreurs et gradients, masquage du padding ; pas de perte textuelle. Avec labels globaux seulement, score/loss au niveau séquence, pas propagation arbitraire d'un label à chaque seconde. Budget d'époques, graine, règle d'arrêt et checkpoint retenu fixés avant comparaison ; une architecture candidate au départ, tous essais/choix comptés.

Prévenir la fuite d'information entre C1 et LSTM : solution simple si possible, C1 gelé appris sur un corpus distinct des séquences LSTM. Sinon scores d'entraînement produits hors groupes par cross-fitting à l'intérieur du train uniquement ; ni scaler, ni C1, ni LSTM n'utilisent les labels de validation/confirmation. Ne pas fournir au LSTM des scores artificiellement parfaits produits sur les exemples d'apprentissage de C1 sans examiner ce biais.

**Preuve attendue :** apprentissage mécanique sur petit sous-ensemble, reload neuf et parité batch/stream, puis résultats sur acquisitions séparées. Mémorisation seule insuffisante. Sans gain vérifié, conserver M1.

## 5 — Comparer le gain acoustique et le gain temporel

Comparer sur les mêmes acquisitions, couverture et protocole de sélection : C1 fenêtre seule, C1 + persistance, C1 + LSTM + même gestion d'événement. Ajouter un petit contrôle utilisant les mêmes dix entrées et la même fenêtre d'historique mais agrégées sans ordre : éviter d'attribuer au temps un gain provenant seulement de l'accès aux neuf mesures. Ce contrôle reste distinct du C1 historique.

Si annotations événementielles disponibles : événements manqués, fausses alertes par durée réellement surveillée, délai de confirmation, répétitions, couverture et latence logicielle. Fixer avant comparaison la tolérance d'appariement alerte/événement, le compromis faux positifs/délai et le gain minimal utile. Une seule alerte ne peut pas créditer plusieurs fuites ; les événements déjà présents au démarrage ont un délai depuis l'observation, pas une origine physique connue. [Référence sur classe versus localisation temporelle](https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments#evaluation).

Si labels globaux seulement : métriques de classification au niveau acquisition et portée annoncée ; pas de délai d'apparition de fuite ni d'extrapolation de fausses alertes/jour depuis un FPR de clips. Rapporter effectifs, durée négative, incertitude par acquisition et limites ; pas compter les fenêtres voisines comme des observations indépendantes. Une différence minuscule/incertaine ne suffit pas à promouvoir le LSTM.

**Décision :** retenir le LSTM seulement s'il améliore le compromis défini sur développement puis résiste à la confirmation réservée. Sinon C1 + persistance reste la version produit candidate. Tout échec de confirmation est publié, pas suivi d'un réglage sur cette réserve.

## 6 — Restituer les faits et préparer WhatsApp

Événement structuré : identifiant, appareil/session, versions, début observé, confirmation, durée/couverture, statut et qualité. Le serveur calcule ces faits et décide de notifier ; Qwen ne reçoit ni autorité de déclenchement ni possibilité d'inventer une durée ou une cause.

M1 : gabarit déterministe, par exemple « Un signal compatible avec une fuite est observé depuis X. Une vérification est recommandée. » Mention replay pour une démonstration. Pas de débit perdu, localisation de fuite ou confiance calibrée sans mesure correspondante.

Qwen optionnel : reformulation/résumé à partir des seuls faits validés, chiffres/date/statut rendus par le serveur, retour au gabarit si sortie invalide, indisponibilité ou timeout. Aucun fine-tuning 27B nécessaire pour ce rôle ; restitution hors chemin critique de l'alerte. Comparer fidélité et compréhension au gabarit avant de la conserver.

WhatsApp : préparer l'adaptateur de notification dans le backend existant, pas un agent autonome. D'abord journal/aperçu sans envoi ; puis destinataire de test autorisé, expéditeur et accès fournis par le propriétaire. Identifiant d'événement pour déduplication, état envoyé/échec, reprise bornée et traitement d'un résultat d'envoi incertain selon les possibilités du fournisseur. Une coupure réseau ne doit pas produire une rafale de doublons. Aucun envoi réel dans cette planification.

## 7 — Intégrer, figer et présenter honnêtement

Coordonner l'interface avec Safoan : remplacer les valeurs de démonstration uniquement là où une vraie inférence est raccordée ; montrer score, événement, durée observée et santé du flux. Le mode replay et le détecteur actif restent visibles. Tester score réel → événement → aperçu notification → erreur/reprise, puis envoi autorisé séparément ; mesurer débit, file bornée et p95 sans promettre le temps réel avant mesure.

Livraison : artefacts C1 et éventuellement LSTM avec empreintes, preprocessing partagé, politique d'alerte, provenance/splits et évaluations, scénario reproductible, commande de lancement et retour au repli V1. Garder `195.242.28.46` intact tant qu'une bascule explicite n'est pas décidée. L'archivage/audit final du rush27B reste un travail distinct inachevé ; ne pas relancer ses fits.

Pour le hackathon, conserver TimeNet et les expériences TSLM déjà réalisées dans leur rôle réel. Ce pipeline de détection séquentielle avec notification n'est pas automatiquement un TSLM entraîné ; vérifier l'acceptation du cadrage avec les organisateurs. La démo produit et les résultats TSLM comparés doivent être présentés séparément, sans promettre que la nouvelle architecture satisfait seule le brief.

## Ordre pratique et décisions avant exécution

0 → 1 → 3 → 6 en mode test → première intégration M1. Audit 2 en parallèle de ce travail indépendant ; si admissible/autorisé, 4 → 5 → intégration M2. La disponibilité de vraies séquences et des accès WhatsApp conditionne ces branches, pas la première boucle locale.

À valider : contrat du placeholder ; rôle d'Aghashahi ou autre corpus séquentiel ; critères opérationnels et budget d'essais ; destinataire/accès de notification ; propriétaire de l'intégration et acceptation hackathon. Aucun accès secret à demander dans le chat public. Aucune durée totale garantie avant ces vérifications.

## Éléments vérifiés pour préparer ce plan

Lecture seule du code `ec73e5f` : `c1_envelope`, helpers C1 et `selected_C=0.01` dans `docs/evidence/tslm-v2/campaign-c13fd47/C1/comparison.json`. Branches récupérées sans fusion : `feat/demo-temporal-building` à `5beb344`, `feat/sensor-replay` à `81c987f`. Docs/API/replay et code `frontend/src/demo/replay.ts` lus ; ancien dossier `/tmp/ehl-sensor-replay` absent, source Git disponible. Historique Entire retrouve les chantiers, dont `110012a` (plan/replay) et `a4665aa` (Monitor). Aucun test, fit, inférence ou intégration rejoué dans cette planification. Sources web relues ; pas de nouvelle acquisition ni réaffectation de dataset exécutée.
