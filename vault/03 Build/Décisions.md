# Décisions

Icham

## 2026-09-13 — Livraison C1, LSTM non promu et endpoint GPU

Icham autorise explicitement le plan C1 et la réaffectation Aghashahi, sans WhatsApp, par endpoint sur le GPU. Nouveau protocole publié avant fit : BR train/validation, LO confirmation, capteurs d'une même condition ensemble ; cible globale30s, jamais onsets inventés. Préparation/résultats historiques intacts. Les signaux à pleine échelle restent inclus dans l'expérience descriptive mais provoquent l'abstention du service, différence de couverture publiée.

Critère fixé avant comparaison non atteint : LSTM AUC groupe0,375/0,590278 contre contrôle sans ordre0,8125/1,0. Ne pas promouvoir le LSTM, recalibrer sur confirmation ou prétendre avoir validé la détection d'événement. Conserver C1/persistance comme démonstration, Qwen optionnel non ajouté et notifications limitées au gabarit/aperçu. La correction TF32 restaure la précision du fit, sans modifier poids ni tolérance de reload. [[C1 temporel - exécution]] et preuves `8baecab`.

Service isolé première H100, loopback8019/SSH, SQLite durable, pas de frontend modifié ni d'accès au repli195. Le raccordement Monitor reste côté application ; acceptation du cadrage hackathon et validation terrain ne sont pas revendiquées.

## 2026-09-13 — Étendre le goal au-delà du diagnostic

Icham demande explicitement d'ajouter les étapes 6, 7, 8, 9 et 10 au vault et au goal général, puis précise que le prompt `/goal` est un renvoi vers le vault. [[Diagnostic causal TSLM vs C1#Plan global en dix étapes]] devient donc le périmètre complet : corrections et comparaison, diversification globale si nécessaire, mesure de son apport, confirmation indépendante, intégration continue et test de valeur des trois approches. La fin du diagnostic seule n'est plus la fin du goal.

Les étapes sont incluses avec leurs conditions et preuves, pas réputées exécutées. Aucune reprise automatique de l'ancienne campagne, réaffectation de la réserve Aghashahi, dépense ou action physique n'est impliquée par cet ajout. Le premier contrôle restant demeure D1 ; chaque campagne ultérieure exige un protocole et un budget annoncés avant calcul. [[Diagnostic causal - exécution]] conserve la checklist et l'état réel.

## 2026-09-13 — Diagnostiquer l'écart avant une nouvelle recette

Icham propose un nouvel objectif de diagnostic causal : données, représentation, supervision, optimisation, utilisation du signal par Qwen et scoring. [[Diagnostic causal TSLM vs C1]] cadre cette proposition ; le protocole détaillé et son budget restent à convenir, aucune nouvelle expérience lancée. Préférer « expliquer l'écart observé entre les pipelines » à l'affirmation préalable « le TSLM exploite moins bien les données ». C1 n'est pas une vérité terrain ; la parité réparée ne démontre pas une amélioration de discrimination.

La suite d'implémentation V2 est en pause, sans refit automatique. Conserver les résultats de l'ancienne campagne et les hypothèses non départagées ; ne pas déclarer l'objectif atteint ni réécrire les résultats historiques. Les trois approches du benchmark produit ci-dessous restent le cap final, pas la prochaine recette à lancer.

## 2026-09-13 — Valeur finale du langage et des séries à démontrer séparément

Icham demande explicitement d'ajouter cet objectif au vault et de reprendre V2 : comprendre l'évolution d'un événement, exploiter les signaux/contextes réellement disponibles et permettre une investigation interactive. Un score + bande DSP + phrase fixe ne démontrent pas l'intérêt d'un TSLM. Le benchmark final comparera trois approches, sans présumer laquelle doit gagner : classifieur/DSP/gabarit ; classifieur/mesures/contexte/Qwen ; TSLM/séries/contexte. [[Plan V2 - fiabilité et parité des scores#7. Objectif final — démontrer la valeur du TSLM]].

Les étapes V2 de fiabilité/détection continuent ; leur campagne A/C ne remplace pas ce benchmark. Si Qwen sur mesures fait aussi bien que le TSLM, aucune valeur propre de l'accès aux séries n'est démontrée. Si le gabarit fait aussi bien, pas de valeur du langage démontrée non plus. Aucune nouvelle modalité, collecte privée ou installation n'est implicitement autorisée.

Choix technique de campagne après diagnostic train : A et C seulement, B omise faute de preuve que le texte domine la perte au checkpoint audité. C ajoute neuf descripteurs C1 mesurés au prompt, sans remplacer les quatre séries ni ajouter labels/métadonnées. Recette commune proposée : quatre époques, seed fixe, lots effectifs de huit avec microbatches de un ; préinscription et nouveaux gates A/C requis avant les six fits comparatifs. Preuves et limites : [[V2 ML - exécution]].

## 2026-09-12 — Répartition confirmée pour la livraison au harness

Message de Nevil transmis par Icham : la partie ML livre les probabilités T0 et leur provenance selon le contrat à `08562e3`, **pas les métriques finales**. Nevil possède l'évaluation sur le split gelé. T1/T2/T3 seulement après T0 si possible, checkpoint identique, aucun réentraînement. Le contrôle d'export (format, couverture, bornes, hash/provenance) ne doit pas être confondu avec le calcul de performances sur test. Voir [[Protocole évaluation]]. Cette instruction prépare une livraison future ; aucun run nouveau autorisé par « regarde ce que Nevil m'a dit ».

## 2026-09-12 — Cas d'usage corrigé par Icham : écoute continue

Décision produit explicite : un appareil écoute les tuyaux en permanence et signale une fuite ; un humain reçoit/examine l'alerte, il ne lance pas l'analyse par upload. [[Plan surveillance continue]] remplace le scénario de relecture manuelle. Les choix un canal, serveur, replay initial, score régulier/texte sur événement et logique de persistance sont des propositions techniques à valider, pas des décisions utilisateur déjà prises.

Conséquences : conserver la V0 comme brique par fenêtre ; ajouter ingestion, santé du flux, suivi d'événement et alertes ; acquérir du continu annoté pour prouver les métriques opérationnelles. TimeNet, entraînement TSLM et baseline restent requis. Aucune autorisation nouvelle de matériel, dépense, collecte privée, notification externe ou commande physique ; aucune implémentation nouvelle dans ce tour.

## 2026-09-12 — Organisation initiale

- Dépôt privé `IchamRaison/ehl-hackathon-zurich`.
- Notes Obsidian versionnées dans `vault/` avec le code.
- Aucun plugin communautaire requis.
- Entire activé en mode manual-commit.
- Télémétrie et push automatique des sessions désactivés.
- Capture des sessions Hermes non vérifiée.

## Organisation courante — plan agents

Le vault dédié remplace la copie du dépôt code comme référence des notes partagées. Direction de travail PIPE acoustique ; faisabilité soumise à G0/G2, pas encore une validation ML. Icham prend TSLM/GPU, Nevil data/TimeNet/protocole, Safoan API/UI, Vincent baseline scikit-learn/tests/exports. Pitch et obligations organisateurs sont partagés.

[[Plan directeur agents]], [[Contrats techniques]], [[Protocole évaluation]] et les quatre fiches agents constituent le plan courant. Pas de localisation géographique, pas de matériel obligatoire. Toute dérogation majeure doit préciser motif, preuve, responsable et impacts sur le brief.

## Prochaines décisions

Utiliser [[Modèle décision]] pour les choix qui engagent le projet.

## 2026-09-12 — Plan ML validé, démarrage différé

Icham valide [[Plan de session Icham - première version TSLM]], puis précise explicitement de ne pas commencer. Périmètre accepté : compte rendu court en une passe, classe et une propriété vérifiable initialement, V0 technique intégrable puis V1 évaluée et améliorations mesurées. Ni réservation GPU ni entraînement autorisé à ce stade. Les classes, la propriété exacte, la base et la configuration d'exécution restent conditionnelles aux vérifications prévues. Discussion GPU en cours dans [[Journal Icham]] ; une H100 n'est pas encore sélectionnée ou disponible par le seul fait d'être proposée.

## 2026-09-12 — Méthode de résolution par le pourquoi

À la demande de Safoan, appliquer la méthode du Cercle d'or de Simon Sinek aux
prochains problèmes : formuler d'abord le **pourquoi** (cause, utilisateur et
changement recherché), expliciter ensuite le **comment** (principes, contraintes
et preuve attendue), puis choisir le **quoi** (fonction ou implémentation).

Application à PIPE :

- **Pourquoi** : aider un technicien à relire et documenter un signal acoustique
  inhabituel afin de décider s'il mérite une investigation, sans promesse de
  diagnostic terrain.
- **Comment** : relier le même signal à l'écoute, aux visuels et à l'inférence ;
  séparer mesures DSP, sorties du modèle et labels ; publier les limites et une
  preuve reproductible ; ne jamais inventer une performance.
- **Quoi** : studio audio/spectrogramme, prédiction TSLM et comparaison baseline,
  dans cet ordre de dépendance.

Cette méthode guide le cadrage, les arbitrages d'interface et le pitch. Elle ne
remplace ni l'audit dataset, ni les contrats, ni les tests. Source : transcription
fournie par Safoan de la conférence « How great leaders inspire action ».
