# Décisions

Icham

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
