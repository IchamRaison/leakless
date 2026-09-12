# Décisions

Icham

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
