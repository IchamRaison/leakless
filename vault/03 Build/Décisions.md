# Décisions

Icham

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

## 2026-09-12 — Méthode de résolution par le pourquoi

À la demande de Safoan, appliquer le Cercle d'or : formuler d'abord le **pourquoi** (cause, utilisateur et changement recherché), expliciter le **comment** (principes, contraintes et preuve attendue), puis choisir le **quoi** (fonction ou implémentation).

Application à PIPE :

- **Pourquoi** : aider un technicien à relire et documenter un signal acoustique inhabituel afin de décider s'il mérite une investigation, sans promesse de diagnostic terrain.
- **Comment** : relier le même signal à l'écoute, aux visuels et à l'inférence ; séparer mesures DSP, sorties du modèle et labels ; publier les limites et une preuve reproductible ; ne jamais inventer une performance.
- **Quoi** : studio audio/spectrogramme, prédiction TSLM et comparaison baseline, dans cet ordre de dépendance.

Cette méthode guide le cadrage, les arbitrages d'interface et le pitch. Elle ne remplace ni l'audit dataset, ni les contrats, ni les tests. Source : transcription fournie par Safoan de la conférence « How great leaders inspire action ».
