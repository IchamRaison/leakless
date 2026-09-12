# Coordination et passation agents

Icham

## Source de vérité et permissions

Le vault partagé est la référence des notes, le dépôt code celle de l'implémentation. Ne jamais recopier la vieille copie vault/ du repo code par-dessus les notes collaboratives. Les scripts, datasets, poids et installations vont dans le dépôt code ou ses répertoires runtime ; ce dépôt contient uniquement documentation/config Obsidian et ressources documentaires.

Avant de travailler : lire AGENTS.md, [[Passation]], [[Plan directeur agents]], contrats et fiche personnelle. Confirmer identité du rôle, dépôt, branche et changements locaux. Vérifier accès GitHub sans révéler de tokens. Si nécessaire demander une invitation, pas des identifiants en chat.

Sur un clone propre : git status, git fetch origin, git pull --ff-only sur main, puis branche de travail. Si changements locaux/divergence, ne pas reset/stash sans comprendre le propriétaire ; préserver et résoudre explicitement. Un agent par clone/worktree et branche pour éviter changements de branche concurrents. Les chemins locaux sont des exemples, pas une obligation pour une autre machine.

## Propriété et revues

Icham : TSLM/config training/bootstrap Python ; Nevil : data/signal/eval et contrats de features ; Safoan : API/frontend et schémas Prediction ; Vincent : baseline/config/tests. Consulter [[Architecture]] pour les chemins cibles.

Changement pyproject/lockfile → concertation Icham. Changement preprocess ou split → Nevil + consommateurs. Changement API Prediction → Safoan + Icham/Vincent/Nevil. Un agent ne modifie pas une interface partagée puis attend que les autres découvrent la casse.

Branches suggérées : feat/icham-tslm, feat/nevil-data, feat/safoan-app, feat/vincent-baseline. Faire des commits étroits, tests et PR avec preuves ; Icham intègre, Nevil relit la méthode ML et Safoan les interfaces UI. Ne pas push des poids/données/secrets ni forcer une branche partagée. Pas de merge du code d'un autre sans lire diff/tests.

Pour le vault, chacun édite principalement son journal et ses notes, sur une branche docs personnelle si plusieurs agents simultanés. Les mises à jour de Passation/Tableau de bord sont coordonnées pour éviter conflits. PR/rebase prudents, jamais écrasement silencieux. Un push n'est validé qu'après lecture du commit/chemin distant.

## Démarrage parallèle

Icham vérifie runtime/modèle/GPU pendant que Nevil audite les fichiers. Safoan définit le contrat et affiche un vrai clip ; Vincent prépare son entraînement baseline et tests. Aucun ne doit attendre une interface parfaite pour avancer mécaniquement.

Les fixtures de développement doivent porter execution_mode=development_fixture. Elles prouvent uniquement une interface/test, jamais une performance. Dès que le premier vrai échantillon arrive, remplacer les fixtures sur le flux P0.

## Message de livraison standard

Rôle et jalon ; commit/branche/PR ; changements ; entrées/versions attendues ; commande exacte exécutée ; résultat observé ; artefacts et checksum ; limitations ; demande à l'autre rôle ; prochaine action. Éviter « tout marche » sans preuve. Copier ce résumé dans le journal personnel avant de rendre la main.

En cas de blocage : opération, erreur exacte sans secret, reproduction minimale, tentatives, impact, action nécessaire et responsable. Ne pas attendre en silence, ne pas contourner le protocole de test pour débloquer un score.

## Mise à jour continue

À chaque jalon, décision, test significatif ou blocage : mettre à jour journal, tâches et notes techniques concernées. Mettre à jour Passation si direction/interface/état global change. Les demandes actuelles du responsable humain priment sur un plan ancien ; documenter les corrections plutôt que garder deux versions contradictoires.

Ne pas réécrire le journal d'un autre pour lui attribuer une réussite. Ne pas marquer une tâche terminée parce qu'un document la décrit. Les tâches « prêt pour intégration », « intégré » et « validé » sont distinctes.

## Budget et sécurité

Icham confirme l'allocation GPU et l'arrêt. Les agents ne créent pas chacun une instance. Surveiller temps, stockage et coût réel depuis console/outils autorisés ; pas d'hypothèse que 1 000 USD sont activés. Secrets par environnement/gestionnaire adapté, jamais git/vault/logs. Contrôler licence avant redistribution des poids et WAV. Données de l'entreprise seulement sur autorisation explicite et sans exposer les clients.

## Résolution d'un conflit

Préserver chaque contribution, lire le contrat concerné, reproduire les deux comportements et demander arbitrage à Icham si choix architectural. Résoudre sans déplacer les labels test ou supprimer les tests qui échouent pour faire passer l'intégration.
