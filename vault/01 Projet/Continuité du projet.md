# Continuité du projet

## Règle permanente demandée par Icham

Pendant tout le projet, mettre continuellement à jour ce vault. Une autre personne ou un autre agent doit pouvoir comprendre immédiatement l'avancée et reprendre le travail sans accéder à la conversation d'origine.

Cette mise à jour fait partie du travail, sans attendre une demande supplémentaire.

## À la reprise

Lire [[Accueil]], [[Passation]] et [[Tableau de bord]], puis les notes liées à la tâche. Vérifier l'état réel du dépôt avant de poursuivre.

## Après chaque avancée significative

Mettre à jour les notes après une décision, une implémentation, un test, un blocage ou un changement de direction, et avant de rendre la main.

- [[Passation]] : état actuel, dernier résultat, travail inachevé, blocages et prochaine action concrète.
- [[Tableau de bord]] : déplacer les tâches selon leur état réel.
- [[Décisions]] : choix, raisons, alternatives écartées et conséquences.
- [[Expériences]] : commandes, résultats observés, preuves et limites.
- [[Architecture]] : changements réels de composants et de fonctionnement.
- Journal du jour : historique court des avancées, avec liens vers les notes et commits pertinents.

Actualiser l'état courant plutôt qu'empiler des résumés contradictoires. Garder l'historique dans le journal. Inclure les mises à jour documentaires dans les commits concernés lorsque possible.

## Critère de qualité

Le lecteur doit pouvoir répondre rapidement à ces questions :

1. Que construit-on et pourquoi ?
2. Qu'est-ce qui fonctionne réellement et quelle preuve le montre ?
3. Qu'est-ce qui est en cours, non testé ou bloqué ?
4. Quels choix ont été faits et pourquoi ?
5. Quelle est la prochaine action et comment la lancer ?

Distinguer explicitement prévu, implémenté, testé et vérifié. Mentionner les fichiers, commandes, commits ou URLs nécessaires à la reprise. Ne jamais inventer une preuve, recopier des secrets ou présenter un test non exécuté comme réussi.

## Partage du vault

Dépôt dédié : https://github.com/IchamRaison/ehl-hackathon-zurich-vault

Le dépôt dédié est la référence pour les notes partagées. Dossier local : `/home/animus/ehl-hackathon-zurich-vault`. Avant chaque mise à jour, récupérer et intégrer les changements des collaborateurs. Mettre à jour et pousser ce dépôt après les avancées significatives. La copie `vault/` du dépôt code est un miroir à actualiser explicitement, jamais une source à recopier aveuglément sur les modifications de l’équipe. Il n’y a pas de synchronisation automatique en arrière-plan.

## Portée

Cette règle concerne ce projet. Elle est aussi rappelée dans `AGENTS.md` à la racine du dépôt pour les agents qui le consultent. Elle ne constitue pas une synchronisation automatique en arrière-plan.
