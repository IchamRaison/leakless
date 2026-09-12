# Passation

Icham

## Dernière reprise — 2026-09-12

Icham s'est identifié dans la session et a demandé la lecture complète du vault. Les 34 fichiers Markdown, réglages Obsidian et 22 pages du PDF ont été lus ; rôle ML/intégration repris. Aucun entraînement ni test produit exécuté pendant cette lecture. Preuves et prochaine action I0 dans [[Journal Icham]].

## Reprise Safoan — 2026-09-12

Branche `feat/safoan-app` créée depuis `3ea769d` et publiée. Entire 0.10.6 installé sur le Mac de Safoan, hooks Git valides et hooks Codex installés ; approbation via `/hooks` encore requise. Capture réelle non vérifiée. Aucun code application démarré. Détails et prochaine action S0 : [[Journal Safoan]].

## Nom proposé

Nevil propose le nom LeakLess et confirme le cadrage software-only acoustique, sous réserve de l’audit. Avis favorable avec maintien obligatoire de TimeNet et entraînement TSLM ; validation finale d’Icham non encore enregistrée. Voir [[Proposition Nevil - LeakLess]]. Le plan PIPE reste applicable, sans renommer ses références avant décision.

## État courant

Direction de travail : PIPE, assistant d'analyse acoustique de fuites avec TimeNet, TSLM réellement entraîné, baseline et démo audio/spectrogramme. Icham a demandé un plan détaillé pour les quatre agents. La direction est organisée ; la faisabilité data/ML reste à valider aux portes G0/G2 de [[Plan directeur agents]].

Documentation et répartition sont prêtes. Aucun modèle PIPE entraîné, score PIPE, split final ou application PIPE n'est vérifié à ce stade. Après `git pull --ff-only`, `main` du code reste à `3ea769d` avec bootstrap/Entire/docs, sans modules produit. La branche distante `nevil/setup` (`c47dc96`) contient de la documentation et trois scripts non implémentés ; elle n'est pas intégrée. Son README annonce TimeNet opérationnel sur le poste de Nevil, non reproduit ici. Aucun lancement GPU payant effectué par cette reprise.

## Lire pour reprendre sans le chat

[[Plan directeur agents]] → [[Contrats techniques]] → [[Protocole évaluation]] → [[Équipe et répartition]] → sa fiche dans 06 Agents. [[Coordination et passation agents]] régit les branches et mises à jour. Les commandes et chemins PIPE proposés sont à implémenter, pas à considérer comme déjà fonctionnels.

## Répartition et premières actions

- Icham : [[Agent Icham - ML]] ; vérifier accès GPU/base, runtime et premier batch. Journal : [[Journal Icham]].
- Nevil : [[Agent Nevil - Data]] ; auditer les trois archives et groupes ; fixture réelle et split. Journal : [[Journal Nevil]].
- Safoan : [[Agent Safoan - Application]] ; figer Prediction avec l'équipe, afficher un vrai WAV. Journal : [[Journal Safoan]].
- Vincent : [[Agent Vincent - Baseline]] ; script Random Forest + tests, puis entraînement sur features de Nevil. Journal : [[Journal Vincent]].

Prise en main documentaire d'Icham terminée ; jalons techniques encore à vérifier. Pitch/règles organisateurs sont partagés avec responsable à désigner par Icham.

## Preuves déjà disponibles

Source dataset https://zenodo.org/records/18631450 consultée ; archive fuite listée avec 500 WAV. Source annonce 1 000 clips au total sur trois catégories, licence CC BY 4.0. Le reste de l'audit est à faire. Mesures sur site expérimental, pas preuve chez des clients. Voir [[PIPE - proposition ML et démo]].

Dépôts privés créés et poussés : code https://github.com/IchamRaison/ehl-hackathon-zurich ; notes https://github.com/IchamRaison/ehl-hackathon-zurich-vault. Bootstrap Entire au commit 373d13f ; activé manual-commit, télémétrie et push automatique de sessions désactivés. Capture effective d'une session non vérifiée.

## Inconnues et risques

- Indépendance des événements/captures, duplications et groupes à établir avant scores.
- Accès modèles de base, compatibilité TimeNet/OpenTSLM et coût/mémoire à mesurer.
- Acceptation explicite du cadrage acoustique, deadline, durée du pitch et accès/crédit Nebius à confirmer.
- TSLM peut ne pas battre la baseline ou ne pas bénéficier de l'ordre temporel ; l'évaluation doit le montrer.
- Capteurs du coéquipier non intégrés et pas nécessaires au MVP. Unités/grandeurs non vérifiées.

## Dépôts et état historique

Sur cette machine : /home/animus/ehl-hackathon-zurich pour le code, /home/animus/ehl-hackathon-zurich-vault pour les notes. Sur un autre poste, cloner les dépôts avec accès autorisé. Le vault dédié est la référence ; le miroir vault/ du code peut être ancien.

Les notes débit/pression et comparaisons de pistes sont conservées comme historique. Elles ne remplacent pas le plan acoustique courant. Mettre cette note à jour au prochain résultat réel, sans accumuler des décisions anciennes sous « état courant ».
