# Passation

Icham

## Entire — état vérifié sur Linux

CLI 0.10.6 et 12 skills installés ; capture Codex locale constatée. La recherche distante de commits fonctionne désormais (audit/splits de Nevil retrouvés), contrairement au contrôle initial non authentifié. Cela ne prouve pas la publication des transcriptions. Trois hooks Codex restent signalés à approuver via `/hooks`. État initial : [[Entire - installation et vérification]] ; nouvelle preuve dans [[Journal Icham]].

## Dernier échange — 2026-09-12

Icham rappelle que les datasets ont déjà été identifiés. Correction importante après recherche Entire et `git fetch origin` : Nevil a publié l'audit des 1 000 WAV et `split_v2` sur `nevil/setup` (`3efa08f`). `split_v1` est invalide et ne doit pas servir ; manifeste v2 présent, SHA-256 vérifié, audit/invariants rapportés mais non réexécutés ici. Pas de conversion TimeNet ni entraînement dans cette livraison. Reprendre ces artefacts, pas recommencer la recherche de source ; revue commune requise, notamment bruit inclus dans le négatif v2 versus protocole du vault. H100 accessible en SSH déjà vérifiée ; aucun nouveau calcul distant ou code ML lancé. Détails, liens et limites : [[Journal Icham]].

## Cadrage précédent

Icham demande une critique structurelle, la première réponse étant trop méthodologique. [[Challenge du cadrage de Nevil]] précise l'avis : direction hackathon soutenue, produit à resserrer ; le WHY de surveillance permanente ne correspond pas au WHAT d'analyse d'un extrait choisi. Recommandation à discuter : seconde lecture pour un technicien, avec bénéfice de qualification/documentation à confronter à son travail réel. Nouvelle note distante [[PROBLEM STATEMENT - LeakLess Temporal AI]] intégrée (`0a70a95`) : direction déclarée approuvée en équipe sous réserve de l'audit ; ses trois classes restent à aligner avec les contrats binaires. Aucun contrat ni chantier technique modifié. G0 et entretien métier peuvent avancer en parallèle ; preuves dans [[Journal Icham]].

## Reprise Safoan — 2026-09-12

Branche `feat/safoan-app` créée depuis `3ea769d` et publiée. Entire 0.10.6 installé sur le Mac de Safoan, hooks Git valides et hooks Codex installés ; approbation via `/hooks` encore requise. Capture réelle non vérifiée. Aucun code application démarré. Détails et prochaine action S0 : [[Journal Safoan]].

## Nom proposé

Nevil propose le nom LeakLess et confirme le cadrage software-only acoustique, sous réserve de l’audit. Avis favorable avec maintien obligatoire de TimeNet et entraînement TSLM ; validation finale d’Icham non encore enregistrée. Voir [[Proposition Nevil - LeakLess]]. Le plan PIPE reste applicable, sans renommer ses références avant décision.

## État courant

Direction de travail : PIPE, assistant d'analyse acoustique de fuites avec TimeNet, TSLM réellement entraîné, baseline et démo audio/spectrogramme. Icham a demandé un plan détaillé pour les quatre agents. La direction est organisée ; la faisabilité data/ML reste à valider aux portes G0/G2 de [[Plan directeur agents]].

Documentation et répartition prêtes. Aucun modèle PIPE entraîné ni score PIPE vérifié. Le checkout principal inspecté reste consacré à l'outillage et aux notes ; `origin/nevil/setup` est désormais à `3efa08f`, avec audit, scripts de groupage/vérification et manifestes v2, non intégrés au checkout principal. Les artefacts ont été lus, pas rejoués sur les WAV. Une conversion TimeNet des données acoustiques n'est pas livrée dans cette branche. L'agent n'a ni provisionné de GPU ni lancé d'entraînement ; il a seulement vérifié la machine fournie par Icham.

## Lire pour reprendre sans le chat

[[Plan directeur agents]] → [[Contrats techniques]] → [[Protocole évaluation]] → [[Équipe et répartition]] → sa fiche dans 06 Agents. [[Coordination et passation agents]] régit les branches et mises à jour. Les commandes et chemins PIPE proposés sont à implémenter, pas à considérer comme déjà fonctionnels.

## Répartition et premières actions

- Icham : [[Agent Icham - ML]] ; vérifier accès GPU/base, runtime et premier batch. Journal : [[Journal Icham]].
- Nevil : [[Agent Nevil - Data]] ; auditer les trois archives et groupes ; fixture réelle et split. Journal : [[Journal Nevil]].
- Safoan : [[Agent Safoan - Application]] ; figer Prediction avec l'équipe, afficher un vrai WAV. Journal : [[Journal Safoan]].
- Vincent : [[Agent Vincent - Baseline]] ; script Random Forest + tests, puis entraînement sur features de Nevil. Journal : [[Journal Vincent]].

Prise en main documentaire d'Icham terminée ; jalons techniques encore à vérifier. Pitch/règles organisateurs sont partagés avec responsable à désigner par Icham.

## Preuves déjà disponibles

Source dataset https://zenodo.org/records/18631450 : 500 fuite, 386 sans fuite, 114 bruits. L'audit publié de Nevil confirme 1 000 WAV mono, 8 kHz, PCM 16 bits, une seconde. `split_v2` publié : 598 train, 208 validation, 194 test, 185 groupes heuristiques ; ses invariants ne prouvent pas une indépendance de sessions réelles. Mesures expérimentales, pas preuve chez des clients. Voir [[Journal Icham]] et [[PIPE - proposition ML et démo]].

Dépôts privés créés et poussés : code https://github.com/IchamRaison/ehl-hackathon-zurich ; notes https://github.com/IchamRaison/ehl-hackathon-zurich-vault. Bootstrap Entire au commit 373d13f ; activé manual-commit, télémétrie et push automatique de sessions désactivés. Capture Codex locale désormais constatée ; publication/indexation distante non vérifiées. Voir [[Entire - installation et vérification]].

## Inconnues et risques

- Indépendance des événements/captures, duplications et groupes à établir avant scores.
- Accès modèles de base, compatibilité TimeNet/OpenTSLM et coût/mémoire à mesurer.
- Acceptation explicite du cadrage acoustique, deadline, durée du pitch et accès/crédit Nebius à confirmer.
- TSLM peut ne pas battre la baseline ou ne pas bénéficier de l'ordre temporel ; l'évaluation doit le montrer.
- Capteurs du coéquipier non intégrés et pas nécessaires au MVP. Unités/grandeurs non vérifiées.

## Dépôts et état historique

Sur cette machine : /home/animus/ehl-hackathon-zurich pour le code, /home/animus/ehl-hackathon-zurich-vault pour les notes. Sur un autre poste, cloner les dépôts avec accès autorisé. Le vault dédié est la référence ; le miroir vault/ du code peut être ancien.

Les notes débit/pression et comparaisons de pistes sont conservées comme historique. Elles ne remplacent pas le plan acoustique courant. Mettre cette note à jour au prochain résultat réel, sans accumuler des décisions anciennes sous « état courant ».
