# Brief et contraintes

Icham

## Sources et statut

Brief Temporal AI Challenge transmis par Icham le 2026-09-12. Copie du PDF `Aionic_Temporal_AI_Hackathon.pdf` dans `assets/` : couche texte extraite et 22 pages inspectées visuellement lors de la reprise documentée dans [[Journal Icham]]. Les pages physiques 14, 16 et 17 décrivent mission, étapes et livrables ; 19–21 le soutien Nebius ; 22 les ressources. Aucune deadline de soumission ni durée de pitch indiquée. La page Notion générale reste non consultée.

## Challenge

Agentic Systems Lab × Aionic Labs × Nebius — Temporal AI Challenge, « Give AI a Sense of Time ».

Utiliser TimeNet d'Aionic pour préparer des séries temporelles ouvertes, puis entraîner ou fine-tuner un TSLM reliant les signaux temporels au langage pour une tâche utile.

## Parcours demandé

1. Problème et données : choisir un utilisateur et un besoin ; trouver des données ouvertes. Une recherche, sélection ou validation agentique des datasets est encouragée.
2. Connexion : utiliser ou construire un connecteur TimeNet pour signaux, métadonnées et annotations. Définir questions, labels ou cibles d'apprentissage.
3. Entraînement et évaluation : entraîner ou fine-tuner un TSLM et le comparer à une baseline sur des données réservées. Éviter les fuites entre périodes, sujets ou appareils.
4. Démonstration : montrer la provenance des données, des entrées réelles, les sorties du modèle, l'utilité, les preuves et les limites.

Le PDF indique un profil idéal de données associant séries temporelles et descriptions ou raisonnement textuel. Il envisage la génération d'annotations par LLM dans certains cas ; cela n'en garantit pas la qualité.

## Livrables obligatoires

- [ ] Démo fonctionnelle
- [ ] Code et configuration d'entraînement
- [ ] Checkpoint ou adapter
- [ ] Documentation du dataset
- [ ] Évaluation courte avec comparaison à une baseline
- [ ] Présentation en direct : problème, approche, résultats et enseignements

## Jury et bonus

Critères : utilité du problème, qualité de préparation des données, entraînement TSLM adéquat, rigueur de l'évaluation, clarté de la démo.

Bonus : datasets bien choisis, connecteurs TimeNet réutilisables et pipelines agentiques de recherche, évaluation, sélection, récupération ou validation de données.

## Compute et soutien

Bon Nebius annoncé de 1 000 USD par équipe. Crédits supplémentaires annoncés pour les équipes les plus prometteuses après dimanche. Accès TimeNet, compute, OpenTSLM comme référence et accompagnement technique Aionic/ASL via Discord.

Crédit non activé ni vérifié sur un compte. GPU disponible, quotas, coût horaire et procédure d'activation à confirmer. Aucun lancement payant effectué.

## À confirmer auprès des organisateurs

- [ ] Deadline exacte, fuseau et plateforme de soumission
- [ ] Durée du pitch et format de la démo
- [x] Composition de l'équipe et rôles documentés dans [[Équipe et répartition]]
- [ ] Accès TimeNet et activation Nebius
- [ ] Contraintes de licence des livrables et modèles

## Direction de travail et choix encore ouverts

Direction courante : PIPE, aide à l'analyse acoustique de fuites pour un technicien, classification fuite/non-fuite et observations mesurables. Dataset candidat Zenodo 18631450, à valider par l'audit G0. Répartition, architecture cible et métriques sont décrites dans [[Plan directeur agents]] et [[Protocole évaluation]] ; versions, modèle de base et faisabilité restent à vérifier. LeakLess est un nom proposé, sans validation finale enregistrée. Santé, industrie et énergie restent des exemples du brief.

## Conséquences pour notre plan

Recommandations de travail, distinctes des exigences organisateurs :

- Valider très tôt une boucle ingestion TimeNet → entraînement court → inférence → évaluation.
- Fixer les groupes train/validation/test avant fenêtrage et ajuster les transformations sur le train uniquement.
- Choisir une tâche dont la baseline et la métrique peuvent être calculées pendant le hackathon.
- Réserver du temps au checkpoint, à la reproductibilité et à la démo, pas seulement à l'interface.

Voir [[Ressources]], [[Tableau de bord]] et [[Passation]].
