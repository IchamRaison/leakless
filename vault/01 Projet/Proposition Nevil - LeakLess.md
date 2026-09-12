# Proposition Nevil - LeakLess

Icham

## Proposition transmise

Nevil propose LeakLess software-only pour le hackathon : dataset acoustique Zenodo, génération de descriptions texte, baseline, protocole anti-leakage et démo claire, sans revendication de validation terrain. L'engagement dépend de l'audit du dataset.

## Avis de cadrage

Option recommandée, cohérente avec le plan acoustique PIPE. LeakLess est le nouveau nom proposé ; aucun changement de périmètre technique nécessaire. La validation finale par Icham n'est pas encore enregistrée : il demande s'il faut valider.

Conditions à expliciter dans l'accord :

- Utiliser réellement TimeNet et entraîner/fine-tuner un TSLM ; génération de descriptions seule ou classifieur + LLM rédacteur ne suffit pas au brief.
- L'audit doit confirmer provenance/licence, labels, groupes de captures et séparation défendable des données, pas uniquement la possibilité de lire les WAV.
- Descriptions générées à partir de propriétés mesurables ; déclarer leur nature générée, aucune cause physique inventée ni texte cible injecté dans les entrées d'inférence.
- Baseline et TSLM sur les mêmes splits ; résultats mesurés, invalidités et limites visibles.
- Mesures réelles sur site expérimental, aucune validation terrain/client revendiquée ; mode bruit explicitement synthétique.

Prochaine porte : Nevil rend le rapport d'audit G0 et ses limites. Icham peut vérifier le runtime TSLM en parallèle ; ne pas attendre l'audit complet pour tester mécaniquement le chargement, sans interpréter ce test comme validation ML.

Voir [[Plan directeur agents]], [[Protocole évaluation]] et [[Agent Nevil - Data]].
