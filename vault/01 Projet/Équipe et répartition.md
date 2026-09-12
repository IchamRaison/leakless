# Équipe et répartition

Icham

Le plan complet à transmettre à chaque agent commence par [[Plan directeur agents]]. Le travail ci-dessous est attribué, non encore exécuté/vérifié.

## Icham — Cœur ML et intégration

Fiche : [[Agent Icham - ML]]. Propriétaire TSLM, entraînement GPU, checkpoint et arbitrages. Livrable : modèle adapté rechargeant réellement, fonction d'inférence et configuration reproductible. Difficulté la plus élevée. Dépend des exemples/transformations de Nevil ; fournit predict à Safoan et modèle figé à Nevil.

## Nevil — Données, TimeNet et preuves

Fiche : [[Agent Nevil - Data]]. Propriétaire ingestion, groupes/splits, DSP partagé, connecteur TimeNet et évaluation finale. Livrable : données documentées, transformations versionnées, baseline/TSLM comparés sans contamination connue. Fournit séries à Icham, features à Vincent et clip/bruit à Safoan.

## Safoan — Application et API

Fiche : [[Agent Safoan - Application]]. Propriétaire UI, API et contrat de sortie commun. Livrable : démo sonore/visuelle reliée aux vrais modèles, erreurs et éventuel replay explicites. Consomme les fonctions d'Icham/Vincent et le DSP de Nevil.

## Vincent — Baseline technique cadrée

Fiche : [[Agent Vincent - Baseline]]. Propriétaire Random Forest scikit-learn, sauvegarde/rechargement, prédiction, tests et exports. Livrable : comparateur solide utilisable par l'API/évaluation. Dépend des features/splits de Nevil. Pas de responsabilité sur l'architecture TSLM ou les décisions de split.

## Première synchronisation

Nevil + Icham fixent forme [C,T], cibles et normalisation ; Vincent valide la matrice de features ; Safoan propose Prediction. Lire [[Contrats techniques]]. Ne pas lancer quatre implémentations incompatibles.

## Responsabilités transverses

Pitch, règles et soumission sont partagés ; Icham désigne le responsable opérationnel au démarrage. Chacun documente son travail, pas uniquement Vincent. Les journaux personnels servent de passation. Les portes G0–G5, revues et règles de blocage sont détaillées dans [[Plan directeur agents]] et [[Coordination et passation agents]].
