# EHL Hackathon Zurich

Icham

## Commencer ici

[[Passation]] donne l'état réel. [[Évaluation qualité V1 - exécution]] donne les résultats mesurés ; [[Plan surveillance continue]] distingue l'ambition produit du prototype. [[Plan directeur agents]] conserve le plan initial et les contraintes communes. Aucune lecture du chat n'est nécessaire.

**Goal élargi le 13 septembre : [[Diagnostic causal TSLM vs C1#Plan global en dix étapes]].** Diagnostic → corrections mesurées → diversification globale si nécessaire → mesure de cet apport → gel et confirmation indépendante → intégration continue et valeur du TSLM. Le diagnostic ne termine plus le goal. [[Diagnostic causal - exécution]] : D0 complet/vérifié, D1 précision de tête testé/préinscrit, prochaine action A puis C sans fit ; étapes 6 à 10 non lancées. Aucun gain final ni cause unique du retard sur C1 démontrés ; pas de reprise automatique de l'ancienne recette V2.

PIPE vise un appareil qui écoute les canalisations en continu et produit des alertes avec preuves acoustiques consultables. V1 entraînée/rechargée, exports T0–T3 publiés et évaluation complète exécutée par Icham. Résultat : détecteur encore insuffisant, sans gain démontré face au contrôle C1 ; descriptions de bandes globalement correctes mais parfois incohérentes avec le score. [[Évaluation qualité V1 - exécution]]. Surveillance et fiabilité terrain non validées.

**Objectif final précisé par Icham le 13 septembre :** aider à comprendre et investiguer l'évolution d'un événement, au-delà de « fuite ou pas fuite ». Démontrer séparément l'apport du langage et celui de l'accès aux séries, en comparant classifieur + DSP + gabarit, classifieur + mesures/contexte + Qwen, et TSLM. Ce sont des capacités à construire et tester, pas des résultats actuels. [[Plan V2 - fiabilité et parité des scores#7. Objectif final — démontrer la valeur du TSLM]].

## Votre mission

- [[Agent Icham - ML]] : modèle, entraînement et intégration finale.
- [[Agent Nevil - Data]] : données, TimeNet, DSP, split et évaluation.
- [[Agent Safoan - Application]] : API, audio/spectrogramme et interaction.
- [[Agent Vincent - Baseline]] : Random Forest, tests et exports.

Lire le contexte commun AVANT de commencer sa fiche. [[Équipe et répartition]] résume les responsabilités et dépendances.

## Documentation commune

- [[Brief et contraintes]] et [[Tableau de bord]]
- [[Architecture]] et [[Contrats techniques]]
- [[Protocole évaluation]]
- [[Coordination et passation agents]] et [[Continuité du projet]]
- [[Démo et pitch]]
- [[Décisions]], [[Expériences]], [[Ressources]]
- Recherche détaillée : [[PIPE - proposition ML et démo]]

## Suivi individuel

[[Journal Icham]] · [[Journal Nevil]] · [[Journal Safoan]] · [[Journal Vincent]]

Chaque agent actualise ses résultats, preuves, blocages et prochaine action après une avancée significative, sans attendre qu'Icham le demande. Un travail décrit n'est pas un travail réalisé.

## Accès

Notes : https://github.com/IchamRaison/ehl-hackathon-zurich-vault
Code : https://github.com/IchamRaison/ehl-hackathon-zurich

Ouvrir la racine du dépôt notes comme vault Obsidian. Pas de sync automatique entre postes/dépôts. Aucun secret, donnée client ou code produit dans ce dépôt.
