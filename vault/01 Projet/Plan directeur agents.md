# Plan directeur agents

Icham

## 0. Lire ceci avant d'exécuter

Ce vault doit suffire pour reprendre le projet sans historique de chat. PIPE est désormais la direction de travail demandée par Icham pour répartir le développement. Sa faisabilité scientifique reste soumise aux portes de validation ci-dessous. Ce document est un plan, pas un compte rendu de code exécuté.

Ordre de lecture obligatoire : [[Passation]] → cette note → [[Brief et contraintes]] → [[Architecture]] → [[Contrats techniques]] → [[Protocole évaluation]] → votre fiche de rôle → [[Coordination et passation agents]]. Les quatre fiches sont [[Agent Icham - ML]], [[Agent Nevil - Data]], [[Agent Safoan - Application]], [[Agent Vincent - Baseline]].

En cas de contradiction, les derniers résultats prouvés dans [[Passation]] décrivent l'état réel ; le présent plan et les contrats définissent le travail attendu. Les anciennes notes d'idéation sont historiques. Un résultat réel qui invalide le plan doit entraîner une décision documentée, pas être dissimulé.

## 1. Projet en une minute

PIPE est un atelier d'analyse acoustique pour la détection de fuites de canalisations. L'utilisateur importe ou choisit un court enregistrement de capteur. Le système montre le son et son spectrogramme, exécute un Time-Series Language Model entraîné, prédit fuite/non-fuite et décrit des propriétés acoustiques mesurables. Le jury peut comparer à une baseline et observer l'effet d'un bruit ajouté de façon contrôlée.

Question métier : ce signal acoustique présente-t-il les caractéristiques d'une fuite dans le domaine étudié ? Utilité visée : aider un technicien à relire et documenter un enregistrement. Utilité terrain non validée. Un membre du groupe possède une entreprise de détection de fuites et fabrique ses capteurs ; son identité, ses unités et la compatibilité de ses appareils ne sont pas établies. Ne pas supposer que ce membre est Vincent.

Le projet n'est ni une carte de localisation de fuite, ni un dispositif diagnostique certifié, ni une mesure du volume perdu, ni une prédiction de rupture. Aucun branchement aux capteurs personnels n'est requis. Le nom PIPE est provisoire, disponibilité commerciale non recherchée.

## 2. Contraintes organisateurs

Temporal AI Challenge : utiliser TimeNet pour préparer des données temporelles ouvertes ; entraîner ou fine-tuner un TSLM ; comparer à une baseline sur des données réservées sans contamination ; montrer entrées réelles, sorties, preuves et limites.

Livrables : démo fonctionnelle, code + configuration d'entraînement, checkpoint/adapter, documentation du dataset, évaluation courte avec baseline, présentation devant le jury. Budget annoncé : bon Nebius de 1 000 USD par équipe, activation/quotas non vérifiés. Deadline, durée du pitch et validation explicite de notre cadrage acoustique restent à confirmer. Voir [[Brief et contraintes]]. Ne pas supprimer TimeNet ou l'entraînement TSLM pour gagner du temps tout en prétendant satisfaire le challenge.

## 3. Ce qui existe vraiment

- Deux dépôts privés existent. Le dépôt code contient le bootstrap Git/Entire et de la documentation ; aucun code produit PIPE vérifié à la rédaction de ce plan.
- Le vault partagé et les recherches sont poussés.
- La source Zenodo a été consultée ; l'archive des fuites a été téléchargée/listée et contient 500 WAV. Les trois catégories n'ont pas encore été intégralement auditées ensemble.
- Aucun split final, baseline, entraînement TSLM, score, endpoint ou UI PIPE n'est encore validé.
- Entire a été activé en mode manual-commit dans le dépôt code ; capture d'une session non testée. Les réglages ne suffisent pas à prouver une capture effective.

Tout chemin de code et commande PIPE décrit dans les autres notes est une cible d'implémentation, à créer et tester. Les installations exactes doivent être vérifiées dans les sources amont et figées, pas devinées.

## 4. Répartition

Icham : cœur TSLM, GPU, apprentissage et intégration finale. Nevil : ingestion, TimeNet, signal, splits et évaluation indépendante. Safoan : application/API, son/spectrogramme et interaction. Vincent : Random Forest scikit-learn, tests et exports, chantier technique guidé par les données de Nevil.

Les obligations organisateurs et le pitch sont partagés ; Icham nomme un responsable pour chacune au démarrage. Chaque agent maintient son journal, aucun ne délègue toute la documentation à Vincent.

## 5. MVP impératif et options

P0 obligatoire : audit data/split ; connecteur TimeNet ; baseline ; entraînement TSLM effectif et rechargement ; endpoint réel ; lecture audio/spectrogramme ; comparaison chiffrée ; documentation/reproduction.

P1 après P0 : mélange de bruit à SNR réglable, score de décision et abstention calibrée si l'évaluation le justifie, test de changement d'appareil, descriptions multi-propriétés, replay de secours clairement étiqueté.

P2 à exclure par défaut : carte géographique, application mobile native, streaming matériel, GNN, multi-datasets fusionnés, agent conversationnel généraliste, réparation recommandée automatiquement. Aucun nouveau chantier ne doit retarder la boucle P0.

## 6. Portes de validation

G0 — données, Nevil : archives lisibles, provenance/licence, groupes documentés, effectifs par split et absence de doublons croisés. Si l'indépendance reste incertaine, documenter et faire arbitrer Icham ; ne pas annoncer un test hors-session/hors-site propre.

G1 — contrat, Icham + Nevil + Safoan + Vincent : un échantillon réel traverse la transformation partagée ; formes, labels, versions et schéma de réponse sont figés. Les quatre agents utilisent la même fixture de contrat, sans partager les labels au modèle.

G2 — preuve d'apprentissage, Icham : mini entraînement sur train, loss/logs, checkpoint sauvegardé, chargement dans un processus neuf et vraie inférence. Test d'overfit minuscule autorisé pour debug, explicitement distinct de l'évaluation.

G3 — boucle produit, Safoan : clip réel → backend → modèle → affichage ; label de démo révélé séparément ; une erreur contrôlée est gérée sans faux résultat.

G4 — évaluation, Nevil : versions figées ; baseline et TSLM sur même test ; résultats au niveau des groupes et limites ; performances faibles publiées sans les embellir.

G5 — livraison, équipe : environnement reproductible, URLs d'artefacts accessibles au jury selon les règles, checkpoint vérifié, note de licence, vidéo/replay étiqueté, répétition et soumission vérifiée.

Pas de promesse de terminer un entraînement en un nombre d'heures avant mesure. Prioriser G0/G2 en parallèle pour découvrir tôt les vrais blocages. Si la donnée ne permet pas une évaluation défendable, changer de source ou de cadrage avec Icham ; une simulation déclarée nécessite accord des organisateurs pour leur critère d'entrées réelles.

## 7. Reprise pratique

Dépôt code : https://github.com/IchamRaison/ehl-hackathon-zurich
Dépôt notes : https://github.com/IchamRaison/ehl-hackathon-zurich-vault

Sur cette machine : `/home/animus/ehl-hackathon-zurich` et `/home/animus/ehl-hackathon-zurich-vault`. Sur une autre machine, cloner les deux dépôts avec les permissions de l'utilisateur et employer ses propres chemins. Si accès refusé, demander une invitation, jamais les mots de passe dans le chat.

Lire les règles Git et les autorisations dans [[Coordination et passation agents]]. Ne pas lancer de machines payantes avant de confirmer le compte, le crédit et le plafond opérationnel avec Icham.
