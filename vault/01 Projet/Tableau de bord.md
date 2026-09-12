# Tableau de bord

Icham

## Prêt

- [x] Dépôts privés code et vault créés.
- [x] Brief, sources et piste acoustique documentés.
- [x] Quatre rôles techniques définis.
- [x] Plan directeur, contrats, protocole et fiches agents rédigés.
- [x] Icham : exécution des étapes 1 à 5 autorisée ; V0 technique avant V1 évaluée.
- [x] Icham : SSH distant et présence d'une H100 80 Go vérifiés ; inventaire initial puis V0 exécutée dans [[Journal Icham]].
- [x] Livrables Nevil retrouvés sur `nevil/setup` : audit des trois archives, scripts et `split_v2` ; SHA-256 du manifeste vérifié, exécution de l'audit non reproduite ici.
- [x] Icham : lecture complète du vault et des 22 pages du PDF ; état des dépôts inspecté, preuves dans [[Journal Icham]].

- [x] Safoan : branche `feat/safoan-app` publiée ; Entire installé et hooks Git vérifiés. Approbation des hooks Codex et capture réelle encore à faire : [[Journal Safoan]].

## En cours et à vérifier

- [x] Icham autorise l'implémentation V1 complète jusqu'à T0/T1–T3 ; runtime H100 et données existantes revérifiés. [[V1 ML - exécution]].
- [ ] En cours : scoring continu, scripts de campagne/reload/export et tests ; trois checkpoints annoncés aux époques 2/4/8, sélection validation seule. Aucun entraînement V1 lancé au démarrage documenté.
- [x] Réserve de graine T2/T3 résolue par Nevil : SHA-256 officiel depuis `b23601a`, version reprise `6dfdf63`. Contrôle interprocessus et exécution des stress encore à réaliser.

- [x] Cadrage explicite Icham : appareil en écoute continue et alertes automatiques, remplacement du scénario d'import manuel. [[Plan surveillance continue]] rédigé ; aucune nouvelle implémentation.
- [ ] Valider le plan continu : capteur/flux, destinataire, délai visé, fausses alertes tolérées, responsable matériel et périmètre replay/réel.
- [ ] Safoan/Icham : flux/replay horodaté, santé de surveillance, événements persistants et tableau d'alertes ; extension de contrat à convenir.
- [ ] Nevil/responsable matériel : acquisitions continues annotées, sessions séparées et métriques événementielles ; aucune preuve opérationnelle déduite des clips.
- [x] Icham : environnement Python/CUDA isolé ; calcul et backward H100 réussis. [[V0 ML - exécution]].
- [x] Icham : reprises sélectives Nevil `1289095` / contrat Safoan `4dd7b88` ; données/TimeNet rejoués sur H100.
- [x] Icham : Qwen 3.5-4B choisi après demande utilisateur, téléchargé anonymement et décodeur chargé sur H100 ; Llama abandonné.
- [x] Icham : trois archives téléchargées/intégrité vérifiée ; tests CPU numériques et frontières d'entrée réussis.
- [x] Livraison Nevil `nevil/temporal-evidence` (`08562e3`) retrouvée et lue : contrat d'export, moteur d'évaluation, C0–C3, T0–T3, rapports. Tests/résultats non reproduits ici ; branche non intégrée. [[Journal Icham#Vérification des nouveaux livrables Nevil]].
- [x] Consigne de livraison Nevil transmise : Icham fournit seulement les prédictions T0 et leur provenance ; Nevil exécute l'évaluation finale. [[Journal Icham#Consigne de livraison T0 et stress — Nevil]].
- [x] Ajouts Nevil `9135754` relus : score continu, contrôleur de conformité existant, petite validation groupée. Plan révisé sans exécution ; réserves longueur/calibration et reproductibilité T2/T3 documentées. [[Journal Icham#Revue des précisions Nevil — scores et stress]].
- [ ] Icham : vérifier les log-probabilités sur développement, préannoncer une campagne initiale d'au plus trois configurations et renseigner le nombre réel comparé ; aucun choix fondé sur le test.
- [ ] Icham : vérifier interprocessus puis exécuter les stress officiels corrigés `6dfdf63` après T0, même checkpoint et preprocessing ; conserver leurs empreintes.
- [ ] Icham : score de classe vérifié, V1 figée puis export T0 conforme sur les 402 clips val/test v2 ; contrôles de contrat sans scoring final. T1/T2/T3 ensuite si possible, même checkpoint, aucun réentraînement.
- [ ] Icham/Nevil : convenir du suivi de développement validation seule ; T0 respecte le binaire/v2 gelé avec bruit, pas de nouvelle vue implicite. Diagnostics texte séparés, métriques finales chez Nevil ; aucune intégration complète du harness requise pour exporter.
- [ ] Prochain bloc Icham proposé : C1 score/export → C2 V1 → C3 gel/reload/T0 ; T1–T3 ensuite si possible. Application/flux et collecte en parallèle, pas prérequis de T0. [[Plan surveillance continue]].
- [ ] G0 Nevil/Icham : revue commune de l'audit et du `split_v2` publiés, avec leurs réserves et alignement du protocole. Ne pas utiliser `split_v1` (invalide).
- [ ] Après G0, Icham/équipe : figer problème et classes ; arbitrer comparaison baseline + mesures/gabarit et test d'utilité proposés dans [[Challenge du cadrage de Nevil]].
- [ ] G1 Tous : fixture réelle et contrats gelés.
- [x] G2 mécanique Icham : runtime, entraînement court, checkpoint complet et reload neuf hors ligne, puis environnement reconstruit. [[V0 ML - exécution]].
- [ ] Vincent : baseline entraînée sur développement, tests et export.
- [ ] G3 Safoan : audio/spectrogramme et vraie inférence de bout en bout.
- [ ] G4 Nevil : évaluation finale commune après gel des modèles.
- [ ] G5 Équipe : checkpoint livré, reproduction, pitch/démo et soumission vérifiée.

## Entire — finaliser l’onboarding du poste Icham

- [x] CLI 0.10.6, 12 skills, hooks actualisés et checkpoints Codex lisibles.
- [ ] Connexion utilisateur avec `entire login`.
- [ ] Approbation des trois nouveaux hooks via `/hooks` dans Codex.
- [ ] Décider de la publication des conversations, puis vérifier recherche/indexation si autorisée.

Voir [[Entire - installation et vérification]].

## À confirmer — responsable opérationnel à nommer par Icham

- [ ] Deadline/fuseau, durée pitch et format de dépôt.
- [ ] Cadrage acoustique accepté par les organisateurs.
- [ ] Accès Nebius, crédit activé, plafond et arrêt instances.
- [ ] Accès aux poids de base et conformité licence de redistribution.

## Blocages identifiés

Aucun blocage restant pour les étapes 1 à 5 de la V0. Qualité, revue d'interfaces G1, intégration application et évaluation finale restent à faire ; elles ne sont pas prouvées par le mini-training. Voir [[V0 ML - exécution]].

## Suivi

[[Journal Icham]], [[Journal Nevil]], [[Journal Safoan]], [[Journal Vincent]]. Mettre une tâche en cours seulement quand démarrée, terminée seulement avec preuve. [[Passation]] conserve la vue globale.
