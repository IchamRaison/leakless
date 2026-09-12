# Journal simulateur

Icham

## 2026-09-12 — livraison locale isolée

Implémentation explicitement demandée par Icham sans déranger l’autre agent. Worktree `/tmp/ehl-sensor-replay`, branche `feat/sensor-replay`, base `83c183c`. Inspection des arbres locaux main/ML/Safoan : aucun simulateur trouvé. Bibliothèque standard, aucun bootstrap partagé modifié. [[Plan simulateur de capteur]] consigne preuves, limites et commandes.

Premier lancement des tests : erreur de racine d’import dans le test, corrigée avant validation. Deux tests passent ensuite ; un test d’arrivée retardée plus ancienne que la fenêtre en attente a été ajouté et passe. Replay réel : cinq réceptions, trois pertes, un doublon, fin normale. Aucun WAV réel sélectionné ni fichier audio publié. Les payloads restent inchangés, séquences/session distinguent captures identiques et retransmission.

Décision : ordonnanceur local unique avec durée du consommateur simulée, pas de serveur/threads inutiles pour cette preuve ; les métadonnées du scénario sont O(n), payloads bornés. Une intégration réelle nécessitera de sortir l’appel bloquant de l’horloge source. Prochaine action : revue de l’enveloppe/transport par Safoan avant raccordement. Aucune communication externe à l’équipe envoyée par cet agent.
