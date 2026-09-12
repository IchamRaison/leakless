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

- [x] Icham demande de planifier une vraie évaluation par nous-mêmes, sans attendre Nevil ; ancienne exclusivité levée. [[Protocole évaluation#Plan qualité V1 — proposé, non exécuté]]. Aucun calcul lancé.
- [ ] Après feu vert : T0/test complet, contrôles comparables, stress puis audit texte val/test séparés, checkpoint figé ; pas une démo de huit clips.
- [x] Icham autorise l'implémentation V1 complète jusqu'à T0/T1–T3 ; runtime H100 et données existantes revérifiés. [[V1 ML - exécution]].
- [x] Scoring continu implémenté, huit tests CPU et contrôle H100 sur quatre clips train réussis ; aucune évaluation de qualité. [[V1 ML - exécution]].
- [x] Scripts de campagne/reload/export publiés et transférés à `1199789f` ; 18 tests CPU réussis dans le runtime reconstruit.
- [x] Campagne V1 terminée : 600 étapes train complet, trois candidats 2/4/8, époque 4 retenue sur validation uniquement, Qwen gelé inchangé. [[V1 ML - exécution]].
- [x] Bundle V1 autonome et reload neuf hors ligne vérifiés ; score/texte/entrées concordants. Exporteur renforcé et ses 19 tests CPU passent.
- [x] T0 produit et contrôlé conforme : 402 IDs val/test, exactement deux colonnes, deux fichiers et provenance vérifiée ; aucune métrique finale côté Icham.
- [x] T0 publié sur `feat/icham-tslm` à `186c45a`, disponible pour Nevil avant les stress.
- [x] T1–T3 terminés, conformes chacun sur 402/402 clips et publiés à `7c04c9a`, même checkpoint/score, sans réentraînement. Audit indépendant des quatre exports réussi, GPU libéré ; instance toujours allumée.
- [x] Réserve de graine T2/T3 résolue par Nevil : SHA-256 officiel depuis `b23601a`, version reprise `6dfdf63`. Contrôle interprocessus réussi dans les tests CPU.

- [x] Cadrage explicite Icham : appareil en écoute continue et alertes automatiques, remplacement du scénario d'import manuel. [[Plan surveillance continue]] rédigé ; aucune nouvelle implémentation.
- [ ] Valider le plan continu : capteur/flux, destinataire, délai visé, fausses alertes tolérées, responsable matériel et périmètre replay/réel.
- [ ] Safoan/Icham : flux/replay horodaté, santé de surveillance, événements persistants et tableau d'alertes ; extension de contrat à convenir.
- [x] [[Plan simulateur de capteur]] étapes 1–4 implémentées sur `feat/sensor-replay` : deux tests CPU réussis, replay réel 8 s (5 reçues, 3 coupures, 1 doublon ignoré). Étape 5 applicative restante, interface à convenir avec Safoan.
- [ ] Nevil/responsable matériel : acquisitions continues annotées, sessions séparées et métriques événementielles ; aucune preuve opérationnelle déduite des clips.
- [x] Icham : environnement Python/CUDA isolé ; calcul et backward H100 réussis. [[V0 ML - exécution]].
- [x] Icham : reprises sélectives Nevil `1289095` / contrat Safoan `4dd7b88` ; données/TimeNet rejoués sur H100.
- [x] Icham : Qwen 3.5-4B choisi après demande utilisateur, téléchargé anonymement et décodeur chargé sur H100 ; Llama abandonné.
- [x] Icham : trois archives téléchargées/intégrité vérifiée ; tests CPU numériques et frontières d'entrée réussis.
- [x] Livraison Nevil `nevil/temporal-evidence` (`08562e3`) retrouvée et lue : contrat d'export, moteur d'évaluation, C0–C3, T0–T3, rapports. Tests/résultats non reproduits ici ; branche non intégrée. [[Journal Icham#Vérification des nouveaux livrables Nevil]].
- [x] Consigne de livraison Nevil transmise : Icham fournit seulement les prédictions T0 et leur provenance ; Nevil exécute l'évaluation finale. [[Journal Icham#Consigne de livraison T0 et stress — Nevil]].
- [x] Ajouts Nevil `9135754` relus : score continu, contrôleur de conformité existant, petite validation groupée. Plan révisé sans exécution ; réserves longueur/calibration et reproductibilité T2/T3 documentées. [[Journal Icham#Revue des précisions Nevil — scores et stress]].
- [x] Trois candidats préannoncés/comparés, nombre réel renseigné, sélection validation seule ; score/log-probabilités et binaire/v2 gelé avec bruit respectés. Diagnostics textuels séparés, métriques finales chez Nevil. [[V1 ML - exécution]].
- [ ] G0 Nevil/Icham : revue commune de l'audit et du `split_v2` publiés, avec leurs réserves et alignement du protocole. Ne pas utiliser `split_v1` (invalide).
- [ ] Après G0, Icham/équipe : figer problème et classes ; arbitrer comparaison baseline + mesures/gabarit et test d'utilité proposés dans [[Challenge du cadrage de Nevil]].
- [ ] G1 Tous : fixture réelle et contrats gelés.
- [x] G2 mécanique Icham : runtime, entraînement court, checkpoint complet et reload neuf hors ligne, puis environnement reconstruit. [[V0 ML - exécution]].
- [ ] Vincent : baseline entraînée sur développement, tests et export.
- [ ] G3 Safoan : audio/spectrogramme et vraie inférence de bout en bout.
- [ ] G4 Icham : évaluation finale commune via le harness existant après gel des modèles ; planifiée seulement, nouvelle consigne utilisateur.
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

Aucun blocage restant pour V0 ni pour la livraison V1/T0–T3, terminées. Qualité finale, revue d'interfaces G1, intégration application et surveillance continue restent à établir ; elles ne sont pas prouvées par la conformité des exports. Voir [[V1 ML - exécution]].

## Suivi

[[Journal Icham]], [[Journal Nevil]], [[Journal Safoan]], [[Journal Vincent]]. Mettre une tâche en cours seulement quand démarrée, terminée seulement avec preuve. [[Passation]] conserve la vue globale.
