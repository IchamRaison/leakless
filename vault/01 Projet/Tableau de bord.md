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
- [x] Recherche de datasets complémentaires documentée : hold-out acoustique Aghashahi, transfert Hong Kong, données d'événements Yorkshire/Wessex et séries hydrauliques auxiliaires. [[Datasets utiles pour PIPE]]

- [x] Safoan : branche `feat/safoan-app` publiée ; Entire installé et hooks Git vérifiés. Approbation des hooks Codex et capture réelle encore à faire : [[Journal Safoan]].

## En cours et à vérifier

- [x] Plan qualité V1 complet exécuté en worktree isolé, sans modification du modèle ; [[Évaluation qualité V1 - exécution]].
- [x] Préinscription/code `c27a43fd` publiés, 32 tests runtime cible réussis, 1 000 MD5/formats WAV et huit SHA de runs vérifiés.
- [x] Cinq contrôles fixes reproduits sans nouvelle recherche TSLM ; comparaison RF Vincent conditionnelle faute d'export accessible.
- [ ] Auditer localement Aghashahi puis Hong Kong avant tout usage ; vérifier licence, formats, sessions et split externe. Ne pas fusionner au train/V1 gelé.
- [x] Évaluation CPU neuf runs et huit comparaisons appariées terminées ; AUC T0 clip/groupe 0,665/0,861, pas de gain établi face à C1. [[Évaluation qualité V1 - exécution]].
- [x] Audit texte 208 val + 194 test terminé : bandes test 184/194 correctes, 36/194 désaccords classe/score, format 194/194 ; aucun changement du modèle.
- [x] Revue indépendante terminée : 396 valeurs, 68 IC et tous les compteurs texte concordants ; rapport/preuves publiés `c8dcb9d`, branche `feat/icham-quality-eval`, SHA distant vérifié.
- [x] Plan V2 proposé à la demande d'Icham, sans commencer : [[Plan V2 - fiabilité et parité des scores]].
- [x] Objectif d'implémentation V2 autorisé ; worktree isolé, SSH/H100 et intégrité V1 revérifiés. [[V2 ML - exécution]].
- [x] Préflight CPU V2 sur 209 entrées : cache/TimeF/arrondi float32 identiques, chemin WAV direct différent ; outil/preuves `e22366f`, cinq tests ML réussis.
- [x] Cause V2 démontrée par traces et reload : passage float32 TimeF ; correctif versionné `aaab4af`, 806 entrées canoniques et 54 tests runtime réussis.
- [x] Aghashahi téléchargé/vérifié/préparé ; audit exhaustif de recouvrement terminé, zéro candidat sur 122 000 paires au seuil fixé, limites explicites. Aucun score externe.
- [x] Seconde cause démontrée : dépendance numérique de l'encodeur au lot ; intervention par clip exacte sur vingt contextes, hooks restaurés et poids inchangés. [[V2 ML - exécution]].
- [x] Second correctif opt-in publié `0b1399e`, 77 tests runtime passent, nouvelle référence mêmes poids V1 créée.
- [x] Nouveau gate 209 et reload neuf PASS, écart maximal zéro ; reçu `parity-gate-reload-002`, SHA `fb44fbb8…`. Ancien gate FAILED conservé.
- [x] Compatibilité V1 `0b1399e` vérifiée sur les quatre témoins, scores historiques exacts ; aucun export V1 modifié.
- [x] Diagnostic 598 train terminé : sonde exacte TimeNet/C1 et autopsie supervision/gradients ; A/C retenues, B omise faute de justification. [[V2 ML - exécution]].
- [x] Manifeste externe vérifié/figé `84007801…`, 3 600 fenêtres primaires ; aucun score externe.
- [x] Code A/C `04d53b6` publié, 117 tests runtime sans skip ; nouveau cache806 bandes/mesures/textes exactement vérifié, références A/C indépendantes mêmes poids créées.
- [x] A `04d53b6` et reload complet PASS, delta zéro, SHA `ed2bfdc5…`.
- [ ] C et reload dans la même séquence active (`98052`) avant les six fits comparatifs.
- [ ] Runner de campagne et évaluateur externe à seuil figé codés/testés localement ; vérifier ensemble dans le runtime avant exécution réelle.
- [x] Objectif final utilisateur ajouté : évolution d'événement/investigation et benchmark gabarit → Qwen sur mesures → TSLM sur séries. [[Plan V2 - fiabilité et parité des scores#7. Objectif final — démontrer la valeur du TSLM]].
- [ ] Construire et évaluer ce benchmark final sur des événements/historiques/contextes réservés ; ne pas confondre campagne A/C et preuve d'utilité du TSLM.
- [x] Reprise dans clone sain `/home/animus/ehl-hackathon-zurich-v2-recovery`, `e9c8ddd` publié ; ancien Git endommagé conservé, aucun reset/destruction.
- [ ] Après parité : cohérence de restitution, diagnostic train et campagne V2 bornée ; confirmation sur de nouvelles données réservées, pas sur le test V1 déjà consulté.
- [x] Icham a levé l'exclusivité Nevil puis lancé l'objectif d'évaluation complète. [[Protocole évaluation]].
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
- [x] G4 Icham : évaluation finale V1 exécutée après gel, vérifiée et publiée ; résultats mesurés insuffisants pour promettre une fiabilité opérationnelle.
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

Aucun blocage d'exécution V0/V1/évaluation. Qualité V1 désormais mesurée et insuffisante pour promettre une surveillance fiable : [[Évaluation qualité V1 - exécution]]. Export RF Vincent encore inaccessible, comparaison conditionnelle omise. Revue d'interfaces G1, intégration application et validation continue restent séparées.

## Suivi

[[Journal Icham]], [[Journal Nevil]], [[Journal Safoan]], [[Journal Vincent]]. Mettre une tâche en cours seulement quand démarrée, terminée seulement avec preuve. [[Passation]] conserve la vue globale.
