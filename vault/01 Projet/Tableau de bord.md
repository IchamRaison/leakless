# Tableau de bord

Icham

## Priorité actuelle — 13 septembre

- [x] Rush27B : même base BF16 sur2H100 vérifiée,223tests, gate mémoire/gradients/reload/parité réussi ; MB4/1époque figés avant comparaison. [[Rush Qwen 27B - exécution]].
- [x] Rush27B : trois fits internes et leurs évaluations terminés, reloads exacts ; contrôle du wrapper sur train réussi. Aucun job restant au dernier contrôle, deux GPU inoccupés. [[Rush Qwen 27B - exécution]].
- [ ] Rush27B : terminer l'audit consolidé et l'archivage/bilan des trois folds ; seul fold0 a été revérifié indépendamment à ce stade. Aucun refit ni nouveau réglage.
- [ ] Discussion demandée, sans implémentation : architecture acoustique → suivi temporel candidat LSTM → notification factuelle. Données continues, utilité du LSTM et compatibilité brief à arbitrer. [[Journal Icham#13 septembre — proposition acoustique, suivi temporel et notification]].

- [x] Nouvel accès `ich@195.242.28.46` réussi : H100 80GB HBM3, GPU inoccupé au contrôle, Python 3.12.3, environ 1,2 To libres. Deux accès H100 vérifiés avec la machine existante. [[Journal Icham#13 septembre — nouveaux essais SSH, une H100 supplémentaire accessible]].
- [x] `iche@89.169.97.196` testé avec clé explicite puis identités normales : SSH répond, authentification refusée ; GPU non vérifié. Les deux autres refus de l'échange précédent restent historiques, sans nouvel essai.
- [x] Deuxième runtime H100 préparé avec uv/lock existants : 132 paquets compatibles, imports, calcul/backward GPU et 214 tests sans skip. `docs/evidence/runtime-h100-2-001/`. Pas encore de poids/données ni de parité Qwen entre machines ; pas de DDP ou essai supplémentaire.
- [x] Socle train ensuite transféré et vérifié sur H100-2 :646fichiers exacts,43SHAQwen/5SHAprovenance/598MD5WAV conformes, aucun val/test/externe. Script `e142f66`, `verify-assets.log` ; vraie parité Qwen entre nœuds encore non mesurée.
- [x] Périmètre du goal étendu aux étapes 6 à 10 à la demande d'Icham ; [[Diagnostic causal TSLM vs C1#Plan global en dix étapes]]. Ajout documentaire, pas exécution.
- [x] Proposition d'Icham consignée : [[Diagnostic causal TSLM vs C1]], expliquer l'écart avant une nouvelle recette ; ancien enchaînement V2 en pause.
- [x] Revue Claude intégrée : contrôles manquants ajoutés et conclusions non démontrées nuancées. [[Diagnostic causal TSLM vs C1#Checklist enrichie après la revue de Claude]]. Aucun contrôle nouvellement exécuté ni logger modifié.
- [x] Six fits précédents complets/vérifiés/rapatriés, derniers artefacts `b65042a`, bilan dans [[Diagnostic causal - exécution]]. Aucun réentraînement.
- [x] Objectif de diagnostic autorisé ; premier bloc D0 borné à1516observations A0/C0, sans apprentissage, protocole fixé avant scores.
- [x] D0 implémenté `733b9c6`, 181 tests runtime sans skip ; préinscription `cd2c3916…` publiée avant observation, preuves `3e425b4`.
- [x] D0 A/C :1516 observations vérifiées et revues, preuves `f64c943` / `9dd5b73`, zéro apprentissage. Reload et échanges complets exacts, aucune relance.
- [x] D1 implémenté/testé/préinscrit : code `306d330`, 203 tests runtime sans skip, préinscription `37cfcae1…` et preuves `1f89933` avant observation.
- [x] D1 A/C exécutés : 3 588 forwards sans fit, artefacts `640d29c` / `9c163d4`, poids/entrées inchangés et reproduction D0 exacte ; [[Diagnostic causal - exécution#D1 — résultats et portée]]. Pas de politique FP32 adoptée ni correction uniforme de l'écart à C1.
- [x] D2 borné : une configuration d'arbres boostés, mêmes 256 valeurs/folds, comparateurs réservés existants, AUC groupe primaire ; revue méthodologique favorable. [[Diagnostic causal - exécution#D2 — sonde non linéaire, protocole avant fit]].
- [x] D2 implémenté `303609e`, revue sans blocage, 11 tests ciblés et 214 tests runtime sans skip ; préinscription `67ed6bd5…` publiée `3505928` avant les fits. Identité exacte des séries/caches vérifiée.
- [x] D2 : trois fits/200 itérations terminés, reçus et résultats revérifiés indépendamment, artefacts `160655a`. AUC groupe réservée moyenne 0,854167, trois gains vs sonde linéaire, encore derrière C1 fixe. [[Diagnostic causal - exécution#D2 — résultats vérifiés et portée]].
- [x] D3 implémenté `c08ff78`, relu, huit tests ciblés puis 222 tests runtime ; préinscription machine `42920f6d…`, aucun fit à ce jalon. [[Diagnostic causal - exécution#D3 — vérifications avant fit]].
- [x] D3 : unique fit terminé au pas400 (100 époques), 32/32corrects et NLL binaire0,000826279 ; Qwen gelé/loss complète inchangés. Reçu train `5e782e54…`, reload neuf `3fe96a1a…`, écart des32scores nul. Mémorisation uniquement, aucun gain de généralisation revendiqué.
- [x] D3 point100 historique :24/32corrects, NLL0,574984 ; preuves `b7ffdc3`, désormais remplacé comme état courant par le point400.
- [x] Extension D3 sur98réservés terminée/vérifiée :67/98corrects,21FN/73fuites,10FP/25non-fuite-bruits ; AUC clip0,741370/groupe0,798077. Préinscription `78e3b455…` publiée `84c1890`, reçu `17f8af51…`,98observations/196forwards, zéro fit. Qualité insuffisante, test officiel/externe fermé ; définir ensuite un contraste à budget comparable.
- [ ] Vérifier la journalisation sur un prochain fit réel : implémentation `66f390b`, neuf tests runtime de transparence réussis, preuves `306d330` ; aucun nouvel entraînement lancé.
- [ ] Produire la matrice causes démontrées / hypothèses non soutenues dans les conditions testées / inconnues. Aucun refit ni confirmation externe automatique.
- [x] Audit train/cibles : les trois entraînements contiennent les deux classes ;242 vrais sans-fuite concentrés dans7 groupes. Limites dans [[Diagnostic causal - exécution#Audit des populations et des cibles]].
- [x] Shortlist relue sur les sources : [[Datasets utiles pour PIPE#Revue du 13 septembre — diversité globale]]. Diversité de l'ensemble des données visée ; Hong Kong candidat à auditer, Aghashahi déjà réservé. Aucun nouveau téléchargement de données ni entraînement.
- [ ] **En dernier recours uniquement :** si les autres pistes n'améliorent pas suffisamment le modèle, diversifier les données acoustiques des deux classes, sites/capteurs/matériaux/conditions. [[Diagnostic causal TSLM vs C1#Dernière étape conditionnelle — diversifier les données acoustiques]]. Les diagnostics restants ne sont pas omis ; aucune collecte supplémentaire lancée.

## Suite incluse dans le goal — après le diagnostic

- [ ] **6.** Corrections justifiées, nouvelle version entraînée et comparaison sur développement avec les références et C1.
- [ ] **7.** Si progrès insuffisant : diversification globale, audit des sources et séparation train/développement/confirmation avant intégration ; sinon omission motivée.
- [ ] **8.** Si ajout de données : comparaison même recette avec/sans ajout, apport réellement mesuré.
- [ ] **9.** Gel complet, reload neuf et évaluation sur réserve indépendante ; aucun réglage sur ses résultats.
- [ ] **10.** Intégration sur poids réels, surveillance/événements/santé testés, replay distinct du terrain et benchmark gabarit/Qwen/TSLM sur événements réservés.

[[Diagnostic causal TSLM vs C1#Critère de complétion du goal élargi]] et [[Diagnostic causal - exécution]] portent les critères et preuves. Une étape prévue n'est pas une étape terminée.

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

- [x] Safoan : code TSLM V2 intégré à l'API/UI sur
  `feat/demo-temporal-building`, commit `bba67ff` ; 23 tests API, 35 frontend,
  build et 12 tests wrapper réussis. [[Journal Safoan]].
- [ ] Safoan/Icham : fournir les trois artefacts V2 cohérents et exécuter la
  recette sur poids réels ; aucun checkpoint/décision/reçu final n'est dans Git.

## À lancer — pas encore vérifié

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
- [x] C et reload PASS, delta zéro, SHA `50ec36e9…` ; séquence `98052` terminée normalement, pas de relance.
- [x] Runner et évaluateur externe publiés `b750f5d`, 144 tests runtime passent ; aucun fit réel encore.
- [x] Assemblage final `c13fd47`, 156 tests runtime sans skip ; campagne A/C préinscrite avant fit, SHA `5c5e0b4b…`, preuve publiée `550c7a6`.
- [x] Bilan complet de l'ancienne campagne six fits TSLM /douze fits C1 : A et C restent derrière C1, [[Diagnostic causal - exécution]]. Aucun refit.
- [x] Douze fits C1 terminés ; `C=0.01` retenu sur train, moyenne AUC groupe0,948718 /clip0,891382, pas une confirmation externe.
- [x] A0 terminé/vérifié, preuve `83bcbe0` : AUC clip/groupe0,710/0,760 sur98 clips réservés ; Qwen inchangé. Pas de conclusion A/C sur ce seul fold.
- [x] A1 et C0 terminés/vérifiés, artefacts `fa1b9b4` ; 3/6 fits TSLM complets, résultats partiels derrière C1.
- [x] Trois derniers reçus et poids distants vérifiés ; CSV/journaux rapatriés et revérifiés, `b65042a`. Ne pas relancer les fits.
- [x] Exporteur inclus dans les156 tests runtime réussis ; audit réel sur modèle final encore à réaliser.
- [x] Générateur Markdown V2 et compte des contradictions affichées publiés `0c99a3a`, 162 tests runtime passent ; rendu réel après comparaison complète, pas de nouveau moteur de métriques.
- [x] Objectif final utilisateur ajouté : évolution d'événement/investigation et benchmark gabarit → Qwen sur mesures → TSLM sur séries. [[Plan V2 - fiabilité et parité des scores#7. Objectif final — démontrer la valeur du TSLM]].
- [ ] Construire et évaluer ce benchmark final sur des événements/historiques/contextes réservés ; ne pas confondre campagne A/C et preuve d'utilité du TSLM.
- [x] Reprise dans clone sain `/home/animus/ehl-hackathon-zurich-v2-recovery`, `e9c8ddd` publié ; ancien Git endommagé conservé, aucun reset/destruction.
- [ ] Suite V2 mise en pause au profit de [[Diagnostic causal TSLM vs C1]] ; restitution réelle et confirmation externe restent non exécutées, pas des tâches terminées.
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
