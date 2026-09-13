# V2 ML - exécution

Icham — mise à jour 2026-09-13

## État courant

Objectif utilisateur actif : implémenter [[Plan V2 - fiabilité et parité des scores]] dans son ensemble. **Parité A et C, rechargements neufs compris : PASS, écarts nuls (`04d53b6`). Diagnostic train et compatibilité V1 terminés ; 144 tests runtime passent à `b750f5d`. Aucun entraînement TSLM V2 réalisé : assemblage final de l'exporteur à vérifier, puis préinscription et six fits comparatifs explicites.** Le 13 septembre, Icham ajoute explicitement le cap final langage/événements/investigation et benchmark à trois approches ; aucune de ces capacités n'est encore validée. Checkout actif sain `/home/animus/ehl-hackathon-zurich-v2-recovery`, branche `feat/icham-v2-reliability`. L'ancien worktree `/home/animus/ehl-hackathon-zurich-v2` et le dépôt code commun sont conservés sans réparation destructive après découverte d'objets Git vides ; ne plus y commiter. Le vault dédié reste sain et autoritaire.

## Conditions vérifiées au démarrage

- Plan lu entièrement ; source vault récupérée à `11ca52b` avant mise à jour. `using-entire` et recherche ciblée utilisés : aucun résultat sur la cause de la parité ; diagnostic fondé sur code/artefacts, pas sur une transcription supposée.
- SSH non interactif à la machine existante réussi. H100 80 Go observée libre, 0 Mio / 0 %. Aucun nouveau GPU/dépendance provisionné.
- Bundle historique `/home/hicham/pipe-v0/artifacts/qwen-v1-1199789f-001/bundle` : `checksums.json` SHA `b95569c50f5e1bbf3533bddc92530b5f285eb25dec999f83999f60ffd912d1ae`, `temporal.pt` SHA `183a1b1c29facf0e27f27efc7adab8c1bb13556cede06abc7f735672cbe7eb82`, inchangés.
- Données/caches existants : WAV `/home/hicham/pipe-v0/data/extracted`, caches `/home/hicham/pipe-v0/artifacts/prepared/{train,val}.npz`, TimeF `/home/hicham/pipe-v0/artifacts/prepared/timef/leakless/acoustic-leak/2.0.0`. Le cache test ne sera pas utilisé pour développer V2.
- API TimeF installée vérifiée en lecture seule : `TimeFReader.iter_records(record_ids=..., with_annotations=False)` filtre avant chargement des valeurs ; cela permet de lire le développement sans décoder les signaux test. L'essai initial d'import `timenet.storage` échoue ; bon module `timenet.reader.reader`, déjà employé dans `prepare.py`.

## Chantiers actifs et limites d'autorité

- Diagnostic numérique terminé : passage float32 et dépendance au lot démontrés/corrigés, preuves ci-dessous. Root possède toute exécution GPU ; ne pas relancer les contrôles terminés.
- Restitution : interface V2 opt-in cohérente, artefact de seuil lié au modèle et conservation du texte brut. Implémentation séparée de V1, pas de déploiement applicatif implicite. Tests synthétiques autorisés, audit réel après parité.
- Données externes : audit des sources/inventaires et conditions de licence, sans consulter de performances et sans entraîner sur le holdout. Aucun import d'un dataset supposé compatible par son seul descriptif.
- Deux questions non bloquantes à Icham : budget de fausses alertes/délai ; disponibilité d'enregistrements continus annotés ou d'une collecte. Leur absence ne bloque pas les travaux numériques, mais la preuve de surveillance réelle reste distincte.

## Préflight réel — 209 entrées, sans score modèle

Diagnostic CPU terminé sur 208 validations et contrôle train `c0128b879694e`. Les octets WAV passent les empreintes gelées. Sur **209/209**, le signal TimeF est exactement la normalisation float64 convertie en float32 ; cache, représentation TimeF et hypothèse float32 sont bit à bit identiques. Le chemin WAV direct diffère sur 209/209 tableaux, maximum `2.384185791015625e-7`. Première divergence localisée avant les bandes : sérialisation float32 TimeF contre normalisation directe float64. Cela ne suffit pas encore à attribuer causalement l'écart du score.

Preuve : `docs/evidence/tslm-v2/parity-preflight-001/report.json`, SHA `2e7ea2004f89a9f961443eb549e78bf4ff478f3a2aee685ea9d5bba79bb3d0a9`, publiée avec l'outil sur `feat/icham-v2-reliability` à `e22366f280bb4161e9437ce1cdf92f90f321374f`. Rapport distant `/home/hicham/pipe-v0/quality-v2-001/parity-preflight-001`. Le préflight emploie une première version de l'outil dont l'empreinte est dans sa provenance, sur code modèle original `16a6df0`.

Cinq tests du diagnostic réussissent dans `.venv-repro` sur la machine ML, hooks PyTorch inclus. Trace GPU des quatre cas fixés en cours depuis le snapshot `/home/hicham/pipe-v0/code-v2-e22366f` : sortie `quality-v2-001/parity-trace-001`, log adjacent. Même bundle V1, aucune modification des poids, aucun entraînement ni métrique de qualité.

## Cause démontrée, correction et nouvelles préparations

Traces GPU sur les trois validations fixées + un train : cache, TimeF et conversion float32 contrôlée produisent exactement mêmes étages/scores ; chemin direct reproduit l'écart maximal historique `0.4688478100893639` contre `0.5312166150213153`, delta `0.06236880493195146`. Trois répétitions par chemin, adaptateurs sur même tableau et hooks transparents : delta zéro. Deuxième processus neuf, mêmes quatre cas : delta zéro également. Rapports `parity-trace-001` SHA `bcd262e011105813025516786bfb86525285b517be8f69d63f8f68bc187c8cf9` et `parity-trace-reload-001` SHA `234c8e530dd36f899c13e4de799a9934c47d30a823241b9039689e2c7fabae76`. Causalité du passage float32 établie sans AUC ni choix de meilleure performance.

Correctif publié `aaab4afd263eec0d89f5c2468daf7ac5f8f091a6` : nouvelle version `rms-f32-hann256-hop128-band4-log1p-v2`, conversion float32 après normalisation et avant FFT. `preprocess_audio(..., version=...)` et `predict.preprocess_for_model` unifient les entrées ; défaut et métadonnées V1 gardent le comportement historique. `prepare_development` lit seulement 598 train + 208 val, exige TimeF/WAV exactement égaux, écrit un nouveau cache. **806/806 entrées vérifiées**, aucun signal/cache test lu.

- Cache `/home/hicham/pipe-v0/artifacts/prepared-v2-aaab4af` : train SHA `8bf27c0ad1ef13056fa7f688a8e091aec027d53f0438db03e88dead0b138f27b`, val SHA `5046b750cfa060a9c21206a3eda9d2f6b5ebe3051b7a69fe9a9cac1af64e78be`.
- Référence corrigée `/home/hicham/pipe-v0/artifacts/qwen-v2-parity-aaab4af`, `pipe-qwen3.5-4b-v2-parity-aaab4afd`, checksums SHA `c653fb294fec32a9610aeeb39a3754512a069c8e837d825a7202439520a28eb0`. **Mêmes poids V1, pas un modèle réentraîné**, interdite comme initialisation des folds V2. Copie indépendante de la base, V1 intacte.
- Restitution opt-in `CoherentPredictor` et reçu de seuil versionné implémentés, sept tests avec doubles CPU ; pas encore audit réel ni intégration applicative.
- **54 tests runtime ML passent** à `aaab4af` : 40 TSLM + 14 évaluation, dont régression canonique, refus d'écrasement et copie indépendante. Preuve `docs/evidence/tslm-v2/canonical-preparation/tests-aaab4af.log`.
- Aghashahi intégralement téléchargé/vérifié/préparé dans `/home/hicham/pipe-v0/data/external/aghashahi-v1` : 120 enregistrements / 3 600 fenêtres primaires / 60 conditions heuristiques ; deux bruits / 60 fenêtres séparés. SHA sources et reçus conservés, code/source `aaab4af`. **Aucun score externe**, audit de recouvrement encore à faire et sessions réelles inconnues. Guide `docs/V2_EXTERNAL_DATA_AUDIT.md`.

## Gate réel ÉCHOUÉ — deuxième divergence à résoudre

`scripts/tslm/check_v2_parity.py` publié à `e9c8ddd`, six tests CPU réussis. Compare 209 entrées sur cinq chemins numériques, trois interfaces et vrais lots 1/2/4 dans les deux ordres. Premier passage ne peut pas produire PASS final : processus neuf de référence obligatoire, tolérance absolue `1e-6` inchangée.

Premier passage terminé, **exit 1 / failed**, depuis `/home/hicham/pipe-v0/code-v2-e9c8ddd`, sortie `/home/hicham/pipe-v0/quality-v2-001/parity-gate-001`, log adjacent. Ancien handle `84453` terminal, ne pas le relancer. Les 209 entrées des cinq chemins sont exactement égales, bundle inchangé, trois interfaces et lots 1 dans les deux ordres donnent des scores identiques. **Lots 2/4 : 195 clips dépassent `1e-6` dans la comparaison globale**, écart maximal `0.062458740766542675`, clip `cf591447e4f2d` : `0.5312671112977968` seul, `0.4688083705312541` en lot. Pas de reload prétendu PASS de ce gate échoué.

La source OpenTSLM épinglée regroupe les `4 × N` séries pour l'encodeur/projecteur avant conversion BF16 ; le décodeur de scoring boucle déjà par clip. Conv1d/LayerNorm/Transformer batch-first, sans BatchNorm ni attention entre clips : une dépendance numérique à la forme du lot est plausible, pas encore prouvée par intervention contrôlée. Nouveau diagnostic quatre cas avec leurs voisins exacts du gate en préparation : hooks encodeur/projecteur, avant/après BF16, puis assemblage temporaire par clip à comparer au bulk et au singleton. **Ne pas supprimer les tailles 2/4 du contrôle ni relever la tolérance pour contourner l'échec.** Une politique interne par clip doit être démontrée, versionnée et préserver les N sorties ainsi que V1.

Revue indépendante : la dispersion de tolérance des 209 est correcte. Compléments requis : lier préparation complète au consommateur avant fit (ajout réalisé et six tests CPU passent ; les 598 train seront aussi revérifiés contre WAV), contrôle des décisions au seuil final une fois choisi, preuve de compatibilité V1 du nouveau runtime distincte du gate canonique. Les fichiers V1 historiques et leurs snapshots exécutables restent inchangés.

## Deuxième cause démontrée — encodage dépendant du lot

Trace GPU `batch-trace-001` terminée depuis le snapshot `ab54940924330b1d0a43e3bb31d98a24f5e8b8b9` : quatre clips fixés et leurs voisins exacts, vingt contextes singleton/lots 2/4/ordres. Les scores originaux reproduisent exactement le gate échoué ; hooks et intervention restaurés, poids inchangés. Une unique intervention, appeler l'assemblage acoustique original séparément par clip, rend **tous les étages et scores exactement égaux aux singletons dans les vingt contextes**.

Pour le cas maximal `cf591447e4f2d`, lot 4 forward : entrées identiques ; première différence à la sortie de l'encodeur `8.046627044677734e-7`, projecteur `1.0728836059570312e-6`, embeddings BF16 `0.00390625`, sommes de log-probabilités `0.1251136788923759`, score `0.062458740766542675`. Causalité numérique du regroupement démontrée, pas un gain de détection. Rapport conservé dans `docs/evidence/tslm-v2/batch-trace-001/report.json`, traces complètes sur la machine ML.

Correction minimale publiée `0b1399e035e2e6d25b6d680bf9c712dcb08930fb` : `single_clip_acoustic_encoding: true`, nouveau scoring `class-continuation-logprob-sum-softmax-single-clip-v2`. Assemblage par clip partagé entre apprentissage et inférence, gradients préservés ; les API acceptent toujours N clips et rendent N scores. V1 reste sur son ancien comportement par défaut. **77 tests runtime passent** (57 TSLM + 20 évaluation), dont régression de lot et gradients. Référence neuve `/home/hicham/pipe-v0/artifacts/qwen-v2-parity-0b1399e`, checksums SHA `ca72fb80a23d6e799edc6824246e69a0a36eaa88360854b694a4b93d5aa52ff4`, mêmes poids temporels V1. Gate complet en cours depuis `code-v2-0b1399e`, sortie `quality-v2-001/parity-gate-002`, handle `65120`. Reload ensuite seulement si ce premier passage réussit ; aucun PASS anticipé et aucune tolérance relevée.

Audit externe exhaustif **terminé, exit 0**, depuis `41dd8b27a558da249e812a49893a6b27bc275070`, sortie `/home/hicham/pipe-v0/artifacts/external-overlap-v2-001`. Les 122 000 paires ont été comparées sur 232 001 décalages entiers chacune : zéro candidat à `|Pearson| >= 0.995`, zéro fenêtre/paire non évaluable, zéro doublon exact. Rapport SHA `5a1a4f090706a74fe7ec65d43017a76fb5841cd5464035294b85e10c5f8233fd`, reçu/provenance et meilleurs rapprochements copiés dans `docs/evidence/tslm-v2/external-overlap-001/`. Ancien handle `7015` terminal, ne pas recommencer. Aucun score de modèle ni exclusion automatique ; filtrage/rééchantillonnage/décalages fractionnaires non couverts, pas une preuve de sessions indépendantes.

## Parité corrigée complète PASS, processus neuf compris

`parity-gate-002` terminé normalement à `0b1399e` : entrées/bundle/interfaces/lots tous vérifiés, **écart maximal strictement zéro** sur les 209 clips et tous les chemins. Deuxième processus terminé depuis le même snapshot avec `--reference quality-v2-001/parity-gate-002/report.json`, sortie `quality-v2-001/parity-gate-reload-002` : **status passed, all_checks_pass=true et cinq contrôles vrais**. Rapport SHA `fb44fbb893f7bd17457b1499bc609d4836d764ac9a9cc9de52f77da6d0a9877c`, conservé dans `docs/evidence/tslm-v2/parity-gate-reload-002/`. Ancien handle `38610` terminal. Aucun seuil sélectionné ; stabilité des décisions au seuil final à contrôler une fois celui-ci fixé. Cela autorise le diagnostic train, pas une revendication de qualité améliorée.

Contrôle distinct de compatibilité V1 en cours depuis le même code numérique : trace des quatre témoins contre `parity-trace-001` historique, bundle V1 inchangé, sortie `quality-v2-001/v1-compatibility-001`, handle `46232`. Il conserve les différences historiques de la voie V1 ; aucun correctif rétroactif de ses exports.

Préparations logicielles parallèles : restitution `predict_waveform` sans requantification publiée `47d6ad5`, dix tests synthétiques réussis ; extraction `campaign.train_epoch` publiée `c4c3808`, **62 tests TSLM passent dans le runtime**, dont comparaison autograd exacte avec l'ancienne boucle V1. Ces changements ne modifient aucune des quatre sources numériques du gate. Pas de variante choisie, de fit ou de score externe lancé.

## Diagnostic train terminé et variantes A/C préparées

`quality-v2-001/train-diagnostic-001` a terminé normalement dans le runtime ML avec le gate complet `0b1399e` et les anciens caches canoniques exacts. Les **598 train / 102 groupes** ont été revérifiés contre les WAV ; aucune qualité val/test consultée. Trois folds groupés fixes (34 groupes réservés chacun, dont 8 non-fuite), SHA `folds.json` **`5c2bea733f8f2a2dc525b9738a5aa40ae3ce220cf6d76d712cab984afb4d5efb`**. Les futures campagnes réutilisent ces IDs, pas de nouveaux folds.

Sondes logistiques fixes, six fits diagnostiques : moyenne AUC groupée **0,636218** sur les quatre séries TimeNet aplaties, **0,937500** sur les neuf descripteurs C1 ; moyenne AUC clip 0,561235 / 0,896558. Preuve `docs/evidence/tslm-v2/train-diagnostic-001/probes.json`, SHA `aa16a7b1230ed0badbaac8db942b721ae07a3d213903645025bc2d80af456fc8`. Ce ne sont pas les scores d'une V2 TSLM ni une confirmation indépendante ; une sonde faible ne prouve pas l'absence d'information.

Autopsie sur huit clips train de huit groupes, sans pas d'optimiseur : la classe représente 18,9 % des tokens mais ~0,036739 de la perte totale ~0,036806, le texte/EOS ~0,000067. Gradients acoustiques présents, Qwen gelé et poids inchangés. Reçu `supervision.json`, SHA `9ca1647218b09cf6d8971d0f1cedce8541e2aaa285f9d5affab04d6f049dec2e`. La domination du texte n'est pas étayée à ce checkpoint ; ne pas extrapoler huit clips à toute l'histoire d'entraînement.

**Décision : comparer A et C, omettre B.** C conserve les quatre séries et ajoute au prompt les neuf mesures officielles C1, rendues à six chiffres significatifs et versionnées ; aucune baseline prédite, classe ou métadonnée de dataset en entrée. Ce choix teste l'accès à une information d'amplitude, pas encore l'utilité spécifique du TSLM. Nouveau code en cours de vérification : cache commun A/C, frontière d'entrée stricte, packing des réponses après le vrai prompt (longueurs variables), accumulation token-pondérée avec lot effectif 8 / microbatch 1. **Le PASS historique ne valide pas automatiquement ces nouvelles sources : refaire les deux gates complets A/C et leurs reloads avant campagne.** V1 conserve le chemin legacy par défaut.

Compatibilité V1 distincte terminée : les quatre témoins retrouvent exactement les scores historiques sous `0b1399e`, SHA du rapport `9a5f065b7c1208c43d85ba0d9f02b7e187c167dd2dbc665fd8ab10fcadf55a1e`. Manifeste externe figé par le lecteur versionné `5a71b7b`, après contrôle des cibles et des 122 000 comparaisons existantes : `/home/hicham/pipe-v0/artifacts/external_aghashahi_v1.json`, SHA **`840078010f4023a045a039167f079ef178f6b0cb56f7f280fd2a5760aad1616b`**, 3 600 fenêtres primaires, bruits annexes séparés, `model_loaded=false`. Aucun score externe ; pas de nouvel audit de corrélation à relancer.

## Vérification A/C — code et cache réels validés, gates en cours

Code A/C publié à **`04d53b6d2e0e444254a77360cd2be90faf301c0e`**, snapshot ML `/home/hicham/pipe-v0/code-v2-04d53b6`. **117 tests runtime passent sans skip** : 80 TSLM et 37 évaluation, dont packing causal/autograd, accumulation pondérée, entrée C commune, rejet du cache altéré et persistance C1. Logs conservés dans `docs/evidence/tslm-v2/ac-preparation-04d53b6/`.

Nouveau cache réel `/home/hicham/pipe-v0/artifacts/prepared-v2-ac-04d53b6` : **806 entrées exactement vérifiées**, bandes, neuf mesures et texte entre WAV canonique et TimeF ; aucun signal test lu. Train SHA `9b7b14e195bc7a82cd0e09907c31d26b75c15ed8fca84f4a9b68895e8c174c44`, val SHA `30042b666adf6cc6a4d80683585d45c91532746b8d0ee1c31149819e7d0e64b8`. Références indépendantes créées : `artifacts/qwen-v2-parity-a-04d53b6` (checksums `cb1f69aa83dea426e6e492ab4431302f7aa521c5f1eec388f09c3f3a5bec504c`) et `artifacts/qwen-v2-parity-c-04d53b6` (`791f7f0fd7c2116f001fb8a4a6a97ec7feb3f532bbb0f35b4c817e8919514055`). Mêmes poids V1 `183a1b1c…`, aucune de ces références n'est un modèle réentraîné ni une initialisation CV autorisée.

Contrôles réels lancés en séquence, arrêt dès échec : A premier passage → A reload → C premier passage → C reload. **Handle SSH `98052` actif**, premier processus A PID `62069` confirmé sur H100 (environ 9,2 Go) ; 201/209 entrées déjà scorées via les trois interfaces au contrôle de progression. Sorties `quality-v2-001/parity-{a,c}-04d53b6` et `parity-{a,c}-reload-04d53b6`, logs adjacents. Ne pas redémarrer ce job d'après une interruption d'observation ; lire le handle/processus. Aucun PASS final ni entraînement anticipé. L'empreinte `amplitude_tokens` est explicitement celle du fragment d'amplitude isolé, pas une inspection des tokens du chat complet.

Orchestration restante en construction séparée : campagne CV/refit/validation, export externe numérique+audit texte, extension opt-in de l'évaluateur au manifeste externe et aux seuils T0 figés. Réutilisation des métriques, du bootstrap, de C1 et des transformations existants ; pas de second moteur, pas de métrique externe calculée à ce stade.

**Résultat suivant : A + reload complet PASS**, cinq contrôles vrais, delta maximal zéro sur les 209 clips et tous les chemins entre processus. Rapport `parity-a-reload-04d53b6/report.json`, SHA **`ed2bfdc511cdb2d45cb7ba7752ee24643b29772551a7fc7793bfe93ae58c60c8`**, copié localement avec le premier passage. La même séquence `98052` poursuit C ; ne pas lancer un second job. Aucun entraînement encore autorisé par les preuves tant que C/reload restent incomplets.

Runner `run_v2_campaign.py` implémenté : préinscription A/C/folds/recette/provenance, six fits explicites, C1 séparé, sélection, refit frais et seuil validation dans un processus neuf ; **17 tests CPU locaux passent**, aucun fit réel. Évaluateur externe opt-in ajouté, **8 tests locaux passent**, réutilise métriques/bootstrap avec reçus de seuil val existants et même reçu pour chaque stress. Derniers ajouts non numériques : `CoherentPredictor.state_hashes()` pour l'audit en mémoire, 12 tests locaux ; le gate final accepte le seuil fini exact de `pick_threshold`, même si borne juste hors [0,1], sans modifier les scores ni `1e-6` (7 tests locaux). Provenance A/C du bundle final personnalisée avant scellement, défaut V1 préservé. Ces ajouts seront testés ensemble dans le runtime ; ils ne modifient ni modèle, ni prédiction, ni preprocessing des gates `04d53b6` en cours.

## Porte A/C complète franchie — prochaine campagne

Séquence `98052` **terminée normalement, exit 0**. C + reload : cinq contrôles vrais et écart maximal zéro sur 209 clips × chemins/lots/ordres, y compris entre processus. Rapport SHA **`50ec36e961ab1248f956161ab9912beb8fbc3e671f444219e1a9236e87500f55`** ; A reload SHA `ed2bfdc5…`. Les deux références de contrôle sont donc vérifiées, sans améliorer ni réentraîner leurs poids. Aucun de ces gates ne contient de seuil sélectionné ; contrôle final au vrai seuil toujours requis après apprentissage.

Orchestration publiée à `b750f5d1bc18681643392b1d760ad4ad8ca29f50`, **144 tests runtime passent sans skip** (99 TSLM +45 évaluation), logs `docs/evidence/tslm-v2/campaign-b750f5d/`. Exporteur externe maintenant codé, 11 tests CPU locaux et relecture indépendante sans écart bloquant : parent neuf `run/` (deux fichiers contractuels) et `audit/` (texte/erreurs/latences et bruits annexes séparés), contrôle des poids en mémoire et des sources avant/après, pas de score de remplacement. Sa validation runtime reste à faire avant tout export réel. L'autopsie historique refuse désormais explicitement C plutôt que d'ignorer ses prompts variables ; aucune sonde n'est relancée.

Prochaine étape : figer/tester ce dernier assemblage logiciel, préinscrire la campagne dans un dossier neuf avec les deux reçus PASS, puis lancer explicitement les six fits train. Recette A/C commune : quatre époques, seed20260912, batch effectif8 / microbatch1, composants acoustiques et optimiseur neufs à chaque fold. Aucun score de validation pour choisir A/C, aucun score externe. Le classement et les erreurs par clip seront inspectés avant autorisation technique du refit final ; aucune chaîne automatique ne prolonge la recherche.

## Incident Git local — historique conservé

Le 13 septembre, le HEAD local pointait vers `0ce65624725a0c44b56334771a16c549bae03155`, objet vide ; vingt objets vides et fin de reflog remplie de NUL constatés. Cause de stockage inconnue, disque non plein. SHA distant `aaab4af` et ses objets intègres. Nouveau clone indépendant depuis cette branche, contrôle `git fsck --connectivity-only` réussi, fichiers non publiés utiles copiés depuis l'ancien worktree, aucun fichier utilisateur supprimé ni ancien Git réinitialisé. Publication saine poursuivie à `e9c8ddd`. Les hooks de l'ancien dépôt ne sont pas supposés installés dans le nouveau clone. Les autres worktrees ne sont pas réparés implicitement.

## Audit de complétion

- [x] Nouveau périmètre isolé, autorisation et intégrité V1 vérifiées.
- [x] Cause de l'écart localisée par expérience contrôlée et reload ; régression synthétique reproduisant la divergence puis la correction.
- [x] Correction causale et parité sur 208 validations + contrôle train, lots/ordre et reload neuf, tolérance `1e-6` inchangée ; écart maximal zéro, rapport `fb44fbb8…`.
- [x] Version nouvelle et copie distincte, poids V1 conservés ; aucune livraison historique modifiée.
- [ ] Restitution cohérente testée réellement, erreurs explicites et audit texte brut/affiché séparé.
- [x] Diagnostic de discrimination sur train, sonde exacte TimeNet/C1, supervision/gradients/groupes inspectés ; aucune V2 entraînée par ce diagnostic.
- [x] Sources A/C nouvelles vérifiées et deux nouveaux gates complets/reloads PASS avant entraînement, `04d53b6`, deltas zéro.
- [ ] Variantes justifiées puis préinscrites, CV groupée train avec réinitialisation, sélection et entraînement final ; seuil du modèle exact sur validation seulement.
- [ ] Reload final et comparaison externe indépendante avec C1, métriques et audit complets ; stress au seuil T0 fixé après gel.
- [ ] Limites de généralisation documentées ; protocole/données continues et critères métier résolus pour toute revendication opérationnelle.
- [ ] Rapport final, artefacts/tests/preuves publiés et vault/mirroir actualisés.

## Prochaine action concrète

Les deux gates/reloads A/C sont terminés PASS ; ne pas les recommencer. Tester le dernier assemblage logiciel incluant l'export, puis préinscrire A/C et lancer les fits comparatifs explicites. Sélection/refit frais/seuil et confirmation externe suivent, sans prolongation cachée d'essais. Ne pas relancer la préparation806, la sonde ou l'audit de recouvrement déjà terminés. Le benchmark final à trois approches et les données d'événements/contexte restent une étape distincte non validée.
