# Protocole évaluation

Icham

**État courant : évaluation complète exécutée par Icham**, après levée de l'exclusivité Nevil puis lancement de l'objectif. Son harness est réutilisé, avec correction prospective de l'AP en cas d'ex aequo et prose de conclusions neutralisée avant calcul. Protocole/code préinscrits à `c27a43fd`. Les résultats réels sont dans [[Évaluation qualité V1 - exécution]] ; V1 et exports T0–T3 demeurent figés (`186c45a`, `7c04c9a`). Les responsabilités/consignes restrictives plus bas décrivent la livraison antérieure, pas le statut courant.

## Plan qualité V1 — proposé, non exécuté

Titre conservé pour les liens historiques : ce plan initial a depuis été exécuté ; voir [[Évaluation qualité V1 - exécution]]. Les temps de réponse texte ont été mesurés ; ni benchmark du score seul ni mémoire de pointe ou débit streaming ne sont revendiqués.

1. **Figer le protocole avant tout calcul test.** Modèle époque 4 et scoring inchangés, SHA des quatre runs et split v2 vérifiés. 598 train exclus du verdict ; 208 validation pour le seuil, 194 test pour l'évaluation finale. Cible binaire actuelle, bruit environnemental parmi les négatifs. Critère principal proposé : ROC-AUC après médiane par groupe ; métriques clip et d'exploitation complémentaires, incertitude par groupes heuristiques. Conserver les trois configurations effectivement comparées et le caractère optimiste possible du score de sélection validation.
2. **Mesurer la détection T0 sur tous les clips test.** Réutiliser `evaluate_predictions.py` et les exports existants, sans nouvelle inférence GPU. Seuil choisi exclusivement sur les médianes de groupes validation par maximum de macro-F1 ; départage du harness au plus petit seuil, décision `score >= seuil`. Publier ROC-AUC, précision moyenne (`pr_auc` dans ce code), macro-F1, balanced accuracy, rappel fuite/FPR, matrice TP/FP/TN/FN et Brier brut non calibré. Précision et F1 fuite peuvent être dérivés des mêmes effectifs, sans second moteur d'évaluation. Aucune moyenne mélangeant validation et test ; pas de seuil arbitraire 0,5 ni réglage sur test.
3. **Comparer aux contrôles gelés.** Même population/IDs et preprocessing prévus par le protocole, runs compatibles C0/C1/C2/C2b/C3 si disponibles/reproductibles ; baseline Vincent seulement si son export comparable existe. Vérifier les sources/empreintes et les règles de développement, pas comparer des chiffres isolés de rapports différents. Réutiliser les comparaisons appariées et bootstrap par groupes : 2000 tirages, graine `20260912`, IC95 implémentés pour ROC-AUC et macro-F1 uniquement. Seuils fixes pendant les tirages, incertitude de sélection du seuil et d'entraînement non couverte. Ne pas annoncer une supériorité si l'incertitude ne la soutient pas, ni ajouter de recherche de nouveaux modèles.
4. **Évaluer T1–T3 sans les régénérer.** Même checkpoint/scoring que T0, mêmes IDs, transformations SHA-256 officielles déjà tracées. Priorité aux écarts appariés d'AUC clip/groupe par rapport à T0. Particularité vérifiée du harness : il choisit un seuil validation distinct pour chaque run/transform ; ses F1/rappels de stress ne sont donc pas des performances à seuil T0 constant. Les étiqueter ainsi ; une analyse à seuil T0 fixe demanderait un complément explicite, pas prétendre qu'elle existe déjà. Une sensibilité à une transformation n'établit pas à elle seule une compréhension temporelle utile ni une robustesse terrain.
5. **Auditer la sortie texte sur 402 clips, pas huit exemples.** Seule nouvelle inférence prévue, checkpoint V1 inchangé. Sur les 208 validation : figer l'extraction et vérifier l'audit, sans améliorer le modèle ; puis audit final distinct sur les 194 test. Comparer la bande générée à la mesure DSP indépendante ; compter format valide, bande correcte sur tous les clips, invalidités/abstentions et désaccords entre classe générée et décision du score au seuil T0. Conserver chaque sortie brute et les erreurs, sans correction silencieuse. La cible textuelle actuelle est une bande dominante parmi quatre, pas une explication causale de fuite ; un calcul DSP direct est le comparateur de référence pour cette propriété. Relever séparément chargement, latence score seul/texte après chauffe et mémoire, sans revendication temps réel avant mesure.
6. **Rendre un verdict traçable.** Rapport séparant détection, apport par rapport aux contrôles, sensibilité aux stress, fidélité du texte et coût d'inférence ; tableaux d'erreurs, intervalles, effectifs et fichiers/commandes liés. `build_final_report.py` réutilise les sorties numériques, mais certaines conclusions Markdown sont codées en dur : prévoir leur neutralisation/relecture avant publication, pas les présenter comme calculées. Aucun ajustement de poids, prompt, seuil ou calibration pour réparer un résultat test. Une V2 ultérieure exige un protocole de développement distinct ; ce test inspecté ne redevient pas un test vierge. Les nouvelles acquisitions avec état connu et le continu réel restent nécessaires pour valider généralisation terrain, délais et fausses alertes par jour.

**Ordre d'exécution proposé après feu vert :** contrôles d'intégrité → détection T0 + comparateurs existants (CPU) → stress (CPU) → audit texte/latence (H100) → synthèse. Aucun de ces calculs n'est exécuté par cette demande de planification ; seuls les scripts/protocoles ont été relus et les notes actualisées.

## Protocole de livraison antérieur — historique de responsabilité

Les mentions ci-dessous réservant le calcul à Nevil décrivent la consigne de livraison antérieure, remplacée par le plan ci-dessus. Le format des exports, le gel du modèle et l'interdiction de réglage sur test restent applicables.

## Livraison finale TSLM — consigne Nevil transmise par Icham

Référence courante : [MODEL_EVAL_CONTRACT à 6dfdf63](https://github.com/IchamRaison/ehl-hackathon-zurich/blob/6dfdf63580bc15ce6a1cf0817b9da3569553aca1/docs/MODEL_EVAL_CONTRACT.md). T0 est publié dans [`artifacts/tslm_runs/tslm-v1/` à `186c45a`](https://github.com/IchamRaison/ehl-hackathon-zurich/tree/186c45a/artifacts/tslm_runs/tslm-v1) : uniquement `metadata.json` + `predictions.csv`, exactement `clip_id,probability_leak`, 402 lignes val/test v2 (208 + 194). Population gelée inchangée, ni label/fold/métadonnées d'acquisition dans le CSV, ni seuil appliqué aux probabilités. Nevil effectue l'évaluation finale, agrégation en groupes, sélection du seuil sur validation et comparaisons ; l'agent ML ne calcule pas les métriques finales. Diagnostics de développement limités à train/validation et contrôles d'intégrité d'export restent distincts.

Score implémenté et vérifié : sommes des log-probabilités de chaque continuation de classe, softmax sur les deux sommes, contexte/signal identiques et tokenisation/terminaison figées dans `scoring_spec.json` ; l'explication n'entre pas dans ce score. Pas de moyenne par token automatique si les longueurs diffèrent : c'est un changement de score, à déclarer et choisir sur développement seulement. Méthode et probabilités brutes non calibrées déclarées dans `threshold_rule`, sans pourcentage généré ni décision 0/1. Pas de calibration requise pour T0 ; cela ne signifie pas invariance universelle des métriques après agrégation par médiane des clips. Détails et contre-exemple dans [[Journal Icham#Revue des précisions Nevil — scores et stress]].

Recherche V1 terminée : 600 étapes sur les 598 clips train, trois checkpoints préannoncés aux époques 2/4/8 d'une même trajectoire ; époque 4 / étape 300 retenue selon la ROC-AUC de validation après médiane par groupe, égalité départagée par l'époque la plus précoce. `n_configs_compared = 3`, Qwen gelé inchangé, encodeur/projecteur adaptés. Validation = 208 clips mais seulement 42 groupes heuristiques, dont 8 non-fuite, pas des sessions indépendantes prouvées. Train/validation seuls motivent tout choix ; aucune adaptation d'après le test ou les sorties de son inspection. Ne pas lancer une nouvelle recherche après remise des prédictions sans protocole distinct.

Gel et reload vérifiés dans un processus neuf hors ligne : bundle `checksums.json` SHA `b95569c50f5e1bbf3533bddc92530b5f285eb25dec999f83999f60ffd912d1ae`, génération identique et scores WAV/brut/séries concordants avec une tolérance absolue de `1e-6`. Code entraînement `1199789f`, exporteur `5a29d6e`, 19 tests CPU passent. Le contrôleur officiel a validé T0 ; la conformité ne constitue pas une mesure de performance.

Procédure d'export conservée : `check_run.py --template` une seule fois dans un dossier neuf (écrasement sinon), puis `--run ... --inspect` après gel pour conformité seulement. L'exporteur refuse les dossiers existants, contrôle les SHA de `split_v2.csv` et de `split_v2_audit.csv`, mappe chaque `clip_id` vers son signal et écrit les probabilités sans arrondi à dix décimales. Les 402 IDs préremplis ne remplacent pas ce mapping. Provenance réelle et exactement deux colonnes/402 lignes val/test exigées même si le contrôleur tolère davantage. L'avertissement de sortie binaire n'entraîne pas d'échec du script. L'inspection inclut le test : ne pas en faire une boucle de réglage.

T1 inversion, T2 permutation des blocs de 250 échantillons à 8 kHz (31,25 ms), T3 randomisation de phase : **exécutés, conformes et publiés après T0**, transformations officielles Nevil avant preprocessing, même bundle `b95569c5…`, exporteur `5a29d6e` et méthode de score, aucun entraînement ou réglage sur ces stress. Chacun couvre 402/402 clips ; fichiers disponibles dans [`artifacts/tslm_runs/` à `7c04c9a`](https://github.com/IchamRaison/ehl-hackathon-zurich/tree/7c04c9a5636b2872334da17c54beb7016ffa00a3/artifacts/tslm_runs), dossiers `tslm-v1-T1/`, `tslm-v1-T2/` et `tslm-v1-T3/`, empreinte du bundle inchangée après exécution. C1 à C4 sont terminés, pas à réexécuter pour produire nous-mêmes les métriques finales. **Ancienne réserve RNG résolue :** correctif officiel `b23601a`, repris depuis `6dfdf63`, avec dérivation SHA-256 canonique et tests entre processus indépendants. Les variantes antérieures au correctif ne sont pas la référence utilisée. T2 ne laisse aucun reste hors permutation, mais peut laisser des blocs fixes. Ne pas requantifier/écrêter les signaux transformés via un retour WAV. Le CSV ne contient pas de description ; qualité du texte et événements continus restent des évaluations séparées. La consigne n'autorise pas à traiter une série de clips comme une chronologie terrain.

## Extension proposée — surveillance continue

Icham confirme le cas d'usage automatique ; [[Plan surveillance continue]] distingue désormais classification de fenêtres, tests logiciels de replay et performance sur événements réels. Les métriques ci-dessous restent utiles pour les clips. Elles ne donnent pas de délai de détection ni de fausses alertes/jour.

Sur acquisitions continues annotées et réservées : définir événements et appariement avant scoring, régler seuils/persistance sur développement seulement, puis rapporter rappel par événement, faux événements par appareil-heure surveillée, délai depuis apparition annotée, doublons, disponibilité et durée non surveillée. Séparer acquisitions/installations avant fenêtrage ; mêmes données et protocole de réglage pour les comparateurs. Un collage de clips ou les stress T1–T3 ne remplacent pas ces acquisitions. Ces extensions restent à convenir/implémenter avec Nevil ; ses résultats test de contrôles sont déjà publiés et ne doivent pas orienter les réglages du TSLM.

## Question testée

Dans le domaine expérimental accessible, le modèle distingue-t-il les clips fuite/non-fuite sur des groupes réservés, et reste-t-il fiable sous perturbation ? La réponse n'établit ni performance client, ni localisation, ni cause physique de la fuite.

## 1. Audit et split avant tout apprentissage

**Périmètre actuel déjà gelé :** T0 V1 utilise `split_v2.csv`, 598/208/194 clips, classification binaire avec les bruits environnementaux inclus parmi les non-fuites. Les principes d'audit et extensions historiques ci-dessous ne rouvrent pas ce split ni cette définition après apprentissage.

Télécharger les trois archives ; établir effectifs, sample rates, amplitudes, durées, anomalies et licences. Ne pas prendre une déclaration de l'archive pour une vérification complète. Dédoublonner par hash du signal décodé, pas seulement du fichier ; rechercher quasi-doublons/corrélations et captures concomitantes.

Les noms contiennent conditions, appareils et suffixes. Regrouper les captures d'un même événement supposé, y compris plusieurs capteurs/matériaux si le même événement les alimente. Documenter les heuristiques et leurs échecs. Des noms identiques en pression/vitesse ne prouvent pas une même session ; inversement des noms différents ne prouvent pas l'indépendance.

Construire les splits par groupe et vérifier que chaque classe est représentée. Choisir les proportions selon le nombre de groupes réellement indépendants ; ne pas forcer 70/15/15 si cela invalide les classes. Publier effectifs clips ET groupes. Mettre les fichiers ambigus en quarantaine ou limiter explicitement la revendication, décision avec Icham.

**Extension historique, non retenue comme remplacement de T0 :** un test distinct limité aux conditions du site expérimental et un challenge hors-domaine séparé pourraient étudier le rôle des bruits externes. Toute variante doit être nommée et convenue séparément ; elle ne change pas la population T0 v2 binaire avec bruit déjà livrée. Ne pas présenter son score comme celui du benchmark principal.

## 2. Développement et test final

Train pour fit, validation pour choix de paramètres et seuils, test final scellé jusqu'au gel. La baseline et le TSLM partagent IDs et cibles. Nevil contrôle les labels du test final ; partager uniquement un petit lot de démonstration explicitement séparé du test scellé pour construire l'UI.

Pour une campagne ultérieure convenue, une validation croisée groupée sur le développement peut être considérée ; elle ne remplace pas la sélection V1 déjà terminée ni le test réservé. Évaluer un appareil/matériau non vu en test complémentaire seulement si le support le permet. Aucun score de généralisation à un nouveau site avec un seul site.

## 3. Comparateurs

Pour T0 livré, les contrôles gelés C0/C1/C2/C2b/C3 et la comparaison finale relèvent du harness Nevil. La liste suivante conserve des options historiques, pas des entraînements supplémentaires requis ni une redéfinition des contrôles publiés :

- Majoritaire comme sanity check, pas comme seule baseline.
- Random Forest de Vincent sur features spectrales, parameters/seed enregistrés.
- TSLM initial avant adaptation, si capable de produire des sorties comparables ; invalidités comptées.
- TSLM adapté.
- Option si temps : CNN acoustique petit ; ne pas ajouter avant G2/G3.

Même prétraitement source et budget d'information. Hyperparamètres ajustés uniquement sur validation. Ne pas sélectionner la meilleure seed sur test.

## 4. Métriques

Les métriques finales ci-dessous sont à calculer par Nevil, pas par l'agent ML. Les diagnostics train/validation ayant sélectionné V1 restent explicitement des mesures de développement.

Macro-F1, précision et rappel fuite, matrice de confusion et taux de faux positifs parmi clips sans fuite. Inclure effectifs, invalidités et abstentions. Ne pas annoncer fausses alertes par jour sur des clips isolés sans chronologie continue. Ne pas annoncer délai d'apparition de fuite sans annotation d'onset.

Scores d'abstention : couverture et erreur parmi les prédictions retenues, avec performance globale incluant les non-réponses. Choisir seuil sur validation ; pas de confiance verbale auto-déclarée. Si comparaison entre modèles, tenir compte de couvertures différentes.

Rapporter incertitude par bootstrap de groupes ou autre méthode justifiée quand taille suffisante ; les clips corrélés ne sont pas des observations indépendantes. Publier limites si groupes trop peu nombreux.

Texte : taux de sorties valides, exactitude des propriétés numériques dans tolérances fixées avant scoring, assertions non supportées. Définir extraction et unités. Ne pas noter uniquement BLEU/ROUGE, longueur ou apparence convaincante.

## 5. Robustesse et ablations

T1–T3 officiels ont été exécutés sur le checkpoint figé, sans réentraînement ; leurs exports sont conformes, publiés et attendent l'évaluation Nevil. Les propositions supplémentaires ci-dessous demandent un protocole distinct ; elles ne sont pas des augmentations ajoutées à l'entraînement V1 après inspection du test.

Fixer sur validation la grille de bruit puis conserver un pool de bruit inédit pour test. Pour chaque perturbation : sample_id parent, noise_id, seed, SNR demandé/réalisé, input hash, modèle, prédiction et runtime. Reporter rappel/FPR/couverture par niveau. Le signal perturbé est une augmentation synthétique, clairement identifiée.

Tester agrégats vs évolution temporelle pour étudier si la dimension temporelle apporte une information utile. Une permutation de l'ordre peut ne pas changer un signal stationnaire ; ce résultat est recevable. Ne pas prétendre que la heatmap d'attention explique causalement la décision.

Audit shortcut : entraînement/prediction sans noms ni métadonnées sources, sondage facultatif métadonnées seules pour révéler confusions, tests de changement de gain pour voir si simple amplitude suffit. Distinguer ablation d'analyse et version du modèle soumise.

## 6. Artefacts d'un run

**Livraison ML finale :** dossier contenant uniquement `metadata.json` et `predictions.csv` au contrat, avec run_id, provenance du checkpoint/code/configuration, SHA du split et de son audit de mapping, méthode de score et transformation. Commandes, logs, preuves de reload et environnement restent hors de ce dossier. Pas de `predictions.jsonl` alternatif ni de `metrics.json` final produit par l'agent ML.

**Évaluation Nevil :** ses métriques et rapports renvoient aux runs reçus et doivent être reproductibles depuis ces prédictions sauvegardées. Cela ne transfère pas la responsabilité du calcul final au producteur ML.

Ne jamais remplacer une sortie ratée par une bonne prédiction inventée. Exemples visuels issus d'un lot démo autorisé, pas cherry-picking caché sur le test final. Montrer au moins une limite réelle dans la présentation si disponible, sans inventer un échec pour le spectacle.

## Acceptation

- [x] V1 figée, reload neuf et export T0 conforme publiés ; 402 IDs, deux colonnes exactes, aucune métrique finale côté ML.
- [x] Stress T1/T2/T3 terminés et contrôlés avec le même checkpoint, chacun 402/402 clips.
- [x] Stress T1/T2/T3 publiés à `7c04c9a`, disponibles pour l'évaluation finale Nevil ; aucun message envoyé par l'agent.
- [ ] Doublons/groupes/splits audités et limites approuvées.
- [ ] Baseline et TSLM évalués sur mêmes exemples.
- [x] Sélection V1 train/validation uniquement ; aucun seuil appliqué aux probabilités livrées.
- [ ] Seuil final et métriques établis par Nevil selon le protocole gelé.
- [ ] Métriques reproductibles, effectifs et échecs inclus.
- [ ] Source expérimentale et perturbations synthétiques visibles.
- [ ] Aucun résultat clinique/terrain/localisation extrapolé.
