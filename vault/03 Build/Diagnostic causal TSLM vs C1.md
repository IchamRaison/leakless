# Diagnostic causal TSLM vs C1

Icham — 2026-09-13

## Point d'entrée du goal

Le prompt reste un simple renvoi vers cette note, comme demandé par Icham :

```text
/goal /home/animus/ehl-hackathon-zurich-vault/03 Build/Diagnostic causal TSLM vs C1.md implémente ça
```

Maintenir ici le périmètre, l'ordre et les critères ; maintenir l'avancement et les preuves dans [[Diagnostic causal - exécution]]. Ne pas dupliquer le plan dans le prompt ni créer un second goal. Les dix étapes ci-dessous restent la référence, y compris les conditions de diversification et la preuve produit finale.

## Statut et objectif

**Périmètre du goal élargi par Icham le 13 septembre aux étapes 1 à 10 ci-dessous.** Le prompt `/goal` renvoie à ce fichier : celui-ci définit donc l'objectif complet, pas seulement le diagnostic. Les étapes 6 à 10 font désormais partie du travail à réaliser, avec leurs conditions de passage ; terminer le diagnostic ne termine plus le goal. [[Diagnostic causal - exécution]] conserve l'état réel et les preuves. Cette extension documentaire ne constitue ni une expérience exécutée ni une reprise automatique de l'ancienne recette V2.

> Diagnostiquer les causes du retard du TSLM face à C1, appliquer et mesurer les corrections justifiées, diversifier globalement les données si le progrès reste insuffisant et mesurer cet apport, figer puis évaluer le modèle sur une réserve indépendante, enfin intégrer et tester la surveillance continue et l'utilité du langage et des séries pour l'investigation.

La première phase conserve son objectif causal :

> Expliquer l'écart de performance observé entre les pipelines TSLM et C1 sur développement. Examiner les causes plausibles — données, représentation, supervision, optimisation, utilisation du signal par Qwen et scoring — puis les départager par des expériences contrôlées. Distinguer causes démontrées, hypothèses non soutenues dans les conditions testées et inconnues avant de proposer une nouvelle recette.

« Exploite moins bien les données » reste une explication possible, pas un fait établi par une AUC inférieure. C1 peut aussi bénéficier de particularités d'acquisition. Le cap produit demeure [[Plan V2 - fiabilité et parité des scores#7. Objectif final — démontrer la valeur du TSLM]] ; le diagnostic n'en démontre pas encore l'utilité.

## Acquis et limites — ne pas recommencer les preuves terminées

- **Démontré :** deux causes d'écarts numériques, conversion float32 avant FFT et dépendance de l'encodage à la taille du lot. Correctifs versionnés, gates A/C sur 209 entrées et rechargements neufs à écart maximal nul (`04d53b6`). Cela démontre la parité dans les contextes vérifiés, pas la qualité du classement ni la justesse de l'objectif appris.
- **Observé :** les six fits sont désormais vérifiés, bilan complet dans [[Diagnostic causal - exécution]], artefacts `b65042a` après `83bcbe0` / `fa1b9b4`. Moyennes AUC groupe : A=0,671474 ; C=0,562500 ; C1=0,948718 sur les mêmes folds de développement. Aucun progrès final ni refit revendiqué.
- **Observé :** la sonde linéaire sur les 256 valeurs TimeNet exactes classe moins bien que les neuf descripteurs C1, sur les trois folds train. Cela ne prouve pas que toute information utile a disparu. Preuve `docs/evidence/tslm-v2/train-diagnostic-001/probes.json`.
- **Observé :** Qwen est gelé, les composants acoustiques changent et leurs gradients sont non nuls. L'autopsie de huit groupes train au checkpoint V1 final n'a pas trouvé une perte dominée par la description. Elle n'écarte ni un problème en début d'apprentissage ni un mauvais signal de gradient.
- **D0/D1 exécutés :** sensibilité aux entrées et apprentissage partiel mesurés ; changer la précision de la tête modifie les scores à poids fixes, mais ne corrige pas uniformément le classement hors groupes. D1 : 3 588 forwards, aucun apprentissage, preuves A `640d29c` / C `9c163d4`. Aucun passage automatique au FP32 ; prochaine question, l'information exploitable par une sonde non linéaire sur les mêmes 256 valeurs. [[Diagnostic causal - exécution#D1 — résultats et portée]].

Les commandes, empreintes et limites détaillées restent dans [[V2 ML - exécution]]. Les résultats V1 test ne sont pas directement comparables aux folds V2 train ; aucun réglage ne doit partir du test déjà consulté.

**D2 désormais exécuté et vérifié :** sur les mêmes 256 valeurs et trois folds, le lecteur HGB obtient 0,854167 d'AUC groupe réservée moyenne, contre 0,636218 pour la sonde linéaire et 0,937500 pour C1 fixe. Gains sur les trois folds face à la sonde linéaire ; aucune absence globale d'information prédictive démontrée. Cela justifie le contrôle suivant du chemin d'apprentissage Qwen, pas une adoption du boosting comme TSLM ni une collecte immédiate. [[Diagnostic causal - exécution#D2 — résultats vérifiés et portée]].

## Questions à départager — pas six chantiers simultanés

**Actualisation D3 :** mémorisation32 réussie au pas400 avec Qwen gelé/loss complète, reload neuf exact. À la demande d'Icham, observer ensuite ce checkpoint fixe sur les98 réservés du fold0, en extension séparée sans apprentissage ni réglage. Ne pas traiter LoRA comme déblocage nécessaire ni32/32 comme preuve de qualité hors entraînement. [[Diagnostic causal - exécution#D3 — résultat final et extension sur groupes réservés]].

| Cause possible | Contraste contrôlé proposé | Limite d'interprétation |
|---|---|---|
| Données / comparaison | Vérifier mapping, cibles et groupes ; analyser C1 et TSLM sur les mêmes sous-populations de développement, avec support suffisant des deux classes. | Les groupes heuristiques ne prouvent pas l'indépendance des sessions ; un effet d'acquisition n'est pas automatiquement un effet de fuite. |
| Représentation | Réutiliser la sonde existante ; si nécessaire, comparer un lecteur non linéaire borné des mêmes séries à un lecteur des descripteurs C1, mêmes folds. | L'échec d'une sonde ne démontre pas l'absence d'information ; un contraste doit déclarer aussi les différences de capacité/budget. |
| Supervision | Auditer positions, masques et pertes/gradients classe versus description, à l'initialisation et après apprentissage ; intervention de pondération seulement si justifiée. | Des gradients non nuls ou une petite perte finale ne prouvent pas un apprentissage discriminant. |
| Optimisation | Comparer comportement train et groupes réservés ; si nécessaire, test de mémorisation sur un petit train équilibré et fixé, budget limité. | Réussir à mémoriser ne valide pas la généralisation ; échouer ne localise pas à lui seul la cause. |
| Utilisation du signal | À checkpoint et prompt figés, échanger les représentations entre clips de développement ; pour C, séparer interventions sur séries et texte des neuf mesures. | Une variation du score montre une sensibilité, pas une utilisation utile. Les entrées désaccordées peuvent être hors distribution ; pas de conclusion physique à partir de ce seul test. |
| Scoring | Contrôler token par token les deux continuations complètes, contexte causal et alignement avec les positions supervisées. | La parité entre interfaces ne détecte pas une erreur commune à toutes. Tout autre libellé ou score est une variante déclarée, jamais un réglage sur test. |

## Plan global en dix étapes

Préalables déjà réalisés : inventaire des six fits, contrôles de parité antérieurs et observations D0. Réutiliser leurs preuves ; ne pas les relancer pour recommencer ce plan. La numérotation ci-dessous reprend l'enchaînement expliqué à Icham.

1. **Précision numérique — exécutée :** D1 compare les têtes BF16, FP32 et FP32 réarrondie sur les mêmes checkpoints et entrées, sans apprentissage ; tests, préinscription et artefacts conservés. Ne pas relancer.
2. **Interprétation :** quantifier l'effet observé et choisir le contraste suivant ; ne pas adopter une correction numérique sur la seule apparence d'une grille plus fine.
3. **Représentation — exécutée :** D2 teste un lecteur non linéaire borné sur les entrées exactes, mêmes groupes de développement et comparaison C1 ; résultats et limites ci-dessus, ne pas relancer.
4. **Apprentissage :** suivant les preuves, mémorisation sur 32 clips fixés, journalisation des pertes/gradients/mises à jour, contournement de Qwen et mesures C1 par voie entraînable. Chaque contrôle doit départager une question ; pas six campagnes simultanées.
5. **Bilan causal :** causes démontrées, hypothèses non soutenues, inconnues et corrections justifiées ; preuve et limite de chaque verdict, justification des contrôles conditionnels omis. Une inconnue ne devient pas une cause acquise ni un motif de recherche illimitée.

Pour les étapes 1 à 5, annoncer avant chaque expérience données, composants fixes, interventions, graines, critères, budget numérique et arrêt. Les étapes 3 et 4 peuvent être ordonnées selon les observations de l'étape 2. Réutiliser reçus, sondes et harness existants.

Conserver séparément qualité du classement, décisions au seuil fixé et cohérence du texte brut/affiché. Une phrase corrigée par DSP/gabarit ne constitue pas un gain du modèle. Une cause de score incorrect n'est pas nécessairement la cause de tout l'écart à C1.

### 6. Appliquer les corrections justifiées et mesurer la nouvelle version

- Partir du bilan de l'étape 5 ; implémenter les corrections étayées avec leurs tests de régression. Préserver les versions historiques et vérifier la parité des chemins préparation, entraînement, inférence et export concernés.
- Préannoncer une campagne bornée : configurations, budget, critère de sélection et définition du progrès suffisant. Entraîner puis comparer la nouvelle version aux références actuelles et à C1 sur les mêmes groupes de développement ; aucune sélection sur le test officiel ou externe.
- Publier classement, rappel/fausses alertes, résultats clip/groupe, incertitudes et cohérence des sorties. Une baisse de loss seule ne suffit pas. Si aucune correction n'est soutenue par le diagnostic, le documenter au lieu d'inventer une recette.
- **Preuve de fin :** corrections et tests liés à leurs causes, recettes/checkpoints/essais tracés, comparaison vérifiée et décision motivée de passer à 7 ou directement à 9. Un résultat insuffisant reste un résultat, pas un succès de qualité.

### 7. Diversifier globalement les données si le progrès reste insuffisant

- Déclencher cette étape après le bilan des diagnostics et corrections, selon les critères annoncés en 6 ; ne pas remplacer les contrôles restants par davantage de données. Si l'élargissement n'est pas nécessaire, consigner pourquoi 7 et 8 ne sont pas activées.
- Auditer la couverture des deux classes : types/intensités de fuite, situations normales, sites, matériaux, capteurs/montages, hydraulique et bruits. Réutiliser [[Datasets utiles pour PIPE]], puis vérifier formats, annotations, provenance, droits, recouvrements et acquisitions liées avant intégration.
- Définir un nouveau protocole versionné train/développement/confirmation, par acquisition/site avant fenêtrage. Aghashahi peut être envisagé pour un futur entraînement et Hong Kong pour une autre réserve après audit ; ce choix reste explicite et conditionnel, pas acquis par cette extension. Ne pas placer les capteurs ou fenêtres d'une même expérience des deux côtés.
- **Preuve de fin :** audit et manifeste du corpus, rôles des sources, séparation vérifiée, réserve de confirmation intacte et décision d'intégration tracée. Collecte matérielle, données privées ou nouvelles dépenses exigent l'autorité et les moyens correspondants ; leur absence est signalée, pas compensée par des données inventées.

### 8. Mesurer l'apport des nouvelles données

- Comparer la même recette avec et sans ajout sur les mêmes groupes de développement réservés. Garder le preprocessing, le scoring et les règles de sélection comparables ; fixer le budget de calcul avant essais pour ne pas confondre ajout de données et allongement d'entraînement.
- Vérifier que les acquisitions ajoutées ne recouvrent pas les réserves, puis rapporter les résultats par source et sous-population lorsque les effectifs le permettent. Déclarer le nombre réel d'essais et les limites de variance.
- **Preuve de fin :** contraste apparié et reproductible, apport positif/nul/négatif mesuré, modèle retenu selon la règle annoncée. Pas d'adoption automatique du modèle enrichi. Étape omise uniquement avec la justification conditionnelle de 7.

### 9. Figer puis confirmer sur une réserve indépendante

- Figer checkpoint, preprocessing, prompt, classes, calcul du score, seuil éventuel, restitution, runtime et empreintes ; recharger dans un processus neuf et vérifier la reproductibilité avant l'évaluation indépendante.
- Évaluer le modèle retenu et C1 avec le harness réutilisé sur la réserve auditée, non utilisée pour choisir les corrections, les données ou les réglages. Publier performances, couverture, incertitudes, erreurs, cohérence du texte et coûts/latences pertinents. Une source réaffectée au train ne peut plus jouer ce rôle.
- Les éventuels stress utilisent le même checkpoint sans réentraînement, avec transformations reproductibles et seuil T0 conservé pour les décisions. Aucun réglage après lecture de la réserve ; une nouvelle adaptation appartiendrait à un nouveau cycle, pas à cette confirmation.
- **Preuve de fin :** artefacts gelés, reload vérifié, prédictions et rapport indépendant reproductible avec verdict favorable ou défavorable. Sans réserve exploitable, cette étape reste incomplète ; ne pas rebaptiser du développement « test indépendant ».

### 10. Intégrer et tester la surveillance continue et la valeur du TSLM

- Réutiliser l'API/UI de Safoan et le simulateur existant : bundle réel, contrat convenu, réception et inférence de bout en bout, reload et pannes testés. Un test local du simulateur n'est pas une preuve de raccordement applicatif.
- Implémenter et tester le flux causal horodaté, les doublons/pertes/retards, la file bornée et la santé de surveillance. Versionner ouverture, persistance et fin d'événement ; ne pas produire une nouvelle alerte par fenêtre ni confondre panne, acquittement et disparition de fuite. Mesurer le débit soutenu et la latence avant toute promesse de cadence.
- Distinguer la recette fonctionnelle en replay de la validation sur acquisitions continues réelles annotées, avec sessions/installations réservées et durée surveillée connue. Mesurer rappel événementiel, faux événements par appareil-heure, délai, répétitions et couverture. Les données continues nécessaires à cette preuve constituent un besoin distinct de l'élargissement conditionnel du train en 7.
- Comparer **classifieur + DSP + gabarit**, **classifieur + mesures/contexte + Qwen**, **TSLM + séries/contexte**, sur les mêmes événements réservés, historiques disponibles et questions fixées avant comparaison. Mesurer fidélité aux preuves/à la chronologie, comparaison des périodes, gestion de l'incertitude, utilité pour l'investigation, détection et coût/latence. Ne pas inventer une modalité ou une durée absente ; conserver texte brut, mesures et fallback identifié.
- **Preuve de fin :** intégration sur poids réels, tests du continu, rapport événementiel et benchmark à trois approches vérifiés, avec limites et passation publiées. Une démo ou un replay seul ne valide pas le terrain. Si le langage ou l'accès aux séries n'apporte pas de gain, le constater ; la supériorité du TSLM n'est pas présumée. Une preuve manquante reste incomplète ou bloquée, pas validée par défaut.

Références d'exécution de 10 : [[Plan surveillance continue]], [[Plan simulateur de capteur]], [[Plan V2 - fiabilité et parité des scores#7. Objectif final — démontrer la valeur du TSLM]]. Aucun achat, déploiement client, notification externe ou commande physique n'est implicite.

## Critère de complétion du goal élargi

Le goal ne s'arrête plus à la matrice causale ou à une proposition de recette. Les preuves des étapes 6, 9 et 10 sont requises, ainsi que celles de 7 et 8 si leur condition est déclenchée. Les omissions ne concernent que les contrôles réellement conditionnels et doivent être justifiées ; aucune étape obligatoire incomplète ne peut être remplacée par un statut « planifié ». Publier code, configurations, commandes, tests, données/provenance autorisées, artefacts, rapports et passation, avec revue des conclusions. Ne pas garantir un gain à l'avance ni transformer une qualité insuffisante en promesse de fiabilité terrain.

## Checklist enrichie après la revue de Claude

Analyse transmise par Icham, portant sur le code à `1b9b3b3`. Icham demande d'intégrer les contrôles manquants sans interrompre le travail en cours. **Ajouts de planification uniquement : aucune journalisation modifiée, aucune nouvelle inférence ou expérience d'apprentissage exécutée.** Le test de mémorisation, la sonde non linéaire et les échanges de signaux étaient déjà prévus ; les préciser ne crée pas une seconde campagne.

- [ ] **Journalisation interprétable avant le prochain apprentissage :** conserver la loss globale, mais ajouter les sommes et effectifs réels pour classe complète, premier token discriminant, description et EOS, ainsi que la NLL par clip. Journaliser séparément la NLL binaire obtenue avec le score officiel des deux continuations complètes. Ne pas supposer 148 tokens par batch ni réduire silencieusement le score au premier token. Agréger l'époque par sommes/effectifs, pas par moyenne non pondérée des batches ; distinguer mesures en ligne et évaluation d'un checkpoint figé.
- [ ] **Mesurer réellement l'apprentissage de classe sur train :** checkpoint identifié, mode eval et ensemble train fixé ; NLL binaire, classement et erreurs séparés, avec contrôle constant 0,5 et contrôle de fréquence de classe appris sur le train concerné. Les pertes de batches successifs ne remplacent pas ce contrôle. Aucun test officiel/externe utilisé.
- [ ] **Préciser le test de mémorisation déjà prévu :** proposition de 32 clips train, 16/16, un clip par groupe distinct si disponible, sélection déterministe sans regarder les scores, initialisation fraîche et budget maximal proposé de 1 000 étapes à préinscrire avant lancement. NLL binaire <0,1 et faibles erreurs sur ces mêmes clips seraient des témoins de capacité de mémorisation, pas une preuve que seul un problème de généralisation demeure. Un échec au budget fixé laisse plusieurs causes possibles ; aucune prolongation automatique.
- [ ] **Contrôle de contournement de Qwen, si nécessaire :** même entrée, encodeur/projecteur, sous-ensemble, initialisation commune des modules partagés et budget ; moyenne des sorties puis tête linéaire/BCE, séparée du modèle livré. Une réussite face à un TSLM qui échoue incriminerait le chemin supplémentaire Qwen/prompt/scoring/objectifs, sans départager à elle seule ces composants. Déclarer que la tête et la fonction de perte changent aussi ; ne pas présenter ce contrôle comme une intervention sur Qwen seul.
- [ ] **Diagnostic d'échelle et de transmission :** mesurer les distributions des normes des embeddings projetés et des tokens Qwen, masques/positions, gradients et mises à jour relatives des modules adaptés, avant/après clipping, avec dtypes et epsilon de l'optimiseur. Une norme différente n'est pas automatiquement un défaut. Réutiliser les échanges de signaux prévus ; une entrée nulle est seulement un témoin hors distribution, jamais une preuve universelle que Qwen ignore le signal. Définir la tolérance et vérifier aussi les entrées réellement injectées.
- [ ] **Sonde non linéaire déjà prévue :** un seul MLP ou modèle de boosting préannoncé sur les 256 valeurs exactes, mêmes folds groupés et budget limité face à C1. Ne pas essayer plusieurs familles sans les compter. Un échec ne démontre pas un plafond informationnel absolu.
- [ ] **Descripteurs C1 par une voie entraînable :** si les contrôles précédents le justifient, transmettre les neuf mêmes mesures numériques à un adaptateur entraînable et comparer à leur exposition textuelle. Figer ordre, normalisation fit sur train et budget. Ce sont neuf descripteurs statiques, pas une nouvelle série temporelle ni neuf nouveaux capteurs ; un gain ne démontrerait pas une compréhension temporelle. Contraste diagnostique distinct, pas une nouvelle recette produit adoptée d'avance.

### Réserves à conserver dans le verdict

**Perte proche de ln 2 ≠ classement au hasard démontré.** `loss × nombre réel de tokens / nombre réel de clips` reconstruit la NLL de la réponse supervisée, pas directement celle du score binaire renormalisé. Avec `a` et `b` les log-probabilités des deux continuations complètes, `NLL_binaire = -logp_classe_vraie + logsumexp(a,b)`. Même une vraie NLL binaire proche de ln 2 ne suffit pas : contre-exemple mathématique, scores 0,5001 pour tous les positifs et 0,4999 pour tous les négatifs donnent une NLL de 0,692947, proche de ln 2 =0,693147, avec classement parfait. Ce n'est pas un résultat PIPE. Sur des classes déséquilibrées, comparer aussi au prédicteur constant de fréquence train, pas seulement au 50/50. [Définition de la log-loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).

**Provenance corrigée :** `supervision.json` examine huit clips/huit groupes avec les poids V1 **époque4**, SHA `183a1b1c…`, et le preprocessing canonique V2. La petite contribution descriptive est constatée sur cette autopsie, pas écartée à tous les stades. `training-report.json:last_loss` concerne le dernier batch de la campagne V1 à huit époques, pas une moyenne des598 clips ni les poids retenus. Les logs V2 sont des batches effectifs token-pondérés : derniers batches publiés A0/C0 =4 clips/74 tokens, A1 =5 clips/92 tokens. A0/A1/C0 ne sont pas trois folds d'un même candidat. Sources : `src/pipe/tslm/campaign.py`, `docs/evidence/tslm-v1/{training-report,selection}.json`, `docs/evidence/tslm-v2/train-diagnostic-001/supervision.json` et les trois `training.jsonl` publiés. Relecture des artefacts, aucune nouvelle métrique modèle calculée.

**Ne pas écarter prématurément Qwen/scoring/optimisation.** Un modèle de langage gelé peut être conditionné par des entrées entraînables ; son gel ne suffit pas à prouver une impossibilité d'apprentissage. Cela ne garantit pas la réussite de notre variante C. [Travail primaire sur le prompt tuning](https://arxiv.org/abs/2104.08691). De même, la quasi-invariance idéale d'Adam à un facteur constant ne prouve pas l'innocuité de notre normalisation : vérifier epsilon, clipping, pondération variable des batches et précision numérique dans le chemin réel avant d'écarter cette hypothèse. [Algorithme AdamW du runtime PyTorch2.8](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html). Le scoring reste donc dans la checklist ; « sous-apprentissage même sur train » est une hypothèse à mesurer, pas une cause déjà démontrée par le plateau seul.

## Discipline de développement

Les trois folds groupés à l'intérieur des 598 train restent la référence de comparaison. Au sein de chaque contraste, tenir fixes données, composants non visés, initialisation lorsqu'un entraînement est requis et protocole de score, sauf le facteur étudié. Prétraitements appris uniquement sur le train du fold. Aucun checkpoint V1 entraîné sur tous les 598 clips ne sert à revendiquer une performance hors-fold.

Consigner tous les essais, même abandonnés, sondes, checkpoints et variantes de scoring compris. La validation déjà utilisée ne redevient pas une confirmation indépendante. Les 194 anciens test restent historiques ; la réserve externe Aghashahi figée (`84007801…`) demeure sans score et ne sert pas à choisir les corrections.

## Reprise pratique de l'ancienne campagne — historique

Dernier contrôle SSH en lecture seule le 13 septembre à **03:23:29 Europe/Paris** : A2, C/fold1 et le dernier C/fold2 annoncent `completed: true` dans leurs logs ; aucun processus `run_v2_campaign.py` encore actif observé. Leurs trois reçus ne sont pas encore intégralement revérifiés/rapatriés. A0/A1/C0 sont déjà vérifiés et publiés. Aucun processus n'a été arrêté dans ce tour ; aucun nouveau fit, classement, refit, seuil ou score externe lancé. Ne pas relancer les fits pour récupérer leurs résultats.

Runtime : `/home/hicham/pipe-v0/artifacts/v2-campaign-c13fd47-001`, logs `/home/hicham/pipe-v0/quality-v2-001/fit-{A,C}{0,1,2}-c13fd47.log`. Utiliser le contrôle existant `finished()` pour vérifier les reçus/empreintes, sans créer de fichier dans les dossiers scellés. Checkout local sain : `/home/animus/ehl-hackathon-zurich-v2-recovery`, branche `feat/icham-v2-reliability` ; ne pas commiter dans l'ancien dépôt Git endommagé.

**Mise à jour :** les trois reçus manquants sont maintenant récupérés/vérifiés, sans réentraîner. Le protocole et la prochaine action du nouvel objectif sont dans [[Diagnostic causal - exécution]].

## Dernière étape conditionnelle — diversifier les données acoustiques

Ce dernier recours d'amélioration correspond à l'étape 7 du goal élargi ; les étapes 8 à 10 viennent ensuite. Ce n'est plus la fin de l'objectif global.

**Ajout demandé par Icham le 13 septembre ; planifié, non lancé.** À activer seulement si les autres pistes diagnostiques et les corrections justifiées n'apportent pas une amélioration suffisante sur développement. Juger cette amélioration avec les critères de qualité annoncés avant comparaison, pas avec la seule baisse de loss ; aucun seuil de gain arbitraire ajouté ici.

**Précision d'Icham : la diversité globale est visée, pas seulement celle des sans-fuite.** La faible diversité des négatifs est un constat particulier. Examiner aussi les types et intensités de fuite, sites, tuyaux/matériaux, capteurs et montages, pressions/débits, bruits et états de fonctionnement. D1 est maintenant exécuté ; les autres diagnostics conditionnels non exécutés ne sont pas des échecs, et leur omission doit être justifiée, pas autorisée par ce complément.

- Élargir les acquisitions **avec fuite et réellement sans fuite** dans ces conditions distinctes. Ne pas remplacer les vrais sans-fuite par davantage de bruits environnementaux, ni supposer que plus de clips signifie plus de situations indépendantes.
- Privilégier des acquisitions **fuite/sans fuite comparables**, avec conditions et sessions documentées. Chercher davantage de situations indépendantes, pas simplement davantage de fenêtres découpées dans les mêmes enregistrements.
- Avant intégration : convenir du responsable, des moyens de collecte et du protocole ; vérifier annotations, provenance et droits, puis définir un nouveau protocole versionné avec séparation par acquisition et une réserve de confirmation intacte. Ne pas modifier rétroactivement les folds actuels ni incorporer le test officiel ou la réserve Aghashahi dans l'apprentissage actuel. Une réaffectation d'Aghashahi dans une future campagne est une option discutée, pas autorisée : elle exige une décision explicite et un autre dispositif de confirmation, sans acquisitions/modalités liées entre les partitions.

Motif vérifié : les598 clips train comptent294 fuites,242 vrais sans-fuite et62 bruits ; les242 sans-fuite ne représentent que7 groupes heuristiques, dont deux totalisent197 clips. La [source Zenodo](https://zenodo.org/records/18631450) situe les deux classes de tuyaux sur le même site principal à Dongguan. Ce sont des limites de diversité, **pas la cause démontrée de tout l'écart TSLM–C1**. Comptages et limites : [[Diagnostic causal - exécution#Audit des populations et des cibles]].

[[Datasets utiles pour PIPE#Revue du 13 septembre — diversité globale]] réexamine, à la demande d'Icham, les sources déjà trouvées : Hong Kong candidat à auditer pour une future extension acoustique ; Aghashahi reste la réserve déjà figée ; les autres modalités ont des usages distincts. Revue documentaire effectuée, aucun nouveau téléchargement de données, collecte, intégration ou réentraînement lancé.
