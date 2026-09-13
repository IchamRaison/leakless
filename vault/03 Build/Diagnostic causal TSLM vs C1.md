# Diagnostic causal TSLM vs C1

Icham — 2026-09-13

## Statut et objectif

**Objectif d'implémentation désormais autorisé par Icham.** [[Diagnostic causal - exécution]] décrit les preuves courantes, le budget D0 fixé et les travaux réellement lancés. Le texte ci-dessous conserve le cadrage et la checklist validés ; les anciennes mentions « planification uniquement » décrivent leur rédaction initiale et ne bloquent plus l'exécution contrôlée. La suite V2 de refit/confirmation reste en pause : pas de nouvelle recette avant le diagnostic.

> Expliquer l'écart de performance observé entre les pipelines TSLM et C1 sur développement. Examiner les causes plausibles — données, représentation, supervision, optimisation, utilisation du signal par Qwen et scoring — puis les départager par des expériences contrôlées. Distinguer causes démontrées, hypothèses non soutenues dans les conditions testées et inconnues avant de proposer une nouvelle recette.

« Exploite moins bien les données » reste une explication possible, pas un fait établi par une AUC inférieure. C1 peut aussi bénéficier de particularités d'acquisition. Le cap produit demeure [[Plan V2 - fiabilité et parité des scores#7. Objectif final — démontrer la valeur du TSLM]] ; le diagnostic n'en démontre pas encore l'utilité.

## Acquis et limites — ne pas recommencer les preuves terminées

- **Démontré :** deux causes d'écarts numériques, conversion float32 avant FFT et dépendance de l'encodage à la taille du lot. Correctifs versionnés, gates A/C sur 209 entrées et rechargements neufs à écart maximal nul (`04d53b6`). Cela démontre la parité dans les contextes vérifiés, pas la qualité du classement ni la justesse de l'objectif appris.
- **Observé :** les six fits sont désormais vérifiés, bilan complet dans [[Diagnostic causal - exécution]], artefacts `b65042a` après `83bcbe0` / `fa1b9b4`. Moyennes AUC groupe : A=0,671474 ; C=0,562500 ; C1=0,948718 sur les mêmes folds de développement. Aucun progrès final ni refit revendiqué.
- **Observé :** la sonde linéaire sur les 256 valeurs TimeNet exactes classe moins bien que les neuf descripteurs C1, sur les trois folds train. Cela ne prouve pas que toute information utile a disparu. Preuve `docs/evidence/tslm-v2/train-diagnostic-001/probes.json`.
- **Observé :** Qwen est gelé, les composants acoustiques changent et leurs gradients sont non nuls. L'autopsie de huit groupes train au checkpoint V1 final n'a pas trouvé une perte dominée par la description. Elle n'écarte ni un problème en début d'apprentissage ni un mauvais signal de gradient.

Les commandes, empreintes et limites détaillées restent dans [[V2 ML - exécution]]. Les résultats V1 test ne sont pas directement comparables aux folds V2 train ; aucun réglage ne doit partir du test déjà consulté.

## Questions à départager — pas six chantiers simultanés

| Cause possible | Contraste contrôlé proposé | Limite d'interprétation |
|---|---|---|
| Données / comparaison | Vérifier mapping, cibles et groupes ; analyser C1 et TSLM sur les mêmes sous-populations de développement, avec support suffisant des deux classes. | Les groupes heuristiques ne prouvent pas l'indépendance des sessions ; un effet d'acquisition n'est pas automatiquement un effet de fuite. |
| Représentation | Réutiliser la sonde existante ; si nécessaire, comparer un lecteur non linéaire borné des mêmes séries à un lecteur des descripteurs C1, mêmes folds. | L'échec d'une sonde ne démontre pas l'absence d'information ; un contraste doit déclarer aussi les différences de capacité/budget. |
| Supervision | Auditer positions, masques et pertes/gradients classe versus description, à l'initialisation et après apprentissage ; intervention de pondération seulement si justifiée. | Des gradients non nuls ou une petite perte finale ne prouvent pas un apprentissage discriminant. |
| Optimisation | Comparer comportement train et groupes réservés ; si nécessaire, test de mémorisation sur un petit train équilibré et fixé, budget limité. | Réussir à mémoriser ne valide pas la généralisation ; échouer ne localise pas à lui seul la cause. |
| Utilisation du signal | À checkpoint et prompt figés, échanger les représentations entre clips de développement ; pour C, séparer interventions sur séries et texte des neuf mesures. | Une variation du score montre une sensibilité, pas une utilisation utile. Les entrées désaccordées peuvent être hors distribution ; pas de conclusion physique à partir de ce seul test. |
| Scoring | Contrôler token par token les deux continuations complètes, contexte causal et alignement avec les positions supervisées. | La parité entre interfaces ne détecte pas une erreur commune à toutes. Tout autre libellé ou score est une variante déclarée, jamais un réglage sur test. |

## Ordre proposé et règle d'arrêt

1. **Inventaire avant calcul :** récupérer/vérifier les résultats de l'ancienne campagne et construire la matrice des preuves déjà disponibles. Réutiliser les reçus, prédictions, sondes et harness existants ; aucun nouveau moteur d'évaluation.
2. **Premier bloc sans réentraînement :** vérifier la comparaison et l'alignement supervision/scoring, puis tester l'utilisation du signal si les vérifications précédentes ne suffisent pas. Fixer à l'avance checkpoint, groupes, interventions, graines, mesures et résultats attendus.
3. **Un seul contraste supplémentaire à la fois :** choisir représentation, supervision ou optimisation selon les observations précédentes, pas selon la recette qu'on aimerait essayer. Annoncer avant lancement un budget numérique de runs/étapes et la condition d'arrêt ; ces budgets restent à convenir, aucune campagne n'est autorisée par cette note.
4. **Restituer avant de corriger :** pour chaque hypothèse, preuve, alternatives encore compatibles, verdict limité aux conditions testées et prochaine expérience réellement discriminante. Si les données ne départagent pas les causes, conclure « inconnu » et préciser l'observation manquante, sans prolongation automatique.
5. **En dernier recours, élargir les acquisitions :** si le diagnostic et les essais de correction justifiés n'améliorent pas suffisamment le modèle sur développement, activer la collecte conditionnelle décrite en fin de plan. Ce chantier vient après les autres pistes, pas en parallèle par défaut.

Conserver séparément qualité du classement, décisions au seuil fixé et cohérence du texte brut/affiché. Une phrase corrigée par DSP/gabarit ne constitue pas un gain du modèle. Une cause de score incorrect n'est pas nécessairement la cause de tout l'écart à C1.

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

## Dernière étape conditionnelle — diversifier les acquisitions sans fuite

**Ajout demandé par Icham le 13 septembre ; planifié, non lancé.** À activer seulement si les autres pistes diagnostiques et les corrections justifiées n'apportent pas une amélioration suffisante sur développement. Juger cette amélioration avec les critères de qualité annoncés avant comparaison, pas avec la seule baisse de loss ; aucun seuil de gain arbitraire ajouté ici.

- Recueillir des enregistrements de **tuyaux réellement sans fuite** dans davantage de conditions distinctes : tuyaux/matériaux, capteurs, installations et états de fonctionnement. Ne pas les remplacer par davantage de bruits environnementaux.
- Privilégier des acquisitions **fuite/sans fuite comparables**, avec conditions et sessions documentées. Chercher davantage de situations indépendantes, pas simplement davantage de fenêtres découpées dans les mêmes enregistrements.
- Avant intégration : convenir du responsable, des moyens de collecte et du protocole ; vérifier annotations, provenance et droits, puis définir un nouveau protocole versionné avec séparation par acquisition et une réserve de confirmation intacte. Ne pas modifier rétroactivement les folds actuels ni incorporer le test officiel ou la réserve Aghashahi dans l'apprentissage.

Motif vérifié : les598 clips train comptent294 fuites,242 vrais sans-fuite et62 bruits ; les242 sans-fuite ne représentent que7 groupes heuristiques, dont deux totalisent197 clips. C'est une limite de diversité des données, **pas la cause démontrée de tout l'écart TSLM–C1**. Comptages et limites : [[Diagnostic causal - exécution#Audit des populations et des cibles]]. Ce complément ne déclenche ni collecte, ni recherche de dataset, ni réentraînement maintenant.
