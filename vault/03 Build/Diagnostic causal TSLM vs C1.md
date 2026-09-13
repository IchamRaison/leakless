# Diagnostic causal TSLM vs C1

Icham — 2026-09-13

## Statut et objectif proposé

Icham propose de remplacer la recherche immédiate d'une meilleure recette par un diagnostic de l'écart à C1. Cette note cadre ce changement ; **aucune nouvelle expérience ni modification du modèle n'est lancée dans ce tour**. Le plan d'implémentation V2 est en pause : pas de refit final automatique ni de nouvelle variante.

> Expliquer l'écart de performance observé entre les pipelines TSLM et C1 sur développement. Examiner les causes plausibles — données, représentation, supervision, optimisation, utilisation du signal par Qwen et scoring — puis les départager par des expériences contrôlées. Distinguer causes démontrées, hypothèses non soutenues dans les conditions testées et inconnues avant de proposer une nouvelle recette.

« Exploite moins bien les données » reste une explication possible, pas un fait établi par une AUC inférieure. C1 peut aussi bénéficier de particularités d'acquisition. Le cap produit demeure [[Plan V2 - fiabilité et parité des scores#7. Objectif final — démontrer la valeur du TSLM]] ; le diagnostic n'en démontre pas encore l'utilité.

## Acquis et limites — ne pas recommencer les preuves terminées

- **Démontré :** deux causes d'écarts numériques, conversion float32 avant FFT et dépendance de l'encodage à la taille du lot. Correctifs versionnés, gates A/C sur 209 entrées et rechargements neufs à écart maximal nul (`04d53b6`). Cela démontre la parité dans les contextes vérifiés, pas la qualité du classement ni la justesse de l'objectif appris.
- **Observé :** les trois folds A0/A1/C0 vérifiés restent derrière C1 sur les mêmes partitions de développement. Preuves `83bcbe0` et `fa1b9b4`, `docs/evidence/tslm-v2/campaign-c13fd47/`. Aucun bilan complet A/C ni progrès final revendiqué.
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

Conserver séparément qualité du classement, décisions au seuil fixé et cohérence du texte brut/affiché. Une phrase corrigée par DSP/gabarit ne constitue pas un gain du modèle. Une cause de score incorrect n'est pas nécessairement la cause de tout l'écart à C1.

## Discipline de développement

Les trois folds groupés à l'intérieur des 598 train restent la référence de comparaison. Au sein de chaque contraste, tenir fixes données, composants non visés, initialisation lorsqu'un entraînement est requis et protocole de score, sauf le facteur étudié. Prétraitements appris uniquement sur le train du fold. Aucun checkpoint V1 entraîné sur tous les 598 clips ne sert à revendiquer une performance hors-fold.

Consigner tous les essais, même abandonnés, sondes, checkpoints et variantes de scoring compris. La validation déjà utilisée ne redevient pas une confirmation indépendante. Les 194 anciens test restent historiques ; la réserve externe Aghashahi figée (`84007801…`) demeure sans score et ne sert pas à choisir les corrections.

## Reprise pratique de l'ancienne campagne

Dernier contrôle SSH en lecture seule le 13 septembre à **03:23:29 Europe/Paris** : A2, C/fold1 et le dernier C/fold2 annoncent `completed: true` dans leurs logs ; aucun processus `run_v2_campaign.py` encore actif observé. Leurs trois reçus ne sont pas encore intégralement revérifiés/rapatriés. A0/A1/C0 sont déjà vérifiés et publiés. Aucun processus n'a été arrêté dans ce tour ; aucun nouveau fit, classement, refit, seuil ou score externe lancé. Ne pas relancer les fits pour récupérer leurs résultats.

Runtime : `/home/hicham/pipe-v0/artifacts/v2-campaign-c13fd47-001`, logs `/home/hicham/pipe-v0/quality-v2-001/fit-{A,C}{0,1,2}-c13fd47.log`. Utiliser le contrôle existant `finished()` pour vérifier les reçus/empreintes, sans créer de fichier dans les dossiers scellés. Checkout local sain : `/home/animus/ehl-hackathon-zurich-v2-recovery`, branche `feat/icham-v2-reliability` ; ne pas commiter dans l'ancien dépôt Git endommagé.

**Prochaine action :** convenir du premier bloc diagnostique borné à partir de cet inventaire, avant toute nouvelle recette. Les résultats manquants de la campagne existante peuvent être récupérés sans réentraîner.
