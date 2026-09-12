# MODEL_EVAL_CONTRACT — ce que le TSLM doit livrer pour être évalué

> Pour Hicham. **Deux fichiers, rien d'autre.** Tout ce qui concerne l'évaluation —
> folds, étiquettes, clusters, seuil, métriques — est de notre côté et ne doit pas
> transiter avec le modèle.

---

## 1. Les deux fichiers

```
<un dossier quelconque>/
    metadata.json
    predictions.csv
```

C'est le **même format que nos contrôles C0 à C3**. Le TSLM n'est pas un cas
particulier : il entre par la même porte, et sort du même moteur.

### `predictions.csv`

```csv
clip_id,probability_leak
c003cd25f5a5f,0.8731
c00c343da6afa,0.1042
```

| Colonne | Contenu |
|---|---|
| `clip_id` | l'identifiant du manifeste, tel quel |
| `probability_leak` | probabilité de la classe *leak*, dans `[0, 1]` |

**Il faut une ligne par clip des folds `val` et `test`.** Les clips de `train`
sont acceptés et ignorés. Aucune autre colonne.

### `metadata.json`

```json
{
  "run_id": "tslm-v1",
  "model_name": "OpenTSLM fine-tuned on leakless/acoustic-leak",
  "checkpoint": "gs://.../step-2400",
  "training_commit": "abc123…",
  "config_hash": "sha256:…",
  "split_filename": "split_v2.csv",
  "split_sha256": "7a8716a35284434292314c10da58663e9f848be60edf18db0f98ef9d63d17896",
  "timestamp": "2026-09-12T18:30:00+00:00",
  "threshold_rule": "aucun seuil appliqué — probabilités brutes",
  "test_labels_not_used_for_tuning": true
}
```

`config_hash` est facultatif (`null` accepté). Tout le reste est obligatoire.

**Ce que désigne chaque commit.** Trois notions distinctes, à ne pas confondre :

| champ | sens | obligatoire |
|---|---|---|
| `training_commit` | le commit checké **au moment de l'entraînement** du checkpoint, capturé à ce moment-là et recopié tel quel dans chaque run qui utilise ce checkpoint. Pas le commit du jour où tu fais l'inférence. | oui |
| `execution_commit` | le commit checké au moment où tu produis `predictions.csv` | non, recommandé pour les runs de stress |
| `model_definition_commit` | le commit où l'architecture ou les descripteurs ont été figés | non |

Si ton worktree avait des modifications non commitées à l'entraînement, dis-le
(`"training_worktree_dirty": true`) : le commit seul ne reproduirait pas le
checkpoint. Le rapport lit ces champs ; il ne les recalcule jamais.

Schéma machine : [`schemas/prediction_run.schema.json`](../schemas/prediction_run.schema.json).

---

## 1bis. Obtenir `probability_leak` depuis un TSLM

C'est le point qui décide de la qualité de toute la comparaison, alors autant
être précis.

### La méthode

Un TSLM produit du texte. Il ne faut **pas** lui faire écrire un pourcentage :
un nombre généré par un décodeur n'est pas une probabilité, c'est un token qui
ressemble à un nombre. La probabilité se lit dans les **log-probabilités**, pas
dans la sortie.

À la position où le modèle décide de la classe, relever les log-probabilités des
deux continuations (`leak` et `no leak`, ou les tokens que ton prompt impose),
puis softmax sur ces deux valeurs uniquement :

```
p(leak) = exp(logp_leak) / (exp(logp_leak) + exp(logp_no_leak))
```

Si la classe tient sur plusieurs tokens, sommer les log-probabilités de la
séquence de chaque classe, puis softmax sur les deux sommes. Normaliser par la
longueur seulement si les deux libellés ont un nombre de tokens différent, et
le déclarer dans `threshold_rule`.

### Pourquoi une sortie dure 0/1 casserait la mesure

Mesuré chez nous, en remplaçant le score continu de C1 par sa décision seuillée :

| | clip AUC | PR-AUC | cluster AUC |
|---|---|---|---|
| score continu | 0,902 | 0,878 | 0,927 |
| sortie dure 0/1 | 0,856 | 0,842 | 0,856 |
| **perte** | **−0,046** | **−0,036** | **−0,071** |

> ### Cette perte est plus grande que l'écart qu'on cherche à mesurer.
> C2b − C1 vaut −0,055 en clip AUC. Un TSLM qui rendrait du 0/1 serait pénalisé
> d'un montant comparable à l'effet étudié, et la comparaison ne voudrait plus
> rien dire. **Une probabilité continue n'est pas un raffinement, c'est la
> condition pour que le run soit exploitable.**

`check_run.py --inspect` te le dit avant l'envoi.

### Calibration

**Elle n'est pas nécessaire.** ROC AUC, PR AUC et l'agrégation par cluster sont
invariants à toute transformation monotone : une probabilité non calibrée donne
exactement les mêmes valeurs. Seul le **score de Brier** en souffre, et nous le
publions en le disant.

Déclare simplement `"threshold_rule": "aucun seuil appliqué — probabilités
brutes non calibrées"`. C'est suffisant, et c'est honnête.

### Vérifier avant d'envoyer

```bash
# fabriquer le squelette : les 402 clip_id sont déjà remplis, dans le bon format
python3 scripts/eval/check_run.py --template mon_run/

# vérifier la conformité, sans qu'aucune métrique ne soit calculée
python3 scripts/eval/check_run.py --run mon_run/ --inspect
```

Le second dit oui ou non, et s'il dit non il dit pourquoi et sur combien de clips
il a regardé. Il ne calcule **aucune** métrique : la conformité et l'évaluation
sont deux choses séparées, et tu n'as pas besoin de voir la seconde pour livrer.

---

## 1ter. Une mise en garde sur la sélection de configurations

La validation compte **208 clips, mais seulement 42 clusters de dépendance, dont
8 du côté non-leak**. Un cluster, pas un clip, est l'unité indépendante.

> Comparer beaucoup de configurations sur 8 unités non-leak surajuste la
> validation, et le test le paiera.

Deux demandes concrètes, ni l'une ni l'autre coûteuse :

1. **Limite le nombre de configurations comparées** et garde-en la trace.
2. **Note combien tu en as essayé** dans `metadata.json`, champ libre, par
   exemple `"n_configs_compared": 6`. Ça ne change rien à l'évaluation, ça change
   ce qu'on a le droit d'affirmer sur l'écart validation → test.

Le train a la même structure : 598 clips mais 102 clusters, dont 24 non-leak.

---

## 1quater. Les transformations de stress, à l'identique

Pour que tes T1/T2/T3 soient **exactement** les nôtres, applique nos fonctions au
signal brut, avec le générateur déterministe par clip :

```python
import sys; sys.path.insert(0, "scripts/temporal")
from stress import TRANSFORMS, clip_rng

# raw = le signal brut du clip, 8000 échantillons, AVANT ton preprocessing
stressed = TRANSFORMS["T2"]["fn"](raw, clip_rng("T2", clip_id))
```

`clip_rng(nom, clip_id)` dérive une graine du couple par **SHA-256** : même clip,
même transformation, même résultat, sur n'importe quelle machine et dans
n'importe quel processus.

> ⚠️ **Prends la version à partir du commit `b23601a`.** Avant lui, `clip_rng`
> utilisait `hash()` de Python, salé par processus : deux exécutions donnaient
> deux permutations différentes. Si tu as déjà récupéré le fichier, reprends-le. Si tu tires ta
propre permutation, tes T2 ne seront pas nos T2 et les invariants publiés ne
s'appliqueront plus.

T2 utilise des blocs de **250 échantillons (31,25 ms)** — 8000 est divisible par
250, donc aucun échantillon ne reste à sa place.

Livre un dossier de run par transformation, avec un `run_id` distinct
(`tslm-v1-T1`, etc.), et passe `--tslm-run-id tslm-v1` au rapport final.

Chaque run de stress ajoute à `metadata.json` :

```json
{
  "stress_transform": "T2",
  "base_run_id": "tslm-v1",
  "retrained": false,
  "checkpoint": "<le même que tslm-v1>",
  "training_commit": "<le même que tslm-v1>",
  "execution_commit": "<HEAD au moment de l'inférence sous stress>",
  "threshold_rule": "aucun seuil appliqué — probabilités brutes"
}
```

`retrained: false` signifie exactement **« not retrained on stressed data »** :
le checkpoint n'a jamais vu T1, T2 ni T3. Le rapport refuse un run de stress
dont le checkpoint ou le `training_commit` diffère de celui du run T0.

> **Le seuil sous stress.** Notre moteur recalcule le seuil sur la validation
> *du run*, donc sur la validation transformée pour un run de stress. Les
> métriques qui dépendent du seuil (macro-F1, exactitude) ne servent donc à
> **aucune** conclusion de sensibilité temporelle. Le rapport ne lit la
> sensibilité que dans l'AUC, la corrélation des probabilités avec T0 et la
> distribution de Δp.

---

## 2. Ce qui est refusé, et pourquoi

Le validateur s'arrête au premier manquement. Chaque refus correspond à une
sentinelle testée dans `tests/run_tests.py`.

| Refusé | Raison |
|---|---|
| une colonne `pressure`, `flow`, `device`, `path`, `filename`, `material`, `region`, `md5` | pression et débit ne sont renseignés que pour la classe *leak* : les laisser circuler donne 93,8 % de rappel sans écouter un son |
| une colonne `fold`, `label`, `group_id` | les folds, les étiquettes et les clusters viennent de `split_v2.csv`. **Un modèle ne redéfinit pas les données sur lesquelles il est jugé.** |
| `split_filename` différent de `split_v2.csv` | `split_v1` est **invalide** (30 conditions physiques traversaient les folds, voir `SPLIT_V2_AUDIT.md`) |
| `split_sha256` différent du gelé | le run a été produit sur un autre découpage ; comparer serait une erreur, pas une approximation |
| un `clip_id` inconnu, en double, ou manquant | la comparaison appariée exige une population de clips identique entre modèles |
| une probabilité hors `[0, 1]` ou non numérique | le score de Brier et l'agrégation exigent une vraie probabilité |
| `test_labels_not_used_for_tuning` absent ou `false` | c'est une déclaration explicite, pas une supposition de notre part |

Chaque contrôle publie sa **couverture** (« 402/402 clips prédits »), pas seulement
un verdict. Un « 0 problème » sans dénominateur ne prouve rien — c'est exactement
l'erreur qui a invalidé `split_v1`.

---

## 3. Ce que nous faisons, et que tu n'as pas à faire

| | |
|---|---|
| Seuil | recalculé de notre côté : argmax du macro-F1 **cluster-level** sur la **validation**. Un seuil déclaré dans `metadata.json` est reporté pour information, jamais utilisé. |
| Agrégation en cluster | **médiane** des probabilités des clips du cluster. Règle gelée avant toute prédiction. |
| Métriques | ROC AUC, PR AUC, macro-F1, exactitude équilibrée, rappel *leak*, taux de faux positifs, Brier — au niveau clip **et** au niveau cluster |
| Incertitude | bootstrap sur les **clusters**, 2000 tirages, graine 20260912 |
| Comparaisons | bootstrap apparié : un seul tirage de clusters sert au TSLM et au contrôle comparé |

> Tu n'as donc **pas** besoin de choisir un seuil, ni de calculer une métrique, ni
> de savoir ce qu'est un cluster de dépendance. Donne des probabilités brutes.

---

## 4. La commande d'intégration

Dès que le dossier existe :

```bash
python3 scripts/eval/build_final_report.py \
    --runs <runs>/c0 <runs>/c1 <runs>/c2 <runs>/c3 <ton-dossier> \
    --out artifacts/final_evaluation
```

Elle produit `metrics.json`, `comparison.json` et `FINAL_EVALUATION.md`, avec les
comparaisons appariées TSLM − C0, TSLM − C1, TSLM − C2, TSLM − C3. Aucun chiffre
n'est recopié à la main.

> Si tu livres **plusieurs** runs (par exemple T0, T1, T2, T3), ajoute
> `--tslm-run-id <le run de référence>`. Sans lui la commande s'arrête et le dit,
> plutôt que de deviner lequel est le run principal.

---

## 5. Le barreau à dépasser n'est pas celui qu'on croyait

Mesuré sur le test gelé, 41 clusters (30 *leak* / 11 *non-leak*) :

| Contrôle | Ce qu'il isole | clip AUC | cluster AUC |
|---|---|---|---|
| C0 | niveau sonore absolu, signal brut | 0,878 | 0,839 |
| **C1** | **forme d'amplitude seule, audio normalisé** | **0,902** | **0,927** |
| C2 | spectre agrégé, sans phase ni ordre | 0,710 | 0,779 |
| **C2b** | **structure temporelle peu profonde** (enveloppe, modulation, flux) | 0,847 | 0,915 |
| C3 | baseline historique (mélange C1 + C2 + ZCR) | 0,824 | 0,900 |

> ### C1 fait mieux que C0.
> La normalisation d'amplitude retire le niveau absolu mais **pas** le raccourci
> d'acquisition : la forme de l'enveloppe en porte davantage. Le contrôle que le
> TSLM doit dépasser est donc **C1 (0,902 / 0,927)**, pas C0.
>
> Et C2b, un contrôle temporel peu profond à six descripteurs, n'est pas démontré
> supérieur à C1 (Δ clip AUC −0,055, IC95 traversant zéro : *inconclusive*).
> Battre C2b ne suffit donc pas.

Et la formulation reste qualitative : avec 41 clusters, aucun écart n'est déclaré
significatif. Les verdicts autorisés sont *compatible with improvement*,
*inconclusive*, *compatible with degradation*.

---

## 6. Test de sensibilité temporelle (optionnel, mais c'est le cœur du sujet)

Quatre jeux TimeF sont prêts, mêmes `clip_id`, mêmes folds, mêmes clusters :

| | Transformation | Ce qui est préservé | Ce qui est détruit |
|---|---|---|---|
| T0 | aucune | tout | rien |
| T1 | inversion temporelle | `\|FFT\|` à 3,5e-16, histogramme d'amplitude | la direction du temps |
| T2 | permutation de blocs de 250 échantillons (31,25 ms) | histogramme d'amplitude | l'ordre au-delà de 31,25 ms (déviation `\|FFT\|` médiane 0,70) |
| T3 | randomisation de phase | `\|FFT\|` à 3,4e-16 | la structure de phase |

Si tu peux faire tourner **le même checkpoint sérialisé**, sans réentraînement,
sur T1, T2 et T3 et nous rendre trois `predictions.csv` de plus, on mesure
directement : **les prédictions du modèle sont-elles sensibles à l'organisation
temporelle, ou restent-elles stables quand seul l'ordre est perturbé ?**

> ⚠️ Ce ne sont **pas** des augmentations physiques démontrées comme préservant
> l'étiquette. Rien ne garantit qu'un clip inversé reste acoustiquement une fuite.
> Ce sont des tests de stress : un score qui bouge montre une sensibilité des
> prédictions à la perturbation, pas la pertinence physique causale de
> l'information détruite. Un score qui ne bouge pas montre une insensibilité à
> *ces* perturbations, pas l'absence de toute information temporelle.
