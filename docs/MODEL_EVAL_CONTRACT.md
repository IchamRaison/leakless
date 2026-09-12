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

Schéma machine : [`schemas/prediction_run.schema.json`](../schemas/prediction_run.schema.json).

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
| C3 | baseline historique (mélange C1 + C2 + ZCR) | 0,824 | 0,900 |

> ### C1 fait mieux que C0.
> La normalisation d'amplitude retire le niveau absolu mais **pas** le raccourci
> d'acquisition : la forme de l'enveloppe en porte davantage. Le contrôle que le
> TSLM doit dépasser est donc **C1 (0,902 / 0,927)**, pas C0.

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

Si tu peux faire tourner le checkpoint sur T1, T2 et T3 et nous rendre trois
`predictions.csv` de plus, on obtient une réponse directe à la question du
hackathon : **le modèle utilise-t-il l'organisation temporelle, ou seulement des
statistiques invariantes à l'ordre ?**

> ⚠️ Ce ne sont **pas** des augmentations préservant l'étiquette. Rien ne garantit
> qu'un clip inversé reste acoustiquement une fuite. Ce sont des tests de stress :
> un modèle dont le score ne bouge pas sous T1/T2/T3 n'utilise pas l'ordre.
