# SPLIT_AUDIT — vérification du split groupé figé `split_v1` ⛔ CADUC

> # 🔴 CE DOCUMENT DÉCRIT UN SPLIT INVALIDE.
>
> `split_v1` laissait **30 conditions physiques traverser les folds (92 clips)**, ainsi que
> 10 paires de quasi-doublons et 5 conditions non-leak. Son contrôle `condition_overlap = 0`
> ne couvrait que **18 %** des clips leak parce qu'il exigeait un champ absent 348 fois sur 500.
>
> **Remplacé par [`SPLIT_V2_AUDIT.md`](SPLIT_V2_AUDIT.md) et `manifests/split_v2.csv`.**
> Tout ce qui suit est conservé pour que l'erreur reste inspectable. **Aucune de ses conclusions
> ne doit être citée**, en particulier « leakage-clean », « condition_overlap = 0 » et
> « deux implémentations indépendantes ».


> 2026-09-12. Le split est **figé et versionné**. Aucun WAV, aucune donnée audio n'entre dans le
> dépôt. **Aucun entraînement, aucune annotation générée, aucune conversion TimeNet, aucun appel
> GPU n'a eu lieu.**
>
> Ces folds sont **les mêmes pour tous les modèles** — baseline, contrôle RMS, TSLM. Les changer
> après avoir vu un score invaliderait toute comparaison.
>
> **Le split n'est pas adapté à la dépendance entre clips — le protocole l'est.** Les règles
> d'évaluation group-aware qui en découlent sont dans `EVAL_PROTOCOL.md` §7ter et résumées au §6.1
> ci-dessous.

---

## 1. Le manifeste

| Fichier | Rôle | Contenu |
|---|---|---|
| [`manifests/split_v1.csv`](../manifests/split_v1.csv) | **Le contrat de split** | `clip_id`, `label`, `label_3c`, `group_id`, `device`, `fold` |
| [`manifests/split_v1_audit.csv`](../manifests/split_v1_audit.csv) | **Audit uniquement** | chemin, matériau, région, pression, débit, catégorie de bruit, fenêtre, répétition, md5, group_id, fold |
| [`manifests/split_v1.meta.json`](../manifests/split_v1.meta.json) | Provenance | seed, procédure, cibles, hachages |

### Pourquoi deux fichiers

> ### 🔴 `split_v1.csv` ne contient **aucune** métadonnée révélant l'étiquette.
> Pression et débit ne sont renseignés que pour la classe *leak* (risque **L1** :
> 100 % de précision, 93,8 % de rappel sans écouter un son). Ils sont donc **absents du fichier
> que le code d'entraînement lit**. La séparation est structurelle, pas une consigne : le code ne
> peut pas fuiter ce qu'il n'a pas chargé.
>
> `split_v1_audit.csv` porte ces colonnes et le chemin du fichier. Il sert à rejouer l'audit et à
> résoudre `clip_id` → forme d'onde. **Aucune de ses colonnes n'entre dans une entrée de modèle.**

### Reproductibilité

```bash
python3 scripts/ingest/build_groups.py \
    --data-root <dossier hors dépôt> \
    --write-manifest manifests
```

Vérifié : deux exécutions successives produisent des fichiers **identiques au bit près**. Un seed
différent produit un split différent (contrôle négatif exécuté).

**Contre-vérification indépendante** — tous les chiffres de ce document sont recalculés depuis
l'audio et les CSV par un second script qui n'importe rien de `build_groups.py` :

```bash
python3 scripts/eval/verify_manifest.py --data-root <dossier hors dépôt>
```

33 contrôles, code de sortie 1 à la première divergence. Deux implémentations indépendantes qui
tombent d'accord valent mieux qu'une qui se relit.

| | |
|---|---|
| **Seed** | **20260912** — unique, fixé d'avance, inscrit dans `split_v1.meta.json` |
| `clip_id` | `"c" + sha1("<dossier>/<fichier>")[:12]` — stable, indépendant de l'ordre de lecture |
| `group_id` | `"g" + sha1(liste triée des membres du groupe)[:10]` — stable, indépendant de l'ordre de parcours |

**Procédure d'affectation** — fixée avant tout score, sans recherche :

1. les 306 groupes sont mélangés avec le seed ;
2. tri **stable** par taille décroissante (les gros groupes, les plus contraignants, en premier) ;
3. chaque groupe entier va au fold dont le déficit relatif est le plus grand **pour sa classe**.

> **Aucune recherche sur plusieurs seeds.** Choisir le « meilleur » split parmi N tirages revient
> à choisir ses données d'évaluation. Un seul tirage, on prend ce qu'il donne.

### Hachages — `sha256`

```
split_v1.csv        89f0624a4543fb1832d70938d5893e50ee131eecaa884e2f2d43b9fe05ca2a36
split_v1_audit.csv  b5b2083abc2bee1b9a75b66c28734e2c7951d8f3d2f0ba67066612daa7191ba8
```

---

## 2. Clips par fold et par classe

| Fold | Clips | % | *leak* | *non-leak* | (dont *no leak*) | (dont *noise*) |
|---|---|---|---|---|---|---|
| **train** | 600 | 60,0 % | 300 | 300 | 229 | 71 |
| **val** | 200 | 20,0 % | 100 | 100 | 79 | 21 |
| **test** | 200 | 20,0 % | 100 | 100 | 78 | 22 |
| **total** | **1000** | 100 % | **500** | **500** | 386 | 114 |

Les cibles 60/20/20 sont atteintes **exactement**, et l'équilibre 500/500 est préservé dans chaque
fold. Ce n'est pas un réglage : c'est la conséquence de la finesse de la classe *leak* (255
groupes pour 500 clips).

---

## 3. Groupes par fold et par classe

| Fold | Groupes | *leak* | *non-leak* |
|---|---|---|---|
| train | 184 | 153 | 31 |
| val | 60 | 51 | 9 |
| test | 62 | 51 | 11 |
| **total** | **306** | **255** | **51** |

---

## 4. Tailles de groupe

| Fold | Classe | Groupes | min | médiane | max | 5 plus gros |
|---|---|---|---|---|---|---|
| train | *leak* | 153 | 1 | 2 | 11 | 11, 8, 7, 7, 6 |
| train | *non-leak* | 31 | 1 | 4 | **74** | 74, 51, 30, 29, 18 |
| val | *leak* | 51 | 1 | 2 | 9 | 9, 4, 4, 4, 3 |
| val | *non-leak* | 9 | 2 | 3 | **74** | **74**, 5, 4, 4, 3 |
| test | *leak* | 51 | 1 | 2 | 8 | 8, 6, 4, 3, 3 |
| test | *non-leak* | 11 | 2 | 4 | **43** | 43, 19, 12, 5, 4 |

---

## 5. Vérifications automatiques

Exécutées par `build_groups.py` à chaque écriture du manifeste. **Chacune doit valoir 0.**

| Contrôle | Ce qu'il interdit | Résultat |
|---|---|---|
| `group_overlap` | un `group_id` présent dans plus d'un fold | **0** ✅ |
| `duplicate_overlap` | deux fichiers octet-identiques dans deux folds différents | **0** ✅ |
| `condition_overlap` | une condition physique complète (matériau+région+pression+débit) à cheval sur deux folds — inclut les 16 conditions bi-device | **0** ✅ |
| `clips_sans_fold` | un clip non affecté | **0** ✅ |
| Total réaffecté | | **1000 / 1000** ✅ |

Les 36 paires de doublons octet-identiques sont donc toutes intra-fold, et les 27 conditions
*leak* complètes — dont les 16 captées par les deux instruments — sont chacune entièrement dans un
seul fold.

---

## 6. ⚠️ Ce que ce split ne règle pas

Trois faiblesses structurelles, mesurées, qui ne sont **pas** des erreurs de construction mais des
propriétés du dataset. Elles doivent accompagner tout chiffre publié.

### 6.1 La validation *non-leak* repose à 74 % sur un seul groupe

> Des 100 clips *non-leak* de validation, **74 viennent d'un seul groupe** — une seule session
> d'enregistrement. Les 8 autres groupes se partagent 26 clips.

`EVAL_PROTOCOL.md` §5 calibre le **seuil d'abstention sur la validation**. Ce seuil reposera donc,
côté *non-leak*, essentiellement sur une session unique.

**Ce n'est pas un leakage** — le groupe est entier dans val, rien ne fuit vers test. C'est une
question de **taille d'échantillon effective** : elle vaut ~9 unités indépendantes, pas 100.

> ### ⛔ ~~Décision d'équipe, 2026-09-12 : `split_v1` est conservé tel quel.~~
> ### Cette décision est **annulée** : `split_v1` s'est révélé invalide et est remplacé par `split_v2`.
>
> Motifs retenus : seed fixée avant tout résultat, aucune recherche de seed, aucun leakage entre
> groupes. Modifier le split maintenant dans le seul but d'équilibrer la taille des groupes
> serait du **split engineering** — on optimiserait la forme de ses données d'évaluation après
> les avoir regardées.
>
> **Ce n'est donc pas le split qui est adapté à la dépendance entre clips, c'est le protocole**
> — voir `EVAL_PROTOCOL.md` §7ter : métriques group-aware, agrégation non pondérée par groupe,
> bootstrap sur les groupes, seuil calibré en pondérant par groupe (le groupe de 74 clips compte
> pour **1 unité sur 9**, pas 74 sur 100), et nombre de groupes publié avec chaque score.

### 6.2 Le test *non-leak* compte 11 unités indépendantes, pas 100

Le plus gros groupe porte 43 des 100 clips (43 %). Tout score *non-leak* doit être publié avec son
nombre de groupes et une dispersion inter-groupes — **jamais un chiffre unique**.

| Fold | Clips *non-leak* | **Groupes indépendants** |
|---|---|---|
| val | 100 | **9** |
| test | 100 | **11** |

C'est le N honnête. `EVAL_PROTOCOL.md` §7ter-E impose de le porter dans tout rapport et dans le
pitch.

### 6.3 Le device est réparti de façon très inégale

| Fold | noise logger | hydrophone | NA |
|---|---|---|---|
| train | 440 | 158 | 2 |
| val | 181 | **19** | 0 |
| test | 163 | **37** | 0 |

La procédure équilibre les classes, pas les instruments. Avec 19 clips hydrophone en validation,
**aucune conclusion sur la généralisation inter-instrument ne peut sortir de ce split.** C'est
précisément le rôle de l'évaluation secondaire en hold-out device (`DATASET_AUDIT.md` §10), qui
est rapportée séparément.

---

## 7. Ce que ce split autorise et n'autorise pas

**Autorise** : conversion TimeNet, génération d'annotations ancrées sur des propriétés de signal
mesurables, entraînement baseline et TSLM, contrôle RMS — tous sur **exactement ces folds**.

**N'autorise pas** :

- refaire un split, changer le seed, ou « réessayer » après avoir vu un score — **`split_v1` est
  figé et n'est plus modifié après le premier résultat** (`EVAL_PROTOCOL.md` §7ter-F) ;
- rapporter un score clip-level seul, ou group-level seul : les deux, ensemble, avec le nombre de
  groupes (§7ter-B) ;
- calculer un intervalle de confiance en rééchantillonnant des clips plutôt que des groupes
  (§7ter-C) ;
- comparer un résultat tenu à l'écart au 0,857 descriptif de `DATASET_AUDIT.md` §11 — **ce n'est
  pas un seuil** ; le seul point de comparaison légitime est le contrôle RMS évalué sur ces mêmes
  folds, calculé sur le signal **brut non normalisé** (`EVAL_PROTOCOL.md` §7bis-A) ;
- affirmer qu'une étiquette est vérifiée. Les distributions de signal présentent des différences
  mesurables associées aux étiquettes fournies ; **la provenance des étiquettes reste celle du
  dataset source.**
