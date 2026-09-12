# SPLIT_V2_AUDIT — cause racine de l'échec de `split_v1`, et le split qui le remplace

> 2026-09-12. **`split_v1` est INVALIDE et remplacé.** Il est conservé tel quel comme artefact
> historique, jamais modifié en place, pour que l'erreur reste inspectable.
>
> **Aucun modèle n'a été entraîné. Aucun résultat de performance n'existe.** Aucun TimeNet,
> aucune baseline, aucune annotation générée, aucun appel GPU. Le split n'a donc été choisi sur
> aucune performance — il ne pouvait pas l'être.

---

## 1. OBSERVATION

Un contrôle adversarial indépendant a signalé que `split_v1` laissait passer des conditions
physiques entre les folds. Recalculé depuis les manifestes publiés :

| Mesure | `split_v1` |
|---|---|
| Conditions leak (matériau, pression, débit) vues par les **deux** devices | **68** |
| dont **à cheval sur plusieurs folds** | **30** |
| Clips concernés | **92**, sur 60 groupes |
| Conditions non-leak (matériau, région) à cheval | **5** |
| Paires quasi-doublons (seuil 0.7) à cheval | **10** |

Or `split_v1` publiait `condition_overlap = 0` et le mot « leakage-clean ». Les deux contrôles
portaient sur les mêmes données et se contredisaient.

---

## 2. INTERPRÉTATION

Le contrôle de `split_v1` ne mesurait pas ce que son nom annonçait. Il ne testait pas
« aucune condition physique ne traverse les folds » mais « aucune condition **entièrement
documentée** ne traverse les folds ». Sur ce dataset, ces deux phrases n'ont pas le même sens :

```python
# scripts/ingest/build_groups.py:534 — le prédicat fautif
if c.cls == "leak" and NA not in (c.material, c.region, c.pressure, c.flow):
```

| Prédicat | Clips leak couverts |
|---|---|
| Les **4** champs non-NA (celui qui a été utilisé) | **91 / 500 — 18 %** |
| Pression **et** débit non-NA (l'observable réel) | **426 / 500 — 85 %** |

La région est absente pour **348 des 500** clips leak. Exiger sa présence n'écartait pas des cas
douteux : cela écartait la majorité des conditions réelles. `condition_overlap = 0` était
arithmétiquement vrai et informativement vide.

---

## 3. CAUSE RACINE DÉMONTRÉE

### RC1 — Une métadonnée manquante a été traitée comme un filtre, pas comme une incertitude

Le prédicat `NA not in (...)` a été écrit pour une **statistique descriptive** (« combien de
conditions sont entièrement documentées ? »), puis réutilisé tel quel comme **invariant de
sécurité** à deux autres endroits. Ce sont deux usages opposés :

- pour décrire, exclure l'incomplet est légitime ;
- pour garantir, l'incomplet doit être traité comme **possiblement identique**, donc contraint.

Le même prédicat apparaissait **quatre fois** : `build_groups.py:431` (fusion bi-device),
`:534` (contrôle), `:612` (statistique descriptive), et recopié dans
`verify_manifest.py:118`.

> ### C'est une seule cause racine, et elle explique les trois symptômes à la fois :
> la fusion incomplète, le contrôle qui ne voyait rien, et le vérificateur qui confirmait.

### RC2 — Le tri par taille détruisait l'effet du seed

```python
random.Random(seed).shuffle(groups)     # mélange
groups.sort(key=lambda g: -sizes[g])    # ... immédiatement écrasé
```

Le plus gros groupe de chaque classe était donc placé **en premier**, alors que tous les
compteurs de sa classe valaient zéro. `argmax(cible − 0)` vaut toujours `train`, le fold de plus
grande cible. Mesuré sur 500 seeds :

| | `train` | `val` | `test` |
|---|---|---|---|
| Taille du plus gros groupe jamais reçu — **avec** le tri | 74 | 74 | **43** |
| Taille du plus gros groupe jamais reçu — **sans** le tri | 74 | 74 | **74** |

`test` ne pouvait structurellement pas recevoir de gros groupe. La phrase « un seul tirage, on
prend ce que le seed donne » de `SPLIT_AUDIT.md` était fausse.

### RC3 — Le vérificateur héritait du prédicat qu'il devait contrôler

`verify_manifest.py` recopiait `NA not in (material, region, pressure, flow)` et lisait les
colonnes `group_id` et `fold` du manifeste. Il vérifiait la **cohérence interne** d'un artefact
avec lui-même. Aucune erreur de construction n'était détectable par principe. Décrire cela comme
« deux implémentations indépendantes » était faux, et c'est ma formulation, pas celle d'un tiers.

---

## 4. HYPOTHÈSES RESTANTES — non démontrées, à ne pas affirmer

1. **Les composantes connexes sont des sessions d'acquisition.** Non prouvé. Le dataset ne
   publie ni identifiant de site, ni de conduite, ni de campagne, ni d'horodatage. Ce sont des
   **grappes de dépendance heuristiques**. `EVAL_PROTOCOL.md` §7ter appelait un groupe « une
   session d'enregistrement » : c'est corrigé, le terme exact est *unité de dépendance*.
2. **Des clips leak et no-leak de la même conduite peuvent rester séparés.** Rien dans les noms
   ne permet de les relier. Une empreinte d'installation partagée entre train et test reste
   possible et **non exclue**.
3. **Les 74 clips leak sans pression ni débit** ne peuvent être rattachés à aucune condition.
   L'invariant I4 ne les couvre pas, et aucune donnée ne permet de le faire.
4. **`dog_1` et `dog_2` sont des enregistrements distincts.** Supposé, non vérifié. Ils ne sont
   pas fusionnés. Les fusionner ferait tomber la classe non-leak à 25 groupes.
5. **La normalisation d'amplitude ne supprime pas le confound de niveau**, elle en supprime la
   forme la plus directe. Rapport signal/bruit, facteur de crête, plancher de quantification et
   réponse du capteur restent corrélés à la chaîne d'acquisition.

---

## 5. RÉPARATION

`scripts/ingest/build_split_v2.py`, trois changements, un par cause racine.

### 5.1 Un graphe de dépendances explicite (RC1)

Les groupes sont les **composantes connexes** d'un graphe dont chaque type d'arête est nommé,
compté, et dont la **couverture** est publiée à côté du résultat.

| Arête | Ce qu'elle lie | Clés | Clips couverts | Fusions |
|---|---|---|---|---|
| `E1` doublon exact | md5 identiques | 964 | **1000 / 1000** | 36 |
| `E2` quasi-doublon | corrélation spectrale > 0.7 | 54 paires | **1000 / 1000** | 12 |
| `E3` session | mêmes métadonnées + device (fenêtres, répétitions) | 327 | **1000 / 1000** | 640 |
| `E4` condition leak inter-device | **(pression, débit)** — région ignorée | 131 | **426 / 1000** | 119 |
| `E5` condition non-leak inter-device | (matériau, région) | 10 | **386 / 1000** | 8 |

> La région ne conditionne plus rien. C'est le correctif direct de RC1. Les couvertures de
> `E4` et `E5` sont inférieures à 100 % parce que les champs concernés n'existent que pour une
> partie des clips — c'est désormais **affiché**, pas masqué.

**Seuil de quasi-doublon : 0.7**, contre 0.8 en v1. Sensibilité publiée dans
`split_v2.meta.json` : 36 paires à 0.9 et 0.8, **54 à 0.7**, 446 à 0.6, 2069 à 0.5. À 0.8 le
contrôle ne faisait que redoubler le test md5 — il n'attrapait aucune paire non identique. 0.7
est le premier seuil qui capture des paires réellement distinctes sans effondrement.

### 5.2 Un vrai tirage aléatoire (RC2)

Échantillonnage par rejet. Chaque groupe reçoit un tirage i.i.d. `p = (0.60, 0.20, 0.20)` ;
l'affectation **entière** est acceptée ou rejetée selon deux contraintes déclarées d'avance :

| Contrainte | Valeur |
|---|---|
| Écart max à la cible, par classe et par fold | **±2 %** |
| Groupes minimum par classe et par fold | **8** |
| Tirage accepté | **n° 7535** (seed 20260912) |

Aucun tri, aucun placement glouton, aucun ordre privilégié. La position d'un groupe s'explique
par ces deux nombres, ou par le hasard — jamais par un effet de bord de l'ordre de parcours.

### 5.3 Un vérificateur qui reconstruit au lieu de faire confiance (RC3)

`scripts/eval/verify_split_invariants.py` ne lit que **`clip_id` et `fold`**. Il ignore
délibérément `group_id`. Tout le reste est recalculé depuis les WAV et leurs noms, puis
confronté aux folds. Il affiche la **couverture** de chaque invariant à côté du résultat.

---

## 6. TEST QUI FALSIFIERAIT LA RÉPARATION

Trois tests. Les trois ont été exécutés.

### T1 — Régression : le vérificateur doit échouer sur `split_v1`

Un contrôle qui passe partout ne contrôle rien.

```bash
python3 scripts/eval/verify_split_invariants.py --data-root <...> --manifest manifests/split_v1.csv
```

| Invariant | `split_v1` | `split_v2` |
|---|---|---|
| I1 doublons exacts à cheval | 0 | **0** |
| I2 quasi-doublons (0.7) à cheval | **10** ❌ | **0** |
| I3 même session à cheval | 0 | **0** |
| I4 condition leak (pression, débit) à cheval | **47** ❌ | **0** |
| I5 condition non-leak (matériau, région) à cheval | **5** ❌ | **0** |
| Code de sortie | **1** | **0** |

✅ Le vérificateur détecte l'erreur qu'il a manquée la première fois.

### T2 — Le biais d'affectation doit avoir disparu

Falsification : si le plus gros groupe reste cloué au même fold quelles que soient les
contraintes, le biais n'est pas levé. Mesuré sur 60 seeds, en relâchant les contraintes :

| Contraintes | Où va le plus gros groupe non-leak (104 clips) |
|---|---|
| ±2 %, 8 groupes min *(retenu)* | train 60/60 |
| ±5 %, 5 groupes min | train 56, **test 3**, **val 1** |
| ±8 %, 3 groupes min | train 46, **val 7**, **test 7** |
| ±12 %, 3 groupes min | train 38, **val 11**, **test 11** |

> ⚠️ **Lecture honnête.** Sous les contraintes retenues, ce groupe va encore toujours dans
> `train`. Mais la cause a changé de nature, et c'est vérifiable : **il pèse 104 clips, soit
> 20,8 % de la classe non-leak, alors que la cible d'un fold à 20 % vaut 100 clips.** Il ne
> *peut* pas entrer dans `val` ou `test` sans faire exploser l'équilibre. C'est une contrainte
> arithmétique des données, déclarée et mesurable — pas un effet de bord de l'ordre de tri.
> Relâcher les contraintes le déplace ; en v1, **aucun** relâchement ne le déplaçait.
>
> Les autres gros groupes, eux, circulent : `val` reçoit un groupe de 80, `test` un de 61.
> En v1, `test` ne dépassait jamais 43.

### T3 — Reproductibilité bit à bit

Deux exécutions successives produisent des fichiers identiques au bit près. ✅
Un seed différent produit un split différent. Sans numpy/scipy, le script **refuse d'écrire**
au lieu de produire un groupage silencieusement différent.

---

## 7. `split_v2` — les chiffres

### 7.1 Manifeste

| Fichier | Rôle |
|---|---|
| [`manifests/split_v2.csv`](../manifests/split_v2.csv) | contrat : `clip_id`, `label`, `label_3c`, `group_id`, `fold` |
| [`manifests/split_v2_audit.csv`](../manifests/split_v2_audit.csv) | audit uniquement : chemin, device, matériau, région, pression, débit, fenêtre, répétition, md5 |
| [`manifests/split_v2.meta.json`](../manifests/split_v2.meta.json) | seed, procédure, contraintes, statistiques d'arêtes, sensibilité du seuil, hachages |

```
sha256  split_v2.csv        7a8716a35284434292314c10da58663e9f848be60edf18db0f98ef9d63d17896
        split_v2_audit.csv  1a3bd3c18ad6d886d42ecc85ba3a5cceaa45a9084102fc112b53e66782e29d61
```

> **`device` a quitté le contrat de split.** En v1 il y figurait, alors qu'il est informatif sur
> l'étiquette en trois classes : la classe *noise* ne contient **aucun** clip hydrophone. Il
> reste dans le fichier d'audit.

### 7.2 Folds

| Fold | Clips | *leak* | *non-leak* | (dont bruit) | Groupes | gr. *leak* | gr. *non-leak* |
|---|---|---|---|---|---|---|---|
| train | 598 | 294 | 304 | 62 | 102 | 78 | 24 |
| val | 208 | 108 | 100 | 20 | 42 | 34 | 8 |
| test | 194 | 98 | 96 | 32 | 41 | 30 | 11 |
| **total** | **1000** | **500** | **500** | 114 | **185** | 152 | 43 |

### 7.3 Tailles de groupe

| Fold | Classe | Groupes | min | médiane | max | 5 plus gros |
|---|---|---|---|---|---|---|
| train | *leak* | 78 | 1 | 3 | 48 | 48, 10, 9, 8, 7 |
| train | *non-leak* | 24 | 2 | 4 | **104** | 104, 93, 13, 10, 10 |
| val | *leak* | 34 | 1 | 3 | 11 | 11, 6, 6, 5, 5 |
| val | *non-leak* | 8 | 2 | 3 | **80** | 80, 4, 3, 3, 3 |
| test | *leak* | 30 | 1 | 3 | 8 | 8, 7, 6, 5, 5 |
| test | *non-leak* | 11 | 2 | 3 | **61** | 61, 5, 5, 4, 4 |

---

## 8. Ce que `split_v2` coûte, et qu'il faut dire

**La correction a rendu le dataset plus petit en unités indépendantes, pas plus grand.** C'est
le prix de l'honnêteté et il doit figurer dans tout rapport.

| | `split_v1` (invalide) | `split_v2` |
|---|---|---|
| Groupes totaux | 306 | **185** |
| Groupes non-leak | 51 | **43** |
| **Groupes non-leak en val** | 9 | **8** |
| **Groupes non-leak en test** | 11 | **11** |
| Part du plus gros groupe dans le non-leak de val | 74 % | **80 %** |
| Part du plus gros groupe dans le non-leak de test | 43 % | **64 %** |

> ### Le N honnête du test côté non-leak est **11 unités**, et l'une d'elles porte 64 % du fold.
> ### Côté validation, **8 unités**, dont une porte 80 %.

Les règles group-aware de `EVAL_PROTOCOL.md` §7ter s'appliquent intégralement et deviennent plus
nécessaires qu'avant : score clip-level **et** group-level, nombre de groupes publié avec chaque
chiffre, intervalles bootstrapés sur les groupes, seuil d'abstention calibré par groupe.

Autres limites qui n'ont pas bougé :

- La classe *noise* n'a aucun clip hydrophone : le hold-out device ne peut toujours pas tester la
  robustesse au bruit.
- Le confound de niveau sonore (≈ 11 dB) est intact. `EVAL_PROTOCOL.md` §7bis-A s'applique.
- Les groupes sont des **unités de dépendance**, pas des sessions d'acquisition démontrées.

---

## 9. Statut

| Artefact | Statut |
|---|---|
| `manifests/split_v1*.csv` | 🔴 **INVALIDE — SUPERSEDED.** Conservé, jamais modifié, ne doit servir à aucun entraînement ni à aucune comparaison. |
| `manifests/split_v2*.csv` | ✅ **actif** — tous les invariants tenus, vérifiés par reconstruction indépendante |
| `docs/SPLIT_AUDIT.md` | 🔴 décrit `split_v1` : **ses conclusions sont caduques**, en-tête corrigé |
| `scripts/ingest/build_groups.py` | 🔴 produit `split_v1` : conservé pour la traçabilité, ne pas utiliser |
| `scripts/eval/verify_manifest.py` | 🔴 vérificateur non indépendant, remplacé par `verify_split_invariants.py` |

**Aucun entraînement n'a été lancé. `split_v2` n'a été comparé à aucune performance.**
Il n'est plus modifié à partir du premier résultat produit sur lui.
