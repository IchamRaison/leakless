# PIPELINE_RESULTS — préparation TimeNet, contrôle RMS, baseline

> 2026-09-12. Tout tourne sur les folds gelés de `manifests/split_v2.csv`
> (`sha256 7a8716a352844342…`, commit `3efa08f`). `split_v1` n'est utilisé nulle part.
>
> **Aucun TSLM entraîné** — Hicham en est propriétaire. **Aucune annotation OpenAI générée.**
> **Aucun GPU utilisé.** Le vault n'est pas touché.

---

## 1. Ce qui a été construit

| # | Livrable | Fichier |
|---|---|---|
| A | Connecteur TimeNet + carte de dataset | `scripts/timenet/leakless_acoustic/` |
| A | Pilote de conversion TimeF | `scripts/timenet/build_timef.py` |
| B | Contrôle RMS seul, signal **brut** | `scripts/eval/rms_control.py` |
| C | Baseline descripteurs + régression logistique | `scripts/eval/baseline_logreg.py` |
| — | Métriques group-aware partagées | `scripts/eval/group_metrics.py` |

Sorties (parquet, JSON) écrites **hors du dépôt**. Aucun WAV, aucun parquet dans Git.

---

## 2. Commandes exactes

```bash
# --- A. Conversion TimeNet (tourne dans le dépôt TimeNet, qui porte l'environnement) ---
cd ~/dev/sandbox/ehl-zurich/TimeNet
LEAKLESS_DATA_ROOT=~/dev/sandbox/ehl-zurich/data-audit/extract \
LEAKLESS_MANIFEST_DIR=~/dev/sandbox/ehl-zurich/ehl-hackathon-zurich/manifests \
uv run python ~/dev/sandbox/ehl-zurich/ehl-hackathon-zurich/scripts/timenet/build_timef.py \
    --out ~/dev/sandbox/ehl-zurich/timef-out

# --- B. Contrôle RMS seul ---
cd ~/dev/sandbox/ehl-zurich/ehl-hackathon-zurich
python3 scripts/eval/rms_control.py \
    --data-root ~/dev/sandbox/ehl-zurich/data-audit/extract \
    --out ~/dev/sandbox/ehl-zurich/eval-out/rms_control.json

# --- C. Baseline ---
python3 scripts/eval/baseline_logreg.py \
    --data-root ~/dev/sandbox/ehl-zurich/data-audit/extract \
    --out ~/dev/sandbox/ehl-zurich/eval-out/baseline_logreg.json
```

---

## 3. A — Préparation TimeNet

Le connecteur minimal valide : une sous-classe de `BaseConnector` avec `download()` et
`convert()`, plus une `dataset.yaml`. Le dépôt TimeNet n'est pas modifié.

| Champ TimeF | Contenu |
|---|---|
| `record_id` | `clip_id` |
| `subject_ids` | `(group_id,)` — la grappe de dépendance est le « sujet » TimeF, l'unité qu'un split ne doit jamais couper |
| série temporelle | forme d'onde **normalisée en amplitude**, 8000 échantillons à 8 kHz |
| `ClassificationTask.target` | `leak` / `no_leak` |

**Ce qui n'y entre pas** : pression, débit, device, matériau, région, chemin, nom de fichier.

Vérifié par relecture depuis le disque (`TimeFReader`) :

| | |
|---|---|
| Records relus | **1000** · tâches **1000** |
| Grappes distinctes en `subject_ids` | **185** — identique au manifeste |
| Cibles | leak **500** / no_leak **500** |
| Longueur de série | 8000 · RMS = 1,0 (normalisé) |
| Clips par fold | train 598 · val 208 · test 194 — identique au manifeste |
| Taille sur disque | 7,9 Mo, hors dépôt |

> Le jeu TimeF étant normalisé par construction, **il ne peut pas servir au contrôle RMS.**
> Celui-ci lit le signal brut, séparément. C'est voulu.

---

## 4. B et C — Résultats sur les folds gelés

Seuil et hyperparamètre choisis **sur la validation uniquement**, pondérés par groupe.
Le test n'a été lu qu'une fois.

### 4.1 Test — le chiffre qui compte

| | Contrôle RMS (brut) | Baseline (normalisée) |
|---|---|---|
| **clip-level** macro-F1 | **0,808** | 0,758 |
| **clip-level** AUC | **0,878** | 0,824 |
| clip-level rappel leak | 0,878 | 0,755 |
| clip-level fausses alarmes | 0,260 | 0,240 |
| **group-level** macro-F1 | 0,802 | **0,856** |
| **group-level** AUC | 0,839 | **0,900** |
| **Grappes derrière le chiffre** | **41** (30 leak / 11 non-leak) | **41** (30 / 11) |
| IC95 AUC clip (bootstrap sur grappes) | [0,679 – 0,932] | [0,751 – 0,962] |
| IC95 AUC groupe | [0,647 – 0,975] | [0,724 – 1,000] |

### 4.2 Validation

| | Contrôle RMS | Baseline |
|---|---|---|
| clip-level macro-F1 / AUC | 0,817 / 0,870 | 0,773 / 0,886 |
| group-level macro-F1 / AUC | 0,846 / 0,820 | 0,963 / 0,996 |
| Grappes | 42 (34 / 8) | 42 (34 / 8) |

### 4.3 Lecture honnête

> ### La baseline ne dépasse pas clairement un volumètre.

- **Au niveau clip, elle fait moins bien que le contrôle RMS** (AUC 0,824 contre 0,878).
- Au niveau groupe elle fait mieux (0,900 contre 0,839), mais **les intervalles de confiance
  se recouvrent largement** : [0,724 – 1,000] contre [0,647 – 0,975]. Avec 11 grappes
  *non-leak* en test, aucun de ces écarts n'est distinguable du bruit.
- L'écart validation → test de la baseline au niveau groupe (0,996 → 0,900) est cohérent
  avec une sélection d'hyperparamètre sur 8 grappes *non-leak* de validation. C'est de
  l'optimisme de sélection, pas une fuite : les folds sont disjoints et vérifiés.
- Le poids le plus fort de la baseline est le **facteur de crête** (−0,76), très au-dessus
  des autres. La normalisation RMS retire le niveau moyen, pas la forme de l'enveloppe :
  le modèle attrape encore un corrélat de la chaîne d'acquisition, exactement le risque
  annoncé dans `EVAL_PROTOCOL.md` §7bis-A.

**Ce que ces chiffres permettent de dire** : le pipeline tourne de bout en bout sur un split
vérifié, et une baseline simple sur audio normalisé n'établit pas d'avantage sur le volume brut.

**Ce qu'ils ne permettent pas de dire** : que la tâche est résolue, qu'un modèle détecte des
fuites, ou qu'un écart de 0,06 d'AUC entre deux approches est réel.

---

## 5. Conformité aux règles fixées

| Règle | État |
|---|---|
| Jamais `split_v1` | ✅ aucun script ne le lit |
| `split_v2` non modifié | ✅ `sha256` inchangé |
| Ni pression, ni débit, ni device, ni chemin comme features | ✅ aucun des trois scripts ne lit ces colonnes |
| `split_v2_audit.csv` seulement pour `clip_id` → chemin | ✅ une seule colonne lue : `path` |
| Audio normalisé pour modèle et baseline | ✅ normalisation RMS par clip |
| Contrôle RMS sur signal brut | ✅ aucune normalisation dans `rms_control.py` |
| Mêmes folds partout | ✅ un seul manifeste lu par les trois scripts |
| Clip-level **et** group-level | ✅ les deux, systématiquement |
| Nombre de grappes avec chaque résultat | ✅ dans chaque bloc |
| Pas de réglage sur le test | ✅ seuil et `C` choisis sur la validation |
| Pas d'annotation OpenAI | ✅ aucune |
| Pas d'entraînement TSLM | ✅ aucun — Hicham en est propriétaire |

---

## 6. Limites qui restent

1. **11 grappes *non-leak* en test, 8 en validation.** Tous les intervalles sont larges et
   se recouvrent. Aucune comparaison entre approches n'est concluante à cette taille.
2. **Le facteur de crête domine la baseline.** La normalisation d'amplitude retire le niveau,
   pas tous ses corrélats. Un contrôle plus sévère consisterait à retirer aussi l'enveloppe.
3. **Le contrôle RMS n'est pas entraîné** : c'est un seul scalaire avec un seuil issu de la
   validation. Il est volontairement primitif, c'est ce qui en fait un plancher lisible.
4. **Aucune ablation temporelle** (`EVAL_PROTOCOL.md` §7) : la question « la structure
   temporelle apporte-t-elle quelque chose » reste ouverte et appartient au TSLM.
5. **Le hold-out device n'a pas été évalué** ici. Il reste une évaluation secondaire, et il
   ne contient aucun clip de bruit.
6. **Le jeu TimeF n'est pas publié dans un registre** : il est écrit localement, hors dépôt.
   Suffisant pour l'entraînement, pas pour une distribution.
