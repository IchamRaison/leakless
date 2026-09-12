# DATASET_AUDIT — audit du dataset acoustique Zenodo 18631450

> Réalisé le 2026-09-12. **Aucun split définitif, aucune baseline, aucun entraînement, aucune
> annotation générée.** Le seul but est de répondre à une question : *un split groupé sans
> leakage évident est-il possible sur ce dataset ?*
>
> Les fichiers audio vivent **hors du dépôt**. Rien de ce qui est décrit ici n'ajoute un `.wav`
> ou un `.rar` à Git.

---

## 0. Méthode et reproductibilité

| | |
|---|---|
| Source | `https://zenodo.org/api/records/18631450` — CC BY 4.0 |
| Titre du record | *Acoustic data for the manuscript «Self-supervised acoustic leakage detection for water distribution systems: A real-time diagnosis framework under data scarcity»* |
| DOI | `10.5281/zenodo.18631450` — publié le 2026-02-13 |
| Emplacement local | `~/dev/sandbox/ehl-zurich/data-audit/` — **hors dépôt** |
| Extraction | `unar` (les archives sont des RAR) |
| Script | [`scripts/ingest/build_groups.py`](../scripts/ingest/build_groups.py) |

```bash
# téléchargement hors dépôt + vérification d'intégrité
curl -sL "https://zenodo.org/api/records/18631450/files/<nom%20url-encodé>.rar/content" -o "<nom>.rar"
md5 -q *.rar          # comparé aux checksums renvoyés par l'API Zenodo
unar -o extract "<nom>.rar"

python3 scripts/ingest/build_groups.py \
    --data-root ~/dev/sandbox/ehl-zurich/data-audit/extract \
    --json ~/dev/sandbox/ehl-zurich/data-audit/groups_manifest.json
```

**Écart assumé par rapport à la consigne :** l'archive `leak acoustic data.rar` (2,9 Mo) a été
téléchargée elle aussi, alors que seules les deux archives jamais ouvertes étaient demandées.
Sans elle, aucune collision inter-classe ni aucun appariement inter-device n'était mesurable —
et ce sont deux des sept mesures attendues. Coût : 2,9 Mo, hors dépôt, même licence, même
record. Les trois MD5 correspondent à ceux publiés par l'API :
`d6b9fcd8…` (no leak), `6080966b…` (environmental noise), `2ec7cef5…` (leak).

---

## 1. OBSERVATIONS — faits bruts, vérifiés

### 1.1 Volumétrie — les comptes annoncés sont exacts

| Classe | Annoncé par Zenodo | **Compté** | Verdict |
|---|---|---|---|
| leak | 500 | **500** | ✅ conforme |
| no leak | 386 | **386** | ✅ conforme — *archive jamais ouverte jusqu'ici* |
| environmental noise | 114 | **114** | ✅ conforme — *archive jamais ouverte jusqu'ici* |
| **total** | 1000 | **1000** | ✅ |

Un seul fichier non-audio : `no leak acoustic data/desktop.ini` (31 ko, artefact Windows). Son
contenu est une simple table `[LocalizedFileNames]` recopiant les noms de fichiers — **aucune
métadonnée supplémentaire exploitable**.

### 1.2 Format audio — parfaitement homogène

**1000 / 1000** fichiers : mono · 8 000 Hz · 16 bits PCM · **1,000 s exactement**. Aucune
exception, aucun fichier tronqué, aucune fréquence d'échantillonnage divergente.

### 1.3 Structure des noms de fichiers — 100 % parsables

La règle annoncée par la source est incomplète. La structure réelle, valide sur **1000/1000**
fichiers, est :

```
<body>[-_]<w0>-<w1>[_<N>].wav
```

| Composant | leak / no leak | environmental noise |
|---|---|---|
| `<body>` | `Matériau-Région-Pression MPa-Débit ms-Device` (5 champs, 886/886) | `Catégorie-Région-Device` (3 champs, 114/114) |
| `<w0>-<w1>` | `0-1` … `4-5` | `0-1` … `4-5` |
| `_<N>` | index optionnel | index optionnel |

> **Le couple `w0-w1` n'est pas documenté par la source.** Il prend exclusivement les valeurs
> `0-1`, `1-2`, `2-3`, `3-4`, `4-5`, et un même `<body>_<N>` apparaît souvent avec les cinq
> valeurs. Exemple intégral : `dog_1-NA-noise logger_{0-1,1-2,2-3,3-4,4-5}.wav`.

### 1.4 Répartition des fenêtres

| Classe | Distribution des fenêtres | Enregistrements parents `(body, N)` | dont couvrant ≥ 2 secondes |
|---|---|---|---|
| leak | `0-1`: 498 · `2-3`: 1 · `4-5`: 1 | 499 | **1** |
| no leak | `0-1`: 119 · `1-2`: 59 · `2-3`: 82 · `3-4`: 60 · `4-5`: 66 | 124 | **88** |
| noise | `0-1`: 33 · `1-2`: 20 · `2-3`: 18 · `3-4`: 22 · `4-5`: 21 | 35 | **34** |

### 1.5 Métadonnées renseignées, par classe

| Champ | leak (500) | no leak (386) | noise (114) |
|---|---|---|---|
| matériau | 494 | 366 | 0 (sans objet) |
| région | 152 | 179 | 5 |
| **pression** | **469** | **0** | **0** |
| **débit** | **426** | **0** | **0** |

| Device | leak | no leak | noise |
|---|---|---|---|
| hydrophone | 107 | 107 | 0 |
| noise logger | 391 | 279 | 114 |
| NA | 2 | 0 | 0 |

16 catégories de bruit : `dog` (22), `drilling` (25), `engine` (13), `rain` (11), `hydrant` (6),
`grass noise` (5), `hydrant_normal` (5), `fan motor` (4), `alarm` (3), `drum` (3), `fan` (3),
`flowing water` (3), `gas pipe` (3), `unknown_noise` (3), `wind` (3), `music` (2).

### 1.6 Doublons octet-identiques

**36 paires, 72 fichiers** (7,2 % du dataset) sont strictement identiques au MD5 près.

- **0 paire inter-classe.** Aucun fichier n'est étiqueté à la fois *leak* et *no leak*.
- **6 paires relient deux clés de métadonnées différentes** — c'est-à-dire que le même audio
  porte deux étiquettes contradictoires :

| Fichier A | Fichier B | Ce qui diffère |
|---|---|---|
| `ductile iron-zone 2-0.312 MPa-0.9 ms-hydrophone-0-1_1.wav` | `pe-zone 2-0.305 MPa-1.6 ms-noise logger-0-1.wav` | **matériau, pression, débit ET device** |
| `pe-NA-0.3110 MPa-0.5 ms-noise logger-0-1.wav` | `pe-NA-NA-NA-noise logger-0-1.wav` | pression et débit effacés |
| `ductile iron-NA-0.1545 MPa-3.36 ms-noise logger-0-1_1.wav` | `ductile iron-NA-0.1630 MPa-3.45 ms-noise logger-0-1.wav` | pression et débit |
| `pe-NA-0.1545 MPa-3.36 ms-noise logger-0-1_3.wav` | `pe-NA-0.1630 MPa-3.45 ms-noise logger-0-1.wav` | pression et débit |
| `hydrant-NA-noise logger_1-2.wav` | `hydrant_normal-NA-noise logger_1-2.wav` | catégorie de bruit |
| `hydrant-NA-noise logger_2-3.wav` | `hydrant_normal-NA-noise logger_2-3.wav` | catégorie de bruit |

### 1.7 Quasi-doublons acoustiques

Corrélation de log-spectrogrammes **centrés sur la moyenne globale** (sans ce centrage, la
similarité cosinus médiane est de 0,96 et ne discrimine rien ; après centrage, la médiane est
de −0,007).

| Seuil | Paires | dont inter-classe | dont même `<body>` |
|---|---|---|---|
| > 0,8 | **36** — exactement les 36 doublons octet-identiques | 0 | 30 |
| > 0,7 | 54 (+18) | 0 | 33 |
| > 0,6 | 446 (+392) | **1** | 58 |

L'unique paire inter-classe à 0,641 : `leak/ductile iron-zone 2-0.3111 MPa-0.15 ms-hydrophone-0-1.wav`
vs `no_leak/steel-zone 1-NA-NA-noise logger_3-4.wav` — matériau, région et device tous
différents, fichiers non identiques.

### 1.8 Corrélation intra-groupe vs inter-groupe

| Comparaison | Corrélation moyenne | Référence (rien en commun) | Ratio |
|---|---|---|---|
| leak, même `<body>` | **0,101** (n=341) | 0,008 | **×12,6** |
| no leak, même `(body, N)` — secondes consécutives | **0,180** (n=584) | 0,016 | ×11,2 |
| no leak, même `(body, fenêtre)` — `N` différents | **0,219** (n=1795) | 0,016 | ×13,7 |
| noise, même `<body>` | **0,356** (n=145) | 0,149 | ×2,4 |
| **leak, même condition physique, device différent** | **−0,081** (n=70) | 0,008 | — |
| leak, même condition physique, même device | 0,189 (n=88) | 0,008 | ×23,6 |

---

## 2. INTERPRÉTATION

*Cette section raisonne à partir du §1. Elle est explicitement séparée des faits.*

1. **`w0-w1` est une fenêtre temporelle**, pas un identifiant. Les valeurs sont exclusivement
   les cinq secondes consécutives `0-1` … `4-5`, et 88 des 124 enregistrements parents de la
   classe *no leak* en couvrent plusieurs. Lecture la plus économique : chaque enregistrement
   parent dure ~5 s et a été découpé en clips d'une seconde.
2. **`_N` indexe des répétitions sous la même condition** — même matériau, même région, même
   pression, même débit, même device. Pour *no leak*, `N` et la fenêtre varient tous deux à
   l'intérieur d'un `<body>`, et les deux axes sont fortement corrélés (0,180 et 0,219 contre
   0,016) : le `<body>` entier se comporte comme **une seule session acoustique**.
3. **Pression et débit ne sont renseignés que pour la classe *leak***. Ce n'est pas un détail
   de remplissage : c'est une information qui **détermine l'étiquette**.
4. **Le device n'est pas un raccourci vers l'étiquette.** 107 hydrophones côté *leak*, 107 côté
   *no leak* — un équilibre trop exact pour être fortuit ; les auteurs semblent avoir apparié
   délibérément. Sur *noise logger*, 391 *leak* contre 279+114 = 393 non-*leak*. Le device seul
   ne prédit rien.
5. **Hydrophone et noise logger d'une même condition ne se ressemblent pas acoustiquement**
   (corrélation moyenne −0,081, *inférieure* au hasard). Les deux chaînes de mesure ont des
   signatures spectrales franchement distinctes. Cela ne supprime pas le lien : c'est le même
   événement physique, donc la même étiquette, capté deux fois.
6. **Les 6 paires de doublons à étiquettes contradictoires signalent une erreur de curation en
   amont**, pas un phénomène physique. Un même fichier a été recopié sous deux noms. Les
   métadonnées ne sont donc pas fiables à 100 %, et une clé de groupe qui s'y fie seule
   sépare des fichiers identiques.

---

## 3. CLÉ DE GROUPE PROPOSÉE

Fondée **uniquement sur des métadonnées observables dans le nom de fichier**, sans aucune
information privilégiée.

```
leak / no leak :  classe | matériau | région | pression | débit | device
noise          :  classe | catégorie | région | device
```

puis **fusion par union-find** de toute paire de clés reliées par un doublon audio
(exact, ou corrélation > 0,8).

Deux décisions explicites :

| Décision | Raison |
|---|---|
| **La fenêtre `w0-w1` est EXCLUE de la clé** | Sinon deux secondes consécutives du même enregistrement tombent de part et d'autre du split. C'est le leakage temporel, textuellement interdit par la consigne du challenge. |
| **L'index `_N` est EXCLU de la clé** | Mesuré : les répétitions d'un même `<body>` sont corrélées 12 à 14 fois plus que le hasard. Les séparer serait du leakage de session. |

La clé **sur-fusionne délibérément**. Sur-fusionner coûte des groupes ; sous-fusionner coûte la
validité du score.

---

## 4. MESURES DEMANDÉES

### 4.1 Pourcentage de clips à groupe non ambigu

Un clip est déclaré **ambigu** si (a) son nom ne parse pas, (b) il est relié à une autre clé par
un doublon audio, ou (c) sa clé est dégénérée — matériau **et** région inconnus, donc
indiscernable d'une autre session.

| | Clips | % |
|---|---|---|
| **Non ambigus** | **972** | **97,2 %** |
| Ambigus | 28 | 2,8 % |
| → clé dégénérée (`NA-NA-…`) | 16 | 6 *leak*, 10 *no leak* |
| → relié à un autre groupe par un doublon audio | 12 | |
| → nom non parsable | **0** | |

À signaler aussi : 2 clips *leak* ont un **device `NA`** (`ductile iron-NA-NA-NA-NA-0-1.wav` et
son `_1`).

### 4.2 Nombre de groupes uniques et tailles

| | Clé métadonnées seule | **Clé finale (après fusion audio)** |
|---|---|---|
| Groupes | 327 | **322** |
| min / médiane / max | 1 / 2 / 74 | **1 / 2 / 74** |
| moyenne | 3,06 | 3,11 |
| Groupes multi-classes | — | **0** |

| Classe | Groupes | min | médiane | max | plus gros groupe |
|---|---|---|---|---|---|
| leak | **271** | 1 | 2 | 9 | 1,8 % de la classe |
| no leak | **18** | 1 | 10 | **74** | **19,2 % de la classe** |
| noise | **33** | 2 | 3 | 7 | 6,1 % de la classe |

> ⚠️ **Le problème est concentré sur la classe *no leak* : 386 clips pour seulement 18 groupes,
> dont deux de 74 clips.** C'est le facteur limitant de tout ce qui suit.

### 4.3 Collisions et noms impossibles à parser

- **0 nom non parsable** sur 1000.
- **6 collisions métadonnées** : audio identique, étiquettes différentes (§1.6).
- **0 collision inter-classe.**
- 1 paire inter-classe à corrélation 0,641, non identique — à surveiller, non démontrée.

### 4.4 Mêmes conditions / devices présentes dans plusieurs fichiers

Sur les conditions *leak* dont les quatre champs sont renseignés :

| | |
|---|---|
| Conditions complètes distinctes | **27** |
| **Vues par les deux devices** | **16** (59 %) |
| Conditions présentes dans ≥ 2 fichiers | 26 / 27 |
| Maximum de fichiers pour une condition | 8 |

Variante sans le device dans la clé — c'est-à-dire en groupant par condition physique :
242 groupes au lieu de 322 ; *no leak* tombe à **10 groupes**, dont un de 104 clips (26,9 % de
la classe).

### 4.5 Faisabilité d'un test à 20 % (borne indicative, aucun split produit)

| Classe | Cible 20 % | Atteignable par groupes entiers | Groupes mobilisés |
|---|---|---|---|
| leak | 100 clips | 100 | 100 |
| no leak | 77 clips | **66** | 11 |
| noise | 23 clips | 22 | 9 |

---

## 5. DÉMONTRÉ

*Vérifiable en relançant le script sur les archives téléchargées. Rien ici ne repose sur une
interprétation.*

1. Les trois archives contiennent **exactement** 500 / 386 / 114 clips WAV. La fiche Zenodo est
   exacte. **Les deux archives jamais ouvertes sont désormais auditées.**
2. Les 1000 fichiers sont mono 8 kHz 16 bits d'exactement 1,000 s. Aucune exception.
3. Les 1000 noms de fichiers parsent avec une seule expression régulière. Aucun cas résiduel.
4. **36 paires de fichiers (72 clips, 7,2 %) sont octet-identiques.** Aucune ne franchit une
   frontière de classe. Six portent des étiquettes contradictoires.
5. **Pression et débit ne sont jamais renseignés hors de la classe *leak*** : 469/500 et 426/500
   côté *leak*, **0/500** côté *no leak* et *noise*. La règle « pression ≠ NA ⇒ leak » a une
   précision de 100 % et un rappel de 93,8 % **sans écouter un seul échantillon audio**.
6. Une clé de groupe fondée sur les seules métadonnées observables couvre **97,2 %** des clips
   sans ambiguïté, et produit **322 groupes** dont **aucun ne mélange deux classes**.
7. Les clips partageant un `<body>` sont corrélés 8 à 14 fois plus que des clips sans rien en
   commun. **Le regroupement par `<body>` correspond à une réalité acoustique mesurée**, pas à
   une convention de nommage.
8. Hydrophone et noise logger d'une même condition physique sont **acoustiquement dissemblables**
   (corrélation moyenne −0,081).

---

## 6. HYPOTHÈSE — non démontré, à ne pas affirmer

1. **`w0-w1` = seconde `w0` à seconde `w1` d'un enregistrement parent de 5 s.** Cohérent avec
   toutes les observations, mais la source ne documente pas ce champ. Personne n'a confirmé.
2. **`_N` = répétition d'une même session.** L'alternative — `N` désigne des sessions distinctes
   sous une condition identique — n'est pas exclue par les données. *Les deux lectures
   conduisent à la même décision de groupage*, donc l'ambiguïté ne bloque rien.
3. **Les 6 doublons contradictoires sont des erreurs de curation.** Plausible, non prouvé.
4. **Les fichiers `NA-NA-…` sont des sessions distinctes.** Invérifiable : c'est précisément
   l'information manquante.
5. **Le déséquilibre matériau (`ductile iron` 281 *leak* / 178 *no leak*) n'est pas exploitable
   comme raccourci par un modèle audio.** Non testé — et ce test exige une baseline, hors
   périmètre aujourd'hui.
6. **18 groupes suffisent à estimer une performance *no leak* stable.** Douteux, et non mesuré.

---

## 7. RISQUES DE LEAKAGE

| # | Risque | Statut | Conséquence si ignoré | Contrôle |
|---|---|---|---|---|
| **L1** | **Métadonnées = étiquette.** Pression/débit renseignés uniquement côté *leak* | **démontré** | 93,8 % de rappel sans audio. Un modèle qui reçoit le nom de fichier ne détecte rien du tout. | Les noms, chemins et champs de métadonnées **ne sont jamais des entrées**. Supervision et groupage uniquement. |
| **L2** | **Fenêtres consécutives.** 88/124 enregistrements *no leak* couvrent plusieurs secondes | **démontré** | Deux secondes voisines du même enregistrement à cheval sur le split. | `w0-w1` exclu de la clé. |
| **L3** | **Répétitions d'une même session** (`_N`), corrélation ×12 à ×14 | **démontré** | Même session des deux côtés. | `_N` exclu de la clé. |
| **L4** | **Doublons octet-identiques** — 36 paires, dont 6 à cheval sur deux clés | **démontré** | Le même fichier en train et en test. | Fusion union-find par MD5 **avant** tout split. |
| **L5** | **Même condition physique, deux devices** — 16 conditions sur 27 | **démontré** | Le même événement physique de part et d'autre du split, via deux instruments. | **Dilemme non résolu, voir ci-dessous.** |
| **L6** | **Granularité *no leak*** : 18 groupes, un de 74 clips | **démontré** | Pas un leakage : une instabilité. Le score *no leak* dépend de quelques groupes. | Reporter des intervalles par groupe, pas un chiffre unique. |
| **L7** | Similarité inter-classe résiduelle (1 paire à 0,64) | **observé** | Marginal. | Mentionné, non traité. |
| **L8** | Bruits issus d'un site public (`dlmeasure.com`), distribution inconnue | **non vérifié** | La classe *noise* pourrait être triviale à séparer pour une raison extrinsèque. | À vérifier avant toute conclusion sur la robustesse au bruit. |

### Le dilemme L5, énoncé sans le trancher

| Option | Effet | Coût |
|---|---|---|
| **Device DANS la clé** (322 groupes) | 16 conditions physiques se retrouvent à cheval sur deux groupes → leakage de condition | Granularité préservée |
| **Device HORS de la clé** (242 groupes) | Chaque condition physique reste entière | *no leak* chute à **10 groupes**, dont un de 104 clips (26,9 %) — l'évaluation devient très instable |

> Ce choix doit être fait **explicitement, et documenté**, pas subi. Il n'est pas tranché ici :
> il dépend du protocole retenu (`docs/EVAL_PROTOCOL.md` §2.5 demande de réserver un device
> entier pour le test — ce que l'option « device hors de la clé » rend impossible).

---

## 8. DÉCISION

> # ✅ **VIABLE AVEC RÉSERVES**

**Pourquoi viable :**

- Les comptes annoncés sont exacts, le format est homogène, **0 nom non parsable sur 1000**.
- Une clé de groupe purement observable couvre **97,2 %** des clips, sans aucun groupe à cheval
  sur deux classes.
- Les cinq vecteurs de leakage principaux (L1–L5) sont **identifiés, mesurés et contrôlables**
  par des règles mécaniques, pas par du jugement au cas par cas.
- Aucun doublon inter-classe : le dataset n'est pas corrompu à sa racine.

**Réserves, à énoncer dans toute communication de résultat :**

1. **La classe *no leak* n'offre que 18 groupes**, dont deux à 74 clips. Tout score *no leak*
   doit être accompagné du nombre de groupes du test, sous peine d'être ininterprétable.
2. **Le dilemme device (L5) n'est pas résolu.** Tant qu'il ne l'est pas, aucun chiffre définitif
   ne doit être publié.
3. **7,2 % du dataset est de la duplication exacte**, et 6 paires portent des étiquettes
   contradictoires. La curation amont n'est pas irréprochable ; le dire nous-mêmes vaut mieux
   que de le voir découvert.
4. **L1 est le risque majeur** et il est indépendant du split : il suffit de laisser fuiter un
   nom de fichier dans un prompt de TSLM pour que tout le protocole s'effondre. Les descriptions
   textuelles générées devront être auditées sur ce point précis.
5. **L'hypothèse « fenêtre de 5 s » n'est pas confirmée par la source.** Elle guide le
   groupage de manière conservatrice, ce qui est sûr ; elle ne doit pas être présentée au jury
   comme un fait établi.

**Ce que cet audit n'établit pas** : que la tâche est apprenable, qu'une baseline atteint un
score utile, que le signal temporel apporte quoi que ce soit. Aucune baseline n'a été entraînée.
Aucun split définitif n'a été produit. Ces questions restent ouvertes.

---

## 9. Prochaine étape unique

Trancher le **dilemme L5** — device dans la clé ou hors de la clé — puis figer les groupes dans
un manifeste versionné (le manifeste seul, **jamais les WAV**). Tout le reste en dépend : split,
baseline, ablation, démo.
