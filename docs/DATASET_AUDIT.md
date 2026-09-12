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

| | Clé métadonnées seule | Après fusion audio | **Clé finale (+ fusion bi-device, §10)** |
|---|---|---|---|
| Groupes | 327 | 322 | **306** |
| min / médiane / max | 1 / 2 / 74 | 1 / 2 / 74 | **1 / 2 / 74** |
| moyenne | 3,06 | 3,11 | 3,27 |
| Groupes multi-classes | — | 0 | **0** |

| Classe | Groupes (clé finale) | min | médiane | max | plus gros groupe |
|---|---|---|---|---|---|
| leak | **255** | 1 | 2 | 12 | 2,4 % de la classe |
| no leak | **18** | 1 | 10 | **74** | **19,2 % de la classe** |
| noise | **33** | 2 | 3 | 7 | 6,1 % de la classe |

**Vue binaire** — la source décrit les 114 clips de bruit comme portant des *« detailed no-leak
labels »*. En tâche *leak* vs *non-leak* :

| | Clips | Groupes |
|---|---|---|
| leak | **500** | 255 |
| non-leak (386 + 114) | **500** | 51 |

> Le dataset est **exactement équilibré, 500 / 500**, en tâche binaire. `EVAL_PROTOCOL.md` §5
> justifie le macro-F1 par un déséquilibre 500/386/114 : cet argument ne tient que pour la
> tâche à trois classes.

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
| **L5** | **Même condition physique, deux devices** — 16 conditions sur 27 | **démontré** | Le même événement physique de part et d'autre du split, via deux instruments. | **Tranché en §10** : fusion ciblée des 16 conditions concernées. |
| **L6** | **Granularité *no leak*** : 18 groupes, un de 74 clips | **démontré** | Pas un leakage : une instabilité. Le score *no leak* dépend de quelques groupes. | Reporter des intervalles par groupe, pas un chiffre unique. |
| **L7** | Similarité inter-classe résiduelle (1 paire à 0,64) | **observé** | Marginal. | Mentionné, non traité. |
| **L8** | Bruits issus d'un site public (`dlmeasure.com`), distribution inconnue | **non vérifié** | La classe *noise* pourrait être triviale à séparer pour une raison extrinsèque. | À vérifier avant toute conclusion sur la robustesse au bruit. |

### Le dilemme L5 — voir §10, il est tranché

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
2. ~~**Le dilemme device (L5) n'est pas résolu.**~~ **Tranché en §10** : les 16 conditions
   bi-device sont fusionnées (67 clips, 6,7 % du dataset). Le hold-out device devient une
   évaluation *secondaire*, sur 87 *leak* + 107 *non-leak* hydrophone, et **sans aucun clip de
   bruit** — la classe *noise* est captée à 100 % au noise logger.
3. **7,2 % du dataset est de la duplication exacte**, et 6 paires portent des étiquettes
   contradictoires. La curation amont n'est pas irréprochable ; le dire nous-mêmes vaut mieux
   que de le voir découvert.
4. **L1 est le risque majeur** et il est indépendant du split : il suffit de laisser fuiter un
   nom de fichier dans un prompt de TSLM pour que tout le protocole s'effondre. Les descriptions
   textuelles générées devront être auditées sur ce point précis.
5. **L'hypothèse « fenêtre de 5 s » n'est pas confirmée par la source.** Elle guide le
   groupage de manière conservatrice, ce qui est sûr ; elle ne doit pas être présentée au jury
   comme un fait établi.

6. **Un raccourci de niveau sonore traverse tout le dataset** (§11, risque **L9**). C'est la
   réserve la plus lourde, et elle est postérieure à la première version de cet audit.

**Ce que cet audit n'établit pas** : que la tâche est apprenable, qu'une baseline atteint un
score utile, que le signal temporel apporte quoi que ce soit. Aucune baseline n'a été entraînée.
Aucun split définitif n'a été écrit. Ces questions restent ouvertes.

---

## 9. Prochaine étape unique

~~Trancher le dilemme L5.~~ **Fait — §10.** Étape suivante : figer les groupes dans un manifeste
versionné (le manifeste seul, **jamais les WAV**), puis traiter **L9** — normalisation
d'amplitude et contrôle « RMS seul » — avant toute mesure de performance.

---

## 10. Résolution du dilemme L5 — device dans la clé, conditions bi-device fusionnées

*Ajouté le 2026-09-12, après la décision d'équipe de valider la direction LeakLess software-only.*

Le dilemme était posé comme un choix binaire. Il n'en est pas un : **le conflit ne porte que sur
67 clips, soit 6,7 % du dataset.**

| Option | Groupes | Défaut |
|---|---|---|
| Device hors de la clé | 242 | *no leak* tombe à 10 groupes, dont un de 104 clips |
| Device dans la clé, sans fusion | 322 | 16 conditions physiques à cheval sur deux groupes |
| **Device dans la clé + fusion des 16 conditions bi-device** | **306** | **le coût réel : 16 fusions, 67 clips** |

**Décision : troisième option.** Le device reste dans la clé — il sépare des sessions réelles et
préserve la granularité — mais les 16 conditions captées par les deux instruments sont fusionnées
par union-find. Le même événement physique ne peut plus se retrouver des deux côtés du split.

Implémenté dans `build_groups.py` (comportement par défaut ; `--no-merge-cross-device` pour
mesurer le coût de l'option inverse).

### Conséquence sur le hold-out device

`EVAL_PROTOCOL.md` §2.5 demande de réserver un device entier pour le test. C'est possible, mais
ce n'est **pas** le split principal — c'est une évaluation secondaire, et elle a deux limites
mesurées :

| | |
|---|---|
| Hydrophone en test | 107 *leak* + 107 *non-leak* |
| Après exclusion des 20 clips hydrophone des conditions bi-device | **87 *leak* + 107 *non-leak*** |
| **Clips de bruit disponibles en hydrophone** | **0** — la classe *noise* est captée à 100 % au noise logger |

> Le hold-out device ne peut donc **pas** tester la robustesse au bruit environnemental. Les deux
> évaluations — split groupé et hold-out device — répondent à deux questions distinctes et
> doivent être rapportées séparément.

---

## 11. Étiquettes vérifiées depuis le contenu des fichiers — et le raccourci qu'elles cachent

> ⚠️ **Ce qui suit n'est pas une baseline.** Rien n'est entraîné, rien n'est ajusté, aucune donnée
> n'est tenue à l'écart. Ce sont des **statistiques descriptives sur la totalité du dataset**,
> destinées à répondre à une seule question : les étiquettes correspondent-elles à quelque chose
> d'audible ? Les AUC ci-dessous **ne peuvent pas être citées comme une performance.**
>
> Reproduction : `build_groups.py --descriptors`.

### 11.1 Descripteurs par classe (médiane, Q1–Q3)

| Descripteur | leak | no leak | noise |
|---|---|---|---|
| **Niveau RMS (dBFS)** | **−11,7** (−14,4 … −8,1) | **−23,1** (−27,4 … −17,9) | −16,8 (−18,7 … −13,7) |
| Centroïde spectral (Hz) | 516 (385 … 752) | 501 (434 … 597) | 875 (625 … 1069) |
| Ratio d'énergie > 1 kHz | 0,060 | 0,073 | 0,276 |
| Taux de passages par zéro | 0,115 | 0,121 | 0,176 |

### 11.2 Séparation par descripteur unique, agrégée **par groupe**

AUC calculée sur les médianes de groupe (255 groupes *leak* contre 51 *non-leak*), pour éviter la
pseudo-réplication des clips d'une même session :

| Descripteur | AUC descriptive | Lecture |
|---|---|---|
| **Niveau RMS** | **0,857** | ⚠️ **un seul scalaire — le volume — sépare la tâche binaire** |
| Centroïde spectral | 0,312 | séparation inverse (0,688 dans l'autre sens), portée par la classe *noise* |
| Ratio > 1 kHz | 0,281 | idem |
| Taux de passages par zéro | 0,261 | idem |

### 11.3 Ce que cela démontre, et ce que cela coûte

✅ **Les étiquettes ne sont pas vides.** Elles correspondent à une différence acoustique réelle et
mesurable. Le critère 2 du GO/NO-GO est satisfait.

🔴 **Mais la différence dominante est un écart de niveau sonore de ~11 dB.** Les enregistrements
*leak* sont systématiquement plus forts que les *no leak*. C'est un artefact de protocole
d'acquisition — capteur posé près d'une fuite active contre capteur posé sur une conduite
silencieuse — pas une signature de fuite.

> ### Nouveau risque **L9 — raccourci de niveau sonore**
>
> | | |
> |---|---|
> | Statut | **démontré** (§11.2) |
> | Mécanisme | Le niveau RMS seul atteint une AUC descriptive de 0,857 au niveau groupe |
> | Conséquence | Un TSLM peut obtenir un score élevé en mesurant le volume. Le score serait réel, la capacité annoncée serait fausse. Et l'amplitude WAV **n'est pas calibrée** : rien ne garantit que cet écart survive à un autre matériel. |
> | Contrôle exigé | **(a)** normalisation d'amplitude par clip, appliquée identiquement à toutes les classes ; **(b)** un contrôle « RMS seul » publié à côté de tout résultat. Un modèle qui ne bat pas 0,857 sur le split groupé n'apporte rien. |

C'est, de tout cet audit, le point le plus important pour la suite : **L1 se contrôle en ne
donnant pas les noms de fichiers au modèle ; L9 ne se contrôle pas, il se mesure et se publie.**

---

## 12. Faisabilité d'un split groupé 60/20/20

*Faisabilité uniquement. **Aucun split n'a été écrit ni figé.*** Tâche binaire, groupes entiers,
clé finale de 306 groupes.

| Fold | Clips | *leak* | *non-leak* | Groupes *leak* | Groupes *non-leak* |
|---|---|---|---|---|---|
| train | 600 | 300 | 300 | 153 | 31 |
| validation | 200 | 100 | 100 | 51 | 9 |
| test | 200 | 100 | 100 | 51 | 11 |

- **0 groupe coupé** entre deux folds.
- Les proportions cibles sont atteintes **exactement**, conséquence directe de l'équilibre
  500 / 500 et de la finesse de la classe *leak*.
- ⚠️ **Le test *non-leak* reste grossier** : ses 100 clips proviennent de 11 groupes, dont un de
  43 clips (43 % du fold). La taille d'échantillon *effective* est bien inférieure à 100.

> Conséquence à appliquer sans exception : **tout score *non-leak* doit être publié avec le
> nombre de groupes du test**, et accompagné d'une dispersion inter-groupes — jamais d'un chiffre
> unique.

---

## 13. GO / NO-GO — avant tout entraînement

| # | Critère | Constat | Verdict |
|---|---|---|---|
| 1 | **Comptes de classes réels** | 500 / 386 / 114 comptés = annoncé. Binaire : **500 / 500 exact** | ✅ |
| 2 | **Étiquettes vérifiées depuis les fichiers** | Séparation acoustique réelle et mesurée (§11) — **mais dominée par le niveau sonore** | ⚠️ **oui, sous condition L9** |
| 3 | **Couverture du parsing des métadonnées** | **1000 / 1000**, une seule expression régulière, 0 échec | ✅ |
| 4 | **Nombre et taille des groupes indépendants** | **306 groupes** (255 *leak*, 51 *non-leak*), médiane 2, max 74 | ⚠️ **fin côté *leak*, grossier côté *non-leak*** |
| 5 | **Cas ambigus / collisions** | **97,2 %** non ambigus ; 16 clés dégénérées, 12 clips reliés par doublon, **0 nom non parsable**, **0 groupe multi-classes** | ✅ |
| 6 | **Doublons / enregistrements dépendants** | 36 paires identiques (7,2 %), **0 inter-classe** ; dépendances de session mesurées ×8 à ×14 et neutralisées par la clé | ✅ **identifiés et contrôlés** |
| 7 | **Split groupé tenable** | 60/20/20 exact, **0 groupe coupé** ; hold-out device possible en secondaire, sans bruit | ✅ **avec réserve de granularité** |

### Verdict

> # 🟢 **GO — conditionnel**

Les sept critères sont franchis. Deux conditions doivent être en place **avant** que le premier
chiffre de performance ne soit produit — pas après :

1. **Normalisation d'amplitude par clip**, identique pour toutes les classes, **et** publication
   d'un contrôle « RMS seul » (AUC descriptive 0,857) à côté de chaque résultat. Sans ce
   contrôle, aucun score n'est interprétable (L9).
2. **Aucune métadonnée de nom de fichier ne doit atteindre le modèle** — ni en entrée, ni dans un
   prompt, ni dans une description générée. Pression et débit valent 93,8 % de rappel à eux
   seuls (L1).

Et deux règles de publication, qui découlent de l'audit :

3. Tout score *non-leak* est accompagné du **nombre de groupes du test** et de leur dispersion.
4. Le hold-out device est rapporté **séparément** du split groupé, en précisant qu'il ne contient
   aucun clip de bruit environnemental.

### Ce que ce GO ne couvre pas

Il autorise la **préparation** : conversion TimeNet, génération d'annotations ancrées sur des
propriétés de signal mesurables, entraînement. Il ne dit rien de l'apprenabilité de la tâche, et
il n'autorise **aucune** affirmation de performance : `CLAIM_LEDGER.md` reste la référence, et
les entrées ⏳ qui y figurent n'ont pas bougé.
