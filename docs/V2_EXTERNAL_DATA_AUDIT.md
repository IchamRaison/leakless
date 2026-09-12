# V2 — audit du nouveau holdout acoustique

Audit du 12 septembre 2026, sources publiques et petits échantillons seulement. Aucun modèle chargé, score produit, entraînement ou métrique calculé. Le test historique de 194 clips reste inchangé et n'est pas un nouveau holdout.

## Conclusion et état réel

**Aghashahi est réellement accessible sans compte** et son sous-ensemble hydrophone peut être préparé sans rééchantillonnage. L'inventaire réel contient **120 signaux étiquetés + 2 bruits ambiants**, pas 280 hydrophones. L'adaptateur de préparation est implémenté ; ses sept tests synthétiques CPU passent. Le contrôle des noms/tailles a également été appliqué au répertoire central ZIP réel, sans télécharger toute l'archive.

**Mise à jour réelle, 13 septembre :** archive intégrale et convertisseur vérifiés,
préparation des 3 600 fenêtres primaires et 60 bruits terminée depuis `aaab4af`.
Artefacts : `/home/hicham/pipe-v0/data/external/aghashahi-v1` ; reçus dans
`docs/evidence/tslm-v2/external-preparation/`. Aucun score de modèle calculé.

Audit exhaustif de recouvrement terminé depuis `41dd8b2` : 122 000 paires
(1 000 WAV historiques × 122 enregistrements externes), 232 001 décalages entiers
par paire dans les trente secondes conservées. **Aucun candidat à
`|Pearson| >= 0.995`, aucune paire/fenêtre non évaluable, aucun doublon exact.**
Gain, composante continue et inversion de polarité sont couverts ; filtrage,
rééchantillonnage et décalages fractionnaires ne le sont pas. Ce résultat n'est
pas une preuve d'indépendance des sessions. Aucune exclusion automatique.
Rapport : `docs/evidence/tslm-v2/external-overlap-001/summary.json`, SHA
`5a1a4f090706a74fe7ec65d43017a76fb5841cd5464035294b85e10c5f8233fd`.
Il reste à figer le manifeste d'évaluation explicite et le modèle/seuil avant
l'inférence externe ; les sections d'audit initial ci-dessous restent historiques.

Hong Kong est également public, mais présente un confondant important : **28 des 40 fichiers hydrophone fuite sont à 4 096 Hz ; les 40 non-fuite et les 12 autres fuite sont à 8 092 Hz**. Ces valeurs viennent des 80 en-têtes WAV, pas d'une supposition à partir de la page. Ce jeu ne remplace pas automatiquement Aghashahi comme confirmation primaire.

**Aucune de ces sources n'apporte une acquisition continue avec instants de début/fin de fuite annotés.** L'objectif continu nécessite encore une collecte ou une source appropriée fournie/confirmée avec Icham.

## 1. Aghashahi : provenance et accès vérifiés

Source officielle : [Dataset of Leak Simulations in Experimental Testbed Water Distribution System, v1](https://data.mendeley.com/datasets/tbrnp6vrnj/1), DOI `10.17632/tbrnp6vrnj.1`, publié le 12 décembre 2022. Auteurs : Mohsen Aghashahi, Lina Sela, M. Katherine Banks. Licence annoncée : [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ; conserver attribution, DOI, licence et description des transformations.

La page décrit un banc PVC unique de 47 m, deux topologies (branched/looped), quatre types de fuite et une condition sans fuite ; hydrophones H1/H2 à 8 kHz, accélération/pression à 51,2 kHz. Les fichiers téléchargés ne doivent pas être présentés comme des acquisitions de bâtiments clients. [Source du dépôt](https://data.mendeley.com/datasets/tbrnp6vrnj/1).

API publique effectivement utilisée par le site, accessible anonymement :

```text
https://data.mendeley.com/public-api/datasets/tbrnp6vrnj/files?folder_id=root&version=1
https://data.mendeley.com/public-api/datasets/tbrnp6vrnj/folders/1
```

En-tête accepté : `Accept: application/vnd.mendeley-public-dataset.1+json`. Le HTML initial contient une liste vide pendant le chargement ; cela ne signifie pas que les données manquent. Les quatre fichiers de l'API ont le statut `COMPLETED`.

| Fichier officiel | Octets annoncés par l'API |
|---|---:|
| Accelerometer.zip | 459 278 311 |
| Dynamic Pressure Sensor.zip | 562 748 973 |
| Hydrophone.zip | 63 994 754 |
| Python code to convert RAW acoustic data to pandas DataFrame.py | 1 722 |
| Total | 1 086 023 760 |

La préparation ne télécharge que les deux derniers fichiers : **63 996 476 octets**, sans les modalités inutilisées. `HEAD` du lien hydrophone renvoie bien la taille annoncée et accepte les lectures HTTP Range.

Archive : [téléchargement officiel Hydrophone.zip](https://data.mendeley.com/public-files/datasets/tbrnp6vrnj/files/db8d1475-7cb4-4c60-b9e2-7d47a7d95971/file_downloaded).

```text
SHA-256 attendu (API, à revérifier après téléchargement intégral) :
d070b62306e1146072bc6929c64203af95e2a1df21a52ce6d93fe1a75395a38d
```

Convertisseur : [fichier officiel, lu mais jamais exécuté](https://data.mendeley.com/public-files/datasets/tbrnp6vrnj/files/dc9d459b-ec6d-4a64-ab69-d2bac7396a5c/file_downloaded).

```text
SHA-256 vérifié sur les 1 722 octets téléchargés :
3dd30f364804d4bfe0bea70ab057cbe5fba73083dcaf2dab3328208c63c0af02
```

## 2. Inventaire interne et réserves de format

Le répertoire central ZIP a été lu par plages HTTP, puis validé avec la grammaire exacte de `prepare_external_holdout.py` : **136 entrées, dont 122 RAW et 14 répertoires** ; somme des RAW décompressés : **150 702 480 octets**. Répertoire central : 18 325 octets, SHA-256 `7a05bec6883d598c1458c79f1618ca5ece419a77c2973f3da7ff84d20583a7fe`. Aucun CRC32 répété parmi les 122 RAW ; ce contrôle n'est pas un audit complet de similarité.

Grammaire effectivement observée :

```text
Hydrophone/{Branched|Looped}/{classe}/{T}_{L}_{condition}_{H1|H2}.raw
T = BR | LO
L = CC | GL | LC | NL | OL
condition = 0.18 LPS_N | 0.47 LPS_N | ND_NN | ND_N | Transient_NN | Transient_N
```

Sous-ensemble primaire : `2 topologies × 5 classes × 6 conditions × 2 hydrophones = 120` RAW, soit **96 fuite et 24 sans fuite**. Deux autres RAW se trouvent dans `Hydrophone/Background Noise/` : ils restent une annexe distincte, exclus du primaire binaire. La licence et la page ne justifient pas de les assimiler silencieusement à une canalisation sans fuite.

Le convertisseur des auteurs spécifie **mono, PCM signé 32 bits little-endian, 8 000 Hz**, puis conserve les **240 000 premiers échantillons**. Point important : les RAW ne font pas exactement 30 s. D'après leurs tailles, ils couvrent **34,9325 à 61,3075 s**. La troncature à 30 s suit donc une règle explicitement présente dans leur code, et non une fenêtre choisie d'après les scores.

Un RAW a été extrait et vérifié par le CRC ZIP, sans lire l'archive entière :

```text
Hydrophone/Branched/No-leak/BR_NL_0.18 LPS_N_H1.raw
1 117 840 octets ; 279 460 échantillons PCM32 ; 34,9325 s
SHA-256 : 0a15619f50bbc1b7172ff6f78a8c19ffc757638bd2890b50f67479a4d8ad3415
```

Cette extraction a transféré 520 913 octets avec le répertoire central. L'échantillon n'a pas servi à choisir une représentation ni un modèle. Aucun compte, archive complète, signal joué à l'oreille ou résultat de modèle n'a été nécessaire à ce contrôle.

### Groupes et doublons

- Un identifiant de **condition heuristique** est dérivé du nom sans le suffixe H1/H2 : les deux capteurs et leurs 30 fenêtres restent ensemble. Cela donne 60 groupes nommés, dont 48 fuite et 12 non-fuite.
- Aucun jour/session/répétition physique indépendant n'est documenté dans les noms. Les différentes conditions partagent le même banc et les mêmes dispositifs de fuite. **60 conditions ne prouvent pas 60 expériences indépendantes** ; des IC construits sur ces groupes devront être explicitement conditionnels à cette approximation. Une lecture conservatrice par topologie/type de fuite ne possède que 10 groupes, dont 2 sans fuite.
- Les deux capteurs peuvent avoir des longueurs brutes différentes, par exemple `BR_GL_ND_N`. Leur appartenance au même groupe n'autorise pas à prétendre un alignement temporel exact ni à les fusionner en stéréo.
- [L'autre dépôt Aghashahi, DOI 10.17632/xw44wv2g88.2](https://data.mendeley.com/datasets/xw44wv2g88/2), publie **les mêmes quatre SHA-256 et tailles d'archives/fichier**. Ce n'est pas un second holdout indépendant.
- Après téléchargement, le préparateur refuse les RAW exacts répétés et les signaux tronqués identiques. Avant confirmation V2, il faut encore vérifier les doublons avec les corpus déjà utilisés, sur contenus décodés et, si nécessaire, sous-séquences/copies transformées. Des extensions/DOI différents ne constituent pas une preuve d'absence de recouvrement.

## 3. Adaptateur implémenté, à geler avant inférence externe

Code : [prepare_external_holdout.py](../scripts/eval/prepare_external_holdout.py). Dépendance de données unique : NumPy, en plus de la bibliothèque standard. Pas de PyTorch, TimeNet, modèle, harness de métriques ou labels de développement importés.

Règle fixée dans le code : entier PCM32 little-endian → `float64 / 2**31`, puis premiers 240 000 points, puis offsets `0, 8000, …, 232000`. **Pas de clipping, gain choisi sur le corpus, padding, rééchantillonnage ou requantification PCM16.** La normalisation/représentation du modèle V2 reste son traitement canonique ultérieur, identique pour les fenêtres de développement et externes. Les valeurs obtenues ne sont pas une calibration physique en pascals.

Le préparateur vérifie taille/SHA des deux sources, grammaire complète des noms, absence de chemins dangereux/doublons/liens symboliques/chiffrement, longueurs RAW et CRC de chaque fichier. Il ne charge jamais les 64 Mo d'archive en une fois en mémoire. Les anciens dossiers, y compris incomplets et liens symboliques, sont refusés. La préparation est réalisée dans un temporaire propre ; le reçu `complete` arrive en dernier dans une nouvelle sortie réservée exclusivement.

Sortie proposée et implémentée :

```text
sources/Hydrophone.zip         # archive originale intégrale vérifiée
sources/author_converter.py   # preuve de décodage/crop ; jamais exécutée
arrays/recording_<sha256>.npy  # float64 LE, 240000 valeurs, nom opaque
inputs.csv                    # primaire, 3600 fenêtres prévues
targets.csv                   # labels/groupes, réservé à l'évaluateur
background_inputs.csv         # annexe distincte, 60 fenêtres prévues
background_targets.csv
recordings.json               # provenance brute complète, hors entrée modèle
metadata.json                 # licence, révision, règles et limites
checksums.json                # empreintes de tous les fichiers précédents
receipt.json                  # état final + empreinte du fichier checksums
```

`inputs.csv` contient exactement `clip_id,array_path,start_sample,n_samples,sample_rate`. Le scoreur futur ne doit recevoir que ces colonnes et les tableaux. `targets.csv` contient `clip_id,recording_id,condition_group_id,label` ; aucune de ces informations de cible/groupe n'entre dans le prompt. Les IDs opaques dérivent du DOI, chemin parent et offset par SHA-256. Toutes les 30 fenêtres d'un enregistrement sont présentes, sans tri par difficulté.

Pour l'évaluation future, charger le `.npy` avec `allow_pickle=False, mmap_mode="r"`, extraire la tranche de 8 000 points, puis utiliser l'entrée numérique canonique vérifiée V2. Ne pas fabriquer un WAV PCM16 pour la compatibilité. Le jeu reste **external_holdout_only**, entraînement et fitting de seuil interdits. Son manifeste ne remplace pas le `split_v2` historique.

### Commandes de reprise

Depuis le commit V2 archivé sur H100 ; le parent du dossier de sortie doit déjà exister. Remplacer `SHA40_DU_COMMIT_TRANSFERE` par la vraie révision complète, jamais par `unknown` ou un SHA de V1 :

```bash
python3 scripts/eval/prepare_external_holdout.py \
  --download \
  --output /home/hicham/pipe-v2/data/external/aghashahi-v1 \
  --code-revision SHA40_DU_COMMIT_TRANSFERE
```

Si l'archive et le convertisseur sont déjà téléchargés, utiliser `--archive /chemin/Hydrophone.zip --converter /chemin/author_converter.py` à la place de `--download`. La vérification cryptographique reste obligatoire. Budget disque indicatif : environ 64 Mo d'archive + 234,3 Mo de tableaux et quelques Mo de manifestes ; laisser de la marge. Aucun fichier source n'est supprimé ou modifié.

Vérification locale exécutée :

```bash
python3 -m unittest discover -s tests/eval -p test_external_holdout.py -v
```

**Sept tests réussis** : inventaire/classes/groupes ; PCM32 exact aux bornes ; archives dangereuses/incomplètes/dupliquées ; liens symboliques ; refus d'une sortie existante avant réseau ; SHA de code explicite ; préparation jouet reproductible avec séparation des labels, sources conservées et tous les checksums vérifiés. Aucun téléchargement complet ni préparation réelle exécuté durant cet audit.

## 4. Hong Kong : alternative terrain, pas primaire immédiat

[Source officielle v1](https://data.mendeley.com/datasets/hkn8mxcjyz/1), DOI `10.17632/hkn8mxcjyz.1`, publié le 7 février 2022, CC BY 4.0. Le dépôt annonce environ 90 sites enterrés, des signaux lors d'une fuite signalée et après réparation. Les fichiers ne comportent pas à eux seuls une table garantissant l'identité des sites/sessions entre conditions. [Description source](https://data.mendeley.com/datasets/hkn8mxcjyz/1).

L'API `/public-api/datasets/hkn8mxcjyz/folders/1`, puis `/files?folder_id=<id>&version=1`, donne l'inventaire complet :

| Modalité | Fichiers | Répartition publiée dans les dossiers | Octets |
|---|---:|---|---:|
| Hydrophones | 80 WAV | 40 Leak / 40 No-Leak | 10 712 960 |
| MEMS Accelerometers | 20 XLSX | 10 / 10 | 239 076 468 |
| Noise Loggers | 60 WAV | 30 / 30, métal/non-métal | 7 449 360 |
| Total | 160 | — | 257 238 788 |

Les 80 en-têtes hydrophone ont été lus par plages de 44 octets : mono PCM16, durée 10 s ; 28 fuite à 4 096 Hz, 12 fuite et 40 non-fuite à **8 092 Hz exactement**. Trois fichiers complets représentatifs ont été téléchargés, leurs SHA vérifiés contre l'API et leurs formats confirmés :

| Exemple officiel | ID fichier | Fréquence | SHA-256 vérifié |
|---|---|---:|---|
| Leak/1.1.01.0330.wav | ec07ddef-bf1b-4c1a-8d27-576cd318dfdd | 4 096 | d7a58812790d3e1cbad146abe45fc3ec88914175c117804c30c5ab9f2232f5bd |
| Leak/3.1.01.0330.wav | 47b668bc-9b47-4e8d-84ac-cf159636b3af | 8 092 | b0a8df7d1eb07a7227794eb1769e71f56579286d5d507d94a89ecc169cd65b21 |
| No-Leak/2.1.01.0330.wav | 167beffd-1778-4928-8762-edfed2730ed3 | 8 092 | 2d7b4dc86c68056d9d657ddf0f56c134323d54bd7463ab948b1a6636d41d181c |

Téléchargement d'un exemple : `https://data.mendeley.com/public-files/datasets/hkn8mxcjyz/files/<ID>/file_downloaded`.

L'inventaire hydrophone canonique (liste triée par dossier/nom, clés `folder,name,id,bytes,sha256`, JSON UTF-8 trié compact) a l'empreinte `9614b4cc522607c6178dbf761e0762bb3a402046c17df003bbc6fd0a761c83f6`. Aucun SHA publié répété dans ce sous-ensemble. En revanche, deux noms de logger ont exactement le même SHA : `12846-20200902-1145.wav` et `12865_20200902_1145.wav`, `7ddcbabd4002337714ee0a8574f2aff3d46a164f7f550dc0e37b8ede728998ee`. Ne pas compter naïvement les fichiers comme observations indépendantes.

Les préfixes hydrophone `1.1`…`1.7`, `3.1`…`3.3`, `2.1`…`2.8` donnent 18 groupes de nommage possibles, pas 18 sites prouvés. Les derniers composants ressemblent à des heures répétées ; leur sens exact et la correspondance avant/après réparation sont à confirmer auprès des auteurs avant une interprétation indépendante par site.

Une adaptation éventuelle devrait décoder en flottants, rééchantillonner avec filtre anti-repliement et paramètres fixes, puis découper à la seconde, sans réécriture PCM16. Mais le sous-ensemble 4 096 Hz est entièrement positif : **le rééchantillonnage n'efface pas le confondant de bande passante/acquisition**. Ne pas choisir le sous-ensemble, filtre ou règle de groupe d'après les performances V2. Aucune conversion ni préparation Hong Kong n'est implémentée ici.

## 5. Gates restants et continuité

1. Télécharger/préparer Aghashahi avec le code gelé, revérifier les artefacts et effectuer l'audit de recouvrement avec le développement sans score. Figer le reçu/manifeste et les règles d'évaluation avant toute inférence externe.
2. Garder exclusivement train/validation historiques pour sélectionner V2 et son seuil. Appliquer ensuite ce modèle/seuil inchangés au nouveau jeu, conserver les résultats même s'ils sont défavorables ; aucun nouveau candidat déclenché par cette confirmation.
3. Présenter séparément résultat par fenêtre/enregistrement/groupe heuristique et ses limites, sans baptiser les fenêtres indépendantes ou transformer un banc unique en preuve terrain.
4. Pour le continu, obtenir une vraie chronologie acoustique, interruptions et périodes sans fuite, avec débuts/fins d'événements, sessions/sites/capteurs et protocole d'annotation. La variation de débit à t≈20 s annoncée dans Aghashahi **n'est pas un début de fuite annoté**. Ni concaténer les clips ni accélérer un replay ne peut valider un délai de détection réel ou des fausses alertes par jour.

Le téléchargement/prétraitement Aghashahi ne nécessite actuellement aucune action de compte d'Icham. La collecte continue, les tolérances métier et, si Hong Kong doit devenir une confirmation par site, la clarification des identifiants auprès des auteurs, restent des dépendances distinctes. Aucun message externe n'a été envoyé.
