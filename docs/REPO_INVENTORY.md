# REPO_INVENTORY — les deux dépôts de Hicham

> 2026-09-12 · CC HQ. Inventaire en **lecture seule stricte** : aucun fichier modifié, aucun commit, aucun changement de branche, aucun script exécuté.
> Clonés dans `~/dev/sandbox/ehl-zurich/` — dossier neutre, hors de tout repo existant.
> Aucune permission n'a manqué : les deux dépôts sont publics et se clonent sans identifiants.

---

## Vue d'ensemble

| | `ehl-hackathon-zurich-vault` | `ehl-hackathon-zurich` |
|---|---|---|
| Nature | Vault Obsidian **de référence** | Workspace + **miroir** du vault |
| Contenu | 17 notes `.md` (619 lignes) + le PDF | 13 notes (miroir) + config outillage |
| Code | **aucun** | **aucun** |
| Commits | 5, tous du 12/09, tous `docs:` | 6, tous du 12/09, tous `docs:`/`chore:` |
| Branches | `main` seule | `main` seule |

**Constat central : il n'y a pas une ligne de code dans les deux dépôts réunis.** Zéro `.py`, zéro notebook, zéro dataset, zéro config d'entraînement, zéro CI. Les deux disent explicitement : *« Aucun code produit ni test fonctionnel du produit pour le moment »* (`01 Projet/Passation.md`) et *« Aucune expérience exécutée pour l'instant »* (`03 Build/Expériences.md`).

**⚠ Piège de coordination** : le `vault/` présent dans `ehl-hackathon-zurich` est un **miroir sans synchronisation automatique** (`AGENTS.md:3`, `Continuité du projet.md:42`). Le dépôt qui fait foi est `ehl-hackathon-zurich-vault`. Travailler sur le mauvais garantit des conflits. `AGENTS.md` avertit : *« Ne pas écraser les contributions des collaborateurs par une copie du dépôt code. »*

---

## 1. Ce qui est déjà prêt — et c'est beaucoup plus qu'il n'y paraît

### ⭐ La pièce maîtresse : `02 Recherche/PIPE - proposition ML et démo.md` (79 lignes)

C'est le dernier commit du vault (`15da3c8`, *« propose evidence-first acoustic TSLM demo with verified sources »*), et **c'est une spec de projet complète, prête à exécuter** : problème, dataset sourcé, pipeline ML, protocole d'évaluation anti-fuite, scénario de démo en 5 étapes, ablations, et alternatives explicitement écartées avec motifs.

**Le sujet qu'elle propose : détection acoustique de fuites d'eau par TSLM.**

### ⭐ Un dataset acoustique réel, ouvert, à moitié audité

**Zenodo `https://zenodo.org/records/18631450` — CC BY 4.0**

| | contenu |
|---|---|
| 500 clips | **fuite** — archive téléchargée et listée par Hicham |
| 386 clips | **sans fuite** — non auditée |
| 114 clips | **bruit environnemental** — non auditée |
| Format | WAV de 1 seconde |
| Métadonnées | encodées dans les noms : **matériau · région · pression · vitesse d'écoulement · appareil (hydrophone vs logger)** |

Les trois archives sont des **`.rar`** : prévoir `unrar`/`7z` et le temps de téléchargement.

### Autres acquis réutilisables

- **Le brief officiel décodé et fidèle** — vérifié contre le PDF slide par slide, les 5 livrables et les 5+2 critères concordent mot pour mot.
- **Trois pistes de repli sourcées** (`02 Recherche/Idées.md`) : UCI hydraulic 447, PTB-XL 1.0.3 (splits officiels par patient, folds 1-8/9/10), Telemanom SMAP/MSL.
- **Un protocole anti-fuite écrit** — la section « Condition scientifique impérative » de PIPE interdit le split aléatoire clip par clip et impose le groupage par événement/appareil. C'est **exactement** le critère jury.
- **Trois templates Obsidian** (`99 Modèles/`), dont un modèle d'expérience qui exige le critère de réussite **avant** le test.
- **`.gitignore` correct sur les secrets** dans les deux dépôts (`.env`, `*.key`, `*.pem`).

---

## 2. Ce qui manque

**0 livrable sur 6 est coché.** Rien n'a été construit.

| Manque | Preuve |
|---|---|
| Stack technique | `03 Build/Architecture.md` : « Statut : non décidée » + 7 rubriques vides |
| Décisions produit | `03 Build/Décisions.md` ne contient que de l'organisation de dépôt |
| Scénario de démo | `04 Démo/Démo et pitch.md` : « scénario à définir », 6 cases vides |
| Accès TimeNet | non vérifié |
| **Voucher Nebius** | **non activé** — « Aucun lancement payant effectué » |
| Deadline · fuseau · plateforme de soumission · durée du pitch | 5 cases vides, « À confirmer auprès des organisateurs » |
| Archives « no leak » et « bruit » | non auditées avant split |
| **Nevil dans l'équipe** | **son nom n'apparaît nulle part dans les deux dépôts** |

---

## 3. Dépendances

**Aucun manifeste de paquet nulle part.** Rien n'est épinglé, rien n'est installé. Tout ci-dessous est **déclaré comme intention**, pas comme état.

| Dépendance | Note |
|---|---|
| **TimeNet** | ingestion/standardisation. ⚠️ **ne fait PAS l'entraînement** (`Idées.md:46`) |
| **OpenTSLM** | checkpoints + scripts de fine-tuning. Variante visée : **OpenTSLM-SP** |
| **Modèle de base** | Llama 3.2 1B ou Gemma — ⚠️ **accès potentiellement gated sur HuggingFace** |
| Baseline | SVM / Random Forest sur features spectrales, éventuellement petit CNN |
| Audio | WAV → énergie par bandes de fréquence (aucune lib nommée) |
| `unrar` / `7z` | les 3 archives Zenodo sont des `.rar` |
| Compte **Nebius** | voucher 1 000 $ non activé |
| Entire 0.7.7 | outillage de **Hicham**, pas une dépendance du projet |

---

## 4. Risques, par capacité à faire perdre la journée

1. **La fuite de données est structurelle dans le dataset acoustique.** PIPE le documente : suffixes `_1`/`_2` nombreux, captures hydrophone et logger sous mêmes conditions ⇒ dépendance évidente entre clips, et *« les noms ne garantissent pas une reconstruction parfaite des sessions »*. Un split naïf donnera un score flatteur **indéfendable** devant ce jury précis. **C'est aussi la plus grosse opportunité — voir §6-B.**
2. **Deadline inconnue.** Ni dans les notes, ni dans les 22 pages du PDF. Le budget temps est fictif tant que ce n'est pas confirmé.
3. **Voucher Nebius non activé.** Chemin critique : sans GPU, pas de TSLM entraîné.
4. **Modèles de base gated.** Une demande d'accès HF en attente bloque le fine-tuning. Prévoir un repli non-gated.
5. **Archives `.rar` lourdes**, deux sur trois jamais ouvertes. Temps réseau + extraction + audit incompressible.
6. **Confusion simulé / réel.** LeakDB est **simulé** et ne satisfait pas l'exigence « real inputs ». Le présenter comme du terrain serait disqualifiant.
7. **Nouveauté faible.** PIPE l'admet : *« La détection acoustique par ML existe déjà ; aucune nouveauté algorithmique établie. »* Et sur ECG : OpenTSLM traite déjà ECG-QA ⇒ le checkpoint a pu voir les données.
8. **Deux copies du vault sans synchro** + hackathon + plusieurs mains = conflits garantis.
9. **Nevil absent des dépôts** ⇒ risque de travail en double avec Hicham dès la première heure.
10. **Apple Silicon** : rien à exécuter ici, mais dès qu'on clonera OpenTSLM, `bitsandbytes` / `flash-attn` / `xformers` / `deepspeed` ne s'installent pas en CUDA sur arm64. ⇒ **sur le Mac : préparation de données, évaluation et démo uniquement. L'entraînement sur Nebius.**

---

## 5. ⚠️ Découverte qui change le calendrier

Slide 20 du PDF : *« The most promising teams get follow-on credits **to keep building after Sunday** »* — et le message Discord dit *« train on Nebius GPUs **all weekend** »*.

⇒ **Le hackathon court probablement jusqu'à dimanche, pas seulement aujourd'hui.** Ça double le temps disponible et change complètement l'arbitrage entre voie sûre et voie ambitieuse. **À confirmer d'urgence auprès de l'orga** — c'est la question la plus rentable de la journée.

---

## 6. Idées exploitables, reliées aux critères du jury

**A. Exécuter PIPE tel qu'écrit.** La conception est faite, il reste l'exécution. Couvre les 5 critères. Coût de démarrage quasi nul.

**B. ⭐ Faire du « leakage audit » un livrable en soi.** Groupage par événement/matériau/appareil dérivé des noms de fichiers, déduplication, puis **comparaison chiffrée split naïf contre split groupé — l'écart de score EST le résultat**. Frappe directement *rigorous evaluation*, le critère le plus discriminant, et différencie de toutes les équipes qui montreront 0,97 d'accuracy sans savoir pourquoi.

**C. Un connecteur TimeNet « audio → séries temporelles » réutilisable.** WAV → énergie par bandes + métadonnées de provenance, packagé proprement. **Bonus explicite du jury**, et le registre TimeNet est vide, donc ce serait le premier.

**D. L'ablation « la temporalité sert-elle vraiment ? »** Features agrégées contre séries temporelles + permutation temporelle. Répond frontalement au thème « Give AI a Sense of Time ». PIPE le cadre honnêtement : *« une permutation temporelle peut ne pas affecter un signal stationnaire ; c'est une conclusion possible, pas un test censé réussir. »*

**E. Mode robustesse au bruit, en live.** Les 114 clips de bruit environnemental servent exactement à ça : intensité réglable **par le jury**, ré-inférence en direct, abstention souhaitée mais *« jamais scriptée »*. Sert *clear demonstration* et la crédibilité.

**F. Abstention calibrée + courbe précision/couverture.** Un technicien a besoin de savoir quand ne pas faire confiance.

**G. Pipeline agentique de sourcing de datasets.** Le raisonnement est **déjà fait en prose** dans `Idées.md` et les sections « alternatives écartées » de PIPE (LeakDB rejeté car simulé ; Intra-Domestic rejeté car pas d'ID logement donc pas de split par usager ; Telemanom rejeté car fuite de preprocessing dans son propre README). L'automatiser en « fetch metadata → check licence → check group-ability → verdict » et le rejouer live = **second bonus jury**, à partir de travail déjà payé.

**H. Évaluer la classification séparément du texte.** Anti-pattern classique : un beau paragraphe qui masque un classifieur médiocre.

**I. Ne pas faire du TSLM un simple rédacteur** qui reçoit la réponse d'un autre classifieur. Le jury *est* l'équipe OpenTSLM — il verra la triche architecturale immédiatement.

---

## 7. Liens avec la track Temporal AI / TimeNet / TSLM

Le vault est **déjà entièrement aligné** : il reformule fidèlement le parcours en 4 étapes et les critères. PIPE pose la conformité comme non négociable : *« TimeNet et entraînement TSLM restent obligatoires. Une solution prompt-only est un prototype de secours, pas un livrable conforme. »* — cadrage juste, beaucoup d'équipes livreront du prompt-only.

Choix déjà arrêtés dans la note : **OpenTSLM-SP**, petit modèle de base, audio traité **comme série numérique** (pas comme transcription ni comme image), cibles textuelles générées depuis labels et mesures du train et **déclarées comme non expertes** — ce que le PDF autorise explicitement (« *you might consider generating text annotations with LLMs* »).

**Ce que le PDF apporte et que les notes n'ont pas intégré** (pages images non lues côté Hicham) : le jury *est* l'équipe OpenTSLM (ICML, ETH + Stanford + Google) ; le format de sortie canonique est **State summary → Recommended action → Expected outcome** avec raisonnement pas à pas ; et l'argument de vente de la techno est l'absence de mur de contexte face à un LLM texte qui sature en ~25 min de signal.

---

## 8. Liens avec LeakLess

**Le recoupement est direct, et il n'est pas de moi : Hicham a lui-même orienté le projet vers la détection de fuites d'eau, en acoustique.** Trois notes y sont consacrées.

Ce qui sert LeakLess au-delà du hackathon :

1. **Le connecteur audio → séries temporelles** (idée C) est réutilisable tel quel.
2. **Le protocole anti-fuite groupé par appareil/événement** (idée B) est exactement celui dont LeakLess aura besoin le jour où il y aura des mesures multi-capteurs multi-sites.
3. **La distinction hydrophone / logger** dans les noms Zenodo donne un **proxy de généralisation inter-appareils** — question centrale pour tout produit capteur.
4. **Les 114 clips de bruit environnemental** sont l'analogue labo du bruit de plomberie et de rue qu'un capteur réel rencontre.
5. L'avertissement *« ne pas prétendre connaître les unités physiques de l'amplitude WAV sans calibration »* est une leçon hardware directement transposable.

**⚠️ Garde-fous déjà posés par le vault, à respecter :**
- **Aucun hardware n'est supposé validé.** C'est explicite : *« Le capteur personnel et les données clients ne sont pas des prérequis du MVP »*, *« Démo par rejeu de données réservées, pas par installation de capteurs. »*
- **Ne pas confondre** le banc hydraulique UCI ni LeakDB simulé avec une preuve de détection sur canalisation réelle.
- **Ne pas interpréter** les m/s des noms de fichiers comme un débit volumique.
- Périmètre honnête déjà borné : **pas de localisation, pas de cause physique inventée, pas de prédiction de rupture, pas de preuve d'absence de fuite.**

**Précision d'attribution** : le mot « LeakLess » **n'apparaît nulle part** dans les deux dépôts. Hicham parle d'« un membre du groupe qui possède une entreprise du secteur ». Le rapprochement est le nôtre, pas le sien.

---

## 9. Les quatre actions qui ne dépendent d'aucune décision de sujet

1. **Confirmer deadline, durée du pitch et format de soumission** auprès de l'orga — débloque le budget temps, et tranche la question « jusqu'à dimanche ? ».
2. **Activer le voucher Nebius** — chemin critique.
3. **Demander l'accès au modèle de base** (Llama/Gemma) au cas où il serait gated, ou choisir un repli non-gated.
4. **Inscrire Nevil dans l'équipe et se répartir les rôles avec Hicham** — sinon double travail dès la première heure.
