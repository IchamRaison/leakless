# HACKATHON_BRIEF — EHL Zurich · Temporal AI Challenge

> Rédigé le 2026-09-12 par CC HQ. **Source unique vérifiée à ce stade : le message de brief de Robert Jakob dans `#eth-agentic-systems-lab` (Discord), posté à 11:59 le 12/09/2026.**
> Le PDF `Aionic_Temporal_AI_Hackathon.pdf` (4,87 Mo) joint à ce message **n'a pas encore pu être lu** — voir §6.
> Tout ce qui n'est pas dans ce document n'a pas été vérifié. Les trous sont nommés, pas comblés.

---

## 1. Identité de la track

**TEMPORAL AI CHALLENGE — Give AI a Sense of Time**
Porté par **Agentic Systems Lab × Aionic Labs × Nebius**.

> *« LLMs read text. The real world speaks in signals: vitals, sensors, markets, grids. This weekend, you'll build AI that understands how the world changes over time. »*

**Mission** : utiliser **TimeNet** (Aionic) pour préparer de la donnée temporelle open-source, puis entraîner un **TSLM** (Time-Series Language Model) qui relie signaux temporels et langage pour une tâche utile.

Exemples donnés par l'énoncé : l'état d'un patient qui évolue, une machine qui commence à tomber en panne, une demande énergétique qui se déplace — *« anywhere people act on signals »*.

---

## 2. Les 4 étapes imposées — « From Problem to Proof »

| # | Étape | Contenu |
|---|---|---|
| 1 | **PROBLEM + DATA** | Choisir un utilisateur cible et un problème, sourcer de la donnée ouverte. *Les pipelines agentiques de chasse aux datasets sont encouragés.* |
| 2 | **CONNECT** | Construire ou utiliser un **connecteur TimeNet** pour signaux, métadonnées et annotations. *Les pipelines d'ingestion réutilisables sont bienvenus.* |
| 3 | **TRAIN + EVALUATE** | Entraîner ou fine-tuner un TSLM, **comparer à une baseline sur du held-out** — ⚠️ *no leakage across time / subjects / devices!* |
| 4 | **DEMONSTRATE** | Inputs réels, sorties du modèle, et en quoi ça aide l'utilisateur. **Montrer les preuves ET les limitations.** |

---

## 3. Livrables imposés — « You Submit »

- Working demo
- Code + training config
- Checkpoint ou adapter
- Dataset docs
- Short eval w/ baseline

**Puis une démo live au jury** : problème, approche, résultats, apprentissages.

---

## 4. Critères de jugement — « What Wins »

> *A useful problem · thoughtful data prep · adequate TSLM training · **rigorous evaluation** · a clear demo.*

**Bonus** : *« for reusable TimeNet connectors & agentic data pipelines others can build on. »*

**Lecture stratégique** — l'évaluation rigoureuse est un critère de victoire **explicite**, et le no-leakage est nommé dans l'énoncé lui-même. C'est l'inverse du hackathon de Paris (Track 3), où le rubric notait couverture / créativité / fun et pas l'honnêteté méthodologique.

---

## 5. Compute

- **Nebius : 1 000 $ de voucher compute par équipe**, GPU accessibles tout le week-end.
- *« Most promising teams get follow-on credits to keep building after Sunday. »*

⚠️ **La procédure de récupération du voucher n'est publiée nulle part sur le Discord.** Recherche « nebius » sur l'ensemble du serveur : **1 seul résultat**, ce message de brief. Aucun channel d'annonces, de support ou d'orga ne décrit la marche à suivre.

→ **Action ouverte** : demander dans `#eth-agentic-systems-lab` (le brief invite explicitement : *« Questions? Just drop them here »*) ou directement à l'orga sur place.

**Règle de sécurité actée pour ce hackathon** : aucune clé API, aucun token, aucun identifiant n'est écrit en clair dans un fichier. Nevil les saisit lui-même. Vérification du GPU une fois l'accès obtenu, par `nvidia-smi` ou équivalent.

---

## 5bis. Apports du PDF officiel (22 pages, LU intégralement)

> Fichier trouvé en local dans le repo de Hicham : `ehl-hackathon-zurich/vault/assets/Aionic_Temporal_AI_Hackathon.pdf`.

### a) « Ideal Data Profile » — p.13 — **le point le plus libérateur du document**

> • *A dataset with time-series **and text reasoning over it / or describing it***
> • *In some cases, you might consider **generating text annotations with LLMs** (as in OpenTSLM paper)*

⇒ **Générer les annotations textuelles avec un LLM est explicitement autorisé.** Un dataset sans texte natif reste donc éligible : on peut construire la couche langage. Cela ouvre en grand le choix du dataset, au lieu de le restreindre aux rares corpus signal+texte.

### b) Les trois domaines cibles — p.12

| Domaine | Formulation officielle |
|---|---|
| **HEALTH** | *A patient's condition changes* |
| **INDUSTRY** | *A machine begins to fail* |
| **ENERGY** | *Demand shifts over time* |

Triptyque affiché : **INTERPRET → ANTICIPATE → ACT**.

### c) Le format de sortie canonique d'un TSLM — p.8 (exemple moteur d'avion)

Entrée : multi-capteurs (température moteur, paramètres huile, électrique, carburant, RPM) + un prompt en 3 questions (*analyser les anomalies · recommander des actions de maintenance · prédire le résultat attendu*).

Sortie, en trois blocs avec **raisonnement pas à pas** :
1. **STATE SUMMARY** — ce que font les signaux
2. **RECOMMENDED ACTION** — quoi faire
3. **EXPECTED OUTCOME** — ce qui devrait se passer ensuite

**Argument de vente de la techno, même page** : un LLM texte (GPT 5.5, 1 M de contexte) sature en **~25 min** à ~200 K tokens / 5 min de signal ; OpenTSLM streame du signal long et multivarié nativement — *« no context wall »*.

### d) Ressources fournies — p.14

*TimeNet access · Access to training compute · **OpenTSLM as a reference** · Technical guidance from Aionic Labs + ASL → via Discord.*

**Présentation finale** : *Live demo to the jury — problem, approach, results, and key learnings.*

**Bonus, formulation complète** : *« Well-chosen datasets and reusable TimeNet connectors that others can build on »* + *« Agentic pipelines for automated search, assessment, selection, retrieval, or validation of datasets are a plus »*.

### e) Nebius — p.19-21

*« $1000 compute voucher per team …and more for the winners »*, et p.20 : *« This weekend: you can train on Nebius GPUs with **sponsored vouchers**. The most promising teams get follow-on credits to keep building after Sunday. »*
Programmes hors hackathon : `nebius.com/startups`, `nebius.com/nebius-research-grants`, `dev.nebius.com/builders`, `nebius.com/fellows`.
Contact Zurich nommé dans le deck : **Jennifer Werthwein** (le bureau de Zurich ouvre mi-septembre).

Support Discord officiel : `discord.com/invite/qXsAt4BHV`

---

## 6. Ce qui N'EST TOUJOURS PAS connu — **après lecture intégrale du PDF**

| Inconnu | Statut |
|---|---|
| **Procédure d'obtention du voucher Nebius** | ❌ **absente du PDF comme du Discord.** Le deck annonce les vouchers mais ne dit nulle part comment les réclamer. → à demander à l'orga / sur Discord |
| **Deadlines précises** (fin du hack, heure de soumission, heure des démos) | ❌ **aucune date ni heure dans les 22 pages** |
| **Format exact de soumission** (plateforme, dépôt, template) | ❌ non spécifié — seule la liste des livrables est donnée |
| **Durée du pitch / format du passage jury** | ❌ non spécifié — on sait seulement que c'est une démo live |

⚠️ **Ces quatre trous ne se combleront pas par la lecture : ils demandent une question à l'orga.**

---

## 7. Ressources officielles

| Ressource | Lien |
|---|---|
| TimeNet docs | https://docs.timenet.ai/ |
| TimeNet GitHub | https://github.com/OpenTSLM/TimeNet |
| OpenTSLM paper | https://arxiv.org/abs/2510.02410 |
| Aionic Labs | https://aioniclabs.ai/ |
| Agentic Systems Lab | https://agenticsystemslab.org/ |

---

## 8. État de l'environnement local (vérifié)

Dossier de travail neutre : `~/dev/sandbox/ehl-zurich/` — **hors de tout repo existant**, aucun lien avec LeakLess ou lab-hardware.

| Élément | État |
|---|---|
| `TimeNet/` cloné (OpenTSLM) | ✅ |
| `uv` mis à jour 0.10.9 → **0.12.13** (TimeNet exige ≥ 0.10.10) | ✅ |
| `uv sync` | ✅ exit 0 |
| CLI `timenet` (`list` / `search` / `info` / `download` / `cache`) | ✅ opérationnel |
| CLI `timenet-build` (`build`) | ✅ opérationnel |
| Python | 3.12.14 |
| torch + MPS (Mac) | 2.12.0, MPS disponible |

**Constat important** : le **registre TimeNet est vide** — en local comme sur le registre hébergé (`timenet://`). TimeNet est en pré-release et ne livre aucun dataset pré-publié. Le registre se remplit en exécutant un connecteur via `timenet-build build`.

**Connecteurs livrés avec le repo** (les trois relient signal et langage) :

| Connecteur | Nature |
|---|---|
| `physionet/ecg_qa_cot` | ECG + question-réponse + chain-of-thought |
| `physionet/sleep_edfx` | Sleep-EDF Expanded — polysomnographie / EEG de sommeil |
| `chengsenwang/tsqa` | TSQA — question-réponse sur séries temporelles |

⚠️ **Aucun de ces connecteurs n'a été exécuté.** Aucune donnée n'a été téléchargée. Aucun entraînement n'a été lancé. Le choix du sujet n'est pas fait.
