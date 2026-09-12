# HACKATHON_OPTIONS — three candidate directions

> 2026-09-12. **Decision taken — see the last section.** The three options below are kept as
> written *before* the choice, so the reasoning can be re-read against what actually happened.
> The risk lines describing the dataset as unaudited are historical: the audit has since been
> done ([`DATASET_AUDIT.md`](DATASET_AUDIT.md)).
>
> ⚠️ **One parameter is missing and it changes every line below: the deadline.** The official
> PDF and the Discord brief give no date and no hour. The phrase *"credits to keep building
> **after Sunday**"* suggests the event runs into Sunday, but **nobody has confirmed it.**
> The time-to-demo figures assume today only; if Sunday is in scope, option B becomes
> substantially safer.

---

## Option A — existing TimeNet connector

Use one of the three connectors shipped with TimeNet — `physionet/ecg_qa_cot`,
`physionet/sleep_edfx`, or `chengsenwang/tsqa` — and put all the effort into training,
evaluation and demo.

| | |
|---|---|
| **Time to first demo** | **shortest.** The connector exists; data acquisition is a command, not a project. |
| **GPU need** | Yes, for fine-tuning. Data prep and eval run on the Mac. |
| **Risk** | **Low technically, high on differentiation.** These are the obvious datasets; other teams will use them too. And on ECG specifically, **OpenTSLM already trains on ECG-QA** — checkpoint exposure must be audited before any novelty claim, and the jury *is* the OpenTSLM team. |
| **Jury criteria** | *useful problem* ⚠️ generic · *data prep* ⚠️ mostly done for us · *TSLM training* ✅ · *rigorous evaluation* ✅ (PTB-XL ships official per-patient splits, folds 1-8/9/10) · *clear demo* ✅ · **bonus connector** ❌ nothing new to contribute |
| **Real link to LeakLess** | **None.** Medical or generic time-series. Nothing transfers to the product. |
| **Provable by tonight** | ✅ Realistically yes: connector run, grouped split, baseline, a first fine-tune, a working demo. |

---

## Option B — LeakLess-adjacent, software-only, Zenodo acoustic dataset

Execute the proposal Hicham already wrote: `02 Recherche/PIPE - proposition ML et démo.md`.
Acoustic leak/no-leak classification with a TSLM, on the Zenodo CC BY 4.0 dataset, plus a
custom TimeNet connector for audio.

| | |
|---|---|
| **Time to first demo** | **longest.** Three `.rar` archives to download (two never opened), audio→time-series conversion to write, a TimeNet connector to build, text annotations to generate. |
| **GPU need** | Yes, for fine-tuning. The connector and the audit run on the Mac. |
| **Risk** | **Highest, and concentrated in data.** 386 + 114 clips unaudited · structural leakage (`_1`/`_2` suffixes, hydrophone *and* logger under identical conditions) · `.rar` download and extraction time · the proposal itself admits *"ML-based acoustic leak detection already exists; no established algorithmic novelty"*. |
| **Jury criteria** | *useful problem* ✅✅ a real user with a real decision · *data prep* ✅✅ the hard part is genuinely hard and visibly done · *TSLM training* ✅ · *rigorous evaluation* ✅✅✅ **the leakage audit becomes a headline result, not a footnote** · *clear demo* ✅✅ audio the jury can hear, spectrogram, live noise injection with a dial they turn · **bonus connector** ✅✅ audio→time-series connector, and **the TimeNet registry is empty, so it would be the first** · **bonus agentic pipeline** ✅ the dataset-sourcing reasoning already exists in prose and can be automated |
| **Real link to LeakLess** | **Strongest, and it survives the weekend.** The audio→time-series connector, the group-clean split protocol, the hydrophone-vs-logger cross-device generalisation proxy, and the environmental-noise robustness test are all directly reusable by the product. |
| **Provable by tonight** | ⚠️ **Partially.** Realistic tonight: archive audit, group reconstruction, naive-vs-grouped split gap, baseline. **The fine-tuned TSLM and the polished demo are tight** — unless the event really runs to Sunday. |

---

## Option C — hybrid: existing connector for the pipeline, LeakLess as the narrative

Build and validate the full pipeline end-to-end on an existing connector (fast, safe), then
show the acoustic leak case as the motivating application — with the connector written but
the TSLM trained on the safe dataset.

| | |
|---|---|
| **Time to first demo** | **short**, close to A. |
| **GPU need** | Yes, once, on the safe dataset. |
| **Risk** | **Lowest overall**, but one specific danger: **if the narrative promises more than the model does, the jury will catch it.** The gap between "here is what we trained" and "here is what it's for" must be stated explicitly, not glossed. |
| **Jury criteria** | *useful problem* ✅ · *data prep* ✅ · *TSLM training* ✅ · *rigorous evaluation* ✅ · *clear demo* ✅ · **bonus connector** ✅ if the audio connector is actually shipped and documented |
| **Real link to LeakLess** | **Medium.** The connector is reusable; the trained model is not. |
| **Provable by tonight** | ✅ Yes, with a working pipeline plus a documented connector. |

---

## Side-by-side

| | **A — existing connector** | **B — acoustic, software-only** | **C — hybrid** |
|---|---|---|---|
| Time to first demo | 🟢 shortest | 🔴 longest | 🟢 short |
| GPU | required | required | required |
| Technical risk | 🟢 low | 🔴 high (data) | 🟢 low |
| Differentiation | 🔴 weak | 🟢 strong | 🟠 medium |
| *rigorous evaluation* | 🟢 | 🟢🟢🟢 | 🟢 |
| Connector bonus | 🔴 none | 🟢🟢 first in an empty registry | 🟢 |
| Real LeakLess value | 🔴 none | 🟢🟢 durable | 🟠 partial |
| Demo the jury can touch | 🟠 | 🟢🟢 audible + interactive | 🟠 |
| Safe if today only | 🟢 | 🔴 | 🟢 |
| Best if Sunday counts | 🟠 | 🟢🟢 | 🟠 |

---

## What is true regardless of the option

These four actions block nothing and unblock everything. They should be running in parallel
with the decision, not after it.

1. **Confirm the deadline, pitch length and submission format** with the organisers — it is
   the input that decides between B and the rest.
2. ~~**Activate the Nebius voucher**~~ — ✅ **done**: H100 80GB HBM3, CUDA 13.0, GPU idle.
3. **Check whether Llama / Gemma are gated** on HuggingFace; line up a non-gated fallback.
4. **Agree who does what with Hicham** — Nevil appears nowhere in either repository.

Also true regardless: **training happens on Nebius, not on the Mac.** `bitsandbytes`,
`flash-attn`, `xformers` and `deepspeed` do not install with CUDA on Apple Silicon. The Mac
does data preparation, evaluation and the demo UI.

---

## DÉCISION — 2026-09-12

> ### ✅ **Option B retenue — LeakLess software-only**, validée par Hicham *sous condition* que
> ### l'audit confirme des étiquettes exploitables et un split défendable entre groupes.

**La condition est levée** : [`DATASET_AUDIT.md`](DATASET_AUDIT.md) §13 conclut à un **GO
conditionnel** — 1000 clips vérifiés, 1000/1000 noms parsés, 306 groupes sans classe mêlée,
split 60/20/20 réalisable sans couper un seul groupe.

Cadre acté avec l'équipe :

| | |
|---|---|
| TimeNet | utilisé |
| TSLM | entraînement réel |
| Descriptions textuelles | ancrées **uniquement** sur des propriétés de signal mesurables |
| Comparaison | TSLM contre baseline, **exactement les mêmes splits** |
| Présentation du dataset | mesures sur un **site d'entraînement expérimental** |
| Revendication terrain / client | **aucune** |

Deux conditions issues de l'audit s'ajoutent et ne sont pas négociables : normalisation
d'amplitude + contrôle « RMS seul » publié (risque L9), et aucune métadonnée de nom de fichier
en entrée du modèle (risque L1). Détail : `EVAL_PROTOCOL.md` §7bis.

**Compute** : Nebius opérationnel — H100 80 Go HBM3, CUDA 13.0, `nvidia-smi` vérifié, GPU au
repos. La ligne « GPU sur le chemin critique » des options ci-dessus n'est plus bloquante.
