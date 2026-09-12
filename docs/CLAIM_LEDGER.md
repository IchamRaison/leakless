# CLAIM_LEDGER

> A machine-readable-in-spirit record of what we are allowed to say.
> **A claim enters the "permitted" section only when the evidence exists and is reproducible.**
> This file is written before the results, so that the results cannot bend it.
>
> Inherited from the Lucky Loop *verify-before-claim* discipline, whose most useful outcome was
> that it caught three false positives **in our own favour** before we published them.

---

## ✅ PERMITTED CLAIMS

Each of these is either already true, or will be true by construction of the work.

> ### **"We build a reproducible software-only pipeline to evaluate temporal acoustic leak-like signals with strict leakage controls."**

Supporting claims, equally permitted:

| Claim | Basis |
|---|---|
| "The dataset consists of physical measurements from an external leak-detection training facility." | stated by the Zenodo record |
| "The three archives contain exactly 500 / 386 / 114 one-second WAV clips, all mono 8 kHz 16-bit." | counted, 2026-09-12, `DATASET_AUDIT.md` §1.1-1.2 |
| "7.2 % of the dataset is byte-identical duplication, and 6 duplicate pairs carry contradictory metadata." | measured, `DATASET_AUDIT.md` §1.6 |
| "Pressure and flow rate are recorded for the leak class only, which makes the filename metadata a near-perfect label leak." | measured, `DATASET_AUDIT.md` §1.5 — 469/500 vs 0/500 |
| "No duplicate and no near-duplicate crosses a class boundary." | measured, `DATASET_AUDIT.md` §1.6-1.7 |
| "The binary task is exactly balanced, 500 leak / 500 non-leak, over 306 groups." | counted, `DATASET_AUDIT.md` §4.2 |
| "A 60/20/20 grouped split is achievable with no group split across folds." | `DATASET_AUDIT.md` §12 — feasibility, no split written |
| "Leak clips are ~11 dB louder than no-leak clips; over the whole dataset, RMS alone separates them with a descriptive AUC of 0.857 at group level." | measured, `DATASET_AUDIT.md` §11 — **a description of the confound, computed with nothing held out. Never a performance, never a threshold.** |
| "Signal distributions show measurable differences associated with the provided labels." | measured, `DATASET_AUDIT.md` §11 — the wording is deliberate: see the forbidden claim below |
| "Our split is grouped by condition, session and device as far as the metadata allows, and we document where it does not." | `EVAL_PROTOCOL.md` §2-3 |
| "We measure and report the gap between a naive random split and a grouped split." | `EVAL_PROTOCOL.md` §4 |
| "The textual descriptions are generated from labels and computed measurements, and are not expert annotations." | `DATASET_CARD.md` §4 |
| "We compare the model against a baseline on held-out data." | challenge requirement, `EVAL_PROTOCOL.md` §5 |
| "We evaluate the classifier separately from the generated text." | `EVAL_PROTOCOL.md` §6 |
| "Abstention is thresholded on validation data." | `EVAL_PROTOCOL.md` §5 |
| "No hardware was used or tested in this work." | factual, and stated in `README.md` |

---

## 🚫 FORBIDDEN CLAIMS

Saying any of these would be false, unprovable, or both — regardless of how good the numbers look.

> ### **"LeakLess detects real leaks in the field."** ❌

| Forbidden claim | Why it is forbidden |
|---|---|
| "Our system detects leaks in customer installations." | No customer data, no deployment, no hardware. |
| "This validates the LeakLess sensor / product." | **No LeakLess hardware was connected or tested today.** The work is software-only. |
| "The model locates the leak." | The dataset does not support localisation. A map here would be fabrication. |
| "The model predicts pipe failure." | Nothing in the data supports forecasting. |
| "A negative output proves there is no leak." | Absence of detection is not evidence of absence. |
| "The flow rate is X L/min." | Filenames give **m/s**, a velocity. No cross-section, no flow rate. |
| "The signal amplitude corresponds to X pascals / g." | WAV amplitude is dimensionless without calibration of the recording chain. |
| "The leak is caused by *[mechanism]*." | We measure a signature, not a cause. |
| "You should repair *[X]*." | We are not qualified to prescribe an intervention. |
| "Our approach is algorithmically novel." | ML-based acoustic leak detection already exists. The contribution is the **evaluation discipline and the reusable pipeline**, not a new algorithm. |
| "Accuracy is *[high number]*" — stated without naming the split | A number without its split is not a result. |
| "The model detects leaks" — stated without the held-out RMS-only control | Loudness is a measured confound. A score published without a control evaluated on the same held-out grouped split may be describing a volume meter. `EVAL_PROTOCOL.md` §7bis-A. |
| A non-leak score stated without the number of test groups | 51 non-leak groups total, a 20 % test fold draws 11, one carrying 43 % of it. |
| "The RMS AUC of 0.857 is our baseline / our control / the bar to beat." | **0.857 is descriptive only** — computed on the whole dataset, nothing held out. It describes the confound; it is **not a performance threshold** and no model result may be compared against it. The control is a separate quantity, evaluated on the held-out grouped split. |
| "Our model beats 0.857." | Compares a held-out score against a number measured on all the data. Two different quantities. |
| An RMS-only control computed on amplitude-normalised audio | Normalisation deletes the very information the control exists to measure. The control runs on the **original un-normalised signal**. `EVAL_PROTOCOL.md` §7bis-A. |
| "The hydraulic test bench / LeakDB demonstrates pipe-leak detection." | One is a bench, the other is simulated. Neither is field evidence. |
| "We verified the labels from the audio content." | **We did not, and cannot.** We measured that signal distributions differ in a way associated with the provided labels. That is a statistical association, not a confirmation that any individual clip really contains a leak. **The provenance of every label remains the source dataset's.** |
| "Clip X is confirmed to contain a leak." | Nothing in this work can confirm the physical truth of an individual label. |

---

## ⏳ CLAIMS TO VERIFY

Neither permitted nor forbidden — **pending evidence.** Nothing from this section may be said
out loud until it moves up to §1, with the check that moved it.

| Claim | What would settle it | Status |
|---|---|---|
| ~~"The no-leak and noise archives contain 386 and 114 usable clips."~~ | archives opened and counted, 2026-09-12 — `scripts/ingest/build_groups.py`, [`DATASET_AUDIT.md`](DATASET_AUDIT.md) §1.1 | ✅ **moved to permitted** — 386 and 114 WAV counted exactly |
| "The filenames allow a clean group reconstruction." | grouping attempted, 2026-09-12 — [`DATASET_AUDIT.md`](DATASET_AUDIT.md) §4 | ⚠️ **partially settled: 97.2 % of clips get an unambiguous group, but the no-leak class yields only 18 groups and the hydrophone/logger dilemma is unresolved.** Say "97.2 %, 322 groups", never "clean". |
| "Temporal structure carries information beyond aggregated features." | the ablation in `EVAL_PROTOCOL.md` §7 | ⏳ **may come back negative, and that is a valid result** |
| "The model abstains under unseen noise." | inject held-out environmental noise and observe | ⏳ desired, **not guaranteed, never scripted** |
| "The model does more than measure loudness." | compare against an RMS-only control **trained and evaluated on the same held-out grouped split**, computed on the un-normalised signal | ⏳ **new, and now the decisive question** — `EVAL_PROTOCOL.md` §7bis-A |
| "The leak/no-leak loudness gap is a property of leaks rather than of the acquisition protocol." | would need a calibrated chain, or field data | ⏳ **cannot be settled with this dataset** |
| "Our TSLM beats the baseline." | grouped-split comparison | ⏳ |
| "The checkpoint has not seen our evaluation data." | audit OpenTSLM's training corpus for overlap | ⏳ **critical if a fallback ECG dataset is used** |
| "The generated descriptions state measurable quantities correctly." | check generated numbers against recomputed values | ⏳ |
| "The hackathon runs until Sunday." | ask the organisers | ⏳ — inferred from *"credits to keep building after Sunday"*, **not confirmed** |

---

## Rules of use

1. **A claim moves from ⏳ to ✅ only with a named, reproducible check** — and the check is
   recorded next to it.
2. **A claim that fails its check moves to 🚫**, it does not quietly disappear.
3. **If a result is favourable and surprising, suspect it first.** That reflex is what caught
   three false positives last time.
4. **Nothing in a pitch, a README, a LinkedIn post or a slide may assert what this ledger does
   not permit.** Especially not under time pressure, and especially not because it sounds good.
