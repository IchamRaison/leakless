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
| "The hydraulic test bench / LeakDB demonstrates pipe-leak detection." | One is a bench, the other is simulated. Neither is field evidence. |

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
