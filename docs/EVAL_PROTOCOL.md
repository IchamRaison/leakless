# EVAL_PROTOCOL — evaluation rules, fixed before any measurement

> Written 2026-09-12, **before any training run**. That is the point: these rules are
> pre-registered so that a favourable result cannot retroactively justify a loose protocol.
> Any change to this file after a result is known must be dated and justified in the same commit.

---

## RULE 0 — THE HARD RULE

> ## ❌ NO RANDOM SPLIT BY FILE.
> ### The split is **grouped by condition / session / device**, as far as the metadata allows.

A random clip-by-clip split on this kind of data produces a flattering score that is
**indefensible**. It is the single most common way a hackathon project loses credibility in
front of a technical jury — and this jury names the requirement explicitly in the challenge
brief:

> *« Compare it with a baseline on held-out data, **avoiding leakage across time, subjects, or devices**. »*
> — Temporal AI Challenge, step 03

**Therefore, as a contract:**

- **no leakage across time**
- **no leakage across subjects**
- **no leakage across devices**

---

## 1. Why the risk is structural here, not hypothetical

The candidate acoustic dataset carries dependencies **by construction**:

| Mechanism | Consequence |
|---|---|
| **Numerous `_1` / `_2` suffixes** | Several clips describe the *same* underlying event |
| **Hydrophone *and* logger capture under identical conditions** | The *same* physical event appears twice, through two devices |
| **Same material / region / pressure / flow-rate combinations recur** | Near-duplicate acoustic conditions across files |
| **1-second clips** | Consecutive clips from one recording session are strongly correlated |

⇒ Two clips can be, in substance, **the same measurement**. Put one in train and the other in
test and the model is scored on data it has already seen. The score goes up; the meaning goes
down.

---

## 2. The split procedure

1. **Derive group keys from the filenames** — material, region, pressure, flow rate, device,
   and the base identifier stripped of its `_1`/`_2` suffix.
2. **Exact deduplication** — identical files removed.
3. **Near-duplicate search** — clips that are acoustically near-identical are assigned to the
   *same* group, never split across folds.
4. **Group-level assignment** — a whole group goes to train, or to validation, or to test.
   **Never across two.**
5. **Hold out devices where support allows it** — if there are enough samples, reserve at
   least one device entirely for test, so that generalisation *across instruments* is measured
   rather than assumed.
6. **Fix the groups BEFORE any windowing or feature extraction.** Windowing a signal before
   splitting leaks information from one fold into another.
7. **Fit every transformation on train only** — normalisation statistics, band definitions,
   calibration constants. Validation and test are transformed with train-fitted parameters,
   never refitted.

---

## 3. When the metadata is not good enough

The proposal note already warns that *« the filenames do not guarantee a perfect
reconstruction of the sessions »*. This will happen. The rule when it does:

> ### If the metadata does not allow a perfect split, **we document the limit — we do not hide it.**
> ### And we **state the real scope of the test** rather than the scope we wish we had.

Concretely, every reported number is accompanied by:

- **what the split actually separates** (events? devices? only file groups?)
- **what it does not separate**, and therefore what the number does *not* prove
- the **residual leakage risk** we could not eliminate

A stated limitation costs a few sentences. An unstated one costs the whole submission the
moment a juror asks the question — and this jury will ask it.

---

## 4. ⭐ The leakage audit is a deliverable, not a precaution

Do not merely avoid the naive split — **measure what it would have bought you**:

| Run | Split | Expected outcome |
|---|---|---|
| A | naive random, clip by clip | optimistic score |
| B | grouped by condition / session / device | honest score |

> ### The gap between A and B **is a result**. Report it.

It demonstrates, on our own work, that the controls are real rather than decorative — and it
distinguishes us from every team that will show a single accuracy figure without knowing why
it is high.

---

## 5. Metrics

| Metric | Why |
|---|---|
| **macro-F1** | class imbalance (500 / 386 / 114) makes plain accuracy misleading |
| **leak recall** | missing a leak is the costly error for the target user |
| **false-alarm rate** | a detector nobody trusts is not deployed |
| **accuracy of the *measurable* statements in the generated text** | catches a fluent model that states wrong numbers |
| **robustness under unseen noise** | the environmental-noise clips are held out for exactly this |
| **precision / coverage curve with abstention** | a technician needs to know when *not* to trust the output |

**Abstention is thresholded on validation**, never on a confidence the model asserts inside
its own generated text.

---

## 6. Separate the classifier from the prose

> **The classification result and the generated text are evaluated separately.**

A convincing paragraph must never be allowed to stand in for a correct decision. This is the
classic failure mode of a TSLM demo: beautiful language over a mediocre classifier.

And the converse architectural shortcut is equally forbidden:

> **The TSLM must not be a mere writer receiving the answer from a separate classifier.**

The jury is the OpenTSLM team. They will recognise that immediately.

---

## 7. Ablation: does temporality actually help?

The challenge is called *Give AI a Sense of Time*. So the question must be asked honestly:

| Comparison | What it tests |
|---|---|
| aggregated features vs. time series | does the temporal structure carry information? |
| original order vs. temporally permuted input | does the model use the order? |

> **A temporal permutation may fail to affect a stationary signal. That is a legitimate
> conclusion, not a test that must succeed.**

A well-measured negative result is worth more than a positive one that does not survive
scrutiny. We have done this before, and it is the most defensible thing we can bring.

---

## 7bis. AMENDMENT — 2026-09-12, after the dataset audit

> Added **before any training run**, and **before any performance number exists**. Justified by
> measurements recorded in [`DATASET_AUDIT.md`](DATASET_AUDIT.md) §11. Nothing in this amendment
> relaxes a rule; it adds two.

**A. Loudness is a confound, and it is measured.** Leak clips are ~11 dB louder than no-leak
clips. The RMS level alone reaches a **descriptive AUC of 0.857** at group level (255 leak groups
vs 51 non-leak groups, whole dataset, nothing held out).

Therefore, as a contract:

- **Per-clip amplitude normalisation**, applied identically to every class, before any model input.
- **An "RMS only" control is published next to every reported score.** A model that does not beat
  0.857 on the grouped split has added nothing over measuring volume.
- WAV amplitude is uncalibrated. **Nothing guarantees this gap survives a different recording
  chain**, which is exactly why the control travels with the result.

**B. The binary task is balanced 500 / 500.** The source labels the 114 environmental-noise clips
as no-leak. §5 above justifies macro-F1 by a 500/386/114 imbalance — that argument holds for the
three-class task only. For leak vs non-leak, report macro-F1 *and* the per-group dispersion,
never accuracy alone.

**C. Group counts travel with every score.** The non-leak side has 51 groups; a 20 % test fold
draws them from 11 groups, one of which carries 43 % of the fold. Any non-leak score published
without its group count is uninterpretable.

**D. Device hold-out is a secondary evaluation, reported separately** — 87 leak + 107 non-leak
hydrophone clips, and **zero environmental-noise clips**. It cannot test noise robustness.

**E. Deviation from §2.2, stated explicitly.** §2.2 says identical files are *removed*. The audit
**merges** their groups instead of deleting the files: 36 identical pairs exist, 6 of which carry
contradictory metadata, and deleting one side of those would silently pick which label is right.
Merging keeps every clip, guarantees duplicates never straddle a fold, and leaves the curation
error visible. Deduplication by removal remains available as an ablation.

---

## 8. Pre-registered failure criteria

Fixed now, so that they cannot be adjusted later:

- If the grouped split gives a score **indistinguishable from chance**, that is the result. We
  report it.
- If abstention **does not** rise under injected unseen noise, we say so. Abstention is
  **desired but not guaranteed, and never scripted.**
- If no split can be defended as group-clean, the run is labelled **exploratory** and no
  performance claim is made from it.
