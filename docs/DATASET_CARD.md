# DATASET_CARD — candidate acoustic dataset

> Status: **candidate, not adopted.** Nothing has been downloaded. The subject is not decided.
> This card exists so that the separation between *measured* and *generated* is fixed **before**
> any data touches the disk.

---

## 1. Source

| | |
|---|---|
| Record | `https://zenodo.org/records/18631450` |
| API | `https://zenodo.org/api/records/18631450` |
| Licence | **CC BY 4.0** |
| Archives | `leak acoustic data.rar` · `no leak acoustic data.rar` · `environmental noise.rar` |

**Provenance, as stated by the source:** the leak and no-leak categories come from an
**external leak-detection training facility in Dongguan**. Part of the environmental noise
comes from a separate public site.

> ⚠️ **These are physical measurements on an experimental facility. They are NOT measurements
> taken at customer sites, and they are not a validation of any deployed product.**

---

## 2. Announced content

| Class | Clips | Audit status |
|---|---|---|
| leak | **500** | ✅ archive downloaded and listed (by Hicham) |
| no leak | **386** | ❌ **never opened** |
| environmental noise | **114** | ❌ **never opened** |

Clip duration: **1 second**.

> ⚠️ **Two of the three archives have never been inspected.** The counts above are what the
> record *announces*, not what we have verified. The no-leak and noise archives must be
> audited before any split is fixed.

---

## 3. ✅ REAL DATA — measured, not invented

| Item | Nature |
|---|---|
| **WAV signals** | physical acoustic measurements |
| **Class labels** | leak / no leak / environmental noise, as provided by the source |
| **Filename metadata** | material · region · pressure · flow rate · device (hydrophone or logger) |
| **Computed signal properties** | band energies, spectral descriptors, and any quantity **calculated** from the waveform |

Everything in this table can be pointed at and recomputed. It is the factual base of the
project.

---

## 4. ⚠️ GENERATED DATA — produced by an LLM

| Item | Nature |
|---|---|
| **Textual descriptions** | **generated from the measurements and the training-set labels** |

**Generation is explicitly permitted by the challenge brief** (official PDF, step 01,
*"Ideal Data Profile"*):

> *« In some cases, you might consider **generating text annotations with LLMs** (as in
> OpenTSLM paper) »*

**Conditions attached to every generated description:**

> ### 🔴 Generated descriptions are NOT expert annotations.
> They are derived from labels and computed measurements. They carry no clinical, engineering
> or diagnostic authority, and they are labelled as generated wherever they are displayed,
> stored, or shown to the jury.

---

## 5. 🚫 Forbidden content in any generated text

These are hard constraints, not stylistic preferences. Each one corresponds to a way this
project could mislead.

| Forbidden | Why |
|---|---|
| **No invented physical cause** | We measure an acoustic signature. We do not know *why* the pipe leaks, and a language model asserting a mechanism is fabricating. |
| **No repair recommendation** | We are not qualified to prescribe an intervention, and a wrong one has a cost in the real world. |
| **No interpretation of the m/s in filenames as a volumetric flow rate** | m/s is a velocity. Converting it to a flow rate requires a cross-section we do not have. |
| **No physical unit attached to WAV amplitude without calibration** | A WAV sample is a dimensionless integer scaled by an unknown recording chain. Calling it a pressure or an acceleration is invention. |
| **No location, no geographic mapping** | The data does not support localisation, and a map is the most convincing lie a demo can tell. |
| **No rupture prediction** | Nothing in the dataset supports forecasting a failure. |
| **No proof of leak absence** | A negative classification is not evidence that no leak exists. |

---

## 6. Structural leakage risk — read with `EVAL_PROTOCOL.md`

| Mechanism | Effect |
|---|---|
| Numerous `_1` / `_2` suffixes | several clips share one underlying event |
| Hydrophone **and** logger under identical conditions | the same event recorded twice, via two devices |
| Recurring material / region / pressure / flow combinations | near-duplicate acoustic conditions |
| 1-second clips from continuous sessions | strong correlation between consecutive clips |

> The source note is explicit: *« the filenames do not guarantee a perfect reconstruction of
> the sessions »*. The split is therefore grouped, and its residual limits are documented.
> See [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md).

---

## 7. Handling rules

- **Nothing is committed to this repository.** `.gitignore` blocks `data/`, `*.rar`, `*.wav`
  and the rest. Archives live outside the repo.
- **Labels, filenames, group identifiers and any target-revealing metadata are never given to
  the model as input.** They are supervision, not features.
- **Original WAVs are kept** alongside derived series, so that any claim can be re-listened to
  and re-checked.
- **Attribution is required** by CC BY 4.0: the Zenodo record must be credited in the
  submission and in the demo.

---

## 8. Rejected alternatives, and why

Recorded so the same ground is not re-explored under time pressure.

| Dataset | Verdict | Reason |
|---|---|---|
| **LeakDB** | rejected | **explicitly simulated** — fails the challenge's "real inputs" requirement |
| **Intra-Domestic Water Leaks** | rejected | no documented dwelling identifiers ⇒ **no per-user split possible** |
| **Telemanom SMAP/MSL** | rejected | its own README describes scaling using **test-set extrema** ⇒ leakage baked into the dataset |
| **UCI Condition Monitoring (hydraulic)** | fallback only | a hydraulic test bench is **not** a water-pipe leak dataset, and must never be presented as one |
| **PTB-XL / ECG-QA** | fallback only | clean per-patient splits, but **OpenTSLM already trains on ECG-QA** ⇒ checkpoint exposure must be audited before any novelty claim |
