# scripts/

**Nothing here downloads data or trains.** `build_groups.py` is implemented and reads an
already-extracted dataset from a path outside the repository.

| Script | Purpose | State |
|---|---|---|
| `ingest/audit_archives.py` | Open the three Zenodo archives, count clips per class, report what the filenames encode | superseded by `build_groups.py`, which does both |
| `ingest/build_split_v2.py` | **Builds `split_v2`** — explicit dependency graph, connected components, rejection-sampling fold assignment | ✅ active — [`docs/SPLIT_V2_AUDIT.md`](../docs/SPLIT_V2_AUDIT.md) |
| `eval/verify_split_invariants.py` | **Independent invariant checker** — reads only `clip_id` and `fold`, recomputes every dependency from the audio. Fails on `split_v1`, passes on `split_v2`. | ✅ active |
| `ingest/build_groups.py` | Produced the invalid `split_v1` | 🔴 **superseded** — kept for traceability, do not use |
| `eval/verify_manifest.py` | Checked `split_v1` — but copied its faulty predicate and trusted its `group_id` column | 🔴 **superseded** by `verify_split_invariants.py` |
| `timenet/leakless_acoustic/` | **TimeNet connector** + dataset card — clips → TimeF, `subject_ids` = dependency cluster, amplitude-normalised | ✅ active |
| `timenet/build_timef.py` | Drives download → convert → write; output goes **outside** the repo | ✅ active |
| `eval/group_metrics.py` | Group-aware metrics: clip-level, group-level, cluster bootstrap, group-weighted threshold | ✅ active |
| `eval/rms_control.py` | **RMS-only control on the raw, un-normalised signal**, same frozen folds | ✅ active — [`docs/PIPELINE_RESULTS.md`](../docs/PIPELINE_RESULTS.md) |
| `eval/baseline_logreg.py` | Spectral descriptors + logistic regression on normalised audio, same frozen folds | ✅ active |
| `eval/harness/` | **Temporal Evidence Harness** — split chokepoint, run contract, group-aware metrics, paired cluster bootstrap, frozen C0-C3 features | ✅ active |
| `eval/run_controls.py` | Produces C0-C3 prediction runs in the same contract shape as the TSLM (`--shuffle-labels` for the negative control) | ✅ active |
| `eval/check_run.py` | **Vérification de conformité pour Hicham** — `--template` génère le squelette aux 402 clip_id, `--run --inspect` valide et diagnostique la distribution. Ne calcule aucune métrique. | ✅ active |
| `eval/evaluate_predictions.py` | **The only evaluation path.** Any run dir in, clip + cluster metrics and paired comparisons out | ✅ active |
| `eval/build_final_report.py` | Generates `artifacts/final_evaluation/` — no number is copied by hand | ✅ active |
| `temporal/stress.py` | T0-T3 transforms and their measured invariants | ✅ active |
| `temporal/build_stress_timef.py` | Writes the T0-T3 TimeF variants outside the repo and verifies mappings | ✅ active |
| `eval/split_report.py` | Run naive-random vs grouped split and report the gap — see `docs/EVAL_PROTOCOL.md` §4 | stub |

Datasets live **outside** this repository (`.gitignore` blocks `data/`, `*.rar`, `*.wav`).
The split manifest (`manifests/`) is tracked — it carries identifiers and folds, never audio.
