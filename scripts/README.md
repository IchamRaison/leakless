# scripts/

**Nothing here downloads data or trains.** `build_groups.py` is implemented and reads an
already-extracted dataset from a path outside the repository.

| Script | Purpose | State |
|---|---|---|
| `ingest/audit_archives.py` | Open the three Zenodo archives, count clips per class, report what the filenames encode | superseded by `build_groups.py`, which does both |
| `ingest/build_groups.py` | Group keys, duplicate merging, ambiguity report, **frozen split manifest** (`--write-manifest`) | ✅ implemented — [`docs/DATASET_AUDIT.md`](../docs/DATASET_AUDIT.md), [`docs/SPLIT_AUDIT.md`](../docs/SPLIT_AUDIT.md) |
| `eval/split_report.py` | Run naive-random vs grouped split and report the gap — see `docs/EVAL_PROTOCOL.md` §4 | stub |

Datasets live **outside** this repository (`.gitignore` blocks `data/`, `*.rar`, `*.wav`).
The split manifest (`manifests/`) is tracked — it carries identifiers and folds, never audio.
