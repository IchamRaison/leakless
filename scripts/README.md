# scripts/

**Stubs only. Nothing here is implemented, and nothing here downloads data or trains.**

| Script | Purpose | State |
|---|---|---|
| `ingest/audit_archives.py` | Open the three Zenodo archives, count clips per class, report what the filenames encode | stub |
| `ingest/build_groups.py` | Reconstruct group keys (material/region/pressure/flow/device, `_1`/`_2` stripped) for a leakage-clean split | stub |
| `eval/split_report.py` | Run naive-random vs grouped split and report the gap — see `docs/EVAL_PROTOCOL.md` §4 | stub |

Datasets live **outside** this repository (`.gitignore` blocks `data/`, `*.rar`, `*.wav`).
