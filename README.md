# EHL Hackathon Zurich

Project workspace for the EHL hackathon in Zurich.

Event information: https://tum-ai.notion.site/ehl-hackathon-zurich

---

## ⚠️ Scope and honesty statement — read first

**This is a software-only hackathon prototype.**

- **No LeakLess hardware is used, connected, or tested today.** No accelerometer, no piezo, no acoustic sensor, no field installation.
- **No claim is made about real leak detection under customer conditions.** Nothing in this repository demonstrates that a deployed product detects leaks in a real building.
- The dataset under consideration consists of **physical measurements taken on an external training facility**, not measurements taken at customer sites.
- **Textual descriptions produced by an LLM are not expert annotations** and are labelled as such wherever they appear.

**Objective of the work:** a **reproducible pipeline** —
`temporal signal → text annotation → baseline → evaluation → demo`
with **strict data-leakage controls** and an explicit record of what is measured versus what is generated.

See [`docs/CLAIM_LEDGER.md`](docs/CLAIM_LEDGER.md) for the exact list of permitted and forbidden claims.

---

## Challenge

**TEMPORAL AI CHALLENGE — "Give AI a Sense of Time"**, by Agentic Systems Lab × Aionic Labs × Nebius.
Use **TimeNet** to prepare open-source time-series data, then train a **TSLM** (Time-Series Language Model) connecting temporal signals with language for a useful task.

Full decoded brief: [`docs/HACKATHON_BRIEF.md`](docs/HACKATHON_BRIEF.md)

---

## Current state — 2026-09-12

| Item | State |
|---|---|
| Subject | ✅ **Option B — LeakLess software-only**, validated by the team (conditional on the dataset audit, now passed) — see [`docs/HACKATHON_OPTIONS.md`](docs/HACKATHON_OPTIONS.md) |
| Nebius compute voucher | ✅ **operational** — H100 80GB HBM3, CUDA 13.0, `nvidia-smi` verified, GPU idle |
| Dataset | ✅ **audited — GO (conditional)** — 1000 WAV counted, 306 leakage-clean groups, [`docs/DATASET_AUDIT.md`](docs/DATASET_AUDIT.md) §13 |
| TimeNet | ✅ cloned and working locally (CLI operational) |
| Split | ✅ **frozen** — `manifests/split_v1.csv`, seed 20260912, all overlap checks at 0 — [`docs/SPLIT_AUDIT.md`](docs/SPLIT_AUDIT.md) |
| Training | ❌ **nothing launched, deliberately** |
| Code | ⏳ `scripts/ingest/build_groups.py` implemented — audit, grouping, **frozen split manifest**. No training, no TimeNet conversion, no generated annotation. |
| Submission deadline / format | ❌ **unknown** — to be confirmed with organisers |

**Nothing has been trained. No API key exists anywhere in this repository.** The dataset
archives have been downloaded and extracted **outside the repository** for the audit; no WAV,
no `.rar` and no derived series is tracked by Git.

---

## Documentation

| File | Contents |
|---|---|
| [`docs/HACKATHON_BRIEF.md`](docs/HACKATHON_BRIEF.md) | The challenge, decoded from the official PDF and Discord brief — including what is still unknown |
| [`docs/NEBIUS_SETUP.md`](docs/NEBIUS_SETUP.md) | Step-by-step GPU setup, including how to avoid the initial $25 charge |
| [`docs/REPO_INVENTORY.md`](docs/REPO_INVENTORY.md) | What both team repositories actually contain |
| [`docs/HACKATHON_OPTIONS.md`](docs/HACKATHON_OPTIONS.md) | Three candidate directions, compared — **no decision taken** |
| [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md) | The evaluation rules, including the hard no-random-split rule |
| [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) | Real data versus generated data, kept strictly separate |
| [`docs/DATASET_AUDIT.md`](docs/DATASET_AUDIT.md) | **Zenodo dataset audit** (FR) — verified counts, duplicates, group key, measured leakage risks, decision |
| [`docs/SPLIT_AUDIT.md`](docs/SPLIT_AUDIT.md) | **Frozen split audit** (FR) — fold counts, group sizes, overlap checks, manifest hashes |
| [`docs/CLAIM_LEDGER.md`](docs/CLAIM_LEDGER.md) | Permitted claims, forbidden claims, claims to verify |

---

## Setup

Install the [Entire CLI](https://docs.entire.io/overview), then run `entire enable` after cloning to install local Git and agent hooks. Hooks are not transferred by Git.

Entire uses manual-commit checkpoints. Telemetry and automatic pushing of session logs are disabled. Review checkpoint content before explicitly sharing it.

Agent integrations are configured for Claude Code, Codex, Copilot CLI, Cursor, Factory AI Droid, Gemini CLI, OpenCode, and Pi. Hermes session capture has not been verified.

## Obsidian vault

Open `vault/` as a vault in Obsidian and start with `Accueil.md`. It contains the brief checklist, tasks, research, architecture, decisions, experiments, demo preparation, daily journal, and templates. No community plugins are required.

Notes and shared settings are tracked in Git. Device-specific workspaces, local plugins, trash, and secrets are excluded. Git is the sharing mechanism; automatic Obsidian Sync is not configured. Pull before editing and commit/push to share changes.

> ⚠️ **The `vault/` directory in this repository is a mirror.** The reference vault is
> `https://github.com/IchamRaison/ehl-hackathon-zurich-vault` and there is **no automatic
> synchronisation** between the two. Write notes in the reference vault, not here.

## Secrets

No API key, token, voucher or promo code belongs in this repository — not in files, not in
notebooks, not in commit messages, not in shell history. `.gitignore` blocks the usual
suspects, but the rule is the discipline, not the file.
