# EHL Hackathon Zurich

Project workspace for the EHL hackathon in Zurich.

## Application Safoan — studio local

Premier jalon disponible : import et lecture WAV, waveform et spectrogramme,
API FastAPI et contrats. Les modèles restent explicitement indisponibles.
Installation, lancement, API, tests et limites : [docs/APPLICATION.md](docs/APPLICATION.md).

```sh
.venv/bin/python -m uvicorn pipe.api.main:app --app-dir src --host 127.0.0.1 --port 8000
# Dans un second terminal, après installation documentée :
npm --prefix frontend run dev
```

Ouvrir http://127.0.0.1:5173.

Event information: https://tum-ai.notion.site/ehl-hackathon-zurich

## Setup

Install the [Entire CLI](https://docs.entire.io/overview), then run `entire enable` after cloning to install local Git and agent hooks. Hooks are not transferred by Git.

Entire uses manual-commit checkpoints. Telemetry and automatic pushing of session logs are disabled. Review checkpoint content before explicitly sharing it.

Agent integrations are configured for Claude Code, Codex, Copilot CLI, Cursor, Factory AI Droid, Gemini CLI, OpenCode, and Pi. Hermes session capture has not been verified.

## Obsidian vault

Open `vault/` as a vault in Obsidian and start with `Accueil.md`. It contains the brief checklist, tasks, research, architecture, decisions, experiments, demo preparation, daily journal, and templates. No community plugins are required.

Notes and shared settings are tracked in Git. Device-specific workspaces, local plugins, trash, and secrets are excluded. Git is the sharing mechanism; automatic Obsidian Sync is not configured. Pull before editing and commit/push to share changes.

La répartition et les contrats de l'équipe sont documentés dans le vault partagé.

## V0 acoustique — Icham

Chaîne réelle WAV → TimeNet → bandes temporelles → OpenTSLM-SP/Qwen 3.5-4B.
Un petit entraînement de diagnostic a été exécuté sur train ; la qualité n'est pas validée.
Installation, checkpoint, preuves et fonction de prédiction : [guide V0](docs/TSLM_V0.md).
La couche fiable V2 est documentée dans [TSLM_V2_OUTPUT.md](docs/TSLM_V2_OUTPUT.md).
Son chargement applicatif exige le bundle sélectionné, l'artefact de décision et
sa preuve de validation ; aucun résultat n'est inventé si ces fichiers manquent.
