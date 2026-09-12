# Entire skills installation

Installed in this project from https://github.com/entireio/skills with:

    npx --yes skills add https://github.com/entireio/skills --all

The upstream repository contains 12 skills. The installer initially skipped `search` because its unquoted YAML description contains a colon followed by a space. A local copy of that description was JSON-quoted (valid YAML), without changing its meaning, then installed with the same installer. `skills-lock.json` records the local source for that repaired skill; do not expect the temporary installation directory to exist on other machines. The repaired skill itself is committed in `.agents/skills/search/`.

All 12 installed SKILL.md frontmatters were parsed successfully. Cross-agent files are in `.agents/skills/`, with installer-created agent links/copies. LICENSE is preserved in `.agents/ENTIRE-LICENSE`. Upstream main observed during verification: fe5266f76d846222c73b080ce39fd803dd705198 (not a claim that all installer downloads were atomically pinned to that revision).

Entire CLI was updated to stable 0.10.6 using its official installer with release checksum verification. Hooks were refreshed, existing branch checkpoint backend preserved, telemetry=false and push_sessions=false retained. Three new Codex hooks require review via `/hooks` on this machine: session_end, subagent_start, subagent_stop. Existing Codex checkpoints remain readable.

Hermes default profile on this machine scans this project's `.agents/skills/` through the documented `skills.external_dirs` setting. `hermes skills list` sees all 12. Other machines need their own discovery configuration; no other Hermes profile was changed. Reload skills or start a fresh session to refresh discovery if needed.

Local capture and checkpoint reads work. Entire login is not configured yet. Remote checkpoint search/indexing and the entire.io session UI are not verified. Automatic session publication remains disabled; publishing recorded conversations requires deliberate review and authorization. Running `entire login` is the remaining account step; review access scopes in the browser yourself. Git hooks are not transferred by cloning: new clones must run `entire enable` and approve their own agent hooks.

Skills may instruct history search first; apply these workflows to recorded development history, not as replacements for scientific dataset research. Respect user intent and approval boundaries for mutations such as session attachment, commit amendment, review resolution, or publishing transcripts.
