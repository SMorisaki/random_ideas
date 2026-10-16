# PROGRESS.md — dist_psi_marg

Iteration log for the Codex-driven development loop (see `TASK.md`). Newest entry last.

## Session 2026-10-16 (cloud): setup and blocker

Environment checks (cloud container, not the local devcontainer):
- Branch `LLM`, working tree clean, `git pull --ff-only` up to date; `LLM` and `main` at the same commit.
- `sh .devcontainer/run_codex.sh login status` -> **"Not logged in"** (exit 1). There is no
  `~/.codex/auth.json` in the cloud container (the local devcontainer keeps it in a Docker volume
  that does not exist here). Codex `codex sandbox -c 'sandbox_mode="workspace-write"'` probe
  succeeded, so `--sandbox workspace-write` is usable once authentication exists.
- No conda/mamba binary and no `environment.yml` existed. `micro.mamba.pm` and GitHub release
  downloads are blocked by the network policy, but `conda.anaconda.org` and PyPI are reachable, so
  micromamba 2.9.0 was extracted from the conda-forge package `micromamba-2.9.0-0.tar.bz2` into
  `/home/user/.local/bin/micromamba`, `environment.yml` was created at the repository root
  (python 3.12, numpy, scipy, matplotlib, numba, h5py, pytest), and the environment was built at
  `/home/user/conda-env` (prepend `/home/user/conda-env/bin` to `PATH`).
- `/projects/data_release` (LVK posterior samples) is not mounted in the cloud container.

**Stop reason (rule: 認証エラーで停止; API キー認証へは切り替えない):** Codex cannot run
without ChatGPT authentication. No implementation iteration was started.

What was prepared so that the loop can start as soon as authentication is available:
- `dist_psi_marg/TASK.md` (goal, completion criteria, forbidden edits, planned iterations) — draft
  to be confirmed by the project owner.
- `dist_psi_marg/AGENTS.md` (rules for Codex), `dist_psi_marg/prompts/`, `dist_psi_marg/logs/`
  (git-ignored), `environment.yml`.

**Second blocker (GitHub push):** `git push -u origin LLM` was refused with HTTP 403:
"Claude doesn't have GitHub access to SMorisaki/random_ideas for your organization". The
repository is attached read-only to the cloud session; pushes need the Claude GitHub App installed
on the repository (https://github.com/apps/claude/installations/select_target) or the GitHub
connection re-linked from claude.ai settings. Commits were made locally on `LLM`. **Resolved later in the same
session:** GitHub access was granted and `git push -u origin LLM` succeeded.

How to unblock (either):
1. Store the content of the local `~/.codex/auth.json` (from the devcontainer volume) as a cloud
   environment secret named `CODEX_AUTH_JSON`. At session start the orchestrator writes it to
   `$HOME/.codex/auth.json` (mode 600) before calling `run_codex.sh login status`. The secret is
   never printed, logged or committed.
2. Or run `sh .devcontainer/run_codex.sh login --device-auth` in a session and approve the device
   code in a browser while the session is attended.
