# PROGRESS.md — multiband_quad

Iteration log for the Codex-driven development loop (see `TASK.md`). Newest entry last.

## Session 2026-10-17 (cloud): setup

- Branch `LLM`, working tree clean, `git pull --ff-only origin LLM` up to date. Push access verified
  (`git push --dry-run` and the session's repository access check: push ok).
- Codex 0.163.0 authenticated with ChatGPT via `sh .devcontainer/run_codex.sh login --device-auth`
  (device code approved by the owner in a browser); `run_codex.sh login status` -> "Logged in using ChatGPT".
  `.devcontainer/codex-check.sh` run by hand: sandbox probe OK -> `~/.codex/sandbox_mode = workspace-write`,
  so Codex is run with `--sandbox workspace-write` (not disabled).
- Conda env rebuilt with `bash dist_psi_marg/tools/cloud_setup.sh` at `/home/user/conda-env`
  (python 3.12.15, numpy 2.5.4, scipy 1.18.1, numba 0.68.0). The orchestrator added
  `python-lalsimulation` 6.2.0 (`micromamba install -c conda-forge python-lalsimulation`) for
  IMRPhenomXHM mode-by-mode waveforms on arbitrary frequency sequences
  (`SimIMRPhenomXHMFrequencySequenceOneMode`: 1.5-8 ms per mode at 1e4 points on this machine;
  full IMRPhenomXHM `ChooseFDWaveformSequence`: 12 ms) and updated `environment.yml` in the same commit.
- `/projects/data_release` is not mounted in the cloud container; experiments use analytic PSDs and
  synthetic signals. No root `CLAUDE.md` exists in the repository.
- The paper (arXiv:2104.07813) was read; its equations are restated in `notes/paper_summary.md` and the
  candidate method is derived in `notes/method_idea.md`. A design review (three adversarial reviewers,
  three independent proposers, one judge) was run before drafting `TASK.md`; its conclusions are folded
  into `TASK.md` and `notes/method_idea.md`.

Owner decisions (2026-10-17, via the question card): accuracy/speed targets as proposed; IMRPhenomXHM
non-precessing as the main model (precession is a stretch item); real PSDs may be downloaded from public
sources (GWOSC / LIGO DCC / zenodo reachable, HTTP 200) into `data/` (not committed); **additional requirement:
also provide a method that works when only `h_+`, `h_x` are available (no mode decomposition)** -> recorded as
method B (baseband IFFT-FFT with half-sample shift and adaptive block length) in `TASK.md` §1-2 and
`notes/method_idea.md` §7, verified on the toy (`notes/orchestrator_toy_methodB.py`).
