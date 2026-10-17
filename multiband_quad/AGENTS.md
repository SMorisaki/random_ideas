# AGENTS.md — guidance for Codex working in `multiband_quad/`

## What this directory is
Research code for a faster way to compute the quadratic term `(h, h)` of the gravitational-wave
likelihood in the *multibanding* scheme of S. Morisaki, PRD 104, 044062 (2021), arXiv:2104.07813,
when the waveform contains higher-order modes. The paper's IFFT-FFT method (its Sec. II.D.2) is the
baseline to beat; the candidate replacement is the pairwise block-weight quadrature described in
`notes/method_idea.md`. Read, in this order:
- `TASK.md`: goal, completion criteria, constraints, iteration plan. `PROGRESS.md`: iteration log.
- `notes/method_idea.md`: the derivation of the new method and the paper's formulas restated.
- `notes/paper_summary.md` (if present): the paper's equations in plain text.

## Hard rules
- Edit, create or delete files **only inside `multiband_quad/`** (this directory and below).
  Never touch files outside it (not even `environment.yml`; report instead).
- Do **not** run `git commit`, `git push`, `git stash`, `git reset`, `git checkout` of other
  branches, or anything that rewrites history. The orchestrator commits.
- Do not create conda environments and do not `pip install` / `mamba install` anything.
  If a package is missing, stop and report the package name and the reason.
- Never write secrets (tokens, keys) into files, logs or code. No network access is needed.
- Keep figures under `multiband_quad/figures/` and data products under `multiband_quad/data/`.
  Committed JSON/MD summaries should stay under ~100 KB; write per-point dumps to
  `data/<name>_full.json` and run logs to `data/<name>_run.log` (both git-ignored).
- Do not edit `TASK.md`, `PROGRESS.md`, `AGENTS.md`, `notes/method_idea.md` or `prompts/`;
  put your report in the final message. The orchestrator records progress.

## Environment
- Use the `python` on `PATH` (a conda environment built from the repository's `environment.yml`:
  numpy, scipy, numba, matplotlib, h5py, pytest, and `lalsimulation` for IMRPhenomXHM/HM).
  Check with `which python` and `python -c "import scipy, numba, lalsimulation"`.
- Tests: `python -m pytest tests -q` from this directory (`multiband_quad/`).
- The machine has 4 cores and 15 GB RAM; keep single experiments under ~10 minutes and
  cache expensive intermediate arrays in `data/*.npz` (git-ignored).

## Code conventions
- Package code lives in `multiband_quad/mbquad/` (importable as `mbquad` from this directory),
  tests in `multiband_quad/tests/`, experiment scripts in `multiband_quad/scripts/`.
- Follow the paper's notation in names and docstrings (`T_b`, `N_b`, `M_b`, `K_s`, `K_e`,
  `w_b`, `L`, `tau`, `D_tilde`, `I_tilde`, `c_k`), and cite the equation number implemented.
- Every approximate method must be checked against the brute-force fine-grid `(h, h)` on a
  documented set of systems; report errors both in `(h, h)` (relative) and in the log-likelihood
  ratio `ln Lambda` (absolute, at the stated SNR), as medians and worst cases.
- Report timings as the median wall-clock per evaluation after warm-up, separately for the
  waveform evaluation and for the `(h, h)` step, on this machine.
- Write docstrings that state the formula implemented and its domain of validity.
