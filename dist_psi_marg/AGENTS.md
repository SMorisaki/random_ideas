# AGENTS.md — guidance for Codex working in `dist_psi_marg/`

## What this directory is
Research code for a fast marginalization of the gravitational-wave likelihood over
luminosity distance `D_L` and polarization angle `psi`.
Background notes (read them first):
- `distmarg.pdf`: the existing distance-only marginalization with a 2-D lookup table in
  `(x0, b)` and the sinh-based grid coordinates `s(x0)`, `t(b)`.
- `dist_psi_marg.tex` / `.pdf`: the distance + psi problem. `ln L` depends on five
  coefficients `A, B, C, D, E` (Eq. 17); after rotating by `xi` and the symmetries of the
  full-circle `psi` integral, it depends on four parameters
  `(lambda1, lambda2, R, eta)` with `lambda1 = E + S`, `lambda2 = E - S`,
  `S = sqrt(C^2 + D^2)`, `R = sqrt(A^2 + B^2)`, `eta = zeta - xi`.
- `TASK.md`: goal, completion criteria and constraints. `PROGRESS.md`: iteration log.

## Hard rules
- Edit, create or delete files **only inside `dist_psi_marg/`** (this directory and below).
  Never touch files outside it (not even `environment.yml`; report instead).
- Do **not** run `git commit`, `git push`, `git stash`, `git reset`, `git checkout` of other
  branches, or anything that rewrites history. The orchestrator commits.
- Do not create new conda environments and do not `pip install` / `mamba install` anything.
  If a package is missing, stop and report the package name and the reason.
- Never write secrets (tokens, keys) into files, logs or code.
- Keep figures and data products under `dist_psi_marg/figures/` and `dist_psi_marg/data/`.
  Keep large generated tables out of git unless `TASK.md` says otherwise.

## Environment
- Use the `python` on `PATH` (a conda environment built from the repository's
  `environment.yml`: numpy, scipy, numba, matplotlib, h5py, pytest). Check with
  `which python` and `python -c "import scipy, numba"`.
- Tests: `python -m pytest dist_psi_marg/tests -q` from the repository root, or
  `python -m pytest tests -q` from this directory.

## Code conventions
- Package code lives in `dist_psi_marg/dpmarg/` (importable as `dpmarg`), tests in
  `dist_psi_marg/tests/`, experiment scripts in `dist_psi_marg/scripts/`.
- Every numerical method must be checked against the brute-force reference
  (`dpmarg.reference`) on a documented set of parameter points, including the extreme
  regimes listed in `TASK.md` (no signal, degenerate `lambda2 = 0`, very large
  `lambda1`, peak outside `[D_min, D_max]`, bimodal `psi` profile).
- Report accuracy as the absolute error in `ln L_marg` and timing as the median
  wall-clock per evaluation after JIT warm-up.
- Write docstrings that state the formula implemented and its domain of validity.
