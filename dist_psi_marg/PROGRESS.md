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

## Session 2026-10-16 (cloud, second session): authentication resolved, loop started

- Branch `LLM`, working tree clean, `git pull --ff-only` up to date (f5041f8). Push access verified
  (`check_repo_access`: push_check ok).
- `sh .devcontainer/run_codex.sh login --device-auth` was approved by the owner in a browser;
  `sh .devcontainer/run_codex.sh login status` -> "Logged in using ChatGPT". Codex 0.163.0.
- Codex sandbox probe in this container: `codex sandbox -c 'sandbox_mode="workspace-write"'` can write
  in the cwd and is denied outside it ("Read-only file system"), so `--sandbox workspace-write` is used.
- Conda env rebuilt with `bash dist_psi_marg/tools/cloud_setup.sh` at `/home/user/conda-env`
  (python 3.12.15, numpy 2.5.4, scipy 1.18.1, numba 0.68.0). `mpmath` added by the orchestrator
  (`micromamba install -c conda-forge mpmath`) for the independent high-precision reference;
  `environment.yml` updated in the same commit.
- `/projects/data_release` is not mounted in the cloud container; experiments use synthetic coefficients.
- Owner instructions (2026-10-16) recorded in `TASK.md` §1, §3, §7, §8: compare three methods
  (full 4-D lookup table / partial table + numerical integration / Gaussian approximation in the steep
  direction + table), find the best one, and determine the SNR below which the Gaussian approximation
  breaks. Coefficients must cover everything a sampler can produce, not only typical GW values.
- Network note: `api.openai.com` is denied by the environment's network policy (CONNECT 403), but
  Codex with ChatGPT auth uses `chatgpt.com` / `auth.openai.com`, which are reachable.

## Iteration 001 — package, independent reference integrals, regression set

Completed the requested package and the full 40-point regression run (20 named
cases and 20 random cases, seed 1). No package installation, specification edits,
commits, or other git state changes were made.

Files created:
- `dpmarg/__init__.py`, `dpmarg/params.py`: coefficient validation, canonical
  reduction and expansion, finite-domain global likelihood maximum, and SNR.
- `dpmarg/reference.py`: full-circle nested SciPy adaptive quadrature and an
  independent arbitrary-precision nested mpmath tanh-sinh integral.
- `dpmarg/regimes.py`: documented named extremes and reproducible random inputs.
- `scripts/make_regression_points.py`: generates the regression JSON, prints the
  complete table, and raises with input parameters if a point fails.
- `tests/test_reference.py`, `pytest.ini`: normalization, symmetry, round trips,
  independent Bessel integrals, mpmath comparisons, and 2-D grid maximum checks.
- `data/regression_points.json`: all inputs, reduced parameters, actual and proxy
  SNR, reference values, errors, and elapsed times.
- `data/regression_table.txt`: complete printed table and generator summary.
- `data/reference_timings.json`: repeated timing samples and per-case medians.

Verification:
- `python -m pytest tests -q`: **41 passed in 69.68 s**, including the slow tests.
- `python scripts/make_regression_points.py`: completed successfully. Every point
  has `abs(ref - mp) <= 1e-8`; worst error **3.3946179200938786e-11**, at
  `large_lam1_rho_0.5`. Total per-point elapsed time was 1208.79 s (about 20 min);
  the median was 30.36 s. The earlier tests ran concurrently with its first cases.
- After generation, reloaded all 40 saved points and compared each to the saved
  mpmath result during repeated SciPy timings; all remained within `1e-8`.
- Timing protocol: one warm-up call and three measured calls per point, median
  wall-clock per point. There is no JIT in these reference methods. Median across
  the 40 per-point medians: **0.137804 s/evaluation**; maximum per-point median:
  **0.290291 s**. The exact zero case takes about 5 microseconds.

Numerical details for the next iteration:
- Both integrations use the same float64 canonical coefficient reduction and
  tiny-negative-eigenvalue clipping convention; their quadrature values are
  independent. The mpmath integrand and normalization use 25-digit arithmetic.
  In isotropic cases the quadratic axes are arbitrary, so eta itself is not an
  invariant under expansion with arbitrary xi.
- Radial partitions include Gaussian widths, exponential endpoint widths
  `1/abs(a-b*r_star)`, and geometric prior-resolution points. The endpoint widths
  are necessary when a high-curvature exponent peaks outside the prior interval.
- Angular searches use a dense scan and bounded refinement. For mpmath partition
  selection, including the radial prior in the peak estimate avoids generating
  hundreds of redundant peaks on a flat degenerate exponent ridge.
- The mpmath integral uses the exact four-quadrant identity
  `4*cosh(R*r*cos(eta)*cos(theta))*cosh(R*r*sin(eta)*sin(theta))`
  on `[0, pi/2]`. It uses a single constant log shift including the prior weight,
  preventing an absolute-error stopping rule from overlooking a small amplitude
  near xmax. Outer tanh-sinh degree increases until its estimated relative error
  is at most `1e-15` at the default precision. No integration tails are truncated.
- Float64 isotropic evaluation uses `b = lam2 + (lam1-lam2)*cos(theta)**2` to
  avoid false angular peaks from roundoff in `cos(theta)**2+sin(theta)**2`.
- The four named `lam1=1e8` cases have actual annular SNR 0.5, 10, 100, 300.
  Random inputs follow the requested proxy `R/sqrt(lam1)` distribution; their
  actual SNR can exceed 300 (the seed-1 maximum is about 655.5). Both quantities
  are stored explicitly. All named and random cases passed without exceptions.

### Orchestrator verification of iteration 001 (accepted)

- Read `dpmarg/params.py`, `dpmarg/reference.py`, `dpmarg/regimes.py`, the tests and the script.
  The coefficient reduction, the normalization `ln[3/(2 pi (xmin^-3 - xmax^-3))]` (checked
  against the `(xmin/x)^4` rescaling used inside the inner integral) and the four-quadrant identity
  `4 cosh(u r) cosh(v r)` used by the mpmath integrand are correct.
- `python -m pytest tests -q`: 41 passed in 69 s (run by the orchestrator, conda env).
- Independent orchestrator check (not using any `dpmarg` quadrature): trapezoid rule on a dense
  grid in `u = ln r` (20001 points) times a 4096-point periodic rule in theta, for six smooth cases
  and both `(xmin, xmax) = (0.1, 100), (0.1, 10)`. Differences from `lnL_marg_ref` are
  `9.0e-8` / `4.0e-8` uniformly across the cases *including the exact-zero case*, i.e. they are the
  discretization error of the crude check, not of the reference. The pure-quadratic case
  `E = 1` agrees with an exact 1-D mpmath integral to 12 digits for `ref` and `mp`.
- Regression set: 40 points, worst `|ref - mp| = 3.4e-11`, actual SNR up to 655, `lam1` up to
  `3.7e7`, `lam2 = 0` cases, peaks outside the prior range on both sides. Reference cost
  0.12–0.29 s per call, so the reference is for validation only; the fast engine for sweeps is the
  goal of iterations 002–003.
- Note for later iterations: with `xmin = 0.1` any interior distance peak at `lam1 = 1e8` has
  SNR >= 1000, so "very large lam1" at moderate SNR necessarily means a peak below `xmin`.
- Next: iteration 002 — numba 1-D distance kernel `lnI(a, b; xmin, xmax)` parametrized by
  `(a, b)` so that `b = 0` and `|a| -> large` are regular inputs.

## Iteration 002 — numba 1-D distance kernel `lnI_direct(a, b, xmin, xmax)` (accepted)

Codex report (summarized): `dpmarg/kernel1d.py` integrates in the peak-centred log coordinate
`v = ln(x/r)` with 24-point Gauss–Legendre panels whose widths start at
`min(r/4, 1/sqrt(b), 1/|f'(r)|)` and double up to 0.5; the window is cut where the exponent drop
reaches `H = 1 - ln(1e-12) + 4 ln(xmax/xmin)`, with a convexity argument certifying a relative
truncation `<= 1e-12` (no erfcx/erfcinv needed). `reference.py` gained `lnI_ref` sharing the inner
integral with `lnL_marg_ref` (plus an offset-coordinate path for layers thinner than `1e-7 r`,
where `x - r` cancels); the 40 regression values are unchanged (max change 0.0).

Orchestrator verification:
- Read `kernel1d.py` (window bound, panel scheme, extreme fallback) and the `reference.py` diff.
- `python -m pytest tests -q`: 262 passed in 75 s.
- `python scripts/check_kernel1d.py` re-run: worst `|Delta lnI|` per regime for `(0.1, 100)` /
  `(0.1, 10)`: below `xmin` 8.8e-11, inside 6.2e-13, above `xmax` 6.7e-10 / 8.3e-10, `b = 0`
  3.0e-10 / 8.3e-10 (target 1e-8). The worst cells (`a ~ 1e5–1e6`, `b << 1`) are the thin
  `xmax`-endpoint layer where the scipy reference itself is near its `rtol = 1e-11` limit.
- Cost: median 3.2–7.5 µs per call (6–11 panels x 24 nodes = 150–260 exponentials). A method-2
  call with ~64 theta nodes will therefore cost ~0.3–0.6 ms with the direct kernel; the 2-D table
  of this kernel (iteration 004) is what brings it to the µs range.
- Next: iteration 003 — method 2 (adaptive theta quadrature over the kernel), accuracy and node
  count first, speed later via the table.
