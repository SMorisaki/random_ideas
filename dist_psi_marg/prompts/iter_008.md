# Iteration 008 — synthesis: `dpmarg.fast` dispatcher, final benchmarks, REPORT.md

You are working in `dist_psi_marg/`. Read `AGENTS.md`, `TASK.md` (all sections, especially §4
completion criteria), `PROGRESS.md` (iterations 001–007), `TABLE1D.md`, `TABLE4D.md`, `METHOD3.md`,
`data/method2_summary.md`, `data/method3_summary.md`, `data/method1_slices.json`,
`data/method1b_slices.json` (iteration 007: transition-aligned coordinates and smooth
normalizations did not change the verdict — every design still needs >512 nodes per axis at the
99th percentile), and the modules `dpmarg/method2.py`, `dpmarg/table1d.py`,
`dpmarg/method3.py`. Report in the final message; do not edit `TASK.md`, `PROGRESS.md`,
`AGENTS.md`, `prompts/`.

## Goal

Turn the findings into (1) a production entry point, (2) a final apples-to-apples comparison of
the three methods on identical parameter sets, and (3) `REPORT.md`, the deliverable of TASK.md §4.

## 1. `dpmarg/fast.py`

- `lnL_marg_fast(A, B, C, D, E, xmin, xmax, table, use_laplace=True) -> float` (numba):
  method 2 with the 1-D table as the general path; when `use_laplace`, first try the 2-D Laplace
  closed form (order 4) **gated** by the measured breakdown thresholds: interior guard from
  `method3`, `lam2 > 0`, and `rho >= margin * rho_break(q)` where `rho_break(q)` is a monotone
  interpolation of the iteration-006 1e-3 thresholds in `q = lam2/lam1` (18 at `q >= 0.3`, 42 at
  0.1, 133 at 0.01, 422 at 0.001; treat `q < 1e-3` as never) and `margin ~ 1.5`; make the gate
  parameters module constants with docstrings. `rho` must be computed cheaply (closed form for
  an interior peak: `rho^2 = 2 lnL_peak`; the guard already implies the peak is interior).
- `lnL_marg_fast_array(parameters, table, use_laplace=True)` numba loop; a diagnostics variant
  returning which branch served each call.
- `dpmarg/__init__.py`: export the public API (`lnL_marg_ref`, `lnL_marg_m2`, `lnL_marg_m2_table`,
  `lnL_marg_m3_2d`, `lnL_marg_fast`, `build_table`, `load_table`, `reduce_params`, `snr`).

## 2. Final comparison (`scripts/final_benchmark.py`)

On identical sets — the 40 regression points, the 300 stored random points, 2000 fresh random
points (seed 808; reference via `lnL_marg_ref`, cache in `data/final_reference_full.json`,
git-ignored, ~5 min), and a "sampler-like" set of 2000 points drawn from a mock posterior
(interior peaks, `rho` log-uniform in `[5, 50]`, `q` uniform in `[0.05, 1]`, `eta` uniform;
state the recipe) — report for each method (method 2 direct kernel, method 2 table, method 3
2-D order 4 where applicable, dispatcher with and without the Laplace branch):
max / 99% / median absolute error vs the reference, fraction of points with error `> 1e-3` and
`> 1e-2`, median and 99% µs per call (warmed, single thread), the branch fractions for the
dispatcher, table build time and memory. Figures: `figures/final_error_cdf.png` (error CDFs per
method and set), `figures/final_cost_vs_error.png` (median µs vs worst error, one marker per
method/set). Summary in `data/final_benchmark.json` (small) and `data/final_benchmark.md`.

## 3. `REPORT.md`

Written for the project owner. English main text with a short Japanese executive summary
(「要約」) at the top. Sections:
1. Problem and notation (Eq. 17 of the tex, reduced parameters, normalization in `(xmin, xmax)`,
   scaling symmetry, SNR definition) — concise, with pointers to `TASK.md`.
2. Reference implementation and validation (two independent integrators, regression set).
3. The three methods: what was implemented, the key design decisions, and the measured results
   (tables from §2 above plus the per-iteration summaries): method 2 (1-D kernel, `(s, v)`
   table, adaptive theta rule), method 1 (both slice studies, prototype errors, why a regular 4-D
   grid fails: the multi-scale transitions along `R` and the cusps of `lnL_max`; whether the
   transition-aligned coordinates of iteration 007 change the verdict), method 3 (breakdown SNR
   per regime, why the 1-D variant has no niche, when the 2-D formula is safe).
4. Recommendation: which method (or combination) to use, the table grid design for the 1-D
   table (coordinates, `sigma0`, `bref`, `smax`, `bmax`, guard), the dispatcher gate, expected
   cost per call, and how to build tables for other priors (`xmax/xmin` ratios, `Dref` scaling).
5. Limitations and future work (guard cells, geometry cost, `bmax`, priors other than
   `D_L^2`, higher-order Laplace, GPU/vectorization), reproduction commands, file map.
Include the figures (relative paths) and keep the numbers consistent with the JSON summaries.

## 4. Tests

`tests/test_fast.py`: dispatcher within `1e-3` of the reference on the regression set and the
stored random set (record the worst), Laplace branch only taken when the gate passes (check the
diagnostics), identical results to `lnL_marg_m2_table` when `use_laplace=False`, the all-zero case
gives 0, finite outputs for `lam2 = 0`, `R = 0`, exterior peaks.

## Rules (AGENTS.md)

Edit only inside this directory; no git state changes; no installs; no secrets; big files and
per-point dumps stay out of git.

## Completion criteria / verification

- `python -m pytest tests -q` passes (keep the suite under ~8 min; mark slow tests).
- `python scripts/final_benchmark.py` completes and writes the figures and summaries.
- `REPORT.md` exists and its numbers match `data/final_benchmark.json`.
- Final message: the comparison table, the dispatcher branch fractions and worst error, and
  anything in TASK.md §4 that is not met (with the reason).
