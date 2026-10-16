# Iteration 004 — 2-D lookup table for the distance kernel lnI(a, b; xmin, xmax)

You are working in `dist_psi_marg/`. Read `AGENTS.md`, `TASK.md` §2–§3 and §7, `dpmarg/kernel1d.py`
(`lnI_direct`, `fmax_1d`), `dpmarg/method2.py` (iteration 003: `_compute` calls `kernel1d._lnJ_pair_count` at each node;
`_geometry` costs ~10–20 µs of proxy scans, which will dominate once the kernel is a lookup —
reduce it where you can without losing the iteration-003 accuracy) and `dpmarg/reference.py`
(`lnI_ref`). `distmarg.pdf` Sec. 3 (the `(s, t)` coordinates) is restated below.
Do not edit `TASK.md`, `PROGRESS.md`, `AGENTS.md`, `prompts/`; report in the final message.

## Goal

Replace `lnI_direct` (3–7 µs) by a table lookup of ~0.1–0.2 µs with `|Delta lnI| <= 1e-5`
(hard limit 1e-4) on the whole `(a, b)` domain used by method 2, for fixed `(xmin, xmax)`, with
a table of at most ~1e6 float64 entries, built in well under a minute with `lnI_direct`.
Then plug it into method 2 (`lnL_marg_m2` with a `use_table` switch or a separate function) and
show that the lnL_marg accuracy of iteration 003 is unchanged while the cost drops to the µs range.

## Design to implement and measure (compare with at least one alternative and justify)

1. **Coordinates.** distmarg.pdf tabulates `ln I(x0, b)` on a regular grid in
   `s(x0) = asinh(sqrt(bmax) (x0 - xmin)) - asinh(sqrt(bmax) (xmax - x0))`, `t(b) = asinh(b / bref)`.
   Generalize so that `b -> 0` and `|a| -> large` are regular: with the endpoint slopes
   `alpha = a - b xmin = f'(xmin)`, `beta = a - b xmax = f'(xmax)` and a scale
   `sigma = max(sqrt(b), sigma0)` (`sigma0 ~ 1/(xmax - xmin)` or similar; tune it), use
       s = asinh(alpha / sigma) + asinh(beta / sigma),       v = asinh(b / bref).
   For large `b` this is distmarg's `s` with `sqrt(b)` in place of `sqrt(bmax)`; for `b -> 0`
   it tends to `2 asinh(a / sigma0)`, finite and smooth. Choose `bref` so that `lnI(a, bref) ~
   lnI(a, 0)` (distmarg Eq. 42 logic) and a `v` range up to `b = 1e8` (and `1e10` if cheap).
2. **Tabulate a smooth remainder, not lnI itself.** Let
       Itilde(a, b) = r^-4 * int_xmin^xmax exp(f(x) - fmax) dx
   with `r` the clipped peak (`fmax_1d` location) and the prior frozen at `r`. The integral is
   analytic: for `b > 0` it is `sqrt(pi/(2b)) [erf(sqrt(b/2)(xmax - x0)) - erf(sqrt(b/2)(xmin - x0))]`
   times `exp(b x0^2/2 - fmax)`, and for `b -> 0` it is `(exp(a xmax) - exp(a xmin))/a` times
   `exp(-fmax)`; implement `ln Itilde` stably in numba (scaled complementary error function
   `erfcx` for arguments where `erf` differences cancel, `expm1`/`log1p` for `b (xmax-xmin)^2 << 1`,
   and the limit `a -> 0`). Then tabulate `Delta(a, b) = lnI(a, b) - ln Itilde(a, b)`
   `= ln <(r/x)^4>` under the truncated Gaussian weight: bounded (between `-4 ln(xmax/r)` and
   `4 ln(r/xmin)`), close to 0 wherever the weight is narrow, and smooth. Show the error maps of
   interpolating `Delta` versus interpolating `lnI` directly on the same grid.
3. **Interpolation.** Regular grid in `(s, v)`; bicubic (Catmull–Rom or cubic B-spline with
   precomputed coefficients) in numba, compared with bilinear on a denser grid. Report the grid
   `(ns, nv)` needed for `1e-5` and `1e-4` for each variant and the lookup time.
4. **Outside the table.** `|s| > smax`: use the asymptotic endpoint-layer expansion
   (`lnI ~ -ln|f'(endpoint)| - 4 ln(endpoint) + series in 1/(|f'| endpoint) and b/f'^2`) or fall
   back to `lnI_direct`; `b > bmax`: fall back to `lnI_direct`. Document the cut-offs and make the
   fallback rate measurable (a counter in the diagnostics path).
5. **API.** `dpmarg/table1d.py`: `build_table(xmin, xmax, ns, nv, smax, bmax, ...)` returning the
   table as a tuple of numpy arrays plus scalar grid parameters that numba functions accept as
   arguments (e.g. `(values, s0, ds, ns, v0, dv, nv, sigma0, bref, xmin, xmax)`), `save/load`
   (`.npz` under `data/`, git-ignored), `lnI_table(a, b, table...)` numba scalar, and an array
   version. `dpmarg/method2.py`: `lnL_marg_m2_table(A, B, C, D, E, xmin, xmax, table...)` reusing
   the iteration-003 node logic (factor the kernel call so both variants share the code), plus an
   array loop for benchmarking. Report the fallback count per call in a diagnostics variant.

## Deliverables

1. `dpmarg/table1d.py` as above; `scripts/build_table1d.py --xmin 0.1 --xmax 100` writing
   `data/table1d_xmin0.1_xmax100.npz` (and the `(0.1, 10)` table).
2. `scripts/check_table1d.py`: error maps `lnI_table - lnI_direct` on the iteration-002 grid plus
   1e5 random `(a, b)` from method 2's actual node distribution (draw `(A..E)` from
   `regimes.random_params` and collect the `(a, b)` that `lnL_marg_m2` evaluates); figures
   `figures/table1d_error_map.png`, `figures/table1d_grid_convergence.png`; `data/table1d_check.json`
   (worst/99% error per regime, grid size, build time, lookup ns, fallback rate).
3. `scripts/check_method2.py`: add the table variant (same regression/random sets as iteration 003;
   reuse `data/regression_points.json` and `data/method2_random_reference.json`, do not recompute
   the reference); report accuracy and median µs per call for direct vs table, and the share of the
   call spent in the geometry (`_geometry`) vs the kernel. Write per-point dumps to
   `data/*_full.json` (git-ignored) and keep `data/table1d_check.json` / `data/method2_check.json`
   small (summaries only).
4. `tests/test_table1d.py`: table vs `lnI_direct` within `1e-4` (and record the worst) on ~300
   points including `b = 0`, `a = 0`, both sides outside, large `b`; `ln Itilde` vs a scipy
   implementation (`1e-12`); fallback path exercised; method 2 with table vs without within `2e-4`
   on the regression set; everything finite.

## Rules (AGENTS.md)

Edit only inside this directory; no git state changes; no installs; no secrets. Keep generated
tables out of git (`.gitignore` already excludes `data/*.npz`).

## Completion criteria / verification

- `python -m pytest tests -q` passes (keep the total under ~5 min).
- `python scripts/build_table1d.py` builds both tables in < 60 s each; `python scripts/check_table1d.py`
  reports worst `|Delta lnI| <= 1e-4` (goal `1e-5`) and lookup time; `python scripts/check_method2.py`
  shows method 2 + table within `1e-3` of the reference on all sets with median µs per call.
- Final message: files, test summary, chosen coordinates and `sigma0`, `bref`, grid size, build
  time, lookup ns, fallback rate, worst error per regime for `Delta`- vs direct-`lnI` tabulation,
  and the method-2 accuracy/cost with the table.
