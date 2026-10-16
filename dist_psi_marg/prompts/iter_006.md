# Iteration 006 — Method 3: Gaussian (Laplace) approximation along the stiff direction; SNR breakdown

You are working in `dist_psi_marg/`. Read `AGENTS.md`, `TASK.md` §2–§3 and §7, `dist_psi_marg.tex`
Sec. 2–3 (the `(s, t)` principal-axis form and the open question about the regulator `k`),
`dpmarg/params.py`, `dpmarg/method2.py`, `dpmarg/table1d.py` (`ln_gaussian`, `_erfcx`), and the
stored reference sets `data/regression_points.json`, `data/method2_random_reference.json`.
Report in the final message; do not edit `TASK.md`, `PROGRESS.md`, `AGENTS.md`, `prompts/`.

## Question to answer (TASK.md §3 method 3)

At high SNR the likelihood is a narrow 2-D Gaussian in the principal coordinates
`(s, t)` (`x = r cos theta`, `y = r sin theta`, rotated by `xi`):
`lnL = lnL_peak - lam1 (s - sbar)^2/2 - lam2 (t - tbar)^2/2`, `sbar = R cos(eta)/lam1`,
`tbar = R sin(eta)/lam2`, `lnL_peak = R^2 cos^2(eta)/(2 lam1) + R^2 sin^2(eta)/(2 lam2)`, and
`lnL_marg = norm + ln iint_{xmin <= r <= xmax} r^-5 exp(lnL) ds dt`, `r = sqrt(s^2 + t^2)`,
`norm = ln[3/(2 pi (xmin^-3 - xmax^-3))]`.
Treat the stiff direction `s` (`lam1 >= lam2`) by a Laplace/Gaussian approximation, keep the
weak direction `t` exact (1-D numerical quadrature; optionally a lookup table), and determine
below which SNR the approximation breaks (`|Delta lnL_marg| > 1e-3` and `> 1e-2`), as a function
of the regime. Do not introduce the tex's regulator `k`: handle the annulus explicitly.

## Method (implement these variants; all numba)

1. **2-D Laplace (closed form, interior peak only)** `lnL_marg_m3_2d(..., order)`:
   `norm + lnL_peak + ln(2 pi) - (ln lam1 + ln lam2)/2 - 5 ln rbar` plus the order-2 (and, if
   cheap, order-4) Laplace corrections from the Taylor expansion of `g(s, t) = -5 ln r` at
   `(sbar, tbar)`: `1 + [g_ss + g_s^2]/(2 lam1) + [g_tt + g_t^2]/(2 lam2) + ...` (derive and
   document; verify numerically on a smooth interior case). Return NaN (or a flag) when the
   peak is within `3/sqrt(lam)` of a boundary in either direction or outside the annulus.
2. **Laplace in `s`, exact in `t`** `lnL_marg_m3(..., order, n_t)`:
   For fixed `t`, the allowed `s` range is `S(t) = {s : xmin^2 <= s^2 + t^2 <= xmax^2}` (one or
   two intervals). Let `s*(t)` be the maximizer of `-lam1 (s - sbar)^2/2 - 5 ln r` restricted to
   `S(t)` (clip `sbar` to the allowed intervals; pick the better interval). Approximate
       int_{S(t)} exp(-lam1 (s - sbar)^2/2) r^-5 ds
       ~ r(s*, t)^-5 * [ truncated-Gaussian integral over S(t) ]            (order 0)
       * [1 + corrections from the Taylor expansion of -5 ln r at s* with the truncated-Gaussian
          moments]                                                           (order 2, 4)
   using `ln_gaussian`-style stable erf/erfcx evaluation for the truncated Gaussian and its
   moments (document the formulas). Then integrate over `t`:
       lnL_marg ~ norm + lnL_peak + ln int exp(-lam2 (t - tbar)^2/2) * [s-result](t) dt
   with a fixed-size rule: locate the maximum of the log-integrand in `t`, use a sinh-stretched
   Gauss–Legendre rule with `n_t` nodes (24–48) over the allowed `t` range, breakpoints at the
   annulus cuts `|t| = xmin`, `|t| = xmax`; accumulate in log space.
3. **Cost**: target a few µs per call for variant 2 and ~0.1 µs for variant 1.

## Deliverables

1. `dpmarg/method3.py` with the two variants, docstrings stating the formulas, orders and
   domains of validity.
2. `scripts/check_method3.py`:
   - accuracy on the stored sets (regression + 300 random; reuse the stored reference values);
   - a structured SNR sweep: for `q = lam2/lam1 in {1, 0.3, 0.1, 0.01, 0.001, 0}`,
     `eta in {0, pi/8, pi/4, 3pi/8, pi/2}`, and peak positions `x0* in {0.3, 3, 30}` (interior),
     `0.7 xmin`, `1.3 xmin`, `0.8 xmax`, `1.5 xmax` (define `x0*` as in iteration 005: the
     distance peak at the dominant angle), SNR `rho` log-spaced from 1 to 1000 (25 values);
     reference values from `lnL_marg_ref` (cache them in `data/method3_sweep_reference.json`,
     ~5000 points x 0.15 s ~ 12 min; reuse if present), errors of every variant/order;
   - the breakdown SNR per `(q, eta, x0*)` and tolerance (1e-3, 1e-2): the smallest `rho` above
     which the error stays below the tolerance for all larger `rho` in the sweep;
   - figures: `figures/method3_error_vs_snr.png` (error vs SNR, panels per `q`, curves per
     order/variant, lines at 1e-3 and 1e-2), `figures/method3_breakdown_snr.png` (breakdown SNR
     vs `q` for the different `eta`, `x0*`);
   - timing: median µs per call per variant/order; `data/method3_check.json` (summary only).
3. `tests/test_method3.py`: variant 1 vs reference within 1e-6 on an interior high-SNR case
   (`rho ~ 300`), variant 2 order 0/2 agree with the reference within the sweep-derived
   tolerance at high SNR, the two variants agree with each other where both apply, finite
   outputs for `lam2 = 0`, `R = 0`, peaks outside the annulus, and the all-zero case.

## Rules (AGENTS.md)

Edit only inside this directory; no git state changes; no installs; no secrets; per-point dumps to
`data/*_full.json` (git-ignored); keep committed JSON small.

## Completion criteria / verification

- `python -m pytest tests -q` passes (keep under ~7 min; mark slow tests).
- `python scripts/check_method3.py` runs to completion and writes the figures and JSON.
- Final message: the breakdown-SNR table per regime and tolerance for each variant/order, the
  accuracy on the stored sets, timings, and a clear statement of where method 3 cannot be used
  (so that the final synthesis can decide on a fallback to method 2).
