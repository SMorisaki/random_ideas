# Iteration 009 — Method 2 with analytic peak prediction: Gauss–Hermite for sharp peaks, fixed grid otherwise

You are working in `dist_psi_marg/`. Read `AGENTS.md`, `TASK.md` §2–§3 and §7, `PROGRESS.md`
(iterations 003, 004, 008 — in particular the two geometry repairs and the cases
`seed-808 #44, #915` and `mock-posterior #124, #876`), `dpmarg/method2.py`, `dpmarg/table1d.py`,
`dpmarg/fast.py`, `scripts/final_benchmark.py` (it caches references in
`data/final_reference_full.json`; reuse that cache, do not recompute references that exist).
Report in the final message; do not edit `TASK.md`, `PROGRESS.md`, `AGENTS.md`, `prompts/`.

## Owner's request

Replace the empirical 64-point proxy scan + golden-section geometry of method 2 by an
**analytic prediction** of the theta peak(s) and width(s) from the Gaussian (Laplace)
approximation in distance, and then use
- a **Gaussian-adapted rule (Gauss–Hermite)** when the predicted peak is sharp, and
- a **fixed, precomputed theta grid** (uniform periodic trapezoid with precomputed trig factors,
  in the spirit of the regular N_phi = 128 phase grid with precomputed phase factors of
  Roulet et al., arXiv:2404.02435 §II.3) when it is broad.
Accuracy target unchanged (`|Delta lnL_marg| <= 1e-3`, goal 1e-4, on all stored sets and the
known failure cases); cost target: well below the current 21 µs median (geometry < 1 µs,
total ~5–10 µs).

## Analytic geometry (derive, verify, document)

Work in the rotated frame: `a(theta) = R cos(theta - eta)`, `b(theta) = E + S cos(2 theta)`,
`E = (lam1 + lam2)/2`, `S = (lam1 - lam2)/2`; pair `h = J(a) + J(-a)` on one period `[0, pi)`.
The Laplace proxy of `ln h` for an interior radial peak is
`p(theta) = R^2 cos^2(theta - eta)/(2 b) - 4 ln x0 - ln(b)/2 + const`, `x0 = a/b`;
for a peak clipped at `xmin` or `xmax` the first term becomes `a x_b - b x_b^2/2` and the prior
term `-4 ln x_b` (then `+ ln` of the endpoint-layer factor `1/|a - b x_b|` instead of `-ln b/2`).

1. **Closed-form interior maximum.** With `phi = 2 theta`,
   `fmax ∝ (1 + cos(phi - 2 eta))/(E + S cos phi)`; its stationarity condition reduces to
       E sin(phi - 2 eta) - S sin(phi) = S sin(2 eta)
       <=>  rho' sin(phi - delta) = S sin(2 eta),
       rho' = sqrt((E cos 2eta - S)^2 + E^2 sin^2 2eta),  delta = atan2(E sin 2eta, E cos 2eta - S),
   so `phi* = delta + asin(S sin 2eta / rho')` (the other root `delta + pi - asin(...)` is the
   minimum; check), `theta* = phi*/2 mod pi`, independent of `R`. (`rho'^2 - S^2 sin^2 2eta =
   (E - S cos 2eta)^2 >= 0`, so the asin is always defined; `rho' = 0` only for `lam2 = 0,
   eta = 0`: a flat ridge, decided by the prior term.) The orchestrator verified on 3000 interior
   cases with SNR 10–316 that this `theta*` differs from the scanned maximum by a median 0.8% of
   the width (99%: 21%), with the offset scaling as `O(1)/rho^2` as expected from the `O(1)`
   prior/width terms; one Newton step on the full proxy removes that offset.
2. **Other candidates.** Boundary-clipped regimes satisfy `R sin(theta - eta) = S x_b sin 2theta`
   (a quartic in `cos theta`); rather than solving it in closed form, refine by Newton (analytic or
   3-point finite-difference derivatives of the proxy, step `~1e-3 w`) from a small candidate set:
   the interior closed-form maximum (both branches), `theta = pi/2` (weak-axis seed — essential
   for the lam2 -> 0 boundary-layer islands found in iteration 008), `eta`, `0`, `eta + pi/2`.
   Keep every candidate whose refined proxy height is within 40 log units of the best, merged
   when closer than their widths. There must be no dense scan in the production path.
3. **Width.** `w = 1/sqrt(-p''(theta*))` from the analytic second derivative of the proxy
   (interior: differentiate the closed form; boundary: likewise), including the prior/width terms;
   cap at `w_max = 0.3`. Also estimate the non-Gaussianity from `p'''` and `p''''` if you use it to
   choose the Gauss–Hermite order.
4. **Validation of the geometry alone** (no kernel calls, cheap): on 1e5 random points from
   `regimes.random_params` (several seeds) plus the named cases and the iteration-008 failure
   cases, compare the analytic maxima/widths with the dense-scan geometry of `method2._geometry`:
   same global maximum within `0.2 w` (or a proxy-height difference < 1e-6), width ratio within
   a factor 1.5, and no missed secondary peak (any scan peak within 40 log units must have an
   analytic counterpart). Report the mismatch rate and the worst cases; a mismatch that changes
   `lnL_marg` by more than 1e-4 is a failure to fix.

## Quadrature rules

- **Sharp branch (`w < w_c`, `w_c ~ 0.2`, tune it):** Gauss–Hermite in `t = (theta - theta*)/(sqrt(2) w)`
  with `N_GH` nodes (precompute nodes/weights for N in {8, 12, 16, 20, 24, 32} as module
  constants via `numpy.polynomial.hermite.hermgauss` at import time):
      int h dtheta ≈ sqrt(2) w * sum_j omega_j exp(t_j^2 + ln h(theta* + sqrt(2) w t_j)),
  accumulated in log space. With two significant peaks use one cluster per peak when they are
  separated by more than ~6 (w1 + w2); otherwise fall back to the existing two-interval sinh
  Gauss–Legendre rule (count how often). The GH nodes must stay inside one period around `theta*`;
  if `sqrt(2) w t_max > pi/2` the branch is not sharp enough — use the grid instead.
  Choose the smallest `N_GH` that meets 1e-4 on all sets; report the error vs `N_GH` curve
  (8–32) and compare with the current sinh-trapezoid at the same number of kernel calls.
- **Broad branch (`w >= w_c`):** fixed uniform periodic grid `theta_j = j pi/N`, `j = 0..N-1`,
  trapezoid weights `pi/N`, with precomputed `cos theta_j`, `sin theta_j`, `cos 2theta_j` so that
  `a_j = R (cos theta_j cos eta + sin theta_j sin eta)`, `b_j = E + S cos 2theta_j` cost no
  transcendental calls. Choose `N` from the predicted width (spacing `<= w/2`, minimum 16,
  maximum 64, rounded to the precomputed sizes {16, 24, 32, 48, 64}); report the error vs `N`.
- Keep the certified `J(-a)` skip and the table lookups (`table1d._lnJ_pair_table_count`).
  No nested 16/32/64 refinement in the production path (it is replaced by the analytic choice);
  keep an optional diagnostic that doubles the nodes to measure the remaining quadrature error.

## Deliverables

1. `dpmarg/geometry.py`: analytic candidates, Newton refinement, widths, derivatives; numba.
2. `dpmarg/method2.py`: `lnL_marg_m2_table(..., rule='analytic')` (new default if it wins) with
   the old `rule='scan'` retained; diagnostics returning branch (GH/grid/fallback), `N`, `w`.
   Update `dpmarg/fast.py` to use the winning rule; `use_laplace` semantics unchanged.
3. `scripts/check_geometry.py` (step 4 above) and `scripts/check_method2b.py`: accuracy on the
   40 regression, 300 stored random, 2000 fresh (seed 808) and 2000 sampler-like points using the
   cached references, plus the iteration-008 failure cases; error vs `N_GH` and vs grid `N`;
   median/99% µs per call for `scan` vs `analytic`, with the geometry share; branch fractions.
   Figures `figures/method2b_error_vs_nodes.png`, `figures/method2b_cost.png`; summaries
   `data/method2b_check.json` (small) and `data/method2b_summary.md`; per-point dumps to
   `data/*_full.json`.
4. `tests/test_method2b.py`: closed-form `theta*` vs brute-force maximization of the proxy
   (1e-10 in height) on 200 random cases; geometry agreement on the named cases; the iteration-008
   failure cases within 1e-4; `analytic` vs `scan` within 2e-4 on the regression set; all-zero
   case exactly 0; finite for `lam2 = 0`, `R = 0`, exterior peaks.
5. Append a section "Iteration 009: analytic geometry, Gauss–Hermite / fixed grid" to `REPORT.md`
   with the measured results and update its recommendation if the new rule wins.

## Rules (AGENTS.md)

Edit only inside this directory; no git state changes; no installs; no secrets; big files and
per-point dumps stay out of git.

## Completion criteria / verification

- `python -m pytest tests -q` passes (keep under ~8 min; mark slow tests).
- `python scripts/check_geometry.py` and `python scripts/check_method2b.py` complete and write the
  figures/summaries; worst `|Delta lnL_marg| <= 1e-3` (report 1e-4 compliance) on every set and on
  the iteration-008 failure cases.
- Final message: geometry mismatch statistics, the chosen `w_c`, `N_GH`, grid `N`, branch
  fractions, accuracy and cost tables (`scan` vs `analytic`), fallback rate, and anything that
  did not meet the target with the reason.
