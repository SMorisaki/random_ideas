# Iteration 003 — Method 2: adaptive theta quadrature over the 1-D distance kernel (numba)

You are working in `dist_psi_marg/`. Read `AGENTS.md`, `TASK.md` §2–§3, §7, `dpmarg/params.py`,
`dpmarg/reference.py`, `dpmarg/kernel1d.py` (iteration 002: `lnI_direct(a, b, xmin, xmax)`, `fmax_1d`,
`_lnI_with_panels`; it costs 3–7 µs per call, so a 64-node method-2 call will cost ~0.3–0.6 ms
with the direct kernel — that is expected; this iteration is about accuracy and node count, the
2-D table of the kernel comes in iteration 004). Do not edit `TASK.md`, `PROGRESS.md`, `AGENTS.md`, `prompts/`; report in the final message.

## Goal

`lnL_marg_m2(A, B, C, D, E, xmin, xmax) -> float` in numba (`@njit(cache=True)`), using
`lnI_direct` at adaptively placed theta nodes, with `|Delta lnL_marg| <= 1e-3` (goal 1e-4) on the
whole regression set and on >= 300 random points (`regimes.random_params`, several seeds), and the
smallest number of kernel evaluations that achieves it (target <= 64 typical, <= 128 worst).

## Mathematical structure to exploit (verify numerically before relying on it)

- Work in the rotated frame: `(lam1, lam2, R, eta) = reduce_params(...)` (eta already folded to
  `[0, pi/2]`; the value is invariant), `a(theta) = R cos(theta - eta)`,
  `b(theta) = lam2 + (lam1 - lam2) cos^2 theta`.
- `theta -> theta + pi` maps `a -> -a`, `b -> b`. Hence
  `int_0^{2pi} J(a(theta), b(theta)) dtheta = int_0^{pi} [J(a, b) + J(-a, b)] dtheta`, and the pair
  `h(theta) = J(a, b) + J(-a, b)` is pi-periodic. Provide `lnJ_pair(a, b, xmin, xmax)` =
  `ln(J(a,b) + J(-a,b))` in `kernel1d.py` computed with `fmax_1d`/`lnI_direct`, skipping the
  second term when `fmax(-a, b) - fmax(a, b)` is below `-40` (document the bound).
- Peak structure of `ln h(theta)` on one period: use the cheap analytic proxy
  `p(theta) = max(fmax(a, b), fmax(-a, b)) - 4 ln r*(theta)` (the prior factor matters when the
  distance peak lies outside `[xmin, xmax]`). There is usually one dominant maximum per period, but
  two comparable maxima occur, e.g. when the distance peak is below `xmin` and the `R xmin cos`
  term competes with the `-(lam1 - lam2) xmin^2 cos^2 theta / 2` term. Find all local maxima of `p`
  on a dense scan (e.g. 128 points over `[0, pi)`), refine the top two by golden-section/Brent
  (about 20 iterations each), estimate each width from the numerical curvature
  `w_k = min(w_max, c / sqrt(|p''(theta_k)|))` with `w_max ~ 0.3`, `c ~ 1`, and keep a secondary
  peak only if its height is within ~40 of the global maximum in `ln`.
- Quadrature (design yourself, but compare at least these two regimes and justify the switch):
  (i) broad regime (`w >= w_c`, roughly `rho <~ 4`): periodic trapezoid rule with `N` uniform nodes
      on `[0, pi)`; it is spectrally accurate for the smooth periodic integrand;
  (ii) sharp regime: sinh-stretched trapezoid around the dominant peak,
      `theta(tau) = theta* + w sinh(tau)`, `tau` uniform on `[-T, T]` with `w sinh(T) = pi/2`,
      integrand `h(theta(tau)) w cosh(tau)`, `N` nodes; add the first Euler–Maclaurin endpoint
      correction `(dtau^2/12)(F'(-T) - F'(T))` or show it is negligible; when a second significant
      sharp peak exists, split the period at the valley points between the peaks and apply the
      sinh rule on each sub-interval (asymmetric `T-`, `T+`).
  Evaluate `h` through `lnJ_pair`, accumulate in log space (subtract the maximum of
  `ln h` over the nodes before exponentiating).
- `lnL_marg = ln[3/(2 pi (xmin^-3 - xmax^-3))] + ln int_0^pi h dtheta`.

## Deliverables

1. `dpmarg/kernel1d.py`: add `lnJ_pair(a, b, xmin, xmax)` (+ test vs `lnI_ref`).
2. `dpmarg/method2.py`: `lnL_marg_m2(A, B, C, D, E, xmin, xmax, n_nodes=64)` numba scalar
   function; a helper returning the node set (for diagnostics/plots); a numba loop over arrays.
3. `scripts/check_method2.py`: compare with `lnL_marg_ref` on `data/regression_points.json` (use the
   stored `lnL_marg_ref`, do not recompute) and on 300 random points (seeds 2, 3, 4; recompute the
   reference, ~0.15 s each); report the absolute error distribution (max, 99%, median) overall and
   per regime (low/intermediate/high SNR, `lam2/lam1` bins, peak inside/outside), for
   `n_nodes in (16, 24, 32, 48, 64, 96, 128)`; save `figures/method2_error_vs_nodes.png`,
   `figures/method2_error_scatter.png` (error vs actual SNR, coloured by `lam2/lam1`), and
   `data/method2_check.json`. Timing: median microseconds per call after warm-up for each `n_nodes`
   (numba, single thread), and the number of kernel evaluations per call.
4. `tests/test_method2.py`: regression set within `1e-3` (and report the worst), symmetry checks
   (rotation `xi`, reflection), `n_nodes=64` vs `128` difference `< 1e-4` on 50 random points,
   finite output for `b = 0`, `R = 0`, and the all-zero case (`== 0` within `1e-12`).

## Rules (AGENTS.md)

Edit only inside this directory; no git state changes; no installs; no secrets.

## Completion criteria / verification

- `python -m pytest tests -q` passes (total under ~4 min; mark slow ones).
- `python scripts/check_method2.py` reports max abs error `<= 1e-3` on the regression set and on the
  random set at `n_nodes = 64`, and the median microseconds per call; figures and JSON exist.
- Final message: files, test summary, the error table vs `n_nodes` (max/99%/median, per regime),
  timing, number of kernel evaluations, which cases dominate the error and why, and the design
  decisions (regime switch, endpoint correction, secondary-peak handling).
