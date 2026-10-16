# Iteration 001 — package skeleton, brute-force reference, regression point set

You are working in `dist_psi_marg/` (the current directory). First read `AGENTS.md`, `TASK.md`
(sections 1, 2, 5, 7) and `dist_psi_marg.tex` (plain LaTeX; it is the specification). The PDF
`distmarg.pdf` describes the distance-only marginalization; its key formulas are restated below, so
you do not need to parse the PDF.

## Problem statement (authoritative for this iteration)

Coefficients `A, B, C, D, E` (floats, with `E >= sqrt(C^2 + D^2)` mathematically; clip tiny
numerical violations), prior limits `xmin < xmax` (`xmin = Dref/Dmax`, `xmax = Dref/Dmin`).
With `r = Dref/D_L in [xmin, xmax]` and `theta = 2 psi in [0, 2 pi)`:

    lnL(r, theta) = r (A cos theta + B sin theta) - 0.5 r^2 (C cos 2theta + D sin 2theta + E)

    lnL_marg = ln[ 3 / (2 pi (xmin^-3 - xmax^-3)) ] + ln( int_xmin^xmax dr int_0^{2pi} dtheta r^-4 exp(lnL) )

Checks: A=B=C=D=E=0 gives lnL_marg = 0 exactly. For fixed theta the integrand in r is
`r^-4 exp(a r - b r^2 / 2)` with `a = A cos theta + B sin theta`, `b = C cos 2theta + D sin 2theta + E >= 0`
(same form as distmarg.pdf Eq. 1 with `x = r`). Its exponent maximum on `[xmin, xmax]` is
(distmarg Eq. 6, `x0 = a/b` when `b > 0`):
`fmax = b x0^2/2 - b (xmin - x0)^2/2` if `x0 < xmin`; `b x0^2/2` if `xmin <= x0 <= xmax`;
`b x0^2/2 - b (x0 - xmax)^2/2` if `x0 > xmax`. For `b = 0` the exponent is linear and the maximum is
at an endpoint. Peak width in r is about `1/sqrt(b)`.

Reduced parameters (TASK.md §2, tex Sec. 2): `R = sqrt(A^2+B^2)`, `zeta = atan2(B, A)`,
`S = sqrt(C^2+D^2)`, `2 xi = atan2(D, C)`, `lam1 = E + S >= lam2 = E - S >= 0`, `eta = zeta - xi`.
After rotating `theta -> theta + xi`: `a = R cos(theta - eta)`, `b = lam1 cos^2 theta + lam2 sin^2 theta`.
Symmetries of the full-circle theta integral: `theta -> theta + pi` flips the sign of `a`
(so `eta ~ eta + pi`), `theta -> -theta` maps `eta -> -eta`. Hence `eta` can be reduced to `[0, pi/2]`.
SNR definition: `rho^2 = 2 max(0, max_{r, theta} lnL)`.

## Deliverables

1. `dpmarg/__init__.py` and `dpmarg/params.py`
   - `reduce_params(A, B, C, D, E) -> (lam1, lam2, R, eta)` with `eta` reduced to `[0, pi/2]`;
     document the symmetry argument in the docstring. Clip `lam2` to `>= 0` if it is slightly
     negative due to rounding (and expose a `check_coefficients` helper that reports the violation
     `E^2 - C^2 - D^2 < 0` beyond a relative tolerance).
   - `expand_params(lam1, lam2, R, eta, xi=0.0) -> (A, B, C, D, E)` (inverse, with an arbitrary
     rotation `xi`) for tests.
   - `lnL_max(A, B, C, D, E, xmin, xmax) -> (fmax, r_star, theta_star)`: global maximum of `lnL` on
     the domain (analytic in `r` per theta via the formula above, then a careful 1-D maximization over
     theta: dense scan of >= 1024 points followed by bounded local refinement). Also `snr(...)`.
   - Vectorization is not needed here; keep it simple and correct.

2. `dpmarg/reference.py` — two independent brute-force implementations.
   - `lnL_marg_ref(A, B, C, D, E, xmin, xmax, rtol=1e-10) -> float` in float64 with scipy:
     nested adaptive quadrature. Inner integral over `r` for fixed theta with `scipy.integrate.quad`
     after subtracting the per-theta exponent maximum `fmax(theta)`; pass the peak location and
     `+-(1, 3, 10)` widths (clipped to the interval) as `points`, `limit >= 200`. The outer integral
     over theta integrates `exp(g(theta) - gmax)` where `g(theta) = fmax(theta) + ln I(theta)`; locate
     the maxima of `g` (dense scan >= 720 points plus local refinement; there can be two lobes at
     `theta*` and `theta* + pi`), subtract `gmax`, and pass the lobe centres and `+-` several widths
     (width about `1/rho`) as `points`. Return `lnL_marg`. It must be robust for `lam1 = 1e8`,
     `rho = 300` (inner width ~1e-4, outer width ~3e-3 rad) and for `b = 0`, `R = 0`.
     Target: well below 1 s per call for typical points; print nothing.
   - `lnL_marg_mp(A, B, C, D, E, xmin, xmax, dps=25) -> float` with `mpmath` (installed):
     an independent nested `mp.quad` (tanh-sinh) implementation in which the integration intervals
     are explicitly split at the peak locations and at `+-(1, 3, 10, 30)` widths, with only a
     constant shift of the exponent (exact in log space). It may be slow (seconds to a minute per
     point); it is used only on the regression set.
   - Both functions must accept Python floats and return a float.

3. `dpmarg/regimes.py` — documented parameter sets.
   - `named_cases(xmin=0.1, xmax=100.0) -> list[dict]` with keys `name, A, B, C, D, E, xmin, xmax,
     note`, covering at least: zero signal; `A=B=0` with `E>0`; psi-independent (`C=D=0`) at low and
     high SNR; degenerate `lam2 = 0` (`S = E`) with `eta = 0`, `pi/4`, `pi/2`; very large
     `lam1 = 1e8` with `rho in {0.5, 10, 100, 300}`; distance peak below `xmin` and above `xmax`
     (`x0 = R/lam` outside `[xmin, xmax]`) at moderate and high SNR; low SNR `rho in {0.3, 1, 3}`;
     a case where the theta profile `g(theta)` has two comparable lobes; a strongly anisotropic case
     `lam2/lam1 = 1e-3` with `eta` near `pi/2`.
   - `random_params(n, seed, xmin=0.1, xmax=100.0) -> list[dict]`: `lam1` log-uniform in
     `[1e-4, 1e8]`, `q = lam2/lam1` uniform in `[0, 1]` (10% exactly 0, 10% exactly 1),
     `rho = R/sqrt(lam1)` log-uniform in `[0.1, 300]` (10% exactly 0), `eta` uniform in `[0, pi/2]`,
     random rotation `xi` in `[0, pi)`, expanded to `(A, ..., E)`.

4. `scripts/make_regression_points.py`: evaluates `lnL_marg_ref` and `lnL_marg_mp` for all named
   cases and 20 random points (seed 1), prints a table (name, lam1, lam2, R, eta, rho, ref, mp,
   |ref - mp|, seconds), and writes `data/regression_points.json` (a list of dicts with the inputs,
   both values and the reduced parameters). Keep the JSON human-readable; it is committed.

5. `tests/test_reference.py` (pytest, from this directory `python -m pytest tests -q`):
   - zero signal gives `|lnL_marg| < 1e-12`;
   - rotation invariance: `expand_params(..., xi)` for several `xi` gives the same `lnL_marg_ref`
     within `1e-9`; reflection `(B, D) -> (-B, -D)` gives the same value;
   - `reduce_params(expand_params(...))` round-trips `lam1, lam2, R, eta` to 1e-12 (eta modulo the
     symmetries);
   - psi-independent case `C = D = 0`: `int_0^{2pi} exp(r R cos(theta - zeta)) dtheta = 2 pi I0(R r)`,
     so `lnL_marg = ln[3/(xmin^-3 - xmax^-3)] + ln int r^-4 I0(R r) exp(-E r^2/2) dr`; implement this
     1-D check with `scipy.special.i0e` (log-scaled) and compare to `lnL_marg_ref` at low and high SNR
     within `1e-8`;
   - `lnL_marg_ref` vs `lnL_marg_mp` within `1e-8` on a subset of the named cases (mark the slow
     ones with `@pytest.mark.slow` and keep the default run under ~2 minutes);
   - `lnL_max` agrees with a brute-force 2-D grid maximum within the grid resolution.

## Rules (from AGENTS.md)

- Create/modify files only inside this directory. Do not run git commands that change state
  (no commit/push/stash/reset). Do not install packages; `python` on PATH is the conda env with
  numpy, scipy, numba, matplotlib, h5py, pytest, mpmath. If something is missing, stop and report.
- Put docstrings stating the implemented formula and its domain of validity.
- Set `MPLCONFIGDIR` is not needed (no plots this iteration).

## Completion criteria / verification

- `python -m pytest tests -q` passes (default run < ~2 min).
- `python scripts/make_regression_points.py` runs to completion, `|ref - mp| <= 1e-8` for every
  point (report any exception with the parameters), and writes `data/regression_points.json`.
- End your work with a short report: files created, test summary, the printed table, the worst
  `|ref - mp|`, the typical runtime of `lnL_marg_ref`, and anything the orchestrator should know
  (e.g. cases where the quadrature needed special handling).
