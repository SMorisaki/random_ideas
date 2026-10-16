# Iteration 002 — fast 1-D distance kernel lnI(a, b; xmin, xmax) in numba, validated to 1e-8

You are working in `dist_psi_marg/` (the current directory). Read `AGENTS.md`, `TASK.md` §2–§3 and
§7, and the iteration-001 code `dpmarg/params.py`, `dpmarg/reference.py` (keep their public APIs;
`_radial_peak(a, b, xmin, xmax) -> (fmax, r*)` already implements the exponent maximum).
Do not edit `TASK.md`, `PROGRESS.md`, `AGENTS.md`, `prompts/`; put your report in your final message.

## Goal

A fast, robust scalar kernel for the distance integral at fixed theta. Every method in TASK.md §3
needs it (method 2 calls it at each theta node; methods 1 and 3 use it to build tables):

    J(a, b)    = int_xmin^xmax x^-4 exp(a x - b x^2/2) dx,      b >= 0, a any real
    fmax(a, b) = max_{x in [xmin, xmax]} (a x - b x^2/2)          (= _radial_peak; b = 0: endpoint)
    lnI(a, b)  = ln J(a, b) - fmax(a, b)

so that `lnL_marg = ln[3/(2 pi (xmin^-3 - xmax^-3))] + ln int_0^{2pi} exp(fmax(theta) + lnI(theta)) dtheta`
with `a = A cos theta + B sin theta`, `b = C cos 2theta + D sin 2theta + E`.
Parametrize by `(a, b)`, not `(x0 = a/b, b)`, so that `b = 0` (degenerate `lam2 = 0` at
theta = pi/2) and `|a| -> large` are regular inputs.

Accuracy target for the direct kernel: `|Delta lnI| <= 1e-8` for `a in [-1e7, 1e7]`,
`b in {0} U [1e-8, 1e8]`, for `(xmin, xmax) = (0.1, 100)` and `(0.1, 10)` (and no failure for
any finite input). Speed target: a few microseconds per call in numba (`@njit(cache=True)`),
scalar signature `lnI_direct(a, b, xmin, xmax) -> float`.

## Method (distmarg.pdf Sec. 2, adapted; design the details yourself and document them)

- Integrate in `u = ln x`: `x^-4 dx = exp(-3u) du`, so the integrand is
  `exp(-3u + f(e^u) - fmax)`, smooth and without the `x^-4` blow-up near `xmin`.
  Use composite Gauss–Legendre panels in `u` (fixed order, e.g. 16–32 nodes per panel; hard-code
  the nodes/weights as constants) on the window where the integrand is non-negligible.
- Window: for `b > 0` with `x0 = a/b`, distmarg gives a half-width `dx` for a relative truncation
  error `eps` (use `eps <= 1e-10`):
    (19)-(20) for x0 < xmin:  dx = 2A / ( sqrt(b^2 (xmin-x0)^2 + 2 b A) + b (xmin-x0) ),
               A = -ln[ eps ( erfcx(alpha1) - exp(-(alpha2^2-alpha1^2)) erfcx(alpha2) ) ],
               alpha1 = sqrt(b/2) (xmin - x0 + 4/(b xmin)), alpha2 = sqrt(b/2) (xmax - x0 + 4/(b xmin))
    (26) for xmin <= x0 <= xmax:  dx = sqrt(2/b) erfcinv[ (eps/2) (xmin/x0)^4 (erf(beta2) - erf(beta1)) ],
               beta1 = sqrt(b/2) (xmin - x0 + 4/(b x0)), beta2 = sqrt(b/2) (xmax - x0 + 4/(b x0))
    (31)-(32) for x0 > xmax:  dx = 2C / ( sqrt(b^2 (x0-xmax)^2 + 2 b C) + b (x0-xmax) ),
               C = -ln[ eps (xmin/xmax)^4 ( erfcx(gamma1) - exp(-(gamma2^2-gamma1^2)) erfcx(gamma2) ) ],
               gamma1 = sqrt(b/2) (x0 - xmax - 4/(b xmax)), gamma2 = sqrt(b/2) (x0 - xmin - 4/(b xmax))
  You may replace these by a simpler conservative bound of your own (e.g. a window of
  `k/sqrt(b)` plus `k/|f'(endpoint)|` layers with `k` chosen from `eps`), provided you document the
  bound and the error map shows it never truncates more than `eps`. numba has no
  `erfcx`/`erfcinv`; if you use them, implement them (e.g. `erfcx` via `math.erfc` for small `z`
  and a continued fraction/asymptotic series for large `z`; `erfcinv` via a rational initial guess
  refined by Newton steps on `math.erfc`), with tests against `scipy.special`.
- Intersect the window with `[xmin, xmax]`; if it is wide in `u` (small `b`), split it into panels
  of width `<= ~1` in `u` so that `exp(-3u)` and the exponent are resolved; always put a panel
  boundary at the peak `x0` when it is inside, and refine the panels near a sharp endpoint layer
  (width `1/|f'(endpoint)|` when the peak is outside). Guard every formula against
  overflow/underflow (`b -> 0`, `|a| -> large`, `x0 -> +-inf`, `b x_range^2 << 1`).
- `b = 0`: exponent linear, fmax at an endpoint; same composite rule on the full interval, with
  the endpoint layer of width `1/|a|` resolved when `|a| (xmax - xmin) >> 1`.
- Provide a numba loop `lnI_direct_array(a, b, xmin, xmax)` over arrays for benchmarking.

## Deliverables

1. `dpmarg/kernel1d.py`: `fmax_1d(a, b, xmin, xmax)` (numba; same values as `_radial_peak`),
   `lnI_direct(a, b, xmin, xmax)`, `lnI_direct_array(...)`, helper special functions, Gauss–Legendre
   constants. Docstrings with the implemented formulas and their domain of validity.
2. `dpmarg/reference.py`: add `lnI_ref(a, b, xmin, xmax, rtol=1e-11)` (scipy `quad` with the same
   breakpoints as the inner integral of `lnL_marg_ref`; refactor so that `lnL_marg_ref`'s inner
   integral and `lnI_ref` share one implementation, without changing `lnL_marg_ref`'s results beyond
   1e-12 — re-run the iteration-001 tests to confirm).
3. `scripts/check_kernel1d.py`: error map `lnI_direct - lnI_ref` on a grid: `b in {0} U
   logspace(-8, 8, 33)`, `a in {0} U +-logspace(-4, 7, 23)`, for both `(xmin, xmax)` settings; print
   the worst error per regime (`x0 < xmin`, inside, `> xmax`, `b = 0`), save
   `figures/kernel1d_error_map.png` (matplotlib `Agg`; `MPLCONFIGDIR` is set in the environment) and
   `data/kernel1d_check.json` (summary numbers only). Timing: median ns per call after warm-up for
   `lnI_direct_array` on 1e5 random `(a, b)` drawn log-uniformly as above, reported separately for
   the three regimes and `b = 0`.
4. `tests/test_kernel1d.py`: `lnI_direct` vs `lnI_ref` within `1e-8` on ~200 points covering all
   regimes (include `b = 0`, `a = 0`, `a = +-1e7`, `b = 1e8`, `x0` just inside/outside both limits,
   `b (xmax-xmin)^2 ~ 1e-6`); `fmax_1d` equals `_radial_peak` exactly; special functions vs scipy
   (`1e-12` relative); consistency: replacing the inner integral of `lnL_marg_ref` by `lnI_direct`
   reproduces `lnL_marg_ref` within `1e-7` on 10 regression points from `data/regression_points.json`;
   a soft runtime check (`warnings.warn` if `> 20 us` per call).

## Rules (AGENTS.md)

- Create/modify files only inside this directory; no git state changes; no package installation
  (`python` on PATH has numpy, scipy, numba 0.68, matplotlib, h5py, pytest, mpmath). If something is
  missing, stop and report. Never write secrets anywhere.

## Completion criteria / verification

- `python -m pytest tests -q` passes (iteration-001 tests included; keep the total under ~3 min —
  you may mark the mpmath comparisons `slow` and run them once).
- `python scripts/check_kernel1d.py` reports worst `|Delta lnI| <= 1e-8` in every regime and the
  timings; the figure and the JSON summary exist.
- Final message: files created/changed, test summary, worst error per regime, median ns/call per
  regime, number of GL nodes per panel and typical panel counts, and any regime where the window
  had to be made more conservative or where the target could not be met (with numbers).
