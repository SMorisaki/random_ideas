# Distance lookup: iteration 004

The implemented production option uses a guarded cubic B-spline table of
`H = lnI - lnG`, where `G = integral exp(f-fmax) dx` is the analytic truncated
Gaussian integral. It preserves the iteration-003 angular accuracy. Numerical
results are in [table1d_check.json](data/table1d_check.json) and
[method2_check.json](data/method2_check.json); the scripts regenerate them.

## Coordinates, analytic kernel, and the frozen-prior issue

The coordinates are

```
s = asinh((a-b*xmin)/sigma) + asinh((a-b*xmax)/sigma)
v = asinh(b/bref)
sigma = sqrt(b + sigma0**2)
sigma0 = 1/(xmax-xmin)
bref = 1/xmax**2
```

Using a quadrature sum instead of `max(sqrt(b),sigma0)` avoids a derivative
kink at the scale transition. For the two priors, `(sigma0,bref)` is
`(0.01001001001,1e-4)` and `(0.10101010101,0.01)`. At `b=bref`, the largest
quadratic exponent is 1/2; the volume prior heavily weights much smaller x,
so the prior-weighted integral is still close to its b=0 limit.

The requested `Delta = lnI-lnItilde`, `Itilde=r^-4 G`, has a discontinuity
at `a=b=0`: r jumps from xmin to xmax when a changes sign at b=0. The jump
in Delta is `4*ln(xmax/xmin)`, or 27.6310 for the main prior. At b>0 the
clipped radius also introduces derivative kinks along both clipping curves.
Thus a globally smooth cubic interpolant of Delta cannot achieve the target
on the proposed whole domain. All three tabulations are implemented:

| Mode | Stored value | Interpretation |
|---:|---|---|
| 0 | lnI | Direct tabulation, including fmax clipping artifacts |
| 1 | lnI-lnItilde | Requested frozen-prior remainder |
| 2 | lnI-lnG | Smooth alternative; recommended |

Mode 2 is `Delta-4*ln(r)`. It is bounded between `-4*ln(xmax)` and
`-4*ln(xmin)` and cancels fmax without depending on the clipped-radius prior
factor. It still has sharp, smooth transitions where the volume-prior and
likelihood peaks exchange dominance. These motivate the cell guard.

`lnItilde` is implemented and independently checked against scaled-offset
SciPy quadrature to 1e-12. The Gaussian calculation uses erf when the peak
is enclosed, erfcx tail differences outside, expm1 for close exponentials,
and a short integrated Taylor series for nearly constant weights. erfcx
uses `exp(z*z)*erfc(z)` below 20 and a 13-term asymptotic series above 20.

## Grid, guard, alternatives, and coverage

Both tables use `996 x 996` physical nodes, `|s|<=24`, `0<=b<=1e8`.
The cubic prefilter has a polynomial extrapolation halo; retaining two
coefficient guards on each edge gives exactly **1,000,000 float64 entries**
(8 MB). A `995 x 995` boolean cell mask adds 0.99 MB. Neither array is
tracked in git. Raw sampled values are discarded after prefiltering.

Five direct-kernel probes per cell (center and four edge midpoints) flag
errors above 2e-6. A flagged cell uses direct quadrature. This is an empirical
error guard, not an analytic uniform certificate. Validation uses a separate
100,000 physical-node sample per prior, 100,000 uniform-coordinate probes
per prior, and the full iteration-002 scalar grid (1,598 points per prior).
The main-prior build flags about 0.82% of cells; the secondary flags 0.34%.

The checker compares unguarded frozen-prior cubic, smooth cubic, direct-lnI
cubic, and smooth bilinear grids at 250, 500, and 996 nodes per axis. Thus the
dense bilinear alternative can also be compared with smaller cubic tables.
It separately measures guarded smooth cubic at all three sizes, including
fallback rates. `smallest_measured_passing_grid` records the smallest tested
size meeting each tolerance, or null when none passes. Those entries are
measurements on the validation set, rather than extrapolated grid estimates.
The default grid favors fewer expensive fallbacks. The error-map figure
uses a symmetric log scale so the failed raw and Delta variants remain
visible alongside the production errors.

Outside the table:

- b>bmax always uses `lnI_direct`.
- |s|>24 uses an 18-term endpoint expansion only if
  `|1/(d*r)|<=1e-3`, `b/d**2<=1e-3`, and
  `d*(xmax-xmin)>80+4*ln(xmax/xmin)`, where d=|f'(endpoint)|.
- Other exterior points and flagged cells use `lnI_direct`.

The expansion integrates the Taylor coefficients of
`(1+p*z)^-4 * exp(-q*z*z/2)` against exp(-z), with
`p=sign/(d*r)`, `q=b/d**2`. Its tests compare with the direct kernel.
Diagnostic fallback flags include **both** endpoint expansion and direct
quadrature. The checker also reports their rates separately.
An additional bmax=1e10 main-prior table is built and stress-checked by the
checker, without changing the two default table files.

## API and reproduction

```python
from dpmarg.table1d import load_table, lnI_table, table_diagnostics
from dpmarg.method2 import lnL_marg_m2_table, method2_table_diagnostics

table = load_table('data/table1d_xmin0.1_xmax100.npz')
log_radial = lnI_table(a, b, table)
log_likelihood = lnL_marg_m2_table(A, B, C, D, E, .1, 100., table)
value, calls, mode, theta, weights, fallbacks = method2_table_diagnostics(
    A, B, C, D, E, .1, 100., table)
```

The table is a tuple containing the coefficient and mask arrays plus scalar
metadata. All scalar/array evaluation functions and method-2 entry points
are numba functions. Saving/loading uses npz with pickle disabled. A table
must match the integration bounds; mismatches raise ValueError.

Method 2 shares the complete adaptive quadrature implementation between
kernel variants. Geometry runs once per call, with a 64-point proxy scan
and 28 golden-section refinement iterations. Coarse nested weights are
formed from the full rule with the correct stride-squared endpoint
correction. The common 32/64-node Legendre rules are precomputed. The saved
iteration-003 reference sets are reused without recomputation.

```
python scripts/build_table1d.py
python scripts/check_table1d.py
python scripts/check_method2.py
python -m pytest tests -q
```

The builder accepts `--xmin`, `--xmax`, `--ns`, `--nv`, `--smax`, and
`--bmax`. Omitting xmax builds both priors. Build times include cell probing
and coefficient construction, exclude the explicitly reported initial JIT
warm-up, and are checked against 60 s per table. Lookup timings are medians
of seven warmed array batches; method-2 timings are medians of per-case
measurements inside repeated numba loops. Geometry and kernel shares are
measured separately on the same actual kernel arguments; remaining time
includes angular trig, pair assembly, node construction, and accumulation.

## Measured results (2026-10-16)

Absolute errors below combine actual method-2 nodes, the iteration-002 grid, and independent coordinate probes.

| Prior xmax | Build seconds | Lookup ns including fallbacks | Worst log error | Physical-node fallback rate |
|---:|---:|---:|---:|---:|
| 100 | 29.07 | 325.2 | 1.9e-06 | 1.547% |
| 10 | 21.58 | 248.4 | 1.99e-06 | 0.675% |

| Regime | Frozen-prior Delta, xmax=100 / 10 | Raw lnI, xmax=100 / 10 | Guarded smooth, xmax=100 / 10 |
|---|---:|---:|---:|
| b = 0 | 13.8 / 9.21 | 0.00412 / 0.00412 | 7.88e-08 / 6.65e-08 |
| x0 < xmin | 15.3 / 9.21 | 0.00412 / 0.00412 | 1.07e-07 / 4.83e-08 |
| inside | 5.19 / 9.42 | 0.000558 / 0.00376 | 1.9e-06 / 1.99e-06 |
| x0 > xmax | 4.77 / 9.08 | 0.00162 / 0.00384 | 1.85e-06 / 1.51e-06 |

Smaller guarded 250 x 250 tables already meet 1e-5 on the validation set, but physical-node fallback rates rise to 22.8% / 15.8%; the selected grid reduces these to 1.55% / 0.675%. Direct-quadrature shares are 1.22% / 0.593%; the rest use the endpoint expansion.
Unguarded smooth cubic needs the largest tested grid for the 1e-4 limit in the secondary prior; no tested unguarded main-prior grid meets 1e-4. Neither raw lnI cubic nor frozen-prior Delta cubic nor smooth bilinear meets 1e-4 within the tested size budget.
The 1e10 extension builds in 25.46 s with worst error 1.75e-6 on 100,000 coordinate probes.

At 64 angular nodes, median direct/table times are 242.26 / 20.96 us. The worst table/reference error is 4.69e-05, and the largest table/direct change is 4.96e-07.
Geometry takes 3.68 us (17.9% median share), and table kernels take 9.43 us (45.9% median share). Across the 340 saved cases, the fallback rate is 1.050%; median zero and maximum 47 fallbacks per call.
Shuffled scalar lookups within the table measure 0.21–0.30 us; including guarded/exterior fallbacks increases the average to about 0.25–0.33 us. Thus the end-to-end angular speed target is met, while the inclusive scalar lookup cost remains above the requested 0.1–0.2 us.

In the warmed per-case method-2 node loops, the median distance cost is 146.7 ns per evaluation (direct: 3516 ns); the shuffled 100,000-node batch benchmark above also includes table-cache misses and rare direct fallbacks.
