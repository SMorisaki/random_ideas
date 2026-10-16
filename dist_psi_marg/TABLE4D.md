The method-1 prototype tests a single regular four-dimensional table for one
fixed prior. The measured verdict, build cost, accuracy and timings are in
[data/method1_check.json](data/method1_check.json); the independent axis study
is in [data/method1_slices.json](data/method1_slices.json). This is an experiment,
not an accuracy-qualified replacement for method 2.

The stored quantity is `Ftilde = lnL_marg - lnL_max`. The coordinates are
`v1 = asinh(lam1/bref)`, `w = asinh(lam2/bref)/v1`,
`u = (s-s(R=0))/(24-s(R=0))`, and `eta/(pi/2)`, with `bref=1/xmax^2`.
All four stored axes run from zero to one. The triangular eigenvalue domain
is covered without filling or sorting an unphysical half of a rectangle.
The rectangular eigenvalue alternative was not built; no measured superiority
of the triangular mapping is claimed.

The suggested unconstrained peak direction is singular at `lam2=0`.
For coordinates only, use the direction of
`(cos(eta)/(lam1+sigma0^2), sin(eta)/(lam2+sigma0^2))`, where
`sigma0=1/(xmax-xmin)`. Its physical `b*` and projection `cos(theta*-eta)`
feed `table1d.coordinates` and `inverse_s`, retaining iteration 004's smooth
quadrature combination of scales. The likelihood and its maximum keep the
original eigenvalues. The coordinate inverse uses Newton correction to avoid
cancellation near `R=0`. Absolute round trips in physical `s` can be limited
to a few times 1e-11 near `xmax` at `lam1=1e8`; the tests explicitly account
for conditioning and verify 1e-12 on the ordinary-scale domain. A global
1e-12 absolute physical-s guarantee is not supported in float64.

The study uses 200 seed-501 random bases plus 20 named cases, 1,025 fine nodes
per axis, and independently generated coarse grids with 8 through 512 nodes.
It compares Catmull–Rom, natural cubic, and linear interpolation. Required
counts must continue to pass on every finer measured grid. Null counts in
JSON mean that no tested resolution establishes the target, rather than an
extrapolated node count. Fine-grid maxima remain sampled diagnostics, not
uniform certificates. All 20 independent brute-force spot checks agree with
the repaired fast generator to 1.0e-5.

At tolerance 1e-3, median Catmull–Rom requirements are 256, 192, >512, and 12
nodes for `(v1,w,u,eta)`. At 1e-4 they are >512, 512, >512, and 24. The 99th
percentile and worst case exceed 512 in every axis at either tolerance. A
straight product of those worst-axis requirements already exceeds 69 billion
entries (554 GB), before accounting for interactions between axes.

Several distinct effects limit the grid:

- `Ftilde` inherits cusps where competing likelihood maxima exchange dominance.
  For `(lam1,lam2,R)=(1e4,0,1)`, its right derivative at `eta=0` is approximately
  -100, although the full marginalized likelihood is smooth and even. Thus
  imposing an even cubic tangent on this remainder is inappropriate there.
- The weak-axis peak coordinate varies over an extremely thin angle layer
  near `(lam2,eta)=(0,0)`, even after its mathematical singularity is regularized.
- Large eigenvalues with `R` near zero produce narrow angular-normalization
  transitions unresolved by a coordinate designed around the radial peak.
- High-curvature lower-boundary cases and nearly equal eigenvalues create
  additional thin transitions; interior Laplace reasoning does not cover them.

The capped prototype uses `66 x 66 x 66 x 69 = 19,837,224` float64 samples
(158,697,792 bytes), without a guard mask. It implements tensor Catmull–Rom
(256 reads), multilinear interpolation (16 reads), linear ghost extrapolation
on the first three axes, and even reflection on eta. Only exterior points
fall back to table-backed method 2. In-box interpolation errors are intentionally
exposed. Full API timings include coefficient reduction, coordinate mapping,
maximum reconstruction, and fallbacks; interpolation-only timings are reported
separately. Timings are medians of seven warmed compiled batches per set.

The wider slice study exposed a method-2 peak-search failure: a golden-section
search over a broad, non-unimodal bracket could abandon the very narrow
weak-axis boundary peak for a much lower prior-driven maximum. The search now
compares its seed, broad refinement, and curvature-scaled refinement. Two
brute-force counterexamples previously wrong by thousands now agree to 1e-5;
regression tests protect them. Existing stored reference sets are unchanged.

A next design should use a smooth normalization that includes angular as well
as radial widths, then separate degeneracies and boundaries into adaptive
patches or guarded domains. Subtracting a clipped interior Laplace expression
would add further switching surfaces. The existing method-2 integration is
the practical whole-domain fallback while those alternatives are studied.
These measurements reject this regular-grid design, not every conceivable
four-dimensional coordinate transformation.

Reproduce from this directory:

```bash
python scripts/slice_study_method1.py
python scripts/build_table4d.py
python -m pytest tests -q
```

Use `--reuse` on the builder to validate the saved table without rebuilding it.
Large `.npz` tables, per-point `*_full.json` dumps, and `*_run.log` files are
ignored by git. Figures are saved as `figures/method1_slices_<axis>.png` and
`figures/method1_points_per_axis.png`.

The measured 19,837,224-entry build took **806.15 s** after a separate 2.52 s
warm-up. Its wall time overlapped some validation. A separate warmed cost
measurement on 10,000 reproducibly sampled actual grid nodes gives
40,317 ns/node and an estimated **799.79 s** serial construction, so overlap
does not explain the missed ten-minute budget. A worst-axis product using
513 nodes on each axis is already 69,257,922,561 entries, 554 GB and roughly
32 days at that measured generation rate; it still has no accuracy certificate.

Measured absolute log-likelihood errors and complete API costs:

| Set | Interpolation | Maximum | 99% | Median | ns/evaluation | Exterior fallback |
|---|---|---:|---:|---:|---:|---:|
| Regression, 40 | Cubic | 4.821 | 3.595 | 0.001908 | 4550 | 2.5% |
| Regression, 40 | Linear | 4.822 | 3.675 | 0.01276 | 4476 | 2.5% |
| Stored random, 200 | Cubic | 4.823 | 4.379 | 0.0007281 | 4255 | 2.5% |
| Stored random, 200 | Linear | 4.832 | 4.458 | 0.01587 | 3916 | 2.5% |
| Fresh random, 2000 | Cubic | 5.406 | 4.190 | 0.001017 | 4579 | 2.2% |
| Fresh random, 2000 | Linear | 5.408 | 4.242 | 0.01559 | 4146 | 2.2% |

Regression/stored errors use brute-force reference values. Fresh errors use
the fast generator (seed 505). The saved random file also contains 100 points
for `(0.1,10)`; these are explicitly excluded from the main-prior table check,
rather than compared using a mismatched prior. The fast generator retains
maximum errors 3.78e-5 and 4.69e-5 on the two matching reference sets.
Interpolation alone costs 973 ns for cubic and 501 ns for linear. Complete
in-box cubic calls cost 3,675–4,074 ns; the remaining full-call cost includes
exterior fallbacks. In-box interpolation accounts for the large errors.

The worst fresh point has `lam1=lam2=6.66e7`, `R=1.13e5`, and its radial
maximum at `xmin=0.1` (zero SNR under the nonnegative SNR convention). Thus
this failure occurs in an explicitly required, accessible regime, not merely
at a large-SNR corner of the rectangular box. Cubic exceeds 1e-3 at 21/40,
94/200 and 1001/2000 points respectively. These error levels reject the
regular-grid prototype even though lookups themselves are fast.

`python -m pytest tests -q`: **339 passed in 122.21 s**. Both study scripts
completed; after the large table was saved, validation was rerun with
`python scripts/build_table4d.py --reuse` to correct the mixed-prior input
selection without regenerating identical samples.
