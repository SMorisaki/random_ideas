# Iteration 005 — Method 1 feasibility: a 4-D lookup table of lnL_marg in (lam1, lam2, R, eta)

You are working in `dist_psi_marg/`. Read `AGENTS.md`, `TASK.md` §2–§3 and §7, `dpmarg/params.py`,
`dpmarg/method2.py`, `dpmarg/table1d.py` and `TABLE1D.md` (iteration 004). The fast generator for
this study is `method2.lnL_marg_m2_table(A, B, C, D, E, xmin, xmax, table)` with
`table = table1d.load_table('data/table1d_xmin0.1_xmax100.npz')` (~21 µs per point, worst error
4.7e-5 vs the reference; rebuild the table with `python scripts/build_table1d.py` if the file is
missing); `lnL_marg_ref` is the slow truth (0.1–0.3 s). Reuse `table1d.coordinates`/`inverse_s`
for the radial coordinate.
Report in the final message; do not edit `TASK.md`, `PROGRESS.md`, `AGENTS.md`, `prompts/`.

## Question to answer (TASK.md §3 method 1)

Can the full marginalized likelihood be tabulated once per prior `(xmin, xmax)` as a regular
4-D grid in well-chosen coordinates of `(lam1, lam2, R, eta)` (eta folded to `[0, pi/2]`), with
cubic (or linear) interpolation giving `|Delta lnL_marg| <= 1e-3` (goal 1e-4) over the whole
domain a sampler can visit, at a table size and build time that are practical
(say <= 2e7 float64 entries, build <= 10 min with the fast generator)? Where does it fail?

## Design (implement, then measure; change what the measurements contradict)

1. **Tabulated quantity.** `Ftilde = lnL_marg - lnL_max`, where `lnL_max = max_{r, theta} lnL`
   on the domain (params.lnL_max; write a numba version using the same proxy scan + golden
   section as method2, without the `-4 ln r` prior term). `Ftilde <= 0` (it is
   `ln` of a prior-weighted average of `exp(lnL - lnL_max)`), and at high SNR with an interior
   peak `Ftilde ~ norm + ln(2 pi / sqrt(lam1 lam2)) - 5 ln r*`, i.e. linear in log coordinates.
   Also test tabulating `Ftilde` minus that Laplace expression where the peak is interior if it
   makes the slices visibly smoother (optional).
2. **Coordinates** (generalize distmarg's `(s, t)` and the iteration-004 `(s, v)`):
   - `v1 = asinh(lam1 / bref)`, `v2 = asinh(lam2 / bref)` with the iteration-004 `bref`, on
     `0 <= v2 <= v1 <= v(bmax)` (use a rectangular grid and leave `v2 > v1` unused, or map
     `v2 = w v1` with `w in [0, 1]`; compare both if cheap);
   - the dominant angle `theta*` (maximizer of `cos^2(theta - eta) / b(theta)`) does not depend on
     `R`, so the radial peak `x0* = R c(lam1, lam2, eta)` with `c = cos(theta* - eta)/b(theta*)` is
     linear in `R`, and `b* = b(theta*)` is `R`-independent. Use the iteration-004 radial coordinate
     `s = asinh(sigma* (x0* - xmin)) - asinh(sigma* (xmax - x0*))`, `sigma* = max(sqrt(b*), sigma0)`,
     which is monotone in `R` (so grid `s` -> `R` is a closed-form inversion); `R = 0` maps to a
     finite `s`, so the `s` range is `[s(R = 0), smax]`;
   - `eta` uniform on `[0, pi/2]` (the function is even about both ends).
   Outside the box (`lam1 > bmax`, `s > smax`) fall back to the table-backed method 2 and count
   the fallbacks. Keep in mind the iteration-004 lesson: avoid tabulated quantities with jumps or
   kinks (clipping switches); check the slices for kinks before choosing the grid.
3. **Slice study** (`scripts/slice_study_method1.py`): draw ~200 base points from
   `regimes.random_params` (reduced parameters) plus the named cases; for each axis, evaluate
   `Ftilde` along the full axis range on a fine grid (>= 1025 points) with the fast generator;
   for each candidate spacing (e.g. 64, 48, 32, 24, 16, 12, 8 points per axis) subsample, apply
   cubic (Catmull–Rom, and a natural cubic spline for comparison) and linear interpolation, and
   record the maximum error on the fine grid. Report, per axis, the points-per-axis needed for
   1e-3 and 1e-4 (distribution over base points: median, 99%, max) and identify the worst slices
   (regime: `lam2/lam1`, SNR, peak position). Figures: `figures/method1_slices_<axis>.png`
   (a few representative slices with the interpolation error), `figures/method1_points_per_axis.png`.
   Spot-check 20 fast-generator values against `lnL_marg_ref` (must agree to the iteration-004
   level) so the study is not fooled by generator error.
4. **Prototype table** (`dpmarg/table4d.py`, `scripts/build_table4d.py`): with the grid implied
   by step 3 (cap the total at 2e7 entries; if the estimate exceeds it, build the largest
   feasible grid and say what accuracy to expect), build the 4-D table with the fast generator
   (numba loop; report build time and size), implement 4-D cubic (tensor Catmull–Rom, 256 reads)
   and 4-D linear lookups in numba (`lnL_marg_m1(A, B, C, D, E, xmin, xmax, table...)`), and
   measure the error on `data/regression_points.json`, `data/method2_random_reference.json`
   (stored reference values) and 2000 fresh random points versus the fast generator; lookup
   time (median ns); fallback rate. Save the table under `data/` (`.npz`, git-ignored) and a
   summary in `data/method1_check.json` (small).
5. `tests/test_method1.py`: coordinate maps invertible (`s -> R -> s`, 1e-12), `lnL_max` numba vs
   `params.lnL_max` (1e-10 on 20 points), a small table (coarse grid, built in < 30 s in the test)
   reproduces the generator within the expected tolerance on grid points exactly and within the
   slice-study bound between them, fallback path exercised, all-zero case gives 0.

## Rules (AGENTS.md)

Edit only inside this directory; no git state changes; no installs; no secrets; large tables and
per-point dumps stay out of git (`data/*.npz`, `data/*_full.json`).

## Completion criteria / verification

- `python -m pytest tests -q` passes (keep the suite under ~6 min; mark slow tests).
- `python scripts/slice_study_method1.py` and `python scripts/build_table4d.py` run to
  completion and write the figures and `data/method1_check.json`.
- Final message: the verdict (feasible or not, with the numbers: points per axis, total entries,
  memory, build time, lookup ns, max/99%/median error on each set, fallback rate), the regimes
  that limit the grid, and what you would change if the target is not met.
