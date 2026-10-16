# Iteration 007 — Method 1 revisited: transition-aligned coordinates and a smooth normalization (slice study only)

You are working in `dist_psi_marg/`. Read `AGENTS.md`, `TASK.md` §2–§3 and §7, `TABLE4D.md`,
`TABLE1D.md`, `dpmarg/table4d.py`, `scripts/slice_study_method1.py` and `data/method1_slices.json`
(iteration 005), `dpmarg/method2.py`, `dpmarg/table1d.py`. Report in the final message; do not edit
`TASK.md`, `PROGRESS.md`, `AGENTS.md`, `prompts/`.

## Question

Iteration 005 rejected a regular 4-D grid in `(v1, w, s, eta)` with `Ftilde = lnL_marg - lnL_max`.
Its diagnosis: (i) `lnL_max` has cusps; (ii) along the radial/`R` axis there are several
transitions at scales that differ by factors up to `sqrt(b) xmin ~ 1e3–1e4` (angular normalization
at `R xmin ~ 1`; radial crossings at `R c ~ xmin +- 1/sqrt(b*)` and `xmax +- 1/sqrt(b*)`), and
one asinh coordinate resolves only one of them; (iii) thin layers near `(lam2, eta) = (0, 0)` and
near `q = 1`. The owner's question is whether *clever grid placement* can rescue a regular grid.
Give a final answer with a second, better-designed slice study. Build a prototype only if the
measured per-axis requirements make a table of `<= 3e7` entries plausible.

## Design to test (measure each change separately so the report can attribute the gains)

1. **Smooth normalization.** Replace `lnL_max` by a smooth, cheap upper envelope
   `N(lam1, lam2, R, eta)` and tabulate `lnL_marg - N`. Candidates:
   (a) `N = ln int_0^pi exp(fmax(theta)) dtheta` evaluated with a fixed cheap rule on the proxy
       (the `_proxy` scan machinery without the `-4 ln r` term, no kernel calls; ~1–3 µs);
   (b) a soft maximum `N = T * ln sum_i exp(fmax(theta_i)/T)` over a fixed 64-point theta grid
       with `T ~ 1` (smooth, periodic);
   (c) iteration 005's `lnL_max` as the baseline.
   Check on the `(1e4, 0, 1)` case of `TABLE4D.md` that the derivative in `eta` is now continuous.
2. **Transition-aligned radial coordinate.** For the dominant angle `theta*` (regularized as in
   iteration 005) with `b* = b(theta*)`, `c = cos(theta* - eta)/b*`, `x0* = R c`, and
   `sigma* = sqrt(b* + sigma0^2)`, use a sum of one asinh per transition:
       s3 = asinh(R xmin k_a)                               (angular transition, k_a ~ 1)
          + asinh(sigma* (x0* - xmin)) + asinh(sigma* (x0* - xmax))   (radial crossings)
   (the last two are iteration 004's `s` up to sign conventions). It is monotone in `R`; invert
   numerically (bisection + Newton) at build time. Compare against iteration 005's `s`.
3. **Eigenvalue axes.** Keep `v1 = asinh(lam1/bref)`; for the weak axis compare
   `w = asinh(lam2/bref)/v1` (iteration 005) with `w' = asinh(lam2 xmax^2)` and with
   `q = lam2/lam1` mapped as `asinh`-stretched near 0 and 1 (`q = 1` is a coordinate singularity
   of `theta*`: show how the function behaves there and whether a symmetric treatment
   (e.g. averaging the two candidate angles) removes the layer).
4. **Angle axis.** `eta` uniform on `[0, pi/2]` with even reflection; also test
   `cos(2 eta)` as the coordinate.

## Slice study (`scripts/slice_study_method1b.py`)

Same protocol as iteration 005 (220 base points: 200 random seed 501 + 20 named; >= 1025 fine
nodes per axis; coarse grids 8–512; Catmull–Rom, natural cubic, linear), run for the baseline
(iteration-005 choices) and for each change in turn (1a/1b, 2, 3, 4, and all combined). Report
per axis the nodes needed for 1e-3 and 1e-4 (median / 99% / max), the worst slices and their
regimes, and figures `figures/method1b_points_per_axis.png`, `figures/method1b_slices_<axis>.png`.
Use the table-backed method 2 as the generator (spot-check 20 values against `lnL_marg_ref`).
Write per-point dumps to `data/*_full.json`; summary to `data/method1b_slices.json`.

## Prototype (only if justified)

If the combined design needs `<= ~70` nodes per axis at 1e-3 (99th percentile), build the table
(`<= 3e7` entries) with `scripts/build_table4d.py`-style code and report accuracy on the stored
sets as in iteration 005. Otherwise state the final verdict with numbers and skip the build.

## Deliverables

`dpmarg/table4d.py` (new coordinate/normalization options, selectable by argument, old behaviour
unchanged by default), `scripts/slice_study_method1b.py`, `tests/test_method1b.py` (coordinate
inversion round trip, smoothness of the normalization at the cusp case, generator spot checks),
figures, `data/method1b_slices.json`, and the final message with the verdict table.

## Rules (AGENTS.md)

Edit only inside this directory; no git state changes; no installs; no secrets; big tables and
per-point dumps stay out of git.

## Completion criteria

- `python -m pytest tests -q` passes (keep the suite under ~8 min; mark slow tests).
- `python scripts/slice_study_method1b.py` completes and writes the figures and JSON.
- Final message: per-axis requirement table for baseline vs each change vs combined, the verdict,
  and (if built) the prototype's accuracy/size/build time/lookup time.
