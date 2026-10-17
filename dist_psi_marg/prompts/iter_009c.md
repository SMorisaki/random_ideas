# Iteration 009 — fix round 2: the degenerate corner (lam2 ~ 0, eta ~ 0) and clipping-cusp maxima

You are continuing iteration 009 in `dist_psi_marg/` (your fix-round-1 code is in the working
tree; read `prompts/iter_009.md`, `prompts/iter_009b.md`, `data/method2b_summary.md`).
Do not edit `TASK.md`, `PROGRESS.md`, `AGENTS.md`, `prompts/`.

## What the orchestrator found (independent sweep, bounds (0.1, 100), xi = 0.3)

144 points: `lam1 in {1e3, 1e5, 1e7}` x `q = lam2/lam1 in {0, 1e-8, 1e-6, 1e-4}` x
`eta in {0, 1e-3, 1e-2, 0.1}` x `R/lam1 in {0.05, 0.3, 3}`. Results versus `lnL_marg_ref`:

- `rule='analytic'`: 15 points > 1e-4, 11 > 1e-3, worst **0.63** (`lam1 = 1e7, q = 1e-4, eta = 0,
  R/lam1 = 0.05`), 0.28 (`q = 1e-6`), 0.10 (`q = 0`), 0.058 (`lam1 = 1e5, q = 0`), and a family of
  1e-3–6e-3 errors for `q = 0, eta = 0` at every `lam1`. All failures are **branch 5 (single-patch
  sinh64) centred on a spurious candidate of width 2.5e-6 … 3e-3** at the `x0 = xmax` crossing
  near `theta = pi/2`, while the true maximum is the **clipping cusp at the `x0 = xmin` crossing**
  `theta_c = acos(R/(lam1 xmin))` (height `+4 ln(xmax/xmin) = 27.6` above the xmax crossing).
  Node skipping is not the cause (identical errors with `skip_margin = inf`).
  Mechanism: on the flat interior ridge of the `lam2 = 0, eta = 0` family,
  `fmax = R^2/(2 lam1)` is constant and `ln h ~ const + 3 ln|cos theta|`; the maximum sits at
  the C0 cusp where the peak enters the prior range, and the geometry assigned it the broad
  one-sided curvature width (0.25) while giving the xmax-crossing candidate a tiny width. The
  sinh64 patch centred at the wrong point under-resolves the cusp.
  The physically meaningful width of a clipping cusp is the radial transition width mapped to
  theta: `w_cusp ~ (1/sqrt(b)) / |dx0/dtheta|` with `x0 = a/b` (for the `eta = 0` ridge:
  `w_cusp = sqrt(lam1)/(R tan theta_c) = 1/(rho_1 tan theta_c)`, `rho_1 = R/sqrt(lam1)`; e.g.
  3.7e-3 at `lam1 = 1e7, R = 5e5`). A kink maximum is not Gaussian: use the sinh-stretched
  trapezoid centred on the cusp (or the grid when `w_cusp` is resolvable), never GH.
- `rule='scan'` (legacy): 4 points > 1e-3, worst **57** (`lam1 = 1e5, q = 1e-6, eta = 1e-3,
  R/lam1 = 0.05`), 3.2 and 3.1 (`lam1 = 1e3, eta = 1e-3`). These are the near-degenerate
  weak-axis islands: for `theta = pi/2 - delta`, `a ~ R (delta + eta)`, `b ~ lam2 + lam1 delta^2`,
  so `fmax(delta) = (R^2/(2 lam1)) (1 + u)^2/(1 + u^2)` with `u = delta/eta` when
  `lam2/lam1 = eta^2`: the island at `delta = eta` is **twice** the ridge height with width ~eta
  (1e-3 rad), invisible to the 64-point scan and to the uniform 16/32-node rule whose nested
  check stopped early. The analytic rule finds it (closed-form interior maximum; branch 3,
  error 6e-7). The scan rule must at least seed from the analytic candidates.

## Required changes

A. **Geometry: clipping-cusp maxima.** When a candidate maximum lies at a radial clipping
   transition (`a = b xmin` or `a = b xmax` crossing), set its width to
   `min(one-sided curvature widths, w_cusp)` with `w_cusp = (1/sqrt(b)) / |dx0/dtheta|`
   (document the formula; handle `b -> 0`), flag it as a *kink* (non-Gaussian), and rank all
   candidates by estimated mass (`height + ln width`, with the pedestal estimate of round 1)
   so that the dominant one defines the centre of a single-patch rule. A kink maximum uses the
   sinh-stretched trapezoid (64 nodes, endpoint-corrected) centred on the kink, or the fixed
   grid when `w_cusp >= 0.017`; GH is never applied to a kink.
B. **Spurious candidates at `a, b -> 0`** (`theta -> pi/2` on the `lam2 = 0, eta = 0` ridge):
   the pair integral is exactly `2 V` there (`V = int r^-4 dr`); the Laplace endpoint proxy
   is singular. Guard the candidate set so that a vanishing-slope endpoint never produces a
   narrow width, and verify with `q = 0, eta = 0` cases that no spurious sharp candidate survives.
C. **Scan rule seeds.** Add the analytic candidate set (closed-form interior maximum, weak-axis
   island solution `delta = eta`-type stationary points, clipping crossings) to `_geometry`'s
   seeds so that `rule='scan'` finds the near-degenerate islands too; keep it as the legacy
   comparison rule. Alternatively document that `scan` is unsupported for `lam2/lam1 < 1e-4`
   if you cannot fix it within this round — but try.
D. **Validation set "degenerate corner".** Add `scripts/check_degenerate.py`: the 144-point grid
   above extended to `q in {0, 1e-8, 1e-6, 1e-4, 1e-2}`, `eta in {0, 1e-4, 1e-3, 1e-2, 0.1}`,
   `R/lam1 in {0.05, 0.3, 3, 30}`, `lam1 in {1e3, 1e5, 1e7}` (300 points) plus 300 random
   near-degenerate points (`log10 q` uniform in [-9, -2] with a 20% atom at 0, `eta` log-uniform
   in [1e-4, 0.3] with a 20% atom at 0, `rho` log-uniform in [1, 300], random `xi`), references
   via `lnL_marg_ref` cached in `data/degenerate_reference_full.json` (git-ignored; ~2 min),
   reporting max / >1e-4 / >1e-3 for `analytic` and `scan`, branch fractions and costs;
   summary `data/degenerate_check.json` (small). Add the worst ~10 cases to `tests/test_method2b.py`
   with the 1e-4 tolerance (and the flat-ridge family `q = 0, eta = 0` at
   `lam1 in {1e3, 1e5, 1e7}`, `R/lam1 in {0.05, 0.3, 3}`).
E. Re-run `scripts/check_method2b.py` and `scripts/check_geometry.py`; the four shared sets must
   stay at <= 1e-4 with costs at least as good as round 1 (mean and p99 below scan on every set);
   the degenerate sets must reach <= 1e-3 everywhere (report 1e-4 compliance). Update
   `data/method2b_summary.md`, the REPORT.md section, figures.

## Rules (AGENTS.md)

Edit only inside this directory; no git state changes; no installs; no secrets; per-point dumps
to `data/*_full.json`.

## Completion criteria / verification

- `python -m pytest tests -q` passes (keep under ~9 min).
- `python scripts/check_degenerate.py` and `python scripts/check_method2b.py` complete; the
  final message gives the degenerate-set tables (before/after, analytic and scan), the shared-set
  table (accuracy, median/mean/p99), and names any remaining point above 1e-4 with its regime.
