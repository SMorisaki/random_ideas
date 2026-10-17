# Iteration 009 — fix round 1: remove the cost tail of the analytic rule

You are continuing iteration 009 in `dist_psi_marg/` (the working tree already contains your
`dpmarg/geometry.py`, the `rule='analytic'` code in `dpmarg/method2.py`, `scripts/check_geometry.py`,
`scripts/check_method2b.py`, `tests/test_method2b.py`, the REPORT.md section). Read
`prompts/iter_009.md` for the original task and `data/method2b_summary.md` for your results.
Do not edit `TASK.md`, `PROGRESS.md`, `AGENTS.md`, `prompts/`.

## What is wrong (orchestrator's analysis of `data/method2b_check_full.json`)

Accuracy is fine (all 4,347 points <= 1e-4). The cost distribution is not: the overall p99 rose
from 177 µs (scan) to 497 µs, and the *mean* cost on the sampler-like set is 51 µs versus 40 µs
for the scan rule, although its median is 3.9 µs. Two causes:

1. **Radial-table fallbacks at pedestal nodes.** On the sampler-like set the grid branch (4)
   sends 20% of its radial calls to direct quadrature (mean 19 of 96 calls, mean cost 123 µs,
   24.6 µs when no radial fallback occurs) and the sinh branch (5) 12% (mean 20 of 170 calls,
   mean 155 µs; 32 µs with none). The GH branch has none (its nodes sit at the peak). The
   expensive nodes are far from the peak where the integrand is negligible (`a ~ 0`, moderate
   `b`: the guarded prior-vs-Gaussian region of the table). Nodes whose contribution is
   provably negligible must not call the table at all.
2. **The sinh fallback fires too often and is too large.** Branch 5 serves 19% of all calls
   (28% on the sampler-like set) with 96 nodes (or 2 x 48), yet its widths are mostly moderate:
   quantiles of `w` on the fallback cases are 5%: 0.008–0.03, 50%: 0.10, 95%: 0.19; only
   0.4–12% have `w < 0.02`. A fixed periodic grid resolves a peak of width `w` with spacing
   `h <~ 1.3 w` (trapezoid error for a periodic Gaussian-like peak ~ `2 exp(-2 pi^2 (w/h)^2)`:
   4e-5 at `w/h = 0.7`), so `N = 64` covers `w >= 0.035` and `N = 128` covers `w >= 0.017`.

## Required changes

A. **Negligible-node skipping (all non-GH rules, and GH too if cheap).** Before any radial
   lookup, evaluate the analytic Laplace proxy `p_L(theta_j)` (`proxy_derivatives(..., laplace=True)`
   where valid, else the continuous proxy) at every node, take `p_max` over the nodes, and skip
   the radial call for nodes with `p_L(theta_j) + ln(weight_j) < p_max + ln(weight at the max) - M`.
   Calibrate `M` (start at 40; the proxy's error relative to the true `ln h` is bounded by the
   width-factor terms, so justify the margin and verify) by comparing skip vs no-skip on all
   sets: the difference must stay below 1e-7 everywhere. Report the fraction of nodes skipped and
   the radial-fallback counts before/after. Evaluate the proxy with the precomputed grid trig
   factors so that skipping costs far less than a lookup.
B. **Replace the sinh fallback by width-selected grids.** For a sharp peak that fails the GH
   guards, use the fixed periodic grid with `N` from the predicted width (spacing `<= 1.3 w`,
   sizes 48/64/96/128; precompute trig for 96 and 128 too) when `w >= 0.017`; keep a sinh rule
   only for `w < 0.017` with a significant pedestal, and use **64** corrected sinh nodes (the
   iteration-003 rule validated at 4.7e-5), not 96. Two overlapping sharp peaks: the two-interval
   Gauss–Legendre rule with 32 nodes per interval. Re-measure the error-vs-N curves for the
   grid on the former fallback cases (not only on the broad branch) and set the spacing factor
   from the data.
C. **GH acceptance by pedestal mass, not tail probes.** Replace the "proxy drop > 20 at
   +-8 w" test by an estimate of the mass outside the GH window: sample the proxy at 8–16
   uniform points over the period (cheap, trig precomputed), estimate
   `mass_pedestal ~ sum exp(p_L) * pi/N_probe` outside `+-6 w`, and accept GH when
   `mass_pedestal < 1e-6 * exp(p*) * sqrt(2 pi) w`; otherwise use the grid of B. Keep the
   iteration-008 failure cases (`seed808_44/915`, `posterior_124/876`) and the `lam2 = 0`
   island cases within 1e-4 — add them to the tests if they are not already there.
D. Re-run `scripts/check_method2b.py` (reuse the cached references) and report, per set:
   max error, fraction > 1e-4, **median, mean and p99 µs** for scan vs analytic, branch
   fractions (GH / grid / sinh / two-interval), radial-fallback counts per call (mean), and the
   fraction of nodes skipped. Targets: every set's mean and p99 below the scan rule's, no point
   above 1e-4 (report any), overall median <= 8 µs. Update `data/method2b_summary.md`,
   `figures/method2b_cost.png`, the REPORT.md section and the tests.

## Rules (AGENTS.md)

Edit only inside this directory; no git state changes; no installs; no secrets; per-point dumps
to `data/*_full.json`.

## Completion criteria / verification

- `python -m pytest tests -q` passes (keep under ~8 min).
- `python scripts/check_method2b.py` completes; the report in your final message contains the
  per-set table above (before/after), the chosen `M`, grid spacing factor, and `w` threshold for
  the residual sinh rule, and names any point above 1e-4.
