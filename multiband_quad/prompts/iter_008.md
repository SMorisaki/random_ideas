# Iteration 008 — band rules for higher-order modes: slack vs error, L scaling, prior grid, and a B0-compatible rule

You are working in `multiband_quad/` (the current directory). Read `AGENTS.md`, `TASK.md` (sections 2, 4 [C3], 7 [008], 9, 10),
`notes/method_idea.md` §8 (items 3-6, 9, 14) and §10, `notes/paper_summary.md` (Eqs. 47-57), and the code of iterations
001-007 (`mbquad/bands.py`, `pairwise.py`, `baseband.py`, `totalweights.py`, `lalwave.py`, `population.py`, `testcases.py`).
Reuse the fine-grid reference caches in `data/` where the systems coincide (`data/xhm_reference_cache.npz`,
`data/population_cache_*.npz`). Iteration 007 found on XHM populations that B0 is as accurate as the pairwise method and that
at `L = 50` all three new methods coincide (adaptive B1 selects B0 everywhere); the per-point costs are in
`data/xhm_timing_iter006.json` and `data/population_iter007.md` (timings section). Use `XHMCoarseEvaluator` for coarse
waveforms, `OMP_NUM_THREADS=1`.

## Goal (completion criterion C3)

Quantify how the accuracy of the new methods depends on the band design, and propose band rules:
(a) the pairwise method A needs, per band and lag class, a block that covers the class's lag interval over the prior;
(b) the total-waveform method B0 is exact up to tails when the two-sided lag support of `|h|^2` fits in `T_b/2`.
The paper's rule (Eq. 51) was designed for `(d, h)`; here we ask what changes for `(h, h)` with HOM.

## Deliverables

1. `mbquad/bands.py`: add an optional argument to `BandSetup` (keep the default behaviour identical) selecting the band rule:
   - `rule="paper"` (Eq. 51 as now);
   - `rule="hh_half"`: additionally require, for every band `b >= 1`, `2 [Lambda_max(f_lo) + L_hh sigma_mmax(f_lo)] <= T_b` with
     `Lambda_max = tau_0pn(2 f_lo/m_max) - tau_0pn(2 f_lo/m_min)` (`m_min` an argument, default 1) and `f_lo = f^(b) - Df^(b)`,
     i.e. the band edge `f^(b)` is the larger of the Eq. 51 root and the root of this condition (document the resulting edges);
   - `rule="hh_full"`: require `Lambda_max(f_lo) + L_hh sigma_mmax(f_lo) + L_hh sigma_mmax(f^(b+1)) <= T_b` (one-sided block
     of method A fits for the extreme class) — this should be weaker than or equal to Eq. 51 in practice; report when it binds.
   Provide `summary_table()` columns for the binding condition per band and `K_MB` for each rule.
2. `scripts/bandrule_iter008.py` (XHM, analytic+lines PSD and real L1 PSD, inclination 1.0, the three nominal systems plus the
   strong-HOM NSBH case `m1 = 23, m2 = 2.3, iota = 1.5` and a `chi_eff = -0.5` BBH variant):
   (a) **L scan** with the paper rule and the same durations: `L in {5, 10, 20, 50}`: relative `(h,h)` errors of linear,
       IFFT-FFT, pairwise, B1 (default and all-half-bins), B0, with `K_MB`; fit the slopes `d log(error)/d log(L)` for
       pairwise and IFFT-FFT (expected ~ -3 and ~ -2) and comment on deviations;
   (b) **B0-compatible rule**: for `rule="hh_half"` with `L_hh in {3, 5, 8}` and `L = 5`: `K_MB` increase relative to the paper
       rule, the adaptive-B1 band modes (expect all "B0"), the errors of B0 / B1 / pairwise / IFFT-FFT, and the implied route
       costs for 1 and 3 detectors using the iteration-006 per-point costs (waveform ms per coarse point from
       `data/xhm_timing_iter006.json`, `(h,h)` costs scaled linearly with `K_MB`): tabulate "IFFT-FFT route with paper bands"
       vs "B0 route with hh_half bands" vs "pairwise route with paper bands";
   (c) **slack vs error for method A**: on the GW190814-like system, artificially shrink the block margins by using
       `mchirp_range = (Mc_true, Mc_true)` vs the prior corners, and shift the blocks by `+-{0.25, 0.5, 1} sigma` (add a
       test-only `block_shift_seconds` option or build weights directly with `BlockWeights.for_lag_interval`) to show how
       the error grows when the lag interval leaves the block; plot error vs slack.
   Save `data/bandrule_iter008.md` (+ JSON) and figures `figures/iter008_L_scan.png`, `figures/iter008_slack.png`.
3. `scripts/prior_slack_iter008.py`: slack audit over a grid of `(Mc, q, chi_eff)` covering the BNS, NSBH and BBH priors of
   iteration 007 (`mbquad/population.py`: `sample_population`), using the MODEL's own time-frequency relation: estimate `tau_lm(f)` for XHM modes from
   `-(1/2 pi) d arg(h_lm)/df` on a fine frequency grid (unwrap; smooth), compare with the Newtonian `tau_0pn(2f/m)` used to
   place the blocks (ratio at the band edges), and report per band/class the realised slack (block edges minus the measured
   lag interval); flag any negative slack; include the non-dyadic duration set `{T, T/4, T/16, T/64}` for the BNS and
   `m_max = 5` (with the (5,5) mode available in XHM via `lalParams` mode array if the one-mode API supports it; otherwise
   use the Newtonian prediction and say so). Save `data/prior_slack_iter008.md`.
4. Tests (`tests/test_bandrules.py`): the new rules reproduce the paper rule when their conditions do not bind; `hh_half`
   edges satisfy the condition; `K_MB` ordering paper <= hh_full <= hh_half (or document exceptions); the slack audit
   function recovers the Newtonian lag within 2% on `ToyModes`.

## Rules (from AGENTS.md)

- Create/modify files only inside this directory; no git state changes; no package installation; no network.
- Do not change the default behaviour of iteration 001-007 code. `OMP_NUM_THREADS=1` in scripts. Keep the default test run
  under ~2 minutes; mark long scans `slow`.

## Completion criteria / verification

- `python -m pytest tests -q` passes; the scripts run and write their tables and figures.
- Final report: the L-scan table with fitted slopes, the B0-compatible-rule table (K_MB increase, errors, route costs for 1
  and 3 detectors), the slack-vs-error result, the prior slack audit (worst slack per population, Newtonian vs measured
  lag ratio), and a one-paragraph recommendation for the band rule(s) to document in METHOD_NOTE.md.
