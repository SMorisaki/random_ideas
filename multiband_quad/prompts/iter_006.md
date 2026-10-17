# Iteration 006 — lean XHM adapter and re-timed routes; pairwise tuning on the GW190814-like system; B1 variants; L = 50

You are working in `multiband_quad/` (the current directory). Read `AGENTS.md`, `TASK.md` (sections 2, 4, 7 and 9),
`notes/method_idea.md` §8-9, `data/xhm_accuracy_iter005.md`, `data/xhm_timing_iter005.md` and the code of iterations
001-005 (`mbquad/lalwave.py`, `pairwise.py`, `baseband.py`, `totalweights.py`, `ifftfft.py`, `coarse.py`, `testcases.py`,
`timing.py`, `scripts/xhm_timing_iter005.py`, `scripts/xhm_accuracy_iter005.py`). The fine-grid XHM references of
iteration 005 are cached in `data/xhm_reference_cache.npz` (reuse them; add to the cache if you need new systems).

## Part 1 — lean adapter and honest route timings (the main deliverable)

The iteration-005 adapter sorts/uniques/unsorts the frequency array and projects modes on every call, costing 4.9 ms (BNS)
and 2.0 ms (GW190814-like) on top of the raw LAL time. The coarse grid is FIXED per band setup, so all of that can be
prepared once.
1. Add to `mbquad/lalwave.py` a `class XHMCoarseEvaluator(model_params_or_model, bands, modes=...)` that precomputes, for
   `bands.all_coarse_frequencies().f`: the sorted unique `lal.REAL8Sequence`, the inverse index map, the per-band slices,
   the window-free `exp(-2 pi i f t_c)` phase (the phase depends on `t_c`, so either precompute `exp(-2 pi i f)` and raise to
   `t_c` cheaply, or recompute the phase vector per call in one vectorised step and report its cost separately), and that
   exposes `coarse_modes(params) -> CoarseModes` (five `OneMode` calls, coefficients and phase applied in one pass each,
   no unsort beyond a single fancy-index per mode) and `coarse_total(params) -> CoarseWaveform` (one polarization call).
   `params` are the intrinsic + extrinsic parameters (a small dataclass or dict); the evaluator must not cache results
   between calls with different parameters.
2. `scripts/xhm_timing_iter006.py`: on the BNS and GW190814-like band setups, median warm times of: raw LAL polarizations;
   raw five `OneMode`; `coarse_total` (lean adapter); `coarse_modes` (lean adapter); the `(h,h)` steps (IFFT-FFT fast,
   pairwise pairs, B1 adaptive, B0, linear); and the routes "polarizations + IFFT-FFT", "modes + pairwise", "polarizations
   + B1", "polarizations + B0" with the lean adapter. Also report the fraction of each route spent in Python outside LAL
   and outside the `(h,h)` kernels. Save `data/xhm_timing_iter006.md` (+ JSON). Include the three-detector picture: with
   three detectors the waveform is evaluated once and the `(h,h)` step three times (sum the numbers; say so).
3. Also time the BBH 60+20 setup (K_MB = 4408) for completeness.

## Part 2 — pairwise (method A) tuning on the GW190814-like system

In iteration 005 the pairwise median relative error on the GW190814-like system (3.6e-7) was 2.2x the IFFT-FFT's (1.6e-7),
while it was 3-5x BETTER on the BNS and BBH. Find out why and whether a setting fixes it:
1. `scripts/pairwise_tuning_iter006.py`: for the GW190814-like XHM system (inclinations 0.3, 1.0, 1.4; analytic+lines and
   real L1 PSDs), scan `L_margin in {1, 2, 3, 5}` x `margin_units in {sigma_pair, sigma_mmax}` x `diag in {block, linear}`
   and report the relative error and `|Delta ln Lambda|` at SNR 25 against the IFFT-FFT and B1 values; include the
   per-band signed error contributions (pairwise vs IFFT-FFT) using `band_contributions`-style decompositions (add a
   `band_contributions(coarse_modes)` method to `PairwiseHH` if missing) and the exact fine-grid band values.
2. Lag audit on the XHM modes (fine grid, `XHMModel.mode` on `psdgrid.f_k`) per class in bands 1 and 2: out-of-block fraction,
   centroid, folding error, for the default blocks; identify the class(es) and band that dominate the pairwise error.
3. If a setting reaches a median at or below the IFFT-FFT's, propose it as the new default (do not change the default in
   code; report). If none does, quantify the floor and say which mechanism (period-`T_b` folding of which class, merger-
   ringdown content in the top band, (3,2) mixing, PSD lines) sets it.

## Part 3 — B1 variants and L = 50 (small)

1. Add to `BasebandIFFTFFTHH` optional `pad_factor` (`p`, default 1: `K'' = next_fast_len(p K_b)`) and `all_half_bins`
   (sum the weights over all `K''` half bins of the baseband period instead of the band's `K_b`). On the three XHM systems
   (analytic+lines PSD, inclination 1.0) tabulate the relative error for `p in {1, 2, 4}` x `all_half_bins in {False, True}`
   next to the IFFT-FFT value and the step time. Keep the defaults unless a variant is better by more than 1.5x at no more
   than 2x the cost; report either way.
2. For one case per system, rebuild the bands with `L = 50` (same durations) and report the relative errors of linear,
   IFFT-FFT, pairwise, B1 at `L = 50` next to `L = 5` (the paper's Fig. 4 used both). Save everything in
   `data/b1_variants_iter006.md`.

## Tests

`tests/test_lean_adapter.py`: `XHMCoarseEvaluator.coarse_modes` reproduces `CoarseModes.from_model(bands, XHMModel, modes)`
to 1e-12 and `coarse_total` reproduces `CoarseWaveform.from_model(bands, model.total_direct)`; the prepared evaluator is
re-usable across parameter changes (two parameter sets, no stale results); `pad_factor`/`all_half_bins` options agree with
the defaults when `p = 1`, `all_half_bins = False`; keep the default test run under ~2 minutes.

## Rules (from AGENTS.md)

- Create/modify files only inside this directory; no git state changes; no package installation; no network.
- Do not change the behaviour or defaults of iteration 001-005 code (adding methods and options is fine; say what you added).
- Keep LAL stderr quiet in logs; report any parameter set LAL refuses.

## Completion criteria / verification

- `python -m pytest tests -q` passes; the three scripts run and write their tables.
- Final report: the lean-adapter timing table with route ratios (and the three-detector sums), the pairwise tuning table
  with the per-band attribution and your diagnosis, the B1 variants table and the L = 50 table, and anything the
  orchestrator should know.
