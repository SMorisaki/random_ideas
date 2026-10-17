# Iteration 007 — population study with IMRPhenomXHM: error distributions (Fig. 4 protocol), real PSDs, t_c scan

You are working in `multiband_quad/` (the current directory). Read `AGENTS.md`, `TASK.md` (sections 3-4: completion
criteria C1 and C1'; section 7), and the code of iterations 001-005 (`mbquad/lalwave.py`, `testcases.py`, `pairwise.py`,
`baseband.py`, `totalweights.py`, `ifftfft.py`, `linear.py`, `reference.py`). Iteration 005 established the XHM interface
(`XHMModel`, `xhm_systems()`, `xhm_case()`), iteration 006 the prepared evaluator `XHMCoarseEvaluator` (use it: it avoids
per-call sorting), the B1 options `pad_factor` / `all_half_bins`, and `PairwiseHH.band_contributions`. Read
`data/xhm_accuracy_iter005.md`, `data/xhm_timing_iter006.md`, `data/pairwise_tuning_iter006.md`, `data/b1_variants_iter006.md`.

## Goal

Produce the accuracy evidence for completion criteria C1 (method A, pairwise) and C1' (method B1), in the paper's Fig. 4
style: distributions of `|Delta ln Lambda| = (SNR^2/2) |Delta (h,h)|/(h,h)_exact` at single-detector SNR 25 (also report the
scale factors for SNR 50 and 100, which are simply x4 and x16) for linear interpolation, IFFT-FFT (fast), pairwise (defaults,
prior-corner `mchirp_range` as below), B1 adaptive (default, band-only) AND B1 with `all_half_bins=True` (p = 1), and B0,
with `L = 5` and `L = 50` (same durations), for three populations and two PSDs. The `L = 50` bands have more coarse points
(report `K_MB` for both); at `L = 50` the adaptive B1 is expected to select B0 in every truncated band (report the modes).

## Populations (seeded, reproducible; `mbquad/population.py`)

- BNS: `m1, m2 ~ U(1.2, 2.0)` Msun (component masses, `m1 >= m2`), `chi1z, chi2z ~ U(-0.05, 0.05)`; `T = 256 s`, `dt = 1/4096`,
  20-2048 Hz, durations 256..4 s, `mchirp_ref = 1.0` (prior minimum for these masses is ~1.04), `mchirp_range = (1.0, 1.8)`.
- NSBH / GW190814-like: `m1 ~ U(15, 35)`, `m2 ~ U(1.5, 4)`, `chi1z ~ U(-0.5, 0.5)`, `chi2z = 0`; `T = 16 s`, 20-1024 Hz,
  durations 16, 8, 4 s, `mchirp_ref = 4.0`, `mchirp_range = (4.0, 9.0)`.
- BBH: `m1 ~ U(30, 80)`, `m2 ~ U(10, 40)` with `q = m2/m1 >= 0.1`, `chi1z, chi2z ~ U(-0.8, 0.8)`; `T = 16 s`, 20-1024 Hz,
  durations 16, 8, 4 s, `mchirp_ref = 12.0`, `mchirp_range = (12.0, 50.0)`.
  (If Eq. 51 has no admissible root or the pairwise slack is negative for a prior corner, say so and adjust the reference
  chirp mass or `T` minimally; report what you changed.)
- Common: inclination isotropic (`cos iota ~ U(-1, 1)`), `phi_ref ~ U(0, 2 pi)`, `F_+ = cos 2 psi`, `F_x = sin 2 psi` with
  `psi ~ U(0, pi)` (a single detector with unit antenna pattern norm), `t_c ~ U(T - 2.12, T - 1.9)`, distance fixed (the SNR
  normalisation removes it). `N = 100` draws per population (seed 20261017), plus 20 "strong-HOM" draws per NSBH/BBH
  population with `q <= 0.15` (but chirp mass still >= the reference, i.e. inside the prior the bands were built for) and
  `iota in (1.3, 1.8)`. Every drawn point must satisfy `mchirp >= mchirp_ref` (reject and redraw otherwise, count rejections).
- PSDs: analytic aLIGO + synthetic lines (use Lorentzian-like narrow lines: add to `mbquad.psd` an `add_lorentzian_lines`
  with widths 0.01-0.5 Hz and heights 1e2-1e4 placed in every band and within `Df` of band edges of the BNS setup; document
  the list), and the real L1 PSD (`data/psd_L1_GW190814_welch16s.txt`, `skip_dc=True`). For the BNS the real PSD's 1/16 Hz
  resolution is interpolated by `PSDGrid`; say so.

## Deliverables

1. `mbquad/population.py`: samplers above (`sample_population(name, n, seed, strong_hom=False)` -> list of parameter dicts)
   and `evaluate_point(params, setup, psdgrid, bands, evaluators, L)` returning a dict of relative errors per method; the
   exact `(h,h)` must be computed on the fine grid through the sequence path (`XHMModel.total_direct`; for the BBH keep the
   iteration-005 tail handling consistent between the reference and the coarse modes), cached per
   (population, index, PSD) in `data/population_cache_<pop>_<psd>.npz` (git-ignored) because it is independent of `L` and
   method; the coarse evaluations for `L = 5` and `L = 50` reuse it. Keep memory bounded (do not hold all BNS fine-grid
   waveforms at once).
2. `scripts/population_iter006.py [--pop bns|nsbh|bbh|all] [--psd lines|L1|both] [--n 100]`: runs the study, writes
   `data/population_iter006_<pop>_<psd>.json` (per point: parameters, SNR-25 `|Delta ln Lambda|` per method and `L`,
   the pairwise minimum slack) and a Markdown summary `data/population_iter006.md` with, per population / PSD / `L`:
   median, 90th, 99th percentile and maximum of `|Delta ln Lambda|` per method, the ratio pairwise/IFFT-FFT and B1/IFFT-FFT
   of the medians and 99th percentiles, and the strong-HOM subset separately. Expected run time: BNS exact references
   dominate (~1-2 s per point); budget ~10 minutes per population per PSD; if it is longer, reduce `n` for the BNS and say so.
3. `scripts/plot_iter006.py`: figures `figures/iter006_errors_<pop>.png` (per population: `|Delta ln Lambda|` at SNR 25 vs
   the point index or vs `ln Lambda = SNR^2/2` scatter as in Fig. 4, colour by method, panels for `L = 5` and `50` and for the
   two PSDs) and `figures/iter006_cdf.png` (empirical CDFs of `|Delta ln Lambda|` per method, all populations). Matplotlib
   `Agg`, `MPLCONFIGDIR=/tmp/mplconfig`.
4. `scripts/tc_scan_iter006.py`: for one NSBH and one BBH point, scan `t_c` over 21 values in `[T - 2.12, T - 1.9]` and report
   the relative `(h,h)` of pairwise (must be constant to ~1e-12 relative), IFFT-FFT, B1 and linear (which vary); table
   `data/tc_scan_iter006.md` and a small figure.
5. `tests/test_population.py`: sampler determinism and prior ranges; `evaluate_point` on one NSBH point agrees with a direct
   evaluation; the cache round-trips; `add_lorentzian_lines` shape; keep the default test run under ~2 minutes (mark the
   population runs `slow`).

## Budget and threading

- Set `OMP_NUM_THREADS=1` in the scripts (iteration 006 found the default 4 threads distort timings; here it only affects
  speed, but keep it for reproducibility). Expected cost: BNS exact references ~1-2 s per point (one 5e5-point XHM call
  per PSD) -> ~100 points x 2 PSDs ~ 5-7 min; coarse evaluations (2 L values x 6 methods) are cheap. If the whole study
  exceeds ~25 minutes, reduce the BNS `n` to 50 and say so.

## Rules (from AGENTS.md)

- Create/modify files only inside this directory; no git state changes; no package installation; no network.
- Do not change the behaviour of iteration 001-005 code (adding helpers is fine; say what you added).
- Large per-point dumps go to `data/*_full.json` or `.npz` (git-ignored); the committed JSON summaries stay under ~100 KB
  (store per-point numbers compactly, e.g. 6 significant digits).

## Completion criteria / verification

- `python -m pytest tests -q` passes; the population script completes for all three populations with both PSDs (or
  documents a reduced `n` for the BNS with the reason); the Markdown summary, JSON files and figures exist.
- Final report: the summary table (median / 99th percentile per method, population, PSD, `L`), the pairwise/IFFT-FFT and
  B1/IFFT-FFT ratios, the strong-HOM subset, the `t_c` scan result, any points where Eq. 51 or the slack check failed, and
  anything else the orchestrator should know.
