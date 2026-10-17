# Iteration 005 — IMRPhenomXHM interface (mode by mode on arbitrary frequency sequences), real systems, timing breakdown

You are working in `multiband_quad/` (the current directory). Read `AGENTS.md`, `TASK.md` (sections 2-4, 7; iteration 005
is the go/no-go point for method A as a practical speed-up), `notes/method_idea.md` §8.9-8.10, and the code of iterations
001-004 (`mbquad/*.py`, especially `coarse.py`, `pairwise.py`, `totalweights.py`, `baseband.py`, `testcases.py`).
`python -c "import lalsimulation, lal"` works in the environment (lalsimulation 6.2.0).

## Iteration-004 API (in addition to iterations 001-003; do not change its behaviour)

- `mbquad.totalweights.CentredBlockHH(bands, psdgrid)` (method B0): `__call__(coarse)`, `band_contributions`,
  `predicted_lag_support(b, m_values, mchirp=None)`, `folding_error_estimate(fine_g, b)`.
- `mbquad.baseband.BasebandIFFTFFTHH(bands, psdgrid, m_values=(1,2,3,4), L_margin=5.0, force_half_points=False)` (method B1):
  `__call__(coarse)`, `band_contributions`, `half_point_waveform`, `fft_lengths()`, `summary()`.
- `mbquad.psd.load_asd_or_psd(path, skip_dc=True)` for the Welch files in `data/` (they contain an f = 0 row).
- Iteration-004 numbers (relative `(h,h)` error, GW190814-like toy geometry with the real L1 PSD, five modes): linear 2.95e-5,
  IFFT-FFT 6.0e-7, pairwise 5.8e-7, B0 4.8e-6, B1 6.7e-7; `(h,h)` step: B1 0.14 ms, B0 0.019 ms, pairwise 0.16 ms,
  IFFT-FFT fast 0.68 ms.

## Facts established by the orchestrator (use them, verify them)

- `lalsimulation.SimIMRPhenomXHMFrequencySequenceOneMode(freqs, m1_SI, m2_SI, chi1z, chi2z, ell, emm, distance_SI, phiRef,
  fRef, lalParams)` returns ONE `lal.COMPLEX16FrequencySeries` with the mode `h_{ell, emm}(f)` on the given
  `lal.REAL8Sequence` of positive frequencies (`emm` may be negative; for (3,3) the orchestrator observed
  `h_{3,-3}(f) = -conj(h_{3,3}(f))` at the sampled points, i.e. the non-precessing relation
  `h_{l,-m}(f) = (-1)^l conj(h_{l,m}(-f))` with the series understood accordingly). Measured cost at 1e4 points for
  23+2.6 Msun: (2,2) 5.7 ms, (2,1) 1.5 ms, (3,3) 1.5 ms, (3,2) 8.1 ms, (4,4) 2.9 ms (sum ~20 ms).
- `lalsimulation.SimInspiralChooseFDWaveformSequence(phiRef, m1_SI, m2_SI, S1x, S1y, S1z, S2x, S2y, S2z, fRef, distance_SI,
  inclination, lalParams, lalsimulation.IMRPhenomXHM, freqs)` returns `(hplus, hcross)` series on the same sequence:
  11.7 ms at 1e4 points. Default XHM mode content: (2,|2|), (2,|1|), (3,|3|), (3,|2|), (4,|4|).
- `lalsimulation.SimInspiralChooseFDModes(...)` and `SimIMRPhenomHMGethlmModes` (IMRPhenomHM, takes a frequency
  sequence) exist; check their signatures with `help()` / `__doc__` and use them only if convenient.
- An independent reviewer verified (lalsimulation 6.2.0) and you must re-verify by test: the `emm < 0` output of
  `OneMode` is the series that combines linearly with `h_+, h_x` at `f > 0` (`h_{l,-m}(f) = (-1)^l conj h_{l,m}(f)` at the
  sampled points); the polarization coefficients are the LAL `AddMode` factors for `(l, -m)` at azimuth `pi/2`:
  `c_+ = 0.5 (Y_{l,-m}(iota, pi/2) + (-1)^l conj Y_{l,m}(iota, pi/2))`, `c_x = 0.5 i (Y_{l,-m} - (-1)^l conj Y_{l,m})` with
  `Y = lal.SpinWeightedSphericalHarmonic(iota, azimuth, -2, l, m)`; `phiRef` enters each mode as `e^{-i m phiRef}`; XHM has
  no (4,3) mode. IMRPhenomHM `SimIMRPhenomHMGethlmModes` (deltaF = 0 for a sequence) uses the opposite convention
  (positive `m` at positive frequencies, AddMode `(l, +m)` at azimuth 0) and is NOT scaled by `(G M/c^3)^2 c/D_L`.
  Reviewer's warm timings: 5x XHM OneMode 11.8 ms vs polarizations 9.5 ms at 19000 points (BNS grid); 12.8 vs 16.5 ms
  at 16064 points; HM GethlmModes 45 ms vs HM polarizations 49 ms; faithful IFFT-FFT (scipy, preallocated) 3.0 ms (BNS)
  and 0.55 ms (GW190814). Your numbers should be in this range; investigate if not (cold calls, LAL warnings, sorting).
- Conventions you must establish by TEST, not by assumption: how `h_+ - i h_x = sum_{l,m} {}_{-2}Y_{lm}(iota, phi) h_{lm}`
  maps to positive-frequency series in LAL (which of `+m` / `-m` contributes at `f > 0`, the role of `phiRef`, the
  `fRef` used by `OneMode` vs `ChooseFDWaveformSequence`), such that the sum over your per-|m| detector-frame
  contributions reproduces `F_+ h_+ + F_x h_x` from `ChooseFDWaveformSequence` to 1e-10 relative (max-abs normalised)
  on a 2000-point random frequency set in [20, 1024] Hz for several `(iota, phi_ref, F_+, F_x)`.

## Deliverables

1. `mbquad/lalwave.py`
   - `class XHMModel(m1, m2, chi1z, chi2z, distance_mpc, inclination, phi_ref, t_c, f_ref=20.0, F_plus=1.0, F_cross=0.0,
     modes=((2,2),(2,1),(3,3),(3,2),(4,4)))` with the same interface as `ToyModes`: `mode(f, lm)` returns the
     detector-frame contribution of the |m| pair (`+m` and `-m` combined with the spin-weighted spherical harmonics
     `lal.SpinWeightedSphericalHarmonic(iota, phi, -2, l, m)` and `F_+`, `F_x`), already multiplied by
     `exp(-2 pi i f t_c)` so that the signal peaks at `t_c` seconds from the start of the segment (LAL series are
     referenced to the coalescence at t = 0; document the sign); `total(f)` = sum of `mode`; `total_direct(f)` =
     `F_+ h_+ + F_x h_x` from `ChooseFDWaveformSequence` times the same phase; `tau_lm(f, lm)` = Newtonian
     `tau_0pn(2 f/m, mchirp)` (document that this is only the leading-order track). Accept unsorted frequency arrays
     (LAL requires increasing sequences: sort, evaluate, unsort). Cache nothing between calls (timings must be honest).
   - `polarizations(f)` returning `(h_+, h_x)` arrays, and `raw_modes(f)` returning the raw LAL series (the `emm < 0`
     outputs, or whichever you established) as a dict `(l, |m|) -> array`, plus `coefficients()` -> dict `(l, |m|) -> complex`
     such that `mode(f, lm) = coefficients()[lm] * raw_modes(f)[lm] * exp(-2 pi i f t_c)` (document the formula).
   - `mchirp` property; `snr_scale(psdgrid, ...)` is NOT needed here (we normalise by the exact `(h,h)`).
1b. `mbquad/pairwise.py`: add `PairwiseHH.mode_matrix(coarse_modes) -> ndarray (n_modes, n_modes) complex Hermitian` such
   that `(h, h) = (4/T) C^dagger M C` for the coefficient vector `C` when `coarse_modes` holds the RAW modes (without
   coefficients and without the `t_c` phase), i.e. `M_{AA'} = sum_b sum_kappa conj(H_A) W^(b)_{AA',kappa} H_{A'}`
   (`W` already contains the window); and a helper `hh_from_matrix(M, C)`. Test (C5 of TASK.md): on system (iii),
   `hh_from_matrix(mode_matrix(raw), C)` equals `PairwiseHH(coarse_modes_with_coefficients)` to 1e-12 relative; `M` built
   from raw modes is identical (to roundoff) for two different `t_c`, distances, `phi_ref`, inclinations and `(F_+, F_x)`
   (only `C` changes); and the pairwise `(h,h)` is independent of `t_c` to 1e-12 relative while the IFFT-FFT value changes
   by more than 1e-9 relative between `t_c = T - 2.1` and `T - 1.9` (document the observed change).
2. `mbquad/testcases.py`: add `xhm_systems()` returning the three documented systems (dicts with all parameters):
   (i) BNS: `m1 = m2 = 1.4`, `chi = 0`, `T = 256 s`, `dt = 1/4096`, 20-2048 Hz, durations 256..4 s, `mchirp_ref = 1.15`,
   `m_max = 4`, `L = 5`, `t_c = T - 2.0`, inclination 1.0 rad, distance 40 Mpc;
   (ii) GW190814-like: `m1 = 23, m2 = 2.6`, `chi1z = 0.1, chi2z = 0`, `T = 16 s`, 20-1024 Hz, durations 16, 8, 4 s,
   `mchirp_ref = 6.4`, inclination 0.8 rad, distance 240 Mpc; (iii) BBH: `m1 = 60, m2 = 20`, `chi1z = 0.3, chi2z = -0.2`,
   `T = 16 s`, 20-1024 Hz, durations 16, 8, 4 s, `mchirp_ref = 25`, inclination 1.3 rad, distance 800 Mpc; all with
   `phi_ref = 0.7`, `F_+ = 0.6`, `F_x = 0.5`. Provide `xhm_case(name, psd="analytic_lines"|"L1_real"|"H1_real")` returning
   `ToyCase`-like `(model, psdgrid, bands)` where the real PSDs come from `data/psd_{L1,H1}_GW190814_welch16s.txt` via
   `load_asd_or_psd` (document that the Welch resolution is 1/16 Hz and that `PSDGrid` interpolates it).
3. `scripts/xhm_accuracy_iter005.py`: for the three systems, with the analytic+lines PSD and the real L1 PSD, and for
   inclinations `{0.3, 1.0, 1.4}` rad (and additionally `q = m2/m1 = 0.1` edge-on variants of (ii) and (iii) at
   `iota = 1.5`), compute the exact `(h,h)` on the fine grid (`total_direct` on `psdgrid.f_k`; for the BNS this is a
   5e5-point XHM call, fine) and the relative error / `|Delta ln Lambda|` at SNR 25 of: linear, IFFT-FFT (fast),
   pairwise (defaults, `mchirp_range` = the prior corners of TASK.md: (1.15, 1.25), (6, 7), (25, 35)), B0 and B1
   adaptive. Also report the pairwise `summary()` minimum slack and the lag-audit out-of-block fractions for the (4,4)x(2,1)
   class in bands 1 and B-1 of system (ii) using the XHM modes on the fine grid (this checks the SPA interval on a real
   model including merger-ringdown). Save `data/xhm_accuracy_iter005.md` (+ JSON) and a figure
   `figures/iter005_xhm_errors.png` (bar chart of `|Delta ln Lambda|` per method and system; `MPLCONFIGDIR=/tmp/mplconfig`).
   Compute the fine-grid exact reference through the SAME sequence path (`total_direct` on `psdgrid.f_k`; for the BNS this
   is one 5e5-point call per PSD and inclination; cache it in `data/xhm_reference_cache.npz`, git-ignored) and check the
   partition of unity (`hh_exact_banded == hh_exact`) before evaluating any coarse method.
4. `scripts/xhm_timing_iter005.py` — the go/no-go table. For systems (i) and (ii) on their coarse grids (K_MB from the
   band setups): median wall time (`mbquad.timing.median_time`, `number` chosen so one batch takes ~1 s) of
   (a) `ChooseFDWaveformSequence` XHM polarizations, (b) the sum of the five `OneMode` calls (+m and -m: ten calls, or
   report both), (c) IMRPhenomHM `SimIMRPhenomHMGethlmModes` on the sequence if its Python binding works (else say so),
   (d) the per-call Python overhead of assembling detector-frame modes (`XHMModel.mode` for all modes minus the raw LAL
   time), and the `(h,h)` steps: linear, IFFT-FFT paper, IFFT-FFT fast, pairwise (pairs / einsum), B0, B1 adaptive.
   Then the derived quantities: total likelihood-side cost "waveform + (h,h) step" for the IFFT-FFT route
   (polarizations + fast IFFT-FFT), the method-A route (modes + pairwise), the B1 route (polarizations + B1) and the
   B0 route; and the ratio to the IFFT-FFT route. Save `data/xhm_timing_iter005.md`.
5. `tests/test_lalwave.py` (keep under ~30 s total): mode-sum vs `total_direct` identity to 1e-10 for two systems and
   two `(iota, phi_ref, F_+, F_x)` each; unsorted-frequency consistency; `t_c` phase convention (the inverse FFT of the
   windowed total on a short BBH segment peaks within 10 ms of `t_c`); pairwise and B1 errors on system (iii) with the
   analytic PSD below 2e-6 relative (document achieved values); linear interpolation worse than IFFT-FFT for system
   (ii) at `iota = 1.4`.

## Rules (from AGENTS.md)

- Create/modify files only inside this directory; no git state changes; no package installation; no network.
- Do not change the behaviour of iteration 001-004 code (adding helpers is fine; say what you added).
- LAL prints warnings to stderr for some parameter corners; keep them from flooding the logs (`lal.swig_redirect_standard_output_error(False)` or
  filtering) and report any parameter set XHM refuses.

## Completion criteria / verification

- `python -m pytest tests -q` passes (mark the BNS fine-grid cases `slow` if needed; the default run stays under ~2 minutes).
- Both scripts run and write their tables; the BNS fine-grid exact `(h,h)` is computed at least once per PSD.
- Final report: files, test summary, the accuracy table, the lag-audit numbers for XHM, the timing table with the route
  comparison, and your own reading of the go/no-go question for method A (where does the per-mode waveform cost leave it?).
