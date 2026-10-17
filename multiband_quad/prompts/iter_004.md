# Iteration 004 — method B for the TOTAL waveform only: B0 (centred block, no FFT) and B1 (baseband IFFT-FFT)

You are working in `multiband_quad/` (the current directory). Read `AGENTS.md`, `TASK.md` (sections 2-4, 7:
"方法 B0" and "方法 B1"), `notes/method_idea.md` (sections 3, 7 and 8.8) and the code of iterations 001-003
(`mbquad/bands.py`, `psd.py`, `coarse.py`, `linear.py`, `ifftfft.py`, `blockweights.py`, `pairwise.py`, `timing.py`,
`testcases.py`). `notes/orchestrator_toy_methodB.py` is a validated quick implementation of B1 (function
`hh_methodB`, with the adaptive rule and the half-sample shift); use it to check conventions.

## Iteration-003 API you must build on (do not change its behaviour)

- `mbquad.blockweights.BlockWeights(bands, psdgrid)`: `A_m` (dict by `N_b`), `fine_inverse_psd(N_b)`,
  `weights(b, m0, Nhat, kappa=None)` (complex `W_kappa(B)`, `B = {m0, ..., m0 + Nhat - 1}` mod `N_b`; `Nhat | N_b`),
  `centred(b, Nhat)` (block `{-Nhat//2, ..., Nhat - Nhat//2 - 1}`), `for_lag_interval(b, lag_lo, lag_hi, Nhat)` ->
  `(W, start_s, end_s)`, `lag_audit(b, g_fine, Nhat, m0)` -> dict with `outside_fraction, lag_centroid, lag_quantiles,
  folding_error, direct_error, fine_sum, quadrature_sum` (`g_fine` on `psdgrid.f_k` or a full length-`N_b` sequence),
  `nbytes()`.
- `mbquad.pairwise.PairwiseHH(bands, psdgrid, modes, L_margin=5.0, margin_units="sigma_pair", mchirp_range=None,
  diag="block", impl="pairs")`: `__call__(coarse_modes)`, `summary()`, `format_summary()`, `class_weights`, `W`.
- `mbquad.coarse.CoarseModes(bands, modes, h_b)`, `.from_model(bands, model, modes)`, `.total()` -> `CoarseWaveform`.
- `BandSetup.lag_pair(f, m_slow, m_fast, mchirp=None)`, `lag_interval_pair(b, m_slow, m_fast, mchirp=None)`,
  `lag_max_pair(b, m_slow, m_fast)`; `IFFTFFTHH.positive_grid_waveform(coarse, b)` -> `(f, h_tilde_c)` on all
  positive bins of the `Nhat` grid (use this, not the band-restricted `half_point_waveform`, for exact identities).
- Iteration-003 numbers on `toy_small()` five modes with lines: pairwise 3.53e-7 (IFFT-FFT 2.22e-7, linear 1.57e-6);
  `(h,h)` step: pairwise pairs 0.22 ms, IFFT-FFT fast 0.71 ms.

## The methods (authoritative summary)

Both take only the TOTAL waveform on the `(d, h)` coarse grids (`CoarseWaveform`), like the paper's methods.

B0 ("centred block on the total"): per band, `(h,h)_b ~= (4/T) sum_kappa Re[W^0_kappa] w^(b)(f_kappa) |h(f_kappa)|^2`
with `W^0 = BlockWeights.centred(b, Nhat=M_b)` (block `{-M_b/2, ..., M_b/2 - 1}` on the coarse grid). It is exact when
the lag support of `w |h|^2` lies within `[-T_b/2, T_b/2]`; otherwise its error is the folding error
`(1/N_b) sum_{m not in B} G_m (conj A_{m~} - conj A_m)` (see `blockweights.lag_audit`), which stays small when the PSD
kernel `A_m` is small at the lags that fold onto small `|m~|`. Cost: one real weight vector per band, `O(K_MB)`, no FFT,
no modes. In this language the paper's linear interpolation is "period `T_b` folding + Fejer taper", B0 is "period
`T_b` folding + rectangular crop", and IFFT-FFT is "period `2 T_b` folding + rectangular crop".

B1 ("baseband IFFT-FFT"): exact under the paper's own time-limitation assumption (Eq. 36), but with the FFT lengths
reduced from `(M_b, 2 M_b)` to `(K', K')`, `K' = scipy.fft.next_fast_len(K_b)`: with `x_j = sqrt(w) h(f_{K_s + j})`
(`j = 0 .. K_b - 1`), `y = ifft(x, K')` is the band waveform demodulated by `f_{K_s}` and sampled with step `T_b/K'`;
under the support assumption `[T - T_b, T]` (indices `n' = n` because `T/T_b` is an integer) the half points are
`z = fft(y * exp(-i pi n / K'))`, `z_j = sqrt(w) h(f_{K_s + j + 1/2})` for `j < K_b`. Then
`(h,h)_b ~= (4/T) [ sum_kappa W^{2M}_{2 kappa} w |h|^2(f_kappa) + sum_kappa W^{2M}_{2 kappa + 1} |z|^2 ]` with
`W^{2M} = BlockWeights.weights(b, m0=-M_b, Nhat=2 M_b)` (the symmetric block of Eq. 38; `W^{2M}` restricted to the
band equals `(That/T) I_tilde_c` of `IFFTFFTHH` up to roundoff — test this). Adaptive rule: if the predicted two-sided
lag support of the total `|h|^2` fits in the centred block of length `M_b`, i.e.
`lag_max(b) + margin <= T_b/2` with `lag_max(b) = bands.lag_max_pair(b, m_max, 1)` (use the model's `m` values actually
present, `m_min` instead of 1 when (2,1) is absent) and `margin = L sigma_mmax(f_lo)` by default, use B0 for that band
(no FFT); band 0 (`M_b = N_b`) is always exact with the centred block. Expose `force_half_points=True` to disable the
rule. Note the IFFT-FFT's `Nhat = min(2 M_b, N_b)` cropping (Eq. 37) is also respected: when `2 M_b >= N_b` the band
needs no interpolation at all.

## Deliverables

1. `mbquad/totalweights.py` — `class CentredBlockHH(bands, psdgrid)` (method B0): per band the real weight vector
   `Re W^0_kappa w(f_kappa)` on the coarse grid; `__call__(coarse)`, `band_contributions(coarse)`; a method
   `predicted_lag_support(b, m_values, mchirp=None)` returning `(lag_max, T_b/2)`; and `folding_error_estimate(coarse_fine_g, b)`
   based on `blockweights.lag_audit` for diagnostics (fine-grid input).
2. `mbquad/baseband.py` — `class BasebandIFFTFFTHH(bands, psdgrid, m_values=(1, 2, 3, 4), L_margin=5.0, force_half_points=False)`
   (method B1 with the adaptive rule): precompute per band `K'`, the half-sample phase `exp(-i pi n/K')`, the two weight
   vectors (`W^{2M}` at even and odd indices restricted to the band, with `w` folded in) or the B0 vector when the rule
   says so; `__call__(coarse)`, `band_contributions(coarse)`, `half_point_waveform(coarse, b)` (compare with
   `IFFTFFTHH.half_point_waveform` on the half points: they must agree to 1e-10 relative in the band interior and to
   ~1e-8 overall), `fft_lengths()` -> per band `(K', mode)` where mode is `"exact"`, `"B0"` or `"B1"`, and a Markdown
   `summary()` listing per band `K_b, K', M_b, 2 M_b, lag_max, T_b/2, mode`. Use `scipy.fft` with preallocated buffers
   (`workers=1`); avoid per-call allocations where easy.
3. `scripts/accuracy_iter004.py`: on `toy_small()` (lines and smooth PSD; quadrupole and five modes) the relative `(h,h)`
   error and `|Delta ln Lambda|` at SNR 25 of: linear, IFFT-FFT (fast), pairwise (iteration 003 defaults), B0 (all bands),
   B1 adaptive, B1 forced half points; plus per-band contributions of the error for B0 and B1 (which bands fail for B0
   and why, with the predicted lag support). Also the same table on a second toy configuration with strong HOM
   amplitudes (ratios (1, 0.3, 0.6, 0.1, 0.4), same geometry) and on the GW190814-like geometry of iteration 001 with
   `ToyModes(mchirp=6.4, t_c=T-2)` and the real PSD `data/psd_L1_GW190814_welch16s.txt` loaded with
   `mbquad.psd.load_asd_or_psd` (20-1024 Hz; note `dt = 1/4096`). Save `data/accuracy_iter004.md`.
4. Extend `scripts/bench_hh_step.py`: add B0 and B1 (adaptive and forced) to the `(h,h)`-step timing table for `toy_small`
   and the GW190814-like geometry; print the FFT-length summary next to the paper's `(M_b, 2 M_b)`.
   Save `data/bench_hh_step_iter004.md`.
5. `tests/test_total_methods.py`:
   - B1 forced half points agrees with IFFT-FFT (fast) to 1e-8 relative on `toy_small()` (both waveforms), and band
     0 is exact;
   - `W^{2M}` restricted to the band equals `(That_b/T) I_tilde_c` of `IFFTFFTHH` to 1e-10 (relative to max);
   - B1 adaptive differs from B1 forced by less than 1e-7 relative on `toy_small()` and uses `"B0"` in the top band(s);
   - B0 relative error `< 1e-5` on `toy_small()` five modes (document the achieved value) and worse than B1;
   - the half-sample-shift interpolation reproduces a pure (2,2) mode evaluated directly at the half points to
     1e-4 relative in the window plateau of band 3;
   - `fft_lengths()` are all `<= M_b/2` for `toy_small()` bands with `b >= 1` (document the achieved ratios).

## Rules (from AGENTS.md)

- Create/modify files only inside this directory; no git state changes; no package installation. Do not change the
  behaviour of iteration 001-003 code (adding methods is fine; say what you added). Keep `python -m pytest tests -q`
  under ~2 minutes. `MPLCONFIGDIR=/tmp/mplconfig` if you plot (plots optional here).

## Completion criteria / verification

- `python -m pytest tests -q` passes; both scripts run and write their Markdown tables.
- Final report: files, test summary, the accuracy tables (toy_small, strong-HOM toy, GW190814-like with the real PSD),
  the FFT-length table (B1 vs paper) and the timing table, and anything the orchestrator should know (in particular
  where B0 fails and whether the real PSD's narrow lines change the picture).
