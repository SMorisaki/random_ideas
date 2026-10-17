# Iteration 002 — the paper's two (h, h) methods: linear interpolation and IFFT-FFT, plus a timing harness

You are working in `multiband_quad/` (the current directory). Read `AGENTS.md`, `TASK.md` (sections 2-4, 7),
`notes/paper_summary.md` (Sec. II.D: Eqs. 26-46) and the code from iteration 001 (`mbquad/bands.py`,
`mbquad/psd.py`, `mbquad/toywave.py`, `mbquad/reference.py`, `mbquad/testcases.py`). `notes/orchestrator_toy.py`
contains a quick-and-dirty but *validated* implementation of both methods (functions `hh_linear`, `hh_ifftfft`
and the precomputation inside the `bands` loop); use it to check your understanding of the conventions,
but write clean code against the iteration-001 API.

## Iteration-001 API you must build on (do not change its behaviour)

- `BandSetup` (`mbquad/bands.py`): attributes `T, dt, N, f_low, f_high, B, T_b, N_b, dt_b, M_b, K_s, K_e, K_b, f_b, Df_b,
  K_MB, K_orig`; methods `window(f, b)`, `sqrt_window(f, b)`, `coarse_frequencies()` (list of `CoarseGrid(f, kappa, w)`),
  `all_coarse_frequencies()` (`AllCoarseGrid(f, kappa, w, band_index)`), `lag_max_pair(b, m_slow, m_fast)`.
  Note that `N_b` may exceed `N` (upper taper above Nyquist) and that band 0 has `M_b = N_b`.
- `PSDGrid` (`mbquad/psd.py`): `f_k` (k = 1 .. N/2 - 1), `S_k` (+inf outside `[f_low, f_high]`), `inv_S`.
- `ToyModes` (`mbquad/toywave.py`): `mode(f, lm)`, `total(f)`, `tau_lm(f, lm)`; `modes` dict `{(l, m): (ratio, phase, growth)}`,
  so a quadrupole-only model is `ToyModes(mchirp, t_c, modes={(2, 2): (1.0, 0.0, 0.0)})`.
- `mbquad.testcases.toy_small()` returns `ToyCase(model, psdgrid, bands)`; `mbquad.reference.hh_exact(h_fine, psdgrid)`,
  `hh_exact_banded(model_total, psdgrid, bands)`, `lnlambda_error(hh_approx, hh_exact, snr)`.

## Conventions (authoritative)

- A "coarse waveform" is the model evaluated on `BandSetup.all_coarse_frequencies()`; provide a small container
  `CoarseWaveform` (or plain helpers in `mbquad/coarse.py`) that holds the per-band complex arrays (one per band,
  indices `kappa = K_s^(b) .. K_e^(b)`) and can be built from (i) a callable `model(f)` or (ii) one concatenated array.
  Both (h, h) methods take a `CoarseWaveform` of the TOTAL waveform.
- All sums are one-sided (positive frequencies) with the paper's `4/T` normalisation; `S_k = inf` outside
  `[f_low, f_high]` so `1/S_k = 0` there, and the windows `w^(b)` are applied on the coarse grid exactly as in Eq. 24/28/46.

## Deliverables

1. `mbquad/linear.py` — `class LinearInterpolationHH` (Eqs. 27-30).
   Precompute `c^(b)_kappa` for every band from the fine grid: with `tilde f` used ONLY in the summation ranges
   (`tilde f_{K_s-1} = tilde f_{K_s} = 0`, `tilde f_{K_e} = tilde f_{K_e+1} = inf`, otherwise `tilde f_k = f_k`) and
   the TRUE coarse frequencies in the weights `(f_{k+1} - f_l)` and `(f_l - f_{k-1})`:
   `c_k = (4 T^(b)/T) [ sum_{tilde f_k <= f_l < tilde f_{k+1}} (f_{k+1} - f_l) w^(b)(f_l)/S_l + sum_{tilde f_{k-1} <= f_l < tilde f_k} (f_l - f_{k-1}) w^(b)(f_l)/S_l ]`
   (for `k = K_s` the second sum is empty; for `k = K_e` the first sum is empty). Vectorise with `np.searchsorted` /
   `np.add.reduceat` or `np.bincount`; no Python loop over fine-grid points. Method `__call__(coarse) -> float`
   returning `sum_b sum_k c_k |h(f_k)|^2`.
2. `mbquad/ifftfft.py` — `class IFFTFFTHH` (Eqs. 31-46), faithful to the paper:
   - precompute per band `I_m = (2/T) Re sum_{k=1}^{floor((N^(b)-1)/2)} e^{2 pi i k m/N^(b)} / S_k` (Eq. 32; compute it
     with one inverse FFT of length `N^(b)`), `Nhat^(b) = min(2 M^(b), N^(b))` (Eq. 37), the cropped `I_c` (Eq. 38: the
     first `ceil(Nhat/2)` and the last `floor(Nhat/2)` samples) and `tilde I_{c,k} = dt^(b) sum_m I_{c,m} e^{-2 pi i k m/Nhat}`
     (Eq. 43); store only the real part for `k = 1 .. floor((Nhat-1)/2)` (document why it is real up to rounding).
   - `__call__(coarse) -> float`: per band build the real time series `hat h_{c,m}` of Eq. 46 from the `M^(b)` coarse
     samples `sqrt(w) h(f_k)` placed at indices `k` of a length-`M^(b)` one-sided spectrum (zero elsewhere), i.e.
     `hat h_m = (2/T^(b)) Re sum_k sqrt(w_k) h_k e^{2 pi i k m/M^(b)}` for `m = 0..M^(b)-1`, placed in the LAST `M^(b)`
     of `Nhat^(b)` samples (zeros before), then `tilde h_{c,k} = dt^(b) sum_m hat h_{c,m} e^{-2 pi i k m/Nhat}` and
     `(h,h)_b = (4/That^(b)) sum_{k=1}^{floor((Nhat-1)/2)} tilde I_{c,k} |tilde h_{c,k}|^2`, `That = Nhat dt^(b)`.
     Implement it twice: `impl="paper"` with complex `ifft`/`fft` exactly as written, and `impl="fast"` using
     `scipy.fft.irfft`/`rfft` on the real time series with preallocated buffers (this is the fair, optimised baseline
     the new methods will be compared with). Both must agree to 1e-12 relative.
   - also expose `half_point_waveform(coarse, b)` returning `tilde h_{c,k}` restricted to the band (needed later to
     compare interpolation schemes).
3. `mbquad/timing.py` — `median_time(fn, *args, number=..., repeat=...)` after a warm-up call; and
   `scripts/bench_hh_step.py` that, for `mbquad.testcases.toy_small()` and for the GW190814-like band setup of
   iteration 001 with `ToyModes(mchirp=6.4, ...)`, evaluates the modes once on the coarse grid and then reports
   the median time of the (h, h) STEP ONLY for: linear interpolation, IFFT-FFT (paper), IFFT-FFT (fast); plus the
   time of the toy waveform evaluation for reference. Write `data/bench_hh_step_iter002.md`.
4. `scripts/accuracy_iter002.py`: relative errors of both methods against `hh_exact` on `toy_small()` for
   (a) quadrupole only (`ToyModes` with only the (2,2) mode) and (b) all five modes; also with the PSD lines switched
   off. Print a table and write `data/accuracy_iter002.md`.
5. `tests/test_paper_methods.py`:
   - linear interpolation: with a PSD that is constant and no window transitions (`f_low`/`f_high` well inside a
     single band) the weights sum to the exact fine-grid sum of `1/S` (document the identity); on `toy_small()` the
     relative errors are `< 5e-6` for both (a) and (b) above (expected ~1.4e-6 and ~1.6e-6);
   - IFFT-FFT: `impl="paper"` and `impl="fast"` agree to 1e-12; relative errors `< 1e-6` on (a) and (b) (expected
     ~2e-7); for the HOM case the IFFT-FFT error is at least 3x smaller than the linear interpolation error;
     for a band with `2 M^(b) >= N^(b)` (band 0 of `toy_small()`), the method reproduces the band's exact fine-grid
     contribution to 1e-10 relative (no cropping happens);
   - `half_point_waveform` for a single pure (2,2) mode agrees with the model evaluated directly at the half points
     to relative 1e-4 or better in the band interior (document the achieved number).

## Rules (from AGENTS.md)

- Create/modify files only inside this directory; no git state changes; no package installation.
- Do not edit the iteration-001 modules except to add small helpers you need (say what you changed in the report).
- Docstrings cite the equation numbers. Keep `python -m pytest tests -q` under ~2 minutes.

## Completion criteria / verification

- `python -m pytest tests -q` passes.
- `python scripts/accuracy_iter002.py` and `python scripts/bench_hh_step.py` run and write their markdown tables.
- Final report: files, test summary, the two tables, and anything that deviates from the paper.
