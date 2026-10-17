# Iteration 001 — package skeleton: PSD, synthetic multi-mode waveform, bands, windows, coarse grids, brute-force (h, h)

You are working in `multiband_quad/` (the current directory). First read `AGENTS.md`, `TASK.md`
(sections 1, 2, 5, 7) and `notes/paper_summary.md` (the paper's equations in plain text; equation
numbers below refer to it). `notes/method_idea.md` explains the new methods (not needed yet).
`notes/orchestrator_toy.py` is a working but quick-and-dirty reference for everything in this
iteration (bands, windows, brute force, the paper's two methods): you may read it, but write clean,
documented, vectorized code with a proper API; do not copy its structure.

## Deliverables

1. `mbquad/__init__.py` and `mbquad/psd.py`
   - `aligo_analytic_psd(f)`: the analytic aLIGO fit
     `S(f) = 1e-49 [x^-4.14 - 5 x^-2 + 111 (1 - x^2 + x^4/2)/(1 + x^2/2)]`, `x = f/215 Hz`
     (Ajith 2011 form; document it).
   - `add_lines(psd_fn, lines)`: returns a new PSD function multiplied by
     `prod (1 + h_i exp(-(f - f_i)^2 / (2 sigma_i^2)))` for a list of `(f_i, sigma_i, h_i)`.
   - `load_asd_or_psd(path)`: two-column text file (frequency, value); decide ASD vs PSD by the
     typical magnitude (ASD ~ 1e-23, PSD ~ 1e-46) and return a PSD interpolator (log-log linear
     interpolation; outside the file range return `+inf`).
   - `class PSDGrid`: built from a PSD function and `(T, dt, f_low, f_high)`; stores the fine grid
     `f_k = k/T` for `k = 1 .. floor((N-1)/2)` (`N = T/dt` must be an integer power of two), `S_k`
     with `S_k = +inf` outside `[f_low, f_high]` (Eq. 6), and `inv_S` (zero outside). Keep memory
     reasonable for `N = 2^20`.
2. `mbquad/toywave.py` — synthetic non-precessing multi-mode waveform in the stationary-phase form
   (Appendix A facts in `notes/paper_summary.md`), usable mode by mode on ANY frequency array:
   - `tau_0pn(f, mchirp)` (Eq. 53), its derivative `dtau_0pn_df`, and `psi_22(f, mchirp, t_c, phi_c)`
     = `2 pi f t_c - phi_c - pi/4 + (3/128) (pi G Mc f / c^3)^(-5/3)` (Newtonian TaylorF2 phase; `t_c`
     measured from the start of the data segment, so the waveform's inverse FT peaks at `t = t_c`).
   - `class ToyModes`: parameters `mchirp` (Msun), `t_c` (s), `phi_c`, `dist_scale` (overall amplitude),
     and a dict `{(l, m): (ratio, phase_offset, growth)}` for the modes `(2,2), (2,1), (3,3), (3,2), (4,4)`
     with defaults `ratio = (1, 0.25, 0.35, 0.10, 0.15)`, `growth = 0` (optional power of `f/100 Hz`
     multiplying the ratio, to mimic PN amplitude growth). Mode `(l, m)`:
     `h_lm(f) = dist_scale * ratio * (f/100)^growth * f^(-7/6) * exp(-i [ (m/2) psi_22(2f/m) + phase_offset ])`.
     Methods: `mode(f, lm) -> complex array`, `total(f) -> complex array` (sum of modes),
     `tau_lm(f, lm) = tau_0pn(2 f / m)`. Optional smooth high-frequency taper per mode
     (`f_taper_lm`, Planck or cosine taper) switched off by default. Document that `t_lm(f) = t_c - tau_lm(f)`.
3. `mbquad/bands.py` — the multibanding geometry of Sec. II.B, II.C, II.E, II.F:
   - `class BandSetup` built from `T, dt, f_low, f_high, durations (list of T^(b), decreasing, each T/2^j),
     mchirp_ref (minimum chirp mass of the prior), m_max, L, tc_min_minus_T (default -2.12 s),
     T_minus_tc_max (default 1.9 s)`. It computes, following the paper exactly:
     `tau(f) = tau_0pn(2 f / m_max, mchirp_ref)` (Eq. 55); `f^(0) = f_low`; for `b >= 1` the root of
     `tau(f) + L sqrt(-tau'(f)) = T^(b) + tc_min_minus_T` (Eq. 51, use `scipy.optimize.brentq`);
     `Df^(b) = 1/sqrt(-tau'(f^(b)))` (Eq. 47) with `Df^(0) = 0`; `Df^(B) = 100 / T_minus_tc_max`,
     `f^(B) = f_high + Df^(B)` (Eq. 56, Eq. 10); check `f^(b) < f^(b+1) - Df^(b+1)` and raise with a clear
     message otherwise; `N^(b)` = smallest power of two with `floor((N^(b)-1)/2) >= f^(b+1) T` (Eq. 13);
     `dt^(b) = T/N^(b)`; `M^(b) = T^(b)/dt^(b)` (must be an integer); `K_s^(b) = ceil((f^(b) - Df^(b)) T^(b))`,
     `K_e^(b) = floor(f^(b+1) T^(b))`; coarse grids `f^(b)_kappa = kappa/T^(b)`; `K_MB` (Eq. 25) and
     `K_orig` (Eq. 8).
   - `window(f, b)` implementing Eq. 9 for arbitrary float arrays, and `sqrt_window`.
   - `coarse_frequencies()` returning, per band, the coarse frequencies, `kappa` indices and window values;
     and `all_coarse_frequencies()` (concatenation, with band index), which is what a waveform model will
     be evaluated on.
   - `summary_table()` returning a list of dicts (per band: `T_b, f_lo, f_hi, Df_lo, Df_hi, N_b, M_b, K_s, K_e,
     K_b`) and a pretty-printer.
   - For later use also provide `lag_max_pair(b, m_slow, m_fast)` =
     `tau_0pn(2 f_lo/m_slow) - tau_0pn(2 f_lo/m_fast)` at the band start `f_lo = f^(b) - Df^(b)`
     (for `b = 0`: `f_low`), i.e. the predicted largest lag between two modes inside the band.
4. `mbquad/reference.py` — brute force:
   - `hh_exact(h_fine, psdgrid)`: Eq. 4 for a waveform array on the fine grid.
   - `hh_exact_banded(model_total, psdgrid, bands)`: Eq. 26 (sum over bands of the windowed fine-grid
     sums) evaluated with the model on the fine grid; must equal `hh_exact` to relative 1e-12 (partition
     of unity).
   - `lnlambda_error(hh_approx, hh_exact, snr)`: `snr^2 (hh_approx - hh_exact) / (2 hh_exact)` — the error
     in `ln Lambda` when the waveform amplitude is rescaled to the given single-detector SNR (document why).
5. `scripts/band_table.py`: prints the band tables and `K_orig / K_MB` for
   (a) BNS: `T = 256 s, dt = 1/4096 s, 20-2048 Hz, durations 256, 128, ..., 4 s, mchirp_ref = 1.15,
   L = 5`, for `m_max = 2` and `m_max = 4` (the paper's Table I gives `K_orig/K_MB = 45` and `27`);
   (b) GW190814-like: `T = 16 s, dt = 1/4096, 20-1024 Hz, durations 16, 8, 4 s, mchirp_ref = 6.4,
   m_max = 4, L = 5` (Sec. III.A);
   (c) BBH: `T = 8 s, dt = 1/4096, 20-1024 Hz, durations 8, 4 s, mchirp_ref = 20, m_max = 4, L = 5`;
   and, per band, `lag_max_pair(b, 4, 1)` compared with `T^(b)/2` and `T^(b)`. Save the tables to
   `data/band_tables.md`.
6. `tests/` (pytest, run from this directory: `python -m pytest tests -q`, keep it under ~1 minute):
   - windows: partition of unity on the fine grid of the BBH configuration (`sum_b w^(b)(f_k) = 1`
     within 1e-12 for `f_low <= f_k <= f_high`), zero outside the band supports, continuity at the
     transition edges;
   - bands: monotone edges, `M^(b)` integer, `K_s <= K_e`, `N^(b)` satisfies Eq. 13 and is a power of two,
     `tau(f^(b)) + L sqrt(-tau')` reproduces `T^(b) + tc_min_minus_T` within 1e-9 s;
   - `K_orig/K_MB` for the BNS configuration within +-15% of 45 (`m_max = 2`) and 27 (`m_max = 4`);
   - toywave: `tau_lm`, numerical derivative of the phase `dpsi/df / (2 pi)` equals `t_c - tau_lm(f)`
     within 1e-6 relative; `total == sum of modes`; mode evaluation on an unsorted frequency array
     matches evaluation on the sorted one;
   - reference: `hh_exact_banded == hh_exact` to 1e-12 relative on the toy configuration below;
     `lnlambda_error` sign/scale sanity.
   Toy configuration for tests: `T = 128 s, dt = 1/1024 s, 20-256 Hz, durations 128, 64, 32, 16, 8, 4 s,
   mchirp_ref = 4.85, m_max = 4, L = 5`, `ToyModes(mchirp = 4.85, t_c = T - 2.0)`, analytic PSD with lines
   `[(26.5, 0.01, 3000), (36, 0.01, 2000), (48.5, 0.01, 1000), (60, 0.02, 500), (120, 0.05, 100)]`
   (frequency Hz, sigma Hz, height). Expose this configuration as a helper (e.g. `mbquad.testcases.toy_small()`)
   because later iterations reuse it.

## Rules (from AGENTS.md)

- Create/modify files only inside this directory. Do not run git commands that change state
  (no commit/push/stash/reset). Do not install packages; `python` on PATH is the conda env with numpy,
  scipy, numba, matplotlib, h5py, pytest, lalsimulation. If something is missing, stop and report.
- Docstrings must state the formula implemented (with the paper's equation number) and its domain of
  validity. Use the paper's notation in names.
- No plots are needed in this iteration (`MPLCONFIGDIR` not needed).

## Completion criteria / verification

- `python -m pytest tests -q` passes (under ~1 minute).
- `python scripts/band_table.py` runs and writes `data/band_tables.md`; report the `K_orig/K_MB`
  values for the BNS configuration with `m_max = 2` and `4` and the lag table for the three systems.
- End your work with a short report: files created, the test summary, the `K_orig/K_MB` numbers,
  anything that differs from the paper or that the orchestrator should know (e.g. if Eq. 51 has no
  root for some band, or if Table I cannot be reproduced within 15% and why).
