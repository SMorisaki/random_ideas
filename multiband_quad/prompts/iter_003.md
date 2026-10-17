# Iteration 003 — method A: pairwise block-weight quadrature on the coarse grid (no run-time FFT)

You are working in `multiband_quad/` (the current directory). Read `AGENTS.md`, `TASK.md` (sections 2-4, 7),
`notes/method_idea.md` (sections 2-6: the derivation, the discrete identity and the toy results; **section 8: the
corrections from the design review, which take precedence where they differ from sections 2-4**) and the code of
iterations 001-002 (`mbquad/bands.py`, `psd.py`, `toywave.py`, `reference.py`, `testcases.py`, `coarse.py`,
`linear.py`, `ifftfft.py`, `timing.py`). `notes/orchestrator_toy.py` has a validated quick implementation of the
block weights (`block_weight`) and of the pairwise sum (`hh_pairwise`); use it to check conventions.

## Iteration-001/002 API you must build on (do not change its behaviour)

- `BandSetup` (`mbquad/bands.py`): `T, dt, N, f_low, f_high, B, T_b, N_b, dt_b, M_b, K_s, K_e, K_b, f_b, Df_b, K_MB, K_orig,
  mchirp_ref, m_max, L`; `tau(f)`, `dtau_df(f)`, `window(f, b)`, `coarse_frequencies()` -> list of `CoarseGrid(f, kappa, w)`,
  `all_coarse_frequencies()` -> `AllCoarseGrid(f, kappa, w, band_index)`, `lag_max_pair(b, m_slow, m_fast)`.
- `PSDGrid`: `f_k` (k = 1 .. N/2 - 1), `S_k`, `inv_S`. `mbquad.toywave`: `tau_0pn`, `dtau_0pn_df`, `ToyModes.mode/total/tau_lm`,
  `ToyModes.modes` dict. `mbquad.testcases.toy_small()` -> `ToyCase(model, psdgrid, bands)`.
- `mbquad.coarse`: `check_setup(bands, psdgrid)`; `CoarseWaveform(bands, h_b)` (tuple of per-band complex arrays of the
  unwindowed TOTAL waveform, indices `K_s..K_e`), `CoarseWaveform.from_array(bands, h)`, `.from_model(bands, model)`,
  `.h_b`, `.check_bands(bands)`.
- `mbquad.linear.LinearInterpolationHH(bands, psdgrid)`: weights `c_b` (tuple per band), `__call__(coarse)`.
- `mbquad.ifftfft.IFFTFFTHH(bands, psdgrid, impl="fast"|"paper")`: `Nhat_b`, `That_b`, `I_tilde_b`, `__call__(coarse)`,
  `band_contributions(coarse)`, `half_point_frequencies(b)`, `half_point_waveform(coarse, b)` (returns `sqrt(w) h` on the
  grid `k/That_b` restricted to the band span; the IFFT-FFT precompute zero-extends `inv_S` when `N_b > N`).
- `mbquad.timing.median_time(fn, *args, number, repeat)`; `mbquad.reference.hh_exact`, `hh_exact_banded`, `lnlambda_error`.
- Iteration-002 numbers on `toy_small()` (relative `(h,h)` errors): linear 1.40e-6 / 1.57e-6 (quadrupole / five modes),
  IFFT-FFT 1.54e-7 / 2.22e-7; `(h,h)` step: linear 0.019 ms, IFFT-FFT fast 0.70 ms, paper 1.41 ms; toy waveform 0.87 ms.

## The method (authoritative summary; equation numbers refer to `notes/paper_summary.md`)

For band `b` (drop the index) let `a_k = 1/S_k` on the fine grid (`k = 1 .. floor((N_b-1)/2)`, zero outside
`[f_low, f_high]`), `A_m = sum_k a_k e^{2 pi i k m/N_b}` (`m = 0 .. N_b-1`, one inverse FFT of length `N_b` of the
one-sided array, times `N_b`). For a block `B = {m_0, ..., m_0 + Nhat - 1}` (indices modulo `N_b`, `Nhat | N_b`):

    W_kappa(B) = (1/Nhat) sum_{m in B} A_m^* e^{2 pi i m kappa/Nhat},   kappa = 0 .. Nhat-1,

and for any complex `g_k` on the fine grid whose inverse transform `G_m = sum_k g_k e^{2 pi i k m/N_b}` vanishes
outside `B`: `sum_k a_k g_k = sum_kappa W_kappa(B) g(kappa/That)`, `That = Nhat dt_b` (`Nhat = M_b` gives the coarse grid
`kappa/T_b`). Checks: `Nhat = N_b` (any `m_0`) gives `W_kappa = a_kappa`; `W(-B) = conj(W(B))`.

Pair terms. With `h = sum_A h_A` (non-precessing modes `A = (l, m)`, `m > 0`; the caller has already combined
`+m` and `-m` and the detector response, see `mbquad/coarse.py`), `g_{AA'} = w^(b) h_A^* h_{A'}` on the coarse grid
(the window multiplies the product; `a_k = 1/S_k` carries no window). With `h_A ~ e^{-i Psi_A}` and
`dPsi_A/df = 2 pi (t_c - tau_A(f))`, the stationary lag of `G_m` is `m dt_b = Lambda_{AA'}(f) = tau_A(f) - tau_{A'}(f)`,
POSITIVE for the slower mode `A` first (larger `m`, `tau_A = tau_0pn(2f/m_A)`). Because `tau` depends only on `m`,
pairs fall into **lag classes**: class 0 = diagonal terms and same-`m` pairs (lag 0), and one class per unordered
`(m_A, m_{A'})` with `m_A > m_{A'}`. For a class the lag interval inside band `b` is
`[Lambda_min, Lambda_max] = [Lambda(f_b[b+1]), Lambda(f_lo)]` with `f_lo = f_b[b] - Df_b[b]` (`f_low` for b = 0);
add `BandSetup.lag_pair(f, m_slow, m_fast)` and `lag_interval_pair(b, m_slow, m_fast, mchirp=None)` (default
`mchirp_ref`). The margins are set by the PAIR's chirp rate `sigma_{AA'}(f) = sqrt(|dtau_A/df - dtau_{A'}/df|)` at the
band edge in question: `delta_lo = L_m sigma_{AA'}(f_b[b+1])` at the `Lambda_min` end and `delta_hi = L_m sigma_{AA'}(f_lo)`
at the `Lambda_max` end (`L_m` a margin parameter, default to be chosen from the study below, expected ~ L = 5);
the conservative alternative uses `sigma_44 = sqrt(-dtau_{m_max}/df)` at both ends. **Prior coverage**: the anchors
depend on the chirp mass, so `PairwiseHH` takes `mchirp_range = (Mc_min, Mc_max)` (default `(mchirp_ref, mchirp_ref)`)
and uses `Lambda_min` at `Mc_max` and `Lambda_max` at `Mc_min`. Rule to implement: **centre a block of length `M_b` on
`[Lambda_min - delta_lo, Lambda_max + delta_hi]`**, i.e. `m_0 = round((Lambda_min - delta_lo + Lambda_max + delta_hi)/(2 dt_b)) - M_b/2`;
class 0 uses the centred block `m_0 = -M_b/2`. Record per band and class the slack
`T_b - (Lambda_max - Lambda_min) - delta_lo - delta_hi`; if negative, warn (do not fail) and clip the margins
symmetrically. Then

    (h, h) ~= (4/T) sum_b sum_kappa  conj(H_kappa)^T  W^(b)_kappa  H_kappa,

`H_kappa` the vector of mode values at `f_kappa`, `W^(b)_kappa` the Hermitian `n_modes x n_modes` matrix with entries
`w^(b)(f_kappa) W_kappa(B_class(A, A'))` for `A` slower than `A'`, the conjugate below the diagonal and the class-0 weight
on the diagonal / same-`m` entries. Equivalent explicit form:
`sum_A W^0 |h_A|^2 + 2 Re sum_{A slower than A'} W(B_{AA'}) h_A^* h_{A'}` (same-`m` pairs with `W^0`).
Take the real part of the centred-block weight (the block `{-M/2, ..., M/2-1}` is not exactly symmetric).

**Exact error mechanism (use it in the audit tool):** for a block quadrature with `Nhat`,
`err = (1/N_b) sum_{m not in B} G_m (conj(A_{m~}) - conj(A_m))`, where `m~` is the element of `B` congruent to `m` modulo
`Nhat`: content of `G` outside `B` is folded with period `That` and weighted by the PSD kernel at the aliased lag. The
dominant harm is `G` content one period away (lag ~ +-That) landing on `A_0`.

## Deliverables

1. `mbquad/blockweights.py`
   - `class BlockWeights(bands, psdgrid)`: computes and caches `A_m` per distinct `N_b` (note several bands can share
     `N_b`; `N_b` may exceed `N`, in which case the fine-grid `a_k` is zero-padded) and provides
     `weights(b, m0, Nhat, kappa=None)` returning `W_kappa(B)` for all `kappa` or for the given `kappa` indices
     (coarse grid: `Nhat = M_b`; half-point grid: `Nhat = 2 M_b`), computed with one FFT of length `Nhat` and a phase
     factor. Cache by `(b, m0, Nhat)`. Report memory use in a method `nbytes()`.
   - helpers: `centred(b, Nhat)`, `for_lag_interval(b, lag_lo, lag_hi, Nhat)` (centres the block on the interval and
     returns the weights plus the block start/end in seconds).
   - `lag_audit(b, g_fine, Nhat, m0)`: given a fine-grid complex sequence `g_k` (e.g. `w h_A^* h_{A'}` of the model on
     `psdgrid.f_k`, zero-padded to `N_b`), compute `G_m` (inverse FFT), the fraction of `sum |G_m|` outside the block, the
     lag centroid and the 1e-6 quantiles of `|G|^2`, and the exact folding error of the formula above (compare it with
     `sum_kappa W g(kappa/That) - sum_k a_k g_k` to 1e-10 relative in a test). Memory: `N_b` up to 2^18 in the toy.
2. `mbquad/pairwise.py`
   - `class PairwiseHH(bands, psdgrid, modes, L_margin=5.0, margin_units="sigma_pair"|"sigma_mmax"|"T_b",
     mchirp_range=None, diag="block"|"linear")`, `modes` a list of `(l, m)`; groups the pairs into lag classes, builds
     one weight vector per (band, class) with `BlockWeights` and the rule above, then the stacked weight tensor `W[b]`
     of shape `(n_modes, n_modes, K_b)` (Hermitian in the first two axes, window included); `diag="linear"` uses the
     iteration-002 linear-interpolation weights `c_k` for class 0 instead (both supported; default `"block"`).
   - `__call__(coarse_modes) -> float`: `coarse_modes` is a mapping `(l, m) -> CoarseWaveform` (or a stacked array per
     band, define it in `mbquad/coarse.py` as `CoarseModes`); implementation with `np.einsum("ik,ijk,jk->", conj(H), W, H).real`
     per band, plus a second implementation `impl="pairs"` that loops over the distinct blocks (pairs sharing a block
     are summed first) — both must agree to 1e-12; keep the faster one as default and say which.
   - `summary()` returning per band and class: `Lambda_min, Lambda_max, delta_lo, delta_hi, block start/end in
     seconds, slack` (and a Markdown formatter).
3. `mbquad/coarse.py`: add `class CoarseModes` holding, per band, a complex array of shape `(n_modes, K_b)` for the
   unwindowed mode samples, with `modes` (list of `(l, m)`), `from_model(bands, model, modes)` (evaluates `model.mode(f, lm)`
   once per mode on `bands.all_coarse_frequencies().f`), `total()` -> `CoarseWaveform`, and `check_bands`.
4. `scripts/margin_study.py`: on `toy_small()` (with lines) and on the same system without lines, for the five-mode
   model: (a) scan `L_margin` in `{0.5, 1, 2, 3, 5, 8}` for `margin_units="sigma_pair"` and `"sigma_mmax"`, and fractions
   of `T_b` `{1/64, 1/32, 1/16, 1/8}` for `"T_b"`; report the relative error of the pairwise method against `hh_exact`,
   next to the IFFT-FFT (fast) and linear-interpolation errors, and two deliberately wrong placements: "all blocks
   centred on lag 0" and "blocks mirrored to negative lags" (which pins the sign convention); (b) the lag audit for the
   (4,4)x(2,1), (4,4)x(2,2), (3,3)x(2,2) and (2,2)x(2,1) classes in bands 1, 3 and 5: out-of-block fraction, centroid vs
   predicted `Lambda`, folding error, for the class block and for the 0-anchored block. Save
   `data/margin_study_iter003.json` (+ `.md` table) and figures `figures/iter003_margin_study.png`,
   `figures/iter003_lag_audit.png` (matplotlib `Agg` backend; set `MPLCONFIGDIR=/tmp/mplconfig`). Choose the default
   `L_margin` and units from this study and justify them in the report.
5. `scripts/accuracy_iter003.py`: table on `toy_small()`: quadrupole only / five modes, with / without lines, for
   linear, IFFT-FFT (fast), pairwise (default margins, `diag="block"` and `diag="linear"`): relative error of `(h,h)` and
   `|Delta ln Lambda|` at SNR 25 (`lnlambda_error`). Save `data/accuracy_iter003.md`.
6. Extend `scripts/bench_hh_step.py`: add the pairwise step (einsum and pairs implementations) to the timing table
   (`data/bench_hh_step_iter003.md`), modes evaluated once beforehand.
7. `tests/test_blockweights.py`, `tests/test_pairwise.py`:
   - `Nhat = N_b` gives `W = a` at the fine indices (band 0 of `toy_small()` where `M_b = N_b`, and a small band with
     `N_b > N`);
   - `W(-B) = conj(W(B))`;
   - the symmetric block `Nhat = 2 M_b`, `m_0 = -M_b`, applied to the total `w |h|^2` on the half-point grid obtained from
     `IFFTFFTHH.half_point_waveform`, reproduces the IFFT-FFT band value to 1e-10 relative (this ties the two formalisms);
   - lag-sign test: one band, a pure (2,2) mode `h_A` and `h_{A'} = h_A e^{-2 pi i f Delta}` with `Delta = 0.3 T_b`
     (a time shift by `+Delta`, so the pair `(A, A')` has lag exactly `Delta`... check the sign numerically against the
     fine-grid sum and document it): the block centred on `+Delta` reproduces the fine-grid pair sum to 1e-8 relative,
     the mirrored block fails;
   - pairwise with default margins on `toy_small()`: relative error `< 1e-6` and at most 2x the IFFT-FFT (fast) error for
     the five-mode case; for the quadrupole-only case it equals the class-0 (centred) result; the "all centred on lag 0"
     placement is at least 5x worse than the default with the lined PSD (expected ~4e-6 vs ~3e-7);
   - `lag_audit` folding error equals the direct error to 1e-10 relative on one pair and band;
   - einsum and pairs implementations agree to 1e-12; the result is real and does not depend on the ordering of `modes`.

## Rules (from AGENTS.md)

- Create/modify files only inside this directory; no git state changes; no package installation. Do not change the
  behaviour of iteration 001-002 code (adding methods is fine; say what you added).
- Docstrings cite the equations / the sections of `notes/method_idea.md`. Keep `python -m pytest tests -q` under ~2 minutes;
  mark anything slower with `@pytest.mark.slow`.

## Completion criteria / verification

- `python -m pytest tests -q` passes; the three scripts run and write their outputs.
- Final report: files, test summary, the accuracy and timing tables, the chosen default margins with the margin-study
  numbers, the per-band/class slack table for `toy_small()` and for the BNS (mchirp_range (1.15, 1.25)), GW190814-like
  (mchirp_range (6.0, 7.0)) and BBH 60+20 (`T = 16 s`, `{16, 8, 4}` s, `mchirp_ref = 25`, mchirp_range (25, 35)) band
  setups (`PairwiseHH.summary()`), and anything the orchestrator should know.
