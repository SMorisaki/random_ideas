# Pairwise (mode-by-mode) multiband quadrature for (h, h) with higher-order modes

Orchestrator's derivation (2026-10-17). Reference: S. Morisaki, PRD 104, 044062 (2021),
arXiv:2104.07813 ("the paper"). Equation numbers below refer to the paper.

## 1. What the paper does for (h, h)

Fine grid `f_k = k/T`, PSD `S_k`, smooth band windows `w^(b)(f)` (Eq. 9) with
`sum_b w^(b) = 1`. Exact quadratic term (Eq. 26):

    (h, h) = sum_b (4/T) sum_k w^(b)(f_k) |h(f_k)|^2 / S_k .

Band `b` has an FFT length `N^(b)` (power of 2, Eq. 13), time step `dt^(b) = T/N^(b)`, duration
`T^(b) = M^(b) dt^(b)` and coarse grid `f^(b)_kappa = kappa/T^(b)`. The windowed waveform
`sqrt(w^(b)) h` is assumed to be time-limited to the last `T^(b)` of data (Eq. 36), so only the
coarse grid is needed for `(d, h)` (Eq. 24).

Two methods for `(h, h)`:

* Linear interpolation (Eqs. 27-30): interpolate `|h|^2` linearly between coarse points;
  precomputed weights `c^(b)_kappa`; `O(K_MB)` cost; inaccurate with higher-order modes
  (HOM) because the cross terms between modes make `|h|^2` oscillatory (Fig. 2).
* IFFT-FFT (Eqs. 31-46): `|h|^2` is the autocorrelation of the time-limited band waveform,
  so its time support is `2 T^(b)`. Cropping to `Nhat^(b) = min(2 M^(b), N^(b))` samples and
  using Parseval gives Eq. 45,
  `(h, h) ~= sum_b (4/That^(b)) sum_k Itilde^(b)_{c,k} |htilde^(b)_{c,k}|^2`,
  where `htilde_c` is the band waveform on the grid `k/(2 T^(b))` obtained by IFFT of the
  coarse samples, zero padding to `2 M^(b)` and FFT (Eq. 46). Cost `O(Nhat log Nhat)` per band
  per likelihood evaluation, dominated in practice by the FFT calls and the associated array
  bookkeeping, and in the paper it is the only accurate option for HOM.

Key observation: the IFFT-FFT step is nothing but an (exact, under the time-limited assumption)
interpolation of the band waveform onto a grid twice as fine. The factor 2 comes from the
autocorrelation of the *total* waveform being supported on `[-T^(b), T^(b)]`.

## 2. Idea: treat mode pairs separately; each ordered pair is one-sided in lag

Write the waveform as a sum of modes (non-precessing, aligned spin, in the paper's setting),
`h(f) = sum_A h_A(f)`, `A = (l, m)`. With the stationary-phase form `h_A = B_A(f) e^{-i Psi_A(f)}`
(Eq. A4) and `dPsi_A/df = 2 pi t_A(f)`, `t_A(f) = t_c - tau_A(f)`, `tau_A(f) = tau_22(2f/m)`,

    |h|^2 = sum_A |h_A|^2 + 2 Re sum_{A > A'} h_A^* h_{A'} .

The inverse Fourier transform (in `f -> lag`) of a single ordered product
`g_{AA'}(f) = w^(b)(f) h_A^*(f) h_{A'}(f)` is, by stationary phase, concentrated at the lag

    lag(f) = t_A(f) - t_{A'}(f) = tau_{A'}(f) - tau_A(f)     (t_c cancels!)

so its lag support inside band `b` is the *range* of this function over the band
(plus tails), and for `m_A > m_{A'}` it is one-sided. With the discrete convention of Sec. 3
(`G_m = sum_k g_k e^{2 pi i k m/N}`, `g = h_A^* h_{A'}`, `h ~ e^{-i Psi}`) the stationary lag is
`m dt = tau_A(f) - tau_{A'}(f) >= 0` for the slower mode `A` first, i.e. `lag in [~0, (tau_A - tau_A')(f_lo)]`.
The length of that interval is at most `tau_{m_max}(f_lo) - tau_{m_min}(f_lo) < T^(b)`,
because `T^(b)` was chosen (Eq. 51 with Eq. 55) to contain `tau_{m_max}(f^(b))` *plus*
`L sqrt(-tau')` *plus* the `t_c` prior range `T - t_c,min` (2.12 s). The `t_c` part is not
needed here because lags do not depend on `t_c`.

Diagonal terms `|h_A|^2` have lag ~ 0 (support = tails only), as in the quadrupole-only case.

Therefore every pair term can be computed **on the coarse grid itself** (no upsampling, no
FFT at run time), provided the precomputed PSD weight uses, for each pair, a time block of
length `T^(b)` positioned over that pair's lag range instead of the symmetric block
`[-T^(b), T^(b)]` of the IFFT-FFT method.

## 3. Exact discrete statement (precomputed "block weights")

Fix a band `b`; drop the superscript. Let `a_k = w(f_k)/S_k` on the fine grid (zero outside the
band) and `g_k = h_A^*(f_k) h_{A'}(f_k)` (any complex sequence). With the `N`-point DFTs
`A_m = sum_k a_k e^{2 pi i k m/N}`, `G_m = sum_k g_k e^{2 pi i k m/N}` (one-sided, complex),
Parseval gives `sum_k a_k g_k = (1/N) sum_m A_m^* G_m`.

If `G_m` vanishes outside a block `B = {m_0, ..., m_0 + Nhat - 1}` (indices mod `N`) with
`Nhat | N`, `r = N/Nhat`, then using the `Nhat`-point DFT of the cropped sequences and
`sum_{m in B} G_m e^{-2 pi i m kappa/Nhat} = N g_{r kappa}` (decimation in frequency =
periodization in time, harmless when the support fits in one period):

    sum_k a_k g_k  =  sum_kappa W_kappa(B) g(kappa/That),      That = Nhat dt,

    W_kappa(B) = (1/Nhat) sum_{m in B} A_m^* e^{2 pi i m kappa/Nhat} .

`W(B)` depends only on the PSD, the window and the block, so it is precomputed once per
(band, block class). Sanity check: `B = ` all of `[0, N)` gives `W_kappa = a_kappa` (the fine
grid sum). The paper's Eq. 45 is the special case of a symmetric block of length `2 M`
applied to the total `|h|^2`.

The quadratic term becomes

    (h, h) ~= sum_b (4/T) sum_kappa  h(f_kappa)^dagger  W^(b)_kappa  h(f_kappa),

where `h(f_kappa)` is the vector of mode values on the coarse grid and `W^(b)_kappa` is a
Hermitian `n_modes x n_modes` matrix whose `(A, A')` entry is `W_kappa(B_{AA'})` (the block for
that ordered pair), i.e. a batched quadratic form, `O(n_modes^2 K_MB)` real operations and no
FFT. Pairs with the same lag class share a block, so only a few distinct weight vectors per
band are needed (e.g. "centred" for diagonal / small-lag pairs, "one-sided" for the rest).

## 4. Where the approximation can fail; what must be measured

1. Tails of the stationary-phase concentration (Appendix A: smooth window gives
   `(lag - lag_sp)^-2` or `^-3` decay). Both modes contribute tails, so the block must have a
   margin on both ends: `[-Dt_max - L' sqrt(-tau'_A), + L' sqrt(-tau'_A')]`. For the extreme
   pair ((4,4) with (2,1)) at the lowest band this margin may slightly exceed the `2.12 s +
   (L-1) sqrt(-tau')` slack left by Eq. 51. Options: (a) accept a tiny error (the product of
   two sub-dominant amplitudes is ~1% of `|h_22|^2`); (b) add a HOM-specific margin to the
   band rule (Eq. 51), costing a few per cent more waveform points; (c) use the `2 T^(b)` grid
   only for the offending pairs. This must be quantified.
2. Merger-ringdown: the modes' phases no longer follow `tau_A(f) = tau_22(2f/m)`, but the
   durations are short there (high bands), so the lags are small compared with `T^(b)`.
3. Mode mixing / precession: inertial-frame modes of precessing models contain several
   co-precessing `m'`, so the one-sided property holds for *co-precessing* mode pairs (the
   twisting coefficients are slowly varying). Scope here: non-precessing HOM; note the
   extension.
4. Requirement: the waveform model must deliver each mode (or each co-precessing mode's
   contribution) on an arbitrary frequency sequence. IMRPhenomXHM provides this
   (`SimIMRPhenomXHMFrequencySequenceOneMode`); IMRPhenomHM via `ChooseFDModes`. The IFFT-FFT
   method only needs the total `h`, which is a practical advantage it keeps.

## 5. Alternatives considered

* Evaluate the total waveform directly on the `2 T^(b)` grid (half points) with the
  IFFT-FFT weights: no FFT, no modes needed, but doubles the waveform evaluations; serves as a
  comparator.
* Filon/Levin-type oscillatory quadrature of the cross terms on the coarse grid: requires the
  lag to be small compared with `T^(b)`; not true for the extreme pairs; the block weights are
  the exact version of this idea.
* Relative binning / heterodyne for `(h, h)`: needs a fiducial waveform, outside the
  multibanding philosophy; not pursued.

## 6. Toy check by the orchestrator (2026-10-17, `notes/orchestrator_toy.py`)

Synthetic Newtonian SPA modes (2,2), (2,1), (3,3), (3,2), (4,4) with exaggerated amplitude ratios
(1, 0.25, 0.35, 0.10, 0.15), `Mc = 4.85 Msun`, `T = 128 s`, `dt = 1/1024 s`, 20-256 Hz, paper-style bands
`{128, 64, 32, 16, 8, 4} s` from Eq. 51 (`m_max = 4`, `L = 5`), analytic aLIGO-like PSD with narrow
Gaussian noise lines (`sigma = 0.01-0.05 Hz`) placed near the low-frequency edges of bands 1-3, where the
(4,4)-(2,1) lag exceeds `T^(b)/2`. `K_orig = 30209`, `K_MB = 3294`.

Relative errors of `(h, h)` against the brute-force fine-grid sum:

| method | quadrupole only | with HOM |
|---|---|---|
| linear interpolation (Eqs. 28-30) | 1.4e-6 | 1.6e-6 |
| IFFT-FFT (Eqs. 45-46), re-implemented | 1.5e-7 | 2.2e-7 |
| pairwise block weights, all blocks centred (deliberately wrong placement) | 2.2e-7 | 4.4e-6 |
| pairwise block weights, one-sided blocks, margin `M/8` | 2.2e-7 | 2.6e-7 |
| pairwise block weights, one-sided blocks, margin `M/16` | 2.2e-7 | 3.6e-7 |
| pairwise block weights, one-sided blocks, margin `M/32` | 2.2e-7 | 1.5e-6 |

Conclusions: (i) the pairwise coarse-grid method reaches the IFFT-FFT accuracy without any run-time FFT;
(ii) the ratio linear/IFFT-FFT ~ 5-7 matches Fig. 4 of the paper (1e-3 vs 2e-4 in ln Lambda);
(iii) block placement only matters where the PSD has narrow lines: the error is
`sum_{m not in B} A_m^* G_m` and `A_m` (the inverse transform of `1/S`) decays quickly for a smooth PSD,
so a smooth PSD hides a wrong block; tests must include narrow lines near band edges (real LVK PSDs have
them); (iv) the margin on the negative-lag side (fast mode's tail) matters more than the positive side.
Natural general rule to implement: centre the block on the predicted lag interval `[0, lag_max]` of the
pair (diagonal: lag 0), with `lag_max` from the band's reference chirp time, and study the residual error
versus the available slack. Timing of the toy is not meaningful (the toy re-evaluates the modes per pair).

Note: with the correct reading of Eq. 30 (tilde-f only in the summation ranges, true `f` in the weights)
the linear interpolation is accurate to ~1e-6 even here; a first implementation that used the tilde
values in the weights gave 3% errors.

## 7. Method B: total waveform only (no mode decomposition), `notes/orchestrator_toy_methodB.py`

When only `h = h_+ F_+ + h_x F_x` is available the lag support of `w |h|^2` is two-sided,
`[-lag_max, lag_max]` with `lag_max ~ T^(b)`, so values at the half points `(kappa + 1/2)/T^(b)` are
genuinely needed (the IFFT-FFT method computes them by exact band-limited interpolation). Two savings keep
the method exact under the same time-limitation assumption:

1. Baseband FFTs. The windowed band waveform vanishes outside `[f_Ks, f_Ke]`, so the `K = K_e - K_s + 1`
   coarse samples `x_j = sqrt(w) h(f_{Ks + j})` can be inverse-transformed with length `K' >= K`
   (`scipy.fft.next_fast_len(K)`) instead of `M^(b) ~ 2 f^(b+1) T^(b)`: `y = ifft(x, K')` is the band waveform
   demodulated by `f_Ks` and sampled with step `T^(b)/K'`, aliased with period `T^(b)` (harmless, the support is
   `[T - T^(b), T]`, which with `T/T^(b)` an integer maps to indices `n' = n`). The half points are
   `z = fft(y * exp(-i pi n/K'))`, `z_j = sqrt(w) h(f_{Ks + j + 1/2})`, `j < K`. Two FFTs of length `K'` per band
   instead of one of length `M` and one of length `2M`; `K'/M ~ (f^(b+1) - f^(b))/(2 f^(b+1)) ~ 1/8 .. 1/4`.
2. Adaptive block length. If the predicted two-sided lag support `lag_max + margin < T^(b)/2` (short
   high-frequency bands, where the `t_c` prior range dominates `T^(b)`), the integer grid with the centred
   block of length `M` suffices and no FFT is needed in that band; otherwise the `2M` symmetric block on the
   `2T^(b)` grid is used, with the weights `W_{2 kappa}` (integer points) and `W_{2 kappa + 1}` (half points),
   which are the paper's `Itilde_c` restricted to the band.

Toy results (same configuration as Sec. 6): relative error 4.3e-7 (paper's IFFT-FFT 2.2e-7, same order);
FFT lengths per band 0 / 594 / 432 / 350 / 360 / 0 versus the paper's (M, 2M) = (8192, 8192), (8192, 16384),
(4096, 8192), (4096, 8192), (2048, 4096), (4096, 8192); `(h, h)` step 0.26 ms versus 1.61 ms.
Further FFT-free variants for the total waveform (local Dirichlet-kernel interpolation of the half points,
non-dyadic `That` between `T^(b)` and `2 T^(b)`) give at most modest gains for the long bands and are left
as optional explorations.
