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

Fix a band `b`; drop the superscript. Let `a_k = 1/S_k` on the fine grid (zero outside
`[f_low, f_high]`, Eq. 6; the window is NOT here) and `g_k = w^(b)(f_k) h_A^*(f_k) h_{A'}(f_k)` (the smooth
window multiplies the mode product, which is what makes `G` compact in lag; see Sec. 8). With the `N`-point DFTs
`A_m = sum_k a_k e^{2 pi i k m/N}`, `G_m = sum_k g_k e^{2 pi i k m/N}` (one-sided, complex),
Parseval gives `sum_k a_k g_k = (1/N) sum_m A_m^* G_m`.

If `G_m` vanishes outside a block `B = {m_0, ..., m_0 + Nhat - 1}` (indices mod `N`) with
`Nhat | N`, `r = N/Nhat`, then using the `Nhat`-point DFT of the cropped sequences and
`sum_{m in B} G_m e^{-2 pi i m kappa/Nhat} = N g_{r kappa}` (decimation in frequency =
periodization in time, harmless when the support fits in one period):

    sum_k a_k g_k  =  sum_kappa W_kappa(B) g(kappa/That),      That = Nhat dt,

    W_kappa(B) = (1/Nhat) sum_{m in B} A_m^* e^{2 pi i m kappa/Nhat} .

`W(B)` depends only on the PSD, `N^(b)` and the block, so it is precomputed once per
(band, lag class); the window enters as `w^(b)(f_kappa)` on the coarse grid, exactly as in Eq. 24. Sanity check: `B = ` all of `[0, N)` gives `W_kappa = a_kappa` (the fine
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

## 8. Corrections and refinements from the design review (2026-10-17)

Three independent adversarial reviewers (signal processing, waveform physics, implementation) re-derived
Sec. 3 and attacked Secs. 2 and 4 numerically with their own toys. Verdicts: "sound with caveats". What changes:

1. **Conventions.** With `G_m = sum_k g_k e^{+2 pi i k m/N}`, `g = w h_A^* h_{A'}` and `h ~ e^{-i Psi}`, the
   stationary lag is `m dt = tau_A(f) - tau_{A'}(f) > 0` for `m_A > m_{A'}` (the higher-`m` mode reaches `f`
   earlier). Blocks for ordered pairs with the slower mode first sit at POSITIVE lags; `B_{A'A} = -B_{AA'}` and
   `W(-B) = conj W(B)` keep the mode matrix Hermitian. (An earlier draft of Sec. 2 had the opposite sign.)
2. **Window placement.** The identity needs `G` compact; this holds only if the smooth window multiplies the
   pair product (`g = w h_A^* h_{A'}`, `a = 1/S`). Putting `w` into `a` (as an earlier draft of Sec. 3 did) gives
   rectangular in-band truncation of `g` (1/lag tails) or the lag range of the whole signal: errors of order 1.
   With `w` on the product the edge tails fall like `lag^-3` (Eq. A16), ~5x smaller in amplitude at equal `L`
   than the IFFT-FFT's `sqrt(w)` tails (Eq. A20).
3. **Lag range.** The lower end is not ~0 but `Lambda_min = Lambda(f^(b+1)) ~ (f^(b+1)/f^(b))^{-8/3} Lambda_max
   ~ 0.45-0.5 Lambda_max` for dyadic bands (only ~0 in the top band). BNS band 1 (T = 128 s): (4,4)x(2,1)
   [52.8, 112.7] s, (4,4)x(2,2) [45.6, 97.4] s, (3,3)x(2,2) [16.6, 35.5] s, (2,2)x(2,1) [7.2, 15.3] s. A block
   anchored at lag 0 overflows for the (4,1) class in BNS bands 1-2 (by 2.6 and 1.0 s); a block centred on the
   class's own interval has 20-50 s of slack. Since `tau_lm` depends only on `m`, there are 7 lag classes per
   band for `m in {1,2,3,4}`: class 0 (diagonal and same-`m` pairs such as (3,2)x(2,2), lag 0, centred block) and
   the six `(m_A, m_{A'})` classes with `Lambda = c tau_44(f)`, `c = 0.133, 0.440, 0.307, 0.975, 0.843, 0.535`
   for (2,1), (3,1), (3,2), (4,1), (4,2), (4,3). One complex weight vector per (band, class).
4. **Margins.** The edge tails of `G_{AA'}` are governed by the pair's own chirp rate
   `sigma_{AA'}^2 = |tau'_A - tau'_{A'}|` at the band edge in question, not by the individual modes:
   margins `L sigma_{AA'}(f^(b+1))` at the `Lambda_min` end and `L sigma_{AA'}(f_lo)` at the `Lambda_max` end
   (conservative: `L sigma_44` at both ends, since `sigma_{AA'} <= sigma_44`). The earlier Sec. 4.1 rule
   (one margin per mode, `sigma_21` at the near end) would cut the tail at ~0.8 sigma and leak ~8%.
5. **Exact error mechanism.** For a block quadrature of period `That`,
   `err = (1/N) sum_{m not in B} G_m (A^*_{m~} - A^*_m)`, `m~` the element of `B` congruent to `m` mod `Nhat`:
   content of `G` outside `B` is folded with period `That` and weighted by the PSD kernel at the aliased lag.
   What hurts most is `G` content one period away (lag ~ +-That), which lands on `A_0`. The diagnostics must
   compute `G_m` per pair on the fine grid and this decomposition, not just "block coverage".
6. **Prior dependence.** The anchors `Lambda(f)` scale with the chirp mass (and PN/spin corrections of
   `tau_lm`). The block must cover the prior: lower edge from `Lambda_min` at `Mc_max` (and the most negative
   `chi_eff`), upper edge from `Lambda_max` at `Mc_min`. The slack then shrinks with `rho = Mc_max/Mc_min`;
   for BNS bands 1-2 the (4,1) class fails for `rho >~ 6` (fine for LVK BNS priors, `rho ~ 1.1`). Use a
   >= 1.5PN (better 3.5PN TaylorF2) `tau` at the prior corners or an explicit extra margin; validate the slack
   over a grid of `(Mc, q, chi_eff)` with the model's own `dPsi_lm/df`. Encode a setup-time slack assertion per
   band and class (as Eq. 51 does for `(d, h)`).
7. **Linear interpolation in this language** = folding with period `T^(b)` + Fejer taper in lag (triangle
   kernel, first zero at lag `T^(b)`); IFFT-FFT = folding with period `2 T^(b)` + rectangular crop; the proposal =
   folding with period `T^(b)` + pair-specific rectangular crop. This explains when linear interpolation fails
   for HOM (pair lag within the PSD coherence time of `j T^(b)`, or attenuated by the Fejer taper at
   `0.3-0.5 T^(b)` while `A(lag)` is not negligible).
8. **Cheapest total-waveform variant (B0), to be tested first.** A centred block of length `M` applied to the
   TOTAL `w |h|^2` on the `(d, h)` coarse grid (no modes, no FFT, one real weight vector per band, `O(K_MB)`)
   gave in a reviewer's toy with strong HOM and 8 narrow lines: band errors 1.5e-6 (64 s band), 1.7e-7 (8 s),
   1.3e-7 (4 s), i.e. `Delta ln Lambda <~ 5e-4` at SNR 25 (linear interpolation: up to 3.4e-5 = 1e-2 at SNR 25;
   IFFT-FFT 2M: 3e-9 to 1e-8; pairwise: 6e-8 to 1.3e-7), because the PSD kernel `A_m` at lags >= 2 s is
   <= 2e-3 of `A_0` even with lines. Hypothesis to test on real waveforms/PSDs; if confirmed it is the natural
   default for short bands (where `Lambda_max + margin < T^(b)/2` for every pair) with B1 (baseband IFFT-FFT)
   or pair-specific blocks only where needed.
9. **Speed-up ceiling (important caveat for the owner).** Table I of the paper bounds the non-waveform
   overhead of the HOM multiband likelihood: `K_orig/K_MB` vs speed-up = 27 vs 21, 57 vs 46, 160 vs 120, i.e.
   the IFFT-FFT plus all bookkeeping is ~22-25% of the likelihood time, so removing the `(h, h)` cost entirely
   gains at most ~1.3x (BNS, 256 s) and ~1.1x (16 s signals). Moreover method A needs every mode on the coarse
   grid at every call: with lalsimulation's `SimIMRPhenomXHMFrequencySequenceOneMode` the six per-mode calls
   repeat the setup (orchestrator's timing at 1e4 points: 1.5-8 ms per mode, ~20 ms in total vs 12 ms for one
   polarization call), which can cost more than the FFTs it removes. IMRPhenomHM's `GethlmModes` returns all
   modes in one call. The timing breakdown (waveform total vs per-mode vs `(h, h)` step for each method) is
   therefore an early deliverable (iteration 005) and a go/no-go point for method A as a *practical* speed-up;
   methods B0/B1 need only the total waveform.
10. **Scope/tests.** The Fig. 4 HOM comparison in the paper used the precessing IMRPhenomPv3HM; a like-for-like
    comparison here re-runs the IFFT-FFT baseline with IMRPhenomXHM. Strong-HOM edge-on `q ~ 0.1` injections at
    SNR 25-50 must be in the population (the extreme pair carries up to 1% of `(h, h)` there, so no pair may be
    dropped: the error budget `Delta ln Lambda ~ 1e-4` means relative `(h, h)` error 3e-7). The BBH test system is
    60+20 Msun with `T = 16 s` (with `T = 8 s` the (4,4) signal from 20 Hz does not fit the segment).
    `T^(b) = T/2^j` is required for `M | N^(b)` (enforced by `BandSetup`).

11. **Per-mode waveform cost (implementation reviewer's measurements, lalsimulation 6.2.0, warm calls).**
    IMRPhenomXHM: 5x `SimIMRPhenomXHMFrequencySequenceOneMode` at 19000 points 11.8 ms vs
    `SimInspiralChooseFDWaveformSequence` 9.5 ms (BNS grid); 12.8 vs 16.5 ms at 16064 points (GW190814 grid).
    IMRPhenomHM: `SimIMRPhenomHMGethlmModes` returns all six modes in one call at the cost of the polarization
    call (45 vs 49 ms). So mode-by-mode evaluation does not penalise method A (the orchestrator's earlier 20 ms
    figure came from single cold calls). Faithful IFFT-FFT with scipy.fft and preallocated buffers: 3.0 ms (BNS,
    K_MB = 20169; 2.6 ms in the FFTs) and 0.55 ms (GW190814), so the honest end-to-end gain from replacing it is
    ~1.25x (BNS, XHM), ~1.03x (GW190814, XHM), < 1.07x (HM). An m-grouped numba kernel for the quadratic form costs
    0.07 / 0.02 ms (numpy m-grouped 0.84 / 0.19 ms; naive 6-mode einsum 1.56 / 0.49 ms).
12. **LAL conventions (verified by the reviewer, to be re-verified by test).** XHM `OneMode(freqs, m1, m2, chi1, chi2,
    ell, emm, distance, phiRef, fRef, params)`: the `emm < 0` output is the one that combines linearly with `h_+, h_x`
    at `f > 0` (`h_{l,-m}(f) = (-1)^l conj h_{l,m}(f)` at the sampled points); the polarization coefficients are the
    AddMode factors for `(l, -m)` at azimuth `pi/2`: `c_+ = 0.5 (Y_{l,-m}(iota, pi/2) + (-1)^l conj Y_{l,m}(iota, pi/2))`,
    `c_x = 0.5 i (Y_{l,-m} - (-1)^l conj Y_{l,m})`; `phiRef` enters each mode as `e^{-i m phiRef}`; XHM has no (4,3).
    HM `GethlmModes` (deltaF = 0 for a sequence): positive `m` at positive frequencies, AddMode `(l, +m)` at azimuth 0,
    NOT scaled by `(G M/c^3)^2 c/D_L`. For a regular-grid fine reference use `ChooseFDWaveformSequence` or set
    `SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(params, 0)` (the sequence path does not use XHM's
    internal multibanding; with threshold 0 both agree exactly, otherwise they differ by 3.8e-6).
13. **Factorisation over extrinsic parameters (new capability of method A).** The per-detector matrix
    `M_{AA'} = sum_kappa conj(h^raw_A) W_kappa(class(A, A')) w(f_kappa) h^raw_{A'}` built from the raw modes is
    independent of `t_c` (the phase `e^{-2 pi i f t_c}` and the detector delay cancel in every pair product), of the
    distance, and, with `e^{-i m phi_c}` and the harmonics `Y_{lm}`, `F_+`, `F_x` pulled into coefficients `C_A`, of
    `phi_c`, inclination, polarisation and sky position: `(h, h) = C^dagger M C`. Modes are evaluated once per intrinsic
    point and `(h, h)` for any extrinsic parameters costs `O(n_modes^2)`. The IFFT-FFT has no such factorisation.
    Also, the pairwise result is `t_c`-independent to roundoff while the IFFT-FFT error varies with `t_c` over the prior.
14. **Block placement, two prescriptions.** "Interval-centred with prior corners" (implemented in iteration 003:
    `[Lambda_min(Mc_max) - delta_lo, Lambda_max(Mc_min) + delta_hi]`) and the implementation reviewer's "contain-0"
    block `[-delta_b, M_b - delta_b)` with `delta_b = L sigma_pair(f^(b+1)) + t_floor` coincide when the prior is wide
    (`Lambda_min(Mc_max) -> 0`); the interval-centred form exploits the slack of narrow priors. Both need a floor on the
    low-lag margin in the top band (merger-ringdown content at lag 0 +- tens of ms; `t_floor ~ 0.05-0.1 s`), where
    `sigma_pair(f^(B))` is tiny. Expected tail scaling: error ~ L^-3 (pairwise, window on the product) vs L^-2
    (IFFT-FFT, sqrt(w) on the waveform); a flat error-vs-L curve indicates a placement bug or PSD-line-dominated error.
15. **Single-mode secondary result.** The centred `M`-block weights are a drop-in replacement for the linear
    interpolation weights of Eq. 29 at identical run-time cost: 2.2e-7 vs 1.4e-6 relative error in the toy for a
    single mode (IMRPhenomD-style analyses, the paper's main use case).

## 9. B1 accuracy floor (orchestrator, after iteration 005; `notes/orchestrator_toy_B1_variants.py`)

On IMRPhenomXHM B1 is 1.7x (GW190814-like) to 10x (BNS) less accurate than the paper's IFFT-FFT, although both compute
half-point values under the same time-limitation assumption. Tested explanations on the toy (paper IFFT-FFT 1.5e-7 /
2.2e-7 for quadrupole / five modes; B1 band-only 4.4e-7 / 4.4e-7):
1. Riemann-sum endpoint error (K' vs M samples per period): the Euler-Maclaurin endpoint correction and zero-padding
   `p = 2..8` leave B1 at 3.9-4.1e-7 / 5.0-5.5e-7 -> not the cause (the integrand is band-limited near the sampling
   Nyquist, and in any case both sums are exact for a time-limited signal).
2. Out-of-band leakage of the cropped signal (the paper sums `I_tilde_c |h_tilde_c|^2` over ALL positive bins, B1 only over
   the band): summing all `K''` half bins of the baseband period gives +1.2e-7, +4.5e-7, -1.8e-7, -3.3e-7, -2.8e-7 for
   `p = 1, 2, 4, 8, 16` (quadrupole), i.e. fluctuations of either sign at the same level, no convergence to the paper's value.
Conclusion: B1 and the IFFT-FFT are two different discrete interpolants of the same approximation; their errors are both
set by the tails that violate Eq. 36 and differ by O(1) factors that depend on the system. B1 keeps its 5-9x cheaper
`(h,h)` step; its accuracy is the same order as the IFFT-FFT (within 2-3x on the toys, within 1.7x on the GW190814-like XHM
case with the real PSD, 10x on the very accurate BNS case where both are below 4e-7). The remaining lever common to both
methods is `L` (the band margin), to be scanned in iteration 008.
