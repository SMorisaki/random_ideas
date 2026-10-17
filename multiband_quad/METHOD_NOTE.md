# Multiband quadratic terms with higher modes

This note describes two replacements for the quadratic step in S. Morisaki,
*Phys. Rev. D* **104**, 044062 (2021), arXiv:2104.07813. Equation numbers refer
to that paper (local [summary](notes/paper_summary.md) and [PDF](notes/arXiv_2104.07813v2.pdf)).
Method A uses mode-pair block weights; methods B0/B1 need only total detector
strain. All accuracy statements concern `(h,h)`, with `(d,h)` held fixed.
[REPORT.md](REPORT.md) gives the criterion verdicts and full headline tables.

## 1. Problem and the paper's two methods

For `f_k = k/T`, one-sided PSD `S_k`, and cosine windows `w_b` (Eq. 9) with
`sum_b w_b = 1` (Eq. 11), Eq. (26) is

    (h,h) = sum_b (4/T) sum_k w_b(f_k) |h(f_k)|^2 / S_k.

The fine sum excludes DC and Nyquist (Eqs. 4, 6). In band `b`,
`dt_b = T/N_b`, `M_b = T_b/dt_b`, and the coarse samples are
`f_kappa = kappa/T_b`, `K_s <= kappa <= K_e` (Eqs. 13, 18, 24–25).
Durations are `T/2^j`, possibly skipping values, so `M_b` divides `N_b`.

Linear interpolation (Eqs. 27–30) replaces total power between coarse
samples by straight lines. Its precomputed `c_k` give an inexpensive dot
product. Higher modes make `|h|^2` oscillate through interference terms;
linear power interpolation can then be inaccurate.

The IFFT-FFT method (Eqs. 31–46) instead assumes the inverse transform of
`sqrt(w_b) h` vanishes outside the last `T_b` seconds (Eq. 36). Its
circular autocorrelation (Eq. 34) occupies `2M_b-1` lags, on both sides of
zero. Thus Eq. (37) uses `Nhat_b = min(2M_b,N_b)` and
`That_b = Nhat_b dt_b`, usually `2T_b`. Eq. (46) reconstructs the band time
series by an `M_b` IFFT, puts it in the last `M_b` entries of a zero array,
and FFTs to `Nhat_b`. Eq. (45) sums its weighted power on the finer grid.
The factor two follows from total-waveform autocorrelation support; it is
not an intrinsic requirement of every ordered mode product.

Our fast baseline uses SciPy real transforms and reusable buffers, while
retaining the paper's approximation. It agrees with the literal complex
implementation; comparing to the latter alone would overstate the saving.

## 2. Ordered products and lag conventions

For non-precessing contributions `h = sum_A h_A`, with `A=(l,m)`,

    |h|^2 = sum_A |h_A|^2 + 2 Re sum_ordered h_A^* h_A'.

Use the paper's SPA convention `h_A ~ exp(-i Psi_A)` (Appendix A),
`Psi_A'/(2 pi) = t_c - tau_A`, and
`tau_A(f) ~= tau_0PN(2f/m)` (Eqs. 53, 55). With the positive-exponential
inverse DFT defined below, the stationary lag of `h_A^* h_A'` is

    Lambda_AA'(f) = tau_A(f) - tau_A'(f).

The slower, higher-`m` mode goes first, giving **positive** lags. The
common `t_c` and detector-delay phase cancels. At leading order a pair's
bulk occupies `[Lambda(f_hi), Lambda(f_lo)]`, not `[0,Lambda(f_lo)]`.
Same-`m` pairs have zero leading-order lag, including distinct `l`.
There is no exact compact support: finite-window tails and merger/mixing
corrections must be checked against the fine sum.

## 3. Block-weight identity and folding error

Fix a band; write `N=N_b`, `dt=dt_b`. Extend the positive fine-grid arrays
to a length-`N` complex DFT grid with zeros elsewhere. Crucially,

    a_k = 1/S_k,                 g_k = w_b(f_k) h_A(f_k)^* h_A'(f_k),
    A_m = sum_k a_k exp(+2 pi i k m/N),
    G_m = sum_k g_k exp(+2 pi i k m/N).

The **window is on the product**, not on `a`. Since `a` is real,
Parseval gives `sum_k a_k g_k = (1/N) sum_m A_m^* G_m`.
Choose a circular consecutive block `B={m0,...,m0+Nhat-1}` with `Nhat|N`,
`r=N/Nhat`, and `That=Nhat dt`. If `G` vanishes outside `B`, decimation
in frequency periodises `G` without overlap:

    sum_{m in B} G_m exp(-2 pi i m kappa/Nhat) = N g_{r kappa},
    W_kappa(B) = (1/Nhat) sum_{m in B} A_m^* exp(+2 pi i m kappa/Nhat),
    sum_k a_k g_k = sum_{kappa=0}^{Nhat-1} W_kappa(B) g(kappa/That).

Only in-band coarse entries are needed when `g` vanishes outside the
window. `W` depends on the PSD, grid and block; multiply it by coarse
`w_b` in the evaluator. Taking `B` to be the entire DFT grid gives
`W_k=a_k`. For the reversed ordered pair use the reflected block;
`W(-B)=conj(W(B))`. On an even-length centred block its single asymmetric
endpoint is handled by taking real diagonal weights and explicitly
conjugating off-diagonal entries, preserving the Hermitian quadratic form.

For general `G`, let `m_tilde` be the unique index in `B` congruent to `m`
modulo `Nhat`. The exact error, quadrature minus fine sum, is

    err = (1/N) sum_{m outside B} G_m (A_m_tilde^* - A_m^*).

Multiply by `4/T` and, for a cross pair, take `2 Re` to get its `(h,h)`
contribution. The omitted tail alone does not determine the error: content
near one period away folds onto the large `A_0`. Smooth PSDs can hide a bad
placement; spectral lines extend the inverse-PSD kernel and expose it.
`BlockWeights.lag_audit` computes this identity independently. Its outside
fraction uses `sum|G|`, while lag centroids/quantiles use `|G|^2`.

The paper's Eq. (45) is the centred `min(2M,N)` version after real-signal
symmetrisation. With our one-sided complex `A`, the weight relation is

    Re[W(k) + W(-k)] = (T/That) I_tilde_c(k).

It is not `W=(That/T) I_tilde_c`. Agreement with Eq. (45) requires its
entire positive interpolated spectrum, including leakage outside the band
([data/accuracy_iter004.md](data/accuracy_iter004.md)).

A useful common description is:

| Method | Lag period | Lag operation / input |
|---|---|---|
| Linear, Eqs. 27–30 | `T_b` | Fejer (triangular) taper associated with linear power interpolation; endpoint conventions of Eq. 30 retained |
| IFFT-FFT, Eqs. 31–46 | `min(2T_b,T)` | Centred crop of total autocorrelation; interpolate `sqrt(w_b) h` |
| A | `T_b` | Class-specific crop of `w_b h_A^* h_A'` |
| B0 | `T_b` | Centred crop of total `w_b |h|^2` |
| B1 | Usually `2T_b` | Same crop idea, half-point interpolation by baseband IFFT-FFT; adaptive B0 shortcut |

The Fejer description explains the lag attenuation and folding of linear
interpolation; Eq. (29), including finite-window and edge details, remains
the implemented definition.

## 4. Method A: prior blocks and extrinsic factorisation

For `m in {1,2,3,4}` there are seven classes: zero lag (diagonal and
same-`m` pairs), and six ordered `(m_slow,m_fast)` classes. Form
`H_m = sum_{A:m_A=m} h_A` to contract each distinct class once. This
retains every mode pair, including weak cross terms. General cost is
`O(n_modes^2 K_MB)` with no evaluation-time FFT.

For a prior `Mc_min <= Mc <= Mc_max`, centre a block of duration `T_b` on

    [Lambda_min(Mc_max) - L_margin sigma_pair(f_hi,Mc_max),
     Lambda_max(Mc_min) + L_margin sigma_pair(f_lo,Mc_min)],
    sigma_pair(f,Mc)^2 = |tau_slow'(f,Mc) - tau_fast'(f,Mc)|,
    f_lo = f_b - Df_b,       f_hi = f_(b+1).

Round its centre to the nearest lag sample; `block_end` is exclusive.
Class zero stays centred. `L_margin` is a block-placement parameter,
distinct from the band-construction `L` in Eq. (51); their current default
values coincide. `sigma_mmax` is an optional conservative margin unit.
The implementation uses Newtonian prior endpoints, not a spin-dependent
prior envelope. Spin and merger coverage therefore needs a model-phase
audit; a new waveform/prior cannot inherit the validation automatically.

Requested slack is

    slack = T_b - (Lambda_max-Lambda_min) - delta_lo - delta_hi.

Also inspect the rounded block's two clearances. The current `PairwiseHH`
constructor warns and clips margins when requested slack is negative; it
does **not** reject the setup. Applications should reject negative
requested slack in truncated bands (as `population.make_evaluators` does).
Band zero has `M_b=N_b` and is exact on its full circular grid: negative
unwrapped SPA slack there is harmless. Do not confuse requested total
spare width with measured minimum one-sided clearance.

With raw modes `u_A` and frequency-independent extrinsic coefficients
`C_A`, `h_A = C_A u_A exp(-2 pi i f t_c)`. Define the physical matrix

    M_phys,AA' = (4/T) sum_b,k conj(u_A) W_k(B_AA') w_b u_A',
    (h,h) = C^dagger M_phys C.

For a fixed detector PSD this matrix is independent of common time/delay,
distance, inclination and antenna factors; it is phase-independent if the
mode API factors orbital phase exactly. The public `mode_matrix` returns
`M_API=(T/4) M_phys`, **excluding** `4/T`;
`hh_from_matrix(M_API,C)` restores it. Matrix construction is more costly
than one grouped norm, but subsequent extrinsic norms cost
`O(n_modes^2)`. Sky position enters `C` through antenna factors; common
delay cancels, assuming the fixed-response approximation used here.

The waveform API must return modes on arbitrary positive frequency
sequences, with consistent phase and projection conventions. In
LALSimulation 6.2.0 XHM the negative-`m` OneMode series is the one used at
positive frequencies. The positive API labels `(l,|m|)` combine the signed
pair through AddMode harmonics at azimuth `pi/2`:

    c_plus  = (Y_l,-m + (-1)^l conj(Y_l,m))/2,
    c_cross = i (Y_l,-m - (-1)^l conj(Y_l,m))/2,
    C_A = (F_plus c_plus + F_cross c_cross) exp(+i |m| phi_ref)/D_Mpc.

Raw modes are normalised to one Mpc with the nominal phase removed; the
five supported labels are `(2,2),(2,1),(3,3),(3,2),(4,4)`. No duplicate
positive/negative calls are needed. The sequence driver avoids the model's
internal regular-grid multibanding discrepancy. Tail recovery and the
mixed `(3,2)` phase residual are essential qualifications (§8).

## 5. Method B: total waveform only

**B0** computes `(4/T) sum_b,k Re W_centred,k w_b |h_k|^2` on the original
coarse grid. It needs one real vector per band and costs `O(K_MB)`, with
no modes or FFT. It is adequate when the two-sided total-power lag bulk
and margins fit inside `[-T_b/2,T_b/2)`, and often works even when weak
outlying pairs do not. Smooth PSD/single-mode controls and the XHM
populations support its use, but exaggerated HOM with narrow lines do not
([data/accuracy_iter004.md](data/accuracy_iter004.md)).

**B1** retains the centred `2M_b` block where necessary. Let
`K=K_e-K_s+1`, `K'=next_fast_len(ceil(p K))` and zero-pad the baseband
spectrum `x_j=sqrt(w_b) h((K_s+j)/T_b)`:

    y = ifft(x,K'),             z = fft(y exp(-i pi n/K')).

`z` supplies half-integer coarse frequencies. Mapping the last `T_b`
seconds to negative times gives a global minus sign for physical complex
half-point samples; powers are unaffected. Integer and half-point powers
use the even/odd entries of the centred `2M_b` weights, with no extra
window on `z`. Two length-`K'` complex FFTs replace length-`M_b` and
`2M_b` transforms spanning from zero frequency.

Adaptive B1 uses B0 if
`Lambda_max(f_lo)+L_margin sigma_mmax(f_lo) <= T_b/2`; a full circular band
is always exact with no transform. It needs only the positive `m` values
present, not their amplitudes or samples. Default `p=1`, band-only half
bins remain unchanged. Optional `pad_factor` and `all_half_bins` test
different finite interpolants; neither improves every system. B1 and the
paper's real full-spectrum interpolant have different tails when Eq. (36)
fails. They are not numerically identical, and endpoint corrections or
padding have not provided a universal cure
([data/b1_variants_iter006.md](data/b1_variants_iter006.md)).

For B0 the recommended optional **`hh_half`** band rule takes each internal
edge to the larger of the Eq. (51) root and the first feasible root of

    2 [Lambda_max(f_lo) + L_hh sigma_mmax(f_lo)] = T_b,
    Lambda_max = tau_0PN(2 f_lo/m_max) - tau_0PN(2 f_lo/m_min).

Eq. (47) recomputes the transition width at the changed edge. Use matched
`L_hh` and B1 `L_margin` if checking its shortcut. This rule leaves the
Eq. (51) time prior margin in place for `(d,h)` while adding the power-lag
condition. `hh_full` instead checks a conservative one-sided width with
both edge margins; it can bind for long BNS bands, but is not required by
the audited method-A blocks. Neither rule is a finite-tail error bound.

## 6. Accuracy and support evidence

Define `epsilon=|(h,h)_approx/(h,h)_fine-1|`. At amplitude rescaled to SNR
`rho`, the quadratic contribution to `|Delta ln Lambda|` is
`rho^2 epsilon/2`, i.e. `312.5 epsilon` at SNR 25. SNR 50/100 multiply the
latter by 4/16. This is not a test of the linear-term error or a full
parameter-estimation run.

| Evidence | Main result | Local source |
|---|---|---|
| Four SPA controls, A | Median/worst `epsilon=2.873e-7 / 3.536e-7`; likelihood `8.977e-5 / 1.105e-4`. IFFT-FFT `2.234e-7 / 2.938e-7` | [accuracy_iter003.md](data/accuracy_iter003.md) |
| XHM fixed cases, A | BNS median/worst `1.035e-8 / 1.159e-8`; GW190814-like `3.553e-7 / 4.281e-7`; BBH `2.031e-8 / 2.702e-5` (BBH worst is outside design prior) | [xhm_accuracy_iter005.md](data/xhm_accuracy_iter005.md) |
| Regular populations, A, paper L=5 | SNR-25 medians `1.263e-6`–`4.420e-5`, maxima up to `1.246e-4`; median ratios to IFFT-FFT `0.079`–`1.038` | [population_iter007.md](data/population_iter007.md) |
| Same populations, B1, L=5 | Median ratios `0.757`–`2.605`; p99 ratios `0.989`–`2.257`; C1' fails for BNS and BBH tails | [population_iter007.md](data/population_iter007.md) |
| L=50, A/B0/B1 | Regular medians roughly `3e-7`–`3e-6`; B1 selects B0 in every truncated band; A remains slightly distinct | [population_iter007.md](data/population_iter007.md) |
| B0 hh_half, L=L_hh=5 | Two-PSD fixed-case worst `epsilon=1.84e-7`, likelihood `5.74e-5`; not a hh_half population study | [bandrule_iter008.md](data/bandrule_iter008.md) |

The nominal GW190814-like chirp mass is below its historical band reference
`6.4`, although inside the pairwise `[6,7]` corners. Low-mass stress
variants are also outside band-design coverage. The populations use lower
reference masses and expose one NSBH draw above its upper block corner.
Keep these distinctions when interpreting C1; the ratio thresholds have
not been replaced by an absolute-error criterion.

Increasing band `L` generally suppresses tails but moves windows relative
to PSD lines and changes cancellations. Four-point fits over `L=5,10,20,50`
give A/IFFT-FFT slopes `-0.84/-0.30` (BNS), `-2.77/-0.97` (GW190814-like),
`-0.81/-0.15` (BBH), not universal `-3/-2` laws. The latter describe the
Appendix A16/A20 tail envelopes, not signed band-summed errors
([data/bandrule_iter008.md](data/bandrule_iter008.md)).

Block shifts within one pair sigma leave the dominant diagonal-fold floor
largely unchanged. Forcing negative clearance produces `epsilon=8.35e-4`,
likelihood `0.261`, at `-5.889 s`. Model-phase prior grids give minimum
measured truncated-band clearances `1.008510 / 0.956317 / 0.871841 s`
(BNS/NSBH/BBH), with no negative cases; these measure bulk tracks, not
window tails ([data/bandrule_iter008.md](data/bandrule_iter008.md),
[data/prior_slack_iter008.md](data/prior_slack_iter008.md)).

## 7. Evaluation cost and setup

Warm median milliseconds, one thread, prepared inputs; waveform and
quadratic step are separate. These are this machine's measurements,
not portable timing promises.

| Component | BNS | GW190814-like | BBH |
|---|---:|---:|---:|
| Prepared total / modes | 6.2072 / 7.5670 | 2.4600 / 3.2867 | 2.4018 / 5.2932 |
| Fast IFFT-FFT step | 3.2570 | 0.5880 | 0.5392 |
| A NumPy / numba step | 0.5576 / 0.2522 | 0.1572 / 0.0610 | 0.1676 / 0.0577 |
| B1 numba step | 0.3638 | 0.1065 | 0.1127 |
| B0 hh_half numba step | 0.0308 | 0.0109 | 0.0109 |
| A route ratio, 1 / 3 detectors | 0.826 / 0.521 | 1.098 / 0.821 | 1.819 / 1.360 |
| B1 route ratio, 1 / 3 detectors | 0.694 / 0.457 | 0.842 / 0.658 | 0.855 / 0.682 |
| B0 hh_half ratio, 1 / 3 detectors | 0.809 / 0.483 | 0.832 / 0.606 | 0.837 / 0.618 |

Source: [data/perf_iter009.md](data/perf_iter009.md), paper bands at L=5
except B0 hh_half. Route ratios are component sums relative to fast
IFFT-FFT: one waveform plus one/three quadratic steps, identical PSD
costs, negligible extra projection assumed. The hh_half waveform term is
scaled by point count; measured one-detector totals are separately listed
in the source. These exclude `(d,h)`, priors and sampler work. A fast
quadratic kernel does not imply a fast full likelihood, particularly with
BBH tail-recovery mode calls.

Weights are built once per fixed PSD/setup/prior. In BNS, A numba setup
costs `0.1514 s` at L=5 / `0.2271 s` at L=50, retaining
`64.726 / 78.946 MB`; traced peaks are `87.561 / 91.755 MB`. Those peaks
exclude caller-owned bands/PSD and JIT first dispatch. PSD changes require
new weights; intrinsic changes do not. Pin OpenMP/BLAS/FFT threads and warm
the numba path before timing. Prepared frequency sorting/unique mappings
remove repeated grid preparation; waveform values are never cached across
calls ([data/perf_iter009.md](data/perf_iter009.md)).

## 8. Limitations and open work

The validated scope is aligned-spin, non-precessing XHM with all supported
modes retained. Extending A to co-precessing modes requires exposing each
frequency-dependent twisting contribution and re-auditing its lag support;
frequency-dependent rotation cannot simply be put in constant `C`.
No precessing accuracy or marginalisation claim follows from these tests.

LAL OneMode cuts higher-mode tails at `Mf=0.3`; the adapter recovers them
through mode-selected polarization calls (also enabling 22 for mixed 32).
Dropping recovery would break the direct-waveform identity and artificially
improve BBH timing. Mixed 32 retains a `1.33194e-5` phase-shape residual
under changes of `phi_ref`; the matrix changes by `6.54606e-13` in the
max-scaled diagnostic, so strict phase independence is not demonstrated
([data/xhm_accuracy_iter005.md](data/xhm_accuracy_iter005.md)).

Wider priors require redesigned blocks and model-aware spin checks.
Newtonian clearances alone are not an error guarantee. The 55 native probe
was unsupported and crashed in a child process: m=5 classes in the prior
audit are labelled Newtonian fallback, not measured XHM evidence. The
skipped-duration audit still uses power-of-two subdivisions, not arbitrary
non-dyadic durations ([data/prior_slack_iter008.md](data/prior_slack_iter008.md)).

Spectral lines, unresolved measured features, diagonal period-T_b folding,
and nonmonotone L dependence remain relevant. The measured Welch PSD has
1/16-Hz resolution; interpolation to the BNS grid cannot recover narrower
features ([data/population_iter007.md](data/population_iter007.md)). B0's
success on these XHM populations does not override the strong-HOM toy
failures. A population study of hh_half and full bilby likelihood tests
remain useful before adopting it as a production default.

## 9. How to use the implementation

Import `BandSetup` and `PSDGrid` from `mbquad`; the specialised evaluators
are in their modules. `BlockWeights(bands,psdgrid)` exposes `weights`,
`centred`, and `lag_audit`. `PairwiseHH` accepts `CoarseModes` or a mapping
of mode labels to `CoarseWaveform`; `CentredBlockHH` and
`BasebandIFFTFFTHH` accept total `CoarseWaveform`. Samples are unwindowed;
evaluators apply the windows. Containers must share the exact `BandSetup`
instance with their evaluator. `XHMCoarseEvaluator` prepares fixed LAL
frequency sequences once and accepts changed model parameters per call.

A runnable 20-line example (setup parameters follow the documented BNS
case; `hh_half` and numba are explicit choices, not changed defaults):

```python
import numpy as np
from mbquad import BandSetup, PSDGrid, aligo_analytic_psd
from mbquad.lalwave import XHMModel, XHMCoarseEvaluator, XHM_MODES
from mbquad.pairwise import PairwiseHH
from mbquad.totalweights import CentredBlockHH
from mbquad.baseband import BasebandIFFTFFTHH
bands = BandSetup(256, 1/4096, 20, 2048, [256,128,64,32,16,8,4],
                  mchirp_ref=1.15, m_max=4, L=5, rule="hh_half")
psd = PSDGrid(aligo_analytic_psd, 256, 1/4096, 20, 2048)
model = XHMModel(1.4, 1.4, 0, 0, 40, 1, .7, 254)
wave = XHMCoarseEvaluator(model, bands)
b0 = CentredBlockHH(bands, psd, impl="numba")
b1 = BasebandIFFTFFTHH(bands, psd, impl="numba")
a = PairwiseHH(bands, psd, XHM_MODES, mchirp_range=(1.15,1.25),
               impl="numba")
assert all(r["full_grid"] or r["slack"] >= 0 for r in a.summary())
total = wave.coarse_total()
modes = wave.coarse_modes()
values = np.array([a(modes), b0(total), b1(total)])
print(values, b1.modes)
```

For matrix reuse, construct `CoarseModes` from `model.raw_modes(f)` in
`a.modes` order, omit `t_c`, then call `a.mode_matrix(raw)` and
`a.hh_from_matrix(M_API,C)`. `coarse_modes()` returns projected modes and
is not the raw input for extrinsic reuse.

NumPy defaults remain A `impl="pairs"`, B0/B1 `impl="numpy"`; numba is
opt-in, with cached compilation and no fast-math. Warm the first call.
Evaluators reuse mutable buffers; use one instance per concurrent worker.
For a fixed paper-band total-waveform route, use B1 and validate its error
at the intended L/PSD; when edges can change, start with B0 + hh_half.
