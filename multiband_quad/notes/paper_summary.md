# Plain-text summary of arXiv:2104.07813 (Morisaki 2021, PRD 104, 044062)

"Accelerating parameter estimation of gravitational waves from compact binary coalescence using
adaptive frequency resolutions". The PDF is `notes/arXiv_2104.07813v2.pdf`. Equation numbers
are the paper's. Notation: data duration `T = N dt`, fine grid `f_k = k/T`, one-sided PSD `S_k`
(`S_k = +inf` outside `[f_low, f_high]`, Eq. 6), waveform `h(f)`.

## Likelihood (Sec. II.A)

    ln L = (d, h) - (h, h)/2 + const                                   (2)
    (d, h) = (4/T) Re sum_{k=1}^{floor((N-1)/2)} d_k^* h(f_k) / S_k      (3)
    (h, h) = (4/T)    sum_{k=1}^{floor((N-1)/2)} |h(f_k)|^2 / S_k        (4)
    K_orig = floor(f_high T) - ceil(f_low T) + 1 ~ (f_high - f_low) T     (8)

## Window functions (Sec. II.B)

Band edges `f_low = f^(0) < f^(1) < ... < f^(B) = f_high + Df^(B)` with transition widths
`Df^(b)` (`f^(b) < f^(b+1) - Df^(b+1)`, `Df^(0) = 0`). Smooth window of band `b` (Eq. 9):

    w^(b)(f) = (1/2)(1 + cos(pi (f - f^(b))/Df^(b)))        f^(b) - Df^(b) < f < f^(b)
             = 1                                            f^(b) <= f <= f^(b+1) - Df^(b+1)
             = (1/2)(1 - cos(pi (f - f^(b+1))/Df^(b+1)))    f^(b+1) - Df^(b+1) < f < f^(b+1)
             = 0                                            otherwise
    sum_{b=0}^{B-1} w^(b)(f) = 1 for f_low <= f <= f_high                                (11)

Appendix A shows why a smooth window is needed: with a rectangular window the band waveform in
the time domain has a tail decaying only like `1/(t(f^(b)) - t)` (Eq. A10); with the smooth
window the tail decays like `(t(f^(b)) - t)^-3` (Eqs. A15-A16), and with the square root of the
smooth window (used for the IFFT-FFT method) like `(t(f^(b)) - t)^-2` (Eqs. A18-A20).

## (d, h) with coarse grids (Sec. II.C)

Per band choose a power of two `N^(b)` with `floor((N^(b)-1)/2) >= f^(b+1) T` (Eq. 13), set
`dt^(b) = N dt / N^(b)` (= `T/N^(b)`). Writing the band's inner product as a time-domain sum
(Eq. 15) with `D^(b)_m` (IFT of `d_k/S_k`, Eq. 16) and `h^(b)_m` (IFT of the windowed waveform,
Eq. 17): because the chirp frequency increases with time, `h^(b)_m ~= 0` for
`m dt^(b) < T - tau(f^(b) - Df^(b))` where `tau(f)` is the time to merger from frequency `f`
(for multi-mode waveforms, of the mode with the longest time to merger). Hence (Eq. 18)

    h^(b)_m ~= 0   for m = 0, ..., N^(b) - M^(b) - 1,     T^(b) = M^(b) dt^(b),

and the sum over `m` reduces to the last `M^(b)` samples (Eq. 20), which is a sum over the
coarse grid `f^(b)_k = k/T^(b)` (Eqs. 21-24):

    (d, h) ~= sum_b (4/T^(b)) Re sum_{k=K_s^(b)}^{K_e^(b)} w^(b)(f^(b)_k) Dtilde^(b)*_k h(f^(b)_k)   (24)
    K_s^(b) = ceil((f^(b) - Df^(b)) T^(b)),   K_e^(b) = floor(f^(b+1) T^(b))
    Dtilde^(b)_k = dt^(b) sum_{m=N^(b)-M^(b)}^{N^(b)-1} D_m e^{-2 pi i k m / M^(b)}            (22)
    K_MB = sum_b (K_e^(b) - K_s^(b) + 1)                                                        (25)

`Dtilde^(b)` is precomputed; the run-time cost is `K_MB` waveform evaluations.

## (h, h) (Sec. II.D)

    (h, h) = sum_b (4/T) sum_k w^(b)(f_k) |h(f_k)|^2 / S_k                                      (26)

### Linear interpolation (Sec. II.D.1)

`|h(f)|^2` is linearly interpolated between coarse points (Eq. 27). Then

    (h, h) ~= sum_b sum_{k=K_s}^{K_e} c^(b)_k |h(f^(b)_k)|^2                                   (28)
    c^(b)_k = (4 T^(b)/T) [ sum_{f_k^(b) <= f_l < f_{k+1}^(b)} (f^(b)_{k+1} - f_l) w^(b)(f_l)/S_l
                          + sum_{f_{k-1}^(b) <= f_l < f_k^(b)} (f_l - f^(b)_{k-1}) w^(b)(f_l)/S_l ]  (29)

with the edge conventions `f^(b)_{K_s-1} = f^(b)_{K_s} = 0`, `f^(b)_{K_e} = f^(b)_{K_e+1} = inf`
(Eq. 30): the first and last coarse bins are extrapolated. Cost `O(K_MB)`. Inaccurate when
`|h|^2` oscillates (cross terms between modes, Fig. 2).

### IFFT-FFT (Sec. II.D.2)

Time-domain form of the band sum (Eqs. 31-33): `(4/T) sum_k w |h|^2/S_k = 2 dt^(b) sum_m I_m H_m`
with `I_m = (2/T) Re sum_k e^{2 pi i k m/N^(b)} / S_k` (precomputed) and `H_m` the IFT of
`w |h|^2`, which is the circular autocorrelation (Eq. 34) of `hhat^(b)_m`, the IFT of
`sqrt(w^(b)) h` (Eq. 35). As for (d, h), `hhat^(b)_m ~= 0` for `m < N^(b) - M^(b)` (Eq. 36), so
`H_m` is supported on `2 M^(b) - 1` samples; crop both sequences to
`Nhat^(b) = min(2 M^(b), N^(b))` samples (Eqs. 37-41: the first `ceil(Nhat/2)` and the last
`floor(Nhat/2)` indices, i.e. a block centred on lag 0) and use Parseval on the cropped
sequences (Eqs. 42-44):

    (h, h) ~= sum_b (4/That^(b)) sum_{k=1}^{floor((Nhat^(b)-1)/2)} Itilde^(b)_{c,k} |htilde^(b)_{c,k}|^2   (45)
    That^(b) = Nhat^(b) dt^(b) (= 2 T^(b) when 2 M^(b) <= N^(b))

`Itilde_c` is precomputed. `htilde_c` is obtained at run time by: IFFT of the `M^(b)` coarse
samples `sqrt(w) h(f^(b)_k)` (Eq. 46) -> time series of length `M^(b)` placed in the last
`M^(b)` of `Nhat^(b)` samples, zeros elsewhere -> FFT of length `Nhat^(b)`. I.e. it is the
coarse band waveform interpolated onto the twice finer grid `k/That^(b)`. Cost
`O(Nhat log Nhat)` per band per likelihood evaluation; the paper notes this is not a fixed
`O(N)` cost but it is more costly than linear interpolation.

## Choice of bands (Sec. II.E, II.F)

Given user-specified `{T^(b)}` (e.g. `T, T/2, T/4, ...`), the edges `f^(b)` solve (Eq. 51)

    tau(f^(b)) + L sqrt(-tau'(f^(b))) = T^(b) + t_c,min - T,       Df^(b) = 1/sqrt(-tau'(f^(b)))   (47)

with `L >> 1` (L = 5 is enough for SNR ~ 25; L = 50 is more accurate), `t_c,min = t_earth,min -
R_earth/c` (LVK prior: `t_earth,min = T - 2.1 s`, so `t_c,min - T = -2.12 s`), and the
leading-order chirp time (Eq. 53)

    tau_0PN(f) = (5/256) (G Mc/c^3) (pi G Mc f/c^3)^(-8/3),   Mc = (m1 m2)^(3/5)/(m1+m2)^(1/5)

evaluated at the minimum chirp mass of the prior; with higher-order modes (Eq. 55)

    tau(f) = tau_0PN(2 f / m_max).

The highest band's upper transition width is `Df^(B) = 100/(T - t_c,max)` (Eq. 56; ~53 Hz for
the LVK prior), needed so that the band waveform decays quickly (Fig. 3).

## Results relevant here (Sec. II.G, III)

- Table I: 1.4-1.4 Msun BNS, f_high = 2048 Hz. f_low = 20 Hz, T = 256 s: K_orig = 5.2e5,
  K_orig/K_MB = 45 (IMRPhenomD) and 27 (IMRPhenomHM, m_max = 4); speed-ups 51 and 21.
- GW190814 (Sec. III.A): 16 s of data, bands {16, 8, 4} s, L = 5 or 50, reference chirp mass
  6.4 Msun, m_max = 4 for IMRPhenomPv3HM. Errors in ln Lambda (Fig. 4): IMRPhenomD with linear
  interpolation: median 4e-3 (L = 5), 2e-4 (L = 50). IMRPhenomPv3HM with IFFT-FFT: median 2e-4
  (L = 5), 5e-5 (L = 50); with linear interpolation and L = 5 the errors are larger (~1e-3).
  Waveform evaluations reduced by 3.3 (L = 5) and 2.8 (L = 50) for Pv3HM.
- Injection study (Sec. III.B): 256 BNS signals, 20-2048 Hz, {256, ..., 4} s, L = 5,
  reference chirp mass 1.15 Msun, median network SNR 24.3: P-P plot consistent.

## Stationary phase facts used (Appendix A)

    h(t) = A(t) cos Phi(t);   h(f) ~= B(f) e^{-i Psi(f)} (f > 0),  B = sqrt(t'(f)) A(t(f))/2,
    Psi(f) = -Phi(t(f)) + 2 pi f t(f) - pi/4,   Phi'(t(f)) = 2 pi f,   so dPsi/df = 2 pi t(f).

For an (l, m) mode the instantaneous GW frequency is `(m/2)` times the (2,2) one, so
`t_lm(f) = t_22(2 f/m)` and `tau_lm(f) = tau_22(2 f/m)`.
