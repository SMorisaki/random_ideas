"""Total-waveform baseband IFFT-FFT, method_idea §7 (method B1)."""

import numpy as np
from scipy.fft import fft, ifft, next_fast_len

from .totalweights import CentredBlockHH, _m_values
from .toywave import tau_0pn


class BasebandIFFTFFTHH:
    """B1: Eq. (45)'s block quadrature with two length-K' baseband FFTs.

    x_j=sqrt(w_b) h((K_s+j)/T_b), y=ifft(x,K'),
    z=fft(y exp(-i pi n/K')), K'=next_fast_len(ceil(pad_factor K_b)).
    all_half_bins sums all K' half bins with weights modulo 2M_b;
    otherwise only the band's K_b half bins are retained. Defaults p=1
    and band-only preserve the original discrete interpolant. Integer and half
    powers use Re W for the centred min(2M_b,N_b) block (§3). If
    lag_max+L_margin sigma_mmax(f_lo)<=T_b/2, use B0 instead; M_b=N_b
    always needs only the original samples. force_half_points bypasses
    the lag rule, but never the exact-grid shortcut of Eq. (37).

    Like Eq. (36), interpolation assumes time support in the last T_b;
    the finite K' periodic baseband interpolant has different edge tails
    from the paper's full-spectrum, real-signal interpolant. Equality of
    those finite interpolants is not guaranteed by Eq. (36). No mode
    amplitudes or mode samples are used; m_values only sets the lag rule.
    Buffers are reused: instances are not safe for concurrent evaluation.
    scipy allocates FFT results (it has no out parameter).
    """

    def __init__(self, bands, psdgrid, m_values=(1, 2, 3, 4), L_margin=5.0,
                 force_half_points=False, pad_factor=1, all_half_bins=False):
        self.m_values = _m_values(m_values, bands.m_max)
        if not np.isfinite(L_margin) or L_margin < 0:
            raise ValueError("L_margin must be finite and nonnegative")
        self.bands = bands
        self.L_margin = float(L_margin)
        self.force_half_points = bool(force_half_points)
        if not np.isfinite(pad_factor) or pad_factor < 1:
            raise ValueError("pad_factor must be finite and >= 1")
        self.pad_factor = float(pad_factor)
        self.all_half_bins = bool(all_half_bins)
        self.centred = CentredBlockHH(bands, psdgrid)
        self.blockweights = self.centred.blockweights
        self.Kprime_b = tuple(next_fast_len(int(np.ceil(pad_factor*k))) for k in bands.K_b)
        self._grids = bands.coarse_frequencies()
        self._phase, self._input, self._sqrt_w, self._power = [], [], [], []
        self.W_even, self.W_odd, self.modes = [], [], []
        self._half_power = []
        self.lag_max, self.margin = [], []
        for b, grid in enumerate(self._grids):
            M, N = int(bands.M_b[b]), int(bands.N_b[b])
            lag, half = self.centred.predicted_lag_support(b, self.m_values)
            flo = bands.f_b[b] - bands.Df_b[b]
            sigma = np.sqrt((8 / 3) * tau_0pn(2 * flo / self.m_values[-1],
                                            bands.mchirp_ref) / flo)
            margin = float(self.L_margin * sigma)
            mode = ("exact" if M == N else
                    "B0" if not force_half_points and lag + margin <= half else "B1")
            self.lag_max.append(lag)
            self.margin.append(margin)
            self.modes.append(mode)
            Kp = self.Kprime_b[b]
            # Retain buffers even in skipped bands for diagnostic half points.
            self._input.append(np.zeros(Kp, complex))
            self._phase.append(np.exp(-1j * np.pi * np.arange(Kp) / Kp))
            self._sqrt_w.append(np.sqrt(grid.w))
            self._power.append(np.empty(len(grid.f)))
            n_half = Kp if self.all_half_bins else len(grid.f)
            self._half_power.append(np.empty(n_half))
            if mode == "B1":
                W = self.blockweights.centred(b, 2 * M)
                self.W_even.append(W[2 * grid.kappa].real * grid.w)
                # z already includes sqrt(w); applying the window twice is wrong.
                idx = 2 * (grid.kappa[0] + np.arange(n_half)) + 1
                self.W_odd.append(W[idx % (2*M)].real.copy())
            else:
                self.W_even.append(self.centred.W[b])
                self.W_odd.append(None)

    def _half_transform(self, coarse, b):
        """§7 half-sample phase on baseband indices n=0,...,K'-1."""
        x = self._input[b]
        K = int(self.bands.K_b[b])
        np.multiply(self._sqrt_w[b], coarse.h_b[b], out=x[:K])
        # The zero tail is never overwritten by scipy's non-in-place IFFT.
        y = ifft(x, workers=1)
        np.multiply(y, self._phase[b], out=y)
        return fft(y, workers=1)[:len(self._half_power[b])]

    def band_contributions(self, coarse):
        """Return B1/B0 band norms, Eq. (26), each including 4/T."""
        coarse.check_bands(self.bands)
        values = np.empty(self.bands.B)
        for b, h in enumerate(coarse.h_b):
            power = self._power[b]
            np.absolute(h, out=power)
            np.square(power, out=power)
            value = np.dot(self.W_even[b], power)
            if self.modes[b] == "B1":
                z = self._half_transform(coarse, b)
                half_power = self._half_power[b]
                np.absolute(z, out=half_power)
                np.square(half_power, out=half_power)
                value += np.dot(self.W_odd[b], half_power)
            values[b] = (4 / self.bands.T) * value
        return values

    def __call__(self, coarse):
        """Evaluate from total CoarseWaveform samples, without mode access."""
        return float(np.sum(self.band_contributions(coarse)))

    def half_point_frequencies(self, b):
        """Return (K_s+j+1/2)/T_b for the retained half bins, in Hz.

        Band-only includes the final half point beyond K_e (§7); all_half_bins
        extends to K'-1 with periodic weights (not a new physical band).
        These are half points only, unlike IFFTFFTHH's interleaved grid.
        """
        self.bands._check_band(b)
        return (self._grids[b].kappa[0] + np.arange(len(self._half_power[b])) + 0.5) / self.bands.T_b[b]

    def half_point_waveform(self, coarse, b):
        """Return -z=sqrt(w) h at half points in the physical phase convention.

        Uses the baseband transform even for bands skipped by the adaptive
        rule. Mapping the last T_b seconds to negative times n-K' gives
        exp(-i pi (n-K')/K')=-exp(-i pi n/K'); this global minus restores
        the paper's/physical phase convention and leaves powers unchanged.
        Finite-grid edge tails differ. Returns owned samples including
        the last half point.
        """
        coarse.check_bands(self.bands)
        self.bands._check_band(b)
        return -self._half_transform(coarse, b)

    def fft_lengths(self):
        """Return per-band (K',mode); exact/B0 bands execute no FFTs.

        K' describes the available diagnostic transform even in skipped
        bands; actual runtime transform lengths there are zero.
        """
        return list(zip(self.Kprime_b, self.modes))

    def summary(self):
        """Markdown FFT lengths and Newtonian lag anchors in seconds, §7."""
        rows = ["| b | K_b | K' | M_b | 2 M_b | lag_max (s) | T_b/2 (s) | margin (s) | mode |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for b, (Kp, mode) in enumerate(self.fft_lengths()):
            M = int(self.bands.M_b[b])
            rows.append(f"| {b} | {self.bands.K_b[b]} | {Kp} | {M} | {2*M} | "
                        f"{self.lag_max[b]:.6f} | {self.bands.T_b[b]/2:.6f} | "
                        f"{self.margin[b]:.6f} | {mode} |")
        return "\n".join(rows)
