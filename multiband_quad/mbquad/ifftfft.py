"""The paper's time-cropped IFFT-FFT quadratic term, Eqs. (31)–(46)."""

import numpy as np
from scipy.fft import irfft, rfft

from .coarse import check_setup


class IFFTFFTHH:
    """Eq. (45) using Eq. (46) to interpolate sqrt(w^(b)) h.

    ``impl='paper'`` uses complex numpy IFFT/FFT literally; ``impl='fast'``
    uses scipy real transforms and reusable spectrum/time/power buffers.
    Both assume Eq. (36): each windowed waveform is concentrated in the
    last T_b seconds. There is no time-cropping approximation when M_b=N_b.
    All frequency sums exclude DC and Nyquist, as in Eqs. (4), (45).

    Instances reuse mutable work buffers and must not be called concurrently.
    scipy FFT results themselves are allocated by scipy (no out parameter).
    """

    def __init__(self, bands, psdgrid, impl="fast"):
        check_setup(bands, psdgrid)
        if impl not in ("paper", "fast"):
            raise ValueError("impl must be 'paper' or 'fast'")
        self.bands, self.impl = bands, impl
        self.Nhat_b = np.minimum(2 * bands.M_b, bands.N_b)
        self.That_b = self.Nhat_b * bands.dt_b
        self.I_tilde_b = []
        self._grids = bands.coarse_frequencies()
        self._sqrt_w = [np.sqrt(grid.w) for grid in self._grids]
        self._spectrum, self._time, self._power = [], [], []
        self._band_slices = []
        for b, grid in enumerate(self._grids):
            N_b, M_b, Nhat = map(int, (bands.N_b[b], bands.M_b[b], self.Nhat_b[b]))
            # Eq. (32): zero-extend inv_S if N_b>N; truncate it if N_b<N.
            a = np.zeros(N_b, dtype=complex)
            count = min(len(psdgrid.inv_S), (N_b - 1) // 2)
            a[1:count + 1] = psdgrid.inv_S[:count]
            I = (2 * N_b / bands.T) * np.fft.ifft(a).real
            # Eq. (38): preserve positive lags first, negative lags last.
            I_c = np.concatenate((I[:(Nhat + 1) // 2], I[N_b - Nhat // 2:]))
            # Eq. (43). I is real and circularly even, as is I_c; its DFT
            # is therefore real up to roundoff. Keep only Eq. (45)'s bins.
            I_tilde = bands.dt_b[b] * np.fft.fft(I_c)
            self.I_tilde_b.append(I_tilde[1:(Nhat + 1) // 2].real.copy())
            self._spectrum.append(np.zeros(M_b if impl == "paper" else M_b // 2 + 1,
                                           dtype=complex))
            self._time.append(np.zeros(Nhat))
            self._power.append(np.empty((Nhat - 1) // 2))
            ratio = Nhat // M_b
            self._band_slices.append(slice(int(grid.kappa[0] * ratio),
                                           int(grid.kappa[-1] * ratio) + 1))
        self.I_tilde_b = tuple(self.I_tilde_b)

    def _transform(self, coarse, b):
        """Eqs. (46), (44): window, inverse transform, pad last M_b, FFT."""
        M_b = int(self.bands.M_b[b])
        dt_b = self.bands.dt_b[b]
        x, h_c = self._spectrum[b], self._time[b]
        grid = self._grids[b]
        np.multiply(self._sqrt_w[b], coarse.h_b[b],
                    out=x[grid.kappa[0]:grid.kappa[-1] + 1])
        if self.impl == "paper":
            h_c[-M_b:] = np.fft.ifft(x).real
            h_c[-M_b:] *= 2 * M_b / self.bands.T_b[b]
            return dt_b * np.fft.fft(h_c)
        # irfft supplies conjugate negative frequencies, hence the factor
        # M_b/T_b rather than 2*M_b/T_b used by the one-sided complex IFFT.
        h_c[-M_b:] = irfft(x, n=M_b, workers=1)
        h_c[-M_b:] *= M_b / self.bands.T_b[b]
        return dt_b * rfft(h_c, workers=1)

    def band_contributions(self, coarse):
        """Return Eq. (45)'s individual band norms (before the band sum)."""
        coarse.check_bands(self.bands)
        values = []
        for b, I_tilde in enumerate(self.I_tilde_b):
            h_tilde = self._transform(coarse, b)[1:len(I_tilde) + 1]
            power = self._power[b]
            np.absolute(h_tilde, out=power)
            np.square(power, out=power)
            values.append((4 / self.That_b[b]) * np.dot(I_tilde, power))
        return np.asarray(values)

    def __call__(self, coarse):
        """Evaluate Eq. (45) from unwindowed total CoarseWaveform samples."""
        return float(np.sum(self.band_contributions(coarse)))

    def half_point_frequencies(self, b):
        """Eq. (44) frequencies within [K_s/T_b,K_e/T_b], in Hz.

        The grid contains integer and half-integer coarse points if Nhat=2M;
        for Nhat=M (no cropping) it contains only the original coarse points.
        """
        self.bands._check_band(b)
        s = self._band_slices[b]
        return np.arange(s.start, s.stop) / self.That_b[b]

    def half_point_waveform(self, coarse, b):
        """Return Eq. (44)'s h_tilde_c restricted to the band's frequency span.

        The result interpolates sqrt(w) h, including both original and half
        points when Nhat=2M; use half_point_frequencies(b) for its grid.
        It agrees directly with unwindowed h in the window's unit plateau.
        """
        coarse.check_bands(self.bands)
        self.bands._check_band(b)
        return self._transform(coarse, b)[self._band_slices[b]].copy()

    def positive_grid_waveform(self, coarse, b):
        """Eq. (44) on ALL positive non-Nyquist bins, including leakage tails.

        Return (f,h_tilde_c). Unlike half_point_waveform's band-span slice,
        this retains interpolation leakage outside the original band and
        is suitable for an exact Eq. (45)/block-weight identity check.
        """
        coarse.check_bands(self.bands)
        self.bands._check_band(b)
        n = int(self.Nhat_b[b])
        k = np.arange(1, (n + 1) // 2)
        return k / self.That_b[b], self._transform(coarse, b)[k].copy()
