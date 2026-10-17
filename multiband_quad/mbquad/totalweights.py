"""FFT-free centred-block quadrature of the total power, method_idea §8.8."""

import numpy as np

from .blockweights import BlockWeights


def _m_values(m_values, m_max):
    """Validate the positive m values actually present in a total waveform."""
    values = tuple(m_values)
    if (not values or any(not isinstance(m, (int, np.integer)) or m < 1
                          for m in values) or max(values) > m_max):
        raise ValueError("m_values must be nonempty positive integers <= BandSetup m_max")
    return tuple(sorted(set(values)))


class CentredBlockHH:
    """B0: (4/T) sum_b,k Re(W_k centred on zero) w_b |h_k|².

    Implements method_idea §§3,8.8 on Eq. (24)'s TOTAL waveform grid.
    Exact when G=N_b ifft(w_b |h|²) is supported in [-T_b/2,T_b/2),
    and always exact if M_b=N_b. Otherwise the error is lag folding (§8.5).
    Instances reuse power buffers and must not be evaluated concurrently.
    impl='numba' fuses power and weighting on a reusable flattened grid;
    the default impl='numpy' preserves the original calculation.
    """

    def __init__(self, bands, psdgrid, impl='numpy'):
        if impl not in ('numpy', 'numba'):
            raise ValueError("impl must be 'numpy' or 'numba'")
        self.impl = impl
        self.blockweights = BlockWeights(bands, psdgrid)
        self.bands = bands
        self.W = tuple(self.blockweights.centred(b, int(bands.M_b[b]))[
            grid.kappa].real * grid.w
            for b, grid in enumerate(bands.coarse_frequencies()))
        self._power = tuple(np.empty(int(k)) for k in bands.K_b)
        if impl == 'numba':
            self.w0 = np.concatenate(self.W) * (4/bands.T)
            self._flat_h = np.empty(bands.K_MB, complex)

    def flatten(self, coarse):
        """Return current Eq. (24) samples, copying independent bands only."""
        from .kernels import flatten_samples
        coarse.check_bands(self.bands)
        return flatten_samples(coarse.h_b, self._flat_h)

    def band_contributions(self, coarse):
        """Return Eq. (26)'s approximate band norms, including the 4/T factor."""
        coarse.check_bands(self.bands)
        if self.impl == 'numba':
            from .kernels import b0_hh
            return np.asarray([(4/self.bands.T)*b0_hh(h, W)
                               for h, W in zip(coarse.h_b, self.W)])
        values = np.empty(self.bands.B)
        for b, (h, W, power) in enumerate(zip(coarse.h_b, self.W, self._power)):
            np.absolute(h, out=power)
            np.square(power, out=power)
            values[b] = (4 / self.bands.T) * np.dot(W, power)
        return values

    def __call__(self, coarse):
        """Evaluate B0 using only unwindowed total CoarseWaveform samples."""
        if self.impl == 'numba':
            from .kernels import b0_hh
            return float(b0_hh(self.flatten(coarse), self.w0))
        return float(np.sum(self.band_contributions(coarse)))

    def predicted_lag_support(self, b, m_values, mchirp=None):
        """Return (lag_max,T_b/2) in seconds, using Eqs. (53),(55).

        lag_max=tau_mmax(f_lo)-tau_mmin(f_lo) predicts the two-sided
        support [-lag_max,+lag_max]; finite-window tails are excluded.
        Optional mchirp is in solar masses; default is the setup reference.
        """
        self.bands._check_band(b)
        ms = _m_values(m_values, self.bands.m_max)
        f_lo = self.bands.f_b[b] - self.bands.Df_b[b]
        lag = self.bands.lag_pair(f_lo, ms[-1], ms[0], mchirp)
        return float(lag), float(self.bands.T_b[b] / 2)

    def folding_error_estimate(self, coarse_fine_g, b):
        """Fine-grid lag audit of g=w_b |h|², method_idea §8.5.

        Input already includes the window, on psdgrid.f_k or a full N_b
        DFT grid. Returns BlockWeights.lag_audit's dictionary plus
        hh_folding_error=(4/T) Re(folding_error). No waveform interpolation
        enters this diagnostic; supply a fine-grid product, not coarse h.
        """
        self.bands._check_band(b)
        M_b = int(self.bands.M_b[b])
        audit = self.blockweights.lag_audit(b, coarse_fine_g, M_b, -M_b // 2)
        audit["hh_folding_error"] = float((4 / self.bands.T)
                                         * audit["folding_error"].real)
        return audit
