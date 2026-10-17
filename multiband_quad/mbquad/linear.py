"""Linear interpolation of total-waveform power, Morisaki Eqs. (27)–(30)."""

import numpy as np

from .coarse import check_setup


class LinearInterpolationHH:
    """Eq. (28): sum_b sum_kappa c_kappa^(b) |h(f_kappa^(b))|².

    Precompute Eq. (29) on PSDGrid's positive fine frequencies, excluding
    DC/Nyquist. Eq. (30)'s modified frequencies affect summation ranges
    only; the weights use the true coarse frequencies, including at the
    extrapolated endpoints. Accuracy requires approximately linear power
    between coarse samples; interference between modes can violate this.
    """

    def __init__(self, bands, psdgrid):
        check_setup(bands, psdgrid)
        self.bands = bands
        self.c_b = []
        f = psdgrid.f_k
        for b, grid in enumerate(bands.coarse_frequencies()):
            if len(grid.f) < 2:
                raise ValueError("Linear interpolation needs at least two coarse points per band")
            f_tilde = grid.f.copy()
            f_tilde[0], f_tilde[-1] = 0, np.inf
            # Each fine point belongs to one modified interval [f~_j,f~_{j+1}).
            # Scatter its two true-frequency interpolation weights to j,j+1.
            j = np.searchsorted(f_tilde, f, side="right") - 1
            a = bands.window(f, b) * psdgrid.inv_S
            right = np.bincount(j, weights=(grid.f[j + 1] - f) * a,
                                minlength=len(grid.f))
            left = np.bincount(j + 1, weights=(f - grid.f[j]) * a,
                               minlength=len(grid.f))
            self.c_b.append((4 * bands.T_b[b] / bands.T) * (right + left))
        self.c_b = tuple(self.c_b)

    def __call__(self, coarse):
        """Evaluate Eq. (28) from unwindowed total CoarseWaveform samples."""
        coarse.check_bands(self.bands)
        return float(sum(np.dot(c, np.abs(h)**2)
                         for c, h in zip(self.c_b, coarse.h_b)))
