"""PSD block weights for Eq. (26); METHOD_NOTE.md §3.

W_k(B)=(1/Nhat) sum_{m in B} conj(A_m) exp(2 pi i m k/Nhat),
A=N_b ifft(1/S). The identity is exact for block-supported inverse
transforms of g=w_b conj(h_A) h_A'; lag_audit measures the folding error.
"""

import numpy as np

from .coarse import check_setup


class BlockWeights:
    """Cache A_m=N_b ifft(a)_m and W(B), method_idea §3.

    a is the one-sided inverse PSD, without a band window; positive bins
    above the original Nyquist are zero. Nhat must divide N_b. The block
    identity is exact when the inverse transform of g is supported in B.
    Cached arrays are read-only. nbytes excludes the caller-owned PSD/grid.
    """

    def __init__(self, bands, psdgrid):
        check_setup(bands, psdgrid)
        self.bands, self.psdgrid = bands, psdgrid
        self.A_m = {}
        self._weights = {}
        for n in map(int, np.unique(bands.N_b)):
            a = self.fine_inverse_psd(n)
            A = n * np.fft.ifft(a)
            A.flags.writeable = False
            self.A_m[n] = A

    def fine_inverse_psd(self, N_b):
        """Eq. (6) a_k on a length-N_b complex DFT grid, excluding Nyquist."""
        a = np.zeros(N_b)
        count = min(len(self.psdgrid.inv_S), (N_b - 1) // 2)
        a[1:count + 1] = self.psdgrid.inv_S[:count]
        return a

    def _check_block(self, b, m0, Nhat):
        self.bands._check_band(b)
        n = int(self.bands.N_b[b])
        if (not isinstance(Nhat, (int, np.integer)) or Nhat < 1
                or Nhat > n or n % Nhat):
            raise ValueError("Nhat must be a positive integer divisor of N_b")
        if not isinstance(m0, (int, np.integer)):
            raise ValueError("m0 must be an integer sample index")
        return n

    def weights(self, b, m0, Nhat, kappa=None):
        """W_k=(1/Nhat) sum_{m in B} conj(A_m) exp(2 pi i m k/Nhat).

        One length-Nhat inverse FFT and a phase implement method_idea §3,
        with the same convention as notes/orchestrator_toy.block_weight.
        kappa, if supplied, contains integer DFT indices (taken modulo Nhat).
        """
        n = self._check_block(b, m0, Nhat)
        key = (int(b), int(m0) % n, int(Nhat))
        if key not in self._weights:
            # Reduce the phase modulo Nhat before exponentiation for accuracy.
            j = np.arange(Nhat)
            idx = (m0 + j) % n
            phase = np.exp(2j * np.pi * ((m0 * j) % Nhat) / Nhat)
            W = phase * np.fft.ifft(self.A_m[n][idx].conj())
            W.flags.writeable = False
            self._weights[key] = W
        W = self._weights[key]
        if kappa is None:
            return W
        kappa = np.asarray(kappa)
        if not np.issubdtype(kappa.dtype, np.integer):
            raise ValueError("kappa must contain integer DFT indices")
        return W[kappa % Nhat]

    def centred(self, b, Nhat):
        """Return weights for B={-floor(Nhat/2),...,ceil(Nhat/2)-1}."""
        return self.weights(b, -(Nhat // 2), Nhat)

    def for_lag_interval(self, b, lag_lo, lag_hi, Nhat):
        """Centre a length-Nhat block on [lag_lo,lag_hi] (seconds), §8.

        Return (W, start, end); end is the exclusive block boundary.
        Rounding uses numpy's nearest-even convention, as in PairwiseHH.
        """
        if not np.all(np.isfinite([lag_lo, lag_hi])) or lag_hi < lag_lo:
            raise ValueError("Require finite lag_lo <= lag_hi")
        dt = self.bands.dt_b[b]
        m0 = int(np.rint((lag_lo + lag_hi) / (2 * dt))) - Nhat // 2
        W = self.weights(b, m0, Nhat)
        return W, float(m0 * dt), float((m0 + Nhat) * dt)

    def nbytes(self):
        """Bytes of cached kernels and weight vectors (each allocation once)."""
        return sum(a.nbytes for a in self.A_m.values()) + sum(
            w.nbytes for w in self._weights.values())

    def lag_audit(self, b, g_fine, Nhat, m0):
        """Audit G=N_b ifft(g) and exact alias error (method_idea §8.5).

        Accept a full length-N_b DFT sequence or samples on psdgrid.f_k;
        the latter are zero-extended/truncated to positive band FFT bins.
        Lags are unwrapped about the block centre. The centroid and 1e-6,
        1-1e-6 quantiles use |G|²; outside_fraction uses |G|. The error is
        quadrature minus fine sum, without 4/T, and is generally complex.
        """
        n = self._check_block(b, m0, Nhat)
        g = np.asarray(g_fine, dtype=complex)
        if g.shape == (n,):
            g = g.copy()
        elif g.shape == self.psdgrid.f_k.shape:
            original = g
            g = np.zeros(n, complex)
            count = min(len(original), (n - 1) // 2)
            g[1:count + 1] = original[:count]
        else:
            raise ValueError("g_fine must use the full DFT or PSDGrid fine grid")
        if np.any(~np.isfinite(g)):
            raise ValueError("g_fine must be finite")
        G = n * np.fft.ifft(g)
        j = np.arange(n)
        block = (m0 + np.arange(Nhat)) % n
        outside = np.ones(n, bool)
        outside[block] = False
        alias = (m0 + (j - m0) % Nhat) % n
        A = self.A_m[n]
        folding_error = np.sum(G[outside] * (A[alias[outside]].conj()
                                            - A[outside].conj())) / n
        exact = np.sum(self.fine_inverse_psd(n) * g)
        approx = np.sum(self.weights(b, m0, Nhat) * g[::n // Nhat])
        centre = m0 + Nhat / 2
        lag = (centre + (j - centre + n / 2) % n - n / 2) * self.bands.dt_b[b]
        power = np.abs(G)**2
        mass = np.abs(G)
        if power.sum() == 0:
            raise ValueError("Cannot audit an identically zero sequence")
        order = np.argsort(lag)
        cdf = np.cumsum(power[order]) / power.sum()
        q = lag[order[np.searchsorted(cdf, [1e-6, 1 - 1e-6])]]
        return dict(outside_fraction=float(mass[outside].sum() / mass.sum()),
                    lag_centroid=float(np.dot(lag, power) / power.sum()),
                    lag_quantiles=q.tolist(), folding_error=complex(folding_error),
                    direct_error=complex(approx - exact), fine_sum=complex(exact),
                    quadrature_sum=complex(approx))
