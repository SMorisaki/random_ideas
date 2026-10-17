"""One-sided noise PSDs and the fine frequency grid of Eqs. (4) and (6)."""

from pathlib import Path

import numpy as np

from ._validation import frequency_limits, sample_count


def aligo_analytic_psd(f):
    """Return the Ajith (2011) analytic aLIGO design fit, in strain²/Hz.

    For f > 0 in Hz and x = f/215, the fit is
    1e-49 [x**(-4.14) - 5*x**(-2)
            + 111*(1 - x**2 + x**4/2)/(1 + x**2/2)].
    This is a synthetic design sensitivity, rather than a measured PSD.
    Nonpositive frequencies return +inf (zero inverse-noise weight).
    """
    f = np.asarray(f, dtype=float)
    positive = f > 0
    x = np.where(positive, f, 215.0) / 215.0
    S = 1e-49 * (x**-4.14 - 5 * x**-2
                 + 111 * (1 - x**2 + x**4 / 2) / (1 + x**2 / 2))
    return np.where(positive, S, np.inf)


def add_lines(psd_fn, lines):
    """Multiply a PSD by prod_i [1 + h_i exp(-(f-f_i)²/(2 sigma_i²))].

    Each (f_i, sigma_i, h_i) specifies a positive-frequency Gaussian noise
    line in Hz, with sigma_i > 0 and h_i >= 0. The input callable is unmodified.
    This synthetic PSD can be used as S_k in Eq. (6).
    """
    lines = tuple(tuple(line) for line in lines)
    for line in lines:
        if (len(line) != 3 or not np.all(np.isfinite(line))
                or line[0] <= 0 or line[1] <= 0 or line[2] < 0):
            raise ValueError("Lines must be finite (f_i > 0, sigma_i > 0, h_i >= 0)")

    def psd_with_lines(f):
        f = np.asarray(f, dtype=float)
        S = np.array(np.broadcast_to(psd_fn(f), f.shape), dtype=float, copy=True)
        for f_i, sigma_i, h_i in lines:
            S *= 1 + h_i * np.exp(-0.5 * ((f - f_i) / sigma_i)**2)
        return S

    return psd_with_lines


def load_asd_or_psd(path, *, skip_dc=False):
    """Read two columns (Hz, noise) and return a one-sided PSD callable.

    Median log10(noise) > -35 identifies strain/sqrt(Hz) ASD, which is
    squared; smaller magnitudes identify strain²/Hz PSD. This heuristic is
    intended for detector noise near ASD ~1e-23 or PSD ~1e-46, not arbitrary
    units. Frequencies must be unique, finite and positive (order is arbitrary),
    with finite positive noise. Interpolate log(S) linearly in log(f) and
    return +inf outside the tabulated range, as in Eq. (6).

    Optional skip_dc=True discards rows at exactly f=0 before validation,
    for Welch PSD files including DC (unused by Eq. 4). The default retains
    the original strict validation, including rejection of zero frequency.
    """
    values = np.loadtxt(Path(path), ndmin=2)
    if values.shape[1] != 2 or values.shape[0] < 2:
        raise ValueError("Expected at least two rows of exactly two columns: frequency, ASD/PSD")
    if skip_dc:
        values = values[values[:, 0] != 0]
        if len(values) < 2:
            raise ValueError("Expected at least two positive-frequency rows after skipping DC")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Frequencies and ASD/PSD values must be finite and positive")
    values = values[np.argsort(values[:, 0])]
    f, noise = values.T
    if np.any(np.diff(f) <= 0):
        raise ValueError("Tabulated frequencies must be unique")
    log_f = np.log(f)
    log_S = np.log(noise)
    if np.median(np.log10(noise)) > -35:
        log_S = 2 * log_S

    def interpolated_psd(f_query):
        f_query = np.asarray(f_query, dtype=float)
        inside = (f_query >= f[0]) & (f_query <= f[-1])
        safe_f = np.where(inside, f_query, f[0])
        S = np.exp(np.interp(np.log(safe_f), log_f, log_S))
        return np.where(inside, S, np.inf)

    return interpolated_psd


class PSDGrid:
    """Fine grid f_k = k/T, k = 1,...,floor((N-1)/2), and Eq. (6) S_k.

    N = T/dt must be a power of two. DC and Nyquist are excluded. On the
    closed interval [f_low, f_high], S_k comes from psd_fn; elsewhere S_k is
    +inf and inv_S is zero. Only in-range frequencies are passed to psd_fn.
    Three float64 arrays use ~12 MiB for N = 2**20; FFT arrays are not stored.
    """

    def __init__(self, psd_fn, T, dt, f_low, f_high):
        self.N = sample_count(T, dt)
        frequency_limits(f_low, f_high, dt)
        self.T, self.dt = float(T), float(dt)
        self.f_low, self.f_high = float(f_low), float(f_high)
        self.f_k = np.arange(1, self.N // 2, dtype=float) / self.T
        self.S_k = np.full(self.f_k.shape, np.inf)
        active = (self.f_k >= f_low) & (self.f_k <= f_high)
        S = np.broadcast_to(np.asarray(psd_fn(self.f_k[active]), dtype=float),
                            self.f_k[active].shape)
        if np.any(np.isnan(S)) or np.any(S <= 0):
            raise ValueError("PSD must be positive or +inf throughout the analysis interval")
        self.S_k[active] = S
        self.inv_S = 1 / self.S_k

    @property
    def f(self):
        """Alias for f_k, in Hz, without copying the grid."""
        return self.f_k
