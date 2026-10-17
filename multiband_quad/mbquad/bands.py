"""Multibanding geometry and cosine windows, Secs. II.B, II.C, II.E, II.F."""

from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq

from ._validation import frequency_limits, sample_count
from .toywave import dtau_0pn_df, tau_0pn


class CoarseGrid(NamedTuple):
    """One band's Eq. (24) arrays: f in Hz, integer kappa, and window w."""

    f: np.ndarray
    kappa: np.ndarray
    w: np.ndarray


class AllCoarseGrid(NamedTuple):
    """Concatenated coarse grids, retaining overlap samples and band indices."""

    f: np.ndarray
    kappa: np.ndarray
    w: np.ndarray
    band_index: np.ndarray


class BandSetup:
    """Construct the paper's bands and coarse grids for a fixed prior.

    Frequencies are in Hz, times in seconds, mchirp_ref in solar masses.
    durations must start at T and strictly decrease through values T/2**j.
    N = T/dt must be a power of two; 0 < f_low < f_high <= Nyquist.
    mchirp_ref is the prior's minimum chirp mass, and m_max its largest
    positive mode m. The Newtonian tau (Eqs. 53, 55) sets internal edges by
    Eq. (51) and widths by Eq. (47); Eq. (56) sets the final upper taper.

    T_b, N_b, dt_b, M_b, K_s, K_e and K_b have B entries; f_b and Df_b have
    B+1 entries, including the final edge f_high + Df_b[B]. N_b is not capped
    at the data's N: Eq. (13) must also contain this extended upper taper.
    K_orig follows Eq. (8), even if f_high is exactly Nyquist (which the
    actual fine grid excludes). K_MB counts overlaps, as in Eq. (25).
    """

    def __init__(self, T, dt, f_low, f_high, durations, mchirp_ref,
                 m_max, L, tc_min_minus_T=-2.12, T_minus_tc_max=1.9):
        self.N = sample_count(T, dt)
        frequency_limits(f_low, f_high, dt)
        self.T, self.dt = float(T), float(dt)
        self.f_low, self.f_high = float(f_low), float(f_high)
        self.T_b = np.asarray(durations, dtype=float).copy()
        if (self.T_b.ndim != 1 or self.T_b.size == 0
                or np.any(~np.isfinite(self.T_b)) or np.any(self.T_b <= 0)):
            raise ValueError("durations must be a nonempty 1D list of finite positive times")
        j = np.log2(self.T / self.T_b)
        if (self.T_b[0] != self.T or np.any(np.diff(self.T_b) >= 0)
                or np.any(j < 0) or not np.allclose(j, np.rint(j), rtol=0, atol=1e-12)):
            raise ValueError("durations must start at T and decrease through T/2**j, integer j >= 0")
        if not np.isfinite(mchirp_ref) or mchirp_ref <= 0:
            raise ValueError("mchirp_ref must be finite and positive (solar masses)")
        if not np.isfinite(m_max) or m_max < 1 or int(m_max) != m_max:
            raise ValueError("m_max must be a positive integer")
        if not np.isfinite(L) or L <= 0:
            raise ValueError("L must be finite and positive")
        if not np.isfinite(tc_min_minus_T):
            raise ValueError("tc_min_minus_T must be finite")
        if not np.isfinite(T_minus_tc_max) or T_minus_tc_max <= 0:
            raise ValueError("T_minus_tc_max must be finite and positive")
        self.mchirp_ref, self.m_max, self.L = float(mchirp_ref), int(m_max), float(L)
        self.tc_min_minus_T = float(tc_min_minus_T)
        self.T_minus_tc_max = float(T_minus_tc_max)
        self.B = len(self.T_b)
        self.f_b = np.empty(self.B + 1)
        self.Df_b = np.empty(self.B + 1)
        self.f_b[0], self.Df_b[0] = self.f_low, 0.0
        for b in range(1, self.B):
            target = self.T_b[b] + self.tc_min_minus_T
            if target <= 0:
                raise ValueError(f"Band {b}: Eq. (51) has no positive root: T_b + tc_min_minus_T <= 0")

            def residual(f):
                return self.tau(f) + self.L * np.sqrt(-self.dtau_df(f)) - target

            if residual(self.f_low) <= 0:
                raise ValueError(f"Band {b}: Eq. (51) root is at or below f_low; use shorter durations")
            upper = self.f_high
            while residual(upper) > 0:
                upper *= 2
            self.f_b[b] = brentq(residual, self.f_low, upper, xtol=1e-12,
                                 rtol=4 * np.finfo(float).eps)
            self.Df_b[b] = 1 / np.sqrt(-self.dtau_df(self.f_b[b]))
        self.Df_b[-1] = 100 / self.T_minus_tc_max
        self.f_b[-1] = self.f_high + self.Df_b[-1]
        for b in range(self.B):
            if not self.f_b[b] < self.f_b[b + 1] - self.Df_b[b + 1]:
                raise ValueError(
                    f"Band {b}: overlapping transitions violate f^(b) < f^(b+1) - Df^(b+1) "
                    f"({self.f_b[b]:.9g} >= {self.f_b[b + 1] - self.Df_b[b + 1]:.9g} Hz); "
                    "adjust durations, L, or frequency limits")
        N_b = []
        for f_hi in self.f_b[1:]:
            n = 2
            while (n - 1) // 2 < f_hi * self.T:
                n *= 2
            N_b.append(n)
        self.N_b = np.asarray(N_b, dtype=np.int64)
        self.dt_b = self.T / self.N_b
        M = self.T_b / self.dt_b
        if np.any(M < 1) or not np.allclose(M, np.rint(M), rtol=0, atol=1e-10):
            raise ValueError("M^(b) = T^(b)/dt^(b) must be a positive integer in every band")
        self.M_b = np.rint(M).astype(np.int64)
        self.K_s = np.ceil((self.f_b[:-1] - self.Df_b[:-1]) * self.T_b).astype(np.int64)
        self.K_e = np.floor(self.f_b[1:] * self.T_b).astype(np.int64)
        self.K_b = self.K_e - self.K_s + 1
        if np.any(self.K_b <= 0):
            raise ValueError("Each band must contain at least one coarse-grid frequency (K_s <= K_e)")
        self.K_MB = int(np.sum(self.K_b))
        self.K_orig = int(np.floor(self.f_high * self.T) - np.ceil(self.f_low * self.T) + 1)

    def tau(self, f):
        """Eqs. (53), (55): tau_0pn(2 f/m_max, mchirp_ref), for positive f in Hz."""
        return tau_0pn(2 * np.asarray(f, dtype=float) / self.m_max, self.mchirp_ref)

    def dtau_df(self, f):
        """Derivative of Eq. (55), including the chain factor 2/m_max, f > 0."""
        return ((2 / self.m_max)
                * dtau_0pn_df(2 * np.asarray(f, dtype=float) / self.m_max, self.mchirp_ref))

    def _check_band(self, b):
        if not isinstance(b, (int, np.integer)) or not 0 <= b < self.B:
            raise IndexError(f"Band index must be an integer in [0, {self.B})")

    def window(self, f, b):
        """Eq. (9) cosine window w^(b)(f), for arbitrary real Hz arrays/scalars.

        The support is [f_b[b] - Df_b[b], f_b[b+1]]; for b = 0 the lower
        edge is a hard boundary at f_low. Sum_b w^(b) = 1 on the analysis
        interval (Eq. 11). Nonzero upper taper values beyond f_high are
        intentional; Eq. (6) removes them from the likelihood.
        """
        self._check_band(b)
        f = np.asarray(f, dtype=float)
        lo, hi = self.f_b[b:b + 2]
        dlo, dhi = self.Df_b[b:b + 2]
        w = np.zeros_like(f)
        if dlo > 0:
            rising = (f > lo - dlo) & (f < lo)
            w[rising] = (1 + np.cos(np.pi * (f[rising] - lo) / dlo)) / 2
        plateau = (f >= lo) & (f <= hi - dhi)
        w[plateau] = 1
        falling = (f > hi - dhi) & (f < hi)
        w[falling] = (1 - np.cos(np.pi * (f[falling] - hi) / dhi)) / 2
        return w

    def sqrt_window(self, f, b):
        """Return sqrt(w^(b)(f)), Eq. (35), on arbitrary real Hz frequencies."""
        return np.sqrt(self.window(f, b))

    def coarse_frequencies(self):
        """Return one CoarseGrid(f, kappa, w) per band (Eqs. 24–25).

        kappa runs inclusively from K_s to K_e; f = kappa/T_b. Boundary
        samples are retained even when their window is zero.
        """
        grids = []
        for b in range(self.B):
            kappa = np.arange(self.K_s[b], self.K_e[b] + 1)
            f = kappa / self.T_b[b]
            grids.append(CoarseGrid(f, kappa, self.window(f, b)))
        return grids

    def all_coarse_frequencies(self):
        """Concatenate coarse grids and band_index, with K_MB entries (Eq. 25).

        The result is an AllCoarseGrid(f, kappa, w, band_index). A waveform
        may be evaluated once on result.f; slices of length K_b recover the
        per-band arrays. Overlap frequencies remain separate entries.
        """
        grids = self.coarse_frequencies()
        return AllCoarseGrid(*(np.concatenate([grid[i] for grid in grids]) for i in range(3)),
                             np.repeat(np.arange(self.B), self.K_b))

    def summary_table(self):
        """Return per-band geometry (Eqs. 9, 13, 18, 24–25); f_lo/hi are edges.

        The full support starts at f_lo - Df_lo and ends at f_hi.
        """
        return [dict(T_b=float(self.T_b[b]), f_lo=float(self.f_b[b]),
                     f_hi=float(self.f_b[b + 1]), Df_lo=float(self.Df_b[b]),
                     Df_hi=float(self.Df_b[b + 1]), N_b=int(self.N_b[b]),
                     M_b=int(self.M_b[b]), K_s=int(self.K_s[b]),
                     K_e=int(self.K_e[b]), K_b=int(self.K_b[b])) for b in range(self.B)]

    def format_summary(self):
        """Format summary_table() as Markdown, with Hz edges and second durations."""
        columns = ["b", "T_b", "f_lo", "f_hi", "Df_lo", "Df_hi", "N_b", "M_b", "K_s", "K_e", "K_b"]
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for b, row in enumerate(self.summary_table()):
            cells = [str(b)] + [f"{value:.6f}" if isinstance(value, float) else str(value)
                                for value in row.values()]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def print_summary(self, file=None):
        """Print the Markdown geometry table to file (stdout by default)."""
        print(self.format_summary(), file=file)

    def lag_max_pair(self, b, m_slow, m_fast):
        """Predict largest SPA lag tau_slow - tau_fast at the band's support start.

        By Appendix A and Eqs. (53), (55), this equals
        tau_0pn(2 f_lo/m_slow) - tau_0pn(2 f_lo/m_fast), with
        f_lo = f_b[b] - Df_b[b], or f_low for b = 0. Positive integer mode
        indices must satisfy m_slow >= m_fast; mchirp_ref sets the timescale.
        This is a Newtonian prediction, not a bound on window-induced tails.
        """
        self._check_band(b)
        if (not np.all(np.isfinite([m_slow, m_fast]))
                or not m_slow >= m_fast >= 1
                or int(m_slow) != m_slow or int(m_fast) != m_fast):
            raise ValueError("Require positive integer m_slow >= m_fast")
        f_lo = self.f_b[b] - self.Df_b[b]
        return float(tau_0pn(2 * f_lo / m_slow, self.mchirp_ref)
                     - tau_0pn(2 * f_lo / m_fast, self.mchirp_ref))

    def lag_pair(self, f, m_slow, m_fast, mchirp=None):
        """Positive SPA lag tau_slow-tau_fast, method_idea §8.1, f > 0.

        Uses Newtonian Eqs. (53), (55); m_slow >= m_fast >= 1 are integers.
        Optional chirp mass is in solar masses; default is mchirp_ref.
        """
        if (not np.all(np.isfinite([m_slow, m_fast]))
                or not m_slow >= m_fast >= 1
                or int(m_slow) != m_slow or int(m_fast) != m_fast):
            raise ValueError("Require positive integer m_slow >= m_fast")
        mc = self.mchirp_ref if mchirp is None else mchirp
        return (tau_0pn(2 * np.asarray(f) / m_slow, mc)
                - tau_0pn(2 * np.asarray(f) / m_fast, mc))

    def lag_interval_pair(self, b, m_slow, m_fast, mchirp=None):
        """Return [Lambda(f_hi),Lambda(f_lo)] in seconds, method_idea §8.3.

        f_lo includes the lower transition; f_hi includes the upper taper.
        These Newtonian anchors omit finite-window tails and PN/spin effects.
        """
        self._check_band(b)
        f_lo = self.f_b[b] - self.Df_b[b]
        return tuple(float(self.lag_pair(f, m_slow, m_fast, mchirp))
                     for f in (self.f_b[b + 1], f_lo))
