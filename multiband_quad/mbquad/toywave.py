"""Synthetic non-precessing SPA modes following Appendix A of Morisaki (2021)."""

from collections.abc import Mapping

import numpy as np


# G M_sun / c**3 in seconds; also used by notes/orchestrator_toy.py.
MSUN_S = 4.925491025543576e-6
DEFAULT_MODES = {
    (2, 2): (1.0, 0.0, 0.0),
    (2, 1): (0.25, 0.7, 0.0),
    (3, 3): (0.35, 1.3, 0.0),
    (3, 2): (0.10, 2.1, 0.0),
    (4, 4): (0.15, 0.4, 0.0),
}


def _positive_f(f):
    f = np.asarray(f, dtype=float)
    if np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("SPA frequencies must be finite and strictly positive")
    return f


def _mass_seconds(mchirp):
    if not np.isfinite(mchirp) or mchirp <= 0:
        raise ValueError("mchirp must be finite and positive, in solar masses")
    return MSUN_S * mchirp


def tau_0pn(f, mchirp):
    """Eq. (53): tau = (5/256) Mc_s (pi Mc_s f)**(-8/3), in seconds.

    f > 0 is in Hz, mchirp > 0 in solar masses, Mc_s = G Mc/c³. This is the
    Newtonian inspiral time to merger, without spin or merger corrections.
    """
    f = _positive_f(f)
    Mc_s = _mass_seconds(mchirp)
    return (5 / 256) * Mc_s * (np.pi * Mc_s * f)**(-8 / 3)


def dtau_0pn_df(f, mchirp):
    """Derivative of Eq. (53): d tau/df = -(8/3) tau(f)/f, in s/Hz.

    Valid for positive Hz frequencies and positive solar-mass chirp masses.
    """
    f = _positive_f(f)
    return -(8 / 3) * tau_0pn(f, mchirp) / f


def psi_22(f, mchirp, t_c=0.0, phi_c=0.0):
    """Newtonian TaylorF2 SPA phase consistent with Eq. (53) and Appendix A.

    Psi_22 = 2 pi f t_c - phi_c - pi/4 + (3/128)(pi Mc_s f)**(-5/3).
    f > 0 is in Hz, mchirp > 0 in solar masses, t_c in seconds from the start
    of the segment, phi_c in radians. With h(f) ~ exp(-i Psi), the inverse
    FT peaks at t = t_c and dPsi/df / (2 pi) = t_c - tau_0pn(f).
    """
    f = _positive_f(f)
    Mc_s = _mass_seconds(mchirp)
    if not np.isfinite(t_c) or not np.isfinite(phi_c):
        raise ValueError("t_c and phi_c must be finite")
    return (2 * np.pi * f * t_c - phi_c - np.pi / 4
            + (3 / 128) * (np.pi * Mc_s * f)**(-5 / 3))


class ToyModes:
    """Synthetic SPA modes h_lm = A_lm exp(-i Psi_lm), for positive f.

    A_lm = dist_scale * ratio * (f/100)**growth * f**(-7/6), and
    Psi_lm = (m/2) Psi_22(2f/m) + phase_offset (Appendix A).
    Thus t_lm(f) = t_c - tau_0pn(2f/m), and Eq. (55) selects the slowest m.
    These inspiral-only modes model phase interference, not physical merger
    amplitudes or detector projections. Unsorted arrays and scalars work.

    modes maps (l,m) to (ratio, phase_offset in radians, growth). Defaults
    use the five modes and phase offsets of the orchestrator's toy example.
    dist_scale is an overall multiplicative amplitude, default 1e-22.
    Optional f_taper_lm maps modes to (start_Hz, stop_Hz); multiply by
    (1 + cos(pi*(f-start)/(stop-start)))/2 in that interval and zero above
    stop. This cosine taper is off by default.
    """

    def __init__(self, mchirp, t_c=0.0, phi_c=0.0, dist_scale=1e-22,
                 modes=None, f_taper_lm=None):
        _mass_seconds(mchirp)
        if not np.all(np.isfinite([t_c, phi_c, dist_scale])) or dist_scale < 0:
            raise ValueError("t_c, phi_c, dist_scale must be finite, with dist_scale >= 0")
        self.mchirp = float(mchirp)
        self.t_c, self.phi_c, self.dist_scale = float(t_c), float(phi_c), float(dist_scale)
        if modes is not None and not isinstance(modes, Mapping):
            raise ValueError("modes must map (l,m) to (ratio, phase_offset, growth)")
        self.modes = dict(DEFAULT_MODES if modes is None else modes)
        for lm, parameters in self.modes.items():
            if lm not in DEFAULT_MODES:
                raise ValueError(f"Unsupported mode {lm}; choose from {tuple(DEFAULT_MODES)}")
            if (len(parameters) != 3 or not np.all(np.isfinite(parameters))
                    or parameters[0] < 0):
                raise ValueError("Each mode needs finite (ratio >= 0, phase_offset, growth)")
            self.modes[lm] = tuple(parameters)
        self.f_taper_lm = dict(f_taper_lm or {})
        for lm, interval in self.f_taper_lm.items():
            if (lm not in self.modes or len(interval) != 2
                    or not np.all(np.isfinite(interval)) or not 0 < interval[0] < interval[1]):
                raise ValueError("Each taper needs an active mode and 0 < start_Hz < stop_Hz")
            self.f_taper_lm[lm] = tuple(interval)

    def phase(self, f, lm):
        """Return Psi_lm = (m/2) Psi_22(2f/m) + offset on f > 0 (Appendix A)."""
        _, phase_offset, _ = self.modes[lm]
        m = lm[1]
        return (m / 2) * psi_22(2 * _positive_f(f) / m, self.mchirp,
                               self.t_c, self.phi_c) + phase_offset

    def mode(self, f, lm):
        """Evaluate one configured positive-m SPA mode (class formula) at f > 0."""
        f = _positive_f(f)
        ratio, _, growth = self.modes[lm]
        amplitude = self.dist_scale * ratio * (f / 100)**growth * f**(-7 / 6)
        if lm in self.f_taper_lm:
            start, stop = self.f_taper_lm[lm]
            u = np.clip((f - start) / (stop - start), 0, 1)
            amplitude = amplitude * (1 + np.cos(np.pi * u)) / 2
        return amplitude * np.exp(-1j * self.phase(f, lm))

    def total(self, f):
        """Sum the configured SPA modes at arbitrary positive frequencies (Appendix A)."""
        f = _positive_f(f)
        h = np.zeros_like(f, dtype=complex)
        for lm in self.modes:
            h += self.mode(f, lm)
        return h

    def tau_lm(self, f, lm):
        """Appendix A: tau_lm(f) = tau_0pn(2f/m), t_lm = t_c - tau_lm, f > 0."""
        if lm not in self.modes:
            raise KeyError(lm)
        return tau_0pn(2 * _positive_f(f) / lm[1], self.mchirp)
