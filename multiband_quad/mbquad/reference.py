"""Brute-force fine-grid likelihood quadratic terms, Eqs. (4) and (26)."""

import numpy as np


def _power(h_fine, psdgrid):
    h = np.asarray(h_fine)
    if h.shape != psdgrid.f_k.shape:
        raise ValueError("Waveform must have exactly the PSDGrid fine-grid shape")
    # Ignore waveform values outside the inverse-PSD support, including NaNs
    # a physical model may use to mark frequencies outside its own domain.
    active = psdgrid.inv_S > 0
    if np.any(~np.isfinite(h[active])):
        raise ValueError("Waveform must be finite where the inverse PSD is nonzero")
    return active, np.abs(h[active])**2 * psdgrid.inv_S[active]


def hh_exact(h_fine, psdgrid):
    """Eq. (4): (h,h) = (4/T) sum_k |h(f_k)|²/S_k, on PSDGrid's grid.

    This one-sided sum excludes DC and Nyquist. h_fine must have the grid's
    shape and finite values wherever inv_S > 0. No waveform approximation
    or band truncation is made beyond the prescribed discrete fine grid.
    """
    _, power = _power(h_fine, psdgrid)
    return float((4 / psdgrid.T) * np.sum(power))


def hh_exact_banded(model_total, psdgrid, bands):
    """Eq. (26): sum_b (4/T) sum_k w^(b)(f_k) |h(f_k)|²/S_k.

    model_total is a callable on the fine frequency array (e.g. model.total),
    evaluated once. The bands and PSDGrid must describe the same segment
    and analysis interval. The Eq. (11) partition of unity makes this equal
    to hh_exact up to floating-point roundoff; no coarse samples are used.
    """
    if any(getattr(psdgrid, key) != getattr(bands, key)
           for key in ("T", "dt", "f_low", "f_high")):
        raise ValueError("PSDGrid and BandSetup must have matching segment and frequency limits")
    active, power = _power(model_total(psdgrid.f_k), psdgrid)
    f = psdgrid.f_k[active]
    return float((4 / psdgrid.T) * sum(
        np.sum(bands.window(f, b) * power) for b in range(bands.B)))


def lnlambda_error(hh_approx, hh_exact, snr):
    """Return snr² (hh_approx - hh_exact)/(2 hh_exact), a signed error diagnostic.

    Eq. (2) contains -(h,h)/2. Rescaling amplitude by snr/sqrt(hh_exact)
    makes the exact norm snr², so the quadratic-term error magnitude in
    ln Lambda is this norm difference divided by two. By the requested
    convention this function returns +Delta(h,h)/2; the signed change in
    ln Lambda itself is its negative if (d,h) is held fixed. Exact norms
    must be positive and finite, approximate norms finite, and SNR >= 0.
    Scalars or broadcast-compatible arrays are accepted.
    """
    approx, exact, snr = np.broadcast_arrays(hh_approx, hh_exact, snr)
    if (np.any(~np.isfinite(exact)) or np.any(exact <= 0)
            or np.any(~np.isfinite(approx)) or np.any(~np.isfinite(snr)) or np.any(snr < 0)):
        raise ValueError("Require finite norms with hh_exact > 0, and finite snr >= 0")
    return snr**2 * (approx - exact) / (2 * exact)
