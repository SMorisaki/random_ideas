"""Prepared-grid XHM regression and parameter reuse; no large fine grids."""
from copy import copy
import numpy as np
import pytest

from mbquad.lalwave import XHMCoarseEvaluator
from mbquad.coarse import CoarseModes, CoarseWaveform
from mbquad.baseband import BasebandIFFTFFTHH
from mbquad.pairwise import PairwiseHH
from mbquad.testcases import xhm_case


@pytest.mark.parametrize('name', ['BNS', 'GW190814', 'BBH'])
def test_prepared_matches_model_and_reuses_parameters(name):
    c = xhm_case(name)
    m, b = c.model, c.bands
    ev = XHMCoarseEvaluator(m, b)
    other = copy(m)
    other.m1 *= 1.01
    other.chi1z = -.1
    other.inclination = .4
    other.phi_ref = 1.1
    other.t_c += .03
    other.distance_mpc *= 1.3
    other.F_plus = -.2
    for params in (m, other, vars(m)):
        model = ev.model(params)
        got = ev.coarse_modes(params)
        expected = CoarseModes.from_model(b, model, model.modes)
        total = ev.coarse_total(params)
        expected_total = CoarseWaveform.from_model(b, model.total_direct)
        for a, e in zip(got.h_b, expected.h_b):
            np.testing.assert_allclose(a, e, rtol=1e-12, atol=0)
        for a, e in zip(total.h_b, expected_total.h_b):
            np.testing.assert_allclose(a, e, rtol=1e-12, atol=0)
    # Partial dictionaries use the initial snapshot, not the preceding call.
    partial = copy(m)
    partial.inclination = .4
    saved = ev.coarse_total({'inclination': .4})
    ev.coarse_total(other)
    expected = CoarseWaveform.from_model(b, partial.total_direct)
    for a, e in zip(saved.h_b, expected.h_b):
        np.testing.assert_allclose(a, e, rtol=1e-12, atol=0)
    from_dict = XHMCoarseEvaluator(vars(m), b)
    np.testing.assert_array_equal(from_dict.coarse_total().h_b[0], ev.coarse_total(m).h_b[0])
    assert np.max(abs(ev.coarse_total(m).h_b[0]-ev.coarse_total(other).h_b[0])) > 0


def test_explicit_b1_defaults_and_pair_band_sum(toy_case):
    c = toy_case
    co = CoarseWaveform.from_model(c.bands, c.model.total)
    default = BasebandIFFTFFTHH(c.bands, c.psdgrid)
    explicit = BasebandIFFTFFTHH(c.bands, c.psdgrid, pad_factor=1, all_half_bins=False)
    np.testing.assert_array_equal(default.band_contributions(co), explicit.band_contributions(co))
    modes = CoarseModes.from_model(c.bands, c.model, c.model.modes)
    pair = PairwiseHH(c.bands, c.psdgrid, modes.modes)
    assert pair.band_contributions(modes).sum() == pytest.approx(pair(modes), rel=1e-13)


@pytest.mark.parametrize('p', [0, -.1, np.nan, np.inf])
def test_invalid_padding(toy_case, p):
    with pytest.raises(ValueError, match='pad_factor'):
        BasebandIFFTFFTHH(toy_case.bands, toy_case.psdgrid, pad_factor=p)


def test_all_half_bins_matches_explicit_transform(toy_case):
    """Check padding and periodic weight indexing independently of the API."""
    from scipy.fft import fft, ifft, next_fast_len
    c = toy_case
    co = CoarseWaveform.from_model(c.bands, c.model.total)
    ev = BasebandIFFTFFTHH(c.bands, c.psdgrid, pad_factor=2, all_half_bins=True)
    band = ev.band_contributions(co)
    for b, g in enumerate(c.bands.coarse_frequencies()):
        if ev.modes[b] != 'B1':
            continue
        k = next_fast_len(2*len(g.f))
        z = fft(ifft(np.sqrt(g.w)*co.h_b[b], k)*np.exp(-1j*np.pi*np.arange(k)/k))
        W = ev.blockweights.centred(b, 2*int(c.bands.M_b[b]))
        idx = (2*(g.kappa[0]+np.arange(k))+1) % len(W)
        expected = (4/c.bands.T)*(np.dot(W[2*g.kappa].real*g.w, abs(co.h_b[b])**2)
                                    + np.dot(W[idx].real, abs(z)**2))
        assert band[b] == pytest.approx(expected, rel=1e-13)
