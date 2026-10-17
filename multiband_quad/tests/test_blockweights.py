"""Discrete identities of method_idea §§3,8; include endpoint conventions."""

import numpy as np
import pytest

from mbquad.bands import BandSetup
from mbquad.blockweights import BlockWeights
from mbquad.coarse import CoarseWaveform
from mbquad.ifftfft import IFFTFFTHH
from mbquad.psd import PSDGrid


@pytest.mark.parametrize('extended', [False, True])
def test_full_block_fine_weights(toy_case, extended):
    if extended:
        bands = BandSetup(8, 1/128, 5, 64, [8], 20, 4, 5)
        psd = PSDGrid(lambda f: 2, 8, 1/128, 5, 64)
        assert bands.N_b[0] > bands.N
    else:
        bands, psd = toy_case.bands, toy_case.psdgrid
        assert bands.M_b[0] == bands.N_b[0]
    bw = BlockWeights(bands, psd)
    n = int(bands.N_b[0])
    expected = bw.fine_inverse_psd(n)
    for m0 in [0, -n//2, 13]:
        np.testing.assert_allclose(bw.weights(0, m0, n), expected,
                                   rtol=1e-12, atol=1e-13 * expected.max())
    assert len(bw.A_m) == len(np.unique(bands.N_b))
    assert bw.nbytes() == sum(a.nbytes for a in bw.A_m.values()) + sum(
        a.nbytes for a in bw._weights.values())


def test_reflection_and_cache(toy_case):
    bw = BlockWeights(toy_case.bands, toy_case.psdgrid)
    b, m0 = 3, 123
    nh = int(toy_case.bands.M_b[b])
    w = bw.weights(b, m0, nh)
    # -{m0,...,m0+nh-1} starts at -m0-nh+1, not -m0-nh.
    np.testing.assert_allclose(bw.weights(b, -m0-nh+1, nh), w.conj(),
                               rtol=1e-12, atol=1e-13 * abs(w).max())
    assert bw.weights(b, m0, nh) is w
    k = np.array([0, 4, nh+2])
    np.testing.assert_array_equal(bw.weights(b, m0, nh, k), w[k % nh])
    _, start, end = bw.for_lag_interval(b, 2, 6, nh)
    assert end - start == toy_case.bands.T_b[b]
    assert abs((start + end)/2 - 4) <= toy_case.bands.dt_b[b]/2
    with pytest.raises(ValueError, match='divisor'):
        bw.weights(b, 0, 3)


def test_symmetric_half_point_identity(toy_case):
    """Eq.45 equals §3 on the full interpolated positive+negative spectrum.

    Positive/negative interpolation powers are equal (a real time series).
    Fold their W weights together. Restricting to the original band span
    omits nonzero interpolation tails and is not an exact identity.
    """
    bands, psd = toy_case.bands, toy_case.psdgrid
    bw, fft = BlockWeights(bands, psd), IFFTFFTHH(bands, psd)
    coarse = CoarseWaveform.from_model(bands, toy_case.model.total)
    values = fft.band_contributions(coarse)
    restricted_errors = []
    for b in range(bands.B):
        nh = int(fft.Nhat_b[b])  # min(2M,N_b); band 0 is the full-block limit.
        w = bw.centred(b, nh)
        f, h = fft.positive_grid_waveform(coarse, b)
        k = np.rint(f * fft.That_b[b]).astype(int)
        value = (4/bands.T) * np.dot((w[k] + w[-k % nh]).real, abs(h)**2)
        assert value == pytest.approx(values[b], rel=1e-10)
        f_span = fft.half_point_frequencies(b)
        k_span = np.rint(f_span * fft.That_b[b]).astype(int)
        h_span = fft.half_point_waveform(coarse, b)
        restricted = (4/bands.T) * np.dot(
            (w[k_span] + w[-k_span % nh]).real, abs(h_span)**2)
        restricted_errors.append(abs(restricted/values[b]-1))
    assert max(restricted_errors) < 3e-6
    assert max(restricted_errors) > 1e-6  # document why the span is insufficient.


def test_lag_sign_and_mirrored_failure(toy_case):
    """h_A'=h_A exp(-2pi i f Delta) gives positive IFT lag +Delta.

    A Gaussian envelope isolates the phase slope of a pure (2,2) mode;
    its common SPA phase cancels exactly from conj(h_A) h_A'.
    """
    bands, psd = toy_case.bands, toy_case.psdgrid
    b = 3
    nh = int(bands.M_b[b]); n = int(bands.N_b[b]); dt = bands.dt_b[b]
    f = np.arange(n)/bands.T
    h_A = np.zeros(n, complex)
    active = f > 0
    h_A[active] = (np.exp(-.5*((f[active]-60)/.2)**2)
                   * toy_case.model.mode(f[active], (2, 2)))
    delta = .3 * bands.T_b[b]
    h_Ap = h_A * np.exp(-2j*np.pi*f*delta)
    g = h_A.conj()*h_Ap
    bw = BlockWeights(bands, psd)
    m0 = int(np.rint(delta/dt)) - nh//2
    audit = bw.lag_audit(b, g, nh, m0)
    assert audit['lag_centroid'] == pytest.approx(delta, abs=1e-10)
    assert abs(audit['direct_error']/audit['fine_sum']) < 1e-8
    wrong = bw.lag_audit(b, g, nh, -m0-nh+1)
    assert abs(wrong['direct_error']/wrong['fine_sum']) > 1e-3


def test_folding_error(toy_case):
    bands, psd, model = toy_case.bands, toy_case.psdgrid, toy_case.model
    b = 3
    g = bands.window(psd.f_k, b)*model.mode(psd.f_k, (4,4)).conj()*model.mode(psd.f_k, (2,1))
    nh = int(bands.M_b[b])
    result = BlockWeights(bands, psd).lag_audit(b, g, nh, -nh//2)
    assert result['folding_error'] == pytest.approx(result['direct_error'], rel=1e-10)
    assert 0 < result['outside_fraction'] < 1
    assert result['lag_quantiles'][0] < result['lag_centroid'] < result['lag_quantiles'][1]
