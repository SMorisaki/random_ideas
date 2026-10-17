"""Fused kernels preserve the iteration 001–008 quadratures and C5 API."""

import warnings

import numpy as np
import pytest

from mbquad.baseband import BasebandIFFTFFTHH
from mbquad.coarse import CoarseModes, CoarseWaveform
from mbquad.kernels import flatten_samples, mode_matrix, pairwise_hh
from mbquad.lalwave import XHMCoarseEvaluator
from mbquad.pairwise import PairwiseHH
from mbquad.testcases import toy_small, xhm_case
from mbquad.totalweights import CentredBlockHH


@pytest.fixture(scope='module', params=['toy', 'XHM'])
def samples(request):
    c = toy_small() if request.param == 'toy' else xhm_case('GW190814')
    h = (CoarseModes.from_model(c.bands, c.model, c.model.modes)
         if request.param == 'toy' else XHMCoarseEvaluator(c.model, c.bands).coarse_modes(c.model))
    return c, h


@pytest.mark.parametrize('indices', [[0], [0, 1], [0, 3], [4, 3, 2, 1, 0]])
def test_pairwise_and_matrix(samples, indices):
    c, original = samples
    modes = [original.modes[i] for i in indices]
    h = CoarseModes(c.bands, modes, [H[indices] for H in original.h_b])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        p = PairwiseHH(c.bands, c.psdgrid, modes, impl='numba')
    assert p(h) == pytest.approx(p(h, impl='pairs'), rel=1e-12)
    assert p(h) == pytest.approx(p(h, impl='einsum'), rel=1e-12)
    M = p.mode_matrix(h)
    expected = p.mode_matrix(h, impl='einsum')
    # Near-zero entries need a norm-scaled roundoff allowance.
    np.testing.assert_allclose(M, expected, rtol=1e-12, atol=np.max(abs(expected))*1e-14)
    np.testing.assert_array_equal(M, M.conj().T)
    H = p.flatten(h)
    np.testing.assert_allclose(mode_matrix(H, p.Wc, p.class_map),
                               M*(4/c.bands.T), rtol=1e-12,
                               atol=np.max(abs(M))*(4/c.bands.T)*1e-14)
    assert pairwise_hh(H, p.Wc, p.class_map) == pytest.approx(p(h), rel=1e-12)
    C = np.exp(1j*np.arange(len(modes)))
    projected = CoarseModes(c.bands, modes, [C[:, None]*a for a in h.h_b])
    assert p.hh_from_matrix(M, C) == pytest.approx(p(projected), rel=1e-12)
    mapping = {lm: CoarseWaveform(c.bands, [a[i] for a in h.h_b])
               for i, lm in enumerate(modes)}
    assert p(mapping) == pytest.approx(p(h), rel=1e-12)
    # A copy into reusable storage must not cache stale waveform samples.
    for a in h.h_b:
        a *= 1.3
    assert p(h) == pytest.approx(p(h, impl='pairs'), rel=1e-12)


@pytest.mark.parametrize('half_options', [{}, {'force_half_points': True},
                                        {'pad_factor': 2, 'all_half_bins': True},
                                        {'L_margin': 0}])
def test_total_methods(samples, half_options):
    c, modes = samples
    h = modes.total()
    for cls, options in [(CentredBlockHH, {}), (BasebandIFFTFFTHH, half_options)]:
        a = cls(c.bands, c.psdgrid, **options)
        b = cls(c.bands, c.psdgrid, impl='numba', **options)
        assert b(h) == pytest.approx(a(h), rel=1e-12)
        np.testing.assert_allclose(b.band_contributions(h), a.band_contributions(h),
                                   rtol=1e-12, atol=0)
        for x in h.h_b:
            x *= 1.01
        assert b(h) == pytest.approx(a(h), rel=1e-12)


def test_linear_class_zero_and_validation(toy_case):
    c = toy_case
    h = CoarseModes.from_model(c.bands, c.model, c.model.modes)
    p = PairwiseHH(c.bands, c.psdgrid, h.modes, diag='linear', impl='numba')
    assert p(h) == pytest.approx(p(h, impl='einsum'), rel=1e-12)
    for cls in (CentredBlockHH, BasebandIFFTFFTHH):
        with pytest.raises(ValueError, match='impl'):
            cls(c.bands, c.psdgrid, impl='bad')


@pytest.mark.parametrize('ms', [(2,), (2, 1), (4, 1, 3, 2), (2, 1, 3, 2, 4),
                                 (4, 2, 2, 1, 3, 2)])
def test_arbitrary_class_order_and_group_sizes(ms):
    """Exercise the four-group fast path and general fallback independently."""
    rng = np.random.default_rng(9009)
    n, K = len(ms), 37
    classes = [(s, f) for s in sorted(set(ms)) for f in sorted(set(ms)) if s > f]
    Wc = rng.normal(size=(1+len(classes), K)) + 1j*rng.normal(size=(1+len(classes), K))
    Wc[0] = 10 + abs(Wc[0].real)
    mapping = np.asarray([[0 if mi == mj else 1+classes.index((mi, mj)) if mi > mj else -1
                           for mj in ms] for mi in ms], dtype=np.int64)
    H = rng.normal(size=(n, K)) + 1j*rng.normal(size=(n, K))
    W = np.empty((n, n, K), complex)
    for i in range(n):
        for j in range(n):
            W[i, j] = Wc[mapping[i, j]] if mapping[i, j] >= 0 else Wc[mapping[j, i]].conj()
    M = np.einsum('ik,ijk,jk->ij', H.conj(), W, H)
    assert pairwise_hh(H, Wc, mapping) == pytest.approx(M.sum().real, rel=1e-12)
    np.testing.assert_allclose(mode_matrix(H, Wc, mapping), M, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('shape,axis', [((23,), 0), ((5, 23), 1)])
def test_flatten_views_and_fallback(shape, axis):
    flat = np.arange(np.prod(shape), dtype=complex).reshape(shape).copy()
    parts = np.split(flat, [7, 12], axis=axis)
    out = np.empty_like(flat)
    assert flatten_samples(parts, out, axis) is flat
    flat *= 2
    np.testing.assert_array_equal(flatten_samples(parts, out, axis), flat)
    independent = [p.copy() for p in parts]
    assert flatten_samples(independent, out, axis) is out
    np.testing.assert_array_equal(out, flat)
    # Shared base alone is insufficient: offsets may be reordered.
    reordered = [parts[2], parts[0], parts[1]]
    assert flatten_samples(reordered, out, axis) is out
    np.testing.assert_array_equal(out, np.concatenate(reordered, axis=axis))
