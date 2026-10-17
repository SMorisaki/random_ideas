"""Pairwise method accuracy, mode grouping, priors and API regression."""

from copy import copy
import warnings

import numpy as np
import pytest

from mbquad.coarse import CoarseModes, CoarseWaveform
from mbquad.ifftfft import IFFTFFTHH
from mbquad.pairwise import PairwiseHH
from mbquad.reference import hh_exact


def test_accuracy_and_implementations(toy_case):
    bands, psd, model = toy_case.bands, toy_case.psdgrid, toy_case.model
    modes = list(model.modes)
    h = CoarseModes.from_model(bands, model, modes)
    p = PairwiseHH(bands, psd, modes)
    exact = hh_exact(model.total(psd.f_k), psd)
    value = p(h)
    fft = IFFTFFTHH(bands, psd)(h.total())
    assert abs(value/exact-1) < 1e-6
    assert abs(value-exact) <= 2*abs(fft-exact)
    assert isinstance(value, float)
    assert value == pytest.approx(p(h, impl='einsum'), rel=1e-12)
    mapping = {lm: CoarseWaveform(bands, [x[i] for x in h.h_b]) for i,lm in enumerate(modes)}
    assert value == pytest.approx(p(mapping), rel=1e-12)
    reverse = CoarseModes.from_model(bands, model, modes[::-1])
    assert value == pytest.approx(p(reverse), rel=1e-12)
    assert value == pytest.approx(PairwiseHH(bands, psd, modes[::-1])(reverse), rel=1e-12)
    np.testing.assert_allclose(h.total().h_b[2], model.total(bands.coarse_frequencies()[2].f),
                               rtol=1e-14, atol=0)
    for W in p.W:
        np.testing.assert_array_equal(W, W.swapaxes(0,1).conj())
    centred = 0
    for b, H in enumerate(h.h_b):
        centred += np.dot(p.class_weights[b][0], abs(H.sum(axis=0))**2).real
    centred *= 4/bands.T
    assert abs(centred-exact) > 5*abs(value-exact)
    assert len(p.summary()) == 7*bands.B
    assert 'slack' in p.format_summary()


def test_quadrupole_class_zero(toy_case):
    bands, psd = toy_case.bands, toy_case.psdgrid
    h = CoarseModes.from_model(bands, toy_case.model, [(2,2)])
    p = PairwiseHH(bands, psd, h.modes)
    value = (4/bands.T)*sum(np.dot(w[0], abs(H[0])**2).real
                           for w,H in zip(p.class_weights,h.h_b))
    assert p(h) == pytest.approx(value, rel=1e-14)
    from mbquad.linear import LinearInterpolationHH
    linear = PairwiseHH(bands, psd, h.modes, diag='linear')
    assert linear(h) == pytest.approx(LinearInterpolationHH(bands,psd)(h.total()),rel=1e-14)


def test_prior_anchors_slack_and_validation(toy_case):
    bands, psd, model = toy_case.bands, toy_case.psdgrid, toy_case.model
    with pytest.warns(RuntimeWarning, match='negative slack'):
        p = PairwiseHH(bands, psd, model.modes, mchirp_range=(4,6))
    row = next(r for r in p.summary() if r['b']==1 and r['lag_class']==(4,1))
    assert row['Lambda_min'] == bands.lag_pair(bands.f_b[2],4,1,6)
    assert row['Lambda_max'] == bands.lag_pair(bands.f_b[1]-bands.Df_b[1],4,1,4)
    assert bands.lag_interval_pair(1,4,1)[1] == bands.lag_max_pair(1,4,1)
    with pytest.warns(RuntimeWarning, match='negative slack'):
        clipped = PairwiseHH(bands,psd,model.modes,L_margin=100)
    assert any(r['slack'] < 0 for r in clipped.summary())
    for r in clipped.summary():
        assert r['delta_lo'] >= 0 and r['delta_hi'] >= 0
        assert r['effective_slack'] >= -1e-12
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        uncovered = PairwiseHH(bands,psd,model.modes,mchirp_range=(.1,10))
    assert any(r['effective_slack']<0 for r in uncovered.summary())
    h = CoarseModes.from_model(copy(bands),model,model.modes)
    with pytest.raises(ValueError,match='BandSetup'):
        p(h)
    for kwargs in [dict(margin_units='bad'),dict(diag='bad'),dict(impl='bad'),
                   dict(mchirp_range=(6,4)),dict(L_margin=-1)]:
        with pytest.raises(ValueError):
            PairwiseHH(bands,psd,model.modes,**kwargs)


def test_linear_class_zero_tensor_and_grouped_agreement(toy_case):
    bands, psd, model = toy_case.bands, toy_case.psdgrid, toy_case.model
    h = CoarseModes.from_model(bands, model, model.modes)
    p = PairwiseHH(bands, psd, model.modes, diag='linear')
    assert p(h, impl='pairs') == pytest.approx(p(h, impl='einsum'), rel=1e-12)
    assert all(row['full_grid'] == (row['b'] == 0) for row in p.summary())
