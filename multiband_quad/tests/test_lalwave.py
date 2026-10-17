"""LAL conventions and iteration-005 real-waveform regressions (~seconds)."""
from copy import copy
import warnings

import lal
import numpy as np
import pytest

from mbquad.coarse import CoarseModes, CoarseWaveform
from mbquad.baseband import BasebandIFFTFFTHH
from mbquad.ifftfft import IFFTFFTHH
from mbquad.linear import LinearInterpolationHH
from mbquad.pairwise import PairwiseHH, hh_from_matrix
from mbquad.reference import hh_exact
from mbquad.testcases import xhm_case


def pair(case):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return PairwiseHH(case.bands, case.psdgrid, case.model.modes)


@pytest.mark.parametrize('name', ['GW190814', 'BBH'])
@pytest.mark.parametrize('geometry', [(.3,.2,.6,.5), (1.4,1.1,-.3,.8)])
def test_mode_sum_conventions(name, geometry):
    m = xhm_case(name).model
    m.inclination, m.phi_ref, m.F_plus, m.F_cross = geometry
    f = np.random.default_rng(41).uniform(20,1024,2000)
    direct = m.total_direct(f)
    assert np.max(abs(m.total(f)-direct))/np.max(abs(direct)) < 1e-10
    raw, C = m.raw_modes(f), m.coefficients()
    for lm in m.modes:
        np.testing.assert_allclose(m.mode(f,lm), C[lm]*raw[lm]*np.exp(-2j*np.pi*f*m.t_c), rtol=1e-14)
    # Negative-m is the positive-frequency AddMode series in XHM.
    _, seq, _ = m._sequence(f)
    for l, mm in m.modes:
        np.testing.assert_allclose(m._one_mode(seq,(l,mm)),
                                   (-1)**l*m._one_mode(seq,(l,mm),sign=1).conj(), rtol=1e-14)


def test_unsorted_and_reference_frequency():
    m = xhm_case('GW190814').model
    f = np.random.default_rng(4).uniform(20,1024,2000)
    order = np.argsort(f)
    m.f_ref = 50.
    for fn in (m.total, m.total_direct):
        np.testing.assert_array_equal(fn(f)[order], fn(f[order]))
    assert max(abs(m.total(f)-m.total_direct(f)))/max(abs(m.total_direct(f))) < 1e-10
    assert m.total(np.array([70.,70.,30.])).shape == (3,)
    assert np.ndim(m.total(70.)) == 0


def test_time_phase_and_accuracy():
    c = xhm_case('BBH');m,p,b = c.model,c.psdgrid,c.bands
    fine = m.total_direct(p.f_k)
    n = p.N
    spectrum = np.zeros(n,complex)
    spectrum[1:n//2] = fine*np.clip((p.f_k-20)/10,0,1)
    peak = np.argmax(abs(np.fft.ifft(spectrum)))*p.dt
    assert abs(peak-m.t_c) < .010
    exact = hh_exact(fine,p)
    co = CoarseModes.from_model(b,m,m.modes)
    # Measured defaults: pairwise ~1.53e-8, adaptive B1 ~4.31e-7.
    assert abs(pair(c)(co)/exact-1) < 2e-6
    assert abs(BasebandIFFTFFTHH(b,p)(co.total())/exact-1) < 2e-6
    m.t_c += .05
    np.testing.assert_allclose(m.total_direct(p.f_k),fine*np.exp(-2j*np.pi*p.f_k*.05),rtol=2e-10)


def test_linear_worse_than_fft():
    c=xhm_case('GW190814');m,p,b=c.model,c.psdgrid,c.bands
    m.inclination=1.4
    exact=hh_exact(m.total_direct(p.f_k),p)
    co=CoarseWaveform.from_model(b,m.total_direct)
    assert abs(LinearInterpolationHH(b,p)(co)-exact) > abs(IFFTFFTHH(b,p,impl='fast')(co)-exact)


def test_matrix_factorization_and_time_independence():
    c=xhm_case('BBH');m,p,b=c.model,c.psdgrid,c.bands
    f=b.all_coarse_frequencies().f
    evaluator=pair(c)
    def rawco(model):
        raw=model.raw_modes(f)
        return CoarseModes(b,model.modes,np.split(np.array(list(raw.values())),np.cumsum(b.K_b)[:-1],axis=1))
    M=evaluator.mode_matrix(rawco(m))
    C=np.array(list(m.coefficients().values()))
    value=evaluator(CoarseModes.from_model(b,m,m.modes))
    assert evaluator.hh_from_matrix(M,C) == pytest.approx(value,rel=1e-12)
    assert hh_from_matrix(M,C,T=b.T) == pytest.approx(value,rel=1e-12)
    np.testing.assert_array_equal(M,M.conj().T)
    other=copy(m)
    other.t_c=14.1;other.distance_mpc=400.;other.inclination=.4
    other.F_plus=-.2;other.F_cross=.9
    np.testing.assert_array_equal(evaluator.mode_matrix(rawco(other)),M)
    # Pure orbital-phase factorization is a numerical approximation in LAL
    # 6.2.0 (mixed 32 mode), explicitly audited instead of hidden by a tolerance.
    other.phi_ref=1.2
    M2=evaluator.mode_matrix(rawco(other))
    residual=np.max(abs(M2-M))/np.max(abs(M))
    assert residual < 1e-12
    fft=IFFTFFTHH(b,p,impl='fast')
    values=[];ffts=[]
    for t in [b.T-2.1,b.T-1.9]:
        m.t_c=t
        co=CoarseModes.from_model(b,m,m.modes)
        values.append(evaluator(co));ffts.append(fft(co.total()))
    assert abs(values[1]/values[0]-1) < 1e-12
    # Measured IFFT-FFT change ~4.55e-9, pairwise ~roundoff.
    assert abs(ffts[1]/ffts[0]-1) > 1e-9
