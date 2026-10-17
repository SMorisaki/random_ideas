"""Optional HOM band support rules and measured stationary-phase slack."""
import numpy as np
import pytest

from mbquad.bands import BandSetup
from mbquad.testcases import xhm_systems, toy_small
from mbquad.slack import phase_time_tracks, measured_slack
from mbquad.pairwise import PairwiseHH
from mbquad.baseband import BasebandIFFTFFTHH


def bands(name='BNS', **kwargs):
    p = xhm_systems()[name]
    return BandSetup(**{k:p[k] for k in ('T','dt','f_low','f_high','durations',
                                        'mchirp_ref','m_max','L')}, **kwargs)


@pytest.mark.parametrize('name', ['BNS','GW190814','BBH'])
@pytest.mark.parametrize('lh', [3,5,8])
def test_support_and_order(name, lh):
    p, full, half = [bands(name,rule=r,L_hh=lh) for r in ('paper','hh_full','hh_half')]
    assert p.K_MB <= full.K_MB <= half.K_MB
    for setup in (full, half):
        for b in range(1, setup.B):
            assert setup.hh_extent(setup.f_b[b], setup.f_b[b+1]) <= setup.T_b[b]+1e-10
        assert all(r['K_MB'] == setup.K_MB for r in setup.summary_table())
    if name != 'BNS' and lh <= 5:
        np.testing.assert_array_equal(p.f_b, full.f_b)


def test_nonbinding_and_default_identical():
    for r in ('hh_full','hh_half'):
        # No cross-mode lag and no added margin: Eq. 51 alone suffices.
        p = bands()
        new = bands(rule=r,m_min=4,L_hh=0)
        np.testing.assert_array_equal(new.f_b,p.f_b)
        np.testing.assert_array_equal(new.Df_b,p.Df_b)
        assert new.K_MB == p.K_MB
    np.testing.assert_array_equal(bands().f_b,bands(rule='paper').f_b)


def test_matched_b1_margin_selects_b0():
    # Equality at the support root must not send B1 down its FFT branch
    # because of round-off in the two equivalent sigma expressions.
    from mbquad.testcases import xhm_case
    c = xhm_case('GW190814')
    for margin in (3,5,8):
        b = bands('GW190814',rule='hh_half',L_hh=margin)
        ev = BasebandIFFTFFTHH(b,c.psdgrid,L_margin=margin)
        assert ev.modes == ['exact','B0','B0']


def test_skipped_powers_and_m5():
    p = xhm_systems()['BNS']
    for rule in ('paper','hh_full','hh_half'):
        setup = BandSetup(p['T'],p['dt'],20,2048,[256,64,16,4],1.,5,5,rule=rule)
        assert setup.K_MB > 0
        assert all(setup.N_b % setup.M_b == 0)


@pytest.mark.parametrize('kw', [dict(rule='x'),dict(L_hh=-1),dict(m_min=0),dict(m_min=5)])
def test_bad_rule_arguments(kw):
    with pytest.raises(ValueError):
        bands(**kw)


def test_infeasible_extra_margin_is_reported():
    with pytest.raises(ValueError,match='no feasible support root'):
        bands('GW190814',rule='hh_half',L_hh=1e6)


def test_toy_phase_lag_and_slack():
    c = toy_small()
    f = np.linspace(30,200,100001)
    samples = {lm:c.model.mode(f,lm) for lm in c.model.modes}
    tracks = phase_time_tracks(f,samples,t_c=c.model.t_c)
    for lm,tau in tracks.items():
        np.testing.assert_allclose(tau,c.model.tau_lm(f,lm),rtol=.02)
    pair = PairwiseHH(c.bands,c.psdgrid,c.model.modes)
    placements = [dict(r,f_lo=30.,f_hi=200.) for r in pair.summary() if r['b']==1]
    audit = measured_slack(f,tracks,placements)
    for row in audit:
        cl = row['lag_class']
        expected = (0.,0.) if cl == 0 else tuple(c.bands.lag_pair(x,*cl) for x in (200.,30.))
        np.testing.assert_allclose([row['measured_lo'],row['measured_hi']],expected,rtol=.02,atol=1e-8)
        assert row['slack'] == min(row['slack_lo'],row['slack_hi'])
