"""Population priors, scalar-cache integrity and independent HH comparisons."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from mbquad.coarse import CoarseModes, CoarseWaveform
from mbquad.lalwave import XHMModel
from mbquad.population import (METHODS, SEED, PopulationCache, chirp_mass, evaluate_point,
    make_bands, make_evaluators, population_lines, population_psd, population_setup, sample_population)
from mbquad.psd import PSDGrid, add_lorentzian_lines, aligo_analytic_psd
from mbquad.reference import hh_exact

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name,strong', [('bns',False),('nsbh',False),('bbh',False),
                                         ('nsbh',True),('bbh',True)])
def test_sampler_determinism_and_priors(name,strong):
    pp=sample_population(name,100,SEED,strong_hom=strong)
    assert pp==sample_population(name,100,SEED,strong_hom=strong)
    assert pp[:5]==sample_population(name,5,SEED,strong_hom=strong)
    assert pp!=sample_population(name,100,SEED+1,strong_hom=strong)
    assert pp.proposals==len(pp)+sum(pp.rejections.values())
    if name=='nsbh':
        assert any(p['m2']/p['m1']<.1 for p in pp)
    cfg=population_setup(name)
    ranges={'bns':(1.2,2.,1.2,2.,.05,.05),
            'nsbh':(15,35,1.5,4,.5,0), 'bbh':(30,80,10,40,.8,.8)}
    a,b,c,d,s1,s2=ranges[name]
    for p in pp:
        assert a<=p['m1']<=b and c<=p['m2']<=d
        assert p['m1']>=p['m2']
        assert abs(p['chi1z'])<=s1 and abs(p['chi2z'])<=s2
        assert chirp_mass(p)>=cfg['mchirp_ref']
        assert cfg['T']-2.12<=p['t_c']<=cfg['T']-1.9
        assert 0<=p['inclination']<=np.pi
        assert 0<=p['phi_ref']<2*np.pi
        assert p['F_plus']**2+p['F_cross']**2==pytest.approx(1,abs=5e-16)
        if strong:
            assert p['m2']/p['m1']<=.15
            if name=='bbh':
                assert p['m2']/p['m1']>=.1
            assert 1.3<p['inclination']<1.8
            assert chirp_mass(p)<=cfg['mchirp_range'][1]
        elif name=='bbh':
            assert p['m2']/p['m1']>=.1


def test_sampler_invalid_arguments():
    for args in [('other',1,SEED),('bns',-1,SEED),('bns',1.5,SEED)]:
        with pytest.raises(ValueError):
            sample_population(*args)
    with pytest.raises(ValueError):
        sample_population('bns',2,SEED,strong_hom=True)
    assert sample_population('bns',0,SEED)==[]


def test_lorentzian_shape_and_width():
    line=add_lorentzian_lines(lambda f:np.full_like(f,2.), [(50,.1,100)])
    f=np.array([[49.9,50.],[50.1,51.]])
    assert line(f).shape==f.shape
    np.testing.assert_allclose(line(f),2*(1+100/(1+((f-50)/.1)**2)))
    assert line(50.)==pytest.approx(202.)
    assert line(50.1)==pytest.approx(102.)
    for spec in [[(0,.1,2)],[(50,0,2)],[(50,.1,-1)],[(50,np.nan,2)]]:
        with pytest.raises(ValueError):
            add_lorentzian_lines(aligo_analytic_psd,spec)
    b=make_bands(population_setup('bns'),5)
    lines=population_lines()
    assert all(.01<=w<=.5 and 100<=h<=10000 for _,w,h in lines)
    for lo,hi in zip(b.f_b[:-1],np.minimum(b.f_b[1:],2048)):
        assert any(lo<=f<=hi for f,_,_ in lines)
    for L in (5,50):
        bb=make_bands(population_setup('bns'),L)
        for f,df in zip(bb.f_b[1:-1],bb.Df_b[1:-1]):
            assert any(abs(centre-f)<df for centre,_,_ in lines)


def test_evaluate_point_and_scalar_cache(monkeypatch):
    cfg=population_setup('nsbh')
    p=population_psd('L1',cfg)
    pars=sample_population('nsbh',1,SEED)[0]
    model=XHMModel(**pars)
    b=make_bands(cfg,5)
    ev=make_evaluators(pars,cfg,p,b)
    exact=hh_exact(model.total_direct(p.f_k),p)
    got=evaluate_point(pars,cfg,p,b,ev,5)
    co=CoarseWaveform.from_model(b,model.total_direct)
    cm=CoarseModes.from_model(b,model,model.modes)
    for m in METHODS:
        direct=(ev[m](cm if m=='pairwise' else co)-exact)/exact
        assert got[m]==pytest.approx(direct,rel=2e-7,abs=2e-14)
    with TemporaryDirectory(dir=ROOT/'data') as d:
        path=Path(d)/'cache.npz'
        cache=PopulationCache(path,p)
        value=cache.get(0,pars)
        assert value==exact
        assert cache.misses==1
        # A fresh object must load exact scalar values without a waveform call.
        second=PopulationCache(path,p)
        def forbidden(*args,**kw):
            raise AssertionError('A cache hit generated a fine waveform')
        monkeypatch.setattr(XHMModel,'total_direct',forbidden)
        assert second.get(0,pars)==value
        assert second.hits==1 and second.misses==0
        cached=evaluate_point(pars,{**cfg,'cache':second,'index':0},p,b,ev,5)
        assert got==cached
        assert second.key(0,pars)!=second.key(0,{**pars,'chi1z':.1})
        assert second.key(0,pars)!=second.key(1,pars)
        changed=PSDGrid(lambda f:2*aligo_analytic_psd(f),**{k:cfg[k] for k in ('T','dt','f_low','f_high')})
        assert PopulationCache(path,changed).key(0,pars)!=second.key(0,pars)
        # L is excluded: a new window partition uses the same fine reference.
        b50=make_bands(cfg,50)
        ev50=make_evaluators(pars,cfg,p,b50)
        evaluate_point(pars,{**cfg,'cache':second,'index':0},p,b50,ev50,50)
        assert second.hits==3
    with pytest.raises(ValueError,match='L must match'):
        evaluate_point(pars,cfg,p,b,ev,50)


@pytest.mark.slow
def test_bns_population_point_both_margins():
    """One real BNS reference and two coarse partitions, not a 100-point run."""
    cfg=population_setup('bns')
    pars=sample_population('bns',1,SEED)[0]
    p=population_psd('lines',cfg)
    exact=hh_exact(XHMModel(**pars).total_direct(p.f_k),p)
    for L in (5,50):
        b=make_bands(cfg,L)
        ev=make_evaluators(pars,cfg,p,b)
        errors=evaluate_point(pars,{**cfg,'exact_hh':exact},p,b,ev,L)
        assert max(abs(v) for v in errors.values())<1e-4
        assert ev['B1'].modes==['exact']+(['B0']*6 if L==50 else ['B1']*6)
