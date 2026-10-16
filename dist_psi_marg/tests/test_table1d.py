"""Analytic Gaussian, guarded interpolation, coverage, and angular regression."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erfcx

from dpmarg.kernel1d import lnI_direct_array
from dpmarg.method2 import lnL_marg_m2_array,lnL_marg_m2_table_array,method2_table_diagnostics
from dpmarg.table1d import (build_table,load_table,save_table,lnItilde,_erfcx,
                            lnI_table_array,table_diagnostics,coordinates,inverse_s,
                            _endpoint_series)
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module',params=[100.,10.])
def table(request):
    # Exercise construction, including guard probes, independently of data/.
    return build_table(.1,request.param)


def scipy_tilde(a,b,lo,hi):
    """Independent scaled-offset SciPy quad; tail below exp(-80)."""
    r=min(hi,max(lo,a/b)) if b else (hi if a>0 else lo)
    d=abs(a-b*r) if r in (lo,hi) else 0.
    integral=0.
    for length in (r-lo,hi-r):
        if length==0:continue
        scale=max(d,math.sqrt(b),1/length)
        upper=min(scale*length,80.)
        integral+=quad(lambda z:math.exp(-(d/scale)*z-.5*(b/scale/scale)*z*z),
                       0,upper,epsabs=1e-13,epsrel=1e-13)[0]/scale
    return math.log(integral)-4*math.log(r)


@pytest.mark.parametrize('z',[0.,.1,1.,10.,19.99,20.,21.,30.,100.,1e8])
def test_erfcx(z):
    assert abs(_erfcx(z)/erfcx(z)-1)<2e-14


@pytest.mark.parametrize('hi',[100.,10.])
def test_tilde(hi):
    rng=np.random.default_rng(53)
    aa=np.r_[0.,1e-12,-1e-12,1e7,-1e7,10**rng.uniform(-10,7,130)*rng.choice([-1,1],130)]
    bb=np.r_[0.,1e-25,1e-20,1e8,1e-30,10**rng.uniform(-20,10,130)]
    bb[20:30]=0
    errors=[abs(lnItilde(a,b,.1,hi)-scipy_tilde(a,b,.1,hi)) for a,b in zip(aa,bb)]
    assert max(errors)<1e-12,max(errors)


def test_accuracy(table):
    rng=np.random.default_rng(54);hi=table[8]
    a=10**rng.uniform(-8,8,300)*rng.choice([-1,1],300)
    b=10**rng.uniform(-12,10,300)
    a[:8]=[0.,0.,1e7,-1e7,1.,-1.,0.,1e4]
    b[:8]=[0.,1e8,0.,0.,0.,0.,1e10,1e8]
    b[8:30]=0
    # Explicitly straddle both clipped-radius curves.
    for j in range(30,60):
        a[j]=b[j]*(.1 if j%2 else hi)+rng.normal()*np.sqrt(b[j])
    value=lnI_table_array(a,b,table);ref=lnI_direct_array(a,b,.1,hi)
    error=abs(value-ref);worst=error.argmax()
    print(f'xmax={hi:g}, worst |Delta lnI|={error[worst]:.9g}, a={a[worst]:g}, b={b[worst]:g}')
    assert np.isfinite(value).all()
    assert error.max()<1e-4


def test_roundtrip_and_storage(table):
    for s in (-23.,-10.,0.,10.,23.):
        for b in (0.,1e-10,1.,1e8):
            a=inverse_s(s,b,table[7],table[8],table[5])
            got,_=coordinates(a,b,table[7],table[8],table[5],table[6])
            assert abs(s-got)<1e-7
    path=ROOT/'data/test_table1d_roundtrip.npz'
    try:
        save_table(path,table);other=load_table(path)
    finally:
        path.unlink(missing_ok=True)
    assert np.array_equal(table[0],other[0]);assert table[1:-1]==other[1:-1]
    assert np.array_equal(table[-1],other[-1])
    assert table[0].size<=1000000


def test_fallback(table):
    a=np.array([-1e9,1e9,1.,0.]);b=np.array([0.,0.,1e10,1e10])
    values,flags=table_diagnostics(a,b,table)
    assert (flags==1).all()
    assert max(abs(values-lnI_direct_array(a,b,table[7],table[8])))<1e-10
    # Guarded cells are separately exercised near a weak-b prior transition.
    if table[8]==100:
        a=np.array([2.920551747195688]);b=np.array([.3106773391503415])
        assert table_diagnostics(a,b,table)[1][0]==1


def test_endpoint_series(table):
    rng=np.random.default_rng(55)
    a=10**rng.uniform(2,10,300)*rng.choice([-1.,1.],300)
    b=10**rng.uniform(-8,9,300)
    reference=lnI_direct_array(a,b,table[7],table[8])
    errs=[]
    for x,y,ref in zip(a,b,reference):
        value=_endpoint_series(x,y,table[7],table[8])
        if np.isfinite(value):errs.append(abs(value-ref))
    assert len(errs)>50
    assert max(errs)<1e-10,max(errs)


def test_method2(table):
    cases=json.loads((ROOT/'data/regression_points.json').read_text())
    p=np.array([[c[k] for k in ('A','B','C','D','E','xmin','xmax')] for c in cases])
    p[:,5]=table[7];p[:,6]=table[8]
    direct=lnL_marg_m2_array(p);look=lnL_marg_m2_table_array(p,table)
    assert np.isfinite(look).all()
    assert max(abs(look-direct))<2e-4
    if table[8]==100.:
        assert max(abs(look-np.array([c['lnL_marg_ref'] for c in cases])))<1e-3
    diag=method2_table_diagnostics(*p[0],table)
    assert 0<=diag[-1]<=diag[1]


def test_invalid():
    with pytest.raises(ValueError):build_table(0.,1.)
    with pytest.raises(ValueError):build_table(mode=5)
