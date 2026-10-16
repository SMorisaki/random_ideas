"""Experimental method-1 geometry, tensor stencils, generator and fallback.

Coarse tables are deliberately inaccurate between nodes over the whole box.
The smooth weak-quadratic test below uses an explicitly measured local bound;
no test treats the experimental production grid as an accurate likelihood.
"""
import json
import math
from pathlib import Path
import numpy as np
import pytest
from dpmarg import table1d,table4d,params,regimes
from dpmarg.method2 import lnL_marg_m2_table
from dpmarg.reference import lnL_marg_ref
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def radial():
    # Build a small independent kernel table when production data are absent.
    path=ROOT/'data/table1d_xmin0.1_xmax100.npz'
    return table1d.load_table(path) if path.exists() else table1d.build_table(ns=32,nv=32)


@pytest.fixture(scope='module')
def coarse(radial):
    return table4d.build_table(radial,(8,8,8,8),bmax=100.,smax=12.)


def test_coordinate_roundtrip():
    rng=np.random.default_rng(515)
    for l1,l2,eta in [(0.,0.,0.),(10.,0.,.7),(1.,.1,0.),(1e8,1e7,.4)]:
        s0=table4d.radial_geometry(l1,l2,eta,.1,100.)[2]
        for u in np.r_[0.,1.,rng.uniform(size=100)]:
            s=s0+u*(24-s0)
            R=table4d.radial_R(s,l1,l2,eta,.1,100.)
            got=table4d.radial_s(R,l1,l2,eta,.1,100.)
            # The stored coordinate is u. Physical s can lose ~3e-11 near
            # xmax at lam1=1e8 because R is represented in float64.
            co,b,_=table4d.radial_geometry(l1,l2,eta,.1,100.)
            sig=math.sqrt(b+1/99.9**2)
            conditioning=8*np.spacing(R)*co*(1/math.hypot(sig,R*co-b*.1)+1/math.hypot(sig,R*co-b*100.))
            assert abs(got-s)<=max(1e-12,conditioning)
            if l1<=10:assert abs(got-s)<1e-12
    for z in rng.uniform(.02,.98,(100,4)):
        physical=table4d.from_unit(z,1e8,24.,.1,100.)
        got=table4d.to_unit(*physical,1e8,24.,.1,100.)
        assert max(abs(z-got))<3e-12


def test_maximum_numba():
    for c in regimes.random_params(20,513):
        p=[c[k] for k in ['A','B','C','D','E']]
        reduced=params.reduce_params(*p)
        want=params.lnL_max(*p,.1,100.)[0]
        value=table4d.lnL_max_numba(*reduced,.1,100.)[0]
        assert abs(value-want)<1e-10


@pytest.mark.parametrize('order',[1,3])
def test_tensor_stencil(order):
    shape=(8,9,10,11)
    axes=np.meshgrid(*(np.linspace(0,1,n) for n in shape),indexing='ij')
    values=(1+axes[0])*(2+axes[1])*(3+axes[2])*(4+axes[3])
    for z in np.random.default_rng(514).uniform(.15,.85,(20,4)):
        assert abs(table4d.interpolate(values,z,order)-np.prod(z+np.arange(1,5)))<1e-12
    # Edge extension should preserve a ramp on axes without even symmetry.
    values=axes[0]+2*axes[1]+3*axes[2]
    for z in [np.array([.001,.999,.99,.01]),np.array([0.,1.,0.,1.])]:
        assert abs(table4d.interpolate(values,z,order)-np.dot(z[:3],[1,2,3]))<1e-12


@pytest.mark.parametrize('order',[1,3])
def test_grid_nodes(coarse,radial,order):
    for index in np.random.default_rng(516).integers(0,8,(30,4)):
        z=index/7
        l1,l2,R,eta=table4d.from_unit(z,100.,12.,.1,100.)
        p=params.expand_params(l1,l2,R,eta)
        value=table4d.lnL_marg_m1(*p,.1,100.,coarse,radial,order)
        truth=lnL_marg_m2_table(*p,.1,100.,radial)
        assert abs(value-truth)<2e-8
    assert table4d.lnL_marg_m1(0.,0.,0.,0.,0.,.1,100.,coarse,radial,order)==0.


@pytest.mark.parametrize('order,tolerance',[(1,.08),(3,.015)])
def test_between_nodes_smooth_domain(radial,order,tolerance):
    # A weak-quadratic subdomain, checked against the complete generator.
    # The main whole-domain slice bound is much larger; this local check
    # verifies useful interpolation accuracy instead of accepting that bound.
    table=table4d.build_table(radial,(8,8,8,8),bmax=1e-4,smax=2.)
    errors=[]
    for z in np.random.default_rng(517).uniform(size=(60,4)):
        l1,l2,R,eta=table4d.from_unit(z,1e-4,2.,.1,100.)
        p=params.expand_params(l1,l2,R,eta)
        value=table4d.lnL_marg_m1(*p,.1,100.,table,radial,order)
        truth=lnL_marg_m2_table(*p,.1,100.,radial)
        errors.append(abs(value-truth))
    assert max(errors)<tolerance


def test_coarse_slice_bound(coarse,radial):
    # Between-node validation along slices of the independently built table.
    # Compare to the measured 8-node full-axis bound from the study, if present.
    path=ROOT/'data/method1_slices.json'
    bound=100.
    if path.exists():
        data=json.loads(path.read_text())
        bound=sum(data['axes'][a]['catmull']['max_error_by_count'][0] for a in ['v1','w','radial','eta'])
    for z in np.random.default_rng(518).uniform(size=(40,4)):
        l1,l2,R,eta=table4d.from_unit(z,100.,12.,.1,100.)
        p=params.expand_params(l1,l2,R,eta)
        got=table4d.lnL_marg_m1(*p,.1,100.,coarse,radial)
        truth=lnL_marg_m2_table(*p,.1,100.,radial)
        assert abs(got-truth)<bound


def test_fallback_and_storage(coarse,radial):
    for l1,l2,R,eta in [(1e9,1e8,300.,.4),(0.,0.,1e9,.7)]:
        p=params.expand_params(l1,l2,R,eta)
        value,flag=table4d.lookup_diagnostics(*p,.1,100.,coarse,radial)
        assert flag==1
        assert value==lnL_marg_m2_table(*p,.1,100.,radial)
    path=ROOT/'data/test_method1_roundtrip.npz'
    try:
        table4d.save_table(path,coarse);loaded=table4d.load_table(path)
        assert np.array_equal(loaded[0],coarse[0]);assert loaded[1:]==coarse[1:]
    finally:path.unlink(missing_ok=True)
    with pytest.raises(ValueError):table4d.lnL_marg_m1(0.,0.,0.,0.,0.,.1,10.,coarse,radial)
    with pytest.raises(ValueError):table4d.build_table(radial,shape=(100,100,100,100))


@pytest.mark.parametrize('l1',[651154.6134045325,633390.2841571145])
def test_generator_narrow_null_axis(radial,l1):
    # Broad golden brackets used to abandon the extremely narrow boundary
    # peak for a much lower prior-driven local maximum (error up to 1700).
    p=params.expand_params(l1,0.,19.2047382462,1.24240566613)
    want=lnL_marg_ref(*p,.1,100.)
    assert abs(lnL_marg_m2_table(*p,.1,100.,radial)-want)<5e-5
