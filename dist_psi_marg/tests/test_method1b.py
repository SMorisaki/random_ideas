"""Iteration-007 maps, genuine cusp removal, and independent generator checks."""
import math
from pathlib import Path
import numpy as np
import pytest
from dpmarg import table4d, table1d, params, regimes
from dpmarg.method2 import lnL_marg_m2_table
from dpmarg.reference import lnL_marg_ref

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def radial():
    path=ROOT/'data/table1d_xmin0.1_xmax100.npz'
    return table1d.load_table(path) if path.exists() else table1d.build_table(ns=32,nv=32)


@pytest.mark.parametrize('options',[(1,0,0,0),(2,0,0,0),(0,1,0,0),(0,2,0,0),(0,0,1,0),(0,0,0,1),(1,2,1,1)])
def test_option_roundtrip(options):
    rng=np.random.default_rng(507)
    for z in rng.uniform(.02,.98,(100,4)):
        if options[1]==1:z[1]*=z[0]
        physical=table4d.from_unit(z,1e8,24.,.1,100.,*options)
        got=table4d.to_unit(*physical,1e8,24.,.1,100.,*options)
        # The two-ended stretch at q~1 amplifies float64 spacing in lam2.
        # Compare against that local condition number, not a fictitious
        # absolute 1e-12 guarantee for a represented physical eigenvalue.
        tolerance=np.full(4,2e-11)
        if options[1]==2:
            l1,l2,_,_=physical;q=l2/l1;eps=1/(1+l1*100**2)
            derivative=(1/math.hypot(eps,q)+1/math.hypot(eps,1-q))/(2*math.asinh(1/eps))
            tolerance[1]=max(2e-11,8*np.finfo(float).eps*derivative)
        assert np.all(abs(got-z)<=tolerance),(z,physical,got,tolerance)


@pytest.mark.parametrize('geometry',[0,1])
@pytest.mark.parametrize('mode',[1,2])
def test_transition_monotonicity_and_inverse(geometry,mode):
    for l1,l2,eta in [(0.,0.,0.),(1e4,0.,0.),(1e4,0.,1e-9),(1e8,0.,math.pi/2),(1e8,1e8,.7),(10.,1.,1.5)]:
        cap=table4d.radial_R(24.,l1,l2,eta,.1,100.)
        R=np.r_[0.,np.geomspace(max(1e-12,cap*1e-14),cap,160)]
        ss=np.array([table4d.transition_s(r,l1,l2,eta,.1,100.,mode,geometry)[0] for r in R])
        assert np.all(np.diff(ss)>0)
        for s in np.linspace(ss[0],ss[-1],71):
            r=table4d.transition_R(s,l1,l2,eta,.1,100.,cap,mode,geometry)
            got,d=table4d.transition_s(r,l1,l2,eta,.1,100.,mode,geometry)
            assert d>0
            assert abs(got-s)<max(2e-11,8*np.spacing(r)*d)


@pytest.mark.parametrize('mode',[1,2])
def test_normalization_removes_cusp(mode):
    # Continuity is checked on N itself so the test cannot be satisfied by
    # method-2 integration errors masking an unremoved cusp.
    f0=table4d.normalization_value(1e4,0.,1.,0.,.1,100.,mode)
    for e in [1e-6,1e-7,1e-8]:
        plus=table4d.normalization_value(1e4,0.,1.,e,.1,100.,mode)
        minus=table4d.normalization_value(1e4,0.,1.,-e,.1,100.,mode)
        assert abs(plus-minus)<1e-12
        assert abs((plus-f0)/e)<1e-5
    raw=table4d.normalization_value(1e4,0.,1.,1e-8,.1,100.,mode+2)
    raw0=table4d.normalization_value(1e4,0.,1.,0.,.1,100.,mode+2)
    assert abs((raw-raw0)/1e-8)>9.


def test_defaults_and_fixed_domain():
    for z in np.random.default_rng(508).uniform(.03,.97,(30,4)):
        old=table4d.from_unit(z,1e8,24.,.1,100.)
        assert old==table4d.from_unit(z,1e8,24.,.1,100.,0,0,0,0)
        for mode in [0,1]:
            for geom in [0,1]:
                new=table4d.from_unit(np.array([z[0],z[1],1.,z[3]]),1e8,24.,.1,100.,mode,0,0,geom)
                assert abs(new[2]-table4d.radial_R(24.,new[0],new[1],new[3],.1,100.))<1e-6
    with pytest.raises(ValueError):table4d.from_unit(np.array([.2,.8,.5,.5]),1e8,24.,.1,100.,0,1)


@pytest.mark.parametrize('geometry',[0,1])
def test_q1_coordinates_have_no_angle_layer(geometry):
    values=[table4d.to_unit(1e8,1e8,5.45e6,e,1e8,24.,.1,100.,1,2,0,geometry)[2] for e in np.linspace(0,np.pi/2,33)]
    assert np.ptp(values)<1e-12


CHECKS=regimes.named_cases()[:18]+[
    dict(zip(['A','B','C','D','E'],params.expand_params(l1,0.,19.2047382462,1.24240566613)),
         name=f'narrow_null_axis_{l1:g}',xmin=.1,xmax=100.)
    for l1 in [651154.6134045325,633390.2841571145]]


@pytest.mark.slow
@pytest.mark.parametrize('case',CHECKS,ids=[c['name'] for c in CHECKS])
def test_generator_spot_checks(case,radial):
    """20 brute-force checks include zero, null, huge, outside, and two lobes.

    no_linear_signal has two equal lobes. Named large-lam1 cases and both
    narrow-null counterexamples protect extreme endpoint/degenerate regimes.
    """
    p=[case[k] for k in ['A','B','C','D','E','xmin','xmax']]
    value=lnL_marg_m2_table(*p,radial);truth=lnL_marg_ref(*p)
    assert abs(value-truth)<5e-5
    reduced=params.reduce_params(*p[:5])
    for mode in [0,1,2]:
        remainder=table4d.reduced_value(*reduced,.1,100.,radial,mode)
        assert abs(remainder+table4d.normalization_value(*reduced,.1,100.,mode)-value)<2e-8
