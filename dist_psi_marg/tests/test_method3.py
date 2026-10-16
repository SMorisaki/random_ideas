"""Independent checks of Taylor coefficients, truncation moments and annuli."""
import math

import numpy as np
import pytest
from scipy.integrate import quad

from dpmarg.method3 import (lnL_marg_m3,lnL_marg_m3_2d,_side_moments,
                            _s_interval,_anchor)
from dpmarg.params import expand_params
from dpmarg.reference import lnL_marg_ref


def interior(q,eta,rho=300.,r=3.):
    h=r/math.hypot(math.cos(eta),math.sin(eta)/q) if q else r
    scale=rho*rho/(h*h*(math.cos(eta)**2+math.sin(eta)**2/q)) if q else (rho/r)**2
    return (*expand_params(scale,q*scale,scale*h,eta),.1,100.)


@pytest.mark.parametrize('order',[2,4])
def test_2d_interior_high_snr(order):
    p=interior(1.,.5)
    assert abs(lnL_marg_m3_2d(*p,order)-lnL_marg_ref(*p))<1e-6


def test_2d_correction_orders_against_smooth_gaussian_expectation():
    # An independent tensor Gauss-Hermite integral checks the coefficients
    # and their asymptotic orders, without the polar reference implementation.
    x,w=np.polynomial.hermite.hermgauss(24)
    errors=[]
    for rho in [30.,60.]:
        l1=(rho/3.)**2;l2=.4*l1;s=2.;t=math.sqrt(5.)
        a=l1*s;bt=l2*t
        p=(*expand_params(l1,l2,math.hypot(a,bt),math.atan2(bt,a)),.1,100.)
        h=((s+math.sqrt(2/l1)*x[:,None])**2+
           (t+math.sqrt(2/l2)*x[None,:])**2)**(-2.5)
        expect=np.dot(w,np.dot(h,w))/math.pi
        norm=math.log(3/(2*math.pi*(.1**-3-100.**-3)))
        truth=norm+.5*(l1*s*s+l2*t*t)+math.log(2*math.pi/math.sqrt(l1*l2))+math.log(expect)
        row=[abs(lnL_marg_m3_2d(*p,o)-truth) for o in [0,2,4]]
        assert row[2]<row[1]<row[0]
        errors.append(row)
    # Doubling curvature scale divides errors by ~4,16,64 respectively.
    assert np.all(np.array(errors[0])/np.array(errors[1])>[3.5,14.,55.])


@pytest.mark.parametrize('q,eta',[(1.,.5),(.1,0.),(.1,math.pi/4),(.001,math.pi/2),(0.,0.)])
@pytest.mark.parametrize('order,tol',[(0,1e-3),(2,1e-3),(4,1e-3)])
def test_1d_high_snr_and_variant_agreement(q,eta,order,tol):
    # The 48-node sweep supports 1e-3 for these high-SNR interior regimes;
    # unlike the 2-D formula, quadrature limits accuracy before 1e-6 here.
    p=interior(q,eta)
    ref=lnL_marg_ref(*p)
    value=lnL_marg_m3(*p,order)
    assert abs(value-ref)<tol
    if q:
        agreement_tol=1e-3
        assert abs(value-lnL_marg_m3_2d(*p,order))<agreement_tol


@pytest.mark.parametrize('pars',[(10.,0.,1.,0.),(10.,0.,1.,math.pi/2),
                                  (20.,3.,0.,0.),(1e8,1e8,5001000.,.3),
                                  (1e4,1e4,1500000.,.4),(0.,0.,.1,.3)])
@pytest.mark.parametrize('order',[0,2,4])
def test_finite_extremes(pars,order):
    p=(*expand_params(*pars),.1,100.)
    assert math.isfinite(lnL_marg_m3(*p,order))


@pytest.mark.parametrize('bounds',[(.1,100.),(.1,10.),(1.,2.)])
def test_all_zero(bounds):
    for order in [0,2,4]:
        assert lnL_marg_m3(0.,0.,0.,0.,0.,*bounds,order)==0.
        assert math.isnan(lnL_marg_m3_2d(0.,0.,0.,0.,0.,*bounds,order))


@pytest.mark.parametrize('pars',[(1.,0.,1.,.5),(100.,100.,1.,.5),
                                  (100.,100.,1e5,.5)])
def test_2d_boundary_rejection(pars):
    assert math.isnan(lnL_marg_m3_2d(*expand_params(*pars),.1,100.))


@pytest.mark.parametrize('b,d,L',[(1.,0.,100.),(1.,3.,100.),(1.,20.,10.),
                                 (1e-8,1.,10.),(0.,0.,1.),(1.,0.,.001)])
def test_truncated_gaussian_moments(b,d,L):
    values=_side_moments(d,b,L)
    for k,value in enumerate(values):
        ref=quad(lambda u:u**k*math.exp(-d*u-b*u*u/2),0,L,epsabs=1e-22,epsrel=1e-11)[0]
        assert abs(value/ref-1)<2e-8


@pytest.mark.parametrize('a,b,t,left,right',[(300.,100.,1.,-10.,10.),
                                           (30.,100.,.1,1.,3.),
                                           (10.,100.,.2,-3.,-1.)])
def test_s_integral_orders(a,b,t,left,right):
    c=min(right,max(left,a/b));shift=c*(a-b*c/2)
    integral=quad(lambda s:math.exp((s-c)*(a-b*c)-.5*b*(s-c)**2)
                  /(s*s+t*t)**2.5,left,right,epsabs=1e-20,epsrel=1e-11)[0]
    ref=shift+math.log(integral)
    err=[abs(_s_interval(a,b,t,left,right,o)-ref) for o in [0,2,4]]
    assert err[2]<err[1]<err[0]
    assert err[2]<2e-5
    mode=_anchor(a,b,t,left,right,c)
    if left<mode<right:
        assert abs(a-b*mode-5*mode/(mode*mode+t*t))<1e-8


def test_rotation_scaling_and_quadrature_convergence():
    pars=(1e4,10.,300.,.7)
    values=[lnL_marg_m3(*expand_params(*pars,xi=xi),.1,100.) for xi in [0.,.4,1.3]]
    assert np.ptp(values)<1e-7
    p=(*expand_params(*pars),.1,100.)
    # High-SNR smooth case: fixed t quadrature is already resolved.
    assert abs(lnL_marg_m3(*p,n_t=48)-lnL_marg_m3(*p,n_t=96))<1e-6
    c=7.
    scaled=(*expand_params(pars[0]/c**2,pars[1]/c**2,pars[2]/c,pars[3]),.1*c,100*c)
    assert abs(lnL_marg_m3(*p)-lnL_marg_m3(*scaled))<1e-7


def test_validation():
    p=interior(1.,.5)
    for fn in [lnL_marg_m3,lnL_marg_m3_2d]:
        with pytest.raises(ValueError):fn(*p,order=3)
        with pytest.raises(ValueError):fn(*p[:5],0.,100.)
        with pytest.raises(ValueError):fn(0.,0.,2.,0.,1.,.1,100.)
    with pytest.raises(ValueError):lnL_marg_m3(*p,n_t=8)
