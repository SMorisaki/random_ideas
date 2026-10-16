"""Regular (s,v) distance tables with cubic interpolation and exact fallbacks.

s=asinh((a-b*xmin)/sigma)+asinh((a-b*xmax)/sigma),
v=asinh(b/bref). The default sigma=hypot(sqrt(b),sigma0) smooths the
max(sqrt(b),sigma0) transition. Mode 2 stores log I minus the log of the
unweighted, peak-shifted Gaussian integral. Unlike mode 1 (the prescribed
frozen-prior Delta), it has no clipped-radius derivative kinks. Mode 0
stores lnI directly. All modes use exactly the same physical coordinates.
"""
import math
from pathlib import Path

import numpy as np
from numba import njit
from scipy.ndimage import spline_filter

from .kernel1d import lnI_direct, fmax_1d


@njit(cache=True)
def _erfcx(z):
    """exp(z*z)*erfc(z), z>=0; asymptotic series above 20 avoids overflow."""
    if z < 20:
        return math.exp(z*z)*math.erfc(z)
    term, total = 1.0, 1.0
    for k in range(1, 14):
        term *= -(2*k-1)/(2*z*z)
        total += term
    return total/(math.sqrt(math.pi)*z)


@njit(cache=True)
def _gaussian_side(d, b, length):
    """Integral_0^length exp(-d*t-b*t*t/2) dt, d,b,length>=0."""
    if length <= 0:
        return 0.0
    drop = d*length+0.5*b*length*length
    if drop < 1e-3:
        # Taylor integrate the polynomial in u=t/length. This also covers
        # a,b -> 0 without a subtraction of almost equal erfc values.
        p, q = d*length, 0.5*b*length*length
        total = 1.0
        factorial = 1.0
        for n in range(1, 5):
            factorial *= n
            inner = 0.0
            choose = 1.0
            for k in range(n+1):
                inner += choose*p**(n-k)*q**k/(n+k+1)
                if k < n:
                    choose *= (n-k)/(k+1)
            total += (-1.0)**n*inner/factorial
        return length*total
    if b == 0:
        return -math.expm1(-d*length)/d
    root = math.sqrt(b/2)
    z = d/(2*root)
    e0 = _erfcx(z)
    e1 = _erfcx(z+root*length)
    # e0-e1*exp(-drop); expm1 preserves a short interval.
    return math.sqrt(math.pi)/(2*root)*((e0-e1)-e1*math.expm1(-drop))


@njit(cache=True)
def ln_gaussian(a, b, xmin, xmax):
    """log integral exp(a*x-b*x*x/2-fmax) dx, finite a and b>=0, 0<xmin<xmax.

    Integrate outwards from the clipped peak using scaled erfc tails.
    No subtraction of large exponents or division by b near b=0 occurs.
    """
    r = min(xmax, max(xmin, a/b)) if b else (xmax if a > 0 else xmin)
    if b>0 and xmin<=a/b<=xmax:
        root=math.sqrt(b/2)
        return (0.5*math.log(math.pi/(2*b))+
                math.log(math.erf(root*(r-xmin))+math.erf(root*(xmax-r))))
    d=abs(a-b*r)
    return math.log(_gaussian_side(d,b,xmax-xmin))


@njit(cache=True)
def lnItilde(a, b, xmin, xmax):
    """log[r^-4 integral exp(f-fmax) dx], r=clipped a/b (linear at b=0)."""
    r = min(xmax,max(xmin,a/b)) if b else (xmax if a > 0 else xmin)
    return ln_gaussian(a,b,xmin,xmax)-4*math.log(r)


@njit(cache=True)
def coordinates(a,b,xmin,xmax,sigma0,bref):
    """Smooth endpoint-slope coordinates for finite a and b>=0."""
    sigma = math.sqrt(b+sigma0*sigma0)
    return (math.asinh((a-b*xmin)/sigma)+math.asinh((a-b*xmax)/sigma),
            math.asinh(b/bref))


@njit(cache=True)
def inverse_s(s,b,xmin,xmax,sigma0):
    """Analytic inverse of s; hypotenuse form avoids squared large slopes."""
    sigma=math.sqrt(b+sigma0*sigma0)
    half=s/2
    z=b*(xmax-xmin)/(2*sigma)
    return b*(xmin+xmax)/2+sigma*math.sinh(half)*math.hypot(1.,z/math.cosh(half))


@njit(cache=True)
def _build(ns,nv,smax,bmax,sigma0,bref,xmin,xmax,mode):
    out=np.empty((ns,nv))
    vmax=math.asinh(bmax/bref)
    for j in range(nv):
        b=bref*math.sinh(vmax*j/(nv-1))
        for i in range(ns):
            a=inverse_s(-smax+2*smax*i/(ns-1),b,xmin,xmax,sigma0)
            value=lnI_direct(a,b,xmin,xmax)
            if mode == 1:
                value-=lnItilde(a,b,xmin,xmax)
            elif mode == 2:
                value-=ln_gaussian(a,b,xmin,xmax)
            out[i,j]=value
    return out


def build_table(xmin=.1,xmax=100.,ns=996,nv=996,smax=24.,bmax=1e8,
                sigma0=None,bref=None,mode=2,interpolation=3,guard=True):
    """Build <=1e6 entries with lnI_direct; return an entirely numba-safe tuple.

    Tuple: (coefficients,s0,ds,v0,dv,sigma0,bref,xmin,xmax,bmax,mode,order,unsafe_cells).
    Cubic coefficients have 2 guard nodes on every edge;
    their storage overhead is included in the entry count. Raw values
    are reconstructed by _build for comparison, rather than kept in RAM.
    bref=1/xmax^2 makes b*xmax^2=1 at v=asinh(1), resolving the weak-b
    transition. The prior-weighted kernel changes much less there.
    """
    if not (0 < xmin < xmax and ns>=8 and nv>=8 and smax>0 and bmax>0):
        raise ValueError('require 0<xmin<xmax, ns,nv>=8, smax,bmax>0')
    if mode not in (0,1,2) or interpolation not in (1,3):
        raise ValueError('mode must be 0,1,2; interpolation must be 1 or 3')
    sigma0=1/(xmax-xmin) if sigma0 is None else float(sigma0)
    bref=1/xmax**2 if bref is None else float(bref)
    if sigma0<=0 or bref<=0:
        raise ValueError('sigma0 and bref must be positive')
    values=_build(ns,nv,smax,bmax,sigma0,bref,xmin,xmax,mode)
    # Polynomial extrapolation avoids imposing a zero derivative on the
    # physical table edges, especially b=0. A long polynomial halo
    # suppresses the prefilter boundary influence before retaining 2 guards.
    if interpolation==3:
        # Prefilter in the extended domain before retaining the guard nodes.
        pad=24
        for axis in (0,1):
            moved=np.moveaxis(values,axis,0)
            left=np.empty((pad,)+moved.shape[1:]); right=left.copy()
            for k in range(1,pad+1):
                left[pad-k]=sum(moved[j]*np.prod([( -k-m)/(j-m) for m in range(4) if m!=j]) for j in range(4))
                right[k-1]=sum(moved[-1-j]*np.prod([( -k-m)/(j-m) for m in range(4) if m!=j]) for j in range(4))
            values=np.moveaxis(np.concatenate((left,moved,right)),0,axis)
        values=np.ascontiguousarray(spline_filter(values,order=3,mode='nearest')[pad-2:-pad+2,pad-2:-pad+2])
    table = (values,-float(smax),2*smax/(ns-1),0.,math.asinh(bmax/bref)/(nv-1),
            sigma0,bref,float(xmin),float(xmax),float(bmax),int(mode),int(interpolation),np.zeros((ns-1,nv-1),dtype=np.bool_))
    if guard:
        _guard_cells(table,2e-6)
    return table


@njit(cache=True,inline='always')
def _weights(t):
    return ((1-t)**3/6,(3*t**3-6*t*t+4)/6,(-3*t**3+3*t*t+3*t+1)/6,t**3/6)


@njit(cache=True)
def _endpoint_series(a,b,lo,hi):
    """Watson endpoint expansion; return NaN when its small parameters fail.

    With inward t and d=|f'(r)|, p=sign/(d*r), q=b/d²,
    I=r^-4/d * sum_n T_n, where T_n are factorial-weighted Taylor
    coefficients of (1+p*z)^-4 exp(-q*z²/2). Use 18 terms only for
    |p|,q<=0.001 and d*(hi-lo)>80+4*log(hi/lo). This deliberately
    conservative asymptotic domain is checked against direct quadrature.
    """
    if a<=b*lo:
        r=lo; d=b*lo-a; sign=1.
    elif a>=b*hi:
        r=hi; d=a-b*hi; sign=-1.
    else:
        return math.nan
    if d<=0:
        return math.nan
    p=sign/(d*r); q=(b/d)/d
    if abs(p)>.001 or q>.001 or d*(hi-lo)<80+4*math.log(hi/lo):
        return math.nan
    total=1.; t1=1.; t2=0.; t3=0.
    for n in range(1,19):
        term=-(n+3)*p*t1-(n-1)*q*t2-(n-1)*(n-2)*p*q*t3
        total+=term
        t1,t2,t3=term,t1,t2
    return -4*math.log(r)-math.log(d)+math.log(total)


@njit(cache=True,inline='always')
def _lookup_count(a,b,table):
    values,s0,ds,v0,dv,sigma0,bref,lo,hi,bmax,mode,order,unsafe=table
    if not (math.isfinite(a) and math.isfinite(b) and b>=0):
        raise ValueError('require finite a,b and b>=0')
    if b>bmax:
        return lnI_direct(a,b,lo,hi),1
    s,v=coordinates(a,b,lo,hi,sigma0,bref)
    ns=values.shape[0]-(4 if order==3 else 0)
    nv=values.shape[1]-(4 if order==3 else 0)
    u=(s-s0)/ds; w=(v-v0)/dv
    if u<0 or u>ns-1 or w<0 or w>nv-1:
        endpoint=_endpoint_series(a,b,lo,hi)
        if math.isfinite(endpoint):
            return endpoint,1
        return lnI_direct(a,b,lo,hi),1
    i=min(int(u),ns-2); j=min(int(w),nv-2)
    if unsafe[i,j]:
        return lnI_direct(a,b,lo,hi),1
    t=u-i; z=w-j
    if order==1:
        result=(1-t)*((1-z)*values[i,j]+z*values[i,j+1])+t*((1-z)*values[i+1,j]+z*values[i+1,j+1])
    else:
        wx=_weights(t); wy=_weights(z)
        result=0.
        for k in range(4):
            row=0.
            for l in range(4):
                row+=wy[l]*values[i+k+1,j+l+1]
            result+=wx[k]*row
    if mode==1:
        result+=lnItilde(a,b,lo,hi)
    elif mode==2:
        result+=ln_gaussian(a,b,lo,hi)
    return result,0


@njit(cache=True,inline='always')
def lnI_table(a,b,table):
    """Peak-shifted distance log integral; exact direct fallback outside table.

    b>=0; table bounds fixed at build time. For b>bmax use lnI_direct.
    For |s|>smax use the conservative endpoint series if applicable, else
    lnI_direct; unsafe cells use lnI_direct. Diagnostics count all three
    paths as fallbacks. The interpolation error bound is empirical.
    """
    return _lookup_count(a,b,table)[0]


@njit(cache=True)
def lnI_table_array(a,b,table):
    """Elementwise lookup for equal one-dimensional arrays, no Python loop."""
    if a.ndim!=1 or b.ndim!=1 or a.size!=b.size:
        raise ValueError('a,b must be equal length one-dimensional arrays')
    result=np.empty(a.size)
    for i in range(a.size):
        result[i]=lnI_table(a[i],b[i],table)
    return result


@njit(cache=True)
def table_diagnostics(a,b,table):
    """Return (values, fallback flags); includes endpoint-series fallbacks."""
    out=np.empty(a.size); flags=np.empty(a.size,dtype=np.int64)
    for i in range(a.size):
        out[i],flags[i]=_lookup_count(a[i],b[i],table)
    return out,flags


def save_table(path,table):
    """Save the coefficient array and scalar metadata, without pickle."""
    np.savez(Path(path),values=table[0],parameters=np.array(table[1:-1]),unsafe=table[-1])


def load_table(path):
    """Read a build_table tuple from .npz, without enabling pickle."""
    with np.load(path,allow_pickle=False) as f:
        p=f['parameters'].tolist()
        p[-2:]=map(int,p[-2:])
        return (np.ascontiguousarray(f['values']),*p,np.ascontiguousarray(f['unsafe']))


@njit(cache=True)
def _guard_cells(table,tolerance):
    """Flag cells whose center/edge probes exceed tolerance (not a proof).

    Runtime uses direct quadrature in flagged cells. A separate independent
    random and physical-node sweep checks the resulting empirical bound.
    """
    values,s0,ds,v0,dv,sigma0,bref,lo,hi,bmax,mode,order,unsafe=table
    flags=unsafe.copy()
    for i in range(unsafe.shape[0]):
        for j in range(unsafe.shape[1]):
            bad=False
            for k in range(5):
                u,z=((.5,.5) if k==0 else (0.,.5) if k==1 else
                     (1.,.5) if k==2 else (.5,0.) if k==3 else (.5,1.))
                b=bref*math.sinh(v0+(j+z)*dv)
                a=inverse_s(s0+(i+u)*ds,b,lo,hi,sigma0)
                error=abs(lnI_table(a,b,table)-lnI_direct(a,b,lo,hi))
                bad=bad or error>tolerance
            flags[i,j]=bad
    unsafe[:,:]=flags


@njit(cache=True)
def _lnJ_pair_table_count(a,b,table):
    """Paired radial integral, same certified sign skip as kernel1d.lnJ_pair."""
    lo,hi=table[7],table[8]
    a=abs(a)
    fp=fmax_1d(a,b,lo,hi); fm=fmax_1d(-a,b,lo,hi)
    positive,fb=_lookup_count(a,b,table)
    positive+=fp
    if a==0:
        return positive+math.log(2.),1,fb
    logvolume=-3*math.log(lo)+math.log1p(-(lo/hi)**3)-math.log(3.)
    if fm-fp < -40 and (2*a*lo>40 or fm+logvolume-positive < -40):
        return positive,1,fb
    negative,fb2=_lookup_count(-a,b,table)
    negative+=fm
    top=max(positive,negative)
    return top+math.log1p(math.exp(min(positive,negative)-top)),2,fb+fb2
