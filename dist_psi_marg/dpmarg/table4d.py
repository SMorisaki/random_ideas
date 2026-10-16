"""Experimental regular 4-D table of log marginal likelihood minus its maximum.

Axes are v1=asinh(lam1/bref), w=v2/v1, u=(s-s(R=0))/(smax-s(R=0)),
and eta/(pi/2). The weak-axis singularity of the unconstrained peak direction
is regularized by adding sigma0**2 to both eigenvalues ONLY when constructing
the coordinates. The likelihood and its maximum retain the physical eigenvalues.
No interpolation accuracy guarantee is implied; see data/method1_check.json.
"""
import math
from pathlib import Path
import numpy as np
from numba import njit
from .params import reduce_params
from .table1d import coordinates, inverse_s
from .method2 import lnL_marg_m2_table


@njit(cache=True)
def _peak(t,l1,l2,R,eta,lo,hi):
    a=abs(R*math.cos(t-eta)); b=l2+(l1-l2)*math.cos(t)**2
    r=min(hi,max(lo,a/b)) if b else hi
    return r*(a-.5*b*r),r


@njit(cache=True)
def lnL_max_numba(l1,l2,R,eta,lo,hi):
    """Maximum of R*r*cos(theta-eta)-b(theta)*r²/2 on a finite annulus.

    Analytic interior candidate plus 64-angle scan, special seeds, and golden
    refinement of all local maxima (without the radial prior term). Returns
    (maximum,r,theta) in the canonical eigenbasis, modulo pi for the pair.
    """
    if l1==l2:
        r=min(hi,max(lo,R/l1)) if l1 else hi
        return r*(R-.5*l1*r),r,eta
    best=-math.inf; radius=lo; angle=0.
    if l2>0:
        x=R*math.cos(eta)/l1; y=R*math.sin(eta)/l2
        r=math.hypot(x,y)
        if lo<=r<=hi:
            return .5*(l1*x*x+l2*y*y),r,math.atan2(y,x)
    step=math.pi/64
    for i in range(67):
        t=i*step if i<64 else (math.pi/2 if i==64 else eta if i==65 else 0.)
        f,r=_peak(t,l1,l2,R,eta,lo,hi)
        if f>best: best,radius,angle=f,r,t
        if f<_peak(t-step,l1,l2,R,eta,lo,hi)[0] or f<_peak(t+step,l1,l2,R,eta,lo,hi)[0]:
            continue
        left=t-step; right=t+step; g=.6180339887498949
        x=right-g*(right-left); y=left+g*(right-left)
        fx=_peak(x,l1,l2,R,eta,lo,hi)[0]; fy=_peak(y,l1,l2,R,eta,lo,hi)[0]
        for k in range(64):
            if fx>fy:
                right,y,fy=y,x,fx; x=right-g*(right-left)
                fx=_peak(x,l1,l2,R,eta,lo,hi)[0]
            else:
                left,x,fx=x,y,fy; y=left+g*(right-left)
                fy=_peak(y,l1,l2,R,eta,lo,hi)[0]
        t=x if fx>fy else y; f,r=_peak(t,l1,l2,R,eta,lo,hi)
        if f>best: best,radius,angle=f,r,t
    return best,radius,angle


@njit(cache=True)
def radial_geometry(l1,l2,eta,lo,hi):
    """Return (cos(theta*-eta), b*, s(R=0)) using a smooth ridge direction.

    theta*=arg((cos eta/(lam1+sigma0²),sin eta/(lam2+sigma0²))).
    This agrees with the unconstrained optimum away from the null-eigenvalue
    layer; it is finite at the all-zero corner. Uses table1d's smooth sigma.
    """
    sigma0=1/(hi-lo)
    x=math.cos(eta)/(l1+sigma0*sigma0)
    y=math.sin(eta)/(l2+sigma0*sigma0)
    norm=math.hypot(x,y); x/=norm; y/=norm
    co=x*math.cos(eta)+y*math.sin(eta)
    b=l2+(l1-l2)*x*x
    s0=coordinates(0.,b,lo,hi,sigma0,1/hi**2)[0]
    return co,b,s0


@njit(cache=True)
def radial_s(R,l1,l2,eta,lo,hi):
    """Physical R>=0 to the iteration-004 smooth radial coordinate s."""
    co,b,_=radial_geometry(l1,l2,eta,lo,hi)
    return coordinates(R*co,b,lo,hi,1/(hi-lo),1/hi**2)[0]


@njit(cache=True)
def radial_R(s,l1,l2,eta,lo,hi):
    """Invert s to R using table1d.inverse_s and Newton roundoff correction."""
    co,b,s0=radial_geometry(l1,l2,eta,lo,hi)
    if s<=s0: return 0.
    sig=math.sqrt(b+1/(hi-lo)**2)
    a=max(0.,inverse_s(s,b,lo,hi,1/(hi-lo)))
    for _ in range(3):
        got=coordinates(a,b,lo,hi,1/(hi-lo),1/hi**2)[0]
        derivative=1/math.hypot(sig,a-b*lo)+1/math.hypot(sig,a-b*hi)
        a=max(0.,a+(s-got)/derivative)
    return a/co


@njit(cache=True)
def from_unit(z,bmax,smax,lo,hi):
    """Regular box [0,1]^4 -> physical (lam1,lam2,R,eta)."""
    vmax=math.asinh(bmax*hi*hi)
    v1=z[0]*vmax; l1=math.sinh(v1)/hi**2
    l2=math.sinh(z[1]*v1)/hi**2; eta=z[3]*math.pi/2
    s0=radial_geometry(l1,l2,eta,lo,hi)[2]
    return l1,l2,radial_R(s0+z[2]*(smax-s0),l1,l2,eta,lo,hi),eta


@njit(cache=True)
def to_unit(l1,l2,R,eta,bmax,smax,lo,hi):
    """Physical coordinates -> unit box (outside coordinates trigger fallback)."""
    v1=math.asinh(l1*hi*hi); v2=math.asinh(l2*hi*hi)
    s0=radial_geometry(l1,l2,eta,lo,hi)[2]
    s=radial_s(R,l1,l2,eta,lo,hi)
    return np.array([v1/math.asinh(bmax*hi*hi),v2/v1 if v1 else 0.,
                     (s-s0)/(smax-s0),eta*2/math.pi])


@njit(cache=True)
def reduced_value(l1,l2,R,eta,lo,hi,radial):
    """Fast generator for Ftilde=lnL_marg-max lnL; domain lam1>=lam2>=0."""
    A=R*math.cos(eta); B=R*math.sin(eta)
    value=lnL_marg_m2_table(A,B,.5*(l1-l2),0.,.5*(l1+l2),lo,hi,radial)
    return value-lnL_max_numba(l1,l2,R,eta,lo,hi)[0]


@njit(cache=True)
def sample_unit(points,bmax,smax,lo,hi,radial):
    """Evaluate Ftilde at rows of unit coordinates in a compiled loop."""
    out=np.empty(len(points))
    for i in range(len(points)):
        l1,l2,R,eta=from_unit(points[i],bmax,smax,lo,hi)
        out[i]=reduced_value(l1,l2,R,eta,lo,hi,radial)
    return out


@njit(cache=True)
def _build(values,bmax,smax,lo,hi,radial):
    z=np.empty(4)
    for i in range(values.shape[0]):
        z[0]=i/(values.shape[0]-1)
        for j in range(values.shape[1]):
            z[1]=j/(values.shape[1]-1)
            for k in range(values.shape[2]):
                z[2]=k/(values.shape[2]-1)
                for m in range(values.shape[3]):
                    z[3]=m/(values.shape[3]-1)
                    l1,l2,R,eta=from_unit(z,bmax,smax,lo,hi)
                    values[i,j,k,m]=reduced_value(l1,l2,R,eta,lo,hi,radial)


def build_table(radial,shape=(66,66,66,69),bmax=1e8,smax=24.):
    """Build regular Ftilde grid; tuple holds values,bmax,smax,xmin,xmax.

    This experimental table has no accuracy guard. Memory cap is 2e7 float64
    entries. Timing excludes the caller's explicit compilation warm-up.
    """
    shape=tuple(int(n) for n in shape)
    if len(shape)!=4 or min(shape)<4 or np.prod(shape)>20000000:
        raise ValueError('require four axes >=4 and <=2e7 total entries')
    if not (math.isfinite(bmax) and bmax>0 and math.isfinite(smax) and smax>0):
        raise ValueError('require positive finite bmax,smax')
    values=np.empty(shape)
    _build(values,bmax,smax,radial[7],radial[8],radial)
    return values,float(bmax),float(smax),radial[7],radial[8]


@njit(cache=True)
def weights(t,order):
    """Linear or tensor Catmull-Rom cardinal weights, t in [0,1]."""
    if order==1: return np.array([1-t,t])
    return np.array([-.5*t+t*t-.5*t**3,1-2.5*t*t+1.5*t**3,
                     .5*t+2*t*t-1.5*t**3,-.5*t*t+.5*t**3])


@njit(cache=True)
def stencil(x,n,order,even=False):
    """Cardinal stencil with even eta reflection and linear edge extension.

    Edge extension is represented as effective weights on physical nodes.
    """
    pos=min(n-1.,max(0.,x*(n-1))); cell=min(n-2,int(pos)); t=pos-cell
    w=weights(t,order); indices=np.empty(len(w),dtype=np.int64)
    for j in range(len(w)):
        idx=cell+j-(1 if order==3 else 0)
        if even:
            if idx<0:idx=-idx
            if idx>=n:idx=2*n-2-idx
        indices[j]=idx
    return indices,w


@njit(cache=True)
def interpolate(values,z,order=3):
    """Tensor Catmull-Rom (256 reads) or multilinear (16) on a unit box.

    Boundary stencils use one-sided Catmull-Rom tangents by extrapolating
    one ghost sample linearly. Eta uses even reflection at both boundaries.
    """
    # Linear extrapolation implemented by shifting out-of-range stencil
    # indices onto the first/last cell, with the corresponding weight split.
    idx=np.empty((4,4),dtype=np.int64); ws=np.zeros((4,4))
    for a in range(4):
        ii,ww=stencil(z[a],values.shape[a],order,a==3)
        n=values.shape[a]
        for j in range(len(ii)):
            p=ii[j]
            if p<0:
                idx[a,j]=0; ws[a,j]=2*ww[j]
                # next index is zero; transfer the negative part to index 1
                ws[a,j+2]-=ww[j]
            elif p>=n:
                idx[a,j]=n-1; ws[a,j]+=2*ww[j]
                ws[a,j-2]-=ww[j]
            else:
                idx[a,j]=p; ws[a,j]+=ww[j]
    value=0.; count=2 if order==1 else 4
    for i in range(count):
        for j in range(count):
            for k in range(count):
                for m in range(count):
                    value+=ws[0,i]*ws[1,j]*ws[2,k]*ws[3,m]*values[idx[0,i],idx[1,j],idx[2,k],idx[3,m]]
    return value


@njit(cache=True)
def lookup_diagnostics(A,B,C,D,E,xmin,xmax,table,radial,order=3):
    """Return (lnL_marg, exterior fallback flag), physical finite inputs only."""
    if xmin!=table[3] or xmax!=table[4] or xmin!=radial[7] or xmax!=radial[8]:
        raise ValueError('table bounds must match integration bounds')
    if order!=1 and order!=3: raise ValueError('order must be 1 or 3')
    l1,l2,R,eta=reduce_params(A,B,C,D,E)
    if l1==0 and R==0:return 0.,0
    z=to_unit(l1,l2,R,eta,table[1],table[2],xmin,xmax)
    if np.min(z)<-1e-13 or np.max(z)>1+1e-13:
        return lnL_marg_m2_table(A,B,C,D,E,xmin,xmax,radial),1
    return interpolate(table[0],z,order)+lnL_max_numba(l1,l2,R,eta,xmin,xmax)[0],0


@njit(cache=True)
def lnL_marg_m1(A,B,C,D,E,xmin,xmax,table,radial,order=3):
    """Experimental table likelihood, with method 2 outside the coordinate box."""
    return lookup_diagnostics(A,B,C,D,E,xmin,xmax,table,radial,order)[0]


@njit(cache=True)
def evaluate_array(parameters,table,radial,order=3):
    """Compiled batch returning values and exterior fallback flags."""
    out=np.empty(len(parameters)); flags=np.empty(len(parameters),dtype=np.int64)
    for i in range(len(parameters)):
        p=parameters[i]
        out[i],flags[i]=lookup_diagnostics(p[0],p[1],p[2],p[3],p[4],p[5],p[6],table,radial,order)
    return out,flags


def save_table(path,table):
    """Store raw float64 samples and scalar metadata with pickle disabled."""
    np.savez(Path(path),values=table[0],parameters=np.array(table[1:]))


def load_table(path):
    """Read a table built for one fixed distance prior."""
    with np.load(path,allow_pickle=False) as f:
        return (np.ascontiguousarray(f['values']),*f['parameters'].tolist())
