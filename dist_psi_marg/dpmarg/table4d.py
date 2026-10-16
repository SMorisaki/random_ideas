"""Experimental regular 4-D table of log marginal likelihood minus its maximum.

Axes are v1=asinh(lam1/bref), w=v2/v1, u=(s-s(R=0))/(smax-s(R=0)),
and eta/(pi/2). The weak-axis singularity of the unconstrained peak direction
is regularized by adding sigma0**2 to both eigenvalues ONLY when constructing
the coordinates. The likelihood and its maximum retain the physical eigenvalues.
No interpolation accuracy guarantee is implied; see data/method1_check.json.
Iteration-007 slice options are exposed by from_unit/to_unit and sample_unit:
radial_mode (0=old, 1=regularized three transitions, 2=literal c=co/b), weak_mode (0=ratio, 1=absolute v2,
2=two-ended q stretch), angle_mode (0=eta, 1=cos(2 eta)), geometry_mode
(0=ridge, 1=average ridge/linear-direction coordinates). normalization_value
documents proxy normalization modes. Saved tables and lookup APIs retain the
iteration-005 format and normalization; the new options are slice-only.
"""
import math
from pathlib import Path
import numpy as np
from numba import njit
from .params import reduce_params
from .table1d import coordinates, inverse_s
from .method2 import lnL_marg_m2_table

# Fixed angular proxy rules; exact symmetry-axis zeros avoid spurious
# nonzero weak-eigenvalue penalties from cos(pi/2) roundoff.
_NORMALIZATION_COS=np.cos(2*np.pi*np.arange(128)/128)
_NORMALIZATION_SIN=np.sin(2*np.pi*np.arange(128)/128)
_NORMALIZATION_COS[np.abs(_NORMALIZATION_COS)<1e-15]=0.
_NORMALIZATION_SIN[np.abs(_NORMALIZATION_SIN)<1e-15]=0.


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
def from_unit(z,bmax,smax,lo,hi,radial_mode=0,weak_mode=0,angle_mode=0,geometry_mode=0):
    """Regular box -> (lam1,lam2,R,eta), with the old mapping by default.

    Optional modes are documented in from_unit_options; absolute-v2 uses
    only the physical triangle z1<=z0, rather than the entire unit box.
    """
    if radial_mode or weak_mode or angle_mode or geometry_mode:
        return from_unit_options(z,bmax,smax,lo,hi,radial_mode,weak_mode,angle_mode,geometry_mode)
    vmax=math.asinh(bmax*hi*hi)
    v1=z[0]*vmax; l1=math.sinh(v1)/hi**2
    l2=math.sinh(z[1]*v1)/hi**2; eta=z[3]*math.pi/2
    s0=radial_geometry(l1,l2,eta,lo,hi)[2]
    return l1,l2,radial_R(s0+z[2]*(smax-s0),l1,l2,eta,lo,hi),eta


@njit(cache=True)
def to_unit(l1,l2,R,eta,bmax,smax,lo,hi,radial_mode=0,weak_mode=0,angle_mode=0,geometry_mode=0):
    """Physical coordinates -> unit box (outside coordinates trigger fallback)."""
    if radial_mode or weak_mode or angle_mode or geometry_mode:
        return to_unit_options(l1,l2,R,eta,bmax,smax,lo,hi,radial_mode,weak_mode,angle_mode,geometry_mode)
    v1=math.asinh(l1*hi*hi); v2=math.asinh(l2*hi*hi)
    s0=radial_geometry(l1,l2,eta,lo,hi)[2]
    s=radial_s(R,l1,l2,eta,lo,hi)
    return np.array([v1/math.asinh(bmax*hi*hi),v2/v1 if v1 else 0.,
                     (s-s0)/(smax-s0),eta*2/math.pi])


@njit(cache=True)
def reduced_value(l1,l2,R,eta,lo,hi,radial,normalization=0):
    """F=lnL_marg-N, lam1>=lam2>=0, R>=0, 0<lo<hi.

    N=max lnL by default; optional modes use normalization_value's proxies.
    """
    A=R*math.cos(eta); B=R*math.sin(eta)
    value=lnL_marg_m2_table(A,B,.5*(l1-l2),0.,.5*(l1+l2),lo,hi,radial)
    return value-normalization_value(l1,l2,R,eta,lo,hi,normalization)


@njit(cache=True)
def sample_unit(points,bmax,smax,lo,hi,radial,normalization=0,radial_mode=0,weak_mode=0,angle_mode=0,geometry_mode=0):
    """Evaluate lnL_marg-N at rows of selectable unit coordinates.

    Defaults exactly retain iteration 005; see from_unit_options for domain.
    """
    out=np.empty(len(points))
    for i in range(len(points)):
        l1,l2,R,eta=from_unit(points[i],bmax,smax,lo,hi,radial_mode,weak_mode,angle_mode,geometry_mode)
        out[i]=reduced_value(l1,l2,R,eta,lo,hi,radial,normalization)
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


@njit(cache=True)
def normalization_value(l1,l2,R,eta,lo,hi,mode=0,temperature=1.):
    """Cheap proxy normalization on lam1>=lam2>=0, R>=0, finite annulus.

    0: old maximum. 1: log(pi/64 sum of paired exp(fmax)) on a fixed
    64-angle pi-periodic rule. 2: T log sum exp(fmax/T), 64 full-circle
    angles. Both use SIGNED a (pairing before log) and the C1 upper proxy
    max_r[a*r-(b+2*T/hi²)*r²/2]+T. This bounds each exact radial maximum
    from above within T, including the null-eigenvalue corner. Clipping
    gives continuous first derivatives, not continuous second derivatives.
    The fixed angular rule can miss narrow peaks: it is NOT a certified
    upper bound on the angular integral. No distance-kernel calls occur.
    Modes 3/4 are unregularized versions of 1/2 for cusp diagnostics.
    """
    if mode==0:return lnL_max_numba(l1,l2,R,eta,lo,hi)[0]
    if mode<1 or mode>4 or temperature<=0:raise ValueError('invalid normalization')
    integral=mode==1 or mode==3
    regularized=mode<=2
    n=128 if integral else 64
    T=1. if integral else temperature
    top=-math.inf; total=0.;ce=math.cos(eta);se=math.sin(eta)
    for i in range(n):
        idx=i if integral else 2*i
        ct=_NORMALIZATION_COS[idx];st=_NORMALIZATION_SIN[idx]
        a=R*(ct*ce+st*se)
        b=l2+(l1-l2)*ct*ct
        if regularized:b+=2*T/(hi*hi)
        r=min(hi,max(lo,a/b)) if b>0 else (hi if a>0 else lo)
        f=(r*(a-.5*b*r)+(T if regularized else 0.))/T
        if f>top:
            total=total*math.exp(top-f)+1.;top=f
        else:total+=math.exp(f-top)
    return T*(top+math.log(total))+(math.log(math.pi/64) if integral else 0.)


@njit(cache=True)
def _direction_geometry(l1,l2,eta,lo,hi,linear=False):
    if not linear:
        co,b,_=radial_geometry(l1,l2,eta,lo,hi)
        return co,b
    return 1.,l2+(l1-l2)*math.cos(eta)**2


@njit(cache=True)
def _transition_s_dR(R,l1,l2,eta,lo,hi,mode,linear=False):
    """Return s and ds/dR for the old s or the three-transition sum.

    s3=asinh(R*xmin)+asinh(sigma*(R*c-xmin))+
    asinh(sigma*(R*c-xmax)), sigma²=b+sigma0², c=co/(b+sigma0²).
    The same ridge regularizer in c makes the all-zero corner finite;
    relative change from c=co/b is sigma0²/(b+sigma0²). Mode 2 instead uses
    the literal c=co/b for b>0, with the regularized convention at b=0
    where that formula is undefined. k_a=1.
    Both maps are strictly increasing for R>=0.
    """
    co,b=_direction_geometry(l1,l2,eta,lo,hi,linear)
    sig2=b+1/(hi-lo)**2;sig=math.sqrt(sig2)
    if mode==0:
        a=R*co
        return (math.asinh((a-b*lo)/sig)+math.asinh((a-b*hi)/sig),
                co*(1/math.hypot(sig,a-b*lo)+1/math.hypot(sig,a-b*hi)))
    c=co/(b if mode==2 and b>0 else sig2);x0=R*c
    y=math.asinh(R*lo)+math.asinh(sig*(x0-lo))+math.asinh(sig*(x0-hi))
    d=lo/math.hypot(1.,R*lo)+sig*c*(1/math.hypot(1.,sig*(x0-lo))+1/math.hypot(1.,sig*(x0-hi)))
    return y,d


@njit(cache=True)
def transition_s(R,l1,l2,eta,lo,hi,mode=1,geometry_mode=0):
    """Monotone s3 (mode=1) or old s (0), optionally average two angles.

    geometry_mode=1 averages coordinates from the regularized ridge and
    theta=eta. At q=1 these coincide; no arbitrary eigenvector is chosen.
    """
    y,d=_transition_s_dR(R,l1,l2,eta,lo,hi,mode)
    if geometry_mode:
        y2,d2=_transition_s_dR(R,l1,l2,eta,lo,hi,mode,True)
        return .5*(y+y2),.5*(d+d2)
    return y,d


@njit(cache=True)
def _transition_polish(R,s,l1,l2,eta,lo,hi,Rcap,mode,geometry_mode):
    """Remove log-coordinate rounding with three physical-R Newton steps."""
    for _ in range(3):
        got,d=transition_s(R,l1,l2,eta,lo,hi,mode,geometry_mode)
        R=min(Rcap,max(0.,R+(s-got)/d))
    return R


@njit(cache=True)
def transition_R(s,l1,l2,eta,lo,hi,Rcap,mode=1,geometry_mode=0):
    """Invert a monotone transition sum with bracketed Newton; 0<=R<=Rcap."""
    s0=transition_s(0.,l1,l2,eta,lo,hi,mode,geometry_mode)[0]
    stop=transition_s(Rcap,l1,l2,eta,lo,hi,mode,geometry_mode)[0]
    if s<=s0:return 0.
    if s>=stop:return Rcap
    if mode==2:
        # Literal co/b can span >50 decades near a null eigenvector.
        # Newton in y=asinh(R/scale), bracketed in y, resolves those roots
        # without a linear-R bisection losing its entire iteration budget.
        scale=1/transition_s(0.,l1,l2,eta,lo,hi,mode,geometry_mode)[1]
        ymax=math.asinh(Rcap/scale);left=0.;right=ymax
        y=ymax*(s-s0)/(stop-s0)
        for _ in range(100):
            R=scale*math.sinh(y)
            got,d=transition_s(R,l1,l2,eta,lo,hi,mode,geometry_mode)
            if abs(got-s)<2e-14*max(1.,abs(s)):
                return _transition_polish(R,s,l1,l2,eta,lo,hi,Rcap,mode,geometry_mode)
            if got>s:right=y
            else:left=y
            candidate=y+(s-got)/(d*scale*math.cosh(y))
            if _%2==1 or candidate<=left or candidate>=right:
                candidate=(left+right)/2
            if candidate==y:return _transition_polish(R,s,l1,l2,eta,lo,hi,Rcap,mode,geometry_mode)
            y=candidate
        return _transition_polish(scale*math.sinh(y),s,l1,l2,eta,lo,hi,Rcap,mode,geometry_mode)
    left=0.;right=Rcap;R=Rcap*(s-s0)/(stop-s0)
    for _ in range(80):
        got,d=transition_s(R,l1,l2,eta,lo,hi,mode,geometry_mode)
        if abs(got-s)<2e-14*max(1.,abs(s)):break
        if got>s:right=R
        else:left=R
        candidate=R+(s-got)/d
        if candidate<=left or candidate>=right:candidate=(left+right)/2
        if candidate==R:break
        R=candidate
    return R


@njit(cache=True)
def weak_q_coordinate(q,l1,hi):
    """Two-ended asinh stretch of q in [0,1], eps=1/(1+lam1*xmax²)."""
    eps=1/(1+l1*hi*hi);a=math.asinh(1/eps)
    return (math.asinh(q/eps)-math.asinh((1-q)/eps)+a)/(2*a)


@njit(cache=True)
def weak_q_inverse(w,l1,hi):
    """Analytic inverse of the two-ended stretch, retaining small q precision."""
    if w<=0:return 0.
    if w>=1:return 1.
    eps=1/(1+l1*hi*hi);h=-abs(w-.5)*math.asinh(1/eps)
    small=eps*math.sinh(math.asinh(1/(2*eps*math.cosh(h)))+h)
    return min(1.,max(0.,1-small if w>.5 else small))


@njit(cache=True)
def from_unit_options(z,bmax,smax,lo,hi,radial_mode=0,weak_mode=0,angle_mode=0,geometry_mode=0):
    """Selectable slice coordinates; weak_mode 1 is a valid triangle in a box.

    weak_mode=0: v2/v1; 1: absolute v2/vmax (requires z1<=z0);
    2: two-ended stretched q. angle_mode=1: (1-cos(2*eta))/2.
    All radial choices share the old s<=smax physical R envelope.
    """
    vmax=math.asinh(bmax*hi*hi);v1=z[0]*vmax;l1=math.sinh(v1)/hi**2
    if weak_mode==1:
        if z[1]>z[0]+1e-13:raise ValueError('absolute weak coordinate requires z1<=z0')
        l2=min(l1,math.sinh(z[1]*vmax)/hi**2)
    elif weak_mode==2:l2=l1*weak_q_inverse(z[1],l1,hi)
    else:l2=math.sinh(z[1]*v1)/hi**2
    eta=math.acos(max(-1.,min(1.,1-2*z[3])))/2 if angle_mode else z[3]*math.pi/2
    if not radial_mode and not geometry_mode:
        s0=radial_geometry(l1,l2,eta,lo,hi)[2]
        R=radial_R(s0+z[2]*(smax-s0),l1,l2,eta,lo,hi)
    else:
        cap=radial_R(smax,l1,l2,eta,lo,hi)
        s0=transition_s(0.,l1,l2,eta,lo,hi,radial_mode,geometry_mode)[0]
        stop=transition_s(cap,l1,l2,eta,lo,hi,radial_mode,geometry_mode)[0]
        R=transition_R(s0+z[2]*(stop-s0),l1,l2,eta,lo,hi,cap,radial_mode,geometry_mode)
    return l1,l2,R,eta


@njit(cache=True)
def to_unit_options(l1,l2,R,eta,bmax,smax,lo,hi,radial_mode=0,weak_mode=0,angle_mode=0,geometry_mode=0):
    """Inverse of from_unit_options on the ordered-eigenvalue annular domain."""
    vmax=math.asinh(bmax*hi*hi);v1=math.asinh(l1*hi*hi);v2=math.asinh(l2*hi*hi)
    w=v2/v1 if v1 else 0.
    if weak_mode==1:w=v2/vmax
    elif weak_mode==2:w=weak_q_coordinate(l2/l1 if l1 else 0.,l1,hi)
    if not radial_mode and not geometry_mode:
        s0=radial_geometry(l1,l2,eta,lo,hi)[2]
        u=(radial_s(R,l1,l2,eta,lo,hi)-s0)/(smax-s0)
    else:
        cap=radial_R(smax,l1,l2,eta,lo,hi)
        s0=transition_s(0.,l1,l2,eta,lo,hi,radial_mode,geometry_mode)[0]
        stop=transition_s(cap,l1,l2,eta,lo,hi,radial_mode,geometry_mode)[0]
        u=(transition_s(R,l1,l2,eta,lo,hi,radial_mode,geometry_mode)[0]-s0)/(stop-s0)
    return np.array([v1/vmax,w,u,(1-math.cos(2*eta))/2 if angle_mode else eta*2/math.pi])
