"""Annulus-aware Laplace approximations, with no small-radius regulator.

The public APIs use A,B,C,D,E,xmin,xmax, as does method2. Orders 0,2,4
mean Taylor degree of h=r**-5 (including odd terms for truncated normals).
These are asymptotic approximations, not accuracy-certified quadratures.
"""
import math

import numpy as np
from numba import njit

from .params import reduce_params
from .table1d import _gaussian_side

GX24, GW24 = np.polynomial.legendre.leggauss(24)
# Precompute variable panel sizes: allocating a fixed total budget must not
# solve for Legendre roots during each likelihood call.
RULE_X=np.zeros((129,128));RULE_W=np.zeros((129,128))
for _n in range(1,129):
    RULE_X[_n,:_n],RULE_W[_n,:_n]=np.polynomial.legendre.leggauss(_n)


@njit(cache=True)
def _logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    return max(a,b)+math.log1p(math.exp(-abs(a-b)))


@njit(cache=True)
def _validate(A,B,C,D,E,xmin,xmax,order):
    if order != 0 and order != 2 and order != 4:
        raise ValueError('order must be 0, 2 or 4')
    if not (math.isfinite(xmin) and math.isfinite(xmax) and 0 < xmin < xmax):
        raise ValueError('require finite 0<xmin<xmax')
    return reduce_params(A,B,C,D,E)


@njit(cache=True)
def lnL_marg_m3_2d(A,B,C,D,E,xmin,xmax,order=4):
    """Interior 2-D Laplace integral of exp(f)*h, h=(s²+t²)**(-5/2).

    With u=sbar²+tbar², v=sbar²/u, w=tbar²/u, z_i=1/(lam_i*u),
    C2=[(-5+35v)z1+(-5+35w)z2]/2,
    C4=[(105-1890v+3465v²)z1²+(105-1890w+3465w²)z2²]/8
       +(-280+3465vw)z1*z2/4.
    Return norm+fpeak+log(2pi/sqrt(lam1*lam2))-5log(rbar)
    +log(1+C2+C4), retaining terms through the requested order.
    This follows from E[h]/h = 1 + h_ii/(2 lam_i h)
    + h_iiii/(8 lam_i² h) + h_sstt/(4 lam1 lam2 h).
    In particular h_ii/h = g_ii+g_i² for g=-5log(r).

    Domain: physical finite coefficients, 0<xmin<xmax, order=0,2,4.
    Returns NaN for lam2=0, a peak outside the annulus, a nonpositive
    correction, or a peak within 3/sqrt(lam2) of either circular boundary.
    This conservative guard contains the entire three-sigma ellipse; it is
    a geometric applicability check, not a 1e-3 error guarantee.
    """
    l1,l2,R,eta=_validate(A,B,C,D,E,xmin,xmax,order)
    if l2 == 0:
        return math.nan
    s=R*math.cos(eta)/l1
    t=R*math.sin(eta)/l2
    r=math.hypot(s,t)
    if r-xmin <= 3/math.sqrt(l2) or xmax-r <= 3/math.sqrt(l2):
        return math.nan
    u=r*r;v=s*s/u;w=t*t/u;z1=1/(l1*u);z2=1/(l2*u)
    correction=1.
    if order >= 2:
        correction+=.5*((-5+35*v)*z1+(-5+35*w)*z2)
    if order >= 4:
        correction+=((105-1890*v+3465*v*v)*z1*z1+
                     (105-1890*w+3465*w*w)*z2*z2)/8
        correction+=(-280+3465*v*w)*z1*z2/4
    if correction <= 0:
        return math.nan
    norm=math.log(3/(2*math.pi))+3*math.log(xmin)-math.log1p(-(xmin/xmax)**3)
    return (norm+.5*(l1*s*s+l2*t*t)+math.log(2*math.pi)
            -.5*(math.log(l1)+math.log(l2))-5*math.log(r)+math.log(correction))


@njit(cache=True)
def _side_moments(d,b,length):
    """M[k]=int_0^length u^k exp(-d*u-b*u²/2)du, k=0..4.

    M0 uses scaled erfcx (or expm1 at b=0). Integration by parts gives
    b M1=1-exp(-drop)-d M0;
    b Mk=(k-1)M[k-2]-d M[k-1]-length**(k-1)exp(-drop).
    For short intervals or far tails this recurrence subtracts nearly equal
    terms. There we evaluate just these Gaussian polynomial moments using
    24-point GL after scaling by 1/(d+sqrt(b)), cutting only at log-drop 48.
    No r^-5 factor is numerically integrated in s.
    """
    out=np.zeros(5)
    if length <= 0:
        return out
    out[0]=_gaussian_side(d,b,length)
    drop=d*length+.5*b*length*length
    if b > 0 and d <= 3*math.sqrt(b) and drop >= 2:
        e=math.exp(-drop)
        out[1]=(-math.expm1(-drop)-d*out[0])/b
        for k in range(2,5):
            out[k]=((k-1)*out[k-2]-d*out[k-1]-length**(k-1)*e)/b
    else:
        width=1/max(d+math.sqrt(b),1/length)
        limit=length/width
        dd=d*width;bb=b*width*width
        cut=96/(dd+math.sqrt(dd*dd+96*bb)) if dd+bb > 0 else limit
        limit=min(limit,cut)
        for j in range(24):
            v=.5*limit*(GX24[j]+1)
            weight=.5*limit*GW24[j]*math.exp(-dd*v-.5*bb*v*v)*width
            u=width*v
            for k in range(1,5):
                out[k]+=weight*u**k
    return out


@njit(cache=True)
def _anchor(a,b,t,left,right,c):
    """Global restricted mode of a*s-b*s²/2-5log(hypot(s,t)).

    Candidates are both endpoints and real roots of
    b*s³-a*s²+(b*t²+5)*s-a*t²=0. Cardano's real/trigonometric
    forms enumerate all stationary points, including competing prior modes.
    c (the clipped Gaussian mode) is also retained for roundoff safety.
    """
    best=c
    top=(a-b*c/2)*c-2.5*math.log(c*c+t*t)
    candidates=np.empty(5)
    candidates[0]=left;candidates[1]=right
    count=2
    if b > 0:
        aa=-a/b;bb=t*t+5/b;cc=-a*t*t/b
        p=bb-aa*aa/3
        q=2*aa**3/27-aa*bb/3+cc
        disc=(q/2)**2+(p/3)**3
        if disc >= 0:
            root=math.sqrt(disc)
            candidates[2]=np.cbrt(-q/2+root)+np.cbrt(-q/2-root)-aa/3
            count=3
        elif p < 0:
            mag=2*math.sqrt(-p/3)
            angle=math.acos(min(1.,max(-1.,3*q/(p*mag))))/3
            for k in range(3):
                candidates[2+k]=mag*math.cos(angle-2*math.pi*k/3)-aa/3
            count=5
    elif a != 0:
        disc=25-4*a*a*t*t
        if disc >= 0:
            candidates[2]=(5-math.sqrt(disc))/(2*a)
            candidates[3]=(5+math.sqrt(disc))/(2*a)
            count=4
    else:
        candidates[2]=0.;count=3
    for k in range(count):
        s=candidates[k]
        if left <= s <= right:
            value=(a-b*s/2)*s-2.5*math.log(s*s+t*t)
            if value > top:
                best,top=s,value
    return best


@njit(cache=True)
def _s_interval(a,b,t,left,right,order):
    if right <= left:
        return -math.inf
    c=min(right,max(left,a/b)) if b else (right if a>0 else left)
    d=a-b*c
    m0=(_gaussian_side(max(0.,d),b,c-left)
        +_gaussian_side(max(0.,-d),b,right-c))
    anchor=_anchor(a,b,t,left,right,c)
    u=anchor*anchor+t*t
    correction=1.
    if order:
        ml=_side_moments(max(0.,d),b,c-left)
        mr=_side_moments(max(0.,-d),b,right-c)
        # E[(s-anchor)^k] by the binomial shift from clipped Gaussian c.
        moments=np.empty(5);moments[0]=1.
        for k in range(1,5):
            moments[k]=(mr[k]+(-1.)**k*ml[k])/m0
        shift=c-anchor
        e1=shift+moments[1]
        e2=shift*shift+2*shift*moments[1]+moments[2]
        correction+=(-5*anchor/u)*e1+.5*(-5/u+35*anchor*anchor/u**2)*e2
        if order==4:
            e3=shift**3+3*shift*shift*moments[1]+3*shift*moments[2]+moments[3]
            e4=(shift**4+4*shift**3*moments[1]+6*shift*shift*moments[2]
                +4*shift*moments[3]+moments[4])
            correction+=(105*anchor/u**2-315*anchor**3/u**3)*e3/6
            correction+=(105/u**2-1890*anchor*anchor/u**3+3465*anchor**4/u**4)*e4/24
    # Nonpositive Taylor integrals are outside asymptotic validity. Return
    # the finite order-0 estimate there (never silently call another method).
    if correction <= 0 or not math.isfinite(correction):
        correction=1.
    return (c*(a-.5*b*c)+math.log(m0)-2.5*math.log(u)+math.log(correction))


@njit(cache=True)
def _t_log(t,l1,l2,a,bt,lo,hi,order):
    if abs(t)>=hi:
        return -math.inf
    outer=math.sqrt(max(0.,(hi-abs(t))*(hi+abs(t))))
    if abs(t)<lo:
        inner=math.sqrt(max(0.,(lo-abs(t))*(lo+abs(t))))
        s=_logadd(_s_interval(a,l1,t,-outer,-inner,order),
                  _s_interval(a,l1,t,inner,outer,order))
    else:
        s=_s_interval(a,l1,t,-outer,outer,order)
    return t*(bt-.5*l2*t)+s


@njit(cache=True)
def _proxy_interval(a,b,t,left,right):
    if right<=left:return -math.inf
    c=min(right,max(left,a/b)) if b else (right if a>0 else left)
    width=min(right-left,1/max(math.sqrt(b),abs(a-b*c),1/(right-left)))
    return c*(a-.5*b*c)-2.5*math.log(c*c+t*t)+math.log(width)


@njit(cache=True)
def _t_proxy(t,l1,l2,a,bt,lo,hi):
    """Cheap exponent/prior/width geometry only; never a quadrature value."""
    if abs(t)>=hi:return -math.inf
    outer=math.sqrt(max(0.,(hi-abs(t))*(hi+abs(t))))
    if abs(t)<lo:
        inner=math.sqrt(max(0.,(lo-abs(t))*(lo+abs(t))))
        s=_logadd(_proxy_interval(a,l1,t,-outer,-inner),
                  _proxy_interval(a,l1,t,inner,outer))
    else:
        s=_proxy_interval(a,l1,t,-outer,outer)
    return t*(bt-.5*l2*t)+s


@njit(cache=True)
def _refine_t(left,right,l1,l2,a,bt,lo,hi):
    g=.6180339887498949
    x=right-g*(right-left);y=left+g*(right-left)
    fx=_t_proxy(x,l1,l2,a,bt,lo,hi);fy=_t_proxy(y,l1,l2,a,bt,lo,hi)
    for _ in range(28):
        if fx>fy:
            right,y,fy=y,x,fx;x=right-g*(right-left)
            fx=_t_proxy(x,l1,l2,a,bt,lo,hi)
        else:
            left,x,fx=x,y,fy;y=left+g*(right-left)
            fy=_t_proxy(y,l1,l2,a,bt,lo,hi)
    return x if fx>fy else y


@njit(cache=True)
def _t_geometry(l1,l2,a,bt,lo,hi):
    """Split at annulus cuts and seeded modes; determine one-sided widths.

    A logarithmic scan resolves the prior near xmin; extra seeds resolve
    Gaussian peaks, sbar/circle intersections and linear endpoint layers.
    Each segment retains its own dominant mode, so both
    +/-t lobes survive when the weak direction is flat.
    """
    raw=np.empty(9)
    raw[0]=-hi;raw[1]=-lo;raw[2]=0.;raw[3]=lo;raw[4]=hi
    count=5
    sb=a/l1 if l1 else 0.
    # Where the stiff Gaussian center meets a circle, its truncation
    # changes over a narrow t layer even though |t| is far from xmin/xmax.
    for radius in (lo,hi):
        if 0<abs(sb)<radius:
            cut=math.sqrt((radius-abs(sb))*(radius+abs(sb)))
            raw[count]=-cut;raw[count+1]=cut;count+=2
    raw[:count].sort()
    unique=1
    for k in range(1,count):
        if raw[k]-raw[unique-1]>1e-14*lo:
            raw[unique]=raw[k];unique+=1
    bounds=raw[:unique].copy()
    centers=np.empty(unique-1);wl=np.empty(unique-1);wr=np.empty(unique-1)
    for j in range(unique-1):
        left,right=bounds[j],bounds[j+1]
        seeds=np.empty(32);n=0
        for k in range(13):
            # Geometric points toward the smaller-|t| endpoint.
            u=lo*math.expm1(math.log1p((right-left)/lo)*k/12)
            seeds[n]=left+u if left>=0 else right-u;n+=1
        sb=a/l1 if l1 else 0.
        tb=bt/l2 if l2 else (hi if bt>0 else 0.)
        seeds[n]=min(right,max(left,tb));n+=1
        for radius in (lo,hi):
            t=math.sqrt(max(0.,radius*radius-sb*sb))
            seeds[n]=min(right,max(left,t if left>=0 else -t));n+=1
        # Boundary-layer seed; endpoints themselves have zero s measure.
        width=min(right-left,1/math.sqrt(l2) if l2 else right-left,
                  1/max(abs(bt-l2*right),1/(right-left)))
        seeds[n]=right-min(width,(right-left)/2);n+=1
        width=min(right-left,1/math.sqrt(l2) if l2 else right-left,
                  1/max(abs(bt-l2*left),1/(right-left)))
        seeds[n]=left+min(width,(right-left)/2);n+=1
        seeds[:n].sort()
        best=0;top=-math.inf
        for k in range(n):
            value=_t_proxy(seeds[k],l1,l2,a,bt,lo,hi)
            if value>top:best,top=k,value
        center=seeds[best]
        aa=seeds[max(0,best-1)];bb=seeds[min(n-1,best+1)]
        candidate=_refine_t(aa,bb,l1,l2,a,bt,lo,hi)
        value=_t_proxy(candidate,l1,l2,a,bt,lo,hi)
        if value>top:center,top=candidate,value
        centers[j]=center
        for side in range(2):
            edge=left if side==0 else right
            distance=abs(edge-center)
            if distance==0:
                width=lo
            else:
                # Locate a half-log-drop with a geometric search and bisection.
                width=min(lo,1/math.sqrt(l2) if l2 else lo,distance)
                sign=-1 if side==0 else 1
                while width<distance and _t_proxy(center+sign*width,l1,l2,a,bt,lo,hi)>top-.5:
                    width=min(distance,width*2)
                low=0.;high=width
                for _ in range(15):
                    mid=(low+high)/2
                    if _t_proxy(center+sign*mid,l1,l2,a,bt,lo,hi)>top-.5:
                        low=mid
                    else:high=mid
                width=max(high,1e-14*lo)
            if side==0:wl[j]=width
            else:wr[j]=width
    return bounds,centers,wl,wr


@njit(cache=True)
def lnL_marg_m3(A,B,C,D,E,xmin,xmax,order=4,n_t=48):
    """Laplace in s and fixed sinh-GL quadrature in t on the exact annulus.

    At fixed t, S(t)=[-outer,outer] minus (-inner,inner) for |t|<xmin,
    outer=sqrt(xmax²-t²), inner=sqrt(xmin²-t²). Each interval is treated
    separately. Its anchor s* is the restricted mode of f_s-5ln(r), found
    from a cubic, not merely clipped sbar (clipping omits the prior slope).
    I_s ~= exp(f_s(c))*r(s*,t)^-5*M0*sum_{k=0}^order h^(k)/h E[(s-s*)^k]/k!.
    c is the clipped Gaussian mode; moments of its truncated Gaussian use
    scaled erfcx and integration-by-parts, with stable polynomial-moment
    quadrature in cancellation-prone tails. Odd moments are retained.
    If a Taylor correction is nonpositive, that interval uses order 0.

    t's exponent is kept as bt*t-lam2*t²/2, so lam2=0 needs no division
    and arbitrary linear terms in the null direction are supported. Split
    at -xmax,-xmin,0,xmin,xmax and where the stiff Gaussian center meets
    either circle: +/-sqrt(radius²-sbar²). These segments share exactly
    n_t GL nodes across the sides of their located maxima, with independent
    sinh scales and one or two nodes minimum per active side. On each
    half-patch v=L*(2u-u²), u=scale*sinh(tau)/(2L), adds quadratic
    clustering at the far endpoint to resolve circle-intersection layers.
    Sizes are assigned in proportion to transformed length, sqrt(width),
    and a tempered height weight. Proxy-negligible patches (60 log units) and tails (48
    log units) are omitted. These empirical cuts are checked by node
    doubling, not an error certificate. Log-sum-exp prevents under/overflow.

    Domain: physical finite coefficients, 0<xmin<xmax, order=0,2,4,
    24<=n_t<=128. The all-zero likelihood is returned exactly. Other low-SNR
    outputs are finite estimates, without an accuracy guarantee; use the
    documented sweep thresholds or method2. No regulator or hidden method2
    fallback is used. This approximation needs small stiff Gaussian width
    relative to the local radial prior scale, not merely a large total SNR.
    """
    l1,l2,R,eta=_validate(A,B,C,D,E,xmin,xmax,order)
    if n_t<24 or n_t>128:
        raise ValueError('n_t must be between 24 and 128')
    if l1==0 and R==0:
        return 0.
    a=R*math.cos(eta);bt=R*math.sin(eta)
    bounds,centers,wl,wr=_t_geometry(l1,l2,a,bt,xmin,xmax)
    # Measure the significance and transformed length of every half-patch.
    # Up to sixteen patches share exactly n_t evaluations. Discard only patches
    # whose height is >60 below the largest, and trim monotone tails at a
    # proxy log-drop of 48 (both are empirical, checked by node doubling).
    segments=len(centers)
    heights=np.empty(segments);top=-math.inf
    for j in range(segments):
        heights[j]=_t_proxy(centers[j],l1,l2,a,bt,xmin,xmax)
        top=max(top,heights[j])
    pcenter=np.empty(16);pscale=np.empty(16);pextent=np.empty(16);pspan=np.empty(16)
    psign=np.empty(16);importance=np.empty(16);count=0
    for j in range(segments):
        if heights[j]<top-60:continue
        center=centers[j]
        for side in range(2):
            edge=bounds[j] if side==0 else bounds[j+1]
            scale=wl[j] if side==0 else wr[j]
            distance=abs(edge-center)
            if distance<=0:continue
            sign=-1 if side==0 else 1
            limit=min(distance,scale)
            while limit<distance and _t_proxy(center+sign*limit,l1,l2,a,bt,xmin,xmax)>heights[j]-48:
                limit=min(distance,2*limit)
            if limit<distance:
                low=limit/2;high=limit
                for _ in range(10):
                    mid=(low+high)/2
                    if _t_proxy(center+sign*mid,l1,l2,a,bt,xmin,xmax)>heights[j]-48:
                        low=mid
                    else:high=mid
                distance=high
            extent=math.asinh(2*distance/scale)
            pcenter[count]=center;pscale[count]=scale;pextent[count]=extent;psign[count]=sign
            pspan[count]=distance
            # Moderate the mass weighting: weak secondary patches still
            # receive nodes; long sinh intervals need more quadrature nodes.
            importance[count]=extent*math.exp(max(-20.,(heights[j]-top)/8))
            count+=1
    # Width matters as well as peak height: an extremely narrow, tall
    # sliver at an inner-circle cut must not starve the adjoining broad
    # boundary layer of nodes. Temper the estimated mass by sqrt(width).
    maxscale=np.max(pscale[:count])
    for j in range(count):
        importance[j]*=math.sqrt(pscale[j]/maxscale)
    minimum=2 if n_t>=2*count else 1
    nodes=np.full(count,minimum)
    # Greedy proportional allocation, with at least one node per patch.
    for _ in range(n_t-minimum*count):
        best=0;score=-1.
        for j in range(count):
            val=importance[j]/(nodes[j]+.5)
            if val>score:best,score=j,val
        nodes[best]+=1
    total=-math.inf
    for j in range(count):
        n=nodes[j]
        x,w=RULE_X[n],RULE_W[n]
        for k in range(n):
            tau=pextent[j]*(x[k]+1)/2
            # A quadratic endpoint map resolves the circle-intersection
            # layer also at the far end of a patch (a plain sinh map
            # clusters only at its maximum). u in [0,1], v=L*(2u-u²).
            u=.5*pscale[j]*math.sinh(tau)/pspan[j]
            t=pcenter[j]+psign[j]*pspan[j]*u*(2-u)
            weight=w[k]*pextent[j]*(1-u)*pscale[j]*math.cosh(tau)/2
            if weight>0:
                value=_t_log(t,l1,l2,a,bt,xmin,xmax,order)+math.log(weight)
                total=_logadd(total,value)
    norm=math.log(3/(2*math.pi))+3*math.log(xmin)-math.log1p(-(xmin/xmax)**3)
    return norm+total


@njit(cache=True)
def method3_array(parameters,variant=1,order=4,n_t=48):
    """Evaluate (n,7) raw coefficient/bound rows; variant 1=1-D, 2=2-D."""
    out=np.empty(len(parameters))
    for i in range(len(parameters)):
        p=parameters[i]
        if variant==2:
            out[i]=lnL_marg_m3_2d(p[0],p[1],p[2],p[3],p[4],p[5],p[6],order)
        else:
            out[i]=lnL_marg_m3(p[0],p[1],p[2],p[3],p[4],p[5],p[6],order,n_t)
    return out
