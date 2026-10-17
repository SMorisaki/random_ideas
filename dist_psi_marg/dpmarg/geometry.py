"""Scan-free angular geometry for the pi-periodic distance-kernel pair.

For a=|R cos(theta-eta)| and b=lam1 cos²theta+lam2 sin²theta,
q=max_r(ar-br²/2)-4 log r is the historical geometry proxy. Its
interior exponent maximum is atan2(lam1 sin eta,lam2 cos eta), modulo pi.
This is the stable Cartesian form of (delta+asin(S sin(2eta)/rho'))/2;
the other stationary root has a=0 and is a minimum (except flat ridges).

``proxy_derivatives(..., laplace=True)`` adds -log(b)/2 in the interior
and -log|a-br| at an endpoint. These asymptotes apply only to an isolated
radial saddle / endpoint layer, away from a clipping transition. The
selector uses them only in that domain; q remains the continuous search
proxy at clipping transitions and in prior-dominated cases. No radial
kernel or angular scan is used here.
"""
import math
import numpy as np
from numba import njit

PI = math.pi


@njit(cache=True)
def clipping_width(t, l1, l2, R, eta, lo, hi):
    """Return (width,kink) at a radial-clipping maximum; (.3,False) otherwise.

    x0=a/b, x0'=(a'-b'x0)/b. Mapping one radial standard deviation
    1/sqrt(b) into angle gives w_cusp=sqrt(b)/|a'-b' x0|.
    Use the minimum of this and both one-sided curvature widths. At b->0
    with a->0 there is no isolated radial saddle: h=2*int r^-4 dr,
    and the singular endpoint Laplace factor must never define a width.
    Domain: l1>=l2>=0, R>=0, 0<lo<hi.
    """
    co=math.cos(t-eta)
    a=abs(R*co);ap=-R*math.sin(t-eta)*(1. if co>=0 else -1.)
    c=math.cos(t);b=l2+(l1-l2)*c*c
    if b*hi*hi<1e-12 and a*hi<1e-8:
        return .3,False
    bp=-(l1-l2)*math.sin(2*t)
    r=lo if abs(a-b*lo)<abs(a-b*hi) else hi
    if b<=0 or abs(a-b*r)>1e-8*max(a,b*r,1e-300):
        return .3,False
    slope=abs(ap-bp*r)
    wc=min(.3,math.sqrt(b)/max(slope,1e-300))
    eps=min(1e-7,1e-3*wc)
    left=proxy_derivatives(t-eps,l1,l2,R,eta,lo,hi)
    right=proxy_derivatives(t+eps,l1,l2,R,eta,lo,hi)
    if left[1]<=0 or right[1]>=0:
        return .3,False
    wl=min(.3,1/math.sqrt(max(abs(left[2]),1e-30)))
    wr=min(.3,1/math.sqrt(max(abs(right[2]),1e-30)))
    return min(wc,wl,wr),True


@njit(cache=True)
def candidate_seeds(l1,l2,R,eta,lo,hi):
    """Interior stationary points, axes and refined clipping crossings.

    Near the weak axis delta=pi/2-theta, solve
    r*(lam2+(lam1-lam2)*delta²)=+/-R*(delta*cos eta+sin eta)
    for seeds, then Newton-refine the exact trigonometric crossing. Both
    signs and both bounds are retained, including delta~eta islands via
    interior_theta. This finite candidate construction contains no scan.
    """
    seeds=np.empty(28)
    seeds[0]=interior_theta(l1,l2,eta)
    seeds[1]=eta+PI/2
    seeds[2],seeds[3],seeds[4],seeds[5]=PI/2,eta,0.,PI
    k=6
    diff=l1-l2
    for r in (lo,hi):
        angle=math.acos(min(1.,R*math.cos(eta)/max(diff*r,1e-300)))
        seeds[k],seeds[k+1]=angle,PI-angle
        k+=2
        for sign in (-1.,1.):
            aa=r*diff;bb=-sign*R*math.cos(eta);cc=r*l2-sign*R*math.sin(eta)
            disc=bb*bb-4*aa*cc
            if aa<=0 or disc<0:continue
            z=-.5*(bb+math.copysign(math.sqrt(disc),bb))
            for delta in (z/aa,cc/z if z else 0.):
                if abs(delta)>1.:continue
                t=PI/2-delta
                for _ in range(12):
                    co=math.cos(t-eta)
                    a=abs(R*co);ap=-R*math.sin(t-eta)*(1. if co>=0 else -1.)
                    b=l2+diff*math.cos(t)**2;bp=-diff*math.sin(2*t)
                    fp=ap-bp*r
                    if abs(fp)<1e-14*max(R,diff*r,1e-300):break
                    step=(a-b*r)/fp
                    t-=min(.2,max(-.2,step))
                    if abs(step)<1e-14:break
                a=abs(R*math.cos(t-eta));b=l2+diff*math.cos(t)**2
                if abs(a-b*r)<1e-8*max(a,b*r,1e-300):
                    seeds[k]=t%PI;k+=1
    seeds[k],seeds[k+1]=eta+PI/2-.01,eta+PI/2+.01
    return seeds[:k+2]


@njit(cache=True)
def interior_theta(l1, l2, eta):
    """Maximum of cos²(theta-eta)/b for l1>=l2>=0; flat ridge returns 0."""
    return math.atan2(l1*math.sin(eta), l2*math.cos(eta)) % PI


@njit(cache=True)
def proxy_derivatives(t, l1, l2, R, eta, lo, hi, laplace=False):
    """Return p,p',p'' analytically, on each smooth radial-clipping branch.

    For interior r=a/b, r'=(a'-b'r)/b and
    r''=(-a-b''r-2b'r')/b. Thus q'=a'r-b'r²/2-4r'/r,
    q''=-ar-b''r²/2+(a'-b'r)r'-4(r''/r-(r'/r)²).
    Endpoint r is constant. Laplace width corrections are differentiated
    analytically; the endpoint formula is undefined at zero slope and is
    deliberately not applied there. At a=0 the pair has a cusp in q.
    """
    co, si = math.cos(t-eta), math.sin(t-eta)
    a = abs(R*co)
    ap = -R*si*(1. if co >= 0 else -1.)
    c, s = math.cos(t), math.sin(t)
    b = l1*c*c+l2*s*s
    bp = -2*(l1-l2)*s*c
    bpp = -2*(l1-l2)*(c*c-s*s)
    r = min(hi, max(lo, a/b)) if b > 0 else (hi if a > 0 else lo)
    p = r*(a-.5*b*r)-4*math.log(r)
    d = ap*r-.5*bp*r*r
    dd = -a*r-.5*bpp*r*r
    if b > 0 and lo < r < hi:
        rp = (ap-bp*r)/b
        rpp = (-a-bpp*r-2*bp*rp)/b
        d -= 4*rp/r
        dd += (ap-bp*r)*rp-4*(rpp/r-(rp/r)**2)
        if laplace:
            p += .5*math.log(2*PI/b)
            d -= .5*bp/b
            dd -= .5*(bpp/b-(bp/b)**2)
    elif laplace:
        slope = a-b*r
        if slope != 0:
            sp, spp = ap-bp*r, -a-bpp*r
            p -= math.log(abs(slope))
            d -= sp/slope
            dd -= spp/slope-(sp/slope)**2
    return p, d, dd


@njit(cache=True)
def _newton(seed, l1, l2, R, eta, lo, hi, laplace=False):
    """Safeguarded ascent Newton, with local curvature-scaled trust steps.

    Backtracking also resolves maxima at radial clipping cusps. Keeping
    the seed prevents the narrow weak-axis island from being overwritten
    by a different broad maximum. No absolute-curvature minimum is accepted.
    """
    t = seed
    p, d, dd = proxy_derivatives(t,l1,l2,R,eta,lo,hi,laplace)
    for _ in range(10):
        w = min(.3, 1/math.sqrt(max(abs(dd),1e-30)))
        if dd < 0:
            step = -d/dd
        else:
            step = math.copysign(w,d)
        step = min(.35,max(-.35,step))
        if abs(step) < 2e-12*max(w,1e-4):
            break
        accepted = False
        for j in range(6):
            u = t+step
            if not laplace:
                # Newton can jump across a C0 maximum at a=b*xmin.
                # Solve that crossing directly instead of backtracking
                # repeatedly against incompatible one-sided curvatures.
                f0=abs(R*math.cos(t-eta))-(l2+(l1-l2)*math.cos(t)**2)*lo
                f1=abs(R*math.cos(u-eta))-(l2+(l1-l2)*math.cos(u)**2)*lo
                if f0*f1<0:
                    root=t+step*abs(f0)/(abs(f0)+abs(f1))
                    for _k in range(6):
                        co=math.cos(root-eta)
                        aa=abs(R*co);ap=-R*math.sin(root-eta)*(1. if co>=0 else -1.)
                        bb=l2+(l1-l2)*math.cos(root)**2
                        bp=-(l1-l2)*math.sin(2*root)
                        fp=ap-bp*lo
                        if fp==0:break
                        root-=(aa-bb*lo)/fp
                    q,qd,qdd=proxy_derivatives(root,l1,l2,R,eta,lo,hi)
                    if q>=p and abs((root-t+PI/2)%PI-PI/2)<=abs(step)*1.01:
                        eps=1e-7
                        dl=proxy_derivatives(root-eps,l1,l2,R,eta,lo,hi)[1]
                        dr=proxy_derivatives(root+eps,l1,l2,R,eta,lo,hi)[1]
                        if dl>0 and dr<0:
                            return root%PI,q,qdd
            v, vd, vdd = proxy_derivatives(u,l1,l2,R,eta,lo,hi,laplace)
            if v >= p-2e-14*max(1.,abs(p)):
                t,p,d,dd = u,v,vd,vdd
                accepted = True
                break
            step *= .5
        if not accepted:
            break
    return t % PI, p, dd


@njit(cache=True)
def analytic_geometry(l1, l2, R, eta, lo, hi):
    """Return all significant q maxima, curvature widths and midpoint cuts.

    Seeds: both interior stationary branches, weak/stiff axes, eta and its
    perpendicular, and both endpoint weak-axis offsets. Retain peaks within
    40 log units of the best. Widths cap at .3; cuts partition one period.
    Domain: finite l1>=l2>=0, R>=0, 0<lo<hi. Broad/flat regimes are valid.
    """
    if R == 0:
        w = min(.3,1/math.sqrt(max((l1-l2)*lo*lo,1e-30)))
        return np.array([PI/2]),np.array([w]),np.array([0.,PI])
    if l2==0 and eta<1e-14 and R*R<=l1:
        # h(theta)=2 int r^-4 exp(-l1*r²*cos²(theta)/2)
        # cosh(R*r*cos(theta)) dr. Since tanh(z)<=z, R²<=l1
        # makes the null axis the global maximum. There h=2 V and
        # -(log h)''=(l1-R²)*<r²>_prior, a finite scale independent
        # of the spurious vanishing-slope xmax endpoint proxy.
        moment2=3*lo*lo*(1-lo/hi)/(1-(lo/hi)**3)
        w=min(.3,1/math.sqrt(max((l1-R*R)*moment2,1e-30)))
        return np.array([PI/2]),np.array([w]),np.array([0.,PI])
    if l1 == l2:
        ts, ws = np.empty(3),np.empty(3)
        n = 0
        p,d,dd = proxy_derivatives(eta,l1,l2,R,eta,lo,hi)
        if dd <= 0:
            ts[n],ws[n] = eta,min(.3,1/math.sqrt(max(-dd,1e-30)))
            n += 1
        if l1*lo < R and l1*lo*lo < 4:
            delta = math.acos(l1*lo/R)
            ts[n],ts[n+1] = (eta-delta)%PI,(eta+delta)%PI
            ws[n]=clipping_width(ts[n],l1,l2,R,eta,lo,hi)[0]
            ws[n+1]=clipping_width(ts[n+1],l1,l2,R,eta,lo,hi)[0]
            n += 2
        if n == 0:
            ts[0],ws[0],n = eta,.3,1
        heights = np.array([proxy_derivatives(t,l1,l2,R,eta,lo,hi)[0] for t in ts[:n]])
        keep = heights >= np.max(heights)-40
        centers,widths = ts[:n][keep],ws[:n][keep]
        order=np.argsort(centers);centers,widths=centers[order],widths[order]
        bounds=np.empty(len(centers)+1)
        bounds[0]=(centers[-1]-PI+centers[0])/2
        for j in range(1,len(centers)):bounds[j]=(centers[j-1]+centers[j])/2
        bounds[-1]=bounds[0]+PI
        return centers,widths,bounds
    # If R<lam2*xmin every ray is clipped below xmin. At large angular
    # curvature the two stationary boundary branches start at the weak
    # axis +/-R*cos(eta)/((lam1-lam2)*xmin). Newton then stays on a smooth
    # endpoint branch; no clipping/prior candidates can compete.
    if R < l2*lo and (l1-l2)*lo*lo>100:
        delta=math.asin(min(.5,R*math.cos(eta)/((l1-l2)*lo)))
        ts,ws,hs=np.empty(2),np.empty(2),np.empty(2)
        n=0
        for seed in (PI/2-delta,PI/2+delta):
            t,p,dd=_newton(seed,l1,l2,R,eta,lo,hi)
            if dd>=0:continue
            w=min(.3,1/math.sqrt(-dd))
            if n and abs((t-ts[0]+PI/2)%PI-PI/2)<min(w,ws[0]):continue
            ts[n],ws[n],hs[n]=t,w,p
            n+=1
        if n:
            keep=hs[:n]>=np.max(hs[:n])-40
            centers,widths=ts[:n][keep],ws[:n][keep]
            order=np.argsort(centers);centers,widths=centers[order],widths[order]
            bounds=np.empty(len(centers)+1)
            bounds[0]=(centers[-1]-PI+centers[0])/2
            for j in range(1,len(centers)):bounds[j]=(centers[j-1]+centers[j])/2
            bounds[-1]=bounds[0]+PI
            return centers,widths,bounds
    # In a well-conditioned, high-SNR interior saddle the other possible
    # maxima are prior/clipping lobes. Retain the general candidate search
    # until the dominant height exceeds their low-SNR range by >40.
    t0 = interior_theta(l1,l2,eta)
    a0=abs(R*math.cos(t0-eta));b0=l2+(l1-l2)*math.cos(t0)**2
    r0=a0/b0 if b0 else hi*2
    if (l2/l1>.01 and lo<r0<hi and min(r0-lo,hi-r0)*math.sqrt(b0)>8
            and proxy_derivatives(t0,l1,l2,R,eta,lo,hi)[0]>60-4*math.log(lo)):
        t,p,dd=_newton(t0,l1,l2,R,eta,lo,hi)
        if dd<0:
            w=min(.3,1/math.sqrt(-dd))
            return np.array([t]),np.array([w]),np.array([t-PI/2,t+PI/2])
    seeds = candidate_seeds(l1,l2,R,eta,lo,hi)
    k=len(seeds)
    centers, heights, widths = np.empty(k), np.empty(k), np.empty(k)
    count = 0
    for i in range(k):
        repeated=False
        for j in range(i):
            if abs((seeds[i]-seeds[j]+PI/2)%PI-PI/2)<1e-10:
                repeated=True
        if repeated:continue
        seed = seeds[i]
        ps,ds,dds = proxy_derivatives(seed,l1,l2,R,eta,lo,hi)
        a=abs(R*math.cos(seed-eta));b=l2+(l1-l2)*math.cos(seed)**2
        # The zero-slope radial pair is regular (2 V), not an endpoint
        # saddle. For R²>lam1 its angular curvature is positive: discard
        # the artificial sharp maximum of the clipped exponent proxy.
        if b*hi*hi<1e-12 and a*hi<1e-8 and R*R>l1:
            continue
        wc,cusp=clipping_width(seed,l1,l2,R,eta,lo,hi)
        if cusp:
            t,p,dd = seed%PI,ps,dds
            w = wc
        else:
            t,p,dd = _newton(seed,l1,l2,R,eta,lo,hi)
            w = min(.3, 1/math.sqrt(max(abs(dd),1e-30)))
        a=abs(R*math.cos(t-eta));b=l2+(l1-l2)*math.cos(t)**2
        if b*hi*hi<1e-12 and a*hi<1e-8 and R*R>l1:
            continue
        wc,cusp=clipping_width(t,l1,l2,R,eta,lo,hi)
        if cusp:w=wc
        if (not cusp and b>0 and
                min(abs(a-b*lo),abs(a-b*hi))<1e-8*max(a,b*lo,1e-300)):
            continue
        # A stopped Newton iterate or an xmax crossing with a nonzero
        # one-sided prior derivative is not a maximum. Curvature alone
        # accepts both, particularly on the flat exponent ridge.
        if not cusp:
            d=proxy_derivatives(t,l1,l2,R,eta,lo,hi)[1]
            if abs(d)*w>1e-3:
                continue
        if dd > 1e-10 and not cusp:
            continue
        duplicate = -1
        for j in range(count):
            if abs((t-centers[j]+PI/2)%PI-PI/2) < min(w,widths[j]):
                duplicate = j
                break
        if duplicate >= 0:
            if p > heights[duplicate]:
                centers[duplicate],heights[duplicate],widths[duplicate] = t,p,w
        else:
            centers[count],heights[count],widths[count] = t,p,w
            count += 1
    if count == 0:
        return np.array([eta]), np.array([.3]), np.array([eta-PI/2,eta+PI/2])
    best = np.max(heights[:count])
    keep = heights[:count] >= best-40
    centers, widths = centers[:count][keep], widths[:count][keep]
    order = np.argsort(centers)
    centers, widths = centers[order], widths[order]
    bounds = np.empty(len(centers)+1)
    bounds[0] = (centers[-1]-PI+centers[0])/2
    for j in range(1,len(centers)):
        bounds[j] = (centers[j-1]+centers[j])/2
    bounds[-1] = bounds[0]+PI
    return centers, widths, bounds


@njit(cache=True)
def laplace_peak(t, l1, l2, R, eta, lo, hi):
    """Refine q maximum by full Laplace p; return center,width,validity.

    Interior: require four radial standard deviations to both bounds and
    r*sqrt(b)>6. Endpoint: require |a-br|>6 sqrt(b) and |a-br|r>20.
    These are empirical guards for geometry, not likelihood approximations;
    the angular selector additionally estimates the pedestal mass outside
    six-width windows using precomputed uniform probes.
    """
    a = abs(R*math.cos(t-eta))
    b = l2+(l1-l2)*math.cos(t)**2
    r = a/b if b else hi*2
    interior = lo < r < hi
    if interior:
        valid = min(r-lo,hi-r)*math.sqrt(b) > 4 and r*math.sqrt(b)>6
    else:
        rb = lo if r<=lo else hi
        slope = abs(a-b*rb)
        valid = slope > 6*math.sqrt(b) and slope*rb>20
    if not valid:
        return t, .3, False
    u,p,dd = _newton(t,l1,l2,R,eta,lo,hi,True)
    return u, min(.3,1/math.sqrt(max(-dd,1e-30))), dd < 0
