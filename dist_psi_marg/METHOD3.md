# Method 3: annulus-aware Laplace approximations

`dpmarg.method3` exports `lnL_marg_m3_2d(A,B,C,D,E,xmin,xmax,order=4)`
and `lnL_marg_m3(A,B,C,D,E,xmin,xmax,order=4,n_t=48)`. Both are Numba
functions, using the same raw coefficients and normalization as method 2.
The permitted Taylor orders are 0, 2 and 4. There is no regulator k.

Write a=R cos(eta), b_t=R sin(eta), f=a s-lambda1 s²/2+b_t t-lambda2 t²/2,
and h=(s²+t²)^(-5/2). The integral is norm + log integral exp(f) h ds dt,
where norm=log[3/(2 pi (xmin^-3-xmax^-3))].

## Interior 2-D formula

With sbar=a/lambda1, tbar=b_t/lambda2, u=sbar²+tbar², v=sbar²/u,
w=tbar²/u, z_i=1/(lambda_i u), the multiplier of the leading Gaussian
integral is 1+C2+C4, truncated at the requested order:

```
C2 = [(-5+35v)z1 + (-5+35w)z2]/2
C4 = [(105-1890v+3465v²)z1² + (105-1890w+3465w²)z2²]/8
     + (-280+3465vw)z1*z2/4
```

This is the Gaussian expectation of h's Taylor polynomial: the fourth
moments are 3/lambda_i² and the mixed second moment is 1/(lambda1 lambda2).
In particular h_ss/h=g_ss+g_s² for g=-5 log(r). The code returns
norm+fpeak+log(2 pi)-log(lambda1 lambda2)/2-5 log(rbar)+log(multiplier).
The independent tensor Gauss-Hermite test verifies error scaling as
rho^-2, rho^-4 and rho^-6 for the three orders on a smooth interior case.

A conservative applicability guard requires
xmin+3/sqrt(lambda2)<rbar<xmax-3/sqrt(lambda2). This contains the entire
three-sigma ellipse. A failed guard, lambda2=0, or a nonpositive correction
returns NaN. Passing the guard does not certify accuracy: at low SNR the
prior-dominated inner region can outweigh the local likelihood peak.

## Integration of the stiff direction

At fixed t, the outer endpoint is sqrt(xmax²-t²). If |t|<xmin, remove
(-sqrt(xmin²-t²),sqrt(xmin²-t²)). Sum the resulting intervals separately;
choosing only one interval would remove a potentially important second mode.

In each interval [l,u], c is the clipped Gaussian center a/lambda1. The
expansion anchor s* maximizes f_s-5 log(r). Endpoints and every real root of

```
lambda1 s³ - a s² + (lambda1 t²+5)s - a t² = 0
```

are compared. Merely clipping sbar does not maximize this prior-weighted
integrand, so the cubic implements the literal maximizer in the request.

On either side of c, use v=|s-c| and Gaussian polynomial moments
M_k=integral_0^L v^k exp(-d v-lambda1 v²/2) dv. For lambda1>0,

```
M0 = sqrt(pi/(2 lambda1)) [erfcx(d/sqrt(2 lambda1))
     - exp(-d L-lambda1 L²/2) erfcx((d+lambda1 L)/sqrt(2 lambda1))]
lambda1 M1 = 1-exp(-d L-lambda1 L²/2)-d M0
lambda1 Mk = (k-1) M[k-2]-d M[k-1]-L^(k-1) exp(-d L-lambda1 L²/2).
```

`_gaussian_side` preserves short intervals with expm1 and supports lambda1=0.
When the moment recurrence would suffer cancellation, a scaled 24-point
Gauss-Legendre rule evaluates the Gaussian polynomial moments only, with a
48-log-unit tail cut. This does not integrate h numerically in s.
The moment tests compare all five moments against independent SciPy quad,
including far tails, short intervals and zero curvature.

Combine the signed left/right moments about c, shift them by c-s*, and
multiply M0 exp(f_s(c)) h(s*,t) by
sum_(k=0)^order [h^(k)(s*,t)/h(s*,t)] E[(s-s*)^k]/k!. The derivative ratios are

```
H1 = -5s/u
H2 = -5/u+35s²/u²
H3 = 105s/u²-315s³/u³
H4 = 105/u²-1890s²/u³+3465s⁴/u⁴,  u=s²+t².
```

Odd moments must be retained for a truncated normal. A nonpositive Taylor
multiplier falls back to that interval's finite order-0 estimate. This is
explicitly an out-of-validity estimate, not an accuracy guarantee or a call
to method 2. The all-zero likelihood returns exactly zero.

## The remaining t integral

The exponent b_t t-lambda2 t²/2 is evaluated directly, so a zero weak
eigenvalue and an arbitrary nonzero weak-axis linear term remain finite.
The geometry scan locates peaks and one-sided half-log-drop widths. Split
at -xmax,-xmin,0,xmin,xmax and at +/-sqrt(radius²-sbar²) for either circle
when real. The extra cuts resolve the change in Gaussian truncation where
its center meets a circle.

Exactly n_t nodes are distributed over the active half-patches (24<=n_t<=128).
Each uses a sinh coordinate centered at its local maximum. A quadratic
endpoint map v=L(2u-u²), u=scale*sinh(tau)/(2L), also clusters points at the
far endpoint. The node allocation uses transformed length, sqrt(width),
and a tempered peak height. All Gaussian and t contributions accumulate
in log space. Proxy heights exclude negligible patches at a 60-log-unit
drop and truncate tails at a 48-log-unit drop; these are empirical cuts,
not rigorous error bounds. Node doubling measures remaining quadrature error.

## Sweep conventions and interpretation

Run `python scripts/check_method3.py`; it reuses the 40 regression and 300
random saved references and a restartable coefficient-keyed sweep cache.
A fresh run computed 4,525 references with `lnL_marg_ref`; the seven SciPy
warning cases agree with independent method-2 angular quadrature at 256
nodes to 3.64e-12. No packages or environments were added.

rho is the actual annular matched-filter SNR. x0* is a(theta*)/b(theta*) at
a dominant angle of the annulus-constrained likelihood, not a prescribed
unconstrained Cartesian radius. Coefficients are scaled together to vary
rho without changing this geometry. With q=0, eta=0, an interior flat ridge
permits theta*=0, which fixes its otherwise arbitrary angular tie.

29 of the 210 requested geometry cells do not exist with this definition.
For q=0 and eta>0, the minimum possible x0*/xmax at a dominant angle is
(1+sin(eta))/(1-sin(eta)): already 2.24 at pi/8, greater than every requested
radius. At pi/2 the radial peak is infinite. For q=0, eta=0, x0*<xmin is
also impossible: the dominant ridge has a/b=r>=xmin. These cells are
explicitly undefined; no nominal radius is substituted. All attainable
cells retain the full 25-point rho=1..1000 sweep.

See [the summary](data/method3_summary.md), [every regime and angle](data/method3_breakdown.md),
[the machine summary](data/method3_check.json), and the two figures in
`figures/method3_*.png`. Breakdowns require every larger sampled rho to pass;
they are measured grid thresholds, not bounds between grid points or beyond
rho=1000. The 96-node diagnostic separates Laplace errors from quadrature
errors in the requested 48-node implementation.

The 2-D formula is suitable for a guarded high-SNR interior branch. Its
three-sigma guard alone is insufficient. The 1-D implementation misses the
few-microsecond cost target and has residual quadrature failures in strongly
anisotropic inner-boundary cases. Low SNR, nonpositive Taylor multipliers,
failed 2-D guards and unresolved t quadrature require a method-2 fallback
in a future dispatcher. This module never applies that fallback silently.
