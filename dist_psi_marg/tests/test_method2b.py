"""Analytic angular geometry and non-nested Gaussian/fixed-grid rules."""
import json
import math
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from dpmarg.geometry import interior_theta, proxy_derivatives, analytic_geometry
from dpmarg.method2 import (lnL_marg_m2_table, method2_analytic_diagnostics,
                           _geometry, _proxy)
from dpmarg.params import reduce_params, expand_params
from dpmarg.regimes import named_cases
from dpmarg.table1d import load_table, build_table
ROOT=Path(__file__).resolve().parents[1]
KEYS=('A','B','C','D','E','xmin','xmax')


@pytest.fixture(scope='module')
def tables():
    return {hi:load_table(ROOT/f'data/table1d_xmin0.1_xmax{hi:g}.npz')
        if (ROOT/f'data/table1d_xmin0.1_xmax{hi:g}.npz').exists() else build_table(.1,hi)
        for hi in (10.,100.)}


def test_closed_form_200():
    """Compare exponent proxy heights with an independent dense+bounded search."""
    rng=np.random.default_rng(9009)
    for _ in range(200):
        l1=10**rng.uniform(-4,8);l2=l1*10**rng.uniform(-3,0)
        eta=rng.uniform(0,np.pi/2);R=10**rng.uniform(0,1.5)*math.sqrt(l2)
        def f(t):return R**2*np.cos(t-eta)**2/(2*(l1*np.cos(t)**2+l2*np.sin(t)**2))
        grid=np.arange(8192)*np.pi/8192;k=np.argmax(f(grid));step=np.pi/8192
        opt=minimize_scalar(lambda t:-f(t),bounds=((k-1)*step,(k+1)*step),
                            method='bounded',options={'xatol':1e-14})
        t=interior_theta(l1,l2,eta)
        assert abs(f(t)+opt.fun)<1e-10
        E,S=(l1+l2)/2,(l1-l2)/2
        delta=math.atan2(E*math.sin(2*eta),E*math.cos(2*eta)-S)
        rho=math.hypot(E*math.cos(2*eta)-S,E*math.sin(2*eta))
        root=(delta+math.asin(np.clip(S*math.sin(2*eta)/rho,-1,1)))/2
        assert abs(f(t)-f(root))<1e-10


@pytest.mark.parametrize('laplace',[False,True])
@pytest.mark.parametrize('pars',[(10.,3.,20.,.4,.1,100.),(1e6,1e5,100.,.4,.1,100.),
                               (.1,.03,200.,.4,.1,100.)])
def test_analytic_derivatives(pars,laplace):
    t=.8;dt=1e-5
    p,d,dd=proxy_derivatives(t,*pars,laplace)
    pm,dm,_=proxy_derivatives(t-dt,*pars,laplace)
    pp,dp,_=proxy_derivatives(t+dt,*pars,laplace)
    assert d==pytest.approx((pp-pm)/(2*dt),rel=2e-8,abs=1e-8)
    assert dd==pytest.approx((dp-dm)/(2*dt),rel=2e-8,abs=1e-8)


def test_named_geometry():
    for c in named_cases():
        pars=(*reduce_params(*(c[k] for k in KEYS[:5])),c['xmin'],c['xmax'])
        a,w,b=analytic_geometry(*pars);centers,widths,_=_geometry(*pars)
        l1,l2,R,eta,lo,hi=pars
        if l2==0 and eta<1e-14 and 0<R*R<=l1:
            # At the regular null axis q's endpoint proxy is singular.
            # The exact pair, by tanh(z)<=z, has its maximum at pi/2.
            assert a==pytest.approx([np.pi/2])
            continue
        for t,v in zip(centers,widths):
            k=np.argmin(abs((a-t+np.pi/2)%np.pi-np.pi/2))
            assert abs((a[k]-t+np.pi/2)%np.pi-np.pi/2)<.2*v or abs(_proxy(a[k],*pars)-_proxy(t,*pars))<1e-6
            if max(w[k]/v,v/w[k])>=1.5:
                # The analytic clipping root is exact; the legacy golden
                # refinement can stop slightly off it. Transition widths
                # deliberately differ from one-sided proxy curvatures.
                l1,l2,R,eta,lo,hi=pars
                aa=abs(R*math.cos(a[k]-eta));bb=l2+(l1-l2)*math.cos(a[k])**2
                assert min(abs(aa-bb*lo),abs(aa-bb*hi))<1e-5*max(aa,bb*lo,1e-30)



def test_regression_scan_and_reference(tables):
    for c in json.loads((ROOT/'data/regression_points.json').read_text()):
        p=[c[k] for k in KEYS];tab=tables[p[6]]
        value=lnL_marg_m2_table(*p,tab,rule='analytic')
        scan=lnL_marg_m2_table(*p,tab,rule='scan')
        assert abs(value-scan)<2e-4,(c['name'],value-scan)
        assert abs(value-c['lnL_marg_ref'])<1e-4


def test_iteration008_and005_repairs(tables):
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from check_geometry import repaired_cases
    from final_benchmark import references
    cases=repaired_cases()
    references({'repaired':cases})
    for c in cases:
        p=[c[k] for k in KEYS]
        value=lnL_marg_m2_table(*p,tables[100.],rule='analytic')
        assert abs(value-c['lnL_marg_ref'])<1e-4,(c['name'],value-c['lnL_marg_ref'])


def test_zero_degenerate_and_exterior(tables):
    for hi,tab in tables.items():
        assert lnL_marg_m2_table(0.,0.,0.,0.,0.,.1,hi,tab,rule='analytic')==0.
        for reduced in [(1e8,0.,0.,0.),(1e8,0.,.1,.3),(1e6,1e6,100.,.3),
                        (.001,0.,100.,.7),(10.,3.,0.,0.)]:
            coeff=expand_params(*reduced)
            d=method2_analytic_diagnostics(*coeff,.1,hi,tab)
            assert math.isfinite(d[0])
            assert d[1] in (3,4,5,6) and d[2]>0 and 0<d[3]<=.3
    with pytest.raises(ValueError):
        lnL_marg_m2_table(0.,0.,0.,0.,0.,.1,100.,tables[100.],rule='typo')


def test_new_geometry_audit_regression(tables):
    # Seed 902 #18544: old scan256 loses a prior/clipping contribution.
    # Cached lnL_marg_ref (rtol=1e-10); the analytic rule agrees to 6.5e-9.
    p=(2.0200153597425166,17.548264393018165,-2856.695477256783,
       741.8864928682437,2951.4580495205587,.1,100.)
    assert abs(lnL_marg_m2_table(*p,tables[100.])-(-2.5190542658990744))<1e-6


def test_optional_doubled_nodes(tables):
    from dpmarg.method2 import method2_analytic_error
    for reduced in [(4.,4.,160.,.37),(10.,0.,1.,0.),(1.,1.,.3,.37)]:
        p=(*expand_params(*reduced),.1,100.)
        value,delta=method2_analytic_error(*p,tables[100.])
        assert value==lnL_marg_m2_table(*p,tables[100.])
        assert abs(delta)<1e-4


@pytest.mark.parametrize('pars,reference',[
    ((0.,0.,58154.54510260031,-10104.618681801097,59025.87936485976,.1,100.),-4.050044824572109),
    ((0.,0.,8399.122295462908,-22504.46323694499,24020.74355462692,.1,100.),-3.6001023900591917),
    ((47.85343646433819,53.04757690757975,90.86919828238011,203.0675110531693,229.08433982342137,.1,100.),16.50296140959407),
    ((-835.9063585086028,1291.591938786055,-87595.35631516055,-342.3816144824504,93434.6272155204,.1,100.),57.75406122057652),
    ((-2481.9364612434238,1355.7418930061444,-3441.490271532935,-5262.704102212494,9270.966678321634,.1,100.),566.0084923520774),
])
def test_grid_and_null_core_regressions(tables,pars,reference):
    # Cached brute-force references: seed4_097, seed808 #779/#904/#964,
    # posterior #269. The uncorrected Gaussian spacing estimate fails here.
    assert abs(lnL_marg_m2_table(*pars,tables[100.])-reference)<1e-4


def test_skipping_matches_identical_unpruned_rule(tables):
    from dpmarg.method2 import _analytic_compute_details,ANALYTIC_GH_ORDER
    skipped=0
    for c in json.loads((ROOT/'data/regression_points.json').read_text()):
        p=[c[k] for k in KEYS];tab=tables[p[6]]
        pruned=_analytic_compute_details(*p,tab)
        full=_analytic_compute_details(*p,tab,ANALYTIC_GH_ORDER,0,False,np.inf)
        assert abs(pruned[0]-full[0])<1e-7,c['name']
        assert pruned[1]<=full[1] and pruned[5]<=full[5]
        skipped+=pruned[6]
    assert skipped>0


def test_skip_integral_certificates():
    from dpmarg.method2 import _node_lower_bound,_node_upper_bound
    from dpmarg.kernel1d import lnJ_pair
    # Independent direct quadrature checks both signs in clipped,
    # interior, degenerate and prior-dominated radial regimes.
    for hi in (10.,100.):
        for a,b in [(0.,0.),(0.,100.),(1.,.01),(-1.,3.),(30.,1.),(1000.,1e5),(1e5,10.)]:
            value=lnJ_pair(a,b,.1,hi)
            assert _node_lower_bound(a,b,.1,hi)<=value+1e-8
            assert value<=_node_upper_bound(a,b,.1,hi)+1e-8


@pytest.mark.parametrize('hi',[10.,100.])
@pytest.mark.parametrize('a,b',[(0.,0.),(18.67935156876167,9.320033921747317),
    (.2944508851254676,2.5622653524820953e-6),(-1.,3.),(100.,1e8)])
def test_guard_quadrature_against_brute_force(a,b,hi):
    from dpmarg.table1d import _guard_integral
    from dpmarg.reference import lnI_ref
    # Includes the worst unsafe-cell probes and an out-of-domain case
    # that must retain the general peak-adapted direct kernel.
    assert abs(_guard_integral(a,b,.1,hi)-lnI_ref(a,b,.1,hi))<1e-8


@pytest.mark.parametrize('pars,reference',[
    ((12.127897987120532,17.94545463672102,16779.477218086064,1469.1461293314687,17009.283537195697,.1,100.),-3.2983729139327584),
    ((-105.30477496728133,121.37576304308025,-10012.909673145161,-69662.87063147344,70378.78874163698,.1,100.),-4.046190075607175),
    ((-22.403030350525526,34.65554624830968,3615.579325325494,18359.90889218251,18969.44030723495,.1,100.),-1.1459222014114534),
    ((-16.03018952786242,18.00932199493541,-3625.858629241456,3230.012313765287,4920.653224806344,.1,100.),-1.1906396258582124),
])
def test_clipped_weak_axis_resolution(tables,pars,reference):
    # Wider geometry audit: a clipping maximum can hide mass from larger
    # radii, or a narrow island can sit on a much broader prior pedestal.
    assert abs(lnL_marg_m2_table(*pars,tables[100.])-reference)<1e-4


# Independent lnL_marg_ref values from the 600-point corner audit.
@pytest.mark.parametrize('pars,reference',[
    ((477668.244562803, 147760.10333066978, 4126265.406740937, 2822930.045738479, 5000500.0, 0.1, 100.0),12488.761805681332),  # grid_260
    ((-73777.70696715605, 245377.32158196907, -4010851.1504504555, -2651600.6426138785, 4808644.649577866, 0.1, 100.0),3403.81125058612),  # random_132
    ((5775.976882604925, 32419.96196293449, -253128.03847387765, 93151.9231812871, 269724.0898659319, 0.1, 100.0),1000.4537541393253),  # random_22
    ((-129559.1545731471, 28887.770902850487, 3109278.56285459, -1459089.4274873696, 3434611.3519337997, 0.1, 100.0),1276.4684384058235),  # random_225
    ((477668.244562803, 147760.10333066978, 4126673.947870317, 2823209.54376281, 5000005.0, 0.1, 100.0),12493.743000583787),  # grid_240
    ((4775.2024564999665, 1482.3769761557571, 41266.739478703166, 28232.095437628097, 50000.05, 0.1, 100.0),220.0870401049465),  # grid_148
    ((-1304.5995864824959, 1934.2491834463826, -5697.433529663651, -14075.754511019664, 15185.11158600509, 0.1, 100.0),124.01760223089116),  # random_18
    ((4776.534661641534, 1478.0786941624594, 41266.78074548391, 28232.123669751767, 50000.0, 0.1, 100.0),140.29071021593975),  # grid_104
    ((4776.534661641534, 1478.0786941624594, 41266.78033281611, 28232.123387430533, 50000.0005, 0.1, 100.0),135.53562734979113),  # grid_124
    ((286.51214738999795, 88.94261856934543, 412.6678074548392, 282.3212366975177, 500.0, 0.1, 100.0),43.229432305961254),  # grid_9
])
def test_degenerate_worst_regressions(tables,pars,reference):
    for rule in ('analytic','scan'):
        assert abs(lnL_marg_m2_table(*pars,tables[100.],rule=rule)-reference)<1e-4


@pytest.mark.parametrize('lam1,ratio,reference',[
    (1000.0,0.05,-0.2390748876667314),
    (1000.0,0.3,40.04039554351209),
    (1000.0,3.0,4485.712385243136),
    (100000.0,0.05,121.12635857561841),
    (100000.0,0.3,4492.620140825587),
    (100000.0,3.0,449983.4086994701),
    (10000000.0,0.05,12493.82305786679),
    (10000000.0,0.3,449990.31645505293),
    (10000000.0,3.0,44999981.10610337),
])
def test_flat_ridge_family(tables,lam1,ratio,reference):
    pars=(*expand_params(lam1,0.,lam1*ratio,0.,xi=.3),.1,100.)
    for rule in ('analytic','scan'):
        assert abs(lnL_marg_m2_table(*pars,tables[100.],rule=rule)-reference)<1e-4
    centers,widths,_=analytic_geometry(lam1,0.,lam1*ratio,0.,.1,100.)
    # No vanishing-slope xmax candidate survives near the null axis.
    assert np.all(abs(centers-np.pi/2)>.01)


def test_clipping_cusp_width_and_non_gaussian_rule():
    from dpmarg.geometry import clipping_width
    from dpmarg.method2 import _analytic_plan
    l1,R=1e7,5e5
    theta=math.acos(R/(l1*.1))
    expected=math.sqrt(l1)/(R*math.tan(theta))
    width,kink=clipping_width(theta,l1,0.,R,0.,.1,100.)
    assert kink and width==pytest.approx(expected,rel=1e-10)
    centers,widths,_=analytic_geometry(l1,0.,R,0.,.1,100.)
    assert len(centers)==2
    assert widths==pytest.approx([expected,expected],rel=1e-10)
    plan=_analytic_plan(l1,0.,R,0.,.1,100.)
    assert plan[3]==5 and plan[4]==64
    # Exactly zero a,b has finite radial mass and no cusp scale.
    assert clipping_width(math.pi/2,l1,0.,0.,0.,.1,100.)==(.3,False)


def test_prior_dominated_formal_cusp(tables):
    # Geometry audit seed 903 #6557: b is too small for an isolated radial
    # saddle. Its formal clipping width must not force a sharp patch.
    pars=(-.3055154665099955,-.004733447176091943,
          4.125718718939891e-5,-3.146214240100136e-5,
          5.477257807442806e-5,.1,100.)
    assert abs(lnL_marg_m2_table(*pars,tables[100.])-4.638091010768807)<1e-4


@pytest.mark.parametrize('lam1,R,reference',[
    (1000.0, 1.0, -1.622464533703599),
    (10000000.0, 0.1, -6.269932822698221),
    (10000000.0, 1.0, -6.269932773197867),
    (10000000.0, 10.0, -6.269927823165107),
])
def test_null_axis_below_unit_snr(tables,lam1,R,reference):
    centers,widths,_=analytic_geometry(lam1,0.,R,0.,.1,100.)
    moment2=3*.1**2*(1-.1/100)/(1-(.1/100)**3)
    expected=min(.3,1/math.sqrt((lam1-R*R)*moment2))
    assert centers==pytest.approx([np.pi/2])
    assert widths==pytest.approx([expected])
    pars=(*expand_params(lam1,0.,R,0.),.1,100.)
    for rule in ('analytic','scan'):
        assert abs(lnL_marg_m2_table(*pars,tables[100.],rule=rule)-reference)<1e-4
