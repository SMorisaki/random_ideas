#!/usr/bin/env python
"""Validate method 2 on saved regression values and 300 fresh references.

Seeds 2,3,4 each contribute 100 points; seed 3 uses the secondary xmax=10
prior and the others xmax=100. --reuse-reference is a development shortcut;
the default always recomputes all 300 independent SciPy reference values.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, set_num_threads

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpmarg.kernel1d import lnJ_pair
from dpmarg.method2 import (_compute, _geometry, _proxy, lnL_marg_m2,
                            lnL_marg_m2_array)
from dpmarg.params import expand_params, lnL_max, reduce_params
from dpmarg.reference import lnL_marg_ref
from dpmarg.regimes import random_params

KEYS = ('A', 'B', 'C', 'D', 'E', 'xmin', 'xmax')
NODES = (16, 24, 32, 48, 64, 96, 128)


def stats(errors):
    return dict(count=len(errors), max=float(np.max(errors)),
                p99=float(np.percentile(errors, 99)), median=float(np.median(errors)))


def describe(case):
    args = [case[k] for k in KEYS]
    l1, l2, R, eta = reduce_params(*args[:5])
    peak, r, _ = lnL_max(*args)
    rho = math.sqrt(2*max(0., peak))
    q = l2/l1 if l1 else 1.
    distance = ('below' if r <= case['xmin'] else
                'above' if r >= case['xmax'] else 'inside')
    return dict(actual_snr=rho, q=q, radial_peak=r,
                snr_regime='low' if rho < 4 else 'intermediate' if rho < 20 else 'high',
                q_regime='q=0' if q < 1e-12 else '0<q<0.01' if q < .01 else
                '0.01<=q<0.5' if q < .5 else '0.5<=q<1' if q < 1-1e-12 else 'q=1',
                peak_regime=distance)


@njit(cache=True)
def diagnostics(parameters, n):
    counts = np.empty(len(parameters), dtype=np.int64)
    modes = np.empty(len(parameters), dtype=np.int64)
    theta_counts = np.empty(len(parameters), dtype=np.int64)
    for i in range(len(parameters)):
        p = parameters[i]
        _, counts[i], modes[i], nodes, _ = _compute(p[0], p[1], p[2], p[3], p[4], p[5], p[6], n)
        theta_counts[i] = len(nodes)
    return counts, modes, theta_counts


def fixed_integral(pars, lo, hi, nodes, weights):
    l1,l2,R,eta = pars
    logs = np.array([lnJ_pair(R*math.cos(t-eta),l2+(l1-l2)*math.cos(t)**2,lo,hi)
                     for t in nodes])
    return (math.log(3/(2*math.pi))+3*math.log(lo)-math.log1p(-(lo/hi)**3)
            +logs.max()+math.log(np.dot(weights,np.exp(logs-logs.max()))))


def design_comparison(cases):
    """Forced uniform/sinh rules plus the difficult boundary two-peak case."""
    grouped = {name: dict(uniform=[], single_sinh=[], endpoint_correction_effect=[], adaptive=[])
               for name in ('broad','single_sharp','two_peaks')}
    n = 64
    for c in cases:
        pars = reduce_params(*(c[k] for k in KEYS[:5]))
        lo,hi = c['xmin'],c['xmax']
        cs,ws,_ = _geometry(*pars,lo,hi)
        uniform = fixed_integral(pars,lo,hi,np.arange(n)*math.pi/n,np.full(n,math.pi/n))
        j = int(np.argmax([_proxy(t,*pars,lo,hi) for t in cs]))
        T = math.asinh(math.pi/(2*ws[j]));dt=2*T/n
        tau = -T+np.arange(n)*dt
        nodes = cs[j]+ws[j]*np.sinh(tau)
        weights = dt*ws[j]*np.cosh(tau)
        uncorrected = fixed_integral(pars,lo,hi,nodes,weights)
        weights[0] -= dt*dt*math.pi/12
        sinh = fixed_integral(pars,lo,hi,nodes,weights)
        mode = c['method2']['64']['mode']
        row = grouped[('broad','single_sharp','two_peaks')[mode]]
        row['uniform'].append(abs(uniform-c['lnL_marg_ref']))
        row['single_sinh'].append(abs(sinh-c['lnL_marg_ref']))
        row['endpoint_correction_effect'].append(abs(sinh-uncorrected))
        row['adaptive'].append(c['method2']['64']['abs_error'])
    output = {name:{key:stats(values) for key,values in row.items()} for name,row in grouped.items()}
    pars=(1e6,1.,5000.,0.)
    coeff=expand_params(*pars)
    ref=lnL_marg_ref(*coeff,.1,100.)
    cs,ws,bs=_geometry(*pars,.1,100.)
    trap_nodes=[];trap_weights=[];corrected_weights=[]
    for center,w,left,right in zip(cs,ws,bs[:-1],bs[1:]):
        tau=np.linspace(-np.arcsinh((center-left)/w),np.arcsinh((right-center)/w),33)
        dt=tau[1]-tau[0]
        weights=dt*w*np.cosh(tau);weights[[0,-1]]*=.5
        corrected=weights.copy()
        # Compare the first EM correction with the original reused-node
        # fourth-order endpoint derivative stencil (no new kernels).
        for k,fac in enumerate((-25.,48.,-36.,16.,-3.)):
            corrected[k]+=dt*fac/144*w*np.cosh(tau[k])
            corrected[-k-1]+=dt*fac/144*w*np.cosh(tau[-k-1])
        trap_nodes.extend(center+w*np.sinh(tau));trap_weights.extend(weights)
        corrected_weights.extend(corrected)
    output['boundary_bimodal_study']=dict(parameters=list(coeff)+[.1,100.],reference=ref,
        split_trapezoid_error=abs(fixed_integral(pars,.1,100.,trap_nodes,trap_weights)-ref),
        split_corrected_trapezoid_error=abs(fixed_integral(pars,.1,100.,trap_nodes,corrected_weights)-ref),
        split_sinh_legendre_error=abs(lnL_marg_m2(*coeff,.1,100.,64)-ref),
        explanation='Valley endpoint layers defeat the reused-node derivative stencil; '
                    'Legendre nodes cluster at asymmetric sinh endpoints without extra kernels.')
    return output


def write_summary(result):
    lines=['Method 2: measured absolute log errors (max / 99% / median).','',
           'Default budget: 64 paired theta nodes; early stopping tolerance 2e-6.',
           'Mode 0: periodic uniform; mode 1: corrected sinh trapezoid; '
           'mode 2: Gauss–Legendre on two asymmetric sinh intervals.','',
           '| Budget | Overall max / 99% / median | Median µs | Kernel calls median / max | Theta nodes median / max |',
           '|---:|---|---:|---|---|']
    for n,row in result['nodes'].items():
        s=row['overall'];k=row['kernel_evaluations'];t=row['theta_nodes']
        lines.append(f'| {n} | {s["max"]:.6g} / {s["p99"]:.6g} / {s["median"]:.6g} | '
                     f'{row["median_us_per_call"]:.2f} | {k["median"]:g} / {k["max"]} | {t["median"]:g} / {t["max"]} |')
    for key in ('dataset','snr_regime','q_regime','peak_regime'):
        lines.extend(['',f'{key}: max / 99% / median','',
                      '| Regime | '+' | '.join(str(n) for n in NODES)+' |',
                      '|---|'+'---|'*len(NODES)])
        for label in result['nodes']['64']['by_regime'][key]:
            cells=[]
            for n in NODES:
                s=result['nodes'][str(n)]['by_regime'][key][label]
                cells.append(f'{s["max"]:.3g} / {s["p99"]:.3g} / {s["median"]:.3g}')
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines.extend(['','The random set contains 100 points for each of seeds 2, 3, 4; '
                  'seed 3 uses xmax=10, the others xmax=100. SNR bins use actual annular SNR: '
                  'low <4, intermediate [4,20), high >=20. Peak bins refer to the clipped '
                  'radial maximum of the full likelihood.','',
                  'Timing is the median of individual scalar wall times over three warmed passes '
                  '(single-thread numba); the JSON also records median batch means.','',
                  'Peak geometry uses the clipped radial exponent maximum minus 4 log(r*), '
                  'a 128-angle scan, and 40 golden-section iterations. At most two maxima '
                  'within 40 log units are retained. Curvature sets w=min(0.3,1/sqrt(abs(p\'\'))); '
                  'split intervals use min(0.3,1.5*w). The width switch is w=0.22.','',
                  'For a single symmetric sinh interval, periodicity gives the exact first '
                  'Euler–Maclaurin correction -pi*dtau^2*h(endpoint)/12. Its maximum effect '
                  'among the single-sharp cases is '
                  f'{result["design_comparison"]["single_sharp"]["endpoint_correction_effect"]["max"]:.6g} '
                  'in log likelihood. The correction is included even when negligible.','',
                  'Ablation at 64 nodes: forced uniform / forced single sinh / adaptive, maximum errors:'])
    for label in ('broad','single_sharp','two_peaks'):
        r=result['design_comparison'][label]
        lines.append(f'- {label}: {r["uniform"]["max"]:.6g} / {r["single_sinh"]["max"]:.6g} / {r["adaptive"]["max"]:.6g}')
    b=result['design_comparison']['boundary_bimodal_study']
    lines.extend(['',f'Additional boundary bimodal case: split trapezoid error '
                  f'{b["split_trapezoid_error"]:.6g}; with reused-node first EM correction '
                  f'{b["split_corrected_trapezoid_error"]:.6g}; split sinh–Legendre '
                  f'{b["split_sinh_legendre_error"]:.6g}.', '',b['explanation'],''])
    worst=sorted(result['cases'],key=lambda c:c['method2']['64']['abs_error'],reverse=True)[:5]
    lines.extend(['Largest errors at the 64-node budget:',''])
    for c in worst:
        lines.append(f'- {c["dataset"]}/{c["name"]}: '
                     f'{c["method2"]["64"]["abs_error"]:.6g}; actual SNR={c["actual_snr"]:.6g}, '
                     f'q={c["q"]:.6g}, radial peak {c["peak_regime"]}, '
                     f'mode={c["method2"]["64"]["mode"]}.')
    lines.extend(['','Degenerate weak-axis profiles and radial endpoint transitions dominate '
                  'the residual angular discretization error. The hardest upper-endpoint '
                  'cases reach actual SNR >1300, despite sampling R/sqrt(lam1)<=300. '
                  'The reference values for regression points are loaded unchanged; '
                  'only the 300 random references and the additional boundary study are computed.',''])
    (ROOT/'data/method2_summary.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-reference', action='store_true')
    options = parser.parse_args()
    set_num_threads(1)
    regression = json.loads((ROOT/'data/regression_points.json').read_text())
    cache = ROOT/'data/method2_random_reference.json'
    if options.reuse_reference and cache.exists():
        random = json.loads(cache.read_text())
    else:
        random = []
        for seed in (2, 3, 4):
            for i, c in enumerate(random_params(100, seed, xmax=10. if seed == 3 else 100.)):
                c['name'] = f'seed{seed}_{i:03d}'
                c['seed'] = seed
                c['lnL_marg_ref'] = lnL_marg_ref(*(c[k] for k in KEYS))
                random.append(c)
            print(f'Computed 100 reference values for seed {seed}.', flush=True)
        cache.write_text(json.dumps(random, indent=2, allow_nan=False)+'\n')
    cases = [dict(c, dataset='regression') for c in regression]
    cases += [dict(c, dataset='random') for c in random]
    for c in cases:
        c.update(describe(c))
    parameters = np.array([[c[k] for k in KEYS] for c in cases])
    reference = np.array([c['lnL_marg_ref'] for c in cases])
    result = dict(random_seeds=[2, 3, 4], random_points_per_seed=100,
                  random_bounds=[[.1,100.], [.1,10.], [.1,100.]],
                  design=dict(proxy_scan=128,peak_refinement_iterations=40,
                              secondary_peak_log_gap=40,width_switch=.22,
                              curvature_width_cap=.3,split_width_scale=1.5,
                              nested_convergence_atol=2e-6),
                  snr_bins='low <4; intermediate [4,20); high >=20 (actual annular SNR)',
                  timing='single-thread scalar: median individual wall time over 3 warmed passes; '
                         'batch: median of 5 warmed batch means',
                  nodes={}, cases=cases)
    allerrors = []
    for n in NODES:
        values = lnL_marg_m2_array(parameters, n)
        counts, modes, theta_counts = diagnostics(parameters, n)
        errors = abs(values-reference)
        allerrors.append(errors)
        samples = []
        for _ in range(5):
            start = time.perf_counter_ns()
            lnL_marg_m2_array(parameters, n)
            samples.append((time.perf_counter_ns()-start)/len(cases)/1000)
        lnL_marg_m2(*parameters[0],n)
        scalar_samples=[]
        for _ in range(3):
            for p in parameters:
                start=time.perf_counter_ns()
                lnL_marg_m2(*p,n)
                scalar_samples.append((time.perf_counter_ns()-start)/1000)
        grouped = {}
        for key in ('dataset', 'snr_regime', 'q_regime', 'peak_regime'):
            grouped[key] = {}
            for label in sorted(set(c[key] for c in cases)):
                mask = np.array([c[key] == label for c in cases])
                grouped[key][label] = stats(errors[mask])
        worst = int(np.argmax(errors))
        row = dict(overall=stats(errors), by_regime=grouped,
                   median_us_per_call=float(np.median(scalar_samples)),
                   median_batch_us_per_call=float(np.median(samples)),
                   theta_nodes=dict(min=int(theta_counts.min()),median=float(np.median(theta_counts)),
                                    max=int(theta_counts.max())),
                   kernel_evaluations=dict(min=int(counts.min()), median=float(np.median(counts)),
                                           p99=float(np.percentile(counts,99)), max=int(counts.max())),
                   modes={str(i): int(np.sum(modes == i)) for i in (0,1,2)},
                   worst_case=cases[worst]['name'])
        result['nodes'][str(n)] = row
        for c, value, error, count, mode, nt in zip(cases, values, errors, counts, modes, theta_counts):
            c.setdefault('method2', {})[str(n)] = dict(value=float(value), abs_error=float(error),
                                                       kernel_evaluations=int(count), mode=int(mode),
                                                       theta_nodes=int(nt))
        print(f'N={n:3d} max/p99/median={errors.max():.6g}/'
              f'{np.percentile(errors,99):.6g}/{np.median(errors):.6g}; '
              f'{row["median_us_per_call"]:.2f} us; kernels '
              f'{np.median(counts):g} median, {counts.max()} max; worst={cases[worst]["name"]}', flush=True)
        for key, groups in grouped.items():
            print(f'  {key}: '+ '; '.join(f'{label}: {s["max"]:.3g}/{s["p99"]:.3g}/{s["median"]:.3g}'
                                        for label,s in groups.items()), flush=True)
    errors = np.array(allerrors)
    fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
    for dataset in ('regression', 'random'):
        mask = np.array([c['dataset'] == dataset for c in cases])
        for quantile, label, style in ((100,'max','-o'),(99,'99%','--s'),(50,'median',':^')):
            ax.semilogy(NODES, np.maximum(1e-14,np.percentile(errors[:,mask],quantile,axis=1)),
                        style, label=f'{dataset}: {label}')
    ax.axhline(1e-3,color='k',lw=.8,label='requirement')
    ax.set(xlabel='Paired theta nodes',ylabel='Absolute error in ln L_marg')
    ax.legend(fontsize=8)
    fig.savefig(ROOT/'figures/method2_error_vs_nodes.png',dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5), layout='constrained')
    artist = ax.scatter([c['actual_snr'] for c in cases],np.maximum(errors[4],1e-14),
                        c=[c['q'] for c in cases],cmap='viridis',s=22)
    ax.set(xscale='symlog',yscale='log',xlabel='Actual annular SNR',
           ylabel='Absolute error in ln L_marg (64 nodes)')
    ax.axhline(1e-3,color='k',lw=.8)
    fig.colorbar(artist,ax=ax,label='lambda2 / lambda1')
    fig.savefig(ROOT/'figures/method2_error_scatter.png',dpi=180)
    plt.close(fig)
    result['design_comparison']=design_comparison(cases)
    (ROOT/'data/method2_check.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_summary(result)
    assert result['nodes']['64']['by_regime']['dataset']['regression']['max'] <= 1e-3
    assert result['nodes']['64']['by_regime']['dataset']['random']['max'] <= 1e-3


if __name__ == '__main__':
    main()
