#!/usr/bin/env python
"""Shared-set accuracy and warmed single-thread cost comparison.

Reference caches are coefficient-keyed and restartable. Timings are per-point
medians of three compiled repeated loops; reported p99 is across point costs,
not the latency p99 of a running sampler. Missing method-1 prototype results
are explicitly unavailable; regenerate it with build_table4d.py if desired.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import warnings

# Set before importing numerical libraries; no parallel kernels are used.
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numba
from numba import njit
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dpmarg import table1d, table4d
from dpmarg.fast import (lnL_marg_fast, fast_array_diagnostics, laplace_gate, LAPLACE_MARGIN,
                         LAPLACE_Q_KNOTS, LAPLACE_RHO_BREAK, LAPLACE_INTERIOR_SIGMA)
from dpmarg.method2 import lnL_marg_m2, lnL_marg_m2_table
from dpmarg.method3 import lnL_marg_m3_2d
from dpmarg.params import expand_params
from dpmarg.reference import lnL_marg_ref
from dpmarg.regimes import random_params

KEYS = ('A', 'B', 'C', 'D', 'E', 'xmin', 'xmax')
METHODS = ('m2_direct', 'm2_table', 'm3_2d_o4', 'fast', 'fast_no_laplace', 'm1_cubic')
CACHE = ROOT/'data/final_reference_full.json'


def write(path, value):
    # Atomic replacement keeps a long reference pass restartable.
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def posterior_points(n=2000, seed=809):
    """Synthetic interior peaks with actual rho log-uniform [5,50].

    q~U(.05,1), eta~U(0,pi/2), rbar log-uniform [.5,5], xi~U(0,pi).
    Given k=cos²eta+sin²eta/q and h=hypot(cos eta,sin eta/q), set
    lam1=(rho*h/(rbar*sqrt(k)))², lam2=q*lam1,
    R=rho*sqrt(lam1/k). Thus rho and rbar are exact Cartesian-peak values.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        rho = math.exp(rng.uniform(math.log(5), math.log(50)))
        q = rng.uniform(.05, 1.)
        eta = rng.uniform(0, math.pi/2)
        radius = math.exp(rng.uniform(math.log(.5), math.log(5)))
        xi = rng.uniform(0, math.pi)
        k = math.cos(eta)**2+math.sin(eta)**2/q
        h = math.hypot(math.cos(eta), math.sin(eta)/q)
        l1 = (rho*h/(radius*math.sqrt(k)))**2
        R = rho*math.sqrt(l1/k)
        out.append(dict(name=f'posterior_{i:04d}', **dict(zip(KEYS,
            (*expand_params(l1, q*l1, R, eta, xi), .1, 100.)))))
    return out


def parameter_sets():
    return dict(regression=json.loads((ROOT/'data/regression_points.json').read_text()),
                stored_random=json.loads((ROOT/'data/method2_random_reference.json').read_text()),
                fresh_random=random_params(2000, 808), sampler_like=posterior_points())


def ref_key(c):
    return ','.join(float(c[k]).hex() for k in KEYS)


def references(sets):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else dict(
        schema=1, integrator='lnL_marg_ref default rtol=1e-10', points={})
    if cache['schema'] != 1:
        raise ValueError('unknown reference cache schema')
    added = 0
    for label, cases in sets.items():
        for i, c in enumerate(cases):
            if 'lnL_marg_ref' in c:
                continue
            key = ref_key(c)
            if key not in cache['points']:
                start = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    value = lnL_marg_ref(*(c[k] for k in KEYS))
                cache['points'][key] = dict(value=value, seconds=time.perf_counter()-start,
                    dataset=label, name=c['name'], warnings=[str(w.message) for w in caught])
                added += 1
                if added % 25 == 0:
                    write(CACHE, cache)
                    print(f'Reference {label}: {i+1}/{len(cases)} ({added} new)', flush=True)
            c['lnL_marg_ref'] = cache['points'][key]['value']
    write(CACHE, cache)
    return cache


@njit(cache=True)
def evaluate(p, table, method):
    if method == 0:
        return lnL_marg_m2(p[0], p[1], p[2], p[3], p[4], p[5], p[6])
    if method == 1:
        return lnL_marg_m2_table(p[0], p[1], p[2], p[3], p[4], p[5], p[6], table)
    if method == 2:
        return lnL_marg_m3_2d(p[0], p[1], p[2], p[3], p[4], p[5], p[6], 4)
    return lnL_marg_fast(p[0], p[1], p[2], p[3], p[4], p[5], p[6], table, method == 3)


@njit(cache=True)
def repeated(p, table, method, repeats):
    total = 0.
    for _ in range(repeats):
        total += evaluate(p, table, method)
    return total


@njit(cache=True)
def repeated_m1(p, table, radial, repeats):
    total = 0.
    for _ in range(repeats):
        total += table4d.lnL_marg_m1(p[0], p[1], p[2], p[3], p[4], p[5], p[6], table, radial)
    return total


def cost(fn, args, repeats):
    samples = []
    for _ in range(3):
        start = time.perf_counter_ns()
        fn(*args, repeats)
        samples.append((time.perf_counter_ns()-start)/1000/repeats)
    return float(np.median(samples))


def statistics(values, refs, times):
    finite = np.isfinite(values)
    errors = abs(values[finite]-refs[finite])
    return dict(count=len(values), available=int(finite.sum()),
                available_fraction=float(finite.mean()),
                error=dict(max=float(errors.max()), p99=float(np.quantile(errors,.99)),
                           median=float(np.median(errors)),
                           fraction_gt_1e_minus3=float(np.mean(errors>1e-3)),
                           fraction_gt_1e_minus2=float(np.mean(errors>1e-2))) if len(errors) else None,
                timing_us=dict(median=float(np.median(times[finite])),
                               p99=float(np.quantile(times[finite],.99))) if finite.any() else None)


def markdown(result):
    lines = ['# Final shared-set benchmark', '',
        'Absolute log errors; error fractions and times use available outputs only. Availability is reported separately.',
        'Costs are median / p99 across per-point warmed cost estimates, single thread (three compiled repeated loops).',
        '', '| Set | Method | Available | Max error | 99% error | Median error | >1e-3 | >1e-2 | Median / 99% us |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, group in result['sets'].items():
        for method, row in group['methods'].items():
            e, t = row['error'], row['timing_us']
            fields = ([f'{e[k]:.6g}' for k in ('max','p99','median')]
                      + [f'{100*e[k]:.3f}%' for k in ('fraction_gt_1e_minus3','fraction_gt_1e_minus2')]
                      + [f"{t['median']:.4g} / {t['p99']:.4g}"]) if e else ['—']*6
            lines.append(f"| {label} | {method} | {row['available']}/{row['count']} | "+' | '.join(fields)+' |')
    lines += ['', 'Dispatcher branch fractions (without Laplace: 100% method 2):', '',
              '| Set | Method 2 | Laplace |', '|---|---:|---:|']
    for label, group in result['sets'].items():
        branches = group['dispatcher_branches']
        lines.append(f"| {label} | {branches['method2']:.3%} | {branches['laplace']:.3%} |")
    lines += ['', 'Table construction (JIT excluded, guards included):', '',
              '| xmax | Build s | Coefficient bytes | Guard bytes | Flagged cells |',
              '|---:|---:|---:|---:|---:|']
    for row in result['tables']:
        lines.append(f"| {row['xmax']:g} | {row['build_s']:.3f} | {row['coefficient_bytes']} | {row['guard_bytes']} | {row['guard_cell_fraction']:.3%} |")
    lines += ['', 'Fresh random: 2000 points from `random_params(2000,808)`; same broad recipe as the saved random sets.',
              'Sampler-like: 2000 points, seed 809; actual rho log-uniform [5,50], q uniform [.05,1], eta uniform [0,pi/2],',
              'Cartesian peak radius log-uniform [.5,5], rotation xi uniform [0,pi], bounds [.1,100]. See `posterior_points` for the exact transform.',
              'Method 3 is shown wherever its native three-sigma guard returns a finite value, including inaccurate low-SNR outputs.',
              'Method 1 is the rejected iteration-005 cubic prototype, only for its [.1,100] prior; the 100 secondary-prior stored points are unavailable.',
              'Per-point references, values, costs and branch codes are in ignored `*_full.json` files.']
    return '\n'.join(lines)+'\n'


def plots(result, full):
    fig, axes = plt.subplots(2,2, figsize=(12,8), constrained_layout=True)
    colors = dict(zip(METHODS, plt.cm.tab10.colors))
    for ax, (label, records) in zip(axes.flat, full.items()):
        for method in METHODS:
            errors = np.array([abs(c['values'][method]-c['reference'])
                               for c in records if c['values'][method] is not None])
            if not len(errors):
                continue
            errors.sort()
            ax.semilogx(np.maximum(errors,1e-13), np.arange(1,len(errors)+1)/len(records),
                        label=method, color=colors[method],
                        linestyle='--' if method=='fast_no_laplace' else '-')
        ax.axvline(1e-3,color='gray',linestyle=':')
        ax.set(title=label, xlabel='Absolute error in ln L_marg', ylabel='Fraction of all points', ylim=(0,1.01))
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(ROOT/'figures/final_error_cdf.png',dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,6),constrained_layout=True)
    markers = ['o','s','^','D']
    for marker, (label, group) in zip(markers,result['sets'].items()):
        for method, row in group['methods'].items():
            if row['error'] is None:
                continue
            ax.scatter(row['timing_us']['median'], max(row['error']['max'],1e-13),
                       edgecolors=[colors[method]], marker=marker, label=f'{method}: {label}', s=60,
                       facecolors='none' if method=='fast_no_laplace' else [colors[method]])
    ax.axhline(1e-3,color='gray',linestyle=':')
    ax.set(xscale='log',yscale='log',xlabel='Median cost (us / evaluation)',ylabel='Worst absolute log error')
    ax.legend(fontsize=7,ncol=2,loc='center left',bbox_to_anchor=(1, .5))
    ax.grid(alpha=.2)
    fig.savefig(ROOT/'figures/final_cost_vs_error.png',dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-only',action='store_true')
    args = parser.parse_args()
    start = time.perf_counter()
    sets = parameter_sets()
    cache = references(sets)
    print(f'References ready ({len(cache["points"])} cached).', flush=True)
    if args.reference_only:
        return
    tables = {}
    table_rows = []
    table1d.build_table(.1,100.,ns=8,nv=8)  # compile build and guard before timing
    for hi in (100.,10.):
        then = time.perf_counter()
        table = table1d.build_table(.1,hi)
        elapsed = time.perf_counter()-then
        path = ROOT/f'data/table1d_xmin0.1_xmax{hi:g}.npz'
        table1d.save_table(path,table)
        tables[hi] = table
        table_rows.append(dict(xmin=.1,xmax=hi,build_s=elapsed,
            coefficient_entries=int(table[0].size),coefficient_bytes=table[0].nbytes,
            guard_bytes=table[-1].nbytes,guard_cell_fraction=float(table[-1].mean()),
            sigma0=table[5],bref=table[6],smax=24.,bmax=table[9],path=str(path.relative_to(ROOT))))
        print(f'Table xmax={hi:g}: {elapsed:.2f}s',flush=True)
    prototype_path = ROOT/'data/table4d_xmin0.1_xmax100.npz'
    prototype = table4d.load_table(prototype_path) if prototype_path.exists() else None
    # Warm every timing specialization, including both dispatcher paths.
    p = np.array([10.,0.,0.,0.,1.,.1,100.])
    for m in range(5):
        repeated(p,tables[100.],m,1)
    if prototype is not None:
        repeated_m1(p,prototype,tables[100.],1)
    result = dict(schema=1,environment=dict(python=platform.python_version(),numpy=np.__version__,
        scipy=scipy.__version__,numba=numba.__version__,machine=platform.machine(),
        processor=platform.processor(),threads=1),
        timing_recipe='3 per-point compiled repeated loops; median of each, then median/p99 across available points; warmed; no concurrent experiment',
        reference_recipe='40+300 saved references; 2000 fresh seed 808 and 2000 mock-posterior seed 809 via lnL_marg_ref, rtol=1e-10',
        sampler_recipe=posterior_points.__doc__,gate=dict(margin=LAPLACE_MARGIN,
            q_knots=LAPLACE_Q_KNOTS.tolist(),rho_break=LAPLACE_RHO_BREAK.tolist(),
            interpolation='monotone log-log; q<.001 never; q>=.3 uses 18',interior_sigma=LAPLACE_INTERIOR_SIGMA),
        tables=table_rows,sets={},
        method1_prototype=dict(available=prototype is not None,
            bytes=prototype[0].nbytes if prototype is not None else None,
            build_s=json.loads((ROOT/'data/method1_check.json').read_text())['build_s'],
            build_policy='historical iteration-005 build; not rebuilt after negative slice verdict'))
    full = {}
    for label,cases in sets.items():
        n = len(cases)
        ps = np.array([[c[k] for k in KEYS] for c in cases])
        refs = np.array([c['lnL_marg_ref'] for c in cases])
        values = {method:np.full(n,np.nan) for method in METHODS}
        timings = {method:np.full(n,np.nan) for method in METHODS}
        branches = np.zeros(n,dtype=int)
        for hi,tab in tables.items():
            mask = ps[:,6]==hi
            if mask.any():
                _, branches[mask] = fast_array_diagnostics(ps[mask],tab)
        for m,method in enumerate(METHODS):
            for i,p in enumerate(ps):
                tab = tables[p[6]]
                if m == 5:
                    if prototype is None or p[6]!=100.:
                        continue
                    values[method][i] = table4d.lnL_marg_m1(*p,prototype,tab)
                    timings[method][i] = cost(repeated_m1,(p,prototype,tab),128)
                else:
                    values[method][i] = evaluate(p,tab,m)
                    if math.isfinite(values[method][i]):
                        timings[method][i] = cost(repeated,(p,tab,m),512 if m==2 else 16 if m==0 else 32)
            print(f'{label} {method}: '+str(statistics(values[method],refs,timings[method])),flush=True)
        np.testing.assert_array_equal(values['fast_no_laplace'],values['m2_table'])
        methods = {m:statistics(values[m],refs,timings[m]) for m in METHODS}
        result['sets'][label] = dict(count=n,methods=methods,
            dispatcher_branches=dict(method2=float(np.mean(branches==0)),laplace=float(np.mean(branches==1))),
            dispatcher_worst_case=cases[int(np.argmax(abs(values['fast']-refs)))]['name'])
        full[label] = [dict(name=c['name'],parameters=ps[i].tolist(),reference=float(refs[i]),
            branch=int(branches[i]),values={m:float(values[m][i]) if np.isfinite(values[m][i]) else None for m in METHODS},
            timing_us={m:float(timings[m][i]) if np.isfinite(timings[m][i]) else None for m in METHODS}) for i,c in enumerate(cases)]
    # Warnings are recorded, then cross-checked by high-budget method 2.
    warned = [r for r in cache['points'].values() if r['warnings']]
    differences = []
    for key,row in cache['points'].items():
        if row['warnings']:
            p = [float.fromhex(v) for v in key.split(',')]
            differences.append(abs(lnL_marg_m2(*p,n_nodes=256)-row['value']))
    result['reference_warnings'] = dict(count=len(warned),
        max_difference_from_m2_256=max(differences,default=0.))
    result['reference_cache'] = dict(points=len(cache['points']),
        total_integration_seconds=sum(r['seconds'] for r in cache['points'].values()))
    all_records = [r for records in full.values() for r in records]
    all_refs = np.array([r['reference'] for r in all_records])
    result['overall'] = {m:statistics(
        np.array([r['values'][m] if r['values'][m] is not None else np.nan for r in all_records]),
        all_refs,
        np.array([r['timing_us'][m] if r['timing_us'][m] is not None else np.nan for r in all_records]))
        for m in METHODS}
    result['overall_dispatcher_branches'] = dict(
        method2=float(np.mean([r['branch']==0 for r in all_records])),
        laplace=float(np.mean([r['branch']==1 for r in all_records])))
    result['validation'] = dict(
        fast_all_points_below_1e_minus3=result['overall']['fast']['error']['max']<1e-3,
        no_laplace_identical_to_m2_table=True)
    result['prior_sweep_gate_audit'] = prior_sweep_gate_audit()
    result['elapsed_s'] = time.perf_counter()-start
    write(ROOT/'data/final_benchmark_full.json',full)
    write(ROOT/'data/final_benchmark.json',result)
    (ROOT/'data/final_benchmark.md').write_text(markdown(result))
    plots(result,full)
    print(f'Completed in {time.perf_counter()-start:.1f}s',flush=True)


def prior_sweep_gate_audit():
    """Optionally check the gate against the saved iteration-006 sweep.

    Uses its independent reference errors for the unchanged order-4 formula.
    This is separate from the four final benchmark input sets and timings.
    """
    path = ROOT/'data/method3_check_full.json'
    if not path.exists():
        return None
    cases = json.loads(path.read_text())['sweep']
    accepted = [c for c in cases if laplace_gate(*(c[k] for k in KEYS))]
    errors = [c['errors']['2d_o4'] for c in accepted]
    return dict(source=str(path.relative_to(ROOT)), count=len(cases),
                accepted=len(accepted), max_absolute_log_error=max(errors,default=0.),
                count_gt_1e_minus3=sum(e>1e-3 for e in errors))


if __name__=='__main__':
    main()
