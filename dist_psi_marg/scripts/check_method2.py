#!/usr/bin/env python
"""Compare direct/table angular quadrature using only the saved references.

All iteration-003 regression/random points and node budgets are reused.
Per-point records go to method2_check_full.json; method2_check.json is small.
Timings are medians after JIT warm-up, with Numba loops avoiding dispatcher
cost. Component timings reuse the actual signed distance arguments.
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
from numba import njit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dpmarg.method2 import (_compute,_geometry,lnL_marg_m2_array,
                           lnL_marg_m2_table_array,method2_table_diagnostics)
from dpmarg.params import reduce_params,lnL_max
from dpmarg.kernel1d import lnI_direct_array
from dpmarg.table1d import load_table,lnI_table_array
from check_table1d import collect_nodes

KEYS=('A','B','C','D','E','xmin','xmax')
NODES=(16,24,32,48,64,96,128)


def stats(errors):
    return dict(count=len(errors),max=float(np.max(errors)),
                p99=float(np.percentile(errors,99)),median=float(np.median(errors)))


def describe(c):
    l1,l2,R,eta=reduce_params(*(c[k] for k in KEYS[:5]))
    peak,r,_=lnL_max(*(c[k] for k in KEYS));rho=math.sqrt(2*max(0,peak));q=l2/l1 if l1 else 1.
    return dict(snr_regime='low' if rho<4 else 'intermediate' if rho<20 else 'high',
                q_regime='q=0' if q<1e-12 else '0<q<0.01' if q<.01 else
                  '0.01<=q<0.5' if q<.5 else '0.5<=q<1' if q<1-1e-12 else 'q=1',
                peak_regime='below' if r<=c['xmin'] else 'above' if r>=c['xmax'] else 'inside')


@njit(cache=True)
def geometry_repeated(pars,lo,hi,repeats):
    total=0.
    for _ in range(repeats):
        c,w,b=_geometry(pars[0],pars[1],pars[2],pars[3],lo,hi)
        total+=c[0]+w[0]+b[0]
    return total


@njit(cache=True)
def direct_repeated(p,n,repeats):
    total=0.
    for _ in range(repeats):
        total+=lnL_marg_m2_array(p,n)[0]
    return total


@njit(cache=True)
def table_repeated(p,tab,n,repeats):
    total=0.
    for _ in range(repeats):
        total+=lnL_marg_m2_table_array(p,tab,n)[0]
    return total


@njit(cache=True)
def kernels_repeated(a,b,lo,hi,tab,use_table,repeats):
    total=0.
    for _ in range(repeats):
        vals=lnI_table_array(a,b,tab) if use_table else lnI_direct_array(a,b,lo,hi)
        total+=vals[0]
    return total


def elapsed_us(fn,*args):
    fn(*args)
    samples=[]
    for _ in range(3):
        start=time.perf_counter_ns();fn(*args)
        samples.append((time.perf_counter_ns()-start)/1000/args[-1])
    return float(np.median(samples))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-reference',action='store_true',help='compatibility option; references are always reused')
    parser.parse_args()
    cs=[dict(c,dataset='regression') for c in json.loads((ROOT/'data/regression_points.json').read_text())]
    cs += [dict(c,dataset='random') for c in json.loads((ROOT/'data/method2_random_reference.json').read_text())]
    for c in cs:c.update(describe(c));c['method2']={}
    ps=np.array([[c[k] for k in KEYS] for c in cs]);refs=np.array([c['lnL_marg_ref'] for c in cs])
    tables={hi:load_table(ROOT/f'data/table1d_xmin0.1_xmax{hi:g}.npz') for hi in (100.,10.)}
    result=dict(reference_policy='saved regression_points.json and method2_random_reference.json; no recomputation',
                geometry=dict(scan=64,refinement_iterations=28,one_geometry_per_call=True,
                              precomputed_legendre_nodes=[32,64]),nodes={})
    for n in NODES:
        direct=lnL_marg_m2_array(ps,n);values=np.empty(len(ps))
        for hi,tab in tables.items():
            mask=ps[:,6]==hi;values[mask]=lnL_marg_m2_table_array(ps[mask],tab,n)
        row=dict(direct=stats(abs(direct-refs)),table=stats(abs(values-refs)),
                 table_minus_direct=stats(abs(values-direct)),by_regime={})
        for key in ('dataset','snr_regime','q_regime','peak_regime'):
            row['by_regime'][key]={}
            for label in sorted(set(c[key] for c in cs)):
                mask=np.array([c[key]==label for c in cs])
                row['by_regime'][key][label]=dict(direct=stats(abs(direct[mask]-refs[mask])),table=stats(abs(values[mask]-refs[mask])))
        for i,c in enumerate(cs):
            c['method2'][str(n)]=dict(direct=float(direct[i]),table=float(values[i]),
                                       direct_error=float(abs(direct[i]-refs[i])),table_error=float(abs(values[i]-refs[i])))
        result['nodes'][str(n)]=row
        print(f'N={n}: direct max {row["direct"]["max"]:.8g}; table max {row["table"]["max"]:.8g}; change {row["table_minus_direct"]["max"]:.8g}',flush=True)
    timings=[]
    for p,c in zip(ps,cs):
        tab=tables[p[6]];pp=p.reshape(1,7);pars=np.array(reduce_params(*p[:5]))
        a,b=collect_nodes(pp)
        geom=elapsed_us(geometry_repeated,pars,p[5],p[6],30)
        d=elapsed_us(direct_repeated,pp,64,10)
        t=elapsed_us(table_repeated,pp,tab,64,30)
        kd=elapsed_us(kernels_repeated,a,b,p[5],p[6],tab,False,10)
        kt=elapsed_us(kernels_repeated,a,b,p[5],p[6],tab,True,30)
        _,count,mode,nodes,weights,fb=method2_table_diagnostics(*p,tab)
        timing=dict(direct_us=d,table_us=t,geometry_us=geom,direct_kernel_us=kd,table_kernel_us=kt,
                    table_geometry_share=geom/t,table_kernel_share=kt/t,
                    table_kernel_ns_per_eval=1000*kt/count,direct_kernel_ns_per_eval=1000*kd/count,kernel_calls=int(count),fallbacks=int(fb),mode=int(mode))
        c['timing']=timing;timings.append(timing)
    result['timing']={key:float(np.median([t[key] for t in timings])) for key in timings[0]}
    result['fallback_rate']=sum(t['fallbacks'] for t in timings)/sum(t['kernel_calls'] for t in timings)
    result['fallbacks_per_call']=stats(np.array([t['fallbacks'] for t in timings]))
    baseline_path=ROOT/'data/method2_iteration003.npz'
    if baseline_path.exists():
        with np.load(baseline_path) as baseline:
            if np.array_equal(baseline['parameters'],ps):
                result['geometry_change_vs_iteration003']=stats(abs(lnL_marg_m2_array(ps)-baseline['values']))
    print(json.dumps({k:v for k,v in result.items() if k!='nodes'},indent=2),flush=True)
    (ROOT/'data/method2_check_full.json').write_text(json.dumps(cs,indent=2,allow_nan=False)+'\n')
    (ROOT/'data/method2_check.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    med=result['timing'];check=result['nodes']['64']
    lines=['# Method 2: direct versus lookup', '',
           'All 40 regression and 300 random reference values are reused unchanged.', '',
           '| Node budget | Direct worst log error | Table worst log error | Largest table change |',
           '|---:|---:|---:|---:|']
    for n,row in result['nodes'].items():
        lines.append(f"| {n} | {row['direct']['max']:.8g} | {row['table']['max']:.8g} | {row['table_minus_direct']['max']:.8g} |")
    lines += ['', f"Median time at 64 nodes: direct {med['direct_us']:.3f} us; table {med['table_us']:.3f} us.",
              f"Geometry: {med['geometry_us']:.3f} us ({100*med['table_geometry_share']:.1f}% median share); "
              f"table kernels: {med['table_kernel_us']:.3f} us ({100*med['table_kernel_share']:.1f}% median share).",
              f"Fallback rate: {100*result['fallback_rate']:.3f}% of radial evaluations; median "
              f"{result['fallbacks_per_call']['median']:g}, maximum {result['fallbacks_per_call']['max']:g} per call.", '',
              'Geometry uses one 64-point scan and 28 refinement iterations; the common Legendre rules are precomputed.',
              'Timing uses warmed repeated numba loops. Component timings are measured separately.',
              'Per-point records: method2_check_full.json (ignored). Small summary: method2_check.json.',
              'Table details and API: ../TABLE1D.md.', '']
    (ROOT/'data/method2_summary.md').write_text('\n'.join(lines))
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for variant in ('direct','table'):
        ax.semilogy(NODES,[result['nodes'][str(n)][variant]['max'] for n in NODES],'-o',label=variant)
    ax.axhline(1e-3,color='k',ls='--');ax.set(xlabel='paired theta node budget',ylabel='worst absolute log error');ax.legend()
    fig.savefig(ROOT/'figures/method2_error_vs_nodes.png',dpi=180);plt.close(fig)
    assert result['nodes']['64']['table']['max']<1e-3
    assert result['nodes']['64']['table_minus_direct']['max']<2e-4


if __name__=='__main__':
    main()
