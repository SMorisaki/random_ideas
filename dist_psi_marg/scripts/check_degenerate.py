#!/usr/bin/env python
"""Reproducible near-null-axis audit against cached lnL_marg_ref.

300 Cartesian grid points and 300 random points (seed 90092). Random
log10(lam1) is uniform on [3,7], rho=R/sqrt(lam1) on [1,300] in log;
q and eta have independent 20% atoms at zero. All priors are (0.1,100).
Use --baseline before a repair to save the original results separately.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dpmarg.params import expand_params, reduce_params
from dpmarg.reference import lnL_marg_ref
from dpmarg.method2 import method2_analytic_diagnostics, method2_table_diagnostics
from dpmarg.table1d import load_table
from check_method2b import repeated, stats
from final_benchmark import cost


def parameter_sets():
    sets={'grid':[], 'random':[]}
    for i,(l1,q,eta,ratio) in enumerate(itertools.product(
            (1e3,1e5,1e7),(0.,1e-8,1e-6,1e-4,1e-2),
            (0.,1e-4,1e-3,1e-2,.1),(.05,.3,3.,30.))):
        sets['grid'].append(dict(name=f'grid_{i}',lam1=l1,q=q,eta=eta,
            R_over_lam1=ratio,xi=.3,parameters=[*expand_params(l1,l1*q,l1*ratio,eta,.3),.1,100.]))
    rng=np.random.default_rng(90092)
    for i in range(300):
        l1=10**rng.uniform(3,7)
        q=0. if rng.random()<.2 else 10**rng.uniform(-9,-2)
        eta=0. if rng.random()<.2 else 10**rng.uniform(-4,np.log10(.3))
        rho=10**rng.uniform(0,np.log10(300));xi=rng.uniform(0,np.pi)
        sets['random'].append(dict(name=f'random_{i}',lam1=l1,q=q,eta=eta,
            rho=rho,R_over_lam1=rho/np.sqrt(l1),xi=xi,
            parameters=[*expand_params(l1,l1*q,rho*np.sqrt(l1),eta,xi),.1,100.]))
    return sets


def references(sets):
    path=ROOT/'data/degenerate_reference_full.json'
    cache=json.loads(path.read_text()) if path.exists() else {}
    for group in sets.values():
        for i,c in enumerate(group):
            key=json.dumps(c['parameters'])
            if key not in cache:
                cache[key]=lnL_marg_ref(*c['parameters'])
                if i%10==0:
                    path.write_text(json.dumps(cache)+'\n')
                    print('reference',c['name'],flush=True)
            c['reference']=cache[key]
    path.write_text(json.dumps(cache)+'\n')


def aggregate(rows):
    return {rule:dict(**stats([r[rule+'_error'] for r in rows],
                [r['timing_us'][rule] for r in rows]),
            branches={str(m):sum(r[rule+'_branch']==m for r in rows)/len(rows)
                for m in ((3,4,5,6) if rule=='analytic' else (0,1,2,3,4,5,6))})
        for rule in ('analytic','scan')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',action='store_true')
    args=parser.parse_args();start=time.perf_counter()
    sets=parameter_sets();references(sets)
    tab=load_table(ROOT/'data/table1d_xmin0.1_xmax100.npz')
    full={};summary=dict(seed=90092,random_lam1_log10_range=[3,7],
        rho_definition='R/sqrt(lam1)',sets={},
        scan_corner_rule='seeded legacy scan geometry with shared fixed quadrature; other regimes use legacy nested quadrature')
    for label,cases in sets.items():
        rows=[]
        for c in cases:
            p=np.array(c['parameters']);ref=c['reference'];r=dict(c)
            a=method2_analytic_diagnostics(*p,tab)
            s=method2_table_diagnostics(*p,tab,64,'scan')
            r.update(analytic=a[0],analytic_error=abs(a[0]-ref),analytic_branch=int(a[1]),
                analytic_nodes=int(a[2]),width=a[3],scan=s[0],scan_error=abs(s[0]-ref),scan_branch=int(s[2]),
                timing_us={rule:cost(repeated,(p,tab,rule),16) for rule in ('analytic','scan')})
            rows.append(r)
        full[label]=rows;summary['sets'][label]=aggregate(rows)
        print(label,json.dumps(summary['sets'][label]),flush=True)
    summary['failures']={rule:[{k:r[k] for k in ('name','lam1','q','eta','R_over_lam1',rule+'_error')}
        for rows in full.values() for r in rows if r[rule+'_error']>1e-4] for rule in ('analytic','scan')}
    summary['elapsed_s']=time.perf_counter()-start
    prefix='degenerate_before' if args.baseline else 'degenerate_check'
    (ROOT/f'data/{prefix}_full.json').write_text(json.dumps(full,indent=2)+'\n')
    if args.baseline:
        (ROOT/'data/degenerate_before_summary_full.json').write_text(json.dumps(summary,indent=2)+'\n')
    else:
        before=ROOT/'data/degenerate_before_summary_full.json'
        if before.exists():summary['before']=json.loads(before.read_text())['sets']
        (ROOT/'data/degenerate_check.json').write_text(json.dumps(summary,indent=2)+'\n')
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for label,rows in full.items():
            for rule in ('analytic','scan'):
                errors=np.sort([r[rule+'_error'] for r in rows])
                axes[0].semilogx(np.maximum(errors,1e-12),np.arange(1,len(rows)+1)/len(rows),label=f'{label} {rule}')
                times=np.sort([r['timing_us'][rule] for r in rows])
                axes[1].semilogx(times,np.arange(1,len(rows)+1)/len(rows),label=f'{label} {rule}')
        axes[0].axvline(1e-4,color='gray',ls=':')
        axes[0].set(xlabel='Absolute log error',ylabel='Cumulative fraction')
        axes[1].set(xlabel='Warmed cost (us)',ylabel='Cumulative fraction');axes[1].legend()
        fig.savefig(ROOT/'figures/degenerate_check.png',dpi=160);plt.close(fig)
        if any(summary['sets'][s][r]['above_1e3'] for s in sets for r in ('analytic','scan')):
            raise SystemExit('degenerate 1e-3 target failed; see per-point dump')


if __name__=='__main__':main()
