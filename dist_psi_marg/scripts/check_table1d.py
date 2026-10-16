#!/usr/bin/env python
"""Compare cubic remainders/raw lnI and bilinear on physical and stress inputs.

The saved small summary contains no per-point data. 100,000 actual method-2
kernel arguments are sampled from random_params(seed=42), separately for
each prior. Independent uniform-coordinate probes also inspect the guard.
Convergence candidates are unguarded to expose interpolation deficiencies.
"""
import json
from pathlib import Path
import sys
import time
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from numba import njit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dpmarg.kernel1d import lnI_direct_array, _lnJ_pair_count
from dpmarg.method2 import _compute
from dpmarg.params import reduce_params
from dpmarg.regimes import random_params
from dpmarg.table1d import (build_table,load_table,lnI_table_array,table_diagnostics,
                            coordinates,inverse_s,_endpoint_series,save_table)


@njit(cache=True)
def collect_nodes(parameters):
    """All actual signed kernel arguments, including the certified pair skip."""
    aa=np.empty(len(parameters)*128);bb=aa.copy();offset=0
    for p in parameters:
        _,_,_,nodes,_=_compute(p[0],p[1],p[2],p[3],p[4],p[5],p[6],64)
        l1,l2,R,eta=reduce_params(p[0],p[1],p[2],p[3],p[4])
        for theta in nodes:
            a=abs(R*math.cos(theta-eta));b=l2+(l1-l2)*math.cos(theta)**2
            calls=_lnJ_pair_count(a,b,p[5],p[6])[1]
            aa[offset]=a;bb[offset]=b;offset+=1
            if calls==2:
                aa[offset]=-a;bb[offset]=b;offset+=1
    return aa[:offset],bb[:offset]


def actual_node_distribution(hi,n=100000):
    cases=random_params(2400,42,xmax=hi)
    ps=np.array([[c[k] for k in ('A','B','C','D','E','xmin','xmax')] for c in cases])
    a,b=collect_nodes(ps)
    if len(a)<n:
        raise RuntimeError(f'only {len(a)} nodes; increase random parameter count')
    ids=np.random.default_rng(43).choice(len(a),n,replace=False)
    return a[ids],b[ids]


def summary(errors):
    return dict(count=len(errors),max=float(np.max(errors)),p99=float(np.percentile(errors,99)))


def labels(a,b,lo,hi):
    return np.where(b==0,'b = 0',np.where(a<b*lo,'x0 < xmin',np.where(a>b*hi,'x0 > xmax','inside')))


def grouped(errors,lab):
    return {key:summary(abs(errors[lab==key])) for key in np.unique(lab)}


def timing(a,b,tab):
    lnI_table_array(a,b,tab)
    samples=[]
    for _ in range(7):
        start=time.perf_counter_ns();lnI_table_array(a,b,tab)
        samples.append((time.perf_counter_ns()-start)/len(a))
    return float(np.median(samples))


def main():
    results=[];maps=[];curves=[];timing_inputs=[]
    build=json.loads((ROOT/'data/table1d_build.json').read_text())
    for hi in (100.,10.):
        lo=.1
        table=load_table(ROOT/f'data/table1d_xmin0.1_xmax{hi:g}.npz')
        a,b=actual_node_distribution(hi)
        aa=np.r_[-np.logspace(-4,7,23)[::-1],0.,np.logspace(-4,7,23)]
        bb=np.r_[0.,np.logspace(-8,8,33)]
        ga,gb=np.meshgrid(aa,bb);ga=ga.ravel();gb=gb.ravel()
        # More extensive independent probes uniform in s,v, including b=0
        # and the two clipped-peak transition curves.
        rng=np.random.default_rng(44)
        sb=rng.uniform(-24,24,100000); vb=rng.uniform(0,np.arcsinh(1e8/table[6]),len(sb))
        xb=table[6]*np.sinh(vb)
        xa=np.array([inverse_s(s,t,lo,hi,table[5]) for s,t in zip(sb,xb)])
        ab=np.r_[a,ga,xa];bs=np.r_[b,gb,xb]
        direct=lnI_direct_array(ab,bs,lo,hi)
        value,flags=table_diagnostics(ab,bs,table)
        error=value-direct
        lab=labels(ab,bs,lo,hi)
        # Distinguish the fast endpoint expansion from direct quadrature.
        fast=0
        for x,y,flag in zip(a,b,flags[:len(a)]):
            if flag and y<=table[9]:
                s,_=coordinates(x,y,lo,hi,table[5],table[6])
                if abs(s)>24 and np.isfinite(_endpoint_series(x,y,lo,hi)):
                    fast+=1
        row=dict(xmin=lo,xmax=hi,build=next(x for x in build if x['xmax']==hi),
                 physical_nodes=summary(abs(error[:len(a)])),
                 iteration002_grid=summary(abs(error[len(a):len(a)+len(ga)])),
                 uniform_coordinate_stress=summary(abs(error[-len(xa):])),
                 by_regime=grouped(error,lab),lookup_ns=timing(a,b,table),
                 in_table_lookup_ns=timing(a[flags[:len(a)]==0],b[flags[:len(a)]==0],table),
                 fallback_rate=float(flags[:len(a)].mean()),endpoint_expansion_rate=fast/len(a),
                 direct_fallback_rate=float(flags[:len(a)].mean())-fast/len(a),
                 comparisons=[],guard='5 center/edge probes per cell; cutoff 2e-6; independent validation')
        selected_errors={}
        # Same-grid comparisons, plus a denser bilinear alternative.
        for mode,order,name in [(1,3,'Delta cubic'),(2,3,'smooth cubic'),(0,3,'lnI cubic'),(2,1,'smooth bilinear')]:
            conv=[]
            for size in (250,500,996):
                start=time.perf_counter()
                trial=build_table(lo,hi,ns=size,nv=size,mode=mode,interpolation=order,guard=False)
                elapsed=time.perf_counter()-start
                errs=lnI_table_array(ab,bs,trial)-direct
                item=dict(name=name,ns=size,nv=size,entries=int(trial[0].size),build_s=elapsed,
                          errors=summary(abs(errs)),by_regime=grouped(errs,lab),
                          lookup_ns=timing(a,b,trial))
                row['comparisons'].append(item);conv.append(item['errors']['max'])
                if size==996:
                    selected_errors[name]=errs[len(a):len(a)+len(ga)].reshape(len(bb),len(aa))
            curves.append((hi,name,conv))
            # Do not claim a grid requirement if none of the measured grids passes.
            for tol in (1e-5,1e-4):
                passed=[c['ns'] for c in row['comparisons'] if c['name']==name and c['errors']['max']<=tol]
                row.setdefault('smallest_measured_passing_grid',{})[name+f' {tol:g}']=min(passed) if passed else None
        guarded_curve=[]
        for size in (250,500,996):
            start=time.perf_counter()
            trial=table if size==996 else build_table(lo,hi,ns=size,nv=size)
            elapsed=time.perf_counter()-start
            vals,fl=table_diagnostics(ab,bs,trial)
            errs=vals-direct
            item=dict(name='guarded smooth cubic',ns=size,nv=size,entries=int(trial[0].size),
                      build_s=row['build']['build_s'] if size==996 else elapsed,
                      errors=summary(abs(errs)),by_regime=grouped(errs,lab),
                      fallback_rate=float(fl[:len(a)].mean()),lookup_ns=timing(a,b,trial))
            row['comparisons'].append(item);guarded_curve.append(item['errors']['max'])
        curves.append((hi,'guarded smooth cubic',guarded_curve))
        for tol in (1e-5,1e-4):
            passed=[c['ns'] for c in row['comparisons'] if c['name']=='guarded smooth cubic' and c['errors']['max']<=tol]
            row['smallest_measured_passing_grid'][f'guarded smooth cubic {tol:g}']=min(passed) if passed else None
        if hi==100.:
            start=time.perf_counter();extended=build_table(lo,hi,bmax=1e10)
            extended_s=time.perf_counter()-start
            save_table(ROOT/'data/table1d_xmin0.1_xmax100_bmax1e10.npz',extended)
            exb=extended[6]*np.sinh(rng.uniform(0,np.arcsinh(1e10/extended[6]),100000))
            exa=np.array([inverse_s(s,t,lo,hi,extended[5]) for s,t in zip(sb,exb)])
            exv,exflags=table_diagnostics(exa,exb,extended)
            row['bmax1e10']=dict(build_s=extended_s,errors=summary(abs(exv-lnI_direct_array(exa,exb,lo,hi))),
                                  fallback_rate=float(exflags.mean()))
            assert extended_s<60 and row['bmax1e10']['errors']['max']<=1e-4
        maps.append((hi,error[len(a):len(a)+len(ga)].reshape(len(bb),len(aa)),selected_errors))
        results.append(row)
        timing_inputs.append((a,b,table,flags[:len(a)]))
        print(json.dumps(row,indent=2),flush=True)
        assert abs(error).max()<=1e-4
    # Measure the selected tables together after all construction work, so
    # CPU warm-up and build-time cache pressure affect both priors similarly.
    for row,(a,b,table,flags) in zip(results,timing_inputs):
        row['lookup_ns']=timing(a,b,table)
        row['in_table_lookup_ns']=timing(a[flags==0],b[flags==0],table)
        selected=next(c for c in row['comparisons'] if c['name']=='guarded smooth cubic' and c['ns']==996)
        selected['lookup_ns']=row['lookup_ns']
        print(f"Final timing xmax={row['xmax']:g}: {row['lookup_ns']:.1f} ns inclusive; "
              f"{row['in_table_lookup_ns']:.1f} ns in-table",flush=True)
    fig,axes=plt.subplots(2,4,figsize=(17,8),layout='constrained')
    for axs,(hi,guarded,raw) in zip(axes,maps):
        for ax,(name,err) in zip(axs,[('guarded smooth',guarded),('Delta cubic',raw['Delta cubic']),('lnI cubic',raw['lnI cubic']),('smooth cubic',raw['smooth cubic'])]):
            artist=ax.imshow(err,origin='lower',aspect='auto',cmap='coolwarm',norm=SymLogNorm(1e-6,vmin=-10,vmax=10))
            ax.set(title=f'{name}; xmax={hi:g}',xlabel='signed logarithmic a',ylabel='zero + log b')
    fig.colorbar(artist,ax=axes,label='lookup − lnI_direct')
    fig.savefig(ROOT/'figures/table1d_error_map.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for hi,name,errors in curves:
        ax=axes[0 if hi==100 else 1];ax.loglog([250,500,996],errors,'-o',label=name)
        ax.set(title=f'xmax={hi:g}',xlabel='nodes per axis',ylabel='worst absolute log error')
        ax.axhline(1e-4,color='k',ls='--');ax.axhline(1e-5,color='k',ls=':');ax.legend()
    fig.savefig(ROOT/'figures/table1d_grid_convergence.png',dpi=180);plt.close(fig)
    (ROOT/'data/table1d_check.json').write_text(json.dumps(dict(bounds=results),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
