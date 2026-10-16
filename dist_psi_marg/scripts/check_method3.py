#!/usr/bin/env python
"""Validate annular Laplace approximations and measure regime-specific SNR limits.

The 25-point SNR grid is actual annulus matched-filter SNR. x0* is a/b
at a dominant angle of the annulus-constrained likelihood (not its prior).
Some q=0 cells cannot have the prescribed x0*: those cells are explicitly
undefined rather than assigning them a fictitious SNR or radius. The cache
is restartable, keyed by full coefficients, and always uses lnL_marg_ref.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dpmarg.method3 import method3_array,lnL_marg_m3,lnL_marg_m3_2d
from dpmarg.params import expand_params,reduce_params
from dpmarg.reference import lnL_marg_ref

KEYS=('A','B','C','D','E','xmin','xmax')
QS=[1.,.3,.1,.01,.001,0.]
ETAS=np.arange(5)*math.pi/8
RADII=[('0.3',.3),('3',3.),('30',30.),('0.7xmin',.07),('1.3xmin',.13),('0.8xmax',80.),('1.5xmax',150.)]
RHOS=np.logspace(0,3,25)
VARIANTS=[(f'{name}_o{o}',v,o) for name,v in [('2d',2),('1d',1)] for o in [0,2,4]]


def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if math.isfinite(x) else None
    return x


def write(path,x):
    path.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')


def annular_peak(q,h,eta):
    """Exact first-quadrant quadratic maximum via its secular multiplier.

    At a circular boundary, s=a/(1+mu), t=bt/(q+mu). The inner
    circle's hard case bt=0 has mu=-q and two reflected t modes.
    """
    a=h*math.cos(eta);bt=h*math.sin(eta)
    if abs(bt)<1e-15*h:bt=0.
    if abs(a)<1e-15*h:a=0.
    if q>0:
        s=a;t=bt/q;r=math.hypot(s,t)
        if .1<=r<=100:return .5*(s*s+q*t*t),r,math.atan2(t,s)
        rb=.1 if r<.1 else 100.
    else:
        if bt==0 and .1<=a<=100:return .5*a*a,a,0.
        rb=.1 if bt==0 and a<.1 else 100.
    if bt==0:
        s=min(rb,a/(1-q)) if q<1 else rb
        t=math.sqrt(max(0.,rb*rb-s*s))
    elif a==0:
        s=0.;t=rb
    else:
        # Bisection in theta avoids a near-singular multiplier for tiny bt.
        from scipy.optimize import minimize_scalar
        opt=minimize_scalar(lambda theta: -(rb*(a*math.cos(theta)+bt*math.sin(theta))-.5*rb*rb*(q+(1-q)*math.cos(theta)**2)),bounds=(0.,math.pi/2),method='bounded',options={'xatol':1e-15})
        th=opt.x;s=rb*math.cos(th);t=rb*math.sin(th)
    return a*s+bt*t-.5*(s*s+q*t*t),rb,math.atan2(t,s)


def shape(q,eta,target):
    """Find R/lam1 giving target a/b at a constrained dominant angle.

    Scaling lam1,lam2,R together leaves this geometry fixed. At q=0,
    eta=0 a flat ridge includes theta=0 if target is inside the annulus;
    select that valid dominant angle to remove the otherwise arbitrary tie.
    For outside q=0 cells the a/b curve need not be monotone: scan all
    brackets and choose the largest-R root. No root means undefined.
    """
    co=math.cos(eta);si=math.sin(eta)
    if q==0 and eta==0:
        if target<.1:return None
        return target
    if q==0 and abs(co)<1e-14:return None
    if q>0 and .1<=target<=100:
        return target/math.hypot(co,si/q)

    def x0(h):
        _,r,t=annular_peak(q,h,eta)
        b=q+(1-q)*math.cos(t)**2
        return h*math.cos(t-eta)/b if b>1e-28 else math.inf
    hs=np.exp(np.linspace(-24,24,241))
    if q>0:
        # Natural radial maximum tends to 0 then infinity.
        root=brentq(lambda h:x0(h)-target,hs[0],hs[-1],xtol=1e-14)
    else:
        ys=[x0(h)-target for h in hs]
        brackets=[(hs[i],hs[i+1]) for i in range(len(hs)-1) if ys[i]*ys[i+1]<0]
        if not brackets:return None
        root=brentq(lambda h:x0(h)-target,*brackets[-1],xtol=1e-13)
    if abs(x0(root)/target-1)>2e-5:
        raise ArithmeticError(('x0 construction',q,eta,target,root,x0(root)))
    return root


def sweep_cases():
    cases=[];undefined=[];geometry=[]
    for q in QS:
        for ie,eta in enumerate(ETAS):
            for label,target in RADII:
                base=dict(q=q,eta=float(eta),eta_index=ie,x0_label=label,x0=target)
                h=shape(q,float(eta),target)
                if h is None:
                    undefined.append(dict(base,reason='No coefficient scale has this radial peak at a dominant annular angle (flat-axis geometry).'))
                    continue
                f,r,theta=annular_peak(q,h,float(eta))
                if f<=0:raise ArithmeticError(('nonpositive base maximum',base,f))
                geometry.append(dict(base,R_over_lam1=h,peak_radius=r,peak_theta=theta))
                for rho in RHOS:
                    scale=rho*rho/(2*f)
                    coeff=expand_params(scale,scale*q,scale*h,float(eta))
                    cases.append(dict(base,rho=float(rho),**dict(zip(KEYS,(*coeff,.1,100.)))))
    return cases,undefined,geometry


def ref_key(c):
    return ','.join(float(c[k]).hex() for k in KEYS)


def reference_job(item):
    key,p=item
    with warnings.catch_warnings(record=True) as caught:
        value=lnL_marg_ref(*p)
    return key,float(value),[str(w.message) for w in caught]


def references(cases,workers):
    path=ROOT/'data/method3_sweep_reference.json'
    cache=json.loads(path.read_text()) if path.exists() else dict(version=1,values={},warnings={})
    missing={ref_key(c):[c[k] for k in KEYS] for c in cases if ref_key(c) not in cache['values']}
    print(f'Reference cache: {len(cases)-len(missing)}/{len(cases)} reused; computing {len(missing)} with {workers} workers',flush=True)
    start=time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i,(key,value,notes) in enumerate(pool.map(reference_job,missing.items(),chunksize=1)):
                cache['values'][key]=value
                if notes:cache['warnings'][key]=notes
                if (i+1)%100==0:
                    write(path,cache)
                    print(f'References {i+1}/{len(missing)}, {time.perf_counter()-start:.0f}s',flush=True)
    finally:
        write(path,cache)
    return np.array([cache['values'][ref_key(c)] for c in cases]),cache['warnings']


def stats(errors):
    finite=errors[np.isfinite(errors)]
    return dict(count=len(errors),finite=len(finite),unavailable=int(len(errors)-len(finite)),
                median=float(np.median(finite)) if len(finite) else None,
                p99=float(np.quantile(finite,.99)) if len(finite) else None,
                max=float(np.max(finite)) if len(finite) else None,
                pass_1e3=int(np.sum(errors<1e-3)),pass_1e2=int(np.sum(errors<1e-2)))


def breakdown(errors):
    result={}
    for tol in [1e-3,1e-2]:
        good=np.maximum.accumulate(np.where(np.isfinite(errors),errors,np.inf)[::-1])[::-1]<tol
        result[str(tol)]=float(RHOS[np.flatnonzero(good)[0]]) if np.any(good) else None
    return result


@njit(cache=True)
def repeated(ps,v,o,n,repeats):
    total=0.
    for _ in range(repeats):
        for p in ps:
            if v==2:
                value=lnL_marg_m3_2d(p[0],p[1],p[2],p[3],p[4],p[5],p[6],o)
            else:
                value=lnL_marg_m3(p[0],p[1],p[2],p[3],p[4],p[5],p[6],o,n)
            if math.isfinite(value):total+=value
    return total


def timing(ps,v,o,n):
    # Per-case median of repeated, warmed compiled calls, then median over
    # cases. This avoids conflating an expensive tail with a median call.
    repeated(ps[:1],v,o,n,1)
    per_case=[]
    repeats=200 if v==2 else 5
    for p in ps:
        one=p.reshape(1,7);times=[]
        for _ in range(3):
            start=time.perf_counter_ns();repeated(one,v,o,n,repeats)
            times.append((time.perf_counter_ns()-start)/1000/repeats)
        per_case.append(float(np.median(times)))
    return float(np.median(per_case))


def plots(cs,errors,thresholds):
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True,layout='constrained')
    for ax,q in zip(axes.flat,QS):
        for name,v,o in VARIANTS:
            arr=np.array([e for c,e in zip(cs,errors[name]) if c['q']==q]).reshape(-1,25)
            # Worst curve across all eta/radius regimes, with unavailable
            # 2-D cells excluded and their counts reported in JSON.
            for i in range(25):
                if not np.isfinite(arr[:,i]).any():arr[:,i]=np.nan
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                curve=np.nanmax(arr,axis=0)
            ax.loglog(RHOS,np.maximum(curve,1e-12),label=name,ls='-' if v==1 else '--')
        ax.axhline(1e-3,color='k',ls=':');ax.axhline(1e-2,color='gray',ls=':')
        ax.set(title=f'q={q:g}',xlabel='actual annular SNR',ylabel='worst absolute log error')
    axes.flat[0].legend(fontsize=8,ncol=2)
    fig.savefig(ROOT/'figures/method3_error_vs_snr.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(6,2,figsize=(14,21),sharex=True,sharey=True,layout='constrained')
    colors=plt.cm.viridis(np.linspace(0,1,5));markers=['o','s','^','v','D','P','X']
    for row,(name,_,_) in enumerate(VARIANTS):
        for col,tol in enumerate(['0.001','0.01']):
            ax=axes[row,col]
            for ie,eta in enumerate(ETAS):
                for ir,(label,_) in enumerate(RADII):
                    yy=[]
                    for q in QS:
                        match=[r for r in thresholds if r['q']==q and r['eta_index']==ie and r['x0_label']==label]
                        yy.append(match[0]['thresholds'][name][tol] if match else None)
                    # Undefined / failed at rho=1000 are intentionally gaps.
                    ax.plot(QS,[np.nan if y is None else y for y in yy],color=colors[ie],marker=markers[ir],ms=4,lw=.7,alpha=.7)
            ax.set(xscale='symlog',yscale='log',title=f'{name}, tolerance {tol}',xlabel='q',ylabel='breakdown SNR',ylim=(.9,1200))
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=colors[i],label=f'eta={i} pi/8') for i in range(5)]
    handles += [Line2D([],[],color='gray',marker=markers[i],ls='',label='x0*='+label) for i,(label,_) in enumerate(RADII)]
    fig.legend(handles=handles,loc='outside lower center',ncol=6,fontsize=9)
    fig.savefig(ROOT/'figures/method3_breakdown_snr.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--n-t',type=int,default=48)
    args=parser.parse_args()
    start=time.perf_counter()
    cs,undefined,geometry=sweep_cases()
    refs,notes=references(cs,args.workers)
    stored=[dict(c,dataset='regression') for c in json.loads((ROOT/'data/regression_points.json').read_text())]
    stored += [dict(c,dataset='random') for c in json.loads((ROOT/'data/method2_random_reference.json').read_text())]
    ps=np.array([[c[k] for k in KEYS] for c in cs]);sp=np.array([[c[k] for k in KEYS] for c in stored])
    sr=np.array([c['lnL_marg_ref'] for c in stored])
    result=dict(n_t=args.n_t,comparison_n_t=min(128,2*args.n_t),quadrature_nodes='n_t total across active sinh half-patches; minimum two nodes (one when n_t<2*patch_count)',
                snr_grid=RHOS.tolist(),x0_definition='a(theta*)/b(theta*) at annulus-constrained dominant likelihood angle; select theta=0 on the q=0,eta=0 flat ridge when possible',
                reference_points=len(cs),undefined_cells=undefined,reference_warning_points=len(notes),variants={},timing_us={})
    errors={};dense_errors={};stored_values={};values={}
    for name,v,o in VARIANTS:
        a=method3_array(ps,v,o,args.n_t);err=abs(a-refs);errors[name]=err;values[name]=a
        sa=method3_array(sp,v,o,args.n_t);se=abs(sa-sr);stored_values[name]=sa
        row=dict(sweep=stats(err),stored={},worst_stored=stored[int(np.nanargmax(se))]['name'],worst_sweep={k:cs[int(np.nanargmax(err))][k] for k in ['q','eta_index','x0_label','rho']})
        for dataset in ['regression','random']:
            mask=np.array([c['dataset']==dataset for c in stored]);row['stored'][dataset]=stats(se[mask])
        if v==1:
            dense=method3_array(ps,v,o,min(128,2*args.n_t))
            delta=abs(a-dense)
            dense_errors[name]=abs(dense-refs)
            row['sweep_2n']=stats(dense_errors[name])
            row['t_quadrature_n_vs_2n']=stats(delta)
            row['t_quadrature_high_snr_n_vs_2n']=stats(delta[np.array([c['rho']>=100 for c in cs])])
            row['t_quadrature_worst']={k:cs[int(np.nanargmax(delta))][k] for k in ['q','eta_index','x0_label','rho']}
        result['variants'][name]=row
        # Time a representative broad set; for 2d also report usable-only.
        sample=sp if v==1 else sp[np.isfinite(sa)]
        result['timing_us'][name]=timing(sample,v,o,args.n_t)
        print(name,json.dumps(clean(dict(row,timing_us=result['timing_us'][name]))),flush=True)
    thresholds=[]
    for k in range(0,len(cs),25):
        c=cs[k]
        thresholds.append(dict(q=c['q'],eta_index=c['eta_index'],x0_label=c['x0_label'],
                               thresholds={name:breakdown(errors[name][k:k+25]) for name,_,_ in VARIANTS},
                               thresholds_2n={name:breakdown(e[k:k+25]) for name,e in dense_errors.items()}))
    # Detailed regime table is separate from the compact machine summary.
    write(ROOT/'data/method3_breakdown_full.json',thresholds)
    warning_checks=[]
    from dpmarg.method2 import lnL_marg_m2
    for key in notes:
        p=[float.fromhex(x) for x in key.split(',')]
        cached=refs[next(i for i,c in enumerate(cs) if ref_key(c)==key)]
        check=lnL_marg_m2(*p,n_nodes=256)
        warning_checks.append(abs(check-cached))
    result['warned_reference_method2_256_max_difference']=max(warning_checks,default=0.)
    result['breakdown_by_q']={} 
    for q in QS:
        result['breakdown_by_q'][str(q)]={}
        for name,_,_ in VARIANTS:
            result['breakdown_by_q'][str(q)][name]={}
            for tol in ['0.001','0.01']:
                ts=[r['thresholds'][name][tol] for r in thresholds if r['q']==q]
                good=[t for t in ts if t is not None]
                result['breakdown_by_q'][str(q)][name][tol]=dict(cells=len(ts),passing=len(good),min=min(good) if good else None,max=max(good) if good else None)
    result['dense_breakdown_by_q']={}
    for q in QS:
        result['dense_breakdown_by_q'][str(q)]={}
        for name in dense_errors:
            result['dense_breakdown_by_q'][str(q)][name]={}
            for tol in ['0.001','0.01']:
                ts=[r['thresholds_2n'][name][tol] for r in thresholds if r['q']==q]
                good=[t for t in ts if t is not None]
                result['dense_breakdown_by_q'][str(q)][name][tol]=dict(cells=len(ts),passing=len(good),min=min(good) if good else None,max=max(good) if good else None)
    full=[]
    for i,c in enumerate(cs):
        full.append(dict(c,lnL_marg_ref=refs[i],values={name:values[name][i] for name,_,_ in VARIANTS},errors={name:errors[name][i] for name,_,_ in VARIANTS}))
    write(ROOT/'data/method3_check_full.json',dict(sweep=full,geometry=geometry,stored=[dict(c,values={name:stored_values[name][i] for name,_,_ in VARIANTS}) for i,c in enumerate(stored)]))
    plots(cs,errors,thresholds)
    result['elapsed_seconds']=time.perf_counter()-start
    write(ROOT/'data/method3_check.json',result)
    # All 210 geometry cells, including unattainable cells, fit in a small
    # reviewable Markdown table. Each entry gives threshold at 1e-3 / 1e-2.
    lines=['# Method 3 SNR thresholds','',
           'Entries are the smallest sampled actual SNR above which every larger sampled SNR passes, at 1e-3 / 1e-2.',
           '`—` means no threshold by 1000 (including unavailable 2-D outputs). `undefined` means the requested geometry cannot exist.',
           'The 2-D three-sigma guard is necessary but does not certify the target accuracy.', '',
           '| q | eta / (pi/8) | x0* | 2d O0 | 2d O2 | 2d O4 | 1d O0 | 1d O2 | 1d O4 |',
           '|---:|---:|---|---|---|---|---|---|---|']
    def fmt(t):return '—' if t is None else f'{t:.3g}'
    for q in QS:
        for ie in range(5):
            for label,_ in RADII:
                matches=[r for r in thresholds if r['q']==q and r['eta_index']==ie and r['x0_label']==label]
                cells=[]
                for name,_,_ in VARIANTS:
                    cells.append(' / '.join(fmt(matches[0]['thresholds'][name][t]) for t in ['0.001','0.01']) if matches else 'undefined')
                lines.append(f'| {q:g} | {ie} | {label} | '+' | '.join(cells)+' |')
    lines += ['',f'Node-doubling diagnostic: {min(128,2*args.n_t)} total nodes (same Laplace formulas).', '',
              '| q | eta / (pi/8) | x0* | 1d O0 | 1d O2 | 1d O4 |',
              '|---:|---:|---|---|---|---|']
    for r in thresholds:
        cells=[' / '.join(fmt(r['thresholds_2n'][name][t]) for t in ['0.001','0.01']) for name in dense_errors]
        lines.append(f"| {r['q']:g} | {r['eta_index']} | {r['x0_label']} | "+' | '.join(cells)+' |')
    (ROOT/'data/method3_breakdown.md').write_text('\n'.join(lines)+'\n')
    lines=['# Method 3 measured results','',
           'All errors are absolute errors in ln L_marg. Timing is the median across cases of warmed compiled per-case median timings.',
           f'Sweep: {len(cs)} reference points, {len(undefined)} unattainable geometry cells. Default {args.n_t} total t nodes; diagnostic {min(128,2*args.n_t)} nodes.', '',
           '| Variant | Median us | Regression median / max | Random median / max | Finite regression / random |',
           '|---|---:|---:|---:|---:|']
    for name,_,_ in VARIANTS:
        a=result['variants'][name]['stored']['regression'];b=result['variants'][name]['stored']['random']
        lines.append(f"| {name} | {result['timing_us'][name]:.3g} | {a['median']:.3g} / {a['max']:.3g} | {b['median']:.3g} / {b['max']:.3g} | {a['finite']}/40, {b['finite']}/300 |")
    lines += ['', 'Worst regime thresholds, entries at 1e-3 / 1e-2. For 2-D only the 25 interior geometries per positive q are included; exterior geometries are unavailable at every SNR.',
              '`>1000` means at least one included cell has no passing threshold. A dash means unavailable. These are grid results, not interpolated error guarantees.', '',
              '| q | 2d O0 | 2d O2 | 2d O4 | 1d O0 | 1d O2 | 1d O4 |',
              '|---:|---|---|---|---|---|---|']
    for q in QS:
        cells=[]
        for name,v,o in VARIANTS:
            selected=[r for r in thresholds if r['q']==q and (v==1 or r['x0_label'] not in ['0.7xmin','1.5xmax'])]
            pair=[]
            for tol in ['0.001','0.01']:
                ts=[r['thresholds'][name][tol] for r in selected]
                pair.append('—' if v==2 and q==0 else '>1000' if any(t is None for t in ts) else f'{max(ts):.3g}')
            cells.append(' / '.join(pair))
        lines.append(f'| {q:g} | '+' | '.join(cells)+' |')
    lines += ['', 'The complete `(q, eta, x0*)` tables for both node budgets are in [method3_breakdown.md](method3_breakdown.md).',
              'At 48 nodes, the q=0.001, eta=3pi/8, x0*=0.7xmin cell still has error 0.00170 at rho=1000. The 96-node sweep resolves this numerical limitation; all attainable 1-D cells have a 1e-3 threshold by SNR 178 (O0), 32 (O2), or 18 (O4).',
              'Higher Taylor order is harmful outside asymptotic validity: the worst stored-set errors grow with order. Positive curvature and a three-sigma interior guard do not by themselves exclude prior-dominated low-SNR regions.',
              '2-D requires positive lambda2 and a well-separated interior peak; 1-D remains finite in degenerate and exterior cases but needs quadrature convergence and stiff-width control.',
              'The 1-D cost target is missed. The existing method-2 table implementation measured ~21 us in iteration 004, below this 1-D implementation. Use method 2 for general fallback; consider only 2-D as a cheap guarded fast branch.', '']
    (ROOT/'data/method3_summary.md').write_text('\n'.join(lines))
    print(f'Completed in {time.perf_counter()-start:.1f}s',flush=True)


if __name__=='__main__':main()
