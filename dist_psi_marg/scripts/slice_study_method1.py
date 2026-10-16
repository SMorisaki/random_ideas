#!/usr/bin/env python
"""Measure one-axis interpolation errors of the full marginalization table.

All coarse grids are evaluated independently; errors use 1025 fresh fine-grid
nodes per slice. Passing counts are measured resolutions, never extrapolations.
A null required count means >512 nodes or unresolved structure at this sampling.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, CubicHermiteSpline
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dpmarg import table1d, table4d, regimes, params
from dpmarg.reference import lnL_marg_ref
from dpmarg.method2 import lnL_marg_m2_table

AXES=['v1','w','radial','eta']
COUNTS=[8,12,16,24,32,48,64,96,128,192,256,384,512]


def interp(x,y,fine,kind,even=False):
    if kind=='linear':return np.interp(fine,x,y)
    if kind=='natural':return CubicSpline(x,y,bc_type='natural')(fine)
    slope=np.empty(len(y));h=x[1]-x[0]
    slope[1:-1]=(y[2:]-y[:-2])/(2*h)
    slope[0]=(y[1]-y[0])/h;slope[-1]=(y[-1]-y[-2])/h
    if even:slope[0]=slope[-1]=0.
    return CubicHermiteSpline(x,y,slope)(fine)


def summary_required(errors,counts,tol):
    needed=[]
    for row in errors:
        # Require all finer measured grids to pass too.
        good=np.maximum.accumulate(row[::-1])[::-1]<=tol
        needed.append(int(counts[np.flatnonzero(good)[0]]) if good.any() else None)
    a=np.array([n if n is not None else np.inf for n in needed])
    def percentile(q):
        value=np.quantile(a,q,method='higher')
        return None if not np.isfinite(value) else int(value)
    return dict(median=percentile(.5),p99=percentile(.99),max=percentile(1),
                unresolved=int(np.isinf(a).sum())),needed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bases',type=int,default=200)
    parser.add_argument('--fine',type=int,default=1025)
    args=parser.parse_args()
    if args.fine<1025:parser.error('--fine must be >=1025')
    radial=table1d.load_table(ROOT/'data/table1d_xmin0.1_xmax100.npz')
    cases=regimes.named_cases()+regimes.random_params(args.bases,501)
    bases=np.array([table4d.to_unit(*params.reduce_params(*(c[k] for k in ['A','B','C','D','E'])),1e8,24.,.1,100.) for c in cases])
    # Base points outside the proposed box are projected for slice diagnostics;
    # their original position is retained in the full data dump.
    bases=np.clip(bases,0,1)
    fine=np.linspace(0,1,args.fine); counts=np.array(COUNTS)
    summary=dict(bmax=1e8,smax=24.,base_count=len(cases),random_seed=501,
                 fine_count=args.fine,counts=COUNTS,axes={},coordinate_choice='v1,w=v2/v1,u=(s-s0)/(24-s0),eta',
                 notes=['Null required counts mean no measured grid passes; no uniform guarantee.',
                        'Regularized peak direction uses lambda+sigma0^2; physical likelihood unchanged.',
                        'Curves include max-subtraction switches and sharp prior/likelihood mode exchanges.'])
    full={};start=time.perf_counter()
    table4d.sample_unit(bases[:1],1e8,24.,.1,100.,radial)
    # Verify the supplied generator independently on 20 documented points.
    checks=[]
    check_cases=cases[:18]+[dict(zip(['A','B','C','D','E'],params.expand_params(l1,0.,19.2047382462,1.24240566613)),name=f'narrow_null_axis_{l1:g}',xmin=.1,xmax=100.) for l1 in [651154.6134045325,633390.2841571145]]
    for c in check_cases:
        p=[c[k] for k in ['A','B','C','D','E','xmin','xmax']]
        value=lnL_marg_m2_table(*p,radial); truth=lnL_marg_ref(*p)
        checks.append(dict(name=c['name'],value=value,reference=truth,error=abs(value-truth)))
    summary['generator_checks']={'n':20,'max_abs_error':max(c['error'] for c in checks)}
    if summary['generator_checks']['max_abs_error']>5e-5:raise AssertionError(checks)
    for axis,name in enumerate(AXES):
        errors={kind:np.empty((len(bases),len(counts))) for kind in ['catmull','natural','linear']}
        curves=[]
        for i,base in enumerate(bases):
            points=np.tile(base,(len(fine),1));points[:,axis]=fine
            truth=table4d.sample_unit(points,1e8,24.,.1,100.,radial)
            for j,n in enumerate(counts):
                x=np.linspace(0,1,n); points=np.tile(base,(n,1));points[:,axis]=x
                y=table4d.sample_unit(points,1e8,24.,.1,100.,radial)
                for kind in errors:
                    prediction=interp(x,y,fine,kind,axis==3)
                    errors[kind][i,j]=np.max(abs(prediction-truth))
                if n==64:representative=interp(x,y,fine,'catmull',axis==3)
            curves.append((truth,representative))
        result={}
        for kind,err in errors.items():
            result[kind]={}
            for tol in [1e-3,1e-4]:
                result[kind][str(tol)],needed=summary_required(err,counts,tol)
            result[kind]['max_error_by_count']=err.max(axis=0).tolist()
        ranked=np.argsort(errors['catmull'][:,COUNTS.index(64)])[::-1]
        worst=[];seen=set()
        for candidate in ranked:
            key=tuple(np.round(np.delete(bases[candidate],axis),12))
            if key in seen:continue
            seen.add(key);worst.append(candidate)
            if len(worst)==3:break
        result['worst_slices']=[]
        fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
        for column,i in enumerate(worst):
            l1,l2,R,eta=params.reduce_params(*(cases[i][k] for k in ['A','B','C','D','E']))
            peak,r,t=table4d.lnL_max_numba(l1,l2,R,eta,.1,100.)
            row=dict(name=cases[i]['name'],base_unit=bases[i].tolist(),lam1=l1,lam2=l2,R=R,eta=eta,
                     q=l2/l1 if l1 else 0.,snr=np.sqrt(max(0.,2*peak)),peak_radius=r,
                     error64=float(errors['catmull'][i,COUNTS.index(64)]))
            result['worst_slices'].append(row)
            y,pred=curves[i];axes[0,column].plot(fine,y);axes[0,column].plot(fine,pred,ls='--')
            axes[1,column].plot(fine,pred-y);axes[1,column].axhline(1e-3,color='k',ls=':')
            axes[0,column].set_title(f"{row['name']}\nq={row['q']:.2g}, SNR={row['snr']:.2g}")
            axes[1,column].set_xlabel(name+' (unit coordinate)')
        axes[0,0].set_ylabel('Ftilde');axes[1,0].set_ylabel('64-node Catmull error')
        fig.tight_layout();fig.savefig(ROOT/f'figures/method1_slices_{name}.png',dpi=150);plt.close(fig)
        summary['axes'][name]=result
        full[name]={kind:err.tolist() for kind,err in errors.items()}
        print(json.dumps({'axis':name,'results':result,'elapsed_s':time.perf_counter()-start}),flush=True)
    # Direct v2 rectangular alternative on valid intervals. This keeps lam1,R,
    # eta fixed, varying v2 from 0 to v1. Also compare w at fixed physical R
    # with the warped-table slices, to expose coordinate coupling.
    # The rectangle wastes half its eigenvalue cells and leaves the same
    # near-zero weak-eigenvalue layer; retain w for the capped prototype.
    summary['rectangular_alternative']='Not built: half the eigenvalue rectangle is unused; w covers the ordered triangle without sorting kinks. No claim of measured superiority.'
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,tol in zip(axs,[1e-3,1e-4]):
        for kind in ['catmull','natural','linear']:
            yy=[summary['axes'][name][kind][str(tol)]['median'] or 700 for name in AXES]
            ax.plot(AXES,yy,'o-',label=kind)
        ax.axhline(512,color='k',ls=':');ax.set_yscale('log');ax.set_ylim(6,900)
        ax.set_title(f'Median required nodes, tolerance {tol:g}\n700 denotes >512/unresolved')
        ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'figures/method1_points_per_axis.png',dpi=150);plt.close(fig)
    # The likelihood is analytic and even at eta=0, but the maximum has
    # an |eta| cusp when its degenerate weak-axis modes exchange dominance.
    f0=table4d.reduced_value(1e4,0.,1.,0.,.1,100.,radial)
    epsilon=1e-7
    slope=(table4d.reduced_value(1e4,0.,1.,epsilon,.1,100.,radial)-f0)/epsilon
    summary['cusp_example']=dict(lam1=1e4,lam2=0.,R=1.,epsilon=epsilon,
                                 measured_Ftilde_slope=slope,expected_slope=-np.sqrt(100.**2-(1/1e4)**2))
    lower=[summary['axes'][a]['catmull']['0.001']['max'] or 513 for a in AXES]
    summary['worst_case_entry_lower_bound']=int(np.prod(lower))
    summary['worst_case_bytes_lower_bound']=8*int(np.prod(lower))
    summary['elapsed_s']=time.perf_counter()-start
    (ROOT/'data/method1_slices.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'data/method1_slices_full.json').write_text(json.dumps(dict(errors=full,cases=cases,bases=bases.tolist(),generator_checks=checks)))


if __name__=='__main__':main()
