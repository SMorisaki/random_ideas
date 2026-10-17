#!/usr/bin/env python
"""Validate optional guarded GL12 quadrature on every masked cell and real nodes."""
import sys,json,time
from pathlib import Path
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dpmarg.table1d import load_table,_guard_integral,inverse_s,_lookup_count
from dpmarg.kernel1d import lnI_direct
from dpmarg.method2 import _analytic_compute
from dpmarg.params import reduce_params
from final_benchmark import write


@njit(cache=False)
def check_cells(tab):
    mask=tab[-1];worst=0.;count=0;bad=0
    worst_ab=np.zeros(2)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not mask[i,j]:continue
            for u,v in ((.1,.1),(.1,.9),(.9,.1),(.9,.9),(.5,.5)):
                b=tab[6]*np.sinh(tab[3]+(j+v)*tab[4])
                a=inverse_s(tab[1]+(i+u)*tab[2],b,tab[7],tab[8],tab[5])
                value=_guard_integral(a,b,tab[7],tab[8]);ref=lnI_direct(a,b,tab[7],tab[8])
                error=abs(value-ref);count+=1;bad+=error>1e-8
                if error>worst:worst=error;worst_ab[0]=a;worst_ab[1]=b
    return count,worst,bad,worst_ab


@njit(cache=False)
def check_nodes(p,tab):
    result=_analytic_compute(p[0],p[1],p[2],p[3],p[4],p[5],p[6],tab)
    l1,l2,R,eta=reduce_params(p[0],p[1],p[2],p[3],p[4]);worst=0.;bad=0;count=0
    for t in result[3]:
        a=R*np.cos(t-eta);b=l2+(l1-l2)*np.cos(t)**2
        for sign in (-1.,1.):
            general,fb=_lookup_count(sign*a,b,tab)
            quick,fb=_lookup_count(sign*a,b,tab,True)
            error=abs(general-quick);worst=max(worst,error);bad+=error>1e-8;count+=1
    return count,worst,bad


def main():
    start=time.perf_counter();tables={h:load_table(ROOT/f'data/table1d_xmin0.1_xmax{h:g}.npz') for h in (10.,100.)}
    summary={}
    for h,tab in tables.items():
        n,w,b,ab=check_cells(tab);summary[str(h)]=dict(probes=n,max_error=w,above_1e8=b,worst_ab=ab.tolist())
        print(h,summary[str(h)],flush=True)
    counts=[]
    for group in json.loads((ROOT/'data/method2b_before_full.json').read_text()).values():
        for r in group:
            p=np.array(r['parameters']);counts.append(check_nodes(p,tables[p[6]]))
    summary['actual_nodes']=dict(probes=sum(c[0] for c in counts),max_error=max(c[1] for c in counts),above_1e8=sum(c[2] for c in counts))
    summary['elapsed_s']=time.perf_counter()-start
    write(ROOT/'data/method2b_guard_check.json',summary);print(summary,flush=True)
    if any(r['above_1e8'] for r in summary.values() if isinstance(r,dict)):raise SystemExit('guard quadrature failed')
if __name__=='__main__':main()
