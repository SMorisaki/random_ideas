#!/usr/bin/env python
"""Build both prior tables by default; all artifacts stay under data/."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dpmarg.table1d import build_table, save_table, _build


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xmin',type=float,default=.1)
    p.add_argument('--xmax',type=float,default=None,help='omit to build 100 and 10')
    p.add_argument('--ns',type=int,default=996)
    p.add_argument('--nv',type=int,default=996)
    p.add_argument('--smax',type=float,default=24.)
    p.add_argument('--bmax',type=float,default=1e8)
    args=p.parse_args()
    # Exclude JIT from numerical construction time; report the warm-up too.
    start=time.perf_counter();_build(8,8,24.,args.bmax,1.,1.,args.xmin,10.,2)
    warmup=time.perf_counter()-start
    results=[]
    for hi in ([args.xmax] if args.xmax is not None else [100.,10.]):
        start=time.perf_counter()
        table=build_table(args.xmin,hi,args.ns,args.nv,args.smax,args.bmax)
        elapsed=time.perf_counter()-start
        path=ROOT/f'data/table1d_xmin{args.xmin:g}_xmax{hi:g}.npz'
        save_table(path,table)
        row=dict(xmin=args.xmin,xmax=hi,ns=args.ns,nv=args.nv,smax=args.smax,
                 bmax=args.bmax,sigma0=table[5],bref=table[6],build_s=elapsed,
                 coefficient_entries=table[0].size,coefficient_bytes=table[0].nbytes,
                 guard_bytes=table[-1].nbytes,guard_cell_fraction=float(table[-1].mean()),
                 warmup_s=warmup,path=str(path.relative_to(ROOT)))
        results.append(row)
        print(json.dumps(row),flush=True)
        assert elapsed<60, f'build exceeded 60 s: {elapsed:g}'
    (ROOT/'data/table1d_build.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
