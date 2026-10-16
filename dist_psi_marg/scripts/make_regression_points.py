#!/usr/bin/env python
"""Generate the committed regression list and report independent quadrature errors."""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpmarg import lnL_marg_mp, lnL_marg_ref, reduce_params, snr
from dpmarg.regimes import named_cases, random_params


def main():
    """Evaluate every named case and 20 seed-1 random cases, failing above 1e-8."""
    output = Path(__file__).resolve().parents[1] / "data" / "regression_points.json"
    output.parent.mkdir(exist_ok=True)
    rows = []
    print(f"{'name':27s} {'lam1':>11s} {'lam2':>11s} {'R':>11s} {'eta':>8s} "
          f"{'rho':>10s} {'ref':>19s} {'mp':>19s} {'|ref-mp|':>10s} {'seconds':>9s}", flush=True)
    for case in named_cases() + random_params(20, seed=1):
        args = [case[k] for k in ("A", "B", "C", "D", "E", "xmin", "xmax")]
        start = time.perf_counter()
        try:
            t0 = time.perf_counter()
            ref = lnL_marg_ref(*args)
            ref_seconds = time.perf_counter()-t0
            high = lnL_marg_mp(*args)
        except Exception as exc:
            raise RuntimeError(f"Failed parameters: {case!r}") from exc
        l1, l2, R, eta = reduce_params(*args[:5])
        rho = snr(*args)
        error = abs(ref-high)
        elapsed = time.perf_counter()-start
        row = dict(case, lam1=l1, lam2=l2, R=R, eta=eta, rho=rho,
                   rho_proxy=R/l1**0.5 if l1 else 0.0,
                   lnL_marg_ref=ref, lnL_marg_mp=high, abs_error=error,
                   ref_seconds=ref_seconds, seconds=elapsed)
        rows.append(row)
        print(f"{case['name']:27s} {l1:11.4g} {l2:11.4g} {R:11.4g} {eta:8.4f} "
              f"{rho:10.4g} {ref:19.11g} {high:19.11g} {error:10.3g} {elapsed:9.2f}", flush=True)
        # Save completed results even if a subsequent point raises or fails.
        output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    failed = [r for r in rows if r["abs_error"] > 1e-8]
    import statistics
    print(f"Worst |ref-mp|: {max(r['abs_error'] for r in rows):.6g}; "
          f"median ref seconds: {statistics.median(r['ref_seconds'] for r in rows):.6g}", flush=True)
    if failed:
        raise RuntimeError(f"Disagreements exceeding 1e-8: {failed!r}")


if __name__ == "__main__":
    main()
