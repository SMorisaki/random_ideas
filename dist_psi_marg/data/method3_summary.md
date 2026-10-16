# Method 3 measured results

All errors are absolute errors in ln L_marg. Timing is the median across cases of warmed compiled per-case median timings.
Sweep: 4525 reference points, 29 unattainable geometry cells. Default 48 total t nodes; diagnostic 96 nodes.

| Variant | Median us | Regression median / max | Random median / max | Finite regression / random |
|---|---:|---:|---:|---:|
| 2d_o0 | 0.131 | 0.00503 / 2.22 | 0.0121 / 13.3 | 7/40, 61/300 |
| 2d_o2 | 0.148 | 2.47e-05 / 1.74 | 0.000181 / 12.5 | 7/40, 61/300 |
| 2d_o4 | 0.149 | 1.33e-07 / 1.54 | 2.59e-06 / 12 | 7/40, 61/300 |
| 1d_o0 | 38.8 | 0.00925 / 6.46 | 0.0183 / 18.6 | 40/40, 300/300 |
| 1d_o2 | 56.6 | 6.26e-05 / 17.2 | 0.000347 / 33.1 | 40/40, 300/300 |
| 1d_o4 | 55.4 | 4.06e-06 / 28.8 | 0.000103 / 47.9 | 40/40, 300/300 |

Worst regime thresholds, entries at 1e-3 / 1e-2. For 2-D only the 25 interior geometries per positive q are included; exterior geometries are unavailable at every SNR.
`>1000` means at least one included cell has no passing threshold. A dash means unavailable. These are grid results, not interpolated error guarantees.

| q | 2d O0 | 2d O2 | 2d O4 | 1d O0 | 1d O2 | 1d O4 |
|---:|---|---|---|---|---|---|
| 1 | 133 / 42.2 | 23.7 / 13.3 | 17.8 / 13.3 | 178 / 56.2 | 31.6 / 17.8 | 17.8 / 13.3 |
| 0.3 | 178 / 56.2 | 31.6 / 31.6 | 31.6 / 31.6 | 178 / 56.2 | 31.6 / 17.8 | 17.8 / 13.3 |
| 0.1 | 178 / 56.2 | 42.2 / 42.2 | 42.2 / 42.2 | 178 / 56.2 | 31.6 / 17.8 | 17.8 / 13.3 |
| 0.01 | 562 / 178 | 133 / 133 | 133 / 133 | 237 / 56.2 | 23.7 / 13.3 | 17.8 / 13.3 |
| 0.001 | >1000 / 562 | 422 / 422 | 422 / 422 | >1000 / 56.2 | >1000 / 13.3 | >1000 / 10 |
| 0 | — / — | — / — | — / — | 133 / 56.2 | 31.6 / 13.3 | 17.8 / 10 |

The complete `(q, eta, x0*)` tables for both node budgets are in [method3_breakdown.md](method3_breakdown.md).
At 48 nodes, the q=0.001, eta=3pi/8, x0*=0.7xmin cell still has error 0.00170 at rho=1000. The 96-node sweep resolves this numerical limitation; all attainable 1-D cells have a 1e-3 threshold by SNR 178 (O0), 32 (O2), or 18 (O4).
Higher Taylor order is harmful outside asymptotic validity: the worst stored-set errors grow with order. Positive curvature and a three-sigma interior guard do not by themselves exclude prior-dominated low-SNR regions.
2-D requires positive lambda2 and a well-separated interior peak; 1-D remains finite in degenerate and exterior cases but needs quadrature convergence and stiff-width control.
The 1-D cost target is missed. The existing method-2 table implementation measured ~21 us in iteration 004, below this 1-D implementation. Use method 2 for general fallback; consider only 2-D as a cheap guarded fast branch.
