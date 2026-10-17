# Iteration 009: GW190814-like overhead profile

OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, OMP_WAIT_POLICY=PASSIVE; 200 warm complete adapter + numba-HH calls. Each call is partitioned before taking medians; individual medians may not sum exactly. LAL means the prepared LAL wrapper regions. Kernel means fused numba calls plus SciPy FFTs in B1. Bookkeeping is the remainder: phase, projection, inverse-index assembly, container validation, copying into flattened buffers, Python dispatch and instrumentation. This is wall-time accounting, not a claim that all bookkeeping executes in Python bytecode.

| Route | total ms | LAL ms | NumPy/Python bookkeeping ms | kernel ms | bookkeeping % |
|---|---:|---:|---:|---:|---:|
| pairwise paper bands | 3.3923 | 2.9843 | 0.3388 | 0.0553 | 10.0% |
| B0 paper bands | 2.4196 | 2.2177 | 0.1886 | 0.0071 | 7.8% |
| B1 paper bands | 2.5439 | 2.2105 | 0.1884 | 0.1316 | 7.4% |
| B0 hh_half bands | 2.4728 | 2.2686 | 0.1931 | 0.0073 | 7.8% |

No waveform results are cached. Timed wrapper regions include their dispatch; the small profiling overhead remains in bookkeeping. Use perf_iter009.md for uninstrumented steady-state medians and the C2 verdict.
