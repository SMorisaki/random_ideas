# Method 2: direct versus lookup

All 40 regression and 300 random reference values are reused unchanged.

| Node budget | Direct worst log error | Table worst log error | Largest table change |
|---:|---:|---:|---:|
| 16 | 0.25307417 | 0.25307417 | 6.0729396e-07 |
| 24 | 0.04826377 | 0.048263715 | 5.2071871e-07 |
| 32 | 0.014165018 | 0.014165018 | 4.9645312e-07 |
| 48 | 0.0010429012 | 0.0010428969 | 5.2255471e-07 |
| 64 | 4.6903035e-05 | 4.6903035e-05 | 4.9645312e-07 |
| 96 | 6.0190307e-06 | 6.0248893e-06 | 5.2255471e-07 |
| 128 | 1.885322e-06 | 1.889013e-06 | 4.9645312e-07 |

Median time at 64 nodes: direct 242.265 us; table 20.961 us.
Geometry: 3.685 us (17.9% median share); table kernels: 9.433 us (45.9% median share).
Fallback rate: 1.050% of radial evaluations; median 0, maximum 47 per call.

Geometry uses one 64-point scan and 28 refinement iterations; the common Legendre rules are precomputed.
Timing uses warmed repeated numba loops. Component timings are measured separately.
Per-point records: method2_check_full.json (ignored). Small summary: method2_check.json.
Table details and API: ../TABLE1D.md.
