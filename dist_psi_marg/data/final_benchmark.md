# Final shared-set benchmark

Absolute log errors; error fractions and times use available outputs only. Availability is reported separately.
Costs are median / p99 across per-point warmed cost estimates, single thread (three compiled repeated loops).

| Set | Method | Available | Max error | 99% error | Median error | >1e-3 | >1e-2 | Median / 99% us |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| regression | m2_direct | 40/40 | 3.78196e-05 | 3.11686e-05 | 3.27766e-13 | 0.000% | 0.000% | 292.9 / 1128 |
| regression | m2_table | 40/40 | 3.78196e-05 | 3.11503e-05 | 3.30455e-09 | 0.000% | 0.000% | 21.14 / 164.5 |
| regression | m3_2d_o4 | 7/40 | 1.54199 | 1.44974 | 1.33338e-07 | 28.571% | 14.286% | 0.1589 / 0.1788 |
| regression | fast | 40/40 | 3.78196e-05 | 3.11503e-05 | 4.07245e-09 | 0.000% | 0.000% | 20.68 / 163.7 |
| regression | fast_no_laplace | 40/40 | 3.78196e-05 | 3.11503e-05 | 3.30455e-09 | 0.000% | 0.000% | 20.58 / 169.3 |
| regression | m1_cubic | 40/40 | 4.82119 | 3.59521 | 0.00190826 | 52.500% | 27.500% | 0.9483 / 39.4 |
| stored_random | m2_direct | 300/300 | 4.6903e-05 | 2.40077e-05 | 1.55431e-15 | 0.000% | 0.000% | 209.7 / 901.7 |
| stored_random | m2_table | 300/300 | 4.6903e-05 | 2.40077e-05 | 7.6848e-10 | 0.000% | 0.000% | 19.63 / 76.08 |
| stored_random | m3_2d_o4 | 61/300 | 11.9674 | 9.71544 | 2.5927e-06 | 13.115% | 9.836% | 0.1508 / 0.1834 |
| stored_random | fast | 300/300 | 0.000325981 | 2.71592e-05 | 9.69816e-10 | 0.000% | 0.000% | 19.76 / 84.26 |
| stored_random | fast_no_laplace | 300/300 | 4.6903e-05 | 2.40077e-05 | 7.6848e-10 | 0.000% | 0.000% | 20.35 / 102.2 |
| stored_random | m1_cubic | 200/300 | 4.82303 | 4.37859 | 0.000728131 | 47.000% | 27.500% | 4.137 / 20.51 |
| fresh_random | m2_direct | 2000/2000 | 4.75703e-05 | 1.66976e-05 | 9.9365e-15 | 0.000% | 0.000% | 213.8 / 1115 |
| fresh_random | m2_table | 2000/2000 | 4.75703e-05 | 1.67026e-05 | 1.02409e-09 | 0.000% | 0.000% | 20.06 / 169.9 |
| fresh_random | m3_2d_o4 | 434/2000 | 12.6355 | 9.69745 | 3.37301e-07 | 20.507% | 13.134% | 0.1645 / 0.188 |
| fresh_random | fast | 2000/2000 | 4.75703e-05 | 1.67026e-05 | 1.04882e-09 | 0.000% | 0.000% | 21.24 / 182.9 |
| fresh_random | fast_no_laplace | 2000/2000 | 4.75703e-05 | 1.67026e-05 | 1.02409e-09 | 0.000% | 0.000% | 20.42 / 169.8 |
| fresh_random | m1_cubic | 2000/2000 | 5.28125 | 4.29022 | 0.00130715 | 52.550% | 32.200% | 4.227 / 20.2 |
| sampler_like | m2_direct | 2000/2000 | 3.31338e-06 | 9.85974e-09 | 1.42109e-14 | 0.000% | 0.000% | 279.5 / 721.9 |
| sampler_like | m2_table | 2000/2000 | 3.31976e-06 | 5.31818e-07 | 2.76159e-09 | 0.000% | 0.000% | 21.38 / 179.5 |
| sampler_like | m3_2d_o4 | 1994/2000 | 1.33145 | 0.366605 | 0.000172877 | 36.861% | 19.860% | 0.1592 / 0.1901 |
| sampler_like | fast | 2000/2000 | 6.95472e-06 | 4.82336e-06 | 1.49752e-08 | 0.000% | 0.000% | 21.74 / 187.9 |
| sampler_like | fast_no_laplace | 2000/2000 | 3.31976e-06 | 5.31818e-07 | 2.76159e-09 | 0.000% | 0.000% | 22.33 / 183.1 |
| sampler_like | m1_cubic | 2000/2000 | 0.840779 | 0.621112 | 0.00125756 | 55.350% | 20.850% | 0.9555 / 1.223 |

Dispatcher branch fractions (without Laplace: 100% method 2):

| Set | Method 2 | Laplace |
|---|---:|---:|
| regression | 90.000% | 10.000% |
| stored_random | 88.000% | 12.000% |
| fresh_random | 87.900% | 12.100% |
| sampler_like | 76.850% | 23.150% |

Table construction (JIT excluded, guards included):

| xmax | Build s | Coefficient bytes | Guard bytes | Flagged cells |
|---:|---:|---:|---:|---:|
| 100 | 20.472 | 8000000 | 990025 | 0.816% |
| 10 | 18.364 | 8000000 | 990025 | 0.343% |

Fresh random: 2000 points from `random_params(2000,808)`; same broad recipe as the saved random sets.
Sampler-like: 2000 points, seed 809; actual rho log-uniform [5,50], q uniform [.05,1], eta uniform [0,pi/2],
Cartesian peak radius log-uniform [.5,5], rotation xi uniform [0,pi], bounds [.1,100]. See `posterior_points` for the exact transform.
Method 3 is shown wherever its native three-sigma guard returns a finite value, including inaccurate low-SNR outputs.
Method 1 is the rejected iteration-005 cubic prototype, only for its [.1,100] prior; the 100 secondary-prior stored points are unavailable.
Per-point references, values, costs and branch codes are in ignored `*_full.json` files.
