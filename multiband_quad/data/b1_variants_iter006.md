# Iteration 006: B1 variants and L=50
Three nominal XHM systems from testcases.py; analytic aLIGO+TOY_LINES PSD, inclination=1.0, other parameters unchanged (t_c=T-2, phi_ref=0.7). Fine-grid totals reuse iteration-005 cache. SNR25 likelihood error magnitude = 312.5 relative HH error. Timings are median warm wall time over five batches targeting 0.15 s each on this machine; waveform evaluation excluded.
OMP_NUM_THREADS=1, OMP_WAIT_POLICY=PASSIVE; scipy FFTs use one worker.
Padding uses K''=next_fast_len(p K_b), p in {1,2,4}; adaptive B0/exact shortcuts remain active. all_half_bins sums K'' half bins with 2M-periodic weights, while integer power remains band-only (padded integer samples are zero). These are distinct finite interpolants; padding need not converge to the paper method.
| System | p | all half bins | relative HH error | abs likelihood error | step ms | cost/default | IFFT-FFT error | IFFT-FFT ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| BNS | 1 | False | 2.123e-07 | 6.636e-05 | 0.4519 | 1.000 | 2.458e-08 | 3.5789 |
| BNS | 1 | True | 4.553e-08 | 1.423e-05 | 0.4455 | 0.986 | 2.458e-08 | 3.5789 |
| BNS | 2 | False | 2.606e-07 | 8.145e-05 | 0.8726 | 1.931 | 2.458e-08 | 3.5789 |
| BNS | 2 | True | 2.985e-08 | 9.327e-06 | 0.9185 | 2.032 | 2.458e-08 | 3.5789 |
| BNS | 4 | False | 2.595e-07 | 8.110e-05 | 1.8479 | 4.089 | 2.458e-08 | 3.5789 |
| BNS | 4 | True | 2.213e-07 | 6.915e-05 | 1.7695 | 3.916 | 2.458e-08 | 3.5789 |
| GW190814 | 1 | False | 2.195e-07 | 6.859e-05 | 0.1272 | 1.000 | 1.446e-07 | 0.5878 |
| GW190814 | 1 | True | 3.839e-08 | 1.200e-05 | 0.1389 | 1.091 | 1.446e-07 | 0.5878 |
| GW190814 | 2 | False | 2.298e-07 | 7.181e-05 | 0.2371 | 1.864 | 1.446e-07 | 0.5878 |
| GW190814 | 2 | True | 2.231e-07 | 6.971e-05 | 0.2389 | 1.878 | 1.446e-07 | 0.5878 |
| GW190814 | 4 | False | 2.248e-07 | 7.026e-05 | 0.5576 | 4.383 | 1.446e-07 | 0.5878 |
| GW190814 | 4 | True | 2.209e-07 | 6.904e-05 | 0.6323 | 4.970 | 1.446e-07 | 0.5878 |
| BBH | 1 | False | 4.050e-07 | 1.266e-04 | 0.1528 | 1.000 | 7.576e-09 | 0.5582 |
| BBH | 1 | True | 4.442e-07 | 1.388e-04 | 0.1526 | 0.999 | 7.576e-09 | 0.5582 |
| BBH | 2 | False | 4.221e-07 | 1.319e-04 | 0.2486 | 1.627 | 7.576e-09 | 0.5582 |
| BBH | 2 | True | 3.867e-08 | 1.208e-05 | 0.2503 | 1.638 | 7.576e-09 | 0.5582 |
| BBH | 4 | False | 4.219e-07 | 1.319e-04 | 0.5820 | 3.809 | 7.576e-09 | 0.5582 |
| BBH | 4 | True | 4.219e-06 | 1.318e-03 | 0.5997 | 3.925 | 7.576e-09 | 0.5582 |

## Variant aggregate over the three fixed systems
| p / all half bins | median rel | worst rel | median likelihood | worst likelihood |
|---|---:|---:|---:|---:|
| 1/False | 2.195e-07 | 4.050e-07 | 6.859e-05 | 1.266e-04 |
| 1/True | 4.553e-08 | 4.442e-07 | 1.423e-05 | 1.388e-04 |
| 2/False | 2.606e-07 | 4.221e-07 | 8.145e-05 | 1.319e-04 |
| 2/True | 3.867e-08 | 2.231e-07 | 1.208e-05 | 6.971e-05 |
| 4/False | 2.595e-07 | 4.219e-07 | 8.110e-05 | 1.319e-04 |
| 4/True | 2.213e-07 | 4.219e-06 | 6.915e-05 | 1.318e-03 |

## Default decision
Eligibility: >1.5x lower error and <=2x step cost relative to p=1, band-only, on each system.
BNS: eligible variants [{'pad_factor': 1, 'all_half_bins': True}].
GW190814: eligible variants [{'pad_factor': 1, 'all_half_bins': True}].
BBH: eligible variants [{'pad_factor': 2, 'all_half_bins': True}].
The defaults remain p=1 and band-only. A system-specific win alone is insufficient to change the shared default; compare all three rows for any proposed variant.

## L=5 versus L=50, unchanged durations
Pairwise retains its default L_margin=5, sigma_pair, block diagonal and the same chirp-mass prior; only the band-construction L changes. B1 retains p=1 and band-only. The same exact fine-grid norm applies to both window partitions.
| System | method | L=5 K_MB | L=5 relative / likelihood | L=50 K_MB | L=50 relative / likelihood |
|---|---|---:|---:|---:|---:|
| BNS | linear | 20167 | 1.005e-07 / 3.140e-05 | 28543 | 8.227e-09 / 2.571e-06 |
| BNS | IFFT-FFT | 20167 | 2.458e-08 / 7.680e-06 | 28543 | 3.397e-09 / 1.062e-06 |
| BNS | pairwise | 20167 | 8.935e-09 / 2.792e-06 | 28543 | 2.283e-09 / 7.133e-07 |
| BNS | B1 | 20167 | 2.123e-07 / 6.636e-05 | 28543 | 2.283e-09 / 7.134e-07 |
| GW190814 | linear | 4865 | 1.017e-05 / 3.179e-03 | 6026 | 8.460e-07 / 2.644e-04 |
| GW190814 | IFFT-FFT | 4865 | 1.446e-07 / 4.518e-05 | 6026 | 1.600e-08 / 5.001e-06 |
| GW190814 | pairwise | 4865 | 3.147e-07 / 9.833e-05 | 6026 | 8.680e-10 / 2.712e-07 |
| GW190814 | B1 | 4865 | 2.195e-07 / 6.859e-05 | 6026 | 8.731e-10 / 2.729e-07 |
| BBH | linear | 4408 | 8.777e-06 / 2.743e-03 | 5146 | 1.953e-06 / 6.104e-04 |
| BBH | IFFT-FFT | 4408 | 7.576e-09 / 2.367e-06 | 5146 | 4.138e-08 / 1.293e-05 |
| BBH | pairwise | 4408 | 1.522e-08 / 4.758e-06 | 5146 | 4.917e-09 / 1.537e-06 |
| BBH | B1 | 4408 | 4.050e-07 / 1.266e-04 | 5146 | 4.935e-09 / 1.542e-06 |

## L aggregate over the three fixed systems
| L / method | median rel | worst rel | median likelihood | worst likelihood |
|---|---:|---:|---:|---:|
| 5/linear | 8.777e-06 | 1.017e-05 | 2.743e-03 | 3.179e-03 |
| 5/IFFT-FFT | 2.458e-08 | 1.446e-07 | 7.680e-06 | 4.518e-05 |
| 5/pairwise | 1.522e-08 | 3.147e-07 | 4.758e-06 | 9.833e-05 |
| 5/B1 | 2.195e-07 | 4.050e-07 | 6.859e-05 | 1.266e-04 |
| 50/linear | 8.460e-07 | 1.953e-06 | 2.644e-04 | 6.104e-04 |
| 50/IFFT-FFT | 1.600e-08 | 4.138e-08 | 5.001e-06 | 1.293e-05 |
| 50/pairwise | 2.283e-09 | 4.917e-09 | 7.133e-07 | 1.537e-06 |
| 50/B1 | 2.283e-09 | 4.935e-09 | 7.134e-07 | 1.542e-06 |

Adaptive B1 band choices (a B0 band executes no FFT):
BNS, L=5: ['exact', 'B1', 'B1', 'B1', 'B1', 'B1', 'B1'].
BNS, L=50: ['exact', 'B0', 'B0', 'B0', 'B0', 'B0', 'B0'].
GW190814, L=5: ['exact', 'B1', 'B1'].
GW190814, L=50: ['exact', 'B0', 'B0'].
BBH, L=5: ['exact', 'B1', 'B1'].
BBH, L=50: ['exact', 'B0', 'B0'].
L=50 reduces the default pairwise and B1 errors in these cases, with B1 selecting B0 in both truncated bands (all six truncated bands for BNS). The BBH IFFT-FFT error increases from 7.58e-9 to 4.14e-8: error cancellation at one fixed point does not imply monotonic improvement with L.

These are one case per system, not a population test. No existing defaults or iteration-001–005 script behaviour changed. LAL refused parameter sets: none.
Fine-reference cache hits/misses: 3/0.
