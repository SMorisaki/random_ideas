# Iteration 008: HOM band rules

XHM, OMP_NUM_THREADS=1. Inclination 1.0 except strong NSBH (23+2.3, 1.5); negative-spin BBH has chi1=chi2=-0.5. All five systems use the nominal durations. Strong NSBH uses the iteration-007 in-prior reference Mc=4 and range [4,9]; using nominal Mc_ref=6.4 would place it outside the band-design prior. Other references/priors are unchanged from testcases.py. Nominal GW190814 itself has Mc below 6.4 (the historical paper reference), explicitly retained for comparison.

PSDs: analytic aLIGO + the five Gaussian TOY_LINES (unchanged from cached iteration 005), and measured L1 Welch16s log-log interpolated to the fine grid. Errors are against Eq. (4). Tables show median / worst over the two PSDs. SNR=25: |Delta ln Lambda|=312.5 times relative error. Full per-PSD results and band edges are in bandrule_iter008_full.json.

## Paper-rule L scan
| System | L | K_MB | linear | IFFT-FFT | pairwise | B1 | B1_all | B0 |
|---|---|---|---|---|---|---|---|---|
| BNS | 5 | 20167 | 1.49e-07 / 1.98e-07 | 3.63e-08 / 4.80e-08 | 9.75e-09 / 1.06e-08 | 3.59e-07 / 5.06e-07 | 2.27e-07 / 4.08e-07 | 2.09e-07 / 3.11e-07 |
| BNS | 10 | 21158 | 1.03e-07 / 1.90e-07 | 3.13e-08 / 3.80e-08 | 1.82e-08 / 2.23e-08 | 2.07e-08 / 3.74e-08 | 2.44e-08 / 4.01e-08 | 1.65e-08 / 2.74e-08 |
| BNS | 20 | 23099 | 1.39e-07 / 2.61e-07 | 4.23e-08 / 7.56e-08 | 3.94e-09 / 7.15e-09 | 1.42e-07 / 1.54e-07 | 6.71e-08 / 6.80e-08 | 4.49e-09 / 7.85e-09 |
| BNS | 50 | 28543 | 2.88e-08 / 4.93e-08 | 1.62e-08 / 2.89e-08 | 1.99e-09 / 2.28e-09 | 2.00e-09 / 2.28e-09 | 2.00e-09 / 2.28e-09 | 2.00e-09 / 2.28e-09 |
| GW190814 | 5 | 4865 | 2.51e-05 / 4.01e-05 | 1.60e-07 / 1.75e-07 | 3.71e-07 / 4.28e-07 | 3.62e-07 / 5.04e-07 | 1.87e-07 / 3.36e-07 | 4.25e-07 / 5.62e-07 |
| GW190814 | 10 | 5019 | 7.03e-06 / 1.14e-05 | 6.33e-08 / 1.12e-07 | 1.12e-07 / 1.18e-07 | 3.89e-07 / 4.61e-07 | 2.52e-07 / 4.40e-07 | 1.09e-07 / 1.13e-07 |
| GW190814 | 20 | 5305 | 1.64e-06 / 2.48e-06 | 5.32e-08 / 9.18e-08 | 2.74e-08 / 2.94e-08 | 2.65e-08 / 2.75e-08 | 2.65e-08 / 2.75e-08 | 2.65e-08 / 2.75e-08 |
| GW190814 | 50 | 6026 | 7.86e-07 / 8.46e-07 | 1.49e-08 / 1.60e-08 | 5.98e-10 / 8.68e-10 | 5.50e-10 / 8.73e-10 | 5.50e-10 / 8.73e-10 | 5.50e-10 / 8.73e-10 |
| BBH | 5 | 4408 | 1.26e-05 / 1.64e-05 | 6.17e-08 / 1.16e-07 | 1.55e-08 / 1.59e-08 | 4.95e-07 / 5.84e-07 | 6.75e-07 / 9.06e-07 | 3.38e-08 / 5.95e-08 |
| BBH | 10 | 4510 | 2.07e-06 / 2.63e-06 | 1.89e-07 / 2.44e-07 | 1.89e-07 / 2.48e-07 | 1.07e-07 / 1.16e-07 | 1.48e-06 / 1.79e-06 | 1.92e-07 / 2.52e-07 |
| BBH | 20 | 4695 | 2.16e-06 / 3.98e-06 | 1.60e-07 / 2.72e-07 | 3.25e-08 / 4.03e-08 | 3.05e-08 / 3.59e-08 | 3.05e-08 / 3.59e-08 | 3.05e-08 / 3.59e-08 |
| BBH | 50 | 5146 | 1.94e-06 / 1.95e-06 | 5.02e-08 / 5.90e-08 | 4.30e-09 / 4.92e-09 | 4.03e-09 / 4.94e-09 | 4.03e-09 / 4.94e-09 | 4.03e-09 / 4.94e-09 |
| NSBH_strong | 5 | 5128 | 1.62e-06 / 2.67e-06 | 3.35e-08 / 4.33e-08 | 3.86e-08 / 4.69e-08 | 1.67e-07 / 2.95e-07 | 1.67e-07 / 2.95e-07 | 4.79e-08 / 6.75e-08 |
| NSBH_strong | 10 | 5306 | 3.19e-06 / 4.39e-06 | 4.81e-08 / 5.43e-08 | 4.58e-08 / 4.76e-08 | 2.02e-08 / 2.24e-08 | 9.01e-08 / 1.05e-07 | 4.73e-08 / 4.74e-08 |
| NSBH_strong | 20 | 5639 | 3.63e-06 / 3.82e-06 | 7.72e-09 / 1.26e-08 | 1.52e-08 / 1.61e-08 | 1.57e-08 / 1.61e-08 | 1.57e-08 / 1.61e-08 | 1.57e-08 / 1.61e-08 |
| NSBH_strong | 50 | 6490 | 4.86e-07 / 6.73e-07 | 8.30e-09 / 8.95e-09 | 5.92e-09 / 7.65e-09 | 5.97e-09 / 7.65e-09 | 5.97e-09 / 7.65e-09 | 5.97e-09 / 7.65e-09 |
| BBH_negative | 5 | 4408 | 2.85e-06 / 2.89e-06 | 1.24e-07 / 2.05e-07 | 5.21e-08 / 7.30e-08 | 6.11e-07 / 9.38e-07 | 1.01e-06 / 1.15e-06 | 6.53e-08 / 8.92e-08 |
| BBH_negative | 10 | 4510 | 5.08e-06 / 6.02e-06 | 1.64e-07 / 2.28e-07 | 1.46e-07 / 1.94e-07 | 2.39e-07 / 4.23e-07 | 2.18e-06 / 2.80e-06 | 1.43e-07 / 1.89e-07 |
| BBH_negative | 20 | 4695 | 2.96e-06 / 3.41e-06 | 1.00e-07 / 1.60e-07 | 9.67e-09 / 9.94e-09 | 1.36e-08 / 1.80e-08 | 1.36e-08 / 1.80e-08 | 1.36e-08 / 1.80e-08 |
| BBH_negative | 50 | 5146 | 8.52e-07 / 1.19e-06 | 1.00e-08 / 1.43e-08 | 1.05e-08 / 1.12e-08 | 9.88e-09 / 1.01e-08 | 9.88e-09 / 1.01e-08 | 9.88e-09 / 1.01e-08 |

Likelihood errors at SNR 25 (median / worst):
| System | L | linear | IFFT-FFT | pairwise | B1 | B1_all | B0 |
|---|---|---|---|---|---|---|---|
| BNS | 5 | 4.67e-05 / 6.19e-05 | 1.13e-05 / 1.50e-05 | 3.05e-06 / 3.30e-06 | 1.12e-04 / 1.58e-04 | 7.08e-05 / 1.27e-04 | 6.53e-05 / 9.71e-05 |
| BNS | 10 | 3.23e-05 / 5.93e-05 | 9.77e-06 / 1.19e-05 | 5.69e-06 / 6.97e-06 | 6.46e-06 / 1.17e-05 | 7.61e-06 / 1.25e-05 | 5.14e-06 / 8.56e-06 |
| BNS | 20 | 4.33e-05 / 8.16e-05 | 1.32e-05 / 2.36e-05 | 1.23e-06 / 2.23e-06 | 4.44e-05 / 4.81e-05 | 2.10e-05 / 2.13e-05 | 1.40e-06 / 2.45e-06 |
| BNS | 50 | 8.99e-06 / 1.54e-05 | 5.05e-06 / 9.04e-06 | 6.23e-07 / 7.13e-07 | 6.25e-07 / 7.13e-07 | 6.25e-07 / 7.13e-07 | 6.25e-07 / 7.13e-07 |
| GW190814 | 5 | 7.85e-03 / 1.25e-02 | 4.99e-05 / 5.46e-05 | 1.16e-04 / 1.34e-04 | 1.13e-04 / 1.57e-04 | 5.85e-05 / 1.05e-04 | 1.33e-04 / 1.76e-04 |
| GW190814 | 10 | 2.20e-03 / 3.58e-03 | 1.98e-05 / 3.49e-05 | 3.49e-05 / 3.69e-05 | 1.22e-04 / 1.44e-04 | 7.86e-05 / 1.38e-04 | 3.40e-05 / 3.52e-05 |
| GW190814 | 20 | 5.13e-04 / 7.76e-04 | 1.66e-05 / 2.87e-05 | 8.56e-06 / 9.18e-06 | 8.27e-06 / 8.60e-06 | 8.27e-06 / 8.60e-06 | 8.27e-06 / 8.60e-06 |
| GW190814 | 50 | 2.46e-04 / 2.64e-04 | 4.65e-06 / 5.00e-06 | 1.87e-07 / 2.71e-07 | 1.72e-07 / 2.73e-07 | 1.72e-07 / 2.73e-07 | 1.72e-07 / 2.73e-07 |
| BBH | 5 | 3.94e-03 / 5.13e-03 | 1.93e-05 / 3.62e-05 | 4.86e-06 / 4.96e-06 | 1.55e-04 / 1.83e-04 | 2.11e-04 / 2.83e-04 | 1.06e-05 / 1.86e-05 |
| BBH | 10 | 6.47e-04 / 8.23e-04 | 5.91e-05 / 7.62e-05 | 5.90e-05 / 7.74e-05 | 3.33e-05 / 3.61e-05 | 4.63e-04 / 5.59e-04 | 6.00e-05 / 7.88e-05 |
| BBH | 20 | 6.74e-04 / 1.24e-03 | 5.00e-05 / 8.50e-05 | 1.02e-05 / 1.26e-05 | 9.54e-06 / 1.12e-05 | 9.54e-06 / 1.12e-05 | 9.54e-06 / 1.12e-05 |
| BBH | 50 | 6.08e-04 / 6.10e-04 | 1.57e-05 / 1.84e-05 | 1.34e-06 / 1.54e-06 | 1.26e-06 / 1.54e-06 | 1.26e-06 / 1.54e-06 | 1.26e-06 / 1.54e-06 |
| NSBH_strong | 5 | 5.06e-04 / 8.36e-04 | 1.05e-05 / 1.35e-05 | 1.21e-05 / 1.46e-05 | 5.23e-05 / 9.23e-05 | 5.23e-05 / 9.23e-05 | 1.50e-05 / 2.11e-05 |
| NSBH_strong | 10 | 9.98e-04 / 1.37e-03 | 1.50e-05 / 1.70e-05 | 1.43e-05 / 1.49e-05 | 6.32e-06 / 7.00e-06 | 2.82e-05 / 3.27e-05 | 1.48e-05 / 1.48e-05 |
| NSBH_strong | 20 | 1.13e-03 / 1.19e-03 | 2.41e-06 / 3.95e-06 | 4.74e-06 / 5.04e-06 | 4.90e-06 / 5.04e-06 | 4.90e-06 / 5.04e-06 | 4.90e-06 / 5.04e-06 |
| NSBH_strong | 50 | 1.52e-04 / 2.10e-04 | 2.59e-06 / 2.80e-06 | 1.85e-06 / 2.39e-06 | 1.87e-06 / 2.39e-06 | 1.87e-06 / 2.39e-06 | 1.87e-06 / 2.39e-06 |
| BBH_negative | 5 | 8.89e-04 / 9.04e-04 | 3.86e-05 / 6.41e-05 | 1.63e-05 / 2.28e-05 | 1.91e-04 / 2.93e-04 | 3.17e-04 / 3.59e-04 | 2.04e-05 / 2.79e-05 |
| BBH_negative | 10 | 1.59e-03 / 1.88e-03 | 5.14e-05 / 7.11e-05 | 4.57e-05 / 6.07e-05 | 7.48e-05 / 1.32e-04 | 6.82e-04 / 8.75e-04 | 4.46e-05 / 5.89e-05 |
| BBH_negative | 20 | 9.25e-04 / 1.07e-03 | 3.13e-05 / 4.99e-05 | 3.02e-06 / 3.11e-06 | 4.25e-06 / 5.62e-06 | 4.25e-06 / 5.62e-06 | 4.25e-06 / 5.62e-06 |
| BBH_negative | 50 | 2.66e-04 / 3.72e-04 | 3.14e-06 / 4.46e-06 | 3.27e-06 / 3.50e-06 | 3.09e-06 / 3.15e-06 | 3.09e-06 / 3.15e-06 | 3.09e-06 / 3.15e-06 |

## Fitted d log(error) / d log(L)
Unweighted fits over all four L values. The expected -3 / -2 are tail-envelope exponents, not guaranteed exponents of the signed, PSD-weighted, band-summed error. Moving band edges changes cancellations and line positions; the dominant diagonal fold can set a different slope. Do not interpret four-point slopes as an asymptotic convergence proof.
| System | Method | median PSD error fit | analytic+lines | L1 |
|---|---|---|---|---|
| BNS | pairwise | -0.84 | -0.72 | -1.07 |
| BNS | IFFT-FFT | -0.30 | -0.92 | -0.13 |
| GW190814 | pairwise | -2.77 | -2.55 | -3.07 |
| GW190814 | IFFT-FFT | -0.97 | -1.11 | -0.78 |
| BBH | pairwise | -0.81 | -0.72 | -0.90 |
| BBH | IFFT-FFT | -0.15 | 0.49 | -0.30 |
| NSBH_strong | pairwise | -0.90 | -0.70 | -1.11 |
| NSBH_strong | IFFT-FFT | -0.78 | -0.87 | -0.70 |
| BBH_negative | pairwise | -0.98 | -0.77 | -1.11 |
| BBH_negative | IFFT-FFT | -1.10 | -0.95 | -1.15 |

## hh_half at L=5: relative errors (median / worst)
B1 uses L_margin=L_hh for the matched support criterion. Its modes include band 0="exact"; every other band is B0. The separate default-B1 column uses the unchanged L_margin=5, which need not select B0 when L_hh=3.
| System | L_hh | K_MB | increase % | matched B1 modes | default B1 modes | B0 | B1 | pairwise | IFFT-FFT |
|---|---|---|---|---|---|---|---|---|---|
| BNS | 3 | 24228 | 20.14 | exact,B0,B0,B0,B0,B0,B0 | exact,B1,B1,B1,B1,B1,B1 | 2.34e-08 / 3.21e-08 | 2.34e-08 / 3.21e-08 | 2.32e-08 / 3.21e-08 | 2.62e-08 / 2.89e-08 |
| BNS | 5 | 24770 | 22.82 | exact,B0,B0,B0,B0,B0,B0 | exact,B0,B0,B0,B0,B0,B0 | 4.39e-10 / 8.09e-10 | 4.39e-10 / 8.09e-10 | 5.76e-10 / 8.08e-10 | 2.54e-08 / 5.02e-08 |
| BNS | 8 | 25666 | 27.27 | exact,B0,B0,B0,B0,B0,B0 | exact,B0,B0,B0,B0,B0,B0 | 1.49e-08 / 2.55e-08 | 1.49e-08 / 2.55e-08 | 1.48e-08 / 2.55e-08 | 3.06e-08 / 3.56e-08 |
| GW190814 | 3 | 4936 | 1.46 | exact,B0,B0 | exact,B1,B1 | 5.12e-08 / 5.42e-08 | 5.12e-08 / 5.42e-08 | 5.31e-08 / 5.56e-08 | 3.39e-08 / 5.61e-08 |
| GW190814 | 5 | 4994 | 2.65 | exact,B0,B0 | exact,B0,B0 | 1.64e-07 / 1.84e-07 | 1.64e-07 / 1.84e-07 | 1.56e-07 / 1.68e-07 | 2.49e-07 / 3.50e-07 |
| GW190814 | 8 | 5112 | 5.08 | exact,B0,B0 | exact,B0,B0 | 8.76e-08 / 8.96e-08 | 8.76e-08 / 8.96e-08 | 8.69e-08 / 8.85e-08 | 5.70e-08 / 8.30e-08 |
| BBH | 3 | 4446 | 0.86 | exact,B0,B0 | exact,B1,B1 | 1.99e-07 / 2.87e-07 | 1.99e-07 / 2.87e-07 | 1.62e-07 / 2.14e-07 | 2.57e-07 / 4.01e-07 |
| BBH | 5 | 4497 | 2.02 | exact,B0,B0 | exact,B0,B0 | 1.97e-08 / 3.57e-08 | 1.97e-08 / 3.57e-08 | 1.87e-08 / 3.50e-08 | 2.16e-07 / 3.96e-07 |
| BBH | 8 | 4584 | 3.99 | exact,B0,B0 | exact,B0,B0 | 6.97e-08 / 8.03e-08 | 6.97e-08 / 8.03e-08 | 7.28e-08 / 8.63e-08 | 9.24e-08 / 1.25e-07 |
| NSBH_strong | 3 | 5219 | 1.77 | exact,B0,B0 | exact,B1,B1 | 1.70e-07 / 1.78e-07 | 1.70e-07 / 1.78e-07 | 1.76e-07 / 1.76e-07 | 2.25e-07 / 2.49e-07 |
| NSBH_strong | 5 | 5280 | 2.96 | exact,B0,B0 | exact,B0,B0 | 4.78e-09 / 6.07e-09 | 4.78e-09 / 6.07e-09 | 7.65e-09 / 1.18e-08 | 5.22e-08 / 6.34e-08 |
| NSBH_strong | 8 | 5414 | 5.58 | exact,B0,B0 | exact,B0,B0 | 6.38e-09 / 1.12e-08 | 6.38e-09 / 1.12e-08 | 9.77e-09 / 1.84e-08 | 4.62e-08 / 6.46e-08 |
| BBH_negative | 3 | 4446 | 0.86 | exact,B0,B0 | exact,B1,B1 | 2.45e-08 / 3.18e-08 | 2.45e-08 / 3.18e-08 | 3.99e-08 / 6.44e-08 | 7.79e-08 / 1.38e-07 |
| BBH_negative | 5 | 4497 | 2.02 | exact,B0,B0 | exact,B0,B0 | 3.39e-08 / 6.38e-08 | 3.39e-08 / 6.38e-08 | 3.84e-08 / 7.33e-08 | 2.12e-08 / 3.60e-08 |
| BBH_negative | 8 | 4584 | 3.99 | exact,B0,B0 | exact,B0,B0 | 5.14e-08 / 6.12e-08 | 5.14e-08 / 6.12e-08 | 5.37e-08 / 6.60e-08 | 6.25e-08 / 8.35e-08 |

hh_half likelihood errors at SNR 25:
| System | L_hh | B0 | B1 | pairwise | IFFT-FFT |
|---|---|---|---|---|---|
| BNS | 3 | 7.31e-06 / 1.00e-05 | 7.31e-06 / 1.00e-05 | 7.24e-06 / 1.00e-05 | 8.19e-06 / 9.03e-06 |
| BNS | 5 | 1.37e-07 / 2.53e-07 | 1.37e-07 / 2.53e-07 | 1.80e-07 / 2.53e-07 | 7.94e-06 / 1.57e-05 |
| BNS | 8 | 4.64e-06 / 7.97e-06 | 4.64e-06 / 7.97e-06 | 4.64e-06 / 7.97e-06 | 9.57e-06 / 1.11e-05 |
| GW190814 | 3 | 1.60e-05 / 1.69e-05 | 1.60e-05 / 1.69e-05 | 1.66e-05 / 1.74e-05 | 1.06e-05 / 1.75e-05 |
| GW190814 | 5 | 5.12e-05 / 5.74e-05 | 5.12e-05 / 5.74e-05 | 4.88e-05 / 5.25e-05 | 7.78e-05 / 1.09e-04 |
| GW190814 | 8 | 2.74e-05 / 2.80e-05 | 2.74e-05 / 2.80e-05 | 2.71e-05 / 2.76e-05 | 1.78e-05 / 2.59e-05 |
| BBH | 3 | 6.23e-05 / 8.98e-05 | 6.23e-05 / 8.98e-05 | 5.07e-05 / 6.69e-05 | 8.03e-05 / 1.25e-04 |
| BBH | 5 | 6.16e-06 / 1.12e-05 | 6.16e-06 / 1.12e-05 | 5.85e-06 / 1.09e-05 | 6.74e-05 / 1.24e-04 |
| BBH | 8 | 2.18e-05 / 2.51e-05 | 2.18e-05 / 2.51e-05 | 2.28e-05 / 2.70e-05 | 2.89e-05 / 3.92e-05 |
| NSBH_strong | 3 | 5.31e-05 / 5.57e-05 | 5.31e-05 / 5.57e-05 | 5.51e-05 / 5.51e-05 | 7.04e-05 / 7.79e-05 |
| NSBH_strong | 5 | 1.49e-06 / 1.90e-06 | 1.49e-06 / 1.90e-06 | 2.39e-06 / 3.67e-06 | 1.63e-05 / 1.98e-05 |
| NSBH_strong | 8 | 1.99e-06 / 3.50e-06 | 1.99e-06 / 3.50e-06 | 3.05e-06 / 5.75e-06 | 1.44e-05 / 2.02e-05 |
| BBH_negative | 3 | 7.65e-06 / 9.93e-06 | 7.65e-06 / 9.93e-06 | 1.25e-05 / 2.01e-05 | 2.43e-05 / 4.32e-05 |
| BBH_negative | 5 | 1.06e-05 / 1.99e-05 | 1.06e-05 / 1.99e-05 | 1.20e-05 / 2.29e-05 | 6.62e-06 / 1.12e-05 |
| BBH_negative | 8 | 1.61e-05 / 1.91e-05 | 1.61e-05 / 1.91e-05 | 1.68e-05 / 2.06e-05 | 1.95e-05 / 2.61e-05 |

## Implied route costs, ms
Estimated from iteration-006 warm wall-clock component medians: waveform_ms_per_point =1000*coarse_total_or_modes/K_MB; each HH cost scales linearly with K_MB as requested. One waveform + D identical-detector HH steps; extra projections/other likelihood work excluded. These are extrapolations, not fresh timings. Strong NSBH uses GW190814 costs; negative BBH uses BBH costs, including OneMode tail recovery. Waveform and HH columns remain separate.
| System | L_hh | route | waveform ms | HH ms/det | 1 detector ms | 3 detectors ms |
|---|---|---|---|---|---|---|
| BNS | 3 | IFFT-FFT paper | 6.456 | 3.302 | 9.758 | 16.361 |
| BNS | 3 | B0 hh_half | 7.756 | 0.073 | 7.829 | 7.975 |
| BNS | 3 | pairwise paper | 8.032 | 0.582 | 8.614 | 9.777 |
| BNS | 5 | IFFT-FFT paper | 6.456 | 3.302 | 9.758 | 16.361 |
| BNS | 5 | B0 hh_half | 7.930 | 0.075 | 8.004 | 8.154 |
| BNS | 5 | pairwise paper | 8.032 | 0.582 | 8.614 | 9.777 |
| BNS | 8 | IFFT-FFT paper | 6.456 | 3.302 | 9.758 | 16.361 |
| BNS | 8 | B0 hh_half | 8.217 | 0.077 | 8.294 | 8.449 |
| BNS | 8 | pairwise paper | 8.032 | 0.582 | 8.614 | 9.777 |
| GW190814 | 3 | IFFT-FFT paper | 2.420 | 0.620 | 3.039 | 4.279 |
| GW190814 | 3 | B0 hh_half | 2.455 | 0.018 | 2.473 | 2.510 |
| GW190814 | 3 | pairwise paper | 3.266 | 0.154 | 3.419 | 3.727 |
| GW190814 | 5 | IFFT-FFT paper | 2.420 | 0.620 | 3.039 | 4.279 |
| GW190814 | 5 | B0 hh_half | 2.484 | 0.019 | 2.502 | 2.540 |
| GW190814 | 5 | pairwise paper | 3.266 | 0.154 | 3.419 | 3.727 |
| GW190814 | 8 | IFFT-FFT paper | 2.420 | 0.620 | 3.039 | 4.279 |
| GW190814 | 8 | B0 hh_half | 2.542 | 0.019 | 2.561 | 2.600 |
| GW190814 | 8 | pairwise paper | 3.266 | 0.154 | 3.419 | 3.727 |
| BBH | 3 | IFFT-FFT paper | 2.209 | 0.566 | 2.775 | 3.907 |
| BBH | 3 | B0 hh_half | 2.228 | 0.017 | 2.245 | 2.280 |
| BBH | 3 | pairwise paper | 5.065 | 0.155 | 5.219 | 5.529 |
| BBH | 5 | IFFT-FFT paper | 2.209 | 0.566 | 2.775 | 3.907 |
| BBH | 5 | B0 hh_half | 2.254 | 0.017 | 2.271 | 2.306 |
| BBH | 5 | pairwise paper | 5.065 | 0.155 | 5.219 | 5.529 |
| BBH | 8 | IFFT-FFT paper | 2.209 | 0.566 | 2.775 | 3.907 |
| BBH | 8 | B0 hh_half | 2.297 | 0.018 | 2.315 | 2.350 |
| BBH | 8 | pairwise paper | 5.065 | 0.155 | 5.219 | 5.529 |
| NSBH_strong | 3 | IFFT-FFT paper | 2.550 | 0.653 | 3.204 | 4.510 |
| NSBH_strong | 3 | B0 hh_half | 2.596 | 0.020 | 2.615 | 2.654 |
| NSBH_strong | 3 | pairwise paper | 3.442 | 0.162 | 3.604 | 3.928 |
| NSBH_strong | 5 | IFFT-FFT paper | 2.550 | 0.653 | 3.204 | 4.510 |
| NSBH_strong | 5 | B0 hh_half | 2.626 | 0.020 | 2.646 | 2.685 |
| NSBH_strong | 5 | pairwise paper | 3.442 | 0.162 | 3.604 | 3.928 |
| NSBH_strong | 8 | IFFT-FFT paper | 2.550 | 0.653 | 3.204 | 4.510 |
| NSBH_strong | 8 | B0 hh_half | 2.693 | 0.020 | 2.713 | 2.753 |
| NSBH_strong | 8 | pairwise paper | 3.442 | 0.162 | 3.604 | 3.928 |
| BBH_negative | 3 | IFFT-FFT paper | 2.209 | 0.566 | 2.775 | 3.907 |
| BBH_negative | 3 | B0 hh_half | 2.228 | 0.017 | 2.245 | 2.280 |
| BBH_negative | 3 | pairwise paper | 5.065 | 0.155 | 5.219 | 5.529 |
| BBH_negative | 5 | IFFT-FFT paper | 2.209 | 0.566 | 2.775 | 3.907 |
| BBH_negative | 5 | B0 hh_half | 2.254 | 0.017 | 2.271 | 2.306 |
| BBH_negative | 5 | pairwise paper | 5.065 | 0.155 | 5.219 | 5.529 |
| BBH_negative | 8 | IFFT-FFT paper | 2.209 | 0.566 | 2.775 | 3.907 |
| BBH_negative | 8 | B0 hh_half | 2.297 | 0.018 | 2.315 | 2.350 |
| BBH_negative | 8 | pairwise paper | 5.065 | 0.155 | 5.219 | 5.529 |

## hh_full binding and resulting edges
hh_full is not always weaker than Eq. 51: evaluating sigma at f_lo and adding the upper-edge margin can make it bind, particularly in long BNS bands. It uses conservative Lambda_max rather than subtracting the positive lower lag anchor. Across these cases K_MB(paper) <= K_MB(hh_full) <= K_MB(hh_half).
| System | L_hh | K_MB | binding per band | edges Hz |
|---|---|---|---|---|
| BNS | 3 | 20167 | full_grid,paper,paper,paper,paper,paper,paper | 20.000, 46.988, 61.919, 82.356, 111.564, 157.485, 251.305, 2100.632 |
| BNS | 5 | 20370 | full_grid,hh_full,hh_full,hh_full,paper,paper,paper | 20.000, 47.766, 62.985, 83.315, 111.564, 157.485, 251.305, 2100.632 |
| BNS | 8 | 21033 | full_grid,hh_full,hh_full,hh_full,hh_full,paper,paper | 20.000, 49.507, 65.775, 87.808, 118.114, 157.485, 251.305, 2100.632 |
| GW190814 | 3 | 4865 | full_grid,paper,paper | 20.000, 57.477, 93.775, 1076.632 |
| GW190814 | 5 | 4865 | full_grid,paper,paper | 20.000, 57.477, 93.775, 1076.632 |
| GW190814 | 8 | 4901 | full_grid,hh_full,paper | 20.000, 61.737, 93.775, 1076.632 |
| BBH | 3 | 4408 | full_grid,paper,paper | 20.000, 26.465, 44.162, 1076.632 |
| BBH | 5 | 4408 | full_grid,paper,paper | 20.000, 26.465, 44.162, 1076.632 |
| BBH | 8 | 4451 | full_grid,hh_full,paper | 20.000, 31.163, 44.162, 1076.632 |
| NSBH_strong | 3 | 5128 | full_grid,paper,paper | 20.000, 75.515, 122.369, 1076.632 |
| NSBH_strong | 5 | 5128 | full_grid,paper,paper | 20.000, 75.515, 122.369, 1076.632 |
| NSBH_strong | 8 | 5159 | full_grid,hh_full,paper | 20.000, 79.135, 122.369, 1076.632 |
| BBH_negative | 3 | 4408 | full_grid,paper,paper | 20.000, 26.465, 44.162, 1076.632 |
| BBH_negative | 5 | 4408 | full_grid,paper,paper | 20.000, 26.465, 44.162, 1076.632 |
| BBH_negative | 8 | 4451 | full_grid,hh_full,paper | 20.000, 31.163, 44.162, 1076.632 |

## hh_half resulting edges (Hz; upper taper included)
| System | L_hh | edges Hz |
|---|---|---|
| BNS | 3 | 20.000, 60.031, 78.615, 103.197, 135.862, 179.519, 251.305, 2100.632 |
| BNS | 5 | 20.000, 61.229, 80.554, 106.335, 140.950, 187.783, 251.734, 2100.632 |
| BNS | 8 | 20.000, 63.024, 83.455, 111.028, 148.547, 200.095, 271.719, 2100.632 |
| GW190814 | 3 | 20.000, 65.745, 93.775, 1076.632 |
| GW190814 | 5 | 20.000, 70.736, 96.951, 1076.632 |
| GW190814 | 8 | 20.000, 78.112, 109.030, 1076.632 |
| BBH | 3 | 20.000, 30.647, 44.162, 1076.632 |
| BBH | 5 | 20.000, 34.069, 48.075, 1076.632 |
| BBH | 8 | 20.000, 39.090, 56.506, 1076.632 |
| NSBH_strong | 3 | 20.000, 86.230, 122.369, 1076.632 |
| NSBH_strong | 5 | 20.000, 91.947, 125.112, 1076.632 |
| NSBH_strong | 8 | 20.000, 100.419, 138.935, 1076.632 |
| BBH_negative | 3 | 20.000, 30.647, 44.162, 1076.632 |
| BBH_negative | 5 | 20.000, 34.069, 48.075, 1076.632 |
| BBH_negative | 8 | 20.000, 39.090, 56.506, 1076.632 |

## Block displacement: slack versus error
GW190814-like, paper L=5, inclination 1.0. Diagonal blocks stay centred; cross-class blocks shift by units of sigma_pair(f_lo,Mc_true). Compare original prior corners [6,7] with [Mc_true,Mc_true], retaining L_margin=5 and fixed block length T_b. Narrowing the prior changes the interval and centre; it does not shorten that block. The requested +/-0.25,0.5,1 sigma generally retain coverage, so shifts extend to 16 sigma to force the lag interval outside. Slack is min(Lambda_min-block_start,block_end-Lambda_max) using true Newtonian anchors; negative means bulk support escapes. Extreme-class error is the band-1 (4,1) cross contribution error divided by total exact HH; full HH also contains the invariant diagonal-tail floor.
| PSD | prior | shift sigma | min slack s | extreme slack s | HH rel error | abs Delta lnLambda | extreme error / HH |
|---|---|---|---|---|---|---|---|
| analytic_lines | prior | -16 | -4.435e+00 | -4.435e+00 | 1.252e-04 | 3.912e-02 | 3.130e-06 |
| analytic_lines | prior | -8 | -7.045e-01 | -7.045e-01 | 7.019e-08 | 2.193e-05 | 7.990e-08 |
| analytic_lines | prior | -4 | 1.075e+00 | 1.159e+00 | 3.169e-07 | 9.902e-05 | 2.426e-10 |
| analytic_lines | prior | -2 | 1.239e+00 | 2.092e+00 | 3.144e-07 | 9.826e-05 | 6.423e-10 |
| analytic_lines | prior | -1 | 1.041e+00 | 2.033e+00 | 3.136e-07 | 9.801e-05 | 6.333e-10 |
| analytic_lines | prior | -0.5 | 9.421e-01 | 1.803e+00 | 3.141e-07 | 9.817e-05 | 4.767e-10 |
| analytic_lines | prior | -0.25 | 8.927e-01 | 1.686e+00 | 3.155e-07 | 9.858e-05 | 7.057e-10 |
| analytic_lines | prior | 0 | 8.432e-01 | 1.568e+00 | 3.147e-07 | 9.833e-05 | 5.584e-10 |
| analytic_lines | prior | 0.25 | 7.936e-01 | 1.451e+00 | 3.155e-07 | 9.858e-05 | 6.941e-10 |
| analytic_lines | prior | 0.5 | 7.443e-01 | 1.334e+00 | 3.173e-07 | 9.917e-05 | 5.957e-10 |
| analytic_lines | prior | 1 | 6.452e-01 | 1.103e+00 | 3.074e-07 | 9.607e-05 | 4.694e-09 |
| analytic_lines | prior | 2 | 4.472e-01 | 6.347e-01 | 3.849e-07 | 1.203e-04 | 7.995e-08 |
| analytic_lines | prior | 4 | -2.989e-01 | -2.989e-01 | 8.825e-07 | 2.758e-04 | 2.967e-07 |
| analytic_lines | prior | 8 | -2.162e+00 | -2.162e+00 | 4.911e-05 | 1.535e-02 | 9.208e-07 |
| analytic_lines | prior | 16 | -5.893e+00 | -5.893e+00 | 1.723e-04 | 5.384e-02 | 3.051e-06 |
| analytic_lines | true_mass | -16 | -4.439e+00 | -4.439e+00 | 8.152e-05 | 2.548e-02 | 3.130e-06 |
| analytic_lines | true_mass | -8 | -7.084e-01 | -7.084e-01 | 3.356e-08 | 1.049e-05 | 4.784e-08 |
| analytic_lines | true_mass | -4 | 1.053e+00 | 1.155e+00 | 3.185e-07 | 9.953e-05 | 6.344e-10 |
| analytic_lines | true_mass | -2 | 1.261e+00 | 2.088e+00 | 3.143e-07 | 9.820e-05 | 5.834e-10 |
| analytic_lines | true_mass | -1 | 1.063e+00 | 2.037e+00 | 3.142e-07 | 9.819e-05 | 7.073e-10 |
| analytic_lines | true_mass | -0.5 | 9.638e-01 | 1.807e+00 | 3.135e-07 | 9.797e-05 | 5.278e-10 |
| analytic_lines | true_mass | -0.25 | 9.145e-01 | 1.689e+00 | 3.163e-07 | 9.884e-05 | 5.893e-10 |
| analytic_lines | true_mass | 0 | 8.649e-01 | 1.572e+00 | 3.142e-07 | 9.820e-05 | 6.407e-10 |
| analytic_lines | true_mass | 0.25 | 8.153e-01 | 1.455e+00 | 3.163e-07 | 9.883e-05 | 6.855e-10 |
| analytic_lines | true_mass | 0.5 | 7.660e-01 | 1.338e+00 | 3.174e-07 | 9.920e-05 | 5.297e-10 |
| analytic_lines | true_mass | 1 | 6.669e-01 | 1.107e+00 | 3.078e-07 | 9.620e-05 | 1.385e-09 |
| analytic_lines | true_mass | 2 | 4.689e-01 | 6.386e-01 | 3.872e-07 | 1.210e-04 | 7.912e-08 |
| analytic_lines | true_mass | 4 | -2.950e-01 | -2.950e-01 | 3.541e-06 | 1.107e-03 | 1.829e-07 |
| analytic_lines | true_mass | 8 | -2.158e+00 | -2.158e+00 | 1.651e-05 | 5.160e-03 | 7.901e-07 |
| analytic_lines | true_mass | 16 | -5.889e+00 | -5.889e+00 | 1.103e-04 | 3.447e-02 | 3.051e-06 |
| L1_real | prior | -16 | -4.435e+00 | -4.435e+00 | 2.867e-04 | 8.959e-02 | 1.138e-05 |
| L1_real | prior | -8 | -7.045e-01 | -7.045e-01 | 1.022e-05 | 3.193e-03 | 2.928e-06 |
| L1_real | prior | -4 | 1.075e+00 | 1.159e+00 | 4.529e-07 | 1.415e-04 | 2.020e-09 |
| L1_real | prior | -2 | 1.239e+00 | 2.092e+00 | 4.503e-07 | 1.407e-04 | 2.027e-09 |
| L1_real | prior | -1 | 1.041e+00 | 2.033e+00 | 4.454e-07 | 1.392e-04 | 1.972e-09 |
| L1_real | prior | -0.5 | 9.421e-01 | 1.803e+00 | 4.291e-07 | 1.341e-04 | 1.643e-09 |
| L1_real | prior | -0.25 | 8.927e-01 | 1.686e+00 | 4.329e-07 | 1.353e-04 | 1.942e-09 |
| L1_real | prior | 0 | 8.432e-01 | 1.568e+00 | 4.281e-07 | 1.338e-04 | 1.703e-09 |
| L1_real | prior | 0.25 | 7.936e-01 | 1.451e+00 | 4.239e-07 | 1.325e-04 | 1.750e-09 |
| L1_real | prior | 0.5 | 7.443e-01 | 1.334e+00 | 4.206e-07 | 1.314e-04 | 1.664e-09 |
| L1_real | prior | 1 | 6.452e-01 | 1.103e+00 | 4.071e-07 | 1.272e-04 | 5.364e-09 |
| L1_real | prior | 2 | 4.472e-01 | 6.347e-01 | 4.690e-07 | 1.466e-04 | 7.309e-08 |
| L1_real | prior | 4 | -2.989e-01 | -2.989e-01 | 7.524e-06 | 2.351e-03 | 4.445e-06 |
| L1_real | prior | 8 | -2.162e+00 | -2.162e+00 | 1.870e-04 | 5.844e-02 | 1.984e-05 |
| L1_real | prior | 16 | -5.893e+00 | -5.893e+00 | 7.609e-04 | 2.378e-01 | 1.130e-05 |
| L1_real | true_mass | -16 | -4.439e+00 | -4.439e+00 | 1.409e-04 | 4.403e-02 | 1.138e-05 |
| L1_real | true_mass | -8 | -7.084e-01 | -7.084e-01 | 1.133e-05 | 3.541e-03 | 2.817e-06 |
| L1_real | true_mass | -4 | 1.053e+00 | 1.155e+00 | 4.420e-07 | 1.381e-04 | 2.119e-09 |
| L1_real | true_mass | -2 | 1.261e+00 | 2.088e+00 | 4.509e-07 | 1.409e-04 | 2.088e-09 |
| L1_real | true_mass | -1 | 1.063e+00 | 2.037e+00 | 4.484e-07 | 1.401e-04 | 2.008e-09 |
| L1_real | true_mass | -0.5 | 9.638e-01 | 1.807e+00 | 4.324e-07 | 1.351e-04 | 1.652e-09 |
| L1_real | true_mass | -0.25 | 9.145e-01 | 1.689e+00 | 4.333e-07 | 1.354e-04 | 1.873e-09 |
| L1_real | true_mass | 0 | 8.649e-01 | 1.572e+00 | 4.275e-07 | 1.336e-04 | 1.759e-09 |
| L1_real | true_mass | 0.25 | 8.153e-01 | 1.455e+00 | 4.257e-07 | 1.330e-04 | 1.707e-09 |
| L1_real | true_mass | 0.5 | 7.660e-01 | 1.338e+00 | 4.252e-07 | 1.329e-04 | 1.639e-09 |
| L1_real | true_mass | 1 | 6.669e-01 | 1.107e+00 | 4.087e-07 | 1.277e-04 | 2.910e-09 |
| L1_real | true_mass | 2 | 4.689e-01 | 6.386e-01 | 4.721e-07 | 1.475e-04 | 7.285e-08 |
| L1_real | true_mass | 4 | -2.950e-01 | -2.950e-01 | 8.592e-06 | 2.685e-03 | 4.443e-06 |
| L1_real | true_mass | 8 | -2.158e+00 | -2.158e+00 | 1.905e-04 | 5.954e-02 | 1.960e-05 |
| L1_real | true_mass | 16 | -5.889e+00 | -5.889e+00 | 8.354e-04 | 2.610e-01 | 1.130e-05 |

For shifts through +/-1 sigma the total relative error stays in [3.07e-07, 4.48e-07]. The largest displaced error is 8.35e-04 at slack -5.889 s (L1_real, true_mass); |Delta ln Lambda|=0.261 at SNR 25. Error is not monotone in signed slack because different class/PSD contributions can cancel.

Reference cache: 10 hits, 2 misses (new spin variant only if no earlier run).
Recommendation: use hh_half with a matched L_hh for B0 when total-waveform FFT-free evaluation is the objective; choose its accuracy setting from the error table, not merely from support. Keep Eq. 51 and prior-centred class blocks for method A, and audit model-phase slack at prior corners including spin. hh_full is an optional conservative geometric check, not evidence that every finite-window tail is covered. See prior_slack_iter008.md for the separate model-track audit and limitations.
