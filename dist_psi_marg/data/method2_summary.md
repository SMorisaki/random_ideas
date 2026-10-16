Method 2: measured absolute log errors (max / 99% / median).

Default budget: 64 paired theta nodes; early stopping tolerance 2e-6.
Mode 0: periodic uniform; mode 1: corrected sinh trapezoid; mode 2: Gauss–Legendre on two asymmetric sinh intervals.

| Budget | Overall max / 99% / median | Median µs | Kernel calls median / max | Theta nodes median / max |
|---:|---|---:|---|---|
| 16 | 0.253074 / 0.198394 / 0.000102923 | 84.47 | 17 / 32 | 16 / 16 |
| 24 | 0.048264 / 0.0383353 / 7.41111e-07 | 118.03 | 25 / 48 | 24 / 24 |
| 32 | 0.0165184 / 0.00770255 / 7.01503e-09 | 158.33 | 34 / 64 | 32 / 32 |
| 48 | 0.00119659 / 0.000528284 / 4.583e-13 | 228.78 | 51 / 96 | 48 / 48 |
| 64 | 4.6903e-05 / 2.59424e-05 / 1.77636e-15 | 249.50 | 64 / 128 | 64 / 64 |
| 96 | 6.01903e-06 / 1.8877e-07 / 1.11022e-15 | 352.00 | 96 / 192 | 48 / 96 |
| 128 | 1.88532e-06 / 1.04675e-08 / 1.11022e-15 | 379.20 | 65 / 256 | 64 / 128 |

dataset: max / 99% / median

| Regime | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|
| random | 0.253 / 0.157 / 9.96e-05 | 0.0396 / 0.0295 / 6.43e-07 | 0.0142 / 0.0056 / 4.52e-09 | 0.00104 / 0.000167 / 2.7e-13 | 4.69e-05 / 2.4e-05 / 1.67e-15 | 6.02e-06 / 2.23e-07 / 1.11e-15 | 1.89e-06 / 4.1e-09 / 1.11e-15 |
| regression | 0.239 / 0.182 / 0.000138 | 0.0483 / 0.0458 / 1.2e-06 | 0.0165 / 0.0133 / 1.84e-08 | 0.0012 / 0.000955 / 1.88e-10 | 3.78e-05 / 3.25e-05 / 4.58e-13 | 1.29e-07 / 1e-07 / 1.42e-14 | 1.76e-08 / 1.08e-08 / 2.44e-15 |

snr_regime: max / 99% / median

| Regime | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|
| high | 0.253 / 0.247 / 0.00193 | 0.0396 / 0.039 / 4.72e-05 | 0.00908 / 0.00824 / 1.33e-06 | 0.000578 / 0.000559 / 1.48e-09 | 4.69e-05 / 3.79e-05 / 1.14e-12 | 2.22e-07 / 1.38e-07 / 1.99e-13 | 1.05e-09 / 5.87e-10 / 2.84e-14 |
| intermediate | 0.0934 / 0.0876 / 0.000113 | 0.0483 / 0.0444 / 6.75e-07 | 0.0165 / 0.0151 / 2.91e-09 | 0.0012 / 0.0011 / 1.42e-13 | 2.71e-05 / 2.53e-05 / 5.77e-15 | 3.03e-07 / 1.55e-07 / 1.42e-14 | 1.55e-09 / 9.01e-10 / 8.88e-16 |
| low | 0.0133 / 0.0106 / 4.15e-14 | 0.00176 / 0.00151 / 1.33e-15 | 0.000474 / 0.00012 / 1.44e-15 | 9.41e-05 / 3.82e-06 / 1.11e-15 | 3.03e-05 / 1.12e-06 / 1.11e-15 | 6.02e-06 / 2.23e-07 / 8.88e-16 | 1.89e-06 / 8.35e-08 / 1.11e-15 |

q_regime: max / 99% / median

| Regime | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|
| 0.01<=q<0.5 | 0.0141 / 0.0112 / 8.17e-05 | 0.00209 / 0.00175 / 6.53e-07 | 0.000149 / 0.000118 / 3.77e-09 | 8.08e-07 / 5.78e-07 / 2.52e-13 | 5.57e-09 / 4.49e-09 / 2.22e-15 | 2.58e-10 / 7.85e-11 / 1.11e-15 | 2.91e-11 / 1.22e-11 / 8.88e-16 |
| 0.5<=q<1 | 0.0091 / 0.00901 / 8.77e-05 | 0.000718 / 0.000596 / 6.75e-07 | 4.74e-05 / 3.97e-05 / 2.15e-09 | 4.45e-07 / 3.59e-07 / 2.27e-13 | 1.46e-08 / 1.9e-09 / 1.33e-15 | 3.27e-09 / 3.87e-10 / 1.11e-15 | 1.46e-08 / 1.69e-09 / 1.11e-15 |
| 0<q<0.01 | 0.0333 / 0.0329 / 0.0172 | 0.0483 / 0.0475 / 0.00911 | 0.00565 / 0.00556 / 0.000987 | 0.000268 / 0.000262 / 1.24e-05 | 2.08e-05 / 2.04e-05 / 3.7e-07 | 5.61e-08 / 5.5e-08 / 2.18e-10 | 1.01e-10 / 9.93e-11 / 0 |
| q=0 | 0.253 / 0.251 / 0.0112 | 0.0418 / 0.0411 / 0.00119 | 0.0165 / 0.0158 / 0.000172 | 0.0012 / 0.00115 / 8.74e-06 | 4.69e-05 / 4.41e-05 / 1.13e-07 | 6.02e-06 / 4.46e-06 / 1.2e-09 | 1.89e-06 / 1.4e-06 / 1.09e-11 |
| q=1 | 0.0118 / 0.0112 / 1.99e-06 | 0.00162 / 0.00157 / 8.16e-10 | 0.000131 / 0.000126 / 1.72e-11 | 2.89e-07 / 2.05e-07 / 1.67e-15 | 5.09e-09 / 4.67e-09 / 1.22e-15 | 1.46e-11 / 1.28e-11 / 8.88e-16 | 3.13e-11 / 2.34e-11 / 1.11e-15 |

peak_regime: max / 99% / median

| Regime | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|
| above | 0.253 / 0.247 / 0.000335 | 0.0418 / 0.0396 / 7.57e-06 | 0.0165 / 0.0142 / 1.13e-07 | 0.0012 / 0.00104 / 3.85e-11 | 4.69e-05 / 3.79e-05 / 2.84e-14 | 3.03e-07 / 2.23e-07 / 1.42e-14 | 1.55e-09 / 1.05e-09 / 7.77e-16 |
| below | 0.0133 / 0.0117 / 3.78e-05 | 0.00176 / 0.00162 / 2.48e-07 | 0.000474 / 0.000131 / 4e-09 | 9.41e-05 / 1.4e-05 / 2.1e-12 | 3.03e-05 / 4.43e-06 / 2.89e-15 | 6.02e-06 / 8.81e-07 / 1.11e-15 | 1.89e-06 / 2.81e-07 / 1.33e-15 |
| inside | 0.0333 / 0.0164 / 1.67e-05 | 0.0483 / 0.00728 / 3.53e-08 | 0.00565 / 0.000769 / 4.38e-10 | 0.000268 / 9.41e-06 / 2.13e-14 | 2.08e-05 / 2.78e-07 / 1.11e-15 | 5.61e-08 / 2.61e-09 / 1.11e-15 | 1.76e-08 / 7.68e-11 / 1.11e-15 |

The random set contains 100 points for each of seeds 2, 3, 4; seed 3 uses xmax=10, the others xmax=100. SNR bins use actual annular SNR: low <4, intermediate [4,20), high >=20. Peak bins refer to the clipped radial maximum of the full likelihood.

Timing is the median of individual scalar wall times over three warmed passes (single-thread numba); the JSON also records median batch means.

Peak geometry uses the clipped radial exponent maximum minus 4 log(r*), a 128-angle scan, and 40 golden-section iterations. At most two maxima within 40 log units are retained. Curvature sets w=min(0.3,1/sqrt(abs(p''))); split intervals use min(0.3,1.5*w). The width switch is w=0.22.

For a single symmetric sinh interval, periodicity gives the exact first Euler–Maclaurin correction -pi*dtau^2*h(endpoint)/12. Its maximum effect among the single-sharp cases is 2.72798e-08 in log likelihood. The correction is included even when negligible.

Ablation at 64 nodes: forced uniform / forced single sinh / adaptive, maximum errors:
- broad: 2.40891e-05 / 0.00452568 / 2.40891e-05
- single_sharp: 409.798 / 4.6903e-05 / 4.6903e-05
- two_peaks: 2.62142 / 4.29297e-06 / 2.71273e-05

Additional boundary bimodal case: split trapezoid error 0.000291325; with reused-node first EM correction 0.00128671; split sinh–Legendre 2.2387e-07.

Valley endpoint layers defeat the reused-node derivative stencil; Legendre nodes cluster at asymmetric sinh endpoints without extra kernels.

Largest errors at the 64-node budget:

- random/seed4_075: 4.6903e-05; actual SNR=1331.55, q=0, radial peak above, mode=1.
- regression/random_05: 3.78202e-05; actual SNR=655.459, q=0, radial peak above, mode=1.
- random/seed3_065: 3.02979e-05; actual SNR=0, q=0, radial peak below, mode=1.
- random/seed3_080: 2.71273e-05; actual SNR=8.57891, q=0, radial peak above, mode=2.
- regression/random_16: 2.40891e-05; actual SNR=6.98975, q=5.73959e-17, radial peak above, mode=0.

Degenerate weak-axis profiles and radial endpoint transitions dominate the residual angular discretization error. The hardest upper-endpoint cases reach actual SNR >1300, despite sampling R/sqrt(lam1)<=300. The reference values for regression points are loaded unchanged; only the 300 random references and the additional boundary study are computed.
