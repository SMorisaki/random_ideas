# Iteration 009: final performance tables

All step/route times are warm median wall-clock milliseconds on this machine. 7 seeded, interleaved batches targeting 0.12 s each; OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, OMP_WAIT_POLICY=PASSIVE; SciPy FFT workers=1. No waveform samples are cached between evaluations. Numba cache=True, fastmath=False. Construction and JIT/cache loading are excluded from steps. Wrappers include validation and flattening: shared contiguous band views are used directly; independent/reordered bands copy into reusable input buffers.

| Component (ms) | BNS L=5 | GW190814 L=5 | BBH L=5 | BNS L=50 |
|---|---:|---:|---:|---:|
| linear | 0.0592 | 0.0160 | 0.0158 | 0.0738 |
| IFFT-FFT paper | 5.5400 | 0.9825 | 0.8880 | 7.3447 |
| IFFT-FFT fast | 3.2570 | 0.5880 | 0.5392 | 4.4908 |
| pairwise pairs | 0.5576 | 0.1572 | 0.1676 | 0.6771 |
| pairwise numba | 0.2522 | 0.0610 | 0.0577 | 0.3394 |
| B0 numpy | 0.0591 | 0.0182 | 0.0169 | 0.0762 |
| B0 numba | 0.0266 | 0.0112 | 0.0104 | 0.0336 |
| B1 numpy | 0.4413 | 0.1326 | 0.1376 | 0.0772 |
| B1 numba | 0.3638 | 0.1065 | 0.1127 | 0.0270 |
| B1 L50 numpy | 0.0777 | 0.0208 | 0.0192 | 0.0765 |
| B1 L50 numba | 0.0271 | 0.0086 | 0.0081 | 0.0276 |
| B0 hh_half numpy | 0.0692 | 0.0184 | 0.0168 | 0.0756 |
| B0 hh_half numba | 0.0308 | 0.0109 | 0.0109 | 0.0315 |
| pairwise einsum | 1.5645 | 0.3704 | 0.3048 | 2.2730 |
| pairwise bare kernel | 0.2222 | 0.0511 | 0.0465 | 0.3122 |
| mode matrix numpy | 1.4411 | 0.3588 | 0.3222 | 2.0925 |
| mode matrix numba | 0.6193 | 0.1467 | 0.1296 | 0.8210 |
| mode matrix bare kernel | 0.5637 | 0.1302 | 0.1170 | 0.7751 |
| B0 bare kernel | 0.0145 | 0.0038 | 0.0036 | 0.0209 |
| raw polarizations | 6.0436 | 2.2299 | 2.1061 | 8.2769 |
| raw five OneMode | 6.3131 | 2.9548 | 2.0034 | 8.9248 |
| LAL tail recovery | 0.0003 | 0.0003 | 2.6015 | 0.0003 |
| coarse_total | 6.2072 | 2.4600 | 2.4018 | 9.0028 |
| coarse_modes | 7.5670 | 3.2867 | 5.2932 | 10.4915 |

| Grid | BNS L=5 | GW190814 L=5 | BBH L=5 | BNS L=50 |
|---|---:|---:|---:|---:|
| K_MB | 20167 | 4865 | 4408 | 28543 |
| K_MB_hh_half | 24770 | 4994 | 4497 | 28543 |
| K_MB_L50 | 28543 | 6026 | 5146 | 28543 |

## 1 detector routes

Component sums: one prepared waveform evaluation + 1 HH steps. Identical PSD costs and negligible extra detector projection are assumed. B0 hh_half waveform cost is scaled by K_MB(hh_half)/K_MB(paper), as requested. Other likelihood work is excluded; ratios use fast IFFT-FFT on paper bands.

| Route: ms / ratio | BNS L=5 | GW190814 L=5 | BBH L=5 | BNS L=50 |
|---|---:|---:|---:|---:|
| IFFT-FFT paper bands | 9.4642 / 1.000 | 3.0480 / 1.000 | 2.9409 / 1.000 | 13.4936 / 1.000 |
| pairwise paper bands | 7.8192 / 0.826 | 3.3477 / 1.098 | 5.3509 / 1.819 | 10.8309 / 0.803 |
| B1 paper bands | 6.5710 / 0.694 | 2.5665 / 0.842 | 2.5145 / 0.855 | 9.0298 / 0.669 |
| B0 hh_half bands | 7.6547 / 0.809 | 2.5361 / 0.832 | 2.4611 / 0.837 | 9.0342 / 0.670 |

## 3 detector routes

Component sums: one prepared waveform evaluation + 3 HH steps. Identical PSD costs and negligible extra detector projection are assumed. B0 hh_half waveform cost is scaled by K_MB(hh_half)/K_MB(paper), as requested. Other likelihood work is excluded; ratios use fast IFFT-FFT on paper bands.

| Route: ms / ratio | BNS L=5 | GW190814 L=5 | BBH L=5 | BNS L=50 |
|---|---:|---:|---:|---:|
| IFFT-FFT paper bands | 15.9782 / 1.000 | 4.2241 / 1.000 | 4.0193 / 1.000 | 22.4751 / 1.000 |
| pairwise paper bands | 8.3236 / 0.521 | 3.4697 / 0.821 | 5.4663 / 1.360 | 11.5098 / 0.512 |
| B1 paper bands | 7.2987 / 0.457 | 2.7794 / 0.658 | 2.7400 / 0.682 | 9.0838 / 0.404 |
| B0 hh_half bands | 7.7163 / 0.483 | 2.5580 / 0.606 | 2.4829 / 0.618 | 9.0972 / 0.405 |

Measured complete one-detector routes are also recorded (including the actual hh_half adapter rather than its scaled estimate):

| Route (measured ms) | BNS L=5 | GW190814 L=5 | BBH L=5 | BNS L=50 |
|---|---:|---:|---:|---:|
| IFFT-FFT paper bands | 9.5072 | 3.1524 | 2.9580 | 14.3541 |
| pairwise paper bands | 7.7935 | 3.4081 | 5.6693 | 10.9557 |
| B1 paper bands | 6.9720 | 2.6157 | 2.5004 | 8.9328 |
| B0 hh_half bands | 7.7506 | 2.4733 | 2.2910 | 9.2691 |

BBH coarse_modes includes all four extra polarization calls recovering OneMode tails above Mf=0.3; raw five OneMode excludes recovery. The matrix timings use the same sample magnitudes; raw intrinsic modes have identical computational cost. The public matrix excludes 4/T; the bare kernel includes it.

## C2 verdict

| System | A / fast (<=0.2) | A / raw five-mode time (<=0.1) | A met | B1 / fast (<=1/3) | B1 met |
|---|---:|---:|---|---:|---|
| BNS L=5 | 0.0774 | 0.0399 | True | 0.1117 | True |
| GW190814 L=5 | 0.1037 | 0.0206 | True | 0.1810 | True |
| BBH L=5 | 0.1070 | 0.0288 | True | 0.2091 | True |
| BNS L=50 | 0.0756 | 0.0380 | True | 0.0060 | True |

## BNS precomputation

Median of three fresh constructors without tracing; tracemalloc peak measured in a separate fresh constructor. Caller-owned bands and PSD are excluded. All internal BlockWeights caches, tensors, flattened tables and buffers are retained. MB means 10^6 bytes. Zero BlockWeights means that evaluator uses its own weights.

| L | Evaluator | seconds | BlockWeights MB | retained arrays MB | traced peak MB | <=60s and <=200MB |
|---:|---|---:|---:|---:|---:|---|
| 5 | linear | 0.1374 | 0.000 | 0.161 | 21.689 | True |
| 5 | IFFT-FFT paper | 0.1057 | 0.000 | 5.339 | 89.391 | True |
| 5 | IFFT-FFT fast | 0.1001 | 0.000 | 4.389 | 88.703 | True |
| 5 | pairwise pairs | 0.1541 | 50.528 | 60.854 | 87.561 | True |
| 5 | pairwise numba | 0.1514 | 50.528 | 64.726 | 87.561 | True |
| 5 | B0 numpy | 0.1042 | 39.125 | 39.448 | 87.560 | True |
| 5 | B0 numba | 0.1092 | 39.125 | 39.932 | 87.560 | True |
| 5 | B1 numpy | 0.1135 | 41.878 | 43.545 | 87.560 | True |
| 5 | B1 numba | 0.1163 | 41.878 | 43.545 | 87.560 | True |
| 5 | B1 L50 numpy | 0.1127 | 43.909 | 45.967 | 91.755 | True |
| 5 | B1 L50 numba | 0.1124 | 43.909 | 45.967 | 91.755 | True |
| 5 | B0 hh_half numpy | 0.1085 | 39.518 | 39.915 | 87.560 | True |
| 5 | B0 hh_half numba | 0.1098 | 39.518 | 40.509 | 87.560 | True |
| 5 | waveform adapter paper | 0.0012 | 0.000 | 0.645 | 1.152 | True |
| 5 | waveform adapter hh_half | 0.0013 | 0.000 | 0.793 | 1.414 | True |
| 50 | linear | 0.1413 | 0.000 | 0.228 | 21.958 | True |
| 50 | IFFT-FFT paper | 0.1276 | 0.000 | 7.175 | 92.674 | True |
| 50 | IFFT-FFT fast | 0.1289 | 0.000 | 5.930 | 91.691 | True |
| 50 | pairwise pairs | 0.2189 | 58.851 | 73.465 | 91.755 | True |
| 50 | pairwise numba | 0.2271 | 58.851 | 78.946 | 91.755 | True |
| 50 | B0 numpy | 0.1181 | 43.909 | 44.366 | 91.754 | True |
| 50 | B0 numba | 0.1142 | 43.909 | 45.051 | 91.754 | True |
| 50 | B1 numpy | 0.1141 | 43.909 | 45.967 | 91.754 | True |
| 50 | B1 numba | 0.1157 | 43.909 | 45.967 | 91.755 | True |
| 50 | B1 L50 numpy | 0.1134 | 43.909 | 45.967 | 91.755 | True |
| 50 | B1 L50 numba | 0.1311 | 43.909 | 45.967 | 91.755 | True |
| 50 | B0 hh_half numpy | 0.1149 | 43.909 | 44.366 | 91.754 | True |
| 50 | B0 hh_half numba | 0.1186 | 43.909 | 45.051 | 91.754 | True |
| 50 | waveform adapter paper | 0.0015 | 0.000 | 0.913 | 1.627 | True |
| 50 | waveform adapter hh_half | 0.0015 | 0.000 | 0.913 | 1.627 | True |

First dispatch (seconds; process-first use may load/compile numba and includes imports):

| Entry | BNS L=5 | GW190814 L=5 | BBH L=5 | BNS L=50 |
|---|---:|---:|---:|---:|
| pairwise numba | 0.1505 | 0.0001 | 0.0001 | 0.0005 |
| mode matrix numba | 0.0036 | 0.0002 | 0.0002 | 0.0009 |
| B0 numba | 0.0016 | 0.0000 | 0.0000 | 0.0001 |
| B1 numba | 0.0044 | 0.0002 | 0.0003 | 0.0001 |

## Accuracy checks

Fixed nominal testcases.py systems, analytic aLIGO PSD + TOY_LINES; brute-force positive fine-grid Eq. (4) from the existing local XHM reference cache. Absolute likelihood error is 312.5 times relative HH error (SNR 25). Population qualifications from iterations 007–008 still apply.

| Method | median relative HH | worst relative HH | median abs Delta ln Lambda | worst abs Delta ln Lambda |
|---|---:|---:|---:|---:|
| linear | 4.881e-06 | 9.682e-06 | 1.525e-03 | 3.026e-03 |
| IFFT-FFT paper | 1.941e-08 | 1.455e-07 | 6.067e-06 | 4.547e-05 |
| IFFT-FFT fast | 1.941e-08 | 1.455e-07 | 6.067e-06 | 4.547e-05 |
| pairwise pairs | 1.537e-08 | 3.188e-07 | 4.803e-06 | 9.963e-05 |
| pairwise numba | 1.537e-08 | 3.188e-07 | 4.803e-06 | 9.963e-05 |
| B0 numpy | 6.017e-08 | 2.972e-07 | 1.880e-05 | 9.288e-05 |
| B0 numba | 6.017e-08 | 2.972e-07 | 1.880e-05 | 9.288e-05 |
| B1 numpy | 2.178e-07 | 4.313e-07 | 6.805e-05 | 1.348e-04 |
| B1 numba | 2.178e-07 | 4.313e-07 | 6.805e-05 | 1.348e-04 |
| B1 L50 numpy | 2.283e-09 | 4.712e-09 | 7.134e-07 | 1.473e-06 |
| B1 L50 numba | 2.283e-09 | 4.712e-09 | 7.134e-07 | 1.473e-06 |
| B0 hh_half numpy | 1.985e-08 | 1.553e-07 | 6.202e-06 | 4.853e-05 |
| B0 hh_half numba | 1.985e-08 | 1.553e-07 | 6.202e-06 | 4.853e-05 |

Statistics above span the four documented configurations (BNS appears at L=5 and 50); per-configuration signed errors are in JSON. Numba and NumPy share the same approximation; tests require relative agreement 1e-12. B1 at L=50 selects B0 in every truncated band. B0 on paper L=5 bands remains diagnostic; the recommended total-waveform route uses hh_half.

Defaults remain unchanged. Recommend opt-in numba for repeated evaluations, especially pairwise; promotion to the default can be considered separately because import/compilation latency and a required numba dependency are visible for short jobs. B0 with hh_half remains the preferred total-waveform configuration.
