# HH-step timing — iteration 004

Machine: x86_64, Python 3.12.15, numpy 2.5.4, scipy 1.18.1. Wall-clock median per call after one warm-up; 7 batches of 100 calls. scipy real FFTs use one worker. Both models use all five default toy modes. toy_small uses analytic aLIGO PSD + TOY_LINES; GW190814-like uses the real L1 Welch-16s PSD. Setup/precomputation is outside all timings.

toy_small: T=128 s, dt=1/1024 s, 20–256 Hz, T_b=(128,64,32,16,8,4) s, mchirp=mchirp_ref=4.85 Msun, t_c=126 s. GW190814-like uses iteration-001 geometry: T=16 s, dt=1/4096 s, 20–1024 Hz, T_b=(16,8,4) s, mchirp=mchirp_ref=6.4 Msun, t_c=14 s. Both use m_max=4, L=5.

The waveform column times model.total(f) on the precomputed concatenated coarse grid; the modes column evaluates each model.mode once on that grid and retains a list (no CoarseModes construction). Each HH column reuses previously evaluated samples and excludes waveform evaluation, container construction, and precomputation. The fast implementation reuses input/work buffers; scipy allocates FFT outputs.

| system | K_MB | waveform total (ms) | waveform modes (ms) | linear HH (ms) | paper IFFT-FFT HH (ms) | fast IFFT-FFT HH (ms) | pairwise einsum HH (ms) | pairwise pairs HH (ms) | B0 HH (ms) | B1 adaptive HH (ms) | B1 forced HH (ms) | fast/adaptive speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toy_small | 3294 | 0.965409 | 0.827774 | 0.019446 | 1.324284 | 0.680781 | 0.319823 | 0.194212 | 0.020410 | 0.130856 | 0.129486 | 5.203x |
| GW190814-like | 4865 | 1.061272 | 1.018860 | 0.017297 | 1.164588 | 0.684404 | 0.410027 | 0.162451 | 0.018863 | 0.144226 | 0.140648 | 4.745x |

Default PairwiseHH implementation: pairs. Same-m modes are added first; each distinct lag class is then contracted once. Both implementations use L_margin=5, sigma_pair, diag=block. The five-mode toy timing does not predict the cost of real per-mode waveform calls. GW190814-like band 0 emits unwrapped-SPA slack warnings, but M_b=N_b there: its quadrature retains every circular lag and is exact for any placement.

## toy_small: B1 adaptive FFT lengths

| b | K_b | K' | M_b | 2 M_b | lag_max (s) | T_b/2 (s) | margin (s) | mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 752 | 756 | 8192 | 16384 | 97.822154 | 64.000000 | 18.285710 | exact |
| 1 | 589 | 594 | 8192 | 16384 | 51.519213 | 32.000000 | 11.766982 | B1 |
| 2 | 429 | 432 | 4096 | 8192 | 23.979002 | 16.000000 | 6.955368 | B1 |
| 3 | 345 | 350 | 4096 | 8192 | 10.630754 | 8.000000 | 3.976024 | B1 |
| 4 | 358 | 360 | 2048 | 4096 | 4.228414 | 4.000000 | 2.109518 | B1 |
| 5 | 821 | 825 | 4096 | 8192 | 1.221304 | 2.000000 | 0.898209 | B1 |

| b | paper IFFT | paper FFT=min(2M,N) | B1 runtime IFFT, FFT | forced runtime IFFT, FFT | K'/M |
| --- | --- | --- | --- | --- | --- |
| 0 | 8192 | 8192 | 0, 0 | 0, 0 | 0.092285 |
| 1 | 8192 | 16384 | 594, 594 | 594, 594 | 0.072510 |
| 2 | 4096 | 8192 | 432, 432 | 432, 432 | 0.105469 |
| 3 | 4096 | 8192 | 350, 350 | 350, 350 | 0.085449 |
| 4 | 2048 | 4096 | 360, 360 | 360, 360 | 0.175781 |
| 5 | 4096 | 8192 | 825, 825 | 825, 825 | 0.201416 |

## GW190814-like: B1 adaptive FFT lengths

| b | K_b | K' | M_b | 2 M_b | lag_max (s) | T_b/2 (s) | margin (s) | mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 600 | 600 | 2048 | 4096 | 61.617891 | 8.000000 | 14.512647 | exact |
| 1 | 310 | 315 | 2048 | 4096 | 4.132803 | 4.000000 | 2.264599 | B1 |
| 2 | 3955 | 3960 | 16384 | 32768 | 1.188182 | 2.000000 | 0.961181 | B1 |

| b | paper IFFT | paper FFT=min(2M,N) | B1 runtime IFFT, FFT | forced runtime IFFT, FFT | K'/M |
| --- | --- | --- | --- | --- | --- |
| 0 | 2048 | 2048 | 0, 0 | 0, 0 | 0.292969 |
| 1 | 2048 | 4096 | 315, 315 | 315, 315 | 0.153809 |
| 2 | 16384 | 32768 | 3960, 3960 | 3960, 3960 | 0.241699 |

B1 adaptive uses the prescribed 5 sigma_mmax margin with m_values=(1,2,3,4). It selects no B0 band in either five-mode geometry: the top bands just fail lag_max+margin<=T_b/2. exact bands execute no FFT even in forced mode. K' in summary() remains the available diagnostic length. M=N is the exact coarse-grid shortcut; 2M=N still needs half points. These timings compare HH costs only, not total likelihood speedups.
