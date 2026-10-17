# HH-step timing — iteration 003

Machine: x86_64, Python 3.12.15, numpy 2.5.4, scipy 1.18.1. Wall-clock median per call after one warm-up; 7 batches of 100 calls. scipy real FFTs use one worker. Both models use all five default toy modes and analytic aLIGO PSD + TOY_LINES. Setup/precomputation is outside all timings.

toy_small: T=128 s, dt=1/1024 s, 20–256 Hz, T_b=(128,64,32,16,8,4) s, mchirp=mchirp_ref=4.85 Msun, t_c=126 s. GW190814-like uses iteration-001 geometry: T=16 s, dt=1/4096 s, 20–1024 Hz, T_b=(16,8,4) s, mchirp=mchirp_ref=6.4 Msun, t_c=14 s. Both use m_max=4, L=5.

The waveform column times model.total(f) on the precomputed concatenated coarse grid; the modes column evaluates each model.mode once on that grid and retains a list (no CoarseModes construction). Each HH column reuses previously evaluated samples and excludes waveform evaluation, container construction, and precomputation. The fast implementation reuses input/work buffers; scipy allocates FFT outputs.

| system | K_MB | waveform total (ms) | waveform modes (ms) | linear HH (ms) | paper IFFT-FFT HH (ms) | fast IFFT-FFT HH (ms) | pairwise einsum HH (ms) | pairwise pairs HH (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toy_small | 3294 | 0.936587 | 0.853732 | 0.018846 | 1.515273 | 0.714001 | 0.348586 | 0.223866 |
| GW190814-like | 4865 | 1.180125 | 1.101644 | 0.017202 | 1.397527 | 0.681636 | 0.402679 | 0.159863 |

Default PairwiseHH implementation: pairs. Same-m modes are added first; each distinct lag class is then contracted once. Both implementations use L_margin=5, sigma_pair, diag=block. The five-mode toy timing does not predict the cost of real per-mode waveform calls. GW190814-like band 0 emits unwrapped-SPA slack warnings, but M_b=N_b there: its quadrature retains every circular lag and is exact for any placement.
