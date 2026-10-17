# HH-step timing — iteration 002

Machine: x86_64, Python 3.12.15, numpy 2.5.4, scipy 1.18.1. Wall-clock median per call after one warm-up; 7 batches of 100 calls. scipy real FFTs use one worker. Both models use all five default toy modes and analytic aLIGO PSD + TOY_LINES. Setup/precomputation is outside all timings.

toy_small: T=128 s, dt=1/1024 s, 20–256 Hz, T_b=(128,64,32,16,8,4) s, mchirp=mchirp_ref=4.85 Msun, t_c=126 s. GW190814-like uses iteration-001 geometry: T=16 s, dt=1/4096 s, 20–1024 Hz, T_b=(16,8,4) s, mchirp=mchirp_ref=6.4 Msun, t_c=14 s. Both use m_max=4, L=5.

The waveform column times model.total(f) on the precomputed concatenated coarse grid. Each HH column reuses those total-waveform samples and excludes waveform evaluation, container construction, and precomputation. The fast implementation reuses input/work buffers; scipy allocates FFT outputs.

| system | K_MB | waveform (ms) | linear HH (ms) | paper IFFT-FFT HH (ms) | fast IFFT-FFT HH (ms) |
| --- | --- | --- | --- | --- | --- |
| toy_small | 3294 | 0.866916 | 0.018595 | 1.409644 | 0.700053 |
| GW190814-like | 4865 | 1.181315 | 0.017855 | 1.294200 | 0.704633 |
