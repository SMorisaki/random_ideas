# Paper-method accuracy — iteration 002

Fine-grid Eq. (4) reference on toy_small(): T=128 s, dt=1/1024 s, 20–256 Hz; T_b=(128,64,32,16,8,4) s; mchirp=mchirp_ref=4.85 Msun, t_c=126 s, m_max=4, L=5. Five modes use ToyModes defaults; quadrupole uses only (2,2) with (ratio,phase,growth)=(1,0,0). PSD: analytic aLIGO with TOY_LINES, or the same PSD without lines. Windows and coarse grids are identical for both waveform choices.

Errors are magnitudes. |Delta ln Lambda| = SNR² |Delta(h,h)/(h,h)|/2 at SNR=25, holding (d,h) fixed. Each row is one deterministic system.

| PSD | waveform | method | exact (h,h) | relative error | abs(Delta ln Lambda), SNR 25 |
| --- | --- | --- | --- | --- | --- |
| lines | (2,2) only | linear | 1.688367930e+01 | 1.403926e-06 | 4.387267e-04 |
| lines | (2,2) only | IFFT-FFT paper | 1.688367930e+01 | 1.543451e-07 | 4.823285e-05 |
| lines | (2,2) only | IFFT-FFT fast | 1.688367930e+01 | 1.543451e-07 | 4.823285e-05 |
| lines | five modes | linear | 1.869698825e+01 | 1.573092e-06 | 4.915912e-04 |
| lines | five modes | IFFT-FFT paper | 1.869698825e+01 | 2.218370e-07 | 6.932407e-05 |
| lines | five modes | IFFT-FFT fast | 1.869698825e+01 | 2.218370e-07 | 6.932407e-05 |
| smooth | (2,2) only | linear | 1.694054553e+01 | 1.404990e-06 | 4.390592e-04 |
| smooth | (2,2) only | IFFT-FFT paper | 1.694054553e+01 | 2.248638e-07 | 7.026994e-05 |
| smooth | (2,2) only | IFFT-FFT fast | 1.694054553e+01 | 2.248638e-07 | 7.026994e-05 |
| smooth | five modes | linear | 1.875949466e+01 | 4.787959e-06 | 1.496237e-03 |
| smooth | five modes | IFFT-FFT paper | 1.875949466e+01 | 2.938171e-07 | 9.181784e-05 |
| smooth | five modes | IFFT-FFT fast | 1.875949466e+01 | 2.938171e-07 | 9.181784e-05 |

Summary over the four waveform/PSD combinations (not a population):

| method | median relative error | worst relative error | median abs(Delta ln Lambda) | worst abs(Delta ln Lambda) |
| --- | --- | --- | --- | --- |
| linear | 1.489041e-06 | 4.787959e-06 | 4.653252e-04 | 1.496237e-03 |
| IFFT-FFT paper | 2.233504e-07 | 2.938171e-07 | 6.979700e-05 | 9.181784e-05 |
| IFFT-FFT fast | 2.233504e-07 | 2.938171e-07 | 6.979700e-05 | 9.181784e-05 |

No changes to the paper's formulas. The real-FFT path changes only the transform implementation. half_point_waveform returns sqrt(w)h; the pure-(2,2) half-point test covers the central 60% of band 5's unit plateau, with maximum pointwise relative error 4.213e-5. This is not a global interpolation bound: the same interior definition gives errors up to 2.5e-4 in lower bands from the Eq. (36) tail approximation.
