# Accuracy — iteration 003

toy_small(): T=128 s, dt=1/1024 s, 20–256 Hz, T_b=(128,64,32,16,8,4) s; Mc=Mc_ref=4.85 Msun, tc=126 s, m_max=4, band L=5. Five modes use ToyModes defaults; quadrupole uses (ratio,phase,growth)=(1,0,0). Analytic aLIGO PSD with TOY_LINES or smooth. Reference is brute-force fine-grid Eq. (4). Pairwise: L_margin=5, sigma_pair, impl=pairs. Each row is one deterministic case. |Delta ln Lambda| = 25² relative error / 2.

| PSD | modes | method | relative (h,h) error | abs(Delta ln Lambda), SNR 25 |
| --- | --- | --- | --- | --- |
| lines | (2,2) | linear | 1.403926e-06 | 4.387267e-04 |
| lines | (2,2) | IFFT-FFT fast | 1.543451e-07 | 4.823285e-05 |
| lines | (2,2) | pairwise block | 2.214771e-07 | 6.921160e-05 |
| lines | (2,2) | pairwise linear | 1.403926e-06 | 4.387267e-04 |
| lines | five modes | linear | 1.573092e-06 | 4.915912e-04 |
| lines | five modes | IFFT-FFT fast | 2.218370e-07 | 6.932407e-05 |
| lines | five modes | pairwise block | 3.530529e-07 | 1.103290e-04 |
| lines | five modes | pairwise linear | 1.285751e-06 | 4.017973e-04 |
| smooth | (2,2) | linear | 1.404990e-06 | 4.390592e-04 |
| smooth | (2,2) | IFFT-FFT fast | 2.248638e-07 | 7.026994e-05 |
| smooth | (2,2) | pairwise block | 2.207669e-07 | 6.898965e-05 |
| smooth | (2,2) | pairwise linear | 1.404990e-06 | 4.390592e-04 |
| smooth | five modes | linear | 4.787959e-06 | 1.496237e-03 |
| smooth | five modes | IFFT-FFT fast | 2.938171e-07 | 9.181784e-05 |
| smooth | five modes | pairwise block | 3.536371e-07 | 1.105116e-04 |
| smooth | five modes | pairwise linear | 1.285565e-06 | 4.017390e-04 |

Summary across four deterministic cases (not a population):

| method | median relative error | worst relative error | median abs(Delta ln Lambda) | worst abs(Delta ln Lambda) |
| --- | --- | --- | --- | --- |
| linear | 1.489041e-06 | 4.787959e-06 | 4.653252e-04 | 1.496237e-03 |
| IFFT-FFT fast | 2.233504e-07 | 2.938171e-07 | 6.979700e-05 | 9.181784e-05 |
| pairwise block | 2.872650e-07 | 3.536371e-07 | 8.977031e-05 | 1.105116e-04 |
| pairwise linear | 1.344839e-06 | 1.404990e-06 | 4.202620e-04 | 4.390592e-04 |
