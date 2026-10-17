# Iteration 005: waveform and HH timing
Median wall clock after warm-up, three batches per entry, each targeted at ~1 second. Python on PATH; LALSimulation 6.2.0; this machine. All times below in ms. BNS K_MB=20167; GW190814 K_MB=4865 (the latter is smaller than the reviewer's 16064-point setup). Same sorted unique sequence for the raw LAL comparisons; adapter calls sort/unsort every time. HH inputs and PSD weights are precomputed. Five negative-m calls suffice; ten are redundant.
| Component | BNS ms | GW190814 ms |
|---|---:|---:|
| XHM polarizations (LAL) | 5.9094 | 2.1170 |
| XHM five OneMode -m (LAL) | 6.2917 | 3.0578 |
| XHM ten OneMode +/-m (LAL) | 13.8843 | 6.3595 |
| HM all modes (LAL) | 45.2515 | 16.2203 |
| XHM assembled detector modes | 11.1971 | 5.0656 |
| XHM total_direct adapter | 7.2513 | 2.6341 |
| linear HH | 0.0587 | 0.0161 |
| IFFT-FFT paper HH | 6.1727 | 1.0350 |
| IFFT-FFT fast HH | 3.6403 | 0.6200 |
| pairwise pairs HH | 0.5409 | 0.1531 |
| pairwise einsum HH | 1.5722 | 0.3708 |
| B0 HH | 0.0563 | 0.0176 |
| B1 adaptive HH | 0.4170 | 0.1273 |
| Python assembly overhead (difference) | 4.9054 | 2.0078 |

## Actual adapter routes (waveform + HH)
| Route | BNS ms | ratio | GW190814 ms | ratio |
|---|---:|---:|---:|---:|
| IFFT-FFT | 10.8916 | 1.0000 | 3.2541 | 1.0000 |
| A | 11.7380 | 1.0777 | 5.2187 | 1.6037 |
| B1 | 7.6683 | 0.7041 | 2.7614 | 0.8486 |
| B0 | 7.3076 | 0.6709 | 2.6517 | 0.8149 |

## Raw LAL route sums (waveform + HH; assembly omitted)
| Route | BNS ms | ratio | GW190814 ms | ratio |
|---|---:|---:|---:|---:|
| IFFT-FFT | 9.5497 | 1.0000 | 2.7370 | 1.0000 |
| A | 6.8326 | 0.7155 | 3.2109 | 1.1732 |
| B1 | 6.3264 | 0.6625 | 2.2443 | 0.8200 |
| B0 | 5.9657 | 0.6247 | 2.1346 | 0.7799 |

## Interpretation
Ratios are costs: below 1 is faster. These route sums include waveform generation and the quadratic term only, not (d,h), priors or sampler work. Python overhead is estimated by subtracting independently timed medians; it includes sorting, unique/unsort, copies, phase and projection, and is not an exact isolated measurement. A reuses raw modes for all extrinsics, but strict orbital-phase independence is limited by the LAL mixed-32 residual documented in the accuracy report. HM all-modes includes (4,3), is a different approximant, and its raw amplitudes need physical scaling; it is not interchangeable with XHM. No ten-call route is needed for XHM.
B0 accuracy is weaker with higher modes and line-rich PSDs. B1 remains the total-waveform candidate; its accuracy relative to IFFT-FFT must be judged alongside the speed table.
HM binding failures: [{}, {}]

Go/no-go: method A is a no-go as a default speed-up with this adapter: BNS costs 1.078x the IFFT-FFT route; GW190814 costs 1.604x the IFFT-FFT route. The raw-LAL BNS route has headroom for shared sequence preparation, but the raw-LAL GW190814 route already loses on per-mode waveform cost. B1 reduces the HH step by 8.73x / 4.87x (BNS / GW190814), though its real-waveform errors do not yet satisfy the requested IFFT-FFT accuracy equivalence. Neither A nor B1 is an unqualified replacement under the current accuracy and total-cost criteria. Extrinsic matrix reuse remains a separate motivation for A.
