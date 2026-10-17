# Accuracy — iteration 004

All references use brute-force Eq. (4) on the PSDGrid fine grid; |Delta ln Lambda| =25² |Delta(h,h)|/(2(h,h)_exact). Each table entry is one deterministic system, not a population. Pairwise uses iteration-003 defaults (pairs, block, 5 sigma_pair). B1 uses the m values actually present, L_margin=5 sigma_mmax, K'=next_fast_len(K_b). No waveform calls enter the HH step.

Toy geometry: T=128 s, dt=1/1024 s, 20–256 Hz, T_b=(128,64,32,16,8,4) s; Mc=Mc_ref=4.85 Msun, tc=126 s. Default ratios=(1,.25,.35,.1,.15); strong-HOM ratios=(1,.3,.6,.1,.4), same phases, no amplitude growth. Both use smooth analytic aLIGO and TOY_LINES PSDs. Quadrupole controls are identical between those two toys. GW190814-like: T=16 s, dt=1/4096 s, 20–1024 Hz, T_b=(16,8,4) s, Mc=Mc_ref=6.4, tc=14 s, default ratios; real L1 PSD loaded with load_asd_or_psd from data/psd_L1_GW190814_welch16s.txt with skip_dc=True (the file includes f=0, unused by Eq. 4), a smooth analytic control, and a matched L1 median16Hz control (running median of log PSD, 257 bins on the original 1/16-Hz grid; this removes narrow features, not just individual known lines). All geometries use m_max=4, L=5.

## toy_small

| PSD | modes | method | relative (h,h) error | abs(Delta ln Lambda), SNR 25 |
| --- | --- | --- | --- | --- |
| lines | 22 | linear | 1.403926e-06 | 4.387267e-04 |
| lines | 22 | IFFT-FFT fast | 1.543451e-07 | 4.823285e-05 |
| lines | 22 | pairwise | 2.214771e-07 | 6.921160e-05 |
| lines | 22 | B0 | 2.214771e-07 | 6.921160e-05 |
| lines | 22 | B1 adaptive | 2.214771e-07 | 6.921160e-05 |
| lines | 22 | B1 forced | 4.434962e-07 | 1.385926e-04 |
| lines | five | linear | 1.573092e-06 | 4.915912e-04 |
| lines | five | IFFT-FFT fast | 2.218370e-07 | 6.932407e-05 |
| lines | five | pairwise | 3.530529e-07 | 1.103290e-04 |
| lines | five | B0 | 4.395005e-06 | 1.373439e-03 |
| lines | five | B1 adaptive | 4.398792e-07 | 1.374622e-04 |
| lines | five | B1 forced | 4.398792e-07 | 1.374622e-04 |
| smooth | 22 | linear | 1.404990e-06 | 4.390592e-04 |
| smooth | 22 | IFFT-FFT fast | 2.248638e-07 | 7.026994e-05 |
| smooth | 22 | pairwise | 2.207669e-07 | 6.898965e-05 |
| smooth | 22 | B0 | 2.207669e-07 | 6.898965e-05 |
| smooth | 22 | B1 adaptive | 2.207669e-07 | 6.898965e-05 |
| smooth | 22 | B1 forced | 4.271317e-07 | 1.334787e-04 |
| smooth | five | linear | 4.787959e-06 | 1.496237e-03 |
| smooth | five | IFFT-FFT fast | 2.938171e-07 | 9.181784e-05 |
| smooth | five | pairwise | 3.536371e-07 | 1.105116e-04 |
| smooth | five | B0 | 1.210697e-07 | 3.783429e-05 |
| smooth | five | B1 adaptive | 5.337178e-07 | 1.667868e-04 |
| smooth | five | B1 forced | 5.337178e-07 | 1.667868e-04 |

## strong-HOM toy

| PSD | modes | method | relative (h,h) error | abs(Delta ln Lambda), SNR 25 |
| --- | --- | --- | --- | --- |
| lines | 22 | linear | 1.403926e-06 | 4.387267e-04 |
| lines | 22 | IFFT-FFT fast | 1.543451e-07 | 4.823285e-05 |
| lines | 22 | pairwise | 2.214771e-07 | 6.921160e-05 |
| lines | 22 | B0 | 2.214771e-07 | 6.921160e-05 |
| lines | 22 | B1 adaptive | 2.214771e-07 | 6.921160e-05 |
| lines | 22 | B1 forced | 4.434962e-07 | 1.385926e-04 |
| lines | five | linear | 3.565085e-05 | 1.114089e-02 |
| lines | five | IFFT-FFT fast | 1.440090e-07 | 4.500282e-05 |
| lines | five | pairwise | 3.051015e-07 | 9.534423e-05 |
| lines | five | B0 | 8.938247e-06 | 2.793202e-03 |
| lines | five | B1 adaptive | 2.731711e-07 | 8.536597e-05 |
| lines | five | B1 forced | 2.731711e-07 | 8.536597e-05 |
| smooth | 22 | linear | 1.404990e-06 | 4.390592e-04 |
| smooth | 22 | IFFT-FFT fast | 2.248638e-07 | 7.026994e-05 |
| smooth | 22 | pairwise | 2.207669e-07 | 6.898965e-05 |
| smooth | 22 | B0 | 2.207669e-07 | 6.898965e-05 |
| smooth | 22 | B1 adaptive | 2.207669e-07 | 6.898965e-05 |
| smooth | 22 | B1 forced | 4.271317e-07 | 1.334787e-04 |
| smooth | five | linear | 5.032164e-06 | 1.572551e-03 |
| smooth | five | IFFT-FFT fast | 1.770895e-07 | 5.534046e-05 |
| smooth | five | pairwise | 3.078875e-07 | 9.621484e-05 |
| smooth | five | B0 | 8.526138e-07 | 2.664418e-04 |
| smooth | five | B1 adaptive | 3.725006e-07 | 1.164064e-04 |
| smooth | five | B1 forced | 3.725006e-07 | 1.164064e-04 |

## GW190814-like

| PSD | modes | method | relative (h,h) error | abs(Delta ln Lambda), SNR 25 |
| --- | --- | --- | --- | --- |
| real L1 | 22 | linear | 1.331900e-06 | 4.162187e-04 |
| real L1 | 22 | IFFT-FFT fast | 3.139544e-07 | 9.811076e-05 |
| real L1 | 22 | pairwise | 2.979860e-07 | 9.312064e-05 |
| real L1 | 22 | B0 | 2.979860e-07 | 9.312064e-05 |
| real L1 | 22 | B1 adaptive | 2.979860e-07 | 9.312064e-05 |
| real L1 | 22 | B1 forced | 6.042534e-07 | 1.888292e-04 |
| real L1 | five | linear | 2.951115e-05 | 9.222233e-03 |
| real L1 | five | IFFT-FFT fast | 5.993453e-07 | 1.872954e-04 |
| real L1 | five | pairwise | 5.796102e-07 | 1.811282e-04 |
| real L1 | five | B0 | 4.765723e-06 | 1.489288e-03 |
| real L1 | five | B1 adaptive | 6.729976e-07 | 2.103117e-04 |
| real L1 | five | B1 forced | 6.729976e-07 | 2.103117e-04 |
| smooth control | 22 | linear | 1.478275e-06 | 4.619609e-04 |
| smooth control | 22 | IFFT-FFT fast | 1.945957e-07 | 6.081115e-05 |
| smooth control | 22 | pairwise | 1.945249e-07 | 6.078903e-05 |
| smooth control | 22 | B0 | 1.945249e-07 | 6.078903e-05 |
| smooth control | 22 | B1 adaptive | 1.945249e-07 | 6.078903e-05 |
| smooth control | 22 | B1 forced | 2.429668e-07 | 7.592713e-05 |
| smooth control | five | linear | 3.835914e-06 | 1.198723e-03 |
| smooth control | five | IFFT-FFT fast | 4.124341e-07 | 1.288856e-04 |
| smooth control | five | pairwise | 4.120280e-07 | 1.287588e-04 |
| smooth control | five | B0 | 5.418589e-07 | 1.693309e-04 |
| smooth control | five | B1 adaptive | 4.628244e-07 | 1.446326e-04 |
| smooth control | five | B1 forced | 4.628244e-07 | 1.446326e-04 |
| L1 median16Hz | 22 | linear | 1.351138e-06 | 4.222305e-04 |
| L1 median16Hz | 22 | IFFT-FFT fast | 2.539085e-07 | 7.934641e-05 |
| L1 median16Hz | 22 | pairwise | 2.545077e-07 | 7.953365e-05 |
| L1 median16Hz | 22 | B0 | 2.545077e-07 | 7.953365e-05 |
| L1 median16Hz | 22 | B1 adaptive | 2.545077e-07 | 7.953365e-05 |
| L1 median16Hz | 22 | B1 forced | 3.122004e-07 | 9.756264e-05 |
| L1 median16Hz | five | linear | 6.450645e-07 | 2.015826e-04 |
| L1 median16Hz | five | IFFT-FFT fast | 5.204162e-07 | 1.626301e-04 |
| L1 median16Hz | five | pairwise | 5.179109e-07 | 1.618472e-04 |
| L1 median16Hz | five | B0 | 6.897365e-07 | 2.155427e-04 |
| L1 median16Hz | five | B1 adaptive | 5.753230e-07 | 1.797884e-04 |
| L1 median16Hz | five | B1 forced | 5.753230e-07 | 1.797884e-04 |

## Real-PSD narrow-feature comparison

For five modes, B0 relative error is 4.765723e-06 on real L1 versus 6.897365e-07 on its matched running-median control (6.91x). B1 adaptive gives 6.729976e-07 versus 5.753230e-07. Narrow PSD features therefore matter much more for B0 in this case. The per-band folding audit below explains the resulting signed error; smoothing changes other local structure too, so this is not a single-line isolation test.

## Summary over fourteen fixed cases

| method | median relative error | worst relative error | median abs(Delta ln Lambda) | worst abs(Delta ln Lambda) |
| --- | --- | --- | --- | --- |
| linear | 1.441632e-06 | 3.565085e-05 | 4.505101e-04 | 1.114089e-02 |
| IFFT-FFT fast | 2.248638e-07 | 5.993453e-07 | 7.026994e-05 | 1.872954e-04 |
| pairwise | 3.015438e-07 | 5.796102e-07 | 9.423243e-05 | 1.811282e-04 |
| B0 | 2.762469e-07 | 8.938247e-06 | 8.632714e-05 | 2.793202e-03 |
| B1 adaptive | 2.855786e-07 | 6.729976e-07 | 8.924330e-05 | 2.103117e-04 |
| B1 forced | 4.416877e-07 | 6.729976e-07 | 1.380274e-04 | 2.103117e-04 |

## Per-band error and fine-grid folding audit

Signed band errors below are divided by the TOTAL fine-grid norm, so columns sum to the signed global relative error. Multiply their magnitudes by 312.5 for SNR-25 log-likelihood contributions. The folding column independently evaluates (4/TN) sum_out G_m (conj A_alias - conj A_m). outside_fraction uses |G|, not |G|²; it alone does not determine accuracy. B0 cuts the predicted extreme-pair bulk where lag_max>T_b/2, especially toy bands 1–4. Band 0 retains all circular lags and is exact even if its unwrapped lag exceeds the limit.

| case | b | B0 signed relative contribution | B1 adaptive signed relative contribution | B0 folding contribution | outside fraction | lag_max (s) | T_b/2 (s) | margin (s) | B1 mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toy_small, lines, 22 | 0 | +3.287859e-18 | +3.287859e-18 | +0.000000e+00 | 0.0000e+00 | 0.000000 | 64.000000 | 7.256689 | exact |
| toy_small, lines, 22 | 1 | +2.987008e-11 | +2.987008e-11 | +2.987009e-11 | 5.0311e-05 | 0.000000 | 32.000000 | 4.669730 | B0 |
| toy_small, lines, 22 | 2 | +9.315460e-09 | +9.315460e-09 | +9.315460e-09 | 1.0073e-04 | 0.000000 | 16.000000 | 2.760240 | B0 |
| toy_small, lines, 22 | 3 | -4.382370e-08 | -4.382370e-08 | -4.382370e-08 | 1.4295e-04 | 0.000000 | 8.000000 | 1.577886 | B0 |
| toy_small, lines, 22 | 4 | -5.881013e-08 | -5.881013e-08 | -5.881013e-08 | 1.6010e-04 | 0.000000 | 4.000000 | 0.837163 | B0 |
| toy_small, lines, 22 | 5 | -1.281886e-07 | -1.281886e-07 | -1.281886e-07 | 1.1817e-04 | 0.000000 | 2.000000 | 0.356455 | B0 |
| toy_small, lines, five | 0 | +2.968989e-18 | +2.968989e-18 | +0.000000e+00 | 0.0000e+00 | 97.822154 | 64.000000 | 18.285710 | exact |
| toy_small, lines, five | 1 | +1.046177e-06 | -1.659108e-08 | +1.046177e-06 | 1.4562e-01 | 51.519213 | 32.000000 | 11.766982 | B1 |
| toy_small, lines, five | 2 | +7.888001e-06 | -1.681066e-08 | +7.888001e-06 | 1.1254e-01 | 23.979002 | 16.000000 | 6.955368 | B1 |
| toy_small, lines, five | 3 | -4.468698e-06 | -1.251901e-07 | -4.468698e-06 | 5.9220e-02 | 10.630754 | 8.000000 | 3.976024 | B1 |
| toy_small, lines, five | 4 | +6.279616e-08 | -1.399917e-07 | +6.279616e-08 | 3.2438e-03 | 4.228414 | 4.000000 | 2.109518 | B1 |
| toy_small, lines, five | 5 | -1.332716e-07 | -1.412956e-07 | -1.332716e-07 | 4.1711e-05 | 1.221304 | 2.000000 | 0.898209 | B1 |
| toy_small, smooth, 22 | 0 | +1.638411e-18 | +1.638411e-18 | +0.000000e+00 | 0.0000e+00 | 0.000000 | 64.000000 | 7.256689 | exact |
| toy_small, smooth, 22 | 1 | +2.994771e-11 | +2.994771e-11 | +2.994771e-11 | 5.0311e-05 | 0.000000 | 32.000000 | 4.669730 | B0 |
| toy_small, smooth, 22 | 2 | +9.277339e-09 | +9.277339e-09 | +9.277339e-09 | 1.0073e-04 | 0.000000 | 16.000000 | 2.760240 | B0 |
| toy_small, smooth, 22 | 3 | -4.370900e-08 | -4.370900e-08 | -4.370900e-08 | 1.4295e-04 | 0.000000 | 8.000000 | 1.577886 | B0 |
| toy_small, smooth, 22 | 4 | -5.857739e-08 | -5.857739e-08 | -5.857739e-08 | 1.6010e-04 | 0.000000 | 4.000000 | 0.837163 | B0 |
| toy_small, smooth, 22 | 5 | -1.277878e-07 | -1.277878e-07 | -1.277878e-07 | 1.1817e-04 | 0.000000 | 2.000000 | 0.356455 | B0 |
| toy_small, smooth, five | 0 | +1.479548e-18 | +1.479548e-18 | +0.000000e+00 | 0.0000e+00 | 97.822154 | 64.000000 | 18.285710 | exact |
| toy_small, smooth, five | 1 | -2.110781e-07 | -1.497555e-08 | -2.110781e-07 | 1.4562e-01 | 51.519213 | 32.000000 | 11.766982 | B1 |
| toy_small, smooth, five | 2 | +2.132567e-07 | -7.537518e-08 | +2.132567e-07 | 1.1254e-01 | 23.979002 | 16.000000 | 6.955368 | B1 |
| toy_small, smooth, five | 3 | +1.764613e-07 | -1.408258e-07 | +1.764613e-07 | 5.9220e-02 | 10.630754 | 8.000000 | 3.976024 | B1 |
| toy_small, smooth, five | 4 | +7.527910e-08 | -1.395264e-07 | +7.527910e-08 | 3.2438e-03 | 4.228414 | 4.000000 | 2.109518 | B1 |
| toy_small, smooth, five | 5 | -1.328492e-07 | -1.630148e-07 | -1.328492e-07 | 4.1711e-05 | 1.221304 | 2.000000 | 0.898209 | B1 |
| strong-HOM toy, lines, 22 | 0 | +3.287859e-18 | +3.287859e-18 | +0.000000e+00 | 0.0000e+00 | 0.000000 | 64.000000 | 7.256689 | exact |
| strong-HOM toy, lines, 22 | 1 | +2.987008e-11 | +2.987008e-11 | +2.987009e-11 | 5.0311e-05 | 0.000000 | 32.000000 | 4.669730 | B0 |
| strong-HOM toy, lines, 22 | 2 | +9.315460e-09 | +9.315460e-09 | +9.315460e-09 | 1.0073e-04 | 0.000000 | 16.000000 | 2.760240 | B0 |
| strong-HOM toy, lines, 22 | 3 | -4.382370e-08 | -4.382370e-08 | -4.382370e-08 | 1.4295e-04 | 0.000000 | 8.000000 | 1.577886 | B0 |
| strong-HOM toy, lines, 22 | 4 | -5.881013e-08 | -5.881013e-08 | -5.881013e-08 | 1.6010e-04 | 0.000000 | 4.000000 | 0.837163 | B0 |
| strong-HOM toy, lines, 22 | 5 | -1.281886e-07 | -1.281886e-07 | -1.281886e-07 | 1.1817e-04 | 0.000000 | 2.000000 | 0.356455 | B0 |
| strong-HOM toy, lines, five | 0 | +2.183283e-18 | +2.183283e-18 | +0.000000e+00 | 0.0000e+00 | 97.822154 | 64.000000 | 18.285710 | exact |
| strong-HOM toy, lines, five | 1 | +2.428269e-06 | -5.892503e-09 | +2.428269e-06 | 2.0902e-01 | 51.519213 | 32.000000 | 11.766982 | B1 |
| strong-HOM toy, lines, five | 2 | +1.743613e-05 | +2.720416e-08 | +1.743613e-05 | 1.6401e-01 | 23.979002 | 16.000000 | 6.955368 | B1 |
| strong-HOM toy, lines, five | 3 | -1.118432e-05 | -1.109214e-07 | -1.118432e-05 | 8.8813e-02 | 10.630754 | 8.000000 | 3.976024 | B1 |
| strong-HOM toy, lines, five | 4 | +3.214429e-07 | -1.124392e-07 | +3.214429e-07 | 5.5028e-03 | 4.228414 | 4.000000 | 2.109518 | B1 |
| strong-HOM toy, lines, five | 5 | -6.327514e-08 | -7.112219e-08 | -6.327514e-08 | 2.1037e-05 | 1.221304 | 2.000000 | 0.898209 | B1 |
| strong-HOM toy, smooth, 22 | 0 | +1.638411e-18 | +1.638411e-18 | +0.000000e+00 | 0.0000e+00 | 0.000000 | 64.000000 | 7.256689 | exact |
| strong-HOM toy, smooth, 22 | 1 | +2.994771e-11 | +2.994771e-11 | +2.994771e-11 | 5.0311e-05 | 0.000000 | 32.000000 | 4.669730 | B0 |
| strong-HOM toy, smooth, 22 | 2 | +9.277339e-09 | +9.277339e-09 | +9.277339e-09 | 1.0073e-04 | 0.000000 | 16.000000 | 2.760240 | B0 |
| strong-HOM toy, smooth, 22 | 3 | -4.370900e-08 | -4.370900e-08 | -4.370900e-08 | 1.4295e-04 | 0.000000 | 8.000000 | 1.577886 | B0 |
| strong-HOM toy, smooth, 22 | 4 | -5.857739e-08 | -5.857739e-08 | -5.857739e-08 | 1.6010e-04 | 0.000000 | 4.000000 | 0.837163 | B0 |
| strong-HOM toy, smooth, 22 | 5 | -1.277878e-07 | -1.277878e-07 | -1.277878e-07 | 1.1817e-04 | 0.000000 | 2.000000 | 0.356455 | B0 |
| strong-HOM toy, smooth, five | 0 | +2.175662e-18 | +2.175662e-18 | +0.000000e+00 | 0.0000e+00 | 97.822154 | 64.000000 | 18.285710 | exact |
| strong-HOM toy, smooth, five | 1 | -4.283911e-07 | -6.313513e-09 | -4.283911e-07 | 2.0902e-01 | 51.519213 | 32.000000 | 11.766982 | B1 |
| strong-HOM toy, smooth, five | 2 | +4.694818e-07 | -4.415513e-08 | +4.694818e-07 | 1.6401e-01 | 23.979002 | 16.000000 | 6.955368 | B1 |
| strong-HOM toy, smooth, five | 3 | +5.251082e-07 | -1.182921e-07 | +5.251082e-07 | 8.8813e-02 | 10.630754 | 8.000000 | 3.976024 | B1 |
| strong-HOM toy, smooth, five | 4 | +3.494642e-07 | -1.120477e-07 | +3.494642e-07 | 5.5028e-03 | 4.228414 | 4.000000 | 2.109518 | B1 |
| strong-HOM toy, smooth, five | 5 | -6.304924e-08 | -9.169218e-08 | -6.304924e-08 | 2.1037e-05 | 1.221304 | 2.000000 | 0.898209 | B1 |
| GW190814-like, real L1, 22 | 0 | +3.457170e-17 | +3.457170e-17 | +0.000000e+00 | 0.0000e+00 | 0.000000 | 8.000000 | 5.759348 | exact |
| GW190814-like, real L1, 22 | 1 | -2.611207e-07 | -2.611207e-07 | -2.611207e-07 | 1.6420e-04 | 0.000000 | 4.000000 | 0.898707 | B0 |
| GW190814-like, real L1, 22 | 2 | -3.686530e-08 | -3.686530e-08 | -3.686530e-08 | 1.3393e-04 | 0.000000 | 2.000000 | 0.381445 | B0 |
| GW190814-like, real L1, five | 0 | +1.241364e-16 | +1.241364e-16 | +0.000000e+00 | 0.0000e+00 | 61.617891 | 8.000000 | 14.512647 | exact |
| GW190814-like, real L1, five | 1 | -4.625543e-06 | -6.702740e-07 | -4.625543e-06 | 2.2278e-03 | 4.132803 | 4.000000 | 2.264599 | B1 |
| GW190814-like, real L1, five | 2 | -1.401796e-07 | -2.723540e-09 | -1.401796e-07 | 6.1106e-05 | 1.188182 | 2.000000 | 0.961181 | B1 |
| GW190814-like, smooth control, 22 | 0 | +2.385648e-17 | +2.385648e-17 | +0.000000e+00 | 0.0000e+00 | 0.000000 | 8.000000 | 5.759348 | exact |
| GW190814-like, smooth control, 22 | 1 | -1.820588e-07 | -1.820588e-07 | -1.820588e-07 | 1.6420e-04 | 0.000000 | 4.000000 | 0.898707 | B0 |
| GW190814-like, smooth control, 22 | 2 | -1.246606e-08 | -1.246606e-08 | -1.246606e-08 | 1.3393e-04 | 0.000000 | 2.000000 | 0.381445 | B0 |
| GW190814-like, smooth control, five | 0 | +0.000000e+00 | +0.000000e+00 | +0.000000e+00 | 0.0000e+00 | 61.617891 | 8.000000 | 14.512647 | exact |
| GW190814-like, smooth control, five | 1 | -4.378875e-07 | -3.210022e-07 | -4.378875e-07 | 2.2278e-03 | 4.132803 | 4.000000 | 2.264599 | B1 |
| GW190814-like, smooth control, five | 2 | -1.039714e-07 | -1.418222e-07 | -1.039714e-07 | 6.1106e-05 | 1.188182 | 2.000000 | 0.961181 | B1 |
| GW190814-like, L1 median16Hz, 22 | 0 | +3.469536e-17 | +3.469536e-17 | +0.000000e+00 | 0.0000e+00 | 0.000000 | 8.000000 | 5.759348 | exact |
| GW190814-like, L1 median16Hz, 22 | 1 | -2.386170e-07 | -2.386170e-07 | -2.386170e-07 | 1.6420e-04 | 0.000000 | 4.000000 | 0.898707 | B0 |
| GW190814-like, L1 median16Hz, 22 | 2 | -1.589065e-08 | -1.589065e-08 | -1.589065e-08 | 1.3393e-04 | 0.000000 | 2.000000 | 0.381445 | B0 |
| GW190814-like, L1 median16Hz, five | 0 | +0.000000e+00 | +0.000000e+00 | +0.000000e+00 | 0.0000e+00 | 61.617891 | 8.000000 | 14.512647 | exact |
| GW190814-like, L1 median16Hz, five | 1 | -5.804715e-07 | -4.261385e-07 | -5.804715e-07 | 2.2278e-03 | 4.132803 | 4.000000 | 2.264599 | B1 |
| GW190814-like, L1 median16Hz, five | 2 | -1.092650e-07 | -1.491845e-07 | -1.092650e-07 | 6.1106e-05 | 1.188182 | 2.000000 | 0.961181 | B1 |

## Convention checks and unmet requested thresholds

The specified algorithms are retained. These checks expose differences in the requested identities rather than changing iteration-001–003 behavior or silently tuning B1. For the one-sided complex A_m, the exact paper-weight identity is Re[W(k)+W(-k)] = (T/That) I_tilde(k), NOT W(k)=(That/T) I_tilde(k). It follows by symmetrizing A_m before the DFT in Eq. (32). The missing negative-frequency weight and the inverted normalization matter. The table checks the correct identity on ALL positive bins and the requested identity on the band bins.

| b | W vs requested (That/T) I, max-scaled | correct even-kernel identity | half points vs paper, interior | half points vs paper, overall | K'/M |
| --- | --- | --- | --- | --- | --- |
| 0 | 5.878901e-16 | 3.578781e-16 | nan | nan | 0.092285 |
| 1 | 2.696098e-15 | 5.836329e-16 | 3.010572e-04 | 3.785555e-03 | 0.072510 |
| 2 | 7.499516e-01 | 3.892167e-16 | 3.696552e-04 | 4.033852e-03 | 0.105469 |
| 3 | 9.374932e-01 | 6.212719e-16 | 1.037285e-04 | 9.267376e-04 | 0.085449 |
| 4 | 9.843688e-01 | 3.806346e-16 | 5.586070e-05 | 1.202548e-03 | 0.175781 |
| 5 | 9.960938e-01 | 3.919787e-16 | 1.589379e-05 | 6.783532e-04 | 0.201416 |

The minimal K' periodic baseband interpolant and the paper's real-signal full-spectrum interpolant have different tails and image contributions. Eq. (36) is approximate for these finite-window signals; it does not make the two discrete interpolants identical. Public half-point samples include a global minus sign relative to raw fft(ifft(x)*phase), from the negative-time origin n-K'; this aligns with the paper and direct ToyModes phases. The norm uses |z|² and is unaffected. Restricted-band quadrature also discards the paper interpolant's out-of-band power; exact weight identities must use its entire positive grid.

Pure (2,2), band 3: max pointwise half-point error against direct evaluation is 4.679710e-04 on the whole plateau and 6.251367e-05 in its central 60%. Only the latter meets 1e-4.

| requested check | achieved | result |
| --- | --- | --- |
| forced B1 vs paper <1e-8 relative | 22: 2.891511e-07; five: 2.180422e-07 | unmet |
| complex W=(That/T) I within 1e-10 | up to 0.996 max-scaled discrepancy; correct symmetrized identity <7e-16 | incorrect identity |
| adaptive vs forced <1e-7, actual m values | 22: 2.220191e-07; five: 0.000000e+00 | unmet for 22 |
| default five-mode adaptive uses upper-band B0 | no B0 with prescribed 5 sigma margin | unmet |
| B0 five-mode relative error <1e-5 and worse than B1 (lines) | 4.395005e-6 vs B1 4.398792e-7 | met |
| pure 22 half points, whole plateau band 3 <1e-4 | 4.679710e-04; central 60% 6.251367e-05 | met only in interior |
| complex half-point agreement with paper, interior 1e-10 / overall ~1e-8 | interior 1.59e-5 to 3.70e-4 / overall 6.78e-4 to 4.03e-3, five modes | unmet |
| K' <= M/2, all toy b>=1 | K'/M=0.07251,0.10547,0.08545,0.17578,0.20142 | met |

Default five-mode adaptive modes: ['exact', 'B1', 'B1', 'B1', 'B1', 'B1']. In the top band, lag_max+5 sigma=2.119513 s > T_b/2=2.000000 s: the requested rule does NOT select B0 there. With L_margin=4 it does; that optional setting is tested separately. Quadrupole-only evaluators use m_values=(2,) and do select B0 in upper bands. Finally, M=N needs no interpolation; 2M=N still needs half points (toy band 1).
