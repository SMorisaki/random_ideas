# Iteration 007: XHM population accuracy

Seed 20261017, independent SeedSequence streams by population/subset, prefix-stable. Regular priors are component-mass boxes conditioned on Mc >= Mc_ref and m1 >= m2; BBH also q >= 0.1. Strong-HOM draws condition the same mass/spin prior on q <= 0.15 and Mc <= Mc_max, and the isotropic inclination prior on 1.3 < iota < 1.8 (20 additional NSBH/BBH points). All angles, spins and t_c vary. Distance = 100 Mpc; antenna norm = 1.

Exact reference: XHMModel.total_direct, full positive-frequency sequence, Eq. (4), DC/Nyquist excluded. Scalar exact norms cached per population/index/PSD independently of L; only one fine waveform is held at a time. Coarse modes retain iteration-005 Mf>0.3 tail recovery. All six methods use unchanged defaults; pairwise uses specified prior corners; B1_all is p=1, all_half_bins=True with adaptive shortcuts.

Numbers below are |Delta ln Lambda| = 312.5 |Delta HH|/HH at SNR25, with the linear term held fixed; this is the quadratic-term Fig. 4 diagnostic. At SNR50 multiply by 4; at SNR100 by 16. Divide by 312.5 for relative HH error. Percentiles use NumPy linear interpolation. Separate regular, strong-HOM, and combined tables prevent the 20 stress points from changing the regular population quantiles. Where regular draws exceed the upper pairwise corner, regular_inside_corners provides a sensitivity check without changing the primary draws.

L1 file: data/psd_L1_GW190814_welch16s.txt, skip_dc=True, log PSD/log frequency interpolation by PSDGrid. Its 1/16 Hz resolution is interpolated to 1/256 Hz for BNS; unresolved measured lines are not recovered. Synthetic PSD: S_analytic * [1 + sum h/(1+((f-f0)/gamma)^2)], gamma is HWHM. The same line list is used for both L values and every population. One line lies in each BNS L=5 band and one within Df of each internal BNS edge at each L.

| f0 (Hz) | HWHM (Hz) | height / baseline |
|---:|---:|---:|
| 35.595704506 | 0.01 | 100 |
| 59.311504474 | 0.03 | 1000 |
| 78.537866121 | 0.1 | 10000 |
| 105.501736834 | 0.5 | 100 |
| 146.260676511 | 0.01 | 1000 |
| 221.943782764 | 0.03 | 10000 |
| 1160.362776933 | 0.1 | 100 |
| 50.985969938 | 0.5 | 1000 |
| 67.091135764 | 0.01 | 10000 |
| 89.070308597 | 0.03 | 100 |
| 120.359442781 | 0.1 | 1000 |
| 169.283840078 | 0.5 | 10000 |
| 268.313381968 | 0.01 | 100 |
| 67.594940606 | 0.03 | 1000 |
| 93.758333247 | 0.1 | 10000 |
| 132.177356314 | 0.5 | 100 |
| 191.348609172 | 0.01 | 1000 |
| 292.148029638 | 0.03 | 10000 |
| 517.754294021 | 0.1 | 100 |

## Setup checks

| Population / PSD | n + strong | rejections regular / strong (Mc low, Mc high, q) | above Mc upper corner | wall s |
|---|---:|---|---|---:|
| bns / lines | 100 + 0 | {'chirp_mass': 0, 'chirp_mass_upper': 0, 'mass_ratio': 0} / {} | [] | 10.4 |
| bns / L1 | 100 + 0 | {'chirp_mass': 0, 'chirp_mass_upper': 0, 'mass_ratio': 0} / {} | [] | 9.7 |
| nsbh / lines | 100 + 20 | {'chirp_mass': 1, 'chirp_mass_upper': 0, 'mass_ratio': 0} / {'chirp_mass': 0, 'chirp_mass_upper': 0, 'mass_ratio': 8} | [51] | 4.6 |
| nsbh / L1 | 100 + 20 | {'chirp_mass': 1, 'chirp_mass_upper': 0, 'mass_ratio': 0} / {'chirp_mass': 0, 'chirp_mass_upper': 0, 'mass_ratio': 8} | [51] | 4.8 |
| bbh / lines | 100 + 20 | {'chirp_mass': 0, 'chirp_mass_upper': 0, 'mass_ratio': 3} / {'chirp_mass': 0, 'chirp_mass_upper': 0, 'mass_ratio': 2934} | [] | 2.4 |
| bbh / L1 | 100 + 20 | {'chirp_mass': 0, 'chirp_mass_upper': 0, 'mass_ratio': 3} / {'chirp_mass': 0, 'chirp_mass_upper': 0, 'mass_ratio': 2934} | [] | 2.4 |

The NSBH mass box reaches Mc=9.32103 Msun whereas the requested pairwise upper anchor is 9.0. Draws are retained and flagged above; these test the specified corners but are outside their upper coverage assumption. No Eq. (51) root failed and no reference mass or duration was adjusted. Negative requested slack occurs only in the full-grid band (exact for any circular block); no truncated-band slack was clipped. Full flags are in each JSON geometry; flagged point parameters and chirp masses are in the per-point JSON records.

| Population | L | K_MB | min slack all / truncated (s) | adaptive B1 band modes |
|---|---:|---:|---:|---|
| bns | 5 | 21653 | -1136.42 / 2.00141 | ['exact', 'B1', 'B1', 'B1', 'B1', 'B1', 'B1'] |
| bns | 50 | 30422 | -1155.5 / 3.56082 | ['exact', 'B0', 'B0', 'B0', 'B0', 'B0', 'B0'] |
| nsbh | 5 | 5128 | -140.004 / 1.61402 | ['exact', 'B1', 'B1'] |
| nsbh | 50 | 6490 | -140.176 / 3.61636 | ['exact', 'B0', 'B0'] |
| bbh | 5 | 4610 | -14.5091 / 1.2685 | ['exact', 'B1', 'B1'] |
| bbh | 50 | 5549 | -14.2283 / 3.6212 | ['exact', 'B0', 'B0'] |

## Error distributions

| Population | PSD | subset | L | method | median | p90 | p99 | max | median relative HH | max relative HH |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| bns | lines | regular | 5 | linear | 1.689e-04 | 5.167e-04 | 8.733e-04 | 9.194e-04 | 5.405e-07 | 2.942e-06 |
| bns | lines | regular | 5 | IFFT-FFT | 1.889e-05 | 3.243e-05 | 3.872e-05 | 4.017e-05 | 6.044e-08 | 1.286e-07 |
| bns | lines | regular | 5 | pairwise | 1.552e-05 | 1.604e-05 | 1.673e-05 | 1.686e-05 | 4.967e-08 | 5.396e-08 |
| bns | lines | regular | 5 | B1 | 2.723e-05 | 5.100e-05 | 6.295e-05 | 7.686e-05 | 8.714e-08 | 2.460e-07 |
| bns | lines | regular | 5 | B1_all | 6.644e-05 | 1.008e-04 | 1.392e-04 | 1.463e-04 | 2.126e-07 | 4.681e-07 |
| bns | lines | regular | 5 | B0 | 1.552e-05 | 1.661e-05 | 1.945e-05 | 1.999e-05 | 4.965e-08 | 6.398e-08 |
| bns | lines | regular | 50 | linear | 2.356e-05 | 7.489e-05 | 1.205e-04 | 1.245e-04 | 7.539e-08 | 3.983e-07 |
| bns | lines | regular | 50 | IFFT-FFT | 6.915e-06 | 1.346e-05 | 1.812e-05 | 1.834e-05 | 2.213e-08 | 5.869e-08 |
| bns | lines | regular | 50 | pairwise | 5.130e-07 | 5.450e-07 | 5.843e-07 | 6.111e-07 | 1.642e-09 | 1.955e-09 |
| bns | lines | regular | 50 | B1 | 5.131e-07 | 5.451e-07 | 5.844e-07 | 6.114e-07 | 1.642e-09 | 1.956e-09 |
| bns | lines | regular | 50 | B1_all | 5.131e-07 | 5.451e-07 | 5.844e-07 | 6.114e-07 | 1.642e-09 | 1.956e-09 |
| bns | lines | regular | 50 | B0 | 5.131e-07 | 5.451e-07 | 5.844e-07 | 6.114e-07 | 1.642e-09 | 1.956e-09 |
| bns | lines | combined | 5 | linear | 1.689e-04 | 5.167e-04 | 8.733e-04 | 9.194e-04 | 5.405e-07 | 2.942e-06 |
| bns | lines | combined | 5 | IFFT-FFT | 1.889e-05 | 3.243e-05 | 3.872e-05 | 4.017e-05 | 6.044e-08 | 1.286e-07 |
| bns | lines | combined | 5 | pairwise | 1.552e-05 | 1.604e-05 | 1.673e-05 | 1.686e-05 | 4.967e-08 | 5.396e-08 |
| bns | lines | combined | 5 | B1 | 2.723e-05 | 5.100e-05 | 6.295e-05 | 7.686e-05 | 8.714e-08 | 2.460e-07 |
| bns | lines | combined | 5 | B1_all | 6.644e-05 | 1.008e-04 | 1.392e-04 | 1.463e-04 | 2.126e-07 | 4.681e-07 |
| bns | lines | combined | 5 | B0 | 1.552e-05 | 1.661e-05 | 1.945e-05 | 1.999e-05 | 4.965e-08 | 6.398e-08 |
| bns | lines | combined | 50 | linear | 2.356e-05 | 7.489e-05 | 1.205e-04 | 1.245e-04 | 7.539e-08 | 3.983e-07 |
| bns | lines | combined | 50 | IFFT-FFT | 6.915e-06 | 1.346e-05 | 1.812e-05 | 1.834e-05 | 2.213e-08 | 5.869e-08 |
| bns | lines | combined | 50 | pairwise | 5.130e-07 | 5.450e-07 | 5.843e-07 | 6.111e-07 | 1.642e-09 | 1.955e-09 |
| bns | lines | combined | 50 | B1 | 5.131e-07 | 5.451e-07 | 5.844e-07 | 6.114e-07 | 1.642e-09 | 1.956e-09 |
| bns | lines | combined | 50 | B1_all | 5.131e-07 | 5.451e-07 | 5.844e-07 | 6.114e-07 | 1.642e-09 | 1.956e-09 |
| bns | lines | combined | 50 | B0 | 5.131e-07 | 5.451e-07 | 5.844e-07 | 6.114e-07 | 1.642e-09 | 1.956e-09 |
| bns | L1 | regular | 5 | linear | 9.144e-05 | 3.700e-04 | 4.937e-04 | 5.300e-04 | 2.926e-07 | 1.696e-06 |
| bns | L1 | regular | 5 | IFFT-FFT | 1.604e-05 | 3.710e-05 | 4.611e-05 | 4.800e-05 | 5.134e-08 | 1.536e-07 |
| bns | L1 | regular | 5 | pairwise | 1.263e-06 | 1.678e-06 | 2.064e-06 | 2.741e-06 | 4.042e-09 | 8.771e-09 |
| bns | L1 | regular | 5 | B1 | 4.179e-05 | 8.168e-05 | 1.041e-04 | 1.098e-04 | 1.337e-07 | 3.513e-07 |
| bns | L1 | regular | 5 | B1_all | 2.791e-05 | 5.840e-05 | 1.135e-04 | 1.165e-04 | 8.932e-08 | 3.728e-07 |
| bns | L1 | regular | 5 | B0 | 1.212e-06 | 1.920e-06 | 2.848e-05 | 2.850e-05 | 3.877e-09 | 9.122e-08 |
| bns | L1 | regular | 50 | linear | 1.439e-05 | 7.017e-05 | 1.354e-04 | 1.507e-04 | 4.605e-08 | 4.822e-07 |
| bns | L1 | regular | 50 | IFFT-FFT | 1.968e-05 | 4.950e-05 | 6.372e-05 | 6.842e-05 | 6.298e-08 | 2.189e-07 |
| bns | L1 | regular | 50 | pairwise | 7.430e-07 | 8.097e-07 | 8.284e-07 | 8.430e-07 | 2.378e-09 | 2.698e-09 |
| bns | L1 | regular | 50 | B1 | 7.443e-07 | 8.058e-07 | 8.236e-07 | 8.384e-07 | 2.382e-09 | 2.683e-09 |
| bns | L1 | regular | 50 | B1_all | 7.443e-07 | 8.058e-07 | 8.236e-07 | 8.384e-07 | 2.382e-09 | 2.683e-09 |
| bns | L1 | regular | 50 | B0 | 7.443e-07 | 8.058e-07 | 8.236e-07 | 8.384e-07 | 2.382e-09 | 2.683e-09 |
| bns | L1 | combined | 5 | linear | 9.144e-05 | 3.700e-04 | 4.937e-04 | 5.300e-04 | 2.926e-07 | 1.696e-06 |
| bns | L1 | combined | 5 | IFFT-FFT | 1.604e-05 | 3.710e-05 | 4.611e-05 | 4.800e-05 | 5.134e-08 | 1.536e-07 |
| bns | L1 | combined | 5 | pairwise | 1.263e-06 | 1.678e-06 | 2.064e-06 | 2.741e-06 | 4.042e-09 | 8.771e-09 |
| bns | L1 | combined | 5 | B1 | 4.179e-05 | 8.168e-05 | 1.041e-04 | 1.098e-04 | 1.337e-07 | 3.513e-07 |
| bns | L1 | combined | 5 | B1_all | 2.791e-05 | 5.840e-05 | 1.135e-04 | 1.165e-04 | 8.932e-08 | 3.728e-07 |
| bns | L1 | combined | 5 | B0 | 1.212e-06 | 1.920e-06 | 2.848e-05 | 2.850e-05 | 3.877e-09 | 9.122e-08 |
| bns | L1 | combined | 50 | linear | 1.439e-05 | 7.017e-05 | 1.354e-04 | 1.507e-04 | 4.605e-08 | 4.822e-07 |
| bns | L1 | combined | 50 | IFFT-FFT | 1.968e-05 | 4.950e-05 | 6.372e-05 | 6.842e-05 | 6.298e-08 | 2.189e-07 |
| bns | L1 | combined | 50 | pairwise | 7.430e-07 | 8.097e-07 | 8.284e-07 | 8.430e-07 | 2.378e-09 | 2.698e-09 |
| bns | L1 | combined | 50 | B1 | 7.443e-07 | 8.058e-07 | 8.236e-07 | 8.384e-07 | 2.382e-09 | 2.683e-09 |
| bns | L1 | combined | 50 | B1_all | 7.443e-07 | 8.058e-07 | 8.236e-07 | 8.384e-07 | 2.382e-09 | 2.683e-09 |
| bns | L1 | combined | 50 | B0 | 7.443e-07 | 8.058e-07 | 8.236e-07 | 8.384e-07 | 2.382e-09 | 2.683e-09 |
| nsbh | lines | regular | 5 | linear | 2.233e-03 | 5.259e-03 | 7.553e-03 | 9.340e-03 | 7.146e-06 | 2.989e-05 |
| nsbh | lines | regular | 5 | IFFT-FFT | 4.259e-05 | 6.763e-05 | 8.754e-05 | 8.808e-05 | 1.363e-07 | 2.818e-07 |
| nsbh | lines | regular | 5 | pairwise | 4.420e-05 | 6.795e-05 | 8.005e-05 | 8.309e-05 | 1.414e-07 | 2.659e-07 |
| nsbh | lines | regular | 5 | B1 | 3.468e-05 | 6.473e-05 | 8.661e-05 | 9.349e-05 | 1.110e-07 | 2.992e-07 |
| nsbh | lines | regular | 5 | B1_all | 3.468e-05 | 6.473e-05 | 8.661e-05 | 9.349e-05 | 1.110e-07 | 2.992e-07 |
| nsbh | lines | regular | 5 | B0 | 4.422e-05 | 6.804e-05 | 7.975e-05 | 8.291e-05 | 1.415e-07 | 2.653e-07 |
| nsbh | lines | regular | 50 | linear | 8.535e-05 | 3.565e-04 | 5.526e-04 | 5.907e-04 | 2.731e-07 | 1.890e-06 |
| nsbh | lines | regular | 50 | IFFT-FFT | 1.393e-06 | 4.200e-06 | 5.100e-06 | 5.208e-06 | 4.456e-09 | 1.667e-08 |
| nsbh | lines | regular | 50 | pairwise | 1.445e-06 | 2.080e-06 | 2.422e-06 | 2.598e-06 | 4.623e-09 | 8.315e-09 |
| nsbh | lines | regular | 50 | B1 | 1.440e-06 | 2.063e-06 | 2.406e-06 | 2.591e-06 | 4.608e-09 | 8.293e-09 |
| nsbh | lines | regular | 50 | B1_all | 1.440e-06 | 2.063e-06 | 2.406e-06 | 2.591e-06 | 4.608e-09 | 8.293e-09 |
| nsbh | lines | regular | 50 | B0 | 1.440e-06 | 2.063e-06 | 2.406e-06 | 2.591e-06 | 4.608e-09 | 8.293e-09 |
| nsbh | lines | strong_hom | 5 | linear | 2.976e-03 | 6.533e-03 | 8.228e-03 | 8.363e-03 | 9.523e-06 | 2.676e-05 |
| nsbh | lines | strong_hom | 5 | IFFT-FFT | 5.094e-05 | 7.908e-05 | 8.600e-05 | 8.705e-05 | 1.630e-07 | 2.785e-07 |
| nsbh | lines | strong_hom | 5 | pairwise | 5.038e-05 | 7.146e-05 | 7.489e-05 | 7.492e-05 | 1.612e-07 | 2.397e-07 |
| nsbh | lines | strong_hom | 5 | B1 | 4.559e-05 | 7.237e-05 | 8.079e-05 | 8.151e-05 | 1.459e-07 | 2.608e-07 |
| nsbh | lines | strong_hom | 5 | B1_all | 4.559e-05 | 7.237e-05 | 8.079e-05 | 8.151e-05 | 1.459e-07 | 2.608e-07 |
| nsbh | lines | strong_hom | 5 | B0 | 5.058e-05 | 7.233e-05 | 7.536e-05 | 7.544e-05 | 1.619e-07 | 2.414e-07 |
| nsbh | lines | strong_hom | 50 | linear | 1.142e-04 | 2.482e-04 | 4.516e-04 | 4.933e-04 | 3.656e-07 | 1.579e-06 |
| nsbh | lines | strong_hom | 50 | IFFT-FFT | 1.550e-06 | 5.026e-06 | 5.678e-06 | 5.778e-06 | 4.961e-09 | 1.849e-08 |
| nsbh | lines | strong_hom | 50 | pairwise | 1.465e-06 | 2.010e-06 | 2.357e-06 | 2.421e-06 | 4.688e-09 | 7.746e-09 |
| nsbh | lines | strong_hom | 50 | B1 | 1.483e-06 | 2.016e-06 | 2.340e-06 | 2.394e-06 | 4.747e-09 | 7.662e-09 |
| nsbh | lines | strong_hom | 50 | B1_all | 1.483e-06 | 2.016e-06 | 2.340e-06 | 2.394e-06 | 4.747e-09 | 7.662e-09 |
| nsbh | lines | strong_hom | 50 | B0 | 1.483e-06 | 2.016e-06 | 2.340e-06 | 2.394e-06 | 4.747e-09 | 7.662e-09 |
| nsbh | lines | combined | 5 | linear | 2.273e-03 | 5.533e-03 | 8.228e-03 | 9.340e-03 | 7.273e-06 | 2.989e-05 |
| nsbh | lines | combined | 5 | IFFT-FFT | 4.324e-05 | 6.923e-05 | 8.744e-05 | 8.808e-05 | 1.384e-07 | 2.818e-07 |
| nsbh | lines | combined | 5 | pairwise | 4.532e-05 | 6.850e-05 | 7.944e-05 | 8.309e-05 | 1.450e-07 | 2.659e-07 |
| nsbh | lines | combined | 5 | B1 | 3.697e-05 | 6.708e-05 | 8.577e-05 | 9.349e-05 | 1.183e-07 | 2.992e-07 |
| nsbh | lines | combined | 5 | B1_all | 3.697e-05 | 6.708e-05 | 8.577e-05 | 9.349e-05 | 1.183e-07 | 2.992e-07 |
| nsbh | lines | combined | 5 | B0 | 4.521e-05 | 6.851e-05 | 7.925e-05 | 8.291e-05 | 1.447e-07 | 2.653e-07 |
| nsbh | lines | combined | 50 | linear | 8.988e-05 | 3.282e-04 | 5.521e-04 | 5.907e-04 | 2.876e-07 | 1.890e-06 |
| nsbh | lines | combined | 50 | IFFT-FFT | 1.421e-06 | 4.452e-06 | 5.243e-06 | 5.778e-06 | 4.548e-09 | 1.849e-08 |
| nsbh | lines | combined | 50 | pairwise | 1.454e-06 | 2.080e-06 | 2.421e-06 | 2.598e-06 | 4.654e-09 | 8.315e-09 |
| nsbh | lines | combined | 50 | B1 | 1.451e-06 | 2.063e-06 | 2.402e-06 | 2.591e-06 | 4.642e-09 | 8.293e-09 |
| nsbh | lines | combined | 50 | B1_all | 1.451e-06 | 2.063e-06 | 2.402e-06 | 2.591e-06 | 4.642e-09 | 8.293e-09 |
| nsbh | lines | combined | 50 | B0 | 1.451e-06 | 2.063e-06 | 2.402e-06 | 2.591e-06 | 4.642e-09 | 8.293e-09 |
| nsbh | lines | regular_inside_corners | 5 | linear | 2.252e-03 | 5.261e-03 | 7.571e-03 | 9.340e-03 | 7.206e-06 | 2.989e-05 |
| nsbh | lines | regular_inside_corners | 5 | IFFT-FFT | 4.251e-05 | 6.629e-05 | 8.754e-05 | 8.808e-05 | 1.360e-07 | 2.818e-07 |
| nsbh | lines | regular_inside_corners | 5 | pairwise | 4.372e-05 | 6.771e-05 | 8.008e-05 | 8.309e-05 | 1.399e-07 | 2.659e-07 |
| nsbh | lines | regular_inside_corners | 5 | B1 | 3.461e-05 | 6.418e-05 | 8.668e-05 | 9.349e-05 | 1.107e-07 | 2.992e-07 |
| nsbh | lines | regular_inside_corners | 5 | B1_all | 3.461e-05 | 6.418e-05 | 8.668e-05 | 9.349e-05 | 1.107e-07 | 2.992e-07 |
| nsbh | lines | regular_inside_corners | 5 | B0 | 4.375e-05 | 6.794e-05 | 7.978e-05 | 8.291e-05 | 1.400e-07 | 2.653e-07 |
| nsbh | lines | regular_inside_corners | 50 | linear | 8.389e-05 | 3.579e-04 | 5.529e-04 | 5.907e-04 | 2.685e-07 | 1.890e-06 |
| nsbh | lines | regular_inside_corners | 50 | IFFT-FFT | 1.497e-06 | 4.223e-06 | 5.101e-06 | 5.208e-06 | 4.790e-09 | 1.667e-08 |
| nsbh | lines | regular_inside_corners | 50 | pairwise | 1.447e-06 | 2.081e-06 | 2.424e-06 | 2.598e-06 | 4.632e-09 | 8.315e-09 |
| nsbh | lines | regular_inside_corners | 50 | B1 | 1.445e-06 | 2.065e-06 | 2.408e-06 | 2.591e-06 | 4.623e-09 | 8.293e-09 |
| nsbh | lines | regular_inside_corners | 50 | B1_all | 1.445e-06 | 2.065e-06 | 2.408e-06 | 2.591e-06 | 4.623e-09 | 8.293e-09 |
| nsbh | lines | regular_inside_corners | 50 | B0 | 1.445e-06 | 2.065e-06 | 2.408e-06 | 2.591e-06 | 4.623e-09 | 8.293e-09 |
| nsbh | L1 | regular | 5 | linear | 1.128e-03 | 3.754e-03 | 9.156e-03 | 9.688e-03 | 3.609e-06 | 3.100e-05 |
| nsbh | L1 | regular | 5 | IFFT-FFT | 5.253e-05 | 1.280e-04 | 1.561e-04 | 1.586e-04 | 1.681e-07 | 5.074e-07 |
| nsbh | L1 | regular | 5 | pairwise | 1.878e-05 | 4.712e-05 | 6.121e-05 | 6.365e-05 | 6.009e-08 | 2.037e-07 |
| nsbh | L1 | regular | 5 | B1 | 5.106e-05 | 1.231e-04 | 2.333e-04 | 2.902e-04 | 1.634e-07 | 9.285e-07 |
| nsbh | L1 | regular | 5 | B1_all | 5.106e-05 | 1.231e-04 | 2.333e-04 | 2.902e-04 | 1.634e-07 | 9.285e-07 |
| nsbh | L1 | regular | 5 | B0 | 1.904e-05 | 4.499e-05 | 5.942e-05 | 6.118e-05 | 6.094e-08 | 1.958e-07 |
| nsbh | L1 | regular | 50 | linear | 4.566e-05 | 1.178e-04 | 1.726e-04 | 1.807e-04 | 1.461e-07 | 5.782e-07 |
| nsbh | L1 | regular | 50 | IFFT-FFT | 1.605e-06 | 4.732e-06 | 5.996e-06 | 6.475e-06 | 5.136e-09 | 2.072e-08 |
| nsbh | L1 | regular | 50 | pairwise | 8.509e-07 | 1.307e-06 | 1.787e-06 | 1.801e-06 | 2.723e-09 | 5.764e-09 |
| nsbh | L1 | regular | 50 | B1 | 8.475e-07 | 1.323e-06 | 1.800e-06 | 1.834e-06 | 2.712e-09 | 5.868e-09 |
| nsbh | L1 | regular | 50 | B1_all | 8.475e-07 | 1.323e-06 | 1.800e-06 | 1.834e-06 | 2.712e-09 | 5.868e-09 |
| nsbh | L1 | regular | 50 | B0 | 8.475e-07 | 1.323e-06 | 1.800e-06 | 1.834e-06 | 2.712e-09 | 5.868e-09 |
| nsbh | L1 | strong_hom | 5 | linear | 9.808e-04 | 7.277e-03 | 1.101e-02 | 1.104e-02 | 3.138e-06 | 3.532e-05 |
| nsbh | L1 | strong_hom | 5 | IFFT-FFT | 5.286e-05 | 9.669e-05 | 1.895e-04 | 2.098e-04 | 1.691e-07 | 6.713e-07 |
| nsbh | L1 | strong_hom | 5 | pairwise | 2.269e-05 | 4.411e-05 | 5.084e-05 | 5.181e-05 | 7.262e-08 | 1.658e-07 |
| nsbh | L1 | strong_hom | 5 | B1 | 4.652e-05 | 9.859e-05 | 1.374e-04 | 1.378e-04 | 1.489e-07 | 4.410e-07 |
| nsbh | L1 | strong_hom | 5 | B1_all | 4.652e-05 | 9.859e-05 | 1.374e-04 | 1.378e-04 | 1.489e-07 | 4.410e-07 |
| nsbh | L1 | strong_hom | 5 | B0 | 2.166e-05 | 4.409e-05 | 4.740e-05 | 4.796e-05 | 6.931e-08 | 1.535e-07 |
| nsbh | L1 | strong_hom | 50 | linear | 7.533e-05 | 1.140e-04 | 1.664e-04 | 1.746e-04 | 2.411e-07 | 5.586e-07 |
| nsbh | L1 | strong_hom | 50 | IFFT-FFT | 1.404e-06 | 3.937e-06 | 7.312e-06 | 7.543e-06 | 4.493e-09 | 2.414e-08 |
| nsbh | L1 | strong_hom | 50 | pairwise | 9.251e-07 | 1.402e-06 | 1.438e-06 | 1.440e-06 | 2.960e-09 | 4.610e-09 |
| nsbh | L1 | strong_hom | 50 | B1 | 9.158e-07 | 1.426e-06 | 1.454e-06 | 1.456e-06 | 2.930e-09 | 4.660e-09 |
| nsbh | L1 | strong_hom | 50 | B1_all | 9.158e-07 | 1.426e-06 | 1.454e-06 | 1.456e-06 | 2.930e-09 | 4.660e-09 |
| nsbh | L1 | strong_hom | 50 | B0 | 9.158e-07 | 1.426e-06 | 1.454e-06 | 1.456e-06 | 2.930e-09 | 4.660e-09 |
| nsbh | L1 | combined | 5 | linear | 1.111e-03 | 4.147e-03 | 1.068e-02 | 1.104e-02 | 3.557e-06 | 3.532e-05 |
| nsbh | L1 | combined | 5 | IFFT-FFT | 5.253e-05 | 1.271e-04 | 1.581e-04 | 2.098e-04 | 1.681e-07 | 6.713e-07 |
| nsbh | L1 | combined | 5 | pairwise | 1.955e-05 | 4.675e-05 | 6.066e-05 | 6.365e-05 | 6.255e-08 | 2.037e-07 |
| nsbh | L1 | combined | 5 | B1 | 5.031e-05 | 1.231e-04 | 2.319e-04 | 2.902e-04 | 1.610e-07 | 9.285e-07 |
| nsbh | L1 | combined | 5 | B1_all | 5.031e-05 | 1.231e-04 | 2.319e-04 | 2.902e-04 | 1.610e-07 | 9.285e-07 |
| nsbh | L1 | combined | 5 | B0 | 1.904e-05 | 4.498e-05 | 5.887e-05 | 6.118e-05 | 6.094e-08 | 1.958e-07 |
| nsbh | L1 | combined | 50 | linear | 4.916e-05 | 1.178e-04 | 1.742e-04 | 1.807e-04 | 1.573e-07 | 5.782e-07 |
| nsbh | L1 | combined | 50 | IFFT-FFT | 1.551e-06 | 4.732e-06 | 6.447e-06 | 7.543e-06 | 4.962e-09 | 2.414e-08 |
| nsbh | L1 | combined | 50 | pairwise | 8.509e-07 | 1.342e-06 | 1.749e-06 | 1.801e-06 | 2.723e-09 | 5.764e-09 |
| nsbh | L1 | combined | 50 | B1 | 8.475e-07 | 1.369e-06 | 1.761e-06 | 1.834e-06 | 2.712e-09 | 5.868e-09 |
| nsbh | L1 | combined | 50 | B1_all | 8.475e-07 | 1.369e-06 | 1.761e-06 | 1.834e-06 | 2.712e-09 | 5.868e-09 |
| nsbh | L1 | combined | 50 | B0 | 8.475e-07 | 1.369e-06 | 1.761e-06 | 1.834e-06 | 2.712e-09 | 5.868e-09 |
| nsbh | L1 | regular_inside_corners | 5 | linear | 1.123e-03 | 3.766e-03 | 9.162e-03 | 9.688e-03 | 3.594e-06 | 3.100e-05 |
| nsbh | L1 | regular_inside_corners | 5 | IFFT-FFT | 5.202e-05 | 1.282e-04 | 1.562e-04 | 1.586e-04 | 1.665e-07 | 5.074e-07 |
| nsbh | L1 | regular_inside_corners | 5 | pairwise | 1.811e-05 | 4.730e-05 | 6.124e-05 | 6.365e-05 | 5.794e-08 | 2.037e-07 |
| nsbh | L1 | regular_inside_corners | 5 | B1 | 5.089e-05 | 1.240e-04 | 2.339e-04 | 2.902e-04 | 1.629e-07 | 9.285e-07 |
| nsbh | L1 | regular_inside_corners | 5 | B1_all | 5.089e-05 | 1.240e-04 | 2.339e-04 | 2.902e-04 | 1.629e-07 | 9.285e-07 |
| nsbh | L1 | regular_inside_corners | 5 | B0 | 1.882e-05 | 4.304e-05 | 5.944e-05 | 6.118e-05 | 6.023e-08 | 1.958e-07 |
| nsbh | L1 | regular_inside_corners | 50 | linear | 4.529e-05 | 1.179e-04 | 1.727e-04 | 1.807e-04 | 1.449e-07 | 5.782e-07 |
| nsbh | L1 | regular_inside_corners | 50 | IFFT-FFT | 1.607e-06 | 4.749e-06 | 6.001e-06 | 6.475e-06 | 5.142e-09 | 2.072e-08 |
| nsbh | L1 | regular_inside_corners | 50 | pairwise | 8.496e-07 | 1.311e-06 | 1.787e-06 | 1.801e-06 | 2.719e-09 | 5.764e-09 |
| nsbh | L1 | regular_inside_corners | 50 | B1 | 8.467e-07 | 1.326e-06 | 1.800e-06 | 1.834e-06 | 2.710e-09 | 5.868e-09 |
| nsbh | L1 | regular_inside_corners | 50 | B1_all | 8.467e-07 | 1.326e-06 | 1.800e-06 | 1.834e-06 | 2.710e-09 | 5.868e-09 |
| nsbh | L1 | regular_inside_corners | 50 | B0 | 8.467e-07 | 1.326e-06 | 1.800e-06 | 1.834e-06 | 2.710e-09 | 5.868e-09 |
| bbh | lines | regular | 5 | linear | 9.429e-04 | 3.914e-03 | 7.695e-03 | 7.770e-03 | 3.017e-06 | 2.486e-05 |
| bbh | lines | regular | 5 | IFFT-FFT | 3.288e-05 | 7.271e-05 | 1.147e-04 | 1.206e-04 | 1.052e-07 | 3.859e-07 |
| bbh | lines | regular | 5 | pairwise | 3.163e-05 | 7.547e-05 | 1.035e-04 | 1.246e-04 | 1.012e-07 | 3.988e-07 |
| bbh | lines | regular | 5 | B1 | 2.490e-05 | 7.164e-05 | 1.335e-04 | 1.532e-04 | 7.967e-08 | 4.901e-07 |
| bbh | lines | regular | 5 | B1_all | 2.487e-05 | 7.158e-05 | 1.335e-04 | 1.531e-04 | 7.960e-08 | 4.901e-07 |
| bbh | lines | regular | 5 | B0 | 3.234e-05 | 7.153e-05 | 1.063e-04 | 1.214e-04 | 1.035e-07 | 3.886e-07 |
| bbh | lines | regular | 50 | linear | 2.704e-04 | 6.360e-04 | 9.554e-04 | 9.912e-04 | 8.651e-07 | 3.172e-06 |
| bbh | lines | regular | 50 | IFFT-FFT | 3.127e-06 | 6.717e-06 | 9.773e-06 | 1.150e-05 | 1.001e-08 | 3.679e-08 |
| bbh | lines | regular | 50 | pairwise | 2.965e-07 | 9.098e-07 | 1.914e-06 | 2.370e-06 | 9.489e-10 | 7.585e-09 |
| bbh | lines | regular | 50 | B1 | 2.768e-07 | 8.881e-07 | 1.967e-06 | 2.427e-06 | 8.858e-10 | 7.768e-09 |
| bbh | lines | regular | 50 | B1_all | 2.768e-07 | 8.881e-07 | 1.967e-06 | 2.427e-06 | 8.858e-10 | 7.768e-09 |
| bbh | lines | regular | 50 | B0 | 2.768e-07 | 8.881e-07 | 1.967e-06 | 2.427e-06 | 8.858e-10 | 7.768e-09 |
| bbh | lines | strong_hom | 5 | linear | 1.945e-03 | 1.065e-02 | 1.501e-02 | 1.579e-02 | 6.225e-06 | 5.054e-05 |
| bbh | lines | strong_hom | 5 | IFFT-FFT | 4.610e-05 | 7.870e-05 | 9.683e-05 | 9.913e-05 | 1.475e-07 | 3.172e-07 |
| bbh | lines | strong_hom | 5 | pairwise | 4.496e-05 | 8.564e-05 | 9.515e-05 | 9.558e-05 | 1.439e-07 | 3.059e-07 |
| bbh | lines | strong_hom | 5 | B1 | 4.711e-05 | 1.210e-04 | 1.990e-04 | 2.145e-04 | 1.507e-07 | 6.863e-07 |
| bbh | lines | strong_hom | 5 | B1_all | 4.712e-05 | 1.210e-04 | 1.990e-04 | 2.145e-04 | 1.508e-07 | 6.863e-07 |
| bbh | lines | strong_hom | 5 | B0 | 4.372e-05 | 7.925e-05 | 9.830e-05 | 1.016e-04 | 1.399e-07 | 3.251e-07 |
| bbh | lines | strong_hom | 50 | linear | 1.892e-04 | 6.433e-04 | 1.081e-03 | 1.113e-03 | 6.054e-07 | 3.562e-06 |
| bbh | lines | strong_hom | 50 | IFFT-FFT | 2.764e-06 | 5.720e-06 | 1.118e-05 | 1.230e-05 | 8.845e-09 | 3.934e-08 |
| bbh | lines | strong_hom | 50 | pairwise | 5.262e-07 | 1.305e-06 | 1.561e-06 | 1.614e-06 | 1.684e-09 | 5.164e-09 |
| bbh | lines | strong_hom | 50 | B1 | 5.082e-07 | 1.322e-06 | 1.648e-06 | 1.702e-06 | 1.626e-09 | 5.446e-09 |
| bbh | lines | strong_hom | 50 | B1_all | 5.082e-07 | 1.322e-06 | 1.648e-06 | 1.702e-06 | 1.626e-09 | 5.446e-09 |
| bbh | lines | strong_hom | 50 | B0 | 5.082e-07 | 1.322e-06 | 1.648e-06 | 1.702e-06 | 1.626e-09 | 5.446e-09 |
| bbh | lines | combined | 5 | linear | 1.000e-03 | 5.521e-03 | 1.146e-02 | 1.579e-02 | 3.202e-06 | 5.054e-05 |
| bbh | lines | combined | 5 | IFFT-FFT | 3.395e-05 | 7.618e-05 | 1.143e-04 | 1.206e-04 | 1.086e-07 | 3.859e-07 |
| bbh | lines | combined | 5 | pairwise | 3.538e-05 | 8.077e-05 | 1.029e-04 | 1.246e-04 | 1.132e-07 | 3.988e-07 |
| bbh | lines | combined | 5 | B1 | 2.747e-05 | 7.743e-05 | 1.494e-04 | 2.145e-04 | 8.789e-08 | 6.863e-07 |
| bbh | lines | combined | 5 | B1_all | 2.745e-05 | 7.739e-05 | 1.494e-04 | 2.145e-04 | 8.783e-08 | 6.863e-07 |
| bbh | lines | combined | 5 | B0 | 3.305e-05 | 7.381e-05 | 1.053e-04 | 1.214e-04 | 1.058e-07 | 3.886e-07 |
| bbh | lines | combined | 50 | linear | 2.390e-04 | 6.360e-04 | 9.843e-04 | 1.113e-03 | 7.648e-07 | 3.562e-06 |
| bbh | lines | combined | 50 | IFFT-FFT | 3.073e-06 | 6.633e-06 | 1.117e-05 | 1.230e-05 | 9.833e-09 | 3.934e-08 |
| bbh | lines | combined | 50 | pairwise | 2.993e-07 | 1.068e-06 | 1.868e-06 | 2.370e-06 | 9.577e-10 | 7.585e-09 |
| bbh | lines | combined | 50 | B1 | 3.127e-07 | 1.030e-06 | 1.913e-06 | 2.427e-06 | 1.001e-09 | 7.768e-09 |
| bbh | lines | combined | 50 | B1_all | 3.127e-07 | 1.030e-06 | 1.913e-06 | 2.427e-06 | 1.001e-09 | 7.768e-09 |
| bbh | lines | combined | 50 | B0 | 3.127e-07 | 1.030e-06 | 1.913e-06 | 2.427e-06 | 1.001e-09 | 7.768e-09 |
| bbh | L1 | regular | 5 | linear | 1.038e-03 | 2.609e-03 | 7.018e-03 | 7.347e-03 | 3.323e-06 | 2.351e-05 |
| bbh | L1 | regular | 5 | IFFT-FFT | 1.873e-04 | 2.756e-04 | 3.123e-04 | 4.599e-04 | 5.994e-07 | 1.472e-06 |
| bbh | L1 | regular | 5 | pairwise | 3.292e-05 | 6.341e-05 | 9.216e-05 | 1.087e-04 | 1.054e-07 | 3.479e-07 |
| bbh | L1 | regular | 5 | B1 | 1.419e-04 | 4.386e-04 | 5.320e-04 | 6.117e-04 | 4.542e-07 | 1.957e-06 |
| bbh | L1 | regular | 5 | B1_all | 1.419e-04 | 4.386e-04 | 5.321e-04 | 6.117e-04 | 4.542e-07 | 1.957e-06 |
| bbh | L1 | regular | 5 | B0 | 3.164e-05 | 6.383e-05 | 1.107e-04 | 1.169e-04 | 1.012e-07 | 3.741e-07 |
| bbh | L1 | regular | 50 | linear | 1.393e-04 | 2.616e-04 | 3.960e-04 | 3.992e-04 | 4.458e-07 | 1.277e-06 |
| bbh | L1 | regular | 50 | IFFT-FFT | 5.771e-06 | 1.598e-05 | 2.773e-05 | 2.833e-05 | 1.847e-08 | 9.066e-08 |
| bbh | L1 | regular | 50 | pairwise | 2.750e-06 | 4.029e-06 | 5.543e-06 | 5.843e-06 | 8.799e-09 | 1.870e-08 |
| bbh | L1 | regular | 50 | B1 | 2.785e-06 | 4.029e-06 | 5.493e-06 | 5.777e-06 | 8.912e-09 | 1.849e-08 |
| bbh | L1 | regular | 50 | B1_all | 2.785e-06 | 4.029e-06 | 5.493e-06 | 5.777e-06 | 8.912e-09 | 1.849e-08 |
| bbh | L1 | regular | 50 | B0 | 2.785e-06 | 4.029e-06 | 5.493e-06 | 5.777e-06 | 8.912e-09 | 1.849e-08 |
| bbh | L1 | strong_hom | 5 | linear | 2.006e-03 | 6.248e-03 | 6.853e-03 | 6.896e-03 | 6.420e-06 | 2.207e-05 |
| bbh | L1 | strong_hom | 5 | IFFT-FFT | 1.984e-04 | 3.165e-04 | 3.725e-04 | 3.755e-04 | 6.350e-07 | 1.202e-06 |
| bbh | L1 | strong_hom | 5 | pairwise | 5.545e-05 | 8.121e-05 | 1.091e-04 | 1.155e-04 | 1.774e-07 | 3.695e-07 |
| bbh | L1 | strong_hom | 5 | B1 | 2.138e-04 | 4.451e-04 | 6.834e-04 | 7.031e-04 | 6.841e-07 | 2.250e-06 |
| bbh | L1 | strong_hom | 5 | B1_all | 2.138e-04 | 4.451e-04 | 6.834e-04 | 7.031e-04 | 6.841e-07 | 2.250e-06 |
| bbh | L1 | strong_hom | 5 | B0 | 3.641e-05 | 8.364e-05 | 1.259e-04 | 1.283e-04 | 1.165e-07 | 4.106e-07 |
| bbh | L1 | strong_hom | 50 | linear | 2.448e-04 | 4.369e-04 | 5.605e-04 | 5.891e-04 | 7.834e-07 | 1.885e-06 |
| bbh | L1 | strong_hom | 50 | IFFT-FFT | 5.540e-06 | 2.226e-05 | 3.114e-05 | 3.131e-05 | 1.773e-08 | 1.002e-07 |
| bbh | L1 | strong_hom | 50 | pairwise | 2.486e-06 | 4.542e-06 | 6.262e-06 | 6.579e-06 | 7.956e-09 | 2.105e-08 |
| bbh | L1 | strong_hom | 50 | B1 | 2.440e-06 | 4.433e-06 | 6.122e-06 | 6.432e-06 | 7.807e-09 | 2.058e-08 |
| bbh | L1 | strong_hom | 50 | B1_all | 2.440e-06 | 4.433e-06 | 6.122e-06 | 6.432e-06 | 7.807e-09 | 2.058e-08 |
| bbh | L1 | strong_hom | 50 | B0 | 2.440e-06 | 4.433e-06 | 6.122e-06 | 6.432e-06 | 7.807e-09 | 2.058e-08 |
| bbh | L1 | combined | 5 | linear | 1.080e-03 | 2.906e-03 | 6.992e-03 | 7.347e-03 | 3.457e-06 | 2.351e-05 |
| bbh | L1 | combined | 5 | IFFT-FFT | 1.874e-04 | 2.922e-04 | 3.725e-04 | 4.599e-04 | 5.998e-07 | 1.472e-06 |
| bbh | L1 | combined | 5 | pairwise | 3.317e-05 | 7.485e-05 | 1.056e-04 | 1.155e-04 | 1.062e-07 | 3.695e-07 |
| bbh | L1 | combined | 5 | B1 | 1.573e-04 | 4.386e-04 | 6.093e-04 | 7.031e-04 | 5.035e-07 | 2.250e-06 |
| bbh | L1 | combined | 5 | B1_all | 1.573e-04 | 4.386e-04 | 6.093e-04 | 7.031e-04 | 5.035e-07 | 2.250e-06 |
| bbh | L1 | combined | 5 | B0 | 3.298e-05 | 6.904e-05 | 1.167e-04 | 1.283e-04 | 1.055e-07 | 4.106e-07 |
| bbh | L1 | combined | 50 | linear | 1.473e-04 | 2.933e-04 | 4.383e-04 | 5.891e-04 | 4.712e-07 | 1.885e-06 |
| bbh | L1 | combined | 50 | IFFT-FFT | 5.709e-06 | 1.826e-05 | 3.000e-05 | 3.131e-05 | 1.827e-08 | 1.002e-07 |
| bbh | L1 | combined | 50 | pairwise | 2.744e-06 | 4.091e-06 | 5.785e-06 | 6.579e-06 | 8.781e-09 | 2.105e-08 |
| bbh | L1 | combined | 50 | B1 | 2.778e-06 | 4.042e-06 | 5.723e-06 | 6.432e-06 | 8.890e-09 | 2.058e-08 |
| bbh | L1 | combined | 50 | B1_all | 2.778e-06 | 4.042e-06 | 5.723e-06 | 6.432e-06 | 8.890e-09 | 2.058e-08 |
| bbh | L1 | combined | 50 | B0 | 2.778e-06 | 4.042e-06 | 5.723e-06 | 6.432e-06 | 8.890e-09 | 2.058e-08 |

## Ratios to IFFT-FFT and criteria

C1: pairwise median <= 1 and p99 <= 2. C1': B1 median and p99 <= 1.5. Ratios compare distribution statistics, not per-point ratios. Each row must pass for a population-wide claim. Strong-HOM is also assessed separately.

| Population | PSD | subset | L | method | median ratio | p99 ratio | criterion met |
|---|---|---|---:|---|---:|---:|---|
| bns | lines | regular | 5 | pairwise | 0.822 | 0.432 | yes |
| bns | lines | regular | 5 | B1 | 1.442 | 1.626 | NO |
| bns | lines | regular | 5 | B1_all | 3.518 | 3.594 | NO |
| bns | lines | regular | 50 | pairwise | 0.074 | 0.032 | yes |
| bns | lines | regular | 50 | B1 | 0.074 | 0.032 | yes |
| bns | lines | regular | 50 | B1_all | 0.074 | 0.032 | yes |
| bns | lines | combined | 5 | pairwise | 0.822 | 0.432 | yes |
| bns | lines | combined | 5 | B1 | 1.442 | 1.626 | NO |
| bns | lines | combined | 5 | B1_all | 3.518 | 3.594 | NO |
| bns | lines | combined | 50 | pairwise | 0.074 | 0.032 | yes |
| bns | lines | combined | 50 | B1 | 0.074 | 0.032 | yes |
| bns | lines | combined | 50 | B1_all | 0.074 | 0.032 | yes |
| bns | L1 | regular | 5 | pairwise | 0.079 | 0.045 | yes |
| bns | L1 | regular | 5 | B1 | 2.605 | 2.257 | NO |
| bns | L1 | regular | 5 | B1_all | 1.740 | 2.462 | NO |
| bns | L1 | regular | 50 | pairwise | 0.038 | 0.013 | yes |
| bns | L1 | regular | 50 | B1 | 0.038 | 0.013 | yes |
| bns | L1 | regular | 50 | B1_all | 0.038 | 0.013 | yes |
| bns | L1 | combined | 5 | pairwise | 0.079 | 0.045 | yes |
| bns | L1 | combined | 5 | B1 | 2.605 | 2.257 | NO |
| bns | L1 | combined | 5 | B1_all | 1.740 | 2.462 | NO |
| bns | L1 | combined | 50 | pairwise | 0.038 | 0.013 | yes |
| bns | L1 | combined | 50 | B1 | 0.038 | 0.013 | yes |
| bns | L1 | combined | 50 | B1_all | 0.038 | 0.013 | yes |
| nsbh | lines | regular | 5 | pairwise | 1.038 | 0.914 | NO |
| nsbh | lines | regular | 5 | B1 | 0.814 | 0.989 | yes |
| nsbh | lines | regular | 5 | B1_all | 0.814 | 0.989 | yes |
| nsbh | lines | regular | 50 | pairwise | 1.038 | 0.475 | NO |
| nsbh | lines | regular | 50 | B1 | 1.034 | 0.472 | yes |
| nsbh | lines | regular | 50 | B1_all | 1.034 | 0.472 | yes |
| nsbh | lines | strong_hom | 5 | pairwise | 0.989 | 0.871 | yes |
| nsbh | lines | strong_hom | 5 | B1 | 0.895 | 0.939 | yes |
| nsbh | lines | strong_hom | 5 | B1_all | 0.895 | 0.939 | yes |
| nsbh | lines | strong_hom | 50 | pairwise | 0.945 | 0.415 | yes |
| nsbh | lines | strong_hom | 50 | B1 | 0.957 | 0.412 | yes |
| nsbh | lines | strong_hom | 50 | B1_all | 0.957 | 0.412 | yes |
| nsbh | lines | combined | 5 | pairwise | 1.048 | 0.909 | NO |
| nsbh | lines | combined | 5 | B1 | 0.855 | 0.981 | yes |
| nsbh | lines | combined | 5 | B1_all | 0.855 | 0.981 | yes |
| nsbh | lines | combined | 50 | pairwise | 1.023 | 0.462 | NO |
| nsbh | lines | combined | 50 | B1 | 1.021 | 0.458 | yes |
| nsbh | lines | combined | 50 | B1_all | 1.021 | 0.458 | yes |
| nsbh | lines | regular_inside_corners | 5 | pairwise | 1.028 | 0.915 | NO |
| nsbh | lines | regular_inside_corners | 5 | B1 | 0.814 | 0.990 | yes |
| nsbh | lines | regular_inside_corners | 5 | B1_all | 0.814 | 0.990 | yes |
| nsbh | lines | regular_inside_corners | 50 | pairwise | 0.967 | 0.475 | yes |
| nsbh | lines | regular_inside_corners | 50 | B1 | 0.965 | 0.472 | yes |
| nsbh | lines | regular_inside_corners | 50 | B1_all | 0.965 | 0.472 | yes |
| nsbh | L1 | regular | 5 | pairwise | 0.357 | 0.392 | yes |
| nsbh | L1 | regular | 5 | B1 | 0.972 | 1.494 | yes |
| nsbh | L1 | regular | 5 | B1_all | 0.972 | 1.494 | yes |
| nsbh | L1 | regular | 50 | pairwise | 0.530 | 0.298 | yes |
| nsbh | L1 | regular | 50 | B1 | 0.528 | 0.300 | yes |
| nsbh | L1 | regular | 50 | B1_all | 0.528 | 0.300 | yes |
| nsbh | L1 | strong_hom | 5 | pairwise | 0.429 | 0.268 | yes |
| nsbh | L1 | strong_hom | 5 | B1 | 0.880 | 0.725 | yes |
| nsbh | L1 | strong_hom | 5 | B1_all | 0.880 | 0.725 | yes |
| nsbh | L1 | strong_hom | 50 | pairwise | 0.659 | 0.197 | yes |
| nsbh | L1 | strong_hom | 50 | B1 | 0.652 | 0.199 | yes |
| nsbh | L1 | strong_hom | 50 | B1_all | 0.652 | 0.199 | yes |
| nsbh | L1 | combined | 5 | pairwise | 0.372 | 0.384 | yes |
| nsbh | L1 | combined | 5 | B1 | 0.958 | 1.467 | yes |
| nsbh | L1 | combined | 5 | B1_all | 0.958 | 1.467 | yes |
| nsbh | L1 | combined | 50 | pairwise | 0.549 | 0.271 | yes |
| nsbh | L1 | combined | 50 | B1 | 0.547 | 0.273 | yes |
| nsbh | L1 | combined | 50 | B1_all | 0.547 | 0.273 | yes |
| nsbh | L1 | regular_inside_corners | 5 | pairwise | 0.348 | 0.392 | yes |
| nsbh | L1 | regular_inside_corners | 5 | B1 | 0.978 | 1.498 | yes |
| nsbh | L1 | regular_inside_corners | 5 | B1_all | 0.978 | 1.498 | yes |
| nsbh | L1 | regular_inside_corners | 50 | pairwise | 0.529 | 0.298 | yes |
| nsbh | L1 | regular_inside_corners | 50 | B1 | 0.527 | 0.300 | yes |
| nsbh | L1 | regular_inside_corners | 50 | B1_all | 0.527 | 0.300 | yes |
| bbh | lines | regular | 5 | pairwise | 0.962 | 0.902 | yes |
| bbh | lines | regular | 5 | B1 | 0.757 | 1.164 | yes |
| bbh | lines | regular | 5 | B1_all | 0.756 | 1.163 | yes |
| bbh | lines | regular | 50 | pairwise | 0.095 | 0.196 | yes |
| bbh | lines | regular | 50 | B1 | 0.089 | 0.201 | yes |
| bbh | lines | regular | 50 | B1_all | 0.089 | 0.201 | yes |
| bbh | lines | strong_hom | 5 | pairwise | 0.975 | 0.983 | yes |
| bbh | lines | strong_hom | 5 | B1 | 1.022 | 2.055 | NO |
| bbh | lines | strong_hom | 5 | B1_all | 1.022 | 2.055 | NO |
| bbh | lines | strong_hom | 50 | pairwise | 0.190 | 0.140 | yes |
| bbh | lines | strong_hom | 50 | B1 | 0.184 | 0.147 | yes |
| bbh | lines | strong_hom | 50 | B1_all | 0.184 | 0.147 | yes |
| bbh | lines | combined | 5 | pairwise | 1.042 | 0.900 | NO |
| bbh | lines | combined | 5 | B1 | 0.809 | 1.307 | yes |
| bbh | lines | combined | 5 | B1_all | 0.808 | 1.307 | yes |
| bbh | lines | combined | 50 | pairwise | 0.097 | 0.167 | yes |
| bbh | lines | combined | 50 | B1 | 0.102 | 0.171 | yes |
| bbh | lines | combined | 50 | B1_all | 0.102 | 0.171 | yes |
| bbh | L1 | regular | 5 | pairwise | 0.176 | 0.295 | yes |
| bbh | L1 | regular | 5 | B1 | 0.758 | 1.704 | NO |
| bbh | L1 | regular | 5 | B1_all | 0.758 | 1.704 | NO |
| bbh | L1 | regular | 50 | pairwise | 0.476 | 0.200 | yes |
| bbh | L1 | regular | 50 | B1 | 0.483 | 0.198 | yes |
| bbh | L1 | regular | 50 | B1_all | 0.483 | 0.198 | yes |
| bbh | L1 | strong_hom | 5 | pairwise | 0.279 | 0.293 | yes |
| bbh | L1 | strong_hom | 5 | B1 | 1.077 | 1.835 | NO |
| bbh | L1 | strong_hom | 5 | B1_all | 1.077 | 1.835 | NO |
| bbh | L1 | strong_hom | 50 | pairwise | 0.449 | 0.201 | yes |
| bbh | L1 | strong_hom | 50 | B1 | 0.440 | 0.197 | yes |
| bbh | L1 | strong_hom | 50 | B1_all | 0.440 | 0.197 | yes |
| bbh | L1 | combined | 5 | pairwise | 0.177 | 0.283 | yes |
| bbh | L1 | combined | 5 | B1 | 0.839 | 1.636 | NO |
| bbh | L1 | combined | 5 | B1_all | 0.840 | 1.636 | NO |
| bbh | L1 | combined | 50 | pairwise | 0.481 | 0.193 | yes |
| bbh | L1 | combined | 50 | B1 | 0.487 | 0.191 | yes |
| bbh | L1 | combined | 50 | B1_all | 0.487 | 0.191 | yes |

Empirical failures on the regular and strong-HOM subsets:

pairwise: nsbh/lines/regular/L=5, nsbh/lines/regular/L=50.

B1: bns/lines/regular/L=5, bns/L1/regular/L=5, bbh/lines/strong_hom/L=5, bbh/L1/regular/L=5, bbh/L1/strong_hom/L=5.

B1_all: bns/lines/regular/L=5, bns/L1/regular/L=5, bbh/lines/strong_hom/L=5, bbh/L1/regular/L=5, bbh/L1/strong_hom/L=5.

## Observed warm per-evaluation timings

OMP_NUM_THREADS=1; OMP_WAIT_POLICY=PASSIVE; scipy FFT workers=1. Medians across population evaluations after one complete untimed warm-up; waveform and HH steps timed separately (no reference or setup cost). These are throughput diagnostics with varying parameters, not a repeated fixed-system microbenchmark. Exact lookup times include cold reference calls on cache misses.

| Population | PSD | L | waveform total / modes (ms) | linear / IFFT-FFT / pair / B1 / B1_all / B0 HH (ms) |
|---|---|---:|---|---|
| bns | lines | 5 | 10.0421 / 12.7497 | 0.1022 / 4.2014 / 0.7596 / 0.6360 / 0.5727 / 0.0809 |
| bns | lines | 50 | 14.0418 / 17.6550 | 0.1162 / 6.5071 / 0.9471 / 0.1122 / 0.0981 / 0.0970 |
| bns | L1 | 5 | 10.2295 / 13.0946 | 0.1036 / 4.3688 / 0.7910 / 0.6522 / 0.5755 / 0.0823 |
| bns | L1 | 50 | 14.2498 / 18.1632 | 0.1254 / 6.5838 / 0.9732 / 0.1141 / 0.0988 / 0.0971 |
| nsbh | lines | 5 | 2.6902 / 3.4992 | 0.0341 / 0.7462 / 0.2626 / 0.2260 / 0.1693 / 0.0257 |
| nsbh | lines | 50 | 3.1869 / 4.1763 | 0.0361 / 0.9375 / 0.2749 / 0.0360 / 0.0283 / 0.0279 |
| nsbh | L1 | 5 | 2.7678 / 3.6063 | 0.0349 / 0.7639 / 0.2639 / 0.2314 / 0.1746 / 0.0266 |
| nsbh | L1 | 50 | 3.2723 / 4.2923 | 0.0375 / 0.9705 / 0.2791 / 0.0364 / 0.0279 / 0.0283 |
| bbh | lines | 5 | 2.4176 / 5.1136 | 0.0289 / 0.6758 / 0.2226 / 0.1868 / 0.1457 / 0.0227 |
| bbh | lines | 50 | 2.7461 / 6.0308 | 0.0313 / 0.7448 / 0.2364 / 0.0307 / 0.0228 / 0.0232 |
| bbh | L1 | 5 | 2.4076 / 5.1277 | 0.0289 / 0.6826 / 0.2237 / 0.1864 / 0.1451 / 0.0230 |
| bbh | L1 | 50 | 2.7267 / 5.8884 | 0.0314 / 0.7418 / 0.2417 / 0.0310 / 0.0237 / 0.0236 |

No LAL-refused parameter sets. All populations used the full requested n=100 unless a command-line override is shown above. Reported run wall times are from the final run; cache hit/miss counts are recorded in the JSON. See tc_scan_iter007.md and figures/iter007_tc_scan.png for the 21-point time scan. Pairwise peak-to-peak relative variation is at most 3.94e-16; linear is also invariant because it uses power. Per-point signed relative errors and exact norms are retained in *_full.json; committed JSON uses six significant digits and remains below 100 KB. Figures show index scatters and empirical CDFs.
