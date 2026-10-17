# Iteration 006: GW190814-like pairwise tuning
Nominal 23+2.6 Msun, chi_z=(0.1,0), t_c=14 s, phi_ref=0.7, F_plus=0.6, F_cross=0.5, distance=240 Mpc; L=5 bands {16,8,4} s, Mc prior [6,7] Msun. Six cases: inclinations {0.3,1.0,1.4} x {analytic+lines, real L1 Welch PSD}. No q=0.1 stress variants; consequently these medians differ from iteration 005's eight-case summary. Fine total references reuse iteration 005; fine mode sequences use XHMModel.mode. |Delta ln Lambda| at SNR25 = 312.5 relative HH error.
| L_margin | units | diag | median rel | worst rel | median likelihood | worst likelihood |
|---:|---|---|---:|---:|---:|---:|
| 1 | sigma_pair | block | 3.675e-07 | 4.443e-07 | 1.149e-04 | 1.388e-04 |
| 1 | sigma_pair | linear | 1.878e-06 | 2.113e-06 | 5.868e-04 | 6.602e-04 |
| 1 | sigma_mmax | block | 3.681e-07 | 4.467e-07 | 1.150e-04 | 1.396e-04 |
| 1 | sigma_mmax | linear | 1.877e-06 | 2.113e-06 | 5.866e-04 | 6.602e-04 |
| 2 | sigma_pair | block | 3.663e-07 | 4.457e-07 | 1.145e-04 | 1.393e-04 |
| 2 | sigma_pair | linear | 1.880e-06 | 2.113e-06 | 5.874e-04 | 6.604e-04 |
| 2 | sigma_mmax | block | 3.625e-07 | 4.290e-07 | 1.133e-04 | 1.341e-04 |
| 2 | sigma_mmax | linear | 1.883e-06 | 2.114e-06 | 5.885e-04 | 6.606e-04 |
| 3 | sigma_pair | block | 3.622e-07 | 4.337e-07 | 1.132e-04 | 1.355e-04 |
| 3 | sigma_pair | linear | 1.884e-06 | 2.115e-06 | 5.887e-04 | 6.608e-04 |
| 3 | sigma_mmax | block | 3.591e-07 | 4.215e-07 | 1.122e-04 | 1.317e-04 |
| 3 | sigma_mmax | linear | 1.884e-06 | 2.113e-06 | 5.888e-04 | 6.604e-04 |
| 5 | sigma_pair | block | 3.626e-07 | 4.281e-07 | 1.133e-04 | 1.338e-04 |
| 5 | sigma_pair | linear | 1.883e-06 | 2.113e-06 | 5.885e-04 | 6.605e-04 |
| 5 | sigma_mmax | block | 3.595e-07 | 4.202e-07 | 1.124e-04 | 1.313e-04 |
| 5 | sigma_mmax | linear | 1.884e-06 | 2.113e-06 | 5.887e-04 | 6.604e-04 |
| baseline | IFFT-FFT | — | 1.539e-07 | 1.976e-07 | 4.809e-05 | 6.174e-05 |
| baseline | B1 | — | 3.473e-07 | 5.199e-07 | 1.085e-04 | 1.625e-04 |

## Per-PSD median / worst relative errors
| Setting (L, units, diag) | analytic+lines | L1 real |
|---|---:|---:|
| 1, sigma_pair, block | 3.060e-07 / 3.176e-07 | 4.312e-07 / 4.443e-07 |
| 1, sigma_pair, linear | 2.036e-06 / 2.113e-06 | 1.677e-06 / 1.728e-06 |
| 1, sigma_mmax, block | 3.060e-07 / 3.178e-07 | 4.342e-07 / 4.467e-07 |
| 1, sigma_mmax, linear | 2.036e-06 / 2.113e-06 | 1.674e-06 / 1.727e-06 |
| 2, sigma_pair, block | 3.054e-07 / 3.147e-07 | 4.330e-07 / 4.457e-07 |
| 2, sigma_pair, linear | 2.039e-06 / 2.113e-06 | 1.675e-06 / 1.727e-06 |
| 2, sigma_mmax, block | 3.050e-07 / 3.137e-07 | 4.138e-07 / 4.290e-07 |
| 2, sigma_mmax, linear | 2.040e-06 / 2.114e-06 | 1.694e-06 / 1.734e-06 |
| 3, sigma_pair, block | 3.042e-07 / 3.110e-07 | 4.194e-07 / 4.337e-07 |
| 3, sigma_pair, linear | 2.042e-06 / 2.115e-06 | 1.689e-06 / 1.732e-06 |
| 3, sigma_mmax, block | 3.056e-07 / 3.146e-07 | 4.088e-07 / 4.215e-07 |
| 3, sigma_mmax, linear | 2.039e-06 / 2.113e-06 | 1.704e-06 / 1.736e-06 |
| 5, sigma_pair, block | 3.053e-07 / 3.147e-07 | 4.133e-07 / 4.281e-07 |
| 5, sigma_pair, linear | 2.039e-06 / 2.113e-06 | 1.695e-06 / 1.735e-06 |
| 5, sigma_mmax, block | 3.055e-07 / 3.158e-07 | 4.074e-07 / 4.202e-07 |
| 5, sigma_mmax, linear | 2.038e-06 / 2.113e-06 | 1.705e-06 / 1.738e-06 |

## Signed per-band attribution
Exact entries are the unnormalized Eq. (26) fine-grid band norms. Signed errors are (approximate band - exact band) / exact TOTAL HH, so bands add to the total relative error. Pairwise here is the unchanged default (5, sigma_pair, block). Per-setting band values and likelihood errors for all 96 points are in pairwise_tuning_iter006_full.json.
| PSD | iota | band | exact HH band | pairwise signed rel | IFFT-FFT signed rel |
|---|---:|---:|---:|---:|---:|
| analytic_lines | 0.3 | 0 | 2.4546660135e+02 | -1.307e-15 | +6.761e-17 |
| analytic_lines | 0.3 | 1 | 3.8057652437e+02 | -2.770e-07 | -9.750e-08 |
| analytic_lines | 0.3 | 2 | 6.3503595051e+02 | -2.826e-08 | -2.910e-08 |
| analytic_lines | 1.0 | 0 | 9.8929987276e+01 | -1.229e-15 | +5.462e-17 |
| analytic_lines | 1.0 | 1 | 1.5323305447e+02 | -2.956e-07 | -1.241e-07 |
| analytic_lines | 1.0 | 2 | 2.6823044068e+02 | -1.909e-08 | -2.047e-08 |
| analytic_lines | 1.4 | 0 | 4.4752052460e+01 | -4.210e-16 | +6.014e-17 |
| analytic_lines | 1.4 | 1 | 6.8733059691e+01 | -2.966e-07 | -1.376e-07 |
| analytic_lines | 1.4 | 2 | 1.2280343544e+02 | -2.613e-09 | -3.802e-09 |
| L1_real | 0.3 | 0 | 1.5942587972e+02 | -3.328e-15 | +5.839e-17 |
| L1_real | 0.3 | 1 | 1.5918183243e+02 | -3.636e-07 | -1.740e-07 |
| L1_real | 0.3 | 2 | 1.6814747924e+02 | -4.692e-08 | +1.080e-08 |
| L1_real | 1.0 | 0 | 6.4217281590e+01 | -2.867e-16 | +7.168e-17 |
| L1_real | 1.0 | 1 | 6.3982572779e+01 | -3.982e-07 | -2.149e-07 |
| L1_real | 1.0 | 2 | 7.0057884009e+01 | -2.981e-08 | +4.004e-08 |
| L1_real | 1.4 | 0 | 2.9053973302e+01 | -6.349e-16 | +1.191e-16 |
| L1_real | 1.4 | 1 | 2.8689372821e+01 | -4.019e-07 | -2.479e-07 |
| L1_real | 1.4 | 2 | 3.1782054832e+01 | -1.148e-08 | +5.027e-08 |

## Default-block class lag audit (iota=1)
Class 0 sums |sum_same_m h_lm|^2. Other classes are oriented slow-m conjugate times fast-m, with factor 2 included in folding errors. Outside fraction uses |G|; centroids use |G|^2. Near-origin alias is the portion folding to |lag|<=0.1 s. All inclinations and individual mode-pair audits are in the full JSON.
| PSD | band | class | block s | outside fraction | centroid s | signed rel folding | near-origin alias rel |
|---|---:|---|---|---:|---:|---:|---:|
| analytic_lines | 1 | 0 | [-4.0000,4.0000) | 1.686e-04 | -0.00000 | -2.458e-07 | -2.382e-07 |
| analytic_lines | 1 | (2, 1) | [-3.3359,4.6641) | 9.847e-05 | 0.30640 | -5.202e-09 | -5.695e-09 |
| analytic_lines | 1 | (3, 1) | [-2.2383,5.7617) | 5.529e-05 | 1.09751 | -6.355e-09 | -6.191e-09 |
| analytic_lines | 1 | (3, 2) | [-2.6992,5.3008) | 6.478e-05 | 0.80515 | -7.993e-08 | -7.742e-08 |
| analytic_lines | 1 | (4, 1) | [-0.4805,7.5195) | 4.158e-05 | 2.49360 | +5.584e-10 | +7.510e-10 |
| analytic_lines | 1 | (4, 2) | [-0.9062,7.0938) | 4.234e-05 | 2.21719 | +3.946e-08 | +3.897e-08 |
| analytic_lines | 1 | (4, 3) | [-1.9141,6.0859) | 4.425e-05 | 1.38597 | +1.703e-09 | +2.111e-09 |
| analytic_lines | 2 | 0 | [-2.0000,2.0000) | 1.432e-04 | -0.00000 | -1.780e-08 | -1.702e-08 |
| analytic_lines | 2 | (2, 1) | [-1.7268,2.2732) | 1.012e-04 | 0.05083 | +2.309e-08 | +2.314e-08 |
| analytic_lines | 2 | (3, 1) | [-1.3684,2.6316) | 5.060e-05 | 0.18954 | -2.911e-10 | -2.859e-10 |
| analytic_lines | 2 | (3, 2) | [-1.5132,2.4868) | 6.780e-05 | 0.17873 | -5.083e-08 | -5.040e-08 |
| analytic_lines | 2 | (4, 1) | [-0.8416,3.1584) | 2.956e-05 | 0.39662 | -6.124e-10 | -6.041e-10 |
| analytic_lines | 2 | (4, 2) | [-0.9666,3.0334) | 3.705e-05 | 0.48692 | +2.036e-08 | +2.017e-08 |
| analytic_lines | 2 | (4, 3) | [-1.2683,2.7317) | 3.903e-05 | 0.28462 | +6.993e-09 | +7.026e-09 |
| L1_real | 1 | 0 | [-4.0000,4.0000) | 1.686e-04 | -0.00000 | -3.223e-07 | -2.738e-07 |
| L1_real | 1 | (2, 1) | [-3.3359,4.6641) | 9.847e-05 | 0.30640 | -7.256e-09 | -8.047e-09 |
| L1_real | 1 | (3, 1) | [-2.2383,5.7617) | 5.529e-05 | 1.09751 | -8.430e-09 | -7.580e-09 |
| L1_real | 1 | (3, 2) | [-2.6992,5.3008) | 6.478e-05 | 0.80515 | -1.190e-07 | -9.580e-08 |
| L1_real | 1 | (4, 1) | [-0.4805,7.5195) | 4.158e-05 | 2.49360 | +1.703e-09 | +1.842e-09 |
| L1_real | 1 | (4, 2) | [-0.9062,7.0938) | 4.234e-05 | 2.21719 | +5.419e-08 | +4.986e-08 |
| L1_real | 1 | (4, 3) | [-1.9141,6.0859) | 4.425e-05 | 1.38597 | +2.820e-09 | +3.639e-09 |
| L1_real | 2 | 0 | [-2.0000,2.0000) | 1.432e-04 | -0.00000 | -4.184e-08 | -2.353e-08 |
| L1_real | 2 | (2, 1) | [-1.7268,2.2732) | 1.012e-04 | 0.05083 | +2.284e-08 | +2.135e-08 |
| L1_real | 2 | (3, 1) | [-1.3684,2.6316) | 5.060e-05 | 0.18954 | -1.183e-09 | -1.759e-10 |
| L1_real | 2 | (3, 2) | [-1.5132,2.4868) | 6.780e-05 | 0.17873 | -3.708e-08 | -4.519e-08 |
| L1_real | 2 | (4, 1) | [-0.8416,3.1584) | 2.956e-05 | 0.39662 | -5.226e-10 | -6.274e-10 |
| L1_real | 2 | (4, 2) | [-0.9666,3.0334) | 3.705e-05 | 0.48692 | +2.043e-08 | +1.953e-08 |
| L1_real | 2 | (4, 3) | [-1.2683,2.7317) | 3.903e-05 | 0.28462 | +7.545e-09 | +6.438e-09 |

## Setting decision and diagnosis
Best scanned setting: L_margin=3, sigma_mmax, diag=block; median 3.591e-07, worst 4.215e-07. No setting reaches the IFFT-FFT median. No code default changed.
The scanned block-diagonal medians span only 3.591e-7–3.681e-7 (best still 2.33x IFFT-FFT); moving the off-diagonal blocks cannot remove the dominant, fixed centred class-0 error. Linear class-0 weights worsen the median to about 1.88e-6. Retain the current default.
Band 1 (T_b=8 s), rather than the top band, dominates: at iota=1 pairwise signed errors are -2.956e-7 / -3.982e-7 (analytic+lines / L1), versus top-band -1.909e-8 / -2.981e-8. In band 1 class 0 contributes -2.458e-7 / -3.223e-7; its near-origin alias portion is -2.382e-7 / -2.738e-7. Thus tails outside the centred [-4,4) s block, especially near +/-T_b folding to A_0, set the main floor. This is period-T_b folding of finite-window power tails, even in the dominant mode.
The elementary |h_22|^2 band-1 contribution is -2.340e-7 / -3.072e-7. The next largest lag class is (m_slow,m_fast)=(3,2), -7.993e-8 / -1.190e-7, almost entirely the (3,3)x(2,2) pair. This lag-class label does not denote the waveform (3,2) mode: (2,2)x(3,2) contributes only +4.528e-9 / +5.819e-9. Mixed-32 content and top-band merger-ringdown are secondary for this floor. The smooth-PSD control remains -3.134e-7; lines change weighting but do not create the failure.
Maximum exact folding-identity residual / total HH: 1.953e-16. This independently checks the attribution against the actual quadrature errors.
Smooth-PSD control at iota=1: -3.134e-07.
Requested negative slack is clipped by the existing implementation. Band 0 has M=N and is exact regardless of its unwrapped slack. Requested/effective margins and blocks are recorded in JSON. LAL refused parameter sets: none.
Cache hits/misses: 6/0.
