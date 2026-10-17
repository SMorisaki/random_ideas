# Iteration 005: XHM fixed-case accuracy

LALSimulation 6.2.0; 22 cases (3 inclinations per system/PSD plus two q=0.1 edge variants per PSD). All references are total_direct on the full PSDGrid.f_k sequence, cached by model and PSD. Every partition-of-unity check passed before coarse evaluation. SNR=25: |Delta ln Lambda|=312.5 relative error.

Analytic aLIGO + TOY_LINES; real L1 Welch resolution 1/16 Hz interpolated in log PSD/log f. These are fixed cases, not the iteration-006 population. The q=0.1 variants keep m1 fixed; their chirp masses (5.67 and 14.79 Msun) lie below the nominal priors.

| System | Method | n | median relative | worst relative | median abs Delta ln Lambda | worst abs Delta ln Lambda |
|---|---|---:|---:|---:|---:|---:|
| BNS | linear | 6 | 1.268e-07 | 2.526e-07 | 3.964e-05 | 7.894e-05 |
| BNS | IFFT-FFT | 6 | 3.489e-08 | 4.906e-08 | 1.090e-05 | 1.533e-05 |
| BNS | pairwise | 6 | 1.035e-08 | 1.159e-08 | 3.233e-06 | 3.622e-06 |
| BNS | B0 | 6 | 1.251e-07 | 4.365e-07 | 3.910e-05 | 1.364e-04 |
| BNS | B1 | 6 | 3.556e-07 | 5.241e-07 | 1.111e-04 | 1.638e-04 |
| GW190814 | linear | 8 | 1.288e-05 | 1.140e-04 | 4.024e-03 | 3.562e-02 |
| GW190814 | IFFT-FFT | 8 | 1.642e-07 | 3.346e-07 | 5.130e-05 | 1.045e-04 |
| GW190814 | pairwise | 8 | 3.553e-07 | 4.281e-07 | 1.110e-04 | 1.338e-04 |
| GW190814 | B0 | 8 | 3.346e-07 | 1.776e-06 | 1.046e-04 | 5.550e-04 |
| GW190814 | B1 | 8 | 2.825e-07 | 5.199e-07 | 8.827e-05 | 1.625e-04 |
| BBH | linear | 8 | 1.393e-05 | 1.255e-04 | 4.354e-03 | 3.922e-02 |
| BBH | IFFT-FFT | 8 | 1.051e-07 | 5.424e-06 | 3.285e-05 | 1.695e-03 |
| BBH | pairwise | 8 | 2.031e-08 | 2.702e-05 | 6.346e-06 | 8.443e-03 |
| BBH | B0 | 8 | 3.771e-08 | 6.230e-05 | 1.178e-05 | 1.947e-02 |
| BBH | B1 | 8 | 6.012e-07 | 2.693e-05 | 1.879e-04 | 8.415e-03 |

## Per-PSD summaries (median / worst relative; parentheses are SNR25 likelihood errors)
| System | PSD | Method | median relative | worst relative | median likelihood | worst likelihood |
|---|---|---|---:|---:|---:|---:|
| BNS | analytic_lines | linear | 1.005e-07 | 1.532e-07 | 3.140e-05 | 4.788e-05 |
| BNS | analytic_lines | IFFT-FFT | 2.458e-08 | 2.661e-08 | 7.680e-06 | 8.317e-06 |
| BNS | analytic_lines | pairwise | 8.935e-09 | 1.136e-08 | 2.792e-06 | 3.551e-06 |
| BNS | analytic_lines | B0 | 1.071e-07 | 1.432e-07 | 3.346e-05 | 4.474e-05 |
| BNS | analytic_lines | B1 | 2.123e-07 | 2.134e-07 | 6.636e-05 | 6.670e-05 |
| BNS | L1_real | linear | 1.981e-07 | 2.526e-07 | 6.191e-05 | 7.894e-05 |
| BNS | L1_real | IFFT-FFT | 4.797e-08 | 4.906e-08 | 1.499e-05 | 1.533e-05 |
| BNS | L1_real | pairwise | 1.056e-08 | 1.159e-08 | 3.299e-06 | 3.622e-06 |
| BNS | L1_real | B0 | 3.109e-07 | 4.365e-07 | 9.715e-05 | 1.364e-04 |
| BNS | L1_real | B1 | 5.061e-07 | 5.241e-07 | 1.581e-04 | 1.638e-04 |
| GW190814 | analytic_lines | linear | 9.789e-06 | 1.558e-05 | 3.059e-03 | 4.869e-03 |
| GW190814 | analytic_lines | IFFT-FFT | 1.430e-07 | 1.651e-07 | 4.469e-05 | 5.160e-05 |
| GW190814 | analytic_lines | pairwise | 3.023e-07 | 3.147e-07 | 9.446e-05 | 9.833e-05 |
| GW190814 | analytic_lines | B0 | 2.747e-07 | 3.012e-07 | 8.583e-05 | 9.412e-05 |
| GW190814 | analytic_lines | B1 | 2.152e-07 | 2.262e-07 | 6.726e-05 | 7.068e-05 |
| GW190814 | L1_real | linear | 4.656e-05 | 1.140e-04 | 1.455e-02 | 3.562e-02 |
| GW190814 | L1_real | IFFT-FFT | 1.862e-07 | 3.346e-07 | 5.819e-05 | 1.045e-04 |
| GW190814 | L1_real | pairwise | 4.119e-07 | 4.281e-07 | 1.287e-04 | 1.338e-04 |
| GW190814 | L1_real | B0 | 8.656e-07 | 1.776e-06 | 2.705e-04 | 5.550e-04 |
| GW190814 | L1_real | B1 | 4.894e-07 | 5.199e-07 | 1.529e-04 | 1.625e-04 |
| BBH | analytic_lines | linear | 9.378e-06 | 1.839e-05 | 2.930e-03 | 5.746e-03 |
| BBH | analytic_lines | IFFT-FFT | 1.409e-08 | 4.703e-06 | 4.402e-06 | 1.470e-03 |
| BBH | analytic_lines | pairwise | 1.999e-08 | 1.556e-05 | 6.247e-06 | 4.862e-03 |
| BBH | analytic_lines | B0 | 1.216e-08 | 6.728e-06 | 3.799e-06 | 2.103e-03 |
| BBH | analytic_lines | B1 | 4.226e-07 | 8.422e-06 | 1.321e-04 | 2.632e-03 |
| BBH | L1_real | linear | 1.715e-05 | 1.255e-04 | 5.359e-03 | 3.922e-02 |
| BBH | L1_real | IFFT-FFT | 1.687e-07 | 5.424e-06 | 5.273e-05 | 1.695e-03 |
| BBH | L1_real | pairwise | 2.878e-08 | 2.702e-05 | 8.993e-06 | 8.443e-03 |
| BBH | L1_real | B0 | 7.543e-08 | 6.230e-05 | 2.357e-05 | 1.947e-02 |
| BBH | L1_real | B1 | 6.258e-07 | 2.693e-05 | 1.956e-04 | 8.415e-03 |

## Prior slack
Band 0 is a complete lag grid; its negative unwrapped slack is harmless.
| System | K_MB | min slack, all (s) | min slack, truncated (s) |
|---|---:|---:|---:|
| BNS | 20167 | -796.418462 | 1.992602 |
| GW190814 | 4865 | -66.480113 | 1.116975 |
| BBH | 4408 | 0.858869 | 0.858869 |

## Real-XHM lag audit: (4,4) x (2,1), iota=1, nominal masses
Outside fraction is sum outside |G| / sum |G|, not |G|^2. The exact folding error includes 4/T.
| PSD | band | predicted interval (s) | block (s) | outside fraction | lag centroid (s) | Re folding HH |
|---|---:|---|---|---:|---:|---:|
| analytic_lines | 1 | [0.861708312144646, 4.6021333254022645] | [-0.48046875, 7.51953125] | 4.157917e-05 | 2.493604 | 1.452894e-07 |
| analytic_lines | 2 | [0.0012845304473350675, 1.3231145825884791] | [-0.841552734375, 3.158447265625] | 2.955830e-05 | 0.396623 | -1.593392e-07 |
| L1_real | 1 | [0.861708312144646, 4.6021333254022645] | [-0.48046875, 7.51953125] | 4.157917e-05 | 2.493604 | 1.687880e-07 |
| L1_real | 2 | [0.0012845304473350675, 1.3231145825884791] | [-0.841552734375, 3.158447265625] | 2.955830e-05 | 0.396623 | -5.180431e-08 |

## C5 and LAL convention findings
XHM negative-m modes use AddMode at azimuth pi/2, with exp(+i |m| phi_ref). Raw modes are normalized to 1 Mpc and this nominal phase is removed; coefficients restore phase and distance. The mixed (3,2) series retains a small numerical phi_ref dependence in LAL 6.2.0. Consequently the strict requirement that matrices be identical to roundoff under phi_ref is unmet. Other extrinsics leave raw matrices identical. This is retained to preserve the direct-waveform identity.
OneMode truncates HM tails at Mf=0.3; on BBH the adapter recovers them using single-mode ChooseFD calls (32 also activates 22 for consistent mixing). No such fallback is needed for timing systems BNS/GW190814.
```json
{
  "other_extrinsics_max_scaled": 0.0,
  "including_phi_ref_max_scaled": 6.546062211871719e-13,
  "factorization_relative": 5.773159728050814e-15,
  "pairwise_tc_relative_change": 0.0,
  "ifftfft_tc_relative_change": 4.553869503709507e-09,
  "GW190814_raw_phase_residuals": {
    "(2, 2)": 7.152041869731632e-14,
    "(2, 1)": 4.55037220131109e-14,
    "(3, 3)": 9.060512530560136e-14,
    "(3, 2)": 1.3319371783445315e-05,
    "(4, 4)": 2.58542983600813e-13
  }
}
```
Refused parameter sets: []
Full per-case errors are in xhm_accuracy_iter005_full.json (git-ignored). No mode pairs were dropped.
