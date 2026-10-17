# Iteration 006: lean XHM timings

Median warm wall time, seven batches targeting 0.2 s each, interleaved in seeded shuffled component order to reduce clock drift; Python on PATH, LALSimulation 6.2.0, this machine. All times in ms. OMP_NUM_THREADS=1, OMP_WAIT_POLICY=PASSIVE; default L=5, nominal testcases.py parameters and analytic+lines PSD. K_MB=20167 / 4865 / 4408 (BNS / GW190814 / BBH). Preparation and waveform samples are outside HH timings. No waveform results are cached between calls.

| Component | BNS | GW190814-like | BBH 60+20 |
|---|---:|---:|---:|
| raw polarizations | 5.8808 | 2.3178 | 2.0441 |
| raw five OneMode | 6.8242 | 2.8974 | 2.0654 |
| LAL tail recovery | 0.0003 | 0.0003 | 2.4762 |
| coarse_total | 6.4562 | 2.4195 | 2.2092 |
| coarse_modes | 8.0316 | 3.2657 | 5.0649 |
| phase vector | 0.3739 | 0.0815 | 0.0742 |
| IFFT-FFT HH | 3.3017 | 0.6198 | 0.5660 |
| pairwise HH | 0.5819 | 0.1537 | 0.1546 |
| B1 HH | 0.4672 | 0.1287 | 0.1407 |
| B0 HH | 0.0608 | 0.0182 | 0.0170 |
| linear HH | 0.0586 | 0.0165 | 0.0153 |

## 1 detector routes
Measured complete adapter + HH calls; other likelihood work is excluded.
| Route | BNS ms / ratio / Python % | GW190814 ms / ratio / Python % | BBH ms / ratio / Python % |
|---|---:|---:|---:|
| IFFT-FFT | 10.4982 / 1.000 / 5.6% | 3.1045 / 1.000 / 5.5% | 2.8560 / 1.000 / 5.1% |
| pairwise | 8.5428 / 0.814 / 11.6% | 3.5041 / 1.129 / 8.0% | 5.1933 / 1.818 / 6.3% |
| B1 | 7.2710 / 0.693 / 8.1% | 2.6311 / 0.847 / 6.5% | 2.4095 / 0.844 / 6.1% |
| B0 | 6.7318 / 0.641 / 8.7% | 2.4854 / 0.801 / 6.9% | 2.2616 / 0.792 / 6.5% |

## 3 detector routes
Waveform evaluated once + HH step three times: sums of component medians. Uses identical PSD/kernel costs and assumes negligible additional detector projection/assembly cost; this is not a measured network evaluation.
| Route | BNS ms / ratio / Python % | GW190814 ms / ratio / Python % | BBH ms / ratio / Python % |
|---|---:|---:|---:|
| IFFT-FFT | 16.3613 / 1.000 / 3.6% | 4.2789 / 1.000 / 4.0% | 3.9073 / 1.000 / 3.7% |
| pairwise | 9.7772 / 0.598 / 10.2% | 3.7268 / 0.871 / 7.6% | 5.5286 / 1.415 / 5.9% |
| B1 | 7.8578 / 0.480 / 7.5% | 2.8058 / 0.656 / 6.1% | 2.6312 / 0.673 / 5.6% |
| B0 | 6.6386 / 0.406 / 8.9% | 2.4741 / 0.578 / 6.9% | 2.2602 / 0.578 / 6.5% |

Python fractions use paired wall-clock accounting in 100 warm adapter calls: time each LAL wrapper region and the enclosing adapter, then subtract within each call and take the median. BBH tail recovery is included in LAL regions. The wrapper regions include negligible dispatch/bookkeeping; fractions are estimates. The phase-vector cost is also measured directly; it is included in adapter time, not added twice. Component medians and complete one-detector routes are measured separately from this instrumentation. JSON also records each warm batch and component-sum route estimates; one-detector ratios use the measured complete routes, three-detector ratios use the sums.
Five OneMode calls suffice for BNS and GW190814. BBH needs four additional polarization calls to preserve iteration-005 high-frequency tails; the raw five-call row excludes them, the coarse_modes and route rows include them. No tail correction was dropped for speed.
Ratios below 1 are faster. Accuracy qualifications remain: B0 loses accuracy for HOM; B1 discrete interpolants have system-dependent errors (see b1_variants_iter006.md). Pairwise retains its extrinsic-matrix reuse capability. These fixed cases do not establish population criteria C1/C1'.
Threading control: with the initial OpenMP default of 4 threads and unset wait policy, direct IFFT-FFT routes measured 16.32 / 10.12 / 10.11 ms (BNS / GW190814 / BBH), well above component sums 11.19 / 3.49 / 3.62 ms. Setting one waveform thread removed most of this interaction in a paired control. The primary table uses the explicit thread settings above; it should not be compared to iteration-005 raw timings as an adapter-only speed-up. The initial control batches are retained in xhm_timing_iter006_threads_full.json (git-ignored). All changes to threading are confined to the new benchmark scripts.
LAL refused parameter sets: none.
