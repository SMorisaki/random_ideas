# PROGRESS.md — multiband_quad

Iteration log for the Codex-driven development loop (see `TASK.md`). Newest entry last.

## Session 2026-10-17 (cloud): setup

- Branch `LLM`, working tree clean, `git pull --ff-only origin LLM` up to date. Push access verified
  (`git push --dry-run` and the session's repository access check: push ok).
- Codex 0.163.0 authenticated with ChatGPT via `sh .devcontainer/run_codex.sh login --device-auth`
  (device code approved by the owner in a browser); `run_codex.sh login status` -> "Logged in using ChatGPT".
  `.devcontainer/codex-check.sh` run by hand: sandbox probe OK -> `~/.codex/sandbox_mode = workspace-write`,
  so Codex is run with `--sandbox workspace-write` (not disabled).
- Conda env rebuilt with `bash dist_psi_marg/tools/cloud_setup.sh` at `/home/user/conda-env`
  (python 3.12.15, numpy 2.5.4, scipy 1.18.1, numba 0.68.0). The orchestrator added
  `python-lalsimulation` 6.2.0 (`micromamba install -c conda-forge python-lalsimulation`) for
  IMRPhenomXHM mode-by-mode waveforms on arbitrary frequency sequences
  (`SimIMRPhenomXHMFrequencySequenceOneMode`: 1.5-8 ms per mode at 1e4 points on this machine;
  full IMRPhenomXHM `ChooseFDWaveformSequence`: 12 ms) and updated `environment.yml` in the same commit.
- `/projects/data_release` is not mounted in the cloud container; experiments use analytic PSDs and
  synthetic signals. No root `CLAUDE.md` exists in the repository.
- The paper (arXiv:2104.07813) was read; its equations are restated in `notes/paper_summary.md` and the
  candidate method is derived in `notes/method_idea.md`. A design review (three adversarial reviewers,
  three independent proposers, one judge) was run before drafting `TASK.md`; its conclusions are folded
  into `TASK.md` and `notes/method_idea.md`.

Owner decisions (2026-10-17, via the question card): accuracy/speed targets as proposed; IMRPhenomXHM
non-precessing as the main model (precession is a stretch item); real PSDs may be downloaded from public
sources (GWOSC / LIGO DCC / zenodo reachable, HTTP 200) into `data/` (not committed); **additional requirement:
also provide a method that works when only `h_+`, `h_x` are available (no mode decomposition)** -> recorded as
method B (baseband IFFT-FFT with half-sample shift and adaptive block length) in `TASK.md` §1-2 and
`notes/method_idea.md` §7, verified on the toy (`notes/orchestrator_toy_methodB.py`).

## Iteration 001 — package skeleton (PASS)

Codex (gpt-6.1-sol, reasoning high, `--sandbox workspace-write`, ~8 min) created `mbquad/` (`psd.py`,
`toywave.py`, `bands.py`, `reference.py`, `testcases.py`, `_validation.py`), `tests/` (65 tests),
`scripts/band_table.py`, `pytest.ini` (`--basetemp=data/pytest_tmp`, git-ignored) and `data/band_tables.md`.

Orchestrator verification:
- `python -m pytest tests -q`: 65 passed in 0.2 s (conda env python).
- Table I of the paper reproduced by the band construction (Eqs. 47, 51, 53, 55, 56) for the BNS setting
  (T = 256 s, 20-2048 Hz, durations 256..4 s, mchirp_ref = 1.15, L = 5): `K_orig/K_MB` = 43.3 (paper 45,
  m_max = 2) and 25.7 (paper 27, m_max = 4); `K_orig = 519169`. The toy geometry matches the orchestrator's
  toy script exactly (`K_orig = 30209`, `K_MB = 3294`).
- Code read in full: clean, vectorised, docstrings cite the equations; windows, N_b, M_b, K_s, K_e as in the
  paper; `N_b` is allowed to exceed `N` when the upper taper `f_high + Df^(B)` is above Nyquist (BNS with
  f_high = 2048 Hz), which follows Eq. 13 literally.
- Predicted largest lag (4,4)-(2,1) per band confirms the design assumption: in every band with b >= 1 it lies
  between `T_b/2` and `T_b` for the long bands (BNS b = 1..5: 113/128, 54/64, 26/32, 11.5/16, 4.7/8 s) and below
  `T_b/2` only in the shortest bands (BNS b = 6: 1.4/4 s; GW190814-like b = 2: 1.2/4 s; BBH b = 1: 1.1/4 s).

Next: iteration 002 (faithful re-implementation of the paper's linear-interpolation and IFFT-FFT methods with
a timing harness).

Real PSDs prepared by the orchestrator (owner approved downloads): GWOSC 4 kHz, 4096 s strain around GW190814
(GPS 1249852257; files `{H,L}-{H1,L1}_GWOSC_4KHZ_R1-1249850209-4096.hdf5`, 129 MB and 99 MB, kept outside the
repository) -> Welch median PSDs with 16 s Tukey segments (alpha = 0.1, 50% overlap, event +-32 s excluded,
median/ln 2 bias correction; L1: 378 segments, H1: 502) saved as `data/psd_{L1,H1}_GW190814_welch16s.txt`
(0.87 MB each, committed as derived products) with the script `notes/make_real_psd.py`. Strongest lines
relative to a 16 Hz running median: L1 at 306-315, 506-516, 1012-1018 Hz (up to 1.4e4x); H1 at 20.1, 22.4,
300-303, 502-517 (up to 3e5x), 992-1011 Hz (up to 4e6x).

## Iteration 002 — the paper's linear-interpolation and IFFT-FFT methods (PASS)

Codex (~12 min) added `mbquad/coarse.py` (`CoarseWaveform`), `mbquad/linear.py` (`LinearInterpolationHH`, Eqs. 27-30 with
the tilde-f ranges / true-f weights convention), `mbquad/ifftfft.py` (`IFFTFFTHH`, Eqs. 31-46, `impl="paper"` complex
FFTs and `impl="fast"` real FFTs with reusable buffers, `half_point_waveform`), `mbquad/timing.py`, two scripts and
`tests/test_paper_methods.py`.

Orchestrator verification: 75 tests pass in 0.5 s; code read in full. Accuracy on `toy_small()` (relative `(h,h)`
error / `|Delta ln Lambda|` at SNR 25): linear 1.40e-6 / 4.4e-4 (quadrupole) and 1.57e-6 / 4.9e-4 (five modes);
IFFT-FFT 1.54e-7 / 4.8e-5 and 2.22e-7 / 6.9e-5; identical to the orchestrator's toy to all printed digits. With the
smooth PSD the five-mode linear error grows to 4.8e-6 (the Fejer-taper attenuation of cross terms, not the lines).
`(h,h)` step timings (median, this machine): linear 0.019 ms, IFFT-FFT paper 1.41 ms, IFFT-FFT fast (rfft) 0.70 ms,
toy waveform 0.87 ms (`toy_small`, K_MB = 3294); GW190814-like geometry: 1.29 / 0.70 ms with waveform 1.18 ms.
The "fast" implementation is the fair baseline for the new methods.

## Design review (workflow: 3 adversarial reviewers, 3 proposers, 1 judge) — partial results folded in

Two reviewers (signal processing, waveform physics) finished; both verdicts "sound with caveats". Their corrections
are recorded in `notes/method_idea.md` §8 and `TASK.md` §2: lag sign convention (positive lags for the slower mode
first), window on the pair product (not in the weights), lag interval `[Lambda(f^(b+1)), Lambda(f_lo)]` with
`Lambda_min ~ 0.5 Lambda_max` (7 lag classes per band), margins from the pair chirp rate `sigma_AA'`, prior coverage of
the anchors (`Mc_min`, `Mc_max`), the exact folding-error formula, the cheap total-waveform variant B0 (centred block
on the total `|h|^2`), and the speed-up ceiling (~1.3x overall from Table I; per-mode XHM calls cost more than one
polarization call) -> timing breakdown and go/no-go for method A in iteration 005. Test BBH changed to 60+20 Msun,
T = 16 s. Remaining reviewer/proposer/judge outputs will be folded in when they finish.

Next: iteration 003 (method A with lag classes, pair margins, prior coverage, lag audit).

## Iteration 003 — method A: pairwise block weights with lag classes (PASS)

Codex (~14 min) added `mbquad/blockweights.py` (`BlockWeights`: cached `A_m` per `N_b`, `weights(b, m0, Nhat)`,
`centred`, `for_lag_interval`, `lag_audit` with the exact folding-error formula), `mbquad/pairwise.py` (`PairwiseHH`:
7 lag classes per band, blocks centred on `[Lambda_min - L sigma_pair(f_hi), Lambda_max + L sigma_pair(f_lo)]` with the
prior corners `Mc_max`/`Mc_min`, slack bookkeeping and clipping with warnings, `einsum` and `pairs` implementations,
`diag="block"|"linear"`), `CoarseModes`, `BandSetup.lag_pair/lag_interval_pair`, `IFFTFFTHH.positive_grid_waveform`,
two scripts, two test files and two figures. 85 tests pass (0.8 s).

Orchestrator verification (code read in full, tables and figures inspected):
- Accuracy on `toy_small()` (relative `(h,h)` error): pairwise block 2.21e-7 (quadrupole) and 3.53e-7 (five modes,
  lines) vs IFFT-FFT 1.54e-7 / 2.22e-7 and linear 1.40e-6 / 1.57e-6; smooth PSD five modes 3.54e-7 vs 2.94e-7 / 4.79e-6.
  Deliberately wrong placements: all blocks centred on lag 0 -> 4.40e-6 (lines), mirrored to negative lags -> 1.35e-4,
  which pins the sign convention (positive lags for the slower mode first).
- Margin scan: error flat at 2.9-3.0e-7 for 0.5-3 pair-sigma, 3.5e-7 at 5, clipping needed at 8 in the toy; default
  kept at 5 sigma_pair (conservative). The residual gap to IFFT-FFT (period `T_b` vs `2 T_b` folding) is 1.5x.
- Lag audit (`figures/iter003_lag_audit.png`): |G|^2 of each ordered pair sits inside the predicted SPA interval with
  `lag^-3`-type tails; class blocks contain it with large slack, centred blocks cut the bulk of (4,4)x(2,1) and
  (4,4)x(2,2) in bands 1, 3 and 5. Class-block out-of-block fraction 1.5e-5 vs 0.69 for the zero-anchored block.
- Prior-coverage slack (`data/slack_iter003.md`): minimum requested slack in truncated bands 1.84 s (toy), 1.99 s (BNS,
  Mc in (1.15, 1.25)), 1.12 s (GW190814-like, Mc in (6, 7)), 0.86 s (BBH 60+20, T = 16 s, Mc in (25, 35)); band 0 has
  `M_b = N_b` and is exact for any placement.
- Timing of the `(h,h)` step (numpy): pairwise `pairs` 0.22 ms (toy) / 0.16 ms (GW190814-like geometry) vs IFFT-FFT fast
  0.71 / 0.68 ms: 3.2x / 4.3x faster; the C2 target (5x) is deferred to the numba implementation of iteration 008.

Next: iteration 004 (methods B0 and B1 for the total waveform).

## Iteration 004 — methods B0 and B1 for the total waveform (PASS, with documented deviations from the prompt)

Codex (~16 min) added `mbquad/totalweights.py` (`CentredBlockHH`, B0), `mbquad/baseband.py` (`BasebandIFFTFFTHH`, B1 with the
adaptive rule, `next_fast_len`, half-sample shift), `tests/test_total_methods.py`, `scripts/accuracy_iter004.py`, the
extended benchmark, and `skip_dc` in `load_asd_or_psd` (the Welch files contain f = 0). 102 tests pass (1.4 s).

Orchestrator verification (code read in full; tables inspected):
- B1 (relative `(h,h)` error): toy lines five modes 4.40e-7 (IFFT-FFT 2.22e-7, pairwise 3.53e-7), strong-HOM toy 2.73e-7
  (1.44e-7), GW190814-like geometry with the real L1 PSD 6.73e-7 (IFFT-FFT 5.99e-7, pairwise 5.80e-7, linear 2.95e-5).
  B1 is within 2x of the IFFT-FFT everywhere (within 1.1x on the real PSD) but is NOT identical to it: the minimal-K'
  periodic baseband interpolant and the paper's full-spectrum real-signal interpolant treat the Eq. 36 tails differently
  (my prompt's expectation of 1e-8 agreement was wrong; Codex documented the correct identity
  `Re[W(k) + W(-k)] = (T/That) I_tilde(k)`, verified to 7e-16).
- B0: quadrupole 2.21e-7 (a drop-in improvement over linear interpolation's 1.40e-6 at identical cost), but with HOM and
  narrow lines ~20x worse than IFFT-FFT (toy 4.40e-6, strong-HOM 8.94e-6, real L1 4.77e-6; 6.9x better on a 16 Hz
  running-median-smoothed L1 PSD) -> B0 is for quadrupole-only or smooth-PSD use; B1 is the total-only default for HOM.
  The 5 sigma adaptive rule selects no B0 band in the five-mode geometries (top band 2.12 s > T_b/2 = 2 s).
- FFT lengths: toy bands 1-5 (594, 432, 350, 360, 825) vs the paper's (8192/16384, ..., 4096/8192), K'/M = 0.07-0.20;
  GW190814-like (315, 3960) vs (2048/4096, 16384/32768). `(h,h)` step: B1 0.131 / 0.144 ms vs IFFT-FFT fast 0.681 / 0.684 ms
  (5.2x / 4.7x; C2 for method B met), B0 0.020 ms, pairwise 0.194 / 0.162 ms.

## Design-review workflow: final status

The three adversarial reviewers completed (their findings are in `notes/method_idea.md` §8 and were acted on in
iterations 003-004). The three proposers and the judge failed with a model usage-limit error ("Fable limit") after
94 minutes; they are not re-run (the orchestrator's own session is unaffected, and the reviews already covered the design;
no further workflow/subagent runs will be started in this session).

Next: iteration 005 (IMRPhenomXHM interface, real systems, mode matrix C5, timing breakdown / go-no-go).

## Iteration 005 — IMRPhenomXHM interface, real systems, mode matrix, timing breakdown (PASS; go/no-go data)

Codex (~25 min) added `mbquad/lalwave.py` (`XHMModel`: detector-frame modes from `SimIMRPhenomXHMFrequencySequenceOneMode`
with the negative-m series, AddMode coefficients at azimuth pi/2 and `exp(+i|m| phi_ref)`, raw modes normalised to 1 Mpc,
`total_direct` through `ChooseFDWaveformSequence`, recovery of the `Mf > 0.3` tails that `OneMode` truncates for the BBH via
mode-selected polarization calls), `PairwiseHH.mode_matrix` / `hh_from_matrix` (C5), `xhm_systems()` / `xhm_case()`, two
scripts, `tests/test_lalwave.py`. 110 tests pass. Mode sums reproduce `F_+ h_+ + F_x h_x` to 1e-10; all 22 partition-of-unity
checks pass; BNS fine-grid references (2^20 grid) computed through the sequence path and cached.

Orchestrator verification and reading of the results (`data/xhm_accuracy_iter005.md`, `data/xhm_timing_iter005.md`):
- Accuracy (relative `(h,h)` error, median over inclinations {0.3, 1.0, 1.4} and both PSDs; worst in parentheses):
  BNS: linear 1.3e-7, IFFT-FFT 3.5e-8, pairwise 1.0e-8 (1.2e-8), B0 1.3e-7, B1 3.6e-7 (5.2e-7).
  GW190814-like: linear 1.3e-5 (1.1e-4), IFFT-FFT 1.6e-7 (3.3e-7), pairwise 3.6e-7 (4.3e-7), B0 3.3e-7 (1.8e-6), B1 2.8e-7 (5.2e-7).
  BBH 60+20: linear 1.4e-5, IFFT-FFT 1.1e-7, pairwise 2.0e-8, B0 3.8e-8, B1 6.0e-7; the worst cases (2.7e-5 pairwise,
  5.4e-6 IFFT-FFT) are the q = 0.1 variants whose chirp masses (5.7, 14.8 Msun) lie BELOW the band-design reference
  (out of the prior the bands were built for; Eq. 51 itself is violated there), so they are excluded from the criteria.
  In `Delta ln Lambda` at SNR 25 all in-prior pairwise and B1 errors are <= 1.6e-4 (the paper's IFFT-FFT median at L = 5
  was 2e-4). Against C1 (pairwise median <= IFFT-FFT median): met for BNS (3x better) and BBH (5x better), NOT met for
  GW190814-like (2.2x worse in the median, 1.3x in the worst case). B1 is 1.7x (GW190814) to 10x (BNS) worse than IFFT-FFT.
- Diagnosis of the B1 gap (orchestrator): B1 and the paper's method compute the same half-point interpolant (a one-period
  Riemann sum of the periodised time series against `e^{-2 pi i f t}` with half-integer `f T_b`, whose integrand flips sign
  at the period boundary) with `K'` versus `M` samples per period; the endpoint error is first order in the sample spacing,
  so B1 loses by ~M/K' (5-14) where the boundary tails are not negligible. Expected cure (to test in iteration 006): the
  Euler-Maclaurin endpoint correction (`g(T_b) = -g(0)` makes the trapezoidal endpoint weight vanish: drop the `n = 0` term
  and, at the next order, correct with finite differences of `y` near `n = 0`) and/or zero-padding the baseband IFFT to
  `p K'` samples (error ~ 1/p, cost ~ p log p).
- Lag audit on real XHM ((4,4)x(2,1), GW190814-like): out-of-block fraction 4.2e-5 (band 1) and 3.0e-5 (top band, which
  contains merger-ringdown), lag centroids inside the predicted SPA intervals: the SPA interval design holds on IMR waveforms.
- C5: `hh_from_matrix(mode_matrix(raw), C)` equals the direct pairwise value to 5.8e-15; the raw-mode matrix is identical
  under changes of `t_c`, distance, inclination, `F_+`, `F_x`; under `phi_ref` it changes by 6.5e-13 (relative, max-scaled)
  because LAL's mixed (3,2) mode carries a 1.3e-5 phase residual beyond `e^{-i m phi}` (LAL detail, negligible).
  Pairwise `(h,h)` is independent of `t_c` to roundoff; the IFFT-FFT value changes by 4.6e-9 over the prior.
- Timing (warm medians, this machine; BNS K_MB = 20167, GW190814 K_MB = 4865): raw LAL XHM polarizations 5.9 / 2.1 ms;
  five `OneMode` calls 6.3 / 3.1 ms (per-mode generation is NOT penalised for the BNS, +45% for the short signal); HM
  all modes 45 / 16 ms. Adapter overhead (per-call sort/unique/unsort, copies, projection) 4.9 / 2.0 ms is an artefact of
  the current adapter (the coarse grid is fixed, so the sorted-unique mapping can be precomputed). `(h,h)` steps:
  IFFT-FFT fast 3.64 / 0.62 ms, pairwise pairs 0.54 / 0.15 ms, B1 0.42 / 0.13 ms, B0 0.056 / 0.018 ms, linear 0.059 / 0.016 ms.
  Route sums (waveform + `(h,h)`), adapter-inclusive: IFFT-FFT 10.9 / 3.25 ms; A 11.7 (1.08x) / 5.2 (1.60x); B1 7.7 (0.70x) /
  2.8 (0.85x); raw-LAL sums without the adapter: A 6.8 (0.72x) / 3.2 (1.17x), B1 6.3 (0.66x) / 2.2 (0.82x).

Go/no-go reading (to be put to the owner): with lalsimulation's IMRPhenomXHM, method A is not a practical speed-up for
short signals (per-mode generation costs more than the whole IFFT-FFT step) and at best ~0.7x of the IFFT-FFT route for the
BNS once the adapter is lean; its value is structural (no FFT, `t_c` independence, the extrinsic factorisation C5).
Method B1 already gives 0.70x (BNS) / 0.85x (GW190814) end-to-end and needs the accuracy fix above to match the IFFT-FFT.

Next (re-planned): iteration 006 = B1 endpoint correction and zero-padding study on the XHM systems, lean adapter
(precomputed grid mapping) with re-timed routes, pairwise margin/diag study on the GW190814-like system; iteration 007 =
population study; 008 = band rule / L-scan / prior grid; 009 = numba kernels; 010 = full-scale BNS; 011 = documents.

## Iteration 006 — lean adapter and re-timed routes, pairwise tuning, B1 variants, L = 50 (PASS)

Codex (~35 min) added `XHMCoarseEvaluator` (prepared sorted-unique LAL sequence, per-band slices, one-pass coefficients
and phase), `PairwiseHH.band_contributions`, B1 options `pad_factor` / `all_half_bins`, three scripts and
`tests/test_lean_adapter.py`. 119 tests pass. Defaults unchanged.

Orchestrator verification (tables inspected, numbers cross-checked with iteration 005):
- Timing with `OMP_NUM_THREADS=1` (the default 4 threads inflated the IFFT-FFT route from 10.5 to 16.3 ms; iteration-005
  route numbers were affected by this interaction too): raw polarizations 5.9 / 2.3 / 2.0 ms, five `OneMode` 6.8 / 2.9 /
  2.1 ms (BNS / GW190814-like / BBH), lean `coarse_total` 6.5 / 2.4 / 2.2 ms, lean `coarse_modes` 8.0 / 3.3 / 5.1 ms (BBH
  includes 2.5 ms of tail-recovery polarization calls). `(h,h)`: IFFT-FFT 3.30 / 0.62 / 0.57, pairwise 0.58 / 0.15 / 0.15,
  B1 0.47 / 0.13 / 0.14, B0 0.06 / 0.02 / 0.02 ms. Routes (ratio to IFFT-FFT), 1 detector: A 0.81 / 1.13 / 1.82, B1 0.69 /
  0.85 / 0.84, B0 0.64 / 0.80 / 0.79; 3 detectors: A 0.60 / 0.87 / 1.42, B1 0.48 / 0.66 / 0.67, B0 0.41 / 0.58 / 0.58.
- Pairwise tuning (GW190814-like, 6 cases): all 16 settings give 3.59-3.68e-7 median (IFFT-FFT 1.54e-7, B1 3.47e-7);
  `diag="linear"` 1.9e-6. Per-band attribution: band 1 carries -3.0e-7 (pairwise) vs -1.2e-7 (IFFT-FFT); class 0 (|h_22|^2)
  contributes -2.5e-7 of it; a smooth-PSD control keeps -3.1e-7, so lines are not the cause. Floor = period-`T_b` folding
  of the diagonal term's finite-window tails onto `A_0`. Default kept.
- B1 variants: `all_half_bins` (p = 1) 2.1e-7 -> 4.6e-8 (BNS), 2.2e-7 -> 3.8e-8 (GW190814-like), 4.1e-7 -> 4.4e-7 (BBH) at
  equal cost; `p = 2` all-bins 3.9e-8 on the BBH. Default kept pending the population study (both evaluated in 007).
- L = 50 with the same durations (K_MB 20167 -> 28543, 4865 -> 6026, 4408 -> 5146): pairwise 2.3e-9 / 8.7e-10 / 4.9e-9,
  B1 2.3e-9 / 8.7e-10 / 4.9e-9 (adaptive B1 selects B0 in every truncated band), IFFT-FFT 3.4e-9 / 1.6e-8 / 4.1e-8,
  linear 8e-9 / 8.5e-7 / 2.0e-6. New design option recorded in TASK.md §7 (008) and notes §10.3: HOM-aware band rule
  so that B0 alone suffices for the total waveform.

Next: iteration 007 (population study, C1/C1'), then 008 (band rule incl. the B0-compatible rule).

## Iteration 007 — population study with IMRPhenomXHM (PASS; C1/C1' evidence)

Codex (~45 min) added `mbquad/population.py` (seeded samplers for BNS / NSBH / BBH priors with `Mc >= Mc_ref`, strong-HOM
subsets with `q <= 0.15` and `iota in (1.3, 1.8)`, cached fine-grid references), `add_lorentzian_lines`, three scripts and
`tests/test_population.py`; 128 tests pass. 100 regular draws per population (+20 strong-HOM for NSBH/BBH), two PSDs
(analytic + Lorentzian lines in every band; real L1), `L = 5` and `50`, six methods. Artifacts were produced under the
prompt's `iter006` names and renamed by the orchestrator to `data/population_iter007*`, `data/tc_scan_iter007.md`,
`figures/iter007_*` (scripts renamed accordingly; `iter006_common.py` belongs to iteration 006).

Results (median / 99th percentile of `|Delta ln Lambda|` at SNR 25; ratios to the IFFT-FFT in parentheses):
- Pairwise (method A), L = 5: BNS lines 1.55e-5 / 1.67e-5 (0.82 / 0.43), BNS L1 1.26e-6 / 2.06e-6 (0.079 / 0.045),
  NSBH lines 4.42e-5 / 8.01e-5 (1.038 / 0.91; 1.028 excluding one draw above the prior's chirp-mass corner), NSBH L1
  1.88e-5 / 6.12e-5 (0.36 / 0.39), BBH lines 3.16e-5 / 1.03e-4 (0.96 / 0.90), BBH L1 3.29e-5 / 9.22e-5 (0.18 / 0.30).
  L = 50: ratios 0.04-0.10 (BNS, BBH) and 1.04 / 0.48 (NSBH lines), 0.53 / 0.30 (NSBH L1). Strong-HOM subsets: all
  pairwise ratios <= 1 (0.28-0.99 medians). **C1 is met everywhere except the NSBH population with the synthetic-line PSD,
  where the median is 3-4% above the IFFT-FFT's (the 99th percentile is below); with the real PSD method A is 3-13x better.**
- B1 (method B, total waveform), L = 5: ratios 1.44 / 1.63 (BNS lines), 2.6 / 2.3 (BNS L1), 0.81 / 0.99 (NSBH lines),
  0.97 / 1.49 (NSBH L1), 0.76 / 1.16 (BBH lines), 0.76 / 1.70 (BBH L1); strong-HOM BBH 1.02 / 2.06 (lines), 1.08 / 1.84 (L1).
  L = 50: 0.04-1.03 everywhere (B1 selects B0 in every truncated band). **C1' (<= 1.5x) fails at L = 5 for the BNS and the
  BBH 99th percentiles, holds at L = 50.** `all_half_bins` is not consistently better (BNS lines 3.5x worse, BNS L1 better).
- B0 on the XHM populations is as accurate as the pairwise method (e.g. BNS L1 1.21e-6, NSBH L1 1.90e-5, BBH L1 3.16e-5 at
  L = 5; strong-HOM subsets within 10% of pairwise): the exaggerated cross terms of the toys (where B0 was 20x worse) do not
  occur in these populations, and B0 costs the same as linear interpolation (0.02-0.08 ms) with no FFT and no modes.
- Absolute levels: all new-method medians <= 1.4e-4 at SNR 25 (the paper's L = 5 IFFT-FFT median on GW190814 was 2e-4);
  at L = 50 they are 3e-7 to 3e-6. Linear interpolation: 1e-4 to 2e-3 (L = 5).
- `t_c` scan (21 values over the prior): pairwise constant to 4e-16 relative; IFFT-FFT varies by 1.3e-6, B1 by 2.0e-6 at
  L = 5; linear and B0 constant (power-only formulas).
- Setup: K_MB 21653 -> 30422 (BNS), 5128 -> 6490 (NSBH), 4610 -> 5549 (BBH) from L = 5 to 50; minimum truncated-band
  slack 2.0 / 1.6 / 1.3 s (L = 5); no Eq. 51 root failures; no clipped margins.

Orchestrator's reading for the owner: C1 holds up to a 4% median excess in one synthetic case; C1' holds only at L = 50;
B0 is the surprise practical winner for the total waveform on realistic populations. Proposal: define the accuracy
criteria in absolute terms (median `|Delta ln Lambda| <= 2e-4` at SNR 25 for L = 5, the paper's own level) in addition to
the ratios, and treat the L = 50 (or HOM-aware) band rule as the recommended configuration for B0/B1.

Next: iteration 008 (band rules, L scaling, prior slack audit, B0-compatible rule).

## Iteration 008 — band rules for HOM: L scan, B0-compatible rule, slack audit (PASS; C3 evidence)

Codex (~40 min) added `BandSetup(rule="paper"|"hh_half"|"hh_full", L_hh, m_min)`, `mbquad/slack.py` (phase-based lag audit
from the model's own `d arg h_lm / df`), two scripts, figures and `tests/test_bandrules.py`; 146 tests pass; defaults unchanged.

Orchestrator verification (tables inspected):
- `hh_half` (edges chosen so that `2 [Lambda_max(f_lo) + L_hh sigma_mmax(f_lo)] <= T_b`), `L = 5`, `L_hh = 5`: adaptive B1 selects
  B0 in every truncated band; K_MB +22.8% (BNS 20167 -> 24770), +2.7% (GW190814-like), +2.0% (BBH), +3.0% (strong NSBH).
  Relative errors (median / worst over the two PSDs) of B0 = B1: BNS 4.4e-10 / 8.1e-10, GW190814-like 1.6e-7 / 1.8e-7,
  BBH 2.0e-8 / 3.6e-8, strong NSBH 4.8e-9 / 6.1e-9, negative-spin BBH 3.4e-8 / 6.4e-8 (paper-band IFFT-FFT: 2.5e-8, 1.6e-7,
  6.2e-8, 3.4e-8, 1.2e-7). Worst `|Delta ln Lambda|` at SNR 25: 5.7e-5. `L_hh = 3` or `8` is not consistently better.
  Estimated route costs (1 / 3 detectors, ms, iteration-006 component costs): BNS IFFT-FFT 9.76 / 16.36 vs B0-hh_half
  8.00 / 8.15 vs pairwise-paper 8.61 / 9.78; GW190814-like 3.04 / 4.28 vs 2.50 / 2.54 vs 3.42 / 3.73; BBH 2.78 / 3.91 vs
  2.27 / 2.31 vs 5.22 / 5.53. **For the total waveform, B0 with the `hh_half` rule is the recommended configuration:
  no FFT, no modes, 0.82x (1 detector) / 0.50-0.59x (3 detectors) of the IFFT-FFT route, IFFT-FFT-level or better accuracy.**
- L scan (paper rule, L = 5, 10, 20, 50; medians over PSDs): pairwise improves from 1e-8..4e-7 to 6e-10..1e-8, IFFT-FFT from
  4e-8..2e-7 to 8e-9..5e-8; fitted slopes are not the tail-envelope -3 / -2 (moving edges, cancellations, PSD lines,
  diagonal fold) but pairwise steepens faster (GW190814-like: -2.8 vs -1.0).
- Slack vs error: shifting the blocks by +-1 sigma leaves the floor unchanged; forcing the support outside the block
  (-5.9 s slack) gives 8.4e-4 relative error (0.26 in ln Lambda at SNR 25), i.e. the design margins matter and are adequate.
- Prior slack audit with the model's own time-frequency relation over (Mc, q, chi_eff) grids (42 / 45 / 54 points): no
  negative measured slack in any truncated band; worst slack 1.01 / 0.96 / 0.87 s (m_max = 4) and 1.41 / 1.36 / 1.29 s
  (m_max = 5, Newtonian prediction: the XHM (5,5) one-mode probe crashed inside LAL); measured/Newtonian lag ratios at the
  lower band edges 0.94-1.02 (BNS) down to 0.07-0.13 (NSBH/BBH near merger: measured lags are shorter than the Newtonian
  prediction, hence inside the blocks); non-dyadic durations {T, T/4, T/16, T/64} keep coverage.

Next: iteration 009 (numba kernels, final timing tables, C2); then 010 = documents (METHOD_NOTE.md, REPORT.md, incl. the
final full-scale BNS table, merged from the former 010/011).
