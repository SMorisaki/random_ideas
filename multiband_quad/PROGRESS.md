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
