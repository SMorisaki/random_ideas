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
