# Iteration 010 — documents: METHOD_NOTE.md and REPORT.md, final figures, cleanup (C4)

You are working in `multiband_quad/` (the current directory). Read `AGENTS.md`, `TASK.md` (all sections, especially 1-4 and
9-12), `notes/method_idea.md` (all sections; §8-12 record the review corrections and the findings of iterations 005-008),
`notes/paper_summary.md`, every `data/*.md` table (iterations 002-009: `accuracy_iter00*.md`, `margin_study_iter003.md`,
`slack_iter003.md`, `xhm_accuracy_iter005.md`, `xhm_timing_iter006.md`, `pairwise_tuning_iter006.md`, `b1_variants_iter006.md`,
`population_iter007.md`, `tc_scan_iter007.md`, `bandrule_iter008.md`, `prior_slack_iter008.md`, `perf_iter009.md`,
`profile_overhead_iter009.md`) and the figures in `figures/`, and the package `mbquad/`. The audience of the
documents is the paper's author (a gravitational-wave data analyst); write in English, concise, with equations in LaTeX-free
plain text or simple Markdown math, and cite the paper's equation numbers.

## Deliverables

1. `METHOD_NOTE.md` (self-contained, ~4-8 pages):
   - the problem (Eq. 26, the paper's two methods and why the IFFT-FFT needs the `2 T_b` grid);
   - the block-weight identity (Sec. 3 of the note, with the corrected conventions: `a_k = 1/S_k`, `g = w h_A^* h_A'`,
     positive lags for the slower mode first), the folding-error formula, and the unified view (linear interpolation =
     period-`T_b` folding + Fejer taper; IFFT-FFT = period-`2 T_b` folding + crop; method A = period-`T_b` folding +
     class-specific crop; B0 = period-`T_b` centred crop on the total; B1 = baseband IFFT-FFT);
   - method A: lag classes, block placement on `[Lambda_min(Mc_max) - L sigma_pair(f_hi), Lambda_max(Mc_min) + L sigma_pair(f_lo)]`,
     slack bookkeeping, the mode matrix factorisation `(h,h) = C^dagger M C` and `t_c` independence, requirements on the
     waveform API (modes on arbitrary frequency sequences; LAL conventions found in iteration 005);
   - method B: B0 (centred block on the total) and B1 (baseband IFFT-FFT with half-sample shift, adaptive B0/B1), the
     `hh_half` band rule and when B0 alone is adequate;
   - accuracy summary (fixed cases, populations, L dependence) and cost summary (steps, routes for 1 and 3 detectors,
     precomputation), each with pointers to the tables;
   - limitations and open points (non-precessing scope and the co-precessing extension, LAL `OneMode` tail truncation,
     the (3,2) phase residual, priors wider than the block design, PSD lines, the measured L slopes);
   - a short "how to use" section with the public API (`BandSetup`, `PSDGrid`, `BlockWeights`, `PairwiseHH`, `CentredBlockHH`,
     `BasebandIFFTFFTHH`, `XHMCoarseEvaluator`, numba options) and a 20-line example.
2. `REPORT.md` (results-oriented, ~3-5 pages): the question, the three test systems and the populations, the headline
   tables (copy the key rows from `data/*.md`: fixed-case accuracy, population medians/99th percentiles with ratios to the
   IFFT-FFT, route costs for 1 and 3 detectors, `hh_half` costs and accuracy, precomputation), the figures
   (`figures/iter007_cdf.png`, `iter007_errors_*.png`, `iter003_lag_audit.png`, `iter008_L_scan.png`, plus one new summary
   figure `figures/report_summary.png` with two panels: accuracy (median `|Delta ln Lambda|` at SNR 25 per method and
   population, L = 5 and 50) and cost (route ms for 1 and 3 detectors per method and system)), the verdicts against the
   completion criteria C1-C5 of `TASK.md` with the exact numbers and the stated exceptions, and the recommendations
   (method A when modes are available and `t_c`/extrinsic factorisation matters; B0 + `hh_half` for the total waveform;
   B1 when the band rule cannot be changed; what to port to bilby's `MBGravitationalWaveTransient`).
3. `scripts/make_report_figure.py` producing `figures/report_summary.png` from the existing JSON tables (no new heavy runs).
4. Cleanup: a `README.md` in `multiband_quad/` (one page: what is here, how to run the tests and the main scripts, where the
   data products are), consistent module docstrings, and `python -m pytest tests -q` green. Do not delete data products or
   figures; do not change numerical defaults.

## Rules (from AGENTS.md)

- Create/modify files only inside this directory; no git state changes; no package installation; no network.
- `TASK.md`, `PROGRESS.md`, `AGENTS.md`, `notes/*` and `prompts/*` stay untouched (the orchestrator maintains them).

## Completion criteria / verification

- `python -m pytest tests -q` passes; `scripts/make_report_figure.py` runs; `METHOD_NOTE.md`, `REPORT.md`, `README.md` exist
  and every number in them can be traced to a table in `data/` (cite the file next to each table).
- Final report: a list of the documents with their section headings, the C1-C5 verdict lines you wrote, and anything you
  found inconsistent between the tables while writing (do not silently fix numbers).
