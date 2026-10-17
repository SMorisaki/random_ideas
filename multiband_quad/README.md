# multiband_quad

Research implementation of faster `(h,h)` evaluation for gravitational-wave
multibanding with higher modes, following Morisaki, PRD 104, 044062 (2021),
arXiv:2104.07813, Eq. (26). The paper's linear interpolation and IFFT-FFT
algorithms are the baselines. Method A uses mode-pair block weights;
B0/B1 use only total detector strain.

## Start here

- [METHOD_NOTE.md](METHOD_NOTE.md): derivation, lag/block conventions,
  prior coverage, waveform API, limitations and a runnable example.
- [REPORT.md](REPORT.md): accuracy, timings, criterion verdicts and figures.
- [TASK.md](TASK.md) and [PROGRESS.md](PROGRESS.md): scope and iteration log.
- [notes/](notes/): paper PDF, plain-text summary and design history.

The recommended total-waveform option is B0 with explicit
`BandSetup(rule="hh_half")`; A adds mode-matrix reuse over extrinsics.
B1 is available for fixed bands. Accuracy qualifications are in the report;
existing numerical defaults remain unchanged.

## Environment and tests

Use the existing `python` on PATH with NumPy, SciPy, numba, matplotlib,
h5py, pytest, lal and lalsimulation. Do not install packages or create a
new environment from this directory. Run from `multiband_quad/`:

```bash
which python
python -c "import scipy, numba, lalsimulation"
python -m pytest tests -q
```

`mbquad/` contains geometry/windows/PSDs, fine-grid reference sums, coarse
sample containers, the five quadrature routes, the XHM adapter, support
audits and opt-in numba kernels. `tests/` covers identities, conventions,
regressions and kernel agreement. Specialised evaluators are imported from
their modules, as in the method note. Evaluators reuse buffers; give each
concurrent worker its own instance and warm numba before timing.

## Main scripts

```bash
python scripts/make_report_figure.py
python scripts/population_iter007.py --help
python scripts/perf_iter009.py --help
```

The first command only reads existing JSON and writes
[figures/report_summary.png](figures/report_summary.png). The remaining
commands show experiment arguments. To regenerate fixed studies, run
`python scripts/xhm_accuracy_iter005.py`,
`python scripts/bandrule_iter008.py`, or
`python scripts/prior_slack_iter008.py`; these have no CLI parser and
compute results even if passed `--help`. They can perform expensive
fine-grid XHM evaluations. `python scripts/band_table.py` regenerates the
historical geometry table, including its earlier BBH setup; use the report
for final systems. For reproducible
benchmark runs set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OMP_WAIT_POLICY=PASSIVE` before Python starts.
The benchmark scripts also pin threads. Accuracy/timing scripts from
iterations 002–006 retain the toy and fixed-case controls;
`tc_scan_iter007.py`, `plot_iter007.py`, and
`profile_overhead_iter009.py` provide the remaining diagnostics.

## Data products

[data/](data/) holds small Markdown tables and JSON summaries. Cite the
specific table when quoting a number; fixed systems, population priors and
band choices differ across iterations. Large `*_full.json` per-point
records, `*_run.log`, reference caches (`*.npz`), raw measured PSD files
and plotting caches are local products; many are git-ignored. Measured-PSD
runs need the local Welch files named in the scripts; the synthetic routes
do not need network access. [figures/](figures/) holds all plots. Existing
products are retained, and the summary figure can be rebuilt without
rerunning the research experiments.
