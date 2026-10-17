# Iteration 009 — performance: numba kernels, flattened weights, precomputation cost, final timing tables (C2)

You are working in `multiband_quad/` (the current directory). Read `AGENTS.md`, `TASK.md` (sections 4 [C2], 7 [009], 9-11),
`notes/method_idea.md` §8.11, §10-11, `data/xhm_timing_iter006.md`, `data/population_iter007.md` (timings section) and the
code of `mbquad/pairwise.py`, `baseband.py`, `totalweights.py`, `ifftfft.py`, `linear.py`, `lalwave.py`, `timing.py`.
`numba` is installed (0.68). `OMP_NUM_THREADS=1` for all timings. Iteration 008 established the `hh_half` band rule
(`BandSetup(..., rule="hh_half", L_hh=5)`) under which B0 alone is adequate for the total waveform; include "B0 with hh_half
bands" as a route in the tables (its K_MB differs: BNS 24770, GW190814-like 4994, BBH 4497).

## Goal (completion criterion C2)

Make the run-time `(h,h)` kernels as cheap as the structure allows and produce the definitive timing tables:
- method A: `(h,h)` step <= 1/5 of the fast IFFT-FFT step and <= 10% of the mode-evaluation time (BNS and GW190814-like);
- method B1: step <= 1/3 of the fast IFFT-FFT step (already met in numpy; keep or improve);
- precomputation <= 60 s and <= 200 MB for the BNS setting (measure and report; expected far below).

## Deliverables

1. `mbquad/kernels.py` (numba, `cache=True`, `fastmath=False`):
   - flattened pairwise kernel: inputs are one concatenated array of raw-or-detector mode samples `H` of shape
     `(n_modes, K_MB)` (band slices contiguous), a per-point class-weight table `Wc` of shape `(n_classes, K_MB)` complex
     (window and `4/T` folded in, class 0 real), and the class index map `(n_modes, n_modes) -> class or -1 (conjugate)`;
     implementation groups modes by `m` on the fly (sum same-`m` modes first) and accumulates
     `sum_k [ sum_m W0_k |G_m,k|^2 + 2 Re sum_{m>m'} Wc_{(m,m'),k} conj(G_m,k) G_m',k ]`; a second entry point returns the
     Hermitian mode matrix `M` (for the factorisation C5) from the same inputs;
   - B0 kernel: `sum_k w0_k |h_k|^2` over the flattened grid (trivial; mostly to remove numpy overhead);
   - B1: keep scipy FFTs but move the window/phase multiply, power and weighted sums into numba; reuse preallocated buffers.
   Provide `PairwiseHH(..., impl="numba")`, `CentredBlockHH(impl=...)`, `BasebandIFFTFFTHH(impl=...)` wrappers that build the
   flattened tables once and dispatch to the kernels; results must agree with the numpy implementations to 1e-12 relative.
2. `scripts/perf_iter009.py`: on the BNS, GW190814-like and BBH setups with `L = 5` (and the BNS at `L = 50`), warm medians
   of all `(h,h)` steps (linear, IFFT-FFT paper/fast, pairwise einsum/pairs/numba, B0 numpy/numba, B1 numpy/numba,
   B1 at `L = 50`, B0 with `hh_half` bands), the mode-matrix kernel, the lean waveform evaluations (polarizations, five modes), and the route sums for
   1 and 3 detectors as in iteration 006 (IFFT-FFT paper bands; pairwise paper bands; B1 paper bands; B0 `hh_half` bands,
   whose waveform cost is scaled by its K_MB); plus the precomputation wall time and memory (`BlockWeights.nbytes`, `tracemalloc`
   peak) for each evaluator at the BNS setting. Save `data/perf_iter009.md` (+ JSON). State explicitly whether the C2
   targets are met, with the numbers.
3. A short `scripts/profile_overhead_iter009.py` that reports, for the GW190814-like setup, the time of the evaluator
   calls split into LAL / numpy bookkeeping / kernel, so the remaining Python overhead is documented.
4. Tests (`tests/test_kernels.py`): numba vs numpy agreement for all three methods on `toy_small()` and one XHM system (1e-12);
   mode matrix from the kernel equals `PairwiseHH.mode_matrix`; kernels handle a single mode, two modes and the five XHM modes;
   keep the default test run under ~2 minutes (numba compile time counts; use `cache=True`).

## Rules (from AGENTS.md)

- Create/modify files only inside this directory; no git state changes; no package installation; no network.
- Do not change the numpy defaults or results of iterations 001-008 (the numba paths are opt-in; say if you recommend
  making one the default and why).

## Completion criteria / verification

- `python -m pytest tests -q` passes; the scripts run and write their tables.
- Final report: the timing tables (steps, routes for 1 and 3 detectors, precomputation cost/memory), the C2 verdict per
  target, the overhead profile, and anything the orchestrator should know.
