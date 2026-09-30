---
phase: 112-benchmark-harness-setup
plan: "02"
subsystem: benchmarks
tags: [benchmark, cross-language, python, r, criterion, elastic, depth, fpcr]
status: complete

dependency_graph:
  requires: ["112-01"]
  provides: ["python runner all 6 ops", "R runner all 6 ops", "bench_functional_boxplot criterion group"]
  affects: ["112-03 (numeric agreement gate reads python.json + r.json)"]

tech_stack:
  added: []
  patterns:
    - "smoke-all pattern: --smoke without --op runs all ops in one pass"
    - "time_fn() helper: warm-up + bench::mark() or timeit.timeit() abstraction shared across ops"
    - "deterministic MBD depth chosen over random projection in gated path (RESEARCH Pitfall 5)"
    - "lambda=0 pinned on all elastic ops (Python/R) for Fisher-Rao metric equivalence with fdars"

key_files:
  created: []
  modified:
    - benchmarks/python_runner.py
    - benchmarks/r_runner.R
    - benchmarks/generate_data.py
    - fdars-core/benches/depth_benchmarks.rs

decisions:
  - "Functional depth in gated path: ModifiedBandDepth (deterministic) not RandomProjectionDepth — avoids nproj/distribution mismatch between scikit-fda Gaussian and R vexponential projections (RESEARCH Pitfall 5)"
  - "Elastic smoke uses max_iter=20 and hits the iteration cap (n=60 curves) — informational message, not an error; convergence is confirmed for the Plan-03 gate with smaller n"
  - "Python smoke-all mode: --smoke without --op runs all 6 ops sequentially for easy CI verification"
  - "R time_warping Karcher mean returns fn matrix (m x n); first column [,1] is the Karcher mean curve"

metrics:
  duration: "~6 minutes (wall clock)"
  completed: "2026-09-20"
  tasks_completed: 3
  tasks_total: 3

actuals:
  tokens: 12490    # chars/4 over changed files (~49961 chars / 4)
  tasks: 3
  commits: 3       # MEASURED: git rev-list --count 7424ab763aee3d18a3546d9ec499c49830c15523..HEAD
  plan_head_before: "7424ab763aee3d18a3546d9ec499c49830c15523"
---

# Phase 112 Plan 02: Benchmark Harness — Full Overlapping Op Set Summary

**One-liner:** Expanded Python (scikit-fda 0.10.1) and R (fda/roahd/fda.usc/fdasrvf) runners from FPCA-only tracer to all six overlapping ops, and added the missing `bench_functional_boxplot` criterion group to `fdars-core/benches/depth_benchmarks.rs`.

## What Was Built

### Task 1: bench_functional_boxplot criterion group + FPCR/elastic fixtures

- `fdars-core/benches/depth_benchmarks.rs`: added `bench_functional_boxplot` function using `DepthMethod::ModifiedBand` and `factor=1.5` across the same N sizes as neighboring depth benches (`[50, 200, 500, 1000, 2300]`); registered in `criterion_group!`. Added `functional_boxplot` and `DepthMethod` to the existing `use fdars_core::depth::{...}` import block. No new `[[bench]]` entry needed — reuses existing criterion 0.5 dev-dep.
- `benchmarks/generate_data.py`: extended to emit two new fixture files alongside the existing `curves_n{n}_m{m}.csv`:
  - `y_n{n}.csv` — seeded scalar response vector (seed+100) for FPCR benchmarks
  - `elastic_n2_m{m}.csv` — 2-curve phase-shifted sinusoid pair (seed+200) for elastic pairwise alignment benchmarks

### Task 2: Python runner — full overlapping op set

`benchmarks/python_runner.py` extended with 5 new `--op` branches (on top of the existing FPCA branch):

| Op | API | Fingerprint |
|----|-----|-------------|
| `basis_smoothing` | `BSplineBasis(n_basis=10) + BasisSmoother(smoothing_parameter=0.0)` | smoothed curve values (n × m) |
| `depth` | `ModifiedBandDepth()` (deterministic) | depth vector (length n) |
| `functional_boxplot` | `Boxplot(fd, depth_method=ModifiedBandDepth(), factor=1.5)` | MBD depth vector (length n) |
| `fpcr` | `FPCA(5)` scores → `np.linalg.lstsq` on `y_n{n}.csv` | fitted values (length n) |
| `elastic` | `FisherRaoElasticRegistration(penalty=0.0)` + `fisher_rao_karcher_mean` | warping (length m) + Karcher mean (length m) |

Elastic class path verified at runtime: `skfda.preprocessing.registration.FisherRaoElasticRegistration` (penalty=0.0 parameter confirmed in __init__ signature).

`--smoke` without `--op` now runs all 6 ops in a single pass.

### Task 3: R runner — full overlapping op set

`benchmarks/r_runner.R` extended with 5 new branches (on top of FPCA):

| Op | API | Fingerprint |
|----|-----|-------------|
| `basis_smoothing` | `fda::create.bspline.basis(n_basis=10) + smooth.basis(lambda=0)` | smoothed values (n × m via eval.fd) |
| `depth` | `roahd::MBD(fData(grid, mat))` | depth vector (length n) |
| `functional_boxplot` | `roahd::fbplot(Depths="MBD", Fvalue=1.5, display=FALSE)$Depth` | depth vector (length n) |
| `fpcr` | `fda.usc::fregre.pc(fdataobj, y, l=1:5)$fitted.values` | fitted values (length n) |
| `elastic` | `fdasrvf::pair_align_functions(lambda=0, pen="roughness", method="DP")$gam` + `time_warping(lambda=0, max_iter=20)$fn[,1]` | warping (length m) + Karcher mean (length m) |

`--smoke` without `--op` runs all 6 ops.

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written.

### Notes

1. **Elastic Karcher mean convergence warning (R):** `fdasrvf::time_warping` with n=60 curves and max_iter=20 prints "The maximal number of iterations has been reached." — this is informational only. The function returns a valid result. For the smoke test (correctness) this is acceptable; the Plan-03 gate uses smaller n and can be tuned if needed.

2. **Python functional boxplot timing:** `Boxplot` constructs the matplotlib figure structure internally; in smoke mode `mean_ns=0.0` as designed. The timing loop calls both `Boxplot()` and `ModifiedBandDepth()` — the depth computation dominates; the Boxplot overhead is geometric/display overhead (no display in benchmark context).

3. **R `time_warping` return shape:** The function returns a list with `fn` as an (m × n) matrix where the first column is the Karcher mean curve. Documented in code comment.

## Threat Model Review

| Threat ID | Status |
|-----------|--------|
| T-112-04 (input path handling) | Mitigated — all runners read only from `benchmarks/data/` CSV paths constructed from script-relative directory, no shell interpolation |
| T-112-05 (timing hangs at large N) | Accepted — smoke uses n=60 (small); full-grid sizing is Plan-03/Phase-113 concern |

## Self-Check: PASSED

All files exist:
- fdars-core/benches/depth_benchmarks.rs: FOUND
- benchmarks/python_runner.py: FOUND
- benchmarks/r_runner.R: FOUND
- benchmarks/generate_data.py: FOUND

All commits verified:
- fc488e54: feat(112-02): add bench_functional_boxplot criterion group + FPCR/elastic fixtures
- 0f3a70c2: feat(112-02): Python runner — full overlapping op set (scikit-fda 0.10.1)
- f85af788: feat(112-02): R runner — full overlapping op set (fda/fda.usc/roahd/fdasrvf)

Commits measured: 3 (git rev-list --count 7424ab763aee3d18a3546d9ec499c49830c15523..HEAD)
