---
phase: 113-cross-language-baseline-hotspot-ranking
plan: "01"
subsystem: benchmarks
tags: [benchmark-harness, numeric-gate, pearson-correlation, criterion, r-fpca]
status: complete

dependency_graph:
  requires:
    - phase: 112-benchmark-harness-setup
      provides: benchmarks/gate.py (full-op agreement gate), benchmarks/r_runner.R (R timing runner), criterion bench infrastructure
  provides:
    - benchmarks/gate.py (recalibrated: depth/functional_boxplot/fpcr use Pearson corr >= threshold)
    - benchmarks/test_gate.py (updated: scale-shifted passes + uncorrelated fails tests)
    - benchmarks/r_runner.R (added run_fpca_native_r() — ecosystem-realistic pca.fd timing)
    - target/criterion/fregre_lm/ (5 cells populated; previously missing)
    - target/criterion/functional_boxplot/ (5 cells populated; previously missing)
  affects: [113-02, 113-03]

actuals:
  tokens: 14200
  tasks: 3
  commits: 2
  plan_head_before: 91e47b7c3e54f87d073f340bdc657a81d9a577f9

tech-stack:
  added: []
  patterns:
    - Pearson correlation gate for scale/shift-invariant cross-library agreement checks
    - Disclosed threshold relaxation pattern: _FPCR_CORR_THRESHOLD=0.95 with inline rationale comment
    - Preprocessing-outside-timed-lambda pattern for fpca_native_r (Pitfall 5)

key-files:
  created: []
  modified:
    - benchmarks/gate.py
    - benchmarks/test_gate.py
    - benchmarks/r_runner.R

key-decisions:
  - "depth/functional_boxplot gate: switched from max-abs-diff to Pearson corr >= 0.99 — the 0.65% MBD normalization diff between scikit-fda and roahd is scale/shift invariant; abs-diff wrongly rejected valid agreement"
  - "FPCR gate threshold: 0.95 (disclosed relaxation) not 0.99 — Phase-112 measured Python-vs-R corr=0.96; fdars as reference may be higher but 0.95 used as conservative disclosed floor; comment in gate.py names the empirical value"
  - "fpca_native_r not in ALL_OPS — keeps --smoke all-ops contract unchanged; invoked only via --op fpca_native_r"
  - "criterion output (target/) is gitignored — fregre_lm and functional_boxplot cells are on disk for Plan 02 to consume; no .rs change was needed"

patterns-established:
  - "Pearson corr gate: (corr, _) = pearsonr(a, b); passed = abs(corr) >= threshold; returns (passed, abs(corr), [{pearson_corr: corr}])"
  - "Disclosed threshold: separate named constant (_FPCR_CORR_THRESHOLD) with inline comment citing the measured empirical value"

requirements-completed: [BASE-01, BASE-02]

coverage:
  - id: D1
    description: "gate.py depth/functional_boxplot/fpcr branches use Pearson correlation >= threshold"
    requirement: BASE-02
    verification:
      - kind: unit
        ref: "benchmarks/test_gate.py::TestDepthAgreement (3 tests), TestFunctionalBoxplotAgreement (3 tests), TestFpcrAgreement (3 tests)"
        status: pass
      - kind: integration
        ref: "benchmarks/gate.py --self-test (all ops: pass-on-agree AND fail-on-mismatch)"
        status: pass
    human_judgment: false
  - id: D2
    description: "r_runner.R run_fpca_native_r() emits fpca_native_r timing record with mean_ns>0 and no result_fingerprint"
    requirement: BASE-01
    verification:
      - kind: integration
        ref: "Rscript benchmarks/r_runner.R --op fpca_native_r --n 60 --m 50 --seed 42 → r.json mean_ns=12978754 ns"
        status: pass
    human_judgment: false
  - id: D3
    description: "criterion output populated for fregre_lm (5 cells) and functional_boxplot (5 cells)"
    requirement: BASE-01
    verification:
      - kind: integration
        ref: "glob target/criterion/fregre_lm/**/new/estimates.json → 5 files, all mean>0; glob target/criterion/functional_boxplot/**/new/estimates.json → 5 files, all mean>0"
        status: pass
    human_judgment: false

duration: 7min
completed: 2026-09-22
---

# Phase 113 Plan 01: Gate Recalibration + fpca_native_r + Criterion Coverage Summary

**Agreement gate recalibrated to Pearson correlation for depth/FPCR (MBD normalization invariance), native R pca.fd timing branch added, and two missing criterion groups populated — all prerequisites for the Wave-2 baseline run.**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-09-22T19:47:37Z
- **Completed:** 2026-09-22T19:54:xx Z
- **Tasks:** 3 of 3
- **Files modified:** 3 (gate.py, test_gate.py, r_runner.R)

## Accomplishments

- Replaced abs-diff gate for depth/functional_boxplot/fpcr with Pearson correlation — fixes the Phase-112 hand-off failures (MBD normalization diff ~0.65% now correctly passes; scale-shifted data passes; uncorrelated data correctly rejected)
- Added `_DEPTH_CORR_THRESHOLD = 0.99` and `_FPCR_CORR_THRESHOLD = 0.95` constants with inline disclosure of empirical basis
- Added `run_fpca_native_r()` to r_runner.R: B-spline smoothing pre-computed outside timed lambda, only `pca.fd()` timed; record has `op="fpca_native_r"`, no fingerprint, NOT in ALL_OPS
- Populated `target/criterion/fregre_lm/` (5 cells: ncomp/3, ncomp/5, ncomp/10, scale/n50_nc5, scale/n200_nc5) by running existing bench
- Populated `target/criterion/functional_boxplot/` (5 cells: N/50, N/200, N/500, N/1000, N/2300)
- All 21 gate tests pass; `gate.py --self-test` exits 0; no fdars-core `.rs` change

## FPCR Threshold Decision Record

| Metric | Value | Source |
|--------|-------|--------|
| Phase-112 measured FPCR corr (Python vs R) | 0.96 | 112-03-SUMMARY.md Task 4 |
| FPCR threshold set | **0.95** | Disclosed relaxation — constant `_FPCR_CORR_THRESHOLD` |
| depth/functional_boxplot threshold | 0.99 | `_DEPTH_CORR_THRESHOLD` |
| Rationale | fdars uses Simpson's weights (closer to R's integration than Python's trapezoid); the 0.96 Python-vs-R result is the empirical floor; 0.95 confirmed agreement up to scale normalization differences while discriminating against genuine numerical failures |

Wave 2/3 and the Phase-116 report should cite: "FPCR gated at Pearson corr ≥ 0.95 (disclosed relaxation; empirical Phase-112 Python-vs-R corr = 0.96; fdars-vs-R expected higher due to Simpson's weight alignment)."

## fpca_native_r Timing Record (n60/m50)

| Run | mean_ns | Notes |
|-----|---------|-------|
| Run 1 | 18,589,085 ns (~18.6 ms) | seed=42, n=60, m=50 |
| Run 2 | 12,978,754 ns (~13.0 ms) | re-run (n_iterations=10 via bench::mark) |

Native `pca.fd()` on n=60, m=50 averaged ~13–19 ms. B-spline smoothing was excluded from timed call (Pitfall 5 per RESEARCH).

## Criterion Cells Populated (Task 3)

| Group | Cell | mean_ns |
|-------|------|---------|
| fregre_lm | ncomp/3 | 629,891 |
| fregre_lm | ncomp/5 | 626,108 |
| fregre_lm | ncomp/10 | 280,080 |
| fregre_lm | scale/n50_nc5 | 368,997 |
| fregre_lm | scale/n200_nc5 | 503,813 |
| functional_boxplot | N/50 | 315,866 |
| functional_boxplot | N/200 | 1,620,693 |
| functional_boxplot | N/500 | 6,047,676 |
| functional_boxplot | N/1000 | 12,712,595 |
| functional_boxplot | N/2300 | 44,704,748 |

All cells have `mean.point_estimate > 0` in their `new/estimates.json`.

## Task Commits

1. **Task 1 (tracer): Recalibrate gate.py + test_gate.py** — `7d4b787d`
2. **Task 2: Add fpca_native_r to r_runner.R** — `afc4a78e`
3. **Task 3: Populate criterion output** — no commit (target/ is gitignored; output on disk)

## Files Created/Modified

- `benchmarks/gate.py` — recalibrated depth/functional_boxplot/fpcr branches; added _DEPTH_CORR_THRESHOLD=0.99, _FPCR_CORR_THRESHOLD=0.95; updated self_test()
- `benchmarks/test_gate.py` — updated TestDepthAgreement, TestFunctionalBoxplotAgreement, TestFpcrAgreement for Pearson metric; added scale_shifted_passes + uncorrelated_fails tests
- `benchmarks/r_runner.R` — added run_fpca_native_r() function + switch case; ALL_OPS unchanged

## Decisions Made

- **FPCR threshold 0.95**: Phase-112 empirical corr=0.96 (Python-vs-R). Using 0.95 as a conservative disclosed floor accommodates scale-normalization differences across three different FPCA weight schemes (fdars/Simpson's, scikit-fda/trapezoid, R fregre.pc). Disclosed in gate.py constant comment and this SUMMARY.
- **fpca_native_r not in ALL_OPS**: Adding it to ALL_OPS would include it in `--smoke` and default all-ops runs, breaking the established smoke contract and potentially causing gate failures (no fingerprint → no gate check → no issue, but it changes run semantics). Invoked explicitly only.
- **No criterion output commit**: `target/` is gitignored project-wide; Plan 02 reads the files directly from disk. No workaround (e.g. `git add -f`) was applied — this was an expected non-issue.

## Deviations from Plan

None — plan executed exactly as written. The FPCR threshold decision of 0.95 was pre-authorized by the plan's critical execution notes ("if <0.99, set a distinct `_FPCR_CORR_THRESHOLD = 0.95` constant and DISCLOSE it").

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes introduced. All changes are local script modifications handling trusted synthetic CSV data (T-113-01 and T-113-SURFACE from threat register; no new surface).

## Self-Check: PASSED

- benchmarks/gate.py: EXISTS, modified
- benchmarks/test_gate.py: EXISTS, modified
- benchmarks/r_runner.R: EXISTS, modified
- target/criterion/fregre_lm/ (5 cells): EXISTS on disk
- target/criterion/functional_boxplot/ (5 cells): EXISTS on disk
- Task 1 commit 7d4b787d: EXISTS
- Task 2 commit afc4a78e: EXISTS
- pytest test_gate.py: 21 passed (0 failed)
- gate.py --self-test: PASSED (all ops)
- fpca_native_r r.json mean_ns>0 and no fingerprint: VERIFIED
- git diff --name-only -- fdars-core/: empty (no .rs changes)

---
*Phase: 113-cross-language-baseline-hotspot-ranking*
*Completed: 2026-09-22*
