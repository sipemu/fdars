---
phase: 113-cross-language-baseline-hotspot-ranking
plan: "02"
subsystem: benchmarks
tags: [benchmark-harness, grid-baseline, comparison-csv, fpcr-gate-failure, pearson-correlation]
status: halted

dependency_graph:
  requires:
    - phase: 113-01
      provides: recalibrated gate.py (FPCR threshold 0.95), fpca_native_r in r_runner.R, criterion output for fregre_lm + functional_boxplot
  provides:
    - benchmarks/run_all.py (--grid mode: N×M loop, per-cell gate-before-timing, per-cell result files)
    - benchmarks/assemble_comparison.py (merge per-cell JSON to comparison.csv with schema + noise fields)
    - benchmarks/test_assemble_comparison.py (10 unit tests, all green)
  affects: [113-03]

actuals:
  tokens: 22000
  tasks: 2
  commits: 2
  plan_head_before: 2016baab814c69bba43355df4135dfbce6cd3c19

tech-stack:
  added: []
  patterns:
    - Per-cell result file pattern: python_n{n}_m{m}.json + r_n{n}_m{m}.json written after gate passes
    - Gate-guarded CSV write: write_comparison_csv(gate_passed=True) raises on gate_passed=False (BENCH-05)
    - TDD (RED→GREEN) for assembler: test_assemble_comparison.py written first, then assembler made green

key-files:
  created:
    - benchmarks/assemble_comparison.py
    - benchmarks/test_assemble_comparison.py
  modified:
    - benchmarks/run_all.py

key-decisions:
  - "--grid mode runs cells strictly sequentially (Pitfall 2 — shared CSV filenames) with gate-before-timing per cell"
  - "fpca_native_r invoked explicitly per cell (not in ALL_OPS) — keeps smoke/default contract unchanged (Plan 01 decision)"
  - "assemble_comparison.py write_comparison_csv() raises ValueError if gate_passed=False (BENCH-05 invariant enforced at write time)"
  - "FPCR gate failure at n=100,m=50 (|corr|=0.9495 < 0.95) is a HARD STOP per plan instructions — threshold not relaxed"

patterns-established:
  - "Grid cell gate-before-timing: run all ops (python + r + fpca_native_r) → collect rust → gate → persist per-cell files → assemble"
  - "Assembler schema: op,library,n,m,mean_ns,ci_lower_ns,ci_upper_ns,std_dev_ns,rsd_pct,ci_width_pct,noise_gate,source,notes"

requirements-completed: []

coverage:
  - id: D1
    description: "--grid mode in run_all.py with per-cell gate-before-timing, sequential loop, fpca_native_r invocation"
    requirement: BASE-01
    verification:
      - kind: integration
        ref: "benchmarks/.venv/bin/python benchmarks/run_all.py --grid --dry-run → exits 0, iterates 6 cells"
        status: pass
      - kind: integration
        ref: "benchmarks/.venv/bin/python benchmarks/run_all.py --help | grep --grid → present"
        status: pass
    human_judgment: false
  - id: D2
    description: "assemble_comparison.py merges JSON to schema-correct comparison.csv with computed noise fields"
    requirement: BASE-01
    verification:
      - kind: unit
        ref: "benchmarks/test_assemble_comparison.py — 10 tests (rsd_pct, ci_width_pct, noise_gate labels, r_native_pca_fd notes, schema header, gate guard)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Grid baseline run + committed comparison.csv (Task 3)"
    requirement: BASE-01
    verification: []
    human_judgment: true
    rationale: "Task 3 halted: FPCR gate failure at n=100,m=50 (|corr|=0.9495 < 0.95); comparison.csv not written; human decision required on threshold or data investigation"

duration: 9min
completed: 2026-09-22
---

# Phase 113 Plan 02: Grid Baseline Run Summary

**--grid mode + assembler implemented (Tasks 1-2 complete); Task 3 halted by FPCR gate failure at cell n=100,m=50 (|Pearson corr|=0.9495 < threshold 0.95) — comparison.csv not written per BENCH-05 mandate.**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-09-22T20:00:15Z
- **Completed (halted):** 2026-09-22T20:09:53Z
- **Tasks:** 2 of 3 complete; Task 3 halted at gate failure
- **Files modified:** 3 (run_all.py, assemble_comparison.py [new], test_assemble_comparison.py [new])

## Accomplishments

- Added `--grid` mode to `run_all.py`: iterates N×{100,300,500} × M×{50,200} (6 cells) strictly sequentially, gate-before-timing per cell, writes `python_n{n}_m{m}.json` + `r_n{n}_m{m}.json` per cell for the assembler
- Added `fpca_native_r` invocation per grid cell (via explicit `--op fpca_native_r` call in `_run_r_with_native_fpca`), separate from ALL_OPS loop
- Created `assemble_comparison.py`: merges per-cell JSON files + rust.json into comparison.csv using pandas.to_csv with the exact RESEARCH Section 5 schema; computes rsd_pct/ci_width_pct/noise_gate for Rust rows; writes with gate_passed guard (BENCH-05)
- Created `test_assemble_comparison.py`: 10 TDD unit tests covering all behavior cases; all green
- **Task 3 gate failure:** FPCR |Pearson corr|=0.9495 < threshold 0.95 at cell n=100,m=50 — aborted per BENCH-05 mandate; comparison.csv NOT written

## Gate Failure Details

| Op | Cell | Python-vs-R |corr| | Threshold | Result |
|----|------|----------------------|-----------|--------|
| fpcr | n=100, m=50 | 0.9495 | 0.95 | FAIL |

**Evidence:**
- Python fpcr fitted_values: range [-1.2189, 0.9541], n=100 values
- R fpcr fitted_values: range [-1.1638, 0.8563], n=100 values  
- Pearson corr: 0.9495 (Spearman: 0.9358, max abs diff: 0.430)
- Both runners used same CSV: `data/curves_n100_m50.csv` + `data/y_n100.csv` (seed=42)

**Context:** Phase-112 measured Python-vs-R FPCR corr=0.96 at n=60, m=50. At n=100, m=50 the correlation drops to 0.9495. This is a genuine algorithmic divergence — scikit-fda and R's fda.usc implement FPCR with different basis representations and fitting procedures; with more curves (n=100 vs 60), the divergence compounds.

**Plan mandate (critical execution notes):** "If a cell's gate genuinely fails on a real cross-library divergence you cannot resolve without relaxing a threshold, STOP and report the failing op/cell/metric in the SUMMARY — do NOT write or commit comparison.csv, and do NOT relax any threshold."

## All ops gate status for cell n=100, m=50

| Op | Result | Notes |
|----|--------|-------|
| fpca (fdars-vs-python) | PASS | min |corr|=0.998253 |
| fpca (fdars-vs-r) | PASS | min |corr|=1.000000 |
| fpcr | FAIL | |corr|=0.9495 < 0.95 |

(Remaining ops — basis_smoothing, depth, functional_boxplot, elastic — were not reached because the gate aborted at the first failure)

## Task Commits

1. **Task 1: --grid mode in run_all.py** — `9f3f7cb5`
2. **Task 2: assemble_comparison.py + TDD tests** — `f036bf8a`
3. **Task 3: Grid baseline run** — HALTED (gate failure; no commit; comparison.csv not written)

## Files Created/Modified

- `benchmarks/run_all.py` — added GRID_N/GRID_M, run_grid(), _run_r_with_native_fpca(), _write_cell_results(), --grid argparse flag
- `benchmarks/assemble_comparison.py` — NEW: assembler for per-cell JSON → comparison.csv
- `benchmarks/test_assemble_comparison.py` — NEW: 10 unit tests (TDD RED→GREEN)

## Decisions Made

- `--grid --dry-run` proves loop wiring without any runner calls (skips all subprocess work)
- `write_comparison_csv(gate_passed=False)` raises ValueError — BENCH-05 enforced at write time, not just orchestrator level
- r_svd library label for R SVD-matched FPCA; r_native_pca_fd for fpca_native_r rows
- Gate failure is a HARD STOP: threshold cannot be relaxed (plan critical execution notes); comparison.csv not written

## Deviations from Plan

### Gate Failure (Not a Deviation — Correct Behavior)

The FPCR gate failure at n=100,m=50 is the gate working correctly. The plan explicitly covers this case: "If a cell's gate genuinely fails on a real cross-library divergence you cannot resolve without relaxing a threshold, STOP and surface the failing op/cell." No threshold was relaxed. The gate aborted before any timing was written (BENCH-05 preserved).

**Root cause investigation needed:**
1. FPCR corr drops from 0.96 (n=60) to 0.9495 (n=100) — algorithmic divergence between scikit-fda and R fda.usc at higher n
2. Options for the next attempt:
   - **Option A:** Reduce the grid to N∈{50,100,500} or use the same n that passed in Phase-112 (n=60) for FPCR validation
   - **Option B:** Use ncomp selection in FPCR to align the number of basis functions used by both libraries
   - **Option C:** Use Spearman rank correlation instead of Pearson (spearman=0.9358, also fails, so this would not help)
   - **Option D (recommended):** Exclude FPCR from the agreement gate for n≥100 since Phase-112 already established that implementations differ; keep gating at n<100 only. Add methodology note to comparison.csv fpcr rows.
   - **Option E:** Lower FPCR threshold to 0.94 (would pass this cell) with disclosed relaxation — BUT plan says "do NOT relax any threshold"

## Known Stubs

- `benchmarks/results/comparison.csv` — NOT WRITTEN (Task 3 halted by gate failure)
- `benchmarks/results/env.json` — stale (not re-captured from a gate-passing grid run)

## Threat Flags

None — no new network endpoints, auth paths, or schema changes. All changes are local script additions handling trusted synthetic CSV data.

## Self-Check: PASSED (Tasks 1-2) / HALTED (Task 3)

- benchmarks/run_all.py has --grid: EXISTS, verified
- benchmarks/assemble_comparison.py: EXISTS
- benchmarks/test_assemble_comparison.py: EXISTS, 10 tests pass
- Task 1 commit 9f3f7cb5: EXISTS
- Task 2 commit f036bf8a: EXISTS
- comparison.csv: NOT WRITTEN (gate failure, correct behavior)
- FPCR gate failure at n=100,m=50: CONFIRMED (|corr|=0.9495 < 0.95)

---
*Phase: 113-cross-language-baseline-hotspot-ranking*
*Halted: 2026-09-22*
