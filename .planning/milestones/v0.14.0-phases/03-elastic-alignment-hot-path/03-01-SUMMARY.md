---
phase: 03-elastic-alignment-hot-path
plan: "01"
subsystem: bench-infrastructure
tags: [benchmarks, elastic-alignment, karcher, criterion, audit]
dependency_graph:
  requires: []
  provides:
    - bench_p3_karcher (criterion bench fn, audit_p3_karcher group)
    - p3_karcher_linalg,parallel_run1.txt (raw criterion artifact)
    - AUDIT-REPORT.md Phase-3 section (one measured row + backlog stub)
  affects:
    - fdars-core/benches/audit_hotpaths.rs
    - .planning/research/AUDIT-REPORT.md
tech_stack:
  added: []
  patterns:
    - criterion benchmark cell (bench_p3_karcher mirrors bench_matrix_sentinel exactly)
    - AUDIT-REPORT append-only pattern (D-05: single growing report)
    - Phase artifact naming p3_<target>_<features>_run<N>.txt
key_files:
  created:
    - .planning/research/bench/p3_karcher_linalg,parallel_run1.txt
  modified:
    - fdars-core/benches/audit_hotpaths.rs
    - .planning/research/AUDIT-REPORT.md
decisions:
  - Deferred karcher_mean_banded import to Plan 02 to keep this wave warning-clean (one unused import would be generated at this stage)
  - Used --no-verify on all 3 commits: /tmp tmpfs full causes LLVM IO linker error in pre-commit doctest hook; this is an infrastructure failure per Pitfall 8 (not a code defect)
  - mean time ~318 ms for n100_m50 unbanded karcher_mean at D-06 params (max_iter=20, tol=1e-4, lambda=0.0)
metrics:
  duration: 6 min
  completed: 2026-08-07
  completed_plans: 1
  total_plans: 2
status: complete
actuals:
  tokens: 9200
  tasks: 3
  commits: 3
---

# Phase 03 Plan 01: Karcher Tracer Cell — Elastic Alignment Hot Path Summary

**One-liner:** karcher_mean unbanded-default tracer cell proven end-to-end via criterion bench fn, release+linalg,parallel run (~318 ms at N=100×M=50), artifact saved, AUDIT-REPORT Phase-3 section appended with measured row and D-07 backlog stub.

## What Was Built

This tracer plan proved the complete measurement pipeline for Phase 3 in one thin vertical slice — bench fn → release run → raw artifact → report row → backlog stub — so Plan 02 can expand to the full grid (N∈{100,500}×M∈{50,200}, banded twins, all three targets) without re-discovering wiring issues.

### Task 1: Add bench_p3_karcher tracer cell

Added a new criterion bench function `bench_p3_karcher` to `fdars-core/benches/audit_hotpaths.rs`, mirroring `bench_matrix_sentinel` exactly in structure. Key details:
- Benchmark group: `audit_p3_karcher`
- Cell: `n100_m50` (N=100, M=50)
- D-06 locked params: `max_iter=20`, `tol=1e-4`, `lambda=0.0` (NOT the sentinel 10/1e-3)
- `black_box` on all inputs and the return value
- Input built outside `b.iter()`
- Doc comment records D-05 Anti-Pattern 2 (band_frac=0.0 default at karcher.rs:300)
- Registered in `criterion_group!` macro after `bench_smooth_sentinel`
- `karcher_mean_banded` import deferred to Plan 02 (keeps this wave warning-clean)

### Task 2: Run tracer cell and save artifact

Ran the tracer cell: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo bench -p fdars-core --features linalg,parallel --bench audit_hotpaths -- audit_p3_karcher`

Result:
- `audit_p3_karcher/n100_m50 time: [315.35 ms 318.04 ms 321.12 ms]`
- Mean: ~318 ms
- 3 outliers (15%): 2 high mild, 1 high severe
- Binary path confirmed: `target/release/deps/audit_hotpaths-aea52eeb0c35d5bd` (Pitfall 1 satisfied)

Artifact saved to `.planning/research/bench/p3_karcher_linalg,parallel_run1.txt` with environment header, run command, full criterion output, and notes.

### Task 3: Append Phase-3 AUDIT-REPORT section

Appended (append-only per D-05) a `## Phase 3: Elastic Alignment Hot Path — Benchmark Results` section to `.planning/research/AUDIT-REPORT.md` containing:
- Results table with 8-column header and one populated `karcher_mean` row linking to the artifact
- D-05 note citing `karcher.rs:300` and naming Anti-Pattern 2
- Draft backlog stub with all D-07 SC4 fields (Function / Current cost / Root cause / Candidate fix)
- Stub marked `[STUB — Plan 02 finalizes with full banded-vs-unbanded numbers]`

Prior Phase 1 and Phase 2 content is intact (append-only).

## Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add bench_p3_karcher tracer cell | f05c773f | fdars-core/benches/audit_hotpaths.rs |
| 2 | Save karcher tracer bench artifact | a36c72c3 | .planning/research/bench/p3_karcher_linalg,parallel_run1.txt |
| 3 | Append Phase-3 AUDIT-REPORT section | b823623e | .planning/research/AUDIT-REPORT.md |

## Key Numbers

| Metric | Value |
|--------|-------|
| karcher_mean N=100, M=50, linalg,parallel, unbanded (D-06) | ~318 ms |
| Phase-1 elastic sentinel (elastic_self_distance_matrix N=100, M=50) | ~790 ms |
| Outliers in tracer run | 3 of 20 (15%) — 2 high mild, 1 high severe |
| Confidence | PENDING (single run; Plan 02 adds run2) |

**Context:** The karcher_mean tracer (~318 ms) is the first criterion evidence of the unbanded cost. At this cell size, `elastic_self_distance_matrix` (Phase-1 sentinel) took ~790 ms — about 2.5× slower than karcher. Plan 02 will measure both functions at the full N∈{100,500}×M∈{50,200} grid plus banded twins for the ~7× reduction comparison.

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written, with two infrastructure-driven deviations:

**1. [Infrastructure — Pitfall 8] --no-verify used on all 3 commits**
- **Found during:** Task 1 commit attempt
- **Issue:** /tmp tmpfs at ~94% capacity. Pre-commit hook runs `cargo test -p fdars-core --features linalg` which includes doctests. Doctest compilation links binaries in /tmp → LLVM ERROR: IO failure on output stream: No space left on device. This is an infrastructure failure per Pitfall 8 (SIGBUS/link fail with NO named `FAILED test_name` line = infra, not code). The issue pre-exists this plan (MEMORY.md, STATE.md decision).
- **Fix:** Used `--no-verify` on all 3 commits. Cargo build succeeded; 1934+ unit/integration tests passed (separately verified before hook ran). No code defects found.
- **Note:** Documented per the sequential_execution block note in PLAN.md: "Retry that single commit with `git commit --no-verify` and note it in SUMMARY.md."

**2. [Executor's choice — import deferral] karcher_mean_banded import deferred to Plan 02**
- **Found during:** Task 1 implementation
- **Issue:** Plan noted executor's choice — import `karcher_mean_banded` here (since Task 3's backlog stub references it) OR defer to Plan 02 to keep wave warning-clean.
- **Fix:** Deferred. Importing `karcher_mean_banded` at this stage produces one unused-import warning (Plan 02's bench cells are the first consumers). Deferring to Plan 02 keeps this wave clean.

## Tracer Feedback Gate

Tracer end-to-end verified:
- Bench fn (`bench_p3_karcher`) compiled clean under `--features linalg,parallel` ✓
- Release run (`cargo bench -- audit_p3_karcher`) completed, produced timing output ✓
- Artifact saved with toolchain tag + /release/ confirmation ✓
- AUDIT-REPORT Phase-3 section appended with populated row + backlog stub ✓

Pipeline proven. Plan 02 can expand to the full grid.

## Known Stubs

None — this tracer plan deliberately leaves Plan 02's grid cells as future work (planned, not unintentional stubs). The AUDIT-REPORT results table row is marked `CONFIDENCE: PENDING (single run — Plan 02 adds run2)` and the backlog stub is marked `[STUB — Plan 02 finalizes with full banded-vs-unbanded numbers]`. These are intentional placeholders per plan design, not gaps.

## Self-Check

Files exist:
- `.planning/research/bench/p3_karcher_linalg,parallel_run1.txt` ✓
- `fdars-core/benches/audit_hotpaths.rs` (modified) ✓
- `.planning/research/AUDIT-REPORT.md` (modified, Phase-3 section appended) ✓

Commits exist:
- f05c773f: feat(03-01) bench fn ✓
- a36c72c3: chore(03-01) artifact ✓
- b823623e: docs(03-01) report ✓
