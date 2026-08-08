---
phase: 06-conditional-svd-library-comparison
plan: 01
subsystem: benchmarking
tags: [faer, nalgebra, svd, fpca, criterion, audit, performance]

requires:
  - phase: 04-fpca-svd-allocation-audit
    provides: GO verdict (SVD ~99.8-99.9% wall-clock, copy 0.14-0.17%) triggering Phase 6

provides:
  - generate_weighted_input helper in audit_hotpaths.rs replicating fdata_to_pc_1d centering + Simpson-sqrt-weight scaling
  - bench_p6_svd_comparison criterion bench function with three sub-groups (nalgebra, faer_seq, conversion) at 7-cell grid
  - svd_equivalence integration test confirming nalgebra and faer agree within 1e-10
  - Five bench artifacts under .planning/research/bench/ (nalgebra run1+2, faer_seq run1+2, conversion run1)
  - Phase 6 section in AUDIT-REPORT.md satisfying all four ROADMAP SC1-SC4 criteria

affects:
  - Phase 9 backlog (P6-1 faer SVD adoption item with measured evidence)
  - Any future audit phase re-examining FPCA performance

actuals:
  tokens: 10839
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "bench_p6_svd_comparison uses cells loop pattern with per-cell sample_size tiering"
    - "faer::set_global_parallelism(Par::Seq) called once outside b.iter() before faer_seq group"
    - "FdMatrix->faer::MatRef via as_slice() (zero-copy) — no to_dmatrix() allocation"
    - "svd_equivalence in separate integration test (not cfg(test) mod in bench) because harness=false"

key-files:
  created:
    - fdars-core/tests/svd_equivalence.rs
    - .planning/research/bench/p6_svd_nalgebra_linalg_run1.txt
    - .planning/research/bench/p6_svd_nalgebra_linalg_run2.txt
    - .planning/research/bench/p6_svd_faer_seq_linalg_run1.txt
    - .planning/research/bench/p6_svd_faer_seq_linalg_run2.txt
    - .planning/research/bench/p6_svd_conversion_linalg_run1.txt
  modified:
    - fdars-core/benches/audit_hotpaths.rs
    - .planning/research/AUDIT-REPORT.md

key-decisions:
  - "svd_equivalence moved to integration test (tests/svd_equivalence.rs) because harness=false bench binaries do not expose #[test] items via cargo test --bench — same pattern as alloc_audit_fpca.rs"
  - "Full 7-cell grid implemented in Task 1 (tracer) rather than expanding incrementally in Task 2 — cleaner code; Task 2 became bench-run-only"
  - "Near-zero singular values (< 1e-8 * sigma_1) excluded from equivalence check — both backends produce floating-point noise at machine-epsilon scale, not a correctness issue"
  - "Severity set to P2 borderline for P6-1 backlog item: measured speedup at primary cell N=500,M=200 is 1.8x (run1) — below 2x research threshold but consistently positive across all 7 cells; high run-to-run variance due to missing governor pinning"
  - "faer parallel path NOT measured — kept scope tight per plan; Phase 9 should add parallel measurement before final P2/P3 decision"

requirements-completed: [PERF-06]

coverage:
  - id: D1
    description: "generate_weighted_input helper replicates fdata_to_pc_1d centering + Simpson-sqrt-weight scaling"
    requirement: PERF-06
    verification:
      - kind: integration
        ref: "fdars-core/tests/svd_equivalence.rs#svd_equivalence (uses generate_weighted_input)"
        status: pass
    human_judgment: false
  - id: D2
    description: "bench_p6_svd_comparison registered in criterion_group! covering all 7 grid cells"
    requirement: PERF-06
    verification:
      - kind: other
        ref: "cargo build -p fdars-core --features linalg --benches exits 0"
        status: pass
    human_judgment: false
  - id: D3
    description: "svd_equivalence confirms nalgebra and faer singular values agree within 1e-10"
    requirement: PERF-06
    verification:
      - kind: integration
        ref: "fdars-core/tests/svd_equivalence.rs#svd_equivalence"
        status: pass
    human_judgment: false
  - id: D4
    description: "Five bench artifacts under .planning/research/bench/ (nalgebra/faer_seq run1+2, conversion run1)"
    requirement: PERF-06
    verification:
      - kind: other
        ref: "ls .planning/research/bench/p6_svd_*_linalg_run*.txt — all 5 present"
        status: pass
    human_judgment: false
  - id: D5
    description: "AUDIT-REPORT.md Phase 6 section with GO verdict, comparison table, adoption note, Phase 9 backlog item"
    requirement: PERF-06
    verification: []
    human_judgment: true
    rationale: "Table numbers must be traced to on-disk artifacts; crossover observation requires human judgment; P2/P3 severity call is a judgment call based on measured 1.8x speedup vs 2x threshold"

duration: 180min
completed: 2026-08-08
status: complete
---

# Phase 6 Plan 01: Conditional SVD Library Comparison Summary

**faer thin_svd is 1.8–4.1x faster than nalgebra SVD at fdars' real FPCA sizes, with zero-copy FdMatrix conversion; P6-1 backlog item drafted at P2/S-effort (borderline — primary cell 1.8x, below 2x threshold)**

## Performance

- **Duration:** ~180 min (bench runs account for ~120 min: 7 cells × 3 groups × 2 runs)
- **Started:** 2026-08-08
- **Completed:** 2026-08-08
- **Tasks:** 3/3
- **Files modified/created:** 8

## Accomplishments

- Added `generate_weighted_input(n, m)` helper to `audit_hotpaths.rs` that replicates the exact matrix `fdata_to_pc_1d` feeds to nalgebra SVD: column centering + Simpson-sqrt-weight scaling. Called outside `b.iter()` so allocation is excluded from the bench window.
- Added `bench_p6_svd_comparison` criterion bench function with three sub-groups (`audit_p6_svd_nalgebra`, `audit_p6_svd_faer_seq`, `audit_p6_svd_conversion`) covering all 7 grid cells including the M=500 crossover probe. Registered in `criterion_group!`.
- Created `tests/svd_equivalence.rs` integration test that confirms nalgebra and faer agree on all significant singular values within 1e-10, and verifies faer thin_svd shape (U is n×m for N>M — Open Question 1 resolved).
- Ran all three sub-groups twice each; saved five artifacts under `.planning/research/bench/` (nalgebra run1+2, faer_seq run1+2, conversion run1). All produced in release profile with `--features linalg`.
- Appended `## Phase 6: Conditional SVD Library Comparison` to `AUDIT-REPORT.md` satisfying all four ROADMAP success criteria (SC1: GO verdict, SC2: 7-cell comparison table, SC3: faer adoption note, SC4: GSD-ready Phase 9 backlog item P6-1).

## Task Commits

1. **Task 1: SVD-comparison tracer + full grid** — `1345a404` (feat)
2. **Task 2: Run bench twice, save p6_svd_* artifacts** — `5536f552` (chore)
3. **Task 3: Append Phase 6 section to AUDIT-REPORT.md** — `60549c59` (docs)

## Files Created/Modified

- `fdars-core/benches/audit_hotpaths.rs` — added `generate_weighted_input`, `bench_p6_svd_comparison`, registered in `criterion_group!`
- `fdars-core/tests/svd_equivalence.rs` — NEW: numerical equivalence integration test (deviation: in tests/ not bench #[cfg(test)], see below)
- `.planning/research/bench/p6_svd_nalgebra_linalg_run1.txt` — NEW: nalgebra SVD timings run 1
- `.planning/research/bench/p6_svd_nalgebra_linalg_run2.txt` — NEW: nalgebra SVD timings run 2
- `.planning/research/bench/p6_svd_faer_seq_linalg_run1.txt` — NEW: faer thin_svd (Par::Seq) run 1
- `.planning/research/bench/p6_svd_faer_seq_linalg_run2.txt` — NEW: faer thin_svd (Par::Seq) run 2
- `.planning/research/bench/p6_svd_conversion_linalg_run1.txt` — NEW: FdMatrix→MatRef conversion cost
- `.planning/research/AUDIT-REPORT.md` — appended Phase 6 section (~86 lines)

## Decisions Made

- **svd_equivalence in integration test:** Bench binaries with `harness = false` use criterion's main, not the Rust test harness — `#[test]` items in `#[cfg(test)]` blocks are not discovered by `cargo test --bench`. Moved to `tests/svd_equivalence.rs` following the `alloc_audit_fpca.rs` precedent. This enables `cargo test -p fdars-core --features linalg --test svd_equivalence` to run it properly.
- **Full grid in Task 1:** The task description said tracer = single cell N=500,M=200. Implemented the full 7-cell grid in Task 1 anyway (cells loop pattern is cleaner than per-cell code). Task 2 became bench-run-only. Not a scope deviation — all 7 cells were always required by Task 2.
- **Significant-values filter in equivalence test:** Near-zero singular values (below 1e-8 × σ₁) excluded from the 1e-10 relative error check. The deterministic sine-curve data has rank ≈ 2, so values beyond index 2 are floating-point noise in both backends — comparing them would produce spurious failures without indicating a correctness issue.
- **P2 borderline severity:** Measured speedup at primary cell N=500,M=200 is 1.8× (run1) — below the RESEARCH "clearly worth it" ≥2× threshold. Set P2 because direction is consistently positive across all 7 cells and absolute saving at N=1000,M=200 is ~27ms/call. High run-to-run variance (missing governor pinning, same issue as Phase 5) means run3 under pinned performance governor could revise to P3 if speedup falls below 1.5×.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] svd_equivalence moved from bench #[cfg(test)] mod to integration test**
- **Found during:** Task 1 (equivalence test verification)
- **Issue:** Bench binaries with `harness = false` use criterion's main function, not the Rust test harness. `#[test]` items in `#[cfg(test)]` blocks inside such bench files are compiled but not discovered by `cargo test --bench`. The plan's verify command `cargo test ... --bench audit_hotpaths svd_equivalence` ran the criterion test mode (not the Rust test harness) and did not execute `svd_equivalence`.
- **Fix:** Created `fdars-core/tests/svd_equivalence.rs` as an integration test (same pattern as `alloc_audit_fpca.rs`). Removed `#[cfg(test)] mod tests` block from bench file. Run via `cargo test -p fdars-core --features linalg --test svd_equivalence`.
- **Files modified:** `fdars-core/benches/audit_hotpaths.rs` (no test mod), `fdars-core/tests/svd_equivalence.rs` (new)
- **Verification:** `test svd_equivalence ... ok` (1 passed)
- **Committed in:** 1345a404 (Task 1 commit)

**2. [Rule 1 - Bug] Near-zero singular value comparison excluded from equivalence check**
- **Found during:** Task 1 (svd_equivalence test, first run)
- **Issue:** Test failed at singular value index 2 with rel_err=1.4e-2 (nalgebra=6.6e-15, faer=2.1e-14). Both values are numerical noise (below machine epsilon × σ₁); the relative error of noise-vs-noise is meaningless.
- **Fix:** Added abs_threshold = 1e-8 × σ₁ filter; both-below-threshold values skipped. Added assertion that at least one significant value was compared.
- **Files modified:** `fdars-core/tests/svd_equivalence.rs`
- **Verification:** Test passes (2 significant singular values compared, both within 1e-10)
- **Committed in:** 1345a404 (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (Rule 3 - blocking infra limitation, Rule 1 - test bug)
**Impact on plan:** Both fixes necessary for correct equivalence testing. No scope creep. The test structure change follows documented project precedent (alloc_audit_fpca.rs pattern).

## Issues Encountered

- **Run-to-run variance in faer bench results:** faer run2 is 20–39% faster than run1 at several cells (e.g., n500_m500: 190ms → 115ms). This is attributable to CPU cache warmup and OS frequency scaling under `powersave` governor (same issue as Phase 5 — `cpupower` requires root for governor pinning). Run1 numbers are used as the conservative baseline in the AUDIT-REPORT table; run2 confirms direction. The 5-artifact discipline (two nalgebra runs + two faer_seq runs + one conversion run) captures the variance for human review.

## User Setup Required

None — audit-only phase, no external service configuration required.

## Known Stubs

None — this phase produces bench artifacts and a report section. No UI, no data stubs.

## Next Phase Readiness

- Phase 6 complete: all four ROADMAP success criteria (SC1-SC4) satisfied
- AUDIT-REPORT.md carries the complete Phase 6 section including GO verdict, measured comparison table, faer adoption note, and GSD-ready P6-1 backlog item
- Phase 9 can promote P6-1 with the evidence artifacts already on disk
- Recommendation for Phase 9: add faer parallel measurement (not done here per plan scope) before finalizing P2/P3 severity

---
*Phase: 06-conditional-svd-library-comparison*
*Completed: 2026-08-08*
