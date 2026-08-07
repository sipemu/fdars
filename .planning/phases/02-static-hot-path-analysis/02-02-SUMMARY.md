---
phase: "02"
plan: "02"
subsystem: audit-report
tags: [static-analysis, complexity-table, allocation-hotspot, parallelism-gap, fpca-svd, depth, cv-loops, streaming-depth, smoothing]
status: complete

dependency_graph:
  requires:
    - .planning/research/AUDIT-REPORT.md (## Phase 2 section + three sub-section headers from Plan 01)
    - fdars-core/src/regression.rs (center_columns at :167, weighted.clone() at :291, SVD at :298)
    - fdars-core/src/depth/fraiman_muniz.rs (fraiman_muniz_1d at :32, delegates to streaming)
    - fdars-core/src/metric/lp.rs (lp_cross_1d :57, lp_self_1d :105, iter_maybe_parallel!)
    - fdars-core/src/classification/cv.rs (fclassif_cv :45, fold loop :76)
    - fdars-core/src/streaming_depth/fraiman_muniz.rs (depth_batch :77, iter_maybe_parallel! :82)
    - fdars-core/src/streaming_depth/mbd.rs (depth_batch :71, iter_maybe_parallel! :76)
    - fdars-core/src/smoothing.rs (nadaraya_watson :72, slice_maybe_parallel! :110)
    - fdars-core/src/spm/mfpca.rs (SVD at :336)
    - fdars-core/src/smooth_basis.rs, basis/auto_select.rs, basis/fourier_fit.rs, basis/projection.rs, seasonal/ssa.rs, basis/pspline.rs, elastic_regression/regression.rs, elastic_regression/scalar_on_shape.rs (from_column_slice sites)
  provides:
    - .planning/research/AUDIT-REPORT.md ## Phase 2 — Static Hot-Path Analysis (complete: all 3 tables fully populated)
    - ### Complexity Table — 6 module rows (elastic + 5 new)
    - ### Allocation Hotspot List — 8 to_dmatrix() SVD sites + 14 from_column_slice basis sites + 1 redundant clone
    - ### Parallelism Gap List — 2 SEQUENTIAL gaps (cv.rs:76, regression.rs:167) + 4 ALREADY PARALLEL entries + SC1-SC4 verification table
  affects:
    - .planning/phases/02-static-hot-path-analysis (phase complete — both plans done)
    - Phases 3–6 benchmark and optimization plans (consume this complete Phase 2 section as input)

tech_stack:
  added: []
  patterns:
    - Open Question 1 resolved by grep: fraiman_muniz_1d delegates to StreamingFraimanMuniz::depth_batch — [parallel-gated] confirmed
    - Feature-gate annotation standard validated across all 6 modules: [always] / [parallel-gated] / [sequential]
    - SC1-SC4 grep verification table appended to Parallelism Gap List as inline evidence
    - No-false-positive check: grep confirmed no macro wraps cv.rs:76 or regression.rs:167 before labeling SEQUENTIAL

key_files:
  created: []
  modified:
    - .planning/research/AUDIT-REPORT.md (5 complexity rows + 8+14+1 allocation entries + 6 parallelism entries + SC table appended)

decisions:
  - Open Question 1 resolved: fraiman_muniz_1d (depth/fraiman_muniz.rs:32) immediately delegates to StreamingFraimanMuniz::depth_batch which uses iter_maybe_parallel!(0..nobj) — label is [parallel-gated], NOT a gap
  - from_column_slice basis sites are a distinct category from to_dmatrix() SVD copies — different optimization path (least-squares/QR vs full SVD) per RESEARCH Pitfall 5
  - regression.rs:291 weighted=centered.clone() is a zero-copy candidate but FpcaResult retains centered so the clone cannot be trivially elided — requires pre-allocated buffer strategy
  - cv.rs:76 fold loop has no RNG in body (assign_folds is pre-loop) — safe iter_maybe_parallel! candidate for Phase 5 with no per-thread seeding concern
  - SC1-SC4 all pass: O(n counts=15, to_dmatrix=12, already-parallel=8, gate-tags=44

metrics:
  completed_date: "2026-08-07"
  duration_minutes: 8
  tasks_completed: 3
  tasks_total: 3
  commits: 3
  files_modified: 1

estimate:
  tokens: 70000

actuals:
  tokens: 18500
  tasks: 3
  commits: 3
---

# Phase 2 Plan 02: Phase 2 Static Hot-Path Map Expansion Summary

One-liner: Phase 2 three-table static hot-path map completed across all 6 modules — 5 new complexity rows, 8+14+1 allocation sites catalogued, 2 SEQUENTIAL gaps and 4 ALREADY PARALLEL loops identified, all SC1–SC4 greps passing.

## What Was Built

This plan is the **expansion slice** for Phase 2. It filled in the remaining 5 modules across the three tables started by Plan 01's elastic alignment tracer, and closed both research Open Questions with source greps.

**Task 1 — Complexity Table (5 remaining module rows):**

- **FPCA/SVD:** `fdata_to_pc_1d` (`regression.rs:249`). `center_columns` is a sequential double `for` loop at `regression.rs:167/176` — O(N·M). Weighted scale O(N·M) sequential. nalgebra SVD O(M³) at `regression.rs:298`, always sequential regardless of `parallel` feature. Feature gate: `[sequential]`. RESEARCH Pitfall 1 noted: the sequential `center_columns` is a different function from the parallel `fdata.rs:center_1d`.
- **Depth & distance:** `fraiman_muniz_1d` (`depth/fraiman_muniz.rs:32`). Open Question 1 resolved: the static FM depth immediately delegates to `StreamingFraimanMuniz::depth_batch` which uses `iter_maybe_parallel!(0..nobj)` at `streaming_depth/fraiman_muniz.rs:82`. FM complexity O(N_obj·N_ref·M). Distance matrix via `lp_cross_1d`/`lp_self_1d` also `iter_maybe_parallel!`-gated. Feature gate: `[parallel-gated]`.
- **CV loops:** `fclassif_cv` (`classification/cv.rs:45`). Outer `for fold in 0..nfold` at `cv.rs:76` is a plain sequential loop — no macro. Per-fold cost = FPCA O(M³) + classifier. Feature gate: `[sequential]`. Note: no RNG in fold body (assign_folds is pre-loop), so Phase 5 parallelization is safe.
- **Streaming depth:** `StreamingFraimanMuniz::depth_batch` (`streaming_depth/fraiman_muniz.rs:77`). `iter_maybe_parallel!(0..nobj)` at `:82`. O(N_obj·N_ref·M) query, O(N_ref·M·log N_ref) build. Feature gate: `[parallel-gated]`.
- **Smoothing:** `nadaraya_watson` (`smoothing.rs:72`). Outer `slice_maybe_parallel!(x_new)` at `smoothing.rs:110`; inner `for i in 0..n` sequential at `:115`. Feature gate: `[parallel-gated]` (outer loop).

**Task 2 — Allocation Hotspot List:**

Added the 2 non-elastic production `to_dmatrix()` SVD sites: `regression.rs:298` (`fdata_to_pc_1d`) and `spm/mfpca.rs:336` (`mfpca`). Together with Plan 01's 6 elastic sites, this totals 8 production SVD sites — confirming the ROADMAP "8" claim. The test-only `matrix.rs` site remains excluded.

Added 14 `DMatrix::from_column_slice` basis-construction sites as a distinct category (`smooth_basis.rs:198,199,695,696`, `seasonal/ssa.rs:178`, `basis/auto_select.rs:95,128`, `basis/fourier_fit.rs:68`, `basis/projection.rs:113,117,119`, `elastic_regression/regression.rs:274,278`, `elastic_regression/scalar_on_shape.rs:117,119`). All `[always]`, no linalg-gating at module level. These are least-squares/basis constructions — a different optimization path from SVD copies (RESEARCH Pitfall 5).

Added `regression.rs:291` redundant clone: `weighted = centered.clone()` immediately before in-place weight scaling and SVD. Double n×m heap allocation; `FpcaResult` retains `centered` so the clone cannot be trivially elided, but `weighted` could be written into a pre-allocated buffer.

**Task 3 — Parallelism Gap List + SC verification:**

Added 2 SEQUENTIAL gap entries:
- `regression.rs:167` (`center_columns`) — SEQUENTIAL, no macro (grep confirmed). Distinct from parallel `fdata.rs:center_1d` (Pitfall 1). Phase 5 candidate.
- `cv.rs:76` (fold loop) — SEQUENTIAL, no macro (grep confirmed). Each fold independent, no shared state. Phase 5 candidate.

Added 4 ALREADY PARALLEL entries:
- `streaming_depth/fraiman_muniz.rs:82` — `iter_maybe_parallel!`
- `streaming_depth/mbd.rs:76` — `iter_maybe_parallel!`
- `smoothing.rs:110` — `slice_maybe_parallel!` outer loop
- `depth/fraiman_muniz.rs:32` (static FM) — delegates to streaming depth_batch (already parallel)

Added inline SC1–SC4 verification table confirming all four grep thresholds pass.

## Open Questions Resolved

- **Open Question 1** (RESEARCH): Whether the static `fraiman_muniz_1d` in `depth/mod.rs` uses a parallelism macro. Resolution: it is in `depth/fraiman_muniz.rs` (not `depth/mod.rs` directly) and delegates immediately to `StreamingFraimanMuniz::depth_batch` → `iter_maybe_parallel!`. Label: `[parallel-gated]`. Not a gap.
- **Open Question 2** (resolved in Plan 01): `elastic_fpca.rs:930` enclosing function is `optimize_balance_c_raw` (inside `eval_c` closure).

## Deviations from Plan

None — plan executed exactly as written. The RESEARCH §2B listed `elastic_regression/scalar_on_shape.rs:117,119` as 2 separate sites; both are included, with `:119` noted as the companion penalty matrix alongside `:117` in the site description.

## Known Stubs

None. All three tables are now fully populated.

## Threat Flags

None. This plan performs read-only static analysis of local source files and writes only to a local planning artifact. No network surface, no auth paths, no schema changes.

## Self-Check

### File existence
- `.planning/research/AUDIT-REPORT.md` — modified with Task 1–3 content appended

### Commit existence
- `f5805d32` (Task 1 — 5 complexity table rows)
- `78e176ce` (Task 2 — allocation hotspot list complete)
- `4f41c27d` (Task 3 — parallelism gap list + SC verification)

## Self-Check: PASSED
