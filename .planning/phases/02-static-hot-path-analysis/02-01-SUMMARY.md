---
phase: "02"
plan: "01"
subsystem: audit-report
tags: [static-analysis, elastic-alignment, complexity, allocations, parallelism, tracer]
status: complete

dependency_graph:
  requires:
    - .planning/research/AUDIT-REPORT.md (Phase 1 sections intact — §Methodology, §Workload Matrix)
    - fdars-core/src/alignment/karcher.rs (source read for complexity and parallelism confirmation)
    - fdars-core/src/alignment/pairwise.rs (source read for pairwise distance parallelism)
    - fdars-core/src/elastic_fpca.rs (source read for SVD sites and sequential inner loops)
    - fdars-core/src/alignment/nd.rs (source read for ND Gram matrix SVD site)
  provides:
    - .planning/research/AUDIT-REPORT.md ## Phase 2 — Static Hot-Path Analysis section
    - ### Complexity Table (seeded with Elastic Alignment row; Plan 02 adds remaining 5 modules)
    - ### Allocation Hotspot List (seeded with 6 elastic to_dmatrix() SVD sites; Plan 02 adds remaining)
    - ### Parallelism Gap List (seeded with elastic findings; Plan 02 adds remaining modules)
  affects:
    - .planning/phases/02-static-hot-path-analysis/02-02-PLAN.md (appends rows into same three tables)

tech_stack:
  added: []
  patterns:
    - Three-list format (Complexity Table + Allocation Hotspot List + Parallelism Gap List) validated end-to-end on elastic alignment tracer slice
    - Feature-gate annotation standard: [always] / [parallel-gated] / [sequential] / [linalg-gated]
    - File:line citation format: `file:line` — function name — context

key_files:
  created: []
  modified:
    - .planning/research/AUDIT-REPORT.md (## Phase 2 section + three sub-sections appended)

decisions:
  - elastic_fpca.rs:930 enclosing function confirmed as optimize_balance_c_raw (inside eval_c closure) — source-verified, resolving RESEARCH Open Question 2
  - elastic_fpca.rs:930 is called on every golden-section iteration (up to 20×), making it a higher-priority dhat target than a single-call SVD site
  - Phase 2 SVD-copy site count: 8 production sites; test-only matrix.rs:682 excluded without citing the line in the skeleton (deferred to Plan 02 Task 2 authority)
  - Three-table format established end-to-end on tracer; Plan 02 can safely expand by adding rows

metrics:
  completed_date: "2026-08-07"
  duration_minutes: 5
  tasks_completed: 2
  tasks_total: 2
  commits: 2
  files_modified: 1

estimate:
  tokens: 55000

actuals:
  tokens: 4200
  tasks: 2
  commits: 2
---

# Phase 2 Plan 01: Phase 2 Section Skeleton + Elastic Alignment Tracer Slice Summary

One-liner: Phase 2 three-table format proven end-to-end on elastic alignment (O(max_iter·N·m²) complexity, 6 to_dmatrix() SVD sites [always], 2 ALREADY PARALLEL loops + 3 SEQUENTIAL gap candidates).

## What Was Built

This plan is the **tracer slice** for Phase 2. It appended a new `## Phase 2 — Static Hot-Path Analysis` section to `.planning/research/AUDIT-REPORT.md` (below Phase 1 content, byte-for-byte intact) with three sub-sections, each populated with the Elastic Alignment module's findings only. This proves the format end-to-end on the highest-complexity, most-fragile module before Plan 02 expands to the remaining five modules.

**Task 1:** Appended the `## Phase 2` section skeleton with:
- Short lead paragraph confirming 8 production SVD-copy sites (test-only `matrix.rs:682` excluded)
- `### Complexity Table` with column contract: Module | Primary function (file:line) | N complexity | M complexity | Feature gate | Fragile flag
- `### Allocation Hotspot List` with column contract: Site (file:line) | Category | Enclosing fn | Alloc size | Feature gate | Phase target
- `### Parallelism Gap List` with column contract: Loop (file:line) | Status | Parallelism macro | Feature gate tag | Gap candidate?

**Task 2:** Populated all three tables with Elastic Alignment content:

*Complexity Table row:* `karcher_mean` → `karcher_mean_impl` (alignment/karcher.rs:323), secondary `elastic_self_distance_matrix` → `self_distance_matrix_impl` (alignment/pairwise.rs:194). N- and M-scaling stated separately per RESEARCH Pitfall 4: O(max_iter·N·m²) unbanded; O(max_iter·N·m·band) banded. N-scaling is unchanged by banding. Feature gate: `[parallel-gated]` (inner N-loop at karcher.rs:185 uses `iter_maybe_parallel!`). Fragile: banding opt-in.

*Allocation Hotspot List — 6 elastic `to_dmatrix()` SVD sites, all `[always]`:*
- `elastic_fpca.rs:214` — `horiz_fpca` — n×m DMatrix
- `elastic_fpca.rs:317` — `joint_fpca` — n×(2m+1) DMatrix
- `elastic_fpca.rs:483` — `horiz_fpca_from_alignment` — n×m DMatrix
- `elastic_fpca.rs:584` — `joint_fpca_from_alignment` — n×(2m+1) DMatrix
- `elastic_fpca.rs:930` — `optimize_balance_c_raw` / `eval_c` closure — n×(m_aug+m) DMatrix, allocated on **every golden-section iteration** (≤20×), making it the highest-frequency SVD site in the elastic path
- `alignment/nd.rs:705` — ND elastic FPCA Gram matrix — m×m DMatrix

*Parallelism Gap List — elastic findings:*
- `karcher.rs:185` — ALREADY PARALLEL (`iter_maybe_parallel!`, `[parallel-gated]`) — not a gap
- `pairwise.rs:227` — ALREADY PARALLEL (`iter_maybe_parallel!`, `[parallel-gated]`) — not a gap
- `elastic_fpca.rs:701` (`shooting_vectors_from_psis`) — SEQUENTIAL (`[sequential]`) — Phase 5 gap candidate
- `elastic_fpca.rs:720` (`build_augmented_srsfs`) — SEQUENTIAL (`[sequential]`) — Phase 5 gap candidate
- `elastic_fpca.rs:764` (`svd_scores_and_eigenvalues` inner i-loop) — SEQUENTIAL (`[sequential]`) — Phase 5 gap candidate
- Banding note: `karcher_mean()` passes `band_frac=0.0` → `None` (full DP, `[always]`); `karcher_mean_banded()` required to enable O(m·band) DP

## Open Question 2 Resolved

RESEARCH Open Question 2 asked which function contains `elastic_fpca.rs:930`. Source read confirmed: it is inside the function `optimize_balance_c_raw`, specifically inside the `eval_c` closure defined at line 919. This closure is called up to 20 times per golden-section search, making this SVD site the highest-frequency allocation in the elastic path — more critical than a single-invocation SVD site.

## Deviations from Plan

None — plan executed exactly as written. The `matrix.rs:682` mention was initially placed in the lead paragraph but was corrected inline (no separate commit) to reference the test-only site by description only, not by line number, per Task 1's directive: "Do NOT write the specific test-only line reference in this skeleton."

## Known Stubs

None. The tables are populated with the elastic tracer content as specified. Plan 02 will append the remaining 5 module rows.

## Threat Flags

None. This plan performs read-only static analysis of local source files and writes only to a local planning artifact (`.planning/research/AUDIT-REPORT.md`). No network surface, no auth paths, no schema changes.

## Self-Check

### File existence
- `.planning/research/AUDIT-REPORT.md` — modified and committed
- `.planning/phases/02-static-hot-path-analysis/02-01-SUMMARY.md` — this file

### Commit existence
- `90d43c8a` (Task 1 — Phase 2 skeleton)
- `a5fd8c1a` (Task 2 — Elastic Alignment table content)

## Self-Check: PASSED
