---
phase: 09-consolidated-report-prioritized-backlog
plan: "02"
subsystem: audit-deliverables
tags: [audit, backlog, report, performance, consolidation]
status: complete

requires:
  - .planning/phases/09-consolidated-report-prioritized-backlog/09-01-SUMMARY.md

provides:
  - .planning/research/BACKLOG.md (performance item blocks: PERF-ELASTIC-BAND, PERF-FPCA-CLONE, PERF-FPCA-TRUNCSVD, PERF-PAR-CV, PERF-PAR-ELFPCA, PERF-PAR-CENTER, ACC-VALIDATE + Ranked Backlog rows)
  - AUDIT-REPORT.md ## Consolidated Findings → ### Performance Findings (PF-2 through PF-5)
affects:
  - .planning/research/BACKLOG.md
  - .planning/research/AUDIT-REPORT.md

tech_stack:
  added: []
  patterns:
    - "7-field backlog item blocks with computed score = value / sqrt(effort) applied to 7 new performance items"
    - "P5-4 cross-reference inside PERF-ELASTIC-BAND (no duplicate item)"
    - "Evidence-linked consolidated findings with traceability to backlog item IDs (PF-N → BACKLOG.md#item-id)"

key_files:
  created: []
  modified:
    - .planning/research/BACKLOG.md
    - .planning/research/AUDIT-REPORT.md

decisions:
  - "PERF-ELASTIC-BAND assigned P1 severity — N=500,M=200 distance matrices are INFEASIBLE (~700s/iter) on the default API, making this a table-stakes capability gap for real workloads"
  - "PERF-PAR-CV assigned P2/S (score=4.00) — highest-scoring new item, one-line macro change targeting a commonly repeated workflow (CV hyperparameter search)"
  - "P5-4 is not a separate item — it is a one-line cross-reference inside PERF-ELASTIC-BAND, per plan artifact constraint"
  - "ACC-VALIDATE assigned P2/M (score=1.73) — cross-cutting accuracy validation item using doc reference (AUDIT-REPORT.md D-02a) as evidence link, per RPT-01 allowance for doc references"
  - "PF-2 and PF-3 both feed PERF-ELASTIC-BAND — PF-2 captures the infeasibility evidence, PF-3 captures the measured banding speedup evidence"
  - "--no-verify used for both commits per MEMORY.md documented exception (/tmp tmpfs exhaustion blocks pre-commit hooks for docs-only commits)"

metrics:
  duration_minutes: 15
  completed_date: "2026-08-09"
  tasks_completed: 2
  commits: 2

estimate:
  tokens: 60000

actuals:
  tokens: 18000
  tasks: 2
  commits: 2
---

# Phase 09 Plan 02: Performance Backlog + Report Findings Summary

Promoted all seven performance draft-backlog slices into full BACKLOG.md items (7 checklist
fields each, computed scores, resolved evidence links), and wrote four additional performance
findings (PF-2 through PF-5) into the AUDIT-REPORT.md Consolidated Findings section, each
with a bench artifact evidence link and a backlog item ID for traceability.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Promote all performance draft-backlog slices into full BACKLOG.md items | 95af4482 | .planning/research/BACKLOG.md |
| 2 | Write remaining performance findings into report Consolidated Findings section | 8efa2b44 | .planning/research/AUDIT-REPORT.md |

## What Was Built

### BACKLOG.md (expanded)

Seven new performance backlog item blocks added with all 7 checklist fields each:

| ID | Severity | Value | Effort | Score | Area |
|----|----------|-------|--------|-------|------|
| PERF-ELASTIC-BAND | P1 | 5 | M | 2.89 | Elastic alignment / `alignment/karcher.rs:300` |
| PERF-PAR-CV | P2 | 4 | S | 4.00 | Classification CV / `classification/cv.rs:76` |
| PERF-FPCA-TRUNCSVD | P2 | 3 | L | 1.00 | FPCA / `regression.rs:298` via `nalgebra::SVD::new` |
| PERF-PAR-ELFPCA | P2 | 3 | M | 1.73 | Elastic FPCA / `elastic_fpca.rs:701/720/764` |
| PERF-PAR-CENTER | P3 | 1 | S | 1.00 | FPCA / `regression.rs:167` |
| PERF-FPCA-CLONE | P3 | 1 | M | 0.58 | FPCA / `regression.rs:291/298` |
| ACC-VALIDATE | P2 | 3 | M | 1.73 | Cross-cutting (Preprocessing / Misc / ML) |

All 7 items carry real measured numbers (no invented figures). P5-4 is a one-line cross-reference
inside PERF-ELASTIC-BAND, not a separate item. The pre-existing P6-1 row was relabeled from
rank `1` to `—` (provisional; final sort deferred to Plan 03).

### AUDIT-REPORT.md (Consolidated Findings expanded)

Four new performance findings added to the `### Performance Findings` subsection:

- **PF-2:** Elastic alignment is fdars' top performance bottleneck — elastic_cross N=500,M=50
  at **37.82 s** (EXCELLENT confidence, 0% variance); N=500,M=200 **INFEASIBLE** (~700 s/iter).
  Evidence: [bench/p3_elastic_cross_linalg,parallel_run1.txt](bench/p3_elastic_cross_linalg,parallel_run1.txt).
  Feeds: PERF-ELASTIC-BAND (P1).

- **PF-3:** The banded fast-path is opt-in, imposing a measured **4.5–5.7×** default-path
  penalty at stable cells (elastic_cross N=100,M=200: 27.85 s unbanded → 6.16 s banded → 4.5×).
  Evidence: [bench/p3_elastic_cross_linalg,parallel_run1.txt](bench/p3_elastic_cross_linalg,parallel_run1.txt) +
  [bench/p3_elastic_cross_banded_linalg,parallel_run1.txt](bench/p3_elastic_cross_banded_linalg,parallel_run1.txt).
  Feeds: PERF-ELASTIC-BAND / P5-4.

- **PF-4:** Parallelism: heavy loops (karcher) pay back at any N — **~4.73× at 8 threads**
  (N=100,M=50: 1.556 s → 329 ms); light loops only past N≈50. Three sequential loops
  safe to parallelize (CV folds, elastic-FPCA inner, center_columns).
  Evidence: [bench/p5_karcher_linalg,parallel_run1.txt](bench/p5_karcher_linalg,parallel_run1.txt).
  Feeds: PERF-PAR-CV / PERF-PAR-ELFPCA / PERF-PAR-CENTER.

- **PF-5:** faer thin_svd is **1.8–4.1× faster** than nalgebra SVD at real FPCA sizes — primary
  cell N=500,M=200: nalgebra **41.026 ms** vs faer **23.084 ms** → **1.8×**. Zero-copy conversion
  eliminates the `to_dmatrix()` allocation. Numerically equivalent within 1e-10.
  Evidence: [bench/p6_svd_faer_seq_linalg_run1.txt](bench/p6_svd_faer_seq_linalg_run1.txt).
  Feeds: P6-1.

PF-1 (Plan 01 FPCA SVD-dominance finding) is preserved unchanged.

## Evidence Links Verified

All evidence links written in this plan resolve to real files under `.planning/research/bench/`:

| Link | Exists | Key number cited |
|------|--------|-----------------|
| bench/p3_elastic_cross_linalg,parallel_run1.txt | yes | N=500,M=50: 37.82 s; N=100,M=200: 27.85 s; N=500,M=200: INFEASIBLE |
| bench/p3_elastic_cross_banded_linalg,parallel_run1.txt | yes | N=100,M=200: 6.16 s banded → 4.5× speedup |
| bench/p4_dhat_fpca_n500_m200.txt | yes | 21 total_blocks, 3,574,424 total_bytes, 3,531,192 peak_bytes |
| bench/p4_fpca_linalg,parallel_run1.txt | yes | N=1000,M=200: 38.307 ms; N=100,M=50: 213.33 µs |
| bench/p1_cv_linalg,parallel_run1.txt | yes | N=100,M=50 fclassif_cv: 947–952 µs |
| bench/p1_fpca_linalg,parallel_run1.txt | yes | N=500,M=200: 16.155 ms total |
| bench/p5_karcher_linalg,parallel_run1.txt | yes | N=100,M=50 8-thread: 329 ms → 4.73× speedup |
| bench/p6_svd_faer_seq_linalg_run1.txt | yes | N=500,M=200: 23.084 ms |

## Deviations from Plan

None — plan executed exactly as written.

The plan specified that P6-1's consolidated finding was already present from Plan 01. PF-5
(faer SVD win, appended here) provides the complementary faer-comparison angle feeding
the same P6-1 item, while PF-1 (Plan 01) covers the allocation-analysis angle. Both are
present; no duplication exists.

## Known Stubs

None. All 7 new BACKLOG.md item blocks carry real benchmark numbers (no placeholder text or
invented figures). All 4 new AUDIT-REPORT.md performance findings cite real bench artifacts
with exact numbers. No stub patterns detected in the modified files.

## Threat Flags

Docs-only audit deliverable — no code, no new attack surface; no applicable threats.

## Self-Check: PASSED

Files modified:
- [x] `.planning/research/BACKLOG.md` — exists, 7 new item blocks present
- [x] `.planning/research/AUDIT-REPORT.md` — exists, PF-2 through PF-5 present

Commits:
- [x] 95af4482 — feat(09-02): promote all performance draft-backlog slices into full BACKLOG.md items
- [x] 8efa2b44 — feat(09-02): write remaining performance findings into AUDIT-REPORT.md Consolidated Findings

Content checks:
- [x] BACKLOG.md: PERF-ELASTIC-BAND, PERF-FPCA-CLONE, PERF-FPCA-TRUNCSVD, PERF-PAR-CV, PERF-PAR-ELFPCA, PERF-PAR-CENTER, ACC-VALIDATE — all 7 IDs present
- [x] BACKLOG.md: 10 "Proposed direction" fields (P6-1 + 7 new = 8 items with 2 legacy plan items in P6-1 block = actually 10 fields total)
- [x] BACKLOG.md: 8 Ranked Backlog rows (P6-1 + 7 new)
- [x] BACKLOG.md: bench/p3_elastic_cross_linalg,parallel_run1.txt link resolves (file exists)
- [x] BACKLOG.md: bench/p4_dhat_fpca_n500_m200.txt link resolves (file exists)
- [x] BACKLOG.md: No P5-4 duplicate item — one-line cross-reference inside PERF-ELASTIC-BAND only
- [x] AUDIT-REPORT.md: ### Performance Findings subsection present
- [x] AUDIT-REPORT.md: 5 PF-N findings total (PF-1 through PF-5)
- [x] AUDIT-REPORT.md: bench/p3_elastic_cross_linalg,parallel_run1.txt link present
- [x] AUDIT-REPORT.md: bench/p6_svd_faer_seq_linalg_run1.txt link present
- [x] AUDIT-REPORT.md: "top bottleneck" / "primary bottleneck" phrase present (PF-2)
- [x] AUDIT-REPORT.md: PF-1 (Plan 01 finding) preserved unchanged
- [x] No fdars-core/ files modified (docs-only deliverable)
