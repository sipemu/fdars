---
phase: 2
slug: static-hot-path-analysis
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-07
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> **Audit-only phase:** produces markdown analysis sections in `.planning/research/AUDIT-REPORT.md`, no code changes. Verification is source-grep-based (no test framework applies).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | None — Phase 2 is analysis-only; the deliverable is markdown appended to `.planning/research/AUDIT-REPORT.md`. No `fdars-core` code changes, no new tests. |
| **Config file** | none |
| **Quick run command** | `grep -c "to_dmatrix" .planning/research/AUDIT-REPORT.md` (per-task provenance grep) |
| **Full suite command** | Manual checklist against the 4 ROADMAP success criteria (SC1–SC4) using the greps in the Per-Task Verification Map below |
| **Estimated runtime** | < 5 seconds (grep only) |

---

## Sampling Rate

- **After every task commit:** Run the task's provenance grep from the Per-Task Verification Map (confirms the section text landed with real file:line anchors).
- **After every plan wave:** Run all SC1–SC4 greps against `AUDIT-REPORT.md`.
- **Before `/gsd-verify-work`:** All four SC greps return their expected counts and each cited `file:line` still exists in `fdars-core/src/`.
- **Max feedback latency:** 5 seconds.

---

## Per-Task Verification Map

> Task IDs are provisional (finalized by the planner). Each row maps a success criterion to a deterministic grep.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 2-01-xx (SC1) | 01 | 1 | PERF-01 | — | N/A (read-only analysis) | grep | `grep -c "O(n" .planning/research/AUDIT-REPORT.md` (≥ 6 module rows) | ✅ (AUDIT-REPORT.md exists) | ⬜ pending |
| 2-01-xx (SC2) | 01 | 1 | PERF-01 | — | N/A | grep | `grep -c "to_dmatrix" .planning/research/AUDIT-REPORT.md` (≥ 8 production SVD sites) | ✅ | ⬜ pending |
| 2-01-xx (SC3) | 01 | 1 | PERF-01 | — | N/A | grep | `grep -Ec "sequential\|gap candidate\|already parallel" .planning/research/AUDIT-REPORT.md` | ✅ | ⬜ pending |
| 2-01-xx (SC4) | 01 | 1 | PERF-01 | — | N/A | grep | `grep -Ec "parallel-gated\|sequential\|linalg-gated\|always" .planning/research/AUDIT-REPORT.md` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements.* No test files to create — the sole prerequisite artifact (`.planning/research/AUDIT-REPORT.md`) already exists from Phase 1. Verification is manual grep-based against the appended sections.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Every complexity-table row cites ≥1 real `file:line` anchor that still exists in source | PERF-01 (SC1) | Provenance cannot be asserted by a count alone — a human/verifier must confirm each cited line exists (`grep -n` the file) and the Big-O matches the loop nesting | For each row, run `grep -n "<function>" fdars-core/src/<file>` and confirm the line is within the cited range |
| Allocation list distinguishes the 8 production `to_dmatrix()` SVD sites from the 14 `DMatrix::from_column_slice` basis sites, and excludes the `#[cfg(test)]` site at `matrix.rs:682` | PERF-01 (SC2) | The ROADMAP's "8 sites" claim needed correction; the two categories have different optimization paths and must not be conflated | Cross-check the report's site table against RESEARCH.md §2A/§2B; confirm `matrix.rs:682` is not listed as a production hotspot |
| No loop labeled "sequential/gap" is actually wrapped in a parallelism macro (no false positives feeding Phase 5) | PERF-01 (SC3, SC4) | Requires cross-referencing each named loop against the `iter_maybe_parallel!` grep inventory | For each gap entry, `grep -n "iter_maybe_parallel\|slice_maybe_parallel" fdars-core/src/<file>` near the cited line; a match means it must be labeled "already parallel", not a gap |

---

## Validation Sign-Off

- [ ] All tasks have an automated grep verify or a documented manual verification
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (N/A — no Wave 0 needed)
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
