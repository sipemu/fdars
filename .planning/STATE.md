---
gsd_state_version: 1.0
milestone: v0.18.0
milestone_name: R-Ecosystem Gap Audit
current_phase: 19
current_phase_name: Consolidated Report & Ranked Backlog
status: planning
stopped_at: Roadmap + REQUIREMENTS traceability + STATE written; 7/7 requirements mapped
last_updated: "2026-08-15T18:35:20.591Z"
last_activity: 2026-08-15
last_activity_desc: Phase 16 complete, transitioned to Phase 17
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 4
  completed_plans: 4
  percent: 75
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-13)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against the reference FDA ecosystems — this milestone maps fdars against the R FDA package ecosystem to produce the next evidence-backed, prioritized backlog.
**Current focus:** Phase 16 — R Ecosystem Inventory

## Current Position

Phase: 19 — Consolidated Report & Ranked Backlog
Plan: Not started
Status: Ready to plan
Last activity: 2026-08-15 — Phase 18 complete, transitioned to Phase 19

## Milestone Roadmap (v0.18.0)

Four phases, seven requirements. Audit-only — zero `fdars-core/src/` edits (mirrors v0.14.0). The R FDA ecosystem replaces scikit-fda as the sole yardstick. Deliverables land in `.planning/research/R-AUDIT-REPORT.md` + `.planning/research/R-BACKLOG.md` (distinct from the archived scikit-fda `AUDIT-REPORT.md`/`BACKLOG.md` — do not overwrite).

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 16 — R Ecosystem Inventory | INV-01, INV-02 | Enumerate the R FDA ecosystem capability-first (versioned, package-tagged, area-organized), then design-goal filter into in-scope / out-of-scope with per-area counts. INV-02 depends on INV-01 (same deliverable). Mirrors v0.14.0 Phase 7. |
| 17 — Parity Matrix & Categorization | GAP-01, GAP-02 | fdars-vs-R present/partial/absent matrix (matched by capability, "searched fdars for:" evidence notes), then categorize every gap table-stakes/differentiator/out-of-scope. GAP-02 depends on GAP-01. Depends on Phase 16 (the in-scope inventory is the row set). Mirrors v0.14.0 Phase 8. |
| 18 — Reverse-Parity Strengths Sweep | GAP-03 | Full module-map walk of fdars-core cataloguing R-unique + fdars-ahead capabilities. Independent of the R-side enumeration → parallelizable with Phases 16–17; must complete before Phase 19. |
| 19 — Consolidated Report & Ranked Backlog | RPT-01, RPT-02 | Consolidate R-AUDIT-REPORT.md (methodology + findings + strengths) and produce R-BACKLOG.md (`score = value/√effort`, master ranked table, 7-field promotion blocks). Depends on 16, 17, 18. Mirrors v0.14.0 Phase 9. |

**Execution order:** 16 → 17 → 19, with 18 parallelizable alongside 16–17 (18 walks the fdars codebase, independent of the R survey; must finish before 19).

## Performance Metrics

**Velocity:**

- Total plans completed: 36 (25 in v0.14.0 + 4 in v0.15.0 + 4 in v0.16.0 + 3 in v0.17.0)
- Average duration: — min
- Total execution time: — hours

**By Phase (prior milestones):**

| Phase | Milestone | Plans |
|-------|-----------|-------|
| 01–09 | v0.14.0 | 21 |
| 10 | v0.15.0 | 2 |
| 11 | v0.15.0 | 2 |
| 12 | v0.16.0 | 1 |
| 13 | v0.16.0 | 3 |
| 14 | v0.17.0 | 2 |
| 15 | v0.17.0 | 1 |

**Recent Trend:**

- Last 5 plans: 13-01, 13-02 (v0.16.0), 14-01, 14-02, 15-01 (v0.17.0) — all completed + verified
- Trend: —

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Relevant to current work (v0.18.0 audit):

- v0.18.0 is an **audit-only** milestone — zero `fdars-core/src/` edits (mirrors v0.14.0). Deliverables are a report + backlog, not code.
- The **R FDA ecosystem** replaces scikit-fda as the sole comparison yardstick — the actionable scikit-fda backlog is exhausted (v0.15.0–v0.17.0). Do NOT re-audit scikit-fda.
- Phase numbering **continues** from v0.17.0 (ended at Phase 15) → v0.18.0 starts at Phase 16.
- There is **no research SUMMARY.md** for this milestone — the R survey IS the milestone's phase work (Phase 16 does the enumeration).
- Deliverable files are **new and distinctly named**: `.planning/research/R-AUDIT-REPORT.md` + `.planning/research/R-BACKLOG.md`. Do NOT overwrite the archived scikit-fda `AUDIT-REPORT.md` / `BACKLOG.md` (reference-only templates).
- Reuse the v0.14.0 deliverable conventions: capability-first enumeration (fit/predict/transform collapsed), verdict rubric (present/partial/absent), category rubric (table-stakes/differentiator/out-of-scope), `score = value/√effort` ranking with 7-field promotion blocks.
- Scope fence (mirrors v0.14.0): numeric algorithms + API ergonomics **in scope**; plotting/visualization + data/IO **out of scope**. The numeric underpinnings of graphical diagnostics (e.g. outliergram/MS-plot statistics) may be in-scope; the plots themselves are not.
- Roadmap grouping: INV-01/INV-02 → Phase 16 (same deliverable, INV-02 filters INV-01's output). GAP-01/GAP-02 → Phase 17 (GAP-02 categorizes GAP-01's gaps). GAP-03 → Phase 18 (fdars-side module-map walk, independent of the R survey → parallelizable). RPT-01/RPT-02 → Phase 19 (consolidation, depends on 16/17/18).

Conventions carried from prior milestones (relevant even to an audit):

- The fdars module map lives in `.planning/codebase/` — use it for the GAP-03 reverse-parity walk (per-module coverage), and for the "searched fdars for:" evidence notes in GAP-01 (map by capability, not API name).
- TMPDIR=/home/simonm/.cache/fdars-bench-tmp required if any grep/build/doctest linking is needed; /tmp tmpfs exhaustion causes bogus "No space left" (MEMORY.md pointer). Audit phases are mostly analysis/write, so build pressure is low.

### Pending Todos

None yet.

### Blockers/Concerns

- **R package versions must be captured explicitly** (INV-01 requires version tags). If R / the packages are not installed locally, versions come from CRAN metadata / package DESCRIPTION at survey time — record the exact version used per package so the inventory is reproducible.
- No runtime R benchmark is in scope — this is a capability/API gap comparison only (R is interpreted; not a meaningful perf baseline). Do not attempt a cross-language speed contest.
- Local main is ahead of origin/HEAD → harness worktrees may fork the wrong base; GSD phases fall back to sequential no-worktree dispatch (MEMORY.md pointer).
- /tmp tmpfs exhaustion can block pre-commit doctest linking with bogus "No space left" — use `--no-verify` for docs and free /tmp before executing (MEMORY.md pointer).

## Deferred Items

v2 backlog items deferred at v0.18.0 definition (2026-08-13): IMPL-* (implementation of any R-parity gap found — deferred to future milestones, the point of the backlog), ACC-VALIDATE (fdars-vs-reference numerical-accuracy validation, could extend to R references) — see REQUIREMENTS.md v2 section.

Advisory tech-debt carried from v0.15.0 (not v0.18.0 work): weakened MEWMA test assertion; `fix_svd_signs` NaN no-op; over-broad Phase 11 test name; Phase 10 & 11 VALIDATION.md `draft` (Nyquist TODO). Phase 14 & 15 VALIDATION.md also remain `draft` (Nyquist TODO, v0.17.0).

## Session Continuity

Last session: 2026-08-13 — created v0.18.0 roadmap (Phases 16–19)
Stopped at: Roadmap + REQUIREMENTS traceability + STATE written; 7/7 requirements mapped
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 16` (R Ecosystem Inventory — INV-01, INV-02).
- Phase 18 (reverse-parity strengths sweep) is independent of the R survey and may be planned/executed in parallel with Phases 16–17.
