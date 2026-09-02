---
gsd_state_version: 1.0
milestone: v0.31.0
milestone_name: Multi-Ecosystem Gap Audit
current_phase: 53
current_phase_name: Consolidation & Backlog
status: planning
stopped_at: Phase 52 complete, ready to plan Phase 53
last_updated: "2026-09-02T06:36:18.063Z"
last_activity: 2026-09-02
last_activity_desc: Phase 52 complete, transitioned to Phase 53
state_head: 2238b5a54b141e020e83e20525a6aa5a9622892b
progress:
  total_phases: 2
  completed_phases: 1
  total_plans: 4
  completed_plans: 4
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-01)

**Core value:** Produce an evidence-backed picture of what fdars is missing relative to four fresh reference ecosystems (MATLAB FDA, Julia FDA, tidyfun/refund, Python-beyond-scikit-fda), turned into a single prioritized, de-duplicated, GSD-ready backlog — so future milestones target the highest-leverage net-new capability work first.
**Current focus:** Phase 52 — Ecosystem Surveys (four independent parallel surveys)

## Current Position

Phase: 53 of 53 (Consolidation & Backlog)
Plan: Not started
Status: Ready to plan
Last activity: 2026-09-02 — Phase 52 complete, transitioned to Phase 53

Progress: [░░░░░░░░░░] 0%

## Milestone Roadmap (v0.31.0)

Two phases, 7 requirements — the next-yardstick gap audit now that both prior parity backlogs (scikit-fda v0.14.0, R core v0.18.0) are exhausted. **Audit-only** (report + backlog, zero `fdars-core/src/` edits). Mirrors the v0.14.0 / v0.18.0 audit shape: parallel enumeration/parity surveys, then a consolidated report + ranked backlog + completeness gate. Phase numbering continues from v0.30.0 (ended at 51) → Phase 52.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 52 — Ecosystem Surveys | MAT-01, JUL-01, TDY-01, PYX-01 | Four mutually-independent surveys (parallel plans): enumerate versioned capability surface → map fdars present/partial/absent → de-dup vs shipped + `BACKLOG.md` + `R-BACKLOG.md` → emit net-new gap list. TDY-01 refund-only-if-not-in-v0.18.0; PYX-01 excludes scikit-fda. Audit-only. |
| 53 — Consolidation & Backlog | RPT-01, RPT-02, RPT-03 | Depends on all four surveys. RPT-01 report (`GAP-AUDIT-REPORT.md`: methodology, per-ecosystem findings, cross-ecosystem convergence, reverse-parity strengths) → RPT-02 ranked backlog (`GAP-BACKLOG.md`: value/√effort, promotion-ready blocks) → RPT-03 de-dup + completeness gate (gate LAST). Audit-only. |

**Execution order (dependency-driven):** 52 → 53. Phase 52's four survey gap-lists are the raw material Phase 53 merges/ranks/de-dups. Within 53 the internal order is RPT-01 → RPT-02 → RPT-03 (gate last).

## Performance Metrics

**Velocity:**

- Total plans completed: 95 (across v0.14.0–v0.30.0)
- Average duration: — min
- Total execution time: — hours

**By Phase (prior milestones):**

| Phase | Milestone | Plans |
|-------|-----------|-------|
| 01–09 | v0.14.0 | 21 |
| 10–45 | v0.15.0–v0.29.0 | 63 |
| 46–51 | v0.30.0 | 23 |
| 52–53 | v0.31.0 | 0/7 (TBD) |

**Recent Trend:**

- Last milestone: v0.30.0 phases 46–51 (23 plans) — audit **tech_debt** 13/13, archived. First internally-driven perf/consolidation pass.
- Trend: v0.31.0 returns to **audit shape** (like v0.14.0 / v0.18.0) — a fresh external gap survey producing a report + backlog, zero code. Deliverables not features; plan sizing is document-scoped.

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.31.0):

- **Next-yardstick audit** — both prior parity backlogs (scikit-fda v0.14.0, R core v0.18.0) are exhausted, so v0.31.0 measures fdars against four *fresh* ecosystems: MATLAB FDA, Julia FDA, tidyfun/refund, Python-beyond-scikit-fda.
- **Audit-only fence** — zero `fdars-core/src/` edits across the whole milestone; deliverables are markdown documents in `.planning/research/`.
- **Distinct filenames** — new deliverables are `GAP-AUDIT-REPORT.md` and `GAP-BACKLOG.md`. Do NOT overwrite the existing `AUDIT-REPORT.md` / `BACKLOG.md` (v0.14.0) or `R-AUDIT-REPORT.md` / `R-BACKLOG.md` (v0.18.0).
- **Hard de-dup rule** — net-new gaps only: every backlog item must be verified absent from shipped fdars AND absent from both prior backlogs. Value is in net-new gaps only.
- **No git tag / no crate publish** — crate is unchanged; a `v*` tag would publish a phantom version (audit-milestone convention — MEMORY.md pointer `audit-milestone-no-git-tag`).
- **Scope exclusions** — no plotting/visualization parity, no data/IO parity; no re-audit of scikit-fda or the core R FDA ecosystem (refund only where NOT captured in v0.18.0; PYX-01 excludes scikit-fda).
- **Phase numbering continues** — v0.30.0 ended at Phase 51 → v0.31.0 starts at Phase 52. No reset.
- **7 requirements → 2 phases** (fine granularity, mirroring the v0.14.0/v0.18.0 audit shape): four independent surveys as parallel plans in Phase 52; RPT-01/02/03 consolidation as sequenced plans in Phase 53. All 7 mapped, no orphans.

### Pending Todos

- **Migrate `fdars-r` R wrapper to use the `FdMatrix` API** (issue `fdars-j75`) — not this milestone (audit-only, no code); carried forward.

### Blockers/Concerns

- Local `main` and `origin/main` are in sync post-v0.30.0 release; prior worktree-base-divergence blocker is currently quiescent. GSD phases have executed inline (not via gsd-executor subagents) in recent milestones — an audit milestone is document-only so subagent build-watchdog issues do not apply.
- **De-dup rigor is the main risk this milestone** — the value gate is "genuinely net-new". Every candidate gap must be checked against the shipped-capabilities list (PROJECT.md Validated section, 40+ entries) AND both prior backlogs before it earns a `GAP-BACKLOG.md` row. RPT-03 is the formal gate.

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| Release | REL-01 done — `fdars-core` 0.30.0 published (tag `v0.30.0`, 2026-09-01) | Done | v0.30.0 | v0.30.0 |
| API-breaking | APIB-01 — breaking removal of the 6 `#[deprecated]` forms from v0.30.0 | Deferred | v0.30.0 | future 1.0-readiness |
| Implementation | Implementing any gap found in this audit — drawn top-first from `GAP-BACKLOG.md` | Deferred | v0.31.0 | future milestone |

## Session Continuity

Last session: 2026-09-02T00:00:00.000Z
Stopped at: Phase 52 complete, ready to plan Phase 53
Resume file: None

## Operator Next Steps

- Plan the first phase: `/gsd-plan-phase 52`
