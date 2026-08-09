# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v0.14.0 — Performance & scikit-fda Gap Audit

**Shipped:** 2026-08-09
**Phases:** 9 | **Plans:** 21 | **Tasks:** 25

### What Was Built
- `.planning/research/AUDIT-REPORT.md` — consolidated audit report: methodology (feature-flag matrix + infra-vs-code triage), 5 performance findings (PF-1..5, each bench-linked), 82 in-scope scikit-fda gaps, and 30 fdars-exclusive strengths.
- `.planning/research/BACKLOG.md` — 32-item value-ranked backlog (`score = value/√effort`), 34 seven-field promotion-ready blocks, completeness gate passed.
- A reproducible criterion benchmark corpus (~51 artifacts under `.planning/research/bench/`) across the 4-combo feature matrix.

### What Worked
- **Tracer-first phase structure.** Every phase opened with a Wave-1 "tracer" plan that proved the measure→artifact→report→backlog pipeline on ONE cell before expanding. Caught schema issues early and made later waves mechanical.
- **Audit-only discipline held.** All 9 phases produced analysis artifacts with zero `fdars-core/src/` edits — scope never leaked into implementation.
- **Milestone audit earned its keep.** The pre-archive `/gsd-audit-milestone` + integration checker caught a real (if cosmetic) defect — a "6 P1 items" miscount contradicted by a 5-item table — that all 9 phase verifications had passed over.
- **Evidence traceability.** Every consolidated finding links back to a real bench artifact with matching numbers; every backlog item to a report section.

### What Was Inefficient
- **`/tmp` tmpfs exhaustion blocked every hook-run commit.** Doctests link in a small `/tmp` and fail with a bogus "No space left"; all docs-only `.planning/` commits had to use `--no-verify`. Recurring friction (see MEMORY.md).
- **Worktree base divergence forced sequential execution.** Local `main` is ahead of `origin/HEAD`, so harness worktrees fork the wrong base (#683). Every phase auto-degraded to sequential single-tree dispatch — correct, but no parallelism benefit.
- **SUMMARY `requirements_completed` frontmatter was under-filled.** Most SUMMARYs left it blank, and the milestone-complete accomplishment auto-extraction pulled junk one-liners (`fdars-core/Cargo.toml`, "8 rows total:") that needed manual curation.

### Patterns Established
- **Tracer plan → expansion wave(s)** per phase, all appending to shared deliverable files (AUDIT-REPORT.md, BACKLOG.md) — inherently sequential, handled cleanly on the main tree.
- **7-field backlog item contract** (location, current cost/gap, root cause, proposed direction, severity P1/P2/P3, effort S/M/L, evidence link) + `value/√effort` ranking — reusable for any future audit.
- **Capability-first parity mapping** (not API-name counting) with "searched fdars for:" notes and known-bug accuracy flags.

### Key Lessons
1. **Run `/gsd-audit-milestone` before `/gsd-complete-milestone`** — phase-level verification does not catch cross-artifact numeric inconsistencies; the milestone audit does.
2. **Fill SUMMARY `requirements_completed` frontmatter during execution** — it feeds the milestone accomplishment list and the 3-source requirement cross-reference; blank frontmatter degrades both.
3. **On this machine, `/tmp` must be freed before hook-verified commits**, or use `--no-verify` for docs-only `.planning/` changes (documented exception).
4. **Set `worktree.baseRef:"head"`** if parallel worktree execution is wanted while `main` is ahead of `origin` — otherwise expect sequential auto-degrade.

### Cost Observations
- Model mix: orchestration on Opus; executors + verifier on Sonnet; integration checker on Haiku.
- Notable: sequential single-tree dispatch throughout (worktree base divergence) — no parallel-wave speedup this milestone.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Phases | Plans | Key Change |
|-----------|--------|-------|------------|
| v0.14.0 | 9 | 21 | First GSD milestone: tracer-first phases, audit-only scope, milestone-audit gate before archive |

### Cumulative Quality

| Milestone | Deliverables | Requirements | Zero-src-edit |
|-----------|--------------|--------------|---------------|
| v0.14.0 | AUDIT-REPORT.md + BACKLOG.md | 13/13 satisfied | yes (audit-only) |

### Top Lessons (Verified Across Milestones)

1. Milestone-level audit catches cross-artifact defects that phase verification misses. *(v0.14.0 — revisit as more milestones ship.)*
