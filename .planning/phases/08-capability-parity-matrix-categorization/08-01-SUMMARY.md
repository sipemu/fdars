---
phase: 08-capability-parity-matrix-categorization
plan: "01"
subsystem: research-documentation
tags: [scikit-fda, capability-parity, audit, preprocessing, verdict-rubric, categorization-rubric, accuracy-flag, tracer]
status: complete

dependency_graph:
  requires:
    - ".planning/research/AUDIT-REPORT.md ## Phase 7 section (scikit-fda capability inventory — the fixed left column; §Area: Preprocessing in-scope rows)"
    - ".planning/codebase/STRUCTURE.md §Where to Add New Code (scikit-fda-task → fdars-module join)"
    - ".planning/codebase/CONCERNS.md §Known Bugs (B-spline round-trip/CV #33 commit 2fb6d3c9; elastic-alignment level encoding #34 commit 6ed62398)"
    - ".planning/research/PITFALLS.md (Pitfall 9 capability-not-API-name; Pitfall 11 searched-note + partial≠missing; Pitfall 12 accuracy flag)"
    - ".planning/phases/08-capability-parity-matrix-categorization/08-CONTEXT.md (D-01/D-01a verdict rubric, D-03 category rubric, D-02 accuracy convention, D-05 single-file)"
  provides:
    - "## Phase 8 — Capability Parity Matrix & Categorization section in AUDIT-REPORT.md (D-05 single-file convention)"
    - "### Verdict Rubric (D-01) — three-value present/partial/absent definitions, D-01a partial retained, searched-note + accuracy-flag conventions"
    - "### Categorization Rubric (D-03) — table-stakes / differentiator / out-of-scope definitions with Pitfall-14 separated-count note"
    - "### Area: Preprocessing — Parity — fully-worked tracer table (39 in-scope rows, all columns populated), proving the row schema for reuse"
    - "Corrected Preprocessing in-scope row count: 39 (recount) supersedes the stale Phase-7 header '29'"
  affects:
    - ".planning/research/AUDIT-REPORT.md (Plans 02/03 append the remaining five area tables, separated counts, reverse-parity sweep, and drafted backlog into this same section, reusing this proven schema)"

tech_stack:
  added: []
  patterns:
    - "Parity row schema: Area | Task | Capability | Verdict | Category | Accuracy? | fdars equivalent | Confidence"
    - "Verdict source-confirmed by grep/read of named fdars-core/src module (STRUCTURE.md points, source confirms) — not decided from the module map alone"
    - "Capability-mapped not API-name-mapped (Pitfall 9): a different call shape (builder + single-call) counts as present"
    - "Mandatory 'searched fdars for: [behavior]. Closest match: [fn/module]. Verdict: [reason]' note on every partial/absent row (Pitfall 11)"
    - "Known-bug rows read 'present — accuracy NOT verified' with CONCERNS.md + fix-commit citation, never a bare check (D-02, Pitfall 12)"
    - "Recount-supersedes-stale-header convention for per-area in-scope counts"

key_files:
  created: []
  modified:
    - ".planning/research/AUDIT-REPORT.md (appended ## Phase 8 section: verdict rubric + category rubric + Preprocessing parity table, 181 insertions)"

decisions:
  - "Task 1 checkpoint resolved to adopt-d01: three-value verdict rubric (present/partial/absent), partial bucket retained per D-01a (matches locked 08-CONTEXT.md default) — confirmed by the user via the orchestrator interactive prompt"
  - "Task 3 human-verify checkpoint: tracer schema approved by the user via the orchestrator interactive prompt (AskUserQuestion); the user reviewed AUDIT-REPORT.md §Phase 8 and accepted the schema for reuse across the remaining five areas"
  - "Preprocessing in-scope row count corrected 29→39: the Phase-7 area header 'In-scope count: 29 rows' is a stale undercount; direct recount of the three Phase-7 task-grouping tables yields 39 in-scope rows (13 Smoothing + 12 Registration/Alignment + 14 Dim-Reduction, minus 2 out-of-scope plumbing). The authoritative Phase-7 tables — not the header — are the fixed left column; all 39 mapped. User explicitly accepted this recount at the human-verify checkpoint"
  - "Task 2 committed with --no-verify per the documented MEMORY.md exception: /tmp tmpfs exhaustion causes SIGBUS in the cargo doctest linker on docs-only commits (infra-not-code; a single isolated doctest passes cleanly)"

metrics:
  duration_minutes: 8
  completed_date: "2026-08-09"
  tasks_completed: 3
  tasks_total: 3
  files_changed: 1

actuals:
  tokens: 18000
  tasks: 3
  commits: 1

requirements-completed: [GAP-02, GAP-03]

coverage:
  - id: D1
    description: "## Phase 8 section with D-01 verdict rubric and D-03 categorization rubric stated once, driving every parity row"
    requirement: "GAP-03"
    verification:
      - kind: other
        ref: "grep -q '^## Phase 8 — Capability Parity Matrix & Categorization' && grep -q '### Verdict Rubric' && grep -q '### Categorization Rubric' .planning/research/AUDIT-REPORT.md"
        status: pass
    human_judgment: false
  - id: D2
    description: "Preprocessing area parity-mapped end-to-end (39 in-scope rows), every column populated, mapped by capability not API name; searched-notes on every partial/absent row; known-bug rows accuracy-flagged with fix-commit citations"
    requirement: "GAP-02"
    verification:
      - kind: other
        ref: "grep -q '### Area: Preprocessing — Parity' && grep -q 'accuracy NOT verified' && grep -Eq '2fb6d3c9|6ed62398' && grep -q 'searched fdars for' .planning/research/AUDIT-REPORT.md"
        status: pass
    human_judgment: true
    rationale: "The tracer schema is a costly-reversibility commitment reused across five downstream areas; correctness of capability-mapping and verdicts requires human judgment. User verified and approved at the Task 3 human-verify checkpoint."
  - id: D3
    description: "Zero fdars-core/src edits (audit-only milestone enforced)"
    verification:
      - kind: other
        ref: "git diff --name-only HEAD~1 HEAD -- fdars-core/src (empty)"
        status: pass
    human_judgment: false
---

# Phase 08 Plan 01: Preprocessing Parity Tracer Summary

**One-liner:** Phase 8 parity apparatus established end-to-end — D-01 verdict rubric (present/partial/absent), D-03 categorization rubric, and a fully-worked 39-row Preprocessing parity table (≈15 present / 9 partial / 15 absent), all source-grep-confirmed with searched-notes and known-bug accuracy flags — proving the row schema for the remaining five areas.

## Performance

- **Duration:** ~8 min (across checkpoint pauses)
- **Completed:** 2026-08-09
- **Tasks:** 3 (1 decision checkpoint, 1 tracer, 1 human-verify checkpoint)
- **Files modified:** 1

## Accomplishments
- Appended the top-level `## Phase 8 — Capability Parity Matrix & Categorization` section to AUDIT-REPORT.md (D-05 single-file convention).
- Wrote the `### Verdict Rubric (D-01)` subsection: three-value present/partial/absent definitions, D-01a partial-retention rationale, the Pitfall-11 searched-note convention, the D-02/Pitfall-12 accuracy-flag convention (naming the three Preprocessing-touching known-bug areas + fix commits), and the HIGH/MEDIUM confidence convention.
- Wrote the `### Categorization Rubric (D-03)` subsection: table-stakes / differentiator / out-of-scope definitions with the Pitfall-14 separated-count note.
- Wrote the `### Area: Preprocessing — Parity` tracer table across three task groupings (Smoothing, Registration/Alignment, Dimensionality-Reduction/Feature-Construction): one row per Phase-7 in-scope Preprocessing capability, every column populated, all verdicts source-confirmed against the named `fdars-core/src` modules.
- Two known-bug rows (`BasisSmoother` #33, `FisherRaoElasticRegistration` #34) correctly read "present — accuracy NOT verified" with CONCERNS.md + fix-commit citations (`2fb6d3c9`, `6ed62398`) — no bare check-marks.
- Zero `fdars-core/src` edits (audit-only milestone enforced).

## Task Commits

1. **Task 1: Confirm D-01 three-value verdict rubric** — decision `adopt-d01` (no commit; blocking decision checkpoint)
2. **Task 2: Seed Phase 8 section + rubrics, parity-map Preprocessing** — `26ad8199` (docs)
3. **Task 3: Human verifies tracer schema** — approved (blocking human-verify checkpoint; no code change)

**Plan metadata:** finalized by orchestrator (this SUMMARY + STATE/ROADMAP updates).

## Files Created/Modified
- `.planning/research/AUDIT-REPORT.md` — appended the `## Phase 8` section (verdict rubric + category rubric + Preprocessing parity table; 181 insertions).

## Decisions Made
- **adopt-d01** (Task 1): three-value verdict rubric with partial retained, matching the locked 08-CONTEXT.md default. User-confirmed.
- **Schema approved** (Task 3): user verified AUDIT-REPORT.md §Phase 8 and approved the row schema for reuse across the remaining five areas.
- See Deviations for the 29→39 recount and the --no-verify commit.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Stale-count correction] Preprocessing in-scope row count 29 → 39**
- **Found during:** Task 2 (parity mapping)
- **Issue:** The plan and the Phase-7 area header state "29 in-scope rows", but a direct recount of the three Phase-7 Preprocessing task-grouping tables yields 39 in-scope rows (13 Smoothing + 12 Registration/Alignment + 14 Dim-Reduction, minus 2 out-of-scope plumbing rows).
- **Fix:** Mapped all 39 authoritative Phase-7 table rows (the tables are the fixed left column per the 1:1 join), and documented the header "29" as a stale undercount using the same recount-supersedes-header convention already recorded for the Representation area.
- **Files modified:** `.planning/research/AUDIT-REPORT.md`
- **Verification:** Row-count note added inline; user explicitly accepted the recount at the Task 3 human-verify checkpoint.
- **Committed in:** `26ad8199`

**2. [MEMORY.md documented exception] Task 2 committed with --no-verify**
- **Found during:** Task 2 commit
- **Issue:** Pre-commit doctest hook failed with the documented `/tmp` tmpfs-exhaustion SIGBUS (53/129 doctests failed on link; a single isolated doctest passes cleanly — confirmed infra-not-code).
- **Fix:** Used `--no-verify` for this docs-only commit per the sanctioned MEMORY.md exception.
- **Files modified:** none beyond the doc.
- **Committed in:** `26ad8199`

---

**Total deviations:** 2 (1 stale-count correction, 1 documented infra exception)
**Impact on plan:** Neither affects scope. The recount makes the tracer more complete (39 vs 29 rows); the --no-verify is the standard project workaround for a known infra flake on docs-only commits. No fdars-core/src edits.

## Issues Encountered

**Orchestrator-finalized (checkpoint channel limitation).** The Task 3 blocking human-verify checkpoint could not be closed by the executor subagent: the subagent's safety rules treat any orchestrator-relayed approval as non-authoritative, but a subagent has no direct channel to the user — every human signal in this runtime arrives via the orchestrator. The user did genuinely approve the schema via the orchestrator's interactive prompt (AskUserQuestion → "Approve — unlock Plans 02/03"), and explicitly accepted the 29→39 recount. Because the executor was structurally unable to accept that approval, the orchestrator finalized the plan (this SUMMARY, STATE.md, ROADMAP.md) after genuine user sign-off. The substantive deliverable (the parity table) was already complete and committed by the executor in `26ad8199`.

## User Setup Required
None — no external service configuration required.

## Next Phase Readiness
- The row schema is proven and user-approved; Plan 02 can expand it to the remaining five areas (Representation, Exploratory, ML, Inference, Misc) using the identical column layout and note conventions.
- Plan 02 inherits the corrected recount convention: recount the Phase-7 tables directly, do not trust stale per-area header counts.
- No blockers.

---
*Phase: 08-capability-parity-matrix-categorization*
*Completed: 2026-08-09*
