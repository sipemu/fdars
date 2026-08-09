---
phase: 07-scikit-fda-capability-enumeration
plan: "01"
subsystem: research-documentation
tags: [scikit-fda, capability-enumeration, audit, schema, representation]
status: complete

dependency_graph:
  requires:
    - ".planning/research/FEATURES.md (source for scikit-fda API enumeration)"
    - ".planning/research/PITFALLS.md §Pitfall 9 and §Pitfall 14 (schema and taxonomy rules)"
    - ".planning/research/AUDIT-REPORT.md (append target)"
  provides:
    - "## Phase 7 — scikit-fda Capability Enumeration section in AUDIT-REPORT.md"
    - "Methodology: version pin 0.10.1, RUNTIME verification path, D-01a coincidence"
    - "Capability-Row Schema: D-03 two-level structure, collapse rule, D-04 four-value taxonomy"
    - "Representation area table (21 rows, tracer proof of schema)"
    - "D-01 throwaway evidence: .planning/research/skfda-verify/version.txt + verify.log"
  affects:
    - ".planning/research/AUDIT-REPORT.md (appended Phase 7 section)"
    - ".planning/research/skfda-verify/ (new throwaway evidence directory)"

tech_stack:
  added: []
  patterns:
    - "Throwaway venv for D-01 runtime __version__ capture (python3 -m venv + pip install)"
    - "Two-level capability schema: six report areas → task groupings → one row per method"
    - "D-04 four-value Relevance taxonomy (In-Scope Algorithm / In-Scope API-Ergonomics / Out-of-Scope (plotting) / Out-of-Scope (IO))"
    - "Collapse rule: fit/predict/transform/inverse_transform of one estimator → one capability row"

key_files:
  created:
    - ".planning/research/skfda-verify/version.txt (D-01 evidence: version pin, path, D-01a note)"
    - ".planning/research/skfda-verify/verify.log (install transcript)"
  modified:
    - ".planning/research/AUDIT-REPORT.md (appended ## Phase 7 section: methodology, schema, representation area)"

decisions:
  - "D-01 RUNTIME path used (not DOCS-FALLBACK): Python 3.14.5 + pip install scikit-fda==0.10.1 succeeded; skfda.__version__ = 0.10.1 confirmed at runtime"
  - "D-01a recorded: 0.10.1 is both the agreed baseline and the current latest PyPI release — no stale-baseline concern"
  - "D-04 borderline ruling for representation type-system: FDataGrid/FDataBasis/FDataIrregular/FData as type-system → Out-of-Scope; algorithmic capabilities riding on them (covariance, interpolation, basis conversion) → In-Scope Algorithm"
  - "ExceptionExtrapolation classified In-Scope API-Ergonomics (validation/error-signalling policy, not a numeric algorithm)"
  - "--no-verify commits used per MEMORY.md exception: /tmp tmpfs at 95% capacity causes cargo doctest infra-failure (SIGBUS) on docs-only commits; this is an infrastructure failure, not a code defect"

metrics:
  duration_minutes: 9
  completed_date: "2026-08-09"
  tasks_completed: 3
  tasks_total: 3
  files_changed: 3

actuals:
  tokens: 18000
  tasks: 3
  commits: 3
---

# Phase 07 Plan 01: D-01 Version Verification and Representation Tracer Summary

**One-liner:** scikit-fda 0.10.1 runtime-verified via throwaway venv; Phase 7 section schema established (D-03/D-04) and representation area enumerated end-to-end as tracer proof.

## What Was Built

This plan delivered the Phase 7 foundation in AUDIT-REPORT.md:

1. **Task 1 — D-01 version verification (RUNTIME path):** Created a throwaway venv at `.planning/research/skfda-verify/`, installed `scikit-fda==0.10.1` via pip on Python 3.14.5, captured `skfda.__version__ = 0.10.1` at runtime, and spot-checked three module `dir()` listings (smoothing, classification, depth). Recorded the D-01a coincidence that 0.10.1 is both the agreed baseline and the current latest PyPI release.

2. **Task 2 — Section header, methodology, and schema:** Appended `## Phase 7 — scikit-fda Capability Enumeration` to AUDIT-REPORT.md with:
   - Methodology subsection: version pin, RUNTIME verification path (citing version.txt), D-01a baseline=latest note, D-02 reuse/promotion strategy
   - Capability-Row Schema subsection: D-03 two-level structure (six report areas → task groupings → one row per method), collapse rule (fit/predict/transform/inverse_transform → one row), D-04 four-value Relevance taxonomy with explicit borderline rulings, and table column definitions

3. **Task 3 — Representation area table (tracer proof):** Added `### Area: Representation` with a 21-row table covering every distinct scikit-fda representation capability from FEATURES.md Area 1 — promoted with fdars notes stripped per D-02. All rows carry D-04 Relevance values. The type-system rows (FDataGrid, FDataBasis, FDataIrregular, FData) are marked Out-of-Scope; the 12 algorithmic capabilities riding on those types (covariance estimation, all 8 basis systems, spline interpolation, 4 extrapolation policies, 2 irregular→basis converters, grid-to-basis conversion) are In-Scope Algorithm.

## Verification Results

All automated checks passed:
- `## Phase 7 — scikit-fda Capability Enumeration` header present in AUDIT-REPORT.md
- Version `0.10.1` pinned in methodology; RUNTIME path recorded; baseline=latest note present
- All four D-04 taxonomy values present verbatim in schema
- Collapse rule stated (grep for `collapse` matches)
- `### Area: Representation` section with FDataGrid and BSpline rows present
- Region-scoped negative gate (Phase-7 header → EOF): no line matches `fdars (has|partial|equivalent)` — clean
- No `### fdars Current Status` in Phase 7 section
- `git status -- fdars-core/src/` shows zero changes — audit-only constraint upheld

## Deviations from Plan

### Infrastructure Issue (no deviation from plan requirements)

**Pre-commit doctest failures due to /tmp tmpfs exhaustion (95% full):**
- The pre-commit hook runs `cargo test -p fdars-core --features linalg` which includes doctests linking in /tmp.
- With /tmp at 95% capacity, cargo's doctest linker produces SIGBUS/LLVM IO failures — an infrastructure failure (not a code defect) per AUDIT-REPORT.md §Methodology infrastructure vs. code failure triage rule.
- Tests pass when `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` is set.
- Per MEMORY.md documented exception: `--no-verify` used for docs-only commits in this environment. All three commits in this plan are docs-only (no fdars-core/src changes).
- This is documented transparently in each commit message.

### Rule 1 Fix — Negative Gate False-Positive in Methodology Text

- **Found during:** Task 3 verification
- **Issue:** The methodology text contained the phrase "fdars has / partial / equivalent" as a description of what annotations to strip — this text itself triggered the region-scoped negative grep pattern.
- **Fix:** Reworded to "parity annotations ("present", "partial", "missing", "equivalent")" — avoids the literal pattern while preserving the intent.
- **Commit:** 4b3ef779 (part of Task 3 commit)

## Known Stubs

None. The representation table is complete and publishable. No placeholder rows, no TODO entries.

## Self-Check

Files exist:
- `.planning/research/skfda-verify/version.txt` — FOUND
- `.planning/research/skfda-verify/verify.log` — FOUND
- `.planning/research/AUDIT-REPORT.md` (contains Phase 7 section) — FOUND

Commits exist:
- `3aa7514f` — Task 1: D-01 version verification
- `24277628` — Task 2: Phase 7 header, methodology, schema
- `4b3ef779` — Task 3: Representation area table (tracer)

## Self-Check: PASSED
