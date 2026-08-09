---
phase: 08-capability-parity-matrix-categorization
plan: "03"
subsystem: research-documentation
tags: [scikit-fda, capability-parity, audit, gap-counts, reverse-parity, strengths, backlog, d-04, accuracy-validation]
status: complete

dependency_graph:
  requires:
    - ".planning/research/AUDIT-REPORT.md ## Phase 8 section (Plans 01+02: all six area parity tables + All-129 Coverage Check)"
    - ".planning/phases/08-capability-parity-matrix-categorization/08-CONTEXT.md (D-02a, D-03, D-04, Pitfall 14)"
    - ".planning/codebase/STRUCTURE.md (fdars module map for reverse-sweep source-confirmation)"
    - ".planning/codebase/CONCERNS.md (fragile areas for D-02a accuracy-validation item)"
  provides:
    - "### Gap Counts (in-scope vs out-of-scope): 82 actionable in-scope gaps / 32 out-of-scope excluded"
    - "### Reverse-Parity Strengths Sweep (D-04): 30 fdars-exclusive capabilities table (22 none / 8 partial-fdars-advantage)"
    - "### Drafted Gap Backlog (unranked): 20 entries (PREP-01 through MISC-04) + ACC-01 D-02a item"
  affects:
    - ".planning/research/AUDIT-REPORT.md (Plan 03 final append completes Phase 8 section)"
    - "Phase 9 (RPT-02/RPT-03): value-ranking input — 82 gap entries + 30 strength entries"

tech_stack:
  added: []
  patterns:
    - "Gap counts table: area × (gaps / table-stakes / differentiator) — derived from per-area summaries"
    - "Reverse table: fdars Capability | fdars module | scikit-fda equivalent: none/partial | Confidence"
    - "Backlog entry schema: Area / Current gap / Root cause (no ranking, no effort, no severity)"
    - "--no-verify used for docs-only commit per MEMORY.md sanctioned exception (/tmp at 95%)"

key_files:
  created: []
  modified:
    - ".planning/research/AUDIT-REPORT.md (appended three subsections: Gap Counts, Reverse-Parity, Drafted Backlog; 386 insertions)"

decisions:
  - "Misc gap count uses 4pt+18a=22 per Misc area summary (not the Coverage Check's 2pt+20a aggregation) — the area summary is the authoritative per-row count; the Coverage Check aggregate reconciles post-recount"
  - "Representation table-stakes count is 6 (not 5 as stated in the area summary text) — ExceptionExtrapolation carries a gap category in the table despite being In-Scope API-Ergonomics in Phase 7; the discrepancy is documented"
  - "Reverse sweep includes 30 rows (not just the four SC3 headliners + 12 D-04 candidates = 16); additional capabilities found by walking the full module map: elastic regression/FPCA/explain, FOSR 1D+2D, Bayesian/constrained alignment, geodesic paths, shape depth, irreg_fdata module, FPCA backbone, warping utilities, FOF regression"
  - "FOF regression row #17 set to 'partial' (not none): fdars has general FOF (fof_regression.rs) while scikit-fda has HistoricalLinearRegression (a specialized subcase); fdars has the broader capability"
  - "Andrews curves row #16 set to 'partial' (not none): scikit-fda has no Andrews module, but Andrews is an Exploratory out-of-scope topic in Phase 7 visualization — classified as partial-fdars-advantage to be precise"
  - "--no-verify used per MEMORY.md sanctioned exception: /tmp tmpfs at 95%, docs-only commit"

metrics:
  duration_minutes: 7
  completed_date: "2026-08-09"
  tasks_completed: 3
  tasks_total: 3
  files_changed: 1

actuals:
  tokens: 75000
  tasks: 3
  commits: 1

requirements-completed: [GAP-03, GAP-04]
---

# Phase 08 Plan 03: Gap Counts, Reverse-Parity Sweep, and Draft Backlog Summary

**One-liner:** Separated 82 actionable in-scope gaps from 32 excluded out-of-scope, enumerated 30 fdars-exclusive capabilities (full D-04 sweep beyond four SC3 headliners), and drafted 20 unranked backlog entries plus D-02a numerical-accuracy validation item — completing Phase 8.

## Performance

- **Duration:** ~7 min
- **Completed:** 2026-08-09
- **Tasks:** 3 (Task 1: Gap Counts; Task 2: Reverse-Parity Sweep; Task 3: Drafted Backlog)
- **Files modified:** 1

## Accomplishments

### Task 1: Gap Counts (in-scope vs out-of-scope)

Appended `### Gap Counts (in-scope vs out-of-scope)` to `.planning/research/AUDIT-REPORT.md ## Phase 8`:

- **32 out-of-scope capabilities** (plotting + IO + type-system plumbing from Phase 7's Design-Goal Filter) reported as a SEPARATE count, explicitly excluded from the actionable total (Pitfall 14 enforced).
- **82 actionable in-scope gaps** across 141 mapped rows: **36 table-stakes** (baseline FDA capabilities fdars lacks or partially covers) and **46 differentiator** (advanced capabilities whose absence is acceptable but whose presence would set fdars apart).
- Per-area breakdown table with (gaps / table-stakes / differentiator) columns; area sums verified against per-area gap-category summaries from Plans 01+02.
- Narrative explains that "fdars is far behind" readings are caused by mixing in the 32 out-of-scope rows.

### Task 2: Reverse-Parity Strengths Sweep (D-04)

Appended `### Reverse-Parity Strengths Sweep (D-04)` with a 30-row table:

**Four SC3 headliners (rows 1–4):**
1. Model explainability: `explain/` + `explain_generic/` — 44+ public functions, PDP/SHAP/ALE/LIME/importance/Sobol/DFbetas/counterfactual/anchor. No scikit-fda equivalent.
2. SPM/control charts: `spm/` (20 files) — EWMA/CUSUM/MEWMA/amEWMA/Hotelling T²/ARL/elastic SPM. No scikit-fda equivalent.
3. Seasonal decomposition: `seasonal/` (12 files) — SAZED/autoperiod/Lomb-Scargle/SSA/matrix-profile/Hilbert/STL. Fragile area noted (Lomb-Scargle NaN). No scikit-fda equivalent.
4. Streaming functional depth: `streaming_depth/` — online FM/BD/MBD/rolling. No scikit-fda equivalent.

**Twelve D-04 candidate list items (rows 5–16):** conformal prediction, tolerance bands, GMM clustering (accuracy-flagged ec17d138), matrix profile, SSA, Hilbert transform, WIRE, FAMM, elastic changepoint, robust L1/Huber regression, multi-response regression, Andrews curves.

**Fourteen additional from full module-map walk (rows 17–30):** general FOF regression (`fof_regression.rs`), elastic regression/FPCA/explain, FOSR 1D+2D, Bayesian alignment, constrained alignment, geodesic paths, shape depth (SRSF-space), shape-space outlier detection, irreg_fdata module, FPCA-in-regression backbone (`FpcPredictor` trait breadth), warping utilities, phase boxplot.

Summary: 22 rows with scikit-fda equivalent = none; 8 rows with partial-fdars-advantage.

### Task 3: Drafted Gap Backlog (unranked)

Appended `### Drafted Gap Backlog (unranked)` with 21 entries, all UNRANKED, each carrying Area / Current gap / Root cause:

| Group | Entries | Areas |
|-------|---------|-------|
| PREP-01 through PREP-09 | 9 entries | Preprocessing: bandwidth criteria, smoother abstraction, missing-value imputation, shift/landmark registration, registration validation, regularized FPCA, variable selection, feature construction, diffusion maps |
| REPR-01 through REPR-04 | 4 entries | Representation: additional basis systems, spline interpolation, extrapolation policies, mixed-effects converters |
| EXPL-01 through EXPL-03 | 3 entries | Exploratory: pluggable-metric depth/outlyingness, summary statistics (trim_mean/depth_median/var/std/cov), Stahel-Donoho outlyingness |
| ML-01 through ML-02 | 2 entries | ML: missing classifiers (MaximumDepth/NearestCentroid/Radius/DTM/DDG), LDO regression + HistoricalLinReg |
| INF-01 through INF-02 | 2 entries | Inference: asymptotic functional ANOVA V-statistic, standalone two-sample Hotelling T² |
| MISC-01 through MISC-04 | 4 entries | Misc: missing metrics/distances, composable LDO/L2Reg operator objects, data-generation wrappers, scoring metrics |
| **ACC-01** (D-02a) | 1 entry | Cross-cutting: comparative numerical-accuracy validation pass (fdars vs scikit-fda) for all four fragile areas (B-spline round-trip #33, elastic-alignment level encoding #34, Lomb-Scargle NaN, GMM over-split ec17d138) |

Explicit note: value ranking, severity, effort estimates, and reproducible-evidence links are Phase 9 (RPT-02/RPT-03, Pitfalls 13/16/17). Closing pointer states the backlog + counts + strengths sweep feed Phase 9.

**Zero `fdars-core/src` edits** (audit-only milestone enforced).

## Task Commits

All three tasks executed atomically in one Edit to the docs-only file:
- **Commit `2bbd7d56`** — "docs(08-03): append gap counts, reverse-parity sweep, and draft backlog" (386 insertions, 0 deletions, 0 source files)

## Files Created/Modified

- `.planning/research/AUDIT-REPORT.md` — appended three subsections (Gap Counts, Reverse-Parity Sweep, Drafted Backlog); 386 insertions.

## Decisions Made

- **Misc gap count (22 from area summary, not 22 from Coverage Check):** The Misc area summary states 4pt+18a=22 gap rows; the Coverage Check aggregate uses 2pt+20a=22 (same total, different breakdown). The area summary is the authoritative per-row source; total gap count 22 is consistent.
- **Representation table-stakes = 6 (including ExceptionExtrapolation):** The area summary text says "table-stakes = 5" but lists six items. ExceptionExtrapolation is listed as a gap because any FDA library should error on out-of-domain queries. The discrepancy is preserved and noted.
- **30 rows in the reverse sweep** (exceeds the 16 initial: 4 headliners + 12 D-04): Full module-map walk found 14 additional fdars-exclusive capabilities (elastic regression/FPCA/explain, FOSR, Bayesian alignment, constrained alignment, geodesic paths, shape depth, irreg_fdata, FPCA backbone, warping utilities, FOF regression).
- **FOF regression = partial-fdars-advantage:** fdars has general FOF (`fof_regression.rs`); scikit-fda has only `HistoricalLinearRegression` (specialized integral subcase). fdars has the broader capability.

## Deviations from Plan

### Auto-fixed Issues

**1. [MEMORY.md documented exception] Commit made with --no-verify**
- **Found during:** Task commit
- **Issue:** Pre-commit doctest/test hook ran 1934 tests; `/tmp` tmpfs at 95% causes the cargo linker to exhaust space (SIGBUS), not a code issue (infra flake documented in MEMORY.md §tmp-exhaustion-blocks-precommit).
- **Fix:** Used `--no-verify` for this docs-only commit per the sanctioned MEMORY.md exception. Only `.planning/research/AUDIT-REPORT.md` was staged — no source files touched.
- **Committed in:** `2bbd7d56`

**Total deviations:** 1 (documented infra exception). Does not affect scope.

## Issues Encountered

None beyond the documented /tmp tmpfs infra flake (resolved with --no-verify per MEMORY.md).

## Known Stubs

None — this plan produces analysis artifacts only (gap-counts table, reverse-parity table, backlog entries); no code stubs exist.

## Threat Flags

None — no network endpoints, auth paths, file access patterns, or schema changes introduced.

## Phase 8 Complete

Phase 8 is now fully complete. The `## Phase 8 — Capability Parity Matrix & Categorization` section of `.planning/research/AUDIT-REPORT.md` now contains:

1. **Verdict rubric + Categorization rubric** (Plan 01 tracer — establishes schema)
2. **All six area parity tables** (Plans 01+02 — 141 literal rows, 59p/19pt/63a)
3. **All-129 Coverage Check** (Plan 02 — confirms 100% Phase-7 row coverage)
4. **Gap Counts (in-scope vs out-of-scope)** (Plan 03 — 82 actionable / 32 excluded)
5. **Reverse-Parity Strengths Sweep (D-04)** (Plan 03 — 30 fdars-exclusive capabilities)
6. **Drafted Gap Backlog (unranked)** (Plan 03 — 20 backlog entries + ACC-01 D-02a)

Zero fdars-core/src edits across all three plans (audit-only milestone enforced).

## Self-Check

**Files exist:**
- `2bbd7d56` committed — confirmed in git log.
- `.planning/research/AUDIT-REPORT.md` contains all three new subsections (grep-verified above).
- All three acceptance criteria grepped and passed.

**No source edits:**
- `git diff --name-only -- fdars-core/src` returns empty (verified).

**Self-Check: PASSED**

---
*Phase: 08-capability-parity-matrix-categorization*
*Completed: 2026-08-09*
