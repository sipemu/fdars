---
phase: 08-capability-parity-matrix-categorization
plan: "02"
subsystem: research-documentation
tags: [scikit-fda, capability-parity, audit, representation, exploratory, ml, inference, misc, verdict-rubric, categorization-rubric]
status: complete

dependency_graph:
  requires:
    - ".planning/research/AUDIT-REPORT.md ## Phase 8 section (Plan 01 tracer: verdict rubric + category rubric + Preprocessing parity table)"
    - ".planning/phases/08-capability-parity-matrix-categorization/08-CONTEXT.md (D-01/D-02/D-03 rubrics)"
    - ".planning/codebase/STRUCTURE.md (fdars module map for source-grep)"
    - ".planning/codebase/CONCERNS.md (GMM ec17d138, seasonal/Lomb-Scargle fragile area)"
    - "fdars-core/src/ (source-grep-confirmed verdicts: basis/, irreg_fdata/, depth/, outliers.rs, fdata.rs, alignment/, classification/, scalar_on_function/, fof_regression.rs, function_on_scalar.rs, clustering.rs, gmm/, famm.rs, spm/stats.rs, distance.rs, metric/, covariance.rs, simulation.rs, utility.rs, helpers.rs)"
  provides:
    - "### Area: Representation — Parity (17 rows mapped, 4p/1pt/12a)"
    - "### Area: Exploratory — Parity (20 rows mapped, 9p/4pt/7a)"
    - "### Area: ML — Parity (20 rows mapped, 11p/2pt/7a)"
    - "### Area: Inference — Parity (5 rows mapped, 0p/2pt/3a)"
    - "### Area: Misc — Parity (40 literal rows mapped, 18p/4pt/18a)"
    - "### All-129 Coverage Check: 141 rows across six tables; aggregate 59p/19pt/63a"
  affects:
    - ".planning/research/AUDIT-REPORT.md (Plan 03 appends separated counts, reverse-parity sweep, drafted backlog into the same section)"

tech_stack:
  added: []
  patterns:
    - "Same row schema as Plan 01: Area | Task | Capability | Verdict | Category | Accuracy? | fdars equivalent | Confidence"
    - "Recount convention: Phase-7 literal table rows are the fixed left column (not stale header numbers)"
    - "Misc area: 40 literal rows mapped (Phase-7 header says 38 due to documented 2-row compression)"
    - "ec17d138 GMM accuracy flag cited in All-129 Coverage Check (not in an ML row — GMM is fdars-exclusive)"
    - "Seasonal/Lomb-Scargle fragile area noted in Coverage Check as fdars-exclusive; no scikit-fda row to flag"
    - "--no-verify used for docs-only commit per MEMORY.md sanctioned exception (/tmp at 95%)"

key_files:
  created: []
  modified:
    - ".planning/research/AUDIT-REPORT.md (appended five area parity tables + All-129 Coverage Check; 396 insertions)"

decisions:
  - "Misc area mapped as 40 literal rows (all distinct Phase-7 table rows), not 38 (the compressed header count); recount-supersedes-header convention applied consistently"
  - "ec17d138 GMM over-split accuracy flag cited in the All-129 Coverage Check rather than in an ML parity row — GMM is fdars-exclusive (not a scikit-fda Inference/ML scikit-fda row); it will carry an accuracy flag in the reverse-parity sweep (Plan 03)"
  - "seasonal/Lomb-Scargle NaN fragile area does not map to any scikit-fda in-scope row; cited in Coverage Check with note that it will appear in Plan 03 reverse-parity sweep"
  - "All-129 aggregate: 59 present / 19 partial / 63 absent across 141 literal rows (after recount corrections)"

metrics:
  duration_minutes: 9
  completed_date: "2026-08-09"
  tasks_completed: 3
  tasks_total: 3
  files_changed: 1

actuals:
  tokens: 95000
  tasks: 3
  commits: 1

requirements-completed: [GAP-02, GAP-03]

coverage:
  - id: SC1
    description: "All five remaining Phase-7 areas parity-mapped using same schema Plan 01 proved (SC1, GAP-02)"
    requirement: "GAP-02"
    verification:
      - kind: other
        ref: "grep -q '### Area: Representation — Parity' && grep -q '### Area: Exploratory — Parity' && grep -q '### Area: ML — Parity' && grep -q '### Area: Inference — Parity' && grep -q '### Area: Misc — Parity' .planning/research/AUDIT-REPORT.md"
        status: pass
    human_judgment: false
  - id: SC2
    description: "All 129+ in-scope capabilities marked present/partial/absent with searched-notes on every partial/absent row"
    requirement: "GAP-02"
    verification:
      - kind: other
        ref: "grep -q '### All-129 Coverage Check' .planning/research/AUDIT-REPORT.md && grep -c 'searched fdars for:' (83 occurrences)"
        status: pass
    human_judgment: false
  - id: SC3
    description: "GMM + seasonal fragile rows accuracy-flagged with citations per D-02"
    requirement: "GAP-02"
    verification:
      - kind: other
        ref: "grep -q 'ec17d138' .planning/research/AUDIT-REPORT.md (present in Coverage Check)"
        status: pass
    human_judgment: false
  - id: SC4
    description: "Every gap row carries a D-03 category (table-stakes / differentiator)"
    requirement: "GAP-03"
    verification:
      - kind: other
        ref: "Per-area gap-category counts documented in each area summary; no gap row missing a category"
        status: pass
    human_judgment: false
  - id: SC5
    description: "Zero fdars-core/src edits (audit-only milestone enforced)"
    verification:
      - kind: other
        ref: "git diff --name-only -- fdars-core/src (empty)"
        status: pass
    human_judgment: false
---

# Phase 08 Plan 02: Five-Area Parity Mapping Summary

**One-liner:** Five remaining scikit-fda capability areas parity-mapped against fdars (141 rows across Representation / Exploratory / ML / Inference / Misc) using the Plan-01 schema — verdicts source-grep-confirmed, gap categories assigned, coverage check written — completing the parity matrix.

## Performance

- **Duration:** ~9 min
- **Completed:** 2026-08-09
- **Tasks:** 3 (Task 1: Representation + Exploratory; Task 2: ML + Inference; Task 3: Misc + Coverage Check)
- **Files modified:** 1

## Accomplishments

- Appended **five area parity tables** into the `## Phase 8` section of `AUDIT-REPORT.md`, each joining 1:1 against the authoritative Phase-7 table rows (not stale headers):
  - `### Area: Representation — Parity`: 17 rows — 4 present, 1 partial, 12 absent. Only B-spline and Fourier basis systems and covariance estimation are present; Monomial/Constant/Custom/Tensor/FE/VV bases absent; all 4 extrapolation policy types absent; both mixed-effects irregular→basis converters absent; grid-to-basis conversion present.
  - `### Area: Exploratory — Parity`: 20 rows — 9 present, 4 partial, 7 absent. Integrated/band/modified-band/projection depth and MS-plot outlier detection present; distance-based depth (L2-only, not metric-pluggable) and simplicial depth (random-Tukey approximation only) partial; trim_mean / depth_based_median / functional var / std absent.
  - `### Area: ML — Parity`: 20 rows — 11 present, 2 partial, 7 absent. kNN/QDA/LDA/DD/logistic classification, agglomerative clustering, k-means, fuzzy c-means, kNN/kernel regressors, and FPLS present; MaximumDepthClassifier and RadiusNeighbors absent; unified-LDO LinearRegression and NearestCentroid-as-nearest-centroid partial.
  - `### Area: Inference — Parity`: 5 rows — 0 present, 2 partial, 3 absent. All five rows are table-stakes gaps; functional ANOVA (permutation, not asymptotic V-statistic) and Hotelling T² (SPM module, not inference module) partial; the two V-statistic functions and independent-sample Hotelling T² absent.
  - `### Area: Misc — Parity`: 40 literal rows — 18 present, 4 partial, 18 absent. All 7 covariance kernels, Lp/Fisher-Rao/inner-product/pairwise-distance present; Mahalanobis/angular/cosine/NormInduced/TransformationMetric absent; LDO and L2Reg partial (matrix present, composable object absent); SRSF operator present; MAE/MSE/MAPE/MSLE scoring absent.
- Appended **`### All-129 Coverage Check`**: documents the 17+39+20+20+5+40 = 141 literal rows mapped across six tables, explains the two recount corrections (Preprocessing 29→39, Misc 38→40), and records the aggregate verdict counts: **59 present / 19 partial / 63 absent**.
- Cited `ec17d138` GMM over-split accuracy flag in the Coverage Check (fdars-exclusive GMM, no scikit-fda row); noted seasonal/Lomb-Scargle fragile area as fdars-exclusive for Plan 03 reverse-parity sweep.
- **Zero `fdars-core/src` edits** (audit-only milestone enforced).

## Task Commits

1. **Task 1: Parity-map Representation + Exploratory** — included in `d9c25c8`
2. **Task 2: Parity-map ML + Inference** — included in `d9c25c8`
3. **Task 3: Parity-map Misc + Coverage Check** — `d9c25c8`

All three tasks were executed atomically in one Edit to the same docs-only file; single commit is appropriate per the sequential executor model.

## Files Created/Modified
- `.planning/research/AUDIT-REPORT.md` — appended five area parity tables + All-129 Coverage Check (396 insertions).

## Decisions Made
- **Misc 38→40 row recount**: The Phase-7 Misc header states 38 in-scope rows; the literal table has 40 distinct rows. Recount-supersedes-header convention applied consistently (same as Preprocessing 29→39 in Plan 01). All 40 rows mapped.
- **ec17d138 in Coverage Check, not in ML table**: GMM is fdars-exclusive (no scikit-fda clustering row for GMM). The accuracy flag belongs in the reverse-parity sweep (Plan 03), not in the scikit-fda-parity ML table. Cited in the Coverage Check so it is not lost.
- **All-129 aggregate**: After recount corrections, 141 literal rows across six areas, with 59p/19pt/63a.

## Deviations from Plan

### Auto-fixed Issues

**1. [MEMORY.md documented exception] Commit made with --no-verify**
- **Found during:** Task commit
- **Issue:** Pre-commit doctest hook failed with 81/129 doctests failing — `/tmp` tmpfs at 95% causes the cargo linker to run out of space (SIGBUS), not a code issue (infra flake documented in MEMORY.md §tmp-exhaustion-blocks-precommit).
- **Fix:** Used `--no-verify` for this docs-only commit per the sanctioned MEMORY.md exception. Only `.planning/research/AUDIT-REPORT.md` was staged — no source files touched.
- **Files modified:** none beyond the doc.
- **Committed in:** `d9c25c8`

**2. [Recount] Misc area 38→40 literal rows**
- **Found during:** Task 3 (Misc mapping)
- **Issue:** The Phase 7 Misc header states "In-scope count: 38 rows" but the literal Misc table has 40 distinct rows. The Design-Goal Filter note acknowledges the 2-row compression. This plan maps all 40 literal rows for completeness, consistent with the recount-supersedes-header convention.
- **Fix:** Mapped all 40 rows; documented the discrepancy in the All-129 Coverage Check.
- **Impact:** The all-area total is 141 literal rows (not 129), with the difference explained by the two recount corrections. Every Phase-7 table row is covered.

**Total deviations:** 2 (1 documented infra exception, 1 recount). Neither affects scope.

## Issues Encountered

None beyond the documented /tmp tmpfs infra flake (resolved with --no-verify per MEMORY.md).

## Known Stubs

None — this plan produces analysis artifacts only (parity tables); no code stubs exist.

## Threat Flags

None — no network endpoints, auth paths, file access patterns, or schema changes introduced.

## User Setup Required
None.

## Next Phase Readiness
- Plan 02 completes the parity matrix (all 141 literal in-scope rows marked). Plan 03 can now proceed to: separated in-scope gap counts, reverse-parity strengths sweep (fdars-exclusive capabilities), drafted gap-backlog entries, and numerical-accuracy validation backlog item (D-02a).
- The ec17d138 GMM accuracy flag and the seasonal/Lomb-Scargle fragile area need to appear in Plan 03's reverse-parity sweep with "present — accuracy NOT verified" flags.
- No blockers.

## Self-Check

**Files exist:**
- `d9c25c8` committed and confirmed in git log.
- `.planning/research/AUDIT-REPORT.md` contains all five area headers and the Coverage Check (grep-verified above).

**Verdicts source-confirmed:** All verdicts were established by grepping/reading the named fdars-core/src modules (not from STRUCTURE.md alone). Confidence HIGH on all rows where a specific public function was named; MEDIUM on rows where the module map was used without a single named symbol (noted per-row).

**Self-Check: PASSED**

---
*Phase: 08-capability-parity-matrix-categorization*
*Completed: 2026-08-09*
