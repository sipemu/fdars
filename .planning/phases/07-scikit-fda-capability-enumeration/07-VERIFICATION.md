---
phase: 07-scikit-fda-capability-enumeration
verified: 2026-08-09T00:00:00Z
status: human_needed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "Reconcile Representation area count discrepancy: the per-area header says '12 in-scope, 7 out-of-scope' (19 total), the Design-Goal Filter table says '13 in-scope, 7 out-of-scope' (20 total), but the actual Representation table has 17 In-Scope rows and 4 Out-of-Scope rows (21 total). All three figures disagree. Confirm whether the actual table (17+4=21) is authoritative and the filter summary (13+7=20) should be corrected, or whether there is a legitimate counting rule that yields 13 and 7."
    expected: "The Design-Goal Filter's per-area Representation row (13 | 7 | 20) and the grand total (125 | 35 | 160) should be corrected to match the actual table, OR a documented counting rule explains the discrepancy."
    why_human: "The actual table rows are unambiguous (17 In-Scope + 4 Out-of-Scope = 21), the report's own notes acknowledge a discrepancy but the explanation does not reconcile the numbers. This is a documentation accuracy issue that affects the filter's stated role as the authoritative count for Phase 8."
---

# Phase 7: scikit-fda Capability Enumeration — Verification Report

**Phase Goal:** Build the scikit-fda side of the comparison — a versioned, area-organized capability inventory.
**Verified:** 2026-08-09
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | scikit-fda's public capability surface is enumerated by area (representation, preprocessing, exploratory, ML, inference, misc) in the report | VERIFIED | All six `### Area:` subsections present under `## Phase 7 — scikit-fda Capability Enumeration` in AUDIT-REPORT.md (grep count = 6) |
| 2 | The exact compared scikit-fda version is pinned and recorded (verified against PyPI `__version__`, baseline 0.10.1) in the methodology | VERIFIED | RUNTIME path used: venv install confirmed `skfda.__version__ = 0.10.1`; D-01a baseline=latest note present; methodology subsection states "0.10.1 is both the agreed sole baseline and the current latest release on PyPI" |
| 3 | Enumeration is capability-oriented, not raw API-name counting: fit/predict/transform families are grouped by user task to avoid the 2-3x gap inflation of Pitfall 9 | VERIFIED | Collapse rule stated explicitly in Capability-Row Schema: "A single scikit-fda estimator's fit(), predict(), transform(), and inverse_transform() are collapsed into one capability row"; e.g. KMeans is one row with "fit / predict / transform" in the Collapsed calls column; Pitfall 9 cited |
| 4 | A one-page design-goal filter is written (in-scope numeric algorithms vs out-of-scope plotting/IO/sklearn-pipeline) to be applied in Phase 8 | VERIFIED | `### Design-Goal Filter` subsection exists within the Phase 7 section of AUDIT-REPORT.md (not a separate file); contains explicit borderline rulings for Visualization, dataset loaders, DataFrame IO, FDAFeatureUnion/PerClassTransformer/sklearn-Pipeline, and type-system; reports in-scope total: 125 and out-of-scope total: 35 as separate figures |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

**Note on SC4 count accuracy:** The Design-Goal Filter exists and fulfills its structural purpose, but contains internal count inconsistency. The Representation area header states "12 in-scope rows, 7 out-of-scope rows" (total 19); the Design-Goal Filter table says "13 in-scope, 7 out-of-scope" (total 20); the actual Representation area table contains 17 In-Scope rows and 4 Out-of-Scope rows (total 21). The report acknowledges the 12 vs 13 discrepancy but not the 17 vs 13 discrepancy, nor the 4 vs 7 Out-of-Scope discrepancy. See Human Verification section.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.planning/research/AUDIT-REPORT.md` | `## Phase 7` section with methodology, schema, six area tables, Design-Goal Filter | VERIFIED | File confirmed at line 863; section runs to EOF (line 1420, 558 lines of Phase 7 content); all required subsections present |
| `.planning/research/skfda-verify/version.txt` | D-01 throwaway evidence file with version pin, path, D-01a note | VERIFIED | File exists; contains "0.10.1", "RUNTIME", and D-01a baseline=latest note |
| `.planning/research/skfda-verify/verify.log` | D-01 venv install transcript | VERIFIED | File exists |
| `.planning/research/design-goal-filter.md` | MUST NOT exist (filter is a subsection, not a file) | VERIFIED | File does not exist |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| Representation area table | FEATURES.md §Area 1 | "Promoted from FEATURES.md §Area 1 (scikit-fda public API table), fdars notes stripped per D-02" stated in methodology; Source column cites FEATURES.md §Area 1 | VERIFIED | Cited per-row in Source column |
| Design-Goal Filter borderline rulings | Six area tables | "Counts are drawn from the six area tables above" | VERIFIED | Filter references each area table; borderline items all accounted for (Visualization, IO loaders, FDAFeatureUnion, PerClassTransformer, sklearn-Pipeline, type-system, ExceptionExtrapolation, scoring metrics) |
| Design-Goal Filter counts | Phase 8 GAP-02/GAP-03 | "The in-scope count is what Phase 8's parity matrix operates on" | VERIFIED | Separated counts stated (125 in-scope, 35 out-of-scope); count accuracy is a WARNING (see Human Verification) |

### Data-Flow Trace (Level 4)

This is a documentation-only phase. No data flows from databases or APIs. The data source is the existing `.planning/research/FEATURES.md` scikit-fda API enumeration, promoted into AUDIT-REPORT.md per D-02. Each table row cites its source in the Source column.

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| AUDIT-REPORT.md § Representation table | 21 capability rows | FEATURES.md §Area 1 + RUNTIME dir() spot-checks | Documentation data, not computed | FLOWING (source cited per row) |
| AUDIT-REPORT.md § Preprocessing table | 31 capability rows | FEATURES.md §Areas 2-4 | Documentation data | FLOWING (source cited per row) |
| AUDIT-REPORT.md § Design-Goal Filter | Capability counts | Six area tables (per filter text) | Computed from tables | WARNING — count in Representation column (13+7=20) does not match actual table rows (17+4=21) |

### Behavioral Spot-Checks

Step 7b SKIPPED: documentation-only phase — no runnable code was produced.

### Probe Execution

No probes declared in PLAN frontmatter for this phase. No `scripts/*/tests/probe-*.sh` files referenced.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| GAP-01 | 07-01-PLAN.md, 07-02-PLAN.md | scikit-fda's public capability surface is enumerated by area with version pinned | SATISFIED | REQUIREMENTS.md marks GAP-01 as Complete at Phase 7; all six areas enumerated; version 0.10.1 pinned with RUNTIME verification |

**Orphaned requirements check:** REQUIREMENTS.md maps GAP-01 to Phase 7 only. No additional requirements are mapped to Phase 7 beyond what appears in the plan's `requirements` field. Coverage is 1/1.

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| AUDIT-REPORT.md (Representation area header) | Internal count inconsistency: "12 in-scope rows, 7 out-of-scope rows" (sum 19) at line 990 | Warning | The per-area note is wrong vs the actual table (17+4=21) and vs the Design-Goal Filter (13+7=20). All three differ. The filter's own notes acknowledge the 12 vs 13 discrepancy but not the deeper 17 vs 13 discrepancy |
| AUDIT-REPORT.md (Design-Goal Filter table) | Representation row shows 13+7=20 but actual table has 17+4=21 rows | Warning | If Phase 8 uses the filter's 125 in-scope total to check completeness, it will expect 125 rows but the actual tables contain 129 in-scope rows (125 - 13 + 17 = 129) |

No debt markers (TBD, FIXME, XXX) found in phase-7 section.
No fdars-side parity notes in Phase 7 section (region-scoped negative gate: CLEAN).
No gap categorization (table-stakes/differentiator) in Phase 7 section: CLEAN.
No fdars-core/src files modified: CONFIRMED (git status shows no changes).

### Human Verification Required

#### 1. Reconcile Representation Area Count Discrepancy

**Test:** Count the rows in the `### Area: Representation` table in AUDIT-REPORT.md (lines 992-1014), categorized by Relevance value. Compare with both the per-area header note ("12 in-scope rows, 7 out-of-scope rows") and the Design-Goal Filter table entry ("Representation | 13 | 7 | 20").

**Expected:** One of:
- (a) The filter table is corrected to "Representation | 17 | 4 | 21" and the grand total updated (in-scope: 129, out-of-scope: 32, total: 161), OR
- (b) A legitimate counting convention explains why 13 and not 17 should be used (e.g. some rows were double-counted from FEATURES.md and should be removed from the table), with the table corrected accordingly, OR
- (c) The per-area header note and filter table are accepted as-is with a documented note that the actual tables are the authoritative source and the counts are approximations.

**Why human:** The actual table row counts are unambiguous (17 In-Scope + 4 Out-of-Scope = 21 rows), but the report's own notes provide an explanation that doesn't arithmetically reconcile. The verifier cannot determine which count is "correct" without a human decision on the intended counting convention. This is a documentation accuracy issue that affects Phase 8's ability to use the 125 in-scope total as a completeness check.

### Gaps Summary

No structural gaps. All four success criteria are satisfied: six areas enumerated, version pinned with RUNTIME verification, capability-oriented grouping with Pitfall 9 addressed, and a Design-Goal Filter with explicit borderline rulings and separated in-scope/out-of-scope counts.

The single human-verification item is a count accuracy issue in the Design-Goal Filter's Representation row: the actual table has 17 In-Scope rows and 4 Out-of-Scope rows, but the filter states 13 and 7 respectively. The report's explanatory notes acknowledge the 12 vs 13 discrepancy but do not explain the further gap to 17, nor the 4 vs 7 Out-of-Scope disagreement. Correcting the filter would change the grand totals from 125/35/160 to approximately 129/32/161.

---

_Verified: 2026-08-09_
_Verifier: Claude (gsd-verifier)_
