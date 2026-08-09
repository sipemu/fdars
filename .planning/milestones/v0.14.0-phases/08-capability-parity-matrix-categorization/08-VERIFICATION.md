---
phase: 08-capability-parity-matrix-categorization
verified: 2026-08-09T22:00:00Z
status: passed
score: 9/9 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: null
---

# Phase 8: Capability Parity Matrix & Categorization — Verification Report

**Phase Goal:** Map fdars vs scikit-fda by capability, categorize gaps, and document fdars strengths
**Verified:** 2026-08-09
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

All must-haves are drawn from the three PLAN frontmatter lists (Plans 01, 02, 03) plus the
ROADMAP success criteria for GAP-02, GAP-03, GAP-04. The sole deliverable is the appended
`## Phase 8 — Capability Parity Matrix & Categorization` section in
`.planning/research/AUDIT-REPORT.md`.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `## Phase 8 — Capability Parity Matrix & Categorization` section exists in AUDIT-REPORT.md | VERIFIED | Confirmed at line 1425 of AUDIT-REPORT.md |
| 2 | The three-value verdict rubric (present / partial / absent) is stated once with definitions and drives every parity row (D-01 / D-01a partial retained) | VERIFIED | `### Verdict Rubric (D-01)` at line 1439; defines all three values plus D-01a partial-retention rationale; accuracy-flag and searched-note conventions stated once |
| 3 | Categorization rubric (table-stakes / differentiator / out-of-scope) is stated once (D-03) | VERIFIED | `### Categorization Rubric (D-03)` at line 1485; three categories defined with Pitfall-14 note on separated counts |
| 4 | All six capability-area parity tables exist (Preprocessing, Representation, Exploratory, ML, Inference, Misc), mapped by capability not API name, with searched-notes on every partial/absent row | VERIFIED | Six `### Area: X — Parity` headers confirmed (one each); 83 occurrences of "searched fdars for:" across partial/absent rows (grep-verified); verdicts explicitly mapped by capability / Pitfall-9 convention |
| 5 | Known-bug rows read "present — accuracy NOT verified" with fix-commit citations (2fb6d3c9 #33, 6ed62398 #34, ec17d138 GMM), never bare checks | VERIFIED | `BasisSmoother` at line 1532: "present — accuracy NOT verified (B-spline round-trip GH #33, CONCERNS.md §Known Bugs, fixed commit `2fb6d3c9`)"; `FisherRaoElasticRegistration` at line 1547: "present — accuracy NOT verified (elastic-alignment level encoding GH #34, CONCERNS.md §Known Bugs, fixed commit `6ed62398`)"; GMM in reverse-parity table row 7 at line 2091: "present — accuracy NOT fully verified ... ec17d138". All three commit hashes confirmed by grep. No bare check-marks on these rows. |
| 6 | In-scope vs out-of-scope gap counts are separated, with plotting/IO excluded from the actionable count | VERIFIED | `### Gap Counts (in-scope vs out-of-scope)` at line 2002; 32 out-of-scope explicitly reported separately and stated excluded; 82 actionable in-scope gaps (36 table-stakes + 46 differentiator) in the counts table at line 2056 |
| 7 | A reverse-parity strengths table documents fdars-only capabilities — at minimum model explainability, SPM/control charts, seasonal decomposition, streaming depth | VERIFIED | `### Reverse-Parity Strengths Sweep (D-04)` at line 2073; 30-row table; rows 1–4 (headliners): Model explainability, SPM/control charts, Seasonal decomposition, Streaming functional depth — all present with HIGH confidence and "none" scikit-fda equivalent. D-04 candidate list (rows 5–16) and 14 additional from full module-map walk (rows 17–30) also confirmed |
| 8 | Draft gap-backlog entries carry area / current-gap / root-cause fields and are UNRANKED | VERIFIED | `### Drafted Gap Backlog (unranked)` at line 2128; 21 entries (PREP-01 through MISC-04 + ACC-01); each uses a three-field table (Area / Current gap / Root cause); 25 root-cause fields counted in the section; explicit "This backlog is UNRANKED" statement at line 2134 with pointer that ranking is Phase 9 |
| 9 | A D-02a backlog item recommends a comparative numerical-accuracy validation pass for the fragile areas | VERIFIED | `ACC-01 — Numerical accuracy validation pass (fdars vs scikit-fda on shared datasets)` at line 2358; covers all four fragile areas: B-spline round-trip/CV #33, elastic-alignment level encoding #34, seasonal/Lomb-Scargle NaN, GMM over-split ec17d138 |

**Score:** 9/9 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.planning/research/AUDIT-REPORT.md` — `## Phase 8` section | Appended section with rubrics, six area tables, gap counts, reverse-parity sweep, drafted backlog | VERIFIED | Section confirmed at line 1425; all 12 subsections present; 963 net insertions across 3 commits (26ad8199, d9c25c86, 2bbd7d56) |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| Phase 8 parity rows | Phase 7 in-scope capability inventory | 1:1 join on Phase-7 table rows (capability axis fixed) | VERIFIED | Each area table notes "joins 1:1 against Phase 7 §Area" rows; recount convention applied consistently (stale headers corrected by direct table recount) |
| Accuracy-flag rows | CONCERNS.md known bugs | Fix commit citations (2fb6d3c9, 6ed62398, ec17d138) | VERIFIED | All three commit hashes present in AUDIT-REPORT.md and cite CONCERNS.md §Known Bugs in each row |
| Gap-backlog entries | Parity tables (partial/absent rows) | Per-area gap derivation | VERIFIED | 21 backlog entries map to partial/absent rows from the six parity tables; grouped by area with root-cause derivable from source |
| Reverse-parity table | fdars-core/src module map | Source-grep-confirmed per row | VERIFIED | Each row names the specific fdars module/file (e.g., `spm/`, `seasonal/`, `streaming_depth/`, `conformal/`, etc.); confidence tags HIGH/MEDIUM assigned per row |

---

### Data-Flow Trace (Level 4)

Not applicable. This is an audit-only documentation phase. The "data" is the analysis content
in AUDIT-REPORT.md — a static document. There are no dynamic data sources, renders, or
runtime flows to trace. The relevant flow is: Phase-7 scikit-fda inventory + fdars-core/src
grep confirmation → parity verdicts → backlog entries. That flow is verified by checking
document content directly (done above).

---

### Behavioral Spot-Checks

Step 7b: SKIPPED (no runnable entry points — this is a documentation-only audit phase; all
deliverables are Markdown sections in AUDIT-REPORT.md, not executable code).

---

### Probe Execution

Step 7c: SKIPPED (no probe scripts declared in PLANs; no `scripts/*/tests/probe-*.sh`
applicable to a documentation phase).

---

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|---------------|-------------|--------|---------|
| GAP-02 | 08-01, 08-02 | fdars capabilities mapped against scikit-fda by capability (not API name), parity matrix with present/partial/absent | SATISFIED | 141 literal rows mapped across six area tables (all 129 Phase-7 in-scope rows covered per Coverage Check at line 1947); capability-not-API-name mapping explicitly stated; all verdicts source-grep-confirmed |
| GAP-03 | 08-01, 08-02, 08-03 | Gaps categorized table-stakes vs differentiator vs out-of-scope; design-goal filter excludes plotting/IO from actionable count | SATISFIED | D-03 categorization rubric stated at line 1485; every gap row carries a category; 32 out-of-scope separated and excluded from the 82 actionable count; per-area gap-category tallies in each area summary |
| GAP-04 | 08-03 | fdars capabilities exceeding scikit-fda documented (model explainability, SPM, seasonal, streaming depth at minimum) | SATISFIED | 30-row reverse-parity table at line 2073; all four SC3 headliners present (rows 1–4); full D-04 candidate list (rows 5–16) plus 14 additional from module-map walk (rows 17–30) |

---

### Anti-Patterns Found

Phase 8 modifies only `.planning/research/AUDIT-REPORT.md` (documentation). All three commits
(26ad8199, d9c25c86, 2bbd7d56) touch zero `fdars-core/src` files — confirmed by
`git diff --name-only HEAD~6 HEAD -- 'fdars-core/src/'` returning empty.

The SUMMARY files for all three plans note `--no-verify` used for docs-only commits due to
the documented `/tmp` tmpfs-exhaustion SIGBUS infra flake. This is a pre-commit hook bypassed
per the MEMORY.md sanctioned exception for docs-only commits (infra-not-code). No source
files were staged in any of those commits. This is NOT a code-quality concern and NOT a gap.

No debt markers (TBD/FIXME/XXX), stubs, or anti-patterns found in the deliverable
(AUDIT-REPORT.md §Phase 8 is a completed analysis document — tables, prose, no placeholders).

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | No anti-patterns found |

---

### Human Verification Required

None. The phase goal is a document deliverable (analysis content in AUDIT-REPORT.md). All
must-haves are checkable by reading the document:

- Rubric definitions are readable as prose.
- All 141 parity rows are present and greppable.
- Accuracy flags and fix-commit citations are grep-verifiable.
- Backlog entries carry the required fields.
- No runtime behavior or visual interface requires human spot-checking.

The one task that formally required human judgment was the Plan 01 Task 3 tracer-schema
approval checkpoint (blocking human-verify gate). Per the 08-01-SUMMARY.md, this was resolved
via the orchestrator's interactive AskUserQuestion prompt — the user approved the schema and
accepted the 29→39 recount. This checkpoint has been closed; its evidence lives in the SUMMARY
frontmatter. There is no outstanding human-verification item.

---

### Gaps Summary

No gaps found. All nine must-haves are VERIFIED against the actual AUDIT-REPORT.md content.

The sole audit-only deliverable is the `## Phase 8 — Capability Parity Matrix &
Categorization` section of `.planning/research/AUDIT-REPORT.md`. That section is:

1. Present and complete (all 12 subsections confirmed).
2. Substantive (141 parity rows, 83 searched-notes, 5 accuracy flags, 30-row strengths
   table, 21 backlog entries — not a stub or placeholder).
3. Consistent with the phase goal (maps fdars vs scikit-fda by capability, categorizes gaps,
   documents fdars strengths as stated in the ROADMAP goal).

Requirements GAP-02, GAP-03, GAP-04 are all satisfied. Zero `fdars-core/src` changes (correct
for an audit-only milestone — the absence of code changes is the intended outcome, not a gap).

---

## Deviation Notes

Two minor process deviations occurred during execution; neither is a content gap:

1. **Preprocessing row-count 29 → 39**: The PLAN stated 29 in-scope rows; direct recount of
   the Phase-7 Preprocessing tables yielded 39 (stale header). All 39 rows were mapped, user
   accepted the recount at the Task 3 checkpoint. The plan's expected count was the stale
   figure; the deliverable is more complete, not less.

2. **`--no-verify` on three docs-only commits**: Pre-commit doctest hook failed due to
   documented `/tmp` tmpfs exhaustion (SIGBUS, infra-not-code per MEMORY.md). No source files
   were bypassed — only the docs file `.planning/research/AUDIT-REPORT.md` was committed each
   time. This is the project-approved workaround for this infra flake.

Neither deviation reduces scope or correctness. Both are documented in the respective SUMMARY
files.

---

_Verified: 2026-08-09T22:00:00Z_
_Verifier: Claude (gsd-verifier)_
