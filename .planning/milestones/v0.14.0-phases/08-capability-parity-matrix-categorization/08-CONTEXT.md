# Phase 8: Capability Parity Matrix & Categorization - Context

**Gathered:** 2026-08-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the **parity comparison** between fdars and scikit-fda 0.10.1 from the two sides already assembled: Phase 7's scikit-fda capability inventory (129 in-scope rows across six areas) and the fdars codebase (`STRUCTURE.md` module map + source). Deliverables are **analysis artifacts only** — a section appended to the single `.planning/research/AUDIT-REPORT.md`; **no `fdars-core/src` changes** (audit-only milestone).

Concretely, this phase appends a `## Phase 8 — Capability Parity Matrix & Categorization` section to `.planning/research/AUDIT-REPORT.md` containing:
1. A **parity matrix** marking each of the 129 in-scope scikit-fda capabilities **present / partial / absent**, mapped by *capability, not API name* (Pitfall 9), each gap candidate carrying an **"fdars equivalent searched"** note (Pitfall 11) and each fragile/known-bug row an **"accuracy verified?"** flag (Pitfall 12) — using `STRUCTURE.md` as the fdars side (SC1).
2. Each gap **categorized table-stakes / differentiator / out-of-scope** via an explicit rubric, with the design-goal filter applied and **separate in-scope vs out-of-scope gap counts** (SC2, GAP-03, Pitfall 14).
3. A **reverse-parity strengths section** documenting every fdars capability that exceeds / has no scikit-fda equivalent (SC3, GAP-04).
4. **Draft gap-backlog entries** with area / current-gap / root-cause fields, ready for value ranking in Phase 9 (SC4).

**Not in scope:**
- Any `fdars-core/src` edits (audit-only).
- **Final value-ranking of the backlog** — that is **Phase 9** (RPT-02). Phase 8 drafts entries; it does not rank them.
- **Running numeric accuracy smoke-tests** against scikit-fda — this milestone flags fragile areas only (see D-03); actual comparative validation is drafted as a backlog item for a future milestone.
- **Re-enumerating scikit-fda** — Phase 7's 129 in-scope rows are the fixed capability axis; do not re-derive or re-count them.

</domain>

<decisions>
## Implementation Decisions

### Parity verdict rubric & fdars-side search depth (SC1, GAP-02, Pitfalls 9/11)
- **D-01:** **Three-value verdict with an explicit rubric, verified by source-grep.** Each of the 129 in-scope capabilities is marked one of:
  - **present** — the core algorithm/capability exists in fdars in *any* call-shape (builder+single-call counts as fit/predict/transform; Pitfall 9);
  - **partial** — the core algorithm is present but key variants/options that scikit-fda offers are missing (e.g. only one hat-matrix strategy where scikit-fda has three; one basis where scikit-fda has eight);
  - **absent** — no fdars equivalent found.
  Verdicts are confirmed by **grepping/reading fdars source** (not STRUCTURE.md alone); STRUCTURE.md is the map that points the search, source confirms the row. Every gap candidate (partial/absent) carries the mandatory "searched fdars for: [behavior]. Closest match: [fn/module]. Verdict: [...]" note (Pitfall 11). — **Reversibility:** costly — the three-value rubric is the schema the whole matrix and the Phase 9 backlog key off; collapsing or re-defining it later forces re-mapping every row.
- **D-01a:** The **"partial" bucket is retained** (not collapsed to binary present/absent) — Pitfall 11 explicitly wants partial-vs-missing separated, and partial rows are distinct backlog candidates (add-a-variant vs implement-from-scratch).

### Accuracy verification (SC4, Pitfall 12)
- **D-02:** **Flag-only accuracy notes — no numeric runs this phase.** The matrix carries an "accuracy verified?" column; fragile / known-bug areas from `CONCERNS.md` (B-spline round-trip & CV GH #33, elastic-alignment level encoding GH #34, seasonal/Lomb-Scargle NaN handling, GMM over-split) are marked **"present — accuracy NOT verified"** with a citation to the CONCERNS.md entry and its fix commit, **never a bare ✓**. No fdars-vs-scikit-fda numeric comparison is run. — **Reversibility:** reversible — a reporting convention; a later phase can run the real smoke-tests.
- **D-02a:** Because accuracy is flagged-not-tested, **draft a backlog item** recommending a comparative numerical-accuracy validation pass (fdars vs scikit-fda on shared datasets) for the fragile areas — so the deferred verification is captured, not lost.

### Gap categorization rubric (SC2, GAP-03, Pitfall 14)
- **D-03:** **Explicit table-stakes / differentiator rubric, Claude applies it across all in-scope gaps.**
  - **table-stakes** — a baseline FDA capability a general-purpose functional-data library is expected to have (e.g. core smoothers, standard depth measures, common basis systems, mainstream regression/classification, Lp metrics);
  - **differentiator** — an advanced / specialised capability whose absence is acceptable but whose presence would set fdars apart (e.g. mixed-effects irregular→basis converters, diffusion maps, RKHS/recursive variable selection, historical/function-on-function regression);
  - **out-of-scope** — carried straight from Phase 7's Relevance taxonomy (plotting/IO/type-system/pipeline plumbing); these are **counted separately** and excluded from the actionable gap total (Pitfall 14).
  Claude classifies each gap and the reviewer checks the finished section; the two-way in-scope split is kept as the roadmap words it (not replaced by a value-band — value ranking is Phase 9). — **Reversibility:** reversible — a classification per row; re-tagging an item is a local edit.

### fdars strengths — reverse-parity sweep (SC3, GAP-04)
- **D-04:** **Full reverse-parity sweep.** Beyond the four headline strengths named in SC3 (model explainability, SPM/control charts, seasonal decomposition, streaming depth), enumerate **every fdars capability that has no scikit-fda 0.10.1 equivalent** — walking the STRUCTURE.md module map for candidates (conformal prediction, tolerance bands, GMM clustering, matrix profile, SSA, Hilbert transform, WIRE, FAMM, elastic changepoint, robust L1/Huber regression, multi-response regression, Andrews curves, etc.). Present as a reverse table (fdars capability → "scikit-fda equivalent: none / partial") so the audit reflects fdars' true advantage surface, not only gaps. — **Reversibility:** reversible — a report table; rows can be added/removed later.

### Output convention (carried forward)
- **D-05:** Append a single `## Phase 8 — Capability Parity Matrix & Categorization` section to `.planning/research/AUDIT-REPORT.md` (Phase 1 D-05 single-file convention, unbroken through Phases 1–7). The categorization rubric, separated counts, strengths sweep, and drafted backlog all live as subsections within that one section.

### Claude's Discretion
- **Matrix layout** — one master table vs six per-area tables mirroring Phase 7's structure; column ordering (Area | Task | Capability | Verdict | Category | Accuracy? | fdars equivalent | Confidence). Planner/executor picks, as long as D-01 (three verdicts), D-03 (categories), and the Pitfall-11/12 notes are all represented.
- **Confidence tagging** on rows (HIGH where source-grep confirmed, MEDIUM where inferred from module map) — mirror the Phase 7 HIGH/MEDIUM/LOW convention.
- **Granularity of drafted backlog entries** — one entry per gap vs sensibly grouped clusters (e.g. "add remaining basis systems" as one entry) — as long as each carries area / current-gap / root-cause (SC4) and stays *unranked* (ranking is Phase 9).
- Ordering within the strengths sweep and whether to group by module or by capability theme.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase definition & requirements
- `.planning/ROADMAP.md` §"Phase 8: Capability Parity Matrix & Categorization" — the 4 success criteria (parity matrix by capability, gap categorization + separated counts, fdars strengths, fragile-area accuracy notes + drafted backlog).
- `.planning/REQUIREMENTS.md` — GAP-02 (parity matrix), GAP-03 (categorization + design-goal filter), GAP-04 (strengths); RPT-01/02/03 (Phase 9) for awareness of what this feeds.

### Primary input — the scikit-fda side (fixed capability axis)
- `.planning/research/AUDIT-REPORT.md` §"Phase 7 — scikit-fda Capability Enumeration" — the **129 in-scope capability rows** across six areas + the Design-Goal Filter (four-value Relevance taxonomy, borderline rulings, separated counts 129 in-scope / 32 out-of-scope / 161 total). This is the left-hand column of the parity matrix; **do not re-derive it**.
- `.planning/research/FEATURES.md` §"Biggest Likely Gaps for fdars (Gap Analysis Head Start)" (line ~700) — Phase-8 head-start material on where gaps are likely; use as a lead, verify against source.

### The fdars side (right-hand column)
- `.planning/codebase/STRUCTURE.md` — the fdars module map; the stated fdars side of the parity search (SC1). Points the source-grep for D-01 verdicts.
- `.planning/codebase/CONCERNS.md` §"Known Bugs (Recent Fixes)" and §"Fragile Areas" — the source list for D-02 "accuracy NOT verified" flags (B-spline #33, elastic alignment #34, seasonal, GMM).
- `.planning/codebase/ARCHITECTURE.md` — module responsibilities, useful for confirming fdars-equivalent verdicts.
- `fdars-core/src/` — grep/read target for confirming present/partial/absent verdicts (D-01) and for the reverse-parity strengths sweep (D-04).

### Methodology / anti-patterns
- `.planning/research/PITFALLS.md` — Pitfall 9 (capability-not-API-name), Pitfall 10 (scikit-fda-has-X ≠ fdars-must), Pitfall 11 ("fdars equivalent searched" note per gap, partial ≠ missing), Pitfall 12 (accuracy-verified column), Pitfall 14 (relevance filter, separated counts). Pitfalls 13/15/16/17 are Phase-9 (ranking/backlog) awareness.
- `.planning/research/SUMMARY.md` §"Highest-impact gaps" and §"Treating scikit-fda as gospel" — context for categorization judgment.

### Project scope anchors
- `.planning/PROJECT.md` §"Out of Scope" and §"Key Decisions" — scikit-fda 0.10.1 sole baseline; plotting/IO/type-system-refactor out of scope; audit-only (no src edits); backlog phrased GSD-ready.

### Output target
- `.planning/research/AUDIT-REPORT.md` — the single append-target report (D-05); Phase 8 adds one `## Phase 8` section.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Phase 7 §Phase 7 tables in AUDIT-REPORT.md** — the 129 in-scope rows with Task/Method/Relevance already tagged; Phase 8 keys the parity matrix directly off these rows (same row order = easy join).
- **`STRUCTURE.md` "Where to Add New Code" + module list** — maps every scikit-fda task grouping to a candidate fdars module (smoothing→`smoothing.rs`/`smooth_basis.rs`, registration→`alignment/`, depth→`depth/`, classification→`classification/`, regression→`scalar_on_function/`, clustering→`clustering.rs`/`gmm/`, metrics→`distance.rs`/`metric.rs`, inference→`famm.rs`/tests, basis→`basis/`).
- **`CONCERNS.md`** — ready-made fragile/known-bug list with file paths and fix commits for the D-02 accuracy flags.
- **Working scikit-fda 0.10.1 venv** at `.planning/research/skfda-verify/venv` — present but **not used** this phase (D-02 flag-only); available if a future phase runs accuracy tests.

### Established Patterns
- Single-file report convention: one `## Phase N` section appended to `AUDIT-REPORT.md` per phase (D-05, Phases 1–7).
- Row confidence tags (HIGH/MEDIUM/LOW) and per-row Source citations — mirror Phase 7's table style.
- Audit-only constraint: zero `fdars-core/src` edits — Phase 8 only reads source and writes `.planning/` docs.

### Integration Points
- Append `## Phase 8 — Capability Parity Matrix & Categorization` to `.planning/research/AUDIT-REPORT.md`.
- Feeds **Phase 9** (RPT-01/02/03): the drafted gap-backlog entries (SC4) + separated in-scope gap counts become the raw material Phase 9 value-ranks; the strengths sweep feeds the consolidated report narrative.

</code_context>

<specifics>
## Specific Ideas

- The parity matrix's left column is **fixed** = Phase 7's 129 in-scope rows; the phase adds the fdars-side columns (verdict, category, accuracy flag, equivalent note). Resist re-enumerating scikit-fda.
- fdars accomplishes tasks via builder-struct + single call returning a result struct — count that as equivalent to scikit-fda's fit/predict/transform (Pitfall 9). A different call shape is **present**, not a gap.
- Known-bug areas must never get a bare ✓ — mark "present — accuracy NOT verified" and cite CONCERNS.md + fix commit (D-02).
- Keep out-of-scope gap counts separate from the actionable in-scope total so the report can't read as "fdars is far behind" (Pitfall 14).
- The reverse-parity strengths sweep should walk the STRUCTURE.md module list for fdars-only capabilities, not just repeat the four SC3 headline areas.

</specifics>

<deferred>
## Deferred Ideas

- **Final value-ranking of the gap backlog** (value/effort, `value/sqrt(effort)`) — **Phase 9** (RPT-02, Pitfall 13). Phase 8 drafts entries unranked.
- **Comparative numerical-accuracy smoke-tests** (fdars vs scikit-fda on shared datasets for fragile areas) — deferred out of this milestone; captured as a **drafted backlog item** this phase (D-02a) rather than executed.
- **Severity / effort estimates and reproducible-evidence links per backlog item** — Phase 9 completeness checklist (RPT-03, Pitfalls 16/17); Phase 8 supplies area/current-gap/root-cause only.
- **Implementing any gap or strength** — future milestone, seeded by the RPT-02 backlog; this is an audit-only milestone.

None of these are scope creep — they are downstream (Phase 9) or explicitly out-of-milestone work, noted to keep Phase 8's parity mapping focused.

</deferred>

---

*Phase: 8-Capability Parity Matrix & Categorization*
*Context gathered: 2026-08-09*
