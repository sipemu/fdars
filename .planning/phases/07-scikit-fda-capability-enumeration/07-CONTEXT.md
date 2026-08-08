# Phase 7: scikit-fda Capability Enumeration - Context

**Gathered:** 2026-08-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the **scikit-fda side** of the eventual parity comparison: a versioned, area-organized, capability-oriented inventory of scikit-fda's public surface, plus a one-page design-goal filter that Phase 8 will apply. Delivers **analysis artifacts only** — a report section appended to the single `.planning/research/AUDIT-REPORT.md`; **no `fdars-core/src` changes** (audit-only milestone).

Concretely, this phase produces, appended as a `## Phase 7 — scikit-fda Capability Enumeration` section to `.planning/research/AUDIT-REPORT.md`:
1. scikit-fda's public capability surface **enumerated by the six report areas** — representation, preprocessing, exploratory, ML, inference, misc (SC1).
2. The **exact compared scikit-fda version pinned and recorded** (verified against PyPI `__version__`, baseline 0.10.1) in the methodology (SC2).
3. Enumeration is **capability-oriented, not API-name counting** — fit/predict/transform families grouped by user task to avoid Pitfall 9's 2–3× gap inflation (SC3).
4. A **one-page design-goal filter** (in-scope numeric algorithms vs out-of-scope plotting/IO/sklearn-pipeline) to be applied in Phase 8 (SC4).

**Not in scope:**
- Mapping scikit-fda against fdars / marking present/partial/absent — that is **Phase 8** (GAP-02). This phase produces the scikit-fda side only; fdars-side gap notes must be **stripped** during promotion of FEATURES.md (see D-02).
- Any `fdars-core/src` edits.
- Accuracy/parity spot-checks against reference datasets (Phase 8 CONCERNS handling).

</domain>

<decisions>
## Implementation Decisions

### Version Pinning & Verification (SC2)
- **D-01:** **Hybrid verification with graceful fallback.** Attempt a throwaway venv install of `scikit-fda==0.10.1` to capture `skfda.__version__` at runtime (and spot-check a few module `dir()` listings for the enumeration). If the install fails on Python 3.14 / heavy-dependency compat, **fall back** to PyPI release metadata + the readthedocs 0.10.1 API reference. **Record explicitly in the report methodology which path was used** and why. Rationale: 0.10.1 is confirmed the latest PyPI release (matches the agreed baseline), so a docs-based pin is already defensible; the venv attempt upgrades the evidence to a literal `__version__` check when the environment allows, without letting install compat block the phase. — **Reversibility:** reversible — verification-method choice, no shipped-code impact.
- **D-01a:** Baseline stays **0.10.1** — it is both the agreed sole baseline (PROJECT.md Key Decisions) and the current latest on PyPI, so no re-pin decision is needed. If a newer version appears at planning time, note it but keep 0.10.1 for consistency with the existing research.

### Reuse of Existing Research (efficiency + confidence)
- **D-02:** **Promote & refactor the existing `.planning/research/FEATURES.md`** into the authoritative report section rather than re-enumerating from scratch. Steps: (a) extract the scikit-fda-only enumeration, (b) **strip out the fdars gap notes** ("fdars has X / partial") — those belong to Phase 8, (c) re-verify entries against the 0.10.1 source per D-01, (d) reorganize FEATURES.md's 12 sub-areas under the **six SC1 report areas**, (e) raise the confidence tag from MEDIUM where the D-01 verification supports it. Rationale: FEATURES.md already enumerates the full 0.10.1 surface by area at MEDIUM confidence; promoting it builds on verified work and avoids drift, while the strip step keeps the scikit-fda-only deliverable clean. — **Reversibility:** reversible — a report-authoring convention; a later phase can re-derive if a claim needs tightening.

### Capability Unit / Grouping (Pitfall 9, SC3)
- **D-03:** **Task-grouped headers with method-family rows.** Two-level structure: the six report areas contain task groupings (smoothing, registration, dimensionality reduction, depth/outlier, classification, regression, clustering, inference, metrics/norms, …), and within each, **one row per distinct method/algorithm** (each smoother, each depth measure, each classifier), **collapsing fit/predict/transform/inverse_transform of the same method into a single row**. Rationale: coarser user-task-only rows would hide *which specific methods* are absent (what Phase 8's parity matrix needs), while one-row-per-class drifts back toward the API-name counting Pitfall 9 warns against. This is the granularity Phase 8 consumes to mark parity per method. — **Reversibility:** costly — the row granularity is the schema Phase 8's parity matrix keys off; changing it later forces re-mapping the matrix.

### Design-Goal Filter (SC4, Pitfall 14)
- **D-04:** **Adopt Pitfall 14's 4-value Relevance taxonomy** as a column/classification on the enumeration: **In-Scope Algorithm / In-Scope API-Ergonomics / Out-of-Scope (plotting) / Out-of-Scope (IO)**. Phase 8's GAP-03 consumes this directly (table-stakes/differentiator/out-of-scope categorization). Borderline items get **explicit rulings** in the one-pager:
  - Plotting / `Visualization` / matplotlib integration → **Out-of-Scope (plotting)**.
  - DataFrame / pandas round-trips, dataset/sample-data loaders → **Out-of-Scope (IO)**.
  - `FDAFeatureUnion` / `PerClassTransformer` / sklearn-Pipeline plumbing → **Out-of-Scope** (Rust equivalent is trait composition, not an API port; PROJECT.md).
  - Representation type-system (`FDataGrid` / `FDataBasis` / `FDataIrregular`) → the **type-system refactor is out of scope** (PROJECT.md), but specific *algorithmic* capabilities riding on it (e.g. irregular-data covariance estimation) are **In-Scope Algorithm** and enumerated as such.
  - Report **in-scope vs out-of-scope counts separately** so the actionable gap count (Phase 8) is not inflated by plotting/IO. — **Reversibility:** reversible — a reporting taxonomy; re-classifying an item is a local edit.

### Output Convention (carried forward)
- **D-05:** Append a single `## Phase 7 — scikit-fda Capability Enumeration` section to `.planning/research/AUDIT-REPORT.md` (Phase 1 D-05 single-file convention). The design-goal filter is written as a one-page subsection within that section (not a separate file), so it travels with the enumeration it filters.

### Claude's Discretion
- Exact table column layout (e.g. Area | Task | Method | fit/predict/transform collapsed-note | Relevance | Confidence | Source) — planner picks, as long as D-03 granularity and D-04 taxonomy are represented.
- Whether the venv install (D-01) uses `python -m venv` + `pip` vs a `uv`/`pipx` ephemeral env — an environment-mechanics call; document the exact command and outcome in the methodology.
- How finely to record `__version__`-verification evidence (captured stdout snippet vs a one-line note) depending on which D-01 path succeeds.
- Ordering of the 12 FEATURES.md sub-areas under the six report areas, and whether any sub-area is split or merged, as long as all six SC1 areas are covered.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase definition & requirements
- `.planning/ROADMAP.md` §"Phase 7: scikit-fda Capability Enumeration" — the 4 success criteria (area enumeration, version pin, capability-not-name grouping, one-page design-goal filter).
- `.planning/REQUIREMENTS.md` — GAP-01 (this phase's requirement); GAP-02/03/04 (Phase 8) for awareness of what this enumeration feeds.

### Primary reusable input (promote, don't re-derive)
- `.planning/research/FEATURES.md` — **the existing 0.10.1 scikit-fda API enumeration across 12 sub-areas**, MEDIUM confidence, verified against readthedocs. This is the source to promote/refactor per D-02 (strip fdars notes, re-verify, reorganize under the six report areas). Contains §"Feature Landscape Summary", §"Feature Dependencies", §"Biggest Likely Gaps for fdars (Gap Analysis Head Start)" (the last is Phase-8 material — defer), and §"Sources".

### Measurement / analysis discipline
- `.planning/research/PITFALLS.md` §"Pitfall 9: Counting API Names Instead of Capabilities" (governs D-03 — group by user task, treat builder+single-call as equivalent to fit/predict/transform) and §"Pitfall 14: Letting Plotting and IO Features Inflate the Gap Count" (governs D-04 — the 4-value Relevance taxonomy, separate in/out-of-scope counts).
- `.planning/research/SUMMARY.md` §"Highest-impact gaps" and §"Treating scikit-fda as gospel" — context on where the enumeration will matter most and the design-goal-filter rationale.

### Project scope anchors
- `.planning/PROJECT.md` §"Out of Scope" and §"Key Decisions" — scikit-fda 0.10.1 as sole baseline; plotting/IO out of scope; representation type-system refactor may be *noted* not built (informs D-04 borderline rulings).

### Output target
- `.planning/research/AUDIT-REPORT.md` — the single append-target report (Phase 1 D-05 convention); Phase 7 adds a `## Phase 7 — scikit-fda Capability Enumeration` section (D-05).

### External source (for D-01 verification)
- scikit-fda 0.10.1 API reference — `fda.readthedocs.io` (docs-fallback path); PyPI project `scikit-fda` (confirmed latest = 0.10.1) for the `__version__` pin.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`.planning/research/FEATURES.md`** — near-complete first draft of the deliverable: full 0.10.1 enumeration by area. Phase 7 promotes/refactors it (D-02) rather than starting blank.
- **`.planning/research/AUDIT-REPORT.md`** — established section/append conventions from Phases 1–6 to mirror for the Phase 7 section header, methodology note, and evidence-linking style.
- **`.planning/research/PITFALLS.md`** — Pitfalls 9 & 14 give ready-made grouping rules and the Relevance taxonomy; no need to invent the schema.
- Working `pip`/PyPI network access and Python 3.14.5 present — enables the D-01 venv attempt; scikit-fda is **not** currently installed (a fresh venv install is required for the runtime `__version__` path).

### Established Patterns
- Single-file report convention: every phase appends one `## Phase N` section to `AUDIT-REPORT.md` (Phase 1 D-05).
- Evidence-linking discipline: each finding cites its source (here: docs URL or captured `__version__`/`dir()` output), mirroring how perf phases linked `bench/` artifacts.
- Audit-only constraint: zero `fdars-core/src` edits across all phases — Phase 7 touches only `.planning/` docs and a throwaway venv.

### Integration Points
- Append `## Phase 7 — scikit-fda Capability Enumeration` to `.planning/research/AUDIT-REPORT.md`.
- Feeds **Phase 8** (GAP-02/03/04): the method-family rows (D-03) are the capability axis of the parity matrix; the 4-value Relevance taxonomy (D-04) is the categorization input for GAP-03.

</code_context>

<specifics>
## Specific Ideas

- The enumeration is the **scikit-fda side only** — resist the urge to write "fdars has this" anywhere; that comparison is Phase 8.
- Six report areas are fixed by SC1: representation, preprocessing, exploratory, ML, inference, misc. FEATURES.md's 12 sub-areas map onto these (e.g. smoothing + registration + dimensionality-reduction → preprocessing; depth/outlier + summary-stats → exploratory; classification + regression + clustering → ML).
- 0.10.1 is confirmed the current latest on PyPI, so the version pin is both "baseline" and "latest" — record that coincidence explicitly to preempt "is this stale?" questions.
- Collapse rule (D-03): a scikit-fda estimator's `fit`/`predict`/`transform`/`inverse_transform` = **one** capability row, matching fdars' builder-struct + single-call shape (Pitfall 9).

</specifics>

<deferred>
## Deferred Ideas

- **fdars-side parity mapping** (present/partial/absent, "fdars equivalent searched" notes) — **Phase 8** (GAP-02). FEATURES.md's §"Biggest Likely Gaps for fdars" is Phase-8 head-start material, not Phase-7 output.
- **Gap categorization** (table-stakes / differentiator / out-of-scope) and **documenting fdars strengths** — **Phase 8** (GAP-03, GAP-04).
- **Numerical-accuracy spot-checks** against scikit-fda reference datasets — Phase 8 CONCERNS handling (B-spline CV, elastic alignment fragility).
- **Re-pinning to a newer scikit-fda** if one releases — noted; keep 0.10.1 for this milestone's consistency.

None of these are scope creep — they are downstream (Phase 8) or explicitly out-of-milestone work, noted to keep Phase 7's enumeration focused on the scikit-fda side.

</deferred>

---

*Phase: 7-scikit-fda Capability Enumeration*
*Context gathered: 2026-08-09*
