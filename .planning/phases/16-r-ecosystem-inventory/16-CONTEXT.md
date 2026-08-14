# Phase 16: R Ecosystem Inventory - Context

**Gathered:** 2026-08-14
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — one material grey area resolved by user; remainder at Claude's discretion per ROADMAP success criteria

<domain>
## Phase Boundary

Produce the **R side** of the parity comparison: a versioned, area-organized, capability-first inventory of the R functional-data-analysis ecosystem, then apply a design-goal filter tagging each capability in-scope (numeric/API-ergonomics) or out-of-scope (plotting/IO). Deliverable is the `§R-Inventory` section of a new `.planning/research/R-AUDIT-REPORT.md` plus a per-area in-scope/out-of-scope count table. This is the R analog of v0.14.0 Phase 7 (scikit-fda capability enumeration). Covers INV-01 and INV-02. Does NOT map against fdars (Phase 17) or walk fdars' own modules (Phase 18).
</domain>

<decisions>
## Implementation Decisions

### Data Source & Versioning (user-resolved)
- **Source R capability data from model knowledge cross-checked against CRAN / pkgdown documentation** — do NOT install R or any packages locally, do NOT run `packageVersion()`.
- Cite each package's version as its **latest CRAN release as of the survey** (record the survey month, e.g. "as of 2026-08"), and state this convention explicitly once in the report methodology. Where a package's exact latest version is uncertain from knowledge, mark it `~<version>` or "latest CRAN" rather than inventing a precise string.
- No live/runtime verification of R APIs — this is a capability/API survey, consistent with the audit-only, no-`fdars-core`-edits fence.

### Area Organization
- Group capabilities into named areas mirroring the v0.14.0 scikit-fda audit, adapted for R's broader surface: **representation/basis-smoothing, preprocessing/registration, exploratory/depth-outlier, ML (regression + classification + clustering), inference/testing, functional-time-series, density/object-data/manifold, misc/utilities**. Merge or split areas as the surfaced capabilities warrant; every area carries a capability count.
- **Capability-first granularity:** one row per capability, collapsing fit/predict/transform and S3/S4 method variants into a single row (Pitfall 9 from the v0.14.0 audit — avoid one-row-per-API-name inflation). Tag each row with its source package(s); a capability offered by several packages lists them.

### In / Out-of-Scope Filter (rule documented once)
- **In-scope** = numeric algorithm OR API-ergonomics capability portable to a numeric Rust library.
- **Out-of-scope** = plotting/visualization (`rainbow` bagplots, `roahd`/`fdaoutlier` graphics, `fda` plot methods, `ggplot`/base-R rendering) OR data/IO (dataset loaders, `read`/`write` round-trips). Note: the *numeric underpinnings* of graphical diagnostics (e.g. outliergram / MS-plot statistics) are in-scope even though the plot is not.
- Produce a per-area in-scope vs out-of-scope count table → the total actionable comparison surface Phase 17 maps against.

### Package Coverage
- Minimum core set (from REQUIREMENTS.md INV-01): `fda`, `fda.usc`, `refund`, `fdapace`, `roahd`, `fdaoutlier`, `ftsa`, `MFPCA`/`funData`, `fdasrvf`, `fdatest`/`fdANOVA`, `frechet`/`fdadensity`, `funHDDC`/`FDboost`.
- Add any further FDA-relevant packages surfaced during the survey (candidates: `face`, `fpca`, `classiFunc`, `funFEM`/`funLBM`, `rainbow` [likely out-of-scope], `FRegSigCom`, `denseFLMM`, `refund.shiny` [out-of-scope]). Document which packages were considered and excluded, and why.

### Claude's Discretion
- Exact area taxonomy boundaries, row-level wording, table column set, and how to present version-uncertainty flags are at Claude's discretion, provided all four Phase-16 success criteria are met.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `.planning/research/AUDIT-REPORT.md` §Phase 7 (scikit-fda capability enumeration) — the structural template for §R-Inventory: capability-first rows, area grouping with counts, and the in-scope/out-of-scope design-goal filter (161-row inventory → 129 in-scope / 32 out-of-scope). Reuse its rubric and table shape; do NOT overwrite it.
- `.planning/research/BACKLOG.md` — the downstream format Phase 19 mirrors (not needed to author the inventory, but sets the destination convention).

### Established Patterns
- Deliverables land in `.planning/research/` under **new, distinct filenames** — `R-AUDIT-REPORT.md` (this + later phases append sections) and, in Phase 19, `R-BACKLOG.md`. Never overwrite the archived scikit-fda `AUDIT-REPORT.md` / `BACKLOG.md`.
- Audit-only: **zero `fdars-core/src/` edits** in any v0.18.0 phase.

### Integration Points
- §R-Inventory (this phase) is consumed directly by Phase 17 (parity matrix maps each in-scope R capability to fdars) and its area taxonomy is reused by Phases 18–19.
</code_context>

<specifics>
## Specific Ideas

- Report section anchor: `## Phase 16 — R Ecosystem Inventory` with `§R-Inventory` and a `§Design-Goal Filter` subsection, mirroring the v0.14.0 report's §Phase 7 layout.
- Include a short methodology preamble stating the knowledge+CRAN sourcing convention and the survey month, so version citations are interpreted correctly downstream.
</specifics>

<deferred>
## Deferred Ideas

- fdars-vs-R parity verdicts and gap categorization → Phase 17.
- Reverse-parity fdars-strengths sweep (fdars module-map walk) → Phase 18.
- Consolidated findings + value-ranked backlog → Phase 19.
</deferred>
