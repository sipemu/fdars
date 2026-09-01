# Phase 52: Ecosystem Surveys - Context

**Gathered:** 2026-09-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Four fresh reference ecosystems (MATLAB FDA, Julia FDA, tidyfun/refund, Python-beyond-scikit-fda) are surveyed capability-first, fdars is mapped present/partial/absent against each, and each survey emits a de-duplicated net-new gap list — the raw material Phase 53 consolidates. Audit-only: zero `fdars-core/src/` edits. Deliverables are markdown documents under `.planning/research/`.

Explicitly out of scope for these surveys: re-auditing scikit-fda (v0.14.0) or the core R FDA ecosystem `fda`/`fda.usc`/`refund`/`fdapace`/`roahd`/`ftsa`/`frechet` (v0.18.0–v0.29.0); plotting/visualization parity; data/IO parity. TDY-01 touches refund ONLY where not captured in v0.18.0; PYX-01 excludes scikit-fda.

</domain>

<decisions>
## Implementation Decisions

### Survey Depth & Scope
- **Capability-first breadth**: cover every major capability *category* per ecosystem, not every function. Enough to surface net-new gaps, not an API-complete catalog.
- **Version pinning**: pin the latest stable release of each package as of the survey date (2026-09) and cite it as `pkg@version` in every inventory row.
- **MATLAB scope**: survey BOTH the Ramsay `fda` MATLAB toolbox AND PACE (MATLAB), per roadmap.
- **Julia scope**: survey the actively-maintained JuliaStats / functional-data packages, and explicitly capture Julia performance/idiom patterns (e.g. type-stable/generic/GPU-friendly designs) as candidate gaps where they represent a capability fdars lacks.

### Gap Identification & De-dup Rigor
- **Granularity**: one gap per distinct capability / method-family (grouped), NOT per-function.
- **Evidence standard**: every absent/partial row carries an explicit "searched fdars for:" note — the grep terms used plus capability reasoning across `src/` modules — cross-checked against the PROJECT.md Validated-capabilities list (40+ entries). Mapped by capability, not API name.
- **De-dup vs prior backlogs**: check each candidate gap by *capability* (not API name) against both `BACKLOG.md` (v0.14.0) and `R-BACKLOG.md` (v0.18.0); drop any match. A gap earns a row only if verified absent from shipped fdars AND absent from both prior backlogs.
- **Partial coverage**: where fdars partially covers a capability, list it as partial and emit the *missing slice* as a net-new gap when warranted.

### Deliverable Layout (parallel-plan safety)
- **Four intermediate per-survey files**: `.planning/research/survey-matlab.md`, `survey-julia.md`, `survey-tidyfun.md`, `survey-pyx.md`. This prevents four parallel plans from colliding on a single file and matches the roadmap's "raw material Phase 53 consolidates". Phase 53 (RPT-01) merges these into `.planning/research/GAP-AUDIT-REPORT.md`.
- **Net-new gap list**: each survey file ends with a "Net-New Gap List" table.
- **Reverse-parity**: each survey notes where fdars *leads* the ecosystem, feeding Phase 53's reverse-parity strengths sweep.
- **Standardized gap-table columns** across all four surveys so Phase 53 can merge mechanically: `Capability | Reference (pkg@ver) | fdars status (present/partial/absent) | Searched-for | Net-new? (vs BACKLOG/R-BACKLOG) | Notes`.

### Claude's Discretion
- Exact package selection within each ecosystem (which Julia/Python packages qualify as in-scope and actively maintained), the specific grep terms per capability, and per-survey section ordering are at Claude's discretion, guided by the decisions above.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Prior audit deliverables** as templates and de-dup baselines: `.planning/research/AUDIT-REPORT.md` + `BACKLOG.md` (v0.14.0 scikit-fda), `.planning/research/R-AUDIT-REPORT.md` + `R-BACKLOG.md` (v0.18.0 R ecosystem). Reuse their table shape and `value/√effort` scoring convention.
- **PROJECT.md Validated section** (40+ shipped-capability entries) — the authoritative present/absent baseline for fdars.
- **fdars-core `src/` module map** (from CLAUDE.md / MEMORY.md): regression, classification, clustering, depth, alignment/elastic, seasonal, spm, explain/explain_generic, streaming_depth, irreg_fdata, function_on_scalar_2d — grep targets for the "searched fdars for" evidence.

### Established Patterns
- Audit milestones mirror the v0.14.0 / v0.18.0 shape: parallel enumeration/parity surveys → consolidated report + ranked backlog + completeness gate (gate last).
- Distinct filenames — new deliverables must NOT overwrite existing `AUDIT-REPORT.md`/`BACKLOG.md` or `R-AUDIT-REPORT.md`/`R-BACKLOG.md`.

### Integration Points
- The four survey files are consumed by Phase 53: RPT-01 (report), RPT-02 (ranked backlog), RPT-03 (de-dup + completeness gate). Standardized columns are the merge contract.

</code_context>

<specifics>
## Specific Ideas

- No git tag / no crate publish for this milestone (crate unchanged — audit-milestone convention; would otherwise publish a phantom version).
- Web research (docs, package registries, source repos) is expected to enumerate the four ecosystems' current capability surfaces with version pins.

</specifics>

<deferred>
## Deferred Ideas

- Implementing any gap found in this audit — drawn top-first from `GAP-BACKLOG.md` in a future implementation milestone.
- Migrating the `fdars-r` R wrapper to the `FdMatrix` API (issue `fdars-j75`) — carried forward, not this (audit-only) milestone.

</deferred>
