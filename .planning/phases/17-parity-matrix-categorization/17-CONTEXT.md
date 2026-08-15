# Phase 17: Parity Matrix & Categorization - Context

**Gathered:** 2026-08-14
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — rubrics + approach reused from the v0.14.0 audit precedent; no new material grey areas.

<domain>
## Phase Boundary

Map every **in-scope** R capability (the 248 from Phase 16's §Design-Goal Filter) against fdars, producing a per-capability **present / partial / absent** verdict matched by capability semantics (NOT API name), each row carrying a "searched fdars for:" evidence note + closest-match fdars reference (or "no match found"). Then **categorize** every absent/partial gap as **table-stakes / differentiator / out-of-scope**. Append `§Parity-Matrix` and `§Categorization` to `.planning/research/R-AUDIT-REPORT.md`. Covers GAP-01, GAP-02. Does NOT do the fdars-side reverse-parity strengths sweep (Phase 18) or the ranked backlog (Phase 19).
</domain>

<decisions>
## Implementation Decisions

### Verdict rubric (reuse v0.14.0 D-01 — documented once in the report)
- **present** — fdars has a capability delivering the same result, even if API/name differs. (Accuracy not re-verified here; flag known-bug areas as "present — accuracy NOT verified" if applicable, per v0.14.0 convention.)
- **partial** — fdars has a related/adjacent capability but not the full behavior (e.g. internal-only, missing a documented sub-mode, or a narrower variant).
- **absent** — no fdars capability delivers the result; closest match noted or "no match found".

### Category rubric (reuse v0.14.0 D-03 — documented once)
- **table-stakes** — a capability a general-purpose FDA library is expected to have; its absence is a competitive deficit.
- **differentiator** — valuable but specialized; nice-to-have, not baseline-expected.
- **out-of-scope** — should not be built (plotting/IO, or outside fdars' numeric-library design goals). Note: in-scope R capabilities are the mapping set, so "out-of-scope" here is rare and reserved for capabilities that, on inspection, are really rendering/IO-adjacent.

### Evidence sourcing for the fdars side
- **Primary reuse:** the v0.14.0 audit already catalogued fdars against ~141 scikit-fda capabilities with "searched fdars for:" notes (`AUDIT-REPORT.md` §Phase 8) AND a 30-item reverse-parity strengths sweep. Reuse those fdars-side findings wherever an R capability overlaps a scikit-fda one already assessed — do not re-derive.
- **Codebase maps:** `.planning/codebase/` (module map, structure, conventions) for the fdars capability surface.
- **Direct search:** `grep`/`glob` over `fdars-core/src/` to confirm present/partial/absent and cite the closest module/function.
- **CRITICAL — credit v0.15.0–v0.17.0 additions** (these post-date the v0.14.0 audit and are now PRESENT in fdars; do not mark them absent):
  - spline interpolation (`spline_interpolate`), functional summary statistics (variance/std/covariance/depth-median/trim_mean), missing-value imputation (`impute_missing_values`), composable `ExtrapolationPolicy`, functional scoring metrics (MAE/MSE/MAPE/MSLE/explained-variance), **least-squares shift registration** (`least_squares_shift_registration`) + three **registration-quality scores** (least_squares / pairwise_correlation / sobolev), banded elastic alignment (`*_with_band`), parallel CV folds, faer FPCA SVD, parallel elastic-FPCA.

### Coverage & matching
- Map ALL 248 in-scope capabilities (GAP-01). Reuse Phase 16's 9-area taxonomy; total verdicts per area + overall; make the actionable-gap count (absent + partial) explicit.
- Match by capability semantics, per Phase 16's Pitfalls (mark "present" if fdars covers the capability regardless of which R package offers it; check the numeric underpinning, not the plot).
- Watch the high-gap areas Phase 16 flagged: Area 7 (`frechet`/`fdadensity` object-data/density — likely large gap), `fdaPDE` FEM smoothing, `fdapace` sparse/PACE FPCA, `funLBM` co-clustering, function-on-function regression, functional time series.

### Claude's Discretion
- Table column layout, per-area sub-grouping, and how to present the "present — accuracy NOT verified" flags are at discretion, provided all 4 ROADMAP SCs are met.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `.planning/research/AUDIT-REPORT.md` §Phase 8 — the scikit-fda parity matrix (141 rows) + 30-item reverse-parity strengths sweep: the fdars-side catalogue to reuse. §Phase 8 rubrics D-01 (verdict) and D-03 (category) are the rubrics to reuse verbatim.
- `.planning/research/R-AUDIT-REPORT.md` §R-Inventory + §Design-Goal Filter — the 248 in-scope R capabilities to map (the input set).
- `.planning/codebase/` — fdars module/structure/convention maps.
- `fdars-core/src/` — direct grep target for present/partial/absent confirmation.
- PROJECT.md "Validated" list — enumerates every shipped fdars capability incl. v0.15.0–v0.17.0 additions.

### Established Patterns
- Append to the existing `.planning/research/R-AUDIT-REPORT.md` (do NOT overwrite Phase 16's sections; do NOT touch the archived scikit-fda `AUDIT-REPORT.md`). Audit-only: zero `fdars-core/src/` edits.

### Integration Points
- §Parity-Matrix + §Categorization feed Phase 19's consolidated findings + ranked backlog (each absent/partial table-stakes/differentiator gap becomes a candidate backlog item).
</code_context>

<specifics>
## Specific Ideas

- Report anchors: `## Phase 17 — Parity Matrix & Categorization`, with `§Parity-Matrix` (per-area verdict tables + a per-area/overall verdict count table) and `§Categorization` (gap → table-stakes/differentiator/out-of-scope).
- Make the headline actionable-gap number explicit (absent + partial, in-scope) — the input to Phase 19 ranking.
</specifics>

<deferred>
## Deferred Ideas

- Reverse-parity fdars-strengths sweep (fdars capabilities with no R equivalent / where fdars leads) → Phase 18.
- Consolidated findings + value-ranked backlog → Phase 19.
</deferred>
