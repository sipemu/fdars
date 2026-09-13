# Phase 106: Top-Level Complexity Hotspots - Context

**Gathered:** 2026-09-13
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure/refactor phase — discuss skipped)

<domain>
## Phase Boundary

Decompose the worst top-level complexity hotspots in `fdars-core` into smaller
private helper functions so each falls under pmat's cognitive/cyclomatic
thresholds — **behavior and numeric output unchanged**. In-scope functions
(pmat 3.18.0 baseline):

- `funfem_cluster` (cyclomatic 59)
- `elastic_net_cd` (cognitive 54)
- `wnet_cv_lambda` (cognitive 45)
- `wnet` (cognitive 31)
- `boost_fofr` (cyclomatic 39)
- `pace_fpca` (cyclomatic 37)
- `bayesian_fosr` (cyclomatic 36)
- `kcfc_cluster` (cyclomatic 28)

Out of scope: the SPM cluster (Phase 107), entropy/SATD/unwrap cleanup (Phase 108),
and the remaining complexity tail (Phase 109).

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure refactor phase.
Constraints (from PROJECT.md / STATE.md, non-negotiable):

- **Strictly non-breaking + numeric-preserving.** Every change is pure internal
  helper extraction. No public API / signature / visibility changes.
- **Reduce complexity by helper extraction only** — never by changing signatures.
- **`fdars-core` only** — protects R + WASM bindings.
- **No new crate dependency** — pmat is a dev/CI tool invoked externally.
- Proof of behavior-preservation = the full existing test suite (2942 lib tests +
  215 doctests + 28 examples) staying green, plus `pmat analyze complexity`
  confirming each target function drops below threshold.
- Extracted helpers are private (`fn`, not `pub fn`), placed adjacent to their
  parent in the same module, following the crate's existing helper conventions.

</decisions>

<code_context>
## Existing Code Insights

Target functions span the regression / clustering / FPCA modules. Codebase
patterns to follow (from CLAUDE.md conventions):

- Private helpers use `snake_case`, `_1d`/`_2d`/`_nd` dimensionality suffixes.
- Column-major `FdMatrix` layout — helpers taking matrix slices must respect
  `data[i + j*nrows]` indexing; prefer existing row methods (`row_to_buf`,
  `row_dot`, `row_l2_sq`).
- Numerical inner loops may carry `#[inline]`; keep hot-path helpers inlinable.
- Precise file locations and per-function decomposition strategy to be
  determined during plan-phase research (pmat pinpoints the offending spans).

</code_context>

<specifics>
## Specific Ideas

No specific requirements beyond the named functions and thresholds — refactor
phase. The pmat complexity report is the authoritative target list; each named
function must fall under pmat's cyclomatic + cognitive thresholds after
decomposition.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
