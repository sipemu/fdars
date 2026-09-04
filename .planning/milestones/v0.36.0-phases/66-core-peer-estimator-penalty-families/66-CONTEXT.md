# Phase 66: Core PEER Estimator & Penalty Families - Context

**Gathered:** 2026-09-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a public `peer()` estimator for structured-penalty scalar-on-function
regression. It estimates the coefficient function β(t) via the
partially-empirical-eigenvector decomposition (null-space + range-space of the
penalty operator) and supports three a-priori penalty families. This is the
estimator that distinguishes PEER from plain FPCR/`pfr`.

In scope (PER-01, PER-02):
- `peer(data, y, argvals, &config)` public entry point returning a result struct
  carrying β(t), intercept, fitted values, and df/selection diagnostics.
- Three penalty families selectable via a penalty-type enum: ridge/identity,
  2nd-difference/roughness (reusing the existing `penalty_matrix` builder), and a
  caller-supplied structured "decree" Q matrix.
- Fixed λ (supplied via config, sensible default). No auto-selection.

Out of scope (later phases):
- Automatic λ selection (GCV / REML) — Phase 67.
- Longitudinal `lpeer`, out-of-sample `predict`, crate-root/prelude exports,
  end-to-end doctest — Phase 68.

</domain>

<decisions>
## Implementation Decisions

### Module Structure & API Surface
- New code lives in a single top-level file `src/peer.rs` (ROADMAP: likely ONE
  new file; may split into a `peer/` submodule later if it grows).
- Public estimator signature follows the crate convention:
  `peer(data, y, argvals, &config)` — functional data first, response second,
  optional evaluation points, then config.
- Configuration carried in a `PeerConfig` builder struct (matches
  `ElasticConfig` / `GmmClusterConfig` pattern), with serde behind the `serde`
  feature per crate convention.
- Result struct `PeerResult` carries: β(t) (`beta`), `intercept`,
  `fitted_values`, `effective_df`, `lambda` used, and the selected `penalty_type`.
  Derives `Debug, Clone, PartialEq`; `#[non_exhaustive]` per crate convention.

### Penalty Family Representation
- Penalty families represented by a `PeerPenalty` enum with three variants:
  `Ridge` (identity), `Difference { order }` (2nd-difference roughness), and
  `Decree(Q)` (caller-supplied raw structured matrix).
- Roughness penalty reuses the existing `penalty_matrix` builder
  (`smooth_basis` / `function_on_scalar`), not a hand-rolled one.
- "Decree" Q is a caller-supplied raw matrix (`Vec<f64>` + dimensions), the most
  general form — matches refund's `pentype="DECREE"`. No partition DSL.
- Default penalty family is `Difference { order: 2 }` (roughness), matching
  refund's default.

### Decomposition & Numerics
- β(t) represented pointwise on the `argvals` grid (Q is p×p on the grid) —
  PEER's original formulation.
- Null-space / range-space split obtained by eigendecomposition/SVD of the
  penalty operator Q via nalgebra (crate SVD convention).
- Linear solve reuses the penalized normal-equations + Cholesky pattern
  (`penalized_solve` in `function_on_scalar.rs`). Claude's discretion whether to
  make that helper crate-visible or replicate the pattern locally.
- Intercept handled by centering y and the design, recovering the intercept as
  ȳ (matches `fregre_lm`).

### Fixed λ, Diagnostics & Errors
- Single `lambda` field in `PeerConfig` with default `1.0`. Automatic selection
  is deferred to Phase 67; an explicit λ is honored verbatim.
- A wrong-dimension Q (or otherwise invalid penalty input) returns
  `FdarError::InvalidDimension { parameter, expected, actual }` — never panics.
- df/selection diagnostic is the effective degrees of freedom via the trace of
  the smoother/hat matrix, plus the λ used.
- Functional inner-product integration uses Simpson's weights
  (`helpers::simpsons_weights`), crate convention.

### Claude's Discretion
- Whether to expose/replicate `penalized_solve`.
- Exact internal helper factoring within `peer.rs`.
- Precise default numeric tolerances for the null/range decomposition.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `function_on_scalar.rs::penalized_solve` (private) — penalized normal-equations
  + Cholesky solve; `penalty_matrix` builder used there and in
  `function_on_scalar_2d.rs`.
- `smooth_basis.rs` — `bspline_penalty_matrix` / `fourier_penalty_matrix`
  roughness penalty builders; `SmoothBasisResult.penalty_matrix`.
- `regression.rs::fdata_to_pc_1d` — FPC basis (available if a basis projection is
  ever needed).
- `linalg.rs` — Cholesky/ridge solve helpers (behind `linalg` feature).
- `helpers.rs::simpsons_weights` — integration weights for functional inner
  products.
- `error.rs::FdarError` — `InvalidDimension`, `InvalidParameter`,
  `ComputationFailed`, `InvalidEnumValue`.

### Established Patterns
- Column-major `FdMatrix`; rows = curves, columns = evaluation points.
- All public fns return `Result<T, FdarError>`; validate dims at entry.
- Config structs use the builder pattern with conditional serde derives.
- Result structs derive `Debug, Clone, PartialEq`, `#[non_exhaustive]`,
  `#[must_use]` on expensive computations.
- Inline `#[cfg(test)] mod tests` per file.

### Integration Points
- Register `pub mod peer;` in `src/lib.rs`. Crate-root + prelude re-exports are
  deferred to Phase 68 (this phase only needs the module reachable for tests);
  add the `pub mod` now.

</code_context>

<specifics>
## Specific Ideas

- Reference baseline: refund@0.1-38 `peer` (pentype ∈ {Ridge, D/Diff, DECREE}).
- Known-answer gates from the ROADMAP success criteria: β(t) recovery on
  synthetic data; all three families fit without error; structured Q yields a
  partition-aware β(t) distinct from plain roughness; wrong-dim Q → descriptive
  `FdarError`, no NaN β(t).

</specifics>

<deferred>
## Deferred Ideas

- Automatic λ selection (GCV / REML) — Phase 67.
- Longitudinal `lpeer`, out-of-sample `predict`, crate-root/prelude exports, and
  the end-to-end module doctest — Phase 68.

</deferred>
