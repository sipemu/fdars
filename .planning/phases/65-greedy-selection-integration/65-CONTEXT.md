# Phase 65: Greedy Selection & Integration - Context

**Gathered:** 2026-09-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Wrap Phase 64's validated `design_criterion` in a deterministic greedy sequential
forward-selection loop, exposing the whole two-stage FOptDes workflow:
`optimal_design(model: &PaceFpcaResult, config: &OptDesConfig) -> Result<OptDesResult, FdarError>`.
Add the `OptDesConfig`/`OptDesResult` types, finalize the additive crate-root + `prelude`
re-exports (full public surface), add a module-level doctest of the end-to-end workflow, and
land a criterion benchmark. A thin orchestration layer over Phase 64 — **adds no new math**.

All new code lands in the existing `fdars-core/src/optimal_design.rs` (extend it) plus additive
`lib.rs`/`prelude.rs` re-exports and one new `benches/` file. Additive/non-breaking: no existing
public signature changes; 28 examples + WASM + R bindings unaffected; no new crate dependency;
MSRV 1.81 preserved; `linalg` feature NOT required.

Out of scope (deferred FOD-BREADTH → future milestone): SR-criterion, exhaustive/branch-and-bound
search, CV-ridge selection, rank-1 Cholesky update, off-grid interpolated candidates.

</domain>

<decisions>
## Implementation Decisions

### OptDesConfig / OptDesResult API Surface
- **`OptDesConfig` carries a single `criterion: DesignCriterion` field** — NOT a separate
  `criterion` + `optimality` pair. `DesignCriterion::Score(OptimalityKind)` already wraps the
  optimality kind, and `Trajectory` needs none; a separate `OptimalityKind` field would be
  redundant and ignored for the trajectory case. (Overrides the roadmap success-criterion's
  literal "DesignCriterion, OptimalityKind" field list, which was descriptive shorthand.)
- **`OptDesConfig` fields:** `candidate_grid: Vec<f64>`, `budget: usize`, `criterion: DesignCriterion`.
  `Default` impl (NOT `#[non_exhaustive]`) = `{ candidate_grid: vec![], budget: 1,
  criterion: DesignCriterion::Trajectory }` — safe minimal defaults; validation catches the empty
  grid at call time. Derives: `Debug, Clone, PartialEq` + serde-gated, following `PaceFpcaConfig`.
- **`OptDesResult` fields:** `selected_indices: Vec<usize>`, `selected_argvals: Vec<f64>`,
  achieved-criterion trace (`Vec<f64>`, one entry per greedy step). `#[non_exhaustive]`. Derives:
  `Debug, Clone, PartialEq` + serde-gated, following `PaceFpcaResult`.

### Greedy Loop Contract (FOD-04)
- Start empty; at each of `config.budget` steps, add the not-yet-selected candidate index that
  most reduces `config.criterion` (evaluated through Phase 64's `design_criterion`), until budget
  reached.
- **Determinism (make-or-break):** parallelize candidate EVALUATION via `iter_maybe_parallel!`, but
  take a SEQUENTIAL argmin with **smallest-index tie-break** (rayon `min_by` is not stable). Two
  identical calls → byte-identical `selected_indices`, and seq == parallel builds agree. Mirror the
  `pace_fpca.rs` determinism-test pattern.
- Exclude already-selected indices each step (no index appears twice — duplicate-free).
- Achieved-criterion trace is monotone non-increasing across steps.
- Candidates constrained to the work grid: every `config.candidate_grid` value must appear (within
  FP tolerance) in `model.argvals`; map to the grid index for exact `design_criterion` evaluation.

### Two-Stage Entry (FOD-05)
- `optimal_design` consumes the supplied `PaceFpcaResult` **read-only** — NO re-estimation of
  eigenstructure or σ². Pure design over the already-fitted model.

### Validation
- Return `Err(FdarError::InvalidParameter)` for: `budget == 0`, `budget > candidate_grid.len()`,
  any candidate not in `model.argvals`, `model.ncomp == 0`, `model.sigma2 <= 0`.

### Re-exports, Doctest, Benchmark
- **Full additive crate-root re-export:** `pub mod optimal_design`; re-export `optimal_design`,
  `design_criterion`, `OptDesConfig`, `OptDesResult`, `DesignCriterion`, `OptimalityKind`. Add the
  same names to `prelude`.
- Module-level doctest demonstrating end-to-end: fit PACE → `optimal_design` → read
  `selected_argvals`. (Watch /tmp tmpfs during doctest linking — see hazards.)
- **One criterion benchmark file** covering both `design_criterion` and `optimal_design` for
  Trajectory and Score(A) on a representative grid/budget. Register as a `[[bench]]` in
  `fdars-core/Cargo.toml` (criterion 0.5 harness).

### Claude's Discretion
- Internal greedy-loop helper factoring, benchmark input sizes, and test-module layout — at
  Claude's discretion, following `pace_fpca.rs` / `kshape.rs` conventions.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 64's `design_criterion(model, selected: &[usize], criterion: DesignCriterion)` — public,
  validated, `#[must_use]`; the greedy loop delegates entirely to it. `DesignCriterion` /
  `OptimalityKind` enums already exist and are re-exported.
- `pace_fpca::PaceFpcaResult` (read-only borrow), `iter_maybe_parallel!` (parallel candidate sweep,
  no RNG — criterion is deterministic), `linalg::cholesky_solve/factor`, `helpers::simpsons_weights`.
- `PaceFpcaConfig`/`PaceFpcaResult` as the config/result derive + `#[non_exhaustive]` precedent.
- Existing `benches/` criterion files as the benchmark template; existing `[[bench]]` entries in
  `fdars-core/Cargo.toml`.

### Established Patterns
- Extend the existing top-level `src/optimal_design.rs` (do not add a submodule dir).
- Additive `lib.rs` peer re-export lines (mirror the kshape/kernel_kmeans peer block) + `prelude.rs`.
- Public fns return `Result<T, FdarError>`; `#[must_use]` (with message) on `optimal_design`.
- Inline `#[cfg(test)] mod tests`; determinism test asserting seq == parallel and stable tie-break.

### Integration Points
- `lib.rs` + `prelude.rs` re-export sites; `fdars-core/Cargo.toml` `[[bench]]` registration.
- The greedy loop reads `model.argvals` to map `candidate_grid` values → grid indices for
  `design_criterion`.

</code_context>

<specifics>
## Specific Ideas

- Determinism WITH and WITHOUT `--features parallel` is a first-class gate (mirror pace_fpca
  determinism test). Smallest-index tie-break on a sequential argmin, never rayon `min_by`.
- Known pre-existing tech debt (NOT this phase): `cargo build --features serde` already fails on
  `shapelet/classifier.rs` (Phase 60) embedding non-serde `ClassifFit` — the FOptDes types are
  serde-clean; do not let a serde-feature build failure be mistaken for a Phase 65 regression.

</specifics>

<deferred>
## Deferred Ideas

- FOD-BREADTH (SR-criterion, exhaustive/branch-and-bound, CV-ridge selection, rank-1 Cholesky
  update, off-grid interpolated candidates) → future milestone.
- Fixing the pre-existing `--features serde` build break (add serde derives to `ClassifFit` +
  fitted sub-structs) → separate backlog item, not FOptDes scope.

</deferred>
