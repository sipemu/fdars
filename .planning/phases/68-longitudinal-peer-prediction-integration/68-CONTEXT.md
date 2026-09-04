# Phase 68: Longitudinal PEER, Prediction & Integration - Context

**Gathered:** 2026-09-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Final phase of the v0.36.0 PEER milestone. Three deliverables:
1. **`lpeer()`** — longitudinal PEER: extends `peer()` to repeated per-subject
   measurements with subject-level random effects, fitted via
   `famm::fit_scalar_mixed_model` (REML EM), returning the (time-varying)
   coefficient function β(t) + variance components (PER-04).
2. **`predict`** — out-of-sample prediction from a fitted `peer`/`lpeer` result;
   coefficient-function + fitted-value accessors (PER-05).
3. **Integration** — full public surface (estimators, result structs, penalty +
   λ enums, predict) reachable from the crate root AND prelude, plus a running
   end-to-end module doctest (PER-05).

Out of scope: none — this closes the milestone. After this phase: milestone
audit → complete → crate bump 0.35.0→0.36.0 → publish on the v0.36.0 tag.

</domain>

<decisions>
## Implementation Decisions

### lpeer API & result
- Signature: `lpeer(data, y, argvals, subject_map, &config) -> Result<LpeerResult, FdarError>`.
  `subject_map: &[usize]` gives the subject index per observation (length n);
  `config` is the existing `PeerConfig` (penalty + `LambdaChoice`).
- Dedicated `LpeerResult` struct: `{ beta, intercept, w_bar, fitted_values,
  sigma2_subject: f64, sigma2_resid: f64, n_subjects, lambda, penalty_type, .. }`.
  Derives `Debug, Clone, PartialEq`; `#[non_exhaustive]`; `#[must_use]`; serde
  behind feature. (Separate from `PeerResult` because it carries variance
  components + subject count.)
- Variance components are non-negative (rely on `famm`'s clamping) — assert σ²≥0.
- Validate `subject_map`: `len == n`, all indices `< n_subjects` (derive
  `n_subjects` as `max(subject_map)+1` or require it) → descriptive `FdarError`.

### famm integration for lpeer
- Reduce the functional predictor to a low-dimensional score representation
  (PEER penalty range-space basis, or `regression::fdata_to_pc_1d` FPC scores),
  pass those scores as `covariates` + `subject_map` to
  `famm::fit_scalar_mixed_model(y, subject_map, n_subjects, Some(scores), p)`, then
  back-project the fixed-effect coefficients to β(t) on the argvals grid. This
  mirrors the existing `fof_regression.rs` famm usage (`x_scores` → mixed model).
- Reuse `fit_scalar_mixed_model` directly — it IS subject-grouped (the correct
  tool here, unlike Phase 67's smoothing random effect which needed a bespoke EM).
- Carry the `PeerPenalty` / λ choice from Phase 67 into the coefficient estimate
  (structured penalty applied to β(t)).

### predict (out-of-sample)
- Method on the result: `PeerResult::predict(&self, new_data: &FdMatrix,
  argvals: &[f64]) -> Result<Vec<f64>, FdarError>` (and the analogous
  `LpeerResult::predict`). Caller supplies `argvals` (they have it from fitting);
  prediction uses `intercept`, `w_bar`, and `beta`:
  `ŷ* = (intercept − w_bar·β) + Σ_j x*[j]·w[j]·β[j]`, `w = simpsons_weights(argvals)`.
- Self-consistency: re-passing the training curves + argvals reproduces the
  training `fitted_values` (the Phase 66 `w_bar` field is what makes this exact) —
  assert in a test.
- lpeer prediction on genuinely new curves is **marginal / population-level**:
  fixed-effect β(t) only, new-subject random effect = 0.
- Return `Result<Vec<f64>, FdarError>` — validate `new_data` ncols == m (the
  training grid length); descriptive error on mismatch.

### Exports & end-to-end doctest
- Crate root (`lib.rs`): `pub use peer::{peer, lpeer, PeerConfig, PeerResult,
  LpeerResult, PeerPenalty, LambdaChoice, LambdaMethod}` (explicit list, matching
  the crate's no-wildcard convention, alphabetical slot near `regression`).
- Prelude (`prelude.rs`): the same set added via `pub use crate::peer::{...}`.
- Replace the current module-header `no_run` snippet with a **running** `//!`
  doctest demonstrating the end-to-end workflow: build data → `peer()` fit → read
  β(t) → `predict` on new curves. Must pass under `cargo test --doc`.
- `lpeer` gets a short separate example (its own `///` doctest or a second block).

### Claude's Discretion
- Exact score-reduction basis for lpeer (penalty range-space vs FPC) — pick
  whichever integrates most cleanly with `fit_scalar_mixed_model`'s covariate
  interface; document the choice.
- Whether `n_subjects` is a parameter or derived from `subject_map`.
- Internal helper factoring; doctest data sizes (keep small + deterministic).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `famm.rs`: `fit_scalar_mixed_model(y, subject_map, n_subjects, covariates:
  Option<&FdMatrix>, p) -> ScalarMixedResult` (pub(crate), ~line 438); result
  carries scalar `sigma2_u` (subject variance) + `sigma2_eps` (residual). Clamps
  variance components non-negative.
- `fof_regression.rs`: existing pattern reducing functional X to scores + calling
  `fit_scalar_mixed_model` — the template for lpeer's famm integration.
- `regression.rs::fdata_to_pc_1d` — FPC scores (~line 331) if score reduction uses FPC.
- `peer.rs` (Phases 66–67): `peer()`, `build_q`, `PeerConfig`, `PeerResult`
  (already carries `w_bar` for prediction), `PeerPenalty`, `LambdaChoice`,
  `LambdaMethod`, `select_lambda_gcv_peer`, `select_lambda_reml_peer`.
- `helpers::simpsons_weights` — for predict's weights.
- `scalar_on_function/mod.rs`: existing `predict(&self, new_data, new_scalar)`
  method pattern on result structs (the convention to follow).

### Established Patterns
- Result structs: `Debug, Clone, PartialEq`, `#[non_exhaustive]`, `#[must_use]`,
  conditional serde. All public fns `Result<T, FdarError>`.
- Crate root + prelude re-exports are explicit `pub use` (no wildcards).
- Inline `#[cfg(test)] mod tests`; deterministic; no RNG in library code.

### Integration Points
- `lib.rs`: `pub mod peer;` already present (Phase 66); ADD the `pub use peer::{...}`
  re-export block this phase.
- `prelude.rs`: ADD `pub use crate::peer::{...}`.

</code_context>

<specifics>
## Specific Ideas

- Reference baseline: refund `lpeer` (longitudinal PEER, subject random effects via
  mgcv/mixed model) + `predict.peer`.
- Known-answer gates (ROADMAP): variance components non-negative; lpeer recovers β(t)
  + tracks injected between-subject variance on synthetic longitudinal data; predict
  self-consistent (re-passed training curves == training fitted values) + finite on
  new curves; full surface crate-root/prelude reachable; module doctest passes under
  `cargo test --doc`.
- STATE note confirmed: `famm::fit_scalar_mixed_model` IS the right tool for lpeer's
  subject random effects (its subject-grouping is exactly the longitudinal structure).

</specifics>

<deferred>
## Deferred Ideas

None — this phase closes the v0.36.0 milestone. Any further PEER enhancements
(user-tunable GCV grid, additional penalty orders, subject-specific prediction with
BLUPs) are future-milestone backlog.

</deferred>
