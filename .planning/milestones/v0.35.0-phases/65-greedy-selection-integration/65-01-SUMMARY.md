# Phase 65 — Summary 65-01: Greedy Selection Loop + Config/Result Types

**Status:** complete
**Requirements:** FOD-04, FOD-05
**Commit (impl):** a7bb7c92

## Files
- **Modified** `fdars-core/src/optimal_design.rs` — added `OptDesConfig`, `OptDesResult`, `optimal_design()`, the private `map_candidates_to_indices()` helper, and 13 inline `#[cfg(test)]` tests. Phase-64 code and its 15 tests left intact.

Additive only — no existing signature changed. No new crate dependency. Not re-exported from `lib.rs`/`prelude.rs` in this plan (that is 65-02).

## Public API added
```rust
pub struct OptDesConfig {              // Debug/Clone/PartialEq, serde-gated, NOT non_exhaustive
    pub candidate_grid: Vec<f64>,
    pub budget: usize,
    pub criterion: DesignCriterion,
}
impl Default for OptDesConfig { /* { vec![], 1, Trajectory } */ }

pub struct OptDesResult {              // Debug/Clone/PartialEq, serde-gated, #[non_exhaustive]
    pub selected_indices: Vec<usize>,
    pub selected_argvals: Vec<f64>,
    pub criterion_trace: Vec<f64>,
}

#[must_use = "..."]
pub fn optimal_design(model: &PaceFpcaResult, config: &OptDesConfig)
    -> Result<OptDesResult, FdarError>;
```
Reachable within the crate as `crate::optimal_design::{OptDesConfig, OptDesResult, optimal_design}`.

## Implementation notes
- Single `criterion: DesignCriterion` field (no separate `OptimalityKind` — `Score(OptimalityKind)` already carries it). `Default` mirrors `PaceFpcaConfig`; `OptDesResult` mirrors `PaceFpcaResult` (`#[non_exhaustive]`).
- Validation fails fast (all `FdarError::InvalidParameter`, never panics): `budget == 0`, `budget > candidate_grid.len()`, `model.ncomp == 0`, `model.sigma2 <= 0.0`, then off-grid candidate during index mapping.
- Candidate→index mapping is a one-time FP-tolerant position search (`|t - cand| < 1e-9`), preserving `candidate_grid` order; off-grid → `InvalidParameter("config.candidate_grid")`.
- Greedy loop: per step, build `remaining` (candidate_indices filtered to exclude `selected`, in order), **parallel-evaluate** each via `iter_maybe_parallel!(remaining)` mapping `idx → (idx, design_criterion(model, &trial, criterion.clone())?)`, `?`-collect into `Vec<(usize,f64)>`, then take a **sequential** `fold`-based argmin with strict `<` (smallest-index tie-break). No rayon `min_by`.
- Model consumed read-only (no re-estimation); each closure captures only `&model`/`&selected` and owns its `trial` — `PaceFpcaResult` is Send+Sync so `--features parallel` compiles.

## Divergences
- `iter_maybe_parallel!` requires an `IntoParallelIterator` argument; a `Copied<Iter<usize>>` does not qualify. Fixed by moving the owned `Vec<usize> remaining` into the macro directly (rebuilt fresh each step, so no clone needed). Behavior unchanged.

## Tests + results (all pass)
13 new: `test_optimal_design_basic`, `test_determinism_two_calls`, `test_duplicate_free`, `test_monotone_trace`, `test_validation_budget_zero`, `test_validation_budget_exceeds_grid`, `test_validation_off_grid_candidate`, `test_validation_ncomp_zero`, `test_validation_sigma2_nonpositive`, `test_trajectory_selects_informative_point`, `test_score_a_selects`, `test_config_default`, `test_prelude_reexport`.
- `cargo test -p fdars-core --lib optimal_design --features linalg,parallel` → **28 passed** (15 Phase-64 + 13 new), 0 failed.
- `cargo test -p fdars-core --lib optimal_design` (default, parallel OFF) → **28 passed** — determinism/seq==parallel confirmed.
- `test_trajectory_selects_informative_point` computes the expected argmin in-test (not hardcoded).

## Gate tails
- `cargo fmt -p fdars-core --check` → clean (exit 0).
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` → Finished, no warnings.

## Seams for 65-02
- `optimal_design`, `OptDesConfig`, `OptDesResult` exist in `optimal_design.rs` ready for additive `lib.rs`/`prelude.rs` re-export, the module doctest, and the benchmark (all 65-02).
- Doctest/bench must build the `PaceFpcaResult` via the real `pace_fpca` fit path (it is `#[non_exhaustive]`, no struct literal from an external crate).
