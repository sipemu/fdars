# Phase 64 — Summary 64-01: Sigma-Design and Trajectory Criterion

**Status:** complete
**Requirements:** FOD-01, FOD-03
**Commit (impl):** c3d551c2

## Files
- **Created** `fdars-core/src/optimal_design.rs` — new top-level peer module (like `kshape.rs`/`kernel_kmeans.rs`): `DesignCriterion` + `OptimalityKind` enums, public `design_criterion` entry point with full input validation, shared private `build_sigma_design`, the ridge-retry factor helper, `build_phi_d`, the trajectory-reconstruction BLUP-MSE branch, and 8 inline trajectory known-answer tests. Score arm is a compile placeholder (`Ok(0.0)`), overwritten in 64-02.

No `lib.rs` change in this plan (the module is committed as an orphan file; wiring lands in 64-02). No version bump, no new dependency, no benchmark.

## Public API added (declared here; re-exported in 64-02)
```rust
pub enum DesignCriterion { Trajectory, Score(OptimalityKind) }   // Debug/Clone/PartialEq, serde-gated
pub enum OptimalityKind  { A, D }                                 // Debug/Clone/PartialEq, serde-gated

#[must_use = "..."]
pub fn design_criterion(
    model: &PaceFpcaResult, selected: &[usize], criterion: DesignCriterion,
) -> Result<f64, FdarError>;
```

## Implementation notes
- `build_sigma_design`: assembles the `p×p` (p = `selected.len()`, NOT K×K) row-major `Σ_d = Φ_d diag(λ) Φ_dᵀ + σ²I_p`, mirroring `pace_fpca.rs:461–474`. The `+= model.sigma2` on the diagonal is applied after each row's inner loop (the classic-bug guard).
- Ridge-retry lives in `factor_sigma_design_with_retry`: `cholesky_factor(Σ_d)` once; on `Err`, add `1e-8` to every diagonal and retry once; second failure → `FdarError::ComputationFailed { operation: "optimal_design Sigma_d Cholesky", ... }`. Never panics.
- Performance: `Σ_d` is factored **once** via `cholesky_factor`, then the trajectory grid loop uses `cholesky_forward_back` per grid point → O(m·p²), not O(m·p³) (Pitfall 6).
- Trajectory branch uses `helpers::simpsons_weights(&model.argvals)` (never uniform `1/m`) — the link that makes `MSE(∅)` grid-invariant. Empty-set fast path returns `Σ_j w_j Σ_k λ_k φ_k(t_j)²` with no solve; non-empty accumulates `w_j·(prior_var_j − rhs_jᵀ Σ_d⁻¹ rhs_j)` where `rhs_j[i] = Σ_k λ_k φ_k(t_j) φ_k(argvals[selected[i]])`.
- Validation at `design_criterion` entry: `ncomp == 0`, `eigenvalues.len() < ncomp`, `sigma2 <= 0.0`, and any out-of-range index → `FdarError::InvalidParameter`. Duplicate indices tolerated (documented).
- Cholesky helpers (`cholesky_factor`/`cholesky_forward_back`) are `pub(crate)` and always compiled — no `#[cfg(feature = "linalg")]` anywhere.

## Tests + results (all pass)
`test_trajectory_empty_set`, `test_trajectory_grid_invariance`, `test_trajectory_reduces_on_point`, `test_monotonicity_trajectory`, `test_validation_index_range`, `test_validation_sigma2`, `test_validation_ncomp`, `test_ridge_retry` — **8/8 pass** under `cargo test -p fdars-core --features linalg --lib optimal_design`.
- `test_trajectory_empty_set`: `MSE(∅) = 3.0 = Σλ_k` within 1e-10.
- `test_trajectory_grid_invariance`: `MSE(∅)` equal within 1e-10 across m=21/51/101 (proves Simpson weights).
- `test_ridge_retry`: `sigma2 = 1e-12`, design `&[10,20,30]` → `Ok` (no panic).

## Divergences
- Added a `model.eigenvalues.len() < model.ncomp` validation guard (not named in the plan) so that a `ncomp>0` model with a too-short eigenvalues vector fails cleanly at entry rather than panicking on an index — defensive, still returns `InvalidParameter`.
- `#[must_use]` carries the codebase-standard message `"expensive computation whose result should not be discarded"` (matching `scalar_on_function/*`), because a bare `#[must_use]` on a `Result` return trips clippy's `double_must_use` under the CI `-D warnings` gate.

## Gate tails
- `cargo fmt -p fdars-core --check` → clean (exit 0).
- `cargo clippy -p fdars-core --features linalg --lib` → clean, 0 warnings.
- `cargo test -p fdars-core --features linalg --lib optimal_design` → `test result: ok. 8 passed; 0 failed`.
- `cargo build -p fdars-core` → compiles (module committed as an orphan file, not yet re-exported).

## Seams for Phase 65
- `build_sigma_design`, `factor_sigma_design_with_retry`, and `build_phi_d` are the shared numerical primitives the greedy sweep reuses per candidate — no new math needed in the wrapper.
- `design_criterion` returns a single minimizable scalar for any hand-chosen `selected`; the greedy loop just calls it over candidate additions and keeps the minimizer.
