# Phase 64 — Summary 64-02: Score Criterion and Re-export

**Status:** complete
**Requirements:** FOD-02, FOD-03
**Commit (impl):** ada84c2e

## Files
- **Modified** `fdars-core/src/optimal_design.rs` — replaced the 64-01 Score placeholder with the real `score_criterion` posterior-covariance branch (A = trace, D = log-det); added 6 Score/dispatch inline tests.
- **Modified** `fdars-core/src/lib.rs` — additive only: `pub mod optimal_design;` (near the `kshape` peer decls) and `pub use optimal_design::{design_criterion, DesignCriterion, OptimalityKind};` (near the other module re-exports). No existing line removed.

No greedy loop, no `OptDesConfig`/`OptDesResult`, no prelude change, no benchmark (all Phase 65).

## Public API added
```rust
// Now reachable at fdars_core::{design_criterion, DesignCriterion, OptimalityKind}
// and fdars_core::optimal_design::*
pub use optimal_design::{design_criterion, DesignCriterion, OptimalityKind};
```

## Implementation notes
- `score_criterion` computes the K×K posterior score covariance `Cov(ξ|Y_S) = Λ − Λ Φ_dᵀ Σ_d⁻¹ Φ_d Λ` via the **shared p×p `build_sigma_design`** solve — the `A_mat`/`Ω_i` pattern from `pace_fpca.rs:547–558`, substituting `Φ_d`/`Σ_d` for `Φ_i`/`Σ_yi`:
  1. Factor `Σ_d` once (same ridge-retry wrapper as 64-01).
  2. Per component k, solve `Σ_d x_k = Φ_d[:,k]` (forward/back), scale `sigma_inv_phi_lam[j,k] = λ_k · x_k[j]`.
  3. `A_mat[k,l] = λ_k · Σ_j Φ_d[j,k]·sigma_inv_phi_lam[j,l]`; `Cov[k,l] = (k==l ? λ_k : 0) − A_mat[k,l]`.
- **A-optimality**: `trace(Cov) = Σ_k Cov[k,k]`.
- **D-optimality**: `log det(Cov)` via `cholesky_factor(Cov)` + `linalg::log_det_from_cholesky` — returned **un-negated** (NEGATIVE for an informative design; Pitfall 4). Not abs'd.
- The per-component solve is against the **p×p `Σ_d`** (never a K×K system; Pitfall 5); the σ²I term lives inside `build_sigma_design`.
- Empty-set fast paths avoid any solve: A → `Σ_k λ_k`; D → `Σ_k log λ_k` directly (with a defensive `λ ≤ 0` → `ComputationFailed` guard), never factoring the diagonal Λ.

## Tests + results (all pass)
Six added this plan: `test_score_a_empty_set`, `test_score_d_empty_set`, `test_score_prior_recovery`, `test_monotonicity_a_opt`, `test_monotonicity_d_opt`, `test_enum_dispatch`. With the 8 trajectory tests from 64-01, **14/14 optimal_design tests pass**.
- `A(∅) = 3.0 = Σλ_k`; `D(∅) = ln 2 = Σ log λ_k` (both within 1e-10).
- `test_score_prior_recovery`: A and D both recover `Cov(ξ|∅) = diag(λ)` exactly.
- A-opt and D-opt monotone non-increasing (`s1 ≤ s0 + 1e-12`).
- Full suite: **2671 lib tests pass** (was 2657; +14) — 0 failed. All integration + doctest suites 0 failed.

## Divergences
- **`test_enum_dispatch` contract corrected (not weakened).** The 64-02 plan's behavior spec said the three variants must yield "three mutually distinct" values. That assumption is **mathematically false for the synthetic orthonormal model**: when eigenfunctions are orthonormal w.r.t. the integration weights, `∫ Var[x̂(t)] dt = Σ_{k,l} Ω_{kl} ∫φ_k φ_l = trace(Ω) = trace(Cov(ξ))`, so **Trajectory ≡ A-optimality is an exact algebraic identity** (observed: `traj=1.35992048625003_27` vs `a=...25`, agreeing to 15 sig-digits). The test now asserts (a) all three `Ok`+finite, (b) the orthonormality identity `|traj − a| < 1e-9` — which itself proves Trajectory runs the real integral, not a stub — and (c) `d` (log-det, a distinct code path) is distinct from and `< a`, confirming separate routing. This encodes the true contract; it does not hide a dispatch bug.
- **`cargo build -p fdars-core --features serde` FAILS — pre-existing, unrelated blocker.** The crate does not compile under `--features serde` because `shapelet::ShapeletTransformClassifier` (added in Phase 60, commit `ea39c623`) embeds a `ClassifFit` field, but `classification::ClassifFit` lacks serde derives (`E0277: ClassifFit: serde::Serialize not satisfied`, ×4). Verified by stashing all Phase-64 changes and rebuilding: the identical 4 errors persist with zero references to `optimal_design`. My enums (`Trajectory`, `Score(OptimalityKind)`, `A`, `D`) are trivially serde-derivable and contribute no error. This is out of Phase 64's additive-only scope; fixing `ClassifFit`/`ShapeletTransformClassifier` serde should be its own backlog item. **FOD-03's serde-gated-derive obligation is met for the new types**, but the crate-wide `--features serde` gate cannot go green until the pre-existing shapelet defect is fixed.

## Gate tails
- `cargo fmt -p fdars-core --check` → clean (exit 0).
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` → `Finished`, no warnings.
- `cargo test -p fdars-core --features linalg,parallel` → lib `test result: ok. 2671 passed; 0 failed`; every integration/doc suite `0 failed`.
- `cargo build -p fdars-core --features serde` → **FAILS** on pre-existing `ClassifFit` serde defect (see Divergences); not introduced by this phase.

## Seams for Phase 65
- `fdars_core::design_criterion` + the `DesignCriterion`/`OptimalityKind` enums are now the public surface for hand-chosen designs and the greedy wrapper's per-candidate scorer.
- The greedy loop stays pure orchestration: repeatedly call `design_criterion(model, &candidate_set, criterion)` and keep the minimizer; both criteria and all three variants are proven monotone non-increasing, so greedy minimization is safe.
- Phase 65 should also add the full re-export surface (prelude entry) and consider filing the `ClassifFit` serde fix so the crate's `--features serde` build is restored.
