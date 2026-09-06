# 76-02 SUMMARY — DIF-03: Differentiable FPCA score projection

**Status:** COMPLETE
**Commit:** `2fce54a9` — `feat(autodiff): differentiable FPCA score projection (DIF-03)`

## Symbols added

- `fdars_core::regression::project_scores_generic<S: Scalar>` (pub free fn; re-exported at crate root via `lib.rs pub use regression::{...}`)
- `fdars_core::regression::FpcaResult::project_generic<S: Scalar>` (pub method wrapper, `#[must_use]`)

`FpcaResult::project` and all `FpcaResult` fields are UNCHANGED.

## Implementation

`project_scores_generic` mirrors `FpcaResult::project`'s inner loop exactly:
`score_k = Σ_j (curve[j] − mean[j]) · rotation[(j,k)] · weights[j]`, with `mean`/`rotation`/`weights`
staying f64 and only `curve: &[S]` generic. The `rotation[(j,k)] * weights[j]` product is folded into a
single f64 before lifting via `S::from_f64`, so the analytic gradient is exactly that f64 constant.

## Analytic-gradient check result (SC #5 strong form)

`project_scores_generic::<Dual>` gradient of `score_k` w.r.t. `curve[j]` equals the closed form
`rotation[(j,k)] * weights[j]` to **≤1e-12** for all j∈0..40, k∈0..3. PASS — the strongest possible AD
check (a linear op → constant gradient), on SPANNING full-rank training curves (random combos of
sin(πt), cos(2πt), sin(3πt) with distinct per-curve coefficients, per project memory on rank-deficient
data silently passing).

## Other gate results for this plan

- SC #5 FD arm: Dual gradient vs central FD (h=1e-8) to ≤1e-6 for all (j,k). PASS.
- SC #6 f64 parity: `project_scores_generic::<f64>` reproduces `FpcaResult::project` (1-row `FdMatrix`)
  to ≤1e-12. PASS.

## Gate results

- Targeted (`fpca`): `fpca_score_gradient_dual`, `fpca_score_generic_f64_parity`, `fpca_score_gradient_vs_fd` — **all PASS**
- Whole crate (`--features linalg,parallel`): **2853 lib passed, 0 failed** (all suites 0 failed)
- clippy `--all-targets --features linalg,parallel -- -D warnings`: **clean**
- `cargo fmt --check`: **clean**
- `git diff --stat fdars-core/Cargo.toml`: **empty** (no new dependency); `grep num-traits` = 0
- MSRV 1.81 preserved

## Constraints honored

No existing public f64 signature changed (`FpcaResult::project` untouched). `FdMatrix` stays `Vec<f64>`.
No new crate dependency. Single-line `lib.rs` re-export addition as specified.

## Deviations

None.
