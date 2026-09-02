# Phase 54 · Summary 54-01: GAK Kernel Core

**Status:** complete
**Requirements delivered:** GAK-01, GAK-02, GAK-03, GAK-04
**Crate version:** unchanged (0.30.0 — bump is a milestone-end step)

## Files changed

| File | Change |
|------|--------|
| `fdars-core/src/metric/gak.rs` | **New.** Full GAK kernel core (log-domain DP, triangular normalization, Gram builder, σ heuristic) + inline tests. |
| `fdars-core/src/metric/mod.rs` | `pub mod gak;` + `pub use gak::{gak, gak_gram_matrix, sigma_gak, GakConfig};` |
| `fdars-core/src/lib.rs` | Crate-root re-export of `gak, gak_gram_matrix, sigma_gak, GakConfig` alongside the `soft_dtw_*` surface. |

No existing signature changed; additive/non-breaking.

## Public API added

- `pub struct GakConfig { pub sigma: Option<f64> }` — `Debug/Clone/PartialEq/Default`, serde-gated, `#[non_exhaustive]`; `GakConfig::with_sigma(f64)`.
- `pub fn gak(x: &[f64], y: &[f64], sigma: f64) -> f64` — normalized pairwise similarity in `[0,1]`, `#[must_use]`. Has a doctest.
- `pub fn gak_gram_matrix(data: &FdMatrix, config: &GakConfig) -> Result<FdMatrix, FdarError>` — n×n symmetric PSD Gram, `#[must_use]`. Has a doctest showing SVM-ready usage.
- `pub fn sigma_gak(data: &FdMatrix) -> f64` — median-distance bandwidth heuristic with positive floor.

Internal seams (for Phase 55):
- `pub(crate) fn loggak(x, y, sigma) -> f64` — unnormalized log-domain forward DP (2-row rolling buffer, O(m) memory).
- `pub(crate) fn logsumexp3(a, b, c) -> f64` — stable 3-way log-sum-exp (soft-MAX; NEG_INFINITY-safe, never NaN).
- private `log_local`, `normalize_log`.

## Implementation notes

- **Log-domain DP is the only path.** `loggak` accumulates `L[i][j] = log_local(i,j) + logsumexp3(L[i-1][j], L[i][j-1], L[i-1][j-1])`, boundary `L[0][0]=0`, else `-inf`. `log_local = -d²/(2σ²) - ln(2 - exp(-d²/(2σ²)))` (Cuturi TGAK triangular local kernel).
- **Normalization in log space:** `exp(logGAK(x,y) - 0.5·(logGAK(x,x)+logGAK(y,y)))`; `-inf` numerator → `0.0` (NaN guard).
- **Gram builder:** pre-collects rows once; precomputes the n diagonal self-log-kernels in one parallel pass (no 2× recompute); computes upper triangle via `iter_maybe_parallel!`; sets diagonal to exactly `1.0`; mirrors `G[j][i]=G[i][j]` (bit-exact symmetry). σ resolved from config or `sigma_gak`.
- **σ heuristic:** exact median of full-curve pairwise Euclidean distances, floored at `1e-8`.

## Divergences

- **Reference test approach:** tslearn is not installed in this environment, so `test_gak_vs_reference` asserts against **hand-derived** analytic values from the Cuturi TGAK formula (documented in-comment as a by-hand DP), not fabricated tslearn numbers. Two cases: `gak([0,1],[0,2],σ=1)=0.44843221961236995` (full DP written out in the doc comment) and `gak([0,1,2],[0,1,3],σ=2)=0.805752775914924`. Both matched a scratch linear-space reference to 1e-9.
- **σ heuristic form:** fdars uses the exact median of full-curve Euclidean distances (deterministic, no RNG) rather than tslearn's random point-pair sampling × `sqrt(median_length)`. Documented in the `sigma_gak` doc comment. Produces a healthy off-diagonal band (verified by `test_sigma_gak_healthy`).

## Tests added (11 lib + 2 doctests, all green)

`test_logsumexp3_basic`, `test_gak_no_underflow` (m=200, off-diag > 1e-10),
`test_gak_normalized_range` ([0,1] + unit diagonal + finite), `test_gak_gram_symmetric` (bit-exact),
`test_gak_gram_psd` (min eigenvalue ≥ −1e-8 via `symmetric_eigenvalues`),
`test_gak_parallel_matches_sequential` (bit-identical recomputation),
`test_sigma_gak_healthy` (auto-σ off-diagonal in healthy range), `test_sigma_gak_floor_on_identical`,
`test_gak_vs_reference` (hand-derived), `test_gak_gram_empty_errors`, `test_gak_gram_bad_sigma_errors`.

## Gate results

- `cargo fmt --check` → clean (exit 0).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → clean (`Finished` in 14.27s, 0 warnings). Fixed two findings: `neg_cmp_op_on_partial_ord` (rewrote σ guard to `sigma.is_nan() || sigma <= 0.0`) and `excessive_precision` (trimmed a reference constant).
- `cargo test -p fdars-core --features linalg gak` → `11 passed; 0 failed`; doctests `2 passed; 0 failed`.
- `cargo test -p fdars-core gak` (default features, parallel) → `11 passed; 0 failed`.
