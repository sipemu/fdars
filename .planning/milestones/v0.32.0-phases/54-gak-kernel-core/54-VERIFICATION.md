---
phase: 54
title: GAK Kernel Core
status: passed
requirements: [GAK-01, GAK-02, GAK-03, GAK-04]
verified: 2026-09-02
---

# Phase 54 Verification — GAK Kernel Core

All 5 ROADMAP success criteria PASS. Evidence is the inline test suite in
`fdars-core/src/metric/gak.rs` (11 lib tests + 2 doctests, all green under both
`--features linalg` and default features), plus fmt/clippy gates.

| # | Success criterion | Verdict | Evidence |
|---|-------------------|---------|----------|
| 1 | Pairwise GAK returns non-zero for long series (m ≥ 100–400), off-diagonal > 1e-10 — proving log-domain, not raw-product | **PASS** | `test_gak_no_underflow` (m=200 sinusoids): asserts `gak > 1e-10`, `≤ 1`, finite. A raw-product recursion would return exactly 0.0 here. DP is `loggak` (log-sum-exp forward pass); no linear-space accumulation exists. |
| 2 | Normalized similarity in `[0,1]`, `k(x,x)==1.0` (±1e-12), no entry <0 or >1, NaN/Inf-free even for dissimilar curves | **PASS** | `test_gak_normalized_range` (incl. a wildly-different-scale curve): asserts diagonal==1 (±1e-12), every entry finite and in `[0,1]`. `normalize_log` maps `-inf` numerator → `0.0` (NaN guard). |
| 3 | n×n Gram symmetric BY ASSIGNMENT (bit-exact `G[i][j]==G[j][i]`) and PSD (min eigenvalue ≥ −1e-8), parallel via `iter_maybe_parallel!` | **PASS** | `test_gak_gram_symmetric` asserts `to_bits()` equality (bit-exact mirror). `test_gak_gram_psd` computes `symmetric_eigenvalues()` on the `to_dmatrix()` Gram, asserts min ≥ −1e-8. `test_gak_parallel_matches_sequential` asserts bit-identical recomputation (order-independent, no RNG). Upper triangle + diagonal built with `iter_maybe_parallel!`. |
| 4 | Auto-σ median heuristic (`sigma_gak`), and with that σ off-diagonal Gram entries land in a healthy (≈0.05–0.95) range, not near-identity/near-constant | **PASS** | `test_sigma_gak_healthy`: `GakConfig::default()` (σ=None → heuristic); asserts `max off-diag < 0.999`, `min off-diag > 1e-4`, span > 0.05. `test_sigma_gak_floor_on_identical` proves the positive floor prevents σ=0 on degenerate data. |
| 5 | GAK matches a reference within 1e-6 on a small hand-checked dataset | **PASS** (with documented divergence) | `test_gak_vs_reference`: tslearn is not installed here, so the reference is **hand-derived analytically** from the Cuturi TGAK formula (DP written out in the comment), NOT fabricated tslearn numbers. `gak([0,1],[0,2],σ=1)=0.44843221961236995` and `gak([0,1,2],[0,1,3],σ=2)=0.805752775914924`, both asserted within 1e-9 (tighter than the 1e-6 bar). Divergence documented in SUMMARY. |

## Gate evidence

- `cargo fmt --check`: clean (exit 0).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`: clean, 0 warnings.
- `cargo test -p fdars-core --features linalg gak`: 11 passed, 0 failed; doctests 2 passed, 0 failed.
- `cargo test -p fdars-core gak` (default/parallel): 11 passed, 0 failed.

## Scope discipline

Nothing from Phase 55 (Gram train/predict export) or Phase 56 (kernel-k-means) was
implemented. The `pub(crate) loggak` seam is left in place for Phase 55's
split-normalization export. No crate version bump. No new dependency.

## Gaps

None.
