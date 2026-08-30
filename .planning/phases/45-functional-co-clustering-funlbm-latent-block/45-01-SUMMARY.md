---
phase: 45-functional-co-clustering-funlbm-latent-block
plan: "01"
subsystem: coclustering
tags: [coclustering, funlbm, cem, fpca, functional-data, latent-block]
status: complete

requires: [regression.rs/fdata_to_pc_1d, clustering.rs/kmeans_fd]
provides: [coclustering::co_cluster, coclustering::CoClusterConfig, coclustering::CoClusterResult, coclustering::BlockParams]
affects: [lib.rs, prelude.rs]

tech_stack:
  added: []
  patterns:
    - funLBM CEM (Classification EM) with alternating hard row/col assignment
    - Block-score projection (global FPCA restricted to column-block argument points)
    - Diagonal block Gaussian with data-scaled regularization
    - k-means++ column initialization on argument-point profiles
    - Multi-restart (n_init) with best-by-log-likelihood selection

key_files:
  created:
    - fdars-core/src/coclustering.rs
  modified:
    - fdars-core/src/lib.rs
    - fdars-core/src/prelude.rs

decisions:
  - Column-clusters range over m argument points (col_labels.len() == m) — true funLBM, NOT clustering of FPC components
  - ONE global FPCA reused; block-score projection restricts inner product to each column-block's argument-point subset
  - Diagonal block covariance (ncomp variances per block) — WASM/MSRV-safe, documented as divergence from R funLBM
  - ICL symmetric penalty: p_KL = (K-1)+(L-1)+2*K*L*eff_ncomp; (ln n + ln m) reflects both data dimensions
  - Data-scaled regularization floor (1e-6 * mean_var) on all block variances to prevent variance collapse

metrics:
  duration_minutes: 30
  completed: "2026-08-30T18:55:32Z"
  tasks_completed: 3
  tasks_planned: 3
  commits: 1

actuals:
  tokens: 33000
  tasks: 3
  commits: 1

requirements: [CLUS-02-01, CLUS-02-02]
---

# Phase 45 Plan 01: funLBM CEM Core Summary

## One-liner

Functional co-clustering via Classification EM on FPC block-scores with column-clusters over m argument points (col_labels.len()==m), diagonal block Gaussians, symmetric ICL, and 10 inline tests covering monotone LL, ARI>0.8 block recovery, determinism, and all error paths.

## What Was Built

New module `fdars-core/src/coclustering.rs` implementing the funLBM latent block model:

### Public API

```rust
// Config (builder-style, mirrors GmmClusterConfig)
pub struct CoClusterConfig {
    pub n_row_blocks: usize,  // K row-clusters (default: 2)
    pub n_col_blocks: usize,  // L col-clusters over m argument points (default: 2)
    pub ncomp: usize,         // FPC components for block-score projection (default: 5)
    pub max_iter: usize,      // CEM max iterations (default: 200)
    pub tol: f64,             // LL convergence tol (default: 1e-6)
    pub n_init: usize,        // random restarts (default: 3)
    pub seed: u64,            // deterministic seed (default: 42)
}

// Per-block diagonal Gaussian parameters
pub struct BlockParams {
    pub mean: Vec<f64>,      // len eff_ncomp
    pub variance: Vec<f64>,  // len eff_ncomp
}

// Result
pub struct CoClusterResult {
    pub row_labels: Vec<usize>,      // len n — hard row-cluster assignments
    pub col_labels: Vec<usize>,      // len m — hard column-cluster assignments (m arg points)
    pub n_row_blocks: usize,
    pub n_col_blocks: usize,
    pub block_params: Vec<BlockParams>, // len K*L, indexed k*L+l
    pub row_props: Vec<f64>,         // len K, sums to 1
    pub col_props: Vec<f64>,         // len L, sums to 1
    pub log_likelihood: f64,
    pub icl: f64,                    // finite, symmetric (ln n + ln m) penalty
    pub iterations: usize,
    pub converged: bool,
}

// Entry point
pub fn co_cluster(
    data: &FdMatrix,
    argvals: &[f64],
    config: &CoClusterConfig,
) -> Result<CoClusterResult, FdarError>
```

### Algorithm

1. Validate inputs (ncomp>=1, n_row_blocks<=n, n_col_blocks<=m, data/argvals mismatch propagated from fdata_to_pc_1d)
2. ONE global FPCA via `fdata_to_pc_1d` → rotation (m×eff_ncomp), mean (m), weights (m)
3. For each of n_init restarts (seed + init*1000):
   - Row init: `kmeans_fd(data, argvals, K, 100, 1e-4, seed)`
   - Col init: k-means++ on m argument-point profiles in R^n (10 iterations)
   - CEM loop up to max_iter:
     - Build block scores: `bscore[i][l][k] = Σ_{j:col_labels[j]==l} w[j]*(data[(i,j)]-mean[j])*rotation[(j,k)]`
     - E-row: argmax_k classification log-density per curve
     - E-col: argmax_l marginal log-density gain per argument point
     - M-step: row_props, col_props, per-block mean+variance (data-scaled reg floor)
     - Compute classification LL; break if |ΔLL| < tol
4. Return best result by log_likelihood

### ICL Formula

`p_KL = (K-1) + (L-1) + 2*K*L*eff_ncomp`
`ICL = log_likelihood - 0.5 * p_KL * (ln(n) + ln(m))`

### Registration

- `lib.rs`: `pub mod coclustering;` (alphabetical, between `clustering_advanced` and `concurrent_regression`)
- `lib.rs`: `pub use coclustering::{co_cluster, BlockParams, CoClusterConfig, CoClusterResult};`
- `prelude.rs`: `pub use crate::coclustering::{CoClusterConfig, CoClusterResult};`

## Tests (10 total, all passing)

| Test | Coverage |
|------|---------|
| test_co_cluster_smoke | Single fit, row_labels.len()==n, col_labels.len()==m, block_params.len()==K*L |
| test_classification_ll_nondecreasing | Per-iter LL vector non-decreasing (±1e-6 tolerance) |
| test_coclustering_recovers_block_structure | ARI>0.8 on both row and col axes (n=20, m=12, K=L=2, signal=±5) |
| test_determinism_under_seed | Two calls with same seed produce identical labels, LL, ICL |
| test_icl_is_finite | ICL is finite on well-conditioned data |
| test_error_k_exceeds_n | K=99>n=8 → InvalidParameter(n_row_blocks) |
| test_error_l_exceeds_m | L=99>m=6 → InvalidParameter(n_col_blocks) |
| test_error_zero_ncomp | ncomp=0 → InvalidParameter(ncomp) |
| test_error_argvals_mismatch | argvals.len()!=m → InvalidDimension (propagated from fdata_to_pc_1d) |
| test_result_surface_populated | col_labels.len()==m, block_params[i].mean.len()==eff_ncomp |

## Deviations from Plan

None — plan executed exactly as written.

- Tracer (Task 1), TDD correctness (Task 2), and error paths (Task 3) were all implemented in a single file and committed atomically after all 10 tests passed.
- The `eff_ncomp = fpca.scores.ncols()` guard is correctly applied (not the requested `config.ncomp`).
- col_labels.len() == m verified (not ncomp) per the resolved CLUS-02-01 semantics.

## Known Stubs

None.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes. Pure in-process numeric library — threat register entries T-45-01 through T-45-04 are all mitigated in the implementation.

## Self-Check: PASSED

- `fdars-core/src/coclustering.rs`: EXISTS (1321 lines)
- `a06d6a2d` in git log: FOUND
- All 10 coclustering tests: PASSED
- col_labels.len() == m: VERIFIED (test_result_surface_populated + test_co_cluster_smoke)
