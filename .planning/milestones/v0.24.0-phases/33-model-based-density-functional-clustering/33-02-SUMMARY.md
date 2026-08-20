---
phase: 33-model-based-density-functional-clustering
plan: "02"
subsystem: clustering_advanced
tags: [clustering, dbscan, kcfc, functional-data, fpca]
dependency_graph:
  requires: ["33-00"]
  provides: ["clustering_advanced.rs", "dbscan_fd", "kcfc_cluster"]
  affects: ["lib.rs"]
tech_stack:
  added: []
  patterns:
    - "BFS graph-search DBSCAN over precomputed l2_distance_matrix"
    - "k-means++ init followed by per-cluster fdata_to_pc_1d reassignment loop (kCFC)"
    - "None-is-noise Vec<Option<usize>> cluster labeling"
    - "Reconstruction-error argmin assignment"
key_files:
  created:
    - fdars-core/src/clustering_advanced.rs
  modified:
    - fdars-core/src/lib.rs
decisions:
  - "DBSCAN uses Vec<Option<usize>> (None=noise) for type-safe noise labeling"
  - "kCFC uses k-means++ init seeded by config.seed for deterministic, good starting points"
  - "Empty-cluster fallback: keep prior FPCA model rather than reinitializing, avoiding divide-by-zero"
  - "Reconstruction error uses Simpson-weighted L2^2 (consistent with functional metric throughout crate)"
  - "BFS neighbor queuing uses Vec::contains check (O(n) per lookup, fine for small n; acceptable for library use)"
metrics:
  duration: "~5 minutes"
  completed: "2026-08-20"
  tasks_completed: 3
  commits: 3
status: complete
actuals:
  tokens: 18000
  tasks: 3
  commits: 3
requirements: [CLUS-01]
---

# Phase 33 Plan 02: DBSCAN + kCFC Clustering Summary

DBSCAN density clustering and kCFC per-cluster FPCA clustering added as new `clustering_advanced` module in fdars-core, re-exported at crate root. Purely additive; no existing signature changed.

## What Was Built

### Task 1: DBSCAN over functional L2 distances

- `DbscanConfig { eps: f64, min_points: usize }` with `Default` (eps=0.5, min_points=3)
- `DbscanResult { cluster: Vec<Option<usize>>, n_clusters, n_noise, distances: FdMatrix }`
- `dbscan_fd(data, argvals, config) -> Result<DbscanResult, FdarError>`
- Algorithm: precompute `l2_distance_matrix`, BFS-expand core points, assign border points, leave unreachable as `None` (noise)
- Inline validation: eps>0, min_points>=1, n>0, m>0, argvals.len()==m

### Task 2: kCFC per-cluster FPCA reassignment loop

- `KcfcConfig { k: usize, ncomp: usize, max_iter: usize, seed: u64 }` with `Default` (k=2, ncomp=3, max_iter=50, seed=42)
- `KcfcResult { cluster: Vec<usize>, fpca_models: Vec<Option<FpcaResult>>, reconstruction_errors: FdMatrix, iterations, converged }`
- `kcfc_cluster(data, argvals, config) -> Result<KcfcResult, FdarError>`
- Algorithm: k-means++ init → per-cluster `fdata_to_pc_1d` → L2^2 reconstruction error → argmin reassignment → repeat until convergence
- Empty-cluster fallback: keep prior FPCA model (avoids NaN/panic on degenerate partitions)
- Actual ncomp clamped via `fdata_to_pc_1d` internals; `rotation.ncols()` used as effective ncomp

### Task 3: Crate-root re-exports

- `pub mod clustering_advanced;` declared in `lib.rs` alongside existing `pub mod clustering;`
- Re-export block: `pub use clustering_advanced::{dbscan_fd, kcfc_cluster, DbscanConfig, DbscanResult, KcfcConfig, KcfcResult};`

## Test Results

All 16 inline tests pass:

| Test | Result |
|------|--------|
| test_dbscan_core_points | ok — 2 clusters, 0 noise on well-separated data |
| test_dbscan_noise_flagging | ok — 2 noise (None) on injected outlier curves |
| test_dbscan_zero_eps_returns_err | ok |
| test_dbscan_negative_eps_returns_err | ok |
| test_dbscan_invalid_min_points_zero | ok |
| test_dbscan_empty_data | ok |
| test_dbscan_mismatched_argvals | ok |
| test_dbscan_distances_shape | ok — n×n matrix |
| test_kcfc_recovery | ok — ARI >= 0.90 on well-separated clusters |
| test_kcfc_errors_ordering | ok — ≥80% of curves have smaller error for true cluster |
| test_kcfc_deterministic | ok — identical seed produces identical assignments |
| test_kcfc_invalid_k_zero | ok |
| test_kcfc_invalid_k_gt_n | ok |
| test_kcfc_empty_data | ok |
| test_kcfc_mismatched_argvals | ok |
| test_kcfc_result_shapes | ok — correct n and k dimensions |

## Deviations from Plan

None — plan executed exactly as written.

The two TDD tasks (Task 1, Task 2) were implemented together in a single file as their implementations are interdependent at the module level (shared imports, test helpers). Tests were verified green before each commit.

## Threat Mitigations Applied

Per plan `<threat_model>`:

| Threat ID | Status |
|-----------|--------|
| T-33-03 DBSCAN input validation | Mitigated — all 5 error paths tested |
| T-33-04 kCFC input validation + empty cluster | Mitigated — empty-cluster keeps prior model; all error paths tested |
| T-33-SC No new dependency | Met — zero new crate imports |

## Known Stubs

None — both algorithms are fully wired. kCFC reconstruction error matrix is live-computed each iteration; no placeholder values.

## Commits

| Hash | Description |
|------|-------------|
| 4e80d9d9 | feat(33-02): implement DBSCAN density clustering for functional data |
| 4db400ed | feat(33-02): add kCFC per-cluster FPCA clustering and declare clustering_advanced module |
| 16bfa183 | feat(33-02): re-export DBSCAN + kCFC at crate root; apply cargo fmt |

## Self-Check: PASSED

- `fdars-core/src/clustering_advanced.rs`: FOUND
- `fdars-core/src/lib.rs`: modified (module + re-exports)
- All 3 commits present in git log
- `cargo build -p fdars-core --features linalg,parallel --lib`: clean
- All 16 `clustering_advanced` tests green
