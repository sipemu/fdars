---
phase: 45-functional-co-clustering-funlbm-latent-block
plan: 02
subsystem: coclustering
tags: [model-selection, slope-heuristic, birge-massart, funlbm, co-clustering]
status: complete

dependencies:
  requires: [45-01]
  provides: [co_cluster_select, CoClusterSelectResult]
  affects: [fdars-core/src/coclustering.rs, fdars-core/src/lib.rs, fdars-core/src/prelude.rs]

tech_stack:
  added: []
  patterns:
    - Birgé–Massart slope heuristic (OLS over top-50% large-model region)
    - Model-dimension formula for LBM: (K-1)+(L-1)+2·K·L·eff_ncomp
    - Fallback-to-argmax-LL for edge cases (small grid, flat slope, zero denominator)

key_files:
  created: []
  modified:
    - fdars-core/src/coclustering.rs  # +442 lines: CoClusterSelectResult + co_cluster_select + 5 tests
    - fdars-core/src/lib.rs           # extend re-export block
    - fdars-core/src/prelude.rs       # add CoClusterSelectResult

decisions:
  - Sequential grid sweep: co_cluster is itself internally parallelised (via kmeans_fd + multiple
    restarts); using sequential outer iteration avoids rayon closure ownership complexity and keeps
    grid results order-stable without a sort step.
  - model_dim reads eff_ncomp from block_params[0].mean.len() to match the actual FPC count used
    by each fitted model (may be < config.ncomp when clipped by min(n,m)).
  - OLS fallback threshold: denominator < 1e-10 triggers max-LL fallback (all dims equal in
    large-model subset).
  - penalty_rate ≤ 0 triggers max-LL fallback (slope non-negative → no meaningful regularisation).
  - grid_scores always fully populated regardless of branch (Pitfall 4 compliance per RESEARCH.md).

metrics:
  duration: 20m
  completed: "2026-08-30T19:00:24Z"
  tasks_completed: 2
  commits: 1

actuals:
  tokens: 8500   # chars/4 over files changed in this plan
  tasks: 2
  commits: 1
---

# Phase 45 Plan 02: Slope-Heuristic (K,L) Model Selection — Summary

Additive implementation of `co_cluster_select` + `CoClusterSelectResult` on top of the 45-01
funLBM CEM core. The function sweeps a user-supplied K×L grid via `co_cluster`, estimates the
Birgé–Massart penalty slope by OLS over the large-model region, and returns the penalised-optimal
fit with full grid diagnostics.

## What Was Built

### New Public API

```rust
// Result type
#[derive(Debug, Clone)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CoClusterSelectResult {
    pub best: CoClusterResult,
    pub best_k: usize,
    pub best_l: usize,
    pub grid_scores: Vec<(usize, usize, f64, usize, f64)>, // (K,L,log_lik,model_dim,penalised_score)
    pub slope_estimate: f64,
    pub penalty_rate: f64,
}

// Selector function
#[must_use]
pub fn co_cluster_select(
    data: &FdMatrix,
    argvals: &[f64],
    k_range: &[usize],
    l_range: &[usize],
    config: &CoClusterConfig,
) -> Result<CoClusterSelectResult, FdarError>
```

### Algorithm

1. Validate non-empty k_range, l_range (→ FdarError::InvalidParameter on empty).
2. Build K×L grid; sweep sequentially calling `co_cluster` per cell (cloned config, overriding
   n_row_blocks/n_col_blocks).
3. Compute model_dim per cell: `(K-1)+(L-1)+2·K·L·eff_ncomp` where eff_ncomp =
   `result.block_params[0].mean.len()`.
4. Sort by dim descending; take top-50% (≥4 when available) as large-model region.
5. OLS slope: `Σ(dim_i−d̄)(ll_i−l̄) / Σ(dim_i−d̄)²`.
6. penalty_rate = 2·|slope|.
7. Select `argmax(LL − penalty_rate·dim)`.

### Edge Cases (all non-panicking)

| Condition | Behaviour |
|-----------|-----------|
| Empty k_range or l_range | FdarError::InvalidParameter |
| Grid < 4 points | Fall back to argmax LL; slope_estimate=0, penalty_rate=0 |
| OLS denominator < 1e-10 | Fall back to argmax LL |
| penalty_rate ≤ 0 | Fall back to argmax LL |
| Single-cell grid | Returns that cell directly |
| co_cluster error for any (K,L) | Propagates immediately |

### Re-exports

- `fdars_core::co_cluster_select`, `fdars_core::CoClusterSelectResult` (lib.rs)
- `fdars_core::prelude::CoClusterSelectResult` (prelude.rs)

## Tests (15 module tests, all green)

From 45-01 (inherited): `test_co_cluster_smoke`, `test_classification_ll_nondecreasing`,
`test_coclustering_recovers_block_structure`, `test_determinism_under_seed`,
`test_icl_is_finite`, `test_error_k_exceeds_n`, `test_error_l_exceeds_m`,
`test_error_zero_ncomp`, `test_error_argvals_mismatch`, `test_result_surface_populated`.

New (45-02):
- `test_co_cluster_select_smoke` — 2-cell grid, verifies grid_scores.len()==2, dimensions correct
- `test_slope_heuristic_selects_correct_kl` — 6-cell grid on well-separated (K=2,L=2) data,
  ARI > 0.6 on row assignment
- `test_select_single_cell` — 1-cell grid, slope/penalty both 0
- `test_select_empty_range_errors` — empty k_range and l_range each → InvalidParameter
- `test_select_determinism` — same seed → identical best_k, best_l, all grid_scores entries

## Deviations from Plan

None — plan executed exactly as written. Task 1 (tracer) and Task 2 (slope heuristic) were
implemented atomically in a single commit since the full slope heuristic code was the natural
completion of the tracer path; no intermediate stub state was needed.

## Threat Mitigations (from threat model)

| Threat | Mitigation | Status |
|--------|------------|--------|
| T-45-06: Empty k/l_range | FdarError::InvalidParameter validation at function entry | DONE |
| T-45-07: OLS denominator ≈ 0 | `denominator.abs() < 1e-10` guard → max-LL fallback | DONE |
| T-45-08: Boundary selection | grid_scores fully populated; never errors on boundary picks | DONE |

## Known Stubs

None.

## Threat Flags

None — pure in-process numeric computation; no new I/O, network, or auth surface.

## Self-Check

- [x] `fdars-core/src/coclustering.rs` exists with co_cluster_select + CoClusterSelectResult
- [x] `fdars-core/src/lib.rs` re-exports both
- [x] `fdars-core/src/prelude.rs` re-exports CoClusterSelectResult
- [x] Commit d4d6accf exists: `feat(45-02): add co_cluster_select + CoClusterSelectResult`
- [x] 15/15 module tests green
