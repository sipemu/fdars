---
phase: 45
slug: functional-co-clustering-funlbm-latent-block
status: passed
verified: 2026-08-30
verifier: orchestrator-inline
reason: >
  Verification performed inline against objective, reproducible evidence
  (crate-wide clippy --all-targets clean, cargo fmt clean, full test suite 2583
  lib unit tests + all integration suites + doctests passing with 0 failures
  after fixing two non_exhaustive-struct-literal doctests the module-scoped
  executor runs did not exercise). Independent gsd-verifier subagent dispatch was
  unreliable this session (transient API 529 / connection drops); the objective
  gate results are authoritative.
requirements_verified: [CLUS-02-01, CLUS-02-02, CLUS-02-03]
---

# Phase 45 — Verification (Functional Co-Clustering, funLBM latent-block)

**Goal:** A user can co-cluster functional data — simultaneously grouping curves into
row-clusters and argument points into column-clusters via a functional latent block model
(funLBM) — and select the number of blocks automatically, a paradigm absent from fdars'
curve-only clustering.

**Verdict: PASSED** — all three requirements are delivered as `Result`-returning public
functions in the new additive module `coclustering.rs`, re-exported at the crate root and
prelude, with inline recovery + error-path tests, and the whole crate passes clippy
`--all-targets`, `cargo fmt --check`, and the full test suite.

## Objective quality gates

| Gate | Command | Result |
|------|---------|--------|
| Lint (incl. test/bench code) | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | ✅ clean |
| Format | `cargo fmt -p fdars-core --check` | ✅ clean |
| Full test suite | `cargo test -p fdars-core --features linalg,parallel` | ✅ 2583 lib + 12/55/50/107/77/1/174/56/16/34 integration + doctests, **0 failures** (2 doctests fixed: `#[non_exhaustive]` `CoClusterConfig` needs `::default()` + field mutation from external code) |

## Per-requirement verdicts (goal-backward)

| Req | Must-have | Delivered symbol | Evidence | Verdict |
|-----|-----------|------------------|----------|---------|
| CLUS-02-01 | Fit funLBM — simultaneously assign curves to row-clusters and **argument points** to column-clusters via block-wise-Gaussian EM on FPC scores, given target (K,L) | `co_cluster` + `CoClusterConfig` (`coclustering.rs`) | tests: recovers a known (K,L) block structure via `adjusted_rand_index`, classification log-likelihood non-decreasing, `col_labels.len()==m` (argument points, not ncomp), determinism under seed | ✅ passed |
| CLUS-02-02 | Retrieve result — row labels, column labels, per-block parameters, converged log-likelihood / ICL | `CoClusterResult { row_labels(n), col_labels(m), block_params, row_props, col_props, log_likelihood, icl, iterations, converged }` + `BlockParams` | tests: result-surface populated; ICL finite (`ICL = ℓ_c − 0.5·p_KL·(ln n + ln m)`, `p_KL=(K−1)+(L−1)+2·K·L·ncomp`) | ✅ passed |
| CLUS-02-03 | Select number of blocks via slope-heuristic over a candidate (K,L) grid | `co_cluster_select` + `CoClusterSelectResult` (`coclustering.rs`) | tests: Birgé–Massart slope heuristic (OLS slope on top-half by model-dimension, argmax(LL − 2·|slope|·dim)) selects the true (K,L) on well-separated data (ARI); grid diagnostics populated; determinism under seed | ✅ passed |

## Resolved decision honored

- **Column-clusters range over the m argument points** (`col_labels.len()==m`), not the ncomp FPC components — the resolved override of the research's initial assumption, per CLUS-02-01. Block scores are the projection of each column-block's restricted raw values onto the global FPC loadings restricted to those points (one global FPCA reused). Verified by `test_error_l_exceeds_m` and `test_result_surface_populated`.

## Additive / non-breaking check

- New module `fdars-core/src/coclustering.rs` only.
- Crate-root re-exports (`src/lib.rs:470`): `co_cluster, co_cluster_select, BlockParams, CoClusterConfig, CoClusterResult, CoClusterSelectResult`.
- Prelude (`src/prelude.rs:86`): `CoClusterConfig, CoClusterResult, CoClusterSelectResult`.
- **Zero changes to existing public signatures** — existing `clustering.rs` / `gmm/` (curve-only clustering) untouched.
- **No new crate dependencies** — reuses `fdata_to_pc_1d`, `kmeans_fd`, `iter_maybe_parallel!`, seeded `StdRng`; diagonal block covariance avoids Cholesky (WASM/MSRV-safe).

## Conventions

Column-major `FdMatrix`; all public fns return `Result<T, FdarError>` (no panics — K>n, L>m,
ncomp<1, dimension mismatch → `FdarError`); `#[must_use]` + `#[derive(Debug, Clone, PartialEq)]`
(where applicable) + `#[non_exhaustive]` + serde cfg_attr on result/config structs; inline
`#[cfg(test)]` tests; seeded `StdRng` determinism; documented divergences from funLBM/funHDDC
(global vs per-block FPCA, CEM vs SEM-Gibbs, diagonal block covariance) in rustdoc.

## Notes / tech debt

- funLBM uses deterministic CEM (hard assignment) rather than SEM-Gibbs — a documented v1
  scope decision.
- Diagonal (not full) block FPC-score covariance — documented divergence.
- `45-VALIDATION.md` remains `status: draft` (Nyquist per-task map seeded pre-plan; consistent
  with prior milestones' deferred Nyquist TODO).
