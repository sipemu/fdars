# Phase 45: Functional Co-Clustering (funLBM latent-block) - Context

**Gathered:** 2026-08-24
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — grey areas proposed in batch, all accepted as recommended

<domain>
## Phase Boundary

Deliver CLUS-02: a new module `fdars-core/src/coclustering.rs` implementing a functional latent
block model (funLBM). Block-wise-Gaussian EM on FPC scores that **simultaneously** assigns
curves to row-clusters and argument points to column-clusters given a target (row, column) block
count; a result exposing row labels, column labels, per-block parameters, and a converged
log-likelihood / model criterion (ICL); plus a slope-heuristic model-selection helper over a
range of candidate (row, column) block counts.

Additive/non-breaking only: `Result`-returning public fns, inline `#[cfg(test)]` tests,
crate-root + prelude re-exports, zero changes to existing public signatures. Numeric outputs
only (no plotting of co-cluster blocks). No new crate dependency. fdars' existing
`clustering.rs`/`gmm/` cluster **curves only** — this adds the row×column co-clustering
paradigm. R baseline: `funLBM` 2.3.1 + `funHDDC` (slope heuristic), matched by capability;
document divergences in rustdoc.

</domain>

<decisions>
## Implementation Decisions

### Block Model & FPC Representation
- Block model: funLBM block-wise Gaussian — each (row-block k, column-block l) block is modelled by a low-dimensional Gaussian on FPC scores.
- FPC scores source: global FPCA via the existing `fdata_to_pc_1d` on the full data; block Gaussians operate on subvectors of the global scores (reuse existing machinery, simpler than per-block FPCA).
- Column-block assignment: an arbitrary cluster label per argument point (L column-clusters); columns need NOT be contiguous.
- FPC components per block: a fixed small `d` taken from config (`ncomp`).

### EM Algorithm
- EM variant: variational / classification EM on the latent block model, alternating updates of row memberships and column memberships (deterministic — no stochastic SEM-Gibbs in v1).
- Initialization: k-means for row clusters (reuse `kmeans_fd`) and k-means on argument-point profiles for column clusters, seeded.
- Convergence: stop when the log-likelihood (or variational lower bound) change < tol, or at `max_iter`.
- Determinism: all randomness seeded via `StdRng::seed_from_u64(seed + offset)`; bit-reproducible given the same seed.

### Result & Model Criterion
- Result struct `CoClusterResult`: `row_labels`, `col_labels`, `n_row_blocks`, `n_col_blocks`, `block_params`, `log_likelihood`, `icl`.
- Model criterion: ICL (Integrated Completed Likelihood) for the latent block model (primary; BIC-style penalty is the fallback deferred).
- Per-block parameters: block mean + FPC-score variance/covariance + row/column mixing proportions.
- Labels: hard labels (argmax posterior) for both rows and columns.

### Model Selection (Slope Heuristic) & API
- Model selection: Birgé–Massart data-driven slope heuristic (dimension-jump / slope estimation) over a user-supplied grid of candidate (K, L) block counts.
- Candidate grid: user supplies a K range and an L range; each (K,L) is fit and the slope heuristic selects.
- Config struct `CoClusterConfig`: `n_row_blocks`, `n_col_blocks`, `ncomp`, `max_iter`, `tol`, `n_init`, `seed` (builder pattern like `GmmClusterConfig`).
- Module layout: single new `coclustering.rs` (factor into a folder only if it exceeds ~500 lines at implement time).

### Claude's Discretion
- Exact block-parameter parameterization (full vs diagonal FPC-score covariance), the precise ICL penalty formula, the slope-heuristic penalty-calibration details, internal helper decomposition, struct field naming, and plan/wave decomposition — at planner/implementer discretion within the accepted decisions and existing conventions.
- Whether the module warrants a folder split at implement time.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdata_to_pc_1d(data: &FdMatrix, ncomp, argvals) -> Result<FpcaResult>` (`src/regression.rs:287`); `FpcaResult { singular_values, rotation, scores, mean, centered, weights }` — global FPC scores for the block Gaussians.
- `kmeans_fd(...)` + `KmeansResult` (`src/clustering.rs:545`) — row-cluster initialization; `fuzzy_cmeans_fd`, `silhouette_score`, `calinski_harabasz` also available.
- `gmm/` (`GmmResult`, `GmmClusterResult`, `GmmClusterConfig`) — Gaussian-mixture EM patterns + config-struct builder style to mirror.
- `adjusted_rand_index(a, b)` + `uniform_grid(n)` (`src/test_helpers.rs`) — test oracles (recover known row/col labels within ARI tolerance).
- `cholesky_*` (`src/linalg.rs`, linalg feature) for block-covariance Gaussians if full covariances are used; column-major `FdMatrix`.

### Established Patterns
- RNG seeding `StdRng::seed_from_u64(seed + k as u64)` per component/replicate (deterministic).
- Feature-gated rayon via `iter_maybe_parallel!` for the (K,L)-grid model-selection sweep.
- All public fns return `Result<T, FdarError>`; `#[must_use]` + `#[derive(Debug, Clone, PartialEq)]` + `#[non_exhaustive]` on result structs; serde cfg_attr; inline `#[cfg(test)]` tests.

### Integration Points
- Register `pub mod coclustering;` in `src/lib.rs` + crate-root `pub use coclustering::{...}` re-exports + key types (`CoClusterResult`, `CoClusterConfig`) added to `src/prelude.rs`.

</code_context>

<specifics>
## Specific Ideas

- R baseline capability parity: `funLBM::funLBM` (block-wise functional Gaussian LBM, ICL, EM) + `funHDDC` slope heuristic. Match by capability, not exact R signatures; document divergences (e.g. global vs per-block FPCA, variational vs SEM-Gibbs EM, diagonal vs full block covariance) in rustdoc.
- Test oracles: (1) on synthetic data generated from a known (K,L) block structure, EM recovers the row and column labels up to permutation with high ARI; (2) log-likelihood is non-decreasing across EM iterations; (3) ICL is finite; (4) determinism — same seed → identical labels/log-lik/ICL; (5) slope heuristic selects the true (K,L) (or near it) on a well-separated synthetic dataset; (6) error paths (K/L larger than n/m, ncomp invalid, dimension mismatch) → FdarError.

</specifics>

<deferred>
## Deferred Ideas

- SEM-Gibbs / stochastic EM variant; per-block FPCA (global FPCA reused in v1); full (vs diagonal) block covariance if the planner chooses diagonal.
- BIC-only model selection (ICL is primary; slope heuristic is the required selector); exhaustive multi-restart consensus beyond `n_init`.
- Non-Gaussian block distributions; missing-data / irregular-grid co-clustering; soft (fuzzy) co-cluster memberships as the primary output (hard labels in v1).
- Plotting/rendering of co-cluster blocks (numeric outputs only).

</deferred>
