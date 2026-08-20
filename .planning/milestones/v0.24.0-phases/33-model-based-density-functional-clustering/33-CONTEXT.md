# Phase 33: Model-Based & Density Functional Clustering - Context

**Gathered:** 2026-08-20
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous)

<domain>
## Phase Boundary

Add functional clustering paradigms beyond the existing k-means/GMM/hierarchical/k-medoids
(CLUS-01), delivering five clusterers as numeric cluster assignments + model outputs only
(no plotting/rendering):

- **funHDDC** — per-group subspace covariance model (extends `gmm/`).
- **funFEM** — discriminative-subspace clustering variant.
- **DBSCAN** — density clusterer over functional distances (reuses `distance.rs`).
- **kCFC** — subspace-embedding (per-cluster FPCA reassignment) loop.
- **joint align-and-cluster** — reuses `alignment/` for shape-invariant grouping.

Strictly additive/non-breaking: no existing clustering / `gmm/` / `distance.rs` / `alignment/`
public signature changes; `Result<T, FdarError>`; inline `#[cfg(test)]` tests; crate-root
re-exports.

</domain>

<decisions>
## Implementation Decisions

### File Placement
- New clustering submodule file(s) for the density/model-based clusterers (e.g.
  `clustering/model_based.rs` + `clustering/density.rs`, or a new `clustering_advanced.rs` —
  planner's discretion on exact file names). funHDDC extends the `gmm/` module (per-group
  subspace covariance builds on the existing GMM EM/covariance machinery).
- Existing `clustering.rs` (`kmeans_fd`, `fuzzy_cmeans_fd`, silhouette/CH metrics) stays
  untouched.

### funHDDC Model
- Simplified per-group subspace covariance: each group has an intrinsic-dimension `d_k`
  subspace (leading eigenvectors) plus an isotropic residual-noise variance on the
  complement — a single representative model, NOT the full funHDDC akjbkqkdk 6-model family.
- Document the divergence from the R `funHDDC` 6-model family in rustdoc.

### DBSCAN
- Neighborhoods computed from `distance.rs::l2_distance_matrix` (functional L2 distance).
- Configurable `eps` and `min_points`; noise curves get an unassigned/noise label
  (e.g. a sentinel cluster id or `Option`-style), distinct from real clusters.

### Correctness Tests
- Recovery up to label permutation on synthetic well-separated functional groups, measured by
  adjusted Rand index / accuracy against a documented threshold.
- The align-and-cluster path tested on data including a shape-shifted group.
- DBSCAN correctly flags injected noise curves as unassigned.

### Claude's Discretion
- Exact new-file names, config/result struct field names, default eps/min_points/d_k,
  kCFC iteration caps, internal helper factoring, and test counts are at Claude's discretion.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdars-core/src/gmm/` — `gmm_em`, `GmmResult`, `CovType`, covariance machinery
  (`covariance.rs`), init (`init.rs`); funHDDC per-group subspace covariance extends this.
- `fdars-core/src/distance.rs` — `l2_distance_matrix`, `euclidean_distance_matrix`,
  `pairwise_distance_matrix`, `cross_distance_matrix` (DBSCAN neighborhoods + generic distances).
- `fdars-core/src/regression.rs::fdata_to_pc_1d` — FPCA for funFEM discriminative subspace and
  kCFC per-cluster embeddings.
- `fdars-core/src/alignment/` — `karcher_mean`, `elastic_align_pair`, `elastic_distance`,
  `amplitude_distance` for the joint align-and-cluster path.
- `fdars-core/src/clustering.rs` — `silhouette_score(_from_distances)`, `calinski_harabasz`
  metrics reusable for cluster diagnostics; existing k-means as an assignment-loop analog.

### Established Patterns
- Column-major `FdMatrix`; `Result<T, FdarError>`; config-struct + result-struct pairing;
  `#[must_use]`, `#[non_exhaustive]`, serde-feature gating; per-thread RNG seeding for any
  randomized init (k-means++-style seeding pattern).

### Integration Points
- New clusterer fns + result structs in new clustering submodule file(s); funHDDC in `gmm/`;
  `pub use` in the module barrels; crate-root re-exports in `src/lib.rs`.

</code_context>

<specifics>
## Specific Ideas

- Reuse `gmm_em`/covariance for funHDDC and `l2_distance_matrix` for DBSCAN rather than adding
  new numeric primitives; **no new crate dependency** (milestone constraint).
- DBSCAN noise label must be distinguishable from cluster ids in the result type.
- Use adjusted Rand index (implement a small helper if not present) as the agreement metric in tests.

</specifics>

<deferred>
## Deferred Ideas

- Plotting/rendering of cluster assignments (out of scope — numeric outputs only).
- Functional co-clustering (funLBM / CLUS-02) — explicitly deferred to a future milestone.
- Full funHDDC akjbkqkdk 6-model family (simplified single model this phase).

</deferred>
