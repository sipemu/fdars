# Phase 33: Model-Based & Density Functional Clustering — Research

**Researched:** 2026-08-20
**Domain:** Functional clustering — subspace-covariance EM, discriminative-subspace clustering, density-based functional DBSCAN, per-cluster FPCA loop, elastic align-and-cluster
**Confidence:** HIGH (all claims verified against source files read this session)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- New clustering submodule file(s) for the density/model-based clusterers (e.g.
  `clustering/model_based.rs` + `clustering/density.rs`, or a new `clustering_advanced.rs` —
  planner's discretion on exact file names). funHDDC extends the `gmm/` module (per-group
  subspace covariance builds on the existing GMM EM/covariance machinery).
- Existing `clustering.rs` (`kmeans_fd`, `fuzzy_cmeans_fd`, silhouette/CH metrics) stays
  untouched.
- Simplified per-group subspace covariance: each group has an intrinsic-dimension `d_k`
  subspace (leading eigenvectors) plus an isotropic residual-noise variance on the
  complement — a single representative model, NOT the full funHDDC akjbkqkdk 6-model family.
  Document the divergence from the R `funHDDC` 6-model family in rustdoc.
- Neighborhoods computed from `distance.rs::l2_distance_matrix` (functional L2 distance).
  Configurable `eps` and `min_points`; noise curves get an unassigned/noise label
  (e.g. a sentinel cluster id or `Option`-style), distinct from real clusters.
- Recovery up to label permutation on synthetic well-separated functional groups, measured by
  adjusted Rand index / accuracy against a documented threshold.
- The align-and-cluster path tested on data including a shape-shifted group.
- DBSCAN correctly flags injected noise curves as unassigned.

### Claude's Discretion
- Exact new-file names, config/result struct field names, default eps/min_points/d_k,
  kCFC iteration caps, internal helper factoring, and test counts are at Claude's discretion.

### Deferred Ideas (OUT OF SCOPE)
- Plotting/rendering of cluster assignments (out of scope — numeric outputs only).
- Functional co-clustering (funLBM / CLUS-02) — explicitly deferred to a future milestone.
- Full funHDDC akjbkqkdk 6-model family (simplified single model this phase).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CLUS-01 | Add functional clustering paradigms beyond the existing k-means/GMM/hierarchical/k-medoids — funHDDC per-group subspace covariance model (extending `gmm/`), funFEM discriminative-subspace clustering variant, DBSCAN density clusterer over functional distances (reusing `distance.rs`), kCFC subspace-embedding loop, and joint align-and-cluster estimator (reusing `alignment/`). Numeric cluster assignments + model outputs only. Additive/non-breaking. | All five algorithms fully formulated below. Reuse map identifies exact existing functions for each. API surface + result struct shapes mirror existing conventions. |
</phase_requirements>

---

## Summary

Phase 33 adds five functional clustering paradigms to fdars-core in a strictly additive, non-breaking way, placing new code in new files within `gmm/` (for funHDDC) and a new module (for funFEM, DBSCAN, kCFC, align-and-cluster). All five algorithms reuse existing fdars-core infrastructure — no new crate dependency is needed. The existing `clustering.rs` is left completely untouched.

The core challenge is correctly formulating each algorithm's EM / assignment loop using available codebase primitives. funHDDC builds directly on `gmm/em.rs`'s E-step and M-step machinery but replaces the covariance update with a per-group subspace-plus-isotropic-noise model. funFEM adds a Fisher-EM discriminative-subspace projection step before the GMM E-step. DBSCAN over precomputed `l2_distance_matrix` output is a straightforward graph-search. kCFC iterates per-cluster `fdata_to_pc_1d` projections with k-means++ reassignment. The joint align-and-cluster alternates `karcher_mean`-based template updates with elastic-distance reassignment.

**Primary recommendation:** Place funHDDC in `fdars-core/src/gmm/subspace.rs` (new file, `pub use` in `gmm/mod.rs`), and all other four in `fdars-core/src/clustering/advanced.rs` (converting `clustering.rs` into a submodule `clustering/mod.rs` + `clustering/advanced.rs`). The planner may keep `clustering.rs` as-is and add `clustering_advanced.rs` as a sibling — both approaches are valid.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| funHDDC per-group subspace EM | `gmm/` module | linalg helpers | Logically extends existing GMM EM machinery; per-group covariance lives alongside existing CovType variants |
| funFEM discriminative-subspace | new clustering file | `gmm/em.rs`, `regression.rs` | Discriminative step on FPC scores; reuses existing E/M-step helpers |
| DBSCAN over L2 distances | new clustering file | `distance.rs` | Pure distance-based; no EM component |
| kCFC subspace-embedding loop | new clustering file | `regression.rs`, `clustering.rs` helpers | Per-cluster FPCA → reassignment cycle |
| Joint align-and-cluster | new clustering file | `alignment/` | Template update via `karcher_mean`; elastic distance for reassignment |
| Validation metrics (ARI) | test-only helper | existing silhouette/CH | ARI not present in repo; implement as `#[cfg(test)]` helper |

---

## Standard Stack

### Core (all already in Cargo.toml — zero new dependencies)

| Library | Version | Purpose in Phase 33 |
|---------|---------|----------------------|
| nalgebra | 0.33 | SVD for per-group FPCA inside funHDDC/funFEM/kCFC via `fdata_to_pc_1d` |
| rand 0.8 / StdRng | 0.8 | Per-thread RNG seeding pattern for randomized inits |
| rayon 1.10 | 1.10 | `iter_maybe_parallel!` for E-step parallelism (already gated) |

### Reused Internal Modules

| Module | Functions Consumed | Purpose |
|--------|--------------------|---------|
| `gmm/em.rs` | `e_step`, `hard_assignments`, `resp_to_membership`, `compute_bic`, `compute_icl`, `finalize_gmm` | funHDDC E-step + model selection |
| `gmm/init.rs` | `kmeans_init_assignments`, `init_params_from_assignments` | funHDDC initialization |
| `gmm/covariance.rs` | `data_scaled_reg`, `regularize_cov`, `identity_cov` | funHDDC covariance regularization |
| `distance.rs` | `l2_distance_matrix` | DBSCAN precomputed distance matrix |
| `regression.rs` | `fdata_to_pc_1d`, `FpcaResult` | funFEM/kCFC subspace projections |
| `alignment/mod.rs` | `karcher_mean`, `elastic_distance`, `amplitude_distance` | Joint align-and-cluster template + reassignment |
| `clustering.rs` | `silhouette_score_from_distances`, `calinski_harabasz_from_distances` | Cluster quality diagnostics |
| `linalg.rs` | `cholesky_d`, `mahalanobis_sq`, `log_det_from_cholesky` | funHDDC log-density computation |
| `helpers.rs` | `simpsons_weights`, `l2_distance` | Integration weights, distances |
| `error.rs` | `FdarError` variants | All public function errors |

**No new packages are installed — this section intentionally has no Package Legitimacy Audit.**

---

## Architecture Patterns

### System Architecture Diagram

```
User calls funhddC_cluster / funfem_cluster / dbscan_fd / kcfc_cluster / align_cluster
          │
          ▼
    Input validation (n, m, k, eps, argvals)
          │
    ┌─────┴──────────────────────────────────────────────────┐
    │                                                          │
 funHDDC path                                    other 4 paths
 (gmm/subspace.rs)                          (clustering_advanced.rs)
    │                                                          │
 build features via                         ┌─────────────────┤
 fdata_to_pc_1d (global)                    │                 │
    │                                   funFEM            DBSCAN
 per-group subspace EM loop:            fdata_to_pc_1d       │
   E-step (gmm/em.rs::e_step)         → FPC scores     l2_distance_matrix
   M-step subspace update:            → Fisher-EM step  → BFS/DFS expand
     per-group SVD → d_k eigenvectors   → GMM E/M-step     clusters
     + isotropic sigma_k on complement       │           → noise label
    │                                   FunFemResult    DbscanResult
 FunHddcResult                                │
 (cluster, subspaces, sigma)         ┌────────┘
                                     │
                                  kCFC            align-and-cluster
                                 per-cluster      karcher_mean per cluster
                                 fdata_to_pc_1d  → elastic_distance
                                 → reassign       → reassign
                                 (iterate)        (iterate)
                                 KcfcResult      AlignClusterResult
          │
    Crate-root re-export in src/lib.rs
```

### Recommended Project Structure

```
fdars-core/src/
├── gmm/
│   ├── mod.rs          # add: pub mod subspace; pub use subspace::{...}
│   ├── subspace.rs     # NEW: funHDDC per-group subspace EM
│   ├── cluster.rs      # unchanged
│   ├── covariance.rs   # unchanged
│   ├── em.rs           # unchanged
│   └── init.rs         # unchanged
├── clustering.rs       # unchanged (or promoted to clustering/mod.rs)
├── clustering_advanced.rs  # NEW: funFEM, DBSCAN, kCFC, align-and-cluster
└── lib.rs              # add pub use for new types
```

Alternatively (if planner prefers a submodule):

```
fdars-core/src/
├── clustering/
│   ├── mod.rs          # re-exports from both files (all existing + new)
│   ├── kmeans.rs       # renamed from clustering.rs content
│   └── advanced.rs     # NEW: funFEM, DBSCAN, kCFC, align-and-cluster
```

**Recommendation:** Keep `clustering.rs` as-is (sibling file approach) and add `clustering_advanced.rs`. This avoids any module-restructuring risk and is purely additive.

---

## Algorithm Formulations

### 1. funHDDC — Per-Group Subspace Covariance EM

**Reference:** Bouveyron & Brunet (2012), "Model-based clustering of high-dimensional data" — simplified to the single model `[a_k b_k]` (intrinsic variance + isotropic noise), NOT the full 6-model `akjbkqkdk` family.

**Intentional divergence from R `funHDDC`:** The R package offers 6 covariance models. This implementation uses the simplest (`intrinsic-dim subspace + isotropic residual`), equivalent to the `AkjBkQkDk` model with the constraint `a_{k,1} = ... = a_{k,d_k}` and `b_k = b` scalar. This divergence is documented in rustdoc.

**Math (simplified `AkBk` model):**

For group k with n_k members and leading eigenvectors U_k (m × d_k):
- Covariance: Σ_k = U_k diag(a_k) U_k^T + b_k (I − U_k U_k^T)
  where a_k ∈ ℝ^{d_k} = within-subspace variances, b_k ∈ ℝ = noise variance
- Density: Gaussian log-density in score space d_k + log-likelihood contribution from complement using b_k
- M-step:
  - Project weighted data to subspace: Z_k = centered_data × U_k (n × d_k)
  - a_k = weighted variance of Z_k columns (n_k effective weight)
  - b_k = (total weighted variance − sum(a_k)) / (m − d_k)
  - U_k = leading d_k eigenvectors of weighted group covariance (via nalgebra SVD)
- Subspace update uses per-group SVD on the m×m weighted covariance matrix of group k.
  Since m (evaluation points) may be large, prefer the thin SVD of the n_k × m data slice.

**Implementation strategy using existing code:**

- Global initialization: run `kmeans_init_assignments` (from `gmm/init.rs`) on FPC-projected features
- E-step: implement a custom `log_component_density_subspace` using the subspace-plus-noise Gaussian log-density; reuse log-sum-exp normalization from `gmm/em.rs::normalizeresponsibilities`
- M-step subspace: per-group SVD via `nalgebra::SVD` (same as `regression.rs::fdata_to_pc_1d` pattern), applied to the weighted centered data slice of group k
- Regularization: reuse `data_scaled_reg` from `gmm/covariance.rs` for the noise floor on b_k
- BIC/ICL: reuse `compute_bic`, `compute_icl` from `gmm/em.rs`; parameter count = k*(m*d_k − d_k*(d_k−1)/2) + k + k + (k−1) (subspace + a_k + b_k + weights)

**Log-density formula (for one observation x_i against group k):**

```rust
// Source: Bouveyron & Brunet (2012), implemented using fdars linalg helpers
fn log_density_subspace(
    x: &[f64],       // centered observation, length m
    mean: &[f64],    // group mean, length m
    u_k: &[f64],     // subspace columns, m * d_k column-major
    a_k: &[f64],     // within-subspace variances, length d_k
    b_k: f64,        // noise variance
    m: usize,
    d_k: usize,
) -> f64 {
    let diff: Vec<f64> = x.iter().zip(mean).map(|(&xi, &mi)| xi - mi).collect();
    // Project diff onto subspace: z = U_k^T diff (length d_k)
    let mut z = vec![0.0; d_k];
    for j in 0..d_k {
        for r in 0..m {
            z[j] += u_k[r + j * m] * diff[r];  // column-major U_k
        }
    }
    // Within-subspace log-likelihood
    let mut ll = 0.0;
    for j in 0..d_k {
        if a_k[j] <= 0.0 { return f64::NEG_INFINITY; }
        ll -= 0.5 * (a_k[j].ln() + z[j].powi(2) / a_k[j]);
    }
    // Complement squared norm: ||diff||^2 - ||z||^2
    let diff_sq: f64 = diff.iter().map(|v| v * v).sum();
    let z_sq: f64 = z.iter().zip(a_k).map(|(&zi, &ai)| zi * zi).sum::<f64>(); // reuse z components
    // complement norm: diff_sq - sum of (U_k^T diff)^2
    let complement_sq = diff_sq - z.iter().map(|v| v * v).sum::<f64>();
    if b_k <= 0.0 { return f64::NEG_INFINITY; }
    ll -= 0.5 * (((m - d_k) as f64) * b_k.ln() + complement_sq / b_k);
    // Constant: -0.5 * m * ln(2π)
    ll -= 0.5 * (m as f64) * std::f64::consts::TAU.ln() * 0.5;
    let _ = z_sq; // silence unused
    ll
}
```

---

### 2. funFEM — Discriminative-Subspace Clustering

**Reference:** Bouveyron & Brunet (2014), "Discriminative functional data clustering." The Fisher-EM algorithm alternates: (1) find the discriminative subspace W that maximizes between-class scatter / within-class scatter on the FPC scores, then (2) run GMM EM in that subspace.

**Simplified implementation (no new crate):**

Since eigenvector decomposition of a scatter-ratio matrix reduces to solving a generalized eigenvalue problem B·w = λ·W·w (where B = between-scatter, W = within-scatter), and we have no `nalgebra::SymmetricEigen` over non-symmetric pairs (only symmetric), the approach is:

1. Compute global FPC scores via `fdata_to_pc_1d` (d = `ncomp` components, large enough to capture variance)
2. Run GMM E-step on current cluster assignments to get soft responsibilities
3. Compute between-cluster scatter B_soft (d×d) and within-cluster scatter W_soft (d×d) from soft responsibilities
4. Use power iteration or SVD of W_soft^{-1} B_soft to find top `p_disc` discriminative directions (p_disc < k−1 ≤ d)
5. Project scores onto discriminative subspace: scores_disc = scores × Vdisc (n × p_disc)
6. Run GMM EM on scores_disc
7. Repeat until convergence of log-likelihood

**Step 4 without generalized eigenvalue solver:** Cholesky-invert W_soft (using existing `cholesky_d` + forward/backward solve) to form W^{-1}B, then SVD to get top eigenvectors. The `nalgebra::SVD` path already available from `regression.rs` is used (convert to `DMatrix`, run SVD, back to `FdMatrix`).

**Key constraint:** p_disc ≤ min(k−1, ncomp). Document in rustdoc that this implementation performs a single-pass Fisher-EM (not the full iterative Fisher-EM with subspace re-estimation per iteration) for tractability without a generalized-eigenvalue crate.

---

### 3. DBSCAN — Density Clusterer over Functional L2 Distances

**Reference:** Ester et al. (1996) DBSCAN. Over functional data, the distance metric is the functional L2 distance (Simpson-integrated). No R-specific deviation; reuses `l2_distance_matrix` directly.

**Algorithm:**

```
1. D = l2_distance_matrix(data, argvals)        // n×n precomputed [VERIFIED: distance.rs:67]
2. labels = vec![NOISE (usize::MAX) or Option::None; n]
3. cluster_id = 0
4. for each point i in 0..n:
     if labels[i] is already set, skip
     neighbors = { j : D[i,j] <= eps && j != i }
     if neighbors.len() < min_points:
       labels[i] = NOISE  // remains unassigned
     else:
       expand_cluster(i, neighbors, cluster_id, eps, min_points, D, labels)
       cluster_id += 1
5. return labels (noise represented as sentinel)
```

**Noise representation decision:** Use `Vec<Option<usize>>` where `None` = noise, `Some(c)` = cluster c. This is the most type-safe approach and avoids magic sentinel values. The planner may choose a `usize::MAX` sentinel for `Vec<usize>` if it simplifies downstream code — both are valid.

**No new primitive needed:** `l2_distance_matrix` returns an `FdMatrix` [VERIFIED: `distance.rs:67-71`], and the DBSCAN BFS/DFS operates on it via `(i, j)` indexing.

---

### 4. kCFC — Subspace-Embedding Clustering via Per-Cluster FPCA

**Reference:** Chiou & Li (2007), "Functional clustering and identifying substructures of longitudinal data." R baseline: `fdapace::kCFC`.

**Algorithm:**

```
1. Initialize cluster labels via kmeans_plusplus_init (reuse from clustering.rs pattern)
2. Repeat until convergence (max_iter):
   a. For each cluster k:
      - Select curves in cluster k: data_k = rows(data, cluster==k)
      - If data_k.nrows() < ncomp_k: fall back to global FPCA or shrink ncomp_k
      - fpca_k = fdata_to_pc_1d(data_k, ncomp_k, argvals)?
      - Compute reconstruction error for ALL curves against cluster k's FPCA:
        err[i, k] = ||x_i - fpca_k.reconstruct(fpca_k.project(x_i))||_L2^2
   b. Reassign: cluster[i] = argmin_k err[i, k]
   c. Check convergence: no reassignment change
3. Return assignments + per-cluster FPCA results
```

**Reuse:**
- `fdata_to_pc_1d(data_k, ncomp_k, argvals)` [VERIFIED: `regression.rs:287-321`]
- `FpcaResult::project(&x_i_mat)` [VERIFIED: `regression.rs:81-103`]
- `FpcaResult::reconstruct(scores, ncomp)` [VERIFIED: `regression.rs:106-170`]

**Implementation note:** `FpcaResult::project` takes `&FdMatrix`, so single-curve projection needs a 1-row FdMatrix. The reconstruction error is computed via `helpers::l2_distance` with Simpson weights. No new math required.

**Pitfall:** When a cluster has fewer curves than `ncomp_k`, `fdata_to_pc_1d` will clamp ncomp via `ncomp.min(n).min(m)` [VERIFIED: `regression.rs:321`], so empty or near-empty clusters degrade gracefully rather than panic.

---

### 5. Joint Align-and-Cluster — Elastic K-Means / Sangalli Joint

**Reference:** Sangalli et al. (2010) joint clustering and alignment; Srivastava et al. `fdasrvf` elastic k-means variant. Alternates template estimation (Karcher mean per cluster in elastic metric) with reassignment by elastic distance to templates.

**Algorithm:**

```
1. Initialize templates via karcher_mean on random subsets, or k-means++ random select
2. Repeat until convergence (max_iter):
   a. Reassign: for each curve i, cluster[i] = argmin_k amplitude_distance(curve_i, template_k)
   b. For each cluster k with >= 1 member:
      template_k = karcher_mean(data_k, argvals, config)
3. Return assignments + final templates
```

**Reuse:**
- `alignment::amplitude_distance(f, g, argvals)` [VERIFIED: `alignment/mod.rs:77-83`] — amplitude (L-amplitude) distance; shape-invariant, measures vertical distance after optimal warping
- `alignment::karcher_mean(data, argvals, config)` [VERIFIED: `alignment/mod.rs:67`] — Karcher mean in elastic metric
- `alignment::elastic_distance(f, g, argvals, lambda)` alternative if the user wants full elastic (amplitude + phase)

**Config choice:** Expose `use_amplitude_only: bool` (default true, using `amplitude_distance`) vs full elastic distance (`elastic_distance` with `lambda=0.0`). This keeps the implementation flexible without new primitives.

**Pitfall:** `karcher_mean` takes full curve sets, not slices — so constructing subset `FdMatrix` from cluster members is required each iteration. Use `FdMatrix::from_column_major` on a gathered row buffer.

---

## Reuse Map Summary

| Clusterer | Exact Functions Called | Source File |
|-----------|------------------------|-------------|
| funHDDC | `gmm/em.rs: e_step, hard_assignments, resp_to_membership, compute_bic, compute_icl` | `gmm/em.rs` |
| funHDDC | `gmm/init.rs: kmeans_init_assignments, init_params_from_assignments` | `gmm/init.rs` |
| funHDDC | `gmm/covariance.rs: data_scaled_reg, identity_cov` | `gmm/covariance.rs` |
| funHDDC | `linalg.rs: cholesky_d, mahalanobis_sq, log_det_from_cholesky` (for log-density) | `linalg.rs` |
| funFEM | `regression.rs: fdata_to_pc_1d, FpcaResult` | `regression.rs` |
| funFEM | `gmm/em.rs: e_step, hard_assignments, resp_to_membership` | `gmm/em.rs` |
| funFEM | `linalg.rs: cholesky_d, cholesky_forward_back, cholesky_solve` (W^{-1}B via Cholesky) | `linalg.rs` |
| DBSCAN | `distance.rs: l2_distance_matrix` | `distance.rs` |
| kCFC | `regression.rs: fdata_to_pc_1d, FpcaResult::project, FpcaResult::reconstruct` | `regression.rs` |
| kCFC | `helpers.rs: l2_distance, simpsons_weights` | `helpers.rs` |
| align-and-cluster | `alignment: karcher_mean, amplitude_distance, elastic_distance` | `alignment/mod.rs` |

**Adjusted Rand Index:** No ARI helper exists in the repo [VERIFIED: grep returned empty on `adjusted_rand`, `rand_index`, `ari`]. Must add as a `#[cfg(test)]` helper in the test module. It is a pure combinatorial computation (contingency table + choose-2 counts) and requires no crate.

---

## API Surface

### funHDDC (in `gmm/subspace.rs`)

```rust
/// Configuration for funHDDC per-group subspace clustering.
///
/// Implements a simplified subspace covariance GMM: each group k has d_k leading
/// eigenvectors (intrinsic subspace) plus an isotropic noise variance b_k on the
/// complement. This is equivalent to the [AkBk] model from Bouveyron & Brunet (2012),
/// simplified from the full 6-model family offered by the R `funHDDC` package.
/// The R `funHDDC` 6-model family (akjbkqkdk through a1b1q1d1) is deliberately
/// NOT implemented — only the single AkBk model is provided for tractability.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FunHddcConfig {
    pub k: usize,             // number of clusters (default: 2)
    pub d_k: usize,           // intrinsic dimension per group (default: 2)
    pub max_iter: usize,      // EM iterations (default: 100)
    pub tol: f64,             // log-likelihood tolerance (default: 1e-6)
    pub n_init: usize,        // random restarts (default: 3)
    pub seed: u64,            // base random seed (default: 42)
    pub ncomp_init: usize,    // global FPCA components for init features (default: 10)
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FunHddcResult {
    pub cluster: Vec<usize>,            // length n
    pub membership: FdMatrix,           // n × k soft responsibilities
    pub subspaces: Vec<FdMatrix>,       // k × (m × d_k) — one subspace matrix per group
    pub within_vars: Vec<Vec<f64>>,     // k × d_k within-subspace variances
    pub noise_vars: Vec<f64>,           // k isotropic noise variances
    pub means: Vec<Vec<f64>>,           // k group means (length m each)
    pub weights: Vec<f64>,              // k mixing proportions
    pub log_likelihood: f64,
    pub bic: f64,
    pub icl: f64,
    pub iterations: usize,
    pub converged: bool,
    pub k: usize,
}

#[must_use = "expensive computation whose result should not be discarded"]
pub fn funhddC_cluster(
    data: &FdMatrix,
    argvals: &[f64],
    config: &FunHddcConfig,
) -> Result<FunHddcResult, FdarError>
```

### funFEM (in `clustering_advanced.rs`)

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FunFemConfig {
    pub k: usize,          // clusters (default: 2)
    pub ncomp: usize,      // global FPC components for score space (default: 10)
    pub p_disc: usize,     // discriminative subspace dimension (default: k-1, clamped to ncomp)
    pub max_iter: usize,   // outer Fisher-EM iterations (default: 50)
    pub tol: f64,          // log-likelihood convergence tolerance (default: 1e-6)
    pub seed: u64,         // random seed (default: 42)
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FunFemResult {
    pub cluster: Vec<usize>,        // length n
    pub membership: FdMatrix,       // n × k
    pub disc_subspace: FdMatrix,    // ncomp × p_disc discriminative directions
    pub log_likelihood: f64,
    pub iterations: usize,
    pub converged: bool,
}

#[must_use = "expensive computation whose result should not be discarded"]
pub fn funfem_cluster(
    data: &FdMatrix,
    argvals: &[f64],
    config: &FunFemConfig,
) -> Result<FunFemResult, FdarError>
```

### DBSCAN (in `clustering_advanced.rs`)

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct DbscanConfig {
    pub eps: f64,           // neighborhood radius (L2 functional distance, default: 0.5)
    pub min_points: usize,  // minimum neighbors to be a core point (default: 3)
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DbscanResult {
    /// Cluster assignment per curve. `None` = noise (unassigned).
    pub cluster: Vec<Option<usize>>,
    /// Number of discovered clusters (excluding noise).
    pub n_clusters: usize,
    /// Number of noise points.
    pub n_noise: usize,
    /// Precomputed distance matrix used (n × n).
    pub distances: FdMatrix,
}

#[must_use = "expensive computation whose result should not be discarded"]
pub fn dbscan_fd(
    data: &FdMatrix,
    argvals: &[f64],
    config: &DbscanConfig,
) -> Result<DbscanResult, FdarError>
```

### kCFC (in `clustering_advanced.rs`)

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct KcfcConfig {
    pub k: usize,           // clusters (default: 2)
    pub ncomp: usize,       // per-cluster FPC components (default: 3)
    pub max_iter: usize,    // outer loop iterations (default: 50)
    pub seed: u64,          // random seed for init (default: 42)
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct KcfcResult {
    pub cluster: Vec<usize>,                    // length n
    pub fpca_models: Vec<Option<FpcaResult>>,   // k — None if cluster was empty
    pub reconstruction_errors: FdMatrix,        // n × k errors
    pub iterations: usize,
    pub converged: bool,
}

#[must_use = "expensive computation whose result should not be discarded"]
pub fn kcfc_cluster(
    data: &FdMatrix,
    argvals: &[f64],
    config: &KcfcConfig,
) -> Result<KcfcResult, FdarError>
```

### Joint Align-and-Cluster (in `clustering_advanced.rs`)

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AlignClusterConfig {
    pub k: usize,               // clusters (default: 2)
    pub max_iter: usize,        // outer iterations (default: 20)
    pub seed: u64,              // random seed for init (default: 42)
    pub use_amplitude_only: bool, // if true, use amplitude_distance; else elastic_distance (default: true)
    pub elastic_lambda: f64,    // penalty for elastic_distance when use_amplitude_only=false (default: 0.0)
    pub karcher_max_iter: usize, // inner Karcher mean iterations (default: 15)
    pub karcher_tol: f64,       // Karcher mean tolerance (default: 1e-4)
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AlignClusterResult {
    pub cluster: Vec<usize>,        // length n
    pub templates: Vec<Vec<f64>>,   // k cluster template curves (each length m)
    pub distances: FdMatrix,        // n × k distances to templates
    pub iterations: usize,
    pub converged: bool,
}

#[must_use = "expensive computation whose result should not be discarded"]
pub fn align_cluster_fd(
    data: &FdMatrix,
    argvals: &[f64],
    config: &AlignClusterConfig,
) -> Result<AlignClusterResult, FdarError>
```

---

## API Conventions (verified against existing code)

All conventions below are `[VERIFIED]` by reading source files this session:

- **Return type:** `Result<T, FdarError>` [VERIFIED: `clustering.rs:546`, `gmm/em.rs:312`]
- **`#[must_use]`:** on all public fitting functions [VERIFIED: `clustering.rs:544`, `regression.rs:286`]
- **`#[non_exhaustive]`:** on all public result/config structs [VERIFIED: `gmm/mod.rs:38,68`]
- **Derives:** `#[derive(Debug, Clone, PartialEq)]` on configs, `#[derive(Debug, Clone)]` on results with `FdMatrix` fields [VERIFIED: `gmm/mod.rs:27,38`]
- **Serde gate:** `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` [VERIFIED: `regression.rs:23-24`]
- **RNG seeding:** `StdRng::seed_from_u64(seed + k as u64)` for per-restart seeding [VERIFIED: `gmm/cluster.rs:23`]
- **Clippy gate:** `cargo clippy --all-targets --features linalg,parallel -- -D warnings` [VERIFIED: MEMORY.md / CI config pointer]
- **Parameter order:** `(data, argvals, config)` or `(data, argvals, k, ...)` — functional data first [VERIFIED: `clustering.rs:545`]
- **Error variants in use:**
  - `FdarError::InvalidDimension` for shape/length mismatches [VERIFIED: `error.rs:7-13`]
  - `FdarError::InvalidParameter` for out-of-range values (k=0, eps<=0, min_points=0) [VERIFIED: `error.rs:14-18`]
  - `FdarError::ComputationFailed` for SVD or numerical failures [VERIFIED: `error.rs:19-23`]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Functional L2 distance matrix | Custom loop | `distance::l2_distance_matrix` |
| FPC score extraction | SVD from scratch | `regression::fdata_to_pc_1d` |
| EM E-step (log-sum-exp) | Custom normalizer | `gmm/em.rs::normalizeresponsibilities` (pub(super), copy pattern) |
| K-means++ init on feature vectors | Custom seeded init | `gmm/init.rs::kmeans_init_assignments` |
| Cholesky inversion | Gaussian elimination | `linalg::cholesky_d` + `forward_solve` |
| Elastic template computation | Custom gradient descent | `alignment::karcher_mean` |
| Elastic curve distance | Custom DP | `alignment::amplitude_distance` or `elastic_distance` |
| Silhouette/CH for quality | Custom metric | `clustering::silhouette_score_from_distances`, `calinski_harabasz_from_distances` |
| Integration weights | Trapz loop | `helpers::simpsons_weights` |
| BIC/ICL | Manual formula | `gmm/em.rs::compute_bic`, `compute_icl` |

**Key insight:** Every numeric primitive needed by all five algorithms already exists in fdars-core. The only genuinely new code is the per-group subspace update (funHDDC M-step), the Fisher-EM scatter computation (funFEM), the BFS/DFS graph search (DBSCAN), the per-cluster FPCA assignment loop (kCFC), and the elastic template reassignment loop (align-and-cluster). None of these require a new crate.

---

## Common Pitfalls

### Pitfall 1: Empty Cluster During EM/kCFC Iteration

**What goes wrong:** When a cluster loses all members during reassignment, the M-step divides by zero (effective weight = 0).

**Why it happens:** k-means++ init doesn't guarantee no empty clusters after first iteration, especially with k close to n.

**How to avoid:** Mirror the existing `gmm/em.rs` pattern — when `nk < 1e-15`, replace the component with `identity_cov` [VERIFIED: `gmm/em.rs:184-186`]. For kCFC, when a cluster becomes empty, keep the previous FPCA model or skip reassignment.

**Warning signs:** Reconstruction error matrix contains NaN/infinity.

### Pitfall 2: DBSCAN Eps in Wrong Units

**What goes wrong:** `eps` is specified in functional L2 distance units (Simpson-integrated). If the user passes a Euclidean eps, neighborhoods will be incorrect.

**Why it happens:** Functional L2 distance magnitude depends on `argvals` range. For argvals on [0,1], a constant-1 curve has L2 norm ≈ 1. For argvals on [0,100], L2 norm is ~10.

**How to avoid:** Document `eps` in the rustdoc as "functional L2 distance (same units as `l2_distance_matrix`)". Suggest a default eps derived from the dataset's median pairwise distance or a fraction of it.

**Warning signs:** All points are noise (eps too small) or one giant cluster (eps too large).

### Pitfall 3: funFEM — Degenerate Within-Scatter Matrix

**What goes wrong:** The within-scatter W_soft can be singular when all responsibilities are concentrated on few clusters (early iterations), making Cholesky inversion fail.

**Why it happens:** Soft responsibilities near 0/1 create a near-singular W.

**How to avoid:** Add a data-scaled regularization floor (same `data_scaled_reg` pattern as GMM) to the diagonal of W_soft before inverting. Fall back to identity if Cholesky fails.

**Warning signs:** `ComputationFailed` from `cholesky_d` on the W matrix.

### Pitfall 4: kCFC — `FpcaResult::project` Takes FdMatrix, Not Slice

**What goes wrong:** Attempting to call `project` on a single curve requires constructing a 1-row `FdMatrix`. Using the wrong slice length causes `InvalidDimension`.

**Why it happens:** `project` expects `data: &FdMatrix` and checks `data.shape().1 == mean.len()` [VERIFIED: `regression.rs:83-90`].

**How to avoid:** Wrap each curve in a 1-row FdMatrix: `FdMatrix::from_slice(&curve, 1, m)?` before calling `fpca_k.project(...)`. Use `reconstruct` on the returned 1-row scores.

### Pitfall 5: funHDDC — d_k > min(n_k, m) Causes SVD Truncation

**What goes wrong:** If `d_k` is larger than the number of group members `n_k`, the per-group SVD cannot yield `d_k` meaningful singular vectors.

**Why it happens:** `fdata_to_pc_1d` clamps `ncomp = ncomp.min(n).min(m)` [VERIFIED: `regression.rs:321`], so the subspace dimension silently shrinks.

**How to avoid:** After per-group FPCA, use the actual returned dimension (rotation.ncols()) rather than `config.d_k`. Validate `d_k < min(n_k, m)` in the input validation step.

### Pitfall 6: align-and-cluster — `karcher_mean` Needs Minimum 1 Curve

**What goes wrong:** If a cluster becomes empty, calling `karcher_mean` on 0 rows panics or errors.

**How to avoid:** Before the template-update step, check each cluster has at least 1 member. If empty, reinitialize the template by picking a random non-member curve (like k-means empty-cluster fallback).

### Pitfall 7: Column-Major FdMatrix Row Extraction

**What goes wrong:** Building a subset FdMatrix from selected rows (e.g., for kCFC per-cluster data) requires correct index arithmetic. Naive slice extraction from `data.as_slice()` gives column chunks, not row chunks.

**Why it happens:** `FdMatrix` is column-major: `data.as_slice()[i + j*n]`. Row-gathering requires transposing memory layout.

**How to avoid:** Use `data.to_row_major()` to get a flat row-major buffer, then select rows, then re-wrap via `FdMatrix::from_column_major` after transposing the selected rows. Or use the existing `data.row(i)` method for individual row access in a loop.

---

## Validation Architecture

Nyquist validation is enabled (`workflow.nyquist_validation: true` in `.planning/config.json` [VERIFIED]).

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in (`#[test]`, `#[cfg(test)]`) |
| Config file | none — inline `#[cfg(test)] mod tests` in each source file |
| Quick run command | `cargo test -p fdars-core --lib --features linalg 2>&1 \| tail -20` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel -- --test-threads=4` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CLUS-01 funHDDC | Well-separated groups recovered with ARI > 0.9 | unit (inline) | `cargo test -p fdars-core --lib funhddC` | Wave 0 gap |
| CLUS-01 funHDDC | BIC/ICL finite and reasonable | unit | `cargo test -p fdars-core --lib funhddC_bic` | Wave 0 gap |
| CLUS-01 funHDDC | Invalid input → FdarError | unit | `cargo test -p fdars-core --lib funhddC_invalid` | Wave 0 gap |
| CLUS-01 funFEM | Well-separated groups recovered | unit | `cargo test -p fdars-core --lib funfem_recovery` | Wave 0 gap |
| CLUS-01 funFEM | Invalid input → FdarError | unit | `cargo test -p fdars-core --lib funfem_invalid` | Wave 0 gap |
| CLUS-01 DBSCAN | Core-point clusters formed correctly | unit | `cargo test -p fdars-core --lib dbscan_core_points` | Wave 0 gap |
| CLUS-01 DBSCAN | Injected noise curves labeled None | unit | `cargo test -p fdars-core --lib dbscan_noise_flagging` | Wave 0 gap |
| CLUS-01 DBSCAN | eps=0 → all noise | unit | `cargo test -p fdars-core --lib dbscan_zero_eps` | Wave 0 gap |
| CLUS-01 DBSCAN | Invalid eps (<=0) → FdarError | unit | `cargo test -p fdars-core --lib dbscan_invalid_eps` | Wave 0 gap |
| CLUS-01 kCFC | Well-separated groups recovered | unit | `cargo test -p fdars-core --lib kcfc_recovery` | Wave 0 gap |
| CLUS-01 kCFC | Reconstruction errors smaller for true cluster | unit | `cargo test -p fdars-core --lib kcfc_errors` | Wave 0 gap |
| CLUS-01 kCFC | k > n → FdarError | unit | `cargo test -p fdars-core --lib kcfc_invalid` | Wave 0 gap |
| CLUS-01 align-cluster | Shape-shifted group found | unit | `cargo test -p fdars-core --lib align_cluster_shape_shift` | Wave 0 gap |
| CLUS-01 align-cluster | Well-separated groups recovered | unit | `cargo test -p fdars-core --lib align_cluster_recovery` | Wave 0 gap |
| CLUS-01 align-cluster | Invalid k → FdarError | unit | `cargo test -p fdars-core --lib align_cluster_invalid` | Wave 0 gap |

### Synthetic Data Generation Patterns

**Cluster-recovery tests** (same pattern as existing `generate_two_clusters` in `clustering.rs` [VERIFIED: `clustering.rs:1037-1059`]):

```rust
// Pattern for 2-cluster well-separated test (reuse in all 5 clusterers)
fn two_separated_clusters(n_per: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<usize>) {
    // Cluster 0: sin waves; Cluster 1: sin waves shifted +5 vertically
    // Ground truth: labels 0..n_per-1 = cluster 0, n_per..2n-1 = cluster 1
    // Source: mirrors clustering.rs:1037-1059 [VERIFIED]
}
```

**Shape-shifted group test** (for align-and-cluster):

```rust
// Cluster 0: sin waves on [0,1]; Cluster 1: time-warped sin waves (phase shift)
// Expected: align-and-cluster finds both groups; k-means does not
fn time_warped_clusters(n_per: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<usize>)
```

**DBSCAN noise test:**

```rust
// 2 tight clusters of 5 curves each + 2 isolated "outlier" curves far from both
// Expected: 2 clusters + 2 noise (None)
fn clusters_with_noise(n_per: usize, m: usize) -> (FdMatrix, Vec<f64>)
```

### Adjusted Rand Index Helper (test-only)

No ARI exists in the repo [VERIFIED: grep returned empty]. Implement as a `#[cfg(test)]` function in a shared `test_helpers.rs` extension or inline in each test module:

```rust
/// Adjusted Rand Index between two label vectors (permutation-invariant agreement).
/// ARI = 1.0 means perfect agreement up to label permutation.
/// ARI near 0 means no better than chance.
#[cfg(test)]
pub(crate) fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    // Build contingency table, compute ARI via choose-2 formula
    // ARI = (RI - E[RI]) / (max(RI) - E[RI])
    // Source: Hubert & Arabie (1985) formula
    // ...
}
```

For DBSCAN where `cluster: Vec<Option<usize>>`, filter out noise points before computing ARI on the remaining labeled subset.

### Recovery Threshold

- **Target:** ARI ≥ 0.90 on well-separated test data (separation ≥ 5σ), consistent with `test_silhouette_score_well_separated` which asserts mean silhouette > 0.5 [VERIFIED: `clustering.rs:1299`].
- **DBSCAN:** n_clusters correct AND noise count correct.
- **align-and-cluster:** ARI ≥ 0.90 on time-warped test data where k-means would fail.

### Wave 0 Gaps

- [ ] `fdars-core/src/gmm/subspace.rs` — funHDDC implementation + inline tests
- [ ] `fdars-core/src/clustering_advanced.rs` — funFEM, DBSCAN, kCFC, align-and-cluster + inline tests
- [ ] `adjusted_rand_index` helper in `src/test_helpers.rs` (or inline in new modules)
- [ ] `pub use` additions in `gmm/mod.rs` and `lib.rs`

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Global covariance GMM only | Per-group subspace covariance (funHDDC) | Handles high-dimensional FPC score spaces without full covariance blow-up |
| Euclidean GMM in score space | Discriminative-subspace GMM (funFEM) | Better separation in between-class directions |
| L2 k-means | DBSCAN over functional L2 distances | Finds arbitrary-shaped clusters, handles noise |
| Global FPCA then k-means on scores | Per-cluster FPCA (kCFC) | Identifies subspace structure per group, not globally |
| Shape-naive k-means | Elastic-distance k-means (align-and-cluster) | Shape-invariant — clusters by amplitude shape, not phase |

---

## Environment Availability

This phase is purely code/config changes with no external runtime dependencies beyond the existing Rust toolchain.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | All | ✓ | 1.97.0 | — |
| nalgebra 0.33 | funHDDC SVD, funFEM scatter | ✓ (in Cargo.lock) | 0.33 | — |
| rand 0.8 | RNG seeding | ✓ (in Cargo.lock) | 0.8 | — |
| rayon 1.10 | Parallel E-step | ✓ (in Cargo.lock) | 1.10 | sequential via feature gate |

---

## Security Domain

Phase 33 is a pure numerical Rust library with no I/O, no network access, no authentication, no session management, and no user-controlled deserialization paths beyond existing `serde` feature gate (which is already audited in prior phases). ASVS V2/V3/V4/V6 do not apply.

| ASVS Category | Applies | Notes |
|---------------|---------|-------|
| V2 Authentication | no | Library crate, no auth |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | No authorization |
| V5 Input Validation | yes | Validate n, m, k, eps, min_points, argvals.len() == m at function entry |
| V6 Cryptography | no | StdRng used for reproducibility, not security |

**V5 validation checklist for new public functions:**

- `n == 0 || m == 0` → `InvalidDimension`
- `k == 0 || k > n` → `InvalidParameter`
- `argvals.len() != m` → `InvalidDimension`
- `eps <= 0.0` → `InvalidParameter`
- `min_points == 0` → `InvalidParameter`
- `ncomp == 0 || ncomp > min(n, m)` → `InvalidParameter`
- `d_k == 0 || d_k >= m` → `InvalidParameter`

---

## Project Constraints (from CLAUDE.md)

- Rust 2021 edition, MSRV 1.81 (linalg feature requires 1.84+; all new code must compile on 1.81 without `linalg`)
- All public functions return `Result<T, FdarError>` — no panics on input validation
- Column-major `FdMatrix` — no row-major storage in public types
- No new crate dependency
- Additive/non-breaking — zero changes to existing public signatures
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
- Inline `#[cfg(test)] mod tests` — no separate test files
- `#[must_use]` on all expensive computation functions
- `#[non_exhaustive]` on all public result/config structs
- Per-thread RNG: `StdRng::seed_from_u64(seed + k as u64)` for randomized init
- Crate-root re-exports in `src/lib.rs`
- Serde gate on new result types: `#[cfg_attr(feature = "serde", derive(...))]`

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `normalizeresponsibilities` in `gmm/em.rs` is `pub(super)` and accessible from `gmm/subspace.rs` | Reuse Map | funHDDC would need to copy the log-sum-exp normalization inline (~15 lines) |
| A2 | `cholesky_forward_back` and `cholesky_solve` in `linalg.rs` are `pub(crate)` and accessible from new files | Reuse Map | funFEM W^{-1}B computation needs to inline Cholesky solve |
| A3 | `kmeans_init_assignments` in `gmm/init.rs` is `pub(super)` | Reuse Map | funHDDC init would need a local copy of the k-means++ algorithm |
| A4 | funFEM Fisher-EM discriminative subspace improves cluster recovery vs plain GMM-on-FPC-scores in practice | Algorithm Formulations | The simplified Fisher-EM (without full iterative subspace re-estimation) may give marginal improvement; acceptable since CONTEXT.md does not require quantitative R-parity |
| A5 | `amplitude_distance` signature: `(f: &[f64], g: &[f64], argvals: &[f64]) -> f64` | Reuse Map / align-and-cluster | If signature differs, align-and-cluster needs adjustment |

Note on A1/A2/A3: These functions have `pub(super)` or `pub(crate)` visibility. The planner should verify access levels. If `pub(super)` blocks cross-module access, the pattern is to either copy the ~15-line helper inline or promote it to `pub(crate)`. Given the established precedent (`linalg.rs` is `pub(crate)`, confirming `pub(crate)` items ARE accessible from new files in the same crate), only `pub(super)` items in `gmm/` are constrained to `gmm/` children.

**Verification needed:** `pub(super)` items in `gmm/em.rs` and `gmm/init.rs` are accessible from `gmm/subspace.rs` because `subspace.rs` is a sibling module under `gmm/` (same parent). This is correct Rust visibility semantics — `pub(super)` means visible to the parent module and all its children. funHDDC in `gmm/subspace.rs` CAN use `pub(super)` items from `gmm/em.rs`. [ASSUMED — standard Rust visibility rule, not grep-verified this session]

---

## Open Questions

1. **Noise representation in DBSCAN result**
   - What we know: `Vec<Option<usize>>` (None=noise) is most type-safe; `Vec<usize>` with `usize::MAX` is simpler for downstream ARI computation.
   - What's unclear: Whether downstream callers (tests, future users) prefer Option or sentinel.
   - Recommendation: Use `Vec<Option<usize>>` in the public API (type-safe, no magic value). Internally convert for ARI computation by filtering None points.

2. **funFEM: single-pass vs multi-pass Fisher-EM subspace estimation**
   - What we know: The full Fisher-EM alternates scatter computation and GMM EM inside each outer iteration. Single-pass (compute subspace once from initial GMM, then run EM in that subspace) is simpler.
   - Recommendation: Implement multi-pass (subspace re-estimated each outer iteration) for correctness, capped at `max_iter`. Document in rustdoc.

3. **align-and-cluster: `karcher_mean` config struct**
   - What we know: `karcher_mean` exists [VERIFIED: `alignment/mod.rs:67`] but the exact signature (config struct fields) was not fully read.
   - Recommendation: Planner should read `alignment/karcher.rs` to confirm the `karcher_mean` signature before writing the wave that calls it.

---

## Sources

### Primary (HIGH confidence — source files read this session)

- `fdars-core/src/gmm/mod.rs:1-79` — GmmResult, GmmClusterResult, CovType definitions; pub use map
- `fdars-core/src/gmm/em.rs:1-393` — e_step, m_step, finalize_gmm, compute_bic, compute_icl, hard_assignments
- `fdars-core/src/gmm/covariance.rs:1-186` — data_scaled_reg, identity_cov, regularize_cov
- `fdars-core/src/gmm/init.rs:1-184` — build_features, kmeans_init_assignments, init_params_from_assignments
- `fdars-core/src/gmm/cluster.rs:1-80` — GmmClusterConfig, run_multiple_inits
- `fdars-core/src/distance.rs:1-218` — l2_distance_matrix, pairwise_distance_matrix signatures
- `fdars-core/src/clustering.rs:1-1637` — KmeansResult, FuzzyCmeansResult, silhouette/CH; k-means++ init pattern
- `fdars-core/src/regression.rs:1-400` — fdata_to_pc_1d, FpcaResult::project, FpcaResult::reconstruct
- `fdars-core/src/alignment/mod.rs:1-135` — karcher_mean, amplitude_distance, elastic_distance pub use
- `fdars-core/src/linalg.rs:1-152` — cholesky_d, forward_solve, mahalanobis_sq, cholesky_factor, cholesky_solve
- `fdars-core/src/matrix.rs:1-80` — FdMatrix column-major layout, indexing
- `fdars-core/src/error.rs:1-51` — FdarError variants
- `fdars-core/src/lib.rs:1-199` — module declarations, crate-root re-exports

### Secondary (ASSUMED — algorithm formulations from training knowledge)

- Bouveyron & Brunet (2012): funHDDC AkBk model formulation
- Bouveyron & Brunet (2014): Fisher-EM discriminative subspace method
- Ester et al. (1996): DBSCAN neighborhood expansion algorithm
- Chiou & Li (2007): kCFC per-cluster FPCA reassignment loop
- Sangalli et al. (2010): Joint clustering and alignment via alternating Karcher mean + elastic distance
- Hubert & Arabie (1985): Adjusted Rand Index formula

---

## Metadata

**Confidence breakdown:**
- Reuse map (exact function names, signatures, visibility): HIGH — all read from source files this session
- Algorithm formulations (math): MEDIUM/ASSUMED — from training knowledge; R packages not fetched (web search disabled)
- API surface (struct fields, fn signatures): HIGH — derived from verified codebase patterns
- Pitfalls: HIGH — derived from verified source code patterns

**Research date:** 2026-08-20
**Valid until:** 2026-09-20 (stable Rust library, no external API drift)
