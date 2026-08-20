# Phase 33: Model-Based & Density Functional Clustering — Pattern Map

**Mapped:** 2026-08-20
**Files analyzed:** 7 new/modified files
**Analogs found:** 7 / 7

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/clustering/model_based.rs` (funHDDC, funFEM, kCFC) | service | CRUD / batch | `src/clustering.rs` (`kmeans_fd`) + `src/gmm/em.rs` | role-match + data-flow-match |
| `src/clustering/density.rs` (DBSCAN, align-and-cluster) | service | CRUD / batch | `src/clustering.rs` + `src/distance.rs` | role-match |
| `src/gmm/funhddc.rs` (funHDDC EM + per-group subspace) | service | batch / CRUD | `src/gmm/em.rs` + `src/gmm/covariance.rs` | exact role, exact data flow |
| `src/clustering/mod.rs` (barrel update) | config / wiring | — | `src/gmm/mod.rs` | exact |
| `src/lib.rs` (crate-root re-export additions) | config / wiring | — | existing `pub use clustering::{...}` block (line 419) | exact |
| Config structs (`FunHddcConfig`, `FunFemConfig`, `DbscanConfig`, `KcfcConfig`, `AlignClusterConfig`) | config | — | `src/gmm/cluster.rs` `GmmClusterConfig` | exact |
| Result structs (`FunHddcResult`, `FunFemResult`, `DbscanResult`, `KcfcResult`, `AlignClusterResult`) | model | — | `src/clustering.rs` `KmeansResult` + `src/gmm/mod.rs` `GmmResult` | exact |

---

## Pattern Assignments

---

### `src/gmm/funhddc.rs` — funHDDC per-group subspace EM

**Analog:** `src/gmm/em.rs` (EM loop structure) + `src/gmm/covariance.rs` (per-group covariance accumulation)

**Imports pattern** (copy from `gmm/em.rs` lines 1–13):
```rust
use super::covariance::{accumulate_full_cov_weighted, regularize_cov, data_scaled_reg};
use super::init::kmeans_init_assignments;
use super::{CovType, GmmResult};
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use rand::prelude::*;
```

**Config struct pattern** (copy from `gmm/cluster.rs` lines 49–86, adapt field names):
```rust
/// Configuration for funHDDC functional clustering.
///
/// Implements a simplified per-group subspace covariance model where each
/// group k has an intrinsic-dimension `d_k` subspace (leading eigenvectors)
/// plus an isotropic residual-noise variance on the complement.
///
/// Note: this is a single representative model, NOT the full six-model
/// akjbkqkdk family from the R `funHDDC` package.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FunHddcConfig {
    /// Intrinsic subspace dimension per group (default: 2).
    pub d_k: usize,
    /// Number of basis functions for FPCA projection (default: 8).
    pub nbasis: usize,
    /// Maximum EM iterations (default: 200).
    pub max_iter: usize,
    /// Convergence tolerance (default: 1e-6).
    pub tol: f64,
    /// Number of random initializations (default: 3).
    pub n_init: usize,
    /// Base random seed (default: 42).
    pub seed: u64,
}

impl Default for FunHddcConfig { ... }
```

**Result struct pattern** (copy from `gmm/mod.rs` lines 36–63, adapt fields):
```rust
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FunHddcResult {
    /// Hard cluster assignments (length n).
    pub cluster: Vec<usize>,
    /// Posterior membership probabilities (n x K).
    pub membership: FdMatrix,
    /// Per-group subspace bases: K vecs of (m x d_k) matrices.
    pub subspace_bases: Vec<FdMatrix>,
    /// Per-group isotropic noise variance.
    pub noise_vars: Vec<f64>,
    /// Log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Number of EM iterations.
    pub iterations: usize,
    /// Whether EM converged.
    pub converged: bool,
    /// Number of clusters K.
    pub k: usize,
}
```

**EM loop core pattern** (copy from `gmm/em.rs` `gmm_em` function — full E/M iteration loop):
```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn funhddc_em(
    data: &FdMatrix,
    argvals: &[f64],
    k: usize,
    config: &FunHddcConfig,
    seed: u64,
) -> Result<FunHddcResult, FdarError> {
    // dimension checks → FdarError::InvalidDimension / InvalidParameter
    // StdRng::seed_from_u64(seed) — same seeding pattern as gmm/em.rs
    // kmeans_init_assignments for warm start (reuse gmm/init.rs)
    // E-step: project each curve onto per-group subspace, compute log-density
    // M-step: update subspace bases (SVD of cluster-centered curves), update noise variance
    // convergence check on log-likelihood delta < tol
    Ok(FunHddcResult { ... })
}
```

**Error handling pattern** (copy from `gmm/em.rs`):
```rust
if n == 0 || m == 0 {
    return Err(FdarError::InvalidDimension {
        parameter: "data",
        expected: "non-empty matrix".into(),
        actual: format!("{n}x{m}"),
    });
}
if k == 0 || k > n {
    return Err(FdarError::InvalidParameter {
        parameter: "k",
        message: format!("k={k} must be in 1..={n}"),
    });
}
// Numerical failure (SVD/convergence):
return Err(FdarError::ComputationFailed {
    operation: "funhddc_em",
    detail: "subspace SVD failed for group ...".to_string(),
});
```

**RNG seeding** (same per-init offset as `gmm/cluster.rs` `run_multiple_inits` lines 22–23):
```rust
let seed = base_seed.wrapping_add(init as u64 * 1000 + k as u64);
let mut rng = StdRng::seed_from_u64(seed);
```

**Deviation from analog:** funHDDC uses per-group FPCA (SVD of within-cluster centered data rows) rather than a parametric Gaussian covariance. The M-step computes leading `d_k` singular vectors per cluster rather than accumulating a full/diagonal covariance matrix. Document this divergence from R `funHDDC` in the module doc comment.

---

### `src/clustering/model_based.rs` — funFEM and kCFC

**Analog:** `src/clustering.rs` (`kmeans_fd` assignment loop, lines 458–515) + `src/regression.rs` (`fdata_to_pc_1d` for FPCA subspace)

**Imports pattern** (model based file):
```rust
use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;
use crate::{iter_maybe_parallel, slice_maybe_parallel};
use rand::prelude::*;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
```

**funFEM config + result** — copy `GmmClusterConfig` struct shape exactly; substitute fields:
```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FunFemConfig {
    /// Number of discriminative subspace dimensions (default: 2).
    pub ncomp: usize,
    /// Number of basis functions for initial FPCA (default: 8).
    pub nbasis: usize,
    /// Maximum EM iterations (default: 200).
    pub max_iter: usize,
    /// Convergence tolerance (default: 1e-6).
    pub tol: f64,
    /// Number of random initializations (default: 3).
    pub n_init: usize,
    /// Base random seed (default: 42).
    pub seed: u64,
}
```

**kCFC assignment-loop pattern** (copy `kmeans_iterate`, lines 482–515, then replace center-update with per-cluster `fdata_to_pc_1d` + score projection):
```rust
// kCFC per-cluster FPCA reassignment loop (cap: config.max_iter iterations)
for iteration in 0..config.max_iter {
    // 1. For each cluster: fit FPCA on member curves via fdata_to_pc_1d(...)
    // 2. Project all curves onto each cluster's subspace → get reconstruction error
    // 3. Assign each curve to cluster with lowest reconstruction error
    // 4. Check convergence: cluster unchanged → break
}
```

**#[must_use] on all public functions** — mandatory, copy from `kmeans_fd` line 544:
```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn kcfc_fd(...) -> Result<KcfcResult, FdarError> { ... }

#[must_use = "expensive computation whose result should not be discarded"]
pub fn funfem_fd(...) -> Result<FunFemResult, FdarError> { ... }
```

---

### `src/clustering/density.rs` — DBSCAN and align-and-cluster

**Analog (DBSCAN):** `src/distance.rs` (`l2_distance_matrix`, lines 66–71) for neighborhood computation; `src/clustering.rs` for noise-label result shape.

**Analog (align-and-cluster):** `src/alignment/` (`karcher_mean`, `elastic_distance`) + `src/clustering.rs` assignment loop.

**DBSCAN imports**:
```rust
use crate::distance::l2_distance_matrix;
use crate::error::FdarError;
use crate::helpers::NUMERICAL_EPS;
use crate::matrix::FdMatrix;
```

**DBSCAN noise label convention** — use `Option<usize>` in result to distinguish noise from cluster membership:
```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct DbscanResult {
    /// Cluster assignment per curve: `None` = noise/unassigned, `Some(k)` = cluster k.
    pub cluster: Vec<Option<usize>>,
    /// Number of clusters found (excluding noise).
    pub n_clusters: usize,
    /// Number of noise curves.
    pub n_noise: usize,
}
```

**DbscanConfig pattern** (copy `GmmClusterConfig` struct shape, simpler fields):
```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DbscanConfig {
    /// Neighborhood radius in functional L2 distance (default: 0.5).
    pub eps: f64,
    /// Minimum points in ε-neighborhood to be a core point (default: 3).
    pub min_points: usize,
}
```

**DBSCAN core using precomputed distance matrix** (reuse `l2_distance_matrix`):
```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn dbscan_fd(
    data: &FdMatrix,
    argvals: &[f64],
    config: &DbscanConfig,
) -> Result<DbscanResult, FdarError> {
    // dimension checks → FdarError::InvalidParameter / InvalidDimension
    let dist_mat = l2_distance_matrix(data, argvals);
    // Standard DBSCAN: find core points (neighbors >= min_points within eps),
    // expand clusters via BFS/DFS from core points, label remaining as None.
    ...
}
```

**align-and-cluster pattern** (reuse elastic distance + k-medoids style assignment):
```rust
use crate::alignment::{elastic_distance, karcher_mean};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct AlignClusterConfig {
    /// Number of clusters (default: 2).
    pub k: usize,
    /// Maximum iterations (default: 50).
    pub max_iter: usize,
    /// Elastic penalty λ (default: 0.1).
    pub lambda: f64,
    /// Base random seed (default: 42).
    pub seed: u64,
}
```

---

### `src/clustering/mod.rs` — barrel update

**Analog:** `src/gmm/mod.rs` (lines 13–19, 77–79)

**Pattern to copy exactly:**
```rust
// Existing submodule declarations stay; add:
pub mod density;
pub mod model_based;

// At bottom of mod.rs, add pub use blocks:
pub use density::{
    dbscan_fd, funhddc_fd, AlignClusterConfig, AlignClusterResult,
    DbscanConfig, DbscanResult,
};
pub use model_based::{
    funfem_fd, kcfc_fd, FunFemConfig, FunFemResult, KcfcConfig, KcfcResult,
};
```

Note: if clustering is currently a single flat file (`src/clustering.rs`), convert to a submodule directory `src/clustering/mod.rs` first, keeping all existing code in `mod.rs` (or in `kmeans.rs` + `fuzzy.rs` sub-files). The existing public API must not change.

---

### `src/lib.rs` — crate-root re-export additions

**Analog:** existing clustering re-export block (lines 418–422) and GMM block (lines 327–331).

**Pattern to add** (append to the existing `pub use clustering::{...}` block):
```rust
// Re-export clustering types (existing + new)
pub use clustering::{
    calinski_harabasz, calinski_harabasz_from_distances, fuzzy_cmeans_fd, kmeans_fd,
    silhouette_score, silhouette_score_from_distances, FuzzyCmeansResult, KmeansResult,
    // NEW:
    dbscan_fd, funfem_fd, kcfc_fd,
    AlignClusterConfig, AlignClusterResult, DbscanConfig, DbscanResult,
    FunFemConfig, FunFemResult, KcfcConfig, KcfcResult,
};

// Re-export GMM clustering types (existing + new funHDDC)
pub use gmm::{
    gmm_cluster, gmm_cluster_with_config, gmm_em, predict_gmm, CovType, GmmClusterConfig,
    GmmClusterResult, GmmResult,
    // NEW (from gmm::funhddc):
    funhddc_fd, FunHddcConfig, FunHddcResult,
};
```

---

## Shared Patterns

### #[must_use] on all expensive public functions
**Source:** `src/clustering.rs` line 544, `src/gmm/cluster.rs` line 104
**Apply to:** `funhddc_fd`, `funfem_fd`, `kcfc_fd`, `dbscan_fd`, `align_cluster_fd`
```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn <name>(...) -> Result<...Result, FdarError> { ... }
```

### #[non_exhaustive] on all result and config structs
**Source:** `src/clustering.rs` lines 16–17, `src/gmm/cluster.rs` line 51
**Apply to:** every new `*Result` and `*Config` struct
```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FooResult { ... }
```

### serde feature gating on config structs
**Source:** `src/gmm/cluster.rs` (note: `GmmClusterConfig` does NOT currently gate serde — check before adding). The project convention per CLAUDE.md is:
```rust
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
```
Apply only to Config structs (not Result structs with `FdMatrix` fields unless `FdMatrix` also implements serde).

### RNG seeding (per init, not per thread)
**Source:** `src/gmm/cluster.rs` lines 21–23
```rust
// For multiple-init loops:
let seed = base_seed.wrapping_add(init as u64 * 1000 + k as u64);
let mut rng = StdRng::seed_from_u64(seed);
```
**Apply to:** funFEM, kCFC, funHDDC (any randomized init).

### Error handling — dimension checks at entry
**Source:** `src/clustering.rs` lines 556–581, `src/gmm/em.rs`
```rust
if n == 0 || m == 0 {
    return Err(FdarError::InvalidDimension {
        parameter: "data",
        expected: "non-empty matrix".into(),
        actual: format!("{n}x{m}"),
    });
}
if k == 0 {
    return Err(FdarError::InvalidParameter {
        parameter: "k",
        message: "number of clusters must be > 0".into(),
    });
}
```
**Apply to:** all five new public clustering functions before any computation.

### Module doc comment style
**Source:** `src/clustering.rs` lines 1–5, `src/gmm/mod.rs` lines 1–11
```rust
//! Short one-line summary.
//!
//! Longer description of approach and key functions.
//!
//! Key functions:
//! - [`fn_name`] — one-line description
```

### Column-major row extraction
**Source:** `src/clustering.rs` line 587 (`data.to_row_major()`)
**Apply to:** all new files that iterate over curves. Extract to a flat row-major buffer once, then index as `&buf[i*m..(i+1)*m]`.

### Parallelism macros
**Source:** `src/clustering.rs` lines 9, 250–265
```rust
use crate::{iter_maybe_parallel, slice_maybe_parallel};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// Usage:
slice_maybe_parallel!(curve_indices)
    .map(|&i| { ... })
    .collect()
```
**Apply to:** any outer loop over n curves in distance computation or assignment steps.

---

## Inline Test Convention

### Synthetic cluster generator
**Source:** `src/clustering.rs` lines 1037–1059 + `src/gmm/tests.rs` lines 12–32

Copy the `generate_two_clusters(n_per, m)` helper pattern for each new test module. For DBSCAN add noise injection; for align-and-cluster add a phase-shifted group.

### Adjusted Rand Index helper
**Status: does not exist in the codebase.** Must be added as a test-only private helper in each new test module (or in `src/test_helpers.rs`).

Implement as a `#[cfg(test)]` free function:
```rust
#[cfg(test)]
fn adjusted_rand_index(labels_true: &[usize], labels_pred: &[usize]) -> f64 {
    // Contingency table → RI → expected RI → ARI
    // ARI = (RI - E[RI]) / (max(RI) - E[RI])
    // Standard combinatorial formula; no external crate needed.
}
```

Place in each new file's `#[cfg(test)] mod tests { ... }` block, or add once to `src/test_helpers.rs` as `pub(crate)`.

### Test structure pattern
**Source:** `src/clustering.rs` lines 1030–1635 (sectioned with `// ===` banners)
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    fn generate_two_clusters(n_per: usize, m: usize) -> (FdMatrix, Vec<f64>) { ... }

    // ============== <AlgorithmName> tests ==============

    #[test]
    fn test_<fn>_basic() { ... }

    #[test]
    fn test_<fn>_finds_clusters() {
        // ARI >= 0.9 threshold on well-separated synthetic data
    }

    #[test]
    fn test_<fn>_deterministic() {
        // same seed → identical result
    }

    #[test]
    fn test_<fn>_invalid_input() {
        // empty data, k > n, etc. → is_err()
    }
}
```

---

## No Analog Found

All new files have close analogs. No files require purely research-based patterns.

---

## Metadata

**Analog search scope:** `fdars-core/src/clustering.rs`, `fdars-core/src/gmm/` (all 6 files), `fdars-core/src/distance.rs`, `fdars-core/src/lib.rs`, `fdars-core/src/regression.rs` (FPCA), `fdars-core/src/alignment/` (names only)
**Files scanned:** 11
**Pattern extraction date:** 2026-08-20
