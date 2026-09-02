# Feature Research: GAK + Kernel-k-Means + Gram-Matrix Export (v0.32.0)

**Domain:** Global Alignment Kernel and kernel clustering for functional curve sets — Rust crate `fdars-core`
**Milestone:** v0.32.0 (promotes GAP-01 from `GAP-BACKLOG.md`)
**Researched:** 2026-09-02
**Confidence:** HIGH — mathematical spec derived from Cuturi 2011 (primary), tslearn@0.9.0 API (cross-checked), Dhillon et al. 2004 kernel-k-means paper (cross-checked), scikit-learn SVC precomputed-kernel convention (HIGH confidence from official docs).

---

## Precise Mathematical Specification

This section gives the exact math the planner must implement against. All claims are sourced from primary references; confidence level shown per claim.

### A. Triangular Global Alignment Kernel (TGAK)

**Reference:** Cuturi (2011), "Fast Global Alignment Kernels," ICML. Implemented in tslearn@0.9.0 `tslearn.metrics.gak`.

#### A.1 Local (Gaussian) Kernel

Given two scalar observations `x_i` and `y_j` (or multivariate, with Euclidean norm):

```
phi(x_i, y_j; sigma) = exp( -||x_i - y_j||^2 / (2 * sigma^2) )
```

`sigma > 0` is the bandwidth parameter. This is the pointwise similarity used at each DP cell. (Confidence: HIGH — direct from tslearn doc formula.)

#### A.2 Unnormalized GAK — the Alignment Sum

Let `x = (x_1, ..., x_n)` and `y = (y_1, ..., y_m)` be two time series / curve samples. Let `A(x, y)` be the set of all monotone alignment paths (causal, endpoint-anchored, each step moves right, down, or diagonally — the same lattice as DTW). The **unnormalized global alignment kernel** is:

```
k(x, y) = sum_{pi in A(x, y)}  product_{t=1}^{|pi|} phi(x_{pi_1(t)}, y_{pi_2(t)}; sigma)
```

This is a *sum* over paths, not a min — turning the DTW min-cost into a soft (log-sum-exp) accumulation. (Confidence: HIGH.)

#### A.3 Dynamic Programming Recursion

The standard DP fills an `(n+1) x (m+1)` accumulator `M`:

```
M[0][0] = 1
M[i][0] = 0   for i >= 1
M[0][j] = 0   for j >= 1
M[i][j] = phi(x_i, y_j; sigma) * (M[i-1][j] + M[i-1][j-1] + M[i][j-1])   for i,j >= 1
k(x, y) = M[n][m]
```

Note the three predecessors match DTW's three moves (up, diagonal, left). The result is the full accumulated path weight. (Confidence: HIGH — from the standard ICASSP/ICML recurrence; also confirmed by tslearn source comments.)

#### A.4 Log-Domain Stable Recursion (REQUIRED for long series)

The product of `phi` values underflows to zero for long series. The stable form operates in log-space. Define `L[i][j] = log M[i][j]`:

```
L[0][0] = 0
L[i][0] = -inf   for i >= 1
L[0][j] = -inf   for j >= 1
L[i][j] = log_phi(i, j) + logsumexp(L[i-1][j], L[i-1][j-1], L[i][j-1])
```

where `log_phi(i, j) = log phi(x_i, y_j; sigma) = -||x_i - y_j||^2 / (2*sigma^2)` (a non-positive value, since phi in (0,1]).

And `logsumexp(a, b, c) = max(a,b,c) + log(exp(a - max) + exp(b - max) + exp(c - max))`.

The final unnormalized kernel in log-space is `L[n][m]`. To recover `k(x,y)` one would exponentiate, but normalization can be done entirely in log-space (see A.5). (Confidence: HIGH — tslearn confirms log-space re-execution to avoid overflow, this is the standard approach.)

Note on the existing `softmin3` in `metric/soft_dtw.rs`: the soft-DTW softmin is `min - gamma * ln(sum(exp(...)))`. The GAK log-domain recursion uses `logsumexp` (no gamma weighting — effectively gamma = 1 in the max-entropy view). These are **different** recursions; the GAK implementation must NOT reuse `softmin3` directly; it uses a plain `logsumexp` accumulator.

#### A.5 Triangular Band Constraint (the "T" in TGAK)

The "Triangular" in TGAK refers to a position kernel `omega(i, j)` that is a **Toeplitz kernel with compact support of width T**:

```
omega(i, j) = 0   if |i - j| >= T
omega(i, j) = 1   otherwise   (the triangular variant uses a flat-top or triangular shape)
```

When `|i - j| >= T` the cell contributes zero, so the DP only fills the `|i - j| < T` band — yielding O(T * min(n, m)) instead of O(n * m). This is the Sakoe-Chiba band constraint applied to the kernel accumulation. For `T = max(n, m)` (no constraint), the full matrix is filled. The dtwclust R package documents: "T is zero whenever the distance exceeds T." (Confidence: MEDIUM — documented in dtwclust and Cuturi 2011 abstract, but exact omega formula details are from the paper body not fully extracted.)

**Default recommendation:** expose `band: Option<usize>` where `None` = full matrix (exact TGAK, O(n*m)) and `Some(T)` = banded (O(T*min(n,m))). The soft_dtw module's banding analogy is `karcher_mean_with_band`'s `band_frac: Option<f64>` parameter — use the same pattern.

**Constraint on series length ratios:** TGAK is only valid when `max(n,m) / min(n,m) <= 2` (i.e., lengths within 2:1 ratio). The tslearn and dtwclust implementations enforce this. An `InvalidParameter` error should be returned when the ratio is violated.

#### A.6 Normalized GAK (the valid similarity in [0, 1])

The raw `k(x, y)` is not bounded. The **normalized GAK** is:

```
gak(x, y) = k(x, y) / sqrt(k(x, x) * k(y, y))
```

In log-space:

```
log_gak(x, y) = L[n][m] - 0.5 * (L_xx[n][n] + L_yy[m][m])
gak(x, y) = exp(log_gak(x, y))
```

Properties:
- `gak(x, x) = 1` for all x (self-similarity is always 1 after normalization).
- `gak(x, y) in (0, 1]` for all x, y (since k(x,y) > 0 always, and by Cauchy-Schwarz `k(x,y) <= sqrt(k(x,x)*k(y,y))`).
- The normalized GAK is a **valid PSD kernel** on time series. (Confidence: HIGH — from tslearn doc and standard kernel normalization theory.)

This normalization is what makes GAK suitable for kernel machines: SVM, kernel-k-means, kernel PCA, etc.

---

### B. Sigma Bandwidth Heuristic

**Reference:** Cuturi (2011) original paper; tslearn `sigma_gak` function (tslearn@0.9.0).

The heuristic (from tslearn's `sigma_gak` docstring and confirmed by the Cuturi 2011 paper):

1. Draw `n_samples` (default 100) random **individual point pairs** from *different* time series in the dataset — i.e., sample point `x_i` from series `x` and point `y_j` from series `y` where `x != y`.
2. Compute the pairwise squared Euclidean distances between these sampled points.
3. Take the **median** of these pointwise squared distances to get `med_sq`.
4. The suggested sigma is: `sigma = sqrt(med_sq) * sqrt(median_length)` where `median_length` is the median series length in the dataset.

Equivalently: `sigma = sqrt(med_sq * median_length)`.

Rationale: a GAK cell spans a sequence of `~median_length` pointwise distances; scaling by `sqrt(median_length)` accounts for the accumulation over the alignment path. (Confidence: MEDIUM — the textual description in tslearn docs confirms "median distance of different points ... scaled by the square root of the median length"; the exact formula above is the plausible implementation; the internal `_sigma_gak` reshapes and samples points. HIGH confidence for the general approach, MEDIUM for the exact formula details.)

**fdars function signature:** `sigma_gak(data: &FdMatrix, n_samples: usize, seed: u64) -> Result<f64, FdarError>`. Expose as a public utility, not tied to any config struct.

---

### C. Kernel K-Means on Curves

**Reference:** Dhillon, Guan, Kulis (2004), "Kernel k-Means, Spectral Clustering and Normalized Cuts," KDD. Implemented in tslearn@0.9.0 `tslearn.clustering.KernelKMeans`.

#### C.1 Objective

Kernel k-means minimizes within-cluster variance in the RKHS (reproducing kernel Hilbert space) induced by the kernel `k`. Given an `n x n` Gram matrix `K` (where `K[i,j] = gak(x_i, x_j)`), the objective is:

```
J(C_1, ..., C_K) = sum_{k=1}^{K} sum_{i in C_k} ||phi(x_i) - mu_k||_H^2
```

where `mu_k = (1/|C_k|) * sum_{j in C_k} phi(x_j)` is the centroid of cluster `k` in feature space H, and `phi` is the (implicit) feature map satisfying `<phi(x), phi(y)>_H = k(x, y)`.

Expanding the squared norm using the kernel (no explicit feature map needed):

```
||phi(x_i) - mu_k||_H^2
  = K[i,i]
  - (2 / |C_k|) * sum_{j in C_k} K[i,j]
  + (1 / |C_k|^2) * sum_{j in C_k} sum_{l in C_k} K[j,l]
```

The three terms are:
- `K[i,i]`: self-kernel of point `i` (equals 1 after normalization, so this term is constant).
- `(2 / |C_k|) * sum_{j in C_k} K[i,j]`: twice the mean kernel similarity of `x_i` to cluster members.
- `(1 / |C_k|^2) * sum_{j,l in C_k} K[j,l]`: within-cluster mean kernel (constant per cluster across iterations).

(Confidence: HIGH — from Dhillon et al. 2004 and confirmed by the arxiv 2011.06461 kernel-k-means paper.)

#### C.2 Assignment Step

Assign each point `x_i` to the cluster `C_k` that minimizes the kernel distance:

```
argmin_k  [ -2/|C_k| * sum_{j in C_k} K[i,j]  +  1/|C_k|^2 * sum_{j,l in C_k} K[j,l] ]
```

(The `K[i,i]` term is constant and does not affect the argmin; with normalized GAK it is exactly 1.)

The cluster-mean terms `(1/|C_k|^2) * sum K[j,l]` can be pre-computed once per iteration and reused for all `i`.

#### C.3 Algorithm

```
Initialize cluster assignments z (random from 1..K, n_init restarts, pick best)
Pre-compute K = GAK Gram matrix (n x n)  — done once, O(n^2) kernel evaluations
Repeat until convergence or max_iter:
    For each cluster k: compute within_k = (1/|C_k|^2) * sum_{j,l in C_k} K[j,l]
    For each point i:
        For each cluster k:
            dist_k = K[i,i] - 2/|C_k| * sum_{j in C_k} K[i,j] + within_k
        z_new[i] = argmin_k dist_k
    If z_new == z: converged; break
    z = z_new
    Handle empty clusters: reassign to the point farthest from its current centroid
Convergence: when no assignment changes between iterations (tol on inertia change also acceptable)
```

**n_init:** Run the full loop `n_init` times with different random initializations; keep the assignment with the lowest total `J`. (Confidence: HIGH — standard kernel-k-means; tslearn default is `n_init=1` but documents that it can be set higher.)

**Empty cluster handling:** When a cluster loses all members, assign the point with the highest distance-to-centroid from any cluster to the empty cluster. This is the standard heuristic. (Confidence: MEDIUM — standard practice; tslearn does not document the specific heuristic in its API docs.)

**Inertia / convergence:** The `inertia` is `J(C_1,...,C_K)`, the sum of kernel distances. Convergence when `|inertia_prev - inertia| < tol` or when assignments do not change. (Confidence: HIGH — matches tslearn `tol=1e-6` default.)

#### C.4 Predict for New Points

After fitting (Gram matrix `K_train` of shape `n_train x n_train` and cluster assignments `z`), predicting cluster for a new point `x_*` requires computing:

```
k_star[j] = gak(x_*, x_j)   for j = 1..n_train  (n_train kernel evaluations)
dist_k(x_*) = 1  -  2/|C_k| * sum_{j in C_k} k_star[j]  +  within_k
predicted_cluster = argmin_k dist_k(x_*)
```

The `within_k` terms come from the training Gram matrix and do not change. The fitted model must store `z`, `within_k` per cluster, and the training data (or at least the training rows, to compute `gak(x_*, x_j)`). (Confidence: HIGH — standard kernel-k-means prediction; confirmed from the structural requirement for kernel machines with precomputed kernels.)

#### C.5 Result Type

```rust
pub struct KernelKmeansResult {
    pub cluster: Vec<usize>,          // n assignments, 0-indexed
    pub inertia: f64,                 // final J value
    pub n_iter: usize,                // iterations taken
    pub converged: bool,
}
```

The fitted model (for predict) is a separate `KernelKmeansFit` struct (or `KernelKmeansResult` with the training data reference and `within_k` stored). The config follows the pattern of `GmmClusterConfig`:

```rust
pub struct KernelKmeansConfig {
    pub n_clusters: usize,           // default 3
    pub max_iter: usize,             // default 50, matches tslearn
    pub tol: f64,                    // default 1e-6
    pub n_init: usize,               // default 5 (higher than tslearn's 1 for safety)
    pub sigma: f64,                  // GAK bandwidth; use sigma_gak() for "auto"
    pub band: Option<usize>,         // triangular band (None = full matrix)
    pub seed: u64,                   // RNG seed for reproducibility
}
```

---

### D. Gram-Matrix Export for External SVM

**Reference:** scikit-learn `SVC(kernel='precomputed')` convention; confirmed from scikit-learn docs (all versions 1.4–1.9).

#### D.1 The Convention

When using a precomputed kernel with scikit-learn's SVC:

- **Training:** pass a **symmetric `n_train x n_train`** Gram matrix to `svc.fit(K_train, y)`.
- **Prediction:** pass an **`n_test x n_train`** matrix to `svc.predict(K_test_train)`, where `K_test_train[i, j] = k(x_test_i, x_train_j)`.
- The SVC does **not** store raw data; it stores support vectors' indices into the training set.

This convention is the de facto standard: sklearn's `SVC(kernel='precomputed')` is exactly what users plug fdars' Gram matrix into. (Confidence: HIGH — directly from scikit-learn docs for versions 1.4, 1.5, 1.6, 1.9.)

#### D.2 fdars Functions Needed

Two public functions:

```rust
// Symmetric n x n Gram matrix (for fitting an SVM)
pub fn gak_gram_train(data: &FdMatrix, sigma: f64, band: Option<usize>)
    -> Result<FdMatrix, FdarError>

// n_test x n_train matrix (for SVM prediction on new data)
pub fn gak_gram_test(
    test_data: &FdMatrix,
    train_data: &FdMatrix,
    sigma: f64,
    band: Option<usize>,
) -> Result<FdMatrix, FdarError>
```

Both return **normalized** GAK values (the `gak(x,y)` in [0,1]) by default — this is what makes the matrix PSD and suitable for kernel machines. The diagonal of `gak_gram_train` is all 1.0 (self-similarity = 1). (Confidence: HIGH.)

#### D.3 PSD Guarantee

The normalized TGAK (with the triangular position kernel with compact support) is a PSD kernel. This is the main theoretical contribution of Cuturi (2011): the triangular constraint ensures the kernel is PSD, unlike certain non-normalized or non-triangular variants. fdars exports normalized values; callers using external SVMs benefit from the PSD guarantee automatically. (Confidence: HIGH — core result of the Cuturi 2011 paper.)

---

## Feature Landscape

### Table Stakes — Users Expect These

These are the features that constitute the GAK capability; missing any one makes the deliverable incomplete.

| Feature | Why Expected | Complexity | Dependency on Existing Code |
|---------|--------------|------------|-----------------------------|
| `gak_distance(x, y, sigma, band)` — unnormalized log-GAK | Core primitive; all other features build on it | MEDIUM | Reuses log-sum-exp idea from `softmin3` (but NOT the same recursion); new `metric/gak.rs` |
| Normalized `gak(x, y, sigma, band)` returning value in [0,1] | PSD kernel property; mandatory for kernel machines | LOW (atop unnormalized) | Calls `gak_distance` for `k(x,x)`, `k(y,y)`, `k(x,y)` |
| Log-domain stable recursion (avoid underflow for long series) | Long series (n>50) will underflow in product space | MEDIUM | Must implement `logsumexp` 3-way — similar to `softmin3` but additive not min |
| `gak_gram_train(data, sigma, band) -> FdMatrix` — symmetric n×n | Entire kernel-k-means and SVM glue require the Gram matrix | MEDIUM | Reuses `self_distance_matrix` from `metric/mod.rs` (renaming to kernel matrix) |
| `gak_gram_test(test, train, sigma, band) -> FdMatrix` — n_test×n_train | SVM prediction requires this exact shape | LOW (atop gram_train pattern) | Reuses `cross_distance_matrix` from `metric/mod.rs` |
| `kernel_kmeans_fd(data, config) -> Result<KernelKmeansFit, FdarError>` | The headline consumer of GAK; without it, GAK is just a metric variant | HIGH | Builds on `gak_gram_train`; cluster logic new (no existing kernel-k-means in clustering.rs) |
| `KernelKmeansFit::predict(new_data) -> Result<Vec<usize>, FdarError>` | Clustering without prediction is not useful for ML workflows | MEDIUM | Stores training data + `within_k`; calls `gak_gram_test` internally |
| `KernelKmeansConfig` struct | fdars convention for complex methods | LOW | Pattern identical to `GmmClusterConfig`, `ElasticConfig` |
| Series-length-ratio guard (reject >2:1 ratio with `InvalidParameter`) | TGAK validity constraint documented by both Cuturi 2011 and dtwclust | LOW | New validation at function entry |
| `KernelKmeansResult` / `KernelKmeansFit` with `Debug + Clone + PartialEq` | fdars convention for all public result types | LOW | Convention from 97 existing types |

### Differentiators — Competitive Advantage

These are not required to ship a correct GAK implementation, but they raise the quality bar and match tslearn's API.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| `sigma_gak(data, n_samples, seed) -> Result<f64, FdarError>` | Removes manual bandwidth tuning; matches tslearn's `sigma_gak`; users can use `auto` | LOW | Sample `n_samples` random point pairs from different series; compute median of squared pointwise distances × sqrt(median series length); reproducible via `seed` |
| `cdist_gak(data1, data2, sigma, band) -> Result<FdMatrix, FdarError>` | Pairwise cross-kernel matrix (n1×n2); needed for `gak_gram_test` and any cross-dataset kernel query | LOW (reuses `cross_distance_matrix`) | Wrapper that calls normalized `gak` over all pairs; `data2=None` → symmetric self-matrix |
| Rayon parallelism for Gram matrix computation | n×n kernel evaluations are embarrassingly parallel; critical for n>100 | LOW | `iter_maybe_parallel!` over rows (same pattern as `soft_dtw_self_1d` and `soft_dtw_cross_1d`) |
| `n_init` restarts in kernel-k-means | Reduces sensitivity to initialization; tslearn default is 1, but 5 is safer | LOW | Outer loop, pick minimum-inertia run |
| Optional `band: Option<usize>` parameter (TGAK triangular constraint) | Trades exactness for speed: O(T·min(n,m)) vs O(n·m); mirrors `band_frac` in `karcher_mean_with_band` | MEDIUM | Band guard in the DP inner loop; same pattern as existing banded alignment |
| Serde support on `KernelKmeansConfig` / `KernelKmeansFit` | Pipeline persistence; standard fdars convention under `serde` feature | LOW | `#[cfg_attr(feature = "serde", ...)]` — same as all other config/result types |
| `#[must_use]` on `gak_gram_train`, `gak_gram_test`, `kernel_kmeans_fd` | fdars convention for expensive computations; 74+ functions already marked | LOW | One annotation per expensive function |
| Criterion benchmark in `benches/` | fdars convention; measures Gram computation vs n (n ∈ {50, 200, 500}) | LOW | Follows existing bench patterns in BENCH-RESULTS.md |

### Anti-Features — Explicitly Out of Scope

| Anti-Feature | Why Requested | Why Excluded | What to Do Instead |
|--------------|---------------|---------------|--------------------|
| Native kernel SVM (SVC in fdars) | Logical next step after Gram-matrix export | Would require a quadratic programming solver (no QP crate currently used; adding one is a new heavy dependency and a separate large feature); violates the "no new crate dependency" convention and scope | Export the `n×n` Gram matrix via `gak_gram_train` and call scikit-learn / libsvm externally |
| GPU / CUDA acceleration | GAK matrix computation is O(n²·n·m) flops — GPU would help at n>1000 | No GPU support exists anywhere in fdars; adding GPU is a new infrastructure story | Use the banded TGAK (`band: Some(T)`) to reduce to O(n²·T·m); rayon parallelism covers typical research-scale datasets |
| Multidimensional (multivariate) curves | tslearn's gak supports `d`-dimensional series | The existing `FdMatrix` is 1D curves (n rows, m eval points); multivariate extension requires a different data representation (`FdCurveSet` or stacked FdMatrix) — out of scope for this milestone | Implement as a follow-on once `FdMatrix` multivariate support is addressed (if needed) |
| Kernel PCA via GAK | Useful but not in the milestone scope (GAP-01 specifies kernel-k-means + SVM glue only) | Adds a new algorithm family; not part of the GAP-01 requirement | Gram matrix export enables external KernelPCA (sklearn) directly |
| Non-normalized GAK as primary output | Some use-cases (similarity scoring without kernel machines) might want raw log-k | Raw kernel violates PSD property; log-scale output is not a similarity in [0,1] | Expose `gak_log_unnormalized` as `pub(crate)` for internal use; the public API is normalized |
| Online / streaming kernel-k-means | Would require incremental Gram matrix update | Significantly more complex; the existing clustering.rs uses batch k-means exclusively | Use the batch `kernel_kmeans_fd` and re-fit when new data arrives |

---

## Feature Dependencies

```
sigma_gak()               (utility — standalone, no kernel dep)
    │
    └──advises──> KernelKmeansConfig.sigma

gak_distance_log()         (log-domain DP recursion — new metric/gak.rs)
    │
    ├──requires──> logsumexp3() helper (similar to softmin3 but additive)
    │
    └──powers──> gak()      (normalized, public)
                    │
                    ├──powers──> gak_gram_train()   (n×n self Gram matrix)
                    │                │
                    │                └──powers──> kernel_kmeans_fd()   (headline consumer)
                    │                                │
                    │                                └──returns──> KernelKmeansFit
                    │                                                  │
                    │                                                  └──predict()
                    │                                                       │
                    │                                                       └──requires──> gak_gram_test()
                    │
                    └──powers──> gak_gram_test()    (n_test×n_train cross Gram)
                                     │
                                     └──enables──> External SVM (scikit-learn SVC(kernel='precomputed'))
```

### Dependency Notes

- **`gak_distance_log` requires `logsumexp3`:** The log-domain recursion is NOT the same as soft-DTW's `softmin3`. Soft-DTW computes `min - gamma*ln(...)` (soft minimum). GAK computes `ln(exp(a)+exp(b)+exp(c))` (log-sum-exp for a kernel sum). They share the log-sum-exp trick but serve opposite purposes. A new `logsumexp3` helper must be added — do NOT reuse `softmin3`.
- **`kernel_kmeans_fd` requires the full training Gram matrix:** The entire `n×n` Gram matrix must be computed and held in memory during fitting. For n=1000 at f64, this is 8 MB — acceptable. For n=10000 it is 800 MB — document the memory scaling in the API docs.
- **`KernelKmeansFit::predict` requires `gak_gram_test`:** The predict method must compute kernel similarities between new points and training points. The training data (or at minimum, all training rows) must be stored in the `KernelKmeansFit` struct to enable this. This is the only case in fdars where a result struct embeds a copy of the training data — document the memory implications.
- **Series-length ratio guard applies to all GAK functions:** Every public function that calls the DP must check `max(n,m) / min(n,m) <= 2` and return `FdarError::InvalidParameter` if violated. Do not silently truncate.

---

## MVP Definition

### Launch With (v0.32.0)

Minimum viable for the milestone to ship:

- [x] `gak_distance_log(x, y, sigma, band)` — log-domain DP recursion, the core primitive
- [x] `gak(x, y, sigma, band)` — normalized kernel value in [0,1]
- [x] `gak_gram_train(data, sigma, band)` — symmetric n×n Gram matrix
- [x] `gak_gram_test(test_data, train_data, sigma, band)` — n_test×n_train matrix
- [x] `kernel_kmeans_fd(data, config)` — kernel-k-means with `n_init` restarts
- [x] `KernelKmeansFit::predict(new_data)` — assign new curves to trained clusters
- [x] `KernelKmeansConfig` — config struct with `n_clusters`, `sigma`, `max_iter`, `tol`, `n_init`, `band`, `seed`
- [x] `sigma_gak(data, n_samples, seed)` — bandwidth heuristic (differentiator but so low-effort it belongs in v1)
- [x] Series-length-ratio validation and all `InvalidParameter` / `InvalidDimension` error paths
- [x] Inline tests: log-domain vs product-space equivalence (small series), self-similarity = 1, Gram symmetry, kernel-k-means convergence smoke test

### Add After Validation (v0.32.x)

- [ ] Criterion benchmark (n × m grid for Gram computation, k-means convergence) — follow BENCH-RESULTS.md convention
- [ ] Example file (`examples/gak_clustering.rs`) with a labeled dataset + cluster recovery check
- [ ] Serde support for `KernelKmeansConfig` / `KernelKmeansFit` (already behind `cfg_attr` — trivial)

### Future Consideration (v0.33+)

- [ ] Multivariate curve support (requires `FdMatrix` multivariate representation decision)
- [ ] Native kernel SVM (requires QP solver dependency decision)
- [ ] Kernel PCA via GAK Gram matrix

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| `gak` normalized kernel | HIGH | MEDIUM | P1 — foundation |
| `gak_gram_train` | HIGH | LOW (uses existing matrix helpers) | P1 — kernel-k-means depends on it |
| `kernel_kmeans_fd` + `predict` | HIGH | HIGH | P1 — headline deliverable |
| `gak_gram_test` | HIGH | LOW | P1 — SVM glue deliverable |
| `sigma_gak` heuristic | MEDIUM | LOW | P1 — too cheap to defer |
| Log-domain stability | HIGH | MEDIUM | P1 — correctness on realistic data |
| Triangular band constraint | MEDIUM | MEDIUM | P2 — performance optimization |
| Rayon parallelism for Gram | MEDIUM | LOW | P2 — follows existing pattern |
| Criterion bench | LOW | LOW | P2 — fdars convention |
| Native kernel SVM | HIGH | HIGH (new dependency) | P3 — out of scope this milestone |

---

## Competitor Feature Analysis

| Feature | tslearn@0.9.0 | dtwclust (R) | fdars v0.32.0 Plan |
|---------|---------------|--------------|-------------------|
| Unnormalized GAK | `_log_unnormalized_gak` (internal) | `GAK(normalize=FALSE)` returns log | `gak_distance_log` (pub(crate) or pub) |
| Normalized GAK | `gak(s1, s2, sigma)` | `GAK(normalize=TRUE)` | `gak(x, y, sigma, band)` |
| Pairwise Gram matrix | `cdist_gak(dataset1, dataset2)` | `proxy::dist(X, method="gak")` | `gak_gram_train` / `gak_gram_test` |
| Bandwidth heuristic | `sigma_gak(dataset, n_samples=100)` | `NULL` triggers built-in estimate | `sigma_gak(data, n_samples, seed)` |
| Triangular band | implicit (window via `triangular` param) | `window.size` parameter | `band: Option<usize>` |
| Kernel k-means | `KernelKMeans(kernel='gak')` | Not provided | `kernel_kmeans_fd(data, config)` |
| SVM with GAK | `tslearn.svm.TimeSeriesSVC(kernel='gak')` | Not provided | Gram-matrix export only (no native SVM) |
| Parallelism | `n_jobs` parameter | Single-threaded R | `iter_maybe_parallel!` under `parallel` feature |
| Log-domain stability | Yes (re-executes in log if overflow) | Yes | Yes (always log-domain from the start) |

---

## Sources

- Cuturi, M. (2011). "Fast Global Alignment Kernels." Proceedings of the 28th ICML, 929–936. https://icml.cc/2011/papers/489_icmlpaper.pdf
- Dhillon, I., Guan, Y., Kulis, B. (2004). "Kernel k-Means, Spectral Clustering and Normalized Cuts." KDD 2004. https://dl.acm.org/doi/10.1145/1014052.1014118
- tslearn@0.9.0 `gak` API: https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.gak.html
- tslearn@0.9.0 `sigma_gak` API: https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.sigma_gak.html
- tslearn@0.9.0 `KernelKMeans` API: https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.KernelKMeans.html
- tslearn@0.9.0 Kernel Methods user guide: https://tslearn.readthedocs.io/en/stable/user_guide/kernel.html
- dtwclust R package `GAK` function: https://rdrr.io/cran/dtwclust/man/GAK.html
- scikit-learn SVC precomputed kernel: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- GAP-BACKLOG.md GAP-01 (v0.31.0): `.planning/research/GAP-BACKLOG.md`
- survey-pyx.md PYX-01 (v0.31.0): `.planning/research/survey-pyx.md`
- fdars-core `metric/soft_dtw.rs` (existing `softmin3`, `soft_dtw_distance`, pairwise matrix helpers): local codebase
- fdars-core `metric/mod.rs` (existing `self_distance_matrix`, `cross_distance_matrix`): local codebase
- fdars-core `clustering.rs` (existing `KmeansResult`, `KmeansConfig` pattern): local codebase

---
*Feature research for: v0.32.0 Global Alignment Kernel + kernel-k-means + Gram-matrix export*
*Researched: 2026-09-02*
