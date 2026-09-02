# Architecture Research — v0.32.0 GAK Integration

**Domain:** Rust functional-data-analysis library extension (GAK kernel + kernel clustering)
**Researched:** 2026-09-02
**Confidence:** HIGH — based on direct reading of all relevant source files (`soft_dtw.rs`, `metric/mod.rs`, `alignment/clustering.rs`, `clustering.rs`, `distance.rs`, `lib.rs`) and the v0.31.0 `GAP-BACKLOG.md` GAP-01 block.

---

## Module Placement Decision

**Place GAK in `src/metric/gak.rs` — a sibling of `soft_dtw.rs`, not a new top-level module.**

Rationale:

1. `soft_dtw.rs` already owns the alignment-lattice DP that GAK reuses. Sibling placement makes the dependency explicit and keeps both variants together for future maintainers.
2. The existing `metric/mod.rs` already declares ten submodules for specialized pairwise operations; `gak` is the eleventh — it fits the pattern exactly.
3. A new `kernel/` top-level module would imply a broader kernel-methods subsystem (SVM, kernel regression, etc.) that is explicitly out of scope this milestone. Naming it `metric/gak.rs` is honest: GAK is a kernel that produces a similarity/distance matrix, consistent with what every sibling module does.
4. Kernel-k-means lives in `src/kernel_kmeans.rs` at the top level (not inside `metric/`), because it is a clustering algorithm that consumes a Gram matrix — the same level as `clustering.rs` and `clustering_advanced.rs`. It is not itself a distance metric.
5. Gram-matrix export is not a separate file — it is a function (`gak_gram_matrix`) that lives in `metric/gak.rs` and is re-exported from `lib.rs`, exactly as `soft_dtw_self_1d` / `soft_dtw_cross_1d` are today.

---

## New vs Modified Files

### New Files

| File | What It Contains |
|------|-----------------|
| `src/metric/gak.rs` | `gak_distance` (pairwise scalar), `gak_gram_matrix` (n x n Gram), `gak_cross_gram` (n_train x n_test), `GakConfig` (sigma, triangular band, normalize flag), `gak_sigma_median` (bandwidth heuristic) |
| `src/kernel_kmeans.rs` | `kernel_kmeans`, `KernelKMeansConfig`, `KernelKMeansResult` |

### Modified Files

| File | Change |
|------|--------|
| `src/metric/mod.rs` | Add `pub mod gak;` + re-export `gak_distance`, `gak_gram_matrix`, `gak_cross_gram`, `GakConfig`, `gak_sigma_median` |
| `src/lib.rs` | `pub mod kernel_kmeans;` + top-level re-exports for all public items from both new modules |

No other existing file changes. The public API of every existing function is untouched.

---

## Reuse Map: What Comes from `soft_dtw.rs`

The GAK forward DP is structurally identical to `soft_dtw_forward` with two substitutions:

| Aspect | soft_dtw | GAK |
|--------|----------|-----|
| Local cost | `(x[i] - y[j])^2` (squared Euclidean) | `exp(-(x[i] - y[j])^2 / (2 * sigma^2))` (triangular kernel) |
| Accumulation operator | `softmin3(r[i-1][j], r[i][j-1], r[i-1][j-1], gamma)` soft-MIN | `logsumexp3(r[i-1][j], r[i][j-1], r[i-1][j-1])` soft-MAX |
| Final value | `r[n][m]` (alignment cost, low = similar) | `exp(r[n][m])` (alignment kernel value, high = similar) |
| Log domain | Implicit (cost scale) | Explicit: accumulate in log space throughout; final = exp(log_gak) |
| Diagonal constraint | Optional band fraction (Phase 12, v0.16.0) | Optional band fraction — same param type, same logic |

**Directly reusable from `soft_dtw.rs` — same algorithm skeleton:**

- The 2-row rolling-buffer DP structure in `soft_dtw_distance` (O(m) memory, `prev`/`curr` swap). GAK uses the same two-buffer approach in log space.
- `pub(super) fn softmin3` demonstrates the stabilized log-sum-exp pattern GAK mirrors. GAK replaces it with `logsumexp3(a, b, c) = log(exp(a) + exp(b) + exp(c))` using the same max-subtraction stabilization trick.
- `pub(super) fn self_distance_matrix` and `cross_distance_matrix` in `metric/mod.rs` — both `pub(super)`, directly usable by the sibling `gak.rs`.
- The `iter_maybe_parallel!` macro import pattern.
- Band-constraint wiring: the `band_frac: Option<f64>` parameter type and the `band_width = ceil(band_frac * m)` calculation, consistent with the banded elastic alignment added in Phase 12.

**What GAK must add (not in `soft_dtw.rs`):**

1. `logsumexp3(a, b, c)` — soft-MAX instead of soft-MIN; new private function sharing the stabilization idea.
2. Triangular local kernel in log space: `log_cost(i, j) = -(x[i] - y[j])^2 / (2 * sigma^2)`.
3. PSD normalization: `k_norm(x, y) = gak(x, y) / sqrt(gak(x, x) * gak(y, y))`. Diagonal terms computed and cached before the off-diagonal fill loop.
4. Sigma bandwidth heuristic: `gak_sigma_median` — the tslearn default is `median(pairwise_L2) * sqrt(m)`. Uses `distance::l2_distance_matrix`.
5. `GakConfig` config struct.

**Not reused from `soft_dtw.rs`:**

- `soft_dtw_forward`, `soft_dtw_backward`, `soft_dtw_accumulate_gradient` — backward/gradient machinery for the barycenter. GAK this milestone has no gradient requirement; these are not ported.
- `SoftDtwBarycenterResult` / `soft_dtw_barycenter` — not needed.
- `softmin3_val` — the backward-pass helper; not needed.

---

## DP Implementation Note

GAK forward DP in log space, 2-row rolling buffer:

```
// log_cost(i, j) = -(x[i-1] - y[j-1])^2 / (2 * sigma^2)
// R[i][j] = log_cost(i, j) + logsumexp3(R[i-1][j], R[i][j-1], R[i-1][j-1])
// GAK(x, y) = exp(R[n][m])
// With band: skip (i,j) where |i - j| > band_width; set to f64::NEG_INFINITY
```

The optional Sakoe-Chiba band: cells outside the band are set to `f64::NEG_INFINITY` in log space (exp gives 0 contribution). `band_width = (band_frac * m.max(n)).ceil() as usize`. The `Option<f64>` parameter type matches `elastic_self_distance_matrix_with_band` / `elastic_cross_distance_matrix_with_band` exactly.

---

## Proposed Public API

All signatures follow fdars conventions: `Result<T, FdarError>`, `#[must_use]` on expensive computations, config struct for multi-parameter functions, column-major `FdMatrix`.

### `src/metric/gak.rs`

```rust
/// Configuration for Global Alignment Kernel computation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct GakConfig {
    /// Bandwidth (sigma > 0). Use gak_sigma_median() to auto-compute.
    pub sigma: f64,
    /// Sakoe-Chiba band as a fraction of series length (0 < f <= 1).
    /// None = full alignment. Some(f) = restrict to band ceil(f * m).
    pub band_frac: Option<f64>,
    /// Normalize to produce a proper kernel in [0, 1].
    /// k_norm(x,y) = gak(x,y) / sqrt(gak(x,x) * gak(y,y)).
    /// Default: true (normalized GAK is PSD; raw is not guaranteed PSD).
    pub normalize: bool,
}

impl Default for GakConfig { ... }  // sigma=1.0, band_frac=None, normalize=true

/// Median-heuristic sigma: median(pairwise_L2_distances) * sqrt(m).
/// Equivalent to tslearn's default bandwidth. O(n^2) — call once.
#[must_use]
pub fn gak_sigma_median(data: &FdMatrix, argvals: &[f64]) -> f64;

/// Scalar GAK value between two curves (public convenience wrapper).
#[must_use]
pub fn gak_distance(x: &[f64], y: &[f64], config: &GakConfig) -> f64;

/// Compute the n x n symmetric GAK Gram matrix (training-set self-kernel).
/// When config.normalize = true the result is a PSD kernel matrix K[i,j] in [0,1].
#[must_use = "expensive computation whose result should not be discarded"]
pub fn gak_gram_matrix(
    data: &FdMatrix,
    config: &GakConfig,
) -> Result<FdMatrix, FdarError>;

/// Compute the n_test x n_train cross-kernel matrix K(X_test, X_train).
/// Feed to an external precomputed-kernel SVM alongside the training Gram matrix.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn gak_cross_gram(
    test: &FdMatrix,
    train: &FdMatrix,
    config: &GakConfig,
) -> Result<FdMatrix, FdarError>;
```

### `src/kernel_kmeans.rs`

```rust
use crate::metric::gak::{GakConfig, gak_gram_matrix, gak_cross_gram};

/// Configuration for kernel k-means clustering via the GAK kernel.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct KernelKMeansConfig {
    /// Number of clusters (k >= 1).
    pub k: usize,
    /// Maximum assignment iterations per run (default: 100).
    pub max_iter: usize,
    /// Number of random restarts; best inertia run is returned (default: 5).
    pub n_init: usize,
    /// Random seed for initialization (default: 42).
    pub seed: u64,
    /// GAK kernel configuration.
    pub gak: GakConfig,
}

impl Default for KernelKMeansConfig { ... }

/// Result of kernel k-means clustering.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct KernelKMeansResult {
    /// Cluster label per observation (0-indexed, length n).
    pub labels: Vec<usize>,
    /// Within-cluster kernel objective value (lower is better).
    pub inertia: f64,
    /// Number of iterations in the winning run.
    pub n_iter: usize,
    /// Whether the winning run converged (labels stabilized).
    pub converged: bool,
    /// The GAK Gram matrix computed during fit (n x n).
    /// Stored for use by predict without recomputation.
    pub gram_train: FdMatrix,
    /// Configuration snapshot used during fit.
    pub config: KernelKMeansConfig,
}

impl KernelKMeansResult {
    /// Assign new curves to the nearest cluster in kernel space.
    /// Computes gak_cross_gram(new_data, training_data, &self.config.gak)
    /// internally; the training Gram matrix is stored in self.gram_train.
    pub fn predict(
        &self,
        new_data: &FdMatrix,
        train_data: &FdMatrix,
    ) -> Result<Vec<usize>, FdarError>;
}

/// Fit kernel k-means on a curve set using the GAK kernel.
///
/// Computes the n x n GAK Gram matrix once, then runs n_init random-partition
/// restarts of the kernel k-means assignment loop, returning the run with
/// lowest kernel inertia.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn kernel_kmeans(
    data: &FdMatrix,
    config: &KernelKMeansConfig,
) -> Result<KernelKMeansResult, FdarError>;
```

---

## Kernel K-Means Assignment Rule

Standard kernel k-means operates purely on the Gram matrix K (no explicit feature map needed):

```
For each observation i, and each cluster c with index set C_c:

objective(i, c) = K[i,i]
                - (2 / |C_c|) * sum_{j in C_c} K[i,j]
                + (1 / |C_c|^2) * sum_{j in C_c, l in C_c} K[j,l]

label[i] = argmin_c  objective(i, c)
```

The third term (cluster kernel mean) is computed once per cluster per iteration and cached. Per-observation, per-cluster assignment is O(n * k). No cluster center vectors are stored — the algorithm is purely index-based on K.

Initialization: random uniform partition across `n_init` restarts, seeded as `seed + restart_idx as u64`. The restart with lowest total inertia is returned. This does not reuse `kmeans_plusplus_init` from `clustering.rs` (that function operates on curve vectors with L2 distances, not on similarity-valued kernel matrices).

---

## Data Flow

```
curves: FdMatrix (n x m, column-major)
    |
    +-- gak_sigma_median(&data, &argvals)
    |       uses: distance::l2_distance_matrix -> median -> * sqrt(m)
    |       returns: f64 (suggested sigma for GakConfig)
    |
    +-- gak_gram_matrix(&data, &GakConfig)           [Gram export path]
    |       |
    |       +-- diagonal pass: gak_pair(xi, xi) for all i -> diag[i]
    |       +-- upper triangle: gak_pair(xi, xj) for i < j
    |       |       log-domain DP (2-row rolling buffer)
    |       |       logsumexp3 accumulation
    |       |       optional Sakoe-Chiba band
    |       |       -> exp(R[n][m]) = unnormalized GAK value
    |       +-- normalize: K[i,j] /= sqrt(diag[i] * diag[j])
    |       returns: FdMatrix (n x n, PSD Gram matrix)
    |
    +-- kernel_kmeans(&data, &KernelKMeansConfig)    [Clustering path]
            |
            +-- calls gak_gram_matrix internally
            +-- n_init restarts:
            |       random partition -> assignment loop -> inertia
            +-- returns best run as KernelKMeansResult
            |       .labels: Vec<usize>
            |       .gram_train: FdMatrix (stored)
            |
            +-- result.predict(&new_data, &train_data):
                    gak_cross_gram(new_data, train_data, config)
                    assignment by cross-kernel objective
                    returns: Vec<usize>

    [SVM export path — no fdars SVM, user-supplied]
    gak_gram_matrix(&train, &config)?        -> K_train (n_train x n_train)
    gak_cross_gram(&test, &train, &config)?  -> K_test  (n_test x n_train)
    User passes both to external SVM (scikit-learn, libsvm, etc.)
```

---

## Dependency-Ordered Build Sequence

Recommended implementation order within the phase (each step unblocks the next):

**Step 1 — GAK pairwise kernel core (`src/metric/gak.rs`, kernel only)**

Deliverables: `logsumexp3` (private), `gak_pair` (pub(crate) pairwise DP), `GakConfig` with defaults, `gak_distance` (public thin wrapper).

Tests: `gak_pair(x, x, ...)` > 0 for any x; normalized result in (0, 1]; monotone decrease as sigma decreases (tighter bandwidth = lower similarity for non-identical pairs); band vs unbanded agree on diagonals.

**Step 2 — Gram matrix builders and sigma heuristic (`src/metric/gak.rs`, matrix surface)**

Deliverables: `gak_sigma_median` (reuses `distance::l2_distance_matrix`), `gak_gram_matrix` (parallel via `self_distance_matrix` + diagonal cache), `gak_cross_gram` (parallel via `cross_distance_matrix`). Wire into `metric/mod.rs` and `lib.rs`.

Tests: symmetry of `gak_gram_matrix`; diagonal = 1.0 when `normalize = true`; positive-definiteness (attempt nalgebra Cholesky on a small n=5 result, expect Ok); `gak_cross_gram` shape (n_test x n_train).

**Step 3 — Kernel k-means (`src/kernel_kmeans.rs`)**

Deliverables: `KernelKMeansConfig` with defaults, `KernelKMeansResult`, `kernel_kmeans` fitting function (calls `gak_gram_matrix`, multi-restart assignment loop), `KernelKMeansResult::predict`.

Tests: two well-separated synthetic curve groups recovered with purity 1.0; `n_init > 1` returns reproducible labels (seeded); `predict` assigns a new curve to its correct group; `k > n` returns `FdarError::InvalidParameter`.

**Step 4 — Integration verification**

- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
- `cargo fmt --check`
- `cargo test --features linalg,parallel`
- Add a rustdoc example in `gak_gram_matrix` showing the SVM export handoff pattern.

---

## Architectural Constraints Respected

| Constraint | How GAK Respects It |
|------------|---------------------|
| Column-major `FdMatrix` | All inputs/outputs are `FdMatrix`; DP inner loop uses `data.row(i)` which calls `row_to_buf` |
| `Result<T, FdarError>` everywhere | `gak_gram_matrix`, `gak_cross_gram`, `kernel_kmeans`, `predict` all return `Result` |
| No new crate dependency | Only uses `rand` (already a dep) for seeded restart initialization; no new entries in `Cargo.toml` |
| Additive/non-breaking | Zero changes to existing public signatures; only `pub mod gak` and `pub mod kernel_kmeans` additions |
| MSRV 1.81 | No const generics, no `let ... else` patterns post-1.81, no stabilized features post-1.81 |
| Parallel feature gate | Gram-matrix parallel dispatch via `iter_maybe_parallel!`; degrades to sequential if `parallel` feature is off |
| `#[non_exhaustive]` on result types | All new result structs marked `#[non_exhaustive]` for forward compatibility |
| Deterministic seeding | Restarts seeded as `StdRng::seed_from_u64(config.seed + restart as u64)` — same pattern as `alignment/clustering.rs` |

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Accumulating in Linear (Non-Log) Space

**What people do:** Multiply local kernel values directly: `R[i][j] = cost(i,j) * (R[i-1][j] + R[i][j-1] + R[i-1][j-1])`.

**Why it's wrong:** For series of moderate length (m >= 50), local kernel values are < 1 and their product underflows to machine zero. The DP result becomes 0 for all pairs regardless of similarity.

**Do this instead:** Accumulate in log space throughout. Set `R[i][j] = log_cost(i,j) + logsumexp3(R[i-1][j], R[i][j-1], R[i-1][j-1])`. Call `exp` exactly once on `R[n][m]`. This mirrors `softmin3`'s `min_val` stabilization in `soft_dtw.rs`.

### Anti-Pattern 2: Computing Gram Matrix Without Caching Diagonal

**What people do:** Compute `K[i,j] / sqrt(gak_pair(xi, xi) * gak_pair(xj, xj))` inline during the off-diagonal fill, calling `gak_pair(xi, xi)` O(n^2/2) times.

**Why it's wrong:** `gak_pair(xi, xi)` is O(m) per call. Without caching, diagonal computation is O(n^2 * m) instead of O(n * m) — unnecessary n-fold slowdown.

**Do this instead:** Pre-compute `diag[i] = gak_pair(xi, xi)` in a single O(n) pass before filling the upper triangle. Use `K[i,j] /= sqrt(diag[i] * diag[j])` in the normalization step.

### Anti-Pattern 3: Applying k-means++ Initialization to Kernel K-Means

**What people do:** Pass the Gram matrix to `kmeans_plusplus_init` from `clustering.rs`, treating K[i,j] as a distance.

**Why it's wrong:** GAK values are similarities (higher = more similar). k-means++ uses D^2 weighting where D is a distance; inverting or negating K[i,j] to fake a distance produces pathological initialization that can seed all centers in the same cluster.

**Do this instead:** Random uniform partition restarts (`n_init` times). The kernel objective landscape is well-behaved enough that multiple random restarts consistently outperform a misapplied similarity-as-distance init.

### Anti-Pattern 4: Placing `kernel_kmeans` Inside `metric/`

**What people do:** Put `kernel_kmeans` in `src/metric/kernel_kmeans.rs` to keep all GAK-related code together.

**Why it's wrong:** `metric/` is for pairwise distance/similarity functions that return `FdMatrix`. A clustering algorithm with config, result type, and `predict` method does not belong there — it belongs at the same level as `clustering.rs` and `clustering_advanced.rs`.

**Do this instead:** `src/kernel_kmeans.rs` at the top level, registered with `pub mod kernel_kmeans;` in `lib.rs`.

### Anti-Pattern 5: Exposing an SVM Wrapper This Milestone

**What people do:** Add a thin `kernel_svm` entry point that takes the Gram matrix and a label vector.

**Why it's wrong:** Native kernel SVM is explicitly out of scope for v0.32.0. An SVM requires a QP solver, which would be a new heavy dependency. The correct boundary is: fdars produces the Gram and cross-Gram matrices; the user feeds them to an external library.

**Do this instead:** `gak_gram_matrix` + `gak_cross_gram` with a rustdoc example showing the handoff. Document the precomputed-kernel interface expected by popular SVM libraries.

---

## Sources

- Direct reading: `fdars-core/src/metric/soft_dtw.rs` — alignment-lattice DP, `softmin3` stabilization, 2-row rolling buffer, parallel dispatch pattern
- Direct reading: `fdars-core/src/metric/mod.rs` — submodule declaration pattern, `self_distance_matrix`, `cross_distance_matrix` helpers
- Direct reading: `fdars-core/src/alignment/clustering.rs` — `KMedoidsConfig`, `kmeans_pp_init`, convergence loop, band-fraction parameter type
- Direct reading: `fdars-core/src/clustering.rs` — `KmeansResult`, `kmeans_plusplus_init`, `KmeansResult::predict` pattern
- Direct reading: `fdars-core/src/clustering_advanced.rs` — config struct pattern, `pub mod` registration
- Direct reading: `fdars-core/src/lib.rs` — re-export conventions, `pub mod` declarations
- `.planning/research/GAP-BACKLOG.md` GAP-01 block — reuse targets: `metric/soft_dtw.rs`, `distance.rs`, existing clustering
- `.planning/PROJECT.md` v0.32.0 section — milestone scope, additive/non-breaking, no new crate dependency
- `.planning/codebase/ARCHITECTURE.md` — layered modular-monolith pattern, column-major `FdMatrix`, `Result` conventions
- Cuturi (2011), "Fast Global Alignment Kernels" (ICML) — algorithmic reference for log-domain DP and triangular kernel
- tslearn@0.9.0 `tslearn/metrics/softdtw_fast.pyx` — reference for log-space accumulation and sigma-median heuristic

---

*Architecture research for: fdars v0.32.0 GAK + kernel clustering integration*
*Researched: 2026-09-02*
