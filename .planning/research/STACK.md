# Stack Research: GAK + Kernel-K-Means + Gram-Matrix Export (v0.32.0)

**Domain:** Rust functional-data-analysis library — adding Global Alignment Kernel, kernel-k-means, and Gram-matrix export to fdars-core
**Researched:** 2026-09-02
**Confidence:** MEDIUM (tslearn API verified against stable docs; GAK algorithm complexity confirmed via Cuturi 2011 via web; kernel k-means centering trick confirmed via multiple sources)

---

## Dependency Verdict: NO NEW CRATE DEPENDENCIES REQUIRED

All three v0.32.0 deliverables — GAK kernel, kernel-k-means, and Gram-matrix export — can be built entirely on the existing stack. This is the most important finding of this research.

Explicit confirmation per feature:

| Feature | Required new dep? | Rationale |
|---------|------------------|-----------|
| GAK DP recursion (log-domain) | NO | Pure scalar arithmetic (`f64::exp`, `f64::ln`); same pattern already used in `metric/soft_dtw.rs` `softmin3` |
| GAK Gram matrix (n×n) | NO | `FdMatrix` (existing column-major matrix) is the output type |
| GAK parallelism (row pairs) | NO | `iter_maybe_parallel!` macro from `parallel.rs` gates `rayon` (already a dep) |
| Sigma heuristic (`sigma_gak` equivalent) | NO | Requires median-of-sampled-pair-distances; only `rand` (existing dep) + plain `f64` needed |
| Kernel-k-means | NO | Works purely on the Gram matrix via the centering trick — scalar arithmetic + `FdMatrix` |
| Gram-matrix export | NO | Return `FdMatrix` directly; caller feeds it to their SVM. No SVM code in fdars |
| FFT | NOT NEEDED | GAK is a pure O(n·m) DP — FFT is irrelevant (see FFT section below) |
| `linalg` / faer | NOT NEEDED | No SVD, Cholesky, or matrix factorization in the GAK/kernel-k-means path |
| MSRV change | NONE | All code is plain `f64` arithmetic; MSRV stays at 1.81 |

---

## Reference Baseline: tslearn@0.9.0

**Version:** tslearn **0.9.0**, released July 2, 2026. This is the version pinned in GAP-BACKLOG.md GAP-01 and was the current stable release at the time of the v0.31.0 audit (2026-09-02).

**Confirmed stable docs URL:** https://tslearn.readthedocs.io/en/stable/ (shows 0.8.1 PDF; 0.9.0 is the PyPI stable release as of July 2026)

**Important caveat on log-domain:** The tslearn changelog notes that a log-space normalization fix for GAK (preventing NaN on series longer than ~405 samples) is slated for v0.10.0, not yet in 0.9.0. The fdars implementation MUST implement log-domain from the start to avoid this class of bug — this is a known deficiency in the 0.9.0 reference that fdars should surpass.

### Exact tslearn 0.9.0 Public API to Match

#### `gak(s1, s2, sigma=1.0, be=None) -> float`

- `s1`, `s2`: time series, shape `(sz, d)` or `(sz,)`. In fdars: one curve = one `FdMatrix` row (slice `&[f64]`).
- `sigma`: Gaussian kernel bandwidth. Default `1.0`. This is the only tuning parameter.
- `be`: backend (NumPy/PyTorch). Not applicable to Rust.
- Returns: normalized float in `[0, 1]`. Normalized means `gak(x,x) == 1.0` for all `x`.
- Formula: `gak(x,y) = k(x,y) / sqrt(k(x,x) * k(y,y))` where `k` is the unnormalized GA kernel.

**fdars counterpart:** `gak(x: &[f64], y: &[f64], sigma: f64) -> f64`

#### `cdist_gak(dataset1, dataset2=None, sigma=1.0, n_jobs=None, verbose=0, be=None) -> ndarray(n1, n2)`

- `dataset1`: shape `(n_ts1, sz1, d)` or `(n_ts1, sz1)`. In fdars: `&FdMatrix` (n1 rows).
- `dataset2`: optional second dataset; if `None`, computes self-similarity (symmetric n×n). In fdars: `Option<&FdMatrix>`.
- `sigma`: kernel bandwidth.
- `n_jobs`: parallelism. In fdars: controlled by `parallel` feature / `rayon`.
- Returns: `(n1, n2)` float array — the Gram matrix. In fdars: `FdMatrix`.

**fdars counterparts:**
- `gak_gram(data: &FdMatrix, sigma: f64) -> FdMatrix` — self Gram (symmetric n×n)
- `gak_cross_gram(data1: &FdMatrix, data2: &FdMatrix, sigma: f64) -> FdMatrix` — cross Gram (n1×n2)

#### `sigma_gak(dataset, n_samples=100, random_state=None, be=None) -> float`

- Computes the suggested GAK bandwidth via Cuturi 2011 heuristic: median of pairwise distances from `n_samples` randomly drawn curve pairs.
- `n_samples`: how many random pairs to sample. Default `100`.
- `random_state`: seed. In fdars: `seed: u64` following the existing convention.
- Returns: `f64` sigma value.

**fdars counterpart:** `sigma_gak(data: &FdMatrix, n_samples: usize, seed: u64) -> Result<f64, FdarError>`

#### `KernelKMeans(n_clusters=3, kernel='gak', max_iter=50, tol=1e-6, n_init=1, kernel_params=None, n_jobs=None, verbose=0, random_state=None)`

- `n_clusters`: number of clusters.
- `kernel`: `'gak'` (primary use case) or any sklearn metric name. In fdars: only GAK is in scope for v0.32.0.
- `max_iter`: iteration cap. Default `50`.
- `tol`: inertia-change convergence threshold. Default `1e-6`.
- `n_init`: number of random restarts. Default `1`.
- `kernel_params`: dict; for GAK: `{'sigma': 1.0}` or `{'sigma': 'auto'}`. In fdars: `sigma: f64` as a plain parameter (auto = call `sigma_gak`).
- `random_state`: seed.
- Methods: `fit(X)`, `predict(X)`, `fit_predict(X)`.
- Attributes after fit: `labels_`, `inertia_`, `n_iter_`.

**fdars counterpart:** `kernel_kmeans_gak(data: &FdMatrix, n_clusters: usize, sigma: f64, max_iter: usize, tol: f64, n_init: usize, seed: u64) -> Result<KernelKmeansResult, FdarError>` with `KernelKmeansResult { labels, inertia, n_iter, converged }`.

---

## The GAK Algorithm — Technical Details

### Is FFT Required?

**No. GAK is a pure O(n·m) dynamic programming recursion.** FFT is used in other kernel-like algorithms (k-Shape uses SBD via FFT cross-correlation — GAP-03), but GAK has nothing to do with frequency domain computation. `rustfft` (already a dependency) is irrelevant to this milestone.

### The Recursion

The unnormalized Global Alignment Kernel `k(x, y)` where `x` has length `n` and `y` has length `m`:

```
k(x, y) = sum over all monotone alignments pi of: prod_{(i,j) in pi} kappa(x_i, y_j)
```

where `kappa(a, b) = exp(-||a - b||^2 / (2 * sigma^2))` is the Gaussian kernel.

This sum is computed via the DP table `M[i][j]` (1-indexed, `M[0][*] = M[*][0] = 0`):

```
M[i][j] = kappa(x_i, y_j) * (M[i-1][j-1] + M[i-1][j] + M[i][j-1])
```

The unnormalized kernel value is `M[n][m]`.

**Triangular variant (TGAK):** Cuturi 2011 introduces an optional triangular constraint — only cells where `|i - j| <= triangular_param` are computed. This reduces computation and prevents alignment of very distant positions (a reasonable inductive bias for time-series kernels). The triangular constraint does not change the PSD property.

### Log-Domain Implementation (MANDATORY)

The raw DP as written above multiplies exponentials. For series of length ~405+ (or smaller series with many near-zero kernel values), floating-point underflow drives `M[n][m]` to 0.0, giving `gak(x,y) = 0/0 = NaN`. This is the bug tslearn 0.9.0 carries and 0.10.0 fixes.

The log-domain recursion computes `log M[i][j]` instead:

```
log_M[i][j] = log_kappa(x_i, y_j) + log_sum_exp(log_M[i-1][j-1], log_M[i-1][j], log_M[i][j-1])
```

where `log_kappa(a, b) = -||a - b||^2 / (2 * sigma^2)` and `log_sum_exp` uses the standard max-subtraction trick for numerical stability.

The unnormalized log-kernel is `log_M[n][m]`. Normalization in log-domain: `log_gak(x,y) = log_M_xy[n][m] - 0.5 * (log_M_xx[n][n] + log_M_yy[m][m])`. The final `gak(x,y) = exp(log_gak(x,y))`.

**This is structurally identical to the existing `softmin3` log-sum-exp in `metric/soft_dtw.rs`.** The pattern is already in the codebase; GAK uses it for `log_sum_exp` of three terms (matching soft-DTW's structure) rather than softmin.

### Memory Layout

The 2-row rolling-buffer optimization applies to the GAK DP, exactly as it does for soft-DTW in `soft_dtw_distance`. For the unnormalized computation, only the current and previous rows need to be kept in memory: `O(min(n, m))` space per pair.

The self-kernel `k(x, x)` must be computed for every curve (for normalization), and the cross-kernel `k(x, y)` for every pair. For an n-curve dataset, this is `n + n*(n-1)/2` DP calls for the symmetric self-Gram matrix — the same upper-triangle loop pattern already used in `metric/mod.rs:self_distance_matrix`.

---

## Existing Module Reuse Map

| v0.32.0 Need | Reuse From | Reuse Type |
|---|---|---|
| DP row-rolling buffer pattern | `metric/soft_dtw.rs: soft_dtw_distance` | Direct pattern copy; same 2-row rolling buffer |
| Log-sum-exp of 3 terms | `metric/soft_dtw.rs: softmin3` | Adapt to log-sum-exp (not softmin) |
| Upper-triangle parallel loop for Gram | `metric/mod.rs: self_distance_matrix` | Direct reuse |
| Cross Gram parallel loop | `metric/mod.rs: cross_distance_matrix` | Direct reuse |
| `iter_maybe_parallel!` / rayon gating | `parallel.rs` | Reuse unchanged |
| Random pair sampling (sigma heuristic) | `rand` dep already present; existing `StdRng::seed_from_u64(seed)` pattern | Direct reuse |
| Result type struct pattern | Any existing `*Result` type | Follow `#[derive(Debug, Clone, PartialEq)] #[non_exhaustive]` convention |
| Kernel-k-means centering trick (scalar `f64` arithmetic on Gram) | No existing analogue — new algorithm | Built on `FdMatrix` arithmetic |
| `FdMatrix` as Gram-matrix output type | `matrix.rs: FdMatrix` | Direct — n×n `FdMatrix` is the Gram matrix |
| Module placement: GAK | `metric/` directory | New `metric/gak.rs` alongside `soft_dtw.rs` |
| Module placement: kernel k-means | `clustering.rs` or new `clustering_kernel.rs` | Add alongside existing `kmeans_fd` |
| Crate-root re-export | `lib.rs` / `prelude.rs` | Follow existing `pub use metric::soft_dtw::*` pattern |

---

## Kernel-K-Means Algorithm: What Is Needed

Kernel k-means operates entirely on the precomputed Gram matrix `K` (n×n). No explicit feature-space coordinates are needed. The algorithm:

1. Pre-compute `K = gak_gram(data, sigma)` — one call, O(n^2 * m).
2. Initialize cluster assignments randomly (use `rand`, existing dep).
3. At each iteration, assign each point `i` to cluster `c` minimizing:
   `K[i,i] - (2/|C_c|) * sum_{j in C_c} K[i,j] + (1/|C_c|^2) * sum_{j,k in C_c} K[j,k]`
4. The third term (cluster kernel average) is precomputed once per cluster per iteration as a scalar: `sum_{j,k in C_c} K[j,k] / |C_c|^2`.
5. Repeat until assignments stabilize or `max_iter` reached.
6. Inertia = sum over all points of the minimum kernel-distance above.

Total complexity: `O(n^2 * m)` to build Gram + `O(n^2 * k * max_iter)` to cluster. No matrix factorization, no SVD, no Cholesky — purely scalar `f64` operations on the Gram matrix entries. **`linalg` / faer is not needed.**

---

## Gram-Matrix Export: What "External-SVM Glue" Means

The Gram-matrix export deliverable is just `gak_gram` returning an `FdMatrix`. That is the complete interface. The caller:

1. Calls `gak_gram(&data, sigma)` → receives an n×n `FdMatrix`.
2. Extracts it (e.g. via the existing column-major buffer, or a future `to_vec2d()` helper) and passes it to their SVM library (libsvm, linfa-svm, an external Python call, etc.).

**No SVM code ships in fdars.** No additional serialization or crate is needed for this deliverable — the existing `FdMatrix` type (with its `data` field or row/column accessors) is sufficient. If the caller wants `serde` JSON export, that is already behind the `serde` feature flag.

---

## Recommended Stack (Unchanged from Existing)

### Core Technologies Used by v0.32.0 Features

| Technology | Version in Cargo.toml | Role in v0.32.0 | Change? |
|------------|----------------------|-----------------|---------|
| Rust (MSRV 1.81) | 1.81 min / 1.97 dev | All implementation | None |
| nalgebra | 0.33 | Not used by GAK path | None |
| rayon | 1.10 (optional, `parallel` feature) | Parallel Gram-matrix row-pair loop | None (reuse `iter_maybe_parallel!`) |
| rand | 0.8 | `sigma_gak` random pair sampling | None |
| rustfft | 6.2 | **Not used** by GAK (pure DP, no FFT) | None |
| faer | 0.23 (`linalg` feature) | **Not used** by GAK/kernel-k-means | None |
| FdMatrix | existing `matrix.rs` | Gram matrix output type | None |

### No New Dependencies

```toml
# Cargo.toml — NO CHANGES NEEDED for v0.32.0
# All three deliverables build on the existing dependency set.
```

---

## Feature-Flag Considerations

| Aspect | Recommendation |
|--------|---------------|
| GAK Gram computation parallel? | YES — gate with `parallel` feature via `iter_maybe_parallel!` on the upper-triangle pairs loop. Same pattern as `soft_dtw_div_self_1d`. |
| Does GAK need `linalg`? | NO — no matrix factorization in the GAK/kernel-k-means path. Both work under the default feature set (no flags). |
| WASM compatibility? | YES — pure `f64` arithmetic, no `rayon` on WASM (rayon is optional), no faer. `gak_gram` will work on WASM. |
| Does the `serde` feature apply? | The result structs should follow the `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` convention, but no new serde work is needed. |

---

## Alternatives Considered

| Decision | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| GAK in raw domain vs log domain | Log domain | Raw product domain | Raw domain silently produces NaN for series length >~405 (confirmed tslearn 0.9.0 bug). Log domain is mandatory for correctness. |
| GAK triangular constraint | Implement as optional `triangular: Option<usize>` | Always use triangular or always skip | Optional gives callers the exact tslearn behavior; `None` = exact, `Some(t)` = constrained. Adds no complexity. |
| Gram matrix output type | `FdMatrix` (existing) | `Vec<Vec<f64>>` or new matrix type | `FdMatrix` is the project's matrix type; re-using it avoids a new abstraction and lets callers use existing row/column accessors. |
| Kernel-k-means sigma | Explicit `f64` parameter | Auto-compute always | Explicit gives the caller full control; `sigma_gak` as a separate function matches tslearn's pattern where sigma is computed separately. |
| New crate for SVM glue | None | linfa-svm, smartcore | Gram-matrix export is just `FdMatrix` return. fdars explicitly does not ship an SVM (per PROJECT.md scope). No dep needed. |
| GAK module location | `metric/gak.rs` (new submodule of `metric/`) | Standalone `gak.rs` at crate root | Consistent with `metric/soft_dtw.rs` and `metric/dtw.rs`; GAK is a kernel/metric, not a top-level domain. |
| Kernel k-means location | `clustering_kernel.rs` or extend `clustering.rs` | `metric/` (wrong abstraction) | Kernel k-means is a clustering algorithm; placing it alongside `clustering.rs` keeps domain cohesion. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Raw-domain GAK DP | Produces NaN for series longer than ~405 samples (float underflow in product of small exponentials) | Log-domain DP with log-sum-exp trick |
| Materializing the full (n+1)×(m+1) DP table in memory | O(n*m) memory per pair; wastes memory when normalization is the goal | 2-row rolling buffer for the forward pass (same as `soft_dtw_distance`) |
| Adding an SVM crate | Out of scope for v0.32.0 (PROJECT.md explicit: "fdars ships NO SVM") | Gram-matrix export only — callers bring their own SVM |
| Adding a wavelet/FFT backend for GAK | GAK has no FFT step; `rustfft` is irrelevant here | Pure DP |
| faer or nalgebra for kernel k-means | The algorithm needs no matrix decomposition — only scalar accumulations over Gram entries | Plain `f64` arithmetic on `FdMatrix` entries |

---

## MSRV and Feature Compatibility Matrix

| Scenario | Works? | Notes |
|----------|--------|-------|
| Default features (`parallel`) | YES | `iter_maybe_parallel!` parallelizes Gram computation; MSRV 1.81 |
| No features (sequential) | YES | All DP is sequential-compatible |
| `linalg` feature | YES | GAK adds nothing to `linalg`; features are orthogonal |
| `serde` feature | YES | Add derive attributes to result structs |
| WASM (`js` feature) | YES | Pure `f64` arithmetic; rayon is off on WASM |
| Rust 1.81 (MSRV) | YES | No const generics, async, or post-1.81 stabilizations needed |

---

## Sources

- tslearn stable docs (gak, cdist_gak, sigma_gak, KernelKMeans) — https://tslearn.readthedocs.io/en/stable/ — MEDIUM confidence (official docs, version 0.8.1 stable PDF; 0.9.0 confirmed via PyPI)
- Cuturi 2011 ICML paper "Fast Global Alignment Kernels" — confirmed O(n·m) DP via multiple secondary sources (R dtwclust::GAK docs, tslearn kernel guide) — MEDIUM confidence
- tslearn changelog (CHANGELOG.md on GitHub) — confirmed log-space NaN fix is toward-v0.10.0, not in 0.9.0 — MEDIUM confidence
- R dtwclust::GAK documentation (rdrr.io) — confirmed triangular constraint, log-domain, sigma-median heuristic — LOW confidence (secondary)
- fdars-core/src/metric/soft_dtw.rs — confirmed existing 2-row rolling buffer, softmin3 log-sum-exp pattern reuse — HIGH confidence (direct codebase read)
- fdars-core/Cargo.toml — confirmed all deps already present, no new deps needed — HIGH confidence (direct codebase read)

---

*Stack research for: v0.32.0 Global Alignment Kernel + kernel-k-means + Gram-matrix export in fdars-core*
*Researched: 2026-09-02*
