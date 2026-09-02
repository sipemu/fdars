# Stack Research: k-Shape Clustering & Shape-Based Distance (v0.34.0)

**Domain:** Rust functional-data-analysis library — adding SBD (Shape-Based Distance) + k-Shape clustering to fdars-core
**Researched:** 2026-09-02
**Confidence:** HIGH (codebase read = HIGH; tslearn source verified via GitHub API = MEDIUM cross-checked with web = MEDIUM-verified; algorithm paper cross-checked)

---

## Dependency Verdict: NO NEW CRATE DEPENDENCIES REQUIRED

All four v0.34.0 deliverables — SBD core, k-Shape fit, out-of-sample predict, and SBD-k-medoids — can be built entirely on the existing crate dependency set. This is the primary finding.

Explicit confirmation per feature:

| Feature | Required new dep? | Rationale |
|---------|------------------|-----------|
| FFT for NCC cross-correlation (SBD core) | NO | `rustfft` 6.2 is already a direct dependency; `FftPlanner::<f64>::new()` + `plan_fft_forward`/`plan_fft_inverse` already called in `src/fts/spectral.rs` and `src/seasonal/mod.rs` |
| Z-normalization of full series | NO | `shapelet::z_normalize_window(&[f64]) -> Vec<f64>` shipped in v0.33.0 — exact per-slice z-norm needed by SBD |
| Symmetric eigendecomposition for shape extraction | NO | `nalgebra::SymmetricEigen` already called in `src/fts/spectral.rs` (line 208: `nalgebra::SymmetricEigen::new(mat)`) — covers the `M = Q^T S Q` problem in shape extraction |
| k-Shape n_init restarts + empty-cluster handling | NO | Direct mirror of `kernel_kmeans.rs` pattern (v0.32.0): `StdRng::seed_from_u64(config.seed.wrapping_add(restart as u64))`, farthest-point empty-cluster recovery, lowest-inertia restart kept |
| Out-of-sample predict | NO | Same contract as `KernelKmeansResult::predict` (v0.32.0): assign new series to nearest centroid via SBD; no re-estimation |
| SBD-based k-medoids | NO | `alignment::clustering::kmedoids_from_distances(&dist_mat, &config)` already accepts any precomputed distance matrix — plug SBD distance matrix in directly |
| Pairwise distance matrix | NO | `src/distance.rs::pairwise_distance_matrix` + `cross_distance_matrix` reusable, or implement inline symmetric loop (upper-triangle fill) — same pattern used in `alignment/clustering.rs` |
| RNG seeding | NO | `rand` 0.8 + `StdRng::seed_from_u64` pattern established in `clustering.rs`, `kernel_kmeans.rs` |
| `linalg` / faer | NOT NEEDED | nalgebra 0.33 `SymmetricEigen` is sufficient for the sz×sz centroid matrix; no Cholesky/SVD path |
| MSRV change | NONE | All code is `f64` arithmetic + existing crate APIs; MSRV stays at 1.81 |

---

## Core Technologies

### Primary Technologies (all existing deps — no changes to Cargo.toml)

| Technology | Version in Cargo.toml | Role in v0.34.0 | Existing Usage Anchor |
|------------|----------------------|-----------------|----------------------|
| Rust (MSRV 1.81) | 1.81 min / 1.97 dev | All implementation | Entire codebase |
| rustfft | 6.2 | FFT forward/inverse for NCC computation in SBD | `src/fts/spectral.rs:42,142-143`; `src/seasonal/mod.rs:286-287,352-354` |
| nalgebra | 0.33 | `SymmetricEigen` for M=Q^T·S·Q shape-extraction centroid step | `src/fts/spectral.rs:208`; `src/regression.rs` (SVD) |
| rand | 0.8 | `StdRng::seed_from_u64(seed + restart)` for deterministic n_init restarts | `src/kernel_kmeans.rs:263`; `src/clustering.rs:584,773` |

### Supporting Existing Infrastructure

| Primitive | Location | v0.34.0 Role | Reuse Type |
|-----------|----------|-------------|------------|
| `z_normalize_window(&[f64]) -> Vec<f64>` | `src/shapelet/distance.rs` | Per-series z-normalization before SBD (the paper requires z-normalized input) | Direct call |
| `z_normalize_into(src, dst)` | `src/shapelet/distance.rs` | In-place variant for hot-loop centroid normalization | Direct call |
| `kmedoids_from_distances(dist_mat, config)` | `src/alignment/clustering.rs` | SBD-k-medoids: pass SBD distance matrix as a drop-in consumer | Direct call (no changes) |
| `KMedoidsConfig` / `KMedoidsResult` | `src/alignment/clustering.rs` | Config + result types for SBD-k-medoids path | Direct reuse |
| `FdMatrix` | `src/matrix.rs` | Input data container; all rows are curve observations | Existing type |
| `FdarError` | `src/error.rs` | `Result<T, FdarError>` error handling throughout | Existing type |
| `iter_maybe_parallel!` / `maybe_par_chunks_mut!` | `src/parallel.rs` | Gate rayon parallelism on pairwise SBD distance matrix computation | Existing macros |
| `seed_for_thread(seed, k)` | `src/helpers.rs` | Per-restart RNG (alternative to inline `seed_from_u64`) | Existing helper |

---

## Reference Baseline — Pinned Version & Exact API

### tslearn@0.9.0 — `KShape` (verified via GitHub API on tslearn-team/tslearn `main`)

**Class signature (stable, from tslearn 0.9.0 docs):**

```python
tslearn.clustering.KShape(
    n_clusters=3,      # number of clusters
    max_iter=100,      # maximum Lloyd iterations
    tol=1e-06,         # inertia variation convergence threshold
    n_init=1,          # restarts (NOTE: default is 1; fdars should use 10 like kernel_kmeans)
    verbose=False,
    random_state=None,
    init='random'      # 'random' or ndarray of shape (n_clusters, ts_size, d)
)
```

**fit(X, y=None)** — X shape `(n_ts, sz, d)`. Returns self with `.cluster_centers_`, `.labels_`, `.inertia_`, `.n_iter_`.

**predict(X)** — X shape `(n_ts, sz, d)`. Returns `labels` array shape `(n_ts,)`.

**Post-fit attributes:** `cluster_centers_` shape `(n_clusters, sz, d)`, `norms_` (training norms), `norms_centroids_` (centroid norms), `labels_`, `inertia_`, `n_iter_`.

**Module path:** `tslearn.clustering.KShape` — reference for the clustering API shape. Distance functions live in `tslearn.metrics`.

### tslearn@0.9.0 — `cdist_normalized_cc` (verified: source read from tslearn/metrics/cycc.py via GitHub API)

**Exact function signature:**

```python
cdist_normalized_cc(
    dataset1,        # shape (n_ts1, sz, d)
    dataset2,        # shape (n_ts2, sz, d)
    norms1,          # shape (n_ts1,) — precomputed L2 norms; negative = compute lazily
    norms2,          # shape (n_ts2,)
    self_similarity  # bool — if True, only compute upper triangle (symmetric case)
) -> dists           # shape (n_ts1, n_ts2) — max NCC values (NOT yet subtracted from 1)
```

**SBD distance** = `1.0 - cdist_normalized_cc(...)` (applied in `KShape._cross_dists`).

### tslearn@0.9.0 — `normalized_cc` (the SBD kernel, from tslearn/metrics/cycc.py)

This is the core NCC computation — the exact formula fdars must replicate:

```python
def normalized_cc(s1, s2, norm1=-1.0, norm2=-1.0):
    sz = s1.shape[0]
    n_bits = 1 + int(np.log2(2 * sz - 1))   # ceil(log2(2*sz - 1))
    fft_sz = 2 ** n_bits                      # next power of two >= 2*sz - 1

    denom = norm1 * norm2                      # precomputed L2 norms product
    if denom < 1e-9: denom = np.inf           # guard: zero-norm series

    cc = np.real(np.fft.ifft(
        np.fft.fft(s1, fft_sz, axis=0) *
        np.conj(np.fft.fft(s2, fft_sz, axis=0)),
        axis=0,
    ))                                         # length fft_sz, circular cross-corr
    cc = np.vstack((cc[-(sz-1):], cc[:sz]))   # rearrange to linear lags: 2*sz-1 entries
    norm_cc = cc.sum(axis=-1) / denom         # scalar (1D) or summed over d dims
    return norm_cc                             # shape (2*sz - 1,); SBD uses max()
```

**Rust translation notes:**

1. `fft_sz = (2 * sz - 1).next_power_of_two()` — exact Rust equivalent; rustfft handles arbitrary sizes but power-of-two is fastest.
2. `FftPlanner::<f64>::new()` + `plan_fft_forward(fft_sz)` and `plan_fft_inverse(fft_sz)` — identical to `seasonal/mod.rs` lines 352–354.
3. The `conj` multiply in frequency domain = pointwise `Complex { re: a.re*b.re + a.im*b.im, im: a.im*b.re - a.re*b.im }` (conjugate of s2 times s1).
4. After IFFT, rearrange: take last `sz-1` elements followed by first `sz` elements of the `fft_sz`-length buffer — this gives the 2*sz-1 linear cross-correlation lags.
5. Divide by `norm1 * norm2` (precomputed scalar norms of z-normalized series).
6. **SBD** = `1.0 - norm_cc.iter().copied().fold(f64::NEG_INFINITY, f64::max)`.
7. **Optimal shift** = `argmax(norm_cc) + 1 - sz` (converts 0-based index to lag in `[-(sz-1), sz-1]`).

**fdars naming convention:** `sbd(s1: &[f64], s2: &[f64]) -> f64` for the distance; `ncc(s1: &[f64], s2: &[f64]) -> (f64, i64)` for `(max_ncc, optimal_shift)`.

### tslearn@0.9.0 — `y_shifted_sbd_vec` (centroid alignment step, from tslearn/metrics/cycc.py)

Aligns each cluster member to the current centroid before shape extraction:

```python
# For each series in cluster: compute NCC, find argmax shift, circularly-shift the series
idx = np.argmax(cc)
shift = idx + 1 - sz    # lag: negative = shift left, positive = shift right
if shift > 0:   dataset_shifted[i, shift:] = dataset[i, :-shift, :]
elif shift < 0: dataset_shifted[i, :shift] = dataset[i, -shift:, :]
else:           dataset_shifted[i] = dataset[i]
```

**Rust translation:** shift a `Vec<f64>` slice in-place; truncation-fill (not wrap-around). Produces a new aligned copy per series in the cluster.

### tslearn@0.9.0 — `KShape._shape_extraction` (centroid update, from tslearn/clustering/kshape.py)

```python
def _shape_extraction(self, X, k):
    Xp = y_shifted_sbd_vec(centroid_k, cluster_members, ...)  # SBD-align
    S = Xp.T @ Xp                          # sz × sz, symmetric positive semidefinite
    Q = I_sz - ones(sz,sz) / sz            # centering matrix, symmetric
    M = Q.T @ S @ Q                        # sz × sz, symmetric
    _, vec = numpy.linalg.eigh(M)          # ascending eigenvalues; vec[:,-1] = top eigenvector
    mu_k = vec[:, -1]                      # largest eigenvector
    # Sign disambiguation: pick sign of mu_k that minimizes sum of SBD distances
    dist_plus  = sum(norm(Xp - mu_k))
    dist_minus = sum(norm(Xp + mu_k))
    if dist_minus < dist_plus: mu_k *= -1
    # After all clusters done: re-z-normalize centroids
```

**Eigendecomposition assessment:** M is symmetric (Q is symmetric idempotent, S is symmetric PSD → M = Q S Q is symmetric). `numpy.linalg.eigh` is the symmetric eigensolver. **`nalgebra::SymmetricEigen::new(mat)`** is the exact Rust equivalent — already proven in `src/fts/spectral.rs` line 208 with the same pattern (build DMatrix, call SymmetricEigen, sort by eigenvalue descending). Use identical pattern: build `DMatrix<f64>` from M (sz×sz), call `SymmetricEigen::new`, pick the eigenvector with the largest eigenvalue.

**Sign disambiguation:** compute sum of L2 distances of aligned cluster members to `+mu_k` vs `-mu_k`; pick the sign with lower total distance. Uses `FdMatrix::row_l2_sq` or inline L2 loop — no dep.

---

## Existing Primitives: Reuse Map

### Already Present in fdars-core (HIGH confidence — direct codebase read)

| Primitive | Location | v0.34.0 Reuse |
|-----------|----------|--------------|
| `FftPlanner::<f64>::new()` + `plan_fft_forward(n)` / `plan_fft_inverse(n)` | `src/fts/spectral.rs:42,143,308` | SBD NCC core — identical API call |
| `(2*n).next_power_of_two()` pattern | `src/seasonal/mod.rs:350` | SBD FFT padding: `(2 * sz - 1).next_power_of_two()` |
| `rustfft::num_complex::Complex<f64>` + buffer pattern | `src/fts/spectral.rs:150` | FFT buffer type for NCC |
| `nalgebra::SymmetricEigen::new(mat)` | `src/fts/spectral.rs:208` | Shape-extraction eigenproblem (M = Q S Q) |
| `DMatrix::from_fn(m, m, ...)` | `src/fts/spectral.rs:198` | Build M for shape extraction |
| Eigenvector sign-alignment pattern | `src/fts/spectral.rs:224-236` | Sign disambiguation of mu_k (existing pattern; adapt for SBD-based sign test instead of largest-magnitude-entry rule) |
| `z_normalize_window(slice: &[f64]) -> Vec<f64>` | `src/shapelet/distance.rs:114` | Pre-normalize series before SBD; z-normalize centroids after each centroid update |
| `z_normalize_into(src, dst)` | `src/shapelet/distance.rs:57` | In-place variant for normalizing centroid buffers |
| `kmedoids_from_distances(dist_mat, config)` | `src/alignment/clustering.rs` | SBD-k-medoids: pass `sbd_distance_matrix(data)` output directly |
| `KMedoidsConfig` / `KMedoidsResult` | `src/alignment/clustering.rs` | Config + result types for SBD-k-medoids — no new types needed |
| `StdRng::seed_from_u64(config.seed.wrapping_add(restart as u64))` | `src/kernel_kmeans.rs:263` | n_init restart seeding — copy verbatim |
| Farthest-point empty-cluster recovery | `src/kernel_kmeans.rs:368,466` | Empty-cluster guard — copy the `ensure_no_empty_random` pattern adapted for SBD distances |
| `iter_maybe_parallel!` | `src/parallel.rs` | Parallelize pairwise SBD distance matrix computation |
| `FdMatrix::row_l2_sq` | `src/matrix.rs` | L2 norm for sign disambiguation in shape extraction |
| `pub fn pairwise_distance_matrix` | `src/distance.rs` | Pairwise SBD distance matrix for `kmedoids` consumer |

### To Be Written for v0.34.0

| Primitive | Proposed Location | Notes |
|-----------|-------------------|-------|
| `sbd(s1: &[f64], s2: &[f64], fft_buf: &mut Vec<Complex<f64>>, planner: &mut FftPlanner<f64>) -> f64` | `src/kshape/sbd.rs` | Core SBD scalar; reuses caller-provided FFT planner + buffer (avoid re-planning per pair) |
| `ncc_with_shift(s1, s2, ...) -> (f64, i64)` | `src/kshape/sbd.rs` | Returns `(max_ncc, optimal_shift)` — needed for the alignment step in shape extraction |
| `sbd_distance_matrix(data: &FdMatrix) -> Vec<f64>` | `src/kshape/sbd.rs` | Symmetric upper-triangle pairwise matrix; gated via `iter_maybe_parallel!` |
| `sbd_cross_distance_matrix(test, centroids) -> Vec<f64>` | `src/kshape/sbd.rs` | Cross-distance for assignment step and predict |
| `shift_to_centroid(series_row, shift) -> Vec<f64>` | `src/kshape/sbd.rs` | Truncation-fill alignment; mirrors `y_shifted_sbd_vec` per-series step |
| `shape_extraction(cluster_members_aligned: &FdMatrix, sz: usize) -> Vec<f64>` | `src/kshape/centroid.rs` | Build M=Q^T S Q, call `nalgebra::SymmetricEigen`, sign-disambiguate |
| `KShapeConfig` | `src/kshape/mod.rs` | `n_clusters`, `n_init` (default 10, not tslearn's 1), `max_iter`, `tol`, `seed` |
| `KShapeResult` struct + `predict(&self, data: &FdMatrix) -> Result<Vec<usize>>` | `src/kshape/mod.rs` | Carries `cluster_centers: FdMatrix`, `labels`, `inertia`, `iter`, `converged`, `n_init_best` |
| `kshape_fd(data: &FdMatrix, config: &KShapeConfig) -> Result<KShapeResult, FdarError>` | `src/kshape/mod.rs` | Top-level entry point |

---

## FFT-Size and Zero-Padding Specifics

**Requirement:** Linear cross-correlation of two length-sz sequences requires a buffer of length `≥ 2*sz - 1`. Circular FFT convolution produces length-N results; to get linear cross-correlation, pad to `N ≥ 2*sz - 1` and discard the wrap-around.

**Power-of-two padding (required for efficiency):**

```rust
let fft_sz = (2 * sz - 1).next_power_of_two();
// Example: sz=100 → 2*100-1=199 → next_power_of_two=256
// Example: sz=500 → 2*500-1=999 → next_power_of_two=1024
// Example: sz=1000 → 2*1000-1=1999 → next_power_of_two=2048
```

`rustfft::FftPlanner` accepts arbitrary sizes (not restricted to power-of-two), but power-of-two is significantly faster due to the Cooley-Tukey radix-2 factorization. This is the same pattern already in `src/seasonal/mod.rs:350`.

**Buffer layout for NCC rearrangement:**

After IFFT into a length-`fft_sz` buffer (indices `0..fft_sz`):
- The linear cross-correlation lags are at indices `0..sz` (non-negative lags) and `fft_sz-(sz-1)..fft_sz` (negative lags).
- Rearrangement to `[-(sz-1), ..., -1, 0, 1, ..., sz-1]` order:
  ```rust
  // lags[0..sz-1] = buf[fft_sz-(sz-1)..fft_sz]   (negative lags)
  // lags[sz-1..2*sz-1] = buf[0..sz]               (non-negative lags)
  ```
- `max_ncc = lags.iter().copied().fold(f64::NEG_INFINITY, f64::max)`
- `argmax_idx` → `optimal_shift = argmax_idx as i64 + 1 - sz as i64` (lag in `[-(sz-1), sz-1]`)

**Norm precomputation:** Compute `||s||_2` once per series at fit time (before the inner-loop pairwise computation). Store in a `Vec<f64>` of length n. For z-normalized inputs, `norm = sqrt(sz)` exactly (since z-norm has unit variance), but compute explicitly to match tslearn's robustness convention.

**IFFT scaling:** `rustfft` does not normalize the IFFT output — divide by `fft_sz` after IFFT to get the correct cross-correlation values (standard FFT convention; numpy's `ifft` normalizes by default). The normalization by `norm1 * norm2` absorbs the scale, but the `fft_sz` factor must be divided out explicitly.

---

## Eigendecomposition Specifics for Shape Extraction

**Matrix M** is `sz × sz` where `sz` is the length of each z-normalized series. For typical time series (`sz` in 50–500):

| sz | M size | nalgebra::SymmetricEigen cost |
|----|--------|-------------------------------|
| 50 | 50×50 | Negligible (<1 ms) |
| 100 | 100×100 | ~1–5 ms |
| 200 | 200×200 | ~5–20 ms |
| 500 | 500×500 | ~100–500 ms — consider power method if slow |

**`nalgebra::SymmetricEigen` correctness:** M = Q^T S Q where Q = I - 11^T/n_k (centering) and S = X_p^T X_p (gram of aligned cluster members). Both Q and S are symmetric (Q is idempotent-symmetric, S is PSD), so M is symmetric. `SymmetricEigen` computes all eigenvalues and eigenvectors. Take the eigenvector with the largest eigenvalue (sort descending, take index 0 — identical to `spectral.rs` lines 211–221).

**Sign disambiguation** differs from the spectral.rs convention (which uses "largest-magnitude entry positive"). For k-Shape, use the tslearn convention: compare `sum_i ||x_p_i - mu_k||_2` vs `sum_i ||x_p_i + mu_k||_2`; negate mu_k if the minus version is smaller. This is mathematically cleaner and matches the reference.

**No `linalg` feature needed:** `nalgebra::SymmetricEigen` is in nalgebra 0.33 core (not behind a faer gate). The shape-extraction path works under default features (MSRV 1.81).

---

## Module Placement

```
src/kshape/
    mod.rs          — KShapeConfig, KShapeResult, kshape_fd() entry point, pub re-exports
    sbd.rs          — sbd(), ncc_with_shift(), sbd_distance_matrix(), sbd_cross_distance_matrix(), shift_to_centroid()
    centroid.rs     — shape_extraction() (the Q^T S Q eigenvector step)
```

Crate-root re-export in `src/lib.rs`:
```rust
pub mod kshape;
pub use kshape::{kshape_fd, KShapeConfig, KShapeResult};
```

Follows the pattern of `pub mod kernel_kmeans` (v0.32.0) and `pub mod shapelet` (v0.33.0).

---

## Core Technologies Table

| Technology | Version | Role in v0.34.0 | Change to Cargo.toml? |
|------------|---------|-----------------|----------------------|
| Rust (MSRV 1.81) | 1.81 min / 1.97 dev | All implementation | None |
| rustfft | 6.2 | FFT forward/inverse for NCC in SBD; `FftPlanner::<f64>` + power-of-two padding | None — already a dep |
| nalgebra | 0.33 | `SymmetricEigen::new(DMatrix)` for shape extraction centroid step | None — already a dep |
| rand | 0.8 | `StdRng::seed_from_u64(seed + restart)` for n_init determinism | None — already a dep |
| rayon | 1.10 (optional, `parallel` feature) | Parallelize pairwise SBD distance matrix via `iter_maybe_parallel!` | None — already a dep |
| shapelet | existing, v0.33.0 | `z_normalize_window` + `z_normalize_into` for series pre-normalization | None — already in crate |
| alignment::clustering | existing | `kmedoids_from_distances` for SBD-k-medoids consumer | None — already in crate |
| kernel_kmeans | existing, v0.32.0 | n_init/restart/empty-cluster pattern to mirror exactly | None — already in crate |

### No New Dependencies

```toml
# Cargo.toml — NO CHANGES NEEDED for v0.34.0
# All k-Shape + SBD deliverables build on the existing dependency set.
# rustfft, nalgebra, rand, rayon are already declared.
```

---

## Alternatives Considered

| Decision | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| FFT for NCC | `rustfft::FftPlanner` (already a dep) | `realfft` crate (real-input FFT) | New dep not justified; `rustfft` handles complex inputs and is already used in 3 modules |
| Eigendecomposition | `nalgebra::SymmetricEigen` | Power iteration (hand-coded) | Only need top-1 eigenvector, but `SymmetricEigen` is already proven in the codebase; power iteration adds untested code for modest gain at typical sz ≤ 200 |
| Eigendecomposition (large sz) | `nalgebra::SymmetricEigen` | `faer` thin SVD (via `linalg` feature) | `SymmetricEigen` is correct and sufficient for sz ≤ 500; faer path would require `linalg` feature gate, complicating MSRV to 1.84 |
| n_init default | 10 (matching `kernel_kmeans.rs`) | 1 (tslearn's default) | k-Shape is as sensitive to local minima as kernel-k-means; 10 restarts matches fdars convention and improves robustness significantly |
| Z-normalization | `shapelet::z_normalize_window` (already in crate) | Inline re-implementation | Avoiding code duplication; v0.33.0 already shipped exactly this helper |
| SBD-k-medoids | Plug SBD distance matrix into `kmedoids_from_distances` | Separate KMedoidsKShape struct | `kmedoids_from_distances` explicitly designed to accept any distance matrix — zero new code needed for this deliverable |
| Module name | `kshape/` | `clustering/kshape.rs` | The module has 3 logical sub-concerns (SBD math, centroid, clustering); a sub-directory matches the `shapelet/`, `metric/`, `alignment/` pattern |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Arbitrary FFT size (non-power-of-two) for NCC | rustfft is correct for arbitrary sizes but significantly slower for non-power-of-two; pairwise SBD on large datasets compounds the cost | `(2 * sz - 1).next_power_of_two()` — same pattern already in `seasonal/mod.rs:350` |
| `nalgebra::SVD` for shape extraction | SVD of M is overkill; M is symmetric so `SymmetricEigen` is the correct and cheaper path | `nalgebra::SymmetricEigen::new(m_matrix)` |
| `faer` for shape extraction | Would require `linalg` feature + MSRV 1.84; no measurable benefit for sz×sz matrices in the clustering hot path | `nalgebra::SymmetricEigen` (default features, MSRV 1.81) |
| z-norm inside the SBD inner loop | If z-normalizing series on every pairwise call, you pay O(n²·sz) normalization; pre-normalize once at fit time | Pre-compute `z_normalize_window` for each series row once; cache norms |
| `tslearn.preprocessing.TimeSeriesScalerMeanVariance` idiom applied per-centroid-update | tslearn z-normalizes centroids after each centroid update as a post-processing step; skipping this causes centroid drift and instability | Re-apply `z_normalize_window` to each centroid after each `shape_extraction` call — one line |
| Adding a new `ndarray` or `faer` dependency | No 2D numerical array op in SBD or shape extraction requires a new matrix library; `nalgebra::DMatrix` + `Vec<f64>` cover it | Existing nalgebra DMatrix for the sz×sz M matrix; `Vec<f64>` for series buffers |

---

## Feature-Flag Considerations

| Aspect | Recommendation |
|--------|---------------|
| Parallel pairwise SBD | YES — gate with `parallel` feature via `iter_maybe_parallel!` on the upper-triangle loop. Each `sbd(s_i, s_j)` call is independent. |
| Does k-Shape path need `linalg`? | NO — `nalgebra::SymmetricEigen` is in nalgebra core, not behind faer gate. Works under default features. |
| WASM compatibility | YES — pure `f64` arithmetic + rustfft (already WASM-compatible per existing uses); rayon is optional and off on WASM. |
| `serde` feature | Follow `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` on `KShapeConfig` and `KShapeResult`; the `cluster_centers_` field (`FdMatrix`) already derives serde conditionally. |

---

## MSRV and Feature Compatibility Matrix

| Scenario | Works? | Notes |
|----------|--------|-------|
| Default features (`parallel`) | YES | pairwise SBD parallelized; MSRV 1.81 |
| No features (sequential) | YES | All loops sequential-compatible |
| `linalg` feature | YES | k-Shape adds nothing to `linalg`; features orthogonal |
| `serde` feature | YES | Add derive attributes to KShapeConfig / KShapeResult |
| WASM (`js` feature) | YES | rustfft is WASM-compatible; rayon off on WASM |
| Rust 1.81 (MSRV) | YES | No post-1.81 stabilizations needed; `SymmetricEigen` stable since nalgebra 0.30+ |

---

## Sources

- tslearn/metrics/cycc.py — `normalized_cc`, `cdist_normalized_cc`, `y_shifted_sbd_vec` exact source — fetched via GitHub API (gh api repos/tslearn-team/tslearn/contents/tslearn/metrics/cycc.py) — HIGH confidence (authoritative source, direct code read)
- tslearn/clustering/kshape.py — `KShape` class: `__init__`, `_shape_extraction`, `_update_centroids`, `_assign`, `_fit_one_init`, `fit`, `predict` — fetched via GitHub API — HIGH confidence (authoritative source, direct code read)
- tslearn 0.9.0 stable docs — `KShape` class signature and parameter descriptions — https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.KShape.html — MEDIUM confidence (verified via WebFetch)
- Paparrizos & Gravano (2015) "k-Shape: Efficient and Accurate Clustering of Time Series" SIGMOD 2015 — abstract + algorithm structure confirmed via web search results + SIGMOD record PDF (abstract) — MEDIUM confidence (algorithm structure confirmed; NCC/SBD/Rayleigh-quotient/eigenvector verified; exact matrix formulas confirmed against tslearn source)
- fdars-core/src/fts/spectral.rs — confirmed `rustfft::FftPlanner` + `nalgebra::SymmetricEigen` usage idiom, FFT planning API, eigenvector sign-alignment pattern — HIGH confidence (direct codebase read)
- fdars-core/src/seasonal/mod.rs:350 — confirmed `(2*n).next_power_of_two()` FFT-padding pattern for cross-correlation — HIGH confidence (direct codebase read)
- fdars-core/src/shapelet/distance.rs — confirmed `z_normalize_window` and `z_normalize_into` APIs (v0.33.0) — HIGH confidence (direct codebase read)
- fdars-core/src/kernel_kmeans.rs — confirmed n_init restart seeding pattern, empty-cluster recovery idiom, predict method structure — HIGH confidence (direct codebase read)
- fdars-core/src/alignment/clustering.rs — confirmed `kmedoids_from_distances(dist_mat, config)` API + `KMedoidsConfig` / `KMedoidsResult` types — HIGH confidence (direct codebase read)

---

*Stack research for: v0.34.0 k-Shape Clustering & Shape-Based Distance in fdars-core*
*Researched: 2026-09-02*
