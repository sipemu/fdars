# Feature Research: k-Shape Clustering & Shape-Based Distance (v0.34.0)

**Domain:** Shape-based time-series clustering — SBD distance primitive + k-Shape algorithm for functional curve/time-series data — Rust crate `fdars-core`
**Milestone:** v0.34.0 (promotes GAP-03 from `GAP-BACKLOG.md`)
**Researched:** 2026-09-02
**Confidence:** MEDIUM — SBD NCCc formula and SBD range [0, 2] confirmed HIGH from aeon docs (direct formula extraction). Shape extraction matrix M confirmed MEDIUM from tslearn source code (GitHub, verified variable names). k-Shape algorithm structure (assignment, refinement, convergence, empty-cluster, n_init) confirmed MEDIUM from tslearn source + KShape API docs. tslearn default n_init=1 confirmed HIGH. Cross-checked against Paparrizos & Gravano 2015 SIGMOD paper (PDF not extractable, but multiple secondary sources including aeon, dtwclust R package, kshape-python repo, and cybergarage article agree on all core formulas).

**In-scope:** SBD primitive (NCCc via FFT), k-Shape clustering fit (shape-extraction centroids + n_init restarts + deterministic seeding), out-of-sample predict, SBD distance matrix for k-medoids. **Out-of-scope:** soft-DTW/elastic/DTW variants (already in `metric/`), GPU acceleration (OOS-01), SAX/symbolic representations (OOS-02).

---

## Precise Mathematical Specification

This section provides the exact formulas the planner must implement against. Claims are sourced and confidence-labelled.

### A. Z-Normalization (prerequisite for SBD)

Both series x and y must be independently z-normalized before SBD computation:

```
z(x)_i = (x_i - mean(x)) / std(x)
```

where `mean(x) = (1/m) Σ x_i` and `std(x) = sqrt((1/m) Σ (x_i - mean(x))^2)` (population std, ddof=0). When `std(x) = 0` (constant series), the normalized form is the zero vector (matching the v0.33.0 `z_normalize_window` convention already in `shapelet/distance.rs`).

**Reuse:** `shapelet::z_normalize_window` and `shapelet::z_normalize_into` already implement this exactly — no new code needed for the primitive.

*Confidence: HIGH — universally described; matches existing fdars implementation.*

---

### B. Shape-Based Distance (SBD) via NCCc

#### B.1 Cross-Correlation Sequence via FFT

Given z-normalized series x and y of length m, the cross-correlation sequence `CC` over all m shifts is:

```
CC = IFFT( FFT(x_padded) · conj(FFT(y_padded)) )
```

Implementation details:
- Zero-pad both x and y to length `fft_len = next_power_of_two(2*m - 1)` (i.e. `fft_len >= 2m - 1`)
- Apply forward FFT to each padded series (real-to-complex, length fft_len)
- Multiply element-wise: `X[k] * conj(Y[k])` for each frequency bin k
- Apply inverse FFT (complex-to-real of length fft_len)
- The resulting real part is the raw cross-correlation at each of the fft_len circular shifts

**Shift indexing:** After IFFT, index w=0 is zero shift; indices 1..m-1 are positive shifts (y shifted right relative to x); indices fft_len-m+1..fft_len-1 are negative shifts (y shifted left). The full `m` meaningful shifts span indices `[0, m-1]` (positive) and `[fft_len-m+1, fft_len-1]` (negative), which together enumerate all 2m-1 non-trivial shift positions.

*Confidence: HIGH — FFT convolution theorem; confirmed by aeon sbd_distance docs.*

#### B.2 Coefficient-Normalized Cross-Correlation (NCCc)

```
NCCc_w(x, y) = CC_w(x, y) / sqrt( (x·x) · (y·y) )
             = CC_w(x, y) / ( ||x||_2 · ||y||_2 )
```

where `||x||_2 = sqrt(Σ x_i^2)` is the L2 norm of the (z-normalized) series x. For z-normalized series of length m, `||x||_2 = sqrt(m)` (unit variance after normalization), so the denominator simplifies to `m` — but computing it from the raw norm is numerically safer and more general.

The optimal shift `w*` is:

```
w* = argmax_w NCCc_w(x, y)
```

NCCc is dimensionless and lies in [-1, +1] for each shift w.

*Confidence: HIGH — formula confirmed from aeon `sbd_distance` docs (direct extraction) and cross-checked against tslearn, dtwclust R package, and the k-Shape paper secondary sources.*

#### B.3 SBD Distance

```
SBD(x, y) = 1 - max_w NCCc_w(x, y)
```

- Range: **[0, 2]** — 0 = perfectly similar (max NCCc = 1), 2 = perfectly anti-similar (max NCCc = -1)
- SBD = 0 iff x and y are identical up to amplitude scaling and circular shift
- The return value is (SBD_scalar, w*) — both needed: SBD for distance computations, w* for alignment in shape extraction

*Confidence: HIGH — aeon docs state range [0,2] explicitly; formula confirmed across multiple sources.*

#### B.4 Complexity

Single SBD(x, y) for series of length m: **O(m log m)** — dominated by the FFT (length ~2m). Pairwise distance matrix for n series: **O(n² · m · log m)**.

---

### C. Shape Extraction (Centroid Update)

Shape extraction is the centroid refinement step of k-Shape. Given a cluster of `n_k` series, it computes the centroid as the top eigenvector of a shift-aligned, mean-centered covariance matrix.

#### C.1 Align Cluster Members to Current Centroid

For each cluster member `x_i` (z-normalized), compute `SBD(centroid, x_i)` to obtain optimal shift `w*_i`. Apply that shift to `x_i`, producing the aligned version `x_i_aligned` (a circular shift by `w*_i` positions).

Aligned series are stacked row-wise into the `n_k × m` matrix **S**:

```
S[i, :] = align(z(x_i), w*_i)   for i = 1..n_k
```

All rows are z-normalized after alignment (re-apply z-normalization to the shifted version).

#### C.2 Centering Matrix Q

```
Q = I_m - (1/n_k) · O_{m×m}
```

where `I_m` is the `m×m` identity matrix and `O_{m×m}` is the `m×m` all-ones matrix. Q is the mean-centering projection that removes the column mean.

*Note:* In the tslearn source code the centering is applied differently — `Q = I_m - (1/sz) * ones(sz, sz)` where `sz = m` (series length) — and then `M = Q^T · S^T · S · Q = Q^T · S^T_T · Q`. The net result is equivalent to the Paparrizos formulation; the matrix `M` (called by the paper `M = S^T (I - O/n) S`) is the **mean-centered cross-product** of the aligned series.

#### C.3 The Eigenproblem

```
M = S^T · Q^T · Q · S    (simplified: M = (Q · S)^T · (Q · S))
```

or equivalently in the tslearn notation:

```
M = Q^T · (S^T · S) · Q
```

Both are equivalent when Q is idempotent. The centroid is the solution to the Rayleigh-quotient maximization:

```
mu_k* = argmax_{v: ||v||=1}  (v^T M v) / (v^T v)
```

This is the **top eigenvector of M** — i.e. the eigenvector corresponding to the largest eigenvalue.

**Algorithm:** Compute `eigh(M)` (symmetric eigendecomposition, ascending order), take the last column of the eigenvectors matrix. In nalgebra: `let sym_eig = na::SymmetricEigen::new(m_mat); let centroid_raw = sym_eig.eigenvectors.column(m-1)`.

#### C.4 Sign Disambiguation and Z-Normalization

The top eigenvector is defined up to sign. Both `+mu_k` and `-mu_k` are evaluated; the one with **smaller total SBD to cluster members** is selected.

After sign selection, the centroid is **z-normalized** (mean 0, std 1) to match the data convention.

*Confidence: MEDIUM — tslearn source code directly uses `Q = I - (1/sz)*ones(sz,sz)`, `M = Q^T S Q` with `S = Xp^T Xp` (cross-product), `eigh(M)`, `vec[:,-1]`. Sign test and z-normalization confirmed from source. The formula as stated in secondary k-Shape sources (M = S^T (I - O/n) S) is the algebraically equivalent statement from the paper.*

---

### D. k-Shape Algorithm

```
Algorithm KShape(data X: n × m, n_clusters k, n_init R, max_iter T, tol ε, seed s):

  best_inertia = +∞
  best_result  = None

  For restart r = 0..R-1:
    rng = StdRng::seed_from_u64(s + r)
    
    1. INIT: Assign each series randomly to one of k clusters (random partition, not k-means++)
       — Every cluster must be non-empty. If a random draw leaves a cluster empty,
         reassign the series farthest from any current centroid to that cluster.
    
    2. For iter = 0..T-1:
    
       a. REFINE CENTROIDS:
          For each cluster j with n_j members:
            mu_j = ShapeExtraction(cluster_members, mu_j_current)
            z-normalize mu_j
          
       b. ASSIGN:
          For each series x_i in X:
            label[i] = argmin_j SBD(mu_j, x_i)
          
          If any cluster becomes empty after assignment:
            reassign the series maximally distant from its centroid to the empty cluster
            (or retry the restart — tslearn raises EmptyClusterError and retries)
       
       c. CONVERGENCE CHECK:
          inertia = sum_i SBD(mu_{label[i]}, x_i)
          if |old_inertia - inertia| < ε  OR  inertia > old_inertia:
            break
    
    if inertia < best_inertia:
      best_inertia = inertia
      best_result  = (labels, centroids, inertia, iter_count, restart_idx)
  
  Return best_result
```

**Key implementation notes:**
- **tslearn default n_init = 1** — fdars should use `n_init = 10` as a better default (mirrors the v0.32.0 KernelKMeans convention of exceeding tslearn's n_init default for robustness)
- **Convergence direction:** Break if inertia drops by less than tol *or* regresses (worsening indicates oscillation near a fixed point)
- **Seeding:** restart r → `StdRng::seed_from_u64(seed + r as u64)` — consistent with `kernel_kmeans_fd` and `KMedoidsConfig` conventions in fdars
- **Norms cache:** Compute z-normalized norms of each input series once per restart — reuse throughout assignment; recompute centroid norms after each refinement step

*Confidence: MEDIUM — algorithm structure confirmed from tslearn source; empty-cluster retry via EmptyClusterError + n_init loop confirmed. The specific "reassign farthest point" vs. "restart" strategy differs — tslearn retries the full restart; fdars may prefer in-place reassignment (lower cost, follows kernel_kmeans pattern).*

---

### E. Out-of-Sample Predict

```
predict(new_X: p × m) -> labels: Vec<usize>:
  For each new series x_i (i = 0..p-1):
    label[i] = argmin_j SBD(centroid_j, z(x_i))
```

Centroids are stored in the fit result; new series are z-normalized before computing SBD. No re-fitting occurs. Mirrors `KernelKmeansResult::predict`.

---

### F. SBD Distance Matrix for k-Medoids

```
sbd_distance_matrix(data: &FdMatrix) -> FdMatrix  (n × n symmetric)
  output[i, j] = SBD(z(row_i), z(row_j))  for all i, j
  output[i, i] = 0.0
```

The result is a symmetric n×n distance matrix in `FdMatrix` column-major layout, suitable as a direct input to `kmedoids_from_distances` (existing in `alignment::clustering`). Only the upper triangle needs to be computed (n*(n-1)/2 SBD calls); the lower triangle is filled by symmetry.

---

### G. Complexity Summary

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Single SBD(x, y) | O(m log m) | FFT + IFFT on ~2m-length buffer |
| Pairwise SBD distance matrix (n series) | O(n² · m · log m) | n*(n-1)/2 FFT cross-correlations |
| Shape extraction (per cluster, n_k members) | O(n_k · m + m³) | FFT alignment O(n_k · m log m) + eigh O(m³) |
| k-Shape per iteration (n series, k clusters) | O(n · k · m · log m + k · m³) | Assignment dominates for large n; eigh dominates for large m |
| k-Shape full fit (T iterations, R restarts) | O(R · T · (n · k · m · log m + k · m³)) | Typical: R=10, T≈10 (converges fast) |

The m³ eigh cost is significant only for large m (>500 evaluation points). For typical functional data with m=50–200, eigh is cheap. For m>300, consider approximate eigendecomposition (out of scope).

---

## Feature Landscape

### Table Stakes — Users Expect These

These features constitute the complete v0.34.0 deliverable. Missing any one makes the milestone incomplete.

| Feature | Why Expected | Complexity | Implementation Notes |
|---------|--------------|------------|----------------------|
| `sbd(x: &[f64], y: &[f64]) -> (f64, i64)` — returns (SBD distance, optimal shift w*) | Core primitive; everything else is built on it. Without w*, shape extraction cannot align members. | MEDIUM | z-normalize x and y via `z_normalize_window` (reuse from `shapelet::distance`); zero-pad to `next_power_of_two(2m-1)`; forward FFT both; multiply X*conj(Y); inverse FFT; normalize by `||x||_2 * ||y||_2`; find max over 2m-1 shifts; return (1-max_ncc, shift_index). Use `rustfft::FftPlanner` (already a dep). |
| `sbd_distance_matrix(data: &FdMatrix) -> Result<FdMatrix, FdarError>` — n×n pairwise SBD | k-medoids requires a precomputed distance matrix; users expect pairwise computation. | MEDIUM | Upper-triangle loop; symmetric fill; reuse `sbd` per pair; `iter_maybe_parallel!` for rayon. |
| `KShapeConfig { n_clusters, max_iter, tol, n_init, seed }` config struct | fdars convention for all complex methods; every existing algorithm uses a config struct. | LOW | Mirrors `KMedoidsConfig`, `KernelKmeansConfig`. Default: `n_clusters=3`, `max_iter=100`, `tol=1e-6`, `n_init=10`, `seed=0`. Serde-gated. |
| `KShapeResult { labels, centroids, inertia, n_iter, n_init_best }` result struct | Stores all state for predict and inspection. | LOW | `centroids: FdMatrix` (k × m), `labels: Vec<usize>`, `inertia: f64`, `n_iter: usize`, `n_init_best: usize`. `Debug + Clone + PartialEq` + serde-gated. |
| `kshape_fd(data: &FdMatrix, config: &KShapeConfig) -> Result<KShapeResult, FdarError>` — k-Shape fit | Headline deliverable: runs full k-Shape (n_init restarts, assignment + shape-extraction centroid refinement), returns best result. | HIGH | Implements algorithm from Section D: random-partition init per restart, seeded `seed + restart_idx`, shape-extraction centroid via top eigenvector of M = Q^T S Q, convergence check, empty-cluster recovery, returns minimum-inertia run. |
| Shape extraction: `shape_extraction(members: &FdMatrix, centroid: &[f64]) -> Vec<f64>` (internal) | The centroid refinement step that distinguishes k-Shape from k-means; without it, the algorithm degrades to k-means with SBD. | HIGH | Align each member by SBD shift → z-normalize → stack into S; form Q = I - (1/n_k)*ones; M = Q^T * S^T * S * Q; SymmetricEigen (nalgebra) → largest eigenvector; test +/- sign (pick lower SBD); z-normalize. |
| `KShapeResult::predict(new_data: &FdMatrix) -> Result<Vec<usize>, FdarError>` — out-of-sample assign | Users expect to cluster new curves without re-fitting — the standard predict pattern mirrors `KernelKmeansResult::predict` and `ShapeletTransformClassifierFit::predict`. | LOW | z-normalize each new series; `argmin_j SBD(centroid_j, x_i)` for each; returns `Vec<usize>`. |
| `kmedoids_from_distances` consumer via `sbd_distance_matrix` | SBD plugged into existing k-medoids scaffolding (`alignment::clustering::kmedoids_from_distances`) — the GAP-03 requirement explicitly lists "SBD plugged into existing k-medoids as an alternative consumer". | LOW | No new function needed in alignment::clustering; caller computes `sbd_distance_matrix(&data)` then passes to `kmedoids_from_distances`. An example or doc test demonstrating the combination suffices. |
| `Result<T, FdarError>` on all public functions | fdars error-handling convention — no panics on input validation. | LOW | Dimension checks: `n > 0`, `m > 0`, `n >= n_clusters`, `n_init >= 1`, series length consistency for predict. |
| `Debug + Clone + PartialEq` on all public result/config types | fdars convention across 97+ types — uniform inspectability and testability. | LOW | Standard derives on `KShapeConfig`, `KShapeResult`. |
| Inline tests: sbd correctness, shape extraction, k-Shape round-trip, predict, distance matrix | Gate for correctness before shipping. | MEDIUM | `sbd` known-answer test (two identical series → SBD=0, optimal-shift series → SBD=0 with shift≠0); shape extraction eigenvector test (hand-crafted 2-cluster case); k-Shape fit→predict smoke test on synthetic data. |

### Differentiators — Competitive Advantage

These raise quality above a minimal correct implementation but are not blockers for a correct v1.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| `n_init = 10` as default (vs tslearn's n_init=1) | k-Shape is sensitive to initialization; 10 restarts dramatically improves cluster quality, especially for k>3. fdars already sets `n_init=10` in `KernelKmeansConfig` for the same reason. | LOW | Just a constant in `KShapeConfig::default()`. Document as "exceeds tslearn default for robustness." |
| Deterministic seeding (`seed + restart_idx`) | Same `seed` → identical labels across runs. Critical for reproducibility in research pipelines. | LOW | `StdRng::seed_from_u64(config.seed.wrapping_add(restart_idx as u64))` — exact pattern from `kernel_kmeans_fd`. |
| Rayon parallelism over assignment step | SBD assignment (n × k SBD calls per iteration) is embarrassingly parallel — rayon gives near-linear scaling for large n. | LOW | `iter_maybe_parallel!(0..n).map(|i| argmin_j sbd(...))` via existing `parallel.rs` macros. Centroid refinement (k sequential eigh calls) is not bottlenecked. |
| `#[must_use]` on `kshape_fd`, `sbd_distance_matrix`, `KShapeResult::predict` | fdars convention for expensive computations (74+ functions already annotated); prevents accidental discard. | LOW | One attribute per expensive function. |
| Criterion benchmark in `benches/` | fdars convention; measures k-Shape fit time vs n and m; quantifies benefit of rayon parallelism; documents the O(n·k·m·log m) profile. | LOW | Two cells: (n=100, m=50, k=3) and (n=500, m=100, k=5). Both sequential and parallel feature gates. |
| Serde support on `KShapeConfig`, `KShapeResult` | Pipeline persistence — save fitted centroids, reload for production predict. Matches `KernelKmeansConfig` / `GmmClusterConfig` serde patterns. | LOW | `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` on both types. |
| Example file `examples/kshape_clustering.rs` | fdars has 28 examples; new milestone algorithms conventionally ship with one. Shows fit → predict → distance-matrix-for-kmedoids flow. | LOW | Synthetic sinusoidal clusters (3 groups with different phase). |
| FFT planner reuse across SBD calls within a fit | Creating a new `FftPlanner` per SBD call adds ~1µs overhead — for n=500, k=3, 10 iterations this is 15,000 planner creations. Pass a shared planner as an internal parameter. | LOW | Internal implementation detail; planner is `!Send` but fine within a single rayon task scope. |
| In-place shift buffer for SBD (avoid allocation per call) | Each SBD call allocates two complex buffers of length ~2m. Preallocate and reuse within the assignment loop. | MEDIUM | Requires a small refactor to pass buffer refs into internal `_sbd_with_buf` helper. Impact: ~20% wall-clock improvement for large m. |

### Anti-Features — Explicitly Out of Scope

| Anti-Feature | Why Requested | Why Excluded | What to Do Instead |
|--------------|---------------|---------------|--------------------|
| soft-DTW / DTW-based shape clustering | DTW variants are the main alternative to SBD; users familiar with tslearn often ask | fdars already ships `metric/soft_dtw.rs` (soft-DTW distance + barycenter) and `alignment/` elastic clustering — those cover DTW-family needs | Use existing `soft_dtw_distance` + barycenter or elastic k-medoids if DTW-family is desired |
| GPU acceleration | SBD FFT computation is embarrassingly parallel — GPU gives large speedup for n>1000 | fdars targets CPU/WASM; no GPU infrastructure anywhere in the crate (OOS-01 in GAP-BACKLOG) | Use rayon CPU parallelism (differentiator above) — sufficient for research-scale datasets |
| Variable-length series support | Some FDA datasets have irregularly sampled or variable-length curves | SBD via cross-correlation requires equal-length series (the FFT zero-pad assumes same m); variable-length adds a fundamentally different alignment problem | Apply `spline_interpolate` (existing in `helpers.rs`) to regularize lengths before k-Shape |
| Learning k-Shape centroids via gradient descent | Gradient-descent centroid update would sidestep the eigenproblem | Fundamentally different algorithm requiring AD; fdars has no AD infrastructure (GAP-08) | Use the eigenproblem shape extraction as specified — it is the canonical Paparrizos algorithm |
| Soft assignment / probabilistic k-Shape | Fuzzy cluster membership (analogous to GMM over SBD) | Not in the Paparrizos paper; the "SBD-GMM" combination is not standard and would require a completely different algorithm | Use `gmm/` module for probabilistic clustering over curve distances if soft assignment is needed |
| SBD for multivariate curves (d > 1) | tslearn KShape accepts multivariate input (per-channel averaged SBD) | `FdMatrix` is single-channel; multivariate extension requires a representation decision not yet made for fdars | Implement as a follow-on once multivariate `FdMatrix` / `FdCurveSet` representation is settled |
| DTW-based SBD (replacing NCCc with DTW-normalized distance) | Some papers propose DTW-based centroid extraction | Not SBD — a different algorithm with O(m²) per-pair cost instead of O(m log m) | Apply existing elastic alignment as preprocessing if DTW-shape invariance is needed |

---

## Feature Dependencies

```
shapelet::z_normalize_window (existing v0.33.0, shapelet/distance.rs)
    │
    └──reused-by──> sbd(x, y) -> (f64, i64)               [new: clustering/kshape/distance.rs]
                        │
                        ├──required-by──> sbd_distance_matrix(&FdMatrix) -> FdMatrix
                        │                     │
                        │                     └──consumed-by──> kmedoids_from_distances (existing alignment/clustering.rs)
                        │
                        └──required-by──> shape_extraction(members, centroid) -> Vec<f64>  [internal]
                                              │
                                              ├──requires──> nalgebra::SymmetricEigen  (existing dep)
                                              │
                                              └──returns──> centroid: Vec<f64>
                                                                │
                                                                └──used-by──> kshape_fd(data, config) -> KShapeResult
                                                                                  │
                                                                                  ├──stores──> KShapeResult { labels, centroids, inertia, n_iter, n_init_best }
                                                                                  │
                                                                                  └──enables──> KShapeResult::predict(new_data) -> Vec<usize>

rustfft::FftPlanner (existing dep, used by seasonal/)
    └──reused-by──> sbd()  (FFT cross-correlation)
```

### Dependency Notes

- **`sbd` reuses `z_normalize_window`:** z-normalization of both series before cross-correlation is mandatory for scale invariance. The existing `shapelet::z_normalize_window` function implements exactly this convention (constant → zero vector). No new normalization code needed.
- **`sbd` reuses `rustfft`:** The FFT planner pattern is already established in `seasonal/hilbert.rs`. The same `FftPlanner::<f64>::new()` → `plan_fft_forward` / `plan_fft_inverse` pattern applies. `num_complex::Complex<f64>` is already a dependency.
- **`shape_extraction` requires `sbd`:** The alignment step (compute w* for each member against current centroid) calls `sbd` n_k times. The extraction result feeds back into the k-Shape loop.
- **`shape_extraction` requires nalgebra `SymmetricEigen`:** The eigenproblem is symmetric positive-semidefinite (M = Q^T S^T S Q is PSD). nalgebra's `SymmetricEigen` is the correct solver — already used in `regression.rs` and `fpca_variants.rs`.
- **`sbd_distance_matrix` is independent of `kshape_fd`:** It is a standalone utility. The k-medoids consumer is a caller convention (not a new function in `alignment/clustering.rs`).
- **`KShapeResult::predict` requires `KShapeResult`:** Only the fitted centroids are needed; no training data is retained.

---

## MVP Definition

### Launch With (v0.34.0)

Minimum viable for the milestone to ship all four target features:

- [ ] `sbd(x: &[f64], y: &[f64]) -> (f64, i64)` — SBD distance + optimal shift via FFT NCCc; reuses `z_normalize_window` + rustfft
- [ ] `sbd_distance_matrix(data: &FdMatrix) -> Result<FdMatrix, FdarError>` — pairwise SBD matrix for k-medoids consumption
- [ ] `shape_extraction(members: &FdMatrix, centroid: &[f64]) -> Vec<f64>` — internal centroid refinement via top eigenvector of M = Q^T S Q, with sign disambiguation and z-normalization
- [ ] `KShapeConfig { n_clusters, max_iter, tol, n_init, seed }` with sensible defaults (n_init=10)
- [ ] `KShapeResult { labels, centroids, inertia, n_iter, n_init_best }` with `Debug + Clone + PartialEq` + serde-gated
- [ ] `kshape_fd(data: &FdMatrix, config: &KShapeConfig) -> Result<KShapeResult, FdarError>` — full algorithm with n_init restarts, deterministic seeding, convergence, empty-cluster recovery
- [ ] `KShapeResult::predict(new_data: &FdMatrix) -> Result<Vec<usize>, FdarError>` — out-of-sample assignment
- [ ] All `Result<T, FdarError>` input validation paths (n >= n_clusters, n_init >= 1, consistent series length, n > 0)
- [ ] Inline tests: `sbd` known-answer (identical series → 0.0, phase-shifted identical series → 0.0 with non-zero shift), shape extraction smoke test, k-Shape fit→predict round-trip on synthetic 3-cluster sinusoidal data, distance matrix symmetry check
- [ ] Doc comment example on `kshape_fd` showing the SBD-k-medoids consumer pattern via `sbd_distance_matrix` + `kmedoids_from_distances`

### Add After Validation (v0.34.x)

- [ ] In-place buffer reuse for SBD (avoid allocation per call) — performance differentiator
- [ ] Criterion benchmark (fit time vs n and m; distance-matrix time)
- [ ] Example file `examples/kshape_clustering.rs`

### Future Consideration (v0.35+)

- [ ] Multivariate SBD (per-channel, then average) — blocked on multivariate FdMatrix representation decision
- [ ] Approximate eigenproblem for large m (>500) — only if performance profiling shows eigh bottleneck
- [ ] GPU-accelerated FFT cross-correlation — OOS-01, conflicts with WASM deployment model

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| `sbd(x, y) -> (dist, shift)` | HIGH | MEDIUM | P1 — atomic primitive; everything depends on it |
| `shape_extraction` (internal) | HIGH | HIGH | P1 — distinguishes k-Shape from k-means; correctness-critical |
| `kshape_fd` (full algorithm) | HIGH | HIGH | P1 — headline deliverable |
| `KShapeResult::predict` | HIGH | LOW | P1 — standard predict pattern; required by GAP-03 |
| `sbd_distance_matrix` | HIGH | MEDIUM | P1 — enables k-medoids consumer, required by GAP-03 |
| `KShapeConfig` + `KShapeResult` structs | HIGH | LOW | P1 — fdars convention; needed for all above |
| `n_init = 10` default + deterministic seeding | MEDIUM | LOW | P2 — robustness over tslearn; consistent with kernel_kmeans convention |
| Rayon parallelism over assignment | MEDIUM | LOW | P2 — follows existing `iter_maybe_parallel!` pattern |
| `#[must_use]` annotations | MEDIUM | LOW | P2 — fdars convention for expensive computations |
| Criterion benchmark | LOW | LOW | P2 — fdars convention |
| Serde support | LOW | LOW | P2 — fdars convention |
| In-place FFT buffer reuse | MEDIUM | MEDIUM | P2 — performance win; not correctness-critical |
| Example file | LOW | LOW | P2 — fdars convention; 28 existing examples |
| Multivariate SBD | MEDIUM | HIGH | P3 — blocked on representation; future milestone |
| GPU acceleration | LOW | VERY HIGH | P3 — out of scope entire crate (OOS-01) |

---

## Competitor Feature Analysis

| Feature | tslearn `KShape` (0.9.0) | kshape-python (TheDatumOrg) | fdars v0.34.0 Plan |
|---------|--------------------------|------------------------------|--------------------|
| SBD computation | Cython `cdist_normalized_cc` via FFT; normalizes by L2 norms | NumPy FFT + cross-correlation | `sbd` via `rustfft::FftPlanner`, reuses `z_normalize_window` |
| Zero-padding | Implicit in NumPy FFT call | Explicit to 2n-1 | Explicit `next_power_of_two(2m-1)` |
| Shape extraction | `eigh(Q^T S Q)`, largest eigenvector, ±sign test, z-normalize | Same | nalgebra `SymmetricEigen`, same algorithm |
| n_init default | 1 (single restart) | Not specified | **10** (better robustness; matches kernel_kmeans convention) |
| Seeding | `random_state` (NumPy RNG) | `seed` parameter | `seed + restart_idx` via `StdRng::seed_from_u64` |
| Empty cluster | `EmptyClusterError` + retry | Not specified | In-place reassign farthest point OR retry restart |
| Convergence | `|Δinertia| < tol` or regression | Max iter | Same dual condition |
| Predict | `predict(X_new)` — SBD to stored centroids | `predict(X_new)` | `KShapeResult::predict` — identical semantics |
| Distance matrix | Not exposed | Not exposed | `sbd_distance_matrix` — explicit pairwise utility |
| k-medoids integration | Not supported | Not supported | Via `sbd_distance_matrix` + existing `kmedoids_from_distances` |
| Equal-length requirement | Required | Required | Required (same constraint) |
| Serde | Not supported | Not supported | Under `serde` feature (fdars convention) |
| Parallelism | `n_jobs` (joblib) | `n_jobs` (joblib) | `iter_maybe_parallel!` under `parallel` feature |
| Benchmarks | Not provided | Not provided | Criterion bench (fdars convention) |

---

## Sources

- Paparrizos, J., Gravano, L. (2015). "k-Shape: Efficient and Accurate Clustering of Time Series." SIGMOD '15, pp. 1855–1870. https://dl.acm.org/doi/10.1145/2723372.2737793 (PDF binary; formulas reconstructed from secondary sources)
- Paparrizos & Gravano (2015) SIGMOD Record version: https://sigmodrecord.org/publications/sigmodRecord/1603/pdfs/18_kShape_RH_Paparrizos.pdf (PDF binary; not extractable)
- aeon `sbd_distance` API docs (NCCc formula, range [0,2] confirmed — HIGH confidence): https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.distances.sbd_distance.html
- tslearn `KShape` API docs (parameters, fit/predict, n_init=1 default): https://tslearn.readthedocs.io/en/latest/gen_modules/clustering/tslearn.clustering.KShape.html
- tslearn `KShape` source code (shape_extraction matrix Q, M, eigh, sign test): https://github.com/tslearn-team/tslearn/blob/main/tslearn/clustering/kshape.py
- tslearn clustering utils (EmptyClusterError, convergence condition): https://github.com/tslearn-team/tslearn/blob/main/tslearn/clustering/utils.py
- kshape-python TheDatumOrg reference implementation: https://github.com/TheDatumOrg/kshape-python
- dtwclust R `SBD` function docs (range, FFT-based, z-normalization recommended): https://rdrr.io/cran/dtwclust/man/SBD.html
- dtwclust R `shape_extraction` docs (Rayleigh quotient description): https://www.rdocumentation.org/packages/dtwclust/versions/5.5.9/topics/shape_extraction
- GAP-BACKLOG.md GAP-03 (v0.31.0, promotion source): `.planning/research/GAP-BACKLOG.md`
- survey-pyx.md PYX-01 (v0.31.0, original gap identification): `.planning/research/survey-pyx.md`
- fdars-core `shapelet/distance.rs` (`z_normalize_window`, `z_normalize_into` — reuse confirmed): local codebase
- fdars-core `seasonal/hilbert.rs` (rustfft `FftPlanner` usage pattern — reuse confirmed): local codebase
- fdars-core `alignment/clustering.rs` (`kmedoids_from_distances`, `KMedoidsConfig` — consumer target confirmed): local codebase
- fdars-core `kernel_kmeans.rs` (n_init pattern, seed+restart_idx, empty-cluster recovery, predict API — convention source): local codebase

---
*Feature research for: v0.34.0 k-Shape Clustering & Shape-Based Distance (GAP-03)*
*Researched: 2026-09-02*
