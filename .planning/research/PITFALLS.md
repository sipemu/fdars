# Pitfalls Research

**Domain:** GAK kernel + kernel-k-means + Gram-matrix export — Rust numerical FDA library (fdars-core v0.32.0)
**Researched:** 2026-09-02
**Confidence:** HIGH

---

## Critical Pitfalls

### Pitfall 1: GAK Raw Recursion Underflows to Zero — Log-Domain Is Mandatory

**What goes wrong:**
The GAK forward recursion accumulates products of `exp(-cost)` terms along every alignment path. For a pair of time series of length m=100, the path length is roughly 2m steps, each contributing a factor `exp(-d²/σ²)`. For moderate σ and any real-valued series, the product of 200 such factors underflows `f64` to 0.0. The kernel then returns 0.0 for all pairs, the Gram matrix is the zero matrix, and kernel-k-means silently assigns all curves to cluster 0.

**Why it happens:**
The direct recursion mirrors the soft-DTW recursion in `metric/soft_dtw.rs` — which sums costs (works fine) — but the GAK recursion multiplies probabilities (catastrophic underflow). Developers porting from soft-DTW to GAK copy the structure and forget that GAK is a sum of exponentials over paths, not a sum of costs.

The triangular GAK from Cuturi (2011) defines:
```
k_GA(x, y) = Σ_{π ∈ A(m,m)} exp( -Σ_{(i,j)∈π} (x_i - y_j)²/σ² )
```
The inner exponential is always < 1, so the sum-of-products underflows unless computed in log domain.

**How to avoid:**
Implement the forward pass entirely in log-space. Define `log_R[i][j]` where `R[i][j] = log(sum of all path probabilities reaching (i,j))`. The recursion becomes:
```
log_R[i][j] = log_cost(i,j) + log_sum_exp(log_R[i-1][j], log_R[i][j-1], log_R[i-1][j-1])
```
where `log_cost(i,j) = -(x[i]-y[j])² / σ²` and `log_sum_exp` uses the standard max-shift trick. The final kernel value is `exp(log_R[n][m])`. This mirrors what `softmin3` does in `metric/soft_dtw.rs` — the pattern is already in the codebase, apply it to the addition-in-log-space case.

**Warning signs:**
- `gak(x, y)` returns exactly `0.0` for any pair of series of length > 50
- The Gram matrix has all-zero off-diagonal entries
- kernel-k-means assigns every curve to the same cluster regardless of σ

**Phase to address:**
Phase 54 (GAK kernel core) — the log-domain recursion must be the implementation path from day one. A `test_gak_no_underflow` test with m=200 series at a realistic σ must pass before the phase closes.

---

### Pitfall 2: GAK Is Only PSD in Its Triangular (Normalized) Form — Raw GAK Breaks Kernel Machines

**What goes wrong:**
Raw (un-normalized) GAK `k_GA(x, y)` is not positive semi-definite. The Gram matrix may have negative eigenvalues. Feeding a non-PSD Gram to an external kernel-SVM (`SVC(kernel='precomputed')` in scikit-learn or equivalent) produces undefined optimization behavior: the SVM solver may diverge, report a support vector count of n (all points), return wrong predictions, or silently succeed with systematically biased outputs. Because scikit-learn does not check PSD by default, the failure is invisible.

The _triangular_ variant (Cuturi 2011, Section 3) normalizes by the geometric mean of self-kernels:
```
k̂_GA(x, y) = k_GA(x, y) / sqrt(k_GA(x,x) · k_GA(y,y))
```
This is the form with a PSD guarantee, and it is what tslearn@0.9.0 exposes as `gak()`.

**Why it happens:**
The normalization step looks optional (the raw kernel "works" for clustering with implicit feature-space distances), so implementers skip it to avoid computing the diagonal self-kernels. The PSD property is only guaranteed for the normalized form.

**How to avoid:**
Always normalize: after computing the n×n log-domain Gram, divide each `G[i][j]` by `sqrt(G[i][i] * G[j][j])` (all in log space: `log_G_norm[i][j] = log_G[i][j] - 0.5*log_G[i][i] - 0.5*log_G[j][j]`, then exp). The diagonal becomes exactly 1.0 by construction. Expose only the normalized form in the public API for the Gram export; document explicitly that the un-normalized value is an internal intermediate.

Provide a post-construction PSD verification test: compute the minimum eigenvalue of the Gram (via nalgebra's symmetric eigendecomposition) and assert it is ≥ −ε for small ε (e.g. 1e-8). This is a one-time check in the test suite, not a runtime guard.

**Warning signs:**
- The Gram matrix diagonal is not all-ones
- `G[i][j] > G[i][i]` or `G[i][j] > G[j][j]` for any pair (impossible after normalization since k̂ ∈ [0,1])
- Any off-diagonal entry is > 1.0 or < 0.0
- Minimum eigenvalue of Gram is strongly negative (< −1e-6)

**Phase to address:**
Phase 54 (GAK kernel core) — normalization is part of the kernel definition, not an option. The PSD eigenvalue check is a required test. Phase 55 (Gram export) must document that the exported matrix is the normalized form.

---

### Pitfall 3: Floating-Point Asymmetry Makes the Gram Matrix Non-Symmetric

**What goes wrong:**
The normalized GAK should satisfy `k̂(x, y) = k̂(y, x)` exactly, but floating-point arithmetic may produce `G[i][j] ≠ G[j][i]` at the ULP level (last bit differs) due to different evaluation orders in the forward recursion. This asymmetry is enough to cause scikit-learn's `SVC(kernel='precomputed')` to raise a `ValueError: kernel matrix is not symmetric` (when it checks), or silently produce wrong results (when it doesn't).

**Why it happens:**
In the log-domain recursion, evaluating `gak(x, y)` follows the path (i increases outer, j increases inner) while `gak(y, x)` follows the transposed path. Even though the underlying computation is mathematically identical, floating-point addition is non-associative, so results differ at ~1e-15 precision.

**How to avoid:**
After computing the upper triangle via the `pairwise_distance_matrix` pattern (see `distance.rs`), explicitly symmetrize: `G[j][i] = G[i][j]` (assign the already-computed value, do not recompute). The `pairwise_distance_matrix` helper in `distance.rs` already does this correctly — mirror that pattern exactly for the Gram computation. Add a test that asserts `(G - G.T).abs().max() < 1e-14` for the returned Gram.

**Warning signs:**
- `G[i][j] != G[j][i]` at bit level (detectable with `assert_eq!(G[(i,j)].to_bits(), G[(j,i)].to_bits())`)
- scikit-learn raises `ValueError: kernel matrix is not symmetric`
- Eigendecomposition reports complex eigenvalues (indicates severe asymmetry)

**Phase to address:**
Phase 54 (GAK kernel core) — enforce symmetry by assignment, not recomputation. Add a symmetry assertion test.

---

### Pitfall 4: Wrong σ Choice Makes the Gram Matrix Degenerate

**What goes wrong:**
GAK has a bandwidth parameter σ. If σ is too small relative to the typical inter-series cost, all off-diagonal kernel values underflow to 0.0 even in log space (`log_G[i][j]` → −∞), and the Gram matrix is near-identity. If σ is too large, all kernel values saturate to 1.0, and the Gram matrix is near-constant (rank-1, all rows equal). Both cases make kernel-k-means assign all points to one cluster regardless of initialization, and make the Gram useless for a downstream SVM.

**Why it happens:**
Users copy σ from a paper or from tslearn defaults without adapting to their data scale. tslearn's default σ is `sigma = np.sqrt(m)` (square root of series length), based on a heuristic from Cuturi (2011) for normalized unit-variance series. fdars users operating on FDA curves with different amplitude scales will produce very different cost magnitudes.

**How to avoid:**
Expose σ as a required explicit parameter (do not default-hide it). Document the Cuturi heuristic in the docstring: for zero-mean, unit-variance series of length m, start with `σ = sqrt(m)`. Provide a companion function `gak_sigma_heuristic(data) -> f64` that computes the median pairwise DTW distance on a subsample and sets `σ = sqrt(median_dtw)` — this is the tslearn `unnormalized_gak` auto-bandwidth recipe. Add a σ-sensitivity sanity-check test: verify that with a reasonable σ, the Gram has off-diagonal entries in (0.1, 0.9) rather than all-near-0 or all-near-1.

**Warning signs:**
- All off-diagonal Gram entries are < 0.01 (σ too small) or all > 0.99 (σ too large)
- Gram eigenvalue spectrum is {1, 1, ..., 1, n-k} (near-identity) or {n, 0, 0, ..., 0} (rank-1)
- kernel-k-means converges in 1 iteration regardless of n_init

**Phase to address:**
Phase 54 (GAK kernel core) — the σ-sensitivity test and the heuristic helper must be delivered in this phase. Phase 55 (Gram export) must document the sensitivity behavior clearly.

---

### Pitfall 5: Diagonal Self-Kernel Dominance Causing Over-Normalization NaN

**What goes wrong:**
If a series `x` is very long and the bandwidth σ is large, the self-kernel `k_GA(x,x)` can become extremely large (many alignment paths, all with cost ≈ 0 since every diagonal element aligns perfectly). In log space `log_G[i][i]` may be very large positive. Normalization computes `G[i][j] / sqrt(G[i][i] * G[j][j])`, which is 0.0/0.0 if two very different series have G[i][j]=0 (underflow) and G[i][i], G[j][j] are large but finite. In log space the analogous failure is `log(0) - large_positive = -inf - large_positive = NaN` or `-Inf`.

**Why it happens:**
The log-normalization `log_G_norm = log_G[i][j] - 0.5*(log_G[i][i] + log_G[j][j])` is undefined when `log_G[i][j] = -Inf` and the denominator is finite positive — the result is `-Inf`, which exponentiates to `0.0` correctly. The true failure is when `log_G[i][i]` is `+Inf` (self-kernel overflow even in log space), which only happens for extraordinarily long series; treat as a recoverable edge case by clamping.

**How to avoid:**
In log-domain normalization: if `log_G[i][i]` is `+Inf` for any i (detectable), return `Err(FdarError::ComputationFailed)` with a message indicating σ is too large relative to series length. If `log_G[i][j] == -Inf` and both diagonals are finite, the normalized result is `0.0` — this is mathematically correct and must not produce NaN. Add an explicit `if log_numer == f64::NEG_INFINITY { 0.0 } else { (log_numer).exp() }` guard. Test the edge case: two completely dissimilar series should produce a normalized GAK near 0 but not NaN.

**Warning signs:**
- Any NaN in the returned Gram matrix
- `k̂(x, x) != 1.0` for any series (self-normalized kernel must be exactly 1.0)
- `k̂(x, y) > 1.0` for any pair

**Phase to address:**
Phase 54 (GAK kernel core) — add `assert!(!gram[(i,j)].is_nan())` as a post-condition check in debug mode. Add a test with dissimilar series to confirm NaN-free behavior.

---

### Pitfall 6: Kernel-K-Means Empty-Cluster Crash or Silent Degeneration

**What goes wrong:**
Kernel-k-means minimizes a kernel-induced objective in feature space. Unlike Euclidean k-means, there is no explicit centroid curve. The assignment step uses kernel evaluations:
```
c*(i) = argmin_c [ k̂(x_i, x_i) - (2/|C_c|) Σ_{j∈C_c} k̂(x_i, x_j) + (1/|C_c|²) Σ_{j,l∈C_c} k̂(x_j, x_l) ]
```
If a cluster becomes empty at any iteration, the `1/|C_c|` terms produce divide-by-zero. The existing k-means in `clustering.rs` handles empty clusters by re-seeding from the dataset, but kernel-k-means has no centroid to reseed from — the naive re-seeding strategy must use the precomputed Gram matrix directly (assign the most distant point to the empty cluster).

**Why it happens:**
Developers port Euclidean k-means logic without recognizing that the "centroid re-seeding" step requires a centroid curve (which does not exist in kernel space). The cluster-sum terms `Σ_{j,l∈C_c} k̂(x_j, x_l)` are precomputed from the Gram; empty-cluster detection is straightforward but the recovery differs fundamentally from Euclidean k-means.

**How to avoid:**
Before each assignment step, check that no cluster is empty. If a cluster empties: assign it the data point currently furthest from its assigned cluster center (using kernel distances from the Gram). This requires a "furthest-from-assigned-center" metric computed from the Gram. Document the recovery strategy in the code. Add a test with n=k (forced-tight scenario) that verifies no panic and valid output.

**Warning signs:**
- Divide-by-zero or NaN in objective after any iteration
- `KernelKMeansResult.cluster` has fewer distinct labels than `k`
- Algorithm converges in 1 iteration with all points in one cluster

**Phase to address:**
Phase 56 (kernel-k-means) — empty-cluster handling is a required feature, not an edge case. The test must force an empty cluster scenario (use k > natural cluster count) and verify recovery.

---

### Pitfall 7: Kernel-K-Means Non-Convergence and Missing N-Init Restarts

**What goes wrong:**
Kernel-k-means is not guaranteed to converge to the global optimum; it finds a local minimum of the kernel objective. A single random initialization routinely lands in a poor local minimum, especially for well-separated clusters in feature space that happen to be nearby in the random initialization. The result: algorithm converges but produces clusters that look nothing like the true structure.

The existing `kmeans_fd` in `clustering.rs` supports a single seed; users wanting multiple restarts call it themselves. For kernel-k-means this is harder because the Gram is precomputed — multiple restarts reuse the same Gram (cheap) but the initialization must be drawn from the Gram entries. Kernel-k-means++ initialization (analogous to k-means++) selects initial cluster assignments proportional to kernel distances from already-selected seeds, which requires reading rows of the Gram.

**Why it happens:**
Developers implement one restart and declare the algorithm done. The n_init=1 problem is invisible on small toy datasets that happen to work.

**How to avoid:**
Expose `n_init: usize` in `KernelKMeansConfig` with default 10 (matching tslearn's default). Implement kernel-k-means++ initialization: select first assignment uniformly at random, then each subsequent seed with probability proportional to `min_c k̂(x_i, x_i) - 2*k̂(x_i, x_c) + k̂(x_c, x_c)` for already-selected centers. Run all restarts, keep the best by kernel objective value. Use the `StdRng::seed_from_u64(seed + run as u64)` pattern (already established for elastic-FPCA and gmm) for deterministic multi-restart seeding.

**Warning signs:**
- n_init=1 in the config struct
- Two runs with different seeds produce very different cluster assignments on the same data
- Algorithm always converges in ≤ 3 iterations (likely stuck at initialization)

**Phase to address:**
Phase 56 (kernel-k-means) — n_init with ≥ 10 restarts and kernel-k-means++ init are non-negotiable. A deterministic-seed test (same seed → same result) must pass.

---

### Pitfall 8: Test-Matrix Orientation / Normalization Mismatch Silently Degrades Prediction

**What goes wrong:**
When using the Gram matrix for a precomputed-kernel SVM, the training phase uses an n_train × n_train symmetric PSD Gram. The prediction (test) phase requires an n_test × n_train matrix where entry `[i, j] = k̂(x_test_i, x_train_j)`. A common mistake is to:
1. Compute an (n_train + n_test) × (n_train + n_test) full Gram and slice incorrectly
2. Compute the test-train Gram transposed (n_train × n_test instead of n_test × n_train)
3. Use a different σ for the test-train Gram than for the training Gram
4. Forget to normalize the test-train entries by the training-set self-kernels (using `k_GA(x_test_i, x_test_i)` and `k_GA(x_train_j, x_train_j)` rather than only training-set diagonals)

All four produce wrong SVM predictions with no error — scikit-learn's `SVC.predict()` accepts any matrix of the right shape.

**Why it happens:**
The training Gram and the test-train Gram are computed separately, often by different calls or at different times. The normalization for the test matrix must use the training-set self-kernels `k̂(x_test_i, x_train_j) = k_GA(x_test_i, x_train_j) / sqrt(k_GA(x_test_i, x_test_i) * k_GA(x_train_j, x_train_j))`. Forgetting that the test-set self-kernels are needed for normalization is the most common mistake.

**How to avoid:**
Expose two separate functions matching the two phases:
- `gak_gram_train(data, sigma) -> GakGramResult` — returns the n×n training Gram plus the precomputed diagonal self-kernels `diag_self_kernels: Vec<f64>`
- `gak_gram_predict(test_data, train_self_kernels, sigma) -> FdMatrix` — computes the n_test × n_train prediction matrix, normalizing by test-set self-kernels (computed internally) and the provided training-set self-kernels

This API makes it impossible to forget the training-set self-kernels because they are part of the result struct. Document in the docstring: "Pass `GakGramResult.diag_self_kernels` to `gak_gram_predict` — using different self-kernels for normalization will silently degrade accuracy." Add an integration test that trains a kernel-SVM with the training Gram and scores predictions against a known-correct reference.

**Warning signs:**
- A single `gak_gram(data)` function that returns only the matrix (no self-kernels)
- Prediction accuracy is far lower than expected (e.g., worse than random) despite correct training
- The test-train matrix has shape (n_train, n_test) rather than (n_test, n_train)

**Phase to address:**
Phase 55 (Gram export) — the split `gak_gram_train` / `gak_gram_predict` API is the design; do not expose a single monolithic function.

---

### Pitfall 9: O(n² · m²) Pairwise Cost With Redundant Self-Kernel Recomputation

**What goes wrong:**
The naive Gram computation calls `gak(x_i, x_j)` for all n² pairs. The forward recursion is O(m²) per pair (the DP table is m×m), so the full Gram is O(n² · m²). For n=500 curves of length m=100, this is 500² × 100² = 25 × 10⁸ operations — already slow at ~25 billion FLOPs. If the diagonal self-kernels `gak(x_i, x_i)` are recomputed naively as part of the upper-triangle loop, they are computed once (correctly). But if the normalization step then re-calls `gak(x_i, x_i)` a second time for each pair, the diagonal cost doubles unnecessarily.

Additionally, the O(n²) pairs must be parallelized. The existing `pairwise_distance_matrix` in `distance.rs` already parallelizes the upper-triangle via `iter_maybe_parallel!` — the GAK Gram must use the same pattern, not a nested loop.

**Why it happens:**
Developers compute the upper-triangle and diagonal separately without caching. The normalization step is written as a post-processing loop that calls `gak(x, x)` again.

**How to avoid:**
1. Precompute all n diagonal self-kernels in a parallel loop before the upper-triangle phase.
2. Reuse the `pairwise_distance_matrix` helper pattern for the upper triangle.
3. Normalization uses the precomputed diagonal vector — no additional kernel evaluations.
4. Cache the `row(i)` materializations (as done in `soft_dtw_self_1d`) to avoid repeated `FdMatrix::row()` calls in the inner loop.

The implementation ordering is: `self_kernels[i] = gak_log(row_i, row_i)` for all i (parallel), then upper-triangle `gram[i][j] = exp(gak_log(row_i, row_j) - 0.5*(self_kernels[i] + self_kernels[j]))` (parallel), then symmetrize. This is three passes, not one, but each pass is embarrassingly parallel.

**Warning signs:**
- Two separate `gak()` call sites for the diagonal (one in the upper-triangle loop, one in normalization)
- Gram computation time scales as n² × m² without the O(n) diagonal precompute saving
- `FdMatrix::row()` called O(n²) times total instead of O(n) times (materializing the same row repeatedly)

**Phase to address:**
Phase 54 (GAK kernel core) for the log-domain computation; Phase 55 (Gram export) for the parallelized Gram construction. A criterion benchmark must measure Gram construction time at n=100 and n=200 to verify O(n²) scaling.

---

### Pitfall 10: Exp-of-Naive-DTW Kernel Is NOT GAK and NOT PSD

**What goes wrong:**
A tempting shortcut is to compute DTW distance `d_DTW(x,y)` and return `exp(-d_DTW(x,y)² / σ²)` as the kernel. This "Gaussian DTW kernel" looks like a kernel, is always in [0,1], and has k(x,x)=1, but it is **not positive semi-definite**. The Gram matrix will have negative eigenvalues, breaking kernel-SVM. The `soft_dtw_divergence` in `metric/soft_dtw.rs` has similar structure — it is also not PSD.

The GAK is specifically designed to be PSD by summing over all alignment paths (not using a single best-path distance). This is the fundamental difference between GAK and "exp(-DTW)".

**Why it happens:**
`exp(-d²)` is the Gaussian (RBF) kernel recipe applied naively to DTW distance. It works for Euclidean distance because the Euclidean distance is Hilbert-embeddable. DTW distance is not.

**How to avoid:**
Do not use soft-DTW distance as input to a Gaussian kernel. The GAK forward recursion must sum over paths, not return a single minimum-cost path. The fact that `soft_dtw.rs` already implements a path-sum in log domain (softmin3) does not mean the soft-DTW value itself is a valid kernel — it is a distance, not a path-sum kernel. Document this distinction explicitly in the `gak` module docstring.

**Warning signs:**
- The GAK implementation calls `soft_dtw_distance` and applies `(-x).exp()` to the result
- The Gram matrix minimum eigenvalue is < −0.01
- The implementation has O(m) cost per pair (a single-path recursion) rather than O(m²) (the full DP table)

**Phase to address:**
Phase 54 (GAK kernel core) — the implementation must follow the O(m²) DP-table path-sum, not the O(m) path-min of soft-DTW. Code review must confirm the recursion sums contributions from all three predecessors, not minimizes.

---

### Pitfall 11: Column-Major FdMatrix Row Access in the Inner Loop Is the Hot Path

**What goes wrong:**
`FdMatrix` stores data column-major. Accessing row i requires either `data.row(i)` (which allocates a Vec<f64>) or `data.row_to_buf(i, buf)` (which fills a pre-allocated buffer). The GAK forward DP accesses both `x[i]` and `y[j]` in a nested loop — if both series are taken as `data.row(i)` Vec allocations, the Gram computation allocates O(n) Vecs, each of size m. For n=500, m=100 this is 500 Vec<f64> allocations of 800 bytes each — manageable, but the pattern is already established in `soft_dtw_self_1d`: pre-collect all rows into `Vec<Vec<f64>>` before the parallel upper-triangle loop.

**Why it happens:**
Developers call `data.row(i)` inside the closure passed to `pairwise_distance_matrix`, triggering allocation inside the rayon parallel scope. Under parallel execution this is safe but slower than necessary.

**How to avoid:**
Mirror the pattern from `soft_dtw_self_1d`: `let rows: Vec<Vec<f64>> = (0..n).map(|i| data.row(i)).collect();` before the parallel upper-triangle closure. The closure then takes `&rows[i]` — no allocation in the hot path. This is already the established convention in `metric/soft_dtw.rs`; apply it to the GAK implementation.

**Warning signs:**
- `data.row(i)` called inside the parallel closure (inside `pairwise_distance_matrix`)
- Memory profiling shows O(n²) allocations during Gram construction (should be O(n))
- Criterion shows high allocation variance for Gram construction

**Phase to address:**
Phase 54 (GAK kernel core) — collect rows before the parallel loop. The criterion benchmark for Gram construction will surface this if allocation cost is significant.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Expose raw (un-normalized) GAK in public API | Simpler initial implementation | Non-PSD kernel breaks kernel-SVM; all downstream code is wrong | Never — normalization is part of the GAK definition |
| Single n_init=1 for kernel-k-means | Fewer restart runs, faster tests | Routinely finds poor local minima; tests pass on toy data, fail on real data | Never for production; acceptable for a debug/profiling mode if labelled |
| Recompute diagonal self-kernels in normalization loop | Code is simpler (one pass) | 2× diagonal work; O(n) redundant O(m²) DP evaluations | Never — pre-compute diagonals in one pass |
| Monolithic `gak_gram()` returning only the matrix | Simpler API surface | Users cannot compute test-train Gram with correct normalization (no access to training self-kernels) | Never if the Gram is intended for precomputed-kernel SVM |
| Use `exp(-soft_dtw_divergence)` as the kernel | Reuses existing code | Not PSD; breaks kernel-SVM silently | Never |
| Skip PSD verification test | One less test to write | PSD failures are invisible until a downstream SVM diverges | Never — the eigenvalue test is the only guard |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| scikit-learn `SVC(kernel='precomputed')` | Pass n_test×n_test matrix for prediction instead of n_test×n_train | Prediction matrix must be n_test×n_train; columns index training points |
| scikit-learn `SVC(kernel='precomputed')` | Use different σ for train Gram and test-train Gram | σ is a hyperparameter of the kernel function — must be identical for train and predict |
| scikit-learn `SVC(kernel='precomputed')` | Normalize test-train entries using test-set self-kernels instead of training-set self-kernels (wrong) | Test-train normalization uses `sqrt(k(x_test_i, x_test_i) * k(x_train_j, x_train_j))` — both parties' self-kernels | 
| tslearn `gak()` reference | tslearn returns a scalar; fdars exports a Gram matrix — the comparison is `tslearn.gak(x, y, sigma) == gram[(i,j)]` | Compute tslearn pairwise and compare element-by-element, not function-by-function |
| rayon + FdMatrix::row() inside closure | Allocates Vec<f64> per row inside parallel scope | Pre-collect rows: `let rows = (0..n).map(|i| data.row(i)).collect::<Vec<_>>()` before the closure |

---

## Performance Traps

| Trap | Symptoms | Prevention | Notes |
|------|----------|------------|-------|
| O(n² · m²) without parallelism | n=200, m=100 takes > 10s on a single core | Use `pairwise_distance_matrix` parallel pattern; full Gram should scale as O(n²·m²/threads) | Benchmark at n=100, n=200; expect ≈ 4× improvement with 4 threads |
| Redundant diagonal self-kernel computation | Gram construction time ≈ 2× theoretical minimum | Precompute all self-kernels in one parallel pass before upper-triangle | Self-kernel `gak(x,x)` is cheaper than `gak(x,y)` (all diagonal = 0) but still O(m²) |
| `data.row()` allocations inside parallel loop | High allocation variance in criterion output | Collect all rows before the parallel closure | Already the established pattern in `soft_dtw_self_1d` |
| kernel-k-means rebuilding cluster-sum terms from scratch each iteration | O(n²·k) per iteration instead of O(n·k) with incremental updates | Precompute intra-cluster kernel sums; update incrementally on assignment change | Acceptable for first implementation; optimize only if benchmark shows needed |
| n_init=10 restarts with fresh Gram per restart | Recomputes Gram 10 times | Gram is fixed for all restarts — compute once, reuse | The Gram depends only on data and σ, not on cluster assignments |

---

## "Looks Done But Isn't" Checklist

- [ ] **GAK kernel core:** Log-domain recursion verified — `gak(x, y)` returns non-zero values for series of length m=200. Verify: `assert!(gram[(0,1)] > 1e-10)`.
- [ ] **GAK kernel core:** Normalization verified — diagonal of returned Gram is all-ones. Verify: `assert!((gram[(i,i)] - 1.0).abs() < 1e-12)` for all i.
- [ ] **GAK kernel core:** PSD verified — minimum eigenvalue of Gram ≥ −1e-8. Verify: compute eigendecomposition via nalgebra, assert on `min_eigenvalue`.
- [ ] **GAK kernel core:** Symmetry verified — `gram[(i,j)] == gram[(j,i)]` exactly (by assignment, not recomputation). Verify: `assert_eq!(gram[(i,j)].to_bits(), gram[(j,i)].to_bits())`.
- [ ] **GAK kernel core:** No NaN or Inf in Gram for any realistic input. Verify: `assert!(gram.data.iter().all(|&x| x.is_finite()))`.
- [ ] **GAK kernel core:** Known-value regression test against tslearn — feed a small n=5, m=10 dataset through both implementations and compare to < 1e-6.
- [ ] **Gram export:** `gak_gram_train` result contains `diag_self_kernels` for use in `gak_gram_predict`. Verify: field is present in the result struct.
- [ ] **Gram export:** `gak_gram_predict` produces an n_test × n_train matrix, not n_train × n_test. Verify: `assert_eq!(pred_gram.shape(), (n_test, n_train))`.
- [ ] **Gram export:** Prediction matrix entries are in [0, 1]. Verify: `assert!(pred_gram[(i,j)] >= 0.0 && pred_gram[(i,j)] <= 1.0)` for all entries.
- [ ] **Kernel-k-means:** n_init ≥ 10 restarts implemented. Verify: `KernelKMeansConfig { n_init: 10, .. }` is the default.
- [ ] **Kernel-k-means:** Empty-cluster recovery implemented. Verify: test with k > natural cluster count does not panic.
- [ ] **Kernel-k-means:** Deterministic seeding verified. Verify: two calls with same seed produce identical `cluster` assignments.
- [ ] **Kernel-k-means:** No explicit centroid curve in result (kernel-k-means has no centroid). Verify: `KernelKMeansResult` does not have a `centers: FdMatrix` field (would be misleading).
- [ ] **σ sensitivity:** σ-sensitivity test passes — with Cuturi heuristic σ, off-diagonal Gram entries are in (0.05, 0.95). Verify: assert on min/max off-diagonal.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Log-domain not implemented (all zeros) | LOW | Rewrite the DP core to operate in log-space; add the underflow test; no API changes needed |
| Non-PSD Gram shipped | HIGH | Must add normalization; if normalization changes the API (no self-kernels in result), a breaking change is needed — fix in Phase 54 before Phase 55 ships |
| Gram asymmetry | LOW | Add explicit `gram[(j,i)] = gram[(i,j)]` assignment after upper-triangle loop; rerun tests |
| Wrong test-train matrix orientation | MEDIUM | Fix `gak_gram_predict` orientation; add integration test with a known-correct SVM prediction; no Gram-train changes needed |
| Empty cluster crash in kernel-k-means | LOW | Add empty-cluster detection and furthest-point recovery before the assignment loop; no API changes |
| n_init=1 producing poor clusters | LOW | Add n_init to config struct; wrap existing loop in an outer restart loop; keep best by objective |
| NaN from self-kernel normalization edge case | LOW | Add the `log_numer == NEG_INFINITY → 0.0` guard; NaN test reveals the gap immediately |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Log-domain not implemented (underflow) | Phase 54 — GAK kernel core | `test_gak_no_underflow`: m=200 series, assert off-diagonal > 1e-10 |
| Non-PSD (raw GAK, no normalization) | Phase 54 — GAK kernel core | `test_gak_psd`: eigendecomposition, assert min_eigenvalue ≥ −1e-8 |
| Floating-point asymmetry | Phase 54 — GAK kernel core | `test_gak_symmetry`: assert bit-exact equality of G[i][j] and G[j][i] |
| Exp-of-DTW shortcut (wrong kernel) | Phase 54 — GAK kernel core | Code review: confirm DP table is O(m²), not O(m); PSD test also catches this |
| Wrong σ (degenerate Gram) | Phase 54 — GAK kernel core | `test_gak_sigma_sensitivity`: Cuturi heuristic σ, assert off-diagonal in (0.05, 0.95) |
| Diagonal self-kernel NaN/overflow | Phase 54 — GAK kernel core | `test_gak_unit_diagonal`: assert all G[i][i] == 1.0; `test_gak_no_nan`: assert all finite |
| Known-value regression vs tslearn | Phase 54 — GAK kernel core | `test_gak_vs_tslearn_reference`: hard-coded reference values from tslearn@0.9.0 |
| Redundant self-kernel recomputation | Phase 55 — Gram export | Criterion benchmark: confirm no 2× overhead vs theoretical minimum |
| Wrong test-train matrix orientation | Phase 55 — Gram export | `test_gram_predict_shape`: assert (n_test, n_train); integration SVM accuracy test |
| Missing training self-kernels in API | Phase 55 — Gram export | API review: `GakGramResult` must include `diag_self_kernels` field |
| Empty-cluster crash in kernel-k-means | Phase 56 — kernel-k-means | `test_kernel_kmeans_empty_cluster`: k > natural clusters, assert no panic, valid output |
| n_init=1 local-minimum problem | Phase 56 — kernel-k-means | `test_kernel_kmeans_n_init`: assert n_init=10 default; multi-restart result ≥ single-restart quality |
| Deterministic seeding missing | Phase 56 — kernel-k-means | `test_kernel_kmeans_deterministic`: two calls same seed → identical assignments |
| Row allocation in parallel loop | Phase 54 and 55 | Code review: `data.row()` must not appear inside the parallel closure |

---

## Sources

- Cuturi, M. (2011). "Fast Global Alignment Kernels." ICML. — Primary GAK reference; Section 3 defines the triangular/normalized form and the PSD proof.
- Cuturi, M. & Blondel, M. (2017). "Soft-DTW: a Differentiable Loss Function for Time-Series." ICML. — Already cited in `metric/soft_dtw.rs`; the soft-min log-domain pattern is reused for GAK.
- tslearn@0.9.0 `gak` implementation — Reference for triangular normalization, σ heuristic, and cross-validation patterns.
- fdars-core `metric/soft_dtw.rs` — Existing log-domain `softmin3` and DP structure that GAK reuses.
- fdars-core `distance.rs` — `pairwise_distance_matrix` parallel upper-triangle pattern that Gram export reuses.
- fdars-core `clustering.rs` — `kmeans_fd` empty-cluster handling and k-means++ init as pattern references for kernel-k-means.
- scikit-learn `SVC(kernel='precomputed')` documentation — training vs prediction matrix shape/normalization contract.
- `.planning/codebase/ARCHITECTURE.md` — Existing anti-patterns (dense matrix copy, unvalidated slice access, NaN inconsistency) that the GAK implementation must not replicate.

---
*Pitfalls research for: fdars v0.32.0 — GAK kernel + kernel-k-means + Gram-matrix export*
*Researched: 2026-09-02*
