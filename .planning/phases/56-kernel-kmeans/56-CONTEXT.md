# Phase 56: Kernel-k-means Clustering - Context

**Gathered:** 2026-09-02
**Status:** Ready for planning
**Mode:** Smart-discuss (autonomous) — decisions resolved from `.planning/research/` + the Phase 54/55 API. No open user decisions. Final phase of milestone v0.32.0.

<domain>
## Phase Boundary

Deliver kernel-k-means clustering of curve sets through the GAK kernel (the headline consumer), plus out-of-sample `predict`. New top-level `src/kernel_kmeans.rs` (peer of `clustering.rs`), re-exported at the crate root. Operates PURELY on the GAK Gram matrix — no explicit centroid curve. Additive/non-breaking, no new dependency.

In scope (GAK-07/08):
- **Fit**: cluster a curve set via kernel-k-means on the GAK Gram — n_init random-partition restarts, empty-cluster recovery, deterministic seeding, best-inertia run returned.
- **Predict**: assign new (out-of-sample) curves to the fitted clusters, reusing the same GAK kernel + normalization (via Phase 55's `gak_gram_predict`).

Out of scope: native SVM; new kernels; any change to Phase 54/55 public behavior.
</domain>

<decisions>
## Implementation Decisions

### Resolved (from research + Phase 54/55 API)

1. **Module:** new `src/kernel_kmeans.rs` at top level; add `pub mod kernel_kmeans;` to `src/lib.rs` and re-export the public surface at the crate root, mirroring how `clustering` is surfaced.

2. **Feature-space assignment (kernel trick — NO centroid).** For cluster `C_k` the squared feature-space distance of point `i` is
   `d²(i,k) = K[i,i] − (2/|C_k|)·Σ_{j∈C_k} K[i,j] + (1/|C_k|²)·Σ_{j,l∈C_k} K[j,l]`.
   With the normalized GAK, `K[i,i]=1`, so it drops out of the argmin; still compute the `2·(...)` cross-term and the per-cluster `within_k = (1/|C_k|²)·Σ_{j,l∈C_k}K[j,l]` term. Precompute `within_k` once per cluster per iteration and reuse across all points → O(n²) per iteration once the Gram is in memory. Assign each point to the cluster minimizing `d²`. The result struct has **NO centroid/`centers` field** (kernel-k-means has no explicit centroid — this is a hard correctness point).

3. **Init = random partition restarts (NOT k-means++).** k-means++ operates on L2 curve vectors; here we operate on similarity-valued Gram entries, so k-means++ is wrong. Use `n_init` random-partition restarts: randomly assign each point to one of k clusters, seeded deterministically per restart as `StdRng::seed_from_u64(seed + restart_idx)`. Keep the run with the lowest total inertia `Σ_i d²(i, label_i)`.

4. **Empty-cluster recovery.** If a cluster empties during iteration (or `k >` natural clusters), reseed it with the point that is currently farthest (max `d²`) from its assigned cluster — never panic. A `k > n_natural_clusters` test must return valid labels without panicking.

5. **Config + result (fdars conventions):**
   - `KernelKmeansConfig { n_clusters: usize, n_init: usize, max_iter: usize, tol: f64, seed: u64, gak: GakConfig }` — Debug/Clone/PartialEq, serde-gated, `Default` (sensible defaults: n_init=10, max_iter=300, tol=1e-4). `n_init` default 10 (robustness over tslearn's default 1 — documented).
   - `KernelKmeansResult { cluster: Vec<usize>, inertia: f64, iter: usize, converged: bool, n_init_best: usize }` + the internal state `predict` needs — Debug/Clone/PartialEq, serde-gated, `#[non_exhaustive]`. **No `centers` field.**
   - `pub fn kernel_kmeans_fd(data: &FdMatrix, config: &KernelKmeansConfig) -> Result<KernelKmeansResult, FdarError>` (#[must_use]) — builds the GAK Gram via Phase 55's `gak_gram_train(data, &config.gak)`, runs the restarts, returns the best result. Store what `predict` requires (see 6).
   - Validation: `n_clusters ≥ 1`, `n_clusters ≤ n`, `n_init ≥ 1`, non-empty data → `FdarError::{InvalidParameter, InvalidDimension}`.

6. **Predict path (GAK-08).** `KernelKmeansResult::predict(&self, new_data: &FdMatrix) -> Result<Vec<usize>, FdarError>`. For each new curve compute `d²(test,k) = k(test,test) − (2/|C_k|)·Σ_{j∈C_k} Kcross[test,j] + within_k`, where `Kcross = gak_gram_predict(&stored_train, new_data)` (n_test × n_train, normalized so `k(test,test)=1` drops out), and `within_k` are the FITTED training within-cluster sums. To make this work, the result stores the fitted `GakGramTrain` (from Phase 55, which carries `train_rows`, `log_self`, `sigma`) plus the final `cluster` labels and the per-cluster `within_k` values (or recompute them from the stored training Gram + labels). Assign each new curve to the min-`d²` cluster. Reuses the same σ + normalization as the fit — no re-estimation.

7. **Determinism:** same `seed` → identical labels (assert in a test). Gram computed ONCE and reused across all restarts (do not rebuild per restart).
</decisions>

<code_context>
## Existing Code Insights
- `src/clustering.rs`: `KmeansResult { cluster, centers, withinss, tot_withinss, iter, converged }` (L17) + a `predict`-style method — mirror the naming/doc style but drop `centers`. `kmeans_fd` (L545) shows the fit-fn shape, argvals handling, and deterministic seeding idiom.
- Phase 55 API in `src/metric/gak.rs`: `gak_gram_train(&FdMatrix,&GakConfig)->Result<GakGramTrain,FdarError>` (Gram + `log_self()` + `sigma` + `pub(crate) train_rows`), `gak_gram_predict(&GakGramTrain,&FdMatrix)->Result<FdMatrix,FdarError>` (n_test×n_train, normalized). `GakConfig{sigma:Option<f64>}`.
- `src/parallel.rs`: `iter_maybe_parallel!` (restart loop is a candidate, but keep RNG determinism — seed per restart, or keep restarts sequential and parallelize the per-point assignment). Prefer whichever keeps results bit-reproducible.
- `src/error.rs`: `FdarError`.
- `kernel_kmeans.rs` will need access to `GakGramTrain`'s training rows for predict — `gak_gram_predict` already encapsulates that, so store the `GakGramTrain` in the result and call `gak_gram_predict`.
</code_context>

<specifics>
## Specific Ideas (verification hooks)
Tests the plan must include:
- `test_kernel_kmeans_recovers_groups`: two well-separated synthetic groups recovered at purity 1.0 (label-permutation-invariant purity check).
- `test_kernel_kmeans_no_centroid_field`: compile-time/structural — the result struct exposes no `centers`/centroid field (documented; assert via the public fields used in the test).
- `test_kernel_kmeans_deterministic`: same seed → identical `cluster` labels across two fits.
- `test_kernel_kmeans_empty_cluster_recovery`: `k` greater than the number of natural clusters returns valid labels, no panic.
- `test_kernel_kmeans_n_init`: n_init>1 runs (Gram built once) and returns the best-inertia labeling; inertia non-increasing vs a single-init baseline on a seeded case.
- `test_kernel_kmeans_predict`: out-of-sample curves near group A route to A's cluster; predict reuses fit σ/normalization (a test curve equal to a training curve gets that curve's cluster).
- `test_kernel_kmeans_validation`: n_clusters=0, n_clusters>n, empty data → errors.
- Doctest on `kernel_kmeans_fd`.
</specifics>

<deferred>
## Deferred Ideas
- Kernel-k-means++ style greedy init (research: random-partition is the correct default here; greedy kernel init is a future refinement).
- Soft/fuzzy kernel clustering; spectral clustering on the GAK Gram — future.
- Parallelizing restarts across threads while preserving determinism — a perf refinement.
</deferred>
