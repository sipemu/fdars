# Phase 56 — Summary 56-01: Kernel-k-means Clustering (GAK-07/08)

**Status:** complete
**Impl commit:** c152ef4f

## Files changed
- **NEW** `fdars-core/src/kernel_kmeans.rs` (~640 LOC incl. tests) — kernel-k-means
  fit + out-of-sample predict, purely Gram-based (no centroid).
- **MOD** `fdars-core/src/lib.rs` — added `pub mod kernel_kmeans;` and crate-root
  re-export of `kernel_kmeans_fd`, `KernelKmeansConfig`, `KernelKmeansResult`.
- **NEW** `.planning/phases/56-kernel-kmeans/56-01-PLAN.md`.

## Public API added
- `KernelKmeansConfig { n_clusters, n_init, max_iter, tol, seed, gak: GakConfig }`
  — Debug/Clone/PartialEq, serde-gated, `Default` (n_clusters=2, n_init=10,
  max_iter=300, tol=1e-4, seed=0). Helper `KernelKmeansConfig::new(k, sigma)`.
- `KernelKmeansResult { cluster, inertia, iter, converged, n_init_best }` +
  private state (`train: GakGramTrain`, `within`, `sizes`) — Debug/Clone/PartialEq,
  serde-gated, `#[non_exhaustive]`. **No `centers`/centroid field.** Methods:
  `n_clusters()`, `predict(&FdMatrix) -> Result<Vec<usize>, FdarError>`.
- `pub fn kernel_kmeans_fd(&FdMatrix, &KernelKmeansConfig) -> Result<KernelKmeansResult, FdarError>`
  (`#[must_use]`).

## Algorithm notes
- GAK Gram built ONCE via `gak_gram_train`; reused across all `n_init` restarts.
- Kernel-trick assignment `d²(i,k) = K[i,i] − (2/|C_k|)Σ_{j∈C_k}K[i,j] + within_k`,
  `within_k = (1/|C_k|²)ΣΣ K[j,l]` precomputed once per cluster per iteration.
- Random-partition init (NOT k-means++), seeded `seed_from_u64(seed + restart_idx)`;
  init repairs empty clusters; lowest-total-inertia run returned.
- Empty-cluster recovery: reseed farthest-point (max d²) from a donor cluster with
  >1 member — never panics (incl. k==n edge, which leaves singletons in place).
- Predict: `Kcross = gak_gram_predict(train, new)` (n_test×n_train, normalized so
  k(test,test)=1) → argmin `1 − (2/|C_k|)ΣKcross[t,j] + within_k` using fitted
  within/sizes. Reuses fit σ/normalization; no re-estimation.

## Tests (all pass)
Inline `#[cfg(test)] mod tests` — 8 tests:
recovers_groups (purity 1.0), deterministic (bit-identical labels+inertia),
empty_cluster_recovery (k=4 > 2 natural groups, no empty cluster survives),
empty_cluster_k_equals_n (extreme edge, no panic), n_init (multi ≤ single inertia),
predict (novel low/high curves route correctly; exact copy matches its label),
validation (n_clusters=0, >n, n_init=0, empty data → errors),
no_centroid (structural — only documented fields; predict needs no centroid).
Plus a doctest on `kernel_kmeans_fd`.

Results: lib `--features linalg` → 8 passed; default features → 8 passed;
doctest → 1 passed.

## Divergences
- Added two convenience/introspection items beyond the literal spec:
  `KernelKmeansConfig::new(k, sigma)` and `KernelKmeansResult::n_clusters()`.
  Additive, non-breaking; keeps tests + doctest ergonomic.
- Added an extra edge test `test_kernel_kmeans_empty_cluster_k_equals_n` on top of
  the required set (belt-and-suspenders for the k==n recovery path).
- No crate-version bump (as instructed).

## Gate tails
- `cargo fmt --check` — clean (exit 0).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean.
- `cargo test -p fdars-core --features linalg kernel_kmeans` — 8 passed.
- `cargo test -p fdars-core kernel_kmeans` (default features) — 8 passed.
