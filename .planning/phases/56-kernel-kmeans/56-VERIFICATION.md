---
phase: 56
title: Kernel-k-means Clustering
status: passed
verified: 2026-09-02
impl_commit: c152ef4f
---

# Phase 56 — Verification: Kernel-k-means Clustering (GAK-07/08)

Each ROADMAP success criterion → PASS/FAIL + evidence.

## Criterion 1 — Cluster via GAK, recover two groups at purity 1.0, NO centroid field
**PASS.**
- `test_kernel_kmeans_recovers_groups`: two well-separated bands (near 0 / near 10),
  permutation-invariant purity == 1.0. Assignments computed from Gram kernel
  distances (`d²(i,k) = K[i,i] − (2/|C_k|)ΣK[i,j] + within_k`).
- No centroid: `KernelKmeansResult` has fields `cluster, inertia, iter, converged,
  n_init_best` (+ private `train/within/sizes`) and **no `centers` field**;
  `test_kernel_kmeans_no_centroid` destructures only the documented public fields
  and predicts from stored Gram state alone. Module doc states the no-centroid
  correctness property.

## Criterion 2 — Robust: n_init restarts (best inertia), empty-cluster recovery no-panic, Gram once
**PASS.**
- n_init restarts: `n_init` random-partition restarts in `kernel_kmeans_fd`, lowest
  total-inertia kept (`n_init_best` recorded). `test_kernel_kmeans_n_init`:
  n_init=10 inertia ≤ n_init=1 baseline.
- Gram once: single `gak_gram_train` call before the restart loop; reused across all
  restarts (`gram = &train.gram`).
- Empty-cluster recovery: `recover_empty_clusters` reseeds the farthest point from a
  donor with >1 member; `ensure_no_empty_random` repairs init. Never panics.
  `test_kernel_kmeans_empty_cluster_recovery` (k=4 > 2 natural groups → every cluster
  non-empty, no panic) and `test_kernel_kmeans_empty_cluster_k_equals_n` (k==n edge)
  both pass.

## Criterion 3 — Reproducible: deterministic per-restart seeding (seed + restart_idx)
**PASS.**
- Each restart seeded `StdRng::seed_from_u64(seed.wrapping_add(restart_idx))`.
- `test_kernel_kmeans_deterministic`: two fits with the same config produce
  bit-identical `cluster` labels AND bit-identical `inertia` (`to_bits()`) AND same
  `n_init_best`.

## Criterion 4 — Out-of-sample predict reuses fit kernel/normalization, routes correctly
**PASS.**
- `KernelKmeansResult::predict` builds `Kcross = gak_gram_predict(train, new_data)`
  (n_test × n_train, normalized so k(test,test)=1, using the STORED training σ and
  self-kernels), then argmin over `1 − (2/|C_k|)ΣKcross[t,j] + within_k` with the
  fitted `within`/`sizes` — no re-estimation.
- `test_kernel_kmeans_predict`: a novel low curve routes to the low cluster, a novel
  high curve to the high cluster, and a test curve equal to training curve 0 gets
  curve 0's label.

## Gates
- `cargo fmt --check` — clean.
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean.
- `cargo test -p fdars-core --features linalg kernel_kmeans` — 8 passed, 0 failed.
- `cargo test -p fdars-core kernel_kmeans` (default features) — 8 passed, 0 failed.
- Doctest on `kernel_kmeans_fd` — passed.

**Verdict: all 4 success criteria PASS → `status: passed`.**
