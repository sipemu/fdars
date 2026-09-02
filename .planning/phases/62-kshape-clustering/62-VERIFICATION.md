---
phase: 62
title: k-Shape Clustering & Predict
status: passed
requirements: [KSH-03, KSH-04]
gaps_found: []
verified: 2026-09-02
---

# Phase 62 Verification: k-Shape Clustering & Predict

Each ROADMAP success criterion → PASS/FAIL with evidence. All 4 pass.

## Criterion 1 — `kshape_fd` runs the full algorithm + validation

> User can call `kshape_fd(data, config)` to cluster: iterative SBD assignment +
> shape-extraction refinement to convergence/max_iter, over n_init restarts
> (default 10), keeping best-by-inertia, returning centroids + labels + inertia +
> iter count. Validation errors (n_clusters=0, >n, n_init=0, empty) → Err.

**PASS.**
- `kshape_fd(data: &FdMatrix, config: &KShapeConfig) -> Result<KShapeResult, FdarError>`
  implemented (`src/kshape.rs`), `#[must_use]`. Result carries `centroids`,
  `cluster`, `inertia`, `iter`, `converged`, `n_init_best`.
- n_init restarts, `seed.wrapping_add(restart)`, min-inertia kept. Default n_init=10
  (`KShapeConfig::default`).
- Convergence: stable labels OR `|Δinertia| < tol`, capped at `max_iter`.
- Validation: n_clusters<1 / >n / n_init<1 → `InvalidParameter`; empty data →
  `InvalidDimension`. Covered by `test_kshape_validation` (green).
- Doctest on `kshape_fd` (rising vs falling ramps) passes.

## Criterion 2 — Centroids are k-Shape shape prototypes (not k-means means)

> Each centroid = TOP eigenvector of M = QᵀSᵀSQ with Q = I − O/n_k, from
> shift-aligned z-normalized members, sign-fixed to correlate positively, re-z-normed.
> On a two-shifted-motif dataset, recovers groups at high purity, centroid corr > 0.99.

**PASS** (with the mathematically-equivalent tslearn formulation).
- `shape_extraction` builds `S = XᵀX`, `Q = I_m − O_m/m` (**divisor m = series
  length**, per 62-CONTEXT decision 5 / FEATURES.md — tslearn `eye(sz)−ones/sz`),
  `M = QᵀSQ`, symmetrized.
- Top eigenvector via `SymmetricEigen` with **argmax over ascending eigenvalues**
  (not index 0).
- Sign fix: picks ±v minimizing Σ SBD(±v, member); centroid re-z-normalized.
- Members shift-aligned by their SBD shift before stacking.
- Evidence: `test_kshape_recovers_shifted_groups` → **purity 1.0** on sine vs
  double-frequency-sine groups with random per-series circular shifts + noise (this
  gate fails on wrong centering/eigenvector/sign). `test_kshape_centroid_sign` →
  corr **> 0.99** on a clean single-shape cluster.

## Criterion 3 — Robust + reproducible

> Empty cluster recovered in place by farthest-point reassignment (documented
> divergence from tslearn full restart) → k > natural-clusters never panics, every
> size ≥ 1; inertia non-increasing on clean data; deterministic seeding → byte-identical
> labels + inertia, sequential and parallel agree.

**PASS.**
- `recover_empty_clusters` + `ensure_no_empty_random` mirror `kernel_kmeans.rs`.
  `test_kshape_empty_cluster_recovery` (k=5, 2 natural groups) → no panic, all
  cluster sizes ≥ 1.
- Deterministic seeding `seed.wrapping_add(restart)`. `test_kshape_deterministic` →
  identical labels, `inertia.to_bits()` equal, and centroid bits equal across two
  fits. SBD is RNG-free, so sequential == parallel (byte-identical).
- `test_kshape_best_of_n_init` → n_init=10 inertia ≤ n_init=1 (best-restart selection
  by inertia correct).

## Criterion 4 — `predict` assigns out-of-sample series

> `KShapeResult::predict(new_data) -> Result<Vec<usize>, FdarError>`: each new series
> z-normalized, SBD to each stored (z-normed) centroid, argmin, centroids used as-is →
> predict(train_data) reproduces training labels.

**PASS.**
- `predict` implemented: z-norm each new series, SBD to each stored centroid, argmin;
  stored centroids used unchanged; dimension guard on `m`.
- Evidence: `test_kshape_predict` → `predict(train) == res.cluster`; a shifted copy of
  training series 0 routes to series 0's cluster. `test_kshape_validation` covers the
  predict dimension-mismatch error path.

## Gate summary

- lib tests (linalg): 7 passed. doctest: 1 passed. default-feature: 7 passed.
- `cargo fmt --check`: clean. `cargo clippy --all-targets --features linalg,parallel
  -- -D warnings`: clean (exit 0).

**Verdict: PASSED (4/4 criteria).**
