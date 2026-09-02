# Phase 62 Plan 01 SUMMARY: k-Shape Clustering & Predict (KSH-03/04)

**Status:** complete
**Requirements:** KSH-03 (k-Shape fit), KSH-04 (out-of-sample predict)
**Commit (impl):** feat(62): k-Shape clustering + shape-extraction centroids + predict

## Files

- **Created** `fdars-core/src/kshape.rs` (~640 LOC incl. tests) — full k-Shape module.
- **Modified** `fdars-core/src/lib.rs` — added `pub mod kshape;` (after `kernel_kmeans`).
- **Created** `.planning/phases/62-kshape-clustering/62-01-PLAN.md`.

No crate-version bump. No new dependency (reuses `rustfft` via `metric::sbd`,
`nalgebra::SymmetricEigen`, `shapelet::z_normalize_window`). No crate-root flat
re-exports (deferred to Phase 63) — public items live under `fdars_core::kshape::`.

## Public API added

```rust
// fdars_core::kshape
pub struct KShapeConfig {
    pub n_clusters: usize,
    pub n_init: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub seed: u64,
}
impl Default for KShapeConfig; // n_clusters=2, n_init=10, max_iter=100, tol=1e-6, seed=0
impl KShapeConfig { pub fn new(n_clusters: usize) -> Self; }

pub struct KShapeResult {
    pub centroids: FdMatrix,   // k×m, z-normalized rows
    pub cluster: Vec<usize>,
    pub inertia: f64,
    pub iter: usize,
    pub converged: bool,
    pub n_init_best: usize,
}
impl KShapeResult {
    pub fn centroids(&self) -> &FdMatrix;
    pub fn cluster(&self) -> &[usize];
    pub fn inertia(&self) -> f64;
    pub fn n_clusters(&self) -> usize;
    pub fn predict(&self, new_data: &FdMatrix) -> Result<Vec<usize>, FdarError>;
}

#[must_use]
pub fn kshape_fd(data: &FdMatrix, config: &KShapeConfig) -> Result<KShapeResult, FdarError>;
```

Both `KShapeConfig` and `KShapeResult` derive `Debug, Clone, PartialEq`, are
serde-gated, and `#[non_exhaustive]`.

## Implementation notes

- **Up-front z-normalization:** every series z-normed once via
  `shapelet::z_normalize_window` into `Vec<Vec<f64>>`; all assignment + extraction
  operate on those rows.
- **Assignment:** per series, min-SBD centroid via `metric::sbd::sbd`, storing the
  distance for empty-cluster recovery.
- **Shape extraction (decision 5):** members aligned to the current centroid by
  their SBD shift (`sbd(centroid, member).shift`), re-z-normalized, stacked into X;
  `S = XᵀX`; `Q = I_m − O_m/m` (**divisor m = series length**, applied as
  column-mean then row-mean subtraction so Q·S·Q is formed without materializing Q);
  `M = QᵀSQ` symmetrized; top eigenvector via `SymmetricEigen` (**argmax over
  ascending eigenvalues**); **sign fix** picks ±v minimizing Σ SBD(±v, member);
  centroid z-normalized before storage.
- **Empty-cluster recovery:** in-place farthest-point reassignment mirroring
  `kernel_kmeans.rs` (`recover_empty_clusters`, `ensure_no_empty_random`).
- **Convergence:** stable labels OR `|Δinertia| < tol`. inertia = Σ SBD(series,
  assigned centroid), computed after refinement.
- **Determinism:** n_init restarts seeded `seed.wrapping_add(restart)`, keep
  min-inertia. SBD is RNG-free, so sequential and parallel builds are byte-identical.
- **predict:** z-norm each new series, SBD to each stored centroid, argmin. Stored
  centroids used as-is (no re-estimation), plus a dimension-mismatch guard.

## Tests + results

Inline `#[cfg(test)] mod tests` — 7 tests + 1 doctest, all green:

- `test_kshape_recovers_shifted_groups` — 2 shape groups (sine vs double-freq sine),
  random per-series circular shifts + noise → **purity 1.0** (the key gate);
  centroids verified zero-mean.
- `test_kshape_centroid_sign` — clean single-shape cluster → centroid corr > 0.99.
- `test_kshape_empty_cluster_recovery` — k=5 > 2 natural groups → no panic, all
  cluster sizes ≥ 1.
- `test_kshape_deterministic` — same seed → identical labels + inertia bits +
  centroid bits.
- `test_kshape_best_of_n_init` — n_init=10 inertia ≤ n_init=1.
- `test_kshape_predict` — predict(train) reproduces cluster; shifted copy of series 0
  routes to its cluster.
- `test_kshape_validation` — n_clusters=0 / >n / n_init=0 / empty data / predict
  dim-mismatch → errors.
- Doctest on `kshape_fd` — rising vs falling ramps separate into two clusters.

## Gate results

- `cargo test -p fdars-core --features linalg kshape`: **7 passed, 0 failed**.
- `cargo test -p fdars-core --features linalg --doc kshape`: **1 passed**.
- `cargo test -p fdars-core kshape` (default features): **7 passed**.
- `cargo fmt --check`: clean.
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`: clean (exit 0).

## Divergences

- **Empty-cluster recovery** is in-place farthest-point reassignment, a documented
  divergence from tslearn's full-restart (matches the Phase 56 `kernel_kmeans`
  precedent). Recorded in `62-CONTEXT.md` decision 6.
- **Convergence tolerance** compares absolute inertia change (`|Δinertia| < tol`,
  default 1e-6) rather than a relative change; suitable given the bounded [0,2] SBD
  scale.

## Seams for Phase 63

- Crate-root re-exports still needed: `pub use kshape::{kshape_fd, KShapeConfig,
  KShapeResult};` + `prelude` additions.
- `sbd_kmedoids` convenience: build via `metric::sbd::sbd_distance_matrix` +
  `alignment::clustering::kmedoids_from_distances` — to live at the bottom of
  `kshape.rs`.
- `circular_shift` helper is currently private in `kshape.rs`; if Phase 63 needs it
  outside, promote it.
