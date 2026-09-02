# Phase 62: k-Shape Clustering & Predict - Context

**Gathered:** 2026-09-02
**Status:** Ready for planning
**Mode:** Smart-discuss (autonomous) — resolved from `.planning/research/` + Phase 61 SBD API + `kernel_kmeans.rs`. No open user decisions. The main algorithmic phase (shape extraction is the only genuinely-new numerical piece).

<domain>
## Phase Boundary

Deliver k-Shape clustering + out-of-sample predict, built on Phase 61's `sbd`. New top-level `src/kshape.rs` (peer of `kernel_kmeans.rs`). Additive/non-breaking, no new dependency. Crate-root re-exports deferred to Phase 63.

In scope (KSH-03/04):
- **`kshape_fd`** — iterative k-Shape: SBD assignment + shape-extraction centroids, `n_init` restarts, in-place empty-cluster recovery, deterministic seeding.
- **`KShapeResult::predict`** — assign new series to fitted centroids by SBD.

Out of scope: `sbd_kmedoids` (Phase 63), crate-root re-exports + bench (Phase 63).
</domain>

<decisions>
## Implementation Decisions

### Resolved (from research + tslearn `KShape` reference)

1. **Module:** new `src/kshape.rs`; add `pub mod kshape;` to `src/lib.rs`. Mirror `kernel_kmeans.rs` structure throughout (config/result/restart-loop/empty-cluster/predict).

2. **Config:** `pub struct KShapeConfig { n_clusters: usize, n_init: usize, max_iter: usize, tol: f64, seed: u64 }` — Debug/Clone/PartialEq, serde-gated, `Default` (`n_init = 10` — fdars convention, NOT tslearn's 1; `max_iter = 100`, `tol = 1e-6`, `seed = 0`). A `KShapeConfig::new(n_clusters)` helper.

3. **Up-front z-normalization:** z-normalize every input series ONCE (reuse `shapelet::z_normalize_window`) into a working `Vec<Vec<f64>>` (or FdMatrix). All assignment + shape-extraction operate on the z-normed series. Constant series → zero vector (guard already in the z-norm helper).

4. **Result:** `pub struct KShapeResult { centroids: FdMatrix /* k×m, z-normalized */, cluster: Vec<usize>, inertia: f64, iter: usize, converged: bool, n_init_best: usize }` — Debug/Clone/PartialEq, serde-gated, `#[non_exhaustive]`; accessors (`centroids()`, `cluster()`, `inertia()`, `n_clusters()`). **inertia** = sum over series of SBD(series, its assigned centroid).

4. **`kshape_fd(data: &FdMatrix, config: &KShapeConfig) -> Result<KShapeResult, FdarError>`** (`#[must_use]`): run `config.n_init` restarts, keep the lowest-inertia result. Each restart seeded `StdRng::seed_from_u64(seed.wrapping_add(restart))` (mirror kernel_kmeans L263). Per restart: random-partition init (assign each series to a random cluster in `0..k`); then iterate to `max_iter`:
   a. **Assignment:** for each series, compute `sbd` to each centroid, assign to the min-distance cluster (store the SBD `shift` for the shape-extraction alignment; a centroid of all-zeros / first iteration is handled — if a centroid is undefined, treat SBD as max).
   b. **Refinement (shape extraction, per cluster):** see decision 5.
   c. **Convergence:** stop when labels unchanged OR total inertia change < `tol`. Track a non-increasing objective.
   Validate: `n_clusters ≥ 1`, `n_clusters ≤ n`, `n_init ≥ 1`, non-empty → `FdarError::{InvalidParameter, InvalidDimension}`.

5. **Shape extraction (THE crux — follow tslearn `_shape_extraction` EXACTLY; `FEATURES.md` has the source-derived spec, treat it as authoritative):** for cluster `k` with member series (z-normed):
   - Align each member to the current centroid by its SBD optimal shift (from the assignment step) → the shift-aligned member vector; stack into `X` (n_k × m).
   - `S = Xᵀ X` (m × m).
   - Centering matrix over the TIME dimension: `Q = I_m − (1/m)·O_m` where `O_m` is the m×m all-ones matrix and **m = series length** (NOT n_k — this is the common transcription error; tslearn: `Q = eye(sz) − ones((sz,sz))/sz`).
   - `M = Qᵀ S Q` (symmetric m × m).
   - Top eigenvector of `M` (LARGEST eigenvalue) via `nalgebra::SymmetricEigen::new(M)` — nalgebra returns eigenvalues ASCENDING, so take the eigenvector at the argmax eigenvalue (mirror the descending-sort in `fts/spectral.rs:208`).
   - **Sign fix:** compare `Σ_i SBD(+v, member_i).distance` vs `Σ_i SBD(−v, member_i).distance`; keep the sign with the SMALLER sum (else the centroid inverts).
   - **z-normalize** the resulting centroid `v` → the new cluster centroid.
   - Empty cluster (n_k = 0): do NOT run extraction; recover in-place (decision 6).

6. **Empty-cluster recovery (in-place — mirror `kernel_kmeans.rs`, documented divergence from tslearn's full-restart):** if a cluster becomes empty, reseed its centroid from the series currently farthest (max SBD) from its assigned centroid. Never panic; a `k > natural clusters` run returns valid labels.

7. **Predict (KSH-04):** `impl KShapeResult { pub fn predict(&self, new_data: &FdMatrix) -> Result<Vec<usize>, FdarError> }` (`#[must_use]`): z-normalize each new series, compute `sbd` to each stored centroid, argmin. Reuses the SAME centroids/normalization as the fit. `predict(train_data)` must reproduce `cluster`.

8. **Determinism:** deterministic seeding → same seed = identical labels; sequential and `parallel` must agree. Parallelize across restarts OR across the per-series assignment (each rayon task its own `FftPlanner`, `!Send`); keep bit-reproducibility (prefer sequential restarts with a parallel assignment/SBD inner loop, or seeded-independent restarts).
</decisions>

<code_context>
## Existing Code Insights
- Phase 61 `src/metric/sbd.rs`: `sbd(x,y) -> Result<SbdResult,FdarError>` (SbdResult{distance, shift: isize}), `sbd_distance_matrix`.
- `src/kernel_kmeans.rs`: `KernelKmeansConfig` (n_init=10 default, L57/76), `KernelKmeansResult` + `predict` (L148), restart loop `for restart in 0..n_init { seed_from_u64(seed.wrapping_add(restart)) }` (L262), in-place empty-cluster recovery — MIRROR this structure.
- `src/fts/spectral.rs:208`: `nalgebra::SymmetricEigen::new(mat)` + descending eigenvalue sort — the shape-extraction idiom.
- `src/shapelet/distance.rs`: `z_normalize_window` for up-front z-norm.
- `src/matrix.rs`: `FdMatrix` (build the k×m centroid matrix; `row_to_buf`), `src/parallel.rs` (`iter_maybe_parallel!`), `src/error.rs`.
- Conventions: config + Default, `#[must_use]`, serde-gated derives, `Result<_,FdarError>`.
</code_context>

<specifics>
## Specific Ideas (verification hooks — from PITFALLS.md)
Tests the plan must include:
- `test_kshape_recovers_shifted_groups`: two synthetic groups, each a distinct base shape with random PER-SERIES CIRCULAR SHIFTS + noise → k-Shape recovers them at purity 1.0 (the key gate; a wrong `Q`/centering or wrong eigenvector fails this).
- `test_kshape_centroid_sign`: the extracted centroid correlates POSITIVELY with its members (sign-fix correct) — corr > 0.99 on a clean single-shape cluster.
- `test_kshape_empty_cluster_recovery`: `k` > natural groups → valid labels, no panic.
- `test_kshape_deterministic`: same seed → identical `cluster`; sequential and parallel agree.
- `test_kshape_inertia_monotone` (or best-of-n_init): n_init>1 returns the min-inertia labeling.
- `test_kshape_predict`: `predict(train_data)` reproduces `cluster`; a new series near group A routes to A.
- `test_kshape_validation`: n_clusters=0 / >n / empty → errors.
- Doctest on `kshape_fd`.
</specifics>

<deferred>
## Deferred Ideas
- `sbd_kmedoids` convenience → Phase 63.
- Crate-root re-exports + `prelude` + criterion bench → Phase 63.
- Parallelizing restarts across threads while preserving determinism — perf refinement.
</deferred>
