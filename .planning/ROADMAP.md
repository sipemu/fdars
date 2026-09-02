# Roadmap: fdars

## Milestones

- ✅ **v0.14.0 — Performance & scikit-fda Gap Audit** — Phases 1–9 (shipped 2026-08-09) — [archive](milestones/v0.14.0-ROADMAP.md)
- ✅ **v0.15.0 — Top-Backlog Quick Wins** — Phases 10–11 (shipped 2026-08-11) — [archive](milestones/v0.15.0-ROADMAP.md)
- ✅ **v0.16.0 — Elastic Feasibility + Parity Quick Wins** — Phases 12–13 (shipped 2026-08-12, PR #40) — [archive](milestones/v0.16.0-ROADMAP.md)
- ✅ **v0.17.0 — Registration Parity & Elastic-FPCA Performance** — Phases 14–15 (shipped 2026-08-12, PR #41) — [archive](milestones/v0.17.0-ROADMAP.md)
- ✅ **v0.18.0 — R-Ecosystem Gap Audit** — Phases 16–19 (shipped 2026-08-15) — [archive](milestones/v0.18.0-ROADMAP.md)
- ✅ **v0.19.0 — Functional Inference Suite** — Phases 20–21 (shipped 2026-08-16) — [archive](milestones/v0.19.0-ROADMAP.md)
- ✅ **v0.20.0 — Table-Stakes Quick Wins** — Phases 22–23 (shipped 2026-08-16) — [archive](milestones/v0.20.0-ROADMAP.md)
- ✅ **v0.21.0 — Functional Regression Completeness** — Phases 24–25 (shipped 2026-08-17) — [archive](milestones/v0.21.0-ROADMAP.md)
- ✅ **v0.22.0 — PACE Sparse FPCA & Elastic Multinomial** — Phases 26–27 (shipped 2026-08-19) — [archive](milestones/v0.22.0-ROADMAP.md)
- ✅ **v0.23.0 — Depth, Outliers & Interval Inference** — Phases 28–30 (shipped 2026-08-20) — [archive](milestones/v0.23.0-ROADMAP.md)
- ✅ **v0.24.0 — Functional Regression & Clustering Breadth** — Phases 31–33 (shipped 2026-08-20) — [archive](milestones/v0.24.0-ROADMAP.md)
- ✅ **v0.25.0 — Serial Dependence, Representation & Density Breadth** — Phases 34–36 (shipped 2026-08-21) — [archive](milestones/v0.25.0-ROADMAP.md)
- ✅ **v0.26.0 — FPCA Breadth & Sparse Covariance** — Phases 37–38 (shipped 2026-08-21) — [archive](milestones/v0.26.0-ROADMAP.md)
- ✅ **v0.27.0 — Functional Time Series & Fréchet Regression** — Phases 39–40 (shipped 2026-08-22) — [archive](milestones/v0.27.0-ROADMAP.md)
- ✅ **v0.28.0 — Spectral Functional Time Series & Object-Data Fréchet Regression** — Phases 41–42 (shipped 2026-08-23) — [archive](milestones/v0.28.0-ROADMAP.md)
- ✅ **v0.29.0 — Boosting/Bayesian Regression, FEM/PDE Smoothing & Functional Co-Clustering** — Phases 43–45 (shipped 2026-08-30) — [archive](milestones/v0.29.0-ROADMAP.md)
- ✅ **v0.30.0 — Performance & Consolidation Pass** — Phases 46–51 (shipped 2026-09-01) — [archive](milestones/v0.30.0-ROADMAP.md)
- ✅ **v0.31.0 — Multi-Ecosystem Gap Audit** — Phases 52–53 (shipped 2026-09-02) — [archive](milestones/v0.31.0-ROADMAP.md)
- ✅ **v0.32.0 — Global Alignment Kernel & Kernel Clustering** — Phases 54–56 (shipped 2026-09-02) — [archive](milestones/v0.32.0-ROADMAP.md)
- ✅ **v0.33.0 — Shapelet Transform & Classification** — Phases 57–60 (shipped 2026-09-02) — [archive](milestones/v0.33.0-ROADMAP.md)
- 🚧 **v0.34.0 — k-Shape Clustering & Shape-Based Distance** — Phases 61–63 (in progress)

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3, …): Planned milestone work — numbering continues across milestones (never resets)
- Decimal phases (61.1, 61.2): Urgent insertions (marked with INSERTED)

<details>
<summary>✅ v0.30.0 — Performance & Consolidation Pass (Phases 46–51) — SHIPPED 2026-09-01</summary>

First internally-driven milestone (both parity backlogs exhausted): measure-first, behavior-preserving depth work. Phase 46 profiling produced three ranked inventories driving 47–51.

- [x] Phase 46: Whole-Crate Profiling & Measurement (PROF-01/02/03, 5 plans) — ranked hot-path/dedup/API inventories
- [x] Phase 47: Hot-Path & Allocation Performance (PERF-01/02, 4 plans) — face_covariance −80.7% wall, dpca −54% alloc blocks; bit-identical
- [x] Phase 48: Parallelism-Gap Closure (PERF-03, 3 plans) — frechet_anova 9.9×, co_cluster 6.4× thread-scaling; payback guards
- [x] Phase 49: Code Consolidation / Dedup (CONS-01/02, 5 plans) — χ²/gamma → distributions.rs, seed_for_thread, permutation_pvalue, SVD sign-core; −358 LOC; bit-identical
- [x] Phase 50: Additive API-Surface Consolidation (API-01/02/03, 3 plans) — 3 Default impls, fanova_seeded, Dim + 5 dispatchers, 6 #[deprecated]; 28 examples + wasm compile
- [x] Phase 51: Benchmark Coverage & Regression Guards (BENCH-01/02, 4 plans) — 9 new module benches + BENCH-RESULTS.md ledger

Milestone audit: **tech_debt** (13/13 requirements satisfied, 6/6 phases verified passed). Full detail: [milestones/v0.30.0-ROADMAP.md](milestones/v0.30.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.31.0 — Multi-Ecosystem Gap Audit (Phases 52–53) — SHIPPED 2026-09-02</summary>

Next-yardstick audit (both prior parity backlogs exhausted): map fdars against four fresh ecosystems and produce a single ranked, de-duplicated, GSD-ready backlog. **Audit-only** — zero `fdars-core/src/` edits, no crate change, no git tag.

- [x] Phase 52: Ecosystem Surveys (MAT-01/JUL-01/TDY-01/PYX-01, 4 plans) — capability-first surveys of MATLAB FDA, Julia FDA, tidyfun/refund, Python-beyond-scikit-fda → four `survey-*.md` with net-new gap lists (completed 2026-09-02)
- [x] Phase 53: Consolidation & Backlog (RPT-01/02/03, 3 plans) — `GAP-AUDIT-REPORT.md` + ranked `GAP-BACKLOG.md` (7 net-new, value/√effort) + RPT-03 completeness gate PASS (completed 2026-09-02)

Milestone audit PASSED 7/7 requirements. Outcome: 7 ranked net-new gaps (top: GAK, shapelets) + 3 recorded out-of-scope; headline = fdars is exceptionally comprehensive, cross-ecosystem convergence LOW. Deliverables in `.planning/research/GAP-AUDIT-REPORT.md` + `GAP-BACKLOG.md`. Full detail: [milestones/v0.31.0-ROADMAP.md](milestones/v0.31.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.32.0 — Global Alignment Kernel & Kernel Clustering (Phases 54–56) — SHIPPED 2026-09-02</summary>

First implementation milestone after three audit/consolidation cycles: a PSD Global Alignment Kernel for curve sets + the kernel machinery it unlocks. Promoted GAP-01 (top-ranked, score 3.00). Strictly sequential dependency spine; all algorithmic risk front-loaded into Phase 54. Real `fdars-core/src/` changes, additive/non-breaking, no new dependency; crate bumped 0.30.0 → 0.32.0, published on the `v0.32.0` tag.

- [x] Phase 54: GAK Kernel Core (GAK-01/02/03/04) — new `metric/gak.rs`; log-domain PSD Triangular GAK + `gak_gram_matrix` + `sigma_gak` (completed 2026-09-02)
- [x] Phase 55: Gram-Matrix Export (GAK-05/06) — split `gak_gram_train`/`gak_gram_predict` for external precomputed-kernel SVM (completed 2026-09-02)
- [x] Phase 56: Kernel-k-means Clustering (GAK-07/08) — new `kernel_kmeans.rs`; kernel-k-means on curves + out-of-sample `predict` (completed 2026-09-02)

Milestone audit PASSED 8/8 requirements. Full detail: [milestones/v0.32.0-ROADMAP.md](milestones/v0.32.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.33.0 — Shapelet Transform & Classification (Phases 57–60) — SHIPPED 2026-09-02</summary>

Interpretable, discovery-based shapelet classification for curves. Promoted GAP-02 (score 2.89, the only backlog gap corroborated across sktime + pyts + tslearn). New `src/shapelet/` submodule along a strict compile-time dependency chain (distance core → discovery → transform → classifier). Real `fdars-core/src/` changes, additive/non-breaking, no new crate dependency; crate bumped 0.32.0 → 0.33.0, published on the `v0.33.0` tag.

- [x] Phase 57: Shapelet Distance Core (SHP-01/02) — new `src/shapelet/distance.rs`; per-window z-normalization + min sliding-window `sdist` with early-abandon + the `Shapelet` type (completed 2026-09-02)
- [x] Phase 58: Discovery & Ranking (SHP-03/04/05) — candidate generation (exhaustive + contracted/seeded), info-gain / F-statistic quality, top-K + self-similarity pruning → `ShapeletSet` (completed 2026-09-02)
- [x] Phase 59: Shapelet Transform (SHP-06) — fitted `ShapeletSet` → n×K distance-feature matrix (train + out-of-sample), transform consistency (completed 2026-09-02)
- [x] Phase 60: Bundled ShapeletTransformClassifier (SHP-07) — end-to-end `fit` (discover → transform → classify; kNN default, LDA optional) + `predict`; crate-root re-exports (completed 2026-09-02)

Milestone audit PASSED 7/7 requirements. Full detail: [milestones/v0.33.0-ROADMAP.md](milestones/v0.33.0-ROADMAP.md)

</details>

### 🚧 v0.34.0 — k-Shape Clustering & Shape-Based Distance (In Progress)

**Milestone Goal:** Add shape-based curve clustering — the SBD (Shape-Based Distance) primitive and the k-Shape algorithm built on it — plus out-of-sample assignment and SBD as a distance backend for existing k-medoids. Promotes GAP-03 (score 2.12, M-effort) from the v0.31.0 `GAP-BACKLOG.md`, rounding out the curve-clustering family alongside v0.32.0's GAK kernel-k-means. Reference baseline: tslearn `KShape`; Paparrizos & Gravano 2015 (k-Shape, SBD). Implementation milestone — real `fdars-core/src/` changes, additive/non-breaking, no new crate dependency; **publishes to crates.io on the `v0.34.0` tag**.

**Phase shape (three phases — a strict sequential dependency chain):** All four researchers converged on a rigid, non-reorderable, non-parallelizable build sequence — SBD distance core → k-Shape fit + predict → SBD-k-medoids + wrap-up — mirroring the v0.32.0 GAK→kernel-k-means precedent. SBD lands in a new `src/metric/sbd.rs` (peer of `gak.rs`/`soft_dtw.rs`); k-Shape in a new top-level `src/kshape.rs` (peer of `kernel_kmeans.rs`); the SBD-k-medoids convenience is a thin adapter at the bottom of `kshape.rs`. `granularity: fine` + disjoint per-phase correctness gates → three phases, each owning a distinct set of silent-correctness gates. **Phase 61** front-loads the five FFT/NCC numerical make-or-break gates (zero-pad to `next_power_of_two(2m−1)`, coefficient-normalized NCC, IFFT scale, signed-lag extraction, z-normalization) — every downstream step inherits any bug here. **Phase 62** is the main algorithmic phase — shape-extraction centroids (top eigenvector, sign fix, shift-alignment, re-z-norm) are the only genuinely-new numerical piece; everything else mirrors `kernel_kmeans.rs`. **Phase 63** is the thin SBD-k-medoids adapter + crate-root re-exports + benchmark. Reuse-first: `rustfft` (FFT), `nalgebra` `SymmetricEigen` (shape extraction), the v0.33.0 `shapelet::z_normalize_window`, existing `kernel_kmeans.rs` patterns, and `alignment::clustering::kmedoids_from_distances`. Multivariate SBD, variable-length series, and other clustering families (KSH-BREADTH) deferred.

- [x] **Phase 61: SBD Distance Core** - FFT normalized-cross-correlation Shape-Based Distance `sbd(x,y) -> (distance, shift)` + public n×n `sbd_distance_matrix`; new `src/metric/sbd.rs` (KSH-01/02) (completed 2026-09-02)
- [ ] **Phase 62: k-Shape Clustering & Predict** - `kshape_fd` (SBD assignment + shape-extraction centroids, n_init restarts, empty-cluster recovery, deterministic seeding) + `KShapeResult::predict`; new `src/kshape.rs` (KSH-03/04)
- [ ] **Phase 63: SBD-based k-medoids & Wrap-up** - `sbd_kmedoids` convenience over the existing `kmedoids_from_distances`; crate-root re-exports + `prelude` + criterion benchmark (KSH-05)

## Phase Details

### Phase 61: SBD Distance Core

**Goal**: Users (and all downstream k-Shape / k-medoids code) can compute the Shape-Based Distance between two equal-length series and build an n×n SBD distance matrix over a curve set — the atomic shape-invariant primitive every later phase consumes. New `src/metric/sbd.rs` (peer of `gak.rs`/`soft_dtw.rs`), pure `&[f64]` FFT arithmetic reusing `rustfft` + `shapelet::z_normalize_window`; no new dependency. Lowest-level phase, but its five FFT/NCC numerical gates are make-or-break — every SBD-dependent result inherits any bug here.
**Depends on**: Nothing (first phase of the milestone; builds only on existing `rustfft`, `shapelet::distance::z_normalize_window`/`z_normalize_into`, `metric::self_distance_matrix`, `matrix.rs` `FdMatrix`/`row_to_buf`, `error.rs`)
**Requirements**: KSH-01, KSH-02
**Success Criteria** (what must be TRUE):

  1. User can call `sbd(x, y) -> Result<(distance, optimal_shift), FdarError>`: both series are z-normalized inside the function (via `shapelet::z_normalize_window`, never trusting the caller), the FFT is zero-padded to `next_power_of_two(2m−1)` (a too-short FFT wraps cross-correlation circularly), the unnormalized rustfft IFFT output is explicitly divided by `fft_len`, NCC is coefficient-normalized by `‖x‖·‖y‖`, and `distance = 1 − max_w NCCc_w` lies in `[0,2]` with the optimal cyclic lag `w*` returned alongside.
  2. The core FFT/NCC correctness gates pass: `sbd(x, x) ≈ 0` (self-distance, exercises IFFT-scale + coefficient-normalization together), `sbd(x, y) == sbd(y, x)` within 1e-10 (symmetry), a right-shifted copy of `x` has `sbd ≈ 0` at the correct signed shift (not a `fft_len − k` wrap-around; exercises zero-padding + signed-lag extraction), and every NCC value lies in `[−1, 1]` within tolerance.
  3. SBD is shape-invariant: `sbd(x, x + c) ≈ 0` for any offset `c` and `sbd(x, a·x) ≈ 0` for any scale `a` (within 1e-10) — the z-normalization gate — and a constant-series input (std ≈ 0, z-norm yields the zero vector, `‖x‖·‖y‖ ≈ 0`) is guarded to return a defined distance (1.0) and shift 0 rather than NaN.
  4. User can call the public `sbd_distance_matrix(data: &FdMatrix) -> Result<FdMatrix, FdarError>` to get an n×n matrix that is symmetric with a zero diagonal, parallelized via `iter_maybe_parallel!` with each rayon task building its own `FftPlanner` (which is `!Send`) — and its output equals the pairwise `sbd` distances computed independently.

**Plans**: TBD

### Phase 62: k-Shape Clustering & Predict

**Goal**: Users can cluster a curve set with k-Shape and assign new out-of-sample curves to the fitted model. New top-level `src/kshape.rs` (peer of `kernel_kmeans.rs`) implementing the iterative SBD-assignment + shape-extraction-centroid algorithm with `n_init` restarts, empty-cluster recovery, deterministic seeding, and an out-of-sample `predict`. This is the main algorithmic phase — shape extraction (top eigenvector of the shift-aligned mean-centered normalized covariance) is the only genuinely-new numerical piece; the restart/seeding/recovery/predict scaffolding mirrors `kernel_kmeans.rs`.
**Depends on**: Phase 61 (`metric::sbd::{sbd, sbd_distance_matrix}` and the returned optimal shift are required for both assignment and centroid shift-alignment)
**Requirements**: KSH-03, KSH-04
**Success Criteria** (what must be TRUE):

  1. User can call `kshape_fd(data, config) -> Result<KShapeResult, FdarError>` to cluster a curve set: it runs iterative SBD assignment + shape-extraction centroid refinement to convergence or `max_iter`, over `n_init` random restarts (default 10, an fdars convention exceeding tslearn's 1), keeping the best restart by inertia, and returns a result carrying centroids + labels + inertia + iteration count. Standard validation errors (`n_clusters = 0`, `n_clusters > n`, `n_init = 0`, empty data) return `Err(FdarError::...)`.
  2. Centroids are correct k-Shape shape prototypes, not k-means means: each centroid is the **top** eigenvector (largest eigenvalue; nalgebra `SymmetricEigen` returns ascending, so take the last column) of `M = Qᵀ Sᵀ S Q` with `Q = I − O/n_k`, built from members that are first shift-aligned (using the SBD-returned shift) and z-normalized, then sign-fixed to correlate positively with the members and re-z-normalized before storage. On a synthetic two-shifted-motif dataset, `kshape_fd` recovers the two groups at high purity and each centroid correlates > 0.99 with its group prototype (up to the resolved sign).
  3. The run is robust and reproducible: an empty cluster mid-iteration is recovered in place by farthest-point reassignment (mirroring `kernel_kmeans.rs`, a documented divergence from tslearn's full restart) so a `k > natural-clusters` fit never panics and every cluster size ≥ 1; inertia is non-increasing across iterations on clean data; and deterministic per-restart seeding (`seed.wrapping_add(restart_idx)`) makes two same-config fits produce byte-identical labels and inertia, with sequential and parallel paths agreeing.
  4. User can call `KShapeResult::predict(new_data) -> Result<Vec<usize>, FdarError>` to assign out-of-sample series: each new series is z-normalized and SBD-compared against each stored (already-z-normalized) centroid, taking the argmin — with centroids used as-is (no re-estimation), so `predict(train_data)` reproduces the training labels exactly.

**Plans**: TBD

### Phase 63: SBD-based k-medoids & Wrap-up

**Goal**: Users can cluster a curve set with k-medoids over the SBD distance — a shape-based clustering consumer distinct from k-Shape — and the full v0.34.0 public surface is exposed at the crate root with a benchmark. `sbd_kmedoids` is a thin convenience (build the SBD distance matrix, feed the existing `kmedoids_from_distances`) at the bottom of `kshape.rs`; this phase also lands the crate-root `pub mod kshape` + `metric::sbd` re-exports, `prelude` additions, and a criterion benchmark. Deferred to the final phase so no partial public API is exposed mid-milestone.
**Depends on**: Phase 62 (crate-root re-exports cover the full `kshape` surface; the wrap-up finalizes the whole milestone's public API) and Phase 61 (`sbd_distance_matrix` feeds the k-medoids adapter); reuses `alignment::clustering::{kmedoids_from_distances, KMedoidsConfig, KMedoidsResult}` unchanged
**Requirements**: KSH-05
**Success Criteria** (what must be TRUE):

  1. User can call `sbd_kmedoids(data, config) -> Result<KMedoidsResult, FdarError>` — a real public function (not merely a doc example) that builds the SBD distance matrix (KSH-02) and feeds it to the existing `kmedoids_from_distances`, returning the standard k-medoids result (medoid indices + labels) as an alternative shape-based clustering path distinct from k-Shape.
  2. `sbd_kmedoids` provably uses the SBD distance (not L2/DTW): an integration test confirms its output equals calling `sbd_distance_matrix` + `kmedoids_from_distances` independently, and a doctest shows the explicit SBD-matrix → k-medoids flow so the shape-based distance backend is unambiguous.
  3. The full v0.34.0 public surface is re-exported at the crate root additively (`pub mod kshape`; `kshape_fd`, `KShapeConfig`, `KShapeResult`, `sbd_kmedoids`; `metric::{sbd, sbd_distance_matrix, SbdResult}`), `prelude` gains `KShapeConfig`/`KShapeResult`, and the crate compiles non-breaking — 28 examples + WASM + R bindings unaffected, no existing public signature changed, no new crate dependency.
  4. A criterion benchmark for the SBD/k-Shape pipeline is added (pairwise SBD + `kshape_fd` on a small n×m×k grid) and whole-crate gates pass: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, and the full lib + doctest suite (including the new SBD/k-Shape/k-medoids tests).

**Plans**: TBD

## Progress

**Execution Order (dependency-driven — strict chain, no reordering or parallelization):**
Phases execute in numeric order: 61 → 62 → 63. Phase 62 cannot begin before Phase 61's `sbd`/`sbd_distance_matrix` (and the returned optimal shift) compile; Phase 63's re-exports and `sbd_kmedoids` adapter depend on both. Phase 61 front-loads the five FFT/NCC numerical make-or-break gates; Phase 62 is the main algorithmic phase (shape-extraction centroids); Phase 63 is the thin k-medoids adapter + crate-root re-exports + benchmark.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 61. SBD Distance Core | v0.34.0 | 1/1 | Complete    | 2026-09-02 |
| 62. k-Shape Clustering & Predict | v0.34.0 | 0/TBD | Not started | - |
| 63. SBD-based k-medoids & Wrap-up | v0.34.0 | 0/TBD | Not started | - |

## Status

All milestones through **v0.33.0 are shipped and archived** under `milestones/`. Milestone **v0.34.0** (Phases 61–63) is the active implementation milestone — it promotes GAP-03 (k-Shape clustering & Shape-Based Distance) out of `.planning/research/GAP-BACKLOG.md` and **will** bump the crate + publish on the `v0.34.0` tag. The remaining four backlog items (GAP-05/06/07/08 — FOptDes, PEER, wavelet regression, differentiable core) carry forward, drawn top-first.

Next: `/gsd-plan-phase 61`
