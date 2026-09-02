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
- 🚧 **v0.33.0 — Shapelet Transform & Classification** — Phases 57–60 (in progress)

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3, …): Planned milestone work — numbering continues across milestones (never resets)
- Decimal phases (57.1, 57.2): Urgent insertions (marked with INSERTED)

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

### 🚧 v0.33.0 — Shapelet Transform & Classification (In Progress)

**Milestone Goal:** Add interpretable, discovery-based shapelet classification for curves/time series — discover discriminative subsequences from labeled training curves, transform curves into a distance-feature space, and classify. Promotes GAP-02 (score 2.89, the only backlog gap corroborated across three reference libraries: sktime, pyts, tslearn) from the v0.31.0 `GAP-BACKLOG.md`. Implementation milestone — real `fdars-core/src/` changes in a new `src/shapelet/` submodule, additive/non-breaking, no new crate dependency; **publishes to crates.io on the `v0.33.0` tag**.

**Phase shape (four phases — a strict compile-time dependency chain):** All four researchers converged on a rigid, non-reorderable, non-parallelizable build sequence: distance core → discovery & ranking → transform → bundled classifier. Each of the four `src/shapelet/` files depends one-way on the previous, so the phase boundaries mirror the file boundaries exactly. Four phases (rather than one phase / four plans) fits `granularity: fine`, and each phase owns a distinct, disjoint set of correctness gates: Phase 57 front-loads the two make-or-break numerical gates (per-window z-normalization scale/offset-invariance; the min-not-mean semantics via known-motif recovery); Phase 58 is the highest-risk phase (most pitfalls — combinatorial tractability, optimal-split information gain, self-similarity pruning, deterministic seeding, float-tie ordering); Phase 59 owns transform consistency + the short-series guard; Phase 60 owns the end-to-end pipeline + train/test leakage discipline. The crate-root `pub mod shapelet` + re-exports are deferred to the final phase to avoid exposing a partial public API mid-milestone. Discovery-based only (Ye & Keogh 2009; Hills–Lines 2014); learning-shapelets (LSH-01) deferred.

- [x] **Phase 57: Shapelet Distance Core** - Per-window z-normalization + min sliding-window z-normalized Euclidean `sdist` with early-abandon and the `Shapelet` type (SHP-01/02) (completed 2026-09-02)
- [ ] **Phase 58: Discovery & Ranking** - Candidate generation (exhaustive + contracted/seeded), information-gain / F-statistic quality scoring, top-K + self-similarity pruning → `ShapeletSet` (SHP-03/04/05)
- [ ] **Phase 59: Shapelet Transform** - Apply a fitted `ShapeletSet` → n×K distance-feature matrix for training and out-of-sample curves, with transform consistency (SHP-06)
- [ ] **Phase 60: Bundled ShapeletTransformClassifier** - End-to-end `fit` (discover → transform → classify; kNN default, LDA optional) + `predict`; crate-root re-exports (SHP-07)

## Phase Details

### Phase 57: Shapelet Distance Core

**Goal**: Users (and downstream shapelet code) can compute the shapelet-to-curve distance `sdist` — the atomic primitive every later phase consumes — with correct per-window z-normalization and early-abandon. New `src/shapelet/distance.rs` + `src/shapelet/mod.rs` skeleton; pure `&[f64]` arithmetic, no new dependency. Lowest-risk phase, but the two numerical gates here are make-or-break — every downstream step inherits any bug.
**Depends on**: Nothing (first phase of the milestone; builds only on existing `matrix.rs` `FdMatrix`/`row_to_buf` + `error.rs`)
**Requirements**: SHP-01, SHP-02
**Success Criteria** (what must be TRUE):

  1. User/internal code can z-normalize a length-L window slice (subtract mean, divide by population std, ddof=0) with a constant-window guard: a constant or near-constant window (std clamped at 1e-10) returns a finite zero-ish vector, never NaN/Inf — verified by constant-window and one-element-perturbed-by-1e-15 tests.
  2. `sdist(S, T)` is the **minimum** over sliding windows of the z-normalized Euclidean distance between the (pre-normalized) shapelet and each per-window-normalized window — asserted **scale- and offset-invariant**: `sdist(S, T) == sdist(S, T + c) == sdist(S, T * a)` within 1e-10 (the make-or-break per-window-normalization gate).
  3. A known-motif recovery test passes: on a synthetic dataset (class_0 = noise, class_1 = noise + a planted length-L motif at a random offset), the shapelet equal to the motif achieves `sdist ≈ 0` on class_1 curves and `sdist` well above threshold on class_0 curves — proving min-not-mean semantics.
  4. `sdist` accepts an explicit `best_so_far` early-abandon parameter: the sequential inner loop short-circuits when the running partial sum-of-squares exceeds `best_so_far`, the answer is identical to the non-abandoned computation, and a benchmark on a non-matching shapelet shows a measurable speedup.
  5. `sdist` returns the minimum distance and the best-match offset (for interpretability), and the `Shapelet` type stores its z-normalized values, source curve, start position, length, and score.

**Plans**: TBD

### Phase 58: Discovery & Ranking

**Goal**: Users can discover a non-redundant `ShapeletSet` from a labeled training curve set — enumerate candidates tractably, score them by discriminative quality, and select the top-K with self-similarity pruning. New `src/shapelet/discovery.rs` (`ShapeletConfig`, `QualityMeasure`). The highest-risk phase — it owns the most pitfalls; the combinatorial contract must be designed into the API from the start, not bolted on.
**Depends on**: Phase 57
**Requirements**: SHP-03, SHP-04, SHP-05
**Success Criteria** (what must be TRUE):

  1. User can discover candidate shapelets across a configurable length range either exhaustively or via deterministic contracted/random sampling bounded by `max_candidates` — and discovery stays tractable: an `n=100, m=200, max_candidates≈1000` fit returns in well under 10 seconds (naive O(n²·M³) exhaustive search is intractable and must never be the only path).
  2. Each candidate is scored by discriminative quality via an **optimal** distance-split threshold: information gain (orderline sort + all-n−1-gap midpoint scan) by default, or the F-statistic alternative, selectable via a `QualityMeasure` enum — a known-discriminative shapelet scores near max entropy while a random candidate scores near 0 (no fixed-threshold shortcut).
  3. The discovery selects the top-K by quality with **self-similarity pruning**: candidates from the same source series whose position range overlaps an already-selected shapelet are dropped, so the selected K span at least `min(K, n_train)` distinct source series and no two transform columns correlate above ~0.95.
  4. Discovery is reproducible: contracted/random candidate sampling is seeded (`config.seed`, `seed_for_thread` for any parallel sub-steps), and two fits with the same config produce byte-identical `Shapelet` sets and thresholds — ranking uses `total_cmp` with a `(series_idx, start_offset)` tie-break (never `partial_cmp(...).unwrap()`), so float ties order deterministically.

**Plans**: TBD

### Phase 59: Shapelet Transform

**Goal**: Users can transform a curve set through a fitted `ShapeletSet` into an n×K distance-feature matrix — applying the identical stored shapelets and z-normalization to both training and out-of-sample curves. New `src/shapelet/transform.rs` (`ShapeletTransformFit`, `shapelet_transform_fit`, `shapelet_transform`); crate-root transform re-exports land here.
**Depends on**: Phase 58
**Requirements**: SHP-06
**Success Criteria** (what must be TRUE):

  1. User can call `shapelet_transform_fit(data, y, config)` to discover shapelets and get back an `n×K` `FdMatrix` where `X[i,j] = sdist(shapelet_j, curve_i)`, alongside the stored (already-z-normalized) `Shapelet` set for reuse.
  2. User can call `shapelet_transform(fit, new_data)` to produce an `n_new×K` feature matrix for out-of-sample curves, applying the exact stored shapelets and stored normalization — no re-discovery, no re-normalization against test-set statistics.
  3. Transform consistency holds: re-transforming the training data reproduces the fit-time distances exactly (each `X[i,j]` matches `sdist(shapelet_j, train_curve_i)` within 1e-12, and two `transform(train)` calls are bit-identical).
  4. Every transform output is finite (an `all(|v| v.is_finite())` assertion passes on all test inputs), and a curve shorter than the minimum shapelet length returns `Err(FdarError::InvalidDimension)` rather than a silent INFINITY row.

**Plans**: TBD

### Phase 60: Bundled ShapeletTransformClassifier

**Goal**: Users can fit an end-to-end `ShapeletTransformClassifier` (discover → transform → classify) and predict labels for new curves — matching sktime's `ShapeletTransformClassifier` pipeline. New `src/shapelet/classifier.rs`; the final crate-root `pub mod shapelet` + all re-exports land here (deferred to this phase to avoid partial public API exposure). Reuses existing `classification/` classifiers as a consumer; `classification/` is left unmodified.
**Depends on**: Phase 59
**Requirements**: SHP-07
**Success Criteria** (what must be TRUE):

  1. User can fit a `ShapeletTransformClassifier` on labeled curves — the pipeline discovers shapelets, transforms to the n×K distance-feature matrix, and trains an inner fdars classifier (kNN default via `fclassif_knn_fit`, LDA selectable via a `ShapeletClassifier` config enum) — returning a result that stores the fitted shapelets + inner classifier.
  2. User can `predict` labels for new curves: the stored shapelets transform the new curves to distance features, and the stored inner classifier produces class labels reusing the identical shapelets + normalization.
  3. Train/test discipline is enforced: the rustdoc example and integration test train on a training split and evaluate on a held-out split (accuracy above chance), and training-set accuracy is never presented as a generalization estimate.
  4. The full public shapelet surface is re-exported at the crate root (`pub mod shapelet` + `Shapelet`, `ShapeletConfig`, `QualityMeasure`, `ShapeletTransformFit`, the classifier config/result, and the `fit`/`transform`/`predict` functions), the crate compiles additively/non-breaking (28 examples + WASM + R bindings unaffected), and a criterion benchmark for the pipeline is added.

**Plans**: TBD

## Progress

**Execution Order (dependency-driven — strict chain, no reordering or parallelization):**
Phases execute in numeric order: 57 → 58 → 59 → 60. Each `src/shapelet/` file depends one-way on the previous, so no phase can begin before its predecessor compiles. Phase 57 front-loads the two numerical make-or-break gates; Phase 58 is the highest-risk phase (most pitfalls); Phase 59 is transform consistency; Phase 60 is the user-facing pipeline + crate-root re-exports.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 57. Shapelet Distance Core | v0.33.0 | 1/1 | Complete    | 2026-09-02 |
| 58. Discovery & Ranking | v0.33.0 | 0/TBD | Not started | - |
| 59. Shapelet Transform | v0.33.0 | 0/TBD | Not started | - |
| 60. Bundled ShapeletTransformClassifier | v0.33.0 | 0/TBD | Not started | - |

## Status

All milestones through **v0.32.0 are shipped and archived** under `milestones/`. The crate is at version 0.32.0. Milestone **v0.33.0** (Phases 57–60) is the active implementation milestone — it promotes GAP-02 (discovery-based shapelet transform & classification) out of `.planning/research/GAP-BACKLOG.md` and **will** bump the crate + publish on the `v0.33.0` tag. The remaining five backlog items (GAP-03/05/06/07/08) carry forward.

Next: `/gsd-plan-phase 57`
