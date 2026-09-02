# Requirements: fdars v0.33.0 — Shapelet Transform & Classification

**Defined:** 2026-09-02
**Core Value:** Add interpretable, discovery-based shapelet classification for curves/time series — discover discriminative subsequences, transform curves into a distance-feature space, and classify. Promotes GAP-02 (score 2.89, the only backlog gap corroborated across three reference libraries: sktime, pyts, tslearn) from the v0.31.0 `GAP-BACKLOG.md`.

## Milestone Requirements

Implementation milestone — real `fdars-core/src/` changes, additive/non-breaking, no new crate dependency. New `src/shapelet/` submodule. Discovery-based (Ye & Keogh 2009; Hills–Lines 2014; sktime `ShapeletTransformClassifier`, pyts `ShapeletTransform`) — **not** learning-shapelets (deferred). Strict compile-time dependency chain: distance core → discovery → transform → classifier.

**Resolved design decisions (from research):** bundled classifier defaults to **kNN** (`fclassif_knn_fit`, canonical Hills/Lines; avoids the FPCA-on-distance-features oddity of LDA), with LDA selectable via a `ShapeletClassifier` config enum. Quality measure defaults to **information gain**, with **F-statistic** selectable (`QualityMeasure` enum; reuses existing `pub(crate) integrated_f_statistic`). Early-abandon exposed as an explicit `best_so_far` parameter. Defaults follow sktime (`max_candidates`≈10000, `max_shapelets`≈min(10·n, 1000), min length 3); z-normalization uses population std (ddof=0, pyts convention) — divergences documented.

### Shapelet Distance Core

- [x] **SHP-01**: User/internal code can z-normalize a length-L window or shapelet slice (subtract mean, divide by population std) with a constant-window guard (std≈0 → zero vector, no NaN). Per-window normalization — the correctness foundation for scale/offset-invariant matching.
- [x] **SHP-02**: User/internal code can compute the shapelet-to-curve distance `sdist(S,T) = min over sliding windows of ‖z(window) − z(S)‖₂`, with an explicit `best_so_far` early-abandon parameter (inner loop breaks when the partial sum exceeds `best_so_far`). Returns the min distance (and best-match offset for interpretability).

### Discovery & Ranking

- [x] **SHP-03**: User can discover candidate shapelets from a labeled training curve set — enumerate candidate subsequences across a configurable length range, either exhaustively or via deterministic contracted/random sampling bounded by `max_candidates` (seeded via `seed_for_thread` for reproducibility).
- [x] **SHP-04**: Each candidate is scored by discriminative quality — **information gain** on the optimal distance-split threshold (orderline sort + midpoint scan) by default, or the **F-statistic** alternative, selectable via a `QualityMeasure` enum.
- [x] **SHP-05**: The discovery selects the top-K shapelets by quality with **self-similarity pruning** — candidates from the same series whose position range overlaps an already-selected shapelet are discarded, yielding a non-redundant `ShapeletSet`.

### Shapelet Transform

- [x] **SHP-06**: User can transform a curve set through a fitted `ShapeletSet` into an n×K distance-feature matrix (`X[i,j] = sdist(shapelet_j, curve_i)`), applying the identical shapelets and z-normalization to training and out-of-sample curves (transform consistency — `transform(train)` reproduces the fit-time distances).

### Bundled Classifier

- [x] **SHP-07**: User can fit an end-to-end `ShapeletTransformClassifier` (discover → transform → classify via an existing fdars classifier — kNN default, LDA optional) and `predict` labels for new curves, reusing the stored shapelets + inner classifier. Matches sktime's `ShapeletTransformClassifier` pipeline.

## Future Requirements

Deferred to a later milestone; tracked but not in this roadmap.

### Learning shapelets

- **LSH-01**: Gradient-learned shapelets (tslearn `LearningShapelets` / Grabocka 2014) — shapelets as model parameters optimized by SGD through a soft-min distance. Deferred; requires a differentiable distance (ties to GAP-08 autodiff core).

### Shapelet breadth

- **SHP-BREADTH**: Multivariate/multi-dimensional shapelets, DTW-based shapelet distance, and ROCKET-style convolutional-kernel alternatives — future.

## Out of Scope

Explicitly excluded for v0.33.0.

| Feature | Reason |
|---------|--------|
| Learning-shapelets (gradient) | Different paradigm needing autodiff through the distance; deferred to LSH-01 (ties to GAP-08). This milestone is discovery-based only. |
| GPU / batched acceleration | fdars targets a portable CPU/WASM numeric core (recorded OOS-01 in `GAP-BACKLOG.md`). |
| SAX / PAA / symbolic & imaging representations | Recorded OOS-02; time-series-ML representations, not the shapelet-numeric method. |
| Native RotationForest (sktime's inner classifier) | fdars reuses its existing kNN/LDA classifiers on the distance features; a new ensemble learner is out of scope. |
| Other GAP-BACKLOG items (GAP-03/05/06/07/08) | k-Shape, FOptDes, PEER, wavelet regression, differentiable core — carry forward, drawn top-first. |

## Traceability

Mapped during roadmap creation (2026-09-02). Every v0.33.0 requirement maps to exactly one phase — 100% coverage, no orphans, no duplicates. Phases 57–60 form a strict compile-time dependency chain (distance core → discovery → transform → classifier).

| Requirement | Phase | Status |
|-------------|-------|--------|
| SHP-01 | Phase 57 | Complete |
| SHP-02 | Phase 57 | Complete |
| SHP-03 | Phase 58 | Complete |
| SHP-04 | Phase 58 | Complete |
| SHP-05 | Phase 58 | Complete |
| SHP-06 | Phase 59 | Complete |
| SHP-07 | Phase 60 | Complete |

**Coverage:**

- Milestone requirements: 7 total
- Mapped to phases: 7 (Phase 57: SHP-01/02 · Phase 58: SHP-03/04/05 · Phase 59: SHP-06 · Phase 60: SHP-07)
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-02*
*Last updated: 2026-09-02 after roadmap creation (traceability populated, Phases 57–60)*
