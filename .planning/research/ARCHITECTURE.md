# Architecture Research

**Domain:** Shapelet Transform & Classification — fdars-core v0.33.0
**Researched:** 2026-09-02
**Confidence:** HIGH

---

## Integration Decision: Module Placement

**Decision: New top-level `src/shapelet/` submodule directory.**

Rationale:

- The shapelet pipeline has four distinct algorithmic layers (distance core, candidate enumeration, discovery/ranking, transform + classifier). At four files minimum, it crosses the 500-line file threshold immediately (shapelet distance + z-norm alone ~150 lines; discovery ~300 lines; bundled STC ~250 lines). Start as a multi-file submodule from day one.
- Shapelets are a distinct algorithmic family (discriminative subsequences → distance features → classification) that does not belong inside `classification/`. The existing `classification/` module implements FPC-space classifiers; shapelets produce a feature matrix *consumed by* those classifiers, not implemented alongside them. Coupling shapelet transform logic into `classification/` would violate component separation and complicate future extension (e.g., GAP-03 k-Shape has a similar consumer relationship).
- Every prior module of comparable scope in fdars uses the submodule pattern: `alignment/`, `depth/`, `spm/`, `seasonal/`, `elastic_regression/`. The shapelet module follows suit.
- `classification/` is left **unmodified** — the bundled `ShapeletTransformClassifier` calls `fclassif_lda_fit` from there as a consumer, not a peer.

---

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Public API Layer                                      │
│  lib.rs re-exports: Shapelet, ShapeletConfig, ShapeletTransformConfig,       │
│  ShapeletTransformFit, ShapeletClassifConfig, ShapeletClassifResult,         │
│  ShapeletScorer, shapelet_transform_fit, shapelet_transform,                 │
│  shapelet_classif_fit, shapelet_classif_predict                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                         src/shapelet/                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  ┌─────────────────┐   │
│  │  distance.rs │  │  discovery.rs│  │transform.rs│  │  classifier.rs  │   │
│  │  (sdist,     │  │  (enumerate, │  │  (fit +    │  │  (STC fit,      │   │
│  │  z-norm      │  │  rank, prune │  │  transform │  │  predict,       │   │
│  │  per window) │  │  candidates) │  │  matrix)   │  │  wraps classif/)│   │
│  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  └────────┬────────┘   │
│         │                 │                │                  │             │
│         └─────────────────┴────────────────┴──────────────────┘             │
│                           mod.rs (re-exports)                               │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
  src/classification/       src/distance.rs           src/helpers.rs
  fclassif_lda_fit()        cross_distance_matrix()   l2_distance()
  ClassifFit, ClassifResult pairwise_distance_matrix()  (no z-norm today — new)
  (consumed by STC,         iter_maybe_parallel! macro
   not modified)            (via distance.rs internals)

          ▼
  src/parallel.rs           src/matrix.rs              src/error.rs
  iter_maybe_parallel!      FdMatrix, row_l2_sq()       FdarError
  seed_for_thread()         row_to_buf()
```

---

## Component Responsibilities

| Component | File | Responsibility | New vs Existing |
|-----------|------|----------------|-----------------|
| Shapelet distance core | `src/shapelet/distance.rs` | Sliding-window z-normalized minimum Euclidean distance between a shapelet (subsequence) and a curve, with early-abandon. Also: z-normalize a window in place. | **New** |
| Candidate enumeration | `src/shapelet/discovery.rs` | Generate all candidate subsequences of lengths `[min_len, max_len]` from training set; score each by information gain (binary class split on distance threshold) or F-statistic (multi-class); self-similarity pruning (remove candidates whose shapelet-distance to an already-selected shapelet is below a correlation threshold); select top-K. Random/contracted search: sample `max_candidates` candidates with `seed_for_thread` seeding. | **New** |
| Shapelet transform | `src/shapelet/transform.rs` | Given a ranked list of `Shapelet` values, compute the n×K `FdMatrix` of minimum z-normalized distances from each curve to each shapelet. Both a fit-time path (returns `ShapeletTransformFit`) and a predict-time path (transform new curves using stored shapelets). | **New** |
| Bundled STC | `src/shapelet/classifier.rs` | `ShapeletTransformClassifier`: wraps the full pipeline — calls `discovery.rs` then `transform.rs` then delegates to `fclassif_lda_fit` (from `src/classification/fit.rs`). `fit` returns `ShapeletClassifResult`; `predict` projects new curves through stored shapelets, then applies the stored `ClassifFit`. | **New** |
| Module barrel | `src/shapelet/mod.rs` | `pub use` all public items; declares the four submodules. | **New** |
| `classification/` module | `src/classification/` | FPC-space LDA/QDA/kNN/kernel/DD classifiers. **Consumed by but not modified for this milestone.** `fclassif_lda_fit` is the bundled STC's default downstream classifier. | **Existing — unmodified** |
| `distance.rs` | `src/distance.rs` | `pairwise_distance_matrix`, `cross_distance_matrix` — reused for STC predict path (cross-distance from new curves to stored shapelets). | **Existing — unmodified** |
| `helpers.rs` | `src/helpers.rs` | `l2_distance`, `seed_for_thread` (via `parallel.rs`). **No z-norm exists today** — z-normalize window logic is new in `src/shapelet/distance.rs`. | **Existing — unmodified** |
| `matrix.rs` | `src/matrix.rs` | `FdMatrix`, `row_l2_sq`, `row_to_buf` — hot-path row access for shapelet distance inner loops. | **Existing — unmodified** |
| `parallel.rs` | `src/parallel.rs` | `iter_maybe_parallel!`, `seed_for_thread` — candidate scan parallelism + deterministic RNG for contracted search. | **Existing — unmodified** |
| `error.rs` | `src/error.rs` | `FdarError` — all new public functions return `Result<_, FdarError>`. | **Existing — unmodified** |

---

## Reuse Map

### What is reused (no changes needed)

| Existing item | Location | How shapelet code uses it |
|---------------|----------|--------------------------|
| `FdMatrix` | `matrix.rs` | Primary data carrier for training curves, transform output (n×K), shapelet storage |
| `row_l2_sq(i, other, j)` | `matrix.rs` | Inner loop of `sdist` for raw squared Euclidean distance between windows |
| `row_to_buf(row, buf)` | `matrix.rs` | Materialize a curve row into a stack buffer for window slicing |
| `cross_distance_matrix` | `distance.rs` | STC predict path: compute n_new × K distances from new curves to K stored shapelets |
| `iter_maybe_parallel!` | `parallel.rs` | Outer loop over candidate (curve, position, length) triples in the O(n·M²) candidate scan; outer loop over new curves in predict |
| `seed_for_thread(seed, k)` | `parallel.rs` | Per-thread RNG for contracted/random shapelet search (same pattern as elastic alignment) |
| `FdarError` | `error.rs` | Return type for all public functions |
| `fclassif_lda_fit(data, y, None, ncomp)` | `classification/fit.rs` | Bundled STC fits LDA on the shapelet feature matrix; returns `ClassifFit` stored in `ShapeletClassifResult` |
| `ClassifFit` | `classification/fit.rs` | Stored in `ShapeletClassifResult`; its `ClassifMethod::Lda` parameters are used directly in the STC predict path |
| `ClassifResult` | `classification/mod.rs` | Embedded in `ShapeletClassifResult` for training accuracy, confusion matrix |

### What is new (net-new code in `src/shapelet/`)

| New item | Where | Purpose |
|----------|-------|---------|
| `z_normalize_window(slice) -> Vec<f64>` | `shapelet/distance.rs` | Z-normalize a subsequence (mean 0, std 1); handles near-zero std gracefully (returns zeros) |
| `shapelet_distance(shapelet, curve) -> f64` | `shapelet/distance.rs` | Sliding-window min z-normalized Euclidean distance; early-abandon on running squared-diff sum exceeding current best |
| `Shapelet` struct | `shapelet/distance.rs` | `{ values: Vec<f64>, source_curve: usize, start_pos: usize, length: usize, score: f64 }` |
| Candidate generation | `shapelet/discovery.rs` | Enumerate all (curve i, position p, length l) triples for l in `[min_len, max_len]`; for contracted search, sample up to `max_candidates` using seeded RNG |
| Information-gain scorer | `shapelet/discovery.rs` | For each candidate: compute distance to all n curves → 1-D split → binary entropy IG or F-statistic; pick optimal split threshold |
| Self-similarity pruner | `shapelet/discovery.rs` | After ranking, greedily remove candidates whose shapelet-distance to any already-selected shapelet is below `similarity_threshold`; ensures diversity |
| `ShapeletConfig` | `shapelet/discovery.rs` | Config for discovery: `min_len`, `max_len`, `n_shapelets` (K), `max_candidates` (contracted search), `similarity_threshold`, `scorer`, `seed: u64` |
| `ShapeletScorer` enum | `shapelet/discovery.rs` | `InfoGain` / `FStat` |
| `ShapeletTransformFit` | `shapelet/transform.rs` | Result of `shapelet_transform_fit`: `{ shapelets: Vec<Shapelet>, train_features: FdMatrix }` |
| `shapelet_transform_fit(data, y, config) -> Result<ShapeletTransformFit, FdarError>` | `shapelet/transform.rs` | Discover shapelets then compute n×K feature matrix for training data |
| `shapelet_transform(fit, new_data) -> Result<FdMatrix, FdarError>` | `shapelet/transform.rs` | Out-of-sample transform: compute n_new×K feature matrix from stored shapelets |
| `ShapeletClassifConfig` | `shapelet/classifier.rs` | Config for bundled STC: `shapelet: ShapeletConfig`, `ncomp: usize` (for downstream LDA) |
| `ShapeletClassifResult` | `shapelet/classifier.rs` | `{ transform: ShapeletTransformFit, classif: ClassifFit, training_accuracy: f64, n_classes: usize }` |
| `shapelet_classif_fit(data, y, config) -> Result<ShapeletClassifResult, FdarError>` | `shapelet/classifier.rs` | Full pipeline: discover → transform → `fclassif_lda_fit` |
| `shapelet_classif_predict(fit, new_data) -> Result<ClassifResult, FdarError>` | `shapelet/classifier.rs` | Project new curves through stored shapelets → LDA predict |

---

## Proposed Public API

All types derive `Debug, Clone, PartialEq`. Serde derives are conditional on `#[cfg_attr(feature = "serde", derive(...))]`. All expensive computations carry `#[must_use]`.

```rust
// shapelet/discovery.rs

#[derive(Debug, Clone, PartialEq)]
pub enum ShapeletScorer { InfoGain, FStat }

#[derive(Debug, Clone, PartialEq)]
pub struct ShapeletConfig {
    pub min_len: usize,
    pub max_len: usize,        // 0 = resolve to m at runtime
    pub n_shapelets: usize,    // K
    pub max_candidates: usize, // 0 = exhaustive
    pub similarity_threshold: f64,
    pub scorer: ShapeletScorer,
    pub seed: u64,
}

// shapelet/distance.rs

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Shapelet {
    pub values: Vec<f64>,      // z-normalized subsequence values
    pub source_curve: usize,
    pub start_pos: usize,
    pub length: usize,
    pub score: f64,            // discriminative score, higher = better
}

// shapelet/transform.rs

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ShapeletTransformFit {
    pub shapelets: Vec<Shapelet>,
    pub train_features: FdMatrix, // n×K
}

#[must_use = "expensive computation whose result should not be discarded"]
pub fn shapelet_transform_fit(
    data: &FdMatrix,
    y: &[usize],
    config: &ShapeletConfig,
) -> Result<ShapeletTransformFit, FdarError>;

#[must_use = "expensive computation whose result should not be discarded"]
pub fn shapelet_transform(
    fit: &ShapeletTransformFit,
    new_data: &FdMatrix,
) -> Result<FdMatrix, FdarError>;  // n_new×K

// shapelet/classifier.rs

#[derive(Debug, Clone, PartialEq)]
pub struct ShapeletClassifConfig {
    pub shapelet: ShapeletConfig,
    pub ncomp: usize,  // FPC components for downstream LDA
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ShapeletClassifResult {
    pub transform: ShapeletTransformFit,
    pub classif: ClassifFit,       // from classification/fit.rs
    pub training_accuracy: f64,
    pub n_classes: usize,
}

#[must_use = "expensive computation whose result should not be discarded"]
pub fn shapelet_classif_fit(
    data: &FdMatrix,
    y: &[usize],
    config: &ShapeletClassifConfig,
) -> Result<ShapeletClassifResult, FdarError>;

pub fn shapelet_classif_predict(
    fit: &ShapeletClassifResult,
    new_data: &FdMatrix,
) -> Result<ClassifResult, FdarError>;
```

---

## Data Flow

```text
Training path:
  curves (n×m FdMatrix) + labels (&[usize])
      |
      |  shapelet_transform_fit(data, y, config)
      v
  [Candidate enumeration — shapelet/discovery.rs]
  For each (curve i, position p, length l) in [min_len..max_len]:
    - Extract window: data row i, columns p..p+l
    - z_normalize_window() -> shapelet_candidate values
    - shapelet_distance(candidate, every other curve j) -> Vec<f64> of n distances
    - Score by InfoGain / FStat on the (distances, y) split
    - Track best split threshold per candidate
      |
      v
  [Ranking + self-similarity pruning — shapelet/discovery.rs]
  Sort candidates by score descending
  Greedy select: skip if shapelet_distance(candidate, accepted[k]) < similarity_threshold
  Retain top n_shapelets (K) -> Vec<Shapelet>
      |
      v
  [Transform — shapelet/transform.rs]
  For each selected shapelet s_k (k = 0..K):
    For each training curve i (parallelized via iter_maybe_parallel!):
      train_features[(i, k)] = shapelet_distance(s_k, curve_i)
  -> n×K FdMatrix (train_features)
      |
  ShapeletTransformFit { shapelets: Vec<Shapelet>, train_features: FdMatrix }
      |
      |  (bundled STC path only — shapelet_classif_fit)
      v
  [Classification — shapelet/classifier.rs]
  fclassif_lda_fit(&train_features, y, None, config.ncomp)
  -> ClassifFit (LDA in shapelet-distance feature space, from classification/fit.rs)
      |
  ShapeletClassifResult { transform, classif, training_accuracy, n_classes }

Prediction path:
  new curves (n_new×m FdMatrix)
      |
  shapelet_transform(fit, new_data)
      |  For each s_k, for each new curve i:
      |    features[(i,k)] = shapelet_distance(s_k, new_data row i)
      v
  n_new×K FdMatrix (new_features)
      |
  [LDA predict on new_features via ClassifFit.method (ClassifMethod::Lda)]
  -> ClassifResult { predicted, accuracy, confusion, n_classes, ncomp }
```

---

## File Structure

```text
fdars-core/src/shapelet/
├── mod.rs         # pub use all public items; declares 4 submodules
├── distance.rs    # z_normalize_window, shapelet_distance (sdist with early-abandon),
│                  # Shapelet struct
├── discovery.rs   # ShapeletConfig, ShapeletScorer, candidate enumeration,
│                  # scoring (InfoGain / FStat), self-similarity pruning
├── transform.rs   # ShapeletTransformFit, shapelet_transform_fit, shapelet_transform
└── classifier.rs  # ShapeletClassifConfig, ShapeletClassifResult,
                   # shapelet_classif_fit, shapelet_classif_predict
```

`src/lib.rs` additions (two lines, additive only):

```rust
pub mod shapelet;
pub use shapelet::{
    Shapelet, ShapeletClassifConfig, ShapeletClassifResult, ShapeletConfig,
    ShapeletScorer, ShapeletTransformFit,
    shapelet_classif_fit, shapelet_classif_predict,
    shapelet_transform, shapelet_transform_fit,
};
```

---

## Dependency-Ordered Build Sequence

The four files have a strict one-way dependency chain that determines phase order.

**Phase 57 — Shapelet distance core** (`shapelet/distance.rs` + `shapelet/mod.rs` skeleton)

Dependencies: `FdMatrix` (`matrix.rs`), `FdarError` (`error.rs`). No shapelet-internal dependency.

Deliverables: `z_normalize_window`, `shapelet_distance` (with early-abandon), `Shapelet` struct, `mod.rs` stub with `pub mod distance`. This is the atomic primitive all later phases consume.

**Phase 58 — Discovery and ranking** (`shapelet/discovery.rs`)

Dependencies: Phase 57 (`shapelet_distance`, `Shapelet`); `iter_maybe_parallel!` + `seed_for_thread` (`parallel.rs`); `FdMatrix`, `FdarError`.

Deliverables: `ShapeletConfig`, `ShapeletScorer`, candidate enumeration (exhaustive + contracted), InfoGain/FStat scorers, self-similarity pruning. Returns `Vec<Shapelet>` from the discovery step. `mod.rs` gains `pub mod discovery`.

**Phase 59 — Shapelet transform** (`shapelet/transform.rs`)

Dependencies: Phase 58 (`Shapelet`, `ShapeletConfig`); `iter_maybe_parallel!`; `FdMatrix`, `FdarError`.

Deliverables: `ShapeletTransformFit`, `shapelet_transform_fit` (calls Phase 58 then builds n×K matrix), `shapelet_transform` (predict path: builds n_new×K matrix from stored shapelets). `lib.rs` gets transform re-exports here. `mod.rs` gains `pub mod transform`.

**Phase 60 — Bundled ShapeletTransformClassifier** (`shapelet/classifier.rs`)

Dependencies: Phase 59 (`ShapeletTransformFit`, `shapelet_transform_fit`, `shapelet_transform`); `fclassif_lda_fit` + `ClassifFit` + `ClassifResult` from `src/classification/fit.rs` (existing, unmodified).

Deliverables: `ShapeletClassifConfig`, `ShapeletClassifResult`, `shapelet_classif_fit`, `shapelet_classif_predict`. `lib.rs` gets final re-exports. Criterion benchmark added. `mod.rs` gains `pub mod classifier`.

---

## Architectural Constraints and Non-Breaking Guarantees

**Additive/non-breaking:** All new code lives in a new `pub mod shapelet`. Zero changes to any existing public signature. The `classification/`, `distance.rs`, `helpers.rs`, `matrix.rs`, `parallel.rs`, and `error.rs` files are consumed as-is. No existing example, R binding, or WASM binding is affected.

**WASM/R-binding safety:** `Shapelet`, `ShapeletTransformFit`, `ShapeletClassifResult`, and all config types are pure Rust structs with no threading primitives in the public surface. The `parallel` feature gate is respected via `iter_maybe_parallel!` — the same macro used in `distance.rs` and `classification/cv.rs`. No `Send + Sync` bounds are added to public API types (keeping WASM compatibility). The `seed_for_thread` pattern from `parallel.rs` is reused verbatim.

**No new crate dependency:** z-normalization, sliding-window Euclidean distance, and candidate scoring are all pure arithmetic over `Vec<f64>` slices. InfoGain requires only entropy computation over class frequency counts — no external dependency. Contracted search uses `rand::rngs::StdRng` already in scope.

**Deterministic seeding:** Contracted search follows the established pattern. The candidate shuffle (when `max_candidates > 0`) is done on a single-threaded index build before the parallel evaluation loop, so the shuffle is identical regardless of thread count. Per-thread sub-steps use `seed_for_thread(config.seed, k)`.

**Column-major access correctness:** Shapelet distance inner loops access rows of `FdMatrix` (evaluation points along a curve). Row access in column-major layout is a stride pattern (stride = nrows). This is acceptable given that the outer parallelism is over candidates/curves rather than over evaluation points, and the windows are short (min_len to max_len). The `row_to_buf` helper can be used to materialize a row into a contiguous buffer before windowing if profiling reveals the stride access is a bottleneck.

**`#[must_use]` discipline:** `shapelet_transform_fit`, `shapelet_transform`, `shapelet_classif_fit` are marked `#[must_use = "expensive computation whose result should not be discarded"]`. `shapelet_classif_predict` is not marked (callers inspect results immediately).

---

## Patterns to Follow

### Pattern: Bundled pipeline delegates to existing classifier

`shapelet_classif_fit` does not re-implement LDA. It calls `fclassif_lda_fit(&transform_fit.train_features, y, None, config.ncomp)` — the exact existing signature from `src/classification/fit.rs`. The `ShapeletClassifResult` stores the returned `ClassifFit` verbatim. `shapelet_classif_predict` calls `shapelet_transform` to get the n_new×K feature matrix, then applies the `ClassifFit`'s stored `ClassifMethod::Lda` parameters directly rather than re-running FPCA projection (the shapelet feature matrix is already the feature space — no further FPC projection is needed or appropriate).

Why LDA as the default downstream classifier: sktime's `ShapeletTransformClassifier` defaults to rotation-forest; fdars has no rotation forest. LDA on the K-dimensional shapelet feature matrix is the most suitable existing classifier in `classification/` — it has stored parameters, a predict path, and `ClassifFit` implements `FpcPredictor` for future explainability hooks. The config struct is designed to be extended in a later milestone to accept a `ClassifMethod` enum if other classifiers are desired.

### Pattern: Early-abandon in sdist inner loop

`shapelet_distance` maintains a running squared Euclidean sum over each window position. When that partial sum exceeds the current best minimum distance found at prior starting positions, it breaks out of the inner window loop early (pruning the remaining comparison points for that position). This prunes O(length) work per candidate position on average when a bad match is detected early. The early-abandon does not change the final answer because the partial sum can only grow — once it exceeds the global best, no additional points can make this position win.

### Pattern: Per-thread seeded RNG for contracted search

When `config.max_candidates > 0`, the candidate triples (i, p, l) are shuffled using `StdRng::seed_from_u64(config.seed)` (single-threaded, before the parallel evaluation loop) then truncated to `max_candidates`. This gives a deterministic, reproducible candidate subset regardless of thread count. The subsequent parallel scoring loop over that fixed list uses `seed_for_thread(config.seed, k)` for any future per-candidate stochastic sub-steps, following the pattern from `elastic_align_pair` and `gmm_em`.

---

## Anti-Patterns to Avoid

### Anti-Pattern: Z-normalizing full curves instead of windows

**What people do:** Apply z-normalization to the full curve before extracting sliding windows, normalizing relative to the whole curve's mean and std.

**Why it is wrong:** The canonical shapelet sdist (Ye & Keogh; Hills/Lines) normalizes each candidate window independently relative to that window's own statistics. Full-curve normalization discards local amplitude information and makes results incomparable with sktime/pyts reference implementations.

**Do this instead:** `z_normalize_window` takes a `&[f64]` slice (the already-extracted window), computes mean and std over that slice only, and returns the normalized values. The caller extracts the window slice before calling.

### Anti-Pattern: Allocating a full n×n distance matrix during discovery

**What people do:** Compute a full n×n shapelet-distance matrix for every candidate before scoring.

**Why it is wrong:** For 100 training curves and 10,000 candidates the intermediate storage is 80 MB per candidate at minimum — orders of magnitude too large. O(n_candidates × n²) memory is prohibitive.

**Do this instead:** For each candidate, compute only a `Vec<f64>` of length n (one distance per training curve), score it, and discard it immediately. Only the final K winners are retained as `Vec<Shapelet>`.

### Anti-Pattern: Placing shapelet transform logic inside `classification/`

**What people do:** Add `shapelet_*` functions to `src/classification/` because the bundled STC eventually produces class labels.

**Why it is wrong:** The shapelet transform is a generic feature-engineering step (curves → distance features) useful beyond any single downstream classifier. Coupling it to `classification/` would require modifying `ClassifMethod` (a `#[non_exhaustive]` enum), breaking the non-exhaustive contract, and would prevent clean reuse with regression or clustering consumers in future milestones (e.g., shapelet features fed into `fregre_lm`).

**Do this instead:** `src/shapelet/` is its own top-level domain module. `classification/` is a leaf dependency — consumed only by `shapelet/classifier.rs`.

### Anti-Pattern: Using `helpers::l2_distance` as the shapelet sdist

**What people do:** Call `helpers::l2_distance(window_a, window_b, weights)` inside the shapelet distance loop, reusing the existing weighted L2 functional distance.

**Why it is wrong:** `helpers::l2_distance` applies Simpson's integration weights — correct for continuous functional inner products over the full domain, but wrong for shapelet distance. Shapelets use plain unweighted squared Euclidean distance over the discrete window values. The weight assumption distorts the distance and breaks comparability with reference implementations.

**Do this instead:** `shapelet/distance.rs` implements its own inner loop with uniform weights (sum of squared differences, no integration weights). It does not call `helpers::l2_distance`.

---

## Sources

- Hills, J., Lines, J., Baranauskas, E., Mapp, J., and Bagnall, A. (2014). Classification of time series by shapelet transformation. Data Mining and Knowledge Discovery, 28(4), 851-881. Canonical discovery algorithm reference.
- Ye, L. and Keogh, E. (2009). Time series shapelets: a new primitive for data mining. ACM KDD 2009. Original shapelet distance definition.
- sktime ShapeletTransformClassifier API (v0.33.x): transform-then-classify architecture, contracted search, self-similarity pruning conventions.
- pyts@0.13.x ShapeletTransform: confirms window enumeration and InfoGain scorer convention.
- fdars-core `src/classification/fit.rs` — `fclassif_lda_fit` exact signature: `(data: &FdMatrix, y: &[usize], scalar_covariates: Option<&FdMatrix>, ncomp: usize) -> Result<ClassifFit, FdarError>`
- fdars-core `src/distance.rs` — `cross_distance_matrix` pattern for the predict path.
- fdars-core `src/parallel.rs` — `iter_maybe_parallel!`, `seed_for_thread` macro/function conventions confirmed by direct source read.

---

*Architecture research for: fdars v0.33.0 Shapelet Transform and Classification*
*Researched: 2026-09-02*
