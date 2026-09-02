# Stack Research: Shapelet Transform & Classification (v0.33.0)

**Domain:** Rust functional-data-analysis library — adding discovery-based shapelet transform + bundled ShapeletTransformClassifier to fdars-core
**Researched:** 2026-09-02
**Confidence:** MEDIUM (sktime/pyts APIs verified via official docs; algorithm cross-checked via published paper abstracts; codebase read HIGH confidence)

---

## Dependency Verdict: NO NEW CRATE DEPENDENCIES REQUIRED

All four v0.33.0 deliverables — shapelet-distance core, discovery & ranking, shapelet transform, and bundled ShapeletTransformClassifier — can be built entirely on the existing stack. This is the primary finding.

Explicit confirmation per feature:

| Feature | Required new dep? | Rationale |
|---------|------------------|-----------|
| Z-normalization of subsequences | NO | `fdata::NormalizationMethod::CurveStandardize` already implements per-row z-norm (subtract mean, divide by std) via `row_normalize(..., RowNorm::Standardize)`. For a subsequence slice a thin inline helper suffices — same arithmetic. |
| Squared Euclidean distance (sliding window min) | NO | `FdMatrix::row_l2_sq` computes inline squared L2 without allocation. For subsequence pairs: plain scalar loop over slice indices. |
| Early-abandon distance | NO | Pure scalar `f64` arithmetic with a running `best_so_far` bound — no dep needed. |
| Candidate subsequence generation | NO | Iterate `(start, length)` pairs over training curve rows; no container dep needed. |
| Information gain scoring | NO | Binary class entropy = `-p*ln(p) - (1-p)*ln(1-p)` using `f64::ln`; no dep. For multi-class: same. |
| F-statistic scoring (ANOVA alternative) | NO | `function_on_scalar::integrated_f_statistic` already exists as `pub(crate)` — the exact pattern is present. |
| Self-similarity pruning | NO | Sort shapelets by position/series index; prune overlapping windows in a single pass — pure logic. |
| Shapelet transform (n×K distance matrix) | NO | `FdMatrix` (existing column-major matrix) is the output type. |
| Parallelism over candidate evaluation | NO | `iter_maybe_parallel!` macro from `parallel.rs` gates `rayon` (already a dep). |
| Bundled classifier (kNN on K distance features) | NO | `knn_classify_from_distances(&dist_mat, y, k_nn)` exists in `classification/knn.rs` — accepts a precomputed distance matrix; the n×K feature matrix can be used directly with `euclidean_distance_matrix` from `distance.rs`. |
| RNG seeding | NO | `rand` 0.8 + `StdRng::seed_from_u64(seed)` pattern already in the codebase. |
| `linalg` / faer | NOT NEEDED | No SVD, Cholesky, or matrix factorization in the shapelet path. |
| MSRV change | NONE | All code is plain `f64` arithmetic; MSRV stays at 1.81. |

---

## Out of Scope: Learning Shapelets

**tslearn `LearningShapelets` (gradient-based) is explicitly NOT in scope for v0.33.0.**

tslearn@0.9.0 ships two distinct shapelet families:

| tslearn class | Approach | In scope? |
|---------------|----------|-----------|
| `ShapeletModel` (alias: `LearningShapelets`) | Gradient descent on shapelet parameters embedded in a differentiable transform + logistic loss; requires autodiff | **NO** — deferred (GAP-08 autodiff is a separate, much larger effort) |
| (no direct equivalent) | Discovery-based: enumerate candidates, score by information gain, select top-K | **YES** — this is v0.33.0 |

The discovery approach is the basis of Hills/Lines/Bagnall (2014) and is what sktime's `RandomShapeletTransform` and pyts's `ShapeletTransform` implement. It requires no gradient machinery.

---

## Reference Baselines — Pinned Versions & Exact APIs

### 1. sktime `ShapeletTransformClassifier` + `RandomShapeletTransform`

**Version:** sktime stable (≥ 0.30.0). The `ShapeletTransformClassifier` wraps `RandomShapeletTransform` + `RotationForest` (default estimator). Verified via https://www.sktime.net/en/stable/.

#### `RandomShapeletTransform` (the transform component)

```python
RandomShapeletTransform(
    n_shapelet_samples=10000,   # candidate shapelets to evaluate
    max_shapelets=None,          # K retained; default = min(10 * n_instances, 1000)
    min_shapelet_length=3,       # minimum subsequence length
    max_shapelet_length=None,    # None = series length
    remove_self_similar=True,    # prune overlapping candidates from same series
    time_limit_in_minutes=0.0,   # 0 = no time budget
    contract_max_n_shapelet_samples=inf,
    n_jobs=1,
    parallel_backend=None,
    batch_size=100,
    random_state=None
)
```

- **Scoring criterion:** information gain (binary or multiclass entropy split). Candidates are abandoned early if their maximum achievable IG cannot beat the current K-th best (early-abandon bound).
- **Transform output:** `n_series × K` matrix of minimum sliding-window distances from each series to each retained shapelet.
- **Fit produces:** `shapelets_` list; `transform(X)` returns the feature matrix.

#### `ShapeletTransformClassifier`

```python
ShapeletTransformClassifier(
    n_shapelet_samples=10000,
    max_shapelets=None,
    max_shapelet_length=None,
    estimator=None,              # default: RotationForest
    transform_limit_in_minutes=0,
    time_limit_in_minutes=0,
    contract_max_n_shapelet_samples=inf,
    save_transformed_data=False,
    n_jobs=1,
    batch_size=100,
    random_state=None
)
```

- Pipeline: `RandomShapeletTransform.fit_transform(X_train, y)` → `estimator.fit(X_transformed, y)`.
- `predict(X_test)`: `RandomShapeletTransform.transform(X_test)` → `estimator.predict(X_transformed)`.

**fdars counterpart naming convention:** `ShapeletTransformClassifier` → `ShapeletTransformClassifier` struct with `fit` + `predict` methods matching this pipeline shape.

### 2. pyts `ShapeletTransform` (pyts@0.13.x)

**Version:** pyts 0.13.0 (PyPI stable as of the v0.31.0 audit). Verified via https://pyts.readthedocs.io/en/latest/.

```python
pyts.transformation.ShapeletTransform(
    n_shapelets='auto',      # int or 'auto' → n_timestamps // 2
    criterion='mutual_info', # 'mutual_info' or 'anova' (F-score)
    window_sizes='auto',     # array-like or 'auto'
    window_steps=None,       # stride; None → 1
    remove_similar=True,
    sort=False,
    verbose=0,
    random_state=None,
    n_jobs=None
)
```

- `fit(X, y)` → self; `transform(X)` → `X_new` shape `(n_samples, n_shapelets)`.
- Post-fit attributes: `shapelets_` (array of shapelet values), `indices_` (n_shapelets × 3 array: `[series_index, start, length]`), `scores_`.
- Two scoring criteria: mutual information (default) and F-statistic via ANOVA. The fdars implementation uses **F-statistic** as the primary discriminative score (simpler, no class-probability estimation; consistent with the `integrated_f_statistic` helper already in the codebase).

**Key difference from sktime:** pyts requires explicit `window_sizes`; sktime uses `n_shapelet_samples` + random length sampling. fdars v0.33.0 follows the sktime random-sampling model (more practical for varying-length curves), with window sizes drawn uniformly from `[min_shapelet_length, max_shapelet_length]`.

### 3. tslearn `LearningShapelets` — NOT IN SCOPE

```python
# tslearn@0.9.0 — DEFERRED; do not implement in v0.33.0
tslearn.shapelets.LearningShapelets(
    n_shapelets_per_size={...},
    max_iter=10000,
    batch_size=256,
    optimizer='sgd',
    weight_regularizer=0.01,
    shapelet_length=0.1,
    verbose_level=0,
    scale=False,
    random_state=None,
    total_lengths=...
)
```

Requires backpropagation through shapelet distances → deferred to GAP-08 (differentiable FDA core, score 1.73). Do not attempt in v0.33.0.

---

## Existing Primitives: Present vs. To-Be-Written

### Already Present in fdars-core (HIGH confidence — direct codebase read)

| Primitive | Location | API | Reuse Type |
|-----------|----------|-----|------------|
| Per-row z-normalization (full curves) | `src/fdata.rs` | `normalize(data, NormalizationMethod::CurveStandardize)` → `FdMatrix` | Adapt for subsequence slices inline |
| Pointwise row mean + std | `src/fdata.rs:row_normalize` | `RowNorm::Standardize`: `mean = sum/m`, `std = sqrt(sum_sq/(m-1))`, safe denom | Copy pattern for subsequence slice |
| Squared row L2 (no alloc) | `src/matrix.rs` | `FdMatrix::row_l2_sq(row_a, other, row_b) -> f64` | Direct reuse where shapes match; for subsequences use inline scalar loop |
| Symmetric distance matrix loop (upper triangle + parallel) | `src/distance.rs` | `pairwise_distance_matrix(n, dist_fn)` + `euclidean_distance_matrix` | Reuse for training-set Euclidean distance on the n×K feature matrix |
| Cross distance matrix | `src/distance.rs` | `cross_distance_matrix(n_new, n_train, dist_fn)` | Reuse for predict path |
| kNN from precomputed distance matrix | `src/classification/knn.rs` | `knn_classify_from_distances(dist_mat, y, k_nn) -> ClassifResult` | Direct — the bundled STC classifier backend |
| kNN Euclidean (feature space) | `src/classification/fit.rs` | `fclassif_knn_fit(data, y, None, ncomp, k_nn) -> ClassifFit` | Alternative: pass n×K feature matrix as `data` with `ncomp = K` |
| F-statistic scoring (1D scalar response) | `src/function_on_scalar.rs` | `pub(crate) integrated_f_statistic(data, groups, labels)` | Adapt pattern for scalar distance split |
| Parallel macro | `src/parallel.rs` | `iter_maybe_parallel!` / `slice_maybe_parallel!` | Gate candidate evaluation loop |
| Per-thread RNG seeding | widespread | `StdRng::seed_from_u64(seed + thread_id as u64)` | Direct reuse |
| `FdMatrix` as output type | `src/matrix.rs` | Column-major `FdMatrix::zeros(n, K)` | The n×K shapelet-distance feature matrix |

### To Be Written for v0.33.0

| Primitive | Where | Notes |
|-----------|-------|-------|
| `z_normalize_slice(s: &[f64]) -> Vec<f64>` | `src/shapelet/core.rs` (new) | Inline per-subsequence z-norm: mean + sample std; zero-std guard (return zeros). 10-line function. |
| `shapelet_min_distance(shapelet: &[f64], curve_row: &[f64]) -> f64` | `src/shapelet/core.rs` | Sliding-window minimum z-normalized squared Euclidean distance with early-abandon. Core hot-path. |
| `generate_candidates(data: &FdMatrix, min_len, max_len, n_samples, seed) -> Vec<Shapelet>` | `src/shapelet/discovery.rs` (new) | Random (series, start, length) sampling from training curves. Returns `Vec<Shapelet>` where `Shapelet { values: Vec<f64>, series: usize, start: usize, length: usize }`. |
| `score_candidate(shapelet, data, y, n_classes) -> f64` | `src/shapelet/discovery.rs` | F-statistic or information gain on the split of per-curve minimum distances. Inline entropy/F computation — no dep. |
| `prune_self_similar(candidates: Vec<Shapelet>, threshold) -> Vec<Shapelet>` | `src/shapelet/discovery.rs` | Remove shapelets from the same series with overlapping windows. |
| `ShapeletTransform` struct + `fit` + `transform` | `src/shapelet/transform.rs` (new) | Fit: discover → rank → prune; transform: compute n×K distance matrix for any curve set. |
| `ShapeletTransformClassifier` struct + `fit` + `predict` | `src/shapelet/classifier.rs` (new) | Bundles `ShapeletTransform` + kNN (or LDA) classifier; matches sktime STC pipeline shape. |
| `ShapeletConfig` | `src/shapelet/mod.rs` (new) | Config struct: `n_shapelet_samples`, `max_shapelets`, `min_shapelet_length`, `max_shapelet_length`, `remove_self_similar`, `seed`, `k_nn`. |

---

## Bundled Classifier: Which fdars Classifier to Wrap

**Recommendation: kNN on the K distance features.**

Rationale:
1. **Direct primitive:** `knn_classify_from_distances(dist_mat, y, k_nn)` already accepts a precomputed n×n matrix. For the shapelet feature case, the n×K shapelet-distance matrix is a tabular Euclidean feature space — pass it through `euclidean_distance_matrix` first, then classify. Alternatively, call `fclassif_knn_fit` directly with the n×K feature matrix as `data` (treating each distance feature as a "channel") — `ncomp = K` bypasses FPCA and uses the raw features.
2. **Precedent:** sktime's `ShapeletTransformClassifier` defaults to RotationForest, but kNN is the classic shapelet paper classifier (Ye & Keogh 2009; Hills/Lines 2014). pyts does not bundle a classifier — it is a transform only.
3. **Simplicity:** kNN on the Euclidean distance over the K-feature space requires no hyperparameter tuning beyond `k`. LDA and QDA are alternatives if the caller wants them — expose `ClassifMethod` as a config enum field.

**Concrete wiring:**

```rust
// In ShapeletTransformClassifier::fit():
let feature_matrix: FdMatrix = self.transform.fit_transform(&data, y)?;  // n × K
// kNN on raw distance features (no FPCA; ncomp = K bypasses FPCA reduction):
self.classif_fit = fclassif_knn_fit(&feature_matrix, y, None, feature_matrix.ncols(), k_nn)?;

// In ShapeletTransformClassifier::predict():
let feature_matrix: FdMatrix = self.transform.transform(&new_data)?;  // n_new × K
// predict_from_scores uses stored training_scores from ClassifMethod::Knn
```

The `fclassif_knn_fit` path stores `ClassifMethod::Knn { training_scores, training_labels, k, n_classes }`, so `predict_from_scores` on a new observation's K-dim feature vector works with no extra code.

**Expose alternatives via `ShapeletConfig.classifier: ShapeletClassifier` enum:** `Knn { k: usize }` (default) | `Lda` | `Qda`. This matches the user-extensibility model of sktime's `estimator` parameter without requiring an external trait object.

---

## Module Placement

```
src/shapelet/
    mod.rs          — ShapeletConfig, public re-exports, crate-root pub use
    core.rs         — z_normalize_slice, shapelet_min_distance (hot path)
    discovery.rs    — generate_candidates, score_candidate, prune_self_similar
    transform.rs    — ShapeletTransform struct (fit, transform, shapelets_)
    classifier.rs   — ShapeletTransformClassifier struct (fit, predict)
```

Crate-root re-export in `src/lib.rs` follows the `pub use shapelet::*` pattern matching `pub use metric::soft_dtw::*` etc.

---

## Core Technologies Used by v0.33.0 Features

| Technology | Version in Cargo.toml | Role in v0.33.0 | Change? |
|------------|----------------------|-----------------|---------|
| Rust (MSRV 1.81) | 1.81 min / 1.97 dev | All implementation | None |
| nalgebra | 0.33 | Not in shapelet path | None |
| rayon | 1.10 (optional, `parallel` feature) | Parallel candidate evaluation loop | None (reuse `iter_maybe_parallel!`) |
| rand | 0.8 | Random candidate sampling (`generate_candidates`) | None |
| rustfft | 6.2 | **Not used** — shapelet distance is pure sliding Euclidean | None |
| faer | 0.23 (`linalg` feature) | **Not used** — no matrix factorization in shapelet path | None |
| statrs | existing | **Not used** — entropy is 2-line inline; F-stat is inline | None |
| FdMatrix | `src/matrix.rs` | n×K shapelet-feature matrix output type | None |

### No New Dependencies

```toml
# Cargo.toml — NO CHANGES NEEDED for v0.33.0
# All shapelet deliverables build on the existing dependency set.
```

---

## Key Algorithm Details

### Z-Normalized Euclidean Distance (the shapelet distance)

For a shapelet `s` of length `L` and a subsequence `t[p..p+L]` of a curve `t`:

1. Z-normalize `s` to zero mean, unit variance (pre-computed at shapelet extraction time, once).
2. Slide over `t`: for each position `p` in `[0, len(t) - L]`:
   a. Z-normalize `t[p..p+L]` in O(L) by computing mean + std online.
   b. Compute squared Euclidean distance between normalized `s` and normalized window.
   c. Early-abandon: maintain `best_so_far`; exit inner loop as soon as accumulated distance exceeds `best_so_far`.
3. Return `sqrt(min_sq_dist)`.

**Early-abandon implementation:** accumulate `sum_sq` term-by-term; break when `sum_sq > best_so_far_sq`. This gives O(1) amortized improvement over the naive O(L) inner loop when candidates are poor (common in large candidate sets). Matches the bound used in Hills/Lines 2014 and implemented in sktime's `RandomShapeletTransform`.

**Z-normalization note:** `NormalizationMethod::CurveStandardize` in `fdata.rs` normalizes complete rows of `FdMatrix`. For sliding-window subsequences, the z-normalization is applied inline to a `&[f64]` slice — a 10-line helper `z_normalize_slice(s: &[f64]) -> Vec<f64>` in `shapelet/core.rs`. No dep needed.

### Discriminative Scoring

**Primary criterion: F-statistic** on the vector of minimum-distances `d_i` (one per training curve).

Split the training distances at each candidate threshold `τ` (try `n` thresholds, one between each adjacent pair of sorted distances). For each split, compute the one-way F-statistic between the two groups (curves with `d_i ≤ τ` vs `d_i > τ`). Take the maximum F over all thresholds as the candidate's score.

The `pointwise_f_statistic` / `integrated_f_statistic` code in `function_on_scalar.rs` implements the same ANOVA arithmetic for full curves — the scalar version for a split is a direct simplification (2-group, 1-dimensional) writable in ~15 lines without touching that function. Keep it inline in `discovery.rs`.

**Alternative: information gain** (sktime default, `mutual_info` in pyts): binary entropy `H = -p*log(p) - (1-p)*log(1-p)` evaluated at each candidate split; `IG = H(parent) - (n_left/n)*H(left) - (n_right/n)*H(right)`. 10-line inline; no dep. Expose as `ScoringCriterion::InformationGain | FStatistic` in `ShapeletConfig`.

---

## Feature-Flag Considerations

| Aspect | Recommendation |
|--------|---------------|
| Parallel candidate evaluation | YES — gate with `parallel` feature via `iter_maybe_parallel!` on the candidate batch loop. Each candidate's distance computation is independent. |
| Does shapelet path need `linalg`? | NO — no matrix factorization. Works under default features. |
| WASM compatibility | YES — pure `f64` arithmetic; rayon is optional and off on WASM. |
| `serde` feature | Result structs follow `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` convention; no new serde work needed. |
| Pre-z-normalizing shapelets at discovery | YES — z-normalize each retained shapelet once at fit time; store normalized values in `ShapeletTransform::shapelets_`. Avoids re-normalizing on each transform call. |

---

## Alternatives Considered

| Decision | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Bundled classifier backend | kNN (`fclassif_knn_fit`) | LDA (`fclassif_lda_fit`) | LDA assumes Gaussian class-conditional; shapelet-distance features are not Gaussian in general. kNN is the canonical choice in the shapelet literature. LDA available as opt-in via config enum. |
| Discriminative scoring | F-statistic (primary), IG (secondary) | Mutual information only | F-statistic is simpler (no probability estimation), already has infrastructure in `function_on_scalar.rs`. IG is also provided to match sktime/pyts defaults. |
| Z-normalization for subsequences | Inline slice helper in `shapelet/core.rs` | Reuse `fdata::normalize` on a single-row `FdMatrix` | Would allocate a 1×L `FdMatrix` per window per candidate — unacceptable O(n·m·L) allocation overhead. Inline slice is allocation-free. |
| Module location | `src/shapelet/` (new submodule directory) | Extend `src/classification/` | Shapelets span transform + classifier — a dedicated submodule matches the `metric/`, `alignment/`, `depth/` pattern. |
| Shapelet storage in transform | Pre-normalized `Vec<f64>` per shapelet | Raw subsequence + normalize at transform time | Pre-normalizing at fit time avoids repeated z-norm on the same shapelet across every `transform` call. |
| Learning shapelets | NOT included | tslearn `LearningShapelets` | Requires gradient-based optimization through the shapelet distance function — deferred to GAP-08 (differentiable FDA core). Out of scope v0.33.0. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| tslearn `LearningShapelets` / gradient approach | Requires autodiff through a non-differentiable argmin operation; deferred to GAP-08 | Discovery-based: enumerate, score, select |
| `fdata::normalize` for per-window z-norm | Allocates a full `FdMatrix` row per window — prohibitive in the inner loop | Inline `z_normalize_slice(&[f64]) -> Vec<f64>` in `shapelet/core.rs` |
| `rustfft` / FFT for shapelet distance | Shapelet distance is a short-window Euclidean operation, not a frequency-domain computation. FFT would be relevant only for k-Shape SBD (GAP-03, separate milestone). | Plain sliding-window Euclidean |
| `faer` or `nalgebra` for feature matrix operations | The n×K shapelet-distance matrix requires no SVD/Cholesky — only `FdMatrix` arithmetic | `FdMatrix::zeros(n, K)` filled by scalar assignment |
| Adding an external information-gain library | Entropy is a 3-line inline function; no dep justified | Inline `binary_entropy(p: f64) -> f64 = -p*ln(p)-(1-p)*ln(1-p)` with zero-guard |

---

## MSRV and Feature Compatibility Matrix

| Scenario | Works? | Notes |
|----------|--------|-------|
| Default features (`parallel`) | YES | `iter_maybe_parallel!` parallelizes candidate evaluation; MSRV 1.81 |
| No features (sequential) | YES | All loops are sequential-compatible |
| `linalg` feature | YES | Shapelet adds nothing to `linalg`; features are orthogonal |
| `serde` feature | YES | Add derive attributes to `ShapeletTransform`, `ShapeletTransformClassifier`, config/result structs |
| WASM (`js` feature) | YES | Pure `f64` arithmetic; rayon is off on WASM |
| Rust 1.81 (MSRV) | YES | No post-1.81 stabilizations needed |

---

## Sources

- sktime stable docs — `ShapeletTransformClassifier`, `RandomShapeletTransform` class signatures and parameter descriptions — https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.classification.shapelet_based.ShapeletTransformClassifier.html — MEDIUM confidence (verified against official sktime.net stable docs via WebFetch)
- sktime stable docs — `RandomShapeletTransform` details — https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.transformations.panel.shapelet_transform.RandomShapeletTransform.html — MEDIUM confidence
- pyts 0.13.0 docs — `ShapeletTransform` class signature, parameters, fit/transform contract — https://pyts.readthedocs.io/en/latest/generated/pyts.transformation.ShapeletTransform.html — MEDIUM confidence
- Hills, J., Lines, J., Baranauskas, E. et al. (2014) "Classification of time series by shapelet transformation" — Data Mining and Knowledge Discovery. DOI: 10.1007/s10618-013-0322-1. Via search result abstracts — MEDIUM confidence (algorithm structure confirmed; early-abandon + information gain + z-norm Euclidean)
- fdars-core/src/fdata.rs — confirmed `NormalizationMethod::CurveStandardize` / `row_normalize` / `RowNorm::Standardize` implementation — HIGH confidence (direct codebase read)
- fdars-core/src/matrix.rs — confirmed `row_l2_sq`, `row_to_buf`, `row_dot` hot-path methods — HIGH confidence (direct codebase read)
- fdars-core/src/classification/knn.rs — confirmed `knn_classify_from_distances(dist_mat, y, k_nn)` API + `fclassif_knn_fit` fit/predict path — HIGH confidence (direct codebase read)
- fdars-core/src/function_on_scalar.rs — confirmed `pub(crate) integrated_f_statistic` exists, arithmetic pattern for F-statistic directly applicable — HIGH confidence (direct codebase read)
- fdars-core/src/distance.rs — confirmed `pairwise_distance_matrix`, `euclidean_distance_matrix`, `cross_distance_matrix` API — HIGH confidence (direct codebase read)

---

*Stack research for: v0.33.0 Shapelet Transform & Classification in fdars-core*
*Researched: 2026-09-02*
