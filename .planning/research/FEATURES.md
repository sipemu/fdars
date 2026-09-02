# Feature Research: Shapelet Transform & Classification (v0.33.0)

**Domain:** Discovery-based shapelet transform + bundled classifier for functional curve/time-series data — Rust crate `fdars-core`
**Milestone:** v0.33.0 (promotes GAP-02 from `GAP-BACKLOG.md`)
**Researched:** 2026-09-02
**Confidence:** MEDIUM — mathematical spec derived from Ye & Keogh 2009 (primary paper; formulas reconstructed from survey literature), Hills/Lines/Bagnall 2014 shapelet transform (Springer DMKD; formulas from secondary survey sources), pyts@0.13.x source code (HIGH confidence for self-similarity pruning — actual code extracted), sktime RandomShapeletTransform API docs (HIGH confidence for parameters and API shape). No direct primary-paper PDF extraction succeeded; all formulas are cross-checked from ≥2 independent secondary sources.

**In-scope:** discovery-based (Ye & Keogh; Hills/Lines shapelet transform). **Out-of-scope:** learning-shapelets / gradient approach (tslearn `ShapeletModel`, Grabocka 2014 — explicitly deferred in milestone context).

---

## Precise Mathematical Specification

This section provides the exact formulas the planner must implement against. Claims are sourced and confidence-labelled per source hierarchy.

### A. Z-Normalization

Before any distance computation, both the shapelet S and every candidate window W are z-normalized independently:

```
z(x) = (x - mean(x)) / std(x)
```

where `mean(x)` is the arithmetic mean of the subsequence and `std(x)` is the sample standard deviation (ddof=0 is conventional in this literature; ddof=1 may also be used — implementations differ but ddof=0 is the more common choice in pyts and sktime). When `std(x) = 0` (constant subsequence), the normalized form is the zero vector.

**Why z-normalization matters:** Without it, a shapelet that is simply a scaled version of another would be considered different. Z-normalization makes the distance scale-invariant and shift-invariant — the shapelet captures **shape** rather than amplitude or offset. This is the defining property of the approach (distinguishing it from raw Euclidean distance over subsequences). (Confidence: HIGH — universally described across Ye&Keogh 2009, Hills 2014, tslearn docs, sktime docs.)

---

### B. Shapelet Distance (sdist) — Sliding-Window Minimum

Given a shapelet S of length L and a time series T of length M (L ≤ M), the **shapelet distance** (sdist) is:

```
sdist(S, T) = min_{t = 0 .. M-L}  ||z(T[t : t+L]) - z(S)||_2
```

Expanded:

```
sdist(S, T) = min_{t = 0 .. M-L}  sqrt( sum_{k=0}^{L-1} ( z(T[t+k]) - z(S[k]) )^2 )
```

where:
- `T[t : t+L]` is the length-L subsequence of T starting at position t.
- `z(·)` applies z-normalization to a length-L vector.
- `|| · ||_2` is the L2 (Euclidean) norm.

The sliding window has `M - L + 1` positions. The minimum over all positions gives the closest match in shape between the shapelet and any contiguous segment of T. **No time-warping** — this is a straight L2 distance after z-normalization (not DTW). (Confidence: HIGH — exact formula confirmed across tslearn docs, multivariate shapelet transform paper ar5iv 1712.06428, and aeon RandomShapeletTransform docs.)

**Implementation note:** The sqrt can be omitted during candidate evaluation (compare squared distances) since the minimum is order-preserving under sqrt. The sqrt is applied to produce the final output features. The minimum is computed incrementally over the sliding window; see early-abandon optimization in Section F.

**Connection to fdars existing code:** `distance.rs` already has Euclidean distance primitives. The z-normalization pass is new but trivial (mean + std of a slice). `FdMatrix` row-access via `row_to_buf` provides zero-copy access to each curve row for the outer loop over training series.

---

### C. Candidate Generation

The complete candidate set for a training dataset of n series each of length M is:

```
Candidates = { (i, t, L) : i ∈ 0..n, t ∈ 0..M-L, L ∈ [min_len, max_len] }
```

The total count is:

```
|Candidates| = n * sum_{L = min_len}^{max_len} (M - L + 1)
             ≈ n * (max_len - min_len + 1) * M  [order of magnitude]
```

For n=100 series, M=300, min_len=3, max_len=300 this is approximately 100 × 298 × 150 ≈ 4.5 million candidates. Each candidate requires computing sdist against all n training series, making the naive full-enumeration search **O(n² · M³)** overall (n candidates × n series per candidate × M² distance operations per pair — counting the sliding window explicitly). (Confidence: MEDIUM — complexity cited consistently in survey literature as O(n²m³) with distance caching; original Ye&Keogh was O(n²m⁴) before caching.)

**Practical implication:** Full enumeration is tractable only for small datasets (n<50, M<100). For the fdars implementation, a **random sampling / contracted** approach must be the default, with full enumeration available via a flag (see Section G: Differentiators).

---

### D. Shapelet Quality: Information Gain on the Distance Split

After computing `d_i = sdist(S, T_i)` for every training series `T_i`, the quality of shapelet S is assessed by how well its distance distribution separates the class labels.

#### D.1 Entropy

For a binary classification problem (generalized below), the entropy of a set D is:

```
H(D) = -p * log2(p) - (1-p) * log2(1-p)
```

where `p = |positive class| / |D|`. For multi-class problems, Shannon entropy:

```
H(D) = - sum_c  (|D_c| / |D|) * log2(|D_c| / |D|)
```

where `|D_c|` is the number of series in class c. (Confidence: HIGH — standard Shannon entropy, universally used in shapelet literature.)

#### D.2 Information Gain

For a given split threshold `θ`, the distance distribution is divided into:

```
D_left  = { (d_i, y_i) : d_i <= θ }
D_right = { (d_i, y_i) : d_i >  θ }
```

The information gain at threshold θ is:

```
IG(S, θ) = H(D) - ( |D_left|/|D| * H(D_left) + |D_right|/|D| * H(D_right) )
```

The quality of shapelet S is the **maximum IG over all possible split thresholds**:

```
quality(S) = max_{θ}  IG(S, θ)
```

#### D.3 Optimal Split Threshold Algorithm

The optimal threshold is found by sorting the distance-label pairs and scanning midpoints:

```
Algorithm FindBestSplit(distances d[0..n-1], labels y[0..n-1]):
    Sort pairs (d_i, y_i) by d_i ascending  → orderline
    best_ig = 0.0
    best_theta = d[0] - 1
    
    For t = 0 .. n-2:
        theta = (d[t] + d[t+1]) / 2   # midpoint between consecutive values
        ig = H(D) - ( (t+1)/n * H(D[0..t]) + (n-t-1)/n * H(D[t+1..n-1]) )
        if ig > best_ig:
            best_ig = ig
            best_theta = theta
    
    Return best_ig, best_theta
```

The split points at class-label boundaries in the orderline are sufficient (IG only changes at class-transition points); efficient implementations skip adjacent same-class pairs. (Confidence: HIGH for the overall algorithm; MEDIUM for the midpoint convention — some implementations use the right edge rather than the midpoint.)

**O(n log n)** per shapelet evaluation (dominated by sorting n distances). With n_candidates candidates, the total cost is O(n_candidates × n log n) for quality scoring, plus O(n_candidates × n × M) for the distance computations.

---

### E. Alternative Quality Measures

The following alternatives to binary information gain are used in practice (Confidence: MEDIUM — sourced from pyts source, Lines & Bagnall 2012, aeon docs):

**F-statistic (ANOVA):** Treats the distance distribution as a continuous feature and computes the one-way F-statistic against class labels. Used in pyts (`criterion='anova'`) and available in Lines & Bagnall (2012). Equivalent to running `scipy.stats.f_oneway` or sklearn `f_classif` on the distance vector. Higher = more discriminant. Advantage: O(n) computation, no sort needed (though sklearn's f_classif sorts internally).

**Mutual Information:** Used in pyts (`criterion='mutual_info'`). sklearn's `mutual_info_classif` with `discrete_features=False`. Captures non-monotone class-distance relationships that IG on a single split may miss.

**Kruskal-Wallis:** Non-parametric analog of F-statistic, listed in Lines & Bagnall (2012) but not commonly used in modern implementations.

**fdars recommendation:** Implement **information gain** (binary IG) as default (matches sktime RandomShapeletTransform and the Hills 2014 paper); expose **F-statistic** as an enum variant for the quality measure config (`QualityMeasure::InformationGain` | `QualityMeasure::FStatistic`). Mutual information deferred (no sklearn dependency; would require implementing MI estimator).

---

### F. Selection: Top-K with Self-Similarity Pruning

#### F.1 Top-K Selection

Maintain a bounded priority queue (max-heap by quality) of size K. After scoring all candidates, return the K highest-quality shapelets. In the contracted / random-sampling variant, the priority queue is updated incrementally as batches of candidates are evaluated.

#### F.2 Self-Similarity Pruning

Two shapelets are **self-similar** if they come from the same training series and their position ranges overlap:

```
Shapelet A: (series i, start a, length L_a)   → covers [a, a + L_a)
Shapelet B: (series i, start b, length L_b)   → covers [b, b + L_b)

A and B are self-similar iff:
    same_series(A, B) = true   AND
    NOT ( a + L_a <= b  OR  b + L_b <= a )   [i.e., intervals overlap]
```

After selecting shapelet A for the top-K list, all other candidates from series i with overlapping position range are removed from consideration. This prevents the top-K from being dominated by slight shifts of the same subsequence from the same curve. (Confidence: HIGH — exact logic extracted from pyts@0.13.x source: `remaining_idx = np.logical_and( np.logical_or(sorted_start_idx >= end, sorted_end_idx <= start), remaining_idx )`.)

**Algorithm:** Process candidates in descending quality order. When adding a candidate to the top-K:
1. Add it.
2. Mark all remaining candidates from the same series that overlap its position range as pruned.

This is O(n_candidates) for the pruning pass per selected shapelet.

---

### G. The Transform and Out-of-Sample Prediction

#### G.1 Shapelet Transform (fit)

After selecting K shapelets `{S_1, ..., S_K}`, the **shapelet transform** maps the training dataset to an n × K feature matrix:

```
X_transformed[i, j] = sdist(S_j, T_i)   for i in 0..n, j in 0..K
```

Each row is a curve, each column is the distance to one shapelet. This is a standard dense matrix — column j is a discriminative 1D feature derived from the local shape proximity to shapelet j.

#### G.2 Out-of-Sample Transform (predict)

For a new curve T_new, apply the same K saved shapelets:

```
x_new[j] = sdist(S_j, T_new)   for j in 0..K
```

The saved shapelets (their z-normalized subsequence data + metadata) must be stored in the fit result. This is the only state needed for out-of-sample application. Then feed `x_new` (shape 1 × K) to the stored classifier.

#### G.3 Bundled ShapeletTransformClassifier

The end-to-end classifier chains the above:

```
fit(training_curves, labels):
    1. Discover K shapelets (search + IG ranking + self-similarity pruning)
    2. Compute X_train = shapelet_transform(training_curves)  [n × K]
    3. Fit inner_classifier on (X_train, labels)
    4. Store shapelets + inner_classifier

predict(new_curves):
    1. X_new = shapelet_transform(new_curves)  [m × K]
    2. Return inner_classifier.predict(X_new)
```

The inner classifier reuses fdars' existing `classification/` module — any of LDA, kNN, kernel classifier, or QDA. This reuse is the primary reason the milestone is rated L-effort (the classifier itself already exists; the work is the discovery machinery). (Confidence: HIGH for the structure — directly from sktime ShapeletTransformClassifier `fit_transform → classifier.fit` pattern and the Hills 2014 paper description.)

---

### H. Complexity Summary

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Full candidate generation | O(n × M × (M - min_len)) | All (series, position, length) triples |
| sdist for one (shapelet, series) pair | O(M × L) | Sliding window of size M-L, each step O(L) |
| Quality (IG) for one shapelet | O(n log n) | n distances, sort + scan |
| Full exhaustive search | O(n² × M³) | All candidates × all training series × all windows |
| With early abandon | O(n² × M² × avg_abandon) | avg_abandon < L in practice |
| Contracted/random search | O(n_candidates × n × M) | n_candidates << total candidates |
| Shapelet transform (fit, no search) | O(K × n × M) | K shapelets × n series |
| Shapelet transform (predict) | O(K × m × M) | K shapelets × m test series |

**Mitigation for tractability:** The random/contracted variant (default in fdars) draws at most `max_candidates` random (series, start, length) triples, limiting search to O(max_candidates × n × M). Empirically, 10,000 candidates gives accuracy comparable to full search (sktime default `n_shapelet_samples=10000`). (Confidence: MEDIUM — cited in contracted shapelet transform literature and sktime defaults.)

---

## Feature Landscape

### Table Stakes — Users Expect These

These features constitute the shapelet transform capability. Missing any one makes the milestone deliverable incomplete.

| Feature | Why Expected | Complexity | Implementation Notes |
|---------|--------------|------------|----------------------|
| Z-normalization of subsequences | Without it the "shape" distance becomes a scaled-amplitude distance; the entire algorithm's validity depends on this step | LOW | `fn znorm_slice(window: &[f64]) -> Vec<f64>`: subtract mean, divide by std; return zero-vec if std=0 |
| `sdist(shapelet: &[f64], curve: &[f64]) -> f64` — sliding-window minimum z-normalized L2 | The atomic primitive; every other feature is built on it | LOW | Sliding window of length `shapelet.len()` over `curve`; z-normalize each window; compute squared L2; return sqrt of min; early-abandon optional |
| Candidate generation over lengths `[min_len, max_len]` | The search space definition; must be configurable | LOW | Enumerate `(series_idx, start, length)` triples; for random/contracted mode, sample uniformly from this space |
| Information gain quality measure with optimal split threshold | The standard discriminative scoring function from Ye & Keogh and Hills 2014; expected by every shapelet practitioner | MEDIUM | Sort n distances + labels into orderline; scan midpoints; compute IG = H(D) - weighted child entropies; O(n log n) per candidate |
| Top-K selection with self-similarity pruning | Without pruning, top-K is filled by shifted variants of the same subsequence | MEDIUM | Max-heap of K; after inserting from series i at [start, start+L), mark overlapping candidates from series i as pruned; process in quality-descending order |
| `ShapeletDiscovery::fit(data, labels, config) -> Result<ShapeletSet, FdarError>` | The fit API that runs candidate search + scoring + selection | HIGH | Outer loop: iterate candidates; inner loop: compute sdist against all n series; score; update top-K; return `ShapeletSet` |
| `shapelet_transform(data: &FdMatrix, shapelets: &ShapeletSet) -> Result<FdMatrix, FdarError>` | The transform step — produces the n×K distance feature matrix for training | MEDIUM | For each (series, shapelet) pair: `sdist`; assemble into column-major FdMatrix of shape (n, K) |
| Out-of-sample `shapelet_transform` on new data (same function, new FdMatrix) | Prediction requires re-applying saved shapelets to test series | LOW | Same function as above; shapelets stored in `ShapeletSet` struct |
| `ShapeletTransformClassifier::fit(data, labels) -> Result<ShapeletTransformClassifierFit, FdarError>` | End-to-end bundled classifier matching sktime's `ShapeletTransformClassifier` | MEDIUM | Calls `ShapeletDiscovery::fit`, `shapelet_transform`, then `fclassif_*` from existing `classification/` |
| `ShapeletTransformClassifierFit::predict(new_data) -> Result<Vec<usize>, FdarError>` | The predict method; transforms new data with stored shapelets, then routes to stored inner classifier | LOW | `shapelet_transform(new_data)` → `classifier.predict(...)` |
| `ShapeletConfig` struct with `min_len`, `max_len`, `max_candidates`, `k_shapelets`, `quality`, `seed` | fdars convention for complex method configuration | LOW | Mirrors `GmmClusterConfig`, `ElasticConfig`; `serde`-gated |
| `ShapeletSet` result type with shapelet data + series/position/length metadata + quality scores | Must store enough to reconstruct transform for new data | LOW | `Vec<Shapelet>` where `Shapelet { data: Vec<f64>, series_idx, start, length, quality, split_threshold }` |
| `ShapeletTransformClassifierFit` storing `ShapeletSet` + inner classifier result | State needed for predict | LOW | Composite struct; the inner classifier result must be one of the existing fdars classifier result types |
| `Debug + Clone + PartialEq` derives on all public result types | fdars convention across 97+ types | LOW | Standard — no exceptions |
| `Result<T, FdarError>` on all public functions | fdars error-handling convention | LOW | Dimension checks at entry: curves same length, labels match, min_len ≤ max_len ≤ M, k_shapelets > 0 |
| Inline tests: sdist correctness, IG computation, self-similarity pruning, transform shape, round-trip | Gates for correctness before shipping | MEDIUM | Synthetic small datasets where optimal shapelet is known analytically |

### Differentiators — Competitive Advantage

These features raise the quality bar and match reference-library behavior, but are not blockers for a correct v1 shapelet transform.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Early-abandon optimization in sdist | Abandons distance computation as soon as running minimum exceeds current best-so-far; reduces O(M × L) to O(avg_abandon × L) per window in practice; critical for tractability in large M | MEDIUM | Maintain `best_so_far: f64`; in the inner z-norm + squared-diff loop, accumulate partial sum and break if partial_sum > best_so_far² (comparing before sqrt). Must z-normalize the full window first (z-norm requires full pass), so early abandon applies only to the Euclidean distance loop not the z-norm pass — still a meaningful win |
| Contracted / random sampling mode (default) | Limits search to `max_candidates` random (series, start, length) triples; makes n=500, M=300 tractable in seconds; sktime's default (`n_shapelet_samples=10000`) | LOW | `ShapeletConfig.max_candidates: Option<usize>` where `None` = full enumeration, `Some(K)` = random sample; use `StdRng::seed_from_u64(seed)` for reproducibility |
| F-statistic quality measure as `QualityMeasure::FStatistic` enum variant | Alternative to IG; simpler to compute (O(n), no sort); good for multiclass; pyts `criterion='anova'` | LOW | `fn f_stat_quality(distances: &[f64], labels: &[usize]) -> f64`: compute between-class and within-class variance; return F = between/within |
| Rayon parallelism over candidate evaluation | Each candidate's sdist-against-all-training-series is independent; parallelism gives near-linear scaling for the discovery phase | LOW | `iter_maybe_parallel!(candidates).map(|c| score_candidate(c, data, labels)).collect()` — same pattern as existing parallel CV |
| `#[must_use]` on `ShapeletDiscovery::fit`, `shapelet_transform`, `ShapeletTransformClassifier::fit` | fdars convention for expensive computations; 74+ functions already annotated | LOW | One attribute per expensive function |
| Criterion benchmark in `benches/` | fdars convention; measures discovery time vs n and M; documents the tractability profile | LOW | Two-cell grid: (n=50, M=100), (n=100, M=200); both contracted (max_candidates=1000) and time (wall-clock for n=100 full enumeration) |
| Serde support on `ShapeletConfig`, `ShapeletSet`, `ShapeletTransformClassifierFit` | Pipeline persistence — save fitted transform, reload for production predict; standard fdars `serde`-feature convention | LOW | `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` on all config/result types |
| Configurable inner classifier (`InnerClassifier` enum or trait object) | Power users want to swap LDA / kNN / kernel without rewriting; matches sktime's `estimator` parameter | MEDIUM | `ShapeletConfig.classifier: InnerClassifierKind` enum (`Lda | Knn(usize) | Kernel`); dispatches to existing fdars classifiers |
| `ShapeletTransformClassifierFit::predict_proba` (posterior probabilities) | Calibrated probabilities useful for ensembles and uncertainty estimation | MEDIUM | Only implementable if inner classifier supports it — LDA/QDA return posteriors natively; kNN proportional |

### Anti-Features — Explicitly Out of Scope

| Anti-Feature | Why Requested | Why Excluded | What to Do Instead |
|--------------|---------------|---------------|--------------------|
| Learning-shapelets (Grabocka 2014 / tslearn `ShapeletModel`) | Gradient-descent learned shapelets often achieve higher accuracy than discovery-based | Fundamentally different algorithm — requires automatic differentiation or manual gradient derivation, a separate optimization loop (SGD), and no reuse of the existing candidate-based infrastructure. Milestone context explicitly states "NOT tslearn learning-shapelets (deferred/out of scope)." | Implement as a future GAP item (differentiable FDA — GAP-08) once an AD-compatible Rust approach is decided |
| GPU acceleration | Distance computation is embarrassingly parallel; GPU gives large speedup for n>1000 | fdars targets CPU/WASM; no GPU infrastructure exists anywhere in the crate | Use rayon CPU parallelism (differentiator above) + early abandon + contracted sampling — sufficient for research-scale datasets |
| SAX / PAA / symbolic representations | Often discussed alongside shapelet methods; pyts has both | SAX/PAA are symbolic/imaging TS-ML representations, not functional numeric methods — explicitly recorded as OOS-02 in GAP-BACKLOG.md | Not applicable — different domain entirely |
| DTW-based shapelet distance (replacing Euclidean with DTW) | Elastic distance may give better shape matching in some domains | Breaks the fixed-length window assumption; substantially more complex; the literature consensus is z-normalized Euclidean is sufficient (and orders of magnitude faster) | Apply elastic alignment (existing fdars `alignment/`) as a preprocessing step on the raw curves before shapelet transform if DTW-like invariance is needed |
| Multivariate shapelet transform (per-channel distances) | Applicable to multivariate time series; in the literature (aeon `RandomShapeletTransform`) | `FdMatrix` is single-channel curves (n × M); multivariate extension requires a different data representation not yet decided — out of scope | Implement as a follow-on once multivariate `FdMatrix` / `FdCurveSet` representation is settled |
| Native support for irregular-length series | Some datasets have varying-length series | Each sdist call requires shapelet length ≤ curve length; irregular lengths need per-curve length checks and more complex candidate generation | Apply `spline_interpolate` (existing in `helpers.rs`) to regularize lengths before shapelet discovery |
| Shapelet ensemble (HIVE-COTE component) | Shapelet Transform Classifier is one component of the HIVE-COTE ensemble in sktime | Ensemble infrastructure would require a separate phase; far beyond the single-algorithm scope of v0.33.0 | Implement STC correctly; ensembling is a future milestone |

---

## Feature Dependencies

```
znorm_slice(window)           (new utility in helpers.rs)
    │
    └──required-by──> sdist(shapelet, curve)    (new in shapelet/distance.rs)
                           │
                           ├──required-by──> ShapeletDiscovery::fit()
                           │                     │
                           │                     ├──requires──> info_gain_quality()
                           │                     │                 │
                           │                     │                 └──requires──> sort + entropy scan (std)
                           │                     │
                           │                     ├──requires──> self_similarity_prune()
                           │                     │
                           │                     └──returns──> ShapeletSet { Vec<Shapelet> }
                           │                                         │
                           └──required-by──> shapelet_transform()    │
                                                 │                   │
                                                 │◄──────────────────┘
                                                 │
                                                 ├──produces──> FdMatrix (n × K distance features)
                                                 │
                                                 └──consumed-by──> ShapeletTransformClassifier::fit()
                                                                         │
                                                                         ├──calls──> fclassif_* (existing classification/)
                                                                         │
                                                                         └──returns──> ShapeletTransformClassifierFit
                                                                                             │
                                                                                             └──predict()
                                                                                                   │
                                                                                                   └──calls──> shapelet_transform() + classifier.predict()
```

### Dependency Notes

- **`sdist` requires `znorm_slice`:** z-normalization is a mandatory preprocessing step on every window. It must be applied to the shapelet once (pre-normalize before storage) and to each candidate window at query time. A constant window (std=0) must not produce NaN — return the zero vector.
- **`ShapeletDiscovery::fit` requires `info_gain_quality`:** quality scoring is the inner-most function called for every candidate. Its performance dominates the discovery runtime for small M. Must be O(n log n) per call.
- **`self_similarity_prune` is called inside `ShapeletDiscovery::fit`:** pruning must happen incrementally during selection, not as a post-processing step on the full top-K. Otherwise self-similar candidates fill the heap before non-similar ones are considered.
- **`shapelet_transform` requires a `ShapeletSet`:** the transform function is stateless — it takes the curve data and the saved shapelets and produces the feature matrix. No fit state beyond the `ShapeletSet` is needed for the transform itself.
- **`ShapeletTransformClassifierFit::predict` requires both `ShapeletSet` and the inner classifier fit result:** both must be stored in the composite result struct. The inner classifier must be one of the existing fdars classifier fit types (e.g., `ClassifFit` from `classification/fit.rs`).
- **Contracted mode (`max_candidates: Some(K)`) is independent of `ShapeletDiscovery::fit`'s interface:** the same public function handles both modes; the internal sampling vs. enumeration is a config-controlled detail.

---

## MVP Definition

### Launch With (v0.33.0)

Minimum viable for the milestone to ship:

- [ ] `znorm_slice(window: &[f64]) -> Vec<f64>` — z-normalization helper (std=0 → zero vec)
- [ ] `sdist(shapelet: &[f64], curve: &[f64]) -> f64` — sliding-window minimum z-normalized Euclidean distance
- [ ] `info_gain_quality(distances: &[f64], labels: &[usize]) -> (f64, f64)` — returns (best_ig, best_split_threshold); O(n log n) via sorting the orderline
- [ ] `ShapeletDiscovery::fit(data: &FdMatrix, labels: &[usize], config: &ShapeletConfig) -> Result<ShapeletSet, FdarError>` — random/contracted candidate generation + IG scoring + top-K + self-similarity pruning
- [ ] `shapelet_transform(data: &FdMatrix, shapelets: &ShapeletSet) -> Result<FdMatrix, FdarError>` — n×K distance feature matrix; works for both training and out-of-sample
- [ ] `ShapeletTransformClassifier::fit(data, labels, config) -> Result<ShapeletTransformClassifierFit, FdarError>` — discover → transform → fit inner classifier (default: kNN with k=1 or LDA as configurable option)
- [ ] `ShapeletTransformClassifierFit::predict(new_data: &FdMatrix) -> Result<Vec<usize>, FdarError>` — transform → classify
- [ ] `ShapeletConfig { min_len, max_len, max_candidates, k_shapelets, quality: QualityMeasure, seed, classifier: InnerClassifierKind }` — all parameters with sensible defaults
- [ ] `ShapeletSet { shapelets: Vec<Shapelet> }` where `Shapelet { data, series_idx, start, length, quality, split_threshold }`
- [ ] All dimension validation error paths (`InvalidDimension`, `InvalidParameter`)
- [ ] Inline tests: sdist (known-answer), IG (hand-computed 2-class split), self-similarity pruning (overlapping candidates removed), transform shape check (n × K), classifier round-trip (fit→predict on synthetic data)

### Add After Validation (v0.33.x)

- [ ] Early-abandon optimization in `sdist` — enables tractability for M > 200
- [ ] F-statistic quality measure (`QualityMeasure::FStatistic`) — alternate criterion
- [ ] Criterion benchmark (discovery time vs n; transform time vs K and M)
- [ ] Example file `examples/shapelet_classification.rs`
- [ ] Serde support on `ShapeletConfig`, `ShapeletSet`, `ShapeletTransformClassifierFit`

### Future Consideration (v0.34+)

- [ ] Learning-shapelets (gradient-based, Grabocka 2014) — requires AD / differentiable FDA (GAP-08)
- [ ] Multivariate shapelet transform — requires multivariate FdMatrix representation decision
- [ ] Shapelet ensemble (HIVE-COTE component) — separate milestone
- [ ] Dilated shapelet transform (RDST) — modern variant, separate milestone

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| `znorm_slice` + `sdist` | HIGH | LOW | P1 — foundational primitive; everything depends on it |
| `info_gain_quality` + optimal split | HIGH | MEDIUM | P1 — quality scoring is the core of discovery |
| Self-similarity pruning | HIGH | LOW | P1 — without it, top-K is meaningless |
| `ShapeletDiscovery::fit` (contracted mode) | HIGH | HIGH | P1 — headline deliverable, discovery machinery |
| `shapelet_transform` (fit + predict) | HIGH | MEDIUM | P1 — the transform step; prediction depends on it |
| `ShapeletTransformClassifierFit::predict` | HIGH | LOW | P1 — the bundled end-to-end classifier |
| `ShapeletConfig` struct | HIGH | LOW | P1 — fdars convention, needed for all above |
| Early-abandon optimization | MEDIUM | MEDIUM | P2 — correctness ships without it; critical for performance |
| F-statistic quality measure | MEDIUM | LOW | P2 — useful for multiclass; trivial once IG exists |
| Rayon parallelism | MEDIUM | LOW | P2 — follows existing pattern; needed for n>100 datasets |
| Criterion benchmark | LOW | LOW | P2 — fdars convention |
| Learning-shapelets | HIGH | VERY HIGH | P3 — out of scope this milestone |
| Multivariate support | MEDIUM | HIGH | P3 — blocked on representation decision |
| GPU acceleration | LOW | VERY HIGH | P3 — out of scope entire crate |

---

## Competitor Feature Analysis

| Feature | sktime `RandomShapeletTransform` | pyts `ShapeletTransform` | fdars v0.33.0 Plan |
|---------|----------------------------------|--------------------------|-------------------|
| Distance | z-normalized Euclidean (min over windows) | Min MSE over windows (equivalent) | `sdist` with z-normalization (explicit) |
| Quality measure | Information gain (default) | Mutual info or ANOVA F-stat | IG (default) + F-stat (enum variant) |
| Candidate selection | Random sample (n_shapelet_samples=10000) | All windows (or windowed) | Random/contracted (max_candidates config) |
| Self-similarity pruning | `remove_self_similar=True` default | `remove_similar=True` default | Self-similarity pruning enabled by default |
| Min/max shapelet length | `min_shapelet_length=3`, `max_shapelet_length=None` | `window_sizes='auto'` | `min_len=3`, `max_len=None` (→ M/2) |
| Top-K | `max_shapelets=None` (→ `min(10*sqrt(n_timeseries)*n_classes, n_shapelet_samples)`) | `n_shapelets='auto'` | `k_shapelets` (explicit) |
| Time contract | `time_limit_in_minutes=0` (disabled) | Not supported | `max_candidates: Option<usize>` |
| Bundled classifier | Via `ShapeletTransformClassifier`; default RotationForest | Separate (transform only) | `ShapeletTransformClassifier`; configurable inner classifier from existing fdars |
| Parallelism | `n_jobs` | `n_jobs` | `iter_maybe_parallel!` under `parallel` feature |
| Serde | Not supported | Not supported | Under `serde` feature (fdars convention) |
| Out-of-sample transform | `transform(X_test)` reusing fitted shapelets | `transform(X_test)` | `shapelet_transform(new_data, &shapelets)` |

---

## Sources

- Ye, L., Keogh, E. (2009). "Time series shapelets: a new primitive for data mining." KDD 2009. https://dl.acm.org/doi/10.1145/1557019.1557122
- Hills, J., Lines, J., Baranauskas, E., Mapp, J., Bagnall, A. (2014). "Classification of time series by shapelet transformation." Data Mining and Knowledge Discovery 28(4), 851–881. https://link.springer.com/article/10.1007/s10618-013-0322-1
- Lines, J., Bagnall, A. (2012). "Alternative quality measures for time series shapelets." Intelligent Data Engineering and Automated Learning – IDEAL 2012.
- sktime `RandomShapeletTransform` API (v0.3x / aeon): https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.transformations.collection.shapelet_based.RandomShapeletTransform.html
- sktime `ShapeletTransformClassifier` API: https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.classification.shapelet_based.ShapeletTransformClassifier.html
- pyts@0.13.x `ShapeletTransform` API docs: https://pyts.readthedocs.io/en/latest/generated/pyts.transformation.ShapeletTransform.html
- pyts@0.13.x `ShapeletTransform` source (self-similarity pruning logic extracted): https://pyts.readthedocs.io/en/stable/_modules/pyts/transformation/shapelet_transform.html
- tslearn@0.9.0 shapelets user guide (sdist formula): https://tslearn.readthedocs.io/en/stable/user_guide/shapelets.html
- GENDIS paper (shapelet survey, sdist formula, complexity): https://arxiv.org/pdf/1910.12948
- Multivariate shapelet transform paper (ar5iv 1712.06428, sdist formula confirmed): https://ar5iv.labs.arxiv.org/html/1712.06428
- GAP-BACKLOG.md GAP-02 (v0.31.0): `.planning/research/GAP-BACKLOG.md`
- survey-pyx.md PYX-01 (v0.31.0): `.planning/research/survey-pyx.md`
- fdars-core `classification/` (existing LDA/QDA/kNN/kernel classifiers): local codebase
- fdars-core `distance.rs` (existing distance primitives): local codebase
- fdars-core `helpers.rs` (`znorm_slice` would be added here alongside existing helpers): local codebase

---
*Feature research for: v0.33.0 Shapelet Transform & Classification (GAP-02)*
*Researched: 2026-09-02*
