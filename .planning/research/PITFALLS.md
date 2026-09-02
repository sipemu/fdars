# Pitfalls Research

**Domain:** Discovery-based shapelet transform + shapelet-based classifier — Rust numerical FDA library (fdars-core v0.33.0)
**Researched:** 2026-09-02
**Confidence:** HIGH

---

## Critical Pitfalls

### Pitfall 1: Per-Window Z-Normalization Applied Only Once to the Whole Series

**What goes wrong:**
The z-normalization in shapelet distance is *per window*, applied independently to every length-L sliding window before computing the Euclidean distance to the (also z-normalized) shapelet. Normalizing the whole series once at the start — or normalizing each curve once and reusing it — produces an algorithm that is sensitive to scale and offset, breaks translation invariance, and returns distances that are meaningless relative to a correctly normalized baseline. Shapelets will appear everywhere or nowhere depending on the absolute magnitude of the input series.

The correct formula for the distance from shapelet `s` (length L, pre-normalized) to a window `w[t..t+L]` from series `x` is:
```
znorm(v) = (v - mean(v)) / max(std(v), ε)
d(s, w) = euclidean(znorm(s), znorm(w[t..t+L]))
```
Both the shapelet and every candidate window are normalized *independently* at the time of comparison — not the whole series up front.

**Why it happens:**
Z-normalization of time series is standard preprocessing (e.g. `functional_std`, `impute_missing_values` in `fdata.rs`), so developers apply the crate's existing series-level normalization pass and assume it covers the shapelet case. It does not. The per-window requirement is unique to shapelet distance and is easy to miss in papers that describe it in passing as "z-normalized" without explicitly saying "per window."

**How to avoid:**
- `shapelet_distance` must call `znorm_window` (a private helper) inside the sliding-window loop on every `&x[t..t+L]` slice — never on the full series.
- The shapelet candidate itself is normalized once at generation time and stored already normalized.
- Guard against constant windows (std < ε) by clamping: `std = std.max(1e-10)` — never `std.max(0.0)` (that still divides by nearly-zero).
- Add a doctest that shows a shifted copy of a series produces the same distance as the original (offset-invariance proof).

**Warning signs:**
- Distance matrix has suspiciously high variance across series of different scales.
- Shapelets discovered from large-valued series never match small-valued ones.
- Unit test: two series differing only by a constant offset have identical shapelet distances → if this fails, per-window normalization is broken.

**Phase to address:** Phase 57 (shapelet-distance core). This is the foundation; every downstream step inherits the bug if it is wrong here.

**Verification hook:**
```
// offset-invariance test
let x = vec![1.0, 2.0, 3.0, 2.0, 1.0];
let x_shifted = x.iter().map(|v| v + 100.0).collect::<Vec<_>>();
assert!((shapelet_dist(&s, &x) - shapelet_dist(&s, &x_shifted)).abs() < 1e-10);
// scale-invariance test
let x_scaled = x.iter().map(|v| v * 50.0).collect::<Vec<_>>();
assert!((shapelet_dist(&s, &x) - shapelet_dist(&s, &x_scaled)).abs() < 1e-10);
```

---

### Pitfall 2: Division by Near-Zero Standard Deviation in Z-Normalization

**What goes wrong:**
Any window that is exactly or nearly constant (all values equal, or a ramp with tiny variation relative to f64 precision) has `std ≈ 0`. Dividing by it produces `±Inf` or `NaN`. One `NaN` in the distance matrix contaminates the info-gain ranking for every shapelet — the whole discovery loop silently produces `NaN` scores, and the top-K selection returns garbage.

This is especially common at series boundaries: a window sliding past the end of a short series may be padded with zeros (if the implementation uses zero-padding instead of clamping), turning a real window into an almost-constant one. It also hits synthetic datasets used in unit tests (constant baseline + a single spike).

**Why it happens:**
The `FdMatrix` column-major layout means row slices are non-contiguous; a hand-written sliding-window loop that computes mean/std over a raw index range can accidentally sum zeros from uninitialized memory if the bounds are off-by-one. The standard deviation formula also involves a subtraction-of-squares that cancels to near-zero for nearly-constant windows in f64 arithmetic — even when the true std is nonzero.

**How to avoid:**
- Always clamp: `let std_clamped = std.max(1e-10);` — the threshold 1e-10 matches the tolerance used in the existing `svd_equivalence` integration test convention in this codebase.
- Separately track whether a window was clamped; if more than X% of windows are clamped for a given series, consider returning a `FdarError::InvalidParameter` or at minimum a doc-noted caveat.
- Never use zero-padding when the window overruns the series — use `InvalidDimension` guard at candidate generation to reject shapelets longer than the shortest series.
- Use a two-pass mean/std (compute mean first, then std from deviations) rather than the single-pass `E[X²] - E[X]²` formula, which is numerically unstable for nearly-constant sequences.

**Warning signs:**
- `shapelet_transform` returns a matrix containing `NaN` or `Inf`.
- Info-gain scores are `NaN` for all candidates.
- Adding a check `assert!(result.iter().all(|v| v.is_finite()))` to the transform catches this at test time.

**Phase to address:** Phase 57 (shapelet-distance core), same function as Pitfall 1.

**Verification hook:**
- Constant-window test: `x = vec![5.0; 20]`, any shapelet; distance must be finite.
- Nearly-constant test: `x = vec![5.0; 20]` with one element perturbed by `1e-15`; distance must be finite.

---

### Pitfall 3: Shapelet Distance Is the Minimum, Not Mean or Sum

**What goes wrong:**
The sliding-window minimum is definitional to shapelet distance:
```
sdist(s, x) = min_{t=0..m-L} euclidean(znorm(s), znorm(x[t..t+L]))
```
Using the mean or sum of window distances makes the measure sensitive to how often the shapelet pattern appears, not whether it appears at all. Shapelets that occur once in the middle of a series are correctly detected by min; they are drowned out by non-matching windows under mean.

**Why it happens:**
`rayon`/`iter_maybe_parallel!` over windows with `.sum()` or `.map().collect::<Vec<_>>().iter().sum()` is a natural reflex for "compute something over all windows." Switching to `.fold(f64::INFINITY, f64::min)` or `.reduce_with(f64::min)` is the correct pattern but requires conscious intent.

**How to avoid:**
- Implement the inner loop as `windows.map(|w| znorm_euclidean(s, w)).fold(f64::INFINITY, f64::min)`.
- For early-abandon (see Pitfall 5), track a running best and short-circuit.
- Unit test: a series containing the shapelet pattern once in a sea of noise → `sdist` must be ~0 (within normalization tolerance); the mean would be large.

**Warning signs:**
- Shapelet classifier accuracy barely above random on a dataset with a clear discriminative motif.
- Known-motif recovery test fails: synthetic dataset `class_0 = noise`, `class_1 = noise + known_pattern at random offset` — the shapelet equal to `known_pattern` must achieve `sdist ≈ 0` for `class_1` and `sdist > threshold` for `class_0`.

**Phase to address:** Phase 57 (shapelet-distance core).

**Verification hook (known-motif recovery):**
```rust
// class_0: 50 curves of pure N(0,1) noise, length 100
// class_1: 50 curves with known_pattern (length 10) inserted at random offset
// shapelet = known_pattern (z-normalized)
// after discovery: top shapelet must achieve mean sdist < 0.2 on class_1
//                  and mean sdist > 1.5 on class_0
```

---

### Pitfall 4: Combinatorial Blowup — Naive All-Subsequences-All-Lengths Is Intractable

**What goes wrong:**
For n training curves of length m, the number of candidate shapelets (all subsequences across all lengths from `min_len` to `max_len`) is:
```
n * Σ_{L=min_len}^{max_len} (m - L + 1) ≈ n * m² / 2
```
For n=200, m=500, this is ~25 million candidates. The full `sdist` computation for each candidate against all n series is O(m) per series, making the naive discovery O(n² * m³) in the worst case. At m=500, this does not complete in human time — it takes hours or days.

Without a time contract or candidate sampling, the `fit` function never returns on any real-world dataset. This is the most common reason shapelet implementations are judged "unusable" in practice.

**Why it happens:**
The algorithm description in Hills et al. (2014) and the original Ye & Keogh paper present the complete search as the formal definition. Practitioners reading papers implement the complete search first. The time contract (sampling candidates uniformly until a wall-clock budget is exhausted) was added to sktime's `ShapeletTransformClassifier` specifically because the complete search is intractable.

**How to avoid:**
- Add a `time_contract_minutes: Option<f64>` (or `max_candidates: Option<usize>`) parameter to the discovery function. Enforce at the outer candidate loop.
- Sample candidate lengths uniformly from `[min_len, max_len]` rather than enumerating all lengths exhaustively.
- Default to a sensible contract (e.g. `max_candidates = 10_000` or `time_contract_minutes = 5.0`).
- Document the default contract in the function's rustdoc so users know it is not an exhaustive search.
- Return the actual number of candidates evaluated in the `ShapeletFitResult` so the caller can tell if the contract was hit.

**Warning signs:**
- `fit` call does not return after 30+ seconds on n=50, m=200 in a test.
- Benchmark shows super-quadratic scaling when doubling m.

**Phase to address:** Phase 58 (discovery & ranking). The contract must be designed into the candidate-loop API from the start, not bolted on later.

**Verification hook:**
- Tractability test: n=100, m=200, `time_contract_minutes=0.1` → `fit` must return within ~10 seconds (contract + overhead).
- Contract-hit flag in result: `result.contract_hit == true` when the budget is exhausted before all candidates are evaluated.

---

### Pitfall 5: No Early-Abandon in the Sliding-Window Distance Scan

**What goes wrong:**
For a shapelet of length L and a series of length m, the sliding-window loop performs (m-L+1) z-normalized Euclidean distance computations. Each takes O(L) time. Without early-abandon, even a hopeless distance computation (already larger than the current best by the first 5 points) runs to completion. Early-abandon is the primary constant-factor speedup for the inner loop.

At discovery time, each candidate shapelet is evaluated against all n series; the expected speedup from early-abandon is 2–8× depending on data, which is the difference between 5 minutes and 40 minutes on medium datasets.

**Why it happens:**
The idiomatic Rust approach — `windows.map(|w| dist(s, w)).fold(INFINITY, min)` — does not naturally early-abandon. Adding early-abandon requires tracking a running best and breaking out of the inner loop early. Rayon iterators do not support break; early-abandon requires a sequential inner loop with an explicit `break` or a custom accumulator.

**How to avoid:**
- The inner z-normalized distance computation must be a sequential loop (not a rayon parallel map) that accumulates partial sum-of-squares and short-circuits when the partial sum already exceeds `current_best²`.
- Note: z-normalization complicates early-abandon because the mean/std of the window must be known before any distance term can be computed. The standard fix is to compute mean/std in a first pass (O(L)), then accumulate the Euclidean distance in a second pass with early-abandon.
- The outer loop over windows (for a fixed shapelet vs. a fixed series) can remain sequential for cache-friendliness.
- Rayon parallelism applies at the level of series (outer: all n series) or candidate shapelets (outer: all candidates), not within the window scan.

**Warning signs:**
- Inner-loop profiling shows near-100% utilization with no early returns.
- Removing the early-abandon condition does not change runtime — the condition is never firing, meaning the threshold is set too loosely or the bound is computed incorrectly.

**Phase to address:** Phase 57 (shapelet-distance core) — the `shapelet_distance` function must expose a `best_so_far: f64` parameter for early-abandon from the start, or the API must be revisited in Phase 58 when discovery calls it in a loop.

**Verification hook:**
- Compare `shapelet_distance` runtime with and without early-abandon on a dataset where the shapelet is not present (worst case for early-abandon); expect 2× or better.

---

### Pitfall 6: Information Gain Computed Without Optimal Split Threshold

**What goes wrong:**
The discriminative power of a shapelet is measured by the information gain (or F-statistic) of the binary split on the distance distribution. The split threshold must be the *optimal* one — searched over the ordered distance values. Using a fixed threshold (e.g., the median, or 0.5) instead of optimizing yields consistently poor shapelet quality scores, causing genuinely discriminative shapelets to be ranked low and weak ones to be ranked high.

The optimal threshold search scans all n-1 gaps in the sorted distance vector, computes the two-class entropy (or F-statistic) for each, and picks the threshold that maximizes information gain. This is O(n log n) per candidate (sorting) + O(n) for the scan.

**Why it happens:**
Information gain in decision trees is normally computed against a fixed split target (class label). The shapelet case is different: the "feature" (distance) and its "split" (threshold) are jointly optimized. Developers familiar with decision tree IG but not shapelet-specific IG often skip the threshold search and use a heuristic.

**How to avoid:**
- Implement `best_ig(distances: &[f64], labels: &[usize]) -> (f64, f64)` (returns `(info_gain, optimal_threshold)`) that sorts by distance, scans all n-1 gaps, and returns the max.
- Never use a fixed threshold in the ranking step.
- Return the optimal threshold alongside the quality score in the candidate struct so it can be reused at transform time (the transform is `sdist(s, x)` which needs no threshold — but the threshold is needed if building a decision stump on top).

**Warning signs:**
- Top-ranked shapelets change drastically depending on a threshold hyperparameter.
- Known-motif recovery test (Pitfall 3) fails even though the correct shapelet has low sdist on the target class.
- Random permutation of the distance vector produces similar IG scores to the real vector (sign that no meaningful split was found).

**Phase to address:** Phase 58 (discovery & ranking).

**Verification hook:**
- Synthetic dataset: two classes differ by a known pattern. After ranking, `best_ig` for the true shapelet must be near `max_entropy` (log2 of class count); a random candidate must score near 0. Validate by comparing `ig_score` between known-discriminative and known-random shapelets.

---

### Pitfall 7: Training-Set Leakage — Quality Computed on Data Used for Final Classifier

**What goes wrong:**
If shapelet quality (info-gain) is computed on the full training set and the shapelet-transform features are then used to train a classifier on the same training set, there is no leakage *in the shapelet quality step* — the quality measure is a univariate score, not a model. However, there is a subtler leakage risk: if the shapelet quality evaluation loop also tunes the internal classifier's hyperparameters (e.g., runs nested CV to tune kNN's k), the whole pipeline overfits.

The more common mistake is reporting accuracy on the training set after `fit` to "check the pipeline works" and treating it as a generalization estimate. The transform + classifier is memorizing the training set via the selected shapelets, which were chosen specifically to separate training classes.

**Why it happens:**
The `ShapeletTransformClassifier::fit` → `.predict(train_data)` evaluation is a natural smoke test during development. The result looks plausible (high accuracy) and is misread as generalization performance.

**How to avoid:**
- The `fit` function must only receive the training split; never pass test data into `fit`.
- Document explicitly that shapelet quality is computed on training data only (this is correct by the Hills/Lines STC design).
- In unit tests and doctests, always evaluate on a held-out test split, not the training split.
- The example in the rustdoc must use `train_data`/`test_data` splits, not the full dataset.

**Warning signs:**
- Accuracy on `predict(train_data)` is 100% or near it.
- Accuracy on held-out test data is much lower than training accuracy (>20% gap on typical datasets).

**Phase to address:** Phase 60 (bundled `ShapeletTransformClassifier`). The split discipline must be enforced in the doctest and example.

**Verification hook:**
- Integration test: train on 80% split, predict on 20% held-out; accuracy must be above chance. Training-set accuracy must not be reported as generalization accuracy anywhere in tests.

---

### Pitfall 8: Self-Similarity Pruning Omission — Top-K Is Redundant

**What goes wrong:**
Without self-similarity pruning, the top-K shapelets are almost always dominated by slight variants of the same subsequence from the same training series. If series `i` contributes the best shapelet at offset `t`, then offset `t+1` from the same series is nearly as good (all z-normalized Euclidean distances differ by one step). The top-K list collapses to K copies of one series' best region, and the transformed feature matrix has K near-identical columns — it carries no more information than one column and wastes computation.

**Why it happens:**
Self-similarity pruning requires tracking which training series and position each selected shapelet came from, then rejecting candidates whose *source series* is already represented in the selected set (or whose distance to an already-selected shapelet is below a threshold). Developers implementing the quality-sorted selection loop forget to add this bookkeeping.

**How to avoid:**
- After sorting candidates by quality, greedily select: add a candidate only if its source series index has not already contributed a shapelet to the selected set (Hills et al. 2014 rule: at most one shapelet per series, or at most one per non-overlapping position).
- Alternatively, compute mutual distance between candidate shapelets and reject if `sdist(s_candidate, s_already_selected) < ε` for any already-selected shapelet.
- Store `(series_idx, start_offset)` in the candidate struct from the start.

**Warning signs:**
- The shapelet transform feature matrix has near-identical (correlation > 0.99) columns.
- `cargo clippy` clean but classifier accuracy does not improve with K > 2 even on well-separated datasets.

**Phase to address:** Phase 58 (discovery & ranking).

**Verification hook:**
- Post-selection correlation check: `max_{i≠j} corr(transform_col_i, transform_col_j) < 0.95` on any dataset with at least two natural classes.
- Series-diversity check: the selected K shapelets must come from at least `min(K, n_train)` distinct training series.

---

### Pitfall 9: Transform Inconsistency — Test Transform Uses Different Shapelets or Re-Normalizes

**What goes wrong:**
The out-of-sample `transform` must apply the *exact* shapelets discovered during `fit` — same sequences, same length, same pre-stored z-normalization. Two failure modes:
1. The transform recomputes shapelets from the test data (completely wrong — shapelets must be fixed from training).
2. The transform applies a different normalization to the shapelet itself on each call (e.g., re-normalizes against the test series statistics instead of using the stored normalization from fitting).

Either failure makes `train_transform ≠ test_transform` in distribution, corrupting the downstream classifier.

**Why it happens:**
In a pure-function codebase (no mutable state after fit), it is tempting to keep the shapelet discovery logic and the transform logic as one combined function. Separating them cleanly requires storing the fitted shapelets in the result struct. If the shapelet is stored un-normalized and re-normalized at transform time using the test series' statistics, the bug is introduced.

The existing `FpcaResult::project` pattern in `regression.rs` is the correct model: store the fitting artifacts (mean, rotation) immutably, then apply them in a `project` call. Apply the same pattern here.

**How to avoid:**
- Store shapelets as already-z-normalized `Vec<Vec<f64>>` in `ShapeletFitResult` (analogous to `FpcaResult`'s `rotation`/`mean`).
- `transform(&self, data: &FdMatrix) -> Result<FdMatrix, FdarError>` uses `self.shapelets` directly — no re-discovery, no re-normalization.
- Add a train/test consistency test: transform the training data using `fit_result.transform(train_data)` and compare column-by-column to the distances stored during fitting (these must match within floating-point tolerance).

**Warning signs:**
- `transform(train_data)` produces different values than those implicitly computed during `fit`.
- `ShapeletFitResult` does not contain the shapelet sequences — they are re-computed on each call.

**Phase to address:** Phase 59 (shapelet transform), which is where `transform` is implemented against the `ShapeletFitResult` struct.

**Verification hook (train/test consistency):**
```rust
let fit = shapelet_discover(&train_data, &labels, config)?;
let train_features = fit.transform(&train_data)?;
let test_features  = fit.transform(&test_data)?;
// re-transforming train must exactly reproduce the training distances
let train_features2 = fit.transform(&train_data)?;
assert!(train_features.approx_eq(&train_features2, 1e-12));
// columns of train_features must match stored distances in fit
for (k, s) in fit.shapelets.iter().enumerate() {
    for i in 0..n_train {
        assert!((train_features[(i,k)] - shapelet_distance(s, train_data.row(i))).abs() < 1e-12);
    }
}
```

---

### Pitfall 10: Test Series Shorter Than a Shapelet — No Defined Policy

**What goes wrong:**
If a test series has length `m_test < L` (shapelet length), the sliding-window loop has zero valid windows. The function must not panic (no index-out-of-bounds in the `&x[t..t+L]` slice) and must return a defined value rather than `NaN` or `INFINITY` silently.

Common options: (a) return `FdarError::InvalidDimension`, (b) return `f64::NAN` (bad — see Pitfall 2), (c) return `f64::INFINITY` (signals "no match" — can be used if documented). The `InvalidDimension` path is most consistent with the fdars `Result<T, FdarError>` convention.

**Why it happens:**
During fitting, the minimum shapelet length is bounded by `min(m_train_i)`, but at prediction time the user may pass shorter series. The sliding-window loop's range `0..m-L+1` wraps to an empty iterator in Rust rather than panicking, but the fold over an empty iterator returns `f64::INFINITY` — which silently becomes an outlier column in the feature matrix.

**How to avoid:**
- In `transform`, check at the start that every row of `data` has `ncols >= min(shapelet_lengths)`.
- Return `FdarError::InvalidDimension { parameter: "data.ncols", expected: min_len, actual: ncols }` if the check fails.
- Document the policy in the rustdoc for `transform`.

**Warning signs:**
- `transform` on shorter series returns a row of all-INFINITY distances without error.
- Test matrix has INFINITY values that silently produce NaN in a downstream dot product.

**Phase to address:** Phase 59 (shapelet transform).

**Verification hook:**
- Test: pass a series of length `min_shapelet_len - 1` to `transform`; expect `Err(FdarError::InvalidDimension)`.

---

### Pitfall 11: Classifier Determinism — Seed Not Threaded Through

**What goes wrong:**
If candidate sampling uses random selection (time-contract mode, which is the default for tractability — see Pitfall 4), the set of selected shapelets changes across runs with different seeds. If the seed is not exposed as a parameter and stored in the fit result, the pipeline is non-reproducible: two calls to `fit` on the same data return different shapelets, different transforms, and different classifier weights.

The fdars convention is per-thread deterministic seeding: `StdRng::seed_from_u64(seed + k as u64)`. If the candidate sampling loop uses a thread-local RNG without a user-supplied seed, reproducibility is broken.

**Why it happens:**
The existing per-thread seeding pattern (documented in `CLAUDE.md`) applies to rayon worker threads, not to a top-level caller seed. When the candidate sampling is sequential (not rayon-parallel), there is no thread-local RNG — the developer must explicitly thread the seed.

**How to avoid:**
- `ShapeletConfig` must include `seed: u64` (default: `42` or a well-known constant).
- The candidate sampling loop uses `rand::SeedableRng::seed_from_u64(config.seed)` to produce a deterministic shuffle/sample.
- Store `seed` in `ShapeletFitResult` so users can reproduce the exact fit.
- If candidate generation is parallelized, use `StdRng::seed_from_u64(config.seed + thread_idx as u64)` — the existing pattern in `parallel.rs`.

**Warning signs:**
- Two calls to `fit` with the same input and config return different shapelets.
- `ShapeletFitResult` does not record which seed was used.

**Phase to address:** Phase 58 (discovery & ranking) — where the random candidate-sampling loop lives.

**Verification hook (determinism test):**
```rust
let cfg = ShapeletConfig { seed: 42, max_candidates: 1000, ..Default::default() };
let fit1 = shapelet_discover(&data, &labels, cfg.clone())?;
let fit2 = shapelet_discover(&data, &labels, cfg.clone())?;
assert_eq!(fit1.shapelets, fit2.shapelets);
```

---

### Pitfall 12: Class Imbalance Biases Information Gain Toward the Majority Class

**What goes wrong:**
Information gain is not class-size-invariant. On an imbalanced training set (e.g. 90/10 split), the prior entropy is low (~0.47 bits) and a shapelet that perfectly separates the 10% minority class achieves less IG than one that splits the majority class — even if the minority-class shapelet is the genuinely discriminative feature. The result is that the top-K selected shapelets capture majority-class variation, not discriminative minority-class shape.

**Why it happens:**
The IG formula treats all training examples equally. Imbalanced datasets with common shapes shared by the majority class will produce many high-IG candidates that exploit majority-class homogeneity, not inter-class separability.

**How to avoid:**
- Document that the default IG ranking assumes balanced classes.
- Optionally support F-statistic (ANOVA-style) as an alternative quality measure: `quality: QualityMeasure { InfoGain | FStatistic }` in `ShapeletConfig`.
- The F-statistic is less sensitive to imbalance and is used in some sktime variants.
- Alternatively, allow class weights in the IG computation (weight each class by `1 / class_size`).
- Minimum viable: document the limitation clearly; do not silently use IG on imbalanced data without warning.

**Warning signs:**
- On an imbalanced dataset, all top-K shapelets come from the majority class.
- Minority-class `sdist` distributions overlap entirely with majority class.

**Phase to address:** Phase 58 (discovery & ranking) — expose `QualityMeasure` enum alongside IG.

**Verification hook:**
- Test: 90/10 imbalanced synthetic dataset with a discriminative motif in the 10% class. F-statistic must rank the true motif higher than IG on this dataset.

---

### Pitfall 13: Float Ties in Min-Distance and Ranking Produce Non-Deterministic Order

**What goes wrong:**
Two candidate shapelets can have exactly equal quality scores (IG or F-stat) in f64 due to identical training distributions. When sorted, their relative order is implementation-defined — this produces different top-K selections across platforms or Rust versions, breaking the determinism guarantee even with a fixed seed.

Similarly, two windows in the sliding-window loop can produce equal distances; taking the first-minimum vs. last-minimum changes the stored threshold for that candidate.

**Why it happens:**
Rust's `sort_by` on `f64` is not stable for NaN (it violates total order), and a naive `.sort_by(|a, b| a.partial_cmp(b).unwrap())` panics on NaN. Even with NaN-free data, equal floats sort non-deterministically under parallel sort.

**How to avoid:**
- Sort quality scores using a stable sort with a tie-break by `(series_idx, start_offset)` — both are deterministic.
- For the sliding-window minimum, always take the first minimum (lowest offset) on ties: `if d < best { best = d; best_offset = t; }` — strict less-than, not `<=`.
- Never use `partial_cmp(...).unwrap()` — use `total_cmp` (stable since Rust 1.62): `a.total_cmp(b)`.

**Warning signs:**
- Test with a duplicate-quality candidate set produces different output on different runs.
- Compilation warning about non-total `Ord` for `f64`.

**Phase to address:** Phase 58 (discovery & ranking).

**Verification hook:**
- Determinism test (covers both Pitfall 11 and 13): run `fit` twice with same seed, assert `fit1.shapelets == fit2.shapelets` and `fit1.thresholds == fit2.thresholds`.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip time contract; enumerate all candidates | Simpler loop, no config parameter | Unusable on any real dataset; discovery never returns | Never — always add the contract |
| Single-pass std formula `E[X²]-E[X]²` | One pass over window | Numerical instability for nearly-constant windows → NaN cascade | Never in a numerical library |
| Store shapelets un-normalized | Smaller struct | Re-normalization at transform time introduces bugs (Pitfall 9) | Never |
| Fixed threshold (e.g. median) for IG | No threshold search | Wrong ranking, poor shapelet selection | Never |
| Series-level normalization only | Simpler preprocessing | Breaks shapelet distance semantics (Pitfall 1) | Never |
| No self-similarity pruning | Simpler selection loop | Redundant features, poor classifier accuracy | Only in a prototype/smoke test, never shipped |
| Hard-code seed=0 | Reproducible output | User cannot vary the search; masks non-determinism bugs | Never — expose seed in config |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| O(n·m³) naive complete search | `fit` hangs after >30s on n=50, m=200 | Time contract + candidate sampling from the start | Any real dataset, m>100 |
| No early-abandon in window scan | Inner loop never exits early; profiler shows full L iterations always | Sequential inner loop with running-best short-circuit | Constant effect; 2–8× slowdown vs. early-abandon |
| Rayon parallelism in the window-scan inner loop | rayon overhead exceeds compute for small L windows | Parallelize at the series or candidate level, not the window level | L < ~200 (rayon task overhead dominates) |
| Materializing all candidate windows as owned `Vec<Vec<f64>>` | Heap allocation per window × millions of windows = GC pressure | Operate on slices; z-normalize into a fixed-length stack buffer | n=100, m=500, any L → ~50M allocations |
| `to_dmatrix()` conversion for shapelet distance | nalgebra matrix copy per distance call | Keep distance in raw `&[f64]` slice arithmetic; never call `to_dmatrix` in the hot loop | Every candidate × every series × every window |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `classification/` existing classifiers | Pass raw `FdMatrix` (curves) to an existing FPC-based classifier | Pass `transform(data)` result (n×K distances) to a classifier that accepts Euclidean features; kNN on Euclidean distances is the standard bundled choice |
| `metric/lp.rs` z-normalized distance | Reuse `lp_cross_1d` with p=2 for shapelet distance | `lp_cross_1d` integrates over `argvals` (L2 functional norm); shapelet distance is discrete Euclidean on z-normalized windows — different function, must not reuse |
| `FdMatrix` row access | Iterate columns (contiguous) for performance | Shapelet windows are row slices (non-contiguous in column-major layout); use `row_to_buf` to copy a row slice before z-normalizing to avoid scattered reads in the inner loop |
| `parallel.rs` macros | Apply `iter_maybe_parallel!` to the window-scan loop | Window scan must be sequential for early-abandon; apply `iter_maybe_parallel!` only to the outer series loop or outer candidate loop |
| `distance.rs` `pairwise_distance_matrix` | Reuse for shapelet candidate pairwise distances (self-similarity pruning) | Valid for small K (comparing selected shapelets to each other); not valid for the main discovery loop (different semantics) |

---

## "Looks Done But Isn't" Checklist

- [ ] **Shapelet distance:** Per-window z-normalization verified with offset-invariance and scale-invariance tests — not series-level normalization.
- [ ] **Constant-window guard:** `std.max(1e-10)` clamping in z-norm helper; constant-window test passes.
- [ ] **Min semantics:** Sliding-window fold uses strict minimum; known-motif recovery test passes (Pitfall 3 hook).
- [ ] **Time contract:** `max_candidates` or `time_contract_minutes` parameter present; tractability test passes (Pitfall 4 hook).
- [ ] **Early-abandon:** Inner distance loop short-circuits; benchmark confirms speedup vs. no-abandon.
- [ ] **Optimal threshold search:** `best_ig` scans all n-1 gaps; no fixed-threshold fallback.
- [ ] **Self-similarity pruning:** Selected shapelets span distinct source series; column-correlation check passes.
- [ ] **Transform consistency:** `transform(train_data)` matches distances computed during `fit`; train/test consistency test passes.
- [ ] **Short-series guard:** `transform` returns `Err(InvalidDimension)` if any series shorter than min shapelet length.
- [ ] **Seed threading:** `ShapeletConfig::seed` present; determinism test passes (two calls, same output).
- [ ] **NaN/Inf guard:** `transform` result passes `all(|v| v.is_finite())` assertion on all test inputs.
- [ ] **Stable sort with tie-break:** Candidate ranking uses `total_cmp` + `(series_idx, offset)` tie-break; no `partial_cmp(...).unwrap()`.
- [ ] **No `to_dmatrix` in hot loop:** Shapelet distance is pure `&[f64]` arithmetic; no nalgebra conversion.
- [ ] **Leakage discipline:** Rustdoc example uses train/test split; training-set accuracy not presented as generalization.
- [ ] **Shapelets stored normalized:** `ShapeletFitResult.shapelets` contains z-normalized sequences; transform never re-normalizes against test data.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Wrong z-normalization (series-level) | HIGH — all downstream results wrong | Rewrite `znorm_window` helper; re-run all tests; the fix is local but invalidates any previously cached results |
| Missing constant-window guard | LOW — one-line fix | Add `std.max(1e-10)` to `znorm_window`; all existing tests still pass |
| Mean instead of min | HIGH — wrong semantics | Rewrite the fold; re-validate known-motif recovery test |
| No time contract | MEDIUM — API addition | Add `max_candidates` to `ShapeletConfig` with a default; no breaking change if `ShapeletConfig` uses `..Default::default()` construction |
| No early-abandon | LOW — optimization only, not correctness | Add running-best parameter to inner loop; no API change |
| Leakage in IG | MEDIUM — re-rank candidates | Switch from full-set IG to per-fold IG or add held-out validation; results change but API does not |
| No self-similarity pruning | MEDIUM — selection logic rewrite | Add source-series tracking to candidate struct; rewrite selection loop |
| Transform inconsistency | HIGH — correctness bug | Ensure shapelets are stored normalized in `ShapeletFitResult`; fix `transform` to use stored sequences; re-run all integration tests |
| Missing short-series guard | LOW — one check + return Err | Add dimension check at start of `transform` |
| Non-determinism (no seed) | MEDIUM — API addition | Add `seed: u64` to `ShapeletConfig`; update all call sites |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| P1: Per-window z-norm | Phase 57 — shapelet-distance core | Offset-invariance + scale-invariance unit tests in `shapelet/mod.rs` |
| P2: Division by ~0 std | Phase 57 — shapelet-distance core | Constant-window and near-constant-window unit tests |
| P3: Min semantics | Phase 57 — shapelet-distance core | Known-motif recovery test (synthetic class_0/class_1 dataset) |
| P4: Combinatorial blowup | Phase 58 — discovery & ranking | Tractability test: n=100, m=200, contract=0.1min returns in time |
| P5: No early-abandon | Phase 57 — shapelet-distance core | Benchmark: early-abandon vs. no-abandon on non-matching shapelet |
| P6: Fixed threshold IG | Phase 58 — discovery & ranking | Synthetic IG validation: known-discriminative shapelet achieves high IG |
| P7: Training-set leakage | Phase 60 — bundled classifier | Rustdoc example uses train/test split; test evaluates on held-out data only |
| P8: Self-similarity omission | Phase 58 — discovery & ranking | Column-correlation check + series-diversity check on selected shapelets |
| P9: Transform inconsistency | Phase 59 — shapelet transform | Train/test consistency test: `transform(train) == stored distances` |
| P10: Short-series policy | Phase 59 — shapelet transform | Unit test: series shorter than min shapelet → `Err(InvalidDimension)` |
| P11: Non-determinism | Phase 58 — discovery & ranking | Determinism test: two same-seed fits produce identical shapelets |
| P12: Class imbalance | Phase 58 — discovery & ranking | F-statistic path; imbalanced dataset test shows true motif ranked higher |
| P13: Float ties | Phase 58 — discovery & ranking | Tie-break sort test: duplicate-score candidates produce deterministic order |

---

## Sources

- Hills, J., Lines, J., Baranauskas, E., Mapp, J., Bagnall, A. (2014). Classification of time series by shapelet transformation. *Data Mining and Knowledge Discovery*, 28(4), 851–881. — Reference algorithm for discovery, ranking, self-similarity pruning, time contract.
- Ye, L., Keogh, E. (2009). Time series shapelets: a new primitive for data mining. *KDD '09*. — Original shapelet paper: defines min-over-windows, z-normalization requirement.
- sktime `ShapeletTransformClassifier` source — time contract convention, `max_shapelets_to_store`, candidate sampling under a contract.
- pyts `ShapeletTransform` — threshold search, IG ranking, self-similarity via `remove_similar_shapelets`.
- Bagnall, A., Lines, J., Bostrom, A., Large, J., Keogh, E. (2017). The great time series classification bake off. *Data Mining and Knowledge Discovery*, 31(3), 606–660. — Accuracy vs. speed trade-offs for shapelet methods; documents early-abandon importance.
- fdars-core `src/metric/lp.rs`, `src/distance.rs`, `src/matrix.rs` — codebase integration constraints (row vs. column access, no `to_dmatrix` in hot loop, `iter_maybe_parallel!` placement).
- fdars-core `src/parallel.rs` — per-thread seed convention `StdRng::seed_from_u64(seed + k as u64)`.
- fdars `CLAUDE.md` — project conventions: `Result<T, FdarError>`, no panics on input validation, `#[must_use]` on expensive computations.

---
*Pitfalls research for: discovery-based shapelet transform + classifier (fdars-core v0.33.0, GAP-02)*
*Researched: 2026-09-02*
