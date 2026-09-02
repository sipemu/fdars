//! Shapelet discovery & ranking: candidate generation, discriminative quality
//! scoring (information gain / F-statistic), and top-K selection with
//! self-similarity pruning.
//!
//! Builds on the Phase 57 distance core ([`shapelet_distance`], [`Shapelet`]):
//! given a labeled training curve set, enumerate candidate subsequences
//! (exhaustively or via deterministic seeded random sampling bounded by
//! `max_candidates`), score each by how well its distance orderline separates
//! the class labels, and greedily select a non-redundant [`ShapeletSet`].
//!
//! # Determinism
//!
//! The candidate SET is fixed by `config.seed` *before* scoring, scoring is
//! pure, and the final ranking uses [`f64::total_cmp`] on quality with a
//! `(series_idx, start, length)` tie-break. Two fits with the same config are
//! therefore byte-identical, and the sequential (`parallel` off) result matches
//! the parallel one exactly.

use crate::error::FdarError;
use crate::helpers::seed_for_thread;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::shapelet::distance::{shapelet_distance, Shapelet};
use rand::Rng;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Discriminative quality measure used to score a candidate shapelet's distance
/// orderline against the class labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum QualityMeasure {
    /// Information gain on the optimal distance-split threshold (Ye & Keogh /
    /// Hills–Lines default). Assumes roughly balanced classes.
    #[default]
    InfoGain,
    /// One-way ANOVA F-statistic of the distance vector grouped by label. Less
    /// sensitive to class imbalance than information gain.
    FStatistic,
}

/// Configuration for [`discover_shapelets`].
///
/// `max_length` and `max_shapelets` accept the sentinel value `0`, which is
/// resolved at fit time: `max_length = 0` clamps to the series length
/// (`ncols`), and `max_shapelets = 0` resolves to `min(10 * n_train, 1000)`
/// (the sktime-style default).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShapeletDiscoveryConfig {
    /// Minimum candidate subsequence length (`>= 1`). Default `3`.
    pub min_length: usize,
    /// Maximum candidate subsequence length. `0` = clamp to series length.
    pub max_length: usize,
    /// Cap on the number of candidates evaluated. `Some(m)` random-samples `m`
    /// (seeded, reproducible) when the exhaustive count exceeds `m`; `None` =
    /// exhaustive. Default `Some(10_000)`.
    pub max_candidates: Option<usize>,
    /// Number of shapelets to keep after selection (`>= 1`). `0` resolves to
    /// `min(10 * n_train, 1000)` at fit time.
    pub max_shapelets: usize,
    /// Discriminative quality measure. Default [`QualityMeasure::InfoGain`].
    pub quality: QualityMeasure,
    /// Seed for deterministic candidate sampling. Default `0`.
    pub seed: u64,
}

impl Default for ShapeletDiscoveryConfig {
    fn default() -> Self {
        Self {
            min_length: 3,
            max_length: 0,
            max_candidates: Some(10_000),
            max_shapelets: 0,
            quality: QualityMeasure::InfoGain,
            seed: 0,
        }
    }
}

/// A discovered, ranked, non-redundant set of shapelets.
///
/// Each contained [`Shapelet`] has its `quality` field populated with the score
/// under [`ShapeletSet::quality`]. The set is ordered by quality descending.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ShapeletSet {
    /// Selected shapelets, ordered by quality descending.
    pub shapelets: Vec<Shapelet>,
    /// The quality measure the shapelets were scored under.
    pub quality: QualityMeasure,
}

impl ShapeletSet {
    /// The selected shapelets (ordered by quality descending).
    #[must_use]
    pub fn shapelets(&self) -> &[Shapelet] {
        &self.shapelets
    }

    /// Number of selected shapelets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shapelets.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shapelets.is_empty()
    }

    /// The quality measure the shapelets were scored under.
    #[must_use]
    pub fn quality(&self) -> QualityMeasure {
        self.quality
    }
}

/// Shannon entropy (base 2) of a label multiset given per-class counts.
fn entropy_from_counts(counts: &[usize], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    let mut h = 0.0;
    for &c in counts {
        if c > 0 {
            let p = c as f64 / n;
            h -= p * p.log2();
        }
    }
    h
}

/// Information gain of the *optimal* distance-split threshold.
///
/// `orderline` holds `(distance, label)` pairs (labels pre-remapped to
/// `0..n_classes`). Sorts by distance (`total_cmp`), then scans candidate
/// thresholds at midpoints between consecutive *distinct* distances, returning
/// `max_θ IG(θ)` with `IG(θ) = H(all) − (|L|/n·H(L) + |R|/n·H(R))`.
fn information_gain(orderline: &mut [(f64, usize)], n_classes: usize) -> f64 {
    let n = orderline.len();
    if n < 2 || n_classes < 2 {
        return 0.0;
    }
    orderline.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Total per-class counts (parent entropy).
    let mut total_counts = vec![0usize; n_classes];
    for &(_, y) in orderline.iter() {
        total_counts[y] += 1;
    }
    let parent_h = entropy_from_counts(&total_counts, n);

    // Incrementally move points from the right side to the left as the split
    // sweeps upward. left_counts starts empty; after including index t, the left
    // side is orderline[0..=t] and the right side is orderline[t+1..].
    let mut left_counts = vec![0usize; n_classes];
    let mut right_counts = total_counts.clone();
    let mut best_ig = 0.0f64;

    for t in 0..(n - 1) {
        let (d, y) = orderline[t];
        left_counts[y] += 1;
        right_counts[y] -= 1;

        // Only a genuine split boundary: the next distance must be strictly
        // larger (a midpoint between distinct distances exists).
        let d_next = orderline[t + 1].0;
        if d_next <= d {
            continue;
        }
        let n_left = t + 1;
        let n_right = n - n_left;
        let h_left = entropy_from_counts(&left_counts, n_left);
        let h_right = entropy_from_counts(&right_counts, n_right);
        let weighted = (n_left as f64 / n as f64) * h_left + (n_right as f64 / n as f64) * h_right;
        let ig = parent_h - weighted;
        if ig > best_ig {
            best_ig = ig;
        }
    }
    best_ig
}

/// One-way ANOVA F-statistic of a 1-D distance vector grouped by class label.
///
/// This is the **scalar / 1-D analogue** of
/// [`crate::function_on_scalar::integrated_f_statistic`], which computes the
/// pointwise F over an `FdMatrix` and integrates it across the grid. Here the
/// "grid" is a single point (the sdist value), so no integration is needed:
///
/// ```text
/// F = MS_between / MS_within
///   = [SS_between / (k − 1)] / [SS_within / (n − k)]
/// ```
///
/// where `k` is the number of classes. Returns `0.0` when the within-group mean
/// square is numerically zero (matching the `integrated_f_statistic` guard).
///
/// `labels` are pre-remapped to `0..n_classes`.
fn f_statistic_1d(distances: &[f64], labels: &[usize], n_classes: usize) -> f64 {
    let n = distances.len();
    if n == 0 || n_classes < 2 || n <= n_classes {
        return 0.0;
    }
    let mut group_sum = vec![0.0f64; n_classes];
    let mut group_cnt = vec![0usize; n_classes];
    let mut grand_sum = 0.0f64;
    for (&d, &y) in distances.iter().zip(labels.iter()) {
        group_sum[y] += d;
        group_cnt[y] += 1;
        grand_sum += d;
    }
    let grand_mean = grand_sum / n as f64;
    let mut group_mean = vec![0.0f64; n_classes];
    for g in 0..n_classes {
        if group_cnt[g] > 0 {
            group_mean[g] = group_sum[g] / group_cnt[g] as f64;
        }
    }
    let mut ss_between = 0.0f64;
    for g in 0..n_classes {
        let diff = group_mean[g] - grand_mean;
        ss_between += group_cnt[g] as f64 * diff * diff;
    }
    let mut ss_within = 0.0f64;
    for (&d, &y) in distances.iter().zip(labels.iter()) {
        let diff = d - group_mean[y];
        ss_within += diff * diff;
    }
    let ms_between = ss_between / (n_classes as f64 - 1.0).max(1.0);
    let ms_within = ss_within / (n as f64 - n_classes as f64).max(1.0);
    if ms_within > 1e-15 {
        ms_between / ms_within
    } else {
        0.0
    }
}

/// A candidate subsequence location in the training set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Candidate {
    series_idx: usize,
    start: usize,
    length: usize,
}

/// Enumerate every `(series_idx, start, length)` triple, or a deterministic
/// seeded random sample of `max_candidates` of them.
///
/// The returned list is sorted by `(series_idx, start, length)` so the candidate
/// order (and hence the whole fit) is fixed by the seed before scoring.
fn generate_candidates(
    n_series: usize,
    ncols: usize,
    min_length: usize,
    max_length: usize,
    max_candidates: Option<usize>,
    seed: u64,
) -> Vec<Candidate> {
    // Exhaustive count = n_series * Σ_{L=min..=max} (ncols - L + 1).
    let per_series: usize = (min_length..=max_length).map(|l| ncols - l + 1).sum();
    let total = n_series.saturating_mul(per_series);

    let exhaustive = match max_candidates {
        Some(m) => total <= m,
        None => true,
    };

    if exhaustive {
        let mut out = Vec::with_capacity(total);
        for series_idx in 0..n_series {
            for length in min_length..=max_length {
                for start in 0..=(ncols - length) {
                    out.push(Candidate {
                        series_idx,
                        start,
                        length,
                    });
                }
            }
        }
        return out;
    }

    // Contracted: sample `m` distinct triples deterministically. We sample a
    // linear index into the flattened enumeration and reject duplicates, which
    // is reproducible from the seed and independent of enumeration order.
    let m = max_candidates.unwrap(); // exhaustive == false ⇒ Some(_)
    let mut rng = seed_for_thread(seed, 0);
    use std::collections::HashSet;
    let mut chosen: HashSet<usize> = HashSet::with_capacity(m);
    // Guard against pathological rejection loops (should not trigger since
    // m < total, but keeps termination provable).
    let max_draws = m.saturating_mul(64).max(total);
    let mut draws = 0usize;
    while chosen.len() < m && draws < max_draws {
        let idx = rng.gen_range(0..total);
        chosen.insert(idx);
        draws += 1;
    }

    let mut out: Vec<Candidate> = chosen
        .into_iter()
        .map(|lin| decode_candidate(lin, n_series, ncols, min_length, max_length))
        .collect();
    // Deterministic candidate order regardless of HashSet iteration order.
    out.sort_by_key(|c| (c.series_idx, c.start, c.length));
    out
}

/// Decode a flat enumeration index back into a `(series_idx, start, length)`
/// triple, matching the exhaustive nesting order (series → length → start).
fn decode_candidate(
    lin: usize,
    _n_series: usize,
    ncols: usize,
    min_length: usize,
    max_length: usize,
) -> Candidate {
    let per_series: usize = (min_length..=max_length).map(|l| ncols - l + 1).sum();
    let series_idx = lin / per_series;
    let mut rem = lin % per_series;
    let mut length = min_length;
    loop {
        let starts = ncols - length + 1;
        if rem < starts {
            return Candidate {
                series_idx,
                start: rem,
                length,
            };
        }
        rem -= starts;
        length += 1;
    }
}

/// Discover a non-redundant [`ShapeletSet`] from a labeled training curve set.
///
/// Enumerates candidate subsequences over `[config.min_length, config.max_length]`
/// (exhaustively, or a deterministic seeded random sample of
/// `config.max_candidates`), scores each candidate by how well its distance
/// orderline separates the class labels (information gain or F-statistic per
/// `config.quality`), then greedily selects the top `config.max_shapelets` with
/// self-similarity pruning: once a shapelet from series `i` spanning
/// `[start, start+length)` is selected, any not-yet-selected candidate from the
/// same series whose range overlaps it is discarded.
///
/// `data` is a column-major [`FdMatrix`] with rows = curves and columns =
/// evaluation points; `labels[i]` is the integer class of curve `i`.
///
/// # Determinism
///
/// The result is byte-identical across runs with the same `config` and identical
/// whether or not the `parallel` feature is enabled.
///
/// # Errors
///
/// - [`FdarError::InvalidDimension`] if `labels.len()` != number of curves.
/// - [`FdarError::InvalidParameter`] if fewer than 2 distinct classes are
///   present, if `min_length < 1`, `min_length > max_length`,
///   `max_length > ncols`, or the resolved `max_shapelets < 1`.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::shapelet::{discover_shapelets, ShapeletDiscoveryConfig};
///
/// // Two classes of length-8 curves; class 1 carries a rising ramp in the
/// // middle that class 0 lacks.
/// let n = 8usize;
/// let m = 8usize;
/// let mut data = vec![0.0f64; n * m];
/// let mut labels = vec![0usize; n];
/// for i in 0..n {
///     let class1 = i % 2 == 1;
///     labels[i] = usize::from(class1);
///     for j in 0..m {
///         // column-major: element (i, j) at i + j*n
///         let base = 0.1 * (i as f64) + 0.05 * (j as f64);
///         let motif = if class1 && (3..6).contains(&j) { (j as f64) * 2.0 } else { 0.0 };
///         data[i + j * n] = base + motif;
///     }
/// }
/// let data = FdMatrix::from_column_major(data, n, m).unwrap();
///
/// let cfg = ShapeletDiscoveryConfig { max_shapelets: 3, ..Default::default() };
/// let set = discover_shapelets(&data, &labels, &cfg).unwrap();
/// assert!(!set.is_empty());
/// assert!(set.len() <= 3);
/// ```
#[must_use = "the discovered shapelet set should not be discarded"]
pub fn discover_shapelets(
    data: &FdMatrix,
    labels: &[usize],
    config: &ShapeletDiscoveryConfig,
) -> Result<ShapeletSet, FdarError> {
    let (n_series, ncols) = data.shape();

    // --- validation ---
    if labels.len() != n_series {
        return Err(FdarError::InvalidDimension {
            parameter: "labels",
            expected: format!("{n_series} labels (one per curve)"),
            actual: format!("{} labels", labels.len()),
        });
    }
    if n_series == 0 || ncols == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least one curve with at least one point".to_string(),
            actual: format!("{n_series}x{ncols}"),
        });
    }

    // Distinct classes + a dense 0..n_classes remap.
    let mut distinct: Vec<usize> = labels.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let n_classes = distinct.len();
    if n_classes < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "labels",
            message: format!("at least 2 distinct classes required, found {n_classes}"),
        });
    }
    let remap = |y: usize| distinct.iter().position(|&d| d == y).unwrap();
    let labels_dense: Vec<usize> = labels.iter().map(|&y| remap(y)).collect();

    if config.min_length < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "min_length",
            message: "min_length must be >= 1".to_string(),
        });
    }
    // Resolve max_length sentinel.
    let max_length = if config.max_length == 0 {
        ncols
    } else {
        config.max_length
    };
    if config.min_length > max_length {
        return Err(FdarError::InvalidParameter {
            parameter: "min_length",
            message: format!(
                "min_length ({}) > max_length ({max_length})",
                config.min_length
            ),
        });
    }
    if max_length > ncols {
        return Err(FdarError::InvalidParameter {
            parameter: "max_length",
            message: format!("max_length ({max_length}) > series length ({ncols})"),
        });
    }
    // Resolve max_shapelets sentinel.
    let max_shapelets = if config.max_shapelets == 0 {
        (10 * n_series).min(1000)
    } else {
        config.max_shapelets
    };
    if max_shapelets < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "max_shapelets",
            message: "max_shapelets must be >= 1".to_string(),
        });
    }

    // --- candidate generation (seed-fixed, order-deterministic) ---
    let candidates = generate_candidates(
        n_series,
        ncols,
        config.min_length,
        max_length,
        config.max_candidates,
        config.seed,
    );

    // Pre-extract each curve row contiguously (rows are non-contiguous in the
    // column-major layout).
    let series_rows: Vec<Vec<f64>> = {
        let mut rows = Vec::with_capacity(n_series);
        let mut buf = vec![0.0f64; ncols];
        for i in 0..n_series {
            data.row_to_buf(i, &mut buf);
            rows.push(buf.clone());
        }
        rows
    };

    // --- score candidates (parallel over the fixed candidate set) ---
    let quality = config.quality;
    let scored: Vec<(f64, Candidate)> = iter_maybe_parallel!(0..candidates.len())
        .map(|ci| {
            let cand = candidates[ci];
            let src = &series_rows[cand.series_idx];
            // Shapelet is z-normalized at construction.
            let shp = Shapelet::from_source(src, cand.series_idx, cand.start, cand.length)
                .expect("candidate window is in-range by construction");
            // Distance orderline: one sdist per training series.
            let mut orderline: Vec<(f64, usize)> = Vec::with_capacity(n_series);
            for (i, row) in series_rows.iter().enumerate() {
                let (d, _off) = shapelet_distance(&shp.values, row, f64::INFINITY)
                    .expect("series length >= shapelet length by construction");
                orderline.push((d, labels_dense[i]));
            }
            let score = match quality {
                QualityMeasure::InfoGain => information_gain(&mut orderline, n_classes),
                QualityMeasure::FStatistic => {
                    let dists: Vec<f64> = orderline.iter().map(|&(d, _)| d).collect();
                    let labs: Vec<usize> = orderline.iter().map(|&(_, y)| y).collect();
                    f_statistic_1d(&dists, &labs, n_classes)
                }
            };
            (score, cand)
        })
        .collect();

    // --- rank: quality desc, tie-break (series_idx, start, length) ---
    let mut ranked = scored;
    ranked.sort_by(|a, b| {
        b.0.total_cmp(&a.0).then_with(|| {
            (a.1.series_idx, a.1.start, a.1.length).cmp(&(b.1.series_idx, b.1.start, b.1.length))
        })
    });

    // --- greedy selection + self-similarity pruning ---
    // Track, per series, the accepted [start, end) intervals so we can reject
    // overlapping same-series candidates.
    let mut accepted_ranges: std::collections::HashMap<usize, Vec<(usize, usize)>> =
        std::collections::HashMap::new();
    let mut selected: Vec<Shapelet> = Vec::with_capacity(max_shapelets);

    for (score, cand) in ranked {
        if selected.len() >= max_shapelets {
            break;
        }
        let start = cand.start;
        let end = cand.start + cand.length;
        let overlaps = accepted_ranges
            .get(&cand.series_idx)
            .is_some_and(|ranges| ranges.iter().any(|&(s, e)| !(end <= s || e <= start)));
        if overlaps {
            continue;
        }
        let src = &series_rows[cand.series_idx];
        let mut shp = Shapelet::from_source(src, cand.series_idx, cand.start, cand.length)
            .expect("candidate window is in-range by construction");
        shp.quality = score;
        accepted_ranges
            .entry(cand.series_idx)
            .or_default()
            .push((start, end));
        selected.push(shp);
    }

    Ok(ShapeletSet {
        shapelets: selected,
        quality: config.quality,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 2-class dataset: class 0 = smooth noise-free baseline, class 1 =
    /// baseline with a distinctive triangular motif planted at a fixed offset.
    /// Returns (data, labels, motif_start, motif_len).
    fn planted_motif_dataset() -> (FdMatrix, Vec<usize>, usize, usize) {
        let n = 20usize;
        let m = 40usize;
        let motif_start = 15usize;
        let motif_len = 8usize;
        let mut flat = vec![0.0f64; n * m];
        let mut labels = vec![0usize; n];
        for i in 0..n {
            let class1 = i % 2 == 1;
            labels[i] = usize::from(class1);
            let offset = 0.01 * (i as f64); // tiny per-curve baseline shift
            for j in 0..m {
                let mut v = offset + (j as f64) * 0.001;
                // Small deterministic shape jitter (survives per-window
                // z-normalization) → nonzero within-class variance in the
                // distance orderline, so the F-statistic is finite/well-defined
                // rather than tripped by a zero within-group mean square.
                let hash = (i.wrapping_mul(2654435761) ^ j.wrapping_mul(40503)) % 211;
                v += 0.05 * (hash as f64 / 211.0 - 0.5);
                if class1 && j >= motif_start && j < motif_start + motif_len {
                    // Triangular spike unique to class 1.
                    let k = j - motif_start;
                    let half = motif_len / 2;
                    let tri = if k <= half {
                        k as f64
                    } else {
                        (motif_len - k) as f64
                    };
                    v += tri;
                }
                flat[i + j * n] = v;
            }
        }
        (
            FdMatrix::from_column_major(flat, n, m).unwrap(),
            labels,
            motif_start,
            motif_len,
        )
    }

    #[test]
    fn test_discover_known_motif() {
        let (data, labels, motif_start, motif_len) = planted_motif_dataset();
        let cfg = ShapeletDiscoveryConfig {
            min_length: motif_len,
            max_length: motif_len,
            max_candidates: None, // exhaustive on this small dataset
            max_shapelets: 3,
            quality: QualityMeasure::InfoGain,
            seed: 0,
        };
        let set = discover_shapelets(&data, &labels, &cfg).unwrap();
        assert!(!set.is_empty(), "no shapelets discovered");
        // The top shapelet should be a class-1 curve and align with the motif.
        let top = &set.shapelets()[0];
        assert!(top.quality > 0.0, "top shapelet has non-positive quality");
        // Overlap with the planted motif region.
        let s = top.start;
        let e = top.start + top.length;
        assert!(
            !(e <= motif_start || motif_start + motif_len <= s),
            "top shapelet [{s},{e}) does not overlap planted motif [{motif_start},{})",
            motif_start + motif_len
        );
        // A perfectly separating shapelet reaches max entropy for a balanced
        // 2-class split = 1.0 bit.
        assert!(
            top.quality > 0.9,
            "top shapelet IG {} not near max entropy 1.0",
            top.quality
        );
    }

    #[test]
    fn test_discover_tractable_contracted() {
        // n=100 series of length 200, contracted to a modest candidate budget.
        let n = 100usize;
        let m = 200usize;
        let mut flat = vec![0.0f64; n * m];
        let mut labels = vec![0usize; n];
        for i in 0..n {
            let class1 = i % 2 == 1;
            labels[i] = usize::from(class1);
            for j in 0..m {
                let mut v = (j as f64) * 0.01 + (i as f64) * 0.001;
                if class1 && (80..90).contains(&j) {
                    v += 5.0;
                }
                flat[i + j * n] = v;
            }
        }
        let data = FdMatrix::from_column_major(flat, n, m).unwrap();
        let cfg = ShapeletDiscoveryConfig {
            min_length: 10,
            max_length: 20,
            max_candidates: Some(800),
            max_shapelets: 5,
            quality: QualityMeasure::InfoGain,
            seed: 7,
        };
        let start = std::time::Instant::now();
        let set = discover_shapelets(&data, &labels, &cfg).unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_secs() < 10,
            "contracted discovery too slow: {elapsed:?}"
        );
        assert!(set.len() <= 5, "returned more than max_shapelets");
        assert!(!set.is_empty());
    }

    #[test]
    fn test_infogain_optimal_split() {
        // Clean separation: class 0 has small distances, class 1 has large.
        let mut orderline = vec![
            (0.1, 0usize),
            (0.2, 0),
            (0.15, 0),
            (5.0, 1),
            (5.5, 1),
            (6.0, 1),
        ];
        let ig = information_gain(&mut orderline, 2);
        // Perfect balanced 2-class split → IG == parent entropy == 1.0 bit.
        assert!((ig - 1.0).abs() < 1e-12, "IG for clean split not 1.0: {ig}");

        // A non-separating orderline (interleaved labels, all equal distance)
        // yields zero gain.
        let mut flat = vec![(1.0, 0usize), (1.0, 1), (1.0, 0), (1.0, 1)];
        let ig0 = information_gain(&mut flat, 2);
        assert!(ig0.abs() < 1e-12, "IG for degenerate split not 0: {ig0}");
    }

    #[test]
    fn test_fstatistic_measure() {
        // Discriminative: distances well separated by class.
        let disc_d = [0.1, 0.12, 0.09, 5.0, 5.1, 4.9];
        let labs = [0usize, 0, 0, 1, 1, 1];
        let f_disc = f_statistic_1d(&disc_d, &labs, 2);
        // Noise: distances uncorrelated with class.
        let noise_d = [1.0, 5.0, 1.0, 5.0, 1.0, 5.0];
        let f_noise = f_statistic_1d(&noise_d, &labs, 2);
        assert!(
            f_disc > f_noise,
            "F-stat did not rank discriminative above noise: {f_disc} vs {f_noise}"
        );
        assert!(
            f_disc > 10.0,
            "discriminative F-stat unexpectedly low: {f_disc}"
        );

        // End-to-end: FStatistic quality path runs and returns a set.
        let (data, labels, _, motif_len) = planted_motif_dataset();
        let cfg = ShapeletDiscoveryConfig {
            min_length: motif_len,
            max_length: motif_len,
            max_candidates: None,
            max_shapelets: 3,
            quality: QualityMeasure::FStatistic,
            seed: 0,
        };
        let set = discover_shapelets(&data, &labels, &cfg).unwrap();
        assert!(!set.is_empty());
        assert_eq!(set.quality(), QualityMeasure::FStatistic);
        assert!(set.shapelets()[0].quality > 0.0);
    }

    #[test]
    fn test_self_similarity_pruning() {
        let (data, labels, _, _) = planted_motif_dataset();
        // Small lengths + many shapelets → without pruning, adjacent overlapping
        // windows from the best series would dominate.
        let cfg = ShapeletDiscoveryConfig {
            min_length: 6,
            max_length: 6,
            max_candidates: None,
            max_shapelets: 8,
            quality: QualityMeasure::InfoGain,
            seed: 0,
        };
        let set = discover_shapelets(&data, &labels, &cfg).unwrap();
        // No two selected shapelets from the SAME series may overlap.
        let shp = set.shapelets();
        for a in 0..shp.len() {
            for b in (a + 1)..shp.len() {
                if shp[a].series_idx == shp[b].series_idx {
                    let (sa, ea) = (shp[a].start, shp[a].start + shp[a].length);
                    let (sb, eb) = (shp[b].start, shp[b].start + shp[b].length);
                    assert!(
                        ea <= sb || eb <= sa,
                        "same-series shapelets overlap: [{sa},{ea}) & [{sb},{eb})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_discover_deterministic() {
        // Larger-than-budget candidate space forces random sampling; same seed
        // must reproduce byte-identical results.
        let n = 30usize;
        let m = 60usize;
        let mut flat = vec![0.0f64; n * m];
        let mut labels = vec![0usize; n];
        for i in 0..n {
            let class1 = i % 2 == 1;
            labels[i] = usize::from(class1);
            for j in 0..m {
                let mut v = (j as f64) * 0.02 + (i as f64) * 0.003;
                if class1 && (20..30).contains(&j) {
                    v += 3.0;
                }
                flat[i + j * n] = v;
            }
        }
        let data = FdMatrix::from_column_major(flat, n, m).unwrap();
        let cfg = ShapeletDiscoveryConfig {
            min_length: 8,
            max_length: 12,
            max_candidates: Some(500),
            max_shapelets: 6,
            quality: QualityMeasure::InfoGain,
            seed: 123,
        };
        let a = discover_shapelets(&data, &labels, &cfg).unwrap();
        let b = discover_shapelets(&data, &labels, &cfg).unwrap();
        assert_eq!(a, b, "same-seed fits not byte-identical");
    }

    #[test]
    fn test_discover_validation() {
        let (data, labels, _, _) = planted_motif_dataset();
        let (_n, ncols) = data.shape();

        // <2 classes.
        let one_class = vec![0usize; labels.len()];
        let cfg = ShapeletDiscoveryConfig::default();
        assert!(matches!(
            discover_shapelets(&data, &one_class, &cfg),
            Err(FdarError::InvalidParameter { .. })
        ));

        // label/row mismatch.
        let short_labels = vec![0usize, 1];
        assert!(matches!(
            discover_shapelets(&data, &short_labels, &cfg),
            Err(FdarError::InvalidDimension { .. })
        ));

        // min > max.
        let cfg_bad = ShapeletDiscoveryConfig {
            min_length: 10,
            max_length: 5,
            ..Default::default()
        };
        assert!(matches!(
            discover_shapelets(&data, &labels, &cfg_bad),
            Err(FdarError::InvalidParameter { .. })
        ));

        // max_length > ncols.
        let cfg_big = ShapeletDiscoveryConfig {
            min_length: 3,
            max_length: ncols + 5,
            ..Default::default()
        };
        assert!(matches!(
            discover_shapelets(&data, &labels, &cfg_big),
            Err(FdarError::InvalidParameter { .. })
        ));
    }
}
