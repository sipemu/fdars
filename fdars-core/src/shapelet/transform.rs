//! Shapelet transform: turn a fitted [`ShapeletSet`] into an `n×K` distance
//! feature matrix, for both training and out-of-sample curves.
//!
//! Given `K` discovered shapelets and a curve set of `n` series, the transform
//! produces an `n×K` [`FdMatrix`] whose entry `X[(i, j)]` is the shapelet
//! distance ([`shapelet_distance`]) from shapelet `j` to curve `i`:
//!
//! ```text
//! X[(i, j)] = sdist(shapelet_j, curve_i)
//! ```
//!
//! **The `K` output columns are shapelet distances, not functional evaluation
//! points.** Downstream (Phase 60) an fdars classifier consumes this matrix as
//! `data` — rows are observations, columns are the `K` shapelet-distance
//! features.
//!
//! # Consistency and normalization
//!
//! The shapelets carried by a [`ShapeletSet`] are stored **already
//! z-normalized** (Phase 57 provenance). The transform reuses those stored
//! values directly and never re-normalizes against the input series' statistics.
//! Every window of each input series is z-normalized independently at comparison
//! time inside [`shapelet_distance`]. Because training and out-of-sample curves
//! flow through the identical code path with the identical stored shapelets and
//! `best_so_far = f64::INFINITY`, re-transforming the training set exactly
//! reproduces the fit-time distances (see [`ShapeletTransformFit`]).

use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::shapelet::discovery::{discover_shapelets, ShapeletDiscoveryConfig, ShapeletSet};
use crate::shapelet::distance::shapelet_distance;

#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Apply a fitted [`ShapeletSet`] to a curve set, producing an `n×K` distance
/// feature matrix.
///
/// Returns an [`FdMatrix`] with `n = data` rows (curves) and `K =
/// shapelets.len()` columns, where
/// `X[(i, j)] = shapelet_distance(&shapelets.shapelets()[j].values, curve_i, f64::INFINITY).0`.
///
/// The shapelet `values` are already z-normalized, so they are reused directly
/// — no re-normalization against `data`. Each input row is z-normalized
/// per-window inside [`shapelet_distance`]. The row loop is parallelized with
/// [`iter_maybe_parallel!`]; distances are order-independent, so the result is
/// identical with or without the `parallel` feature.
///
/// **Output columns are shapelet distances, not evaluation points.**
///
/// # Errors
///
/// - [`FdarError::InvalidParameter`] if the shapelet set is empty (`K == 0`): a
///   zero-column feature matrix carries no information.
/// - [`FdarError::InvalidDimension`] (propagated from [`shapelet_distance`]) if
///   any series is shorter than a shapelet — no valid sliding window exists.
///
/// All returned entries are finite (guaranteed by the Phase 57 z-normalization
/// constant-window guard).
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::shapelet::{discover_shapelets, shapelet_transform, ShapeletDiscoveryConfig};
///
/// let n = 8usize;
/// let m = 8usize;
/// let mut flat = vec![0.0f64; n * m];
/// let mut labels = vec![0usize; n];
/// for i in 0..n {
///     let class1 = i % 2 == 1;
///     labels[i] = usize::from(class1);
///     for j in 0..m {
///         let base = 0.1 * (i as f64) + 0.05 * (j as f64);
///         let motif = if class1 && (3..6).contains(&j) { (j as f64) * 2.0 } else { 0.0 };
///         flat[i + j * n] = base + motif;
///     }
/// }
/// let data = FdMatrix::from_column_major(flat, n, m).unwrap();
///
/// let cfg = ShapeletDiscoveryConfig { max_shapelets: 3, ..Default::default() };
/// let set = discover_shapelets(&data, &labels, &cfg).unwrap();
///
/// let features = shapelet_transform(&set, &data).unwrap();
/// assert_eq!(features.shape(), (n, set.len()));
/// // Every entry is a finite shapelet distance.
/// for j in 0..set.len() {
///     for i in 0..n {
///         assert!(features.get(i, j).unwrap().is_finite());
///     }
/// }
/// ```
#[must_use = "the transformed feature matrix should not be discarded"]
pub fn shapelet_transform(shapelets: &ShapeletSet, data: &FdMatrix) -> Result<FdMatrix, FdarError> {
    let k = shapelets.len();
    if k == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "shapelets",
            message: "shapelet set is empty (K = 0); a 0-column feature matrix is not useful"
                .to_string(),
        });
    }

    let (n, ncols) = data.shape();
    let shp = shapelets.shapelets();

    // Compute one row of K distances per curve, in parallel over curves.
    // Each row is independent, so parallelism is deterministic. Any per-row
    // error (e.g. a series shorter than a shapelet) is captured as an `Err`.
    let rows: Vec<Result<Vec<f64>, FdarError>> = iter_maybe_parallel!(0..n)
        .map(|i| {
            // Copy the (non-contiguous, column-major) row contiguously once,
            // then scan it against every shapelet.
            let mut buf = vec![0.0f64; ncols];
            data.row_to_buf(i, &mut buf);
            let mut row = Vec::with_capacity(k);
            for s in shp {
                let (dist, _off) = shapelet_distance(&s.values, &buf, f64::INFINITY)?;
                row.push(dist);
            }
            Ok(row)
        })
        .collect();

    // Bubble the first error (deterministic: rows are order-independent).
    let mut per_row = Vec::with_capacity(n);
    for r in rows {
        per_row.push(r?);
    }

    // Assemble the n×K matrix in column-major order: element (i, j) at i + j*n.
    let mut flat = vec![0.0f64; n * k];
    for (i, row) in per_row.iter().enumerate() {
        for (j, &d) in row.iter().enumerate() {
            flat[i + j * n] = d;
        }
    }
    FdMatrix::from_column_major(flat, n, k)
}

/// A fitted shapelet transform: the discovered [`ShapeletSet`] plus the training
/// feature matrix produced by applying it to the training curves.
///
/// Stores the already-z-normalized shapelets so that out-of-sample
/// [`transform`](Self::transform) reuses the exact same shapelets and
/// normalization as the fit — never re-discovering or re-normalizing against
/// test-set statistics.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ShapeletTransformFit {
    /// The discovered, already-z-normalized shapelet set.
    pub shapelets: ShapeletSet,
    /// The training `n×K` distance feature matrix (columns = shapelet distances).
    pub features: FdMatrix,
}

impl ShapeletTransformFit {
    /// The fitted shapelet set (already z-normalized, ordered by quality).
    #[must_use]
    pub fn shapelets(&self) -> &ShapeletSet {
        &self.shapelets
    }

    /// The training `n×K` feature matrix (columns are shapelet distances).
    #[must_use]
    pub fn features(&self) -> &FdMatrix {
        &self.features
    }

    /// Transform out-of-sample curves using the stored shapelets.
    ///
    /// Applies the exact fitted shapelets (same sequences, same stored
    /// z-normalization) to `new_data`, yielding an `n_new×K` feature matrix.
    /// This is [`shapelet_transform`] against the stored set — no re-discovery,
    /// no re-normalization against `new_data`.
    ///
    /// # Errors
    ///
    /// Same as [`shapelet_transform`]: [`FdarError::InvalidDimension`] if a
    /// series is shorter than a shapelet, [`FdarError::InvalidParameter`] if the
    /// stored set is empty.
    #[must_use = "the transformed feature matrix should not be discarded"]
    pub fn transform(&self, new_data: &FdMatrix) -> Result<FdMatrix, FdarError> {
        shapelet_transform(self.shapelets(), new_data)
    }
}

/// Fit a shapelet transform: discover shapelets from a labeled training set,
/// then transform that training set into an `n×K` distance feature matrix.
///
/// Calls [`discover_shapelets`] (Phase 58) with `config`, then
/// [`shapelet_transform`] on the training `data`, storing both the shapelet set
/// and the training features in the returned [`ShapeletTransformFit`]. Reuse the
/// stored set on new curves via [`ShapeletTransformFit::transform`].
///
/// `data` is a column-major [`FdMatrix`] with rows = curves, columns =
/// evaluation points; `labels[i]` is the integer class of curve `i`.
///
/// # Errors
///
/// - Any error from [`discover_shapelets`] (e.g. [`FdarError::InvalidDimension`]
///   on label/row mismatch, [`FdarError::InvalidParameter`] on fewer than 2
///   classes or bad length bounds).
/// - Any error from [`shapelet_transform`] (empty discovered set, or a series
///   shorter than a discovered shapelet).
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::shapelet::{shapelet_transform_fit, ShapeletDiscoveryConfig};
///
/// // Two classes of length-8 curves; class 1 carries a motif class 0 lacks.
/// let n = 8usize;
/// let m = 8usize;
/// let mut flat = vec![0.0f64; n * m];
/// let mut labels = vec![0usize; n];
/// for i in 0..n {
///     let class1 = i % 2 == 1;
///     labels[i] = usize::from(class1);
///     for j in 0..m {
///         let base = 0.1 * (i as f64) + 0.05 * (j as f64);
///         let motif = if class1 && (3..6).contains(&j) { (j as f64) * 2.0 } else { 0.0 };
///         flat[i + j * n] = base + motif;
///     }
/// }
/// let data = FdMatrix::from_column_major(flat, n, m).unwrap();
///
/// let cfg = ShapeletDiscoveryConfig { max_shapelets: 3, ..Default::default() };
/// let fit = shapelet_transform_fit(&data, &labels, &cfg).unwrap();
///
/// // Training features are n×K; reuse the fitted shapelets on new curves.
/// let k = fit.shapelets().len();
/// assert_eq!(fit.features().shape(), (n, k));
/// let new_features = fit.transform(&data).unwrap();
/// assert_eq!(new_features.shape(), (n, k));
/// ```
#[must_use = "the fitted shapelet transform should not be discarded"]
pub fn shapelet_transform_fit(
    data: &FdMatrix,
    labels: &[usize],
    config: &ShapeletDiscoveryConfig,
) -> Result<ShapeletTransformFit, FdarError> {
    let shapelets = discover_shapelets(data, labels, config)?;
    let features = shapelet_transform(&shapelets, data)?;
    Ok(ShapeletTransformFit {
        shapelets,
        features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapelet::distance::shapelet_distance;

    /// Two-class dataset: class 1 carries a triangular motif class 0 lacks.
    /// `rows` optionally overrides the number of curves (labels alternate).
    fn labeled_dataset(n: usize, m: usize) -> (FdMatrix, Vec<usize>) {
        let mut flat = vec![0.0f64; n * m];
        let mut labels = vec![0usize; n];
        let motif_start = m / 2;
        let motif_len = (m / 4).max(1);
        for i in 0..n {
            let class1 = i % 2 == 1;
            labels[i] = usize::from(class1);
            let offset = 0.01 * (i as f64);
            for j in 0..m {
                let mut v = offset + (j as f64) * 0.001;
                let hash = (i.wrapping_mul(2654435761) ^ j.wrapping_mul(40503)) % 211;
                v += 0.05 * (hash as f64 / 211.0 - 0.5);
                if class1 && j >= motif_start && j < motif_start + motif_len {
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
        (FdMatrix::from_column_major(flat, n, m).unwrap(), labels)
    }

    fn default_cfg() -> ShapeletDiscoveryConfig {
        ShapeletDiscoveryConfig {
            min_length: 3,
            max_length: 6,
            max_candidates: None,
            max_shapelets: 4,
            seed: 0,
            ..Default::default()
        }
    }

    #[test]
    fn test_transform_fit_shape() {
        let (data, labels) = labeled_dataset(16, 24);
        let cfg = default_cfg();
        let fit = shapelet_transform_fit(&data, &labels, &cfg).unwrap();

        let k = fit.shapelets().len();
        assert!(k > 0, "no shapelets discovered");
        assert_eq!(fit.features().shape(), (16, k), "training features not n×K");
        // All entries finite.
        for idx in 0..fit.features().len() {
            let (i, j) = (idx % 16, idx / 16);
            let v = fit.features().get(i, j).unwrap();
            assert!(v.is_finite(), "non-finite feature at ({i},{j}): {v}");
        }
    }

    #[test]
    fn test_transform_out_of_sample_shape() {
        let (train, labels) = labeled_dataset(16, 24);
        let cfg = default_cfg();
        let fit = shapelet_transform_fit(&train, &labels, &cfg).unwrap();
        let k = fit.shapelets().len();

        // New set with a DIFFERENT row count (catches a transpose bug).
        let n_new = 9usize;
        let (new_data, _new_labels) = labeled_dataset(n_new, 24);
        let features = fit.transform(&new_data).unwrap();
        assert_eq!(features.nrows(), n_new, "wrong row count (transpose?)");
        assert_eq!(features.ncols(), k, "wrong column count (transpose?)");
        assert_ne!(n_new, 16, "test setup: n_new must differ from n_train");
        for idx in 0..features.len() {
            let (i, j) = (idx % n_new, idx / n_new);
            assert!(features.get(i, j).unwrap().is_finite());
        }
    }

    #[test]
    fn test_transform_consistency() {
        let (train, labels) = labeled_dataset(16, 24);
        let cfg = default_cfg();
        let fit = shapelet_transform_fit(&train, &labels, &cfg).unwrap();

        // Re-transforming the training data reproduces the stored features.
        let re = fit.transform(&train).unwrap();
        assert_eq!(re.shape(), fit.features().shape());
        for idx in 0..re.len() {
            let (i, j) = (idx % 16, idx / 16);
            let a = re.get(i, j).unwrap();
            let b = fit.features().get(i, j).unwrap();
            assert!(
                (a - b).abs() < 1e-12,
                "transform not consistent at ({i},{j}): {a} vs {b}"
            );
        }
        // Two transform calls are bit-identical.
        let re2 = fit.transform(&train).unwrap();
        assert_eq!(re, re2, "two transform(train) calls differ");
    }

    #[test]
    fn test_transform_values_are_sdist() {
        // Tiny hand-checked case: build a set with two known shapelets and
        // verify each X[i,j] equals the direct shapelet_distance.
        let n = 3usize;
        let m = 7usize;
        let mut flat = vec![0.0f64; n * m];
        let mut labels = vec![0usize; n];
        for i in 0..n {
            labels[i] = i % 2;
            for j in 0..m {
                flat[i + j * n] = (i as f64) + (j as f64) * (1.0 + i as f64);
            }
        }
        let data = FdMatrix::from_column_major(flat, n, m).unwrap();
        let cfg = ShapeletDiscoveryConfig {
            min_length: 3,
            max_length: 4,
            max_candidates: None,
            max_shapelets: 2,
            seed: 0,
            ..Default::default()
        };
        let set = discover_shapelets(&data, &labels, &cfg).unwrap();
        let features = shapelet_transform(&set, &data).unwrap();

        for j in 0..set.len() {
            let s = &set.shapelets()[j];
            for i in 0..n {
                let row = data.row(i);
                let (expected, _off) = shapelet_distance(&s.values, &row, f64::INFINITY).unwrap();
                let got = features.get(i, j).unwrap();
                assert_eq!(got, expected, "X[{i},{j}] != direct sdist");
            }
        }
    }

    #[test]
    fn test_transform_short_series_error() {
        let (train, labels) = labeled_dataset(12, 24);
        let cfg = default_cfg();
        let fit = shapelet_transform_fit(&train, &labels, &cfg).unwrap();

        // A new series shorter than the longest shapelet must error.
        let longest = fit
            .shapelets()
            .shapelets()
            .iter()
            .map(|s| s.length)
            .max()
            .unwrap();
        let short_m = longest - 1;
        assert!(short_m >= 1, "test setup: need a positive short length");
        let (short_data, _) = labeled_dataset(4, short_m);
        let err = fit.transform(&short_data).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidDimension { .. }),
            "expected InvalidDimension, got {err:?}"
        );
    }

    #[test]
    fn test_transform_empty_set_error() {
        // Hand-build an empty ShapeletSet (K = 0).
        let empty = ShapeletSet {
            shapelets: Vec::new(),
            quality: crate::shapelet::QualityMeasure::InfoGain,
        };
        let (data, _labels) = labeled_dataset(4, 10);
        let err = shapelet_transform(&empty, &data).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidParameter { .. }),
            "expected InvalidParameter, got {err:?}"
        );
    }
}
