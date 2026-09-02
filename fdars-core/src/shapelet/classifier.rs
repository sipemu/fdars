//! Bundled shapelet-transform classifier: discover → transform → classify.
//!
//! This is the end-to-end pipeline matching sktime's `ShapeletTransformClassifier`.
//! [`shapelet_classifier_fit`] discovers a [`ShapeletSet`] from labeled training
//! curves (Phase 58), transforms the training set into an `n×K` shapelet-distance
//! feature matrix (Phase 59), and fits an existing fdars classifier on those
//! features (k-NN by default, LDA optionally). [`ShapeletClassifierFit::predict`]
//! transforms new curves through the identical stored shapelets and classifies
//! them with the stored inner model — no re-discovery, no re-normalization.
//!
//! # Divergence from the inner classifier's usual input
//!
//! The inner classifiers ([`fclassif_knn_fit`], [`fclassif_lda_fit`]) run FPCA on
//! their `data` argument. Here `data` is the `n×K` **shapelet-distance** matrix,
//! not functional evaluation points — so the inner FPCA is applied to distance
//! features rather than curves. With the default `ncomp = None` we use `ncomp = K`
//! (full rank, clamped to `min(K, n-1)`), so the FPCA rotation is a full-rank
//! orthonormal change of basis that preserves all feature information; the k-NN /
//! LDA decision then operates on an information-preserving rotation of the raw
//! shapelet-distance features. sktime uses a RotationForest on the transformed
//! features; fdars deliberately reuses its existing k-NN / LDA machinery instead.
//!
//! # Train/test discipline
//!
//! Shapelet quality is computed on the training split only (correct by the
//! Hills/Lines design). Never pass test data into [`shapelet_classifier_fit`], and
//! never report [`ShapeletClassifierFit::train_accuracy`] as a generalization
//! estimate — evaluate on a held-out split via [`ShapeletClassifierFit::predict`].

use crate::classification::{fclassif_knn_fit, fclassif_lda_fit, ClassifFit};
use crate::error::FdarError;
use crate::explain_generic::{FpcPredictor, TaskType};
use crate::matrix::FdMatrix;
use crate::shapelet::discovery::{ShapeletDiscoveryConfig, ShapeletSet};
use crate::shapelet::transform::{shapelet_transform_fit, ShapeletTransformFit};

/// The inner classifier the shapelet-transform pipeline trains on the `n×K`
/// distance-feature matrix.
///
/// Defaults to canonical 1-nearest-neighbor (Hills/Lines).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ShapeletClassifier {
    /// k-nearest-neighbors on the shapelet-distance features.
    Knn {
        /// Number of neighbors.
        k: usize,
    },
    /// Linear discriminant analysis on the shapelet-distance features.
    Lda,
}

impl Default for ShapeletClassifier {
    fn default() -> Self {
        Self::Knn { k: 1 }
    }
}

/// Configuration for [`shapelet_classifier_fit`].
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShapeletClassifierConfig {
    /// Shapelet discovery configuration (length range, candidate cap, K, seed).
    pub discovery: ShapeletDiscoveryConfig,
    /// Inner classifier trained on the `n×K` distance features.
    pub classifier: ShapeletClassifier,
    /// FPCA components for the inner classifier on the `K`-column feature matrix.
    ///
    /// `None` (the default) uses `ncomp = K` (full rank, clamped to `min(K, n-1)`),
    /// so the inner FPCA is an information-preserving rotation of the raw
    /// shapelet-distance features. See the module docs for the divergence note.
    pub ncomp: Option<usize>,
}

/// A fitted shapelet-transform classifier: the discovered shapelet transform plus
/// the inner classifier trained on the `n×K` distance features.
///
/// Stores the (already z-normalized) shapelets and the fitted inner [`ClassifFit`]
/// so that [`predict`](Self::predict) reuses the identical shapelets, normalization,
/// and FPCA rotation as the fit.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ShapeletClassifierFit {
    /// The fitted shapelet transform (discovered shapelets + training features).
    pub transform: ShapeletTransformFit,
    /// The inner classifier fitted on the `n×K` shapelet-distance features.
    pub classifier: ClassifFit,
    /// The configuration used to produce this fit.
    pub config: ShapeletClassifierConfig,
    /// Distinct original labels in sorted order; index = remapped class produced by
    /// the inner classifier, value = the caller's original label. Used to map inner
    /// predictions back to the caller's label space.
    pub classes: Vec<usize>,
}

impl ShapeletClassifierFit {
    /// The fitted shapelet set (already z-normalized, ordered by quality).
    #[must_use]
    pub fn shapelets(&self) -> &ShapeletSet {
        self.transform.shapelets()
    }

    /// The fitted shapelet transform.
    #[must_use]
    pub fn transform(&self) -> &ShapeletTransformFit {
        &self.transform
    }

    /// The inner classifier fitted on the shapelet-distance features.
    #[must_use]
    pub fn classifier(&self) -> &ClassifFit {
        &self.classifier
    }

    /// Training-set accuracy of the inner classifier on the shapelet-distance
    /// features.
    ///
    /// **This is not a generalization estimate.** The shapelets were selected to
    /// separate the training classes; evaluate on a held-out split via
    /// [`predict`](Self::predict) instead.
    #[must_use]
    pub fn train_accuracy(&self) -> f64 {
        self.classifier.result.accuracy
    }

    /// Predict class labels for out-of-sample curves.
    ///
    /// Transforms `new_data` through the stored shapelets (identical sequences,
    /// stored z-normalization) into an `n_new×K` distance-feature matrix, then
    /// classifies each row with the stored inner [`ClassifFit`] by reusing the
    /// [`FpcPredictor`] projection path (project features → FPC scores →
    /// `predict_from_scores`). Predictions are mapped back to the caller's original
    /// label space.
    ///
    /// # Errors
    ///
    /// - [`FdarError::InvalidDimension`] (propagated from the transform) if any
    ///   series in `new_data` is shorter than a discovered shapelet.
    /// - [`FdarError::InvalidParameter`] if the stored shapelet set is empty.
    #[must_use = "predicted labels should not be discarded"]
    pub fn predict(&self, new_data: &FdMatrix) -> Result<Vec<usize>, FdarError> {
        let features = self.transform.transform(new_data)?;
        let scores = self.classifier.project(&features);
        let d = scores.ncols();
        let n_new = scores.nrows();
        let task = self.classifier.task_type();

        let mut out = Vec::with_capacity(n_new);
        for i in 0..n_new {
            let row: Vec<f64> = (0..d).map(|j| scores[(i, j)]).collect();
            let raw = self.classifier.predict_from_scores(&row, None);
            // Map the FpcPredictor output to a remapped class index.
            let remapped = match task {
                TaskType::BinaryClassification => usize::from(raw >= 0.5),
                TaskType::MulticlassClassification(_) => raw.round() as usize,
                // The inner model is always a classifier here; regression is
                // unreachable, but fall back to a rounded class index.
                TaskType::Regression => raw.round() as usize,
            };
            // Map the remapped class back to the caller's original label.
            let label = self.classes.get(remapped).copied().unwrap_or(remapped);
            out.push(label);
        }
        Ok(out)
    }
}

/// Fit a bundled shapelet-transform classifier: discover shapelets, transform the
/// training curves to an `n×K` distance-feature matrix, and train an inner fdars
/// classifier (k-NN default, LDA optional) on those features.
///
/// The returned [`ShapeletClassifierFit`] stores the discovered shapelets and the
/// fitted inner model; reuse them on new curves via
/// [`ShapeletClassifierFit::predict`].
///
/// `data` is a column-major [`FdMatrix`] with rows = curves, columns = evaluation
/// points; `labels[i]` is the integer class of curve `i`.
///
/// The inner classifier's FPCA `ncomp` is resolved from `config.ncomp` as
/// `ncomp.unwrap_or(K).min(K).min(n-1).max(1)` where `K` is the number of
/// discovered shapelets — full rank by default, so the inner FPCA is an
/// information-preserving rotation of the raw shapelet-distance features. See the
/// module docs for the divergence note (sktime uses RotationForest; fdars reuses
/// k-NN / LDA).
///
/// # Errors
///
/// - Any error from shapelet discovery/transform (e.g. label/row mismatch,
///   fewer than 2 classes, a series shorter than a discovered shapelet).
/// - Any error from the inner classifier fit.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::{shapelet_classifier_fit, ShapeletClassifierConfig, ShapeletDiscoveryConfig};
///
/// // Build a 2-class dataset: class 1 carries a triangular motif class 0 lacks.
/// fn make(n: usize, m: usize) -> (FdMatrix, Vec<usize>) {
///     let mut flat = vec![0.0f64; n * m];
///     let mut labels = vec![0usize; n];
///     let (start, len) = (m / 2, (m / 4).max(1));
///     for i in 0..n {
///         let class1 = i % 2 == 1;
///         labels[i] = usize::from(class1);
///         for j in 0..m {
///             let hash = (i.wrapping_mul(2654435761) ^ j.wrapping_mul(40503)) % 211;
///             let mut v = 0.01 * (i as f64) + (j as f64) * 0.001 + 0.05 * (hash as f64 / 211.0 - 0.5);
///             if class1 && j >= start && j < start + len {
///                 let k = j - start;
///                 let half = len / 2;
///                 v += if k <= half { k as f64 } else { (len - k) as f64 };
///             }
///             flat[i + j * n] = v;
///         }
///     }
///     (FdMatrix::from_column_major(flat, n, m).unwrap(), labels)
/// }
///
/// // TRAIN/TEST discipline: discover on train only, evaluate on held-out test.
/// let (train, train_y) = make(24, 24);
/// let (test, test_y) = make(12, 24);
///
/// let cfg = ShapeletClassifierConfig {
///     discovery: ShapeletDiscoveryConfig { min_length: 3, max_length: 6, max_shapelets: 4, ..Default::default() },
///     ..Default::default()
/// };
/// let fit = shapelet_classifier_fit(&train, &train_y, &cfg).unwrap();
///
/// let preds = fit.predict(&test).unwrap();
/// let correct = preds.iter().zip(&test_y).filter(|(p, t)| p == t).count();
/// let acc = correct as f64 / test_y.len() as f64;
/// assert!(acc > 0.5, "held-out accuracy {acc} should beat chance");
/// ```
#[must_use = "the fitted classifier should not be discarded"]
pub fn shapelet_classifier_fit(
    data: &FdMatrix,
    labels: &[usize],
    config: &ShapeletClassifierConfig,
) -> Result<ShapeletClassifierFit, FdarError> {
    // Discover shapelets on the training split + transform to n×K features.
    let transform = shapelet_transform_fit(data, labels, &config.discovery)?;
    let features = transform.features().clone();
    let k = transform.shapelets().len();
    let n = features.nrows();

    // Resolve inner FPCA components: full rank (=K) by default, clamped to the
    // FPCA bound min(K, n-1), never below 1.
    let ncomp = config
        .ncomp
        .unwrap_or(k)
        .min(k)
        .min(n.saturating_sub(1))
        .max(1);

    // Fit the inner classifier on the shapelet-distance features.
    let classifier = match config.classifier {
        ShapeletClassifier::Knn { k: k_nn } => {
            fclassif_knn_fit(&features, labels, None, ncomp, k_nn)?
        }
        ShapeletClassifier::Lda => fclassif_lda_fit(&features, labels, None, ncomp)?,
    };

    // Sorted-unique original labels: the inner classifier remaps labels to 0..G-1
    // in this order, so this vector maps a remapped class back to the caller's label.
    let mut classes: Vec<usize> = labels.to_vec();
    classes.sort_unstable();
    classes.dedup();

    Ok(ShapeletClassifierFit {
        transform,
        classifier,
        config: config.clone(),
        classes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapelet::discovery::ShapeletDiscoveryConfig;

    /// Two-class dataset: class 1 carries a triangular motif class 0 lacks.
    /// Labels alternate (even → class 0, odd → class 1).
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

    fn discovery_cfg() -> ShapeletDiscoveryConfig {
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
    fn test_stc_fit_predict_end_to_end() {
        // TRAIN/TEST discipline: discover on train only, predict on held-out test.
        let (train, train_y) = labeled_dataset(24, 24);
        let (test, test_y) = labeled_dataset(12, 24);
        let cfg = ShapeletClassifierConfig {
            discovery: discovery_cfg(),
            ..Default::default()
        };
        let fit = shapelet_classifier_fit(&train, &train_y, &cfg).unwrap();

        let preds = fit.predict(&test).unwrap();
        assert_eq!(preds.len(), test_y.len());
        let correct = preds.iter().zip(&test_y).filter(|(p, t)| p == t).count();
        let acc = correct as f64 / test_y.len() as f64;
        assert!(
            acc > 0.6,
            "held-out accuracy {acc} should be well above chance (0.5)"
        );
    }

    #[test]
    fn test_stc_knn_default() {
        // Default config uses 1-NN.
        assert_eq!(
            ShapeletClassifierConfig::default().classifier,
            ShapeletClassifier::Knn { k: 1 }
        );
        let (train, train_y) = labeled_dataset(20, 24);
        let cfg = ShapeletClassifierConfig {
            discovery: discovery_cfg(),
            ..Default::default()
        };
        let fit = shapelet_classifier_fit(&train, &train_y, &cfg).unwrap();
        let acc = fit.train_accuracy();
        assert!(
            (0.0..=1.0).contains(&acc),
            "train_accuracy out of range: {acc}"
        );
    }

    #[test]
    fn test_stc_lda_option() {
        let (train, train_y) = labeled_dataset(24, 24);
        let (test, _test_y) = labeled_dataset(10, 24);
        let cfg = ShapeletClassifierConfig {
            discovery: discovery_cfg(),
            classifier: ShapeletClassifier::Lda,
            ncomp: None,
        };
        let fit = shapelet_classifier_fit(&train, &train_y, &cfg).unwrap();
        let preds = fit.predict(&test).unwrap();
        assert_eq!(preds.len(), 10);
        for &p in &preds {
            assert!(p == 0 || p == 1, "unexpected label {p}");
        }
    }

    #[test]
    fn test_stc_predict_consistency() {
        // LDA is deterministic at fit and predict on the same projected scores, so
        // predicting the training curves reproduces the fit-time training predictions.
        let (train, train_y) = labeled_dataset(24, 24);
        let cfg = ShapeletClassifierConfig {
            discovery: discovery_cfg(),
            classifier: ShapeletClassifier::Lda,
            ncomp: None,
        };
        let fit = shapelet_classifier_fit(&train, &train_y, &cfg).unwrap();

        // Fit-time predictions are stored in remapped space; map them back to the
        // caller's labels via the stored `classes` list.
        let fit_time: Vec<usize> = fit
            .classifier
            .result
            .predicted
            .iter()
            .map(|&r| fit.classes[r])
            .collect();

        let re = fit.predict(&train).unwrap();
        assert_eq!(
            re, fit_time,
            "predict(train) != fit-time training predictions"
        );
    }

    #[test]
    fn test_stc_validation() {
        // Single-class labels → error (propagated from discovery/classifier).
        let (data, _labels) = labeled_dataset(8, 24);
        let single = vec![0usize; 8];
        let cfg = ShapeletClassifierConfig {
            discovery: discovery_cfg(),
            ..Default::default()
        };
        assert!(shapelet_classifier_fit(&data, &single, &cfg).is_err());

        // Label/row length mismatch → error.
        let short = vec![0usize, 1, 0];
        assert!(shapelet_classifier_fit(&data, &short, &cfg).is_err());
    }

    #[test]
    fn test_shapelet_reexports() {
        // Crate-root re-exported names must be reachable.
        use crate::{
            discover_shapelets, shapelet_classifier_fit as _scf, shapelet_distance,
            shapelet_transform, shapelet_transform_fit, QualityMeasure, Shapelet,
            ShapeletClassifier, ShapeletClassifierConfig, ShapeletClassifierFit,
            ShapeletDiscoveryConfig, ShapeletSet, ShapeletTransformFit,
        };
        // Reference each to prove the path resolves (compile-level).
        let _ = _scf;
        let _ = shapelet_distance;
        let _ = discover_shapelets;
        let _ = shapelet_transform;
        let _ = shapelet_transform_fit;
        let _c: fn() -> ShapeletClassifierConfig = ShapeletClassifierConfig::default;
        let _q = QualityMeasure::InfoGain;
        let _cl = ShapeletClassifier::default();
        // Type-level references.
        fn _takes(
            _a: &Shapelet,
            _b: &ShapeletSet,
            _c: &ShapeletDiscoveryConfig,
            _d: &ShapeletTransformFit,
            _e: &ShapeletClassifierFit,
        ) {
        }
    }
}
