//! SRVF-based outlier detection using elastic distances.
//!
//! Detects outlier curves by computing elastic distances from a reference
//! (Karcher mean or median) and applying a threshold rule: the Tukey fence
//! (default) or an empirical `(1 − alpha)` quantile (see [`OutlierThreshold`]).

use super::karcher::karcher_mean;
use super::pairwise::{amplitude_distance, elastic_distance, phase_distance_pair};
use super::robust_karcher::{karcher_median, RobustKarcherConfig};
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Rule turning the per-curve elastic distances into an outlier cutoff.
///
/// # Example
/// ```
/// use fdars_core::alignment::{ElasticOutlierConfig, OutlierThreshold};
///
/// let mut cfg = ElasticOutlierConfig::default();
/// assert_eq!(cfg.threshold_method, OutlierThreshold::TukeyFence { k: 1.5 });
/// // Flag curves beyond the empirical 90% quantile of the distances.
/// cfg.threshold_method = OutlierThreshold::Quantile;
/// cfg.alpha = 0.10;
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutlierThreshold {
    /// Tukey fence `Q3 + k · IQR` on the distances (quartiles by linear
    /// interpolation). `k = 1.5` is the default and reproduces the fixed fence
    /// used before this option existed (fdasrvf `outlier.detection(k = 1.5)`
    /// uses the same boxplot rule). With fewer than 4 curves the cutoff is
    /// `max(distance) + 1`, i.e. nothing is flagged. Ignores `alpha`.
    TukeyFence {
        /// Fence multiplier (finite, `>= 0`).
        k: f64,
    },
    /// Empirical `(1 − alpha)` quantile of the distances (linear
    /// interpolation between order statistics), using
    /// [`ElasticOutlierConfig::alpha`] (`0 < alpha < 1`). Curves strictly
    /// above it are flagged, so roughly a fraction `alpha` of the sample is
    /// flagged whatever its shape: this is a screening rule, not a test.
    Quantile,
}

impl Default for OutlierThreshold {
    fn default() -> Self {
        Self::TukeyFence { k: 1.5 }
    }
}

/// Configuration for elastic outlier detection.
///
/// Construct via `ElasticOutlierConfig::default()`, then assign the fields you need (e.g. `let mut c = ElasticOutlierConfig::default(); c.field = …;`). This struct is `#[non_exhaustive]`, so external crates cannot build it with a struct literal — not even functional-update `..Default::default()` form.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticOutlierConfig {
    /// Roughness penalty for elastic alignment (0.0 = no penalty).
    pub lambda: f64,
    /// Tail fraction used by [`OutlierThreshold::Quantile`] (default 0.05):
    /// the cutoff is the empirical `(1 − alpha)` quantile of the distances.
    /// It has **no effect** with the default [`OutlierThreshold::TukeyFence`].
    pub alpha: f64,
    /// If `true`, use the Karcher median as reference (more robust).
    /// If `false`, use the Karcher mean.
    pub use_median: bool,
    /// Threshold rule (default: Tukey fence with `k = 1.5`, identical to
    /// the behaviour before this field existed).
    pub threshold_method: OutlierThreshold,
}

impl Default for ElasticOutlierConfig {
    fn default() -> Self {
        Self {
            lambda: 0.0,
            alpha: 0.05,
            use_median: true,
            threshold_method: OutlierThreshold::default(),
        }
    }
}

/// Result of elastic outlier detection.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ElasticOutlierResult {
    /// Indices of detected outlier curves.
    pub outlier_indices: Vec<usize>,
    /// Total elastic distance from each curve to the reference (length n).
    pub distances: Vec<f64>,
    /// Cutoff distance (Tukey fence `Q3 + k·IQR`, or the `(1 − alpha)`
    /// quantile, per [`ElasticOutlierConfig::threshold_method`]).
    pub threshold: f64,
    /// Amplitude component of elastic distance for each curve (length n).
    pub amplitude_distances: Vec<f64>,
    /// Phase component of elastic distance for each curve (length n).
    pub phase_distances: Vec<f64>,
}

/// Detect outlier curves using elastic distances and a threshold rule.
///
/// Computes a reference curve (Karcher mean or median), then measures the
/// elastic distance from each curve to the reference. Curves whose distance
/// exceeds the cutoff are flagged. The cutoff is the Tukey fence
/// `Q3 + 1.5·IQR` by default, or an empirical `(1 − alpha)` quantile with
/// [`OutlierThreshold::Quantile`]. The per-curve distances are computed in
/// parallel (results are identical to a sequential run).
///
/// # Arguments
/// * `data`    — Functional data matrix (n x m).
/// * `argvals` — Evaluation points (length m).
/// * `config`  — Configuration parameters.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if `argvals` length does not match `m`
/// or `n < 2`, and [`FdarError::InvalidParameter`] if the Tukey `k` is negative
/// or non-finite, or if `alpha` is not in `(0, 1)` with
/// [`OutlierThreshold::Quantile`].
///
/// # Example
/// ```
/// use fdars_core::alignment::{elastic_outlier_detection, ElasticOutlierConfig};
/// use fdars_core::matrix::FdMatrix;
///
/// let (n, m) = (8, 30);
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let mut data = FdMatrix::zeros(n, m);
/// for i in 0..n {
///     for j in 0..m {
///         data[(i, j)] = ((t[j] + 0.02 * i as f64) * 4.0).sin();
///     }
/// }
/// for j in 0..m {
///     data[(7, j)] = 10.0 * (20.0 * t[j]).cos(); // gross outlier
/// }
/// let res = elastic_outlier_detection(&data, &t, &ElasticOutlierConfig::default()).unwrap();
/// assert!(res.outlier_indices.contains(&7));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_outlier_detection(
    data: &FdMatrix,
    argvals: &[f64],
    config: &ElasticOutlierConfig,
) -> Result<ElasticOutlierResult, FdarError> {
    let (n, m) = data.shape();

    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 rows".to_string(),
            actual: format!("{n} rows"),
        });
    }

    validate_threshold(config)?;

    // Step 1: Compute reference curve.
    let reference = if config.use_median {
        let median_config = RobustKarcherConfig {
            max_iter: 15,
            tol: 1e-3,
            lambda: config.lambda,
            trim_fraction: 0.1,
        };
        let result = karcher_median(data, argvals, &median_config)?;
        result.mean
    } else {
        let result = karcher_mean(data, argvals, 15, 1e-3, config.lambda);
        result.mean
    };

    // Step 2: Compute elastic distances from reference.
    let distances: Vec<f64> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let fi = data.row(i);
            elastic_distance(&reference, &fi, argvals, config.lambda)
        })
        .collect();

    // Step 3: Compute amplitude and phase distances.
    let amplitude_distances: Vec<f64> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let fi = data.row(i);
            amplitude_distance(&reference, &fi, argvals, config.lambda)
        })
        .collect();

    let phase_distances: Vec<f64> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let fi = data.row(i);
            phase_distance_pair(&reference, &fi, argvals, config.lambda)
        })
        .collect();

    // Step 4: Threshold on total distances.
    let threshold = match config.threshold_method {
        OutlierThreshold::TukeyFence { k } => tukey_fence(&distances, k),
        OutlierThreshold::Quantile => quantile_threshold(&distances, config.alpha),
    };

    // Step 5: Identify outliers.
    let outlier_indices: Vec<usize> = (0..n).filter(|&i| distances[i] > threshold).collect();

    Ok(ElasticOutlierResult {
        outlier_indices,
        distances,
        threshold,
        amplitude_distances,
        phase_distances,
    })
}

/// Validate the threshold-rule parameters.
fn validate_threshold(config: &ElasticOutlierConfig) -> Result<(), FdarError> {
    match config.threshold_method {
        OutlierThreshold::TukeyFence { k } if !(k.is_finite() && k >= 0.0) => {
            Err(FdarError::InvalidParameter {
                parameter: "threshold_method",
                message: format!("Tukey fence k must be finite and >= 0, got {k}"),
            })
        }
        OutlierThreshold::Quantile if !(config.alpha > 0.0 && config.alpha < 1.0) => {
            Err(FdarError::InvalidParameter {
                parameter: "alpha",
                message: format!(
                    "alpha must be in (0, 1) for the quantile threshold, got {}",
                    config.alpha
                ),
            })
        }
        _ => Ok(()),
    }
}

/// Empirical `(1 − alpha)` quantile of `values` (linear interpolation).
pub(super) fn quantile_threshold(values: &[f64], alpha: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile_sorted(&sorted, 100.0 * (1.0 - alpha))
}

/// Compute the Tukey fence threshold: `Q3 + k · IQR`.
pub(super) fn tukey_fence(values: &[f64], k: f64) -> f64 {
    let n = values.len();
    if n < 4 {
        // With very few values, use max + epsilon as a permissive threshold.
        return values.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 1.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q1 = percentile_sorted(&sorted, 25.0);
    let q3 = percentile_sorted(&sorted, 75.0);
    let iqr = q3 - q1;

    q3 + k * iqr
}

/// Compute a percentile from a sorted slice using linear interpolation.
fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }

    let rank = pct / 100.0 * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;

    if lo >= n || hi >= n {
        sorted[n - 1]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]
    use super::*;
    use crate::test_helpers::uniform_grid;

    fn make_clean_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let t = uniform_grid(m);
        let mut data_vec = vec![0.0; n * m];
        for i in 0..n {
            let phase = 0.02 * i as f64;
            for j in 0..m {
                data_vec[i + j * n] = ((t[j] + phase) * 4.0).sin();
            }
        }
        let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();
        (data, t)
    }

    #[test]
    fn outlier_detection_no_outliers() {
        let (data, t) = make_clean_data(6, 20);
        let config = ElasticOutlierConfig::default();
        let result = elastic_outlier_detection(&data, &t, &config).unwrap();

        assert_eq!(result.distances.len(), 6);
        assert_eq!(result.amplitude_distances.len(), 6);
        assert_eq!(result.phase_distances.len(), 6);
        assert!(result.threshold > 0.0);

        // With clean homogeneous data, expect no (or very few) outliers.
        assert!(
            result.outlier_indices.len() <= 1,
            "clean data should have at most 1 outlier, got {}",
            result.outlier_indices.len()
        );
    }

    #[test]
    fn outlier_detection_finds_extreme() {
        let m = 20;
        let t = uniform_grid(m);
        let n = 8;
        let mut data_vec = vec![0.0; n * m];

        // 7 clean curves.
        for i in 0..7 {
            let phase = 0.02 * i as f64;
            for j in 0..m {
                data_vec[i + j * n] = ((t[j] + phase) * 4.0).sin();
            }
        }
        // 1 extreme outlier (curve 7).
        for j in 0..m {
            data_vec[7 + j * n] = (t[j] * 20.0).cos() * 10.0;
        }
        let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();

        let config = ElasticOutlierConfig::default();
        let result = elastic_outlier_detection(&data, &t, &config).unwrap();

        // The outlier (index 7) should be detected.
        assert!(
            result.outlier_indices.contains(&7),
            "should detect curve 7 as outlier, detected: {:?}",
            result.outlier_indices
        );

        // The outlier's distance should exceed the threshold.
        assert!(
            result.distances[7] > result.threshold,
            "outlier distance ({}) should exceed threshold ({})",
            result.distances[7],
            result.threshold
        );
    }

    #[test]
    fn outlier_detection_config_default() {
        let cfg = ElasticOutlierConfig::default();
        assert!((cfg.lambda - 0.0).abs() < f64::EPSILON);
        assert!((cfg.alpha - 0.05).abs() < f64::EPSILON);
        assert!(cfg.use_median);
        assert_eq!(
            cfg.threshold_method,
            OutlierThreshold::TukeyFence { k: 1.5 }
        );
    }

    #[test]
    fn tukey_fence_hand_computed() {
        // Sorted 1..=8: Q1 at rank 1.75 -> 2.75, Q3 at rank 5.25 -> 6.25, IQR 3.5.
        let v: Vec<f64> = (1..=8).rev().map(f64::from).collect();
        assert!((tukey_fence(&v, 1.5) - 11.5).abs() < 1e-12);
        assert!((tukey_fence(&v, 0.0) - 6.25).abs() < 1e-12);
        assert!((tukey_fence(&v, 3.0) - 16.75).abs() < 1e-12);
    }

    #[test]
    fn quantile_threshold_hand_computed() {
        // Sorted 1..=10, alpha = 0.1: rank 0.9 * 9 = 8.1 -> 9 + 0.1 * (10 - 9).
        let v: Vec<f64> = (1..=10).rev().map(f64::from).collect();
        assert!((quantile_threshold(&v, 0.1) - 9.1).abs() < 1e-12);
        // alpha = 0.5 is the median of 1..=10.
        assert!((quantile_threshold(&v, 0.5) - 5.5).abs() < 1e-12);
    }

    fn data_with_outlier() -> (FdMatrix, Vec<f64>) {
        let (n, m) = (9, 20);
        let t = uniform_grid(m);
        let mut d = FdMatrix::zeros(n, m);
        for i in 0..n - 1 {
            for j in 0..m {
                d[(i, j)] = ((t[j] + 0.02 * i as f64) * 4.0).sin() * (1.0 + 0.03 * i as f64);
            }
        }
        for j in 0..m {
            d[(n - 1, j)] = (t[j] * 20.0).cos() * 10.0;
        }
        (d, t)
    }

    #[test]
    fn quantile_threshold_option_end_to_end() {
        let (d, t) = data_with_outlier();
        let mut cfg = ElasticOutlierConfig::default();
        cfg.threshold_method = OutlierThreshold::Quantile;
        cfg.alpha = 0.12;
        let r = elastic_outlier_detection(&d, &t, &cfg).unwrap();

        // The cutoff is the empirical 88% quantile of the returned distances.
        let mut s = r.distances.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let rank = 0.88 * 8.0; // 7.04: between the 8th and 9th order statistics
        let expect = s[7] + (rank - 7.0) * (s[8] - s[7]);
        assert!((r.threshold - expect).abs() <= 1e-12 * expect.abs().max(1.0));
        // Only the largest distance exceeds it: the gross outlier.
        assert_eq!(r.outlier_indices, vec![8]);

        // Distances do not depend on the threshold rule.
        let r0 = elastic_outlier_detection(&d, &t, &ElasticOutlierConfig::default()).unwrap();
        assert_eq!(r0.distances, r.distances);
        assert_eq!(r0.amplitude_distances, r.amplitude_distances);
    }

    #[test]
    fn tukey_k_and_alpha_independence() {
        let (d, t) = data_with_outlier();
        let base = elastic_outlier_detection(&d, &t, &ElasticOutlierConfig::default()).unwrap();
        // alpha has no effect on the default Tukey rule.
        let mut cfg = ElasticOutlierConfig::default();
        cfg.alpha = 0.3;
        assert_eq!(elastic_outlier_detection(&d, &t, &cfg).unwrap(), base);
        // An explicit k = 1.5 is the default; k = 0 gives the plain Q3 cutoff.
        cfg.threshold_method = OutlierThreshold::TukeyFence { k: 0.0 };
        let r = elastic_outlier_detection(&d, &t, &cfg).unwrap();
        let mut s = base.distances.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q3 = percentile_sorted(&s, 75.0);
        assert_eq!(r.threshold.to_bits(), q3.to_bits());
        assert!(r.threshold <= base.threshold);
        assert!(r.outlier_indices.len() >= base.outlier_indices.len());
    }

    #[test]
    fn threshold_parameter_validation() {
        let (d, t) = data_with_outlier();
        let mut cfg = ElasticOutlierConfig::default();
        cfg.threshold_method = OutlierThreshold::TukeyFence { k: -1.0 };
        assert!(matches!(
            elastic_outlier_detection(&d, &t, &cfg),
            Err(FdarError::InvalidParameter { .. })
        ));
        cfg.threshold_method = OutlierThreshold::Quantile;
        for a in [0.0, 1.0, f64::NAN] {
            cfg.alpha = a;
            assert!(matches!(
                elastic_outlier_detection(&d, &t, &cfg),
                Err(FdarError::InvalidParameter { .. })
            ));
        }
    }
}
