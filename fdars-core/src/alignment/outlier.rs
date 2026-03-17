//! SRVF-based outlier detection using elastic distances.
//!
//! Detects outlier curves by computing elastic distances from a reference
//! (Karcher mean or median) and applying the Tukey fence rule.

use super::karcher::karcher_mean;
use super::pairwise::{amplitude_distance, elastic_distance, phase_distance_pair};
use super::robust_karcher::{karcher_median, RobustKarcherConfig};
use crate::error::FdarError;
use crate::matrix::FdMatrix;

/// Configuration for elastic outlier detection.
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticOutlierConfig {
    /// Roughness penalty for elastic alignment (0.0 = no penalty).
    pub lambda: f64,
    /// Significance level (controls threshold sensitivity; currently used
    /// to document intent — the actual threshold uses the Tukey fence).
    pub alpha: f64,
    /// If `true`, use the Karcher median as reference (more robust).
    /// If `false`, use the Karcher mean.
    pub use_median: bool,
}

impl Default for ElasticOutlierConfig {
    fn default() -> Self {
        Self {
            lambda: 0.0,
            alpha: 0.05,
            use_median: true,
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
    /// Cutoff distance (Tukey fence: Q3 + 1.5 * IQR).
    pub threshold: f64,
    /// Amplitude component of elastic distance for each curve (length n).
    pub amplitude_distances: Vec<f64>,
    /// Phase component of elastic distance for each curve (length n).
    pub phase_distances: Vec<f64>,
}

/// Detect outlier curves using elastic distances and the Tukey fence.
///
/// Computes a reference curve (Karcher mean or median), then measures the
/// elastic distance from each curve to the reference. Curves exceeding the
/// Tukey fence threshold (Q3 + 1.5 * IQR) are flagged as outliers.
///
/// # Arguments
/// * `data`    — Functional data matrix (n x m).
/// * `argvals` — Evaluation points (length m).
/// * `config`  — Configuration parameters.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if `argvals` length does not match `m`
/// or `n < 2`.
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
    let distances: Vec<f64> = (0..n)
        .map(|i| {
            let fi = data.row(i);
            elastic_distance(&reference, &fi, argvals, config.lambda)
        })
        .collect();

    // Step 3: Compute amplitude and phase distances.
    let amplitude_distances: Vec<f64> = (0..n)
        .map(|i| {
            let fi = data.row(i);
            amplitude_distance(&reference, &fi, argvals, config.lambda)
        })
        .collect();

    let phase_distances: Vec<f64> = (0..n)
        .map(|i| {
            let fi = data.row(i);
            phase_distance_pair(&reference, &fi, argvals, config.lambda)
        })
        .collect();

    // Step 4: Tukey fence on total distances.
    let threshold = tukey_fence(&distances);

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

/// Compute the Tukey fence threshold: Q3 + 1.5 * IQR.
fn tukey_fence(values: &[f64]) -> f64 {
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

    q3 + 1.5 * iqr
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
    }
}
