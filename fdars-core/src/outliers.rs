//! Outlier detection for functional data.
//!
//! This module provides methods for detecting outliers in functional data
//! based on depth measures and likelihood ratio tests.

use crate::depth::band::{modified_band_1d, modified_epigraph_index_1d};
use crate::depth::{functional_boxplot, total_variation_depth_1d, DepthMethod};
use crate::error::FdarError;
use crate::helpers::{quantile_sorted, sort_nan_safe};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::streaming_depth::{SortedReferenceState, StreamingDepth, StreamingFraimanMuniz};
use rand::prelude::*;
use rand_distr::StandardNormal;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Compute trimmed mean and variance from data using depth-based trimming.
///
/// Returns (trimmed_mean, trimmed_var) each of length m.
fn compute_trimmed_stats(data: &FdMatrix, depths: &[f64], n_keep: usize) -> (Vec<f64>, Vec<f64>) {
    let m = data.ncols();

    let mut depth_idx: Vec<(usize, f64)> =
        depths.iter().enumerate().map(|(i, &d)| (i, d)).collect();
    // O(n) partial sort instead of O(n log n) full sort — we only need the top n_keep elements
    if n_keep < depth_idx.len() {
        depth_idx.select_nth_unstable_by(n_keep - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let keep_idx: Vec<usize> = depth_idx[..n_keep].iter().map(|(i, _)| *i).collect();

    let results: Vec<(f64, f64)> = iter_maybe_parallel!(0..m)
        .map(|j| {
            let mut mean_j = 0.0;
            for &i in &keep_idx {
                mean_j += data[(i, j)];
            }
            mean_j /= n_keep as f64;

            let mut var_j = 0.0;
            for &i in &keep_idx {
                let diff = data[(i, j)] - mean_j;
                var_j += diff * diff;
            }
            var_j /= n_keep as f64;
            var_j = var_j.max(1e-10);

            (mean_j, var_j)
        })
        .collect();

    let trimmed_mean: Vec<f64> = results.iter().map(|&(m, _)| m).collect();
    let trimmed_var: Vec<f64> = results.iter().map(|&(_, v)| v).collect();

    (trimmed_mean, trimmed_var)
}

/// Compute normalized Mahalanobis-like distance for a single observation.
fn normalized_distance(
    data: &FdMatrix,
    i: usize,
    trimmed_mean: &[f64],
    trimmed_var: &[f64],
) -> f64 {
    let m = data.ncols();
    let mut dist = 0.0;
    for j in 0..m {
        let diff = data[(i, j)] - trimmed_mean[j];
        dist += diff * diff / trimmed_var[j];
    }
    (dist / m as f64).sqrt()
}

/// Compute bootstrap threshold for LRT outlier detection.
///
/// # Arguments
/// * `data` - Functional data matrix (n observations x m evaluation points)
/// * `nb` - Number of bootstrap iterations
/// * `smo` - Smoothing parameter for bootstrap
/// * `trim` - Trimming proportion
/// * `seed` - Random seed
/// * `percentile` - Percentile for threshold (e.g., 0.99 for 99th percentile)
///
/// # Returns
/// Threshold at specified percentile for outlier detection
#[must_use = "expensive computation whose result should not be discarded"]
pub fn outliers_threshold_lrt(
    data: &FdMatrix,
    nb: usize,
    smo: f64,
    trim: f64,
    seed: u64,
    percentile: f64,
) -> f64 {
    outliers_threshold_lrt_with_dist(data, nb, smo, trim, seed, percentile).0
}

/// Compute bootstrap threshold and full null distribution for LRT outlier detection.
///
/// Same as [`outliers_threshold_lrt`] but also returns the sorted bootstrap
/// distribution of max-distances, enabling per-curve p-value computation:
/// `p = (sum(boot_dist >= d) + 1) / (B + 1)`.
///
/// # Arguments
/// * `data` - Functional data matrix (n observations x m evaluation points)
/// * `nb` - Number of bootstrap iterations
/// * `smo` - Smoothing parameter for bootstrap
/// * `trim` - Trimming proportion
/// * `seed` - Random seed
/// * `percentile` - Percentile for threshold (e.g., 0.99 for 99th percentile)
///
/// # Returns
/// `(threshold, sorted_distribution)` — threshold at specified percentile and the
/// full sorted bootstrap null distribution of max-distances (length `nb`).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn outliers_threshold_lrt_with_dist(
    data: &FdMatrix,
    nb: usize,
    smo: f64,
    trim: f64,
    seed: u64,
    percentile: f64,
) -> (f64, Vec<f64>) {
    let n = data.nrows();
    let m = data.ncols();

    if n < 3 || m == 0 {
        return (0.0, vec![]);
    }

    let n_keep = ((1.0 - trim) * n as f64).ceil().max(1.0) as usize;
    let n_keep = n_keep.min(n);

    // Compute column standard deviations for smoothing
    let col_vars: Vec<f64> = iter_maybe_parallel!(0..m)
        .map(|j| {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for i in 0..n {
                let val = data[(i, j)];
                sum += val;
                sum_sq += val * val;
            }
            let mean = sum / n as f64;
            // Bootstrap variance, population formula
            let var = sum_sq / n as f64 - mean * mean;
            var.max(0.0).sqrt()
        })
        .collect();

    // Run bootstrap iterations in parallel
    let max_dists: Vec<f64> = iter_maybe_parallel!(0..nb)
        .map(|b| {
            let mut rng = crate::helpers::seed_for_thread(seed, b);

            // Resample with replacement and add smoothing noise
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
            // Pre-generate all noise values to preserve RNG sequence
            let noise_vals: Vec<f64> = (0..n * m)
                .map(|_| rng.sample::<f64, _>(StandardNormal))
                .collect();
            let mut boot_data = FdMatrix::zeros(n, m);
            // Column-first iteration for cache-friendly writes in column-major layout
            for j in 0..m {
                let smo_var = smo * col_vars[j];
                for (new_i, &old_i) in indices.iter().enumerate() {
                    let noise = noise_vals[new_i * m + j] * smo_var;
                    boot_data[(new_i, j)] = data[(old_i, j)] + noise;
                }
            }

            // Compute trimmed stats from bootstrap sample
            let state = SortedReferenceState::from_reference(&boot_data);
            let streaming_fm = StreamingFraimanMuniz::new(state, true);
            let depths = streaming_fm.depth_batch(&boot_data);
            let (trimmed_mean, trimmed_var) = compute_trimmed_stats(&boot_data, &depths, n_keep);

            // Find max normalized distance across all observations
            (0..n)
                .map(|i| normalized_distance(&boot_data, i, &trimmed_mean, &trimmed_var))
                .fold(0.0_f64, f64::max)
        })
        .collect();

    // Sort and extract threshold at specified percentile
    let mut sorted_dists = max_dists;
    crate::helpers::sort_nan_safe(&mut sorted_dists);
    let idx =
        crate::utility::f64_to_usize_clamped(nb as f64 * percentile).min(nb.saturating_sub(1));
    let threshold = sorted_dists.get(idx).copied().unwrap_or(0.0);
    (threshold, sorted_dists)
}

/// Detect outliers using LRT method.
///
/// # Arguments
/// * `data` - Functional data matrix (n observations x m evaluation points)
/// * `threshold` - Outlier threshold
/// * `trim` - Trimming proportion
///
/// # Returns
/// Vector of booleans indicating outliers
#[must_use = "expensive computation whose result should not be discarded"]
/// Detect outliers in functional data using the Likelihood Ratio Test.
///
/// Compares each observation's normalized distance against a threshold.
/// Observations exceeding the threshold are flagged as outliers.
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
/// * `threshold` - Decision threshold (from [`outliers_threshold_lrt`])
/// * `trim` - Trimming proportion for robust estimation
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::outliers::detect_outliers_lrt;
///
/// let data = FdMatrix::from_column_major(
///     (0..50).map(|i| (i as f64 * 0.1).sin()).collect(),
///     5, 10,
/// ).unwrap();
/// let outliers = detect_outliers_lrt(&data, 3.0, 0.1);
/// assert_eq!(outliers.len(), 5);
/// ```
pub fn detect_outliers_lrt(data: &FdMatrix, threshold: f64, trim: f64) -> Vec<bool> {
    let n = data.nrows();
    let m = data.ncols();

    if n < 3 || m == 0 {
        return vec![false; n];
    }

    let n_keep = ((1.0 - trim) * n as f64).ceil().max(1.0) as usize;
    let n_keep = n_keep.min(n);

    let state = SortedReferenceState::from_reference(data);
    let streaming_fm = StreamingFraimanMuniz::new(state, true);
    let depths = streaming_fm.depth_batch(data);
    let (trimmed_mean, trimmed_var) = compute_trimmed_stats(data, &depths, n_keep);

    iter_maybe_parallel!(0..n)
        .map(|i| normalized_distance(data, i, &trimmed_mean, &trimmed_var) > threshold)
        .collect()
}

/// Result of the outliergram analysis.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct OutligramResult {
    /// Modified Epigraph Index for each curve
    pub mei: Vec<f64>,
    /// Modified Band Depth for each curve
    pub mbd: Vec<f64>,
    /// Parabola coefficients: MBD = a0 + a1*MEI + a2*MEI²
    pub a0: f64,
    pub a1: f64,
    pub a2: f64,
    /// Outlier threshold (IQR-based)
    pub threshold: f64,
    /// Outlier flags (true = outlier)
    pub outlier_flags: Vec<bool>,
}

/// Compute outliergram with parabolic outlier boundary.
///
/// Combines modified band depth and modified epigraph index to detect
/// shape outliers. Fits a parabolic upper bound `MBD = a0 + a1*MEI + a2*MEI²`
/// and flags curves whose residuals fall below an IQR-based threshold.
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
/// * `factor` - IQR multiplier for the outlier threshold (typically 1.5)
///
/// # Returns
/// Outliergram result with depths, parabola coefficients, and outlier flags.
pub fn outliergram(data: &FdMatrix, factor: f64) -> Result<OutligramResult, FdarError> {
    let n = data.nrows();
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 rows".to_string(),
            actual: format!("{n} rows"),
        });
    }

    let mei = modified_epigraph_index_1d(data, data);
    let mbd = modified_band_1d(data, data);

    // Fit parabola: MBD = a0 + a1*MEI + a2*MEI²
    // Using normal equations: X = [1, mei, mei²], y = mbd
    let mut xtx = [[0.0; 3]; 3];
    let mut xty = [0.0; 3];
    for i in 0..n {
        let x = [1.0, mei[i], mei[i] * mei[i]];
        for r in 0..3 {
            for c in 0..3 {
                xtx[r][c] += x[r] * x[c];
            }
            xty[r] += x[r] * mbd[i];
        }
    }

    // Solve 3x3 system via Cramer's rule
    let (a0, a1, a2) = solve_3x3(xtx, xty);

    // Compute residuals below the parabola
    let residuals: Vec<f64> = (0..n)
        .map(|i| mbd[i] - (a0 + a1 * mei[i] + a2 * mei[i] * mei[i]))
        .collect();

    // IQR-based threshold on residuals
    let mut sorted_resid = residuals.clone();
    sorted_resid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = sorted_resid[n / 4];
    let q3 = sorted_resid[3 * n / 4];
    let iqr = q3 - q1;
    let threshold = q1 - factor * iqr;

    let outlier_flags: Vec<bool> = residuals.iter().map(|&r| r < threshold).collect();

    Ok(OutligramResult {
        mei,
        mbd,
        a0,
        a1,
        a2,
        threshold,
        outlier_flags,
    })
}

/// Result of magnitude-shape outlyingness decomposition.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MagnitudeShapeResult {
    /// Magnitude outlyingness: 1 - MBD (how far from the center)
    pub magnitude: Vec<f64>,
    /// Shape outlyingness: L2 distance of normalized curve direction from mean direction
    pub shape: Vec<f64>,
}

/// Decompose outlyingness into magnitude and shape components.
///
/// Magnitude measures how far a curve is from the center (1 - modified band depth).
/// Shape measures how different the curve's direction is from the mean direction
/// (L2 distance of normalized centered curves).
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
pub fn magnitude_shape_outlyingness(data: &FdMatrix) -> Result<MagnitudeShapeResult, FdarError> {
    let (n, m) = data.shape();
    if n < 2 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 rows and 1 column".to_string(),
            actual: format!("{n} rows, {m} columns"),
        });
    }

    // Magnitude: 1 - modified band depth
    let mbd = modified_band_1d(data, data);
    let magnitude: Vec<f64> = mbd.iter().map(|&d| 1.0 - d).collect();

    // Shape: compute centered curves, normalize directions, compare to mean direction
    // Step 1: compute column means
    let mut col_means = vec![0.0; m];
    for j in 0..m {
        for i in 0..n {
            col_means[j] += data[(i, j)];
        }
        col_means[j] /= n as f64;
    }

    // Step 2: center and normalize each curve (direction)
    let mut directions = vec![vec![0.0; m]; n];
    for i in 0..n {
        let mut norm_sq = 0.0;
        for j in 0..m {
            let c = data[(i, j)] - col_means[j];
            directions[i][j] = c;
            norm_sq += c * c;
        }
        let norm = norm_sq.sqrt().max(1e-15);
        for j in 0..m {
            directions[i][j] /= norm;
        }
    }

    // Step 3: mean direction
    let mut mean_dir = vec![0.0; m];
    for i in 0..n {
        for j in 0..m {
            mean_dir[j] += directions[i][j];
        }
    }
    let mut mean_norm_sq = 0.0;
    for j in 0..m {
        mean_dir[j] /= n as f64;
        mean_norm_sq += mean_dir[j] * mean_dir[j];
    }
    let mean_norm = mean_norm_sq.sqrt().max(1e-15);
    for j in 0..m {
        mean_dir[j] /= mean_norm;
    }

    // Step 4: shape = L2 distance from mean direction
    let shape: Vec<f64> = (0..n)
        .map(|i| {
            let dist_sq: f64 = (0..m)
                .map(|j| {
                    let d = directions[i][j] - mean_dir[j];
                    d * d
                })
                .sum();
            dist_sq.sqrt()
        })
        .collect();

    Ok(MagnitudeShapeResult { magnitude, shape })
}

/// Solve a 3x3 linear system via Cramer's rule.
fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> (f64, f64, f64) {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-15 {
        return (0.0, 0.0, 0.0);
    }

    let det_x = b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]);

    let det_y = a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]);

    let det_z = a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    (det_x / det, det_y / det, det_z / det)
}

/// Lower and upper IQR fences `(Q1 − factor·IQR, Q3 + factor·IQR)` for a value slice.
///
/// Shared cutoff helper for the outlier detectors. Uses linear-interpolation quantiles
/// ([`quantile_sorted`]) rather than R's floor-index quartiles — the two agree for n > 20 and
/// differ only marginally for small samples.
fn iqr_fence(values: &[f64], factor: f64) -> (f64, f64) {
    let mut sorted = values.to_vec();
    sort_nan_safe(&mut sorted);
    let q1 = quantile_sorted(&sorted, 0.25);
    let q3 = quantile_sorted(&sorted, 0.75);
    let iqr = q3 - q1;
    (q1 - factor * iqr, q3 + factor * iqr)
}

/// Configuration for [`tvdmss`]. Defaults reproduce the `fdaoutlier` `tvdmss` defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TvdMssConfig {
    /// IQR multiplier for the shape (MSS lower-fence) stage. Default `1.5`.
    pub emp_factor_mss: f64,
    /// Inflation factor for the magnitude-stage functional boxplot. Default `1.5`.
    pub emp_factor_tvd: f64,
    /// Fraction of the original n used as the stage-2 central region. Default `0.5`.
    /// Documented for parity with `fdaoutlier`; fdars' [`functional_boxplot`] fixes the
    /// central region at the deepest 50%, so this field is currently informational.
    pub central_region_tvd: f64,
}

impl Default for TvdMssConfig {
    fn default() -> Self {
        Self {
            emp_factor_mss: 1.5,
            emp_factor_tvd: 1.5,
            central_region_tvd: 0.5,
        }
    }
}

/// Numeric output of the [`tvdmss`] two-stage detector (no rendering).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct TvdMssOutliers {
    /// Row indices flagged as magnitude outliers (stage 2: functional boxplot on TVD).
    pub magnitude_outliers: Vec<usize>,
    /// Row indices flagged as shape outliers (stage 1: lower IQR fence on MSS).
    pub shape_outliers: Vec<usize>,
    /// Total variation depth per curve (over original indices).
    pub tvd: Vec<f64>,
    /// Modified shape similarity index per curve (over original indices).
    pub mss: Vec<f64>,
}

/// TVD + MSSI two-stage outlier detector (Huang & Sun 2019 / `fdaoutlier::tvdmss`).
///
/// Stage 1 flags **shape outliers** as curves whose MSS falls below the lower IQR fence
/// (`Q1 − emp_factor_mss·IQR`) and below the mean MSS. Stage 2 removes those curves and runs a
/// López-Pintado–Romo depth-fence [`functional_boxplot`] on the remaining curves' values to flag
/// **magnitude outliers**, re-mapping the reduced indices back to the original sample.
///
/// Consumes Phase 28's [`total_variation_depth_1d`] directly for the `tvd`/`mss` vectors — no
/// reimplementation. **Divergence:** `fdaoutlier` scales the stage-2 central region by
/// `n_orig / n_reduced`; fdars' `functional_boxplot` fixes the central region at the deepest 50%
/// of the reduced set, so `central_region_tvd` is informational only.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if the sample has fewer than 3 curves or zero columns.
#[must_use = "outlier detection results should not be discarded"]
pub fn tvdmss(data: &FdMatrix, config: TvdMssConfig) -> Result<TvdMssOutliers, FdarError> {
    let (n, m) = data.shape();
    if n < 3 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 curves and 1 column".to_string(),
            actual: format!("{n} rows, {m} columns"),
        });
    }

    let depth = total_variation_depth_1d(data, data)?;

    // Stage 1: shape outliers = low-MSS lower tail.
    let (lower_mss, _) = iqr_fence(&depth.mss, config.emp_factor_mss);
    let mean_mss = depth.mss.iter().sum::<f64>() / n as f64;
    let shape_outliers: Vec<usize> = (0..n)
        .filter(|&i| depth.mss[i] < lower_mss && depth.mss[i] < mean_mss)
        .collect();

    // Stage 2: magnitude outliers via functional boxplot on the non-shape curves.
    let keep: Vec<usize> = (0..n).filter(|i| !shape_outliers.contains(i)).collect();
    let mut magnitude_outliers = Vec::new();
    if keep.len() >= 3 {
        let kn = keep.len();
        let mut col_major = vec![0.0; kn * m];
        for (r, &orig) in keep.iter().enumerate() {
            for j in 0..m {
                col_major[r + j * kn] = data[(orig, j)];
            }
        }
        let reduced = FdMatrix::from_column_major(col_major, kn, m)?;
        let fbp = functional_boxplot(&reduced, DepthMethod::ModifiedBand, config.emp_factor_tvd)?;
        magnitude_outliers = fbp.outliers.iter().map(|&r| keep[r]).collect();
    }

    Ok(TvdMssOutliers {
        magnitude_outliers,
        shape_outliers,
        tvd: depth.tvd,
        mss: depth.mss,
    })
}

/// Configuration for [`muod`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MuodConfig {
    /// IQR multiplier for the per-index upper boxplot cutoff. Default `1.5`.
    pub factor: f64,
}

impl Default for MuodConfig {
    fn default() -> Self {
        Self { factor: 1.5 }
    }
}

/// Numeric output of the [`muod`] detector: per-curve indices and flagged sets (no rendering).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct MuodResult {
    /// Row indices flagged as shape outliers (upper fence on the shape index).
    pub shape_outliers: Vec<usize>,
    /// Row indices flagged as magnitude outliers (upper fence on the magnitude index).
    pub magnitude_outliers: Vec<usize>,
    /// Row indices flagged as amplitude outliers (upper fence on the amplitude index).
    pub amplitude_outliers: Vec<usize>,
    /// Shape index per curve: `|corr(X_i, μ) − 1|` (0 = same shape as the mean).
    pub shape_index: Vec<f64>,
    /// Magnitude index per curve: `|intercept_i|` (0 = same location as the mean).
    pub magnitude_index: Vec<f64>,
    /// Amplitude index per curve: `|slope_i − 1|` (0 = same amplitude as the mean).
    pub amplitude_index: Vec<f64>,
}

/// Per-curve MUOD indices `(shape, magnitude, amplitude)` from OLS regression on the pointwise mean.
fn muod_indices(data: &FdMatrix) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (n, m) = data.shape();

    // Pointwise mean μ_t and its moments.
    let mut mu = vec![0.0; m];
    for (j, mu_j) in mu.iter_mut().enumerate() {
        let mut s = 0.0;
        for i in 0..n {
            s += data[(i, j)];
        }
        *mu_j = s / n as f64;
    }
    let mu_mean = mu.iter().sum::<f64>() / m as f64;
    let mu_var = mu.iter().map(|&v| (v - mu_mean).powi(2)).sum::<f64>() / (m as f64 - 1.0);
    let mu_std = mu_var.sqrt();

    let triples: Vec<(f64, f64, f64)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let mut xi_mean = 0.0;
            for j in 0..m {
                xi_mean += data[(i, j)];
            }
            xi_mean /= m as f64;

            let mut cov = 0.0;
            let mut xi_var = 0.0;
            for j in 0..m {
                let dx = data[(i, j)] - xi_mean;
                let dmu = mu[j] - mu_mean;
                cov += dx * dmu;
                xi_var += dx * dx;
            }
            cov /= m as f64 - 1.0;
            xi_var /= m as f64 - 1.0;
            let xi_std = xi_var.sqrt();

            let slope = if mu_var < 1e-15 { 1.0 } else { cov / mu_var };
            let intercept = xi_mean - slope * mu_mean;
            let corr = if xi_std < 1e-15 || mu_std < 1e-15 {
                1.0
            } else {
                cov / (xi_std * mu_std)
            };

            ((corr - 1.0).abs(), intercept.abs(), (slope - 1.0).abs())
        })
        .collect();

    let mut shape = Vec::with_capacity(n);
    let mut magnitude = Vec::with_capacity(n);
    let mut amplitude = Vec::with_capacity(n);
    for (s, mg, a) in triples {
        shape.push(s);
        magnitude.push(mg);
        amplitude.push(a);
    }
    (shape, magnitude, amplitude)
}

/// Fast-MUOD (Massive Unsupervised Outlier Detection) via per-curve regression on the pointwise mean.
///
/// For each curve, regresses it against the sample's pointwise mean μ and forms three indices —
/// `shape = |corr(X_i, μ) − 1|`, `magnitude = |intercept_i|`, `amplitude = |slope_i − 1|` — then
/// flags outliers per index with the upper IQR boxplot whisker (`Q3 + factor·IQR`).
///
/// **Divergence:** this is the Fast-MUOD variant (regression against the pointwise mean), not
/// `fdaoutlier`'s pairwise C++ block; results match on the standard synthetic benchmarks. Only the
/// `boxplot` cutoff is implemented (the `tangent` cutoff is deferred to the backlog). Degenerate
/// variances (`< 1e-15`) fall back to a neutral index so a constant curve yields no NaN.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if the sample has fewer than 3 curves or fewer than 2 columns.
#[must_use = "outlier detection results should not be discarded"]
pub fn muod(data: &FdMatrix, config: MuodConfig) -> Result<MuodResult, FdarError> {
    let (n, m) = data.shape();
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 curves".to_string(),
            actual: format!("{n} rows"),
        });
    }
    if m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 columns".to_string(),
            actual: format!("{m} columns"),
        });
    }

    let (shape_index, magnitude_index, amplitude_index) = muod_indices(data);
    let flag_upper = |idx: &[f64]| -> Vec<usize> {
        let (_, upper) = iqr_fence(idx, config.factor);
        (0..n).filter(|&i| idx[i] > upper).collect()
    };
    let shape_outliers = flag_upper(&shape_index);
    let magnitude_outliers = flag_upper(&magnitude_index);
    let amplitude_outliers = flag_upper(&amplitude_index);

    Ok(MuodResult {
        shape_outliers,
        magnitude_outliers,
        amplitude_outliers,
        shape_index,
        magnitude_index,
        amplitude_index,
    })
}

/// A single step in a [`sequential_transform_outliers`] pipeline.
///
/// Transforms are applied **cumulatively** (each step feeds the next):
/// - `T0` — identity (raw data)
/// - `T1` — vertical centering: subtract each curve's mean
/// - `T2` — L2 normalization: divide each curve by its L2 norm
/// - `D1` — lag-1 first difference (`m → m−1` columns)
/// - `D2` — identical to `D1` (re-differences when applied after `D1`)
///
/// The multivariate outlyingness transform `O` is out of scope for this phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum SeqTransform {
    /// Identity (raw data).
    T0,
    /// Vertical centering: subtract each curve's mean.
    T1,
    /// L2 normalization: divide each curve by its L2 norm.
    T2,
    /// Lag-1 first difference.
    D1,
    /// Identical to [`SeqTransform::D1`].
    D2,
}

/// Configuration for [`sequential_transform_outliers`].
///
/// Not `serde`-serializable — it carries a [`DepthMethod`], which does not derive serde.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeqTransformConfig {
    /// Depth method for the per-step functional boxplot. Default [`DepthMethod::ModifiedBand`].
    pub depth_method: DepthMethod,
    /// Fence inflation factor for the per-step functional boxplot. Default `1.5`.
    pub emp_factor: f64,
}

impl Default for SeqTransformConfig {
    fn default() -> Self {
        Self {
            depth_method: DepthMethod::ModifiedBand,
            emp_factor: 1.5,
        }
    }
}

/// Numeric output of [`sequential_transform_outliers`] (no rendering).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct SeqTransformOutliers {
    /// Outlier row indices flagged after each transform step, in sequence order.
    pub per_transform_outliers: Vec<(SeqTransform, Vec<usize>)>,
    /// Union of all per-transform outlier indices (sorted, deduplicated).
    pub union_outliers: Vec<usize>,
}

/// Apply a single [`SeqTransform`] to a matrix, returning the transformed matrix.
fn seq_transform_apply(current: &FdMatrix, t: SeqTransform) -> Result<FdMatrix, FdarError> {
    let (n, m) = current.shape();
    match t {
        SeqTransform::T0 => Ok(current.clone()),
        SeqTransform::T1 => {
            let mut cm = vec![0.0; n * m];
            for i in 0..n {
                let mut mean = 0.0;
                for j in 0..m {
                    mean += current[(i, j)];
                }
                mean /= m as f64;
                for j in 0..m {
                    cm[i + j * n] = current[(i, j)] - mean;
                }
            }
            FdMatrix::from_column_major(cm, n, m)
        }
        SeqTransform::T2 => {
            let mut cm = vec![0.0; n * m];
            for i in 0..n {
                let mut norm = 0.0;
                for j in 0..m {
                    norm += current[(i, j)].powi(2);
                }
                let norm = norm.sqrt();
                if norm < 1e-15 {
                    return Err(FdarError::ComputationFailed {
                        operation: "T2 normalization",
                        detail: format!("zero-norm curve at row {i}"),
                    });
                }
                for j in 0..m {
                    cm[i + j * n] = current[(i, j)] / norm;
                }
            }
            FdMatrix::from_column_major(cm, n, m)
        }
        SeqTransform::D1 | SeqTransform::D2 => {
            if m < 2 {
                return Err(FdarError::InvalidDimension {
                    parameter: "data",
                    expected: "at least 2 columns for lag-1 differencing".to_string(),
                    actual: format!("{m} columns"),
                });
            }
            let m2 = m - 1;
            let mut cm = vec![0.0; n * m2];
            for i in 0..n {
                for k in 0..m2 {
                    cm[i + k * n] = current[(i, k + 1)] - current[(i, k)];
                }
            }
            FdMatrix::from_column_major(cm, n, m2)
        }
    }
}

/// Sequential-transformation outlier detection (Dai et al. 2020 / `fdaoutlier::seq_transform`).
///
/// Applies each transform in `sequence` **cumulatively** (every step's output feeds the next) and
/// runs a functional boxplot after each step, collecting the per-step outlier sets. The
/// `union_outliers` field (all indices flagged by at least one step) is an fdars convenience — the R
/// baseline returns only the per-transform sets.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if the sample has fewer than 2 curves or if a `D1`/`D2`
/// step is reached with fewer than 2 columns; returns [`FdarError::ComputationFailed`] if a `T2`
/// step encounters a zero-norm curve.
#[must_use = "outlier detection results should not be discarded"]
pub fn sequential_transform_outliers(
    data: &FdMatrix,
    sequence: &[SeqTransform],
    config: SeqTransformConfig,
) -> Result<SeqTransformOutliers, FdarError> {
    let n = data.nrows();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 curves".to_string(),
            actual: format!("{n} rows"),
        });
    }

    let mut current = data.clone();
    let mut per_transform_outliers = Vec::with_capacity(sequence.len());
    for &t in sequence {
        current = seq_transform_apply(&current, t)?;
        let fbp = functional_boxplot(&current, config.depth_method, config.emp_factor)?;
        per_transform_outliers.push((t, fbp.outliers));
    }

    let mut union_outliers: Vec<usize> = per_transform_outliers
        .iter()
        .flat_map(|(_, v)| v.iter().copied())
        .collect();
    union_outliers.sort_unstable();
    union_outliers.dedup();

    Ok(SeqTransformOutliers {
        per_transform_outliers,
        union_outliers,
    })
}

/// Configuration for [`depthgram`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DepthgramConfig {
    /// IQR factor for the parabola (shape) upper fence. Default `1.5`.
    pub outliergram_factor: f64,
    /// Inflation factor for the MBD magnitude functional boxplot. Default `1.5`.
    pub boxplot_factor: f64,
}

impl Default for DepthgramConfig {
    fn default() -> Self {
        Self {
            outliergram_factor: 1.5,
            boxplot_factor: 1.5,
        }
    }
}

/// Numeric output of the [`depthgram`] statistic (no rendering).
///
/// For univariate (p=1) functional data the three representations (dimension-wise `_d`, time-wise
/// `_t`, correlation-corrected `_t2`) are identical.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DepthgramResult {
    /// MBD of the MEI vector, dimension-wise.
    pub mbd_mei_d: Vec<f64>,
    /// MEI of the MBD vector, dimension-wise.
    pub mei_mbd_d: Vec<f64>,
    /// MBD of the MEI vector, time-wise (equals `_d` for p=1).
    pub mbd_mei_t: Vec<f64>,
    /// MEI of the MBD vector, time-wise (equals `_d` for p=1).
    pub mei_mbd_t: Vec<f64>,
    /// MBD of the MEI vector, correlation-corrected (equals `_d` for p=1).
    pub mbd_mei_t2: Vec<f64>,
    /// MEI of the MBD vector, correlation-corrected (equals `_d` for p=1).
    pub mei_mbd_t2: Vec<f64>,
    /// Row indices flagged as shape outliers (upper fence on the parabola deviation).
    pub shape_outliers: Vec<usize>,
    /// Row indices flagged as magnitude outliers (functional boxplot on MBD).
    pub magnitude_outliers: Vec<usize>,
    /// Modified band depth per curve.
    pub mbd: Vec<f64>,
    /// Modified epigraph index per curve.
    pub mei: Vec<f64>,
}

/// Depthgram statistic (roahd `depthGram`) — numeric coordinates + outlier flags (no rendering).
///
/// Computes the modified band depth (MBD) and modified epigraph index (MEI) of the sample, then the
/// `(MBD-of-MEI, MEI-of-MBD)` index pairs. Shape outliers are curves whose MBD falls above the
/// outliergram parabola's upper IQR fence; magnitude outliers come from a functional boxplot on the
/// MBD values.
///
/// **Divergence:** roahd's depthgram is defined for p-variate data and returns three distinct
/// representations. This implementation handles univariate (p=1) functional data only, where all
/// three representations are equivalent (the `_d`, `_t`, `_t2` fields are identical). At least ~4
/// curves are needed for a meaningful parabola IQR.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if the sample has fewer than 2 curves or zero columns.
#[must_use = "outlier detection results should not be discarded"]
pub fn depthgram(data: &FdMatrix, config: DepthgramConfig) -> Result<DepthgramResult, FdarError> {
    let (n, m) = data.shape();
    if n < 2 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 curves and 1 column".to_string(),
            actual: format!("{n} rows, {m} columns"),
        });
    }

    let mbd = modified_band_1d(data, data);
    let mei = modified_epigraph_index_1d(data, data);

    // (MBD of MEI, MEI of MBD) via n×1 matrix wrapping.
    let mei_mat = FdMatrix::from_column_major(mei.clone(), n, 1)?;
    let mbd_mat = FdMatrix::from_column_major(mbd.clone(), n, 1)?;
    let mbd_mei = modified_band_1d(&mei_mat, &mei_mat);
    let mei_mbd = modified_epigraph_index_1d(&mbd_mat, &mbd_mat);

    // Shape outliers: deviation above the outliergram parabola, upper IQR fence.
    let nf = n as f64;
    let a2 = -2.0 / (nf * (nf - 1.0));
    let a0 = a2;
    let a1 = 2.0 * (nf + 1.0) / (nf - 1.0);
    let dist: Vec<f64> = (0..n)
        .map(|i| (a0 + a1 * mei[i] + a2 * nf * nf * mei[i] * mei[i]) - mbd[i])
        .collect();
    let (_, upper) = iqr_fence(&dist, config.outliergram_factor);
    let shape_outliers: Vec<usize> = (0..n).filter(|&i| dist[i] > upper).collect();

    // Magnitude outliers: functional boxplot on the MBD values.
    let mbd_mat2 = FdMatrix::from_column_major(mbd.clone(), n, 1)?;
    let fbp = functional_boxplot(&mbd_mat2, DepthMethod::ModifiedBand, config.boxplot_factor)?;
    let magnitude_outliers = fbp.outliers;

    Ok(DepthgramResult {
        mbd_mei_d: mbd_mei.clone(),
        mei_mbd_d: mei_mbd.clone(),
        mbd_mei_t: mbd_mei.clone(),
        mei_mbd_t: mei_mbd.clone(),
        mbd_mei_t2: mbd_mei,
        mei_mbd_t2: mei_mbd,
        shape_outliers,
        magnitude_outliers,
        mbd,
        mei,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate homogeneous functional data
    fn generate_normal_fdata(n: usize, m: usize, seed: u64) -> FdMatrix {
        let mut rng = StdRng::seed_from_u64(seed);
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            let phase: f64 = rng.gen::<f64>() * 0.2;
            let amp: f64 = 1.0 + rng.gen::<f64>() * 0.1;
            for j in 0..m {
                let noise: f64 = rng.sample::<f64, _>(StandardNormal) * 0.05;
                data[(i, j)] = amp * (2.0 * PI * t[j] + phase).sin() + noise;
            }
        }
        data
    }

    /// Generate data with obvious outliers
    fn generate_data_with_outlier(n: usize, m: usize, n_outliers: usize) -> FdMatrix {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

        let mut data = FdMatrix::zeros(n, m);

        // Normal curves
        for i in 0..(n - n_outliers) {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin();
            }
        }

        // Outlier curves (shifted up by 10)
        for i in (n - n_outliers)..n {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin() + 10.0;
            }
        }

        data
    }

    // ============== Threshold tests ==============

    #[test]
    fn test_outliers_threshold_lrt_returns_positive() {
        let n = 20;
        let m = 30;
        let data = generate_normal_fdata(n, m, 42);

        let threshold = outliers_threshold_lrt(&data, 50, 0.1, 0.1, 42, 0.95);

        assert!(threshold > 0.0, "Threshold should be positive");
    }

    #[test]
    fn test_outliers_threshold_lrt_deterministic() {
        let n = 15;
        let m = 25;
        let data = generate_normal_fdata(n, m, 42);

        let t1 = outliers_threshold_lrt(&data, 30, 0.1, 0.1, 123, 0.95);
        let t2 = outliers_threshold_lrt(&data, 30, 0.1, 0.1, 123, 0.95);

        assert!(
            (t1 - t2).abs() < 1e-10,
            "Same seed should give same threshold"
        );
    }

    #[test]
    fn test_outliers_threshold_lrt_percentile_effect() {
        let n = 20;
        let m = 30;
        let data = generate_normal_fdata(n, m, 42);

        let t_low = outliers_threshold_lrt(&data, 50, 0.1, 0.1, 42, 0.50);
        let t_high = outliers_threshold_lrt(&data, 50, 0.1, 0.1, 42, 0.99);

        assert!(
            t_high >= t_low,
            "Higher percentile should give higher or equal threshold"
        );
    }

    #[test]
    fn test_outliers_threshold_lrt_invalid_input() {
        // Too few observations
        let data = FdMatrix::zeros(2, 30);
        let threshold = outliers_threshold_lrt(&data, 50, 0.1, 0.1, 42, 0.95);
        assert!(threshold.abs() < 1e-10, "Should return 0 for n < 3");

        // Empty m
        let data = FdMatrix::zeros(10, 0);
        let threshold = outliers_threshold_lrt(&data, 50, 0.1, 0.1, 42, 0.95);
        assert!(threshold.abs() < 1e-10);
    }

    // ============== Detection tests ==============

    #[test]
    fn test_detect_outliers_lrt_finds_obvious_outlier() {
        let n = 20;
        let m = 30;
        let data = generate_data_with_outlier(n, m, 1);

        // Use a reasonable threshold
        let outliers = detect_outliers_lrt(&data, 3.0, 0.1);

        assert_eq!(outliers.len(), n);

        // The last curve (outlier) should be detected
        assert!(outliers[n - 1], "Obvious outlier should be detected");

        // Most normal curves should not be outliers
        let n_detected: usize = outliers.iter().filter(|&&x| x).count();
        assert!(n_detected <= 3, "Should not detect too many outliers");
    }

    #[test]
    fn test_detect_outliers_lrt_homogeneous_data() {
        let n = 20;
        let m = 30;
        let data = generate_normal_fdata(n, m, 42);

        // With very high threshold, no outliers
        let outliers = detect_outliers_lrt(&data, 100.0, 0.1);

        let n_detected: usize = outliers.iter().filter(|&&x| x).count();
        assert_eq!(
            n_detected, 0,
            "Very high threshold should detect no outliers"
        );
    }

    #[test]
    fn test_detect_outliers_lrt_threshold_effect() {
        let n = 20;
        let m = 30;
        let data = generate_data_with_outlier(n, m, 3);

        let low_thresh = detect_outliers_lrt(&data, 2.0, 0.1);
        let high_thresh = detect_outliers_lrt(&data, 10.0, 0.1);

        let n_low: usize = low_thresh.iter().filter(|&&x| x).count();
        let n_high: usize = high_thresh.iter().filter(|&&x| x).count();

        assert!(
            n_low >= n_high,
            "Lower threshold should detect more or equal outliers"
        );
    }

    #[test]
    fn test_detect_outliers_lrt_invalid_input() {
        // Too few observations
        let data = FdMatrix::zeros(2, 30);
        let outliers = detect_outliers_lrt(&data, 3.0, 0.1);
        assert_eq!(outliers.len(), 2);
        assert!(
            outliers.iter().all(|&x| !x),
            "Should return all false for n < 3"
        );
    }

    #[test]
    fn test_identical_data_outliers() {
        let n = 10;
        let m = 20;
        let data = FdMatrix::from_column_major(vec![1.0; n * m], n, m).unwrap();
        let flags = detect_outliers_lrt(&data, 1.0, 0.15);
        assert_eq!(flags.len(), n);
        // All identical → no outliers
        for &f in &flags {
            assert!(!f);
        }
    }

    #[test]
    fn test_n3_minimal_outliers() {
        // Minimum viable: 3 curves
        let n = 3;
        let m = 10;
        let mut data_vec = vec![0.0; n * m];
        // Third curve is an outlier
        for j in 0..m {
            data_vec[j * n] = 0.0;
            data_vec[1 + j * n] = 0.1;
            data_vec[2 + j * n] = 100.0;
        }
        let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();
        let flags = detect_outliers_lrt(&data, 0.5, 0.15);
        assert_eq!(flags.len(), n);
    }

    // ============== With-distribution tests ==============

    #[test]
    fn test_with_dist_returns_sorted_distribution() {
        let data = generate_normal_fdata(20, 30, 42);
        let nb = 50;
        let (threshold, dist) = outliers_threshold_lrt_with_dist(&data, nb, 0.1, 0.1, 42, 0.95);

        assert_eq!(dist.len(), nb, "Distribution length should equal nb");
        for w in dist.windows(2) {
            assert!(w[0] <= w[1], "Distribution should be sorted");
        }
        let idx = ((nb as f64 * 0.95) as usize).min(nb - 1);
        assert!(
            (threshold - dist[idx]).abs() < 1e-10,
            "Threshold should match distribution at percentile index"
        );
    }

    #[test]
    fn test_with_dist_matches_scalar() {
        let data = generate_normal_fdata(15, 25, 99);
        let scalar = outliers_threshold_lrt(&data, 40, 0.1, 0.1, 123, 0.95);
        let (with_dist, _) = outliers_threshold_lrt_with_dist(&data, 40, 0.1, 0.1, 123, 0.95);
        assert!(
            (scalar - with_dist).abs() < 1e-10,
            "Scalar version should match with_dist version"
        );
    }

    #[test]
    fn test_bootstrap_dist_enables_pvalue() {
        let n = 20;
        let m = 30;
        let data = generate_data_with_outlier(n, m, 1);
        let trim = 0.1;

        let (_, dist) = outliers_threshold_lrt_with_dist(&data, 200, 0.1, trim, 42, 0.99);
        let nb = dist.len();

        // Compute per-curve distances
        let n_keep = ((1.0 - trim) * n as f64).ceil() as usize;
        let state = SortedReferenceState::from_reference(&data);
        let streaming_fm = StreamingFraimanMuniz::new(state, true);
        let depths = streaming_fm.depth_batch(&data);
        let (tmean, tvar) = compute_trimmed_stats(&data, &depths, n_keep);

        // p-value for the outlier curve (last one)
        let d_outlier = normalized_distance(&data, n - 1, &tmean, &tvar);
        let p_outlier =
            (dist.iter().filter(|&&v| v >= d_outlier).count() as f64 + 1.0) / (nb as f64 + 1.0);

        // p-value for a normal curve (first one)
        let d_normal = normalized_distance(&data, 0, &tmean, &tvar);
        let p_normal =
            (dist.iter().filter(|&&v| v >= d_normal).count() as f64 + 1.0) / (nb as f64 + 1.0);

        assert!(
            p_outlier < 0.05,
            "Outlier should have small p-value, got {p_outlier}"
        );
        assert!(
            p_normal > 0.05,
            "Normal curve should have large p-value, got {p_normal}"
        );
    }

    #[test]
    fn test_with_dist_invalid_input() {
        let data = FdMatrix::zeros(2, 30);
        let (threshold, dist) = outliers_threshold_lrt_with_dist(&data, 50, 0.1, 0.1, 42, 0.95);
        assert!(threshold.abs() < 1e-10);
        assert!(dist.is_empty(), "Should return empty dist for n < 3");
    }

    #[test]
    fn test_all_false_high_threshold() {
        let n = 10;
        let m = 20;
        let data_vec: Vec<f64> = (0..n * m).map(|i| (i as f64 * 0.1).sin()).collect();
        let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();
        // Very high threshold → no outliers
        let flags = detect_outliers_lrt(&data, 1e10, 0.15);
        for &f in &flags {
            assert!(!f, "High threshold should produce no outliers");
        }
    }

    // ============== n_keep clamping tests ==============

    #[test]
    fn test_trim_zero_no_trimming() {
        // trim=0 → n_keep=n, exercises the skip-partial-sort branch in compute_trimmed_stats
        let data = generate_normal_fdata(10, 20, 42);
        let threshold = outliers_threshold_lrt(&data, 30, 0.1, 0.0, 42, 0.95);
        assert!(threshold > 0.0);
        let flags = detect_outliers_lrt(&data, threshold, 0.0);
        assert_eq!(flags.len(), 10);
    }

    #[test]
    fn test_trim_near_one_heavy_trimming() {
        // trim=0.9 → n_keep=1, exercises minimal trim set (single deepest curve)
        let data = generate_normal_fdata(10, 20, 42);
        let threshold = outliers_threshold_lrt(&data, 30, 0.1, 0.9, 42, 0.95);
        assert!(threshold >= 0.0);
        let flags = detect_outliers_lrt(&data, threshold, 0.9);
        assert_eq!(flags.len(), 10);
    }

    #[test]
    fn test_trim_one_clamps_to_one() {
        // trim=1.0 → n_keep would be 0, must clamp to 1 (was a panic before fix)
        let data = generate_normal_fdata(10, 20, 42);
        let threshold = outliers_threshold_lrt(&data, 30, 0.1, 1.0, 42, 0.95);
        assert!(threshold >= 0.0);
        let flags = detect_outliers_lrt(&data, threshold, 1.0);
        assert_eq!(flags.len(), 10);
    }

    #[test]
    fn test_trim_negative_clamps_to_n() {
        // trim=-0.5 → n_keep would exceed n, must clamp to n (was a panic before fix)
        let data = generate_normal_fdata(10, 20, 42);
        let threshold = outliers_threshold_lrt(&data, 30, 0.1, -0.5, 42, 0.95);
        assert!(threshold > 0.0);
        let flags = detect_outliers_lrt(&data, threshold, -0.5);
        assert_eq!(flags.len(), 10);
    }

    // ============== Bootstrap parameter edge cases ==============

    #[test]
    fn test_smo_zero_no_noise() {
        // smo=0 → bootstrap resamples without smoothing noise
        let data = generate_normal_fdata(10, 20, 42);
        let (threshold, dist) = outliers_threshold_lrt_with_dist(&data, 30, 0.0, 0.1, 42, 0.95);
        assert!(threshold > 0.0);
        assert_eq!(dist.len(), 30);
    }

    #[test]
    fn test_nb_zero_empty_bootstrap() {
        let data = generate_normal_fdata(10, 20, 42);
        let (threshold, dist) = outliers_threshold_lrt_with_dist(&data, 0, 0.1, 0.1, 42, 0.95);
        assert!(threshold.abs() < 1e-10);
        assert!(dist.is_empty());
    }

    #[test]
    fn test_nb_one_single_bootstrap() {
        let data = generate_normal_fdata(10, 20, 42);
        let (threshold, dist) = outliers_threshold_lrt_with_dist(&data, 1, 0.1, 0.1, 42, 0.95);
        assert_eq!(dist.len(), 1);
        // With 1 iteration, threshold must equal the single value
        assert!((threshold - dist[0]).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_zero_returns_minimum() {
        let data = generate_normal_fdata(15, 20, 42);
        let nb = 50;
        let (_, dist) = outliers_threshold_lrt_with_dist(&data, nb, 0.1, 0.1, 42, 0.95);
        let t_zero = outliers_threshold_lrt(&data, nb, 0.1, 0.1, 42, 0.0);
        assert!(
            (t_zero - dist[0]).abs() < 1e-10,
            "percentile=0 should return the minimum of the distribution"
        );
    }

    #[test]
    fn test_percentile_one_returns_maximum() {
        let data = generate_normal_fdata(15, 20, 42);
        let nb = 50;
        let (_, dist) = outliers_threshold_lrt_with_dist(&data, nb, 0.1, 0.1, 42, 0.95);
        let t_one = outliers_threshold_lrt(&data, nb, 0.1, 0.1, 42, 1.0);
        assert!(
            (t_one - *dist.last().unwrap()).abs() < 1e-10,
            "percentile=1 should return the maximum of the distribution"
        );
    }

    // ============== Distribution invariant tests ==============

    #[test]
    fn test_distribution_values_non_negative() {
        let data = generate_normal_fdata(15, 20, 42);
        let (_, dist) = outliers_threshold_lrt_with_dist(&data, 50, 0.1, 0.1, 42, 0.95);
        for &v in &dist {
            assert!(v >= 0.0, "Max-distances must be non-negative, got {v}");
        }
    }

    // ============== detect_outliers_lrt edge cases ==============

    #[test]
    fn test_detect_m_zero_returns_all_false() {
        let data = FdMatrix::zeros(10, 0);
        let flags = detect_outliers_lrt(&data, 3.0, 0.1);
        assert_eq!(flags.len(), 10);
        assert!(flags.iter().all(|&f| !f));
    }

    #[test]
    fn test_detect_multiple_outliers() {
        let data = generate_data_with_outlier(20, 30, 3);
        let flags = detect_outliers_lrt(&data, 3.0, 0.1);
        // All three outlier curves (indices 17, 18, 19) should be detected
        let outlier_count = flags[17..20].iter().filter(|&&x| x).count();
        assert!(
            outlier_count >= 2,
            "At least 2 of 3 outliers should be detected, got {outlier_count}"
        );
    }

    // ============== End-to-end integration ==============

    #[test]
    fn test_end_to_end_threshold_then_detect() {
        let data = generate_data_with_outlier(20, 30, 2);
        let threshold = outliers_threshold_lrt(&data, 100, 0.1, 0.1, 42, 0.99);
        let flags = detect_outliers_lrt(&data, threshold, 0.1);

        // Outlier curves (last 2) should be flagged
        assert!(
            flags[18] || flags[19],
            "At least one outlier should be detected in end-to-end flow"
        );
        // Normal curves should mostly not be flagged
        let false_positives = flags[..18].iter().filter(|&&x| x).count();
        assert!(
            false_positives <= 2,
            "False positive count should be low, got {false_positives}"
        );
    }

    #[test]
    fn test_end_to_end_with_dist_pvalues_all_curves() {
        // Full pipeline: bootstrap dist → per-curve p-values → outlier classification
        let n = 25;
        let m = 30;
        let data = generate_data_with_outlier(n, m, 2);
        let trim = 0.1;

        let (_, dist) = outliers_threshold_lrt_with_dist(&data, 200, 0.1, trim, 42, 0.99);
        let nb = dist.len();

        let n_keep = ((1.0 - trim) * n as f64).ceil().max(1.0) as usize;
        let n_keep = n_keep.min(n);
        let state = SortedReferenceState::from_reference(&data);
        let streaming_fm = StreamingFraimanMuniz::new(state, true);
        let depths = streaming_fm.depth_batch(&data);
        let (tmean, tvar) = compute_trimmed_stats(&data, &depths, n_keep);

        // Compute p-values for all curves
        let pvalues: Vec<f64> = (0..n)
            .map(|i| {
                let d = normalized_distance(&data, i, &tmean, &tvar);
                (dist.iter().filter(|&&v| v >= d).count() as f64 + 1.0) / (nb as f64 + 1.0)
            })
            .collect();

        // Normal curves (0..23) should have large p-values
        let normal_small_p = pvalues[..23].iter().filter(|&&p| p < 0.01).count();
        assert_eq!(
            normal_small_p, 0,
            "Normal curves should not have tiny p-values"
        );

        // Outlier curves (23, 24) should have small p-values
        for &i in &[23, 24] {
            assert!(
                pvalues[i] < 0.05,
                "Outlier curve {i} should have small p-value, got {}",
                pvalues[i]
            );
        }
    }

    // ============== Outliergram tests ==============

    fn outliergram_test_data() -> FdMatrix {
        // 20 curves on 30 grid points, with 2 outliers
        let n = 20;
        let m = 30;
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut vals = vec![0.0; n * m];
        for i in 0..n {
            for (j, &tj) in t.iter().enumerate() {
                let base = tj.sin();
                vals[i + j * n] = if i < 18 {
                    base + 0.1 * (i as f64 * 0.5).sin()
                } else {
                    // Outliers: large deviation
                    base + 2.0 * (if i == 18 { 1.0 } else { -1.0 })
                };
            }
        }
        FdMatrix::from_column_major(vals, n, m).unwrap()
    }

    #[test]
    fn outliergram_runs() {
        let data = outliergram_test_data();
        let result = outliergram(&data, 1.5).unwrap();
        assert_eq!(result.mei.len(), 20);
        assert_eq!(result.mbd.len(), 20);
        assert_eq!(result.outlier_flags.len(), 20);
        // The two outlier curves should have lower MBD
        let central_mbd: f64 = result.mbd[..18].iter().sum::<f64>() / 18.0;
        assert!(result.mbd[18] < central_mbd || result.mbd[19] < central_mbd);
    }

    #[test]
    fn outliergram_parabola_coefficients() {
        let data = outliergram_test_data();
        let result = outliergram(&data, 1.5).unwrap();
        // Parabola should have a2 <= 0 (concave) for the theoretical relationship
        // (this is typical but not strictly required)
        assert!(result.a0.is_finite());
        assert!(result.a1.is_finite());
        assert!(result.a2.is_finite());
    }

    #[test]
    fn magnitude_shape_dimensions() {
        let data = outliergram_test_data();
        let result = magnitude_shape_outlyingness(&data).unwrap();
        assert_eq!(result.magnitude.len(), 20);
        assert_eq!(result.shape.len(), 20);
        // All values should be non-negative
        assert!(result.magnitude.iter().all(|&v| v >= 0.0));
        assert!(result.shape.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn magnitude_outliers_have_high_magnitude() {
        let data = outliergram_test_data();
        let result = magnitude_shape_outlyingness(&data).unwrap();
        let central_mag: f64 = result.magnitude[..18].iter().sum::<f64>() / 18.0;
        // At least one outlier should have higher magnitude
        assert!(result.magnitude[18] > central_mag || result.magnitude[19] > central_mag);
    }

    #[test]
    fn outliergram_too_few_curves() {
        let data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        assert!(outliergram(&data, 1.5).is_err());
    }

    // --- Phase 29: tvdmss + muod ---

    /// `n` sinusoids each with a genuine per-curve shape perturbation (so the derivative
    /// ranks that drive MSS are well-defined, not floating-point noise). `outlier_idx` is
    /// replaced per `kind`: "magnitude" = same shape +10 vertical shift, "amplitude" = ×5
    /// scale, "shape" = a genuinely different high-frequency wiggle, "constant" = a flat curve.
    fn outlier_sample(n: usize, m: usize, outlier_idx: usize, kind: &str) -> FdMatrix {
        // A normal inlier-shaped curve for row i: a primary sine plus a small curve-specific
        // secondary harmonic that gives each curve real (non-degenerate) derivative variation.
        let inlier_shape = |i: usize, x: f64| -> f64 {
            (x * PI).sin() + 0.1 * (x * 4.0 * PI + 0.3 * i as f64).sin()
        };
        let mut cm = vec![0.0; n * m];
        for i in 0..n {
            for t in 0..m {
                let x = t as f64 / (m as f64 - 1.0);
                let val = if i == outlier_idx {
                    match kind {
                        // Same shape family as its own inlier, only shifted / scaled.
                        "magnitude" => inlier_shape(i, x) + 10.0,
                        "amplitude" => 5.0 * inlier_shape(i, x),
                        // A genuinely different shape: centered on the sin trend (same
                        // vertical band as the inliers) but with a high-frequency wiggle
                        // that weaves through the bundle — low MBD, mid MEI, low MSS.
                        "shape" => (x * PI).sin() + 0.4 * (x * 12.0 * PI).sin(),
                        "constant" => 0.5,
                        _ => inlier_shape(i, x),
                    }
                } else {
                    inlier_shape(i, x)
                };
                cm[i + t * n] = val;
            }
        }
        FdMatrix::from_column_major(cm, n, m).unwrap()
    }

    #[test]
    fn tvdmss_flags_magnitude_outlier() {
        let idx = 4usize;
        let data = outlier_sample(12, 40, idx, "magnitude");
        let res = tvdmss(&data, TvdMssConfig::default()).unwrap();
        assert_eq!(res.tvd.len(), 12);
        assert_eq!(res.mss.len(), 12);
        assert!(
            res.magnitude_outliers.contains(&idx),
            "magnitude outlier {idx} not flagged: {:?}",
            res.magnitude_outliers
        );
    }

    #[test]
    fn tvdmss_flags_shape_outlier() {
        let idx = 7usize;
        let data = outlier_sample(12, 60, idx, "shape");
        let res = tvdmss(&data, TvdMssConfig::default()).unwrap();
        assert!(
            res.shape_outliers.contains(&idx),
            "shape outlier {idx} not flagged: {:?}",
            res.shape_outliers
        );
    }

    #[test]
    fn tvdmss_rejects_empty_and_too_few() {
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(matches!(
            tvdmss(&empty, TvdMssConfig::default()),
            Err(FdarError::InvalidDimension { .. })
        ));
        let two = outlier_sample(2, 8, 0, "none");
        assert!(matches!(
            tvdmss(&two, TvdMssConfig::default()),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn muod_flags_magnitude_amplitude_shape() {
        let mag = outlier_sample(12, 40, 3, "magnitude");
        let r = muod(&mag, MuodConfig::default()).unwrap();
        assert_eq!(r.shape_index.len(), 12);
        assert!(
            r.magnitude_outliers.contains(&3),
            "magnitude: {:?}",
            r.magnitude_outliers
        );

        let amp = outlier_sample(12, 40, 5, "amplitude");
        let r = muod(&amp, MuodConfig::default()).unwrap();
        assert!(
            r.amplitude_outliers.contains(&5),
            "amplitude: {:?}",
            r.amplitude_outliers
        );

        let shp = outlier_sample(12, 60, 8, "shape");
        let r = muod(&shp, MuodConfig::default()).unwrap();
        assert!(
            r.shape_outliers.contains(&8),
            "shape: {:?}",
            r.shape_outliers
        );
    }

    #[test]
    fn muod_constant_curve_no_nan() {
        let data = outlier_sample(12, 40, 6, "constant");
        let r = muod(&data, MuodConfig::default()).unwrap();
        for v in r
            .shape_index
            .iter()
            .chain(&r.magnitude_index)
            .chain(&r.amplitude_index)
        {
            assert!(!v.is_nan(), "index produced NaN");
        }
    }

    #[test]
    fn muod_rejects_bad_dims() {
        let two = outlier_sample(2, 8, 0, "none");
        assert!(matches!(
            muod(&two, MuodConfig::default()),
            Err(FdarError::InvalidDimension { .. })
        ));
        let one_col = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 3, 1).unwrap();
        assert!(matches!(
            muod(&one_col, MuodConfig::default()),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    // --- Phase 29 Plan 02: sequential_transform_outliers + depthgram ---

    #[test]
    fn seq_transform_default_sequence_flags_outlier_and_union_is_flatten() {
        let idx = 4usize;
        let data = outlier_sample(12, 40, idx, "magnitude");
        let seq = [SeqTransform::T0, SeqTransform::T1, SeqTransform::D1];
        let res =
            sequential_transform_outliers(&data, &seq, SeqTransformConfig::default()).unwrap();

        assert_eq!(res.per_transform_outliers.len(), 3);
        assert!(
            res.per_transform_outliers
                .iter()
                .any(|(_, v)| !v.is_empty()),
            "no transform flagged anything"
        );
        assert!(
            res.union_outliers.contains(&idx),
            "union {:?} missing outlier {idx}",
            res.union_outliers
        );
        // union == sorted, deduped flatten of the per-transform sets.
        let mut expected: Vec<usize> = res
            .per_transform_outliers
            .iter()
            .flat_map(|(_, v)| v.iter().copied())
            .collect();
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(res.union_outliers, expected);
    }

    #[test]
    fn seq_transform_error_paths() {
        // D1 with a single column.
        let one_col = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 3, 1).unwrap();
        assert!(matches!(
            sequential_transform_outliers(
                &one_col,
                &[SeqTransform::D1],
                SeqTransformConfig::default()
            ),
            Err(FdarError::InvalidDimension { .. })
        ));
        // T2 with an all-zero curve → ComputationFailed.
        let mut cm = vec![0.0; 3 * 4];
        for i in [0usize, 2] {
            for t in 0..4 {
                cm[i + t * 3] = 1.0 + i as f64 + t as f64;
            }
        }
        // row 1 stays all-zero
        let zero_row = FdMatrix::from_column_major(cm, 3, 4).unwrap();
        assert!(matches!(
            sequential_transform_outliers(
                &zero_row,
                &[SeqTransform::T2],
                SeqTransformConfig::default()
            ),
            Err(FdarError::ComputationFailed { .. })
        ));
        // n == 1.
        let one_curve = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
        assert!(matches!(
            sequential_transform_outliers(
                &one_curve,
                &[SeqTransform::T0],
                SeqTransformConfig::default()
            ),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn depthgram_flags_magnitude_and_shape() {
        let mag = outlier_sample(12, 40, 3, "magnitude");
        let r = depthgram(&mag, DepthgramConfig::default()).unwrap();
        assert_eq!(r.mbd.len(), 12);
        assert_eq!(r.mei.len(), 12);
        assert_eq!(r.mbd_mei_d.len(), 12);
        assert!(
            r.magnitude_outliers.contains(&3),
            "magnitude: {:?}",
            r.magnitude_outliers
        );

        let shp = outlier_sample(14, 60, 9, "shape");
        let r = depthgram(&shp, DepthgramConfig::default()).unwrap();
        assert!(
            r.shape_outliers.contains(&9),
            "shape: {:?}",
            r.shape_outliers
        );
    }

    #[test]
    fn depthgram_p1_representations_equivalent() {
        let data = outlier_sample(10, 30, 2, "magnitude");
        let r = depthgram(&data, DepthgramConfig::default()).unwrap();
        assert_eq!(r.mbd_mei_d, r.mbd_mei_t);
        assert_eq!(r.mbd_mei_d, r.mbd_mei_t2);
        assert_eq!(r.mei_mbd_d, r.mei_mbd_t);
        assert_eq!(r.mei_mbd_d, r.mei_mbd_t2);
    }

    #[test]
    fn depthgram_rejects_bad_dims() {
        let one = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
        assert!(matches!(
            depthgram(&one, DepthgramConfig::default()),
            Err(FdarError::InvalidDimension { .. })
        ));
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(matches!(
            depthgram(&empty, DepthgramConfig::default()),
            Err(FdarError::InvalidDimension { .. })
        ));
    }
}
