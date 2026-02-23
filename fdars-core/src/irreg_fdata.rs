//! Irregular functional data operations.
//!
//! This module provides data structures and algorithms for functional data
//! where observations have different evaluation points (irregular/sparse sampling).
//!
//! ## Storage Format
//!
//! Uses a CSR-like (Compressed Sparse Row) format for efficient storage:
//! - `offsets[i]..offsets[i+1]` gives the slice indices for observation i
//! - `argvals` and `values` store all data contiguously
//!
//! This format is memory-efficient and enables parallel processing of observations.
//!
//! ## Example
//!
//! For 3 curves with varying numbers of observation points:
//! - Curve 0: 5 points
//! - Curve 1: 3 points
//! - Curve 2: 7 points
//!
//! The offsets would be: [0, 5, 8, 15]

use crate::matrix::FdMatrix;
use crate::{iter_maybe_parallel, slice_maybe_parallel};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Kernel function type for smoothing operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelType {
    /// Epanechnikov kernel: K(u) = 0.75(1 - u²) for |u| ≤ 1
    Epanechnikov,
    /// Gaussian kernel: K(u) = exp(-u²/2) / √(2π)
    Gaussian,
}

/// Compressed storage for irregular functional data.
///
/// Uses CSR-style layout where each observation can have a different
/// number of evaluation points.
#[derive(Clone, Debug)]
pub struct IrregFdata {
    /// Start indices for each observation (length n+1)
    /// `offsets[i]..offsets[i+1]` gives the range for observation i
    pub offsets: Vec<usize>,
    /// All observation points concatenated
    pub argvals: Vec<f64>,
    /// All values concatenated
    pub values: Vec<f64>,
    /// Domain range `[min, max]`
    pub rangeval: [f64; 2],
}

impl IrregFdata {
    /// Create from lists of argvals and values (one per observation).
    ///
    /// # Arguments
    /// * `argvals_list` - List of observation point vectors
    /// * `values_list` - List of value vectors (same lengths as argvals_list)
    ///
    /// # Panics
    /// Panics if the lists have different lengths or if any pair has mismatched lengths.
    pub fn from_lists(argvals_list: &[Vec<f64>], values_list: &[Vec<f64>]) -> Self {
        let n = argvals_list.len();
        assert_eq!(
            n,
            values_list.len(),
            "argvals_list and values_list must have same length"
        );

        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);

        let total_points: usize = argvals_list.iter().map(|v| v.len()).sum();
        let mut argvals = Vec::with_capacity(total_points);
        let mut values = Vec::with_capacity(total_points);

        let mut range_min = f64::INFINITY;
        let mut range_max = f64::NEG_INFINITY;

        for i in 0..n {
            assert_eq!(
                argvals_list[i].len(),
                values_list[i].len(),
                "Observation {} has mismatched argvals/values lengths",
                i
            );

            argvals.extend_from_slice(&argvals_list[i]);
            values.extend_from_slice(&values_list[i]);
            offsets.push(argvals.len());

            if let (Some(&min), Some(&max)) = (argvals_list[i].first(), argvals_list[i].last()) {
                range_min = range_min.min(min);
                range_max = range_max.max(max);
            }
        }

        IrregFdata {
            offsets,
            argvals,
            values,
            rangeval: [range_min, range_max],
        }
    }

    /// Create from flattened representation (for R interop).
    ///
    /// Returns `None` if offsets are empty, argvals/values lengths differ,
    /// the last offset doesn't match argvals length, or offsets are non-monotonic.
    ///
    /// # Arguments
    /// * `offsets` - Start indices (length n+1)
    /// * `argvals` - All observation points concatenated
    /// * `values` - All values concatenated
    /// * `rangeval` - Domain range `[min, max]`
    pub fn from_flat(
        offsets: Vec<usize>,
        argvals: Vec<f64>,
        values: Vec<f64>,
        rangeval: [f64; 2],
    ) -> Option<Self> {
        if offsets.is_empty()
            || argvals.len() != values.len()
            || *offsets.last()? != argvals.len()
            || offsets.windows(2).any(|w| w[0] > w[1])
        {
            return None;
        }
        Some(IrregFdata {
            offsets,
            argvals,
            values,
            rangeval,
        })
    }

    /// Number of observations.
    #[inline]
    pub fn n_obs(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Number of points for observation i.
    #[inline]
    pub fn n_points(&self, i: usize) -> usize {
        self.offsets[i + 1] - self.offsets[i]
    }

    /// Get observation i as a pair of slices (argvals, values).
    #[inline]
    pub fn get_obs(&self, i: usize) -> (&[f64], &[f64]) {
        let start = self.offsets[i];
        let end = self.offsets[i + 1];
        (&self.argvals[start..end], &self.values[start..end])
    }

    /// Total number of observation points across all curves.
    #[inline]
    pub fn total_points(&self) -> usize {
        self.argvals.len()
    }

    /// Get observation counts for all curves.
    pub fn obs_counts(&self) -> Vec<usize> {
        (0..self.n_obs()).map(|i| self.n_points(i)).collect()
    }

    /// Get minimum number of observations per curve.
    pub fn min_obs(&self) -> usize {
        (0..self.n_obs())
            .map(|i| self.n_points(i))
            .min()
            .unwrap_or(0)
    }

    /// Get maximum number of observations per curve.
    pub fn max_obs(&self) -> usize {
        (0..self.n_obs())
            .map(|i| self.n_points(i))
            .max()
            .unwrap_or(0)
    }
}

// =============================================================================
// Integration
// =============================================================================

/// Compute integral of each curve using trapezoidal rule.
///
/// For curve i with observation points t_1, ..., t_m and values x_1, ..., x_m:
/// ∫f_i(t)dt ≈ Σ_{j=1}^{m-1} (t_{j+1} - t_j) * (x_{j+1} + x_j) / 2
///
/// # Returns
/// Vector of integrals, one per curve
pub fn integrate_irreg(ifd: &IrregFdata) -> Vec<f64> {
    let n = ifd.n_obs();

    iter_maybe_parallel!(0..n)
        .map(|i| {
            let (t, x) = ifd.get_obs(i);

            if t.len() < 2 {
                return 0.0;
            }

            let mut integral = 0.0;
            for j in 1..t.len() {
                let h = t[j] - t[j - 1];
                integral += 0.5 * h * (x[j] + x[j - 1]);
            }
            integral
        })
        .collect()
}

// =============================================================================
// Norms
// =============================================================================

/// Compute Lp norm for each curve in irregular functional data.
///
/// ||f_i||_p = (∫|f_i(t)|^p dt)^{1/p}
///
/// Uses trapezoidal rule for integration.
///
/// # Arguments
/// * `ifd` - Irregular functional data
/// * `p` - Norm order (p >= 1)
///
/// # Returns
/// Vector of norms, one per curve
pub fn norm_lp_irreg(ifd: &IrregFdata, p: f64) -> Vec<f64> {
    let n = ifd.n_obs();

    iter_maybe_parallel!(0..n)
        .map(|i| {
            let (t, x) = ifd.get_obs(i);

            if t.len() < 2 {
                return 0.0;
            }

            let mut integral = 0.0;
            for j in 1..t.len() {
                let h = t[j] - t[j - 1];
                let val_left = x[j - 1].abs().powf(p);
                let val_right = x[j].abs().powf(p);
                integral += 0.5 * h * (val_left + val_right);
            }
            integral.powf(1.0 / p)
        })
        .collect()
}

// =============================================================================
// Mean Estimation
// =============================================================================

/// Epanechnikov kernel function.
#[inline]
fn kernel_epanechnikov(u: f64) -> f64 {
    if u.abs() <= 1.0 {
        0.75 * (1.0 - u * u)
    } else {
        0.0
    }
}

/// Gaussian kernel function.
#[inline]
fn kernel_gaussian(u: f64) -> f64 {
    (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

impl KernelType {
    #[inline]
    fn as_fn(self) -> fn(f64) -> f64 {
        match self {
            KernelType::Epanechnikov => kernel_epanechnikov,
            KernelType::Gaussian => kernel_gaussian,
        }
    }
}

/// Estimate mean function at specified target points using kernel smoothing.
///
/// Uses local weighted averaging (Nadaraya-Watson estimator) at each target point:
/// μ̂(t) = Σ_{i,j} K_h(t - t_{ij}) x_{ij} / Σ_{i,j} K_h(t - t_{ij})
///
/// # Arguments
/// * `ifd` - Irregular functional data
/// * `target_argvals` - Points at which to estimate the mean
/// * `bandwidth` - Kernel bandwidth
/// * `kernel_type` - Kernel function to use
///
/// # Returns
/// Estimated mean function values at target points
pub fn mean_irreg(
    ifd: &IrregFdata,
    target_argvals: &[f64],
    bandwidth: f64,
    kernel_type: KernelType,
) -> Vec<f64> {
    let n = ifd.n_obs();
    let kernel = kernel_type.as_fn();

    slice_maybe_parallel!(target_argvals)
        .map(|&t| {
            let mut sum_weights = 0.0;
            let mut sum_values = 0.0;

            for i in 0..n {
                let (obs_t, obs_x) = ifd.get_obs(i);

                for (&ti, &xi) in obs_t.iter().zip(obs_x.iter()) {
                    let u = (ti - t) / bandwidth;
                    let w = kernel(u);
                    sum_weights += w;
                    sum_values += w * xi;
                }
            }

            if sum_weights > 0.0 {
                sum_values / sum_weights
            } else {
                f64::NAN
            }
        })
        .collect()
}

// =============================================================================
// Covariance Estimation
// =============================================================================

/// Estimate covariance at a grid of points using local linear smoothing.
///
/// # Arguments
/// * `ifd` - Irregular functional data
/// * `s_grid` - First grid points for covariance
/// * `t_grid` - Second grid points for covariance
/// * `bandwidth` - Kernel bandwidth
///
/// # Returns
/// Covariance matrix estimate at (s_grid, t_grid) points as `FdMatrix`
pub fn cov_irreg(ifd: &IrregFdata, s_grid: &[f64], t_grid: &[f64], bandwidth: f64) -> FdMatrix {
    let n = ifd.n_obs();
    let ns = s_grid.len();
    let nt = t_grid.len();

    // First estimate mean at all observation points
    let mean_at_obs = mean_irreg(ifd, &ifd.argvals, bandwidth, KernelType::Gaussian);

    // Centered values
    let centered: Vec<f64> = ifd
        .values
        .iter()
        .zip(mean_at_obs.iter())
        .map(|(&v, &m)| v - m)
        .collect();

    // Estimate covariance at each (s, t) pair
    let mut cov = vec![0.0; ns * nt];

    for (si, &s) in s_grid.iter().enumerate() {
        for (ti, &t) in t_grid.iter().enumerate() {
            cov[si + ti * ns] =
                accumulate_cov_at_point(&ifd.offsets, &ifd.argvals, &centered, n, s, t, bandwidth);
        }
    }

    FdMatrix::from_column_major(cov, ns, nt).unwrap()
}

/// Compute kernel-weighted covariance estimate at a single (s, t) grid point.
fn accumulate_cov_at_point(
    offsets: &[usize],
    obs_times: &[f64],
    centered: &[f64],
    n: usize,
    s: f64,
    t: f64,
    bandwidth: f64,
) -> f64 {
    let mut sum_weights = 0.0;
    let mut sum_products = 0.0;

    for i in 0..n {
        let start = offsets[i];
        let end = offsets[i + 1];
        let obs_t = &obs_times[start..end];
        let obs_c = &centered[start..end];

        for j1 in 0..obs_t.len() {
            for j2 in 0..obs_t.len() {
                let w1 = kernel_gaussian((obs_t[j1] - s) / bandwidth);
                let w2 = kernel_gaussian((obs_t[j2] - t) / bandwidth);
                let w = w1 * w2;

                sum_weights += w;
                sum_products += w * obs_c[j1] * obs_c[j2];
            }
        }
    }

    if sum_weights > 0.0 {
        sum_products / sum_weights
    } else {
        0.0
    }
}

// =============================================================================
// Distance Metrics
// =============================================================================

/// Compute Lp distance between two irregular curves.
///
/// Uses the union of observation points and linear interpolation.
fn lp_distance_pair(t1: &[f64], x1: &[f64], t2: &[f64], x2: &[f64], p: f64) -> f64 {
    // Create union of time points
    let mut all_t: Vec<f64> = t1.iter().chain(t2.iter()).copied().collect();
    all_t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_t.dedup();

    // Filter to common range
    let (t_min, t_max) = match (t1.first(), t1.last(), t2.first(), t2.last()) {
        (Some(&a), Some(&b), Some(&c), Some(&d)) => (a.max(c), b.min(d)),
        _ => return f64::NAN,
    };

    let common_t: Vec<f64> = all_t
        .into_iter()
        .filter(|&t| t >= t_min && t <= t_max)
        .collect();

    if common_t.len() < 2 {
        return f64::NAN;
    }

    // Interpolate both curves at common points
    let y1: Vec<f64> = common_t.iter().map(|&t| linear_interp(t1, x1, t)).collect();
    let y2: Vec<f64> = common_t.iter().map(|&t| linear_interp(t2, x2, t)).collect();

    // Compute integral of |y1 - y2|^p
    let mut integral = 0.0;
    for j in 1..common_t.len() {
        let h = common_t[j] - common_t[j - 1];
        let val_left = (y1[j - 1] - y2[j - 1]).abs().powf(p);
        let val_right = (y1[j] - y2[j]).abs().powf(p);
        integral += 0.5 * h * (val_left + val_right);
    }

    integral.powf(1.0 / p)
}

/// Linear interpolation at point t.
fn linear_interp(argvals: &[f64], values: &[f64], t: f64) -> f64 {
    if t <= argvals[0] {
        return values[0];
    }
    if t >= argvals[argvals.len() - 1] {
        return values[values.len() - 1];
    }

    // Find the interval
    let idx = argvals.iter().position(|&x| x > t).unwrap();
    let t0 = argvals[idx - 1];
    let t1 = argvals[idx];
    let x0 = values[idx - 1];
    let x1 = values[idx];

    // Linear interpolation
    x0 + (x1 - x0) * (t - t0) / (t1 - t0)
}

/// Compute pairwise Lp distances for irregular functional data.
///
/// # Arguments
/// * `ifd` - Irregular functional data
/// * `p` - Norm order
///
/// # Returns
/// Distance matrix (n × n) as `FdMatrix`
pub fn metric_lp_irreg(ifd: &IrregFdata, p: f64) -> FdMatrix {
    let n = ifd.n_obs();
    let mut dist = vec![0.0; n * n];

    // Compute upper triangle in parallel
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .collect();

    let distances: Vec<f64> = slice_maybe_parallel!(pairs)
        .map(|&(i, j)| {
            let (t_i, x_i) = ifd.get_obs(i);
            let (t_j, x_j) = ifd.get_obs(j);
            lp_distance_pair(t_i, x_i, t_j, x_j, p)
        })
        .collect();

    // Fill symmetric matrix
    for (k, &(i, j)) in pairs.iter().enumerate() {
        dist[i + j * n] = distances[k];
        dist[j + i * n] = distances[k];
    }

    FdMatrix::from_column_major(dist, n, n).unwrap()
}

// =============================================================================
// Conversion to Regular Grid
// =============================================================================

/// Convert irregular data to regular grid via linear interpolation.
///
/// Missing values (outside observation range) are marked as NaN.
///
/// # Arguments
/// * `ifd` - Irregular functional data
/// * `target_grid` - Regular grid to interpolate to
///
/// # Returns
/// Data matrix (n × len(target_grid)) as `FdMatrix`
pub fn to_regular_grid(ifd: &IrregFdata, target_grid: &[f64]) -> FdMatrix {
    let n = ifd.n_obs();
    let m = target_grid.len();

    let result = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            let (obs_t, obs_x) = ifd.get_obs(i);

            target_grid
                .iter()
                .map(|&t| {
                    if obs_t.is_empty() || t < obs_t[0] || t > obs_t[obs_t.len() - 1] {
                        f64::NAN
                    } else {
                        linear_interp(obs_t, obs_x, t)
                    }
                })
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<f64>>()
        // Transpose to column-major
        .chunks(m)
        .enumerate()
        .fold(vec![0.0; n * m], |mut acc, (i, row)| {
            for (j, &val) in row.iter().enumerate() {
                acc[i + j * n] = val;
            }
            acc
        });

    FdMatrix::from_column_major(result, n, m).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ifd(offsets: Vec<usize>, argvals: Vec<f64>, values: Vec<f64>) -> IrregFdata {
        let range_min = argvals.iter().cloned().fold(f64::INFINITY, f64::min);
        let range_max = argvals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        IrregFdata::from_flat(offsets, argvals, values, [range_min, range_max]).unwrap()
    }

    #[test]
    fn test_from_lists() {
        let argvals_list = vec![vec![0.0, 0.5, 1.0], vec![0.0, 1.0]];
        let values_list = vec![vec![1.0, 2.0, 3.0], vec![1.0, 3.0]];

        let ifd = IrregFdata::from_lists(&argvals_list, &values_list);

        assert_eq!(ifd.n_obs(), 2);
        assert_eq!(ifd.n_points(0), 3);
        assert_eq!(ifd.n_points(1), 2);
        assert_eq!(ifd.total_points(), 5);
    }

    #[test]
    fn test_get_obs() {
        let argvals_list = vec![vec![0.0, 0.5, 1.0], vec![0.0, 1.0]];
        let values_list = vec![vec![1.0, 2.0, 3.0], vec![1.0, 3.0]];

        let ifd = IrregFdata::from_lists(&argvals_list, &values_list);

        let (t0, x0) = ifd.get_obs(0);
        assert_eq!(t0, &[0.0, 0.5, 1.0]);
        assert_eq!(x0, &[1.0, 2.0, 3.0]);

        let (t1, x1) = ifd.get_obs(1);
        assert_eq!(t1, &[0.0, 1.0]);
        assert_eq!(x1, &[1.0, 3.0]);
    }

    #[test]
    fn test_integrate_irreg() {
        // Integrate constant function = 1 over [0, 1]
        let ifd = make_ifd(
            vec![0, 3, 6],
            vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        );

        let integrals = integrate_irreg(&ifd);

        assert!((integrals[0] - 1.0).abs() < 1e-10);
        assert!((integrals[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_lp_irreg() {
        // L2 norm of constant function = c is c (on \[0,1\])
        let ifd = make_ifd(vec![0, 3], vec![0.0, 0.5, 1.0], vec![2.0, 2.0, 2.0]);

        let norms = norm_lp_irreg(&ifd, 2.0);

        assert!((norms[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_interp() {
        let t = vec![0.0, 1.0, 2.0];
        let x = vec![0.0, 2.0, 4.0];

        assert!((linear_interp(&t, &x, 0.5) - 1.0).abs() < 1e-10);
        assert!((linear_interp(&t, &x, 1.5) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_irreg() {
        // Two identical curves should give exact mean
        let ifd = make_ifd(
            vec![0, 3, 6],
            vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        );

        let target = vec![0.0, 0.5, 1.0];
        let mean = mean_irreg(&ifd, &target, 0.5, KernelType::Gaussian);

        // Mean should be close to the common values
        assert!((mean[1] - 1.0).abs() < 0.3);
    }

    // ========================================================================
    // Tests for from_flat and accessors
    // ========================================================================

    #[test]
    fn test_from_flat() {
        let offsets = vec![0, 3, 5, 10];
        let argvals = vec![0.0, 0.5, 1.0, 0.0, 1.0, 0.0, 0.2, 0.4, 0.6, 0.8];
        let values = vec![1.0, 2.0, 3.0, 1.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let rangeval = [0.0, 1.0];

        let ifd = IrregFdata::from_flat(offsets.clone(), argvals.clone(), values.clone(), rangeval)
            .unwrap();

        assert_eq!(ifd.n_obs(), 3);
        assert_eq!(ifd.offsets, offsets);
        assert_eq!(ifd.argvals, argvals);
        assert_eq!(ifd.values, values);
        assert_eq!(ifd.rangeval, rangeval);
    }

    #[test]
    fn test_from_flat_invalid() {
        // Empty offsets
        assert!(IrregFdata::from_flat(vec![], vec![], vec![], [0.0, 1.0]).is_none());
        // Mismatched argvals/values lengths
        assert!(IrregFdata::from_flat(vec![0, 2], vec![0.0, 1.0], vec![1.0], [0.0, 1.0]).is_none());
        // Last offset doesn't match argvals length
        assert!(
            IrregFdata::from_flat(vec![0, 5], vec![0.0, 1.0], vec![1.0, 2.0], [0.0, 1.0]).is_none()
        );
        // Non-monotonic offsets
        assert!(IrregFdata::from_flat(
            vec![0, 3, 1],
            vec![0.0, 1.0, 2.0],
            vec![1.0, 2.0, 3.0],
            [0.0, 2.0]
        )
        .is_none());
    }

    #[test]
    fn test_accessors_n_obs_n_points_total() {
        let argvals_list = vec![
            vec![0.0, 0.5, 1.0],             // 3 points
            vec![0.0, 1.0],                  // 2 points
            vec![0.0, 0.25, 0.5, 0.75, 1.0], // 5 points
        ];
        let values_list = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 3.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
        ];

        let ifd = IrregFdata::from_lists(&argvals_list, &values_list);

        // Test n_obs
        assert_eq!(ifd.n_obs(), 3);

        // Test n_points for each curve
        assert_eq!(ifd.n_points(0), 3);
        assert_eq!(ifd.n_points(1), 2);
        assert_eq!(ifd.n_points(2), 5);

        // Test total_points
        assert_eq!(ifd.total_points(), 10);
    }

    #[test]
    fn test_obs_counts() {
        let argvals_list = vec![
            vec![0.0, 0.5, 1.0],             // 3 points
            vec![0.0, 1.0],                  // 2 points
            vec![0.0, 0.25, 0.5, 0.75, 1.0], // 5 points
        ];
        let values_list = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 3.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
        ];

        let ifd = IrregFdata::from_lists(&argvals_list, &values_list);
        let counts = ifd.obs_counts();

        assert_eq!(counts, vec![3, 2, 5]);
    }

    #[test]
    fn test_min_max_obs() {
        let argvals_list = vec![
            vec![0.0, 0.5, 1.0],             // 3 points
            vec![0.0, 1.0],                  // 2 points
            vec![0.0, 0.25, 0.5, 0.75, 1.0], // 5 points
        ];
        let values_list = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 3.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
        ];

        let ifd = IrregFdata::from_lists(&argvals_list, &values_list);

        assert_eq!(ifd.min_obs(), 2);
        assert_eq!(ifd.max_obs(), 5);
    }

    #[test]
    fn test_min_max_obs_empty() {
        let ifd = IrregFdata::from_lists(&[], &[]);
        assert_eq!(ifd.min_obs(), 0);
        assert_eq!(ifd.max_obs(), 0);
    }

    // ========================================================================
    // Tests for cov_irreg
    // ========================================================================

    #[test]
    fn test_cov_irreg_identical_curves() {
        // Two identical curves should have zero covariance (no variability)
        let ifd = make_ifd(
            vec![0, 5, 10],
            vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0],
            vec![1.0, 2.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 2.0, 1.0],
        );

        let grid = vec![0.25, 0.5, 0.75];
        let cov = cov_irreg(&ifd, &grid, &grid, 0.3);

        // Covariance should be close to 0 (identical curves)
        assert_eq!(cov.nrows(), 3);
        assert_eq!(cov.ncols(), 3);
        // Diagonal should be variance (close to 0 for identical curves)
        for i in 0..3 {
            assert!(
                cov[(i, i)].abs() < 0.5,
                "Diagonal cov[{},{}] = {} should be near 0",
                i,
                i,
                cov[(i, i)]
            );
        }
    }

    #[test]
    fn test_cov_irreg_symmetry() {
        // Covariance matrix should be symmetric
        let ifd = make_ifd(
            vec![0, 5, 10, 15],
            vec![
                0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0,
            ],
            vec![
                1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 2.0, 3.0, 2.0, 3.0, 2.0,
            ],
        );

        let grid = vec![0.25, 0.5, 0.75];
        let cov = cov_irreg(&ifd, &grid, &grid, 0.3);

        // Check symmetry: cov[i,j] = cov[j,i]
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (cov[(i, j)] - cov[(j, i)]).abs() < 1e-10,
                    "Cov[{},{}] = {} != Cov[{},{}] = {}",
                    i,
                    j,
                    cov[(i, j)],
                    j,
                    i,
                    cov[(j, i)]
                );
            }
        }
    }

    #[test]
    fn test_cov_irreg_diagonal_positive() {
        // Diagonal (variances) should be non-negative
        let ifd = make_ifd(
            vec![0, 5, 10, 15],
            vec![
                0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0,
            ],
            vec![
                1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 2.0, 3.0, 2.0, 3.0, 2.0,
            ],
        );

        let grid = vec![0.25, 0.5, 0.75];
        let cov = cov_irreg(&ifd, &grid, &grid, 0.3);

        for i in 0..3 {
            assert!(
                cov[(i, i)] >= -1e-10,
                "Variance at {} should be non-negative: {}",
                i,
                cov[(i, i)]
            );
        }
    }

    #[test]
    fn test_cov_irreg_different_grids() {
        let ifd = make_ifd(
            vec![0, 5, 10],
            vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0],
            vec![1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
        );

        let s_grid = vec![0.25, 0.5];
        let t_grid = vec![0.5, 0.75];
        let cov = cov_irreg(&ifd, &s_grid, &t_grid, 0.3);

        // Should produce a 2x2 matrix
        assert_eq!(cov.nrows(), 2);
        assert_eq!(cov.ncols(), 2);
    }

    // ========================================================================
    // Tests for metric_lp_irreg
    // ========================================================================

    #[test]
    fn test_metric_lp_irreg_self_distance_zero() {
        // Distance to self should be 0
        let ifd = make_ifd(
            vec![0, 5, 10],
            vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0],
            vec![1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
        );

        let dist = metric_lp_irreg(&ifd, 2.0);

        // Diagonal should be 0
        let n = 2;
        for i in 0..n {
            assert!(
                dist[(i, i)].abs() < 1e-10,
                "Self-distance d[{},{}] = {} should be 0",
                i,
                i,
                dist[(i, i)]
            );
        }
    }

    #[test]
    fn test_metric_lp_irreg_symmetry() {
        // Distance matrix should be symmetric
        let ifd = make_ifd(
            vec![0, 5, 10, 15],
            vec![
                0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0,
            ],
            vec![
                1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 3.0, 4.0, 3.0, 2.0,
            ],
        );

        let dist = metric_lp_irreg(&ifd, 2.0);
        let n = 3;

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                    "Dist[{},{}] = {} != Dist[{},{}] = {}",
                    i,
                    j,
                    dist[(i, j)],
                    j,
                    i,
                    dist[(j, i)]
                );
            }
        }
    }

    #[test]
    fn test_metric_lp_irreg_triangle_inequality() {
        // Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        let ifd = make_ifd(
            vec![0, 5, 10, 15],
            vec![
                0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, // curve a
                1.0, 1.0, 1.0, 1.0, 1.0, // curve b
                2.0, 2.0, 2.0, 2.0, 2.0, // curve c
            ],
        );

        let dist = metric_lp_irreg(&ifd, 2.0);

        // d(a,c) <= d(a,b) + d(b,c)
        let d_ac = dist[(0, 2)];
        let d_ab = dist[(0, 1)];
        let d_bc = dist[(1, 2)];

        assert!(
            d_ac <= d_ab + d_bc + 1e-10,
            "Triangle inequality violated: {} > {} + {}",
            d_ac,
            d_ab,
            d_bc
        );
    }

    // ========================================================================
    // Tests for to_regular_grid
    // ========================================================================

    #[test]
    fn test_to_regular_grid_basic() {
        let ifd = make_ifd(
            vec![0, 5],
            vec![0.0, 0.25, 0.5, 0.75, 1.0],
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
        );

        let grid = vec![0.0, 0.5, 1.0];
        let result = to_regular_grid(&ifd, &grid);

        // Should produce 1 curve x 3 points
        assert_eq!(result.nrows(), 1);
        assert_eq!(result.ncols(), 3);

        // Check interpolated values
        assert!(
            (result[(0, 0)] - 0.0).abs() < 1e-10,
            "At t=0: {}",
            result[(0, 0)]
        );
        assert!(
            (result[(0, 1)] - 2.0).abs() < 1e-10,
            "At t=0.5: {}",
            result[(0, 1)]
        );
        assert!(
            (result[(0, 2)] - 4.0).abs() < 1e-10,
            "At t=1: {}",
            result[(0, 2)]
        );
    }

    #[test]
    fn test_to_regular_grid_multiple_curves() {
        let ifd = make_ifd(
            vec![0, 5, 10],
            vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0],
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, // Linear: y = 4t
                4.0, 3.0, 2.0, 1.0, 0.0, // Linear: y = 4 - 4t
            ],
        );

        let grid = vec![0.0, 0.5, 1.0];
        let result = to_regular_grid(&ifd, &grid);

        // Should produce 2 curves x 3 points
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 3);

        // Curve 0 at t=0.5 should be 2.0
        assert!((result[(0, 1)] - 2.0).abs() < 1e-10);
        // Curve 1 at t=0.5 should be 2.0
        assert!((result[(1, 1)] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_regular_grid_boundary_nan() {
        let ifd = make_ifd(
            vec![0, 3],
            vec![0.2, 0.5, 0.8], // Curve only defined on [0.2, 0.8]
            vec![1.0, 2.0, 3.0],
        );

        let grid = vec![0.0, 0.5, 1.0]; // Grid extends beyond curve range
        let result = to_regular_grid(&ifd, &grid);

        // At t=0.0 (before curve starts), should be NaN
        assert!(result[(0, 0)].is_nan(), "t=0 should be NaN");
        // At t=0.5 (within range), should be valid
        assert!(
            (result[(0, 1)] - 2.0).abs() < 1e-10,
            "t=0.5: {}",
            result[(0, 1)]
        );
        // At t=1.0 (after curve ends), should be NaN
        assert!(result[(0, 2)].is_nan(), "t=1 should be NaN");
    }
}
