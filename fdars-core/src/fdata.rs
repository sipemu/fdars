//! Functional data operations: mean, center, derivatives, norms, and geometric median.

use crate::helpers::{simpsons_weights, simpsons_weights_2d, NUMERICAL_EPS};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Compute finite difference for a 1D function at a given index.
///
/// Uses forward difference at left boundary, backward difference at right boundary,
/// and central difference for interior points.
fn finite_diff_1d(
    values: impl Fn(usize) -> f64,
    idx: usize,
    n_points: usize,
    step_sizes: &[f64],
) -> f64 {
    if idx == 0 {
        (values(1) - values(0)) / step_sizes[0]
    } else if idx == n_points - 1 {
        (values(n_points - 1) - values(n_points - 2)) / step_sizes[n_points - 1]
    } else {
        (values(idx + 1) - values(idx - 1)) / step_sizes[idx]
    }
}

/// Compute 2D partial derivatives at a single grid point.
///
/// Returns (∂f/∂s, ∂f/∂t, ∂²f/∂s∂t) using finite differences.
fn compute_2d_derivatives(
    get_val: impl Fn(usize, usize) -> f64,
    si: usize,
    ti: usize,
    m1: usize,
    m2: usize,
    hs: &[f64],
    ht: &[f64],
) -> (f64, f64, f64) {
    // ∂f/∂s
    let ds = finite_diff_1d(|s| get_val(s, ti), si, m1, hs);

    // ∂f/∂t
    let dt = finite_diff_1d(|t| get_val(si, t), ti, m2, ht);

    // ∂²f/∂s∂t (mixed partial)
    let denom = hs[si] * ht[ti];

    // Get the appropriate indices for s and t differences
    let (s_lo, s_hi) = if si == 0 {
        (0, 1)
    } else if si == m1 - 1 {
        (m1 - 2, m1 - 1)
    } else {
        (si - 1, si + 1)
    };

    let (t_lo, t_hi) = if ti == 0 {
        (0, 1)
    } else if ti == m2 - 1 {
        (m2 - 2, m2 - 1)
    } else {
        (ti - 1, ti + 1)
    };

    let dsdt = (get_val(s_hi, t_hi) - get_val(s_lo, t_hi) - get_val(s_hi, t_lo)
        + get_val(s_lo, t_lo))
        / denom;

    (ds, dt, dsdt)
}

/// Perform Weiszfeld iteration to compute geometric median.
///
/// This is the core algorithm shared by 1D and 2D geometric median computations.
fn weiszfeld_iteration(data: &FdMatrix, weights: &[f64], max_iter: usize, tol: f64) -> Vec<f64> {
    let (n, m) = data.shape();

    // Initialize with the mean
    let mut median: Vec<f64> = (0..m)
        .map(|j| {
            let col = data.column(j);
            col.iter().sum::<f64>() / n as f64
        })
        .collect();

    for _ in 0..max_iter {
        // Compute distances from current median to all curves
        let distances: Vec<f64> = (0..n)
            .map(|i| {
                let mut dist_sq = 0.0;
                for j in 0..m {
                    let diff = data[(i, j)] - median[j];
                    dist_sq += diff * diff * weights[j];
                }
                dist_sq.sqrt()
            })
            .collect();

        // Compute weights (1/distance), handling zero distances
        let inv_distances: Vec<f64> = distances
            .iter()
            .map(|d| {
                if *d > NUMERICAL_EPS {
                    1.0 / d
                } else {
                    1.0 / NUMERICAL_EPS
                }
            })
            .collect();

        let sum_inv_dist: f64 = inv_distances.iter().sum();

        // Update median using Weiszfeld iteration
        let new_median: Vec<f64> = (0..m)
            .map(|j| {
                let mut weighted_sum = 0.0;
                for i in 0..n {
                    weighted_sum += data[(i, j)] * inv_distances[i];
                }
                weighted_sum / sum_inv_dist
            })
            .collect();

        // Check convergence
        let diff: f64 = median
            .iter()
            .zip(new_median.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / m as f64;

        median = new_median;

        if diff < tol {
            break;
        }
    }

    median
}

/// Compute the mean function across all samples (1D).
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
///
/// # Returns
/// Mean function values at each evaluation point
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::fdata::mean_1d;
///
/// // 3 curves at 4 evaluation points
/// let data = FdMatrix::from_column_major(
///     vec![1.0, 2.0, 3.0,  4.0, 5.0, 6.0,  7.0, 8.0, 9.0,  10.0, 11.0, 12.0],
///     3, 4,
/// ).unwrap();
/// let mean = mean_1d(&data);
/// assert_eq!(mean.len(), 4);
/// assert!((mean[0] - 2.0).abs() < 1e-10); // mean of [1, 2, 3]
/// ```
pub fn mean_1d(data: &FdMatrix) -> Vec<f64> {
    let (n, m) = data.shape();
    if n == 0 || m == 0 {
        return Vec::new();
    }

    iter_maybe_parallel!(0..m)
        .map(|j| {
            let col = data.column(j);
            col.iter().sum::<f64>() / n as f64
        })
        .collect()
}

/// Compute the mean function for 2D surfaces.
///
/// Data is stored as n x (m1*m2) matrix where each row is a flattened surface.
pub fn mean_2d(data: &FdMatrix) -> Vec<f64> {
    // Same computation as 1D - just compute pointwise mean
    mean_1d(data)
}

/// Center functional data by subtracting the mean function.
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
///
/// # Returns
/// Centered data matrix
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::fdata::{center_1d, mean_1d};
///
/// let data = FdMatrix::from_column_major(
///     vec![1.0, 3.0, 2.0, 4.0, 3.0, 5.0], 2, 3,
/// ).unwrap();
/// let centered = center_1d(&data);
/// assert_eq!(centered.shape(), (2, 3));
/// // Column means of centered data should be zero
/// let means = mean_1d(&centered);
/// assert!(means.iter().all(|m| m.abs() < 1e-10));
/// ```
pub fn center_1d(data: &FdMatrix) -> FdMatrix {
    let (n, m) = data.shape();
    if n == 0 || m == 0 {
        return FdMatrix::zeros(0, 0);
    }

    // First compute the mean for each column (parallelized)
    let means: Vec<f64> = iter_maybe_parallel!(0..m)
        .map(|j| {
            let col = data.column(j);
            col.iter().sum::<f64>() / n as f64
        })
        .collect();

    // Create centered data
    let mut centered = FdMatrix::zeros(n, m);
    for j in 0..m {
        let col = centered.column_mut(j);
        let src = data.column(j);
        for i in 0..n {
            col[i] = src[i] - means[j];
        }
    }

    centered
}

/// Normalization method for functional data.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum NormalizationMethod {
    /// Center columns (subtract per-time-point mean across curves).
    Center,
    /// Autoscale columns (center + divide by per-time-point std dev). UV scaling.
    Autoscale,
    /// Pareto scaling (center + divide by sqrt of per-time-point std dev).
    Pareto,
    /// Range scaling (center + divide by per-time-point range).
    Range,
    /// Per-curve centering (subtract each curve's own mean).
    CurveCenter,
    /// Per-curve standardization (subtract mean, divide by std dev per curve).
    CurveStandardize,
    /// Per-curve range normalization to [0, 1].
    CurveRange,
}

/// Normalize functional data using the specified method.
///
/// **Column-wise methods** (across curves at each time point):
/// - `Center`: subtract column means (same as [`center_1d`])
/// - `Autoscale`: center + divide by column std dev (unit variance per time point)
/// - `Pareto`: center + divide by sqrt(column std dev)
/// - `Range`: center + divide by column range (max - min)
///
/// **Row-wise methods** (per curve):
/// - `CurveCenter`: subtract each curve's own mean
/// - `CurveStandardize`: subtract mean, divide by std dev per curve
/// - `CurveRange`: scale each curve to [0, 1]
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::fdata::{normalize, NormalizationMethod};
///
/// let data = FdMatrix::from_column_major(
///     vec![1.0, 3.0, 2.0, 6.0, 3.0, 9.0], 2, 3,
/// ).unwrap();
///
/// // Autoscale: zero mean, unit variance per time point
/// let scaled = normalize(&data, NormalizationMethod::Autoscale);
/// assert_eq!(scaled.shape(), (2, 3));
/// ```
pub fn normalize(data: &FdMatrix, method: NormalizationMethod) -> FdMatrix {
    let (n, m) = data.shape();
    if n == 0 || m == 0 {
        return FdMatrix::zeros(n, m);
    }

    match method {
        NormalizationMethod::Center => center_1d(data),
        NormalizationMethod::Autoscale => column_scale(data, n, m, ScaleKind::StdDev),
        NormalizationMethod::Pareto => column_scale(data, n, m, ScaleKind::SqrtStdDev),
        NormalizationMethod::Range => column_scale(data, n, m, ScaleKind::Range),
        NormalizationMethod::CurveCenter => row_normalize(data, n, m, RowNorm::Center),
        NormalizationMethod::CurveStandardize => row_normalize(data, n, m, RowNorm::Standardize),
        NormalizationMethod::CurveRange => row_normalize(data, n, m, RowNorm::Range),
    }
}

#[derive(Clone, Copy)]
enum ScaleKind {
    StdDev,
    SqrtStdDev,
    Range,
}

fn column_scale(data: &FdMatrix, n: usize, m: usize, kind: ScaleKind) -> FdMatrix {
    let mut result = FdMatrix::zeros(n, m);
    for j in 0..m {
        let col = data.column(j);
        let mean = col.iter().sum::<f64>() / n as f64;
        let scale = match kind {
            ScaleKind::StdDev => {
                let var =
                    col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
                var.sqrt()
            }
            ScaleKind::SqrtStdDev => {
                let var =
                    col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
                var.sqrt().sqrt()
            }
            ScaleKind::Range => {
                let min = col.iter().copied().fold(f64::INFINITY, f64::min);
                let max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                max - min
            }
        };
        let out = result.column_mut(j);
        let denom = if scale > 1e-15 { scale } else { 1.0 };
        for i in 0..n {
            out[i] = (col[i] - mean) / denom;
        }
    }
    result
}

#[derive(Clone, Copy)]
enum RowNorm {
    Center,
    Standardize,
    Range,
}

fn row_normalize(data: &FdMatrix, n: usize, m: usize, kind: RowNorm) -> FdMatrix {
    let mut result = FdMatrix::zeros(n, m);
    for i in 0..n {
        let row: Vec<f64> = (0..m).map(|j| data[(i, j)]).collect();
        let mean = row.iter().sum::<f64>() / m as f64;
        match kind {
            RowNorm::Center => {
                for j in 0..m {
                    result[(i, j)] = row[j] - mean;
                }
            }
            RowNorm::Standardize => {
                let std = (row.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
                    / (m - 1).max(1) as f64)
                    .sqrt();
                let denom = if std > 1e-15 { std } else { 1.0 };
                for j in 0..m {
                    result[(i, j)] = (row[j] - mean) / denom;
                }
            }
            RowNorm::Range => {
                let min = row.iter().copied().fold(f64::INFINITY, f64::min);
                let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let range = max - min;
                let denom = if range > 1e-15 { range } else { 1.0 };
                for j in 0..m {
                    result[(i, j)] = (row[j] - min) / denom;
                }
            }
        }
    }
    result
}

/// Compute Lp norm for each sample.
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
/// * `argvals` - Evaluation points for integration
/// * `p` - Order of the norm (e.g., 2.0 for L2)
///
/// # Returns
/// Vector of Lp norms for each sample
pub fn norm_lp_1d(data: &FdMatrix, argvals: &[f64], p: f64) -> Vec<f64> {
    let (n, m) = data.shape();
    if n == 0 || m == 0 || argvals.len() != m {
        return Vec::new();
    }

    let weights = simpsons_weights(argvals);

    if (p - 2.0).abs() < 1e-14 {
        iter_maybe_parallel!(0..n)
            .map(|i| {
                let mut integral = 0.0;
                for j in 0..m {
                    let v = data[(i, j)];
                    integral += v * v * weights[j];
                }
                integral.sqrt()
            })
            .collect()
    } else if (p - 1.0).abs() < 1e-14 {
        iter_maybe_parallel!(0..n)
            .map(|i| {
                let mut integral = 0.0;
                for j in 0..m {
                    integral += data[(i, j)].abs() * weights[j];
                }
                integral
            })
            .collect()
    } else {
        iter_maybe_parallel!(0..n)
            .map(|i| {
                let mut integral = 0.0;
                for j in 0..m {
                    integral += data[(i, j)].abs().powf(p) * weights[j];
                }
                integral.powf(1.0 / p)
            })
            .collect()
    }
}

/// Compute numerical derivative of functional data (parallelized over rows).
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
/// * `argvals` - Evaluation points
/// * `nderiv` - Order of derivative
///
/// # Returns
/// Derivative data matrix
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::fdata::deriv_1d;
///
/// // Linear function f(t) = t on [0, 1], derivative should be ~1
/// let argvals: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let data = FdMatrix::from_column_major(argvals.clone(), 1, 20).unwrap();
/// let deriv = deriv_1d(&data, &argvals, 1);
/// assert_eq!(deriv.shape(), (1, 20));
/// // Interior points should have derivative close to 1.0
/// assert!((deriv[(0, 10)] - 1.0).abs() < 0.1);
/// ```
/// Compute one derivative step: forward/central/backward differences written column-wise.
fn deriv_1d_step(
    current: &FdMatrix,
    n: usize,
    m: usize,
    h0: f64,
    hn: f64,
    h_central: &[f64],
) -> FdMatrix {
    let mut next = FdMatrix::zeros(n, m);
    // Column 0: forward difference
    let src_col0 = current.column(0);
    let src_col1 = current.column(1);
    let dst = next.column_mut(0);
    for i in 0..n {
        dst[i] = (src_col1[i] - src_col0[i]) / h0;
    }
    // Interior columns: central difference
    for j in 1..(m - 1) {
        let src_prev = current.column(j - 1);
        let src_next = current.column(j + 1);
        let dst = next.column_mut(j);
        let h = h_central[j - 1];
        for i in 0..n {
            dst[i] = (src_next[i] - src_prev[i]) / h;
        }
    }
    // Column m-1: backward difference
    let src_colm2 = current.column(m - 2);
    let src_colm1 = current.column(m - 1);
    let dst = next.column_mut(m - 1);
    for i in 0..n {
        dst[i] = (src_colm1[i] - src_colm2[i]) / hn;
    }
    next
}

pub fn deriv_1d(data: &FdMatrix, argvals: &[f64], nderiv: usize) -> FdMatrix {
    let (n, m) = data.shape();
    if n == 0 || m < 2 || argvals.len() != m {
        return FdMatrix::zeros(n, m);
    }
    if nderiv == 0 {
        return data.clone();
    }

    let mut current = data.clone();

    // Pre-compute step sizes for central differences
    let h0 = argvals[1] - argvals[0];
    let hn = argvals[m - 1] - argvals[m - 2];
    let h_central: Vec<f64> = (1..(m - 1))
        .map(|j| argvals[j + 1] - argvals[j - 1])
        .collect();

    for _ in 0..nderiv {
        current = deriv_1d_step(&current, n, m, h0, hn, &h_central);
    }

    current
}

/// Result of 2D partial derivatives.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Deriv2DResult {
    /// Partial derivative with respect to s (∂f/∂s)
    pub ds: FdMatrix,
    /// Partial derivative with respect to t (∂f/∂t)
    pub dt: FdMatrix,
    /// Mixed partial derivative (∂²f/∂s∂t)
    pub dsdt: FdMatrix,
}

/// Compute finite-difference step sizes for a grid.
///
/// Uses forward/backward difference at boundaries and central difference for interior.
fn compute_step_sizes(argvals: &[f64]) -> Vec<f64> {
    let m = argvals.len();
    if m < 2 {
        return vec![1.0; m];
    }
    (0..m)
        .map(|j| {
            if j == 0 {
                argvals[1] - argvals[0]
            } else if j == m - 1 {
                argvals[m - 1] - argvals[m - 2]
            } else {
                argvals[j + 1] - argvals[j - 1]
            }
        })
        .collect()
}

/// Collect per-curve row vectors into a column-major FdMatrix.
fn reassemble_colmajor(rows: &[Vec<f64>], n: usize, ncol: usize) -> FdMatrix {
    let mut mat = FdMatrix::zeros(n, ncol);
    for i in 0..n {
        for j in 0..ncol {
            mat[(i, j)] = rows[i][j];
        }
    }
    mat
}

/// Compute 2D partial derivatives for surface data.
///
/// For a surface f(s,t), computes:
/// - ds: partial derivative with respect to s (∂f/∂s)
/// - dt: partial derivative with respect to t (∂f/∂t)
/// - dsdt: mixed partial derivative (∂²f/∂s∂t)
///
/// # Arguments
/// * `data` - Functional data matrix, n surfaces, each stored as m1*m2 values
/// * `argvals_s` - Grid points in s direction (length m1)
/// * `argvals_t` - Grid points in t direction (length m2)
/// * `m1` - Grid size in s direction
/// * `m2` - Grid size in t direction
pub fn deriv_2d(
    data: &FdMatrix,
    argvals_s: &[f64],
    argvals_t: &[f64],
    m1: usize,
    m2: usize,
) -> Option<Deriv2DResult> {
    let n = data.nrows();
    let ncol = m1 * m2;
    if n == 0
        || ncol == 0
        || m1 < 2
        || m2 < 2
        || data.ncols() != ncol
        || argvals_s.len() != m1
        || argvals_t.len() != m2
    {
        return None;
    }

    let hs = compute_step_sizes(argvals_s);
    let ht = compute_step_sizes(argvals_t);

    // Compute all derivatives in parallel over surfaces
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let mut ds = vec![0.0; ncol];
            let mut dt = vec![0.0; ncol];
            let mut dsdt = vec![0.0; ncol];

            let get_val = |si: usize, ti: usize| -> f64 { data[(i, si + ti * m1)] };

            for ti in 0..m2 {
                for si in 0..m1 {
                    let idx = si + ti * m1;
                    let (ds_val, dt_val, dsdt_val) =
                        compute_2d_derivatives(get_val, si, ti, m1, m2, &hs, &ht);
                    ds[idx] = ds_val;
                    dt[idx] = dt_val;
                    dsdt[idx] = dsdt_val;
                }
            }

            (ds, dt, dsdt)
        })
        .collect();

    let (ds_vecs, (dt_vecs, dsdt_vecs)): (Vec<Vec<f64>>, (Vec<Vec<f64>>, Vec<Vec<f64>>)) =
        results.into_iter().map(|(a, b, c)| (a, (b, c))).unzip();

    Some(Deriv2DResult {
        ds: reassemble_colmajor(&ds_vecs, n, ncol),
        dt: reassemble_colmajor(&dt_vecs, n, ncol),
        dsdt: reassemble_colmajor(&dsdt_vecs, n, ncol),
    })
}

/// Compute the geometric median (L1 median) of functional data using Weiszfeld's algorithm.
///
/// The geometric median minimizes sum of L2 distances to all curves.
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
/// * `argvals` - Evaluation points for integration
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
pub fn geometric_median_1d(
    data: &FdMatrix,
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let (n, m) = data.shape();
    if n == 0 || m == 0 || argvals.len() != m {
        return Vec::new();
    }

    let weights = simpsons_weights(argvals);
    weiszfeld_iteration(data, &weights, max_iter, tol)
}

/// Compute the geometric median for 2D functional data.
///
/// Data is stored as n x (m1*m2) matrix where each row is a flattened surface.
///
/// # Arguments
/// * `data` - Functional data matrix (n x m) where m = m1*m2
/// * `argvals_s` - Grid points in s direction (length m1)
/// * `argvals_t` - Grid points in t direction (length m2)
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
pub fn geometric_median_2d(
    data: &FdMatrix,
    argvals_s: &[f64],
    argvals_t: &[f64],
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let (n, m) = data.shape();
    let expected_cols = argvals_s.len() * argvals_t.len();
    if n == 0 || m == 0 || m != expected_cols {
        return Vec::new();
    }

    let weights = simpsons_weights_2d(argvals_s, argvals_t);
    weiszfeld_iteration(data, &weights, max_iter, tol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;
    use std::f64::consts::PI;

    // ============== Mean tests ==============

    #[test]
    fn test_mean_1d() {
        // 2 samples, 3 points each
        // Sample 1: [1, 2, 3]
        // Sample 2: [3, 4, 5]
        // Mean should be [2, 3, 4]
        let data = vec![1.0, 3.0, 2.0, 4.0, 3.0, 5.0]; // column-major
        let mat = FdMatrix::from_column_major(data, 2, 3).unwrap();
        let mean = mean_1d(&mat);
        assert_eq!(mean, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_mean_1d_single_sample() {
        let data = vec![1.0, 2.0, 3.0];
        let mat = FdMatrix::from_column_major(data, 1, 3).unwrap();
        let mean = mean_1d(&mat);
        assert_eq!(mean, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mean_1d_invalid() {
        assert!(mean_1d(&FdMatrix::zeros(0, 0)).is_empty());
    }

    #[test]
    fn test_mean_2d_delegates() {
        let data = vec![1.0, 3.0, 2.0, 4.0];
        let mat = FdMatrix::from_column_major(data, 2, 2).unwrap();
        let mean1d = mean_1d(&mat);
        let mean2d = mean_2d(&mat);
        assert_eq!(mean1d, mean2d);
    }

    // ============== Center tests ==============

    #[test]
    fn test_center_1d() {
        let data = vec![1.0, 3.0, 2.0, 4.0, 3.0, 5.0]; // column-major
        let mat = FdMatrix::from_column_major(data, 2, 3).unwrap();
        let centered = center_1d(&mat);
        // Mean is [2, 3, 4], so centered should be [-1, 1, -1, 1, -1, 1]
        assert_eq!(centered.as_slice(), &[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
    }

    #[test]
    fn test_center_1d_mean_zero() {
        let data = vec![1.0, 3.0, 2.0, 4.0, 3.0, 5.0];
        let mat = FdMatrix::from_column_major(data, 2, 3).unwrap();
        let centered = center_1d(&mat);
        let centered_mean = mean_1d(&centered);
        for m in centered_mean {
            assert!(m.abs() < 1e-10, "Centered data should have zero mean");
        }
    }

    #[test]
    fn test_center_1d_invalid() {
        let centered = center_1d(&FdMatrix::zeros(0, 0));
        assert!(centered.is_empty());
    }

    // ============== Norm tests ==============

    #[test]
    fn test_norm_lp_1d_constant() {
        // Constant function 2 on [0, 1] has L2 norm = 2
        let argvals = uniform_grid(21);
        let data: Vec<f64> = vec![2.0; 21];
        let mat = FdMatrix::from_column_major(data, 1, 21).unwrap();
        let norms = norm_lp_1d(&mat, &argvals, 2.0);
        assert_eq!(norms.len(), 1);
        assert!(
            (norms[0] - 2.0).abs() < 0.1,
            "L2 norm of constant 2 should be 2"
        );
    }

    #[test]
    fn test_norm_lp_1d_sine() {
        // L2 norm of sin(pi*x) on [0, 1] = sqrt(0.5)
        let argvals = uniform_grid(101);
        let data: Vec<f64> = argvals.iter().map(|&x| (PI * x).sin()).collect();
        let mat = FdMatrix::from_column_major(data, 1, 101).unwrap();
        let norms = norm_lp_1d(&mat, &argvals, 2.0);
        let expected = 0.5_f64.sqrt();
        assert!(
            (norms[0] - expected).abs() < 0.05,
            "Expected {}, got {}",
            expected,
            norms[0]
        );
    }

    #[test]
    fn test_norm_lp_1d_invalid() {
        assert!(norm_lp_1d(&FdMatrix::zeros(0, 0), &[], 2.0).is_empty());
    }

    // ============== Derivative tests ==============

    #[test]
    fn test_deriv_1d_linear() {
        // Derivative of linear function x should be 1
        let argvals = uniform_grid(21);
        let data = argvals.clone();
        let mat = FdMatrix::from_column_major(data, 1, 21).unwrap();
        let deriv = deriv_1d(&mat, &argvals, 1);
        // Interior points should have derivative close to 1
        for j in 2..19 {
            assert!(
                (deriv[(0, j)] - 1.0).abs() < 0.1,
                "Derivative of x should be 1"
            );
        }
    }

    #[test]
    fn test_deriv_1d_quadratic() {
        // Derivative of x^2 should be 2x
        let argvals = uniform_grid(51);
        let data: Vec<f64> = argvals.iter().map(|&x| x * x).collect();
        let mat = FdMatrix::from_column_major(data, 1, 51).unwrap();
        let deriv = deriv_1d(&mat, &argvals, 1);
        // Check interior points
        for j in 5..45 {
            let expected = 2.0 * argvals[j];
            assert!(
                (deriv[(0, j)] - expected).abs() < 0.1,
                "Derivative of x^2 should be 2x"
            );
        }
    }

    #[test]
    fn test_deriv_1d_invalid() {
        let result = deriv_1d(&FdMatrix::zeros(0, 0), &[], 1);
        assert!(result.is_empty() || result.as_slice().iter().all(|&x| x == 0.0));
    }

    // ============== Geometric median tests ==============

    #[test]
    fn test_geometric_median_identical_curves() {
        // All curves identical -> median = that curve
        let argvals = uniform_grid(21);
        let n = 5;
        let m = 21;
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                data[i + j * n] = (2.0 * PI * argvals[j]).sin();
            }
        }
        let mat = FdMatrix::from_column_major(data, n, m).unwrap();
        let median = geometric_median_1d(&mat, &argvals, 100, 1e-6);
        for j in 0..m {
            let expected = (2.0 * PI * argvals[j]).sin();
            assert!(
                (median[j] - expected).abs() < 0.01,
                "Median should equal all curves"
            );
        }
    }

    #[test]
    fn test_geometric_median_converges() {
        let argvals = uniform_grid(21);
        let n = 10;
        let m = 21;
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                data[i + j * n] = (i as f64 / n as f64) * argvals[j];
            }
        }
        let mat = FdMatrix::from_column_major(data, n, m).unwrap();
        let median = geometric_median_1d(&mat, &argvals, 100, 1e-6);
        assert_eq!(median.len(), m);
        assert!(median.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_geometric_median_invalid() {
        assert!(geometric_median_1d(&FdMatrix::zeros(0, 0), &[], 100, 1e-6).is_empty());
    }

    // ============== 2D derivative tests ==============

    #[test]
    fn test_deriv_2d_linear_surface() {
        // f(s, t) = 2*s + 3*t
        // ∂f/∂s = 2, ∂f/∂t = 3, ∂²f/∂s∂t = 0
        let m1 = 11;
        let m2 = 11;
        let argvals_s: Vec<f64> = (0..m1).map(|i| i as f64 / (m1 - 1) as f64).collect();
        let argvals_t: Vec<f64> = (0..m2).map(|i| i as f64 / (m2 - 1) as f64).collect();

        let n = 1; // single surface
        let ncol = m1 * m2;
        let mut data = vec![0.0; n * ncol];

        for si in 0..m1 {
            for ti in 0..m2 {
                let s = argvals_s[si];
                let t = argvals_t[ti];
                let idx = si + ti * m1;
                data[idx] = 2.0 * s + 3.0 * t;
            }
        }

        let mat = FdMatrix::from_column_major(data, n, ncol).unwrap();
        let result = deriv_2d(&mat, &argvals_s, &argvals_t, m1, m2).unwrap();

        // Check interior points for ∂f/∂s ≈ 2
        for si in 2..(m1 - 2) {
            for ti in 2..(m2 - 2) {
                let idx = si + ti * m1;
                assert!(
                    (result.ds[(0, idx)] - 2.0).abs() < 0.2,
                    "∂f/∂s at ({}, {}) = {}, expected 2",
                    si,
                    ti,
                    result.ds[(0, idx)]
                );
            }
        }

        // Check interior points for ∂f/∂t ≈ 3
        for si in 2..(m1 - 2) {
            for ti in 2..(m2 - 2) {
                let idx = si + ti * m1;
                assert!(
                    (result.dt[(0, idx)] - 3.0).abs() < 0.2,
                    "∂f/∂t at ({}, {}) = {}, expected 3",
                    si,
                    ti,
                    result.dt[(0, idx)]
                );
            }
        }

        // Check interior points for mixed partial ≈ 0
        for si in 2..(m1 - 2) {
            for ti in 2..(m2 - 2) {
                let idx = si + ti * m1;
                assert!(
                    result.dsdt[(0, idx)].abs() < 0.5,
                    "∂²f/∂s∂t at ({}, {}) = {}, expected 0",
                    si,
                    ti,
                    result.dsdt[(0, idx)]
                );
            }
        }
    }

    #[test]
    fn test_deriv_2d_quadratic_surface() {
        // f(s, t) = s*t
        // ∂f/∂s = t, ∂f/∂t = s, ∂²f/∂s∂t = 1
        let m1 = 21;
        let m2 = 21;
        let argvals_s: Vec<f64> = (0..m1).map(|i| i as f64 / (m1 - 1) as f64).collect();
        let argvals_t: Vec<f64> = (0..m2).map(|i| i as f64 / (m2 - 1) as f64).collect();

        let n = 1;
        let ncol = m1 * m2;
        let mut data = vec![0.0; n * ncol];

        for si in 0..m1 {
            for ti in 0..m2 {
                let s = argvals_s[si];
                let t = argvals_t[ti];
                let idx = si + ti * m1;
                data[idx] = s * t;
            }
        }

        let mat = FdMatrix::from_column_major(data, n, ncol).unwrap();
        let result = deriv_2d(&mat, &argvals_s, &argvals_t, m1, m2).unwrap();

        // Check interior points for ∂f/∂s ≈ t
        for si in 3..(m1 - 3) {
            for ti in 3..(m2 - 3) {
                let idx = si + ti * m1;
                let expected = argvals_t[ti];
                assert!(
                    (result.ds[(0, idx)] - expected).abs() < 0.1,
                    "∂f/∂s at ({}, {}) = {}, expected {}",
                    si,
                    ti,
                    result.ds[(0, idx)],
                    expected
                );
            }
        }

        // Check interior points for ∂f/∂t ≈ s
        for si in 3..(m1 - 3) {
            for ti in 3..(m2 - 3) {
                let idx = si + ti * m1;
                let expected = argvals_s[si];
                assert!(
                    (result.dt[(0, idx)] - expected).abs() < 0.1,
                    "∂f/∂t at ({}, {}) = {}, expected {}",
                    si,
                    ti,
                    result.dt[(0, idx)],
                    expected
                );
            }
        }

        // Check interior points for mixed partial ≈ 1
        for si in 3..(m1 - 3) {
            for ti in 3..(m2 - 3) {
                let idx = si + ti * m1;
                assert!(
                    (result.dsdt[(0, idx)] - 1.0).abs() < 0.3,
                    "∂²f/∂s∂t at ({}, {}) = {}, expected 1",
                    si,
                    ti,
                    result.dsdt[(0, idx)]
                );
            }
        }
    }

    #[test]
    fn test_deriv_2d_invalid_input() {
        // Empty data
        let result = deriv_2d(&FdMatrix::zeros(0, 0), &[], &[], 0, 0);
        assert!(result.is_none());

        // Mismatched dimensions
        let mat = FdMatrix::from_column_major(vec![1.0; 4], 1, 4).unwrap();
        let argvals = vec![0.0, 1.0];
        let result = deriv_2d(&mat, &argvals, &[0.0, 0.5, 1.0], 2, 2);
        assert!(result.is_none());
    }

    // ============== 2D geometric median tests ==============

    #[test]
    fn test_geometric_median_2d_basic() {
        // Three identical surfaces -> median = that surface
        let m1 = 5;
        let m2 = 5;
        let m = m1 * m2;
        let n = 3;
        let argvals_s: Vec<f64> = (0..m1).map(|i| i as f64 / (m1 - 1) as f64).collect();
        let argvals_t: Vec<f64> = (0..m2).map(|i| i as f64 / (m2 - 1) as f64).collect();

        let mut data = vec![0.0; n * m];

        // Create identical surfaces: f(s, t) = s + t
        for i in 0..n {
            for si in 0..m1 {
                for ti in 0..m2 {
                    let idx = si + ti * m1;
                    let s = argvals_s[si];
                    let t = argvals_t[ti];
                    data[i + idx * n] = s + t;
                }
            }
        }

        let mat = FdMatrix::from_column_major(data, n, m).unwrap();
        let median = geometric_median_2d(&mat, &argvals_s, &argvals_t, 100, 1e-6);
        assert_eq!(median.len(), m);

        // Check that median equals the surface
        for si in 0..m1 {
            for ti in 0..m2 {
                let idx = si + ti * m1;
                let expected = argvals_s[si] + argvals_t[ti];
                assert!(
                    (median[idx] - expected).abs() < 0.01,
                    "Median at ({}, {}) = {}, expected {}",
                    si,
                    ti,
                    median[idx],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_nan_mean_no_panic() {
        let mut data_vec = vec![1.0; 6];
        data_vec[2] = f64::NAN;
        let data = FdMatrix::from_column_major(data_vec, 2, 3).unwrap();
        let m = mean_1d(&data);
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn test_nan_center_no_panic() {
        let mut data_vec = vec![1.0; 6];
        data_vec[2] = f64::NAN;
        let data = FdMatrix::from_column_major(data_vec, 2, 3).unwrap();
        let c = center_1d(&data);
        assert_eq!(c.nrows(), 2);
    }

    #[test]
    fn test_nan_norm_no_panic() {
        let mut data_vec = vec![1.0; 6];
        data_vec[2] = f64::NAN;
        let data = FdMatrix::from_column_major(data_vec, 2, 3).unwrap();
        let argvals = vec![0.0, 0.5, 1.0];
        let norms = norm_lp_1d(&data, &argvals, 2.0);
        assert_eq!(norms.len(), 2);
    }

    #[test]
    fn test_n1_norm() {
        let data = FdMatrix::from_column_major(vec![0.0, 1.0, 0.0], 1, 3).unwrap();
        let argvals = vec![0.0, 0.5, 1.0];
        let norms = norm_lp_1d(&data, &argvals, 2.0);
        assert_eq!(norms.len(), 1);
        assert!(norms[0] > 0.0);
    }

    #[test]
    fn test_n2_center() {
        let data = FdMatrix::from_column_major(vec![1.0, 3.0, 2.0, 4.0], 2, 2).unwrap();
        let centered = center_1d(&data);
        // Mean at each point: [2.0, 3.0]
        // centered[0,0] = 1.0 - 2.0 = -1.0
        assert!((centered[(0, 0)] - (-1.0)).abs() < 1e-12);
        assert!((centered[(1, 0)] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_deriv_nderiv0() {
        // nderiv=0 returns the original data (0th derivative = identity)
        let data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let argvals = vec![0.0, 0.5, 1.0];
        let result = deriv_1d(&data, &argvals, 0);
        assert_eq!(result.shape(), data.shape());
        for i in 0..2 {
            for j in 0..3 {
                assert!((result[(i, j)] - data[(i, j)]).abs() < 1e-12);
            }
        }
    }

    // ============== Normalize tests ==============

    #[test]
    fn test_normalize_autoscale() {
        // 3 curves, 4 time points
        let data = FdMatrix::from_column_major(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            3,
            4,
        )
        .unwrap();
        let scaled = normalize(&data, NormalizationMethod::Autoscale);
        // Each column should have mean ≈ 0 and std ≈ 1
        for j in 0..4 {
            let col = scaled.column(j);
            let mean = col.iter().sum::<f64>() / 3.0;
            assert!(
                mean.abs() < 1e-10,
                "column {j} mean should be 0, got {mean}"
            );
            let var = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / 2.0;
            assert!(
                (var - 1.0).abs() < 1e-10,
                "column {j} variance should be 1, got {var}"
            );
        }
    }

    #[test]
    fn test_normalize_pareto() {
        let data =
            FdMatrix::from_column_major(vec![1.0, 5.0, 3.0, 10.0, 20.0, 30.0], 2, 3).unwrap();
        let scaled = normalize(&data, NormalizationMethod::Pareto);
        // Columns should be centered and scaled by sqrt(std)
        for j in 0..3 {
            let col = scaled.column(j);
            let mean = col.iter().sum::<f64>() / 2.0;
            assert!(mean.abs() < 1e-10, "column {j} mean should be 0");
        }
    }

    #[test]
    fn test_normalize_range() {
        let data = FdMatrix::from_column_major(vec![0.0, 10.0, 2.0, 8.0], 2, 2).unwrap();
        let scaled = normalize(&data, NormalizationMethod::Range);
        // Column 0: values [0, 10], range 10, centered [-5, 5], scaled [-0.5, 0.5]
        assert!((scaled[(0, 0)] - (-0.5)).abs() < 1e-10);
        assert!((scaled[(1, 0)] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_curve_center() {
        let data = FdMatrix::from_column_major(vec![1.0, 4.0, 3.0, 6.0, 5.0, 8.0], 2, 3).unwrap();
        let result = normalize(&data, NormalizationMethod::CurveCenter);
        // Row 0: [1, 3, 5], mean=3, centered=[-2, 0, 2]
        assert!((result[(0, 0)] - (-2.0)).abs() < 1e-10);
        assert!((result[(0, 1)] - 0.0).abs() < 1e-10);
        assert!((result[(0, 2)] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_curve_standardize() {
        let data = FdMatrix::from_column_major(vec![1.0, 4.0, 3.0, 6.0, 5.0, 8.0], 2, 3).unwrap();
        let result = normalize(&data, NormalizationMethod::CurveStandardize);
        // Each row should have mean ≈ 0 and std ≈ 1
        for i in 0..2 {
            let row: Vec<f64> = (0..3).map(|j| result[(i, j)]).collect();
            let mean = row.iter().sum::<f64>() / 3.0;
            assert!(mean.abs() < 1e-10, "row {i} mean should be 0");
            let var = row.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / 2.0;
            assert!((var - 1.0).abs() < 1e-10, "row {i} variance should be 1");
        }
    }

    #[test]
    fn test_normalize_curve_range() {
        let data =
            FdMatrix::from_column_major(vec![2.0, 10.0, 4.0, 20.0, 6.0, 30.0], 2, 3).unwrap();
        let result = normalize(&data, NormalizationMethod::CurveRange);
        // Row 0: [2, 4, 6] -> [0.0, 0.5, 1.0]
        assert!((result[(0, 0)] - 0.0).abs() < 1e-10);
        assert!((result[(0, 1)] - 0.5).abs() < 1e-10);
        assert!((result[(0, 2)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_center_matches_center_1d() {
        let data = FdMatrix::from_column_major(vec![1.0, 3.0, 2.0, 4.0, 3.0, 5.0], 2, 3).unwrap();
        let a = center_1d(&data);
        let b = normalize(&data, NormalizationMethod::Center);
        assert_eq!(a.as_slice(), b.as_slice());
    }
}
