//! Smoothing functions for functional data.
//!
//! This module provides kernel-based smoothing methods including
//! Nadaraya-Watson, local linear, and local polynomial regression.

use crate::slice_maybe_parallel;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Gaussian kernel function.
fn gaussian_kernel(u: f64) -> f64 {
    (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Epanechnikov kernel function.
fn epanechnikov_kernel(u: f64) -> f64 {
    if u.abs() <= 1.0 {
        0.75 * (1.0 - u * u)
    } else {
        0.0
    }
}

/// Tri-cube kernel function (used by R's loess()).
fn tricube_kernel(u: f64) -> f64 {
    let abs_u = u.abs();
    if abs_u < 1.0 {
        (1.0 - abs_u.powi(3)).powi(3)
    } else {
        0.0
    }
}

/// Get kernel function by name.
fn get_kernel(kernel_type: &str) -> fn(f64) -> f64 {
    match kernel_type.to_lowercase().as_str() {
        "epanechnikov" | "epan" => epanechnikov_kernel,
        "tricube" | "tri-cube" => tricube_kernel,
        _ => gaussian_kernel,
    }
}

/// Nadaraya-Watson kernel smoother.
///
/// # Arguments
/// * `x` - Predictor values
/// * `y` - Response values
/// * `x_new` - Points at which to evaluate the smoother
/// * `bandwidth` - Kernel bandwidth
/// * `kernel` - Kernel type ("gaussian" or "epanechnikov")
///
/// # Returns
/// Smoothed values at x_new
pub fn nadaraya_watson(
    x: &[f64],
    y: &[f64],
    x_new: &[f64],
    bandwidth: f64,
    kernel: &str,
) -> Vec<f64> {
    let n = x.len();
    if n == 0 || y.len() != n || x_new.is_empty() || bandwidth <= 0.0 {
        return vec![0.0; x_new.len()];
    }

    let kernel_fn = get_kernel(kernel);

    slice_maybe_parallel!(x_new)
        .map(|&x0| {
            let mut num = 0.0;
            let mut denom = 0.0;

            for i in 0..n {
                let u = (x[i] - x0) / bandwidth;
                let w = kernel_fn(u);
                num += w * y[i];
                denom += w;
            }

            if denom > 1e-10 {
                num / denom
            } else {
                0.0
            }
        })
        .collect()
}

/// Local linear regression smoother.
///
/// # Arguments
/// * `x` - Predictor values
/// * `y` - Response values
/// * `x_new` - Points at which to evaluate the smoother
/// * `bandwidth` - Kernel bandwidth
/// * `kernel` - Kernel type
///
/// # Returns
/// Smoothed values at x_new
pub fn local_linear(x: &[f64], y: &[f64], x_new: &[f64], bandwidth: f64, kernel: &str) -> Vec<f64> {
    let n = x.len();
    if n == 0 || y.len() != n || x_new.is_empty() || bandwidth <= 0.0 {
        return vec![0.0; x_new.len()];
    }

    let kernel_fn = get_kernel(kernel);

    slice_maybe_parallel!(x_new)
        .map(|&x0| {
            // Compute weighted moments
            let mut s0 = 0.0;
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            let mut t0 = 0.0;
            let mut t1 = 0.0;

            for i in 0..n {
                let u = (x[i] - x0) / bandwidth;
                let w = kernel_fn(u);
                let d = x[i] - x0;

                s0 += w;
                s1 += w * d;
                s2 += w * d * d;
                t0 += w * y[i];
                t1 += w * y[i] * d;
            }

            // Solve local linear regression
            let det = s0 * s2 - s1 * s1;
            if det.abs() > 1e-10 {
                (s2 * t0 - s1 * t1) / det
            } else if s0 > 1e-10 {
                t0 / s0
            } else {
                0.0
            }
        })
        .collect()
}

/// Accumulate weighted normal equations (X'WX and X'Wy) for local polynomial fit.
fn accumulate_weighted_normal_equations(
    x: &[f64],
    y: &[f64],
    x0: f64,
    bandwidth: f64,
    p: usize,
    kernel_fn: impl Fn(f64) -> f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];

    for i in 0..n {
        let u = (x[i] - x0) / bandwidth;
        let w = kernel_fn(u);
        let d = x[i] - x0;

        for j in 0..p {
            let w_dj = w * d.powi(j as i32);
            for k in 0..p {
                xtx[j * p + k] += w_dj * d.powi(k as i32);
            }
            xty[j] += w_dj * y[i];
        }
    }

    (xtx, xty)
}

/// Solve a linear system using Gaussian elimination with partial pivoting.
/// Returns the solution vector, or a zero vector if the system is singular.
/// Gaussian elimination with partial pivoting (forward pass).
/// Find the row with the largest absolute value in column `col` at or below the diagonal.
fn find_pivot(a: &[f64], p: usize, col: usize) -> usize {
    let mut max_idx = col;
    for j in (col + 1)..p {
        if a[j * p + col].abs() > a[max_idx * p + col].abs() {
            max_idx = j;
        }
    }
    max_idx
}

/// Swap two rows in both the matrix `a` and the RHS vector `b`.
fn swap_rows(a: &mut [f64], b: &mut [f64], p: usize, row_a: usize, row_b: usize) {
    for k in 0..p {
        a.swap(row_a * p + k, row_b * p + k);
    }
    b.swap(row_a, row_b);
}

/// Subtract a scaled copy of the pivot row from all rows below it.
fn eliminate_below(a: &mut [f64], b: &mut [f64], p: usize, pivot_row: usize) {
    let pivot = a[pivot_row * p + pivot_row];
    for j in (pivot_row + 1)..p {
        let factor = a[j * p + pivot_row] / pivot;
        for k in pivot_row..p {
            a[j * p + k] -= factor * a[pivot_row * p + k];
        }
        b[j] -= factor * b[pivot_row];
    }
}

fn forward_eliminate(a: &mut [f64], b: &mut [f64], p: usize) {
    for i in 0..p {
        let max_idx = find_pivot(a, p, i);
        if max_idx != i {
            swap_rows(a, b, p, i, max_idx);
        }

        if a[i * p + i].abs() < 1e-10 {
            continue;
        }

        eliminate_below(a, b, p, i);
    }
}

/// Back substitution for an upper-triangular system.
fn back_substitute(a: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let mut coefs = vec![0.0; p];
    for i in (0..p).rev() {
        let mut sum = b[i];
        for j in (i + 1)..p {
            sum -= a[i * p + j] * coefs[j];
        }
        if a[i * p + i].abs() > 1e-10 {
            coefs[i] = sum / a[i * p + i];
        }
    }
    coefs
}

fn solve_gaussian(a: &mut [f64], b: &mut [f64], p: usize) -> Vec<f64> {
    forward_eliminate(a, b, p);
    back_substitute(a, b, p)
}

/// Solve a linear system Ax = b via Gaussian elimination with partial pivoting.
///
/// Public wrapper for use by other modules (e.g., `fregre_basis_cv`).
/// `a` is a p×p matrix in row-major order, `b` is the RHS vector of length p.
/// Both are modified in place.
pub fn solve_gaussian_pub(a: &mut [f64], b: &mut [f64], p: usize) -> Vec<f64> {
    solve_gaussian(a, b, p)
}

/// Local polynomial regression smoother.
///
/// # Arguments
/// * `x` - Predictor values
/// * `y` - Response values
/// * `x_new` - Points at which to evaluate the smoother
/// * `bandwidth` - Kernel bandwidth
/// * `degree` - Polynomial degree
/// * `kernel` - Kernel type
///
/// # Returns
/// Smoothed values at x_new
pub fn local_polynomial(
    x: &[f64],
    y: &[f64],
    x_new: &[f64],
    bandwidth: f64,
    degree: usize,
    kernel: &str,
) -> Vec<f64> {
    let n = x.len();
    if n == 0 || y.len() != n || x_new.is_empty() || bandwidth <= 0.0 {
        return vec![0.0; x_new.len()];
    }
    if degree == 0 {
        return nadaraya_watson(x, y, x_new, bandwidth, kernel);
    }

    if degree == 1 {
        return local_linear(x, y, x_new, bandwidth, kernel);
    }

    let kernel_fn = get_kernel(kernel);
    let p = degree + 1; // Number of coefficients

    slice_maybe_parallel!(x_new)
        .map(|&x0| {
            let (mut xtx, mut xty) =
                accumulate_weighted_normal_equations(x, y, x0, bandwidth, p, kernel_fn);
            let coefs = solve_gaussian(&mut xtx, &mut xty, p);
            coefs[0]
        })
        .collect()
}

/// k-Nearest Neighbors smoother.
///
/// # Arguments
/// * `x` - Predictor values
/// * `y` - Response values
/// * `x_new` - Points at which to evaluate the smoother
/// * `k` - Number of neighbors
///
/// # Returns
/// Smoothed values at x_new
pub fn knn_smoother(x: &[f64], y: &[f64], x_new: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 || y.len() != n || x_new.is_empty() || k == 0 {
        return vec![0.0; x_new.len()];
    }

    let k = k.min(n);

    slice_maybe_parallel!(x_new)
        .map(|&x0| {
            // Compute distances
            let mut distances: Vec<(usize, f64)> = x
                .iter()
                .enumerate()
                .map(|(i, &xi)| (i, (xi - x0).abs()))
                .collect();

            // Partial sort to get k nearest
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Average of k nearest neighbors
            let sum: f64 = distances.iter().take(k).map(|(i, _)| y[*i]).sum();
            sum / k as f64
        })
        .collect()
}

/// Compute smoothing matrix for Nadaraya-Watson.
///
/// Returns the smoother matrix S such that y_hat = S * y.
pub fn smoothing_matrix_nw(x: &[f64], bandwidth: f64, kernel: &str) -> Vec<f64> {
    let n = x.len();
    if n == 0 || bandwidth <= 0.0 {
        return Vec::new();
    }

    let kernel_fn = get_kernel(kernel);
    let mut s = vec![0.0; n * n];

    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            let u = (x[j] - x[i]) / bandwidth;
            s[i + j * n] = kernel_fn(u);
            row_sum += s[i + j * n];
        }
        if row_sum > 1e-10 {
            for j in 0..n {
                s[i + j * n] /= row_sum;
            }
        }
    }

    s
}

// ─── Cross-Validation for Kernel Smoothers ──────────────────────────────────

/// CV criterion type for bandwidth selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CvCriterion {
    /// Leave-one-out cross-validation (R's `CV.S`).
    Cv,
    /// Generalized cross-validation (R's `GCV.S`).
    Gcv,
}

/// Result of bandwidth optimization.
#[derive(Debug, Clone)]
pub struct OptimBandwidthResult {
    /// Optimal bandwidth.
    pub h_opt: f64,
    /// Criterion used.
    pub criterion: CvCriterion,
    /// Criterion value at optimal h.
    pub value: f64,
}

/// LOO-CV score for a kernel smoother (R's `CV.S`).
///
/// Computes the leave-one-out CV score by zeroing the diagonal of the
/// smoothing matrix, re-normalizing rows, and computing mean squared error.
///
/// # Arguments
/// * `x` — Predictor values
/// * `y` — Response values
/// * `bandwidth` — Kernel bandwidth
/// * `kernel` — Kernel type ("gaussian", "epanechnikov", "tricube")
///
/// # Returns
/// Mean squared LOO prediction error, or `INFINITY` if inputs are invalid.
pub fn cv_smoother(x: &[f64], y: &[f64], bandwidth: f64, kernel: &str) -> f64 {
    let n = x.len();
    if n < 2 || y.len() != n || bandwidth <= 0.0 {
        return f64::INFINITY;
    }

    // Get the smoother matrix S
    let mut s = smoothing_matrix_nw(x, bandwidth, kernel);
    if s.is_empty() {
        return f64::INFINITY;
    }

    // Zero the diagonal → S_cv (LOO smoother)
    for i in 0..n {
        s[i + i * n] = 0.0;
    }

    // Re-normalize each row so it sums to 1
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| s[i + j * n]).sum();
        if row_sum > 1e-10 {
            for j in 0..n {
                s[i + j * n] /= row_sum;
            }
        }
    }

    // Compute y_hat = S_cv * y, then MSE
    let mut mse = 0.0;
    for i in 0..n {
        let y_hat: f64 = (0..n).map(|j| s[i + j * n] * y[j]).sum();
        let resid = y[i] - y_hat;
        mse += resid * resid;
    }
    mse / n as f64
}

/// GCV score for a kernel smoother (R's `GCV.S`).
///
/// Computes `(RSS / n) / (1 - tr(S) / n)²`.
///
/// # Arguments
/// * `x` — Predictor values
/// * `y` — Response values
/// * `bandwidth` — Kernel bandwidth
/// * `kernel` — Kernel type
///
/// # Returns
/// GCV score, or `INFINITY` if inputs are invalid or denominator is near zero.
pub fn gcv_smoother(x: &[f64], y: &[f64], bandwidth: f64, kernel: &str) -> f64 {
    let n = x.len();
    if n < 2 || y.len() != n || bandwidth <= 0.0 {
        return f64::INFINITY;
    }

    let s = smoothing_matrix_nw(x, bandwidth, kernel);
    if s.is_empty() {
        return f64::INFINITY;
    }

    // y_hat = S * y
    let mut rss = 0.0;
    for i in 0..n {
        let y_hat: f64 = (0..n).map(|j| s[i + j * n] * y[j]).sum();
        let resid = y[i] - y_hat;
        rss += resid * resid;
    }

    // trace(S) = sum of diagonal
    let trace_s: f64 = (0..n).map(|i| s[i + i * n]).sum();

    let denom = 1.0 - trace_s / n as f64;
    if denom.abs() < 1e-10 {
        f64::INFINITY
    } else {
        (rss / n as f64) / (denom * denom)
    }
}

/// Bandwidth optimizer for kernel smoothers (R's `optim.np`).
///
/// Grid search over evenly-spaced bandwidths, selecting the one that
/// minimizes the specified criterion (CV or GCV).
///
/// # Arguments
/// * `x` — Predictor values
/// * `y` — Response values
/// * `h_range` — Optional `(h_min, h_max)`. Defaults to `(h_default / 5, h_default * 5)`
///   where `h_default = (x_max - x_min) / n^0.2`.
/// * `criterion` — CV or GCV
/// * `kernel` — Kernel type
/// * `n_grid` — Number of grid points (default: 50)
pub fn optim_bandwidth(
    x: &[f64],
    y: &[f64],
    h_range: Option<(f64, f64)>,
    criterion: CvCriterion,
    kernel: &str,
    n_grid: usize,
) -> OptimBandwidthResult {
    let n = x.len();
    let n_grid = n_grid.max(2);

    // Determine search range
    let (h_min, h_max) = match h_range {
        Some((lo, hi)) if lo > 0.0 && hi > lo => (lo, hi),
        _ => {
            let x_min = x.iter().copied().fold(f64::INFINITY, f64::min);
            let x_max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let h_default = (x_max - x_min) / (n as f64).powf(0.2);
            let h_default = h_default.max(1e-10);
            (h_default / 5.0, h_default * 5.0)
        }
    };

    let score_fn = match criterion {
        CvCriterion::Cv => cv_smoother,
        CvCriterion::Gcv => gcv_smoother,
    };

    let mut best_h = h_min;
    let mut best_score = f64::INFINITY;

    for i in 0..n_grid {
        let h = h_min + (h_max - h_min) * i as f64 / (n_grid - 1) as f64;
        let score = score_fn(x, y, h, kernel);
        if score < best_score {
            best_score = score;
            best_h = h;
        }
    }

    OptimBandwidthResult {
        h_opt: best_h,
        criterion,
        value: best_score,
    }
}

// ─── kNN CV Functions ───────────────────────────────────────────────────────

/// Result of kNN k-selection by cross-validation.
#[derive(Debug, Clone)]
pub struct KnnCvResult {
    /// Optimal k (number of neighbors).
    pub optimal_k: usize,
    /// CV error for each k tested (index 0 = k=1).
    pub cv_errors: Vec<f64>,
}

/// Global LOO-CV for kNN k selection (R's `knn.gcv`).
///
/// For each candidate k, computes LOO prediction error using a
/// kernel-weighted kNN estimator with Epanechnikov kernel.
///
/// # Arguments
/// * `x` — Predictor values
/// * `y` — Response values
/// * `max_k` — Maximum k to test (tests k = 1, 2, …, max_k)
pub fn knn_gcv(x: &[f64], y: &[f64], max_k: usize) -> KnnCvResult {
    let n = x.len();
    let max_k = max_k.min(n.saturating_sub(1)).max(1);

    // Precompute sorted distances from each point to all others
    let mut sorted_neighbors: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, (x[j] - x[i]).abs()))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_neighbors.push(dists);
    }

    let mut cv_errors = Vec::with_capacity(max_k);

    for k in 1..=max_k {
        let mut mse = 0.0;
        for i in 0..n {
            let neighbors = &sorted_neighbors[i];
            // Bandwidth: midpoint between k-th and (k+1)-th NN distances
            let d_k = if k <= neighbors.len() {
                neighbors[k - 1].1
            } else {
                neighbors.last().map_or(1.0, |x| x.1)
            };
            let d_k1 = if k < neighbors.len() {
                neighbors[k].1
            } else {
                d_k * 2.0
            };
            let h = (d_k + d_k1) / 2.0;
            let h = h.max(1e-10);

            // Epanechnikov kernel weighted prediction
            let mut num = 0.0;
            let mut den = 0.0;
            for &(j, dist) in neighbors.iter().take(k) {
                let u = dist / h;
                let w = epanechnikov_kernel(u);
                num += w * y[j];
                den += w;
            }
            let y_hat = if den > 1e-10 { num / den } else { y[i] };
            mse += (y[i] - y_hat).powi(2);
        }
        cv_errors.push(mse / n as f64);
    }

    let optimal_k = cv_errors
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(1, |(i, _)| i + 1);

    KnnCvResult {
        optimal_k,
        cv_errors,
    }
}

/// Local (per-observation) LOO-CV for kNN k selection (R's `knn.lcv`).
///
/// For each observation, independently selects the best k by minimizing
/// the absolute LOO prediction error at that point.
///
/// # Arguments
/// * `x` — Predictor values
/// * `y` — Response values
/// * `max_k` — Maximum k to test
///
/// # Returns
/// Vector of per-observation optimal k values (length n).
pub fn knn_lcv(x: &[f64], y: &[f64], max_k: usize) -> Vec<usize> {
    let n = x.len();
    let max_k = max_k.min(n.saturating_sub(1)).max(1);

    let mut per_obs_k = vec![1usize; n];

    for i in 0..n {
        // Sort neighbors by distance (excluding self)
        let mut neighbors: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, (x[j] - x[i]).abs()))
            .collect();
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut best_k = 1;
        let mut best_err = f64::INFINITY;

        for k in 1..=max_k {
            // Simple kNN average of k nearest neighbors
            let sum: f64 = neighbors.iter().take(k).map(|&(j, _)| y[j]).sum();
            let y_hat = sum / k as f64;
            let err = (y[i] - y_hat).abs();
            if err < best_err {
                best_err = err;
                best_k = k;
            }
        }
        per_obs_k[i] = best_k;
    }

    per_obs_k
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    // ============== Nadaraya-Watson tests ==============

    #[test]
    fn test_nw_constant_data() {
        let x = uniform_grid(20);
        let y: Vec<f64> = vec![5.0; 20];

        let y_smooth = nadaraya_watson(&x, &y, &x, 0.1, "gaussian");

        // Smoothing constant data should return constant
        for &yi in &y_smooth {
            assert!(
                (yi - 5.0).abs() < 0.1,
                "Constant data should remain constant"
            );
        }
    }

    #[test]
    fn test_nw_linear_data() {
        let x = uniform_grid(50);
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        let y_smooth = nadaraya_watson(&x, &y, &x, 0.2, "gaussian");

        // Linear data should be approximately preserved (with some edge effects)
        for i in 10..40 {
            let expected = 2.0 * x[i] + 1.0;
            assert!(
                (y_smooth[i] - expected).abs() < 0.2,
                "Linear trend should be approximately preserved"
            );
        }
    }

    #[test]
    fn test_nw_gaussian_vs_epanechnikov() {
        let x = uniform_grid(30);
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| (2.0 * std::f64::consts::PI * xi).sin())
            .collect();

        let y_gauss = nadaraya_watson(&x, &y, &x, 0.1, "gaussian");
        let y_epan = nadaraya_watson(&x, &y, &x, 0.1, "epanechnikov");

        // Both should produce valid output
        assert_eq!(y_gauss.len(), 30);
        assert_eq!(y_epan.len(), 30);

        // They should be different (different kernels)
        let diff: f64 = y_gauss
            .iter()
            .zip(&y_epan)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.0,
            "Different kernels should give different results"
        );
    }

    #[test]
    fn test_nw_invalid_input() {
        // Empty input
        let result = nadaraya_watson(&[], &[], &[0.5], 0.1, "gaussian");
        assert_eq!(result, vec![0.0]);

        // Mismatched lengths
        let result = nadaraya_watson(&[0.0, 1.0], &[1.0], &[0.5], 0.1, "gaussian");
        assert_eq!(result, vec![0.0]);

        // Zero bandwidth
        let result = nadaraya_watson(&[0.0, 1.0], &[1.0, 2.0], &[0.5], 0.0, "gaussian");
        assert_eq!(result, vec![0.0]);
    }

    // ============== Local linear tests ==============

    #[test]
    fn test_ll_constant_data() {
        let x = uniform_grid(20);
        let y: Vec<f64> = vec![3.0; 20];

        let y_smooth = local_linear(&x, &y, &x, 0.15, "gaussian");

        for &yi in &y_smooth {
            assert!((yi - 3.0).abs() < 0.1, "Constant should remain constant");
        }
    }

    #[test]
    fn test_ll_linear_data_exact() {
        let x = uniform_grid(30);
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi + 2.0).collect();

        let y_smooth = local_linear(&x, &y, &x, 0.2, "gaussian");

        // Local linear should fit linear data exactly (in interior)
        for i in 5..25 {
            let expected = 3.0 * x[i] + 2.0;
            assert!(
                (y_smooth[i] - expected).abs() < 0.1,
                "Local linear should fit linear data well"
            );
        }
    }

    #[test]
    fn test_ll_invalid_input() {
        let result = local_linear(&[], &[], &[0.5], 0.1, "gaussian");
        assert_eq!(result, vec![0.0]);

        let result = local_linear(&[0.0, 1.0], &[1.0, 2.0], &[0.5], -0.1, "gaussian");
        assert_eq!(result, vec![0.0]);
    }

    // ============== Local polynomial tests ==============

    #[test]
    fn test_lp_degree1_equals_local_linear() {
        let x = uniform_grid(25);
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let y_ll = local_linear(&x, &y, &x, 0.15, "gaussian");
        let y_lp = local_polynomial(&x, &y, &x, 0.15, 1, "gaussian");

        for i in 0..25 {
            assert!(
                (y_ll[i] - y_lp[i]).abs() < 1e-10,
                "Degree 1 should equal local linear"
            );
        }
    }

    #[test]
    fn test_lp_quadratic_data() {
        let x = uniform_grid(40);
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let y_smooth = local_polynomial(&x, &y, &x, 0.15, 2, "gaussian");

        // Local quadratic should fit quadratic data well in interior
        for i in 8..32 {
            let expected = x[i] * x[i];
            assert!(
                (y_smooth[i] - expected).abs() < 0.1,
                "Local quadratic should fit quadratic data"
            );
        }
    }

    #[test]
    fn test_lp_invalid_input() {
        // Zero degree delegates to Nadaraya-Watson (not zeros)
        let result = local_polynomial(&[0.0, 1.0], &[1.0, 2.0], &[0.5], 0.1, 0, "gaussian");
        let nw = nadaraya_watson(&[0.0, 1.0], &[1.0, 2.0], &[0.5], 0.1, "gaussian");
        assert_eq!(result, nw);

        // Empty input
        let result = local_polynomial(&[], &[], &[0.5], 0.1, 2, "gaussian");
        assert_eq!(result, vec![0.0]);
    }

    // ============== KNN smoother tests ==============

    #[test]
    fn test_knn_k1_nearest() {
        let x = vec![0.0, 0.5, 1.0];
        let y = vec![1.0, 2.0, 3.0];

        let result = knn_smoother(&x, &y, &[0.1, 0.6, 0.9], 1);

        // k=1 should return the nearest neighbor's y value
        assert!((result[0] - 1.0).abs() < 1e-10, "0.1 nearest to 0.0 -> 1.0");
        assert!((result[1] - 2.0).abs() < 1e-10, "0.6 nearest to 0.5 -> 2.0");
        assert!((result[2] - 3.0).abs() < 1e-10, "0.9 nearest to 1.0 -> 3.0");
    }

    #[test]
    fn test_knn_k_equals_n_is_mean() {
        let x = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let expected_mean = 3.0;

        let result = knn_smoother(&x, &y, &[0.5], 5);

        assert!(
            (result[0] - expected_mean).abs() < 1e-10,
            "k=n should return mean"
        );
    }

    #[test]
    fn test_knn_invalid_input() {
        let result = knn_smoother(&[], &[], &[0.5], 3);
        assert_eq!(result, vec![0.0]);

        let result = knn_smoother(&[0.0, 1.0], &[1.0, 2.0], &[0.5], 0);
        assert_eq!(result, vec![0.0]);
    }

    // ============== Smoothing matrix tests ==============

    #[test]
    fn test_smoothing_matrix_row_stochastic() {
        let x = uniform_grid(10);
        let s = smoothing_matrix_nw(&x, 0.2, "gaussian");

        assert_eq!(s.len(), 100);

        // Each row should sum to 1 (row stochastic)
        for i in 0..10 {
            let row_sum: f64 = (0..10).map(|j| s[i + j * 10]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "Row {} should sum to 1, got {}",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_smoothing_matrix_invalid_input() {
        let result = smoothing_matrix_nw(&[], 0.1, "gaussian");
        assert!(result.is_empty());

        let result = smoothing_matrix_nw(&[0.0, 1.0], 0.0, "gaussian");
        assert!(result.is_empty());
    }

    #[test]
    fn test_nan_nw_no_panic() {
        let x = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let mut y = vec![0.0, 1.0, 2.0, 1.0, 0.0];
        y[2] = f64::NAN;
        let result = nadaraya_watson(&x, &y, &x, 0.3, "gaussian");
        assert_eq!(result.len(), x.len());
        // NaN should propagate but not panic
    }

    #[test]
    fn test_n1_smoother() {
        // Single data point
        let x = vec![0.5];
        let y = vec![3.0];
        let x_new = vec![0.5];
        let result = nadaraya_watson(&x, &y, &x_new, 0.3, "gaussian");
        assert_eq!(result.len(), 1);
        assert!(
            (result[0] - 3.0).abs() < 1e-6,
            "Single point smoother should return the value"
        );
    }

    #[test]
    fn test_duplicate_x_smoother() {
        // Duplicate x values
        let x = vec![0.0, 0.0, 0.5, 1.0, 1.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x_new = vec![0.0, 0.5, 1.0];
        let result = nadaraya_watson(&x, &y, &x_new, 0.3, "gaussian");
        assert_eq!(result.len(), 3);
        for v in &result {
            assert!(v.is_finite());
        }
    }

    // ============== CV smoother tests ==============

    #[test]
    fn test_cv_smoother_linear_data() {
        let x = uniform_grid(30);
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let cv = cv_smoother(&x, &y, 0.2, "gaussian");
        assert!(cv.is_finite());
        assert!(cv >= 0.0);
        assert!(cv < 1.0, "CV error for smooth linear data should be small");
    }

    #[test]
    fn test_cv_smoother_invalid() {
        assert_eq!(cv_smoother(&[], &[], 0.1, "gaussian"), f64::INFINITY);
        assert_eq!(
            cv_smoother(&[0.0, 1.0], &[1.0, 2.0], -0.1, "gaussian"),
            f64::INFINITY
        );
    }

    #[test]
    fn test_gcv_smoother_linear_data() {
        let x = uniform_grid(30);
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let gcv = gcv_smoother(&x, &y, 0.2, "gaussian");
        assert!(gcv.is_finite());
        assert!(gcv >= 0.0);
    }

    #[test]
    fn test_gcv_smoother_invalid() {
        assert_eq!(gcv_smoother(&[], &[], 0.1, "gaussian"), f64::INFINITY);
    }

    #[test]
    fn test_optim_bandwidth_returns_valid() {
        let x = uniform_grid(30);
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| (2.0 * std::f64::consts::PI * xi).sin())
            .collect();

        let result = optim_bandwidth(&x, &y, None, CvCriterion::Gcv, "gaussian", 20);
        assert!(result.h_opt > 0.0);
        assert!(result.value.is_finite());
        assert_eq!(result.criterion, CvCriterion::Gcv);
    }

    #[test]
    fn test_optim_bandwidth_cv_vs_gcv() {
        let x = uniform_grid(25);
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let cv_result = optim_bandwidth(&x, &y, None, CvCriterion::Cv, "gaussian", 20);
        let gcv_result = optim_bandwidth(&x, &y, None, CvCriterion::Gcv, "gaussian", 20);

        assert!(cv_result.h_opt > 0.0);
        assert!(gcv_result.h_opt > 0.0);
    }

    #[test]
    fn test_optim_bandwidth_custom_range() {
        let x = uniform_grid(20);
        let y: Vec<f64> = x.to_vec();
        let result = optim_bandwidth(&x, &y, Some((0.05, 0.5)), CvCriterion::Cv, "gaussian", 10);
        assert!(result.h_opt >= 0.05);
        assert!(result.h_opt <= 0.5);
    }

    // ============== kNN CV tests ==============

    #[test]
    fn test_knn_gcv_returns_valid() {
        let x = uniform_grid(20);
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let result = knn_gcv(&x, &y, 10);
        assert!(result.optimal_k >= 1);
        assert!(result.optimal_k <= 10);
        assert_eq!(result.cv_errors.len(), 10);
        for &e in &result.cv_errors {
            assert!(e.is_finite());
            assert!(e >= 0.0);
        }
    }

    #[test]
    fn test_knn_gcv_constant_data() {
        let x = uniform_grid(15);
        let y = vec![5.0; 15];
        let result = knn_gcv(&x, &y, 5);
        // For constant data, error should be near zero for any k
        for &e in &result.cv_errors {
            assert!(e < 0.01, "Constant data: CV error should be near zero");
        }
    }

    #[test]
    fn test_knn_lcv_returns_valid() {
        let x = uniform_grid(15);
        let y: Vec<f64> = x.to_vec();

        let result = knn_lcv(&x, &y, 5);
        assert_eq!(result.len(), 15);
        for &k in &result {
            assert!(k >= 1);
            assert!(k <= 5);
        }
    }

    #[test]
    fn test_knn_lcv_constant_data() {
        let x = uniform_grid(10);
        let y = vec![3.0; 10];
        let result = knn_lcv(&x, &y, 5);
        assert_eq!(result.len(), 10);
        // For constant data, all k values should give zero error
        // so k=1 is optimal (first tested)
        for &k in &result {
            assert!(k >= 1);
        }
    }

    // ============== Tricube kernel tests ==============

    #[test]
    fn test_tricube_kernel_values() {
        // At u=0, tricube should be 1.0
        let k0 = tricube_kernel(0.0);
        assert!((k0 - 1.0).abs() < 1e-10, "tricube(0) should be 1.0");

        // At |u| >= 1, tricube should be 0.0
        assert_eq!(tricube_kernel(1.0), 0.0, "tricube(1) should be 0");
        assert_eq!(tricube_kernel(-1.0), 0.0, "tricube(-1) should be 0");
        assert_eq!(tricube_kernel(2.0), 0.0, "tricube(2) should be 0");

        // At u=0.5, should be positive and less than 1
        let k05 = tricube_kernel(0.5);
        assert!(k05 > 0.0 && k05 < 1.0, "tricube(0.5) should be in (0, 1)");
    }

    #[test]
    fn test_nw_tricube_constant_data() {
        let x = uniform_grid(20);
        let y = vec![5.0; 20];

        let y_smooth = nadaraya_watson(&x, &y, &x, 0.2, "tricube");

        for &yi in &y_smooth {
            assert!(
                (yi - 5.0).abs() < 0.1,
                "Tricube NW: constant data should remain constant"
            );
        }
    }

    #[test]
    fn test_nw_tricube_linear_data() {
        let x = uniform_grid(50);
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        let y_smooth = nadaraya_watson(&x, &y, &x, 0.2, "tricube");

        // Interior points should be approximately correct
        for i in 10..40 {
            let expected = 2.0 * x[i] + 1.0;
            assert!(
                (y_smooth[i] - expected).abs() < 0.3,
                "Tricube NW: linear trend should be approximately preserved at i={i}"
            );
        }
    }

    #[test]
    fn test_nw_tricube_vs_gaussian() {
        let x = uniform_grid(30);
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| (2.0 * std::f64::consts::PI * xi).sin())
            .collect();

        let y_gauss = nadaraya_watson(&x, &y, &x, 0.15, "gaussian");
        let y_tri = nadaraya_watson(&x, &y, &x, 0.15, "tricube");

        assert_eq!(y_gauss.len(), y_tri.len());

        // Both should produce valid output
        assert!(y_tri.iter().all(|v| v.is_finite()));

        // They should be different
        let diff: f64 = y_gauss.iter().zip(&y_tri).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.0,
            "Gaussian and tricube kernels should give different results"
        );
    }

    #[test]
    fn test_nw_tricube_vs_epanechnikov() {
        let x = uniform_grid(30);
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| (2.0 * std::f64::consts::PI * xi).sin())
            .collect();

        let y_epan = nadaraya_watson(&x, &y, &x, 0.15, "epanechnikov");
        let y_tri = nadaraya_watson(&x, &y, &x, 0.15, "tricube");

        // Both compact support kernels should produce finite output
        assert!(y_epan.iter().all(|v| v.is_finite()));
        assert!(y_tri.iter().all(|v| v.is_finite()));

        // Should differ since kernel shapes are different
        let diff: f64 = y_epan.iter().zip(&y_tri).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.0,
            "Epanechnikov and tricube should give different results"
        );
    }

    #[test]
    fn test_ll_tricube_constant_data() {
        let x = uniform_grid(20);
        let y = vec![3.0; 20];

        let y_smooth = local_linear(&x, &y, &x, 0.2, "tricube");

        for &yi in &y_smooth {
            assert!(
                (yi - 3.0).abs() < 0.1,
                "Tricube LL: constant should remain constant"
            );
        }
    }

    #[test]
    fn test_ll_tricube_linear_data() {
        let x = uniform_grid(30);
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi + 2.0).collect();

        let y_smooth = local_linear(&x, &y, &x, 0.2, "tricube");

        // Local linear should fit linear data well in the interior
        for i in 5..25 {
            let expected = 3.0 * x[i] + 2.0;
            assert!(
                (y_smooth[i] - expected).abs() < 0.2,
                "Tricube LL: should fit linear data well at i={i}"
            );
        }
    }

    #[test]
    fn test_ll_tricube_vs_gaussian() {
        let x = uniform_grid(30);
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let y_gauss = local_linear(&x, &y, &x, 0.15, "gaussian");
        let y_tri = local_linear(&x, &y, &x, 0.15, "tricube");

        assert_eq!(y_gauss.len(), y_tri.len());
        assert!(y_tri.iter().all(|v| v.is_finite()));

        let diff: f64 = y_gauss.iter().zip(&y_tri).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.0,
            "Gaussian and tricube local linear should differ"
        );
    }

    #[test]
    fn test_lp_tricube_quadratic() {
        let x = uniform_grid(40);
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let y_smooth = local_polynomial(&x, &y, &x, 0.15, 2, "tricube");

        // Local quadratic with tricube should fit well in interior
        for i in 8..32 {
            let expected = x[i] * x[i];
            assert!(
                (y_smooth[i] - expected).abs() < 0.15,
                "Tricube LP: should fit quadratic data at i={i}"
            );
        }
    }

    #[test]
    fn test_get_kernel_tricube_aliases() {
        // "tricube" and "tri-cube" should both resolve to the tricube kernel
        let k1 = get_kernel("tricube");
        let k2 = get_kernel("tri-cube");

        let test_val = 0.5;
        assert!(
            (k1(test_val) - k2(test_val)).abs() < 1e-15,
            "Both tricube aliases should give the same result"
        );
    }

    #[test]
    fn test_smoothing_matrix_tricube() {
        let x = uniform_grid(10);
        let s = smoothing_matrix_nw(&x, 0.2, "tricube");

        assert_eq!(s.len(), 100);

        // Each row should sum to 1 (row stochastic)
        for i in 0..10 {
            let row_sum: f64 = (0..10).map(|j| s[i + j * 10]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "Tricube: row {} should sum to 1, got {}",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_cv_smoother_tricube() {
        let x = uniform_grid(30);
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let cv = cv_smoother(&x, &y, 0.2, "tricube");
        assert!(cv.is_finite());
        assert!(cv >= 0.0);
    }

    #[test]
    fn test_optim_bandwidth_tricube() {
        let x = uniform_grid(25);
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| (2.0 * std::f64::consts::PI * xi).sin())
            .collect();

        let result = optim_bandwidth(&x, &y, None, CvCriterion::Gcv, "tricube", 20);
        assert!(result.h_opt > 0.0);
        assert!(result.value.is_finite());
    }
}
