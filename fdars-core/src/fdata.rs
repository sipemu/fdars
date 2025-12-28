//! Functional data operations: mean, center, derivatives, norms, and geometric median.

use crate::helpers::{simpsons_weights, simpsons_weights_2d};
use rayon::prelude::*;

/// Compute the mean function across all samples (1D).
///
/// # Arguments
/// * `data` - Column-major matrix (n_samples x n_points)
/// * `nrow` - Number of samples
/// * `ncol` - Number of evaluation points
///
/// # Returns
/// Mean function values at each evaluation point
pub fn mean_1d(data: &[f64], nrow: usize, ncol: usize) -> Vec<f64> {
    if nrow == 0 || ncol == 0 || data.len() != nrow * ncol {
        return Vec::new();
    }

    (0..ncol)
        .into_par_iter()
        .map(|j| {
            let mut sum = 0.0;
            for i in 0..nrow {
                sum += data[i + j * nrow];
            }
            sum / nrow as f64
        })
        .collect()
}

/// Compute the mean function for 2D surfaces.
///
/// Data is stored as n x (m1*m2) matrix where each row is a flattened surface.
pub fn mean_2d(data: &[f64], nrow: usize, ncol: usize) -> Vec<f64> {
    // Same computation as 1D - just compute pointwise mean
    mean_1d(data, nrow, ncol)
}

/// Center functional data by subtracting the mean function.
///
/// # Arguments
/// * `data` - Column-major matrix (n_samples x n_points)
/// * `nrow` - Number of samples
/// * `ncol` - Number of evaluation points
///
/// # Returns
/// Centered data matrix (column-major)
pub fn center_1d(data: &[f64], nrow: usize, ncol: usize) -> Vec<f64> {
    if nrow == 0 || ncol == 0 || data.len() != nrow * ncol {
        return Vec::new();
    }

    // First compute the mean for each column (parallelized)
    let means: Vec<f64> = (0..ncol)
        .into_par_iter()
        .map(|j| {
            let mut sum = 0.0;
            for i in 0..nrow {
                sum += data[i + j * nrow];
            }
            sum / nrow as f64
        })
        .collect();

    // Create centered data (parallelized by column)
    let mut centered = vec![0.0; nrow * ncol];
    for j in 0..ncol {
        for i in 0..nrow {
            centered[i + j * nrow] = data[i + j * nrow] - means[j];
        }
    }

    centered
}

/// Compute Lp norm for each sample.
///
/// # Arguments
/// * `data` - Column-major matrix (n_samples x n_points)
/// * `nrow` - Number of samples
/// * `ncol` - Number of evaluation points
/// * `argvals` - Evaluation points for integration
/// * `p` - Order of the norm (e.g., 2.0 for L2)
///
/// # Returns
/// Vector of Lp norms for each sample
pub fn norm_lp_1d(data: &[f64], nrow: usize, ncol: usize, argvals: &[f64], p: f64) -> Vec<f64> {
    if nrow == 0 || ncol == 0 || argvals.len() != ncol || data.len() != nrow * ncol {
        return Vec::new();
    }

    let weights = simpsons_weights(argvals);

    (0..nrow)
        .into_par_iter()
        .map(|i| {
            let mut integral = 0.0;
            for j in 0..ncol {
                let val = data[i + j * nrow].abs().powf(p);
                integral += val * weights[j];
            }
            integral.powf(1.0 / p)
        })
        .collect()
}

/// Compute numerical derivative of functional data (parallelized over rows).
///
/// # Arguments
/// * `data` - Column-major matrix (n_samples x n_points)
/// * `nrow` - Number of samples
/// * `ncol` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `nderiv` - Order of derivative
///
/// # Returns
/// Derivative data matrix (column-major)
pub fn deriv_1d(
    data: &[f64],
    nrow: usize,
    ncol: usize,
    argvals: &[f64],
    nderiv: usize,
) -> Vec<f64> {
    if nrow == 0 || ncol == 0 || argvals.len() != ncol || nderiv < 1 || data.len() != nrow * ncol {
        return vec![0.0; nrow * ncol];
    }

    let mut current = data.to_vec();

    // Pre-compute step sizes for central differences
    let h0 = argvals[1] - argvals[0];
    let hn = argvals[ncol - 1] - argvals[ncol - 2];
    let h_central: Vec<f64> = (1..(ncol - 1))
        .map(|j| argvals[j + 1] - argvals[j - 1])
        .collect();

    for _ in 0..nderiv {
        // Compute derivative for each row in parallel
        let deriv: Vec<f64> = (0..nrow)
            .into_par_iter()
            .flat_map(|i| {
                let mut row_deriv = vec![0.0; ncol];

                // Forward difference at left boundary
                row_deriv[0] = (current[i + nrow] - current[i]) / h0;

                // Central differences for interior points
                for j in 1..(ncol - 1) {
                    row_deriv[j] = (current[i + (j + 1) * nrow] - current[i + (j - 1) * nrow])
                        / h_central[j - 1];
                }

                // Backward difference at right boundary
                row_deriv[ncol - 1] =
                    (current[i + (ncol - 1) * nrow] - current[i + (ncol - 2) * nrow]) / hn;

                row_deriv
            })
            .collect();

        // Reorder from row-major to column-major order
        current = vec![0.0; nrow * ncol];
        for i in 0..nrow {
            for j in 0..ncol {
                current[i + j * nrow] = deriv[i * ncol + j];
            }
        }
    }

    current
}

/// Result of 2D partial derivatives.
pub struct Deriv2DResult {
    /// Partial derivative with respect to s (∂f/∂s)
    pub ds: Vec<f64>,
    /// Partial derivative with respect to t (∂f/∂t)
    pub dt: Vec<f64>,
    /// Mixed partial derivative (∂²f/∂s∂t)
    pub dsdt: Vec<f64>,
}

/// Compute 2D partial derivatives for surface data.
///
/// For a surface f(s,t), computes:
/// - ds: partial derivative with respect to s (∂f/∂s)
/// - dt: partial derivative with respect to t (∂f/∂t)
/// - dsdt: mixed partial derivative (∂²f/∂s∂t)
///
/// # Arguments
/// * `data` - Column-major matrix, n surfaces, each stored as m1*m2 values
/// * `n` - Number of surfaces
/// * `argvals_s` - Grid points in s direction (length m1)
/// * `argvals_t` - Grid points in t direction (length m2)
/// * `m1` - Grid size in s direction
/// * `m2` - Grid size in t direction
pub fn deriv_2d(
    data: &[f64],
    n: usize,
    argvals_s: &[f64],
    argvals_t: &[f64],
    m1: usize,
    m2: usize,
) -> Option<Deriv2DResult> {
    let ncol = m1 * m2;
    if n == 0 || ncol == 0 || argvals_s.len() != m1 || argvals_t.len() != m2 {
        return None;
    }

    // Helper to access data: surface i, position (si, ti)
    let get_val = |i: usize, si: usize, ti: usize| -> f64 { data[i + (si + ti * m1) * n] };

    // Pre-compute step sizes for s direction
    let hs: Vec<f64> = (0..m1)
        .map(|j| {
            if j == 0 {
                argvals_s[1] - argvals_s[0]
            } else if j == m1 - 1 {
                argvals_s[m1 - 1] - argvals_s[m1 - 2]
            } else {
                argvals_s[j + 1] - argvals_s[j - 1]
            }
        })
        .collect();

    // Pre-compute step sizes for t direction
    let ht: Vec<f64> = (0..m2)
        .map(|j| {
            if j == 0 {
                argvals_t[1] - argvals_t[0]
            } else if j == m2 - 1 {
                argvals_t[m2 - 1] - argvals_t[m2 - 2]
            } else {
                argvals_t[j + 1] - argvals_t[j - 1]
            }
        })
        .collect();

    // Compute all derivatives in parallel over surfaces
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut ds = vec![0.0; m1 * m2];
            let mut dt = vec![0.0; m1 * m2];
            let mut dsdt = vec![0.0; m1 * m2];

            for ti in 0..m2 {
                for si in 0..m1 {
                    let idx = si + ti * m1;

                    // ∂f/∂s using finite differences
                    if si == 0 {
                        ds[idx] = (get_val(i, 1, ti) - get_val(i, 0, ti)) / hs[si];
                    } else if si == m1 - 1 {
                        ds[idx] = (get_val(i, m1 - 1, ti) - get_val(i, m1 - 2, ti)) / hs[si];
                    } else {
                        ds[idx] = (get_val(i, si + 1, ti) - get_val(i, si - 1, ti)) / hs[si];
                    }

                    // ∂f/∂t using finite differences
                    if ti == 0 {
                        dt[idx] = (get_val(i, si, 1) - get_val(i, si, 0)) / ht[ti];
                    } else if ti == m2 - 1 {
                        dt[idx] = (get_val(i, si, m2 - 1) - get_val(i, si, m2 - 2)) / ht[ti];
                    } else {
                        dt[idx] = (get_val(i, si, ti + 1) - get_val(i, si, ti - 1)) / ht[ti];
                    }

                    // ∂²f/∂s∂t (mixed partial)
                    let denom = hs[si] * ht[ti];

                    if si == 0 && ti == 0 {
                        dsdt[idx] = (get_val(i, 1, 1) - get_val(i, 0, 1) - get_val(i, 1, 0)
                            + get_val(i, 0, 0))
                            / denom;
                    } else if si == m1 - 1 && ti == 0 {
                        dsdt[idx] =
                            (get_val(i, m1 - 1, 1) - get_val(i, m1 - 2, 1) - get_val(i, m1 - 1, 0)
                                + get_val(i, m1 - 2, 0))
                                / denom;
                    } else if si == 0 && ti == m2 - 1 {
                        dsdt[idx] =
                            (get_val(i, 1, m2 - 1) - get_val(i, 0, m2 - 1) - get_val(i, 1, m2 - 2)
                                + get_val(i, 0, m2 - 2))
                                / denom;
                    } else if si == m1 - 1 && ti == m2 - 1 {
                        dsdt[idx] = (get_val(i, m1 - 1, m2 - 1)
                            - get_val(i, m1 - 2, m2 - 1)
                            - get_val(i, m1 - 1, m2 - 2)
                            + get_val(i, m1 - 2, m2 - 2))
                            / denom;
                    } else if si == 0 {
                        dsdt[idx] =
                            (get_val(i, 1, ti + 1) - get_val(i, 0, ti + 1) - get_val(i, 1, ti - 1)
                                + get_val(i, 0, ti - 1))
                                / denom;
                    } else if si == m1 - 1 {
                        dsdt[idx] = (get_val(i, m1 - 1, ti + 1)
                            - get_val(i, m1 - 2, ti + 1)
                            - get_val(i, m1 - 1, ti - 1)
                            + get_val(i, m1 - 2, ti - 1))
                            / denom;
                    } else if ti == 0 {
                        dsdt[idx] =
                            (get_val(i, si + 1, 1) - get_val(i, si - 1, 1) - get_val(i, si + 1, 0)
                                + get_val(i, si - 1, 0))
                                / denom;
                    } else if ti == m2 - 1 {
                        dsdt[idx] = (get_val(i, si + 1, m2 - 1)
                            - get_val(i, si - 1, m2 - 1)
                            - get_val(i, si + 1, m2 - 2)
                            + get_val(i, si - 1, m2 - 2))
                            / denom;
                    } else {
                        dsdt[idx] = (get_val(i, si + 1, ti + 1)
                            - get_val(i, si - 1, ti + 1)
                            - get_val(i, si + 1, ti - 1)
                            + get_val(i, si - 1, ti - 1))
                            / denom;
                    }
                }
            }

            (ds, dt, dsdt)
        })
        .collect();

    // Convert to column-major matrices
    let mut ds_mat = vec![0.0; n * ncol];
    let mut dt_mat = vec![0.0; n * ncol];
    let mut dsdt_mat = vec![0.0; n * ncol];

    for i in 0..n {
        for j in 0..ncol {
            ds_mat[i + j * n] = results[i].0[j];
            dt_mat[i + j * n] = results[i].1[j];
            dsdt_mat[i + j * n] = results[i].2[j];
        }
    }

    Some(Deriv2DResult {
        ds: ds_mat,
        dt: dt_mat,
        dsdt: dsdt_mat,
    })
}

/// Compute the geometric median (L1 median) of functional data using Weiszfeld's algorithm.
///
/// The geometric median minimizes sum of L2 distances to all curves.
///
/// # Arguments
/// * `data` - Column-major matrix (n_samples x n_points)
/// * `nrow` - Number of samples
/// * `ncol` - Number of evaluation points
/// * `argvals` - Evaluation points for integration
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
pub fn geometric_median_1d(
    data: &[f64],
    nrow: usize,
    ncol: usize,
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    if nrow == 0 || ncol == 0 || argvals.len() != ncol || data.len() != nrow * ncol {
        return Vec::new();
    }

    let weights = simpsons_weights(argvals);

    // Initialize with the mean
    let mut median: Vec<f64> = (0..ncol)
        .map(|j| {
            let mut sum = 0.0;
            for i in 0..nrow {
                sum += data[i + j * nrow];
            }
            sum / nrow as f64
        })
        .collect();

    for _ in 0..max_iter {
        // Compute distances from current median to all curves
        let distances: Vec<f64> = (0..nrow)
            .map(|i| {
                let mut dist_sq = 0.0;
                for j in 0..ncol {
                    let diff = data[i + j * nrow] - median[j];
                    dist_sq += diff * diff * weights[j];
                }
                dist_sq.sqrt()
            })
            .collect();

        // Compute weights (1/distance), handling zero distances
        let eps = 1e-10;
        let inv_distances: Vec<f64> = distances
            .iter()
            .map(|d| if *d > eps { 1.0 / d } else { 1.0 / eps })
            .collect();

        let sum_inv_dist: f64 = inv_distances.iter().sum();

        // Update median using Weiszfeld iteration
        let new_median: Vec<f64> = (0..ncol)
            .map(|j| {
                let mut weighted_sum = 0.0;
                for i in 0..nrow {
                    weighted_sum += data[i + j * nrow] * inv_distances[i];
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
            / ncol as f64;

        median = new_median;

        if diff < tol {
            break;
        }
    }

    median
}

/// Compute the geometric median for 2D functional data.
///
/// Data is stored as n x (m1*m2) matrix where each row is a flattened surface.
pub fn geometric_median_2d(
    data: &[f64],
    nrow: usize,
    ncol: usize,
    argvals_s: &[f64],
    argvals_t: &[f64],
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let expected_cols = argvals_s.len() * argvals_t.len();
    if nrow == 0 || ncol == 0 || ncol != expected_cols || data.len() != nrow * ncol {
        return Vec::new();
    }

    let weights = simpsons_weights_2d(argvals_s, argvals_t);

    // Initialize with the mean
    let mut median: Vec<f64> = (0..ncol)
        .map(|j| {
            let mut sum = 0.0;
            for i in 0..nrow {
                sum += data[i + j * nrow];
            }
            sum / nrow as f64
        })
        .collect();

    for _ in 0..max_iter {
        // Compute distances from current median to all surfaces
        let distances: Vec<f64> = (0..nrow)
            .map(|i| {
                let mut dist_sq = 0.0;
                for j in 0..ncol {
                    let diff = data[i + j * nrow] - median[j];
                    dist_sq += diff * diff * weights[j];
                }
                dist_sq.sqrt()
            })
            .collect();

        // Compute weights (1/distance), handling zero distances
        let eps = 1e-10;
        let inv_distances: Vec<f64> = distances
            .iter()
            .map(|d| if *d > eps { 1.0 / d } else { 1.0 / eps })
            .collect();

        let sum_inv_dist: f64 = inv_distances.iter().sum();

        // Update median using Weiszfeld iteration
        let new_median: Vec<f64> = (0..ncol)
            .map(|j| {
                let mut weighted_sum = 0.0;
                for i in 0..nrow {
                    weighted_sum += data[i + j * nrow] * inv_distances[i];
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
            / ncol as f64;

        median = new_median;

        if diff < tol {
            break;
        }
    }

    median
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn uniform_grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    // ============== Mean tests ==============

    #[test]
    fn test_mean_1d() {
        // 2 samples, 3 points each
        // Sample 1: [1, 2, 3]
        // Sample 2: [3, 4, 5]
        // Mean should be [2, 3, 4]
        let data = vec![1.0, 3.0, 2.0, 4.0, 3.0, 5.0]; // column-major
        let mean = mean_1d(&data, 2, 3);
        assert_eq!(mean, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_mean_1d_single_sample() {
        let data = vec![1.0, 2.0, 3.0];
        let mean = mean_1d(&data, 1, 3);
        assert_eq!(mean, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mean_1d_invalid() {
        assert!(mean_1d(&[], 0, 0).is_empty());
        assert!(mean_1d(&[1.0], 1, 2).is_empty()); // wrong data length
    }

    #[test]
    fn test_mean_2d_delegates() {
        let data = vec![1.0, 3.0, 2.0, 4.0];
        let mean1d = mean_1d(&data, 2, 2);
        let mean2d = mean_2d(&data, 2, 2);
        assert_eq!(mean1d, mean2d);
    }

    // ============== Center tests ==============

    #[test]
    fn test_center_1d() {
        let data = vec![1.0, 3.0, 2.0, 4.0, 3.0, 5.0]; // column-major
        let centered = center_1d(&data, 2, 3);
        // Mean is [2, 3, 4], so centered should be [-1, 1, -1, 1, -1, 1]
        assert_eq!(centered, vec![-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
    }

    #[test]
    fn test_center_1d_mean_zero() {
        let data = vec![1.0, 3.0, 2.0, 4.0, 3.0, 5.0];
        let centered = center_1d(&data, 2, 3);
        let centered_mean = mean_1d(&centered, 2, 3);
        for m in centered_mean {
            assert!(m.abs() < 1e-10, "Centered data should have zero mean");
        }
    }

    #[test]
    fn test_center_1d_invalid() {
        assert!(center_1d(&[], 0, 0).is_empty());
    }

    // ============== Norm tests ==============

    #[test]
    fn test_norm_lp_1d_constant() {
        // Constant function 2 on [0, 1] has L2 norm = 2
        let argvals = uniform_grid(21);
        let mut data = vec![0.0; 21];
        for j in 0..21 {
            data[j] = 2.0;
        }
        let norms = norm_lp_1d(&data, 1, 21, &argvals, 2.0);
        assert_eq!(norms.len(), 1);
        assert!((norms[0] - 2.0).abs() < 0.1, "L2 norm of constant 2 should be 2");
    }

    #[test]
    fn test_norm_lp_1d_sine() {
        // L2 norm of sin(pi*x) on [0, 1] = sqrt(0.5)
        let argvals = uniform_grid(101);
        let mut data = vec![0.0; 101];
        for j in 0..101 {
            data[j] = (PI * argvals[j]).sin();
        }
        let norms = norm_lp_1d(&data, 1, 101, &argvals, 2.0);
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
        assert!(norm_lp_1d(&[], 0, 0, &[], 2.0).is_empty());
    }

    // ============== Derivative tests ==============

    #[test]
    fn test_deriv_1d_linear() {
        // Derivative of linear function x should be 1
        let argvals = uniform_grid(21);
        let mut data = vec![0.0; 21];
        for j in 0..21 {
            data[j] = argvals[j];
        }
        let deriv = deriv_1d(&data, 1, 21, &argvals, 1);
        // Interior points should have derivative close to 1
        for j in 2..19 {
            assert!(
                (deriv[j] - 1.0).abs() < 0.1,
                "Derivative of x should be 1"
            );
        }
    }

    #[test]
    fn test_deriv_1d_quadratic() {
        // Derivative of x^2 should be 2x
        let argvals = uniform_grid(51);
        let mut data = vec![0.0; 51];
        for j in 0..51 {
            data[j] = argvals[j] * argvals[j];
        }
        let deriv = deriv_1d(&data, 1, 51, &argvals, 1);
        // Check interior points
        for j in 5..45 {
            let expected = 2.0 * argvals[j];
            assert!(
                (deriv[j] - expected).abs() < 0.1,
                "Derivative of x^2 should be 2x"
            );
        }
    }

    #[test]
    fn test_deriv_1d_invalid() {
        let result = deriv_1d(&[], 0, 0, &[], 1);
        assert!(result.is_empty() || result.iter().all(|&x| x == 0.0));
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
        let median = geometric_median_1d(&data, n, m, &argvals, 100, 1e-6);
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
        let median = geometric_median_1d(&data, n, m, &argvals, 100, 1e-6);
        assert_eq!(median.len(), m);
        assert!(median.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_geometric_median_invalid() {
        assert!(geometric_median_1d(&[], 0, 0, &[], 100, 1e-6).is_empty());
    }
}
