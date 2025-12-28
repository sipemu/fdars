//! Utility functions for functional data analysis.

use crate::helpers::simpsons_weights;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Compute Simpson's rule integration for a single function.
///
/// # Arguments
/// * `values` - Function values at evaluation points
/// * `argvals` - Evaluation points
pub fn integrate_simpson(values: &[f64], argvals: &[f64]) -> f64 {
    if values.len() != argvals.len() || values.is_empty() {
        return 0.0;
    }

    let weights = simpsons_weights(argvals);
    values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| v * w)
        .sum()
}

/// Compute inner product between two functional data curves.
///
/// # Arguments
/// * `curve1` - First curve values
/// * `curve2` - Second curve values
/// * `argvals` - Evaluation points
pub fn inner_product(curve1: &[f64], curve2: &[f64], argvals: &[f64]) -> f64 {
    if curve1.len() != curve2.len() || curve1.len() != argvals.len() || curve1.is_empty() {
        return 0.0;
    }

    let weights = simpsons_weights(argvals);
    curve1
        .iter()
        .zip(curve2.iter())
        .zip(weights.iter())
        .map(|((&c1, &c2), &w)| c1 * c2 * w)
        .sum()
}

/// Compute inner product matrix for functional data.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of observations
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
///
/// # Returns
/// Symmetric inner product matrix (n x n), column-major
pub fn inner_product_matrix(data: &[f64], n: usize, m: usize, argvals: &[f64]) -> Vec<f64> {
    if n == 0 || m == 0 || argvals.len() != m || data.len() != n * m {
        return Vec::new();
    }

    let weights = simpsons_weights(argvals);

    // Compute upper triangle in parallel
    let upper_triangle: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            (i..n)
                .map(|j| {
                    let mut ip = 0.0;
                    for k in 0..m {
                        ip += data[i + k * n] * data[j + k * n] * weights[k];
                    }
                    (i, j, ip)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Build symmetric matrix
    let mut result = vec![0.0; n * n];
    for (i, j, ip) in upper_triangle {
        result[i + j * n] = ip;
        result[j + i * n] = ip;
    }

    result
}

/// Compute the Adot matrix used in PCvM statistic.
pub fn compute_adot(n: usize, inprod: &[f64]) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    let expected_len = (n * n + n) / 2;
    if inprod.len() != expected_len {
        return Vec::new();
    }

    let out_len = (n * n - n + 2) / 2;
    let mut adot_vec = vec![0.0; out_len];

    adot_vec[0] = PI * (n + 1) as f64;

    // Collect all (i, j) pairs for parallel processing
    let pairs: Vec<(usize, usize)> = (2..=n).flat_map(|i| (1..i).map(move |j| (i, j))).collect();

    // Compute adot values in parallel
    let results: Vec<(usize, f64)> = pairs
        .into_par_iter()
        .map(|(i, j)| {
            let mut sumr = 0.0;

            for r in 1..=n {
                if i == r || j == r {
                    sumr += PI;
                } else {
                    let auxi = i * (i - 1) / 2;
                    let auxj = j * (j - 1) / 2;
                    let auxr = r * (r - 1) / 2;

                    let ij = auxi + j - 1;
                    let ii = auxi + i - 1;
                    let jj = auxj + j - 1;
                    let rr = auxr + r - 1;

                    let ir = if i > r { auxi + r - 1 } else { auxr + i - 1 };
                    let rj = if r > j { auxr + j - 1 } else { auxj + r - 1 };
                    let jr = rj;

                    let num = inprod[ij] - inprod[ir] - inprod[rj] + inprod[rr];
                    let aux1 = (inprod[ii] - 2.0 * inprod[ir] + inprod[rr]).sqrt();
                    let aux2 = (inprod[jj] - 2.0 * inprod[jr] + inprod[rr]).sqrt();
                    let den = aux1 * aux2;

                    let mut quo = if den.abs() > 1e-10 { num / den } else { 0.0 };
                    quo = quo.clamp(-1.0, 1.0);

                    sumr += (PI - quo.acos()).abs();
                }
            }

            let idx = 1 + ((i - 1) * (i - 2) / 2) + j - 1;
            (idx, sumr)
        })
        .collect();

    // Fill in the results
    for (idx, val) in results {
        if idx < adot_vec.len() {
            adot_vec[idx] = val;
        }
    }

    adot_vec
}

/// Compute the PCvM statistic.
pub fn pcvm_statistic(adot_vec: &[f64], residuals: &[f64]) -> f64 {
    let n = residuals.len();

    if n == 0 || adot_vec.is_empty() {
        return 0.0;
    }

    let mut sums = 0.0;
    for i in 2..=n {
        for j in 1..i {
            let idx = 1 + ((i - 1) * (i - 2) / 2) + j - 1;
            if idx < adot_vec.len() {
                sums += residuals[i - 1] * adot_vec[idx] * residuals[j - 1];
            }
        }
    }

    let diag_sum: f64 = residuals.iter().map(|r| r * r).sum();
    adot_vec[0] * diag_sum + 2.0 * sums
}

/// Result of random projection statistics.
pub struct RpStatResult {
    /// CvM statistics for each projection
    pub cvm: Vec<f64>,
    /// KS statistics for each projection
    pub ks: Vec<f64>,
}

/// Compute random projection statistics.
pub fn rp_stat(proj_x_ord: &[i32], residuals: &[f64], n_proj: usize) -> RpStatResult {
    let n = residuals.len();

    if n == 0 || n_proj == 0 || proj_x_ord.len() != n * n_proj {
        return RpStatResult {
            cvm: Vec::new(),
            ks: Vec::new(),
        };
    }

    // Process projections in parallel
    let stats: Vec<(f64, f64)> = (0..n_proj)
        .into_par_iter()
        .map(|p| {
            let mut y = vec![0.0; n];
            let mut cumsum = 0.0;

            for i in 0..n {
                let idx = proj_x_ord[p * n + i] as usize;
                if idx > 0 && idx <= n {
                    cumsum += residuals[idx - 1];
                }
                y[i] = cumsum;
            }

            let sum_y_sq: f64 = y.iter().map(|yi| yi * yi).sum();
            let cvm = sum_y_sq / (n * n) as f64;

            let max_abs_y = y.iter().map(|yi| yi.abs()).fold(0.0, f64::max);
            let ks = max_abs_y / (n as f64).sqrt();

            (cvm, ks)
        })
        .collect();

    let cvm_stats: Vec<f64> = stats.iter().map(|(cvm, _)| *cvm).collect();
    let ks_stats: Vec<f64> = stats.iter().map(|(_, ks)| *ks).collect();

    RpStatResult {
        cvm: cvm_stats,
        ks: ks_stats,
    }
}

/// k-NN prediction for functional regression.
pub fn knn_predict(
    distance_matrix: &[f64],
    y: &[f64],
    n_train: usize,
    n_test: usize,
    k: usize,
) -> Vec<f64> {
    if n_train == 0 || n_test == 0 || k == 0 || y.len() != n_train {
        return vec![0.0; n_test];
    }

    let k = k.min(n_train);

    (0..n_test)
        .into_par_iter()
        .map(|i| {
            // Get distances from test point i to all training points
            let mut distances: Vec<(usize, f64)> = (0..n_train)
                .map(|j| (j, distance_matrix[i + j * n_test]))
                .collect();

            // Sort by distance
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Average of k nearest neighbors
            let sum: f64 = distances.iter().take(k).map(|(j, _)| y[*j]).sum();
            sum / k as f64
        })
        .collect()
}

/// Compute leave-one-out cross-validation error for k-NN.
pub fn knn_loocv(distance_matrix: &[f64], y: &[f64], n: usize, k: usize) -> f64 {
    if n == 0 || k == 0 || y.len() != n || distance_matrix.len() != n * n {
        return f64::INFINITY;
    }

    let k = k.min(n - 1);

    let errors: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            // Get distances from point i to all other points
            let mut distances: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, distance_matrix[i + j * n]))
                .collect();

            // Sort by distance
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Prediction
            let pred: f64 = distances.iter().take(k).map(|(j, _)| y[*j]).sum::<f64>() / k as f64;

            // Squared error
            (y[i] - pred).powi(2)
        })
        .collect();

    errors.iter().sum::<f64>() / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    // ============== Integration tests ==============

    #[test]
    fn test_integrate_simpson_constant() {
        // Integral of constant 2 over [0, 1] should be 2
        let argvals = uniform_grid(21);
        let values: Vec<f64> = vec![2.0; 21];

        let result = integrate_simpson(&values, &argvals);
        assert!(
            (result - 2.0).abs() < 0.01,
            "Integral of constant 2 should be 2, got {}",
            result
        );
    }

    #[test]
    fn test_integrate_simpson_linear() {
        // Integral of x over [0, 1] should be 0.5
        let argvals = uniform_grid(51);
        let values: Vec<f64> = argvals.clone();

        let result = integrate_simpson(&values, &argvals);
        assert!(
            (result - 0.5).abs() < 0.01,
            "Integral of x should be 0.5, got {}",
            result
        );
    }

    #[test]
    fn test_integrate_simpson_invalid() {
        assert!(integrate_simpson(&[], &[]).abs() < 1e-10);
        assert!(integrate_simpson(&[1.0, 2.0], &[0.0]).abs() < 1e-10);
    }

    // ============== Inner product tests ==============

    #[test]
    fn test_inner_product_orthogonal() {
        let argvals = uniform_grid(101);
        // sin(2*pi*x) and sin(4*pi*x) are orthogonal on [0,1]
        let curve1: Vec<f64> = argvals.iter().map(|&t| (2.0 * PI * t).sin()).collect();
        let curve2: Vec<f64> = argvals.iter().map(|&t| (4.0 * PI * t).sin()).collect();

        let ip = inner_product(&curve1, &curve2, &argvals);
        assert!(
            ip.abs() < 0.05,
            "Orthogonal functions should have near-zero inner product, got {}",
            ip
        );
    }

    #[test]
    fn test_inner_product_self_positive() {
        let argvals = uniform_grid(51);
        let curve: Vec<f64> = argvals.iter().map(|&t| (2.0 * PI * t).sin()).collect();

        let ip = inner_product(&curve, &curve, &argvals);
        assert!(ip > 0.0, "Self inner product should be positive");
        // Integral of sin^2(2*pi*x) over [0,1] is 0.5
        assert!((ip - 0.5).abs() < 0.05, "Expected ~0.5, got {}", ip);
    }

    #[test]
    fn test_inner_product_invalid() {
        assert!(inner_product(&[], &[], &[]).abs() < 1e-10);
        assert!(inner_product(&[1.0], &[1.0, 2.0], &[0.0]).abs() < 1e-10);
    }

    // ============== Inner product matrix tests ==============

    #[test]
    fn test_inner_product_matrix_symmetric() {
        let n = 5;
        let m = 21;
        let argvals = uniform_grid(m);

        // Create simple test data
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                data[i + j * n] = (i as f64 + 1.0) * argvals[j];
            }
        }

        let matrix = inner_product_matrix(&data, n, m, &argvals);
        assert_eq!(matrix.len(), n * n);

        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (matrix[i + j * n] - matrix[j + i * n]).abs() < 1e-10,
                    "Matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_inner_product_matrix_diagonal_positive() {
        let n = 3;
        let m = 21;
        let argvals = uniform_grid(m);

        let mut data = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                data[i + j * n] = (i as f64 + 1.0) * argvals[j];
            }
        }

        let matrix = inner_product_matrix(&data, n, m, &argvals);

        // Diagonal elements should be positive
        for i in 0..n {
            assert!(matrix[i + i * n] > 0.0, "Diagonal should be positive");
        }
    }

    #[test]
    fn test_inner_product_matrix_invalid() {
        let result = inner_product_matrix(&[], 0, 0, &[]);
        assert!(result.is_empty());
    }

    // ============== PCvM statistic tests ==============

    #[test]
    fn test_pcvm_statistic_zero_residuals() {
        let n = 5;
        let inprod_len = (n * n + n) / 2;
        let inprod: Vec<f64> = (0..inprod_len).map(|i| i as f64 * 0.1).collect();
        let adot = compute_adot(n, &inprod);

        let residuals = vec![0.0; n];
        let stat = pcvm_statistic(&adot, &residuals);

        assert!(
            stat.abs() < 1e-10,
            "Zero residuals should give zero statistic"
        );
    }

    #[test]
    fn test_pcvm_statistic_positive() {
        let n = 5;
        let inprod_len = (n * n + n) / 2;
        let inprod: Vec<f64> = (0..inprod_len).map(|i| (i as f64 * 0.1).max(0.1)).collect();
        let adot = compute_adot(n, &inprod);

        let residuals = vec![1.0, -0.5, 0.3, -0.2, 0.4];
        let stat = pcvm_statistic(&adot, &residuals);

        assert!(stat.is_finite(), "Statistic should be finite");
    }

    // ============== k-NN prediction tests ==============

    #[test]
    fn test_knn_predict_k1() {
        // Simple case: k=1 returns nearest neighbor's value
        let n_train = 3;
        let n_test = 2;
        let y = vec![1.0, 2.0, 3.0];

        // Distance matrix layout: [n_test x n_train] column-major
        // Element [i + j * n_test] = distance from test i to train j
        let distance_matrix = vec![
            0.1, 0.9, // dist(test0, train0), dist(test1, train0)
            0.5, 0.2, // dist(test0, train1), dist(test1, train1)
            0.8, 0.1, // dist(test0, train2), dist(test1, train2)
        ];

        let predictions = knn_predict(&distance_matrix, &y, n_train, n_test, 1);

        assert_eq!(predictions.len(), 2);
        assert!(
            (predictions[0] - 1.0).abs() < 1e-10,
            "Test 0 nearest to train 0"
        );
        assert!(
            (predictions[1] - 3.0).abs() < 1e-10,
            "Test 1 nearest to train 2"
        );
    }

    #[test]
    fn test_knn_predict_k_all() {
        // k = n_train should return mean of all training values
        let n_train = 4;
        let n_test = 1;
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let expected_mean = 2.5;

        let distance_matrix = vec![0.1, 0.2, 0.3, 0.4]; // arbitrary distances

        let predictions = knn_predict(&distance_matrix, &y, n_train, n_test, n_train);

        assert!(
            (predictions[0] - expected_mean).abs() < 1e-10,
            "k=n should return mean"
        );
    }

    #[test]
    fn test_knn_predict_invalid() {
        let result = knn_predict(&[], &[], 0, 1, 1);
        assert_eq!(result, vec![0.0]);
    }

    // ============== k-NN LOOCV tests ==============

    #[test]
    fn test_knn_loocv_returns_finite() {
        let n = 5;
        let y = vec![1.0, 2.0, 1.5, 2.5, 1.8];

        // Create distance matrix (symmetric with zero diagonal)
        let mut distance_matrix = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let dist = ((i as f64) - (j as f64)).abs();
                distance_matrix[i + j * n] = dist;
            }
        }

        let error = knn_loocv(&distance_matrix, &y, n, 2);

        assert!(error.is_finite(), "LOOCV error should be finite");
        assert!(error >= 0.0, "LOOCV error should be non-negative");
    }

    #[test]
    fn test_knn_loocv_invalid() {
        let result = knn_loocv(&[], &[], 0, 1);
        assert!(result.is_infinite());

        let result = knn_loocv(&[0.0; 4], &[1.0, 2.0], 2, 0);
        assert!(result.is_infinite());
    }
}
