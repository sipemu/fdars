//! Regression functions for functional data.
//!
//! This module provides functional PCA, PLS, and ridge regression.

use crate::iter_maybe_parallel;
#[cfg(feature = "linalg")]
use anofox_regression::solvers::RidgeRegressor;
#[cfg(feature = "linalg")]
use anofox_regression::{FittedRegressor, Regressor};
use nalgebra::{DMatrix, SVD};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Result of functional PCA.
pub struct FpcaResult {
    /// Singular values
    pub singular_values: Vec<f64>,
    /// Rotation matrix (loadings), m x ncomp, column-major
    pub rotation: Vec<f64>,
    /// Scores matrix, n x ncomp, column-major
    pub scores: Vec<f64>,
    /// Mean function
    pub mean: Vec<f64>,
    /// Centered data
    pub centered: Vec<f64>,
}

/// Perform functional PCA via SVD on centered data.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of observations
/// * `m` - Number of evaluation points
/// * `ncomp` - Number of components to extract
pub fn fdata_to_pc_1d(data: &[f64], n: usize, m: usize, ncomp: usize) -> Option<FpcaResult> {
    if n == 0 || m == 0 || ncomp < 1 || data.len() != n * m {
        return None;
    }

    let ncomp = ncomp.min(n).min(m);

    // Compute column means
    let means: Vec<f64> = iter_maybe_parallel!(0..m)
        .map(|j| {
            let mut sum = 0.0;
            for i in 0..n {
                sum += data[i + j * n];
            }
            sum / n as f64
        })
        .collect();

    // Center the data
    let centered_data: Vec<f64> = (0..(n * m))
        .map(|idx| {
            let i = idx % n;
            let j = idx / n;
            data[i + j * n] - means[j]
        })
        .collect();

    // Create nalgebra DMatrix
    let matrix = DMatrix::from_column_slice(n, m, &centered_data);

    // Compute SVD
    let svd = SVD::new(matrix, true, true);

    // Extract singular values
    let singular_values: Vec<f64> = svd.singular_values.iter().take(ncomp).cloned().collect();

    // Extract V (right singular vectors)
    let v_t = svd.v_t.as_ref()?;
    let rotation_data: Vec<f64> = (0..ncomp)
        .flat_map(|k| (0..m).map(move |j| v_t[(k, j)]))
        .collect();

    // Compute scores: X_centered * V = U * S
    let u = svd.u.as_ref()?;
    let mut scores_data: Vec<f64> = Vec::with_capacity(n * ncomp);
    for k in 0..ncomp {
        let sv_k = singular_values[k];
        for i in 0..n {
            scores_data.push(u[(i, k)] * sv_k);
        }
    }

    Some(FpcaResult {
        singular_values,
        rotation: rotation_data,
        scores: scores_data,
        mean: means,
        centered: centered_data,
    })
}

/// Result of PLS regression.
pub struct PlsResult {
    /// Weight vectors, m x ncomp
    pub weights: Vec<f64>,
    /// Score vectors, n x ncomp
    pub scores: Vec<f64>,
    /// Loading vectors, m x ncomp
    pub loadings: Vec<f64>,
}

/// Perform PLS via NIPALS algorithm.
pub fn fdata_to_pls_1d(
    data: &[f64],
    n: usize,
    m: usize,
    y: &[f64],
    ncomp: usize,
) -> Option<PlsResult> {
    if n == 0 || m == 0 || y.len() != n || ncomp < 1 || data.len() != n * m {
        return None;
    }

    let ncomp = ncomp.min(n).min(m);

    // Center X and y
    let x_means: Vec<f64> = (0..m)
        .map(|j| {
            let mut sum = 0.0;
            for i in 0..n {
                sum += data[i + j * n];
            }
            sum / n as f64
        })
        .collect();

    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

    let mut x_cen: Vec<f64> = (0..(n * m))
        .map(|idx| {
            let i = idx % n;
            let j = idx / n;
            data[i + j * n] - x_means[j]
        })
        .collect();

    let mut y_cen: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();

    let mut weights = vec![0.0; m * ncomp];
    let mut scores = vec![0.0; n * ncomp];
    let mut loadings = vec![0.0; m * ncomp];

    // NIPALS algorithm
    for k in 0..ncomp {
        // w = X'y / ||X'y||
        let mut w: Vec<f64> = (0..m)
            .map(|j| {
                let mut sum = 0.0;
                for i in 0..n {
                    sum += x_cen[i + j * n] * y_cen[i];
                }
                sum
            })
            .collect();

        let w_norm: f64 = w.iter().map(|&wi| wi * wi).sum::<f64>().sqrt();
        if w_norm > 1e-10 {
            for wi in &mut w {
                *wi /= w_norm;
            }
        }

        // t = Xw
        let t: Vec<f64> = (0..n)
            .map(|i| {
                let mut sum = 0.0;
                for j in 0..m {
                    sum += x_cen[i + j * n] * w[j];
                }
                sum
            })
            .collect();

        let t_norm_sq: f64 = t.iter().map(|&ti| ti * ti).sum();

        // p = X't / (t't)
        let p: Vec<f64> = (0..m)
            .map(|j| {
                let mut sum = 0.0;
                for i in 0..n {
                    sum += x_cen[i + j * n] * t[i];
                }
                sum / t_norm_sq.max(1e-10)
            })
            .collect();

        // Store results
        for j in 0..m {
            weights[j + k * m] = w[j];
            loadings[j + k * m] = p[j];
        }
        for i in 0..n {
            scores[i + k * n] = t[i];
        }

        // Deflate X
        for j in 0..m {
            for i in 0..n {
                x_cen[i + j * n] -= t[i] * p[j];
            }
        }

        // Deflate y
        let t_y: f64 = t.iter().zip(y_cen.iter()).map(|(&ti, &yi)| ti * yi).sum();
        let q = t_y / t_norm_sq.max(1e-10);
        for i in 0..n {
            y_cen[i] -= t[i] * q;
        }
    }

    Some(PlsResult {
        weights,
        scores,
        loadings,
    })
}

/// Result of ridge regression fit.
#[cfg(feature = "linalg")]
pub struct RidgeResult {
    /// Coefficients
    pub coefficients: Vec<f64>,
    /// Intercept
    pub intercept: f64,
    /// Fitted values
    pub fitted_values: Vec<f64>,
    /// Residuals
    pub residuals: Vec<f64>,
    /// R-squared
    pub r_squared: f64,
    /// Lambda used
    pub lambda: f64,
    /// Error message if any
    pub error: Option<String>,
}

/// Fit ridge regression.
///
/// # Arguments
/// * `x` - Predictor matrix (column-major, n x m)
/// * `y` - Response vector
/// * `n` - Number of observations
/// * `m` - Number of predictors
/// * `lambda` - Regularization parameter
/// * `with_intercept` - Whether to include intercept
#[cfg(feature = "linalg")]
pub fn ridge_regression_fit(
    x: &[f64],
    y: &[f64],
    n: usize,
    m: usize,
    lambda: f64,
    with_intercept: bool,
) -> RidgeResult {
    if n == 0 || m == 0 || y.len() != n || x.len() != n * m {
        return RidgeResult {
            coefficients: Vec::new(),
            intercept: 0.0,
            fitted_values: Vec::new(),
            residuals: Vec::new(),
            r_squared: 0.0,
            lambda,
            error: Some("Invalid input dimensions".to_string()),
        };
    }

    // Convert to faer Mat format
    let x_faer = faer::Mat::from_fn(n, m, |i, j| x[i + j * n]);
    let y_faer = faer::Col::from_fn(n, |i| y[i]);

    // Build and fit the ridge regressor
    let regressor = RidgeRegressor::builder()
        .with_intercept(with_intercept)
        .lambda(lambda)
        .build();

    let fitted = match regressor.fit(&x_faer, &y_faer) {
        Ok(f) => f,
        Err(e) => {
            return RidgeResult {
                coefficients: Vec::new(),
                intercept: 0.0,
                fitted_values: Vec::new(),
                residuals: Vec::new(),
                r_squared: 0.0,
                lambda,
                error: Some(format!("Fit failed: {:?}", e)),
            }
        }
    };

    // Extract coefficients
    let coefs = fitted.coefficients();
    let coefficients: Vec<f64> = (0..coefs.nrows()).map(|i| coefs[i]).collect();

    // Get intercept
    let intercept = fitted.intercept().unwrap_or(0.0);

    // Compute fitted values
    let mut fitted_values = vec![0.0; n];
    for i in 0..n {
        let mut pred = intercept;
        for j in 0..m {
            pred += x[i + j * n] * coefficients[j];
        }
        fitted_values[i] = pred;
    }

    // Compute residuals
    let residuals: Vec<f64> = y
        .iter()
        .zip(fitted_values.iter())
        .map(|(&yi, &yhat)| yi - yhat)
        .collect();

    // Compute R-squared
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = residuals.iter().map(|&r| r.powi(2)).sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    RidgeResult {
        coefficients,
        intercept,
        fitted_values,
        residuals,
        r_squared,
        lambda,
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate functional data with known structure for testing
    fn generate_test_fdata(n: usize, m: usize) -> (Vec<f64>, Vec<f64>) {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

        // Create n curves: sine waves with varying phase
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            let phase = (i as f64 / n as f64) * PI;
            for j in 0..m {
                data[i + j * n] = (2.0 * PI * t[j] + phase).sin();
            }
        }

        (data, t)
    }

    // ============== FPCA tests ==============

    #[test]
    fn test_fdata_to_pc_1d_basic() {
        let n = 20;
        let m = 50;
        let ncomp = 3;
        let (data, _) = generate_test_fdata(n, m);

        let result = fdata_to_pc_1d(&data, n, m, ncomp);
        assert!(result.is_some());

        let fpca = result.unwrap();
        assert_eq!(fpca.singular_values.len(), ncomp);
        assert_eq!(fpca.rotation.len(), m * ncomp);
        assert_eq!(fpca.scores.len(), n * ncomp);
        assert_eq!(fpca.mean.len(), m);
        assert_eq!(fpca.centered.len(), n * m);
    }

    #[test]
    fn test_fdata_to_pc_1d_singular_values_decreasing() {
        let n = 20;
        let m = 50;
        let ncomp = 5;
        let (data, _) = generate_test_fdata(n, m);

        let fpca = fdata_to_pc_1d(&data, n, m, ncomp).unwrap();

        // Singular values should be in decreasing order
        for i in 1..fpca.singular_values.len() {
            assert!(
                fpca.singular_values[i] <= fpca.singular_values[i - 1] + 1e-10,
                "Singular values should be decreasing"
            );
        }
    }

    #[test]
    fn test_fdata_to_pc_1d_centered_has_zero_mean() {
        let n = 20;
        let m = 50;
        let (data, _) = generate_test_fdata(n, m);

        let fpca = fdata_to_pc_1d(&data, n, m, 3).unwrap();

        // Column means of centered data should be zero
        for j in 0..m {
            let col_mean: f64 = (0..n).map(|i| fpca.centered[i + j * n]).sum::<f64>() / n as f64;
            assert!(
                col_mean.abs() < 1e-10,
                "Centered data should have zero column mean"
            );
        }
    }

    #[test]
    fn test_fdata_to_pc_1d_ncomp_limits() {
        let n = 10;
        let m = 50;
        let (data, _) = generate_test_fdata(n, m);

        // Request more components than n - should cap at n
        let fpca = fdata_to_pc_1d(&data, n, m, 20).unwrap();
        assert!(fpca.singular_values.len() <= n);
    }

    #[test]
    fn test_fdata_to_pc_1d_invalid_input() {
        // Empty data
        let result = fdata_to_pc_1d(&[], 0, 50, 3);
        assert!(result.is_none());

        // Wrong data length
        let data = vec![0.0; 100];
        let result = fdata_to_pc_1d(&data, 10, 20, 3); // Should be 10*20=200
        assert!(result.is_none());

        // Zero components
        let (data, _) = generate_test_fdata(10, 50);
        let result = fdata_to_pc_1d(&data, 10, 50, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_fdata_to_pc_1d_reconstruction() {
        let n = 10;
        let m = 30;
        let (data, _) = generate_test_fdata(n, m);

        // Use all components for perfect reconstruction
        let ncomp = n.min(m);
        let fpca = fdata_to_pc_1d(&data, n, m, ncomp).unwrap();

        // Reconstruct: X_centered = scores * rotation^T
        // But scores = U * S, rotation = V
        // So X_centered ≈ sum_k (score_k * loading_k)
        for i in 0..n {
            for j in 0..m {
                let mut reconstructed = 0.0;
                for k in 0..ncomp {
                    let score = fpca.scores[i + k * n];
                    let loading = fpca.rotation[j + k * m];
                    reconstructed += score * loading;
                }
                let original_centered = fpca.centered[i + j * n];
                assert!(
                    (reconstructed - original_centered).abs() < 0.1,
                    "Reconstruction error at ({}, {}): {} vs {}",
                    i,
                    j,
                    reconstructed,
                    original_centered
                );
            }
        }
    }

    // ============== PLS tests ==============

    #[test]
    fn test_fdata_to_pls_1d_basic() {
        let n = 20;
        let m = 30;
        let ncomp = 3;
        let (x, _) = generate_test_fdata(n, m);

        // Create y with some relationship to x
        let y: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) + 0.1).collect();

        let result = fdata_to_pls_1d(&x, n, m, &y, ncomp);
        assert!(result.is_some());

        let pls = result.unwrap();
        assert_eq!(pls.weights.len(), m * ncomp);
        assert_eq!(pls.scores.len(), n * ncomp);
        assert_eq!(pls.loadings.len(), m * ncomp);
    }

    #[test]
    fn test_fdata_to_pls_1d_weights_normalized() {
        let n = 20;
        let m = 30;
        let ncomp = 2;
        let (x, _) = generate_test_fdata(n, m);
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let pls = fdata_to_pls_1d(&x, n, m, &y, ncomp).unwrap();

        // Weight vectors should be approximately unit norm
        for k in 0..ncomp {
            let norm: f64 = (0..m)
                .map(|j| pls.weights[j + k * m].powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() < 0.1,
                "Weight vector {} should be unit norm, got {}",
                k,
                norm
            );
        }
    }

    #[test]
    fn test_fdata_to_pls_1d_invalid_input() {
        let (x, _) = generate_test_fdata(10, 30);
        let y = vec![0.0; 10];

        // Wrong y length
        let result = fdata_to_pls_1d(&x, 10, 30, &[0.0; 5], 2);
        assert!(result.is_none());

        // Zero components
        let result = fdata_to_pls_1d(&x, 10, 30, &y, 0);
        assert!(result.is_none());
    }

    // ============== Ridge regression tests ==============

    #[test]
    fn test_ridge_regression_fit_basic() {
        let n = 50;
        let m = 5;

        // Create X with known structure
        let mut x = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                x[i + j * n] = (i as f64 + j as f64) / (n + m) as f64;
            }
        }

        // Create y = sum of x columns + noise
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let mut sum = 0.0;
                for j in 0..m {
                    sum += x[i + j * n];
                }
                sum + 0.01 * (i as f64 % 10.0)
            })
            .collect();

        let result = ridge_regression_fit(&x, &y, n, m, 0.1, true);

        assert!(result.error.is_none(), "Ridge should fit without error");
        assert_eq!(result.coefficients.len(), m);
        assert_eq!(result.fitted_values.len(), n);
        assert_eq!(result.residuals.len(), n);
    }

    #[test]
    fn test_ridge_regression_fit_r_squared() {
        let n = 50;
        let m = 3;

        // Create perfect linear relationship
        let x: Vec<f64> = (0..n * m).map(|i| i as f64 / (n * m) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

        let result = ridge_regression_fit(&x, &y, n, m, 0.01, true);

        // R-squared should be high for good fit
        assert!(
            result.r_squared > 0.5,
            "R-squared should be high, got {}",
            result.r_squared
        );
        assert!(result.r_squared <= 1.0 + 1e-10, "R-squared should be <= 1");
    }

    #[test]
    fn test_ridge_regression_fit_regularization() {
        let n = 30;
        let m = 10;

        // Create X
        let x: Vec<f64> = (0..n * m)
            .map(|i| ((i * 17) % 100) as f64 / 100.0)
            .collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();

        let low_lambda = ridge_regression_fit(&x, &y, n, m, 0.001, true);
        let high_lambda = ridge_regression_fit(&x, &y, n, m, 100.0, true);

        // Higher lambda should give smaller coefficient norm
        let norm_low: f64 = low_lambda
            .coefficients
            .iter()
            .map(|c| c.powi(2))
            .sum::<f64>()
            .sqrt();
        let norm_high: f64 = high_lambda
            .coefficients
            .iter()
            .map(|c| c.powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(
            norm_high <= norm_low + 1e-6,
            "Higher lambda should shrink coefficients: {} vs {}",
            norm_high,
            norm_low
        );
    }

    #[test]
    fn test_ridge_regression_fit_residuals() {
        let n = 20;
        let m = 3;

        let x: Vec<f64> = (0..n * m).map(|i| i as f64 / (n * m) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

        let result = ridge_regression_fit(&x, &y, n, m, 0.1, true);

        // Check residuals = y - fitted
        for i in 0..n {
            let expected_resid = y[i] - result.fitted_values[i];
            assert!(
                (result.residuals[i] - expected_resid).abs() < 1e-10,
                "Residual mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_ridge_regression_fit_no_intercept() {
        let n = 30;
        let m = 5;

        let x: Vec<f64> = (0..n * m).map(|i| i as f64 / (n * m) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

        let result = ridge_regression_fit(&x, &y, n, m, 0.1, false);

        assert!(result.error.is_none());
        // Intercept should be 0 when not fitting
        assert!(
            result.intercept.abs() < 1e-10,
            "Intercept should be 0, got {}",
            result.intercept
        );
    }

    #[test]
    fn test_ridge_regression_fit_invalid_input() {
        // Empty data
        let result = ridge_regression_fit(&[], &[], 0, 5, 0.1, true);
        assert!(result.error.is_some());

        // Mismatched dimensions
        let x = vec![0.0; 50];
        let y = vec![0.0; 10];
        let result = ridge_regression_fit(&x, &y, 10, 10, 0.1, true); // x should be 10*10=100
        assert!(result.error.is_some());
    }
}
