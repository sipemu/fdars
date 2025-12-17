//! Regression functions for functional data.
//!
//! This module provides functional PCA, PLS, and ridge regression.

use anofox_regression::solvers::RidgeRegressor;
use anofox_regression::{FittedRegressor, Regressor};
use nalgebra::{DMatrix, SVD};
use rayon::prelude::*;

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
    let means: Vec<f64> = (0..m)
        .into_par_iter()
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
