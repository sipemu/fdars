//! Regression functions for functional data.
//!
//! This module provides functional PCA, PLS, and ridge regression.

use crate::error::FdarError;
use crate::matrix::FdMatrix;
#[cfg(feature = "linalg")]
use anofox_regression::solvers::RidgeRegressor;
#[cfg(feature = "linalg")]
use anofox_regression::{FittedRegressor, Regressor};
use nalgebra::SVD;

/// Result of functional PCA.
#[derive(Debug, Clone, PartialEq)]
pub struct FpcaResult {
    /// Singular values
    pub singular_values: Vec<f64>,
    /// Rotation matrix (loadings), m x ncomp
    pub rotation: FdMatrix,
    /// Scores matrix, n x ncomp
    pub scores: FdMatrix,
    /// Mean function
    pub mean: Vec<f64>,
    /// Centered data, n x m
    pub centered: FdMatrix,
}

impl FpcaResult {
    /// Project new functional data onto the FPC score space.
    ///
    /// Centers the input data by subtracting the mean function estimated
    /// during FPCA, then multiplies by the rotation (loadings) matrix to
    /// obtain FPC scores for the new observations.
    ///
    /// # Arguments
    /// * `data` - Matrix (n_new x m) of new observations
    ///
    /// # Errors
    ///
    /// Returns [`FdarError::InvalidDimension`] if the number of columns in
    /// `data` does not match the length of the mean vector (i.e. the number
    /// of evaluation points used during FPCA).
    ///
    /// # Examples
    ///
    /// ```
    /// use fdars_core::matrix::FdMatrix;
    /// use fdars_core::regression::fdata_to_pc_1d;
    ///
    /// let data = FdMatrix::from_column_major(
    ///     (0..50).map(|i| (i as f64 * 0.1).sin()).collect(),
    ///     5, 10,
    /// ).unwrap();
    /// let fpca = fdata_to_pc_1d(&data, 3).unwrap();
    ///
    /// // Project the original data (scores should match)
    /// let scores = fpca.project(&data).unwrap();
    /// assert_eq!(scores.shape(), (5, 3));
    ///
    /// // Project new data
    /// let new_data = FdMatrix::from_column_major(
    ///     (0..20).map(|i| (i as f64 * 0.2).cos()).collect(),
    ///     2, 10,
    /// ).unwrap();
    /// let new_scores = fpca.project(&new_data).unwrap();
    /// assert_eq!(new_scores.shape(), (2, 3));
    /// ```
    pub fn project(&self, data: &FdMatrix) -> Result<FdMatrix, FdarError> {
        let (n, m) = data.shape();
        let ncomp = self.rotation.ncols();
        if m != self.mean.len() {
            return Err(FdarError::InvalidDimension {
                parameter: "data",
                expected: format!("{} columns", self.mean.len()),
                actual: format!("{m} columns"),
            });
        }

        let mut scores = FdMatrix::zeros(n, ncomp);
        for i in 0..n {
            for k in 0..ncomp {
                let mut sum = 0.0;
                for j in 0..m {
                    sum += (data[(i, j)] - self.mean[j]) * self.rotation[(j, k)];
                }
                scores[(i, k)] = sum;
            }
        }
        Ok(scores)
    }

    /// Reconstruct functional data from FPC scores.
    ///
    /// Computes the approximation of functional data using the first
    /// `ncomp` principal components:
    /// `data[i, j] = mean[j] + sum_k scores[i, k] * rotation[j, k]`
    ///
    /// # Arguments
    /// * `scores` - Score matrix (n x p) where p >= `ncomp`
    /// * `ncomp` - Number of components to use for reconstruction
    ///
    /// # Errors
    ///
    /// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero or exceeds
    /// the number of columns in `scores` or the number of available components
    /// in the rotation matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use fdars_core::matrix::FdMatrix;
    /// use fdars_core::regression::fdata_to_pc_1d;
    ///
    /// let data = FdMatrix::from_column_major(
    ///     (0..100).map(|i| (i as f64 * 0.1).sin()).collect(),
    ///     10, 10,
    /// ).unwrap();
    /// let fpca = fdata_to_pc_1d(&data, 5).unwrap();
    ///
    /// // Reconstruct using all 5 components
    /// let recon = fpca.reconstruct(&fpca.scores, 5).unwrap();
    /// assert_eq!(recon.shape(), (10, 10));
    ///
    /// // Reconstruct using fewer components
    /// let recon2 = fpca.reconstruct(&fpca.scores, 2).unwrap();
    /// assert_eq!(recon2.shape(), (10, 10));
    /// ```
    pub fn reconstruct(&self, scores: &FdMatrix, ncomp: usize) -> Result<FdMatrix, FdarError> {
        let (n, p) = scores.shape();
        let m = self.mean.len();
        let max_comp = self.rotation.ncols().min(p);
        if ncomp == 0 {
            return Err(FdarError::InvalidParameter {
                parameter: "ncomp",
                message: "ncomp must be >= 1".to_string(),
            });
        }
        if ncomp > max_comp {
            return Err(FdarError::InvalidParameter {
                parameter: "ncomp",
                message: format!("ncomp={ncomp} exceeds available components ({max_comp})"),
            });
        }

        let mut recon = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                let mut val = self.mean[j];
                for k in 0..ncomp {
                    val += scores[(i, k)] * self.rotation[(j, k)];
                }
                recon[(i, j)] = val;
            }
        }
        Ok(recon)
    }
}

/// Center columns of a matrix and return (centered_matrix, column_means).
fn center_columns(data: &FdMatrix) -> (FdMatrix, Vec<f64>) {
    let (n, m) = data.shape();
    let mut centered = FdMatrix::zeros(n, m);
    let mut means = vec![0.0; m];
    for j in 0..m {
        let col = data.column(j);
        let mean = col.iter().sum::<f64>() / n as f64;
        means[j] = mean;
        let out_col = centered.column_mut(j);
        for i in 0..n {
            out_col[i] = col[i] - mean;
        }
    }
    (centered, means)
}

/// Extract rotation (V) and scores (U*S) from SVD results.
fn extract_pc_components(
    svd: &SVD<f64, nalgebra::Dyn, nalgebra::Dyn>,
    n: usize,
    m: usize,
    ncomp: usize,
) -> Option<(Vec<f64>, FdMatrix, FdMatrix)> {
    let singular_values: Vec<f64> = svd.singular_values.iter().take(ncomp).copied().collect();

    let v_t = svd.v_t.as_ref()?;
    let mut rotation = FdMatrix::zeros(m, ncomp);
    for k in 0..ncomp {
        for j in 0..m {
            rotation[(j, k)] = v_t[(k, j)];
        }
    }

    let u = svd.u.as_ref()?;
    let mut scores = FdMatrix::zeros(n, ncomp);
    for k in 0..ncomp {
        let sv_k = singular_values[k];
        for i in 0..n {
            scores[(i, k)] = u[(i, k)] * sv_k;
        }
    }

    Some((singular_values, rotation, scores))
}

/// Perform functional PCA via SVD on centered data.
///
/// # Arguments
/// * `data` - Matrix (n x m): n observations, m evaluation points
/// * `ncomp` - Number of components to extract
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or zero columns.
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero.
/// Returns [`FdarError::ComputationFailed`] if the SVD decomposition fails to
/// produce U or V_t matrices.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::regression::fdata_to_pc_1d;
///
/// // 5 curves, each evaluated at 10 points
/// let data = FdMatrix::from_column_major(
///     (0..50).map(|i| (i as f64 * 0.1).sin()).collect(),
///     5, 10,
/// ).unwrap();
/// let result = fdata_to_pc_1d(&data, 3).unwrap();
/// assert_eq!(result.scores.shape(), (5, 3));
/// assert_eq!(result.rotation.shape(), (10, 3));
/// assert_eq!(result.mean.len(), 10);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fdata_to_pc_1d(data: &FdMatrix, ncomp: usize) -> Result<FpcaResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0 rows".to_string(),
            actual: format!("n = {n}"),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "m > 0 columns".to_string(),
            actual: format!("m = {m}"),
        });
    }
    if ncomp < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: format!("ncomp must be >= 1, got {ncomp}"),
        });
    }

    let ncomp = ncomp.min(n).min(m);
    let (centered, means) = center_columns(data);
    let svd = SVD::new(centered.to_dmatrix(), true, true);
    let (singular_values, rotation, scores) =
        extract_pc_components(&svd, n, m, ncomp).ok_or_else(|| FdarError::ComputationFailed {
            operation: "SVD",
            detail: "failed to extract U or V_t from SVD decomposition".to_string(),
        })?;

    Ok(FpcaResult {
        singular_values,
        rotation,
        scores,
        mean: means,
        centered,
    })
}

/// Result of PLS regression.
#[derive(Debug, Clone, PartialEq)]
pub struct PlsResult {
    /// Weight vectors, m x ncomp
    pub weights: FdMatrix,
    /// Score vectors, n x ncomp
    pub scores: FdMatrix,
    /// Loading vectors, m x ncomp
    pub loadings: FdMatrix,
    /// Column means of the training data, length m
    pub x_means: Vec<f64>,
}

impl PlsResult {
    /// Project new functional data onto the PLS score space.
    ///
    /// Centers the input data by subtracting the column means estimated
    /// during PLS fitting, then iteratively projects and deflates through
    /// each PLS component using the stored weight and loading vectors.
    ///
    /// # Arguments
    /// * `data` - Matrix (n_new x m) of new observations
    ///
    /// # Errors
    ///
    /// Returns [`FdarError::InvalidDimension`] if the number of columns in
    /// `data` does not match the number of predictor variables used during
    /// PLS fitting.
    ///
    /// # Examples
    ///
    /// ```
    /// use fdars_core::matrix::FdMatrix;
    /// use fdars_core::regression::fdata_to_pls_1d;
    ///
    /// let x = FdMatrix::from_column_major(
    ///     (0..100).map(|i| (i as f64 * 0.1).sin()).collect(),
    ///     10, 10,
    /// ).unwrap();
    /// let y: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
    /// let pls = fdata_to_pls_1d(&x, &y, 3).unwrap();
    ///
    /// // Project the original data
    /// let scores = pls.project(&x).unwrap();
    /// assert_eq!(scores.shape(), (10, 3));
    ///
    /// // Project new data
    /// let new_x = FdMatrix::from_column_major(
    ///     (0..20).map(|i| (i as f64 * 0.2).cos()).collect(),
    ///     2, 10,
    /// ).unwrap();
    /// let new_scores = pls.project(&new_x).unwrap();
    /// assert_eq!(new_scores.shape(), (2, 3));
    /// ```
    pub fn project(&self, data: &FdMatrix) -> Result<FdMatrix, FdarError> {
        let (n, m) = data.shape();
        let ncomp = self.weights.ncols();
        if m != self.x_means.len() {
            return Err(FdarError::InvalidDimension {
                parameter: "data",
                expected: format!("{} columns", self.x_means.len()),
                actual: format!("{m} columns"),
            });
        }

        // Center data
        let mut x_cen = FdMatrix::zeros(n, m);
        for j in 0..m {
            for i in 0..n {
                x_cen[(i, j)] = data[(i, j)] - self.x_means[j];
            }
        }

        // Iteratively project and deflate through each component
        let mut scores = FdMatrix::zeros(n, ncomp);
        for k in 0..ncomp {
            // Compute scores: t = X_cen * w_k
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..m {
                    sum += x_cen[(i, j)] * self.weights[(j, k)];
                }
                scores[(i, k)] = sum;
            }

            // Deflate: X_cen -= t * p_k'
            for j in 0..m {
                let p_jk = self.loadings[(j, k)];
                for i in 0..n {
                    x_cen[(i, j)] -= scores[(i, k)] * p_jk;
                }
            }
        }

        Ok(scores)
    }
}

/// Compute PLS weight vector: w = X'y / ||X'y||
fn pls_compute_weights(x_cen: &FdMatrix, y_cen: &[f64]) -> Vec<f64> {
    let (n, m) = x_cen.shape();
    let mut w: Vec<f64> = (0..m)
        .map(|j| {
            let mut sum = 0.0;
            for i in 0..n {
                sum += x_cen[(i, j)] * y_cen[i];
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
    w
}

/// Compute PLS scores: t = Xw
fn pls_compute_scores(x_cen: &FdMatrix, w: &[f64]) -> Vec<f64> {
    let (n, m) = x_cen.shape();
    (0..n)
        .map(|i| {
            let mut sum = 0.0;
            for j in 0..m {
                sum += x_cen[(i, j)] * w[j];
            }
            sum
        })
        .collect()
}

/// Compute PLS loadings: p = X't / (t't)
fn pls_compute_loadings(x_cen: &FdMatrix, t: &[f64], t_norm_sq: f64) -> Vec<f64> {
    let (n, m) = x_cen.shape();
    (0..m)
        .map(|j| {
            let mut sum = 0.0;
            for i in 0..n {
                sum += x_cen[(i, j)] * t[i];
            }
            sum / t_norm_sq.max(1e-10)
        })
        .collect()
}

/// Deflate X by removing the rank-1 component t * p'
fn pls_deflate_x(x_cen: &mut FdMatrix, t: &[f64], p: &[f64]) {
    let (n, m) = x_cen.shape();
    for j in 0..m {
        for i in 0..n {
            x_cen[(i, j)] -= t[i] * p[j];
        }
    }
}

/// Execute one NIPALS step: compute weights/scores/loadings and deflate X and y.
fn pls_nipals_step(
    k: usize,
    x_cen: &mut FdMatrix,
    y_cen: &mut [f64],
    weights: &mut FdMatrix,
    scores: &mut FdMatrix,
    loadings: &mut FdMatrix,
) {
    let n = x_cen.nrows();
    let m = x_cen.ncols();

    let w = pls_compute_weights(x_cen, y_cen);
    let t = pls_compute_scores(x_cen, &w);
    let t_norm_sq: f64 = t.iter().map(|&ti| ti * ti).sum();
    let p = pls_compute_loadings(x_cen, &t, t_norm_sq);

    for j in 0..m {
        weights[(j, k)] = w[j];
        loadings[(j, k)] = p[j];
    }
    for i in 0..n {
        scores[(i, k)] = t[i];
    }

    pls_deflate_x(x_cen, &t, &p);
    let t_y: f64 = t.iter().zip(y_cen.iter()).map(|(&ti, &yi)| ti * yi).sum();
    let q = t_y / t_norm_sq.max(1e-10);
    for i in 0..n {
        y_cen[i] -= t[i] * q;
    }
}

/// Perform PLS via NIPALS algorithm.
///
/// # Arguments
/// * `data` - Matrix (n x m): n observations, m evaluation points
/// * `y` - Response vector (length n)
/// * `ncomp` - Number of components to extract
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or zero
/// columns, or if `y.len()` does not equal the number of rows in `data`.
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fdata_to_pls_1d(data: &FdMatrix, y: &[f64], ncomp: usize) -> Result<PlsResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0 rows".to_string(),
            actual: format!("n = {n}"),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "m > 0 columns".to_string(),
            actual: format!("m = {m}"),
        });
    }
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("length {n}"),
            actual: format!("length {}", y.len()),
        });
    }
    if ncomp < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: format!("ncomp must be >= 1, got {ncomp}"),
        });
    }

    let ncomp = ncomp.min(n).min(m);

    // Center X and y
    let x_means: Vec<f64> = (0..m)
        .map(|j| {
            let col = data.column(j);
            let sum: f64 = col.iter().sum();
            sum / n as f64
        })
        .collect();

    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;

    let mut x_cen = FdMatrix::zeros(n, m);
    for j in 0..m {
        for i in 0..n {
            x_cen[(i, j)] = data[(i, j)] - x_means[j];
        }
    }

    let mut y_cen: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();

    let mut weights = FdMatrix::zeros(m, ncomp);
    let mut scores = FdMatrix::zeros(n, ncomp);
    let mut loadings = FdMatrix::zeros(m, ncomp);

    // NIPALS algorithm
    for k in 0..ncomp {
        pls_nipals_step(
            k,
            &mut x_cen,
            &mut y_cen,
            &mut weights,
            &mut scores,
            &mut loadings,
        );
    }

    Ok(PlsResult {
        weights,
        scores,
        loadings,
        x_means,
    })
}

/// Result of ridge regression fit.
#[derive(Debug, Clone)]
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
/// * `x` - Predictor matrix (n x m)
/// * `y` - Response vector
/// * `lambda` - Regularization parameter
/// * `with_intercept` - Whether to include intercept
#[cfg(feature = "linalg")]
#[must_use = "expensive computation whose result should not be discarded"]
pub fn ridge_regression_fit(
    x: &FdMatrix,
    y: &[f64],
    lambda: f64,
    with_intercept: bool,
) -> RidgeResult {
    let (n, m) = x.shape();
    if n == 0 || m == 0 || y.len() != n {
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
    let x_faer = faer::Mat::from_fn(n, m, |i, j| x[(i, j)]);
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
                error: Some(format!("Fit failed: {e:?}")),
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
            pred += x[(i, j)] * coefficients[j];
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
    fn generate_test_fdata(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

        // Create n curves: sine waves with varying phase
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            let phase = (i as f64 / n as f64) * PI;
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j] + phase).sin();
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

        let result = fdata_to_pc_1d(&data, ncomp);
        assert!(result.is_ok());

        let fpca = result.unwrap();
        assert_eq!(fpca.singular_values.len(), ncomp);
        assert_eq!(fpca.rotation.shape(), (m, ncomp));
        assert_eq!(fpca.scores.shape(), (n, ncomp));
        assert_eq!(fpca.mean.len(), m);
        assert_eq!(fpca.centered.shape(), (n, m));
    }

    #[test]
    fn test_fdata_to_pc_1d_singular_values_decreasing() {
        let n = 20;
        let m = 50;
        let ncomp = 5;
        let (data, _) = generate_test_fdata(n, m);

        let fpca = fdata_to_pc_1d(&data, ncomp).unwrap();

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

        let fpca = fdata_to_pc_1d(&data, 3).unwrap();

        // Column means of centered data should be zero
        for j in 0..m {
            let col_mean: f64 = (0..n).map(|i| fpca.centered[(i, j)]).sum::<f64>() / n as f64;
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
        let fpca = fdata_to_pc_1d(&data, 20).unwrap();
        assert!(fpca.singular_values.len() <= n);
    }

    #[test]
    fn test_fdata_to_pc_1d_invalid_input() {
        // Empty data
        let empty = FdMatrix::zeros(0, 50);
        let result = fdata_to_pc_1d(&empty, 3);
        assert!(result.is_err());

        // Zero components
        let (data, _) = generate_test_fdata(10, 50);
        let result = fdata_to_pc_1d(&data, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_fdata_to_pc_1d_reconstruction() {
        let n = 10;
        let m = 30;
        let (data, _) = generate_test_fdata(n, m);

        // Use all components for perfect reconstruction
        let ncomp = n.min(m);
        let fpca = fdata_to_pc_1d(&data, ncomp).unwrap();

        // Reconstruct: X_centered = scores * rotation^T
        for i in 0..n {
            for j in 0..m {
                let mut reconstructed = 0.0;
                for k in 0..ncomp {
                    let score = fpca.scores[(i, k)];
                    let loading = fpca.rotation[(j, k)];
                    reconstructed += score * loading;
                }
                let original_centered = fpca.centered[(i, j)];
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

        let result = fdata_to_pls_1d(&x, &y, ncomp);
        assert!(result.is_ok());

        let pls = result.unwrap();
        assert_eq!(pls.weights.shape(), (m, ncomp));
        assert_eq!(pls.scores.shape(), (n, ncomp));
        assert_eq!(pls.loadings.shape(), (m, ncomp));
    }

    #[test]
    fn test_fdata_to_pls_1d_weights_normalized() {
        let n = 20;
        let m = 30;
        let ncomp = 2;
        let (x, _) = generate_test_fdata(n, m);
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let pls = fdata_to_pls_1d(&x, &y, ncomp).unwrap();

        // Weight vectors should be approximately unit norm
        for k in 0..ncomp {
            let norm: f64 = (0..m)
                .map(|j| pls.weights[(j, k)].powi(2))
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

        // Wrong y length
        let result = fdata_to_pls_1d(&x, &[0.0; 5], 2);
        assert!(result.is_err());

        // Zero components
        let y = vec![0.0; 10];
        let result = fdata_to_pls_1d(&x, &y, 0);
        assert!(result.is_err());
    }

    // ============== Ridge regression tests ==============

    #[cfg(feature = "linalg")]
    #[test]
    fn test_ridge_regression_fit_basic() {
        let n = 50;
        let m = 5;

        // Create X with known structure
        let mut x = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                x[(i, j)] = (i as f64 + j as f64) / (n + m) as f64;
            }
        }

        // Create y = sum of x columns + noise
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let mut sum = 0.0;
                for j in 0..m {
                    sum += x[(i, j)];
                }
                sum + 0.01 * (i as f64 % 10.0)
            })
            .collect();

        let result = ridge_regression_fit(&x, &y, 0.1, true);

        assert!(result.error.is_none(), "Ridge should fit without error");
        assert_eq!(result.coefficients.len(), m);
        assert_eq!(result.fitted_values.len(), n);
        assert_eq!(result.residuals.len(), n);
    }

    #[cfg(feature = "linalg")]
    #[test]
    fn test_ridge_regression_fit_r_squared() {
        let n = 50;
        let m = 3;

        let x = FdMatrix::from_column_major(
            (0..n * m).map(|i| i as f64 / (n * m) as f64).collect(),
            n,
            m,
        )
        .unwrap();
        let y: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

        let result = ridge_regression_fit(&x, &y, 0.01, true);

        assert!(
            result.r_squared > 0.5,
            "R-squared should be high, got {}",
            result.r_squared
        );
        assert!(result.r_squared <= 1.0 + 1e-10, "R-squared should be <= 1");
    }

    #[cfg(feature = "linalg")]
    #[test]
    fn test_ridge_regression_fit_regularization() {
        let n = 30;
        let m = 10;

        let x = FdMatrix::from_column_major(
            (0..n * m)
                .map(|i| ((i * 17) % 100) as f64 / 100.0)
                .collect(),
            n,
            m,
        )
        .unwrap();
        let y: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();

        let low_lambda = ridge_regression_fit(&x, &y, 0.001, true);
        let high_lambda = ridge_regression_fit(&x, &y, 100.0, true);

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

    #[cfg(feature = "linalg")]
    #[test]
    fn test_ridge_regression_fit_residuals() {
        let n = 20;
        let m = 3;

        let x = FdMatrix::from_column_major(
            (0..n * m).map(|i| i as f64 / (n * m) as f64).collect(),
            n,
            m,
        )
        .unwrap();
        let y: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

        let result = ridge_regression_fit(&x, &y, 0.1, true);

        for i in 0..n {
            let expected_resid = y[i] - result.fitted_values[i];
            assert!(
                (result.residuals[i] - expected_resid).abs() < 1e-10,
                "Residual mismatch at {}",
                i
            );
        }
    }

    #[cfg(feature = "linalg")]
    #[test]
    fn test_ridge_regression_fit_no_intercept() {
        let n = 30;
        let m = 5;

        let x = FdMatrix::from_column_major(
            (0..n * m).map(|i| i as f64 / (n * m) as f64).collect(),
            n,
            m,
        )
        .unwrap();
        let y: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

        let result = ridge_regression_fit(&x, &y, 0.1, false);

        assert!(result.error.is_none());
        assert!(
            result.intercept.abs() < 1e-10,
            "Intercept should be 0, got {}",
            result.intercept
        );
    }

    #[cfg(feature = "linalg")]
    #[test]
    fn test_ridge_regression_fit_invalid_input() {
        let empty = FdMatrix::zeros(0, 5);
        let result = ridge_regression_fit(&empty, &[], 0.1, true);
        assert!(result.error.is_some());

        let x = FdMatrix::zeros(10, 10);
        let y = vec![0.0; 5];
        let result = ridge_regression_fit(&x, &y, 0.1, true);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_all_zero_fpca() {
        // All-zero data: centering leaves zeros, SVD should return trivial result
        let n = 5;
        let m = 20;
        let data = FdMatrix::zeros(n, m);
        let result = fdata_to_pc_1d(&data, 2);
        // Should not panic; may return Ok with zero singular values
        if let Ok(res) = result {
            assert_eq!(res.scores.nrows(), n);
            for &sv in &res.singular_values {
                assert!(
                    sv.abs() < 1e-10,
                    "All-zero data should have zero singular values"
                );
            }
        }
    }

    #[test]
    fn test_n1_pca() {
        // Single observation: centering leaves all zeros, SVD may return trivial result
        let data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
        let result = fdata_to_pc_1d(&data, 1);
        // With n=1, centering leaves all zeros, so SVD may fail or return trivial result
        // Just ensure no panic
        let _ = result;
    }

    #[test]
    fn test_constant_y_pls() {
        let n = 10;
        let m = 20;
        let data_vec: Vec<f64> = (0..n * m).map(|i| (i as f64 * 0.1).sin()).collect();
        let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();
        let y = vec![5.0; n]; // Constant response
        let result = fdata_to_pls_1d(&data, &y, 2);
        // Constant y → centering makes y all zeros, PLS may fail
        // Just ensure no panic
        let _ = result;
    }

    // ============== FpcaResult::project tests ==============

    #[test]
    fn test_fpca_project_shape() {
        let n = 20;
        let m = 30;
        let ncomp = 3;
        let (data, _) = generate_test_fdata(n, m);
        let fpca = fdata_to_pc_1d(&data, ncomp).unwrap();

        let new_data = FdMatrix::zeros(5, m);
        let scores = fpca.project(&new_data).unwrap();
        assert_eq!(scores.shape(), (5, ncomp));
    }

    #[test]
    fn test_fpca_project_reproduces_training_scores() {
        let n = 20;
        let m = 30;
        let ncomp = 3;
        let (data, _) = generate_test_fdata(n, m);
        let fpca = fdata_to_pc_1d(&data, ncomp).unwrap();

        // Projecting the training data should reproduce the original scores
        let scores = fpca.project(&data).unwrap();
        for i in 0..n {
            for k in 0..ncomp {
                assert!(
                    (scores[(i, k)] - fpca.scores[(i, k)]).abs() < 1e-8,
                    "Score mismatch at ({}, {}): {} vs {}",
                    i,
                    k,
                    scores[(i, k)],
                    fpca.scores[(i, k)]
                );
            }
        }
    }

    #[test]
    fn test_fpca_project_dimension_mismatch() {
        let (data, _) = generate_test_fdata(20, 30);
        let fpca = fdata_to_pc_1d(&data, 3).unwrap();

        let wrong_m = FdMatrix::zeros(5, 20); // wrong number of columns
        assert!(fpca.project(&wrong_m).is_err());
    }

    // ============== FpcaResult::reconstruct tests ==============

    #[test]
    fn test_fpca_reconstruct_shape() {
        let n = 10;
        let m = 30;
        let ncomp = 5;
        let (data, _) = generate_test_fdata(n, m);
        let fpca = fdata_to_pc_1d(&data, ncomp).unwrap();

        let recon = fpca.reconstruct(&fpca.scores, 3).unwrap();
        assert_eq!(recon.shape(), (n, m));
    }

    #[test]
    fn test_fpca_reconstruct_full_matches_original() {
        let n = 10;
        let m = 30;
        let ncomp = n.min(m);
        let (data, _) = generate_test_fdata(n, m);
        let fpca = fdata_to_pc_1d(&data, ncomp).unwrap();

        // Full reconstruction should recover original data
        let recon = fpca.reconstruct(&fpca.scores, ncomp).unwrap();
        for i in 0..n {
            for j in 0..m {
                assert!(
                    (recon[(i, j)] - data[(i, j)]).abs() < 0.1,
                    "Reconstruction error at ({}, {}): {} vs {}",
                    i,
                    j,
                    recon[(i, j)],
                    data[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_fpca_reconstruct_fewer_components() {
        let n = 20;
        let m = 30;
        let ncomp = 5;
        let (data, _) = generate_test_fdata(n, m);
        let fpca = fdata_to_pc_1d(&data, ncomp).unwrap();

        let recon2 = fpca.reconstruct(&fpca.scores, 2).unwrap();
        let recon5 = fpca.reconstruct(&fpca.scores, 5).unwrap();
        assert_eq!(recon2.shape(), (n, m));
        assert_eq!(recon5.shape(), (n, m));
    }

    #[test]
    fn test_fpca_reconstruct_invalid_ncomp() {
        let (data, _) = generate_test_fdata(10, 30);
        let fpca = fdata_to_pc_1d(&data, 3).unwrap();

        // Zero components
        assert!(fpca.reconstruct(&fpca.scores, 0).is_err());
        // More components than available
        assert!(fpca.reconstruct(&fpca.scores, 10).is_err());
    }

    // ============== PlsResult::project tests ==============

    #[test]
    fn test_pls_project_shape() {
        let n = 20;
        let m = 30;
        let ncomp = 3;
        let (x, _) = generate_test_fdata(n, m);
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let pls = fdata_to_pls_1d(&x, &y, ncomp).unwrap();

        let new_x = FdMatrix::zeros(5, m);
        let scores = pls.project(&new_x).unwrap();
        assert_eq!(scores.shape(), (5, ncomp));
    }

    #[test]
    fn test_pls_project_reproduces_training_scores() {
        let n = 20;
        let m = 30;
        let ncomp = 3;
        let (x, _) = generate_test_fdata(n, m);
        let y: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) + 0.1).collect();
        let pls = fdata_to_pls_1d(&x, &y, ncomp).unwrap();

        // Projecting the training data should reproduce the original scores
        let scores = pls.project(&x).unwrap();
        for i in 0..n {
            for k in 0..ncomp {
                assert!(
                    (scores[(i, k)] - pls.scores[(i, k)]).abs() < 1e-8,
                    "Score mismatch at ({}, {}): {} vs {}",
                    i,
                    k,
                    scores[(i, k)],
                    pls.scores[(i, k)]
                );
            }
        }
    }

    #[test]
    fn test_pls_project_dimension_mismatch() {
        let (x, _) = generate_test_fdata(20, 30);
        let y: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let pls = fdata_to_pls_1d(&x, &y, 3).unwrap();

        let wrong_m = FdMatrix::zeros(5, 20); // wrong number of columns
        assert!(pls.project(&wrong_m).is_err());
    }

    #[test]
    fn test_pls_x_means_stored() {
        let n = 20;
        let m = 30;
        let (x, _) = generate_test_fdata(n, m);
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let pls = fdata_to_pls_1d(&x, &y, 3).unwrap();

        // x_means should be stored and have correct length
        assert_eq!(pls.x_means.len(), m);
    }
}
