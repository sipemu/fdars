//! Fourier basis fitting and GCV-based selection.

use super::fourier::fourier_basis;
use super::helpers::{compute_fit_criteria, svd_pseudoinverse};
use crate::matrix::FdMatrix;
use nalgebra::{DMatrix, DVector};

/// Result of Fourier basis fitting.
#[derive(Debug, Clone)]
pub struct FourierFitResult {
    /// Coefficient matrix (n x nbasis)
    pub coefficients: FdMatrix,
    /// Fitted values (n x m)
    pub fitted: FdMatrix,
    /// Effective degrees of freedom (equals nbasis for unpenalized fit)
    pub edf: f64,
    /// Residual sum of squares
    pub rss: f64,
    /// GCV score
    pub gcv: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of basis functions
    pub n_basis: usize,
}

/// Fit Fourier basis to functional data using least squares.
///
/// Projects data onto Fourier basis and reconstructs fitted values.
/// Unlike P-splines, this uses unpenalized least squares projection.
///
/// # Arguments
/// * `data` - Column-major FdMatrix (n x m)
/// * `argvals` - Evaluation points
/// * `nbasis` - Number of Fourier basis functions (should be odd: 1 constant + pairs of sin/cos)
///
/// # Returns
/// FourierFitResult with coefficients, fitted values, and model selection criteria
pub fn fourier_fit_1d(
    data: &FdMatrix,
    argvals: &[f64],
    nbasis: usize,
) -> Result<FourierFitResult, crate::FdarError> {
    let n = data.nrows();
    let m = data.ncols();
    if n == 0 || m == 0 || nbasis < 3 || argvals.len() != m {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "data/argvals/nbasis",
            expected: "n > 0, m > 0, nbasis >= 3, argvals.len() == m".to_string(),
            actual: format!(
                "n={}, m={}, nbasis={}, argvals.len()={}",
                n,
                m,
                nbasis,
                argvals.len()
            ),
        });
    }

    // Ensure nbasis is odd (1 constant + pairs of sin/cos)
    let nbasis = if nbasis % 2 == 0 { nbasis + 1 } else { nbasis };

    let basis = fourier_basis(argvals, nbasis);
    let actual_nbasis = basis.len() / m;
    let b_mat = DMatrix::from_column_slice(m, actual_nbasis, &basis);

    let btb = &b_mat.transpose() * &b_mat;
    let btb_inv = svd_pseudoinverse(&btb).ok_or_else(|| crate::FdarError::ComputationFailed {
        operation: "SVD pseudoinverse",
        detail: "failed to compute pseudoinverse of B^T B in fourier_fit_1d".to_string(),
    })?;
    let proj = &btb_inv * b_mat.transpose();
    let h_mat = &b_mat * &proj;
    let edf: f64 = (0..m).map(|i| h_mat[(i, i)]).sum();

    let mut all_coefs = FdMatrix::zeros(n, actual_nbasis);
    let mut all_fitted = FdMatrix::zeros(n, m);
    let mut total_rss = 0.0;

    for i in 0..n {
        let curve: Vec<f64> = (0..m).map(|j| data[(i, j)]).collect();
        let curve_vec = DVector::from_vec(curve.clone());

        let bt_y = b_mat.transpose() * &curve_vec;
        let coefs = &btb_inv * bt_y;

        for k in 0..actual_nbasis {
            all_coefs[(i, k)] = coefs[k];
        }

        let fitted = &b_mat * &coefs;
        for j in 0..m {
            all_fitted[(i, j)] = fitted[j];
            let resid = curve[j] - fitted[j];
            total_rss += resid * resid;
        }
    }

    let total_points = (n * m) as f64;
    let (gcv, aic, bic) = compute_fit_criteria(total_rss, total_points, edf, m);

    Ok(FourierFitResult {
        coefficients: all_coefs,
        fitted: all_fitted,
        edf,
        rss: total_rss,
        gcv,
        aic,
        bic,
        n_basis: actual_nbasis,
    })
}

/// Select optimal number of Fourier basis functions using GCV.
///
/// Performs grid search over nbasis values and returns the one with minimum GCV.
///
/// # Arguments
/// * `data` - Column-major FdMatrix (n x m)
/// * `argvals` - Evaluation points
/// * `min_nbasis` - Minimum number of basis functions to try
/// * `max_nbasis` - Maximum number of basis functions to try
///
/// # Returns
/// Optimal number of basis functions
pub fn select_fourier_nbasis_gcv(
    data: &FdMatrix,
    argvals: &[f64],
    min_nbasis: usize,
    max_nbasis: usize,
) -> usize {
    let m = data.ncols();
    let min_nb = min_nbasis.max(3);
    // Ensure max doesn't exceed m (can't have more parameters than data points)
    let max_nb = max_nbasis.min(m / 2);

    if max_nb <= min_nb {
        return min_nb;
    }

    let mut best_nbasis = min_nb;
    let mut best_gcv = f64::INFINITY;

    // Test odd values only (1 constant + pairs of sin/cos)
    let mut nbasis = if min_nb % 2 == 0 { min_nb + 1 } else { min_nb };
    while nbasis <= max_nb {
        if let Ok(result) = fourier_fit_1d(data, argvals, nbasis) {
            if result.gcv < best_gcv && result.gcv.is_finite() {
                best_gcv = result.gcv;
                best_nbasis = nbasis;
            }
        }
        nbasis += 2;
    }

    best_nbasis
}
