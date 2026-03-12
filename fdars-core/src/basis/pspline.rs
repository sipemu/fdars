//! P-spline fitting and penalty (difference) matrices.

use super::bspline::bspline_basis;
use super::helpers::svd_pseudoinverse;
use crate::matrix::FdMatrix;
use nalgebra::{DMatrix, DVector};

/// Compute difference matrix for P-spline penalty.
pub fn difference_matrix(n: usize, order: usize) -> DMatrix<f64> {
    if order == 0 {
        return DMatrix::identity(n, n);
    }

    let mut d = DMatrix::zeros(n - 1, n);
    for i in 0..(n - 1) {
        d[(i, i)] = -1.0;
        d[(i, i + 1)] = 1.0;
    }

    let mut result = d;
    for _ in 1..order {
        if result.nrows() <= 1 {
            break;
        }
        let rows = result.nrows() - 1;
        let cols = result.ncols();
        let mut d_next = DMatrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                d_next[(i, j)] = -result[(i, j)] + result[(i + 1, j)];
            }
        }
        result = d_next;
    }

    result
}

/// Result of P-spline fitting.
#[derive(Debug, Clone, PartialEq)]
pub struct PsplineFitResult {
    /// Coefficient matrix (n x nbasis)
    pub coefficients: FdMatrix,
    /// Fitted values (n x m)
    pub fitted: FdMatrix,
    /// Effective degrees of freedom
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

/// Fit P-splines to functional data.
pub fn pspline_fit_1d(
    data: &FdMatrix,
    argvals: &[f64],
    nbasis: usize,
    lambda: f64,
    order: usize,
) -> Option<PsplineFitResult> {
    let n = data.nrows();
    let m = data.ncols();
    if n == 0 || m == 0 || nbasis < 2 || argvals.len() != m {
        return None;
    }

    // For order 4 B-splines: nbasis = nknots + order, so nknots = nbasis - 4
    let basis = bspline_basis(argvals, nbasis.saturating_sub(4).max(2), 4);
    let actual_nbasis = basis.len() / m;
    let b_mat = DMatrix::from_column_slice(m, actual_nbasis, &basis);

    let d = difference_matrix(actual_nbasis, order);
    let penalty = &d.transpose() * &d;

    let btb = &b_mat.transpose() * &b_mat;
    let btb_penalized = &btb + lambda * &penalty;

    let btb_inv = svd_pseudoinverse(&btb_penalized)?;
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

    let gcv_denom = 1.0 - edf / m as f64;
    let gcv = if gcv_denom.abs() > 1e-10 {
        (total_rss / total_points) / (gcv_denom * gcv_denom)
    } else {
        f64::INFINITY
    };

    let mse = total_rss / total_points;
    let aic = total_points * mse.ln() + 2.0 * edf;
    let bic = total_points * mse.ln() + total_points.ln() * edf;

    Some(PsplineFitResult {
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
