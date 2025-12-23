//! Basis representation functions for functional data.
//!
//! This module provides B-spline and Fourier basis expansions for representing
//! functional data in a finite-dimensional basis.

use nalgebra::{DMatrix, DVector, SVD};
use rayon::prelude::*;
use std::f64::consts::PI;

/// Compute B-spline basis matrix for given knots and grid points.
///
/// Creates a B-spline basis with uniformly spaced knots extended beyond the data range.
/// For order k and nknots interior knots, produces nknots + order basis functions.
pub fn bspline_basis(t: &[f64], nknots: usize, order: usize) -> Vec<f64> {
    let n = t.len();
    // Total knots: order (left) + nknots (interior) + order (right) = 2*order + nknots
    // Number of B-spline basis functions: total_knots - order = nknots + order
    let nbasis = nknots + order;

    let t_min = t.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let dt = (t_max - t_min) / (nknots - 1) as f64;

    let mut knots = Vec::with_capacity(nknots + 2 * order);
    for i in 0..order {
        knots.push(t_min - (order - i) as f64 * dt);
    }
    for i in 0..nknots {
        knots.push(t_min + i as f64 * dt);
    }
    for i in 1..=order {
        knots.push(t_max + i as f64 * dt);
    }

    // Index of t_max in knot vector: it's the last interior knot
    let t_max_knot_idx = order + nknots - 1;

    let mut basis = vec![0.0; n * nbasis];

    for (ti, &t_val) in t.iter().enumerate() {
        let mut b0 = vec![0.0; knots.len() - 1];

        // Find which interval t_val belongs to
        // Use half-open intervals [knots[j], knots[j+1]) except at t_max
        // where we use the closed interval [knots[j], knots[j+1]]
        for j in 0..(knots.len() - 1) {
            let in_interval = if j == t_max_knot_idx - 1 {
                // Last interior interval: use closed [t_max - dt, t_max]
                t_val >= knots[j] && t_val <= knots[j + 1]
            } else {
                // Normal half-open interval [knots[j], knots[j+1])
                t_val >= knots[j] && t_val < knots[j + 1]
            };

            if in_interval {
                b0[j] = 1.0;
                break;
            }
        }

        let mut b = b0;
        for k in 2..=order {
            let mut b_new = vec![0.0; knots.len() - k];
            for j in 0..(knots.len() - k) {
                let d1 = knots[j + k - 1] - knots[j];
                let d2 = knots[j + k] - knots[j + 1];

                let left = if d1.abs() > 1e-10 {
                    (t_val - knots[j]) / d1 * b[j]
                } else {
                    0.0
                };
                let right = if d2.abs() > 1e-10 {
                    (knots[j + k] - t_val) / d2 * b[j + 1]
                } else {
                    0.0
                };
                b_new[j] = left + right;
            }
            b = b_new;
        }

        for j in 0..nbasis {
            basis[ti + j * n] = b[j];
        }
    }

    basis
}

/// Compute Fourier basis matrix.
pub fn fourier_basis(t: &[f64], nbasis: usize) -> Vec<f64> {
    let n = t.len();
    let t_min = t.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let period = t_max - t_min;

    let mut basis = vec![0.0; n * nbasis];

    for (i, &ti) in t.iter().enumerate() {
        let x = 2.0 * PI * (ti - t_min) / period;

        basis[i] = 1.0;

        let mut k = 1;
        let mut freq = 1;
        while k < nbasis {
            if k < nbasis {
                basis[i + k * n] = (freq as f64 * x).sin();
                k += 1;
            }
            if k < nbasis {
                basis[i + k * n] = (freq as f64 * x).cos();
                k += 1;
            }
            freq += 1;
        }
    }

    basis
}

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

/// Result of basis projection.
pub struct BasisProjectionResult {
    /// Coefficient matrix (n_samples x n_basis)
    pub coefficients: Vec<f64>,
    /// Number of basis functions used
    pub n_basis: usize,
}

/// Project functional data to basis coefficients.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `nbasis` - Number of basis functions
/// * `basis_type` - 0 = B-spline, 1 = Fourier
pub fn fdata_to_basis_1d(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    nbasis: usize,
    basis_type: i32,
) -> Option<BasisProjectionResult> {
    if n == 0 || m == 0 || argvals.len() != m || nbasis < 2 {
        return None;
    }

    let basis = if basis_type == 1 {
        fourier_basis(argvals, nbasis)
    } else {
        bspline_basis(argvals, nbasis + 2, 4)
    };

    let actual_nbasis = basis.len() / m;
    let b_mat = DMatrix::from_column_slice(m, actual_nbasis, &basis);

    let btb = &b_mat.transpose() * &b_mat;
    let btb_svd = SVD::new(btb, true, true);

    let max_sv = btb_svd.singular_values.iter().cloned().fold(0.0, f64::max);
    let eps = 1e-10 * max_sv;

    let s_inv: Vec<f64> = btb_svd
        .singular_values
        .iter()
        .map(|&s| if s > eps { 1.0 / s } else { 0.0 })
        .collect();

    let v = btb_svd.v_t.as_ref()?.transpose();
    let u_t = btb_svd.u.as_ref()?.transpose();

    let mut btb_inv = DMatrix::zeros(actual_nbasis, actual_nbasis);
    for i in 0..actual_nbasis {
        for j in 0..actual_nbasis {
            let mut sum = 0.0;
            for k in 0..actual_nbasis.min(s_inv.len()) {
                sum += v[(i, k)] * s_inv[k] * u_t[(k, j)];
            }
            btb_inv[(i, j)] = sum;
        }
    }

    let proj = btb_inv * b_mat.transpose();

    let coefs: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let curve: Vec<f64> = (0..m).map(|j| data[i + j * n]).collect();
            (0..actual_nbasis)
                .map(|k| {
                    let mut sum = 0.0;
                    for j in 0..m {
                        sum += proj[(k, j)] * curve[j];
                    }
                    sum
                })
                .collect::<Vec<_>>()
        })
        .collect();

    Some(BasisProjectionResult {
        coefficients: coefs,
        n_basis: actual_nbasis,
    })
}

/// Reconstruct functional data from basis coefficients.
pub fn basis_to_fdata_1d(
    coefs: &[f64],
    n: usize,
    coefs_ncols: usize,
    argvals: &[f64],
    nbasis: usize,
    basis_type: i32,
) -> Vec<f64> {
    let m = argvals.len();
    if n == 0 || m == 0 || nbasis < 2 {
        return Vec::new();
    }

    let basis = if basis_type == 1 {
        fourier_basis(argvals, nbasis)
    } else {
        bspline_basis(argvals, nbasis + 2, 4)
    };

    let actual_nbasis = basis.len() / m;

    (0..n)
        .into_par_iter()
        .flat_map(|i| {
            (0..m)
                .map(|j| {
                    let mut sum = 0.0;
                    for k in 0..actual_nbasis.min(coefs_ncols) {
                        sum += coefs[i + k * n] * basis[j + k * m];
                    }
                    sum
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Result of P-spline fitting.
pub struct PsplineFitResult {
    /// Coefficient matrix
    pub coefficients: Vec<f64>,
    /// Fitted values
    pub fitted: Vec<f64>,
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
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    nbasis: usize,
    lambda: f64,
    order: usize,
) -> Option<PsplineFitResult> {
    if n == 0 || m == 0 || nbasis < 2 || argvals.len() != m {
        return None;
    }

    let basis = bspline_basis(argvals, nbasis + 2, 4);
    let actual_nbasis = basis.len() / m;
    let b_mat = DMatrix::from_column_slice(m, actual_nbasis, &basis);

    let d = difference_matrix(actual_nbasis, order);
    let penalty = &d.transpose() * &d;

    let btb = &b_mat.transpose() * &b_mat;
    let btb_penalized = &btb + lambda * &penalty;

    let btb_inv = btb_penalized.clone().try_inverse()?;

    let proj = &btb_inv * b_mat.transpose();
    let h_mat = &b_mat * &proj;
    let edf: f64 = (0..m).map(|i| h_mat[(i, i)]).sum();

    let mut all_coefs = vec![0.0; n * actual_nbasis];
    let mut all_fitted = vec![0.0; n * m];
    let mut total_rss = 0.0;

    for i in 0..n {
        let curve: Vec<f64> = (0..m).map(|j| data[i + j * n]).collect();
        let curve_vec = DVector::from_vec(curve.clone());

        let bt_y = b_mat.transpose() * &curve_vec;
        let coefs = &btb_inv * bt_y;

        for k in 0..actual_nbasis {
            all_coefs[i + k * n] = coefs[k];
        }

        let fitted = &b_mat * &coefs;
        for j in 0..m {
            all_fitted[i + j * n] = fitted[j];
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
