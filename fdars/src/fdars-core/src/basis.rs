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
///
/// The period is automatically set to the range of evaluation points (t_max - t_min).
/// For explicit period control, use `fourier_basis_with_period`.
pub fn fourier_basis(t: &[f64], nbasis: usize) -> Vec<f64> {
    let t_min = t.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let period = t_max - t_min;
    fourier_basis_with_period(t, nbasis, period)
}

/// Compute Fourier basis matrix with explicit period.
///
/// This function creates a Fourier basis expansion where the period can be specified
/// independently of the evaluation range. This is essential for seasonal analysis
/// where the seasonal period may differ from the observation window.
///
/// # Arguments
/// * `t` - Evaluation points
/// * `nbasis` - Number of basis functions (1 constant + pairs of sin/cos)
/// * `period` - The period for the Fourier basis
///
/// # Returns
/// Column-major matrix (n_points x nbasis) stored as flat vector
pub fn fourier_basis_with_period(t: &[f64], nbasis: usize, period: f64) -> Vec<f64> {
    let n = t.len();
    let t_min = t.iter().cloned().fold(f64::INFINITY, f64::min);

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
        // For order 4 B-splines: nbasis = nknots + order, so nknots = nbasis - 4
        bspline_basis(argvals, nbasis.saturating_sub(4).max(2), 4)
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
        // For order 4 B-splines: nbasis = nknots + order, so nknots = nbasis - 4
        bspline_basis(argvals, nbasis.saturating_sub(4).max(2), 4)
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

    // For order 4 B-splines: nbasis = nknots + order, so nknots = nbasis - 4
    let basis = bspline_basis(argvals, nbasis.saturating_sub(4).max(2), 4);
    let actual_nbasis = basis.len() / m;
    let b_mat = DMatrix::from_column_slice(m, actual_nbasis, &basis);

    let d = difference_matrix(actual_nbasis, order);
    let penalty = &d.transpose() * &d;

    let btb = &b_mat.transpose() * &b_mat;
    let btb_penalized = &btb + lambda * &penalty;

    // Use SVD pseudoinverse for robustness with singular matrices
    let svd = SVD::new(btb_penalized.clone(), true, true);
    let max_sv = svd.singular_values.iter().cloned().fold(0.0_f64, f64::max);
    let eps = 1e-10 * max_sv;

    let u = svd.u.as_ref()?;
    let v_t = svd.v_t.as_ref()?;

    let s_inv: Vec<f64> = svd.singular_values.iter()
        .map(|&s| if s > eps { 1.0 / s } else { 0.0 })
        .collect();

    let mut btb_inv = DMatrix::zeros(actual_nbasis, actual_nbasis);
    for i in 0..actual_nbasis {
        for j in 0..actual_nbasis {
            let mut sum = 0.0;
            for k in 0..actual_nbasis.min(s_inv.len()) {
                sum += v_t[(k, i)] * s_inv[k] * u[(j, k)];
            }
            btb_inv[(i, j)] = sum;
        }
    }

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

/// Result of Fourier basis fitting.
pub struct FourierFitResult {
    /// Coefficient matrix
    pub coefficients: Vec<f64>,
    /// Fitted values
    pub fitted: Vec<f64>,
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
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `nbasis` - Number of Fourier basis functions (should be odd: 1 constant + pairs of sin/cos)
///
/// # Returns
/// FourierFitResult with coefficients, fitted values, and model selection criteria
pub fn fourier_fit_1d(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    nbasis: usize,
) -> Option<FourierFitResult> {
    if n == 0 || m == 0 || nbasis < 3 || argvals.len() != m {
        return None;
    }

    // Ensure nbasis is odd (1 constant + pairs of sin/cos)
    let nbasis = if nbasis % 2 == 0 { nbasis + 1 } else { nbasis };

    let basis = fourier_basis(argvals, nbasis);
    let actual_nbasis = basis.len() / m;
    let b_mat = DMatrix::from_column_slice(m, actual_nbasis, &basis);

    let btb = &b_mat.transpose() * &b_mat;

    // Use SVD pseudoinverse for robustness
    let svd = SVD::new(btb.clone(), true, true);
    let max_sv = svd.singular_values.iter().cloned().fold(0.0_f64, f64::max);
    let eps = 1e-10 * max_sv;

    let u = svd.u.as_ref()?;
    let v_t = svd.v_t.as_ref()?;

    let s_inv: Vec<f64> = svd
        .singular_values
        .iter()
        .map(|&s| if s > eps { 1.0 / s } else { 0.0 })
        .collect();

    let mut btb_inv = DMatrix::zeros(actual_nbasis, actual_nbasis);
    for i in 0..actual_nbasis {
        for j in 0..actual_nbasis {
            let mut sum = 0.0;
            for k in 0..actual_nbasis.min(s_inv.len()) {
                sum += v_t[(k, i)] * s_inv[k] * u[(j, k)];
            }
            btb_inv[(i, j)] = sum;
        }
    }

    let proj = &btb_inv * b_mat.transpose();

    // For unpenalized fit, hat matrix H = B * (B'B)^{-1} * B'
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

    // GCV: RSS / n * (1 - edf/m)^2
    let gcv_denom = 1.0 - edf / m as f64;
    let gcv = if gcv_denom.abs() > 1e-10 {
        (total_rss / total_points) / (gcv_denom * gcv_denom)
    } else {
        f64::INFINITY
    };

    let mse = total_rss / total_points;
    let aic = total_points * mse.ln() + 2.0 * edf;
    let bic = total_points * mse.ln() + total_points.ln() * edf;

    Some(FourierFitResult {
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
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `min_nbasis` - Minimum number of basis functions to try
/// * `max_nbasis` - Maximum number of basis functions to try
///
/// # Returns
/// Optimal number of basis functions
pub fn select_fourier_nbasis_gcv(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    min_nbasis: usize,
    max_nbasis: usize,
) -> usize {
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
        if let Some(result) = fourier_fit_1d(data, n, m, argvals, nbasis) {
            if result.gcv < best_gcv && result.gcv.is_finite() {
                best_gcv = result.gcv;
                best_nbasis = nbasis;
            }
        }
        nbasis += 2;
    }

    best_nbasis
}

/// Result of automatic basis selection for a single curve.
#[derive(Clone)]
pub struct SingleCurveSelection {
    /// Selected basis type: 0 = P-spline, 1 = Fourier
    pub basis_type: i32,
    /// Selected number of basis functions
    pub nbasis: usize,
    /// Best criterion score (GCV, AIC, or BIC)
    pub score: f64,
    /// Coefficients for the selected basis
    pub coefficients: Vec<f64>,
    /// Fitted values
    pub fitted: Vec<f64>,
    /// Effective degrees of freedom
    pub edf: f64,
    /// Whether seasonal pattern was detected (if use_seasonal_hint)
    pub seasonal_detected: bool,
    /// Lambda value (for P-spline, NaN for Fourier)
    pub lambda: f64,
}

/// Result of automatic basis selection for all curves.
pub struct BasisAutoSelectionResult {
    /// Per-curve selection results
    pub selections: Vec<SingleCurveSelection>,
    /// Criterion used (0=GCV, 1=AIC, 2=BIC)
    pub criterion: i32,
}

/// Detect if a curve has seasonal/periodic pattern using FFT.
///
/// Returns true if the peak power in the periodogram is significantly
/// above the mean power level.
fn detect_seasonality_fft(curve: &[f64]) -> bool {
    use rustfft::{FftPlanner, num_complex::Complex};

    let n = curve.len();
    if n < 8 {
        return false;
    }

    // Remove mean
    let mean: f64 = curve.iter().sum::<f64>() / n as f64;
    let mut input: Vec<Complex<f64>> = curve.iter()
        .map(|&x| Complex::new(x - mean, 0.0))
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut input);

    // Compute power spectrum (skip DC component and Nyquist)
    let powers: Vec<f64> = input[1..n/2].iter()
        .map(|c| c.norm_sqr())
        .collect();

    if powers.is_empty() {
        return false;
    }

    let max_power = powers.iter().cloned().fold(0.0_f64, f64::max);
    let mean_power = powers.iter().sum::<f64>() / powers.len() as f64;

    // Seasonal if peak is significantly above mean
    max_power > 2.0 * mean_power
}

/// Fit a single curve with Fourier basis and compute criterion score.
fn fit_curve_fourier(
    curve: &[f64],
    m: usize,
    argvals: &[f64],
    nbasis: usize,
    criterion: i32,
) -> Option<(f64, Vec<f64>, Vec<f64>, f64)> {
    // Ensure nbasis is odd
    let nbasis = if nbasis % 2 == 0 { nbasis + 1 } else { nbasis };

    let basis = fourier_basis(argvals, nbasis);
    let actual_nbasis = basis.len() / m;
    let b_mat = DMatrix::from_column_slice(m, actual_nbasis, &basis);

    let btb = &b_mat.transpose() * &b_mat;
    let svd = SVD::new(btb.clone(), true, true);
    let max_sv = svd.singular_values.iter().cloned().fold(0.0_f64, f64::max);
    let eps = 1e-10 * max_sv;

    let u = svd.u.as_ref()?;
    let v_t = svd.v_t.as_ref()?;

    let s_inv: Vec<f64> = svd.singular_values.iter()
        .map(|&s| if s > eps { 1.0 / s } else { 0.0 })
        .collect();

    let mut btb_inv = DMatrix::zeros(actual_nbasis, actual_nbasis);
    for i in 0..actual_nbasis {
        for j in 0..actual_nbasis {
            let mut sum = 0.0;
            for k in 0..actual_nbasis.min(s_inv.len()) {
                sum += v_t[(k, i)] * s_inv[k] * u[(j, k)];
            }
            btb_inv[(i, j)] = sum;
        }
    }

    let proj = &btb_inv * b_mat.transpose();
    let h_mat = &b_mat * &proj;
    let edf: f64 = (0..m).map(|i| h_mat[(i, i)]).sum();

    let curve_vec = DVector::from_column_slice(curve);
    let bt_y = b_mat.transpose() * &curve_vec;
    let coefs = &btb_inv * bt_y;

    let fitted = &b_mat * &coefs;
    let mut rss = 0.0;
    for j in 0..m {
        let resid = curve[j] - fitted[j];
        rss += resid * resid;
    }

    let n_points = m as f64;
    let score = match criterion {
        0 => { // GCV
            let gcv_denom = 1.0 - edf / n_points;
            if gcv_denom.abs() > 1e-10 {
                (rss / n_points) / (gcv_denom * gcv_denom)
            } else {
                f64::INFINITY
            }
        }
        1 => { // AIC
            let mse = rss / n_points;
            n_points * mse.ln() + 2.0 * edf
        }
        _ => { // BIC
            let mse = rss / n_points;
            n_points * mse.ln() + n_points.ln() * edf
        }
    };

    let coef_vec: Vec<f64> = (0..actual_nbasis).map(|k| coefs[k]).collect();
    let fitted_vec: Vec<f64> = (0..m).map(|j| fitted[j]).collect();

    Some((score, coef_vec, fitted_vec, edf))
}

/// Fit a single curve with P-spline basis and compute criterion score.
fn fit_curve_pspline(
    curve: &[f64],
    m: usize,
    argvals: &[f64],
    nbasis: usize,
    lambda: f64,
    order: usize,
    criterion: i32,
) -> Option<(f64, Vec<f64>, Vec<f64>, f64)> {
    let basis = bspline_basis(argvals, nbasis.saturating_sub(4).max(2), 4);
    let actual_nbasis = basis.len() / m;
    let b_mat = DMatrix::from_column_slice(m, actual_nbasis, &basis);

    let d = difference_matrix(actual_nbasis, order);
    let penalty = &d.transpose() * &d;

    let btb = &b_mat.transpose() * &b_mat;
    let btb_penalized = &btb + lambda * &penalty;

    let svd = SVD::new(btb_penalized.clone(), true, true);
    let max_sv = svd.singular_values.iter().cloned().fold(0.0_f64, f64::max);
    let eps = 1e-10 * max_sv;

    let u = svd.u.as_ref()?;
    let v_t = svd.v_t.as_ref()?;

    let s_inv: Vec<f64> = svd.singular_values.iter()
        .map(|&s| if s > eps { 1.0 / s } else { 0.0 })
        .collect();

    let mut btb_inv = DMatrix::zeros(actual_nbasis, actual_nbasis);
    for i in 0..actual_nbasis {
        for j in 0..actual_nbasis {
            let mut sum = 0.0;
            for k in 0..actual_nbasis.min(s_inv.len()) {
                sum += v_t[(k, i)] * s_inv[k] * u[(j, k)];
            }
            btb_inv[(i, j)] = sum;
        }
    }

    let proj = &btb_inv * b_mat.transpose();
    let h_mat = &b_mat * &proj;
    let edf: f64 = (0..m).map(|i| h_mat[(i, i)]).sum();

    let curve_vec = DVector::from_column_slice(curve);
    let bt_y = b_mat.transpose() * &curve_vec;
    let coefs = &btb_inv * bt_y;

    let fitted = &b_mat * &coefs;
    let mut rss = 0.0;
    for j in 0..m {
        let resid = curve[j] - fitted[j];
        rss += resid * resid;
    }

    let n_points = m as f64;
    let score = match criterion {
        0 => { // GCV
            let gcv_denom = 1.0 - edf / n_points;
            if gcv_denom.abs() > 1e-10 {
                (rss / n_points) / (gcv_denom * gcv_denom)
            } else {
                f64::INFINITY
            }
        }
        1 => { // AIC
            let mse = rss / n_points;
            n_points * mse.ln() + 2.0 * edf
        }
        _ => { // BIC
            let mse = rss / n_points;
            n_points * mse.ln() + n_points.ln() * edf
        }
    };

    let coef_vec: Vec<f64> = (0..actual_nbasis).map(|k| coefs[k]).collect();
    let fitted_vec: Vec<f64> = (0..m).map(|j| fitted[j]).collect();

    Some((score, coef_vec, fitted_vec, edf))
}

/// Select optimal basis type and parameters for each curve individually.
///
/// This function compares Fourier and P-spline bases for each curve,
/// selecting the optimal basis type and number of basis functions using
/// model selection criteria (GCV, AIC, or BIC).
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of curves
/// * `m` - Number of evaluation points per curve
/// * `argvals` - Evaluation points
/// * `criterion` - Model selection criterion: 0=GCV, 1=AIC, 2=BIC
/// * `nbasis_min` - Minimum number of basis functions (0 for auto)
/// * `nbasis_max` - Maximum number of basis functions (0 for auto)
/// * `lambda_pspline` - Smoothing parameter for P-spline (negative for auto-select)
/// * `use_seasonal_hint` - Whether to use FFT to detect seasonality
///
/// # Returns
/// BasisAutoSelectionResult with per-curve selections
pub fn select_basis_auto_1d(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    criterion: i32,
    nbasis_min: usize,
    nbasis_max: usize,
    lambda_pspline: f64,
    use_seasonal_hint: bool,
) -> BasisAutoSelectionResult {
    // Determine nbasis ranges
    let fourier_min = if nbasis_min > 0 { nbasis_min.max(3) } else { 3 };
    let fourier_max = if nbasis_max > 0 { nbasis_max.min(m / 3).min(25) } else { (m / 3).min(25) };

    let pspline_min = if nbasis_min > 0 { nbasis_min.max(6) } else { 6 };
    let pspline_max = if nbasis_max > 0 { nbasis_max.min(m / 2).min(40) } else { (m / 2).min(40) };

    // Lambda grid for P-spline when auto-selecting
    let lambda_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0];
    let auto_lambda = lambda_pspline < 0.0;

    let selections: Vec<SingleCurveSelection> = (0..n)
        .into_par_iter()
        .map(|i| {
            // Extract single curve
            let curve: Vec<f64> = (0..m).map(|j| data[i + j * n]).collect();

            // Detect seasonality if requested
            let seasonal_detected = if use_seasonal_hint {
                detect_seasonality_fft(&curve)
            } else {
                false
            };

            let mut best_score = f64::INFINITY;
            let mut best_basis_type = 0i32; // P-spline
            let mut best_nbasis = pspline_min;
            let mut best_coefs = Vec::new();
            let mut best_fitted = Vec::new();
            let mut best_edf = 0.0;
            let mut best_lambda = f64::NAN;

            // Try Fourier bases
            let fourier_start = if seasonal_detected { fourier_min.max(5) } else { fourier_min };
            let mut nb = if fourier_start % 2 == 0 { fourier_start + 1 } else { fourier_start };
            while nb <= fourier_max {
                if let Some((score, coefs, fitted, edf)) =
                    fit_curve_fourier(&curve, m, argvals, nb, criterion)
                {
                    if score < best_score && score.is_finite() {
                        best_score = score;
                        best_basis_type = 1; // Fourier
                        best_nbasis = nb;
                        best_coefs = coefs;
                        best_fitted = fitted;
                        best_edf = edf;
                        best_lambda = f64::NAN;
                    }
                }
                nb += 2;
            }

            // Try P-spline bases
            for nb in pspline_min..=pspline_max {
                if auto_lambda {
                    // Search over lambda grid
                    for &lam in &lambda_grid {
                        if let Some((score, coefs, fitted, edf)) =
                            fit_curve_pspline(&curve, m, argvals, nb, lam, 2, criterion)
                        {
                            if score < best_score && score.is_finite() {
                                best_score = score;
                                best_basis_type = 0; // P-spline
                                best_nbasis = nb;
                                best_coefs = coefs;
                                best_fitted = fitted;
                                best_edf = edf;
                                best_lambda = lam;
                            }
                        }
                    }
                } else {
                    // Use fixed lambda
                    if let Some((score, coefs, fitted, edf)) =
                        fit_curve_pspline(&curve, m, argvals, nb, lambda_pspline, 2, criterion)
                    {
                        if score < best_score && score.is_finite() {
                            best_score = score;
                            best_basis_type = 0; // P-spline
                            best_nbasis = nb;
                            best_coefs = coefs;
                            best_fitted = fitted;
                            best_edf = edf;
                            best_lambda = lambda_pspline;
                        }
                    }
                }
            }

            SingleCurveSelection {
                basis_type: best_basis_type,
                nbasis: best_nbasis,
                score: best_score,
                coefficients: best_coefs,
                fitted: best_fitted,
                edf: best_edf,
                seasonal_detected,
                lambda: best_lambda,
            }
        })
        .collect();

    BasisAutoSelectionResult {
        selections,
        criterion,
    }
}
