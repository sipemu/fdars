//! Basis-penalized smoothing with continuous derivative penalties.
//!
//! This module implements `smooth.basis` from R's fda package. Unlike the
//! discrete difference penalty used in P-splines (`basis.rs`), this uses
//! continuous derivative penalties: `min ||y - Φc||² + λ·∫(Lf)² dt`.
//!
//! Key capabilities:
//! - [`smooth_basis`] — Penalized least squares with continuous roughness penalty
//! - [`smooth_basis_gcv`] — GCV-optimal smoothing parameter selection
//! - [`bspline_penalty_matrix`] / [`fourier_penalty_matrix`] — Roughness penalty matrices

use crate::basis::{bspline_basis, fourier_basis_with_period};
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;
use nalgebra::DMatrix;
use std::f64::consts::PI;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Basis type for penalized smoothing.
#[derive(Debug, Clone, PartialEq)]
pub enum BasisType {
    /// B-spline basis with given order (typically 4 for cubic).
    Bspline { order: usize },
    /// Fourier basis with given period.
    Fourier { period: f64 },
}

/// Functional data parameter object (basis + penalty specification).
#[derive(Debug, Clone, PartialEq)]
pub struct FdPar {
    /// Type of basis system.
    pub basis_type: BasisType,
    /// Number of basis functions.
    pub nbasis: usize,
    /// Smoothing parameter.
    pub lambda: f64,
    /// Derivative order for the penalty (default: 2).
    pub lfd_order: usize,
    /// Precomputed K×K penalty matrix (column-major).
    pub penalty_matrix: Vec<f64>,
}

/// Result of basis-penalized smoothing.
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothBasisResult {
    /// Basis coefficients (n × K).
    pub coefficients: FdMatrix,
    /// Fitted values (n × m).
    pub fitted: FdMatrix,
    /// Effective degrees of freedom.
    pub edf: f64,
    /// Generalized cross-validation score.
    pub gcv: f64,
    /// AIC.
    pub aic: f64,
    /// BIC.
    pub bic: f64,
    /// Roughness penalty matrix (K × K, column-major).
    pub penalty_matrix: Vec<f64>,
    /// Number of basis functions used.
    pub nbasis: usize,
}

// ─── Penalty Matrices ───────────────────────────────────────────────────────

/// Compute the roughness penalty matrix for B-splines via numerical quadrature.
///
/// R\[j,k\] = ∫ D^m B_j(t) · D^m B_k(t) dt
///
/// Uses Simpson's rule on a fine sub-grid for each knot interval.
///
/// # Arguments
/// * `argvals` — Evaluation points (length m)
/// * `nbasis` — Number of basis functions
/// * `order` — B-spline order (typically 4 for cubic)
/// * `lfd_order` — Derivative order for penalty (typically 2)
///
/// # Returns
/// K × K penalty matrix in column-major layout (K = nbasis)
pub fn bspline_penalty_matrix(
    argvals: &[f64],
    nbasis: usize,
    order: usize,
    lfd_order: usize,
) -> Vec<f64> {
    if nbasis < 2 || order < 1 || lfd_order >= order || argvals.len() < 2 {
        return vec![0.0; nbasis * nbasis];
    }

    let nknots = nbasis.saturating_sub(order).max(2);

    // Create a fine quadrature grid (10 sub-points per original interval)
    let n_sub = 10;
    let t_min = argvals[0];
    let t_max = argvals[argvals.len() - 1];
    let n_quad = (argvals.len() - 1) * n_sub + 1;
    let quad_t: Vec<f64> = (0..n_quad)
        .map(|i| t_min + (t_max - t_min) * i as f64 / (n_quad - 1) as f64)
        .collect();

    // Evaluate B-spline basis on fine grid
    let basis_fine = bspline_basis(&quad_t, nknots, order);
    let actual_nbasis = basis_fine.len() / n_quad;

    // Compute derivatives of B-spline basis numerically
    let h = (t_max - t_min) / (n_quad - 1) as f64;
    let deriv_basis = differentiate_basis_columns(&basis_fine, n_quad, actual_nbasis, h, lfd_order);

    // Integration weights on fine grid
    let weights = simpsons_weights(&quad_t);

    // Compute penalty matrix: R[j,k] = ∫ D^m B_j · D^m B_k dt
    integrate_symmetric_penalty(&deriv_basis, &weights, actual_nbasis, n_quad)
}

/// Compute the roughness penalty matrix for a Fourier basis.
///
/// For Fourier basis, the penalty is diagonal with eigenvalues `(2πk/T)^(2m)`.
///
/// # Arguments
/// * `nbasis` — Number of basis functions
/// * `period` — Period of the Fourier basis
/// * `lfd_order` — Derivative order for penalty
///
/// # Returns
/// K × K penalty matrix in column-major layout
pub fn fourier_penalty_matrix(nbasis: usize, period: f64, lfd_order: usize) -> Vec<f64> {
    let k = nbasis;
    let mut penalty = vec![0.0; k * k];

    // First basis function is constant → lfd_order-th derivative is 0
    // penalty[0] = 0 (already zero)

    // For sin/cos pairs: eigenvalue is (2πk/T)^(2m)
    // Matches R's fda package convention (sqrt(2)-normalized basis)
    let mut freq = 1;
    let mut idx = 1;
    while idx < k {
        let omega = 2.0 * PI * f64::from(freq) / period;
        let eigenval = omega.powi(2 * lfd_order as i32);

        // sin component
        if idx < k {
            penalty[idx + idx * k] = eigenval;
            idx += 1;
        }
        // cos component
        if idx < k {
            penalty[idx + idx * k] = eigenval;
            idx += 1;
        }
        freq += 1;
    }

    penalty
}

// ─── Smoothing Functions ────────────────────────────────────────────────────

/// Perform basis-penalized smoothing.
///
/// Solves `(Φ'Φ + λR)c = Φ'y` per curve via Cholesky decomposition.
/// This implements `smooth.basis` from R's fda package.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `fdpar` — Functional parameter object specifying basis and penalty
///
/// # Returns
/// [`SmoothBasisResult`] with coefficients, fitted values, and diagnostics.
pub fn smooth_basis(
    data: &FdMatrix,
    argvals: &[f64],
    fdpar: &FdPar,
) -> Result<SmoothBasisResult, crate::FdarError> {
    let (n, m) = data.shape();
    if n == 0 || m == 0 || argvals.len() != m || fdpar.nbasis < 2 {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "data/argvals/fdpar",
            expected: "n > 0, m > 0, argvals.len() == m, nbasis >= 2".to_string(),
            actual: format!(
                "n={}, m={}, argvals.len()={}, nbasis={}",
                n,
                m,
                argvals.len(),
                fdpar.nbasis
            ),
        });
    }

    // Evaluate basis on argvals
    let (basis_flat, actual_nbasis) = evaluate_basis(argvals, &fdpar.basis_type, fdpar.nbasis);
    let k = actual_nbasis;

    let b_mat = DMatrix::from_column_slice(m, k, &basis_flat);
    let r_mat = DMatrix::from_column_slice(k, k, &fdpar.penalty_matrix);

    // (Φ'Φ + λR + εI) — small ridge ensures positive definiteness
    let btb = b_mat.transpose() * &b_mat;
    let ridge_eps = 1e-10;
    let system: DMatrix<f64> =
        &btb + fdpar.lambda * &r_mat + ridge_eps * DMatrix::<f64>::identity(k, k);

    // Invert the penalized system
    let system_inv =
        invert_penalized_system(&system, k).ok_or_else(|| crate::FdarError::ComputationFailed {
            operation: "matrix inversion",
            detail: "failed to invert penalized system (Φ'Φ + λR)".to_string(),
        })?;

    // Hat matrix: H = Φ (Φ'Φ + λR)^{-1} Φ'  →  EDF = tr(H)
    let h_mat = &b_mat * &system_inv * b_mat.transpose();
    let edf: f64 = (0..m).map(|i| h_mat[(i, i)]).sum();

    // Project all curves
    let proj = &system_inv * b_mat.transpose();
    let (all_coefs, all_fitted, total_rss) = project_all_curves(data, &b_mat, &proj, n, m, k);

    let total_points = (n * m) as f64;
    let gcv = compute_gcv(total_rss, total_points, edf, m);
    let mse = total_rss / total_points;
    // Total effective degrees of freedom = n curves * per-curve edf
    let total_edf = n as f64 * edf;
    let aic = total_points * mse.max(1e-300).ln() + 2.0 * total_edf;
    let bic = total_points * mse.max(1e-300).ln() + total_points.ln() * total_edf;

    Ok(SmoothBasisResult {
        coefficients: all_coefs,
        fitted: all_fitted,
        edf,
        gcv,
        aic,
        bic,
        penalty_matrix: fdpar.penalty_matrix.clone(),
        nbasis: k,
    })
}

/// Perform basis-penalized smoothing with GCV-optimal lambda.
///
/// Searches over a log-lambda grid and selects the lambda minimizing GCV.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `basis_type` — Type of basis system
/// * `nbasis` — Number of basis functions
/// * `lfd_order` — Derivative order for penalty
/// * `log_lambda_range` — Range of log10(lambda) to search, e.g. (-8.0, 4.0)
/// * `n_grid` — Number of grid points for the search
pub fn smooth_basis_gcv(
    data: &FdMatrix,
    argvals: &[f64],
    basis_type: &BasisType,
    nbasis: usize,
    lfd_order: usize,
    log_lambda_range: (f64, f64),
    n_grid: usize,
) -> Option<SmoothBasisResult> {
    let m = argvals.len();
    if m == 0 || nbasis < 2 || n_grid < 2 {
        return None;
    }

    // Compute penalty matrix once
    let penalty = match basis_type {
        BasisType::Bspline { order } => bspline_penalty_matrix(argvals, nbasis, *order, lfd_order),
        BasisType::Fourier { period } => fourier_penalty_matrix(nbasis, *period, lfd_order),
    };

    let (lo, hi) = log_lambda_range;
    let mut best_gcv = f64::INFINITY;
    let mut best_result: Option<SmoothBasisResult> = None;

    for i in 0..n_grid {
        let log_lam = lo + (hi - lo) * i as f64 / (n_grid - 1) as f64;
        let lam = 10.0_f64.powf(log_lam);

        let fdpar = FdPar {
            basis_type: basis_type.clone(),
            nbasis,
            lambda: lam,
            lfd_order,
            penalty_matrix: penalty.clone(),
        };

        if let Ok(result) = smooth_basis(data, argvals, &fdpar) {
            if result.gcv < best_gcv {
                best_gcv = result.gcv;
                best_result = Some(result);
            }
        }
    }

    best_result
}

// ─── Internal Helpers ───────────────────────────────────────────────────────

/// Differentiate column-major basis matrix `lfd_order` times using gradient_uniform.
fn differentiate_basis_columns(
    basis: &[f64],
    n_quad: usize,
    nbasis: usize,
    h: f64,
    lfd_order: usize,
) -> Vec<f64> {
    let mut deriv = basis.to_vec();
    for _ in 0..lfd_order {
        let mut new_deriv = vec![0.0; n_quad * nbasis];
        for j in 0..nbasis {
            let col: Vec<f64> = (0..n_quad).map(|i| deriv[i + j * n_quad]).collect();
            let grad = crate::helpers::gradient_uniform(&col, h);
            for i in 0..n_quad {
                new_deriv[i + j * n_quad] = grad[i];
            }
        }
        deriv = new_deriv;
    }
    deriv
}

/// Integrate symmetric penalty: R[j,k] = ∫ D^m B_j · D^m B_k dt.
fn integrate_symmetric_penalty(
    deriv_basis: &[f64],
    weights: &[f64],
    k: usize,
    n_quad: usize,
) -> Vec<f64> {
    let mut penalty = vec![0.0; k * k];
    for j in 0..k {
        for l in j..k {
            let mut val = 0.0;
            for i in 0..n_quad {
                val += deriv_basis[i + j * n_quad] * deriv_basis[i + l * n_quad] * weights[i];
            }
            penalty[j + l * k] = val;
            penalty[l + j * k] = val;
        }
    }
    penalty
}

/// Evaluate basis functions on argvals, returning (flat column-major, actual_nbasis).
fn evaluate_basis(argvals: &[f64], basis_type: &BasisType, nbasis: usize) -> (Vec<f64>, usize) {
    let m = argvals.len();
    match basis_type {
        BasisType::Bspline { order } => {
            let nknots = nbasis.saturating_sub(*order).max(2);
            let basis = bspline_basis(argvals, nknots, *order);
            let actual = basis.len() / m;
            (basis, actual)
        }
        BasisType::Fourier { period } => {
            let basis = fourier_basis_with_period(argvals, nbasis, *period);
            (basis, nbasis)
        }
    }
}

/// Invert the penalized system matrix via Cholesky or SVD pseudoinverse.
fn invert_penalized_system(system: &DMatrix<f64>, k: usize) -> Option<DMatrix<f64>> {
    if let Some(chol) = system.clone().cholesky() {
        return Some(chol.inverse());
    }
    // SVD fallback
    let svd = nalgebra::SVD::new(system.clone(), true, true);
    let u = svd.u.as_ref()?;
    let v_t = svd.v_t.as_ref()?;
    let max_sv: f64 = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let eps = 1e-10 * max_sv;
    let mut inv = DMatrix::<f64>::zeros(k, k);
    for ii in 0..k {
        for jj in 0..k {
            let mut sum = 0.0;
            for s in 0..k.min(svd.singular_values.len()) {
                if svd.singular_values[s] > eps {
                    sum += v_t[(s, ii)] / svd.singular_values[s] * u[(jj, s)];
                }
            }
            inv[(ii, jj)] = sum;
        }
    }
    Some(inv)
}

/// Project all curves onto basis, returning (coefficients, fitted, total_rss).
fn project_all_curves(
    data: &FdMatrix,
    b_mat: &DMatrix<f64>,
    proj: &DMatrix<f64>,
    n: usize,
    m: usize,
    k: usize,
) -> (FdMatrix, FdMatrix, f64) {
    let mut all_coefs = FdMatrix::zeros(n, k);
    let mut all_fitted = FdMatrix::zeros(n, m);
    let mut total_rss = 0.0;

    for i in 0..n {
        let curve: Vec<f64> = (0..m).map(|j| data[(i, j)]).collect();
        let y_vec = nalgebra::DVector::from_vec(curve.clone());
        let coefs = proj * &y_vec;

        for j in 0..k {
            all_coefs[(i, j)] = coefs[j];
        }
        let fitted = b_mat * &coefs;
        for j in 0..m {
            all_fitted[(i, j)] = fitted[j];
            let resid = curve[j] - fitted[j];
            total_rss += resid * resid;
        }
    }

    (all_coefs, all_fitted, total_rss)
}

/// Compute GCV score.
fn compute_gcv(rss: f64, n_points: f64, edf: f64, m: usize) -> f64 {
    let gcv_denom = 1.0 - edf / m as f64;
    if gcv_denom.abs() > 1e-10 {
        (rss / n_points) / (gcv_denom * gcv_denom)
    } else {
        f64::INFINITY
    }
}

// ─── Nbasis Selection via CV ────────────────────────────────────────────────

/// Criterion for nbasis selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BasisCriterion {
    /// Generalized cross-validation.
    Gcv,
    /// Leave-one-out cross-validation (k-fold).
    Cv,
    /// Akaike Information Criterion.
    Aic,
    /// Bayesian Information Criterion.
    Bic,
}

/// Result of nbasis selection.
#[derive(Debug, Clone, PartialEq)]
pub struct BasisNbasisCvResult {
    /// Optimal number of basis functions.
    pub optimal_nbasis: usize,
    /// Score for each nbasis tested.
    pub scores: Vec<f64>,
    /// Range of nbasis values tested.
    pub nbasis_range: Vec<usize>,
    /// Criterion used.
    pub criterion: BasisCriterion,
}

/// Evaluate information criterion (GCV/AIC/BIC) for a range of nbasis values.
fn evaluate_nbasis_info_criterion(
    data: &FdMatrix,
    argvals: &[f64],
    nbasis_range: &[usize],
    basis_type: &BasisType,
    criterion: BasisCriterion,
    lambda: f64,
) -> Vec<f64> {
    let mut scores = Vec::with_capacity(nbasis_range.len());
    for &nb in nbasis_range {
        if nb < 2 {
            scores.push(f64::INFINITY);
            continue;
        }
        let penalty = match basis_type {
            BasisType::Bspline { order } => bspline_penalty_matrix(argvals, nb, *order, 2),
            BasisType::Fourier { period } => fourier_penalty_matrix(nb, *period, 2),
        };
        let fdpar = FdPar {
            basis_type: basis_type.clone(),
            nbasis: nb,
            lambda,
            lfd_order: 2,
            penalty_matrix: penalty,
        };
        match smooth_basis(data, argvals, &fdpar) {
            Ok(result) => {
                let score = match criterion {
                    BasisCriterion::Gcv => result.gcv,
                    BasisCriterion::Aic => result.aic,
                    BasisCriterion::Bic => result.bic,
                    _ => unreachable!(),
                };
                scores.push(score);
            }
            Err(_) => scores.push(f64::INFINITY),
        }
    }
    scores
}

/// Evaluate nbasis via k-fold cross-validation of reconstruction error.
fn evaluate_nbasis_cv(
    data: &FdMatrix,
    argvals: &[f64],
    nbasis_range: &[usize],
    basis_type: &BasisType,
    lambda: f64,
    n_folds: usize,
) -> Vec<f64> {
    let (n, m) = data.shape();
    let n_folds = n_folds.max(2);
    let folds = crate::cv::create_folds(n, n_folds, 42);
    let mut scores = Vec::with_capacity(nbasis_range.len());

    for &nb in nbasis_range {
        if nb < 2 {
            scores.push(f64::INFINITY);
            continue;
        }
        let penalty = match basis_type {
            BasisType::Bspline { order } => bspline_penalty_matrix(argvals, nb, *order, 2),
            BasisType::Fourier { period } => fourier_penalty_matrix(nb, *period, 2),
        };

        let mut total_mse = 0.0;
        let mut total_count = 0;

        for fold in 0..n_folds {
            let (train_idx, test_idx) = crate::cv::fold_indices(&folds, fold);
            if train_idx.is_empty() || test_idx.is_empty() {
                continue;
            }
            let train_data = crate::cv::subset_rows(data, &train_idx);
            let fdpar = FdPar {
                basis_type: basis_type.clone(),
                nbasis: nb,
                lambda,
                lfd_order: 2,
                penalty_matrix: penalty.clone(),
            };

            if let Ok(train_result) = smooth_basis(&train_data, argvals, &fdpar) {
                let (basis_flat, actual_k) = evaluate_basis(argvals, basis_type, nb);
                let b_mat = DMatrix::from_column_slice(m, actual_k, &basis_flat);
                let r_mat =
                    DMatrix::from_column_slice(actual_k, actual_k, &train_result.penalty_matrix);
                let btb = b_mat.transpose() * &b_mat;
                let ridge_eps = 1e-10;
                let system: DMatrix<f64> = &btb
                    + lambda * &r_mat
                    + ridge_eps * DMatrix::<f64>::identity(actual_k, actual_k);

                if let Some(system_inv) = invert_penalized_system(&system, actual_k) {
                    let proj = &system_inv * b_mat.transpose();
                    for &ti in &test_idx {
                        let curve: Vec<f64> = (0..m).map(|j| data[(ti, j)]).collect();
                        let y_vec = nalgebra::DVector::from_vec(curve.clone());
                        let coefs = &proj * &y_vec;
                        let fitted = &b_mat * &coefs;
                        let mse: f64 =
                            (0..m).map(|j| (curve[j] - fitted[j]).powi(2)).sum::<f64>() / m as f64;
                        total_mse += mse;
                        total_count += 1;
                    }
                }
            }
        }

        if total_count > 0 {
            scores.push(total_mse / f64::from(total_count));
        } else {
            scores.push(f64::INFINITY);
        }
    }
    scores
}

/// Select the optimal number of basis functions using multiple criteria
/// (R's `fdata2basis_cv`).
pub fn basis_nbasis_cv(
    data: &FdMatrix,
    argvals: &[f64],
    nbasis_range: &[usize],
    basis_type: &BasisType,
    criterion: BasisCriterion,
    n_folds: usize,
    lambda: f64,
) -> Option<BasisNbasisCvResult> {
    let (n, m) = data.shape();
    if n == 0 || m == 0 || argvals.len() != m || nbasis_range.is_empty() {
        return None;
    }

    let scores = match criterion {
        BasisCriterion::Gcv | BasisCriterion::Aic | BasisCriterion::Bic => {
            evaluate_nbasis_info_criterion(
                data,
                argvals,
                nbasis_range,
                basis_type,
                criterion,
                lambda,
            )
        }
        BasisCriterion::Cv => {
            evaluate_nbasis_cv(data, argvals, nbasis_range, basis_type, lambda, n_folds)
        }
    };

    let (best_idx, _) = scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;

    Some(BasisNbasisCvResult {
        optimal_nbasis: nbasis_range[best_idx],
        scores,
        nbasis_range: nbasis_range.to_vec(),
        criterion,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn uniform_grid(m: usize) -> Vec<f64> {
        (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
    }

    #[test]
    fn test_bspline_penalty_matrix_symmetric() {
        let t = uniform_grid(101);
        let penalty = bspline_penalty_matrix(&t, 15, 4, 2);
        let _k = 15; // may differ from actual due to knot construction
        let actual_k = (penalty.len() as f64).sqrt() as usize;
        for i in 0..actual_k {
            for j in 0..actual_k {
                assert!(
                    (penalty[i + j * actual_k] - penalty[j + i * actual_k]).abs() < 1e-10,
                    "Penalty matrix not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_bspline_penalty_matrix_positive_semidefinite() {
        let t = uniform_grid(101);
        let penalty = bspline_penalty_matrix(&t, 10, 4, 2);
        let k = (penalty.len() as f64).sqrt() as usize;
        // Diagonal elements should be non-negative
        for i in 0..k {
            assert!(
                penalty[i + i * k] >= -1e-10,
                "Diagonal element {} is negative: {}",
                i,
                penalty[i + i * k]
            );
        }
    }

    #[test]
    fn test_fourier_penalty_diagonal() {
        let penalty = fourier_penalty_matrix(7, 1.0, 2);
        // Should be diagonal
        for i in 0..7 {
            for j in 0..7 {
                if i != j {
                    assert!(
                        penalty[i + j * 7].abs() < 1e-10,
                        "Off-diagonal ({},{}) = {}",
                        i,
                        j,
                        penalty[i + j * 7]
                    );
                }
            }
        }
        // Constant term should have zero penalty
        assert!(penalty[0].abs() < 1e-10);
        // Higher frequency terms should have larger penalties
        assert!(penalty[1 + 7] > 0.0);
        assert!(penalty[3 + 3 * 7] > penalty[1 + 7]);
    }

    #[test]
    fn test_smooth_basis_bspline() {
        let m = 101;
        let n = 5;
        let t = uniform_grid(m);

        // Generate noisy sine curves
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin() + 0.1 * (i as f64 * 0.3 + j as f64 * 0.01);
            }
        }

        let nbasis = 15;
        let penalty = bspline_penalty_matrix(&t, nbasis, 4, 2);
        let _actual_k = (penalty.len() as f64).sqrt() as usize;

        let fdpar = FdPar {
            basis_type: BasisType::Bspline { order: 4 },
            nbasis,
            lambda: 1e-4,
            lfd_order: 2,
            penalty_matrix: penalty,
        };

        let result = smooth_basis(&data, &t, &fdpar);
        assert!(result.is_ok(), "smooth_basis should succeed");

        let res = result.unwrap();
        assert_eq!(res.fitted.shape(), (n, m));
        assert_eq!(res.coefficients.nrows(), n);
        assert!(res.edf > 0.0, "EDF should be positive");
        assert!(res.gcv > 0.0, "GCV should be positive");
    }

    #[test]
    fn test_smooth_basis_fourier() {
        let m = 101;
        let n = 3;
        let t = uniform_grid(m);

        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin() + (4.0 * PI * t[j]).cos();
            }
        }

        let nbasis = 7;
        let period = 1.0;
        let penalty = fourier_penalty_matrix(nbasis, period, 2);

        let fdpar = FdPar {
            basis_type: BasisType::Fourier { period },
            nbasis,
            lambda: 1e-6,
            lfd_order: 2,
            penalty_matrix: penalty,
        };

        let result = smooth_basis(&data, &t, &fdpar);
        assert!(result.is_ok());

        let res = result.unwrap();
        // Fourier basis should fit periodic data well
        for j in 0..m {
            let expected = (2.0 * PI * t[j]).sin() + (4.0 * PI * t[j]).cos();
            assert!(
                (res.fitted[(0, j)] - expected).abs() < 0.1,
                "Fourier fit poor at j={}: got {}, expected {}",
                j,
                res.fitted[(0, j)],
                expected
            );
        }
    }

    #[test]
    fn test_smooth_basis_gcv_selects_reasonable_lambda() {
        let m = 101;
        let n = 5;
        let t = uniform_grid(m);

        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] =
                    (2.0 * PI * t[j]).sin() + 0.1 * ((i * 37 + j * 13) % 20) as f64 / 20.0;
            }
        }

        let basis_type = BasisType::Bspline { order: 4 };
        let result = smooth_basis_gcv(&data, &t, &basis_type, 15, 2, (-8.0, 4.0), 25);
        assert!(result.is_some(), "GCV search should succeed");
    }

    #[test]
    fn test_smooth_basis_large_lambda_reduces_edf() {
        let m = 101;
        let n = 3;
        let t = uniform_grid(m);

        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin();
            }
        }

        let nbasis = 15;
        let penalty = bspline_penalty_matrix(&t, nbasis, 4, 2);
        let _actual_k = (penalty.len() as f64).sqrt() as usize;

        let fdpar_small = FdPar {
            basis_type: BasisType::Bspline { order: 4 },
            nbasis,
            lambda: 1e-8,
            lfd_order: 2,
            penalty_matrix: penalty.clone(),
        };
        let fdpar_large = FdPar {
            basis_type: BasisType::Bspline { order: 4 },
            nbasis,
            lambda: 1e2,
            lfd_order: 2,
            penalty_matrix: penalty,
        };

        let res_small = smooth_basis(&data, &t, &fdpar_small).unwrap();
        let res_large = smooth_basis(&data, &t, &fdpar_large).unwrap();

        assert!(
            res_large.edf < res_small.edf,
            "Larger lambda should reduce EDF: {} vs {}",
            res_large.edf,
            res_small.edf
        );
    }

    // ============== basis_nbasis_cv tests ==============

    #[test]
    fn test_basis_nbasis_cv_gcv() {
        let m = 101;
        let n = 5;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] =
                    (2.0 * PI * t[j]).sin() + 0.1 * ((i * 37 + j * 13) % 20) as f64 / 20.0;
            }
        }

        let nbasis_range: Vec<usize> = (4..=20).step_by(2).collect();
        let result = basis_nbasis_cv(
            &data,
            &t,
            &nbasis_range,
            &BasisType::Bspline { order: 4 },
            BasisCriterion::Gcv,
            5,
            1e-4,
        );
        assert!(result.is_some());
        let res = result.unwrap();
        assert!(nbasis_range.contains(&res.optimal_nbasis));
        assert_eq!(res.scores.len(), nbasis_range.len());
        assert_eq!(res.criterion, BasisCriterion::Gcv);
    }

    #[test]
    fn test_basis_nbasis_cv_aic_bic() {
        let m = 51;
        let n = 5;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin();
            }
        }

        let nbasis_range: Vec<usize> = vec![5, 7, 9, 11];
        let aic_result = basis_nbasis_cv(
            &data,
            &t,
            &nbasis_range,
            &BasisType::Bspline { order: 4 },
            BasisCriterion::Aic,
            5,
            0.0,
        );
        let bic_result = basis_nbasis_cv(
            &data,
            &t,
            &nbasis_range,
            &BasisType::Bspline { order: 4 },
            BasisCriterion::Bic,
            5,
            0.0,
        );
        assert!(aic_result.is_some());
        assert!(bic_result.is_some());
    }

    #[test]
    fn test_basis_nbasis_cv_kfold() {
        let m = 51;
        let n = 10;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin() + 0.05 * ((i * 7 + j * 3) % 10) as f64;
            }
        }

        let nbasis_range: Vec<usize> = vec![5, 7, 9];
        let result = basis_nbasis_cv(
            &data,
            &t,
            &nbasis_range,
            &BasisType::Bspline { order: 4 },
            BasisCriterion::Cv,
            5,
            1e-4,
        );
        assert!(result.is_some());
        let res = result.unwrap();
        assert!(nbasis_range.contains(&res.optimal_nbasis));
        assert_eq!(res.criterion, BasisCriterion::Cv);
    }
}
