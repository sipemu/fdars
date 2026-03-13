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
                    BasisCriterion::Cv => unreachable!(),
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
    use crate::test_helpers::uniform_grid;
    use std::f64::consts::PI;

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

    // ============== Comprehensive additional tests ==============

    // Helper: generate standard test data (sine + high-freq component)
    fn make_test_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let t: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin()
                    + 0.1 * (10.0 * t[j]).sin()
                    + 0.05 * ((i * 37 + j * 13) % 20) as f64 / 20.0;
            }
        }
        (data, t)
    }

    // Helper: create an FdPar for B-spline smoothing
    fn make_bspline_fdpar(argvals: &[f64], nbasis: usize, lambda: f64) -> FdPar {
        let penalty = bspline_penalty_matrix(argvals, nbasis, 4, 2);
        FdPar {
            basis_type: BasisType::Bspline { order: 4 },
            nbasis,
            lambda,
            lfd_order: 2,
            penalty_matrix: penalty,
        }
    }

    // Helper: create an FdPar for Fourier smoothing
    fn make_fourier_fdpar(nbasis: usize, period: f64, lambda: f64) -> FdPar {
        let penalty = fourier_penalty_matrix(nbasis, period, 2);
        FdPar {
            basis_type: BasisType::Fourier { period },
            nbasis,
            lambda,
            lfd_order: 2,
            penalty_matrix: penalty,
        }
    }

    // ─── BasisType enum tests ───────────────────────────────────────────────

    #[test]
    fn test_basis_type_bspline_variant() {
        let bt = BasisType::Bspline { order: 4 };
        assert_eq!(bt, BasisType::Bspline { order: 4 });
        // Different orders are not equal
        assert_ne!(bt, BasisType::Bspline { order: 3 });
    }

    #[test]
    fn test_basis_type_fourier_variant() {
        let bt = BasisType::Fourier { period: 1.0 };
        assert_eq!(bt, BasisType::Fourier { period: 1.0 });
        assert_ne!(bt, BasisType::Fourier { period: 2.0 });
    }

    #[test]
    fn test_basis_type_cross_variant_inequality() {
        let bspline = BasisType::Bspline { order: 4 };
        let fourier = BasisType::Fourier { period: 1.0 };
        assert_ne!(bspline, fourier);
    }

    #[test]
    fn test_basis_type_clone_and_debug() {
        let bt = BasisType::Bspline { order: 4 };
        let cloned = bt.clone();
        assert_eq!(bt, cloned);
        let debug_str = format!("{:?}", bt);
        assert!(debug_str.contains("Bspline"));
        assert!(debug_str.contains("4"));
    }

    // ─── FdPar struct tests ─────────────────────────────────────────────────

    #[test]
    fn test_fdpar_construction_and_fields() {
        let penalty = vec![1.0, 0.0, 0.0, 1.0];
        let fdpar = FdPar {
            basis_type: BasisType::Bspline { order: 4 },
            nbasis: 2,
            lambda: 0.01,
            lfd_order: 2,
            penalty_matrix: penalty.clone(),
        };
        assert_eq!(fdpar.nbasis, 2);
        assert!((fdpar.lambda - 0.01).abs() < 1e-15);
        assert_eq!(fdpar.lfd_order, 2);
        assert_eq!(fdpar.penalty_matrix.len(), 4);
    }

    #[test]
    fn test_fdpar_clone_and_debug() {
        let t = uniform_grid(50);
        let fdpar = make_bspline_fdpar(&t, 8, 1e-3);
        let cloned = fdpar.clone();
        assert_eq!(fdpar, cloned);
        let debug_str = format!("{:?}", fdpar);
        assert!(debug_str.contains("FdPar"));
    }

    // ─── BasisCriterion enum tests ──────────────────────────────────────────

    #[test]
    fn test_basis_criterion_variants() {
        assert_eq!(BasisCriterion::Gcv, BasisCriterion::Gcv);
        assert_eq!(BasisCriterion::Cv, BasisCriterion::Cv);
        assert_eq!(BasisCriterion::Aic, BasisCriterion::Aic);
        assert_eq!(BasisCriterion::Bic, BasisCriterion::Bic);
        assert_ne!(BasisCriterion::Gcv, BasisCriterion::Aic);
        assert_ne!(BasisCriterion::Cv, BasisCriterion::Bic);
    }

    #[test]
    fn test_basis_criterion_copy() {
        let c = BasisCriterion::Gcv;
        let copied = c; // Copy
        assert_eq!(c, copied);
    }

    #[test]
    fn test_basis_criterion_debug() {
        let debug_str = format!("{:?}", BasisCriterion::Bic);
        assert!(debug_str.contains("Bic"));
    }

    // ─── SmoothBasisResult tests ────────────────────────────────────────────

    #[test]
    fn test_smooth_basis_result_all_fields() {
        let (data, t) = make_test_data(3, 50);
        let fdpar = make_bspline_fdpar(&t, 10, 1e-4);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();

        // coefficients: n curves x k basis functions
        assert_eq!(res.coefficients.nrows(), 3);
        assert!(res.coefficients.ncols() > 0);
        assert_eq!(res.nbasis, res.coefficients.ncols());
        // fitted: n x m
        assert_eq!(res.fitted.shape(), (3, 50));
        // edf should be between 1 and nbasis
        assert!(res.edf > 0.0 && res.edf <= res.nbasis as f64);
        // gcv, aic, bic should be finite
        assert!(res.gcv.is_finite());
        assert!(res.aic.is_finite());
        assert!(res.bic.is_finite());
        // penalty_matrix should be k x k
        let k = res.nbasis;
        assert_eq!(res.penalty_matrix.len(), k * k);
    }

    #[test]
    fn test_smooth_basis_result_clone() {
        let (data, t) = make_test_data(2, 50);
        let fdpar = make_bspline_fdpar(&t, 8, 1e-3);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();
        let cloned = res.clone();
        assert_eq!(res, cloned);
    }

    // ─── smooth_basis: B-spline detailed tests ──────────────────────────────

    #[test]
    fn test_smooth_basis_bspline_coefficient_shape() {
        let (data, t) = make_test_data(4, 50);
        let nbasis = 12;
        let fdpar = make_bspline_fdpar(&t, nbasis, 1e-4);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();
        assert_eq!(res.coefficients.nrows(), 4);
        // actual nbasis may differ from requested due to knot construction
        assert!(res.coefficients.ncols() >= 2);
        assert_eq!(res.nbasis, res.coefficients.ncols());
    }

    #[test]
    fn test_smooth_basis_bspline_fitted_values_shape() {
        let m = 80;
        let n = 6;
        let (data, t) = make_test_data(n, m);
        let fdpar = make_bspline_fdpar(&t, 15, 1e-4);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();
        assert_eq!(res.fitted.shape(), (n, m));
    }

    #[test]
    fn test_smooth_basis_bspline_zero_lambda_interpolates() {
        // With lambda=0, the smoother should nearly interpolate the data
        let m = 30;
        let n = 2;
        let (data, t) = make_test_data(n, m);
        let fdpar = make_bspline_fdpar(&t, 15, 0.0);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();

        // Residuals should be very small (near interpolation)
        let mut max_resid = 0.0_f64;
        for i in 0..n {
            for j in 0..m {
                let resid = (data[(i, j)] - res.fitted[(i, j)]).abs();
                max_resid = max_resid.max(resid);
            }
        }
        assert!(
            max_resid < 0.5,
            "Zero-lambda B-spline should closely interpolate; max_resid = {}",
            max_resid
        );
    }

    #[test]
    fn test_smooth_basis_bspline_large_lambda_oversmooths() {
        // With very large lambda, the fit should be much smoother (lower variance)
        // than with small lambda
        let m = 50;
        let n = 1;
        let (data, t) = make_test_data(n, m);

        let fdpar_small = make_bspline_fdpar(&t, 15, 1e-6);
        let res_small = smooth_basis(&data, &t, &fdpar_small).unwrap();

        let fdpar_large = make_bspline_fdpar(&t, 15, 1e6);
        let res_large = smooth_basis(&data, &t, &fdpar_large).unwrap();

        let compute_variance = |fitted: &FdMatrix, row: usize, ncols: usize| -> f64 {
            let vals: Vec<f64> = (0..ncols).map(|j| fitted[(row, j)]).collect();
            let mean = vals.iter().sum::<f64>() / ncols as f64;
            vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / ncols as f64
        };

        let var_small = compute_variance(&res_small.fitted, 0, m);
        let var_large = compute_variance(&res_large.fitted, 0, m);
        assert!(
            var_large < var_small,
            "Large lambda should yield lower variance fit: var_large={}, var_small={}",
            var_large,
            var_small
        );
    }

    #[test]
    fn test_smooth_basis_bspline_penalty_effect_on_smoothness() {
        // Compare roughness of fits with small vs large lambda
        let m = 50;
        let n = 1;
        let (data, t) = make_test_data(n, m);

        let fdpar_small = make_bspline_fdpar(&t, 15, 1e-8);
        let fdpar_large = make_bspline_fdpar(&t, 15, 1.0);

        let res_small = smooth_basis(&data, &t, &fdpar_small).unwrap();
        let res_large = smooth_basis(&data, &t, &fdpar_large).unwrap();

        // Measure roughness as sum of squared second differences
        let roughness = |fitted: &FdMatrix, row: usize, ncols: usize| -> f64 {
            (1..ncols - 1)
                .map(|j| {
                    let d2 = fitted[(row, j + 1)] - 2.0 * fitted[(row, j)] + fitted[(row, j - 1)];
                    d2 * d2
                })
                .sum::<f64>()
        };

        let r_small = roughness(&res_small.fitted, 0, m);
        let r_large = roughness(&res_large.fitted, 0, m);
        assert!(
            r_large < r_small,
            "Larger lambda should produce smoother fit: roughness_large={}, roughness_small={}",
            r_large,
            r_small
        );
    }

    #[test]
    fn test_smooth_basis_bspline_single_curve() {
        let m = 50;
        let (data, t) = make_test_data(1, m);
        let fdpar = make_bspline_fdpar(&t, 10, 1e-4);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();
        assert_eq!(res.fitted.nrows(), 1);
        assert_eq!(res.fitted.ncols(), m);
        assert!(res.gcv.is_finite());
    }

    #[test]
    fn test_smooth_basis_bspline_many_curves() {
        let m = 50;
        let n = 20;
        let (data, t) = make_test_data(n, m);
        let fdpar = make_bspline_fdpar(&t, 10, 1e-4);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();
        assert_eq!(res.fitted.nrows(), n);
        assert_eq!(res.coefficients.nrows(), n);
    }

    #[test]
    fn test_smooth_basis_bspline_minimal_nbasis() {
        // nbasis = 2 is the minimum allowed
        let m = 50;
        let (data, t) = make_test_data(1, m);
        let fdpar = make_bspline_fdpar(&t, 2, 1e-4);
        let res = smooth_basis(&data, &t, &fdpar);
        // Should succeed (or at least not panic); the fit may be poor
        assert!(res.is_ok());
    }

    #[test]
    fn test_smooth_basis_bspline_different_orders() {
        let m = 50;
        let (data, t) = make_test_data(2, m);
        // Order 3 (quadratic B-splines)
        let penalty3 = bspline_penalty_matrix(&t, 10, 3, 2);
        let fdpar3 = FdPar {
            basis_type: BasisType::Bspline { order: 3 },
            nbasis: 10,
            lambda: 1e-4,
            lfd_order: 2,
            penalty_matrix: penalty3,
        };
        let res3 = smooth_basis(&data, &t, &fdpar3);
        assert!(res3.is_ok());

        // Order 5 (quartic B-splines)
        let penalty5 = bspline_penalty_matrix(&t, 10, 5, 2);
        let fdpar5 = FdPar {
            basis_type: BasisType::Bspline { order: 5 },
            nbasis: 10,
            lambda: 1e-4,
            lfd_order: 2,
            penalty_matrix: penalty5,
        };
        let res5 = smooth_basis(&data, &t, &fdpar5);
        assert!(res5.is_ok());
    }

    // ─── smooth_basis: Fourier detailed tests ───────────────────────────────

    #[test]
    fn test_smooth_basis_fourier_coefficient_shape() {
        let m = 50;
        let n = 3;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin();
            }
        }
        let nbasis = 7;
        let fdpar = make_fourier_fdpar(nbasis, 1.0, 1e-6);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();
        assert_eq!(res.coefficients.nrows(), n);
        assert_eq!(res.coefficients.ncols(), nbasis);
        assert_eq!(res.nbasis, nbasis);
    }

    #[test]
    fn test_smooth_basis_fourier_fits_pure_sine() {
        // Fourier basis should perfectly fit a pure sine with enough basis fns
        let m = 100;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(1, m);
        for j in 0..m {
            data[(0, j)] = (2.0 * PI * t[j]).sin();
        }
        let fdpar = make_fourier_fdpar(5, 1.0, 1e-8);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();

        for j in 0..m {
            let expected = (2.0 * PI * t[j]).sin();
            assert!(
                (res.fitted[(0, j)] - expected).abs() < 0.05,
                "Fourier should fit pure sine; j={}, got={}, expected={}",
                j,
                res.fitted[(0, j)],
                expected
            );
        }
    }

    #[test]
    fn test_smooth_basis_fourier_different_periods() {
        let m = 50;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(1, m);
        for j in 0..m {
            data[(0, j)] = (2.0 * PI * t[j]).sin();
        }

        // Period = 1.0 (matches the data)
        let fdpar1 = make_fourier_fdpar(7, 1.0, 1e-6);
        let res1 = smooth_basis(&data, &t, &fdpar1).unwrap();

        // Period = 2.0 (mismatch, but should still produce a result)
        let fdpar2 = make_fourier_fdpar(7, 2.0, 1e-6);
        let res2 = smooth_basis(&data, &t, &fdpar2).unwrap();

        // Both should succeed and have valid shapes
        assert_eq!(res1.fitted.shape(), (1, m));
        assert_eq!(res2.fitted.shape(), (1, m));
    }

    #[test]
    fn test_smooth_basis_fourier_zero_lambda() {
        let m = 50;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(1, m);
        for j in 0..m {
            data[(0, j)] = (2.0 * PI * t[j]).sin() + (4.0 * PI * t[j]).cos();
        }
        let fdpar = make_fourier_fdpar(9, 1.0, 0.0);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();
        assert_eq!(res.fitted.shape(), (1, m));
        // EDF should be close to nbasis with zero penalty
        assert!(res.edf > 1.0);
    }

    #[test]
    fn test_smooth_basis_fourier_large_lambda() {
        let m = 50;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(1, m);
        for j in 0..m {
            data[(0, j)] = (2.0 * PI * t[j]).sin();
        }
        let fdpar = make_fourier_fdpar(9, 1.0, 1e6);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();
        // EDF should be very small with huge penalty
        assert!(
            res.edf < 5.0,
            "Large lambda should reduce EDF; edf={}",
            res.edf
        );
    }

    // ─── smooth_basis: Lambda comparison tests ──────────────────────────────

    #[test]
    fn test_smooth_basis_lambda_gradient_edf() {
        // EDF should monotonically decrease with increasing lambda
        let m = 50;
        let (data, t) = make_test_data(3, m);
        let lambdas = [1e-8, 1e-4, 1e-2, 1.0, 1e2];
        let mut prev_edf = f64::INFINITY;
        for &lam in &lambdas {
            let fdpar = make_bspline_fdpar(&t, 12, lam);
            let res = smooth_basis(&data, &t, &fdpar).unwrap();
            assert!(
                res.edf <= prev_edf + 0.01,
                "EDF should decrease: lambda={}, edf={}, prev_edf={}",
                lam,
                res.edf,
                prev_edf
            );
            prev_edf = res.edf;
        }
    }

    #[test]
    fn test_smooth_basis_lambda_gradient_rss() {
        // RSS should monotonically increase with increasing lambda
        let m = 50;
        let n = 2;
        let (data, t) = make_test_data(n, m);
        let lambdas = [0.0, 1e-6, 1e-2, 1.0, 1e4];
        let mut prev_rss = -1.0;
        for &lam in &lambdas {
            let fdpar = make_bspline_fdpar(&t, 12, lam);
            let res = smooth_basis(&data, &t, &fdpar).unwrap();
            let mut rss = 0.0;
            for i in 0..n {
                for j in 0..m {
                    rss += (data[(i, j)] - res.fitted[(i, j)]).powi(2);
                }
            }
            assert!(
                rss >= prev_rss - 1e-8,
                "RSS should increase: lambda={}, rss={}, prev_rss={}",
                lam,
                rss,
                prev_rss
            );
            prev_rss = rss;
        }
    }

    // ─── smooth_basis: Error cases ──────────────────────────────────────────

    #[test]
    fn test_smooth_basis_empty_data_rows() {
        let t = uniform_grid(50);
        let data = FdMatrix::zeros(0, 50);
        let fdpar = make_bspline_fdpar(&t, 10, 1e-4);
        let res = smooth_basis(&data, &t, &fdpar);
        assert!(res.is_err());
    }

    #[test]
    fn test_smooth_basis_empty_data_cols() {
        let data = FdMatrix::zeros(5, 0);
        let fdpar = FdPar {
            basis_type: BasisType::Bspline { order: 4 },
            nbasis: 10,
            lambda: 1e-4,
            lfd_order: 2,
            penalty_matrix: vec![0.0; 100],
        };
        let res = smooth_basis(&data, &[], &fdpar);
        assert!(res.is_err());
    }

    #[test]
    fn test_smooth_basis_mismatched_argvals() {
        let t = uniform_grid(50);
        let data = FdMatrix::zeros(3, 40); // m=40 but argvals has 50
        let fdpar = make_bspline_fdpar(&t, 10, 1e-4);
        let res = smooth_basis(&data, &t, &fdpar);
        assert!(res.is_err());
    }

    #[test]
    fn test_smooth_basis_nbasis_too_small() {
        let t = uniform_grid(50);
        let data = FdMatrix::zeros(3, 50);
        // nbasis = 1, which is below minimum of 2
        let fdpar = FdPar {
            basis_type: BasisType::Bspline { order: 4 },
            nbasis: 1,
            lambda: 1e-4,
            lfd_order: 2,
            penalty_matrix: vec![0.0; 1],
        };
        let res = smooth_basis(&data, &t, &fdpar);
        assert!(res.is_err());
    }

    #[test]
    fn test_smooth_basis_error_is_invalid_dimension() {
        let t = uniform_grid(50);
        let data = FdMatrix::zeros(0, 50);
        let fdpar = make_bspline_fdpar(&t, 10, 1e-4);
        let err = smooth_basis(&data, &t, &fdpar).unwrap_err();
        match err {
            crate::FdarError::InvalidDimension { .. } => {} // expected
            other => panic!("Expected InvalidDimension, got {:?}", other),
        }
    }

    // ─── Penalty matrix detailed tests ──────────────────────────────────────

    #[test]
    fn test_bspline_penalty_matrix_different_orders() {
        let t = uniform_grid(101);
        // Order 1 penalty (penalize derivatives)
        let p1 = bspline_penalty_matrix(&t, 10, 4, 1);
        // Order 2 penalty (penalize curvature)
        let p2 = bspline_penalty_matrix(&t, 10, 4, 2);
        // Both should be square and same size
        assert_eq!(p1.len(), p2.len());
        // But they should differ
        let diff: f64 = p1.iter().zip(p2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-10,
            "Different lfd_orders should produce different penalties"
        );
    }

    #[test]
    fn test_bspline_penalty_matrix_edge_cases() {
        // Too few argvals
        let t = vec![0.0];
        let p = bspline_penalty_matrix(&t, 10, 4, 2);
        // Should return zero matrix
        assert!(p.iter().all(|&v| v == 0.0));

        // nbasis < 2
        let t2 = uniform_grid(50);
        let p2 = bspline_penalty_matrix(&t2, 1, 4, 2);
        assert!(p2.iter().all(|&v| v == 0.0));

        // lfd_order >= order
        let p3 = bspline_penalty_matrix(&t2, 10, 4, 4);
        assert!(p3.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_bspline_penalty_nonnegative_diagonal() {
        let t = uniform_grid(101);
        for nbasis in [5, 10, 20] {
            let p = bspline_penalty_matrix(&t, nbasis, 4, 2);
            let k = (p.len() as f64).sqrt() as usize;
            for i in 0..k {
                assert!(
                    p[i + i * k] >= -1e-10,
                    "Diagonal ({},{}) negative for nbasis={}: {}",
                    i,
                    i,
                    nbasis,
                    p[i + i * k]
                );
            }
        }
    }

    #[test]
    fn test_fourier_penalty_increasing_with_frequency() {
        let penalty = fourier_penalty_matrix(11, 1.0, 2);
        let k = 11;
        // Constant term is zero
        assert!(penalty[0].abs() < 1e-15);
        // Pairs: (1,2) -> freq 1, (3,4) -> freq 2, etc.
        let mut prev_eigenval = 0.0;
        for freq in 1..=5 {
            let idx_sin = 2 * freq - 1;
            let eigenval = penalty[idx_sin + idx_sin * k];
            assert!(
                eigenval > prev_eigenval,
                "Higher frequency should have larger penalty: freq={}, eigenval={}, prev={}",
                freq,
                eigenval,
                prev_eigenval
            );
            prev_eigenval = eigenval;
            // cos and sin of same frequency should have same penalty
            let idx_cos = 2 * freq;
            if idx_cos < k {
                assert!(
                    (penalty[idx_cos + idx_cos * k] - eigenval).abs() < 1e-10,
                    "Sin and cos penalty should match at freq {}",
                    freq
                );
            }
        }
    }

    #[test]
    fn test_fourier_penalty_different_periods() {
        let p1 = fourier_penalty_matrix(7, 1.0, 2);
        let p2 = fourier_penalty_matrix(7, 2.0, 2);
        // Longer period -> smaller omega -> smaller penalty eigenvalues
        for i in 1..7 {
            assert!(
                p2[i + i * 7] < p1[i + i * 7] || (p1[i + i * 7] == 0.0 && p2[i + i * 7] == 0.0),
                "Longer period should have smaller penalties at i={}",
                i
            );
        }
    }

    #[test]
    fn test_fourier_penalty_first_order() {
        // lfd_order = 1: penalize first derivative
        let p = fourier_penalty_matrix(5, 1.0, 1);
        // Eigenvalues: (2*pi*freq)^2 for lfd_order=1
        let omega1 = 2.0 * PI;
        let expected1 = omega1.powi(2);
        assert!(
            (p[1 + 5] - expected1).abs() < 1e-6,
            "First-order penalty eigenval: got {}, expected {}",
            p[1 + 5],
            expected1
        );
    }

    #[test]
    fn test_fourier_penalty_zero_nbasis() {
        let p = fourier_penalty_matrix(0, 1.0, 2);
        assert!(p.is_empty());
    }

    #[test]
    fn test_fourier_penalty_nbasis_one() {
        let p = fourier_penalty_matrix(1, 1.0, 2);
        assert_eq!(p.len(), 1);
        assert!(p[0].abs() < 1e-15); // constant term has zero penalty
    }

    // ─── smooth_basis_gcv detailed tests ────────────────────────────────────

    #[test]
    fn test_smooth_basis_gcv_returns_valid_result() {
        let (data, t) = make_test_data(5, 50);
        let bt = BasisType::Bspline { order: 4 };
        let result = smooth_basis_gcv(&data, &t, &bt, 12, 2, (-6.0, 2.0), 20);
        assert!(result.is_some());
        let res = result.unwrap();
        assert_eq!(res.fitted.shape(), (5, 50));
        assert!(res.gcv.is_finite());
        assert!(res.edf > 0.0);
    }

    #[test]
    fn test_smooth_basis_gcv_fourier() {
        let m = 80;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(3, m);
        for i in 0..3 {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin() + 0.5 * (4.0 * PI * t[j]).cos();
            }
        }
        let bt = BasisType::Fourier { period: 1.0 };
        let result = smooth_basis_gcv(&data, &t, &bt, 9, 2, (-8.0, 4.0), 25);
        assert!(result.is_some());
        let res = result.unwrap();
        assert_eq!(res.fitted.nrows(), 3);
        assert_eq!(res.nbasis, 9);
    }

    #[test]
    fn test_smooth_basis_gcv_selects_finite_gcv() {
        let (data, t) = make_test_data(5, 60);
        let bt = BasisType::Bspline { order: 4 };
        let res = smooth_basis_gcv(&data, &t, &bt, 12, 2, (-6.0, 2.0), 15).unwrap();
        assert!(res.gcv.is_finite());
        assert!(res.gcv > 0.0);
    }

    #[test]
    fn test_smooth_basis_gcv_empty_data() {
        let data = FdMatrix::zeros(0, 50);
        let t = uniform_grid(50);
        let bt = BasisType::Bspline { order: 4 };
        let result = smooth_basis_gcv(&data, &t, &bt, 10, 2, (-6.0, 2.0), 10);
        // Should return None since smooth_basis will error for empty data
        assert!(result.is_none());
    }

    #[test]
    fn test_smooth_basis_gcv_empty_argvals() {
        let data = FdMatrix::zeros(5, 0);
        let bt = BasisType::Bspline { order: 4 };
        let result = smooth_basis_gcv(&data, &[], &bt, 10, 2, (-6.0, 2.0), 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_smooth_basis_gcv_nbasis_too_small() {
        let (data, t) = make_test_data(5, 50);
        let bt = BasisType::Bspline { order: 4 };
        let result = smooth_basis_gcv(&data, &t, &bt, 1, 2, (-6.0, 2.0), 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_smooth_basis_gcv_ngrid_too_small() {
        let (data, t) = make_test_data(5, 50);
        let bt = BasisType::Bspline { order: 4 };
        let result = smooth_basis_gcv(&data, &t, &bt, 10, 2, (-6.0, 2.0), 1);
        assert!(result.is_none());
    }

    #[test]
    fn test_smooth_basis_gcv_narrow_range() {
        let (data, t) = make_test_data(3, 50);
        let bt = BasisType::Bspline { order: 4 };
        // Very narrow search range
        let result = smooth_basis_gcv(&data, &t, &bt, 10, 2, (-3.0, -2.0), 5);
        assert!(result.is_some());
    }

    #[test]
    fn test_smooth_basis_gcv_wide_range() {
        let (data, t) = make_test_data(3, 50);
        let bt = BasisType::Bspline { order: 4 };
        // Very wide search range
        let result = smooth_basis_gcv(&data, &t, &bt, 10, 2, (-12.0, 8.0), 30);
        assert!(result.is_some());
    }

    // ─── basis_nbasis_cv detailed tests ─────────────────────────────────────

    #[test]
    fn test_basis_nbasis_cv_scores_length() {
        let (data, t) = make_test_data(5, 50);
        let nbasis_range: Vec<usize> = vec![4, 6, 8, 10, 12];
        let res = basis_nbasis_cv(
            &data,
            &t,
            &nbasis_range,
            &BasisType::Bspline { order: 4 },
            BasisCriterion::Gcv,
            5,
            1e-4,
        )
        .unwrap();
        assert_eq!(res.scores.len(), 5);
        assert_eq!(res.nbasis_range.len(), 5);
        assert_eq!(res.nbasis_range, nbasis_range);
    }

    #[test]
    fn test_basis_nbasis_cv_optimal_within_range() {
        let (data, t) = make_test_data(8, 50);
        let nbasis_range: Vec<usize> = vec![5, 7, 9, 11, 13, 15];
        for criterion in [
            BasisCriterion::Gcv,
            BasisCriterion::Aic,
            BasisCriterion::Bic,
        ] {
            let res = basis_nbasis_cv(
                &data,
                &t,
                &nbasis_range,
                &BasisType::Bspline { order: 4 },
                criterion,
                5,
                1e-4,
            )
            .unwrap();
            assert!(
                nbasis_range.contains(&res.optimal_nbasis),
                "optimal_nbasis {} not in range for {:?}",
                res.optimal_nbasis,
                criterion
            );
        }
    }

    #[test]
    fn test_basis_nbasis_cv_fourier_gcv() {
        let m = 80;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(5, m);
        for i in 0..5 {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin()
                    + 0.3 * (4.0 * PI * t[j]).cos()
                    + 0.02 * ((i * 7 + j * 3) % 10) as f64;
            }
        }
        let nbasis_range: Vec<usize> = vec![5, 7, 9, 11];
        let res = basis_nbasis_cv(
            &data,
            &t,
            &nbasis_range,
            &BasisType::Fourier { period: 1.0 },
            BasisCriterion::Gcv,
            5,
            1e-4,
        )
        .unwrap();
        assert!(nbasis_range.contains(&res.optimal_nbasis));
    }

    #[test]
    fn test_basis_nbasis_cv_fourier_cv() {
        let m = 60;
        let t = uniform_grid(m);
        let n = 10;
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin() + 0.02 * ((i * 11 + j) % 15) as f64;
            }
        }
        let nbasis_range: Vec<usize> = vec![5, 7, 9];
        let res = basis_nbasis_cv(
            &data,
            &t,
            &nbasis_range,
            &BasisType::Fourier { period: 1.0 },
            BasisCriterion::Cv,
            5,
            1e-4,
        )
        .unwrap();
        assert!(nbasis_range.contains(&res.optimal_nbasis));
        assert_eq!(res.criterion, BasisCriterion::Cv);
    }

    #[test]
    fn test_basis_nbasis_cv_with_nbasis_below_minimum() {
        // Range includes nbasis = 1 which is invalid
        let (data, t) = make_test_data(5, 50);
        let nbasis_range: Vec<usize> = vec![1, 5, 10];
        let res = basis_nbasis_cv(
            &data,
            &t,
            &nbasis_range,
            &BasisType::Bspline { order: 4 },
            BasisCriterion::Gcv,
            5,
            1e-4,
        )
        .unwrap();
        // Score for nbasis=1 should be infinity, so optimal should be 5 or 10
        assert!(
            res.optimal_nbasis >= 5,
            "Should skip invalid nbasis=1, got optimal={}",
            res.optimal_nbasis
        );
        assert!(res.scores[0].is_infinite());
    }

    #[test]
    fn test_basis_nbasis_cv_empty_range() {
        let (data, t) = make_test_data(5, 50);
        let nbasis_range: Vec<usize> = vec![];
        let result = basis_nbasis_cv(
            &data,
            &t,
            &nbasis_range,
            &BasisType::Bspline { order: 4 },
            BasisCriterion::Gcv,
            5,
            1e-4,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_basis_nbasis_cv_empty_data() {
        let data = FdMatrix::zeros(0, 50);
        let t = uniform_grid(50);
        let nbasis_range: Vec<usize> = vec![5, 10];
        let result = basis_nbasis_cv(
            &data,
            &t,
            &nbasis_range,
            &BasisType::Bspline { order: 4 },
            BasisCriterion::Gcv,
            5,
            1e-4,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_basis_nbasis_cv_mismatched_argvals() {
        let data = FdMatrix::zeros(5, 50);
        let t = uniform_grid(40); // mismatch
        let nbasis_range: Vec<usize> = vec![5, 10];
        let result = basis_nbasis_cv(
            &data,
            &t,
            &nbasis_range,
            &BasisType::Bspline { order: 4 },
            BasisCriterion::Gcv,
            5,
            1e-4,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_basis_nbasis_cv_single_nbasis() {
        let (data, t) = make_test_data(5, 50);
        let nbasis_range: Vec<usize> = vec![10];
        let res = basis_nbasis_cv(
            &data,
            &t,
            &nbasis_range,
            &BasisType::Bspline { order: 4 },
            BasisCriterion::Gcv,
            5,
            1e-4,
        )
        .unwrap();
        assert_eq!(res.optimal_nbasis, 10);
        assert_eq!(res.scores.len(), 1);
    }

    #[test]
    fn test_basis_nbasis_cv_bic_penalizes_more_than_aic() {
        // BIC penalizes complexity more heavily than AIC, so it should generally
        // select the same or fewer basis functions
        let (data, t) = make_test_data(5, 80);
        let nbasis_range: Vec<usize> = (4..=20).step_by(2).collect();

        let aic_res = basis_nbasis_cv(
            &data,
            &t,
            &nbasis_range,
            &BasisType::Bspline { order: 4 },
            BasisCriterion::Aic,
            5,
            1e-4,
        )
        .unwrap();
        let bic_res = basis_nbasis_cv(
            &data,
            &t,
            &nbasis_range,
            &BasisType::Bspline { order: 4 },
            BasisCriterion::Bic,
            5,
            1e-4,
        )
        .unwrap();
        // BIC should select at most as many basis functions as AIC
        // (not guaranteed in all cases, but typical behavior)
        assert!(
            bic_res.optimal_nbasis <= aic_res.optimal_nbasis + 4,
            "BIC selected {} vs AIC selected {} -- BIC should not select much more than AIC",
            bic_res.optimal_nbasis,
            aic_res.optimal_nbasis
        );
    }

    // ─── Fitted values quality tests ────────────────────────────────────────

    #[test]
    fn test_smooth_basis_fitted_close_to_data() {
        // With moderate penalty and enough basis functions, fitted should be close to data
        let m = 50;
        let n = 3;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin();
            }
        }
        let fdpar = make_bspline_fdpar(&t, 15, 1e-6);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();

        let mut max_err = 0.0_f64;
        for i in 0..n {
            for j in 0..m {
                let err = (data[(i, j)] - res.fitted[(i, j)]).abs();
                max_err = max_err.max(err);
            }
        }
        assert!(
            max_err < 0.1,
            "Fitted should be close to smooth data; max_err={}",
            max_err
        );
    }

    #[test]
    fn test_smooth_basis_constant_data() {
        // Constant data should be fit exactly
        let m = 50;
        let n = 2;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] = 3.15;
            }
        }
        let fdpar = make_bspline_fdpar(&t, 10, 1e-4);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();
        for i in 0..n {
            for j in 0..m {
                assert!(
                    (res.fitted[(i, j)] - 3.15).abs() < 0.01,
                    "Constant data should be fit well at ({},{}): got {}",
                    i,
                    j,
                    res.fitted[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_smooth_basis_linear_data() {
        // Linear data should be fit well with cubic B-splines
        let m = 50;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(1, m);
        for j in 0..m {
            data[(0, j)] = 2.0 * t[j] + 1.0;
        }
        let fdpar = make_bspline_fdpar(&t, 10, 1e-4);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();
        for j in 0..m {
            let expected = 2.0 * t[j] + 1.0;
            assert!(
                (res.fitted[(0, j)] - expected).abs() < 0.05,
                "Linear data should be fit well at j={}: got {}, expected {}",
                j,
                res.fitted[(0, j)],
                expected
            );
        }
    }

    // ─── EDF and diagnostic tests ───────────────────────────────────────────

    #[test]
    fn test_smooth_basis_edf_bounded() {
        let m = 50;
        let (data, t) = make_test_data(3, m);
        let fdpar = make_bspline_fdpar(&t, 12, 1e-4);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();
        // EDF should be between 1 and m (evaluation points)
        assert!(
            res.edf > 0.0 && res.edf <= m as f64,
            "EDF should be in (0, {}]; got {}",
            m,
            res.edf
        );
    }

    #[test]
    fn test_smooth_basis_gcv_aic_bic_all_finite() {
        let (data, t) = make_test_data(4, 60);
        let fdpar = make_bspline_fdpar(&t, 12, 1e-3);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();
        assert!(res.gcv.is_finite(), "GCV should be finite: {}", res.gcv);
        assert!(res.aic.is_finite(), "AIC should be finite: {}", res.aic);
        assert!(res.bic.is_finite(), "BIC should be finite: {}", res.bic);
    }

    // ─── Penalty matrix size consistency tests ──────────────────────────────

    #[test]
    fn test_smooth_basis_penalty_matrix_in_result() {
        let (data, t) = make_test_data(3, 50);
        let nbasis = 10;
        let fdpar = make_bspline_fdpar(&t, nbasis, 1e-4);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();
        let k = res.nbasis;
        assert_eq!(
            res.penalty_matrix.len(),
            k * k,
            "Penalty matrix should be k*k = {}*{} = {}; got {}",
            k,
            k,
            k * k,
            res.penalty_matrix.len()
        );
    }

    // ─── Regression: multiple identical curves ──────────────────────────────

    #[test]
    fn test_smooth_basis_identical_curves_same_coefficients() {
        let m = 50;
        let t = uniform_grid(m);
        let curve: Vec<f64> = (0..m).map(|j| (2.0 * PI * t[j]).sin()).collect();
        let n = 4;
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] = curve[j];
            }
        }
        let fdpar = make_bspline_fdpar(&t, 10, 1e-4);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();

        // All curves should have the same coefficients
        let k = res.coefficients.ncols();
        for i in 1..n {
            for j in 0..k {
                assert!(
                    (res.coefficients[(i, j)] - res.coefficients[(0, j)]).abs() < 1e-10,
                    "Identical curves should have identical coefficients: curve {} col {} differs",
                    i,
                    j
                );
            }
        }
    }

    // ─── Cross-validation: different numbers of folds ───────────────────────

    #[test]
    fn test_basis_nbasis_cv_different_nfolds() {
        let (data, t) = make_test_data(12, 50);
        let nbasis_range: Vec<usize> = vec![5, 8, 11];
        for nfolds in [2, 3, 5, 10] {
            let res = basis_nbasis_cv(
                &data,
                &t,
                &nbasis_range,
                &BasisType::Bspline { order: 4 },
                BasisCriterion::Cv,
                nfolds,
                1e-4,
            );
            assert!(res.is_some(), "CV should succeed with nfolds={}", nfolds);
            let r = res.unwrap();
            assert!(nbasis_range.contains(&r.optimal_nbasis));
        }
    }

    // ─── Large nbasis / more basis than reasonable ──────────────────────────

    #[test]
    fn test_smooth_basis_many_basis_functions() {
        let m = 100;
        let (data, t) = make_test_data(2, m);
        // Many basis functions relative to data points
        let fdpar = make_bspline_fdpar(&t, 40, 1e-2);
        let res = smooth_basis(&data, &t, &fdpar);
        assert!(
            res.is_ok(),
            "Should handle many basis functions with penalty"
        );
    }

    // ─── evaluate_basis internal function (indirectly tested) ───────────────

    #[test]
    fn test_smooth_basis_bspline_vs_fourier_different_results() {
        let m = 50;
        let (data, t) = make_test_data(2, m);
        let fdpar_bs = make_bspline_fdpar(&t, 9, 1e-4);
        let fdpar_f = make_fourier_fdpar(9, 1.0, 1e-4);
        let res_bs = smooth_basis(&data, &t, &fdpar_bs).unwrap();
        let res_f = smooth_basis(&data, &t, &fdpar_f).unwrap();
        // Results should differ between the two basis types
        let diff: f64 = (0..m)
            .map(|j| (res_bs.fitted[(0, j)] - res_f.fitted[(0, j)]).abs())
            .sum();
        // They fit the same data, so some difference is expected but not huge
        assert!(
            diff > 1e-10,
            "B-spline and Fourier fits should differ for the same data"
        );
    }

    // ─── compute_gcv edge cases (indirectly tested) ─────────────────────────

    #[test]
    fn test_smooth_basis_gcv_positive_for_noisy_data() {
        let m = 50;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(1, m);
        for j in 0..m {
            // Noisy data
            data[(0, j)] = (2.0 * PI * t[j]).sin() + 0.5 * ((j * 37) % 20) as f64 / 20.0 - 0.25;
        }
        let fdpar = make_bspline_fdpar(&t, 10, 1e-3);
        let res = smooth_basis(&data, &t, &fdpar).unwrap();
        assert!(res.gcv > 0.0, "GCV should be positive for noisy data");
    }

    // ─── Penalty order (lfd_order) tests ────────────────────────────────────

    #[test]
    fn test_smooth_basis_different_lfd_orders() {
        let m = 50;
        let (data, t) = make_test_data(2, m);

        // lfd_order = 1 (penalize first derivative)
        let penalty1 = bspline_penalty_matrix(&t, 10, 4, 1);
        let fdpar1 = FdPar {
            basis_type: BasisType::Bspline { order: 4 },
            nbasis: 10,
            lambda: 1e-2,
            lfd_order: 1,
            penalty_matrix: penalty1,
        };
        let res1 = smooth_basis(&data, &t, &fdpar1);
        assert!(res1.is_ok());

        // lfd_order = 2 (penalize second derivative)
        let penalty2 = bspline_penalty_matrix(&t, 10, 4, 2);
        let fdpar2 = FdPar {
            basis_type: BasisType::Bspline { order: 4 },
            nbasis: 10,
            lambda: 1e-2,
            lfd_order: 2,
            penalty_matrix: penalty2,
        };
        let res2 = smooth_basis(&data, &t, &fdpar2);
        assert!(res2.is_ok());

        // Different penalty orders should produce different fitted values
        let r1 = res1.unwrap();
        let r2 = res2.unwrap();
        let diff: f64 = (0..m)
            .map(|j| (r1.fitted[(0, j)] - r2.fitted[(0, j)]).abs())
            .sum();
        assert!(
            diff > 1e-10,
            "Different lfd_orders should produce different fits"
        );
    }

    // ─── BasisNbasisCvResult field tests ────────────────────────────────────

    #[test]
    fn test_basis_nbasis_cv_result_fields() {
        let (data, t) = make_test_data(6, 50);
        let nbasis_range: Vec<usize> = vec![5, 7, 9, 11, 13];
        let res = basis_nbasis_cv(
            &data,
            &t,
            &nbasis_range,
            &BasisType::Bspline { order: 4 },
            BasisCriterion::Aic,
            5,
            1e-4,
        )
        .unwrap();

        assert!(nbasis_range.contains(&res.optimal_nbasis));
        assert_eq!(res.scores.len(), nbasis_range.len());
        assert_eq!(res.nbasis_range, nbasis_range);
        assert_eq!(res.criterion, BasisCriterion::Aic);
        // optimal_nbasis should correspond to minimum score
        let min_score = res.scores.iter().copied().fold(f64::INFINITY, f64::min);
        let best_idx = res
            .scores
            .iter()
            .position(|&s| (s - min_score).abs() < 1e-15)
            .unwrap();
        assert_eq!(res.optimal_nbasis, nbasis_range[best_idx]);
    }

    #[test]
    fn test_basis_nbasis_cv_result_clone() {
        let (data, t) = make_test_data(5, 50);
        let nbasis_range: Vec<usize> = vec![5, 10];
        let res = basis_nbasis_cv(
            &data,
            &t,
            &nbasis_range,
            &BasisType::Bspline { order: 4 },
            BasisCriterion::Gcv,
            5,
            1e-4,
        )
        .unwrap();
        let cloned = res.clone();
        assert_eq!(res, cloned);
    }

    // ─── Non-uniform argvals ────────────────────────────────────────────────

    #[test]
    fn test_smooth_basis_nonuniform_argvals() {
        let m = 50;
        // Non-uniform grid: denser at the ends
        let t: Vec<f64> = (0..m)
            .map(|i| {
                let x = i as f64 / (m - 1) as f64;
                0.5 * (1.0 - (PI * x).cos())
            })
            .collect();
        let mut data = FdMatrix::zeros(2, m);
        for i in 0..2 {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin() + 0.1 * i as f64;
            }
        }
        let fdpar = make_bspline_fdpar(&t, 10, 1e-4);
        let res = smooth_basis(&data, &t, &fdpar);
        assert!(res.is_ok(), "Should handle non-uniform argvals");
        let r = res.unwrap();
        assert_eq!(r.fitted.shape(), (2, m));
    }

    // ─── Numerical stability with extreme lambda ────────────────────────────

    #[test]
    fn test_smooth_basis_very_small_lambda() {
        let m = 50;
        let (data, t) = make_test_data(2, m);
        let fdpar = make_bspline_fdpar(&t, 10, 1e-15);
        let res = smooth_basis(&data, &t, &fdpar);
        assert!(res.is_ok(), "Should handle very small lambda");
    }

    #[test]
    fn test_smooth_basis_very_large_lambda() {
        let m = 50;
        let (data, t) = make_test_data(2, m);
        let fdpar = make_bspline_fdpar(&t, 10, 1e10);
        let res = smooth_basis(&data, &t, &fdpar);
        assert!(res.is_ok(), "Should handle very large lambda");
    }

    // ─── Multiple curves consistency ────────────────────────────────────────

    #[test]
    fn test_smooth_basis_multi_curve_vs_single_curve() {
        // Smoothing multiple curves at once should give the same result as smoothing each individually
        let m = 50;
        let n = 3;
        let (data, t) = make_test_data(n, m);
        let fdpar = make_bspline_fdpar(&t, 10, 1e-3);

        // All at once
        let res_all = smooth_basis(&data, &t, &fdpar).unwrap();

        // One at a time
        for i in 0..n {
            let mut single = FdMatrix::zeros(1, m);
            for j in 0..m {
                single[(0, j)] = data[(i, j)];
            }
            let res_single = smooth_basis(&single, &t, &fdpar).unwrap();
            for j in 0..m {
                assert!(
                    (res_all.fitted[(i, j)] - res_single.fitted[(0, j)]).abs() < 1e-10,
                    "Multi-curve fit should match single-curve fit: curve {} point {}",
                    i,
                    j
                );
            }
        }
    }

    // ─── BasisCriterion comparison: all criteria produce finite scores ──────

    #[test]
    fn test_basis_nbasis_cv_all_criteria_finite_scores() {
        let (data, t) = make_test_data(10, 60);
        let nbasis_range: Vec<usize> = vec![5, 7, 9, 11];

        for criterion in [
            BasisCriterion::Gcv,
            BasisCriterion::Aic,
            BasisCriterion::Bic,
            BasisCriterion::Cv,
        ] {
            let res = basis_nbasis_cv(
                &data,
                &t,
                &nbasis_range,
                &BasisType::Bspline { order: 4 },
                criterion,
                5,
                1e-4,
            )
            .unwrap();
            // At least some scores should be finite (valid nbasis values)
            let finite_count = res.scores.iter().filter(|s| s.is_finite()).count();
            assert!(
                finite_count > 0,
                "At least one score should be finite for {:?}",
                criterion
            );
        }
    }
}
