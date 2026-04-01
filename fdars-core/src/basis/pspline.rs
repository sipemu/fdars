//! P-spline fitting and penalty (difference) matrices.

use super::bspline::{bspline_basis, bspline_basis_from_knots, construct_bspline_knots};
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
#[non_exhaustive]
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
    /// B-spline knot vector used during fitting
    pub knots: Vec<f64>,
    /// B-spline order used during fitting (typically 4 for cubic)
    pub order: usize,
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
    let nknots = nbasis.saturating_sub(4).max(2);
    let spline_order = 4;
    let t_min = argvals.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = argvals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let knots = construct_bspline_knots(t_min, t_max, nknots, spline_order);
    let basis = bspline_basis(argvals, nknots, spline_order);
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
        knots,
        order: spline_order,
    })
}

/// Evaluate P-spline fit on a new grid using stored knot vector.
///
/// Uses the B-spline coefficients and knot vector from [`pspline_fit_1d`]
/// to evaluate the fitted curves at arbitrary evaluation points within
/// the original domain.
///
/// # Arguments
/// * `result` - P-spline fit result containing coefficients and knots
/// * `new_argvals` - New evaluation points
///
/// # Returns
/// Matrix (n x m_new) of evaluated curves
#[must_use]
pub fn pspline_evaluate(result: &PsplineFitResult, new_argvals: &[f64]) -> FdMatrix {
    let n = result.coefficients.nrows();
    let m_new = new_argvals.len();
    let nbasis = result.n_basis;

    let basis = bspline_basis_from_knots(new_argvals, &result.knots, result.order);
    let actual_nbasis = basis.len() / m_new;

    // Column-major: iterate columns then rows
    let flat: Vec<f64> = (0..m_new)
        .flat_map(|j| {
            (0..n)
                .map(|i| {
                    let mut sum = 0.0;
                    for k in 0..actual_nbasis.min(nbasis) {
                        sum += result.coefficients[(i, k)] * basis[j + k * m_new];
                    }
                    sum
                })
                .collect::<Vec<_>>()
        })
        .collect();

    FdMatrix::from_column_major(flat, n, m_new).expect("dimension invariant")
}

/// Fit P-splines with automatic lambda selection via GCV minimization.
///
/// Searches over a logarithmic grid of lambda values and returns the
/// result with the lowest GCV score.
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
/// * `argvals` - Evaluation points (length m)
/// * `nbasis` - Number of basis functions
/// * `order` - Difference penalty order (typically 2)
///
/// # Returns
/// The P-spline fit result with optimal lambda, or `None` if all
/// lambda values produce invalid fits.
#[must_use]
pub fn pspline_fit_gcv(
    data: &FdMatrix,
    argvals: &[f64],
    nbasis: usize,
    order: usize,
) -> Option<PsplineFitResult> {
    let lambdas: Vec<f64> = (-6..=6)
        .flat_map(|exp| {
            [1.0, 3.16]
                .iter()
                .map(move |&base| base * 10.0f64.powi(exp))
        })
        .collect();

    let mut best: Option<PsplineFitResult> = None;
    let mut best_gcv = f64::INFINITY;

    for &lambda in &lambdas {
        if let Some(result) = pspline_fit_1d(data, argvals, nbasis, lambda, order) {
            if result.gcv.is_finite() && result.gcv < best_gcv {
                best_gcv = result.gcv;
                best = Some(result);
            }
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::FdMatrix;

    fn sine_data(argvals: &[f64]) -> FdMatrix {
        let n = 3;
        let m = argvals.len();
        let vals: Vec<f64> = (0..n)
            .flat_map(|i| argvals.iter().map(move |&t| ((i + 1) as f64 * t).sin()))
            .collect();
        FdMatrix::from_column_major(vals, n, m).unwrap()
    }

    #[test]
    fn pspline_stores_knots() {
        let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
        let data = sine_data(&t);
        let result = pspline_fit_1d(&data, &t, 10, 0.001, 2).unwrap();
        assert!(!result.knots.is_empty());
        assert_eq!(result.order, 4);
    }

    #[test]
    fn pspline_evaluate_on_original_grid() {
        let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
        let data = sine_data(&t);
        let result = pspline_fit_1d(&data, &t, 10, 0.001, 2).unwrap();
        let eval = pspline_evaluate(&result, &t);
        // Should match fitted values closely
        for i in 0..data.nrows() {
            for j in 0..t.len() {
                assert!(
                    (eval[(i, j)] - result.fitted[(i, j)]).abs() < 1e-10,
                    "mismatch at ({i}, {j}): eval={} fitted={}",
                    eval[(i, j)],
                    result.fitted[(i, j)]
                );
            }
        }
    }

    #[test]
    fn pspline_evaluate_on_finer_grid() {
        let t: Vec<f64> = (0..30).map(|i| i as f64 / 29.0).collect();
        let data = sine_data(&t);
        let result = pspline_fit_1d(&data, &t, 12, 0.001, 2).unwrap();

        // Evaluate on a finer grid
        let t_fine: Vec<f64> = (0..200).map(|i| i as f64 / 199.0).collect();
        let eval = pspline_evaluate(&result, &t_fine);
        assert_eq!(eval.nrows(), data.nrows());
        assert_eq!(eval.ncols(), 200);

        // Values should be in a reasonable range (sin is in [-1, 1])
        for i in 0..eval.nrows() {
            for j in 0..eval.ncols() {
                assert!(
                    eval[(i, j)].abs() < 2.0,
                    "wild value at ({i}, {j}): {}",
                    eval[(i, j)]
                );
            }
        }
    }

    /// Regression test for issue #21: pspline_evaluate on a different grid
    /// matches the fitted values when evaluated at original grid points,
    /// and produces bounded values on a finer grid (no wild oscillations).
    #[test]
    fn regression_issue_21_cross_grid_consistency() {
        // Fit on 31-point grid in [1, 18] (non-[0,1] domain like growth data)
        let t: Vec<f64> = (0..31).map(|i| 1.0 + 17.0 * i as f64 / 30.0).collect();
        let data = sine_data(&t);
        let result = pspline_fit_1d(&data, &t, 12, 1.0, 2).unwrap();

        // Evaluate at original grid points — must match fitted exactly
        let eval_orig = pspline_evaluate(&result, &t);
        for i in 0..data.nrows() {
            for j in 0..t.len() {
                assert!(
                    (eval_orig[(i, j)] - result.fitted[(i, j)]).abs() < 1e-10,
                    "mismatch at ({i}, {j})"
                );
            }
        }

        // Evaluate on 200-point finer grid — values must stay bounded
        let t_fine: Vec<f64> = (0..200).map(|i| 1.0 + 17.0 * i as f64 / 199.0).collect();
        let eval_fine = pspline_evaluate(&result, &t_fine);
        assert_eq!(eval_fine.shape(), (data.nrows(), 200));

        // Find range of fitted values for reference
        let mut fit_min = f64::INFINITY;
        let mut fit_max = f64::NEG_INFINITY;
        for i in 0..data.nrows() {
            for j in 0..t.len() {
                let v = result.fitted[(i, j)];
                fit_min = fit_min.min(v);
                fit_max = fit_max.max(v);
            }
        }
        let margin = (fit_max - fit_min) * 0.5;

        for i in 0..eval_fine.nrows() {
            for j in 0..200 {
                assert!(
                    eval_fine[(i, j)] > fit_min - margin && eval_fine[(i, j)] < fit_max + margin,
                    "curve {i} at t={:.2}: value {:.2} outside [{:.1}, {:.1}]",
                    t_fine[j],
                    eval_fine[(i, j)],
                    fit_min - margin,
                    fit_max + margin,
                );
            }
        }
    }

    /// Regression test: pspline_evaluate on a non-[0,1] domain produces
    /// smooth interpolation (not wild oscillations as in old basis_to_fdata).
    #[test]
    fn regression_issue_21_non_unit_domain_interp() {
        // Fit on [5, 15] (arbitrary non-[0,1] domain)
        let t: Vec<f64> = (0..40).map(|i| 5.0 + 10.0 * i as f64 / 39.0).collect();
        let data = sine_data(&t);
        let result = pspline_fit_1d(&data, &t, 10, 1.0, 2).unwrap();

        // Evaluate at midpoints between original grid points
        let t_mid: Vec<f64> = t.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
        let eval_mid = pspline_evaluate(&result, &t_mid);

        // Midpoint values should be between neighboring fitted values (approximately)
        for i in 0..data.nrows() {
            for j in 0..t_mid.len() {
                let lo = result.fitted[(i, j)].min(result.fitted[(i, j + 1)]);
                let hi = result.fitted[(i, j)].max(result.fitted[(i, j + 1)]);
                let margin = (hi - lo).abs() + 0.5; // generous margin
                assert!(
                    eval_mid[(i, j)] >= lo - margin && eval_mid[(i, j)] <= hi + margin,
                    "curve {i} midpoint {j}: value {:.4} far from [{:.4}, {:.4}]",
                    eval_mid[(i, j)],
                    lo,
                    hi
                );
            }
        }
    }

    #[test]
    fn pspline_evaluate_matches_at_subset_points() {
        // Fit on fine grid, evaluate on coarser subset
        let t: Vec<f64> = (0..100).map(|i| i as f64 / 99.0).collect();
        let data = sine_data(&t);
        let result = pspline_fit_1d(&data, &t, 15, 0.0001, 2).unwrap();

        // Evaluate on every 10th point
        let t_coarse: Vec<f64> = (0..10).map(|i| i as f64 * 10.0 / 99.0).collect();
        let eval = pspline_evaluate(&result, &t_coarse);

        for i in 0..data.nrows() {
            for (jc, &_tc) in t_coarse.iter().enumerate() {
                let j_orig = jc * 10;
                let diff = (eval[(i, jc)] - result.fitted[(i, j_orig)]).abs();
                assert!(
                    diff < 1e-8,
                    "mismatch at curve {i}, coarse idx {jc} (orig {j_orig}): diff={diff}"
                );
            }
        }
    }
}
