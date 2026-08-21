//! Exponential basis: `B_j(t) = exp(rates[j] · t)`.
//!
//! ## Evaluation matrix
//!
//! The evaluation matrix is column-major of shape `(n × nbasis)`:
//!
//! ```text
//! eval_matrix[i + j * n] = exp(rates[j] * argvals[i])
//! ```
//!
//! When `rates[j] = 0`, the j-th basis function is identically 1 (constant).
//!
//! ## Roughness penalty
//!
//! The penalty matrix is computed via **numeric quadrature** (Gram of the
//! `lfd_order`-th derivatives on a fine sub-grid).  An analytic formula exists
//! (`r_i^d * r_j^d * ∫ exp((r_i+r_j)t) dt`) but requires special-casing
//! `r_i + r_j = 0`.  The numeric approach is simpler and consistent with the
//! B-spline penalty pattern; see Research Open-Q3.
//!
//! Default `lfd_order = 2` (curvature roughness).
//!
//! Reference: R `fda` package `create.exponential.basis`.

use crate::basis::basis_system::BasisSystem;
use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::smooth_basis::{differentiate_basis_columns, integrate_symmetric_penalty};

// ─── Public factory ──────────────────────────────────────────────────────────

/// Construct an exponential basis over `argvals` with one function per entry in `rates`.
///
/// The j-th basis function is `B_j(t) = exp(rates[j] * t)`.  When `rates[j] = 0`
/// the function is identically 1 (constant).
///
/// Returns a [`BasisSystem`] containing:
/// - a column-major evaluation matrix of shape `(n × nbasis)`, and
/// - a numeric 2nd-derivative Gram penalty matrix of shape `(nbasis × nbasis)`.
///
/// # Arguments
///
/// * `argvals` — Evaluation points (length ≥ 2).
/// * `rates`   — Rate parameters (one per basis function; length ≥ 1).
///
/// # Errors
///
/// - [`FdarError::InvalidDimension`] if `argvals.len() < 2`.
/// - [`FdarError::InvalidParameter`] if `rates` is empty.
///
/// # Examples
///
/// ```
/// use fdars_core::exponential_basis;
///
/// let t = vec![0.0, 1.0];
/// let bs = exponential_basis(&t, &[0.0, -1.0]).unwrap();
/// // Column 0 (rate = 0): exp(0) = 1 at all points
/// assert!((bs.eval_matrix[0] - 1.0).abs() < 1e-12);
/// assert!((bs.eval_matrix[1] - 1.0).abs() < 1e-12);
/// // Column 1 (rate = -1): exp(-t) → [1, exp(-1)]
/// let n = bs.n_eval;
/// assert!((bs.eval_matrix[n] - 1.0).abs() < 1e-12);
/// assert!((bs.eval_matrix[n + 1] - (-1.0_f64).exp()).abs() < 1e-12);
/// ```
pub fn exponential_basis(argvals: &[f64], rates: &[f64]) -> Result<BasisSystem, FdarError> {
    let n = argvals.len();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: ">= 2".to_string(),
            actual: n.to_string(),
        });
    }
    let nbasis = rates.len();
    if nbasis < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "rates",
            message: "must be non-empty".to_string(),
        });
    }

    // Build column-major evaluation matrix: eval_matrix[i + j*n] = exp(rates[j] * t_i)
    let mut eval_matrix = vec![0.0_f64; n * nbasis];
    for (ti, &t) in argvals.iter().enumerate() {
        for j in 0..nbasis {
            eval_matrix[ti + j * n] = (rates[j] * t).exp();
        }
    }

    // Numeric Gram penalty (lfd_order = 2) on a fine quadrature grid.
    // Grid density: 10 sub-points per original interval (same as bspline_penalty_matrix).
    let lfd_order = 2_usize;
    let penalty_matrix = exponential_penalty_numeric(argvals, rates, nbasis, lfd_order);

    Ok(BasisSystem {
        eval_matrix,
        penalty_matrix,
        nbasis,
        n_eval: n,
        lfd_order,
    })
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Compute the numeric Gram penalty for the exponential basis.
///
/// Builds a fine quadrature grid (10 sub-points per original interval),
/// evaluates the exponential basis on that grid, differentiates `lfd_order`
/// times, then integrates the symmetric outer product.
fn exponential_penalty_numeric(
    argvals: &[f64],
    rates: &[f64],
    nbasis: usize,
    lfd_order: usize,
) -> Vec<f64> {
    if argvals.len() < 2 {
        return vec![0.0; nbasis * nbasis];
    }
    let t_min = argvals[0];
    let t_max = argvals[argvals.len() - 1];
    let n_sub = 10;
    let n_quad = (argvals.len() - 1) * n_sub + 1;

    // Fine uniform grid
    let quad_t: Vec<f64> = (0..n_quad)
        .map(|i| t_min + (t_max - t_min) * i as f64 / (n_quad - 1) as f64)
        .collect();

    // Evaluate basis on fine grid
    let mut basis_fine = vec![0.0_f64; n_quad * nbasis];
    for (ti, &t) in quad_t.iter().enumerate() {
        for j in 0..nbasis {
            basis_fine[ti + j * n_quad] = (rates[j] * t).exp();
        }
    }

    let h = (t_max - t_min) / (n_quad - 1) as f64;
    let deriv_basis = differentiate_basis_columns(&basis_fine, n_quad, nbasis, h, lfd_order);
    let weights = simpsons_weights(&quad_t);
    integrate_symmetric_penalty(&deriv_basis, &weights, nbasis, n_quad)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns FdarError::InvalidDimension for argvals.len() < 2.
    #[test]
    fn exp_invalid_argvals_too_short() {
        let result = exponential_basis(&[0.5], &[0.0, -1.0]);
        assert!(matches!(result, Err(FdarError::InvalidDimension { .. })));
    }

    /// Returns FdarError::InvalidParameter for empty rates.
    #[test]
    fn exp_invalid_empty_rates() {
        let result = exponential_basis(&[0.0, 1.0], &[]);
        assert!(matches!(result, Err(FdarError::InvalidParameter { .. })));
    }

    /// At t=0 all basis functions equal 1 (exp(rate * 0) == 1 for any rate).
    #[test]
    fn exp_eval_at_zero_is_one() {
        let t = vec![0.0, 1.0];
        let rates = [0.0, -1.0, -5.0, 2.0];
        let bs = exponential_basis(&t, &rates).unwrap();
        // Row 0 of each column (ti=0): eval_matrix[0 + j*n_eval]
        let n = bs.n_eval;
        for j in 0..bs.nbasis {
            let val = bs.eval_matrix[j * n]; // ti=0
            assert!((val - 1.0).abs() < 1e-12, "B_{j}(0)={val}, expected 1.0");
        }
    }

    /// rate=0 column is identically 1.0 (constant function).
    #[test]
    fn exp_rate_zero_is_constant_one() {
        let t: Vec<f64> = (0..5).map(|i| i as f64 * 0.5).collect();
        let bs = exponential_basis(&t, &[0.0]).unwrap();
        let n = bs.n_eval;
        for ti in 0..n {
            let val = bs.eval_matrix[ti]; // j=0
            assert!((val - 1.0).abs() < 1e-12, "B_0(t_{ti})={val}, expected 1.0");
        }
    }

    /// Closed-form eval: t=[0,1], rates=[0,-1] → col1 = [1, exp(-1)].
    #[test]
    fn exp_eval_closed_form() {
        let t = vec![0.0, 1.0];
        let bs = exponential_basis(&t, &[0.0, -1.0]).unwrap();
        let n = bs.n_eval; // == 2
        // col 0 (rate=0): [1, 1]
        assert!((bs.eval_matrix[0] - 1.0).abs() < 1e-12, "B_0(0)");
        assert!((bs.eval_matrix[1] - 1.0).abs() < 1e-12, "B_0(1)");
        // col 1 (rate=-1): [1, exp(-1)]
        assert!((bs.eval_matrix[n] - 1.0).abs() < 1e-12, "B_1(0)");
        assert!(
            (bs.eval_matrix[n + 1] - (-1.0_f64).exp()).abs() < 1e-12,
            "B_1(1)"
        );
    }

    /// eval_matrix and penalty_matrix have correct shapes.
    #[test]
    fn exp_shape_invariants() {
        let t: Vec<f64> = (0..5).map(|i| i as f64 / 4.0).collect();
        let rates = [0.0, -1.0, -2.0];
        let bs = exponential_basis(&t, &rates).unwrap();
        assert_eq!(bs.eval_matrix.len(), 5 * 3);
        assert_eq!(bs.penalty_matrix.len(), 3 * 3);
        assert_eq!(bs.nbasis, 3);
        assert_eq!(bs.n_eval, 5);
        assert_eq!(bs.lfd_order, 2);
    }

    /// Penalty matrix is symmetric.
    #[test]
    fn exp_penalty_symmetric() {
        let t: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
        let rates = [0.0, -1.0, -3.0];
        let bs = exponential_basis(&t, &rates).unwrap();
        let k = bs.nbasis;
        for j in 0..k {
            for l in 0..k {
                let pjl = bs.penalty_matrix[j + l * k];
                let plj = bs.penalty_matrix[l + j * k];
                assert!(
                    (pjl - plj).abs() < 1e-10,
                    "P[{j},{l}]={pjl} != P[{l},{j}]={plj}"
                );
            }
        }
    }

    /// Penalty diagonal is non-negative (PSD check).
    #[test]
    fn exp_penalty_diagonal_nonneg() {
        let t: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
        let bs = exponential_basis(&t, &[0.0, -1.0, 1.0]).unwrap();
        let k = bs.nbasis;
        for j in 0..k {
            let diag = bs.penalty_matrix[j + j * k];
            assert!(diag >= -1e-10, "P[{j},{j}]={diag} is negative");
        }
    }

    /// BasisSystem derives Debug, Clone, PartialEq.
    #[test]
    fn exp_basis_system_derives() {
        let t = vec![0.0, 1.0];
        let bs = exponential_basis(&t, &[0.0]).unwrap();
        let bs2 = bs.clone();
        assert_eq!(bs, bs2);
        let _ = format!("{bs:?}");
    }
}
