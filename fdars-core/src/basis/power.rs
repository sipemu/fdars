//! Power basis: `B_j(t) = t^{exponents[j]}` with general (possibly non-integer) exponents.
//!
//! ## Evaluation matrix
//!
//! The evaluation matrix is column-major of shape `(n × nbasis)`:
//!
//! ```text
//! eval_matrix[i + j * n] = argvals[i].powf(exponents[j])
//! ```
//!
//! ## Domain constraint
//!
//! When any exponent is **non-integer** or **negative**, the basis is undefined
//! at `t ≤ 0` (e.g., `0.0_f64.powf(-0.5) = ∞`, `(-1.0_f64).powf(0.5) = NaN`).
//! In that case **all** `argvals` must be strictly positive; otherwise
//! [`FdarError::InvalidParameter`] is returned (Pitfall 1 from Research).
//!
//! When all exponents are non-negative integers (i.e., the monomial case),
//! `t = 0` is permitted.
//!
//! ## Roughness penalty
//!
//! - **All exponents are non-negative integers**: analytic falling-factorial Gram
//!   (same formula as `monomial_basis`; exact, no quadrature).
//! - **Otherwise**: numeric Gram via the `pub(crate)` helpers promoted in 35-01,
//!   restricted to the strictly-positive domain.
//!
//! Default `lfd_order = 2` (curvature roughness).
//!
//! Reference: R `fda` package `create.power.basis`.

use crate::basis::basis_system::BasisSystem;
use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::smooth_basis::{differentiate_basis_columns, integrate_symmetric_penalty};

// ─── Public factory ──────────────────────────────────────────────────────────

/// Construct a power basis over `argvals` with one function per entry in `exponents`.
///
/// The j-th basis function is `B_j(t) = t^{exponents[j]}`.
///
/// When all exponents are non-negative integers the penalty is analytic (same as
/// [`monomial_basis`]); otherwise a numeric Gram is used on the fine quadrature
/// grid.
///
/// # Arguments
///
/// * `argvals`   — Evaluation points (length ≥ 2).
/// * `exponents` — Exponent values (one per basis function; length ≥ 1).
///
/// # Errors
///
/// - [`FdarError::InvalidDimension`] if `argvals.len() < 2`.
/// - [`FdarError::InvalidParameter`] if `exponents` is empty.
/// - [`FdarError::InvalidParameter`] if any exponent is non-integer or negative
///   **and** any `argvals[i] ≤ 0.0` (non-positive domain forbidden for such exponents).
///
/// # Examples
///
/// ```
/// use fdars_core::power_basis;
///
/// // Integer exponents match monomial behaviour
/// let t = vec![0.0, 1.0, 2.0];
/// let bs = power_basis(&t, &[0.0, 1.0, 2.0]).unwrap();
/// let n = bs.n_eval;
/// // B₂(t) = t²  →  column 2: [0, 1, 4]
/// assert!((bs.eval_matrix[2 * n] - 0.0).abs() < 1e-12);
/// assert!((bs.eval_matrix[2 * n + 1] - 1.0).abs() < 1e-12);
/// assert!((bs.eval_matrix[2 * n + 2] - 4.0).abs() < 1e-12);
/// ```
pub fn power_basis(argvals: &[f64], exponents: &[f64]) -> Result<BasisSystem, FdarError> {
    let n = argvals.len();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: ">= 2".to_string(),
            actual: n.to_string(),
        });
    }
    let nbasis = exponents.len();
    if nbasis < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "exponents",
            message: "must be non-empty".to_string(),
        });
    }

    // Determine whether any exponent requires a strictly-positive domain.
    let requires_positive = exponents
        .iter()
        .any(|&e| !is_nonneg_integer(e));

    if requires_positive {
        // Reject non-positive argvals to prevent NaN/Inf leaking into eval/penalty.
        let bad = argvals.iter().any(|&t| t <= 0.0);
        if bad {
            return Err(FdarError::InvalidParameter {
                parameter: "argvals",
                message: "all argvals must be strictly positive when any exponent is non-integer \
                           or negative (t ≤ 0 produces NaN/Inf for such exponents)"
                    .to_string(),
            });
        }
    }

    // Build column-major evaluation matrix: eval_matrix[i + j*n] = t^exponents[j]
    let mut eval_matrix = vec![0.0_f64; n * nbasis];
    for (ti, &t) in argvals.iter().enumerate() {
        for j in 0..nbasis {
            eval_matrix[ti + j * n] = t.powf(exponents[j]);
        }
    }

    let lfd_order = 2_usize;
    let a = argvals[0];
    let b = argvals[n - 1];

    let penalty_matrix = if !requires_positive {
        // All non-negative integer exponents → analytic Gram (exact).
        power_penalty_analytic(exponents, lfd_order, a, b)
    } else {
        // Non-integer or negative exponents → numeric Gram.
        power_penalty_numeric(argvals, exponents, nbasis, lfd_order)
    };

    Ok(BasisSystem {
        eval_matrix,
        penalty_matrix,
        nbasis,
        n_eval: n,
        lfd_order,
    })
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Returns true iff `e` represents a non-negative integer (0, 1, 2, …).
fn is_nonneg_integer(e: f64) -> bool {
    e >= 0.0 && e == e.floor() && e.is_finite()
}

/// Falling factorial: `e * (e-1) * … * (e-d+1)`.
/// Returns 1.0 if `d == 0`.
fn falling_factorial(e: f64, d: usize) -> f64 {
    if d == 0 {
        return 1.0;
    }
    (0..d).fold(1.0_f64, |acc, k| acc * (e - k as f64))
}

/// Analytic Gram entry for the `d`-th derivative of `t^{e_i}` and `t^{e_j}` on `[a, b]`.
///
/// Formula:
/// ```text
/// p = e_i + e_j - 2d + 1
/// if |p| < 1e-15: c_i * c_j * ln(b/a)
/// else:           c_i * c_j * (b^p - a^p) / p
/// ```
fn gram_entry(ei: f64, ej: f64, d: usize, a: f64, b: f64) -> f64 {
    let ci = falling_factorial(ei, d);
    let cj = falling_factorial(ej, d);
    if ci.abs() < 1e-15 || cj.abs() < 1e-15 {
        return 0.0;
    }
    let p = ei + ej - 2.0 * d as f64 + 1.0;
    if p.abs() < 1e-15 {
        if a <= 0.0 {
            return 0.0; // improper integral at a=0; set to 0 conservatively
        }
        ci * cj * (b.ln() - a.ln())
    } else {
        ci * cj * (b.powf(p) - a.powf(p)) / p
    }
}

/// Analytic penalty matrix for integer exponents (falling-factorial Gram).
fn power_penalty_analytic(exponents: &[f64], lfd_order: usize, a: f64, b: f64) -> Vec<f64> {
    let k = exponents.len();
    let mut penalty = vec![0.0_f64; k * k];
    for j in 0..k {
        for l in j..k {
            let val = gram_entry(exponents[j], exponents[l], lfd_order, a, b);
            penalty[j + l * k] = val;
            penalty[l + j * k] = val;
        }
    }
    penalty
}

/// Numeric Gram penalty for non-integer/negative exponents.
///
/// Uses a fine quadrature grid (10 sub-points per original interval)
/// over the supplied argvals range.
fn power_penalty_numeric(
    argvals: &[f64],
    exponents: &[f64],
    nbasis: usize,
    lfd_order: usize,
) -> Vec<f64> {
    if argvals.len() < 2 {
        return vec![0.0; nbasis * nbasis];
    }
    let t_min = argvals[0];
    let t_max = argvals[argvals.len() - 1];
    // Strictly positive grid (domain check already done in caller)
    let t_min_safe = t_min.max(1e-10);
    let n_sub = 10;
    let n_quad = (argvals.len() - 1) * n_sub + 1;

    let quad_t: Vec<f64> = (0..n_quad)
        .map(|i| t_min_safe + (t_max - t_min_safe) * i as f64 / (n_quad - 1) as f64)
        .collect();

    let mut basis_fine = vec![0.0_f64; n_quad * nbasis];
    for (ti, &t) in quad_t.iter().enumerate() {
        for j in 0..nbasis {
            basis_fine[ti + j * n_quad] = t.powf(exponents[j]);
        }
    }

    let h = (t_max - t_min_safe) / (n_quad - 1) as f64;
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
    fn power_invalid_argvals_too_short() {
        let result = power_basis(&[0.5], &[1.0, 2.0]);
        assert!(matches!(result, Err(FdarError::InvalidDimension { .. })));
    }

    /// Returns FdarError::InvalidParameter for empty exponents.
    #[test]
    fn power_invalid_empty_exponents() {
        let result = power_basis(&[0.0, 1.0], &[]);
        assert!(matches!(result, Err(FdarError::InvalidParameter { .. })));
    }

    /// Non-positive argval with a negative exponent → InvalidParameter (no NaN/Inf).
    #[test]
    fn power_rejects_nonpositive_argval_with_negative_exponent() {
        let result = power_basis(&[-1.0, 1.0], &[-1.0]);
        assert!(matches!(result, Err(FdarError::InvalidParameter { .. })));
    }

    /// Non-positive argval (zero) with a fractional exponent → InvalidParameter.
    #[test]
    fn power_rejects_zero_argval_with_fractional_exponent() {
        let result = power_basis(&[0.0, 1.0], &[0.5]);
        assert!(matches!(result, Err(FdarError::InvalidParameter { .. })));
    }

    /// Integer exponents [0,1,2] on t=[0,1,2]: eval matches monomial_basis exactly.
    #[test]
    fn power_integer_exponents_match_monomial() {
        use crate::basis::monomial::monomial_basis;
        let t = vec![0.0, 1.0, 2.0];
        let bs_power = power_basis(&t, &[0.0, 1.0, 2.0]).unwrap();
        let bs_mono = monomial_basis(&t, 3).unwrap();
        assert_eq!(bs_power.eval_matrix.len(), bs_mono.eval_matrix.len());
        for (a, b) in bs_power.eval_matrix.iter().zip(bs_mono.eval_matrix.iter()) {
            assert!((a - b).abs() < 1e-12, "power eval {a} != monomial eval {b}");
        }
    }

    /// B_j(t_i) == t_i^exponents[j] for non-integer exponents (closed-form spot check).
    #[test]
    fn power_noninteger_eval_closed_form() {
        let t = vec![1.0, 1.5, 2.0];
        let exponents = [0.5, 1.5];
        let bs = power_basis(&t, &exponents).unwrap();
        let n = bs.n_eval;
        // col 0 (exp=0.5): [1^0.5, 1.5^0.5, 2^0.5]
        assert!((bs.eval_matrix[0] - 1.0_f64.powf(0.5)).abs() < 1e-12, "B_0(1)");
        assert!(
            (bs.eval_matrix[1] - 1.5_f64.powf(0.5)).abs() < 1e-12,
            "B_0(1.5)"
        );
        assert!(
            (bs.eval_matrix[2] - 2.0_f64.powf(0.5)).abs() < 1e-12,
            "B_0(2)"
        );
        // col 1 (exp=1.5): at t=1.5
        assert!(
            (bs.eval_matrix[n + 1] - 1.5_f64.powf(1.5)).abs() < 1e-12,
            "B_1(1.5)"
        );
    }

    /// eval_matrix and penalty_matrix have correct shapes.
    #[test]
    fn power_shape_invariants() {
        let t: Vec<f64> = (1..=5).map(|i| i as f64).collect();
        let exponents = [0.5, 1.0, 1.5];
        let bs = power_basis(&t, &exponents).unwrap();
        assert_eq!(bs.eval_matrix.len(), 5 * 3);
        assert_eq!(bs.penalty_matrix.len(), 3 * 3);
        assert_eq!(bs.nbasis, 3);
        assert_eq!(bs.n_eval, 5);
        assert_eq!(bs.lfd_order, 2);
    }

    /// Penalty matrix is symmetric (integer exponent path).
    #[test]
    fn power_penalty_symmetric_integer() {
        let t: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
        let bs = power_basis(&t, &[0.0, 1.0, 2.0, 3.0]).unwrap();
        let k = bs.nbasis;
        for j in 0..k {
            for l in 0..k {
                let pjl = bs.penalty_matrix[j + l * k];
                let plj = bs.penalty_matrix[l + j * k];
                assert!(
                    (pjl - plj).abs() < 1e-12,
                    "P[{j},{l}]={pjl} != P[{l},{j}]={plj}"
                );
            }
        }
    }

    /// Penalty matrix is symmetric (non-integer exponent path).
    #[test]
    fn power_penalty_symmetric_fractional() {
        let t: Vec<f64> = (1..=10).map(|i| i as f64 / 10.0).collect();
        let bs = power_basis(&t, &[0.5, 1.5]).unwrap();
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

    /// Penalty diagonal is non-negative (PSD check), integer path.
    #[test]
    fn power_penalty_diagonal_nonneg_integer() {
        let t: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
        let bs = power_basis(&t, &[0.0, 1.0, 2.0]).unwrap();
        let k = bs.nbasis;
        for j in 0..k {
            let diag = bs.penalty_matrix[j + j * k];
            assert!(diag >= -1e-10, "P[{j},{j}]={diag} is negative");
        }
    }

    /// P[0,0]==0 and P[1,1]==0 for integer exponents [0,1,2], lfd_order=2.
    #[test]
    fn power_penalty_low_exponents_zero() {
        let t = vec![0.0, 0.5, 1.0];
        let bs = power_basis(&t, &[0.0, 1.0, 2.0]).unwrap();
        let k = bs.nbasis;
        assert!(bs.penalty_matrix[0].abs() < 1e-12, "P[0,0] should be 0");
        assert!(bs.penalty_matrix[1 + k].abs() < 1e-12, "P[1,1] should be 0");
    }

    /// BasisSystem derives Debug, Clone, PartialEq.
    #[test]
    fn power_basis_system_derives() {
        let t = vec![1.0, 2.0];
        let bs = power_basis(&t, &[0.5]).unwrap();
        let bs2 = bs.clone();
        assert_eq!(bs, bs2);
        let _ = format!("{bs:?}");
    }
}
