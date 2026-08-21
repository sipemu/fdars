//! Monomial basis: `B_j(t) = t^j` for `j = 0, 1, …, nbasis-1`.
//!
//! ## Evaluation matrix
//!
//! The evaluation matrix is column-major of shape `(n × nbasis)`:
//!
//! ```text
//! eval_matrix[i + j * n] = argvals[i].powi(j as i32)
//! ```
//!
//! ## Roughness penalty
//!
//! The default roughness order is `lfd_order = 2` (curvature).  The penalty
//! matrix is computed analytically using the exact Gram integral of the
//! `lfd_order`-th derivative of each basis function.
//!
//! For integer exponents `e_i`, `e_j` and domain `[a, b]`:
//!
//! ```text
//! c_i = e_i * (e_i-1) * … * (e_i-d+1)   (falling factorial)
//! c_j = e_j * (e_j-1) * … * (e_j-d+1)
//!
//! R[i,j] = 0                                  if c_i ≈ 0 or c_j ≈ 0
//!         = c_i * c_j * ln(b/a)               if |e_i + e_j - 2d + 1| < 1e-15
//!         = c_i * c_j * (b^p - a^p) / p       otherwise, p = e_i + e_j - 2d + 1
//! ```
//!
//! Reference: standard polynomial calculus, same semantics as R's `fda` package
//! `create.monomial.basis`.

use crate::basis::basis_system::BasisSystem;
use crate::error::FdarError;

// ─── Public factory ──────────────────────────────────────────────────────────

/// Construct a monomial (polynomial power) basis over `argvals` with `nbasis` functions.
///
/// The j-th basis function is `B_j(t) = t^j` for `j = 0, …, nbasis-1`.
///
/// Returns a [`BasisSystem`] containing:
/// - a column-major evaluation matrix of shape `(n × nbasis)`,
/// - an analytic 2nd-derivative Gram penalty matrix of shape `(nbasis × nbasis)`.
///
/// # Errors
///
/// - [`FdarError::InvalidDimension`] if `argvals.len() < 2`.
/// - [`FdarError::InvalidParameter`] if `nbasis < 1`.
///
/// # Examples
///
/// ```
/// use fdars_core::monomial_basis;
///
/// let t = vec![0.0, 1.0, 2.0];
/// let bs = monomial_basis(&t, 3).unwrap();
/// // Column 0: B₀ = 1 → [1, 1, 1]
/// assert!((bs.eval_matrix[0] - 1.0).abs() < 1e-12);
/// assert!((bs.eval_matrix[1] - 1.0).abs() < 1e-12);
/// assert!((bs.eval_matrix[2] - 1.0).abs() < 1e-12);
/// // Column 1: B₁ = t → [0, 1, 2]
/// assert!((bs.eval_matrix[3] - 0.0).abs() < 1e-12);
/// assert!((bs.eval_matrix[4] - 1.0).abs() < 1e-12);
/// assert!((bs.eval_matrix[5] - 2.0).abs() < 1e-12);
/// // Column 2: B₂ = t² → [0, 1, 4]
/// assert!((bs.eval_matrix[6] - 0.0).abs() < 1e-12);
/// assert!((bs.eval_matrix[7] - 1.0).abs() < 1e-12);
/// assert!((bs.eval_matrix[8] - 4.0).abs() < 1e-12);
/// // P[2,2] for exponents [0,1,2], lfd_order=2, domain [0,2]:
/// // c₂=2, c₂=2, power=2+2-4+1=1, P=2*2*(2^1-0^1)/1 = 8.0
/// let p22 = bs.penalty_matrix[2 + 2 * 3];
/// assert!((p22 - 8.0).abs() < 1e-9, "P[2,2]={p22}");
/// ```
pub fn monomial_basis(argvals: &[f64], nbasis: usize) -> Result<BasisSystem, FdarError> {
    let n = argvals.len();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: ">= 2".to_string(),
            actual: n.to_string(),
        });
    }
    if nbasis < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "nbasis",
            message: "must be >= 1".to_string(),
        });
    }

    // Build column-major evaluation matrix: eval_matrix[i + j*n] = t^j
    let mut eval_matrix = vec![0.0_f64; n * nbasis];
    for (ti, &t) in argvals.iter().enumerate() {
        for j in 0..nbasis {
            eval_matrix[ti + j * n] = t.powi(j as i32);
        }
    }

    // Compute analytic penalty matrix with lfd_order = 2
    let lfd_order = 2_usize;
    let a = argvals[0];
    let b = argvals[n - 1];
    let penalty_matrix = monomial_penalty_analytic(nbasis, lfd_order, a, b);

    Ok(BasisSystem {
        eval_matrix,
        penalty_matrix,
        nbasis,
        n_eval: n,
        lfd_order,
    })
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Falling factorial: `e * (e-1) * … * (e-d+1)`.
/// Returns 1.0 if `d == 0`, and 0.0 if any factor is zero (integer exponent < d).
fn falling_factorial(e: f64, d: usize) -> f64 {
    if d == 0 {
        return 1.0;
    }
    let mut acc = 1.0_f64;
    for k in 0..d {
        acc *= e - k as f64;
    }
    acc
}

/// Analytic Gram entry for the `d`-th derivative penalty of `t^{e_i}` and `t^{e_j}`
/// on domain `[a, b]`.
///
/// Formula:
/// ```text
/// p = e_i + e_j - 2*d + 1
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
        // Integral of t^{-1} over [a,b] — requires a > 0
        if a <= 0.0 {
            // Integral ∫₀ᵇ t⁻¹ dt is improper — this path is unreachable for the
            // current lfd_order=2 with non-negative integer exponents (all falling
            // factorials yield ei,ej >= 2, so p = ei+ej-3 >= 1 > 0).
            // If lfd_order is ever made user-configurable, this branch WILL be reached
            // and must return Err, not 0 — the correct value is +∞.
            debug_assert!(
                false,
                "gram_entry: improper integral t^(-1) encountered (a={a}, b={b}); \
                 penalty result would be wrong if lfd_order < 2"
            );
            return 0.0;
        }
        ci * cj * (b.ln() - a.ln())
    } else {
        ci * cj * (b.powf(p) - a.powf(p)) / p
    }
}

/// Build the `nbasis × nbasis` analytic penalty matrix (column-major).
fn monomial_penalty_analytic(nbasis: usize, lfd_order: usize, a: f64, b: f64) -> Vec<f64> {
    let mut penalty = vec![0.0_f64; nbasis * nbasis];
    for j in 0..nbasis {
        for k in j..nbasis {
            let val = gram_entry(j as f64, k as f64, lfd_order, a, b);
            penalty[j + k * nbasis] = val;
            penalty[k + j * nbasis] = val; // symmetry
        }
    }
    penalty
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// monomial_basis returns FdarError for argvals.len() < 2.
    #[test]
    fn monomial_invalid_argvals_too_short() {
        let result = monomial_basis(&[0.5], 2);
        assert!(matches!(result, Err(FdarError::InvalidDimension { .. })));
    }

    /// monomial_basis returns FdarError for nbasis == 0.
    #[test]
    fn monomial_invalid_nbasis_zero() {
        let result = monomial_basis(&[0.0, 1.0], 0);
        assert!(matches!(result, Err(FdarError::InvalidParameter { .. })));
    }

    /// Closed-form eval: t=[0,1,2], nbasis=3 → cols [1,1,1], [0,1,2], [0,1,4].
    ///
    /// Column-major layout: element (ti, j) is at index ti + j * n_eval.
    #[test]
    fn monomial_eval_matrix_closed_form() {
        let t = vec![0.0, 1.0, 2.0];
        let bs = monomial_basis(&t, 3).unwrap();
        let n = bs.n_eval; // == 3
                           // col 0: B₀(t) = 1 — indices 0, 1, 2
        assert!((bs.eval_matrix[0] - 1.0).abs() < 1e-12, "B₀(0)");
        assert!((bs.eval_matrix[1] - 1.0).abs() < 1e-12, "B₀(1)");
        assert!((bs.eval_matrix[2] - 1.0).abs() < 1e-12, "B₀(2)");
        // col 1: B₁(t) = t — indices n, n+1, n+2
        assert!((bs.eval_matrix[n] - 0.0).abs() < 1e-12, "B₁(0)");
        assert!((bs.eval_matrix[n + 1] - 1.0).abs() < 1e-12, "B₁(1)");
        assert!((bs.eval_matrix[n + 2] - 2.0).abs() < 1e-12, "B₁(2)");
        // col 2: B₂(t) = t² — indices 2*n, 2*n+1, 2*n+2
        assert!((bs.eval_matrix[2 * n] - 0.0).abs() < 1e-12, "B₂(0)");
        assert!((bs.eval_matrix[2 * n + 1] - 1.0).abs() < 1e-12, "B₂(1)");
        assert!((bs.eval_matrix[2 * n + 2] - 4.0).abs() < 1e-12, "B₂(2)");
    }

    /// eval_matrix shape invariant.
    #[test]
    fn monomial_eval_matrix_shape() {
        let t = vec![0.0, 0.5, 1.0];
        let bs = monomial_basis(&t, 4).unwrap();
        assert_eq!(bs.eval_matrix.len(), 3 * 4);
        assert_eq!(bs.penalty_matrix.len(), 4 * 4);
        assert_eq!(bs.nbasis, 4);
        assert_eq!(bs.n_eval, 3);
        assert_eq!(bs.lfd_order, 2);
    }

    /// Penalty P[2,2] == 4.0 for exponents [0,1,2], lfd_order=2, domain [0,1].
    #[test]
    fn monomial_penalty_p22_standard_domain() {
        let t = vec![0.0, 0.5, 1.0];
        let bs = monomial_basis(&t, 3).unwrap();
        let k = 3;
        let p22 = bs.penalty_matrix[2 + 2 * k];
        // c₂ = 2, c₂ = 2; power = 2+2-4+1 = 1; P = 2*2*(1^1-0^1)/1 = 4.0
        assert!((p22 - 4.0).abs() < 1e-9, "P[2,2] = {p22}, expected 4.0");
    }

    /// Penalty is symmetric.
    #[test]
    fn monomial_penalty_symmetry() {
        let t: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
        let bs = monomial_basis(&t, 5).unwrap();
        let k = 5;
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

    /// Penalty diagonal entries are non-negative (PSD check).
    #[test]
    fn monomial_penalty_diagonal_psd() {
        let t: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
        let bs = monomial_basis(&t, 6).unwrap();
        let k = 6;
        for j in 0..k {
            let diag = bs.penalty_matrix[j + j * k];
            assert!(diag >= -1e-10, "P[{j},{j}]={diag} is negative");
        }
    }

    /// P[0,0] == 0 and P[1,1] == 0 for lfd_order=2 (D²(1)=0, D²(t)=0).
    ///
    /// Column-major layout: P[j,k] at index j + k * nbasis.
    #[test]
    fn monomial_penalty_low_exponents_zero() {
        let t = vec![0.0, 0.5, 1.0];
        let bs = monomial_basis(&t, 3).unwrap();
        let k = bs.nbasis; // == 3
                           // P[0,0] at index 0 + 0*k = 0
        assert!(bs.penalty_matrix[0].abs() < 1e-12, "P[0,0] should be 0");
        // P[1,1] at index 1 + 1*k = 1 + k
        assert!(bs.penalty_matrix[1 + k].abs() < 1e-12, "P[1,1] should be 0");
    }

    /// BasisSystem derives Debug, Clone, PartialEq.
    #[test]
    fn basis_system_derives() {
        let t = vec![0.0, 1.0];
        let bs = monomial_basis(&t, 2).unwrap();
        let bs2 = bs.clone();
        assert_eq!(bs, bs2);
        let _ = format!("{bs:?}"); // Debug
    }
}
