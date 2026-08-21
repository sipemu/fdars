//! Polygonal basis: piecewise-linear hat functions.
//!
//! Each basis function `B_j(t)` is a "hat function" centred at `knots[j]`:
//! it equals 1 at its own knot, 0 at all other knots, and varies linearly
//! between adjacent knots.  The collection satisfies the **partition-of-unity**
//! property: `Σ_j B_j(t) = 1` for all `t` in the knot span.
//!
//! ## Evaluation matrix
//!
//! The evaluation matrix is column-major of shape `(n × nbasis)`:
//!
//! ```text
//! eval_matrix[i + j * n] = hat_j(argvals[i])
//! ```
//!
//! The hat-function formula for basis function `j` (0-indexed):
//!
//! ```text
//! B_j(t) = (t - knots[j-1]) / (knots[j] - knots[j-1])   if knots[j-1] ≤ t ≤ knots[j]
//!         + (knots[j+1] - t) / (knots[j+1] - knots[j])   if knots[j]   ≤ t ≤ knots[j+1]
//! ```
//!
//! Boundary knots (`j = 0` or `j = nbasis-1`) have only one ramp; elsewhere 0.
//!
//! ## Roughness penalty
//!
//! The **1st-order roughness penalty** (`lfd_order = 1`) is used because the
//! 2nd derivative of a piecewise-linear function is 0 almost everywhere
//! (D² = delta function at knots, measure zero).  Using `lfd_order ≥ 2` would
//! yield an all-zero penalty matrix and is documented as incorrect for this
//! basis family.
//!
//! The penalty matrix is computed via **numeric quadrature** (same Gram pattern
//! as `bspline_penalty_matrix`) to correctly handle non-uniform knot spacing.
//!
//! ## Validation
//!
//! Knots must be strictly increasing: `knots[i+1] > knots[i]` for all i.
//! Non-monotone or duplicate knots produce division-by-zero in the hat formulas
//! and are rejected with [`FdarError::InvalidParameter`].
//!
//! Reference: R `fda` package `create.polygonal.basis` (equivalent to B-splines of order 2).

use crate::basis::basis_system::BasisSystem;
use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::smooth_basis::{differentiate_basis_columns, integrate_symmetric_penalty};

// ─── Public factory ──────────────────────────────────────────────────────────

/// Construct a polygonal (piecewise-linear hat function) basis.
///
/// The number of basis functions equals `knots.len()`.  Each basis function
/// `B_j` is the hat function centred at `knots[j]`, satisfying:
/// - `B_j(knots[j]) = 1`
/// - `B_j(knots[k]) = 0` for `k ≠ j`
/// - Linear interpolation between knots
/// - Partition of unity: `Σ_j B_j(t) = 1` for all `t` in the knot span.
///
/// Returns a [`BasisSystem`] with `lfd_order = 1` (1st-order roughness penalty).
///
/// # Arguments
///
/// * `argvals` — Evaluation points (length ≥ 2).
/// * `knots`   — Strictly-increasing knot sequence (length ≥ 2).
///
/// # Errors
///
/// - [`FdarError::InvalidDimension`] if `argvals.len() < 2`.
/// - [`FdarError::InvalidParameter`] if `knots.len() < 2`.
/// - [`FdarError::InvalidParameter`] if any `knots[i+1] ≤ knots[i]`
///   (non-monotone or duplicate knots; would produce division-by-zero in hat formulas).
///
/// # Examples
///
/// ```
/// use fdars_core::polygonal_basis;
///
/// let knots = vec![0.0, 0.5, 1.0];
/// let argvals = vec![0.0, 0.25, 0.5, 0.75, 1.0];
/// let bs = polygonal_basis(&argvals, &knots).unwrap();
/// // Partition of unity at t=0.25
/// let n = bs.n_eval;
/// let nbasis = bs.nbasis;
/// let sum: f64 = (0..nbasis).map(|j| bs.eval_matrix[1 + j * n]).sum();
/// assert!((sum - 1.0).abs() < 1e-12, "partition-of-unity sum={sum}");
/// ```
pub fn polygonal_basis(argvals: &[f64], knots: &[f64]) -> Result<BasisSystem, FdarError> {
    let n = argvals.len();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: ">= 2".to_string(),
            actual: n.to_string(),
        });
    }
    let nbasis = knots.len();
    if nbasis < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "knots",
            message: "must have length >= 2".to_string(),
        });
    }
    // Validate strictly increasing knots (Pitfall 5)
    if !knots.windows(2).all(|w| w[1] > w[0]) {
        return Err(FdarError::InvalidParameter {
            parameter: "knots",
            message: "knots must be strictly increasing (no duplicate or out-of-order values)"
                .to_string(),
        });
    }

    // Build column-major evaluation matrix: eval_matrix[ti + j*n] = hat_j(t_i)
    let mut eval_matrix = vec![0.0_f64; n * nbasis];
    for (ti, &t) in argvals.iter().enumerate() {
        for j in 0..nbasis {
            eval_matrix[ti + j * n] = hat_function(t, knots, j);
        }
    }

    // 1st-order numeric Gram penalty (D^2 of piecewise-linear is 0 a.e.)
    let lfd_order = 1_usize;
    let penalty_matrix = polygonal_penalty_numeric(argvals, knots, nbasis, lfd_order);

    Ok(BasisSystem {
        eval_matrix,
        penalty_matrix,
        nbasis,
        n_eval: n,
        lfd_order,
    })
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Evaluate the j-th hat function at point `t`.
///
/// Uses a half-open interval convention to avoid double-counting at shared
/// knot boundaries:
///
/// - Left ramp:  `(t - knots[j-1]) / (knots[j] - knots[j-1])` on `[knots[j-1], knots[j])`
/// - Right ramp: `(knots[j+1] - t) / (knots[j+1] - knots[j])` on `[knots[j], knots[j+1]]`
///
/// At `t == knots[j]` only the right ramp is active (evaluates to 1.0).
/// For the last knot the right ramp includes the closed endpoint.
/// Boundary knots have only one ramp; the value is 0 outside the support.
fn hat_function(t: f64, knots: &[f64], j: usize) -> f64 {
    let nk = knots.len();
    // Left ramp: half-open [knots[j-1], knots[j]) — excludes the peak knot
    // so the right ramp owns it.  For the last knot (j == nk-1), use closed [knots[j-1], knots[j]].
    let at_last_knot = j == nk - 1;
    let left = if j > 0 {
        let in_left = if at_last_knot {
            t >= knots[j - 1] && t <= knots[j]
        } else {
            t >= knots[j - 1] && t < knots[j]
        };
        if in_left {
            (t - knots[j - 1]) / (knots[j] - knots[j - 1])
        } else {
            0.0
        }
    } else {
        0.0
    };
    // Right ramp: closed [knots[j], knots[j+1]]
    let right = if j + 1 < nk && t >= knots[j] && t <= knots[j + 1] {
        (knots[j + 1] - t) / (knots[j + 1] - knots[j])
    } else {
        0.0
    };
    left + right
}

/// Compute the numeric 1st-order Gram penalty for the polygonal basis.
///
/// Uses a fine quadrature grid (10 sub-points per original interval)
/// over the knot span.
fn polygonal_penalty_numeric(
    argvals: &[f64],
    knots: &[f64],
    nbasis: usize,
    lfd_order: usize,
) -> Vec<f64> {
    if argvals.len() < 2 {
        return vec![0.0; nbasis * nbasis];
    }
    // Use knot span for the penalty domain (avoids evaluating outside the hat support)
    let t_min = knots[0];
    let t_max = knots[knots.len() - 1];
    let n_sub = 10;
    let n_quad = (argvals.len() - 1) * n_sub + 1;

    let quad_t: Vec<f64> = (0..n_quad)
        .map(|i| t_min + (t_max - t_min) * i as f64 / (n_quad - 1) as f64)
        .collect();

    // Evaluate polygonal basis on fine grid
    let mut basis_fine = vec![0.0_f64; n_quad * nbasis];
    for (ti, &t) in quad_t.iter().enumerate() {
        for j in 0..nbasis {
            basis_fine[ti + j * n_quad] = hat_function(t, knots, j);
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
    fn poly_invalid_argvals_too_short() {
        let result = polygonal_basis(&[0.5], &[0.0, 0.5, 1.0]);
        assert!(matches!(result, Err(FdarError::InvalidDimension { .. })));
    }

    /// Returns FdarError::InvalidParameter for knots.len() < 2.
    #[test]
    fn poly_invalid_knots_too_few() {
        let result = polygonal_basis(&[0.0, 1.0], &[0.5]);
        assert!(matches!(result, Err(FdarError::InvalidParameter { .. })));
    }

    /// Returns FdarError::InvalidParameter for duplicate knots.
    #[test]
    fn poly_invalid_duplicate_knots() {
        let result = polygonal_basis(&[0.0, 0.5, 1.0], &[0.0, 0.5, 0.5]);
        assert!(matches!(result, Err(FdarError::InvalidParameter { .. })));
    }

    /// Returns FdarError::InvalidParameter for non-monotone knots.
    #[test]
    fn poly_invalid_nonmonotone_knots() {
        let result = polygonal_basis(&[0.0, 0.5, 1.0], &[0.0, 1.0, 0.5]);
        assert!(matches!(result, Err(FdarError::InvalidParameter { .. })));
    }

    /// Hat-peak test: B_j(knots[j]) == 1.0 for each j.
    #[test]
    fn poly_hat_peaks_at_knot() {
        let knots = vec![0.0, 0.5, 1.0];
        let argvals = knots.clone();
        let bs = polygonal_basis(&argvals, &knots).unwrap();
        let n = bs.n_eval;
        for j in 0..bs.nbasis {
            // argvals[j] == knots[j] in this test, so row j, col j
            let val = bs.eval_matrix[j + j * n];
            assert!(
                (val - 1.0).abs() < 1e-12,
                "B_{j}(knots[{j}])={val}, expected 1.0"
            );
        }
    }

    /// At t=0.25 with knots [0,0.5,1]: B₀=0.5, B₁=0.5, B₂=0.0.
    #[test]
    fn poly_midpoint_values() {
        let knots = vec![0.0, 0.5, 1.0];
        let argvals = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let bs = polygonal_basis(&argvals, &knots).unwrap();
        let n = bs.n_eval; // == 5
        // ti=1 is t=0.25
        let b0 = bs.eval_matrix[1]; // j=0: index 1 + 0*5 = 1
        let b1 = bs.eval_matrix[1 + n]; // j=1: index 1 + 1*5 = 6
        let b2 = bs.eval_matrix[1 + 2 * n]; // j=2: index 1 + 2*5 = 11
        assert!((b0 - 0.5).abs() < 1e-12, "B₀(0.25)={b0}");
        assert!((b1 - 0.5).abs() < 1e-12, "B₁(0.25)={b1}");
        assert!(b2.abs() < 1e-12, "B₂(0.25)={b2}");
    }

    /// At t=0.5 with knots [0,0.5,1]: B₀=0, B₁=1, B₂=0.
    #[test]
    fn poly_eval_at_interior_knot() {
        let knots = vec![0.0, 0.5, 1.0];
        let argvals = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let bs = polygonal_basis(&argvals, &knots).unwrap();
        let n = bs.n_eval;
        // ti=2 is t=0.5
        let b0 = bs.eval_matrix[2];
        let b1 = bs.eval_matrix[2 + n];
        let b2 = bs.eval_matrix[2 + 2 * n];
        assert!(b0.abs() < 1e-12, "B₀(0.5)={b0}");
        assert!((b1 - 1.0).abs() < 1e-12, "B₁(0.5)={b1}");
        assert!(b2.abs() < 1e-12, "B₂(0.5)={b2}");
    }

    /// Partition of unity: Σ_j B_j(t) == 1.0 (within 1e-12) at each argval in the knot span.
    #[test]
    fn poly_partition_of_unity() {
        let knots = vec![0.0, 0.5, 1.0];
        let argvals: Vec<f64> = (0..=20).map(|i| i as f64 / 20.0).collect();
        let bs = polygonal_basis(&argvals, &knots).unwrap();
        let n = bs.n_eval;
        for ti in 0..n {
            let sum: f64 = (0..bs.nbasis)
                .map(|j| bs.eval_matrix[ti + j * n])
                .sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "partition-of-unity violated at ti={ti}: sum={sum}"
            );
        }
    }

    /// eval_matrix and penalty_matrix have correct shapes.
    #[test]
    fn poly_shape_invariants() {
        let knots = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let argvals: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
        let bs = polygonal_basis(&argvals, &knots).unwrap();
        assert_eq!(bs.eval_matrix.len(), 11 * 5);
        assert_eq!(bs.penalty_matrix.len(), 5 * 5);
        assert_eq!(bs.nbasis, 5);
        assert_eq!(bs.n_eval, 11);
        assert_eq!(bs.lfd_order, 1);
    }

    /// Penalty matrix is symmetric.
    #[test]
    fn poly_penalty_symmetric() {
        let knots = vec![0.0, 0.5, 1.0];
        let argvals: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
        let bs = polygonal_basis(&argvals, &knots).unwrap();
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
    fn poly_penalty_diagonal_nonneg() {
        let knots = vec![0.0, 0.5, 1.0];
        let argvals: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
        let bs = polygonal_basis(&argvals, &knots).unwrap();
        let k = bs.nbasis;
        for j in 0..k {
            let diag = bs.penalty_matrix[j + j * k];
            assert!(diag >= -1e-10, "P[{j},{j}]={diag} is negative");
        }
    }

    /// BasisSystem derives Debug, Clone, PartialEq.
    #[test]
    fn poly_basis_system_derives() {
        let knots = vec![0.0, 1.0];
        let bs = polygonal_basis(&knots, &knots).unwrap();
        let bs2 = bs.clone();
        assert_eq!(bs, bs2);
        let _ = format!("{bs:?}");
    }
}
