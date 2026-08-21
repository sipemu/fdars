//! Linear differential operators and principal differential analysis.
//!
//! This module provides two complementary tools for working with linear ordinary
//! differential equations (ODEs) in functional data analysis:
//!
//! - [`Lfd`]: A linear differential operator that can be *applied* to a set of
//!   observed curves, forming `Lx = D^m x + β_{m-1}(t)·D^{m-1}x + … + β₀(t)·x`.
//! - [`principal_differential_analysis`]: Estimates the coefficient functions of a
//!   linear ODE from a collection of observed solution curves.
//!
//! # Relationship to the R `fda` package
//!
//! The [`Lfd`] struct corresponds to the `Lfd` object in the R `fda` package
//! (see <https://rdrr.io/cran/fda/man/Lfd.html>).  The PDA estimator corresponds
//! to `pda.fd` and recovers the weight functions β₀(t), …, β_{m-1}(t) of the
//! order-*m* ODE by independent least squares at each grid point
//! (see <https://arxiv.org/abs/2406.18484>).
//!
//! # Examples
//!
//! ```
//! use fdars_core::pda::{Lfd, PdaResult, principal_differential_analysis};
//! use fdars_core::matrix::FdMatrix;
//! use std::f64::consts::PI;
//!
//! // Build a harmonic-oscillator dataset: x_i(t) = cos(2π t)
//! let omega = 2.0 * PI;
//! let n_pts = 101;
//! let argvals: Vec<f64> = (0..n_pts).map(|i| i as f64 / (n_pts - 1) as f64).collect();
//! let n_curves = 5;
//! let mut data = FdMatrix::zeros(n_curves, n_pts);
//! for i in 0..n_curves {
//!     let a = (i + 1) as f64;
//!     for (j, &t) in argvals.iter().enumerate() {
//!         data[(i, j)] = a * (omega * t).cos();
//!     }
//! }
//!
//! // Apply the identity-like Lfd (m=1, β₀ = 0) to verify shape invariance.
//! let lfd = Lfd { coefs: vec![vec![0.0]] };
//! let lx = lfd.apply(&data, &argvals).unwrap();
//! assert_eq!(lx.shape(), data.shape());
//! ```

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use nalgebra::DMatrix;

// ─── Lfd ────────────────────────────────────────────────────────────────────

/// A linear differential operator of the form
/// `Lx(t) = D^m x(t) + β_{m-1}(t)·D^{m-1}x(t) + … + β₀(t)·x(t)`.
///
/// The operator holds `m` weight functions `β₀, …, β_{m-1}` (where `m` is the
/// *order* of `L`).  Each weight function is sampled on the same evaluation
/// grid that is later supplied to [`Lfd::apply`].
///
/// # Constant-coefficient operators
///
/// A length-1 inner `Vec<f64>` in `coefs` is treated as a constant and broadcast
/// to all grid points.  For example, `coefs = vec![vec![-9.87]]` represents the
/// operator `Lx = Dx - 9.87·x` (order 1, constant negative spring constant).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Lfd {
    /// Weight functions β₀(t), …, β_{m-1}(t), each sampled on the evaluation grid.
    ///
    /// `coefs.len()` equals the operator order *m*.
    /// `coefs[k]` must have length either 1 (constant, broadcast) or `n_pts`
    /// (grid-sampled), where `n_pts = argvals.len()` when [`apply`](Lfd::apply) is
    /// called.
    pub coefs: Vec<Vec<f64>>,
}

impl Lfd {
    /// Apply the operator to each curve in `data`.
    ///
    /// For a curve `xᵢ`, computes the scalar sequence
    /// `Lxᵢ(t_j) = D^m xᵢ(t_j) + Σ_{k=0}^{m-1} βₖ(t_j) · D^k xᵢ(t_j)`.
    ///
    /// Derivatives are estimated via the iterated finite-difference scheme in
    /// [`crate::helpers::gradient`] (5-point stencil on uniform grids, 3-point
    /// Lagrange on non-uniform grids).
    ///
    /// # Arguments
    ///
    /// * `data`    — Functional data matrix `(n × n_pts)`; each row is one curve.
    /// * `argvals` — Evaluation points (length `n_pts`, must be sorted ascending).
    ///
    /// # Errors
    ///
    /// * [`FdarError::InvalidDimension`] if `argvals.len() != data.ncols()`.
    /// * [`FdarError::InvalidDimension`] if any `coefs[k]` has a length other than
    ///   `1` or `n_pts`.
    ///
    /// # Returns
    ///
    /// An `FdMatrix` of the same shape as `data`, containing the operator output.
    pub fn apply(&self, data: &FdMatrix, argvals: &[f64]) -> Result<FdMatrix, FdarError> {
        let (n, n_pts) = data.shape();
        let m = self.coefs.len(); // operator order

        // Validate argvals length matches data columns.
        if argvals.len() != n_pts {
            return Err(FdarError::InvalidDimension {
                parameter: "argvals",
                expected: n_pts.to_string(),
                actual: argvals.len().to_string(),
            });
        }

        // Validate coefs[k] lengths: each must be 1 (constant) or n_pts (grid-sampled).
        for (k, coef) in self.coefs.iter().enumerate() {
            if coef.len() != 1 && coef.len() != n_pts {
                return Err(FdarError::InvalidDimension {
                    parameter: "coefs[k]",
                    expected: format!("1 or {n_pts}"),
                    actual: format!("coefs[{k}].len() = {}", coef.len()),
                });
            }
        }

        let mut out = FdMatrix::zeros(n, n_pts);

        for i in 0..n {
            // Extract curve i as a Vec<f64>.
            let mut derivs: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
            let curve: Vec<f64> = (0..n_pts).map(|j| data[(i, j)]).collect();
            derivs.push(curve);

            // Compute D¹x, D²x, …, D^m x by iterating gradient.
            for _ in 0..m {
                let prev = derivs.last().unwrap();
                let d = crate::helpers::gradient(prev, argvals);
                derivs.push(d);
            }

            // Lx_i(t_j) = D^m x_i(t_j) + Σ_{k=0}^{m-1} β_k(t_j) · D^k x_i(t_j)
            for j in 0..n_pts {
                let mut lx_j = derivs[m][j];
                for k in 0..m {
                    // Broadcast length-1 coefficients (Pitfall 3).
                    let beta_k = if self.coefs[k].len() == 1 {
                        self.coefs[k][0]
                    } else {
                        self.coefs[k][j]
                    };
                    lx_j += beta_k * derivs[k][j];
                }
                // Write into column-major output: element (i, j) at i + j*n.
                out[(i, j)] = lx_j;
            }
        }

        Ok(out)
    }
}

// ─── PdaResult ───────────────────────────────────────────────────────────────

/// Result of [`principal_differential_analysis`].
///
/// Holds the recovered pointwise coefficient functions β₀(t), …, β_{m-1}(t)
/// of the order-`m` linear ODE:
///
/// `D^m x(t) = -β₀(t)·x(t) - β₁(t)·Dx(t) - … - β_{m-1}(t)·D^{m-1}x(t)`.
///
/// # Fields
///
/// * `coefficients` — Length-`order` outer Vec; `coefficients[k]` is β_k(t) sampled
///   at `argvals` (length `n_pts`).
/// * `order`        — ODE order *m*.
/// * `residuals`    — Optional residual matrix (currently always `None`; can be
///   computed from [`Lfd::apply`] with the recovered coefficients).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PdaResult {
    /// Recovered coefficient functions.
    ///
    /// `coefficients[k]` is β_k(t) sampled at the evaluation grid supplied to
    /// [`principal_differential_analysis`].  Length of outer Vec is `order`;
    /// each inner Vec has length `n_pts`.
    pub coefficients: Vec<Vec<f64>>,

    /// Order of the estimated ODE.
    pub order: usize,

    /// Optional residuals matrix (not computed by default; kept for extensibility).
    pub residuals: Option<FdMatrix>,
}

// ─── principal_differential_analysis ────────────────────────────────────────

/// Estimate the coefficient functions of a linear ODE from observed solution curves.
///
/// Given `n` curves on a common grid of `n_pts` points, PDA estimates the weight
/// functions β₀(t), …, β_{m-1}(t) such that
///
/// `D^m x_i(t) ≈ -β₀(t)·x_i(t) - β₁(t)·Dx_i(t) - … - β_{m-1}(t)·D^{m-1}x_i(t)`.
///
/// At each grid point `t_j` an independent ordinary least-squares problem is solved:
///
/// ```text
/// X_j = [x_i(t_j), Dx_i(t_j), …, D^{m-1}x_i(t_j)]  (n × m design matrix)
/// y_j = -[D^m x_i(t_j)]                               (n-vector)
/// β(t_j) = pinv(X_j) y_j   via SVD pseudoinverse
/// ```
///
/// # Arguments
///
/// * `data`    — Functional data matrix `(n × n_pts)`; each row is one solution curve.
/// * `argvals` — Evaluation grid (length `n_pts`, must be sorted ascending).
/// * `order`   — ODE order *m* (`≥ 1`).
///
/// # Errors
///
/// * [`FdarError::InvalidDimension`] if `argvals.len() != data.ncols()`.
/// * [`FdarError::InvalidParameter`] if `order == 0`.
/// * [`FdarError::InvalidDimension`] if `n_curves < order + 1` (underdetermined system).
///
/// # Returns
///
/// A [`PdaResult`] containing the recovered coefficient functions.
///
/// # Notes on singular designs
///
/// When the pointwise design matrix `X_j` is rank-deficient (e.g. due to nearly
/// identical curves or ill-posed boundary grid points), the SVD pseudoinverse
/// threshold `1e-10 · max_singular_value` is applied, yielding zero coefficients
/// for the degenerate directions rather than `NaN`/panic.
pub fn principal_differential_analysis(
    data: &FdMatrix,
    argvals: &[f64],
    order: usize,
) -> Result<PdaResult, FdarError> {
    let (n, n_pts) = data.shape();

    // Validate order.
    if order == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "order",
            message: "must be >= 1".to_string(),
        });
    }

    // Validate argvals length.
    if argvals.len() != n_pts {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: n_pts.to_string(),
            actual: argvals.len().to_string(),
        });
    }

    // Guard: Pitfall 4 — too few curves for a well-posed pointwise regression.
    if n < order + 1 {
        return Err(FdarError::InvalidDimension {
            parameter: "data (n_curves)",
            expected: format!(">= order + 1 = {}", order + 1),
            actual: n.to_string(),
        });
    }

    // Compute derivative FdMatrices D⁰x, D¹x, …, D^{order}x.
    // derivs[k] is the k-th derivative of all curves, same shape as data.
    let mut derivs: Vec<FdMatrix> = Vec::with_capacity(order + 1);
    derivs.push(data.clone()); // D⁰x = x

    for _ in 0..order {
        let prev = derivs.last().unwrap();
        let mut next = FdMatrix::zeros(n, n_pts);
        for i in 0..n {
            let row: Vec<f64> = (0..n_pts).map(|j| prev[(i, j)]).collect();
            let grad = crate::helpers::gradient(&row, argvals);
            for j in 0..n_pts {
                next[(i, j)] = grad[j];
            }
        }
        derivs.push(next);
    }

    // Initialize coefficient storage: coefficients[k] = β_k(t), length n_pts.
    let mut coefficients: Vec<Vec<f64>> = vec![vec![0.0; n_pts]; order];

    // At each grid point t_j, solve the pointwise least-squares system.
    for j in 0..n_pts {
        // Build n × order design matrix X_j: column k = D^k x at t_j, for k=0..order-1.
        let mut x_j = DMatrix::<f64>::zeros(n, order);
        for i in 0..n {
            for k in 0..order {
                x_j[(i, k)] = derivs[k][(i, j)];
            }
        }

        // Target y_j = -(D^order x) at t_j.
        let y_j: Vec<f64> = (0..n).map(|i| -derivs[order][(i, j)]).collect();
        let y_vec = nalgebra::DVector::from_vec(y_j);

        // Solve via SVD pseudoinverse: β = pinv(X_j) · y_j.
        let svd = nalgebra::SVD::new(x_j, true, true);
        let max_sv = svd
            .singular_values
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        let threshold = 1e-10 * max_sv;

        // Compute the pseudoinverse action: pinv(X) y = V · diag(1/σ) · U^T · y.
        if let (Some(u), Some(v_t)) = (svd.u.as_ref(), svd.v_t.as_ref()) {
            // u_t_y = U^T · y  (shape: order × 1, using only relevant rows)
            let u_t_y: Vec<f64> = (0..order.min(svd.singular_values.len()))
                .map(|s| (0..n).map(|i| u[(i, s)] * y_vec[i]).sum::<f64>())
                .collect();

            // Apply 1/σ filter and accumulate via V^T rows.
            let mut beta_j = vec![0.0_f64; order];
            for k in 0..order {
                let mut val = 0.0_f64;
                for s in 0..order.min(svd.singular_values.len()) {
                    if svd.singular_values[s] > threshold {
                        // v_t[(s, k)] is the (s, k) entry of V^T.
                        val += v_t[(s, k)] * u_t_y[s] / svd.singular_values[s];
                    }
                }
                beta_j[k] = val;
            }

            for k in 0..order {
                coefficients[k][j] = beta_j[k];
            }
        }
        // If SVD factorization failed entirely (u or v_t missing), coefficients
        // remain zero — a conservative fallback that avoids NaN/panic.
    }

    Ok(PdaResult {
        coefficients,
        order,
        residuals: None,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── Lfd tests ────────────────────────────────────────────────────────────

    /// A constant operator (m=1, β₀ = c) applied to a constant curve:
    /// D¹(const) = 0, so Lx(t) = 0 + c · const = c · const.
    #[test]
    fn lfd_constant_operator_on_constant_curve() {
        let n_pts = 11;
        let argvals: Vec<f64> = (0..n_pts).map(|i| i as f64 / (n_pts - 1) as f64).collect();
        let curve_val = 3.0_f64;
        let c = 2.0_f64;

        // One curve, all values = curve_val.
        let mut data = FdMatrix::zeros(1, n_pts);
        for j in 0..n_pts {
            data[(0, j)] = curve_val;
        }

        let lfd = Lfd {
            coefs: vec![vec![c]], // constant β₀ = c, broadcast
        };
        let lx = lfd.apply(&data, &argvals).unwrap();

        assert_eq!(lx.shape(), (1, n_pts));
        // D¹ const ≈ 0, so Lx ≈ c * curve_val everywhere.
        let expected = c * curve_val;
        for j in 0..n_pts {
            assert!(
                (lx[(0, j)] - expected).abs() < 1e-6,
                "lx[{j}] = {} but expected {} (diff = {})",
                lx[(0, j)],
                expected,
                (lx[(0, j)] - expected).abs()
            );
        }
    }

    /// Mismatched argvals length returns Err(InvalidDimension).
    #[test]
    fn lfd_mismatched_argvals_returns_err() {
        let n_pts = 10;
        let _argvals: Vec<f64> = (0..n_pts).map(|i| i as f64).collect();
        let data = FdMatrix::zeros(2, n_pts);

        // Supply argvals of wrong length.
        let lfd = Lfd {
            coefs: vec![vec![1.0]],
        };
        let wrong_argvals: Vec<f64> = (0..n_pts + 3).map(|i| i as f64).collect();
        let result = lfd.apply(&data, &wrong_argvals);
        assert!(
            matches!(result, Err(FdarError::InvalidDimension { .. })),
            "expected InvalidDimension, got: {:?}",
            result
        );
    }

    /// A coefs[k] of a length other than 1 or n_pts returns InvalidDimension.
    #[test]
    fn lfd_bad_coefs_length_returns_err() {
        let n_pts = 10;
        let argvals: Vec<f64> = (0..n_pts).map(|i| i as f64).collect();
        let data = FdMatrix::zeros(2, n_pts);

        // coefs[0].len() = 5, which is neither 1 nor n_pts=10.
        let lfd = Lfd {
            coefs: vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]],
        };
        let result = lfd.apply(&data, &argvals);
        assert!(
            matches!(result, Err(FdarError::InvalidDimension { .. })),
            "expected InvalidDimension, got: {:?}",
            result
        );
    }

    /// apply returns FdMatrix of same shape as input data.
    #[test]
    fn lfd_apply_shape_preserved() {
        let n_pts = 20;
        let n_curves = 4;
        let argvals: Vec<f64> = (0..n_pts).map(|i| i as f64 / (n_pts - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n_curves, n_pts);
        for i in 0..n_curves {
            for j in 0..n_pts {
                data[(i, j)] = argvals[j].powi(2) + i as f64;
            }
        }

        // m=2 operator with constant coefficients.
        let lfd = Lfd {
            coefs: vec![vec![1.0], vec![0.5]],
        };
        let lx = lfd.apply(&data, &argvals).unwrap();
        assert_eq!(lx.shape(), (n_curves, n_pts));
    }

    // ── PDA tests ────────────────────────────────────────────────────────────

    /// Harmonic oscillator: x''(t) = -ω²x(t), ω = 2π.
    /// PDA with order=2 should recover β₀ ≈ ω² and β₁ ≈ 0.
    #[test]
    fn pda_recovers_harmonic_oscillator() {
        let omega = 2.0 * PI;
        let n_pts = 101;
        let argvals: Vec<f64> = (0..n_pts)
            .map(|i| i as f64 / (n_pts - 1) as f64)
            .collect();
        let n_curves = 20;

        // x_i(t) = A_i · cos(ω t) + B_i · sin(ω t), with varied A_i, B_i.
        let mut data = FdMatrix::zeros(n_curves, n_pts);
        for i in 0..n_curves {
            let a = (i + 1) as f64;
            let b = (i + 2) as f64;
            for (j, &t) in argvals.iter().enumerate() {
                data[(i, j)] = a * (omega * t).cos() + b * (omega * t).sin();
            }
        }

        let result = principal_differential_analysis(&data, &argvals, 2).unwrap();
        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(result.coefficients[0].len(), n_pts);

        let omega_sq = omega * omega; // ≈ 39.478
        let tolerance = 1.0;

        for (j, &beta0_j) in result.coefficients[0].iter().enumerate() {
            assert!(
                (beta0_j - omega_sq).abs() < tolerance,
                "β₀[{j}] = {beta0_j}, expected ≈ {omega_sq}, diff = {}",
                (beta0_j - omega_sq).abs()
            );
        }
        for (j, &beta1_j) in result.coefficients[1].iter().enumerate() {
            assert!(
                beta1_j.abs() < tolerance,
                "β₁[{j}] = {beta1_j}, expected ≈ 0"
            );
        }
    }

    /// n < order+1 returns Err(InvalidDimension).
    #[test]
    fn pda_too_few_curves_returns_err() {
        // order=2 requires n >= 3; supply only 2 curves.
        let n_pts = 51;
        let argvals: Vec<f64> = (0..n_pts).map(|i| i as f64 / (n_pts - 1) as f64).collect();
        let data = FdMatrix::zeros(2, n_pts); // 2 curves, order=2 needs ≥3

        let result = principal_differential_analysis(&data, &argvals, 2);
        assert!(
            matches!(result, Err(FdarError::InvalidDimension { .. })),
            "expected InvalidDimension, got: {:?}",
            result
        );
    }

    /// Mismatched argvals length returns FdarError.
    #[test]
    fn pda_mismatched_argvals_returns_err() {
        let n_pts = 51;
        let _argvals: Vec<f64> = (0..n_pts).map(|i| i as f64 / (n_pts - 1) as f64).collect();
        let data = FdMatrix::zeros(5, n_pts);

        // Supply argvals of wrong length.
        let wrong_argvals: Vec<f64> = (0..n_pts + 5).map(|i| i as f64).collect();
        let result = principal_differential_analysis(&data, &wrong_argvals, 2);
        assert!(
            matches!(result, Err(FdarError::InvalidDimension { .. })),
            "expected InvalidDimension, got: {:?}",
            result
        );
    }

    /// order=0 returns Err(InvalidParameter).
    #[test]
    fn pda_zero_order_returns_err() {
        let n_pts = 51;
        let argvals: Vec<f64> = (0..n_pts).map(|i| i as f64 / (n_pts - 1) as f64).collect();
        let data = FdMatrix::zeros(5, n_pts);

        let result = principal_differential_analysis(&data, &argvals, 0);
        assert!(
            matches!(result, Err(FdarError::InvalidParameter { .. })),
            "expected InvalidParameter, got: {:?}",
            result
        );
    }

    /// PdaResult struct invariants: coefficients.len()==order, each inner len==n_pts.
    #[test]
    fn pda_result_shape_invariants() {
        let omega = 2.0 * PI;
        let n_pts = 51;
        let argvals: Vec<f64> = (0..n_pts).map(|i| i as f64 / (n_pts - 1) as f64).collect();
        let n_curves = 10;
        let order = 2;

        let mut data = FdMatrix::zeros(n_curves, n_pts);
        for i in 0..n_curves {
            let a = (i + 1) as f64;
            for (j, &t) in argvals.iter().enumerate() {
                data[(i, j)] = a * (omega * t).cos();
            }
        }

        let result = principal_differential_analysis(&data, &argvals, order).unwrap();
        assert_eq!(result.order, order);
        assert_eq!(result.coefficients.len(), order);
        for k in 0..order {
            assert_eq!(
                result.coefficients[k].len(),
                n_pts,
                "coefficients[{k}] should have length n_pts={n_pts}"
            );
        }
        // residuals default is None.
        assert!(result.residuals.is_none());
    }
}
