//! Functional scoring metrics — MAE, MSE, MAPE, MSLE, explained variance.
//!
//! All metrics integrate the pointwise error function over `argvals` using
//! Simpson's rule, producing a single scalar score per metric. Each metric
//! averages over all curves (rows of the input matrices).
//!
//! # Shape Contract (all five functions)
//!
//! - `y_true.shape() == y_pred.shape()` — else `InvalidDimension { parameter: "y_pred" }`
//! - `argvals.len() == y_true.ncols()` — else `InvalidDimension { parameter: "argvals" }`
//! - `y_true.nrows() >= 1` and `argvals.len() >= 2` — else `InvalidDimension { parameter: "data" }`

use crate::helpers::{simpsons_weights, NUMERICAL_EPS};
use crate::matrix::FdMatrix;
use crate::FdarError;

/// Validate that `y_true`, `y_pred`, and `argvals` have consistent shapes.
///
/// Returns `(n, m)` — number of curves and evaluation points — on success.
fn validate_shapes(
    y_true: &FdMatrix,
    y_pred: &FdMatrix,
    argvals: &[f64],
) -> Result<(usize, usize), FdarError> {
    let (n, m) = y_true.shape();
    if y_pred.shape() != (n, m) {
        return Err(FdarError::InvalidDimension {
            parameter: "y_pred",
            expected: format!("({n}, {m})"),
            actual: format!("{:?}", y_pred.shape()),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if n == 0 || m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n >= 1 and m >= 2".to_string(),
            actual: format!("n={n}, m={m}"),
        });
    }
    Ok((n, m))
}

/// Functional Mean Absolute Error integrated over `argvals`.
///
/// Computes `functional_mae = (1/n) * sum_i ∫ |y_true_i(t) - y_pred_i(t)| dt`
/// where the integral is approximated by Simpson's rule over `argvals`.
///
/// # Errors
///
/// - [`FdarError::InvalidDimension`] if shapes of `y_true`, `y_pred`, or `argvals`
///   are inconsistent or `n < 1` / `m < 2`.
pub fn functional_mae(
    y_true: &FdMatrix,
    y_pred: &FdMatrix,
    argvals: &[f64],
) -> Result<f64, FdarError> {
    let (n, m) = validate_shapes(y_true, y_pred, argvals)?;
    let weights = simpsons_weights(argvals);
    let mut total = 0.0_f64;
    for i in 0..n {
        for j in 0..m {
            total += (y_true[(i, j)] - y_pred[(i, j)]).abs() * weights[j];
        }
    }
    Ok(total / n as f64)
}

/// Functional Mean Squared Error integrated over `argvals`.
///
/// Computes `functional_mse = (1/n) * sum_i ∫ (y_true_i(t) - y_pred_i(t))^2 dt`
/// where the integral is approximated by Simpson's rule over `argvals`.
///
/// # Errors
///
/// - [`FdarError::InvalidDimension`] if shapes of `y_true`, `y_pred`, or `argvals`
///   are inconsistent or `n < 1` / `m < 2`.
pub fn functional_mse(
    y_true: &FdMatrix,
    y_pred: &FdMatrix,
    argvals: &[f64],
) -> Result<f64, FdarError> {
    let (n, m) = validate_shapes(y_true, y_pred, argvals)?;
    let weights = simpsons_weights(argvals);
    let mut total = 0.0_f64;
    for i in 0..n {
        for j in 0..m {
            let diff = y_true[(i, j)] - y_pred[(i, j)];
            total += diff * diff * weights[j];
        }
    }
    Ok(total / n as f64)
}

/// Functional Mean Absolute Percentage Error integrated over `argvals`.
///
/// Computes `functional_mape = (1/n) * sum_i ∫ |y_true_i(t) - y_pred_i(t)| / |y_true_i(t)| dt`
/// where the integral is approximated by Simpson's rule over `argvals`.
///
/// # Errors
///
/// - [`FdarError::InvalidDimension`] if shapes are inconsistent.
/// - [`FdarError::InvalidParameter`] if any value of `y_true` is near zero
///   (`|y_true| < NUMERICAL_EPS`), which would cause division by zero.
pub fn functional_mape(
    y_true: &FdMatrix,
    y_pred: &FdMatrix,
    argvals: &[f64],
) -> Result<f64, FdarError> {
    let (n, m) = validate_shapes(y_true, y_pred, argvals)?;
    // Pre-scan for near-zero denominators before computing
    for i in 0..n {
        for j in 0..m {
            if y_true[(i, j)].abs() < NUMERICAL_EPS {
                return Err(FdarError::InvalidParameter {
                    parameter: "y_true",
                    message: format!(
                        "MAPE is undefined when y_true contains values near zero \
                         (found |y_true[{i},{j}]| = {} < NUMERICAL_EPS)",
                        y_true[(i, j)].abs()
                    ),
                });
            }
        }
    }
    let weights = simpsons_weights(argvals);
    let mut total = 0.0_f64;
    for i in 0..n {
        for j in 0..m {
            let pct_err = (y_true[(i, j)] - y_pred[(i, j)]).abs() / y_true[(i, j)].abs();
            total += pct_err * weights[j];
        }
    }
    Ok(total / n as f64)
}

/// Functional Mean Squared Logarithmic Error integrated over `argvals`.
///
/// Computes `functional_msle = (1/n) * sum_i ∫ (ln(1+y_true_i(t)) - ln(1+y_pred_i(t)))^2 dt`
/// where the integral is approximated by Simpson's rule over `argvals`.
///
/// MSLE is designed for non-negative targets (e.g. counts, prices). Applying it to
/// values below -1 yields undefined logarithms.
///
/// # Errors
///
/// - [`FdarError::InvalidDimension`] if shapes are inconsistent.
/// - [`FdarError::InvalidParameter`] if any value of `y_true` or `y_pred` is `<= -1`
///   (making `ln(1+x)` undefined).
pub fn functional_msle(
    y_true: &FdMatrix,
    y_pred: &FdMatrix,
    argvals: &[f64],
) -> Result<f64, FdarError> {
    let (n, m) = validate_shapes(y_true, y_pred, argvals)?;
    // Pre-scan for domain violations: y > -1 required for ln_1p
    let threshold = -1.0 + NUMERICAL_EPS;
    for i in 0..n {
        for j in 0..m {
            if y_true[(i, j)] <= threshold {
                return Err(FdarError::InvalidParameter {
                    parameter: "y_true",
                    message: format!(
                        "MSLE requires y_true > -1; found y_true[{i},{j}] = {}",
                        y_true[(i, j)]
                    ),
                });
            }
            if y_pred[(i, j)] <= threshold {
                return Err(FdarError::InvalidParameter {
                    parameter: "y_pred",
                    message: format!(
                        "MSLE requires y_pred > -1; found y_pred[{i},{j}] = {}",
                        y_pred[(i, j)]
                    ),
                });
            }
        }
    }
    let weights = simpsons_weights(argvals);
    let mut total = 0.0_f64;
    for i in 0..n {
        for j in 0..m {
            let log_diff = f64::ln_1p(y_true[(i, j)]) - f64::ln_1p(y_pred[(i, j)]);
            total += log_diff.powi(2) * weights[j];
        }
    }
    Ok(total / n as f64)
}

/// Functional Explained Variance Score integrated over `argvals`.
///
/// Computes the explained variance per curve as:
/// `EV_i = 1 - SS_res_i / SS_tot_i`
/// where:
/// - `SS_res_i = ∫ (residual_i(t) - mean_residual_i)^2 dt`
/// - `SS_tot_i = ∫ (y_true_i(t) - mean_true_i)^2 dt`
/// - Means are computed as the weighted integral divided by the domain length.
///
/// The score is then averaged over all curves. Returns 1.0 for perfect prediction,
/// 0.0 when prediction equals the mean of `y_true`, and can be negative for
/// predictions worse than the mean baseline.
///
/// When `SS_tot_i ≈ 0` (constant true curve) and `SS_res_i ≈ 0` (perfect fit),
/// returns 1.0 for that curve (trivial perfect prediction).
/// When `SS_tot_i ≈ 0` but `SS_res_i > 0`, returns 0.0 (prediction adds no value
/// over a constant).
///
/// # Errors
///
/// - [`FdarError::InvalidDimension`] if shapes are inconsistent.
pub fn functional_explained_variance(
    y_true: &FdMatrix,
    y_pred: &FdMatrix,
    argvals: &[f64],
) -> Result<f64, FdarError> {
    let (n, m) = validate_shapes(y_true, y_pred, argvals)?;
    let weights = simpsons_weights(argvals);
    let domain_len: f64 = weights.iter().sum();

    let mut ev_sum = 0.0_f64;
    for i in 0..n {
        // Compute residuals and weighted means
        let mut res_sum = 0.0_f64;
        let mut true_sum = 0.0_f64;
        for j in 0..m {
            let residual = y_true[(i, j)] - y_pred[(i, j)];
            res_sum += residual * weights[j];
            true_sum += y_true[(i, j)] * weights[j];
        }
        // Weighted means (integral / domain length)
        let mean_res = if domain_len > NUMERICAL_EPS {
            res_sum / domain_len
        } else {
            0.0
        };
        let mean_true = if domain_len > NUMERICAL_EPS {
            true_sum / domain_len
        } else {
            0.0
        };
        // Compute SS_res and SS_tot via Simpson weights
        let mut ss_res = 0.0_f64;
        let mut ss_tot = 0.0_f64;
        for j in 0..m {
            let res_centered = (y_true[(i, j)] - y_pred[(i, j)]) - mean_res;
            let true_centered = y_true[(i, j)] - mean_true;
            ss_res += res_centered.powi(2) * weights[j];
            ss_tot += true_centered.powi(2) * weights[j];
        }
        // Guard near-zero SS_tot (constant true curve).
        // CR-02: the old inner check `ss_res < NUMERICAL_EPS` was an absolute threshold
        // comparison that returned 1.0 even when ss_res > ss_tot (both below NUMERICAL_EPS).
        // Replace with a relative test: "perfect fit" only when the residual variance is
        // no larger than ss_tot up to a small relative tolerance.
        let ev_i = if ss_tot < NUMERICAL_EPS {
            if ss_res <= ss_tot * (1.0 + 1e-6) {
                1.0
            } else {
                0.0
            }
        } else {
            1.0 - ss_res / ss_tot
        };
        ev_sum += ev_i;
    }
    Ok(ev_sum / n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_helpers::uniform_grid;

    // Helper: create an FdMatrix from row-major input (for test convenience).
    fn mat_from_rows(rows: &[Vec<f64>]) -> FdMatrix {
        let n = rows.len();
        let m = rows[0].len();
        let mut col_major = vec![0.0_f64; n * m];
        for (i, row) in rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                col_major[i + j * n] = v;
            }
        }
        FdMatrix::from_column_major(col_major, n, m).unwrap()
    }

    // ------------------------------------------------------------------ MAE --

    #[test]
    fn test_functional_mae_constant_error() {
        // y_true = 0, y_pred = c => |error| = c everywhere
        // Integral over [0,1] (uniform 5-point grid) = c * 1.0
        // For 1 curve: mae = c.
        let c = 2.0_f64;
        let argvals = uniform_grid(5); // [0, 0.25, 0.5, 0.75, 1.0]
        let y_true = mat_from_rows(&[vec![0.0; 5]]);
        let y_pred = mat_from_rows(&[vec![c; 5]]);
        let mae = functional_mae(&y_true, &y_pred, &argvals).unwrap();
        // ∫ c dt over [0,1] = c * 1.0 = 2.0
        assert!((mae - c).abs() < 1e-10, "mae={mae}, expected {c}");
    }

    #[test]
    fn test_functional_mae_multi_curve() {
        // Two curves: errors c=1 and c=2. Average = 1.5
        let argvals = uniform_grid(5);
        let y_true = mat_from_rows(&[vec![0.0; 5], vec![0.0; 5]]);
        let y_pred = mat_from_rows(&[vec![1.0; 5], vec![2.0; 5]]);
        let mae = functional_mae(&y_true, &y_pred, &argvals).unwrap();
        // curve1 integral = 1.0, curve2 integral = 2.0, average = 1.5
        assert!((mae - 1.5).abs() < 1e-10, "mae={mae}, expected 1.5");
    }

    #[test]
    fn test_functional_mae_shape_mismatch_y_pred() {
        let argvals = uniform_grid(5);
        let y_true = mat_from_rows(&[vec![0.0; 5]]);
        let y_pred = mat_from_rows(&[vec![0.0; 4]]); // wrong ncols
        let result = functional_mae(&y_true, &y_pred, &argvals);
        assert!(matches!(
            result,
            Err(FdarError::InvalidDimension {
                parameter: "y_pred",
                ..
            })
        ));
    }

    #[test]
    fn test_functional_mae_shape_mismatch_argvals() {
        let argvals = uniform_grid(4); // len=4 but matrix has 5 cols
        let y_true = mat_from_rows(&[vec![0.0; 5]]);
        let y_pred = mat_from_rows(&[vec![0.0; 5]]);
        let result = functional_mae(&y_true, &y_pred, &argvals);
        assert!(matches!(
            result,
            Err(FdarError::InvalidDimension {
                parameter: "argvals",
                ..
            })
        ));
    }

    // ------------------------------------------------------------------ MSE --

    #[test]
    fn test_functional_mse_constant_error() {
        // y_true = 0, y_pred = c => error^2 = c^2 everywhere
        // Integral over [0,1] (5-point grid) = c^2
        let c = 3.0_f64;
        let argvals = uniform_grid(5);
        let y_true = mat_from_rows(&[vec![0.0; 5]]);
        let y_pred = mat_from_rows(&[vec![c; 5]]);
        let mse = functional_mse(&y_true, &y_pred, &argvals).unwrap();
        assert!((mse - c * c).abs() < 1e-10, "mse={mse}, expected {}", c * c);
    }

    #[test]
    fn test_functional_mse_zero_error() {
        // Perfect prediction => MSE = 0
        let argvals = uniform_grid(5);
        let y_true = mat_from_rows(&[vec![1.0, 2.0, 3.0, 4.0, 5.0]]);
        let y_pred = y_true.clone();
        let mse = functional_mse(&y_true, &y_pred, &argvals).unwrap();
        assert!(mse.abs() < 1e-14, "mse={mse}, expected 0");
    }

    // ---------------------------------------------------------------- MAPE --

    #[test]
    fn test_functional_mape_constant_error() {
        // y_true = 4.0, y_pred = 5.0 => |error|/|y_true| = 0.25 everywhere
        // Integral over [0,1] = 0.25, 1 curve => mape = 0.25
        let argvals = uniform_grid(5);
        let y_true = mat_from_rows(&[vec![4.0; 5]]);
        let y_pred = mat_from_rows(&[vec![5.0; 5]]);
        let mape = functional_mape(&y_true, &y_pred, &argvals).unwrap();
        assert!((mape - 0.25).abs() < 1e-10, "mape={mape}, expected 0.25");
    }

    #[test]
    fn test_functional_mape_zero_y_true() {
        // Should return Err(InvalidParameter) when y_true contains near-zero value
        let argvals = uniform_grid(5);
        let y_true = mat_from_rows(&[vec![0.0; 5]]);
        let y_pred = mat_from_rows(&[vec![1.0; 5]]);
        let result = functional_mape(&y_true, &y_pred, &argvals);
        assert!(matches!(
            result,
            Err(FdarError::InvalidParameter {
                parameter: "y_true",
                ..
            })
        ));
    }

    // ---------------------------------------------------------------- MSLE --

    #[test]
    fn test_functional_msle_constant() {
        // y_true = 1.0, y_pred = 1.0 => log_diff = 0 everywhere => msle = 0
        let argvals = uniform_grid(5);
        let y_true = mat_from_rows(&[vec![1.0; 5]]);
        let y_pred = y_true.clone();
        let msle = functional_msle(&y_true, &y_pred, &argvals).unwrap();
        assert!(msle.abs() < 1e-14, "msle={msle}, expected 0");
    }

    #[test]
    fn test_functional_msle_hand_computed() {
        // y_true = 3.0, y_pred = 1.0 over [0,1] with 5 pts
        // log_diff = ln(4) - ln(2) = ln(2) everywhere
        // integral of ln(2)^2 over [0,1] = ln(2)^2 ≈ 0.480453
        let argvals = uniform_grid(5);
        let y_true = mat_from_rows(&[vec![3.0; 5]]);
        let y_pred = mat_from_rows(&[vec![1.0; 5]]);
        let msle = functional_msle(&y_true, &y_pred, &argvals).unwrap();
        let expected = f64::ln(2.0).powi(2);
        assert!(
            (msle - expected).abs() < 1e-10,
            "msle={msle}, expected={expected}"
        );
    }

    #[test]
    fn test_functional_msle_domain_y_true() {
        // y_true = -1.5 should cause Err(InvalidParameter)
        let argvals = uniform_grid(5);
        let y_true = mat_from_rows(&[vec![-1.5; 5]]);
        let y_pred = mat_from_rows(&[vec![1.0; 5]]);
        let result = functional_msle(&y_true, &y_pred, &argvals);
        assert!(matches!(
            result,
            Err(FdarError::InvalidParameter {
                parameter: "y_true",
                ..
            })
        ));
    }

    #[test]
    fn test_functional_msle_domain_y_pred() {
        // y_pred = -2.0 should cause Err(InvalidParameter)
        let argvals = uniform_grid(5);
        let y_true = mat_from_rows(&[vec![1.0; 5]]);
        let y_pred = mat_from_rows(&[vec![-2.0; 5]]);
        let result = functional_msle(&y_true, &y_pred, &argvals);
        assert!(matches!(
            result,
            Err(FdarError::InvalidParameter {
                parameter: "y_pred",
                ..
            })
        ));
    }

    // -------------------------------------------------------- Explained Variance --

    #[test]
    fn test_explained_variance_perfect() {
        // Perfect prediction => EV = 1.0
        let argvals = uniform_grid(5);
        let y_true = mat_from_rows(&[vec![1.0, 2.0, 3.0, 4.0, 5.0]]);
        let y_pred = y_true.clone();
        let ev = functional_explained_variance(&y_true, &y_pred, &argvals).unwrap();
        assert!((ev - 1.0).abs() < 1e-10, "ev={ev}, expected 1.0");
    }

    #[test]
    fn test_explained_variance_constant_true() {
        // Constant y_true => SS_tot ≈ 0; if y_pred == y_true: EV = 1.0
        let argvals = uniform_grid(5);
        let y_true = mat_from_rows(&[vec![3.0; 5]]);
        let y_pred = y_true.clone();
        let ev = functional_explained_variance(&y_true, &y_pred, &argvals).unwrap();
        assert!((ev - 1.0).abs() < 1e-10, "ev={ev}, expected 1.0");
    }

    #[test]
    fn test_explained_variance_shape_mismatch() {
        let argvals = uniform_grid(5);
        let y_true = mat_from_rows(&[vec![1.0; 5]]);
        let y_pred = mat_from_rows(&[vec![0.0; 4]]); // wrong shape
        let result = functional_explained_variance(&y_true, &y_pred, &argvals);
        assert!(matches!(
            result,
            Err(FdarError::InvalidDimension {
                parameter: "y_pred",
                ..
            })
        ));
    }

    // CR-02 regression test: constant y_true + tiny-amplitude y_pred must NOT return 1.0.
    // Before the fix, both ss_tot and ss_res fell below NUMERICAL_EPS and the function
    // returned 1.0 even though ss_res > ss_tot (the prediction had MORE variation than the
    // constant baseline).
    #[test]
    fn test_explained_variance_constant_true_perturbed_pred() {
        // y_true is a constant curve (all 5.0) => SS_tot = 0.
        // y_pred = 5.0 + 1e-6 * sin(t) has a tiny positive SS_res > SS_tot.
        // Correct EV must be <= 0.0 (the pred is strictly worse than the baseline).
        let m = 100_usize;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let y_true_row: Vec<f64> = vec![5.0_f64; m];
        let y_pred_row: Vec<f64> = argvals
            .iter()
            .map(|&t| 5.0 + 1e-6 * (t * std::f64::consts::PI * 2.0).sin())
            .collect();
        let y_true = mat_from_rows(&[y_true_row]);
        let y_pred = mat_from_rows(&[y_pred_row]);
        let ev = functional_explained_variance(&y_true, &y_pred, &argvals).unwrap();
        assert!(
            ev <= 0.0,
            "EV for constant true + oscillating pred must be <= 0.0, got {ev}"
        );
    }
}
