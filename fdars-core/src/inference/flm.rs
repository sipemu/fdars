//! Functional-linear-model inference: overall-significance F-test and a
//! residual-based goodness-of-fit test on a fitted [`FregreLmResult`].
//!
//! Both tests read the public fields of a fitted FLM
//! ([`FregreLmResult::residuals`], [`FregreLmResult::fitted_values`],
//! [`FregreLmResult::r_squared`], [`FregreLmResult::ncomp`]) and convert the
//! observed statistic to a p-value via the self-contained F-distribution
//! survival function in [`super::dist`]. They are additive, `Result`-returning,
//! and validate their inputs at entry.

use super::dist::f_sf;
use super::TestResult;
use crate::error::FdarError;
use crate::scalar_on_function::FregreLmResult;

/// Overall-significance F-test for a fitted functional linear model.
///
/// Tests the null hypothesis H0 that the functional coefficient has no effect —
/// i.e. the FLM reduces to an intercept-only model. The statistic is the
/// classical regression F built from the model R²:
///
/// ```text
/// F = (R² / p) / ((1 − R²) / (n − p − 1))
/// ```
///
/// where `p = fit.ncomp` is the number of effective FPC parameters and
/// `n = fit.residuals.len()` is the sample size. Under H0, F follows an
/// F(p, n − p − 1) distribution, so the p-value is the F upper-tail
/// (survival) probability of the observed statistic. A small p-value rejects
/// H0 in favour of a genuine functional effect.
///
/// Returns a [`TestResult`] with `n_perm = 0` (this is an asymptotic /
/// closed-form test, not a permutation test).
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] when the fit is degenerate: `ncomp`
/// is zero, the denominator degrees of freedom `n − p − 1` are non-positive,
/// or `r_squared` is not finite or is `>= 1.0` (a perfect fit makes the F
/// statistic ill-defined).
pub fn flm_f_test(fit: &FregreLmResult) -> Result<TestResult, FdarError> {
    let p = fit.ncomp;
    let n = fit.residuals.len();
    let r2 = fit.r_squared;

    if p == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "fit.ncomp",
            message: "flm_f_test requires ncomp >= 1 (at least one FPC parameter)".to_string(),
        });
    }
    // Denominator degrees of freedom: n - p - 1 must be positive.
    if n <= p + 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "fit",
            message: format!(
                "degenerate degrees of freedom: n - p - 1 = {} - {} - 1 <= 0",
                n, p
            ),
        });
    }
    if !r2.is_finite() || r2 >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "fit.r_squared",
            message: format!(
                "r_squared must be finite and < 1.0 for a well-defined F statistic, got {r2}"
            ),
        });
    }

    let d1 = p as f64;
    let d2 = (n - p - 1) as f64;
    let f_stat = (r2 / d1) / ((1.0 - r2) / d2);
    let p_value = f_sf(f_stat, d1, d2);

    Ok(TestResult {
        statistic: f_stat,
        p_value,
        n_perm: 0,
    })
}

/// Residual-based lack-of-fit (goodness-of-fit) test for a fitted functional
/// linear model.
///
/// # Chosen null method (F-form lack-of-fit)
///
/// This test targets the null hypothesis H0 that the linear FLM is
/// **well specified** — that is, the conditional mean of the response is
/// captured by the fitted linear-functional relationship, so the residuals
/// carry no remaining structure associated with the fitted values. Rejection
/// (small p) is evidence of **lack of fit / mis-specification**.
///
/// The statistic is an F-form lack-of-fit test that regresses the fitted
/// model's residuals on a low-order polynomial expansion of the fitted values
/// (a Ramsey-RESET-style specification test). If the linear FLM is adequate,
/// the fitted values explain none of the residual variation and the extra
/// regressors are jointly insignificant; a strong nonlinearity the linear model
/// cannot capture leaves curvature in the residual-vs-fitted relationship that
/// the polynomial terms pick up.
///
/// Concretely, with residuals `e_i` and fitted values `ŷ_i`, we fit the
/// auxiliary regression
///
/// ```text
/// e_i = a0 + a1·ŷ_i + a2·ŷ_i² + a3·ŷ_i³ + u_i
/// ```
///
/// and report the F-statistic for H0: `a1 = a2 = a3 = 0` (the intercept is a
/// nuisance term, so `q = 3` restrictions). Under H0 the statistic is
/// F(q, n − q − 1); the p-value is its upper-tail probability, and a small
/// p-value rejects adequacy of the linear FLM.
///
/// Returns a [`TestResult`] with `statistic = F`, the F-tail `p_value`, and
/// `n_perm = 0`.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] when the sample is too small for the
/// auxiliary regression (`n − q − 1 <= 0` with `q = 3`), when the residual and
/// fitted-value vectors have mismatched lengths, or when the inputs contain
/// non-finite values / the fitted values are (numerically) constant so the
/// polynomial design is rank-deficient.
pub fn flm_gof_test(fit: &FregreLmResult) -> Result<TestResult, FdarError> {
    // Number of auxiliary (polynomial) restrictions being tested.
    const Q: usize = 3;

    let e = &fit.residuals;
    let yhat = &fit.fitted_values;
    let n = e.len();

    if yhat.len() != n {
        return Err(FdarError::InvalidParameter {
            parameter: "fit",
            message: format!(
                "residuals ({}) and fitted_values ({}) must have equal length",
                n,
                yhat.len()
            ),
        });
    }
    // Auxiliary regression has Q + 1 coefficients (intercept + Q powers);
    // need n - (Q + 1) > 0 residual df, i.e. n > Q + 1.
    if n <= Q + 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "fit",
            message: format!(
                "degenerate degrees of freedom for GoF: n = {n} <= {}",
                Q + 1
            ),
        });
    }
    if e.iter().chain(yhat.iter()).any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: "fit",
            message: "residuals / fitted_values contain non-finite values".to_string(),
        });
    }

    // Standardize fitted values to a stable scale before forming powers, so the
    // polynomial design is well-conditioned regardless of the response scale.
    let mean_yhat = yhat.iter().sum::<f64>() / n as f64;
    let var_yhat = yhat.iter().map(|&v| (v - mean_yhat).powi(2)).sum::<f64>() / n as f64;
    if var_yhat <= 1e-30 {
        return Err(FdarError::InvalidParameter {
            parameter: "fit.fitted_values",
            message: "fitted values are (numerically) constant; GoF design is rank-deficient"
                .to_string(),
        });
    }
    let sd_yhat = var_yhat.sqrt();

    // Design matrix columns: [1, z, z^2, z^3] with z the standardized fitted value.
    let ncoef = Q + 1;
    let mut x: Vec<[f64; 4]> = Vec::with_capacity(n);
    for &yh in yhat {
        let z = (yh - mean_yhat) / sd_yhat;
        x.push([1.0, z, z * z, z * z * z]);
    }

    // Normal equations X'X (ncoef x ncoef) and X'e (ncoef).
    let mut xtx = [[0.0f64; 4]; 4];
    let mut xte = [0.0f64; 4];
    for (row, &ei) in x.iter().zip(e.iter()) {
        for a in 0..ncoef {
            xte[a] += row[a] * ei;
            for b in 0..ncoef {
                xtx[a][b] += row[a] * row[b];
            }
        }
    }

    // Solve (X'X) coef = X'e via Gaussian elimination with partial pivoting.
    let coef = match solve_linear(xtx, xte, ncoef) {
        Some(c) => c,
        None => {
            return Err(FdarError::InvalidParameter {
                parameter: "fit.fitted_values",
                message: "GoF auxiliary design is singular (rank-deficient fitted values)"
                    .to_string(),
            });
        }
    };

    // Residual sum of squares of the FULL auxiliary model.
    let mut rss_full = 0.0;
    for (row, &ei) in x.iter().zip(e.iter()) {
        let pred = (0..ncoef).map(|a| coef[a] * row[a]).sum::<f64>();
        rss_full += (ei - pred).powi(2);
    }

    // Restricted model: intercept only (a1 = a2 = a3 = 0). Its fitted value is
    // the mean of e; RSS_restricted = Σ (e_i − ē)².
    let mean_e = e.iter().sum::<f64>() / n as f64;
    let rss_restricted = e.iter().map(|&ei| (ei - mean_e).powi(2)).sum::<f64>();

    let df_num = Q as f64; // restrictions
    let df_den = (n - ncoef) as f64; // n - (Q + 1)

    // Guard against an essentially-zero residual model (perfect auxiliary fit).
    if rss_full <= 1e-30 || rss_restricted <= 1e-30 {
        // No residual variation to explain → no evidence of lack of fit.
        return Ok(TestResult {
            statistic: 0.0,
            p_value: 1.0,
            n_perm: 0,
        });
    }

    let f_stat = ((rss_restricted - rss_full) / df_num) / (rss_full / df_den);
    let f_stat = f_stat.max(0.0);
    let p_value = f_sf(f_stat, df_num, df_den);

    Ok(TestResult {
        statistic: f_stat,
        p_value,
        n_perm: 0,
    })
}

/// Solve an `n x n` linear system `A x = b` (with `n <= 4`) via Gaussian
/// elimination with partial pivoting. Returns `None` if the matrix is
/// (numerically) singular.
fn solve_linear(a: [[f64; 4]; 4], b: [f64; 4], n: usize) -> Option<[f64; 4]> {
    let mut m = a;
    let mut rhs = b;
    for col in 0..n {
        // Partial pivot.
        let mut pivot = col;
        let mut best = m[col][col].abs();
        for r in (col + 1)..n {
            if m[r][col].abs() > best {
                best = m[r][col].abs();
                pivot = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if pivot != col {
            m.swap(pivot, col);
            rhs.swap(pivot, col);
        }
        let diag = m[col][col];
        for r in (col + 1)..n {
            let factor = m[r][col] / diag;
            for c in col..n {
                m[r][c] -= factor * m[col][c];
            }
            rhs[r] -= factor * rhs[col];
        }
    }
    // Back-substitution.
    let mut sol = [0.0f64; 4];
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for c in (i + 1)..n {
            s -= m[i][c] * sol[c];
        }
        sol[i] = s / m[i][i];
    }
    Some(sol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::FdMatrix;
    use crate::scalar_on_function::fregre_lm;
    use crate::test_helpers::uniform_grid;

    /// Deterministic pseudo-noise in [-1, 1] from a splitmix-style counter.
    fn noise(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let z = (*seed >> 33) as f64 / (1u64 << 31) as f64; // in [0, 2)
        z - 1.0
    }

    /// Build a functional predictor: curve i is a sine of amplitude a_i.
    fn make_curves(n: usize, argvals: &[f64], amps: &[f64], seed: u64) -> FdMatrix {
        let m = argvals.len();
        let mut s = seed;
        let mut mat = FdMatrix::zeros(n, m);
        for i in 0..n {
            for (j, &t) in argvals.iter().enumerate() {
                let base = amps[i] * (2.0 * std::f64::consts::PI * t).sin();
                mat[(i, j)] = base + 0.05 * noise(&mut s);
            }
        }
        mat
    }

    #[test]
    fn f_test_rejects_genuine_effect() {
        let argvals = uniform_grid(40);
        let n = 40;
        // Amplitudes vary across curves; response is a strong linear function
        // of the amplitude (the dominant functional signal) + small noise.
        let amps: Vec<f64> = (0..n).map(|i| 0.5 + 2.0 * (i as f64) / n as f64).collect();
        let data = make_curves(n, &argvals, &amps, 7);
        let mut s = 1234u64;
        let y: Vec<f64> = (0..n)
            .map(|i| 3.0 * amps[i] + 0.1 * noise(&mut s))
            .collect();

        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let res = flm_f_test(&fit).unwrap();
        assert!(
            res.p_value < 0.05,
            "genuine functional effect should reject H0, got p={} (F={})",
            res.p_value,
            res.statistic
        );
    }

    #[test]
    fn f_test_fails_to_reject_null_effect() {
        let argvals = uniform_grid(40);
        let n = 40;
        // Amplitudes vary, but the response is constructed INDEPENDENTLY of the
        // functional predictor (deterministic pseudo-noise with a separate
        // stream), so there is no genuine effect to detect.
        let amps: Vec<f64> = (0..n).map(|i| 0.5 + 2.0 * (i as f64) / n as f64).collect();
        let data = make_curves(n, &argvals, &amps, 71);
        let mut s = 98765u64;
        let y: Vec<f64> = (0..n).map(|_| 5.0 + noise(&mut s)).collect();

        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let res = flm_f_test(&fit).unwrap();
        assert!(
            res.p_value > 0.20,
            "null-effect fit should fail to reject H0, got p={} (F={}, R²={})",
            res.p_value,
            res.statistic,
            fit.r_squared
        );
    }

    #[test]
    fn f_test_guards_degenerate_df() {
        // n - p - 1 <= 0: with n = 5 curves and ncomp = 3 -> df_den = 1 ok,
        // but ncomp = 4 forces n - p - 1 = 0 -> Err.
        let argvals = uniform_grid(20);
        let n = 5;
        let amps: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
        let data = make_curves(n, &argvals, &amps, 3);
        let mut s = 5u64;
        let y: Vec<f64> = (0..n)
            .map(|i| 2.0 * amps[i] + 0.1 * noise(&mut s))
            .collect();
        let fit = fregre_lm(&data, &y, None, 4).unwrap();
        assert!(
            matches!(flm_f_test(&fit), Err(FdarError::InvalidParameter { .. })),
            "degenerate df must return Err"
        );
    }

    #[test]
    fn gof_fails_to_reject_well_specified() {
        let argvals = uniform_grid(50);
        let n = 60;
        let amps: Vec<f64> = (0..n).map(|i| 0.5 + 2.0 * (i as f64) / n as f64).collect();
        let data = make_curves(n, &argvals, &amps, 21);
        // Truly linear relationship + small noise.
        let mut s = 55u64;
        let y: Vec<f64> = (0..n)
            .map(|i| 2.0 * amps[i] + 0.05 * noise(&mut s))
            .collect();
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let res = flm_gof_test(&fit).unwrap();
        assert!(
            res.p_value > 0.10,
            "well-specified linear FLM should not be flagged, got p={} (F={})",
            res.p_value,
            res.statistic
        );
    }

    #[test]
    fn gof_rejects_mis_specified() {
        let argvals = uniform_grid(50);
        let n = 60;
        let amps: Vec<f64> = (0..n).map(|i| 0.5 + 2.0 * (i as f64) / n as f64).collect();
        let data = make_curves(n, &argvals, &amps, 22);
        // Strongly nonlinear (quadratic) relationship the linear FLM cannot
        // capture -> residual-vs-fitted curvature -> lack of fit.
        let mut s = 66u64;
        let y: Vec<f64> = (0..n)
            .map(|i| 4.0 * amps[i] * amps[i] + 0.05 * noise(&mut s))
            .collect();
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let res = flm_gof_test(&fit).unwrap();
        assert!(
            res.p_value < 0.05,
            "mis-specified (nonlinear) FLM should be flagged, got p={} (F={})",
            res.p_value,
            res.statistic
        );
    }

    #[test]
    fn gof_guards_degenerate_df() {
        // n = 4 <= Q + 1 = 4 -> Err.
        let argvals = uniform_grid(20);
        let n = 4;
        let amps: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
        let data = make_curves(n, &argvals, &amps, 9);
        let mut s = 4u64;
        let y: Vec<f64> = (0..n).map(|i| amps[i] + 0.1 * noise(&mut s)).collect();
        let fit = fregre_lm(&data, &y, None, 2).unwrap();
        assert!(matches!(
            flm_gof_test(&fit),
            Err(FdarError::InvalidParameter { .. })
        ));
    }
}
