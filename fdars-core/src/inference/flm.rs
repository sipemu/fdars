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
}
