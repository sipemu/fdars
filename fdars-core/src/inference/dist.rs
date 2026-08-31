//! Self-contained distribution survival functions for the inference module.
//!
//! This is the single crate-internal home for the special-function machinery
//! used by the functional-inference tests: the χ² survival function (via the
//! regularized incomplete gamma function) and the F survival function (via the
//! regularized incomplete beta function). Both are hand-rolled in the
//! Numerical-Recipes style so the crate takes on **no new dependency** for
//! distribution tails.
//!
//! Consumers: `hotelling.rs` (χ² tail for the Hotelling-T² p-value),
//! `flm.rs` (F tail for the FLM significance / goodness-of-fit tests), and
//! `anova.rs` (χ² tail for the scaled-χ² V-statistic p-value).
//!
//! Phase-49 CONS-01: the χ² SF machinery (`ln_gamma` + the Q-direct series/CF
//! primitives) was consolidated into `crate::distributions`; the χ² survival
//! wrappers here now delegate to `distributions::chi2_sf`, and the F-tail path
//! (`betai`/`betacf`) shares the consolidated `distributions::ln_gamma`. This is
//! a bit-identical code-motion (locked by `tests/equivalence_phase49.rs`).

use crate::distributions::{chi2_sf, ln_gamma};

/// Chi-square survival function: P(X > x) for X ~ χ²(k degrees of freedom).
///
/// Equivalent to the regularized upper incomplete gamma Q(k/2, x/2). Delegates
/// to the consolidated [`crate::distributions::chi2_sf`] (SF-direct tail policy).
pub(crate) fn chi_square_sf(x: f64, k: usize) -> f64 {
    chi2_sf(x, k as f64)
}

/// Chi-square survival function with a real-valued degrees-of-freedom argument.
///
/// The scaled-χ² (Satterthwaite/Box) approximation used by the V-statistic
/// needs a non-integer `df`, so this variant takes `df: f64` directly. Delegates
/// to the consolidated [`crate::distributions::chi2_sf`].
pub(crate) fn chi_square_sf_df(x: f64, df: f64) -> f64 {
    chi2_sf(x, df)
}

/// Continued-fraction evaluation of the incomplete beta function
/// (Numerical-Recipes `betacf`, Lentz form). Used only inside [`betai`].
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..300 {
        let m = m as f64;
        let m2 = 2.0 * m;
        // Even step.
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step.
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

/// Regularized incomplete beta function I_x(a, b), for 0 <= x <= 1.
///
/// Uses the continued-fraction form with the symmetry swap
/// `I_x(a,b) = 1 − I_{1−x}(b,a)` when `x >= (a+1)/(a+b+2)`, which keeps the
/// continued fraction in its fast-converging regime.
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Prefactor x^a (1-x)^b / (a B(a,b)).
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let bt = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// F-distribution survival function: P(X > f) for X ~ F(d1, d2).
///
/// Uses the identity `SF_F(f) = I_{d2 / (d2 + d1·f)}(d2/2, d1/2)` where
/// `I_x(a,b)` is the regularized incomplete beta function. Returns `1.0` for
/// `f <= 0` (the whole distribution lies to the right of 0) and is
/// monotonically decreasing in `f`. Self-contained — reuses the module's
/// [`ln_gamma`]; no new crate dependency.
///
/// `d1` (numerator) and `d2` (denominator) are the degrees of freedom and must
/// be positive; a non-positive df yields `1.0` (no evidence against the null).
pub(crate) fn f_sf(f: f64, d1: f64, d2: f64) -> f64 {
    if f <= 0.0 {
        return 1.0;
    }
    if d1 <= 0.0 || d2 <= 0.0 {
        return 1.0;
    }
    let x = d2 / (d2 + d1 * f);
    betai(d2 / 2.0, d1 / 2.0, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chi_square_sf_sane() {
        // χ²(1): P(X > 3.841) ≈ 0.05; P(X > 0) = 1.
        assert!((chi_square_sf(3.8415, 1) - 0.05).abs() < 1e-3);
        assert!((chi_square_sf(0.0, 3) - 1.0).abs() < 1e-12);
        // χ²(2): P(X > 5.991) ≈ 0.05.
        assert!((chi_square_sf(5.9915, 2) - 0.05).abs() < 1e-3);
        // Monotone decreasing.
        assert!(chi_square_sf(1.0, 3) > chi_square_sf(5.0, 3));
        // Real-df variant agrees with the integer variant at integer df.
        assert!((chi_square_sf_df(5.9915, 2.0) - chi_square_sf(5.9915, 2)).abs() < 1e-12);
    }

    #[test]
    fn f_sf_matches_tabulated_quantiles() {
        // f <= 0 short-circuits to 1.0.
        assert!((f_sf(0.0, 3.0, 10.0) - 1.0).abs() < 1e-12);
        assert!((f_sf(-1.0, 3.0, 10.0) - 1.0).abs() < 1e-12);

        // F(1, 10) 0.95-quantile ≈ 4.9646 → SF ≈ 0.05.
        assert!(
            (f_sf(4.9646, 1.0, 10.0) - 0.05).abs() < 1e-2,
            "F(1,10) SF at 4.9646 = {}",
            f_sf(4.9646, 1.0, 10.0)
        );
        // F(5, 20) 0.95-quantile ≈ 2.7109 → SF ≈ 0.05.
        assert!(
            (f_sf(2.7109, 5.0, 20.0) - 0.05).abs() < 1e-2,
            "F(5,20) SF at 2.7109 = {}",
            f_sf(2.7109, 5.0, 20.0)
        );
        // F(3, 30) 0.99-quantile ≈ 4.5097 → SF ≈ 0.01.
        assert!(
            (f_sf(4.5097, 3.0, 30.0) - 0.01).abs() < 1e-2,
            "F(3,30) SF at 4.5097 = {}",
            f_sf(4.5097, 3.0, 30.0)
        );

        // Large f in the far right tail → small p (F(5,20) at 3.4).
        assert!(f_sf(3.4, 5.0, 20.0) < 0.05);

        // Monotone decreasing in f.
        assert!(f_sf(1.0, 3.0, 20.0) > f_sf(5.0, 3.0, 20.0));
        // Bounded in [0, 1].
        let p = f_sf(2.5, 4.0, 15.0);
        assert!((0.0..=1.0).contains(&p));
    }
}
