//! Chi-squared distribution functions implemented from scratch.
//!
//! Provides CDF and quantile functions for the chi-squared distribution
//! via the regularized incomplete gamma function with Lanczos approximation.
//!
//! The Lanczos approximation with g=7 coefficients achieves relative error
//! < 1e-10 for x > 0.5 (Pugh, 2004, Table 4). Combined with the reflection
//! formula for x < 0.5, this covers the full domain. The chi-squared CDF
//! inherits this precision through the regularized incomplete gamma function.
//!
//! Phase-49 CONS-01: the CDF-family regularized incomplete gamma kernel and the
//! Lanczos `ln_gamma` were consolidated into `crate::distributions`. The
//! `pub(crate)` names below are preserved (spm-internal callers and `spm/tests.rs`
//! are unchanged) but now delegate to the shared home — a bit-identical
//! code-motion locked by `tests/equivalence_phase49.rs`.
//!
//! # References
//!
//! - Abramowitz, M. & Stegun, I.A. (1964). *Handbook of Mathematical
//!   Functions*. Dover. Formula 26.2.23.
//! - Johnson, N.L., Kotz, S. & Balakrishnan, N. (1994). *Continuous
//!   Univariate Distributions*, Vol. 1. Wiley. §18.6.2.
//! - Pugh, G.R. (2004). *An Analysis of the Lanczos Gamma Approximation*.
//!   Ph.D. thesis, University of British Columbia. Table 4, g=7, n=9.

/// Regularized lower incomplete gamma function P(a, x). Delegates to the
/// consolidated [`crate::distributions::reg_gamma_p`] (P-direct tail policy).
pub(crate) fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    crate::distributions::reg_gamma_p(a, x)
}

/// Chi-squared CDF: P(χ²(k) <= x). Delegates to [`crate::distributions::chi2_cdf`].
pub(crate) fn chi2_cdf(x: f64, k: usize) -> f64 {
    crate::distributions::chi2_cdf(x, k)
}

/// Chi-squared quantile function (inverse CDF). Delegates to
/// [`crate::distributions::chi2_quantile`] (Wilson-Hilferty + Newton).
pub(crate) fn chi2_quantile(p: f64, k: usize) -> f64 {
    crate::distributions::chi2_quantile(p, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::ln_gamma;
    use std::f64::consts::PI;

    #[test]
    fn test_ln_gamma_known_values() {
        // Gamma(1) = 1, ln(1) = 0
        assert!((ln_gamma(1.0)).abs() < 1e-10);
        // Gamma(2) = 1, ln(1) = 0
        assert!((ln_gamma(2.0)).abs() < 1e-10);
        // Gamma(5) = 24, ln(24) ≈ 3.1781
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-6);
        // Gamma(0.5) = sqrt(pi), ln(sqrt(pi)) ≈ 0.5724
        assert!((ln_gamma(0.5) - 0.5 * PI.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_chi2_cdf_zero() {
        assert_eq!(chi2_cdf(0.0, 1), 0.0);
        assert_eq!(chi2_cdf(0.0, 5), 0.0);
        assert_eq!(chi2_cdf(-1.0, 3), 0.0);
    }

    #[test]
    fn test_chi2_cdf_known_values() {
        // chi2_cdf(1.386, 2) ≈ 0.5
        let val = chi2_cdf(1.3862943611198906, 2);
        assert!(
            (val - 0.5).abs() < 1e-4,
            "chi2_cdf(1.386, 2) should be ~0.5, got {val}"
        );

        // chi2_cdf(5.991, 2) ≈ 0.95
        let val = chi2_cdf(5.991464547107979, 2);
        assert!(
            (val - 0.95).abs() < 1e-3,
            "chi2_cdf(5.991, 2) should be ~0.95, got {val}"
        );
    }

    #[test]
    fn test_chi2_quantile_median() {
        // Median of chi2(2) ≈ 1.3863
        let q = chi2_quantile(0.5, 2);
        assert!(
            (q - 1.3862943611198906).abs() < 0.01,
            "chi2_quantile(0.5, 2) should be ~1.3863, got {q}"
        );
    }

    #[test]
    fn test_chi2_quantile_95th() {
        // 95th percentile of chi2(2) ≈ 5.991
        let q = chi2_quantile(0.95, 2);
        assert!(
            (q - 5.991464547107979).abs() < 0.01,
            "chi2_quantile(0.95, 2) should be ~5.991, got {q}"
        );
    }

    #[test]
    fn test_chi2_roundtrip() {
        for k in &[1, 2, 5, 10, 20] {
            for &x in &[0.5, 1.0, 3.0, 5.0, 10.0, 20.0] {
                let p = chi2_cdf(x, *k);
                if p > 0.001 && p < 0.999 {
                    let x_back = chi2_quantile(p, *k);
                    assert!(
                        (x_back - x).abs() < 0.05,
                        "Round-trip failed for k={k}, x={x}: got p={p}, x_back={x_back}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_chi2_quantile_boundary() {
        assert_eq!(chi2_quantile(0.0, 5), 0.0);
        assert!(chi2_quantile(1.0, 5).is_infinite());
    }

    #[test]
    fn test_regularized_gamma_boundary() {
        assert_eq!(regularized_gamma_p(1.0, 0.0), 0.0);
        assert_eq!(regularized_gamma_p(5.0, 0.0), 0.0);
    }
}
