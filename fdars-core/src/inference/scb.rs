//! Simultaneous confidence bands (Degras) for the mean and mean difference.
//!
//! [`mean_scb`] is a thin inference-facing wrapper over
//! [`crate::tolerance::scb_mean_degras`]. [`scb_two_sample_test`] builds a
//! simultaneous confidence band around the *difference* of the two sample-mean
//! functions using the same Degras multiplier-bootstrap machinery and rejects
//! the null of equal means when that band excludes zero at any grid point
//! (`SCBmeanfd`-style two-sample test).

use super::TestResult;
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::tolerance::{scb_mean_degras, MultiplierDistribution, ToleranceBand};

/// Simultaneous confidence band for the mean function (Degras).
///
/// Thin inference-facing wrapper over [`crate::tolerance::scb_mean_degras`];
/// the band math is not reimplemented here. Returns a [`ToleranceBand`] whose
/// `[lower, upper]` covers the true mean at approximately `confidence`
/// coverage.
///
/// # Arguments
/// * `data` - Functional data matrix (`n x m`, `n >= 3` for Degras).
/// * `argvals` - Evaluation points (length `m`).
/// * `bandwidth` - Kernel bandwidth for local-polynomial smoothing (`> 0`).
/// * `nb` - Number of bootstrap replicates (`>= 1`).
/// * `confidence` - Confidence level in `(0, 1)`, e.g. `0.95`.
/// * `multiplier` - Multiplier distribution (Gaussian or Rademacher).
///
/// # Errors
///
/// Forwards all validation errors from [`scb_mean_degras`]:
/// [`FdarError::InvalidDimension`] for `< 3` rows, zero columns, or an
/// `argvals` length mismatch; [`FdarError::InvalidParameter`] for a
/// non-positive bandwidth, `nb == 0`, or `confidence` outside `(0, 1)`.
pub fn mean_scb(
    data: &FdMatrix,
    argvals: &[f64],
    bandwidth: f64,
    nb: usize,
    confidence: f64,
    multiplier: MultiplierDistribution,
) -> Result<ToleranceBand, FdarError> {
    scb_mean_degras(data, argvals, bandwidth, nb, confidence, multiplier)
}

/// Two-sample mean-equality test via a simultaneous confidence band for the
/// mean difference (`SCBmeanfd`-style).
///
/// Forms the paired difference matrix `d[i] = data_a[i] − data_b[i]` (over the
/// first `min(n_a, n_b)` rows) and reuses the Degras multiplier bootstrap
/// ([`scb_mean_degras`]) to produce a simultaneous band for the mean
/// difference. The null of equal means is rejected when that band excludes
/// zero at any grid point.
///
/// The returned [`TestResult`] encodes the decision:
/// * `statistic` is the maximum standardized excursion of the difference band
///   from zero, `max_t (|center(t)| / half_width(t))`; it exceeds `1.0` exactly
///   when the band excludes zero somewhere (i.e. when the null is rejected).
/// * `p_value` is `0.0` when the null is rejected (band excludes zero) and
///   `1.0` otherwise — a conservative encoding of the simultaneous-band
///   decision at the requested `confidence` (there is no finer p-value from a
///   single band). `n_perm` is `0`.
///
/// # Arguments
/// * `data_a` - First sample (`n_a x m`).
/// * `data_b` - Second sample (`n_b x m`).
/// * `argvals` - Evaluation points (length `m`).
/// * `bandwidth` - Kernel bandwidth (`> 0`).
/// * `nb` - Number of bootstrap replicates (`>= 1`).
/// * `confidence` - Confidence level in `(0, 1)`.
/// * `multiplier` - Multiplier distribution.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if the two samples have unequal or
/// zero column counts or an `argvals` length mismatch. Forwards
/// [`scb_mean_degras`] validation errors otherwise (including the `>= 3` rows
/// requirement on the difference matrix, `bandwidth > 0`, `nb >= 1`,
/// `confidence in (0, 1)`).
pub fn scb_two_sample_test(
    data_a: &FdMatrix,
    data_b: &FdMatrix,
    argvals: &[f64],
    bandwidth: f64,
    nb: usize,
    confidence: f64,
    multiplier: MultiplierDistribution,
) -> Result<TestResult, FdarError> {
    let (n_a, m_a) = data_a.shape();
    let (n_b, m_b) = data_b.shape();
    if m_a == 0 || m_b == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 column (grid points)".to_string(),
            actual: format!("data_a has {m_a} columns, data_b has {m_b} columns"),
        });
    }
    if m_a != m_b {
        return Err(FdarError::InvalidDimension {
            parameter: "data_b",
            expected: format!("{m_a} columns (matching data_a)"),
            actual: format!("{m_b} columns"),
        });
    }
    if argvals.len() != m_a {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m_a} elements (matching data columns)"),
            actual: format!("{} elements", argvals.len()),
        });
    }

    // Paired difference matrix over the first min(n_a, n_b) rows. The Degras
    // band on this matrix is a simultaneous confidence band for the mean
    // difference mean_a - mean_b.
    let n = n_a.min(n_b);
    let m = m_a;
    let mut diff = FdMatrix::zeros(n, m);
    for j in 0..m {
        for i in 0..n {
            diff[(i, j)] = data_a[(i, j)] - data_b[(i, j)];
        }
    }

    // Delegates remaining validation (>= 3 rows, bandwidth, nb, confidence).
    let band = scb_mean_degras(&diff, argvals, bandwidth, nb, confidence, multiplier)?;

    // Reject when the band excludes zero at any grid point, i.e. when the
    // maximum standardized excursion |center| / half_width exceeds 1.
    let mut max_excursion = 0.0_f64;
    let mut excludes_zero = false;
    for j in 0..m {
        let hw = band.half_width[j].max(1e-300);
        let excursion = band.center[j].abs() / hw;
        if excursion > max_excursion {
            max_excursion = excursion;
        }
        if band.lower[j] > 0.0 || band.upper[j] < 0.0 {
            excludes_zero = true;
        }
    }

    let p_value = if excludes_zero { 0.0 } else { 1.0 };
    Ok(TestResult {
        statistic: max_excursion,
        p_value,
        n_perm: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    /// Curves around a known mean function `mean_fn(t) + shift`, with
    /// deterministic bounded noise.
    fn make_sample(
        n: usize,
        argvals: &[f64],
        mean_fn: impl Fn(f64) -> f64,
        shift: f64,
        noise_amp: f64,
        seed: u64,
    ) -> FdMatrix {
        let m = argvals.len();
        let mut mat = FdMatrix::zeros(n, m);
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        for i in 0..n {
            for (j, &t) in argvals.iter().enumerate() {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // Zero-mean noise in [-1, 1): 2*u - 1 with u in [0, 1).
                let u = (state >> 33) as f64 / (1u64 << 31) as f64;
                let noise = 2.0 * u - 1.0;
                mat[(i, j)] = mean_fn(t) + shift + noise_amp * noise;
            }
        }
        mat
    }

    #[test]
    fn mean_scb_covers_true_mean() {
        let argvals = uniform_grid(40);
        // A gentle (near-linear) mean so that local-polynomial smoothing bias
        // is negligible and the simultaneous band should cover the true mean at
        // every grid point. (A high-curvature mean like sin(2πt) with a wide
        // bandwidth would be dominated by smoothing bias, not coverage.)
        let mean_fn = |t: f64| 0.5 + 0.3 * t;
        let data = make_sample(80, &argvals, mean_fn, 0.0, 0.4, 42);
        let band = mean_scb(
            &data,
            &argvals,
            0.1,
            400,
            0.95,
            MultiplierDistribution::Gaussian,
        )
        .unwrap();
        // The true mean should lie within [lower, upper] at every grid point.
        let mut covered = 0usize;
        for (j, &t) in argvals.iter().enumerate() {
            let truth = mean_fn(t);
            if band.lower[j] <= truth && truth <= band.upper[j] {
                covered += 1;
            }
        }
        let n = argvals.len();
        assert_eq!(
            covered, n,
            "true mean should be covered at every grid point ({covered}/{n})"
        );
    }

    #[test]
    fn scb_two_sample_detects_difference() {
        let argvals = uniform_grid(40);
        let mean_fn = |t: f64| (2.0 * std::f64::consts::PI * t).sin();
        // Same mean shape, but sample b shifted by a clear constant.
        let a = make_sample(50, &argvals, mean_fn, 0.0, 0.2, 11);
        let b = make_sample(50, &argvals, mean_fn, 1.5, 0.2, 22);
        let res = scb_two_sample_test(
            &a,
            &b,
            &argvals,
            0.15,
            300,
            0.95,
            MultiplierDistribution::Gaussian,
        )
        .unwrap();
        assert_eq!(res.p_value, 0.0, "clear difference should reject the null");
        assert!(res.statistic > 1.0);
    }

    #[test]
    fn scb_two_sample_no_difference() {
        let argvals = uniform_grid(40);
        let mean_fn = |t: f64| (2.0 * std::f64::consts::PI * t).sin();
        // Same generator (up to seed) -> no genuine mean difference.
        let a = make_sample(50, &argvals, mean_fn, 0.0, 0.3, 101);
        let b = make_sample(50, &argvals, mean_fn, 0.0, 0.3, 202);
        let res = scb_two_sample_test(
            &a,
            &b,
            &argvals,
            0.15,
            300,
            0.95,
            MultiplierDistribution::Gaussian,
        )
        .unwrap();
        assert_eq!(
            res.p_value, 1.0,
            "no genuine difference should fail to reject, got statistic={}",
            res.statistic
        );
    }

    #[test]
    fn scb_invalid_input() {
        let argvals = uniform_grid(20);
        let mean_fn = |t: f64| t;
        let a = make_sample(10, &argvals, mean_fn, 0.0, 0.1, 5);
        // Mismatched columns.
        let argvals_b = uniform_grid(15);
        let b = make_sample(10, &argvals_b, mean_fn, 0.0, 0.1, 6);
        assert!(matches!(
            scb_two_sample_test(
                &a,
                &b,
                &argvals,
                0.15,
                100,
                0.95,
                MultiplierDistribution::Gaussian
            ),
            Err(FdarError::InvalidDimension { .. })
        ));
        // Forwarded param validation: bandwidth <= 0.
        let b2 = make_sample(10, &argvals, mean_fn, 0.0, 0.1, 7);
        assert!(matches!(
            mean_scb(
                &a,
                &argvals,
                0.0,
                100,
                0.95,
                MultiplierDistribution::Gaussian
            ),
            Err(FdarError::InvalidParameter { .. })
        ));
        // confidence out of range, forwarded.
        assert!(matches!(
            scb_two_sample_test(
                &a,
                &b2,
                &argvals,
                0.15,
                100,
                1.5,
                MultiplierDistribution::Gaussian
            ),
            Err(FdarError::InvalidParameter { .. })
        ));
    }
}
