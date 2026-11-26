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
use crate::regions::{difference_regions, DifferenceRegion};
use crate::tolerance::{
    scb_mean_degras, scb_mean_degras_with_critical, MultiplierDistribution, ToleranceBand,
};

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
    let (band, _) = two_sample_band_core(
        data_a, data_b, argvals, bandwidth, nb, confidence, multiplier,
    )?;
    let (max_excursion, excludes_zero) = band_excursion(&band);
    let p_value = if excludes_zero { 0.0 } else { 1.0 };
    Ok(TestResult {
        statistic: max_excursion,
        p_value,
        n_perm: 0,
    })
}

/// Validate two samples, form the paired difference matrix and run the Degras
/// band on it. Shared by [`scb_two_sample_test`] and [`scb_two_sample_band`].
fn two_sample_band_core(
    data_a: &FdMatrix,
    data_b: &FdMatrix,
    argvals: &[f64],
    bandwidth: f64,
    nb: usize,
    confidence: f64,
    multiplier: MultiplierDistribution,
) -> Result<(ToleranceBand, f64), FdarError> {
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
    scb_mean_degras_with_critical(&diff, argvals, bandwidth, nb, confidence, multiplier)
}

/// Maximum standardized excursion `|center| / half_width` of a band and
/// whether the band excludes zero at any grid point.
fn band_excursion(band: &ToleranceBand) -> (f64, bool) {
    let mut max_excursion = 0.0_f64;
    let mut excludes_zero = false;
    for j in 0..band.center.len() {
        let hw = band.half_width[j].max(1e-300);
        let excursion = band.center[j].abs() / hw;
        if excursion > max_excursion {
            max_excursion = excursion;
        }
        if band.lower[j] > 0.0 || band.upper[j] < 0.0 {
            excludes_zero = true;
        }
    }
    (max_excursion, excludes_zero)
}

/// Simultaneous confidence band for the two-sample mean difference, with the
/// regions where it excludes zero. Result of [`scb_two_sample_band`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ScbTwoSampleBand {
    /// Smoothed estimate of `mean_a(t) − mean_b(t)` (band centre).
    pub estimate: Vec<f64>,
    /// Lower band limit.
    pub lower: Vec<f64>,
    /// Upper band limit.
    pub upper: Vec<f64>,
    /// Half-width `critical_value · σ̂(t) / √n` at each grid point.
    pub half_width: Vec<f64>,
    /// Multiplier-bootstrap critical value `c` (the `confidence`-quantile of
    /// `sup_t |Z*(t)|`).
    pub critical_value: f64,
    /// Confidence level used.
    pub confidence: f64,
    /// Maximum standardized excursion `max_t |estimate| / half_width`
    /// (identical to the `statistic` of [`scb_two_sample_test`]).
    pub statistic: f64,
    /// Whether the band excludes zero somewhere (identical to the decision of
    /// [`scb_two_sample_test`]: rejected ⇔ its `p_value == 0`).
    pub rejected: bool,
    /// Contiguous grid regions where the band excludes zero. `Positive` ⇔
    /// `mean_a > mean_b` there; `min_adjusted_pvalue` is `None`.
    pub regions: Vec<DifferenceRegion>,
}

/// Two-sample simultaneous confidence band for the mean difference, exposing
/// the band and its significant regions.
///
/// Same construction and arguments as [`scb_two_sample_test`] (paired
/// difference of the first `min(n_a, n_b)` rows, Degras multiplier bootstrap),
/// which returns only the decision; this function additionally returns the
/// band (`estimate`, `lower`, `upper`, `half_width`, `critical_value`) and the
/// contiguous regions where it excludes zero — the regions where the two mean
/// functions differ with simultaneous (family-wise over the grid) coverage
/// `confidence`.
///
/// The `statistic` and `rejected` fields reproduce [`scb_two_sample_test`]
/// exactly for the same inputs.
///
/// # Errors
///
/// Same as [`scb_two_sample_test`].
///
/// # Examples
///
/// ```
/// use fdars_core::inference::scb_two_sample_band;
/// use fdars_core::tolerance::MultiplierDistribution;
/// use fdars_core::FdMatrix;
///
/// let m = 40;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let (n, mut a, mut b) = (30, FdMatrix::zeros(30, m), FdMatrix::zeros(30, m));
/// for i in 0..n {
///     for j in 0..m {
///         let noise = 0.1 * (((i * 31 + j * 17) % 13) as f64 / 6.0 - 1.0);
///         let bump = (-((t[j] - 0.7) / 0.08).powi(2)).exp();
///         a[(i, j)] = t[j] + bump + noise;
///         b[(i, j)] = t[j] - 0.5 * noise;
///     }
/// }
/// let band = scb_two_sample_band(&a, &b, &t, 0.05, 300, 0.95,
///     MultiplierDistribution::Gaussian).unwrap();
/// assert!(band.rejected);
/// // The region where A > B sits around the bump at t = 0.7.
/// let r = &band.regions[0];
/// assert!(r.t_start > 0.45 && r.t_end < 0.95);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn scb_two_sample_band(
    data_a: &FdMatrix,
    data_b: &FdMatrix,
    argvals: &[f64],
    bandwidth: f64,
    nb: usize,
    confidence: f64,
    multiplier: MultiplierDistribution,
) -> Result<ScbTwoSampleBand, FdarError> {
    let (band, critical_value) = two_sample_band_core(
        data_a, data_b, argvals, bandwidth, nb, confidence, multiplier,
    )?;
    let (statistic, rejected) = band_excursion(&band);
    let significant: Vec<bool> = band
        .lower
        .iter()
        .zip(&band.upper)
        .map(|(&l, &u)| l > 0.0 || u < 0.0)
        .collect();
    let regions = difference_regions(argvals, &band.center, &significant, None)?;
    Ok(ScbTwoSampleBand {
        estimate: band.center,
        lower: band.lower,
        upper: band.upper,
        half_width: band.half_width,
        critical_value,
        confidence,
        statistic,
        rejected,
        regions,
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

    /// Verbatim copy of the pre-#76 `scb_two_sample_test` body (before the
    /// band core was factored out), used as a same-process bit-identity
    /// reference.
    fn legacy_scb_two_sample_test(
        data_a: &FdMatrix,
        data_b: &FdMatrix,
        argvals: &[f64],
        bandwidth: f64,
        nb: usize,
        confidence: f64,
        multiplier: MultiplierDistribution,
    ) -> TestResult {
        let n = data_a.nrows().min(data_b.nrows());
        let m = data_a.ncols();
        let mut diff = FdMatrix::zeros(n, m);
        for j in 0..m {
            for i in 0..n {
                diff[(i, j)] = data_a[(i, j)] - data_b[(i, j)];
            }
        }
        let band = scb_mean_degras(&diff, argvals, bandwidth, nb, confidence, multiplier).unwrap();
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
        TestResult {
            statistic: max_excursion,
            p_value: if excludes_zero { 0.0 } else { 1.0 },
            n_perm: 0,
        }
    }

    fn bump_samples(height: f64, seed: u64) -> (FdMatrix, FdMatrix, Vec<f64>) {
        let argvals = uniform_grid(50);
        let a = make_sample(
            40,
            &argvals,
            |t| t + height * (-((t - 0.3) / 0.06).powi(2)).exp(),
            0.0,
            0.3,
            seed,
        );
        let b = make_sample(36, &argvals, |t| t, 0.0, 0.3, seed + 1000);
        (a, b, argvals)
    }

    #[test]
    fn scb_test_bit_identical_to_legacy_and_band_agrees() {
        for (height, seed) in [(0.0, 3u64), (0.4, 4), (1.5, 5), (0.0, 6)] {
            let (a, b, t) = bump_samples(height, seed);
            for mult in [
                MultiplierDistribution::Gaussian,
                MultiplierDistribution::Rademacher,
            ] {
                let new = scb_two_sample_test(&a, &b, &t, 0.04, 250, 0.95, mult).unwrap();
                let old = legacy_scb_two_sample_test(&a, &b, &t, 0.04, 250, 0.95, mult);
                assert_eq!(new.statistic.to_bits(), old.statistic.to_bits());
                assert_eq!(new.p_value.to_bits(), old.p_value.to_bits());
                let band = scb_two_sample_band(&a, &b, &t, 0.04, 250, 0.95, mult).unwrap();
                assert_eq!(band.statistic.to_bits(), new.statistic.to_bits());
                assert_eq!(band.rejected, new.p_value == 0.0);
                assert_eq!(band.rejected, !band.regions.is_empty());
                // Band invariants.
                for j in 0..t.len() {
                    assert!(
                        (band.upper[j] - band.lower[j] - 2.0 * band.half_width[j]).abs() < 1e-12
                    );
                    assert!(band.lower[j] <= band.estimate[j] && band.estimate[j] <= band.upper[j]);
                }
                assert!(band.critical_value > 0.0);
            }
        }
    }

    #[test]
    fn scb_band_matches_mean_scb_of_paired_difference() {
        let (a, b, t) = bump_samples(1.0, 9);
        let band =
            scb_two_sample_band(&a, &b, &t, 0.05, 200, 0.9, MultiplierDistribution::Gaussian)
                .unwrap();
        let n = b.nrows();
        let mut diff = FdMatrix::zeros(n, t.len());
        for j in 0..t.len() {
            for i in 0..n {
                diff[(i, j)] = a[(i, j)] - b[(i, j)];
            }
        }
        let reference =
            mean_scb(&diff, &t, 0.05, 200, 0.9, MultiplierDistribution::Gaussian).unwrap();
        assert_eq!(band.estimate, reference.center);
        assert_eq!(band.lower, reference.lower);
        assert_eq!(band.upper, reference.upper);
        assert_eq!(band.half_width, reference.half_width);
    }

    #[test]
    fn scb_band_localises_bump() {
        let (a, b, t) = bump_samples(1.5, 21);
        let band = scb_two_sample_band(
            &a,
            &b,
            &t,
            0.03,
            400,
            0.95,
            MultiplierDistribution::Gaussian,
        )
        .unwrap();
        assert!(band.rejected);
        assert_eq!(band.regions.len(), 1, "regions: {:?}", band.regions);
        let r = &band.regions[0];
        assert_eq!(r.direction, crate::regions::SignificanceDirection::Positive);
        // Bump centred at 0.3 with sd ~0.04: region inside [0.15, 0.45] and
        // covering the centre.
        assert!(r.t_start >= 0.15 && r.t_end <= 0.45, "{r:?}");
        assert!(r.t_start <= 0.3 && r.t_end >= 0.3, "{r:?}");
        assert!(r.min_adjusted_pvalue.is_none());
    }
}
