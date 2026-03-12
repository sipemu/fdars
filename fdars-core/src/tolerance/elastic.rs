use super::fpca::fpca_tolerance_band;
use super::helpers::valid_band_params;
use super::{BandType, ToleranceBand};
use crate::error::FdarError;
use crate::matrix::FdMatrix;

// ─── Elastic Tolerance Band ─────────────────────────────────────────────────

/// Compute a tolerance band in the elastic (aligned) space.
///
/// First computes the Karcher mean to align all curves, then applies the
/// FPCA tolerance band on the aligned data. This separates amplitude
/// variability from phase variability, giving bands that reflect shape
/// variation without contamination from timing differences.
///
/// # Arguments
/// * `data` — Functional data matrix (n x m)
/// * `argvals` — Evaluation points (length m)
/// * `ncomp` — Number of principal components to retain
/// * `nb` — Number of bootstrap replicates
/// * `coverage` — Target coverage probability (e.g., 0.95)
/// * `band_type` — [`BandType::Pointwise`] or [`BandType::Simultaneous`]
/// * `max_iter` — Maximum iterations for Karcher mean convergence
/// * `seed` — Random seed for reproducibility
///
/// # Returns
/// `Ok(ToleranceBand)` in the aligned space, or `Err(FdarError)` if inputs are invalid
/// or the underlying FPCA fails.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 3 rows,
/// zero columns, or if `argvals` length does not match the number of columns.
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero, `nb` is zero,
/// `coverage` is not in the open interval (0, 1), or `max_iter` is zero.
/// Returns [`FdarError::ComputationFailed`] if the underlying FPCA on the
/// aligned data fails.
///
/// # Examples
///
/// ```
/// use fdars_core::simulation::{sim_fundata, EFunType, EValType};
/// use fdars_core::tolerance::{elastic_tolerance_band, BandType};
///
/// let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
/// let data = sim_fundata(30, &t, 5, EFunType::Fourier, EValType::Exponential, Some(42));
///
/// let band = elastic_tolerance_band(&data, &t, 3, 100, 0.95, BandType::Pointwise, 10, 42).unwrap();
/// assert!(band.lower.iter().zip(band.upper.iter()).all(|(l, u)| l < u));
/// ```
pub fn elastic_tolerance_band(
    data: &FdMatrix,
    argvals: &[f64],
    ncomp: usize,
    nb: usize,
    coverage: f64,
    band_type: BandType,
    max_iter: usize,
    seed: u64,
) -> Result<ToleranceBand, FdarError> {
    let (n, m) = data.shape();
    if n < 3 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 rows and 1 column".to_string(),
            actual: format!("{n} x {m}"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {m} (matching data columns)"),
            actual: format!("length {}", argvals.len()),
        });
    }
    if !valid_band_params(n, m, ncomp, nb, coverage) {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp/nb/coverage",
            message: "ncomp and nb must be >= 1, coverage must be in (0, 1)".to_string(),
        });
    }
    if max_iter == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "max_iter",
            message: "must be >= 1".to_string(),
        });
    }

    // Step 1: Karcher mean -> aligned data
    let karcher = crate::alignment::karcher_mean(data, argvals, max_iter, 1e-4, 0.0);

    // Step 2: FPCA tolerance band on aligned data
    fpca_tolerance_band(&karcher.aligned_data, ncomp, nb, coverage, band_type, seed)
}
