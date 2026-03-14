use super::fpca::fpca_tolerance_band;
use super::helpers::valid_band_params;
use super::{
    BandType, ElasticToleranceBandResult, ElasticToleranceConfig, PhaseToleranceBand, ToleranceBand,
};
use crate::alignment::KarcherMeanResult;
use crate::elastic_fpca::{
    shooting_vectors_from_psis, sphere_karcher_mean, warps_to_normalized_psi,
};
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::warping::{exp_map_sphere, normalize_warp, psi_to_gam};

// ─── Validation Helper ─────────────────────────────────────────────────────

/// Validate common elastic tolerance band inputs.
fn validate_elastic_inputs(
    data: &FdMatrix,
    argvals: &[f64],
    ncomp: usize,
    nb: usize,
    coverage: f64,
    max_iter: usize,
) -> Result<(usize, usize), FdarError> {
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
    Ok((n, m))
}

// ─── Elastic Tolerance Band (amplitude only) ───────────────────────────────

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
#[must_use = "expensive computation whose result should not be discarded"]
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
    validate_elastic_inputs(data, argvals, ncomp, nb, coverage, max_iter)?;

    // Step 1: Karcher mean -> aligned data
    let karcher = crate::alignment::karcher_mean(data, argvals, max_iter, 1e-4, 0.0);

    // Step 2: FPCA tolerance band on aligned data
    fpca_tolerance_band(&karcher.aligned_data, ncomp, nb, coverage, band_type, seed)
}

// ─── Phase Tolerance Band ──────────────────────────────────────────────────

/// Compute phase band from pre-computed Karcher mean result.
fn phase_band_from_karcher(
    karcher: &KarcherMeanResult,
    argvals: &[f64],
    ncomp: usize,
    nb: usize,
    coverage: f64,
    band_type: BandType,
    seed: u64,
) -> Result<PhaseToleranceBand, FdarError> {
    let m = argvals.len();
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Step 1: Convert warps γ → ψ = √γ' (normalized to Hilbert sphere)
    let psis = warps_to_normalized_psi(&karcher.gammas, argvals);

    // Step 2: Karcher mean of ψ vectors on the sphere
    let mu_psi = sphere_karcher_mean(&psis, &time, 20);

    // Step 3: Shooting vectors (tangent space at mu_psi)
    let shooting = shooting_vectors_from_psis(&psis, &mu_psi, &time);

    // Step 4: FPCA tolerance band in tangent space
    let tangent_band = fpca_tolerance_band(&shooting, ncomp, nb, coverage, band_type, seed)?;

    // Step 5: Map tangent-space bounds back to warping functions
    //   lower tangent vector = center - half_width
    //   upper tangent vector = center + half_width
    let tangent_lower: Vec<f64> = tangent_band
        .center
        .iter()
        .zip(tangent_band.half_width.iter())
        .map(|(&c, &h)| c - h)
        .collect();
    let tangent_upper: Vec<f64> = tangent_band
        .center
        .iter()
        .zip(tangent_band.half_width.iter())
        .map(|(&c, &h)| c + h)
        .collect();

    // Map back through exponential map on the sphere
    let psi_lower = exp_map_sphere(&mu_psi, &tangent_lower, &time);
    let psi_upper = exp_map_sphere(&mu_psi, &tangent_upper, &time);

    // Convert ψ → γ (cumulative integral of ψ²)
    let mut gamma_lower = psi_to_gam(&psi_lower, &time);
    let mut gamma_upper = psi_to_gam(&psi_upper, &time);

    // Scale from [0,1] to original domain and ensure monotonicity
    let t0 = argvals[0];
    let domain = argvals[m - 1] - t0;
    for j in 0..m {
        gamma_lower[j] = t0 + gamma_lower[j] * domain;
        gamma_upper[j] = t0 + gamma_upper[j] * domain;
    }
    normalize_warp(&mut gamma_lower, argvals);
    normalize_warp(&mut gamma_upper, argvals);

    // Center gamma is the identity warp
    let gamma_center = argvals.to_vec();

    Ok(PhaseToleranceBand {
        gamma_lower,
        gamma_upper,
        gamma_center,
        tangent_band,
    })
}

/// Compute a phase tolerance band on warping functions.
///
/// Characterizes acceptable timing (phase) variation by projecting warping
/// functions onto the tangent space of the Hilbert sphere via shooting vectors,
/// computing FPCA tolerance bands in that space, and mapping the bounds back
/// to warping functions.
///
/// # Arguments
/// * `data` — Functional data matrix (n x m)
/// * `argvals` — Evaluation points (length m)
/// * `ncomp` — Number of principal components for phase variation
/// * `nb` — Number of bootstrap replicates
/// * `coverage` — Target coverage probability (e.g., 0.95)
/// * `band_type` — [`BandType::Pointwise`] or [`BandType::Simultaneous`]
/// * `max_iter` — Maximum iterations for Karcher mean convergence
/// * `seed` — Random seed for reproducibility
///
/// # Returns
/// A [`PhaseToleranceBand`] with lower/upper warping function bounds and the
/// tangent-space band.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 3 rows,
/// zero columns, or if `argvals` length does not match the number of columns.
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero, `nb` is zero,
/// `coverage` is not in the open interval (0, 1), or `max_iter` is zero.
/// Returns [`FdarError::ComputationFailed`] if the underlying FPCA fails.
///
/// # Examples
///
/// ```
/// use fdars_core::simulation::{sim_fundata, EFunType, EValType};
/// use fdars_core::tolerance::{phase_tolerance_band, BandType};
///
/// let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
/// let data = sim_fundata(30, &t, 5, EFunType::Fourier, EValType::Exponential, Some(42));
///
/// let phase = phase_tolerance_band(&data, &t, 3, 100, 0.95, BandType::Pointwise, 10, 42).unwrap();
/// // Phase bounds fix the domain endpoints
/// assert!((phase.gamma_lower[0] - t[0]).abs() < 1e-10);
/// assert!((phase.gamma_upper[49] - t[49]).abs() < 1e-10);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn phase_tolerance_band(
    data: &FdMatrix,
    argvals: &[f64],
    ncomp: usize,
    nb: usize,
    coverage: f64,
    band_type: BandType,
    max_iter: usize,
    seed: u64,
) -> Result<PhaseToleranceBand, FdarError> {
    validate_elastic_inputs(data, argvals, ncomp, nb, coverage, max_iter)?;

    let karcher = crate::alignment::karcher_mean(data, argvals, max_iter, 1e-4, 0.0);
    phase_band_from_karcher(&karcher, argvals, ncomp, nb, coverage, band_type, seed)
}

// ─── Joint Elastic Tolerance Band ──────────────────────────────────────────

/// Compute joint amplitude and phase elastic tolerance bands.
///
/// Runs a single Karcher mean alignment and then computes both:
/// - An **amplitude band** on the aligned curves (shape variation)
/// - A **phase band** on the warping functions (timing variation)
///
/// This is more efficient than calling [`elastic_tolerance_band`] and
/// [`phase_tolerance_band`] separately, since the Karcher mean is computed once.
///
/// # Arguments
/// * `data` — Functional data matrix (n x m)
/// * `argvals` — Evaluation points (length m)
/// * `config` — [`ElasticToleranceConfig`] with all parameters
///
/// # Returns
/// An [`ElasticToleranceBandResult`] containing both amplitude and phase bands.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 3 rows,
/// zero columns, or if `argvals` length does not match the number of columns.
/// Returns [`FdarError::InvalidParameter`] if `ncomp_amplitude` or `ncomp_phase`
/// is zero, `nb` is zero, `coverage` is not in (0, 1), or `max_iter` is zero.
/// Returns [`FdarError::ComputationFailed`] if the underlying FPCA fails.
///
/// # Examples
///
/// ```
/// use fdars_core::simulation::{sim_fundata, EFunType, EValType};
/// use fdars_core::tolerance::{elastic_tolerance_band_with_config, ElasticToleranceConfig};
///
/// let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
/// let data = sim_fundata(30, &t, 5, EFunType::Fourier, EValType::Exponential, Some(42));
///
/// let result = elastic_tolerance_band_with_config(&data, &t, &ElasticToleranceConfig::default()).unwrap();
/// assert_eq!(result.amplitude.lower.len(), 50);
/// assert_eq!(result.phase.gamma_lower.len(), 50);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_tolerance_band_with_config(
    data: &FdMatrix,
    argvals: &[f64],
    config: &ElasticToleranceConfig,
) -> Result<ElasticToleranceBandResult, FdarError> {
    // Validate for amplitude
    validate_elastic_inputs(
        data,
        argvals,
        config.ncomp_amplitude,
        config.nb,
        config.coverage,
        config.max_iter,
    )?;
    // Also validate phase ncomp
    if config.ncomp_phase == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp_phase",
            message: "must be >= 1".to_string(),
        });
    }

    // Single Karcher mean computation shared by both bands
    let karcher = crate::alignment::karcher_mean(data, argvals, config.max_iter, config.tol, 0.0);

    // Amplitude band on aligned data
    let amplitude = fpca_tolerance_band(
        &karcher.aligned_data,
        config.ncomp_amplitude,
        config.nb,
        config.coverage,
        config.band_type,
        config.seed,
    )?;

    // Phase band on warping functions
    let phase = phase_band_from_karcher(
        &karcher,
        argvals,
        config.ncomp_phase,
        config.nb,
        config.coverage,
        config.band_type,
        config.seed,
    )?;

    Ok(ElasticToleranceBandResult { amplitude, phase })
}
