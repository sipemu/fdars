use super::fpca::fpca_tolerance_band;
use super::helpers::valid_band_params;
use super::{BandType, ExponentialFamily, ToleranceBand};
use crate::error::FdarError;
use crate::matrix::FdMatrix;

// ─── Exponential Family Tolerance Band ──────────────────────────────────────

/// Apply the link function for an exponential family.
fn apply_link(value: f64, family: ExponentialFamily) -> f64 {
    match family {
        ExponentialFamily::Gaussian => value,
        ExponentialFamily::Binomial => {
            // logit: log(p / (1-p)), clamp to avoid infinities
            let p = value.clamp(1e-10, 1.0 - 1e-10);
            (p / (1.0 - p)).ln()
        }
        ExponentialFamily::Poisson => {
            // log link, clamp to avoid log(0)
            value.max(1e-10).ln()
        }
    }
}

/// Apply the inverse link function for an exponential family.
fn apply_inverse_link(value: f64, family: ExponentialFamily) -> f64 {
    match family {
        ExponentialFamily::Gaussian => value,
        ExponentialFamily::Binomial => {
            // inverse logit: 1 / (1 + exp(-x))
            1.0 / (1.0 + (-value).exp())
        }
        ExponentialFamily::Poisson => {
            // exp
            value.exp()
        }
    }
}

/// Apply a link function element-wise to all data entries.
fn transform_data(data: &FdMatrix, family: ExponentialFamily) -> FdMatrix {
    let (n, m) = data.shape();
    let mut out = FdMatrix::zeros(n, m);
    for j in 0..m {
        for i in 0..n {
            out[(i, j)] = apply_link(data[(i, j)], family);
        }
    }
    out
}

/// Apply the inverse link to a band, recomputing half-widths on the response scale.
fn inverse_link_band(band: &ToleranceBand, family: ExponentialFamily) -> ToleranceBand {
    let lower: Vec<f64> = band
        .lower
        .iter()
        .map(|&v| apply_inverse_link(v, family))
        .collect();
    let upper: Vec<f64> = band
        .upper
        .iter()
        .map(|&v| apply_inverse_link(v, family))
        .collect();
    let center: Vec<f64> = band
        .center
        .iter()
        .map(|&v| apply_inverse_link(v, family))
        .collect();
    let half_width: Vec<f64> = upper
        .iter()
        .zip(lower.iter())
        .map(|(&u, &l)| (u - l) / 2.0)
        .collect();
    ToleranceBand {
        lower,
        upper,
        center,
        half_width,
    }
}

/// Compute a tolerance band for exponential family functional data.
///
/// Transforms data via the canonical link function, applies FPCA + bootstrap
/// on the transformed scale, then maps the band back via the inverse link.
///
/// # Arguments
/// * `data` — Functional data matrix (n x m), values in natural parameter space
/// * `family` — [`ExponentialFamily`] specifying the distribution
/// * `ncomp` — Number of principal components to retain
/// * `nb` — Number of bootstrap replicates
/// * `coverage` — Target coverage probability (e.g., 0.95)
/// * `seed` — Random seed for reproducibility
///
/// # Returns
/// `Ok(ToleranceBand)` on success, or `Err(FdarError)` if inputs are invalid or FPCA fails.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 3 rows
/// or zero columns.
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero, `nb` is zero,
/// or `coverage` is not in the open interval (0, 1).
/// Returns [`FdarError::ComputationFailed`] if the underlying FPCA on the
/// link-transformed data fails.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::simulation::{sim_fundata, EFunType, EValType};
/// use fdars_core::tolerance::{exponential_family_tolerance_band, ExponentialFamily};
///
/// // Create positive data suitable for Poisson family
/// let t: Vec<f64> = (0..30).map(|i| i as f64 / 29.0).collect();
/// let raw = sim_fundata(30, &t, 3, EFunType::Fourier, EValType::Exponential, Some(42));
/// let mut data = FdMatrix::zeros(30, 30);
/// for j in 0..30 {
///     for i in 0..30 {
///         data[(i, j)] = (raw[(i, j)] + 5.0).max(0.1);
///     }
/// }
///
/// let band = exponential_family_tolerance_band(
///     &data, ExponentialFamily::Poisson, 3, 50, 0.95, 42,
/// ).unwrap();
/// // Poisson inverse link (exp) ensures all bounds are positive
/// assert!(band.lower.iter().all(|&v| v > 0.0));
/// ```
pub fn exponential_family_tolerance_band(
    data: &FdMatrix,
    family: ExponentialFamily,
    ncomp: usize,
    nb: usize,
    coverage: f64,
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
    if !valid_band_params(n, m, ncomp, nb, coverage) {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp/nb/coverage",
            message: "ncomp and nb must be >= 1, coverage must be in (0, 1)".to_string(),
        });
    }

    let transformed = transform_data(data, family);
    let band = fpca_tolerance_band(
        &transformed,
        ncomp,
        nb,
        coverage,
        BandType::Simultaneous,
        seed,
    )?;
    Ok(inverse_link_band(&band, family))
}
