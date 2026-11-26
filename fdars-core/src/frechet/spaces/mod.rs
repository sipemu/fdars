//! Non-density object-data `MetricSpace` backends (FRE-02).
//!
//! Each backend implements [`crate::frechet::MetricSpace`] for a specific object
//! geometry, consumed generically by the Fréchet regression / ANOVA solvers:
//!
//! * [`SpdMatrixSpace`] / [`SpdMetric`] — SPD covariance matrices (FRE-02-01).
//! * [`CorrelationMatrixSpace`] — correlation matrices (FRE-02-02).
//! * [`SphericalSpace`] — spherical data (FRE-02-03).
//! * [`NetworkSpace`] — graph-Laplacian networks (FRE-02-04).
//! * [`PointProcessSpace`] — intensity/count point processes (FRE-02-05).

mod correlation;
mod network;
mod point_process;
mod spd;
mod spherical;

pub use correlation::CorrelationMatrixSpace;
pub use network::NetworkSpace;
pub use point_process::PointProcessSpace;
pub use spd::{SpdMatrixSpace, SpdMetric};
pub use spherical::SphericalSpace;

use crate::error::FdarError;

/// Shared `weighted_frechet_mean` input validation for flat-vector backends:
/// non-empty `objects`, `weights.len() == objects.len()`, and every object of
/// length `len`.
pub(super) fn check_mean_inputs(
    objects: &[Vec<f64>],
    weights: &[f64],
    len: usize,
) -> Result<(), FdarError> {
    if objects.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "objects",
            expected: "at least 1 object".to_string(),
            actual: "0 objects".to_string(),
        });
    }
    if weights.len() != objects.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "weights",
            expected: format!("{} weights (matching objects)", objects.len()),
            actual: format!("{} weights", weights.len()),
        });
    }
    for (i, o) in objects.iter().enumerate() {
        if o.len() != len {
            return Err(FdarError::InvalidDimension {
                parameter: "objects",
                expected: format!("each object has {len} elements"),
                actual: format!("object {i} has {} elements", o.len()),
            });
        }
    }
    Ok(())
}
