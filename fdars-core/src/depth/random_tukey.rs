//! Random Tukey depth measures.

use crate::dim::Dim;
use crate::matrix::FdMatrix;

use super::random_depth_core;

/// Compute random Tukey depth for 1D functional data.
///
/// Takes the minimum over all random projections (more conservative than RP depth).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn random_tukey_1d(data_obj: &FdMatrix, data_ori: &FdMatrix, nproj: usize) -> Vec<f64> {
    random_tukey_1d_seeded(data_obj, data_ori, nproj, None)
}

/// Compute random Tukey depth with optional seed for reproducibility.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn random_tukey_1d_seeded(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
    nproj: usize,
    seed: Option<u64>,
) -> Vec<f64> {
    random_depth_core(
        data_obj,
        data_ori,
        nproj,
        seed,
        f64::INFINITY,
        f64::min,
        |acc, _| acc,
    )
}

/// Compute random Tukey depth for 1D or 2D functional data via a unified [`Dim`] dispatch.
///
/// The 2D path never diverged from the 1D one, so both [`Dim`] arms forward to
/// [`random_tukey_1d`]. Because `random_tukey_1d` draws fresh entropy from
/// `thread_rng()` (no public seed), the forwarding is a compile-time guarantee
/// (single-arm `match`), not a runtime-equality one; use
/// [`random_tukey_1d_seeded`] for reproducible results.
///
/// # Arguments
/// * `data_obj` - Data to compute depth for
/// * `data_ori` - Reference data
/// * `nproj` - Number of random projections
/// * `dim` - Dimensionality selector ([`Dim::One`] or [`Dim::Two`])
#[must_use = "expensive computation whose result should not be discarded"]
pub fn random_tukey(data_obj: &FdMatrix, data_ori: &FdMatrix, nproj: usize, dim: Dim) -> Vec<f64> {
    match dim {
        Dim::One | Dim::Two => random_tukey_1d(data_obj, data_ori, nproj),
    }
}

/// Compute random Tukey depth for 2D functional data.
#[deprecated(
    since = "0.30.0",
    note = "redundant with `random_tukey(…, Dim::Two)`; body just forwards to `random_tukey_1d`"
)]
#[must_use = "expensive computation whose result should not be discarded"]
pub fn random_tukey_2d(data_obj: &FdMatrix, data_ori: &FdMatrix, nproj: usize) -> Vec<f64> {
    random_tukey_1d(data_obj, data_ori, nproj)
}
