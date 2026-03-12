//! Random Tukey depth measures.

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

/// Compute random Tukey depth for 2D functional data.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn random_tukey_2d(data_obj: &FdMatrix, data_ori: &FdMatrix, nproj: usize) -> Vec<f64> {
    random_tukey_1d(data_obj, data_ori, nproj)
}
