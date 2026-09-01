//! Modal depth measures.

use crate::dim::Dim;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Compute modal depth for 1D functional data.
///
/// Uses a Gaussian kernel to measure density around each curve.
///
/// # Arguments
/// * `data_obj` - Data to compute depth for
/// * `data_ori` - Reference data
/// * `h` - Bandwidth parameter
#[must_use = "expensive computation whose result should not be discarded"]
pub fn modal_1d(data_obj: &FdMatrix, data_ori: &FdMatrix, h: f64) -> Vec<f64> {
    let nobj = data_obj.nrows();
    let nori = data_ori.nrows();
    let n_points = data_obj.ncols();

    if nobj == 0 || nori == 0 || n_points == 0 {
        return Vec::new();
    }

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut depth = 0.0;

            for j in 0..nori {
                let dist_sq = data_obj.row_l2_sq(i, data_ori, j);
                let dist = (dist_sq / n_points as f64).sqrt();
                let kernel_val = (-0.5 * (dist / h).powi(2)).exp();
                depth += kernel_val;
            }

            depth / nori as f64
        })
        .collect()
}

/// Compute modal depth for 1D or 2D functional data via a unified [`Dim`] dispatch.
///
/// The 2D path never diverged from the 1D one, so both [`Dim`] arms forward to
/// [`modal_1d`]. The `dim` argument makes caller intent explicit and provides a
/// single future seam should a real 2D specialization ever be needed.
///
/// # Arguments
/// * `data_obj` - Data to compute depth for
/// * `data_ori` - Reference data
/// * `h` - Bandwidth parameter
/// * `dim` - Dimensionality selector ([`Dim::One`] or [`Dim::Two`])
#[must_use = "expensive computation whose result should not be discarded"]
pub fn modal(data_obj: &FdMatrix, data_ori: &FdMatrix, h: f64, dim: Dim) -> Vec<f64> {
    match dim {
        Dim::One | Dim::Two => modal_1d(data_obj, data_ori, h),
    }
}

/// Compute modal depth for 2D functional data.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn modal_2d(data_obj: &FdMatrix, data_ori: &FdMatrix, h: f64) -> Vec<f64> {
    modal_1d(data_obj, data_ori, h)
}
