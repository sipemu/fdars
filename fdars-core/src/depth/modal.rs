//! Modal depth measures.

use crate::autodiff::Scalar;
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
pub(crate) fn modal_1d(data_obj: &FdMatrix, data_ori: &FdMatrix, h: f64) -> Vec<f64> {
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

/// Generic-over-`Scalar`, differentiable modal depth of a single curve (DOP-04).
///
/// Kernel-density depth of `curve` against the `reference` set:
/// `depth = (1/nref) · Σ_j exp(-0.5·(dist_j / h)²)`, where
/// `dist_j = sqrt( Σ_t (curve[t] - reference[j,t])² / n )`.
///
/// The **object curve** carries the scalar type `T`, so at `T = Dual` / `T = Var`
/// the returned depth propagates exact gradients w.r.t. the curve values. The
/// reference curves and bandwidth `h` stay `f64` (lifted via `T::from_f64`). Modal
/// depth is smooth everywhere (Gaussian kernel + L2, no rank/sort/indicator ops),
/// which is why it is the differentiable depth measure.
///
/// At `T = f64` this reproduces one row of [`modal_1d`] to within rounding
/// (~1e-12; `modal_1d` uses `f64::powi(2)` which is not bit-guaranteed equal to
/// `u*u`). Returns `T::zero()` for an empty reference set.
///
/// # Examples
///
/// ```
/// use fdars_core::depth::modal_depth_generic;
/// use fdars_core::matrix::FdMatrix;
///
/// // 3 reference curves × 4 points (column-major), plus a query curve.
/// let reference = FdMatrix::from_column_major(
///     vec![0.0, 1.0, -1.0, 0.1, 1.1, -0.9, 0.0, 1.0, -1.0, -0.1, 0.9, -1.1],
///     3,
///     4,
/// )
/// .unwrap();
/// let curve = vec![0.05, 1.05, -0.95, 0.0];
/// let d = modal_depth_generic::<f64>(&curve, &reference, 0.5);
/// assert!(d > 0.0 && d <= 1.0);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn modal_depth_generic<T: Scalar>(curve: &[T], reference: &FdMatrix, h: f64) -> T {
    let nref = reference.nrows();
    let n_points = curve.len();
    if nref == 0 || n_points == 0 {
        return T::zero();
    }
    let inv_n = T::from_f64(n_points as f64);
    let h_t = T::from_f64(h);
    let neg_half = T::from_f64(-0.5);

    let mut depth = T::zero();
    for j in 0..nref {
        let mut dist_sq = T::zero();
        for t in 0..n_points {
            let diff = curve[t] - T::from_f64(reference[(j, t)]);
            dist_sq += diff * diff;
        }
        // Match modal_1d's arithmetic order for f64 parity:
        // dist = sqrt(dist_sq / n); kernel = exp(-0.5 * (dist/h)^2).
        let dist = (dist_sq / inv_n).sqrt();
        let u = dist / h_t;
        let kernel = (neg_half * (u * u)).exp();
        depth += kernel;
    }
    depth / T::from_f64(nref as f64)
}
