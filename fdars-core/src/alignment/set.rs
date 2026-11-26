//! Set-level alignment operations and elastic decomposition.

use super::optim::{check_grid, WarpOptimConfig};
use super::pairwise::{align_pair_cfg, elastic_align_pair};
use super::srsf::reparameterize_curve;
use super::{AlignmentResult, AlignmentSetResult};
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Result of elastic phase-amplitude decomposition.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct DecompositionResult {
    /// Full alignment result.
    pub alignment: AlignmentResult,
    /// Amplitude distance: SRSF distance after alignment.
    pub d_amplitude: f64,
    /// Phase distance: geodesic distance of warp from identity.
    pub d_phase: f64,
}

/// Align all curves in `data` to a single target curve.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `target` — Target curve to align to (length m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight on warp deviation from identity (0.0 = no penalty)
///
/// # Returns
/// [`AlignmentSetResult`] with all warping functions, aligned curves, and distances.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::alignment::align_to_target;
///
/// let argvals: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let target: Vec<f64> = argvals.iter().map(|&t| (t * 6.0).sin()).collect();
/// let data = FdMatrix::from_column_major(
///     (0..60).map(|i| ((i as f64 * 0.05) + 0.1).sin()).collect(),
///     3, 20,
/// ).unwrap();
/// let result = align_to_target(&data, &target, &argvals, 0.0);
/// assert_eq!(result.aligned_data.shape(), (3, 20));
/// assert_eq!(result.distances.len(), 3);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn align_to_target(
    data: &FdMatrix,
    target: &[f64],
    argvals: &[f64],
    lambda: f64,
) -> AlignmentSetResult {
    let align = |fi: &[f64]| elastic_align_pair(target, fi, argvals, lambda);
    collect_alignments(data, &align)
}

/// [`align_to_target`] with a configurable warp optimiser
/// ([`WarpOptimConfig`]: RBFGS, the DP options, a Sakoe–Chiba band, the λ
/// conversion). [`WarpOptimConfig::default`] with `lambda` set gives exactly
/// [`align_to_target`]`(data, target, argvals, lambda)`.
///
/// # Errors
/// [`FdarError::InvalidDimension`] if `target` or `data.ncols()` does not
/// match `argvals` (or `m < 2`); [`FdarError::InvalidParameter`] for a
/// non-increasing `argvals` or an invalid `cfg`.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{align_to_target_with_config, WarpOptimConfig, WarpOptimMethod};
/// use fdars_core::matrix::FdMatrix;
///
/// let t: Vec<f64> = (0..40).map(|i| i as f64 / 39.0).collect();
/// let target: Vec<f64> = t.iter().map(|&s| (6.0 * s).sin()).collect();
/// let mut data = FdMatrix::zeros(3, 40);
/// for i in 0..3 {
///     for j in 0..40 {
///         data[(i, j)] = (6.0 * (t[j] + 0.03 * i as f64)).sin();
///     }
/// }
/// let mut cfg = WarpOptimConfig::default();
/// cfg.method = WarpOptimMethod::Rbfgs;
/// let res = align_to_target_with_config(&data, &target, &t, &cfg).unwrap();
/// assert_eq!(res.gammas.shape(), (3, 40));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn align_to_target_with_config(
    data: &FdMatrix,
    target: &[f64],
    argvals: &[f64],
    cfg: &WarpOptimConfig,
) -> Result<AlignmentSetResult, FdarError> {
    check_grid(argvals, data.ncols(), "data")?;
    check_grid(argvals, target.len(), "target")?;
    cfg.validate()?;
    Ok(align_to_target_cfg(data, target, argvals, cfg))
}

/// [`align_to_target_with_config`] without validation.
pub(super) fn align_to_target_cfg(
    data: &FdMatrix,
    target: &[f64],
    argvals: &[f64],
    cfg: &WarpOptimConfig,
) -> AlignmentSetResult {
    let align = |fi: &[f64]| align_pair_cfg(target, fi, argvals, cfg);
    collect_alignments(data, &align)
}

/// Align every row of `data` with `align` and collect the results.
fn collect_alignments<A>(data: &FdMatrix, align: &A) -> AlignmentSetResult
where
    A: Fn(&[f64]) -> AlignmentResult + Sync,
{
    let (n, m) = data.shape();

    let results: Vec<AlignmentResult> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let fi = data.row(i);
            align(&fi)
        })
        .collect();

    let mut gammas = FdMatrix::zeros(n, m);
    let mut aligned_data = FdMatrix::zeros(n, m);
    let mut distances = Vec::with_capacity(n);

    for (i, r) in results.into_iter().enumerate() {
        for j in 0..m {
            gammas[(i, j)] = r.gamma[j];
            aligned_data[(i, j)] = r.f_aligned[j];
        }
        distances.push(r.distance);
    }

    AlignmentSetResult {
        gammas,
        aligned_data,
        distances,
    }
}

/// Perform elastic phase-amplitude decomposition of two curves.
///
/// Returns both the alignment result and the separate amplitude and phase distances.
///
/// # Arguments
/// * `f1` — Target curve (length m)
/// * `f2` — Curve to decompose against f1 (length m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight on warp deviation from identity (0.0 = no penalty)
pub fn elastic_decomposition(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    lambda: f64,
) -> DecompositionResult {
    let alignment = elastic_align_pair(f1, f2, argvals, lambda);
    let d_amplitude = alignment.distance;
    let d_phase = crate::warping::phase_distance(&alignment.gamma, argvals);
    DecompositionResult {
        alignment,
        d_amplitude,
        d_phase,
    }
}

/// Apply stored warps to original curves to produce aligned data.
pub(super) fn apply_stored_warps(data: &FdMatrix, gammas: &FdMatrix, argvals: &[f64]) -> FdMatrix {
    let (n, m) = data.shape();
    let mut aligned = FdMatrix::zeros(n, m);
    for i in 0..n {
        let fi = data.row(i);
        let gamma: Vec<f64> = (0..m).map(|j| gammas[(i, j)]).collect();
        let f_aligned = reparameterize_curve(&fi, argvals, &gamma);
        for j in 0..m {
            aligned[(i, j)] = f_aligned[j];
        }
    }
    aligned
}
