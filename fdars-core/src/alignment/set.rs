//! Set-level alignment operations and elastic decomposition.

use super::pairwise::elastic_align_pair;
use super::srsf::reparameterize_curve;
use super::{AlignmentResult, AlignmentSetResult};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Result of elastic phase-amplitude decomposition.
#[derive(Debug, Clone, PartialEq)]
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
#[must_use = "expensive computation whose result should not be discarded"]
pub fn align_to_target(
    data: &FdMatrix,
    target: &[f64],
    argvals: &[f64],
    lambda: f64,
) -> AlignmentSetResult {
    let (n, m) = data.shape();

    let results: Vec<AlignmentResult> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let fi = data.row(i);
            elastic_align_pair(target, &fi, argvals, lambda)
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
