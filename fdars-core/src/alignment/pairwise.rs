//! Pairwise elastic alignment, distance computation, and distance matrices.

use super::srsf::{reparameterize_curve, srsf_single, srsf_transform};
use super::{dp_alignment_core, AlignmentResult};
use crate::helpers::{l2_distance, simpsons_weights};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─── Public Alignment Functions ─────────────────────────────────────────────

/// Align curve f2 to curve f1 using the elastic framework.
///
/// Computes the optimal warping γ such that f2∘γ is as close as possible
/// to f1 in the elastic (Fisher-Rao) metric.
///
/// # Arguments
/// * `f1` — Target curve (length m)
/// * `f2` — Curve to align (length m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight on warp deviation from identity (0.0 = no penalty)
///
/// # Returns
/// [`AlignmentResult`] with warping function, aligned curve, and elastic distance.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_align_pair(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64) -> AlignmentResult {
    let q1 = srsf_single(f1, argvals);
    let q2 = srsf_single(f2, argvals);
    elastic_align_pair_from_srsf(f2, &q1, &q2, argvals, lambda)
}

/// Compute the elastic distance between two curves.
///
/// This is shorthand for aligning the pair and returning only the distance.
///
/// # Arguments
/// * `f1` — First curve (length m)
/// * `f2` — Second curve (length m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight on warp deviation from identity (0.0 = no penalty)
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_distance(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64) -> f64 {
    elastic_align_pair(f1, f2, argvals, lambda).distance
}

// ─── Internal Helpers with Pre-computed SRSFs ────────────────────────────────

/// Align curve f2 to curve f1 given their pre-computed SRSFs.
///
/// This avoids redundant SRSF computation when calling from distance matrix
/// routines where the same curve's SRSF would otherwise be recomputed for
/// every pair.
fn elastic_align_pair_from_srsf(
    f2: &[f64],
    q1: &[f64],
    q2: &[f64],
    argvals: &[f64],
    lambda: f64,
) -> AlignmentResult {
    // Find optimal warping via DP
    let gamma = dp_alignment_core(q1, q2, argvals, lambda);

    // Apply warping to f2
    let f_aligned = reparameterize_curve(f2, argvals, &gamma);

    // Compute elastic distance: L2 distance between q1 and aligned q2 SRSF
    let q_aligned = srsf_single(&f_aligned, argvals);

    let weights = simpsons_weights(argvals);
    let distance = l2_distance(q1, &q_aligned, &weights);

    AlignmentResult {
        gamma,
        f_aligned,
        distance,
    }
}

/// Compute elastic distance given a raw curve f2, and pre-computed SRSFs q1, q2.
///
/// The raw f2 is needed to reparameterize before computing the aligned SRSF.
fn elastic_distance_from_srsf(
    f2: &[f64],
    q1: &[f64],
    q2: &[f64],
    argvals: &[f64],
    lambda: f64,
) -> f64 {
    let gamma = dp_alignment_core(q1, q2, argvals, lambda);
    let f_aligned = reparameterize_curve(f2, argvals, &gamma);
    let q_aligned = srsf_single(&f_aligned, argvals);
    let weights = simpsons_weights(argvals);
    l2_distance(q1, &q_aligned, &weights)
}

// ─── Distance Matrices ──────────────────────────────────────────────────────

/// Compute the symmetric elastic distance matrix for a set of curves.
///
/// Pre-computes SRSF transforms for all curves once (O(n)) instead of
/// recomputing each curve's SRSF for every pair (O(n²)).
///
/// Uses upper-triangle computation with parallelism, following the
/// `self_distance_matrix` pattern from `metric.rs`.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight on warp deviation from identity (0.0 = no penalty)
///
/// # Returns
/// Symmetric n × n distance matrix.
pub fn elastic_self_distance_matrix(data: &FdMatrix, argvals: &[f64], lambda: f64) -> FdMatrix {
    let n = data.nrows();

    // Pre-compute all SRSF transforms once
    let srsfs = srsf_transform(data, argvals);

    let upper_vals: Vec<f64> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            let qi = srsfs.row(i);
            ((i + 1)..n)
                .map(|j| {
                    let fj = data.row(j);
                    let qj = srsfs.row(j);
                    elastic_distance_from_srsf(&fj, &qi, &qj, argvals, lambda)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut dist = FdMatrix::zeros(n, n);
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = upper_vals[idx];
            dist[(i, j)] = d;
            dist[(j, i)] = d;
            idx += 1;
        }
    }
    dist
}

/// Compute the elastic distance matrix between two sets of curves.
///
/// Pre-computes SRSF transforms for both datasets once instead of
/// recomputing each curve's SRSF for every pair.
///
/// # Arguments
/// * `data1` — First dataset (n1 × m)
/// * `data2` — Second dataset (n2 × m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight on warp deviation from identity (0.0 = no penalty)
///
/// # Returns
/// n1 × n2 distance matrix.
pub fn elastic_cross_distance_matrix(
    data1: &FdMatrix,
    data2: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();

    // Pre-compute all SRSF transforms once for both datasets
    let srsfs1 = srsf_transform(data1, argvals);
    let srsfs2 = srsf_transform(data2, argvals);

    let vals: Vec<f64> = iter_maybe_parallel!(0..n1)
        .flat_map(|i| {
            let qi = srsfs1.row(i);
            (0..n2)
                .map(|j| {
                    let fj = data2.row(j);
                    let qj = srsfs2.row(j);
                    elastic_distance_from_srsf(&fj, &qi, &qj, argvals, lambda)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut dist = FdMatrix::zeros(n1, n2);
    for i in 0..n1 {
        for j in 0..n2 {
            dist[(i, j)] = vals[i * n2 + j];
        }
    }
    dist
}

/// Compute the amplitude distance between two curves (= elastic distance after alignment).
pub fn amplitude_distance(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64) -> f64 {
    elastic_distance(f1, f2, argvals, lambda)
}

/// Compute the phase distance between two curves (geodesic distance of optimal warp from identity).
pub fn phase_distance_pair(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64) -> f64 {
    let alignment = elastic_align_pair(f1, f2, argvals, lambda);
    crate::warping::phase_distance(&alignment.gamma, argvals)
}

/// Compute the symmetric phase distance matrix for a set of curves.
pub fn phase_self_distance_matrix(data: &FdMatrix, argvals: &[f64], lambda: f64) -> FdMatrix {
    let n = data.nrows();

    let upper_vals: Vec<f64> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            let fi = data.row(i);
            ((i + 1)..n)
                .map(|j| {
                    let fj = data.row(j);
                    phase_distance_pair(&fi, &fj, argvals, lambda)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut dist = FdMatrix::zeros(n, n);
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = upper_vals[idx];
            dist[(i, j)] = d;
            dist[(j, i)] = d;
            idx += 1;
        }
    }
    dist
}

/// Compute the symmetric amplitude distance matrix (= elastic self distance matrix).
pub fn amplitude_self_distance_matrix(data: &FdMatrix, argvals: &[f64], lambda: f64) -> FdMatrix {
    elastic_self_distance_matrix(data, argvals, lambda)
}
