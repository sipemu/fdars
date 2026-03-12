//! Pairwise elastic alignment, distance computation, and distance matrices.

use super::srsf::{reparameterize_curve, srsf_transform};
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
    let m = f1.len();

    // Build single-row FdMatrices for SRSF computation
    let f1_mat = FdMatrix::from_slice(f1, 1, m).expect("dimension invariant: data.len() == n * m");
    let f2_mat = FdMatrix::from_slice(f2, 1, m).expect("dimension invariant: data.len() == n * m");

    let q1_mat = srsf_transform(&f1_mat, argvals);
    let q2_mat = srsf_transform(&f2_mat, argvals);

    let q1: Vec<f64> = q1_mat.row(0);
    let q2: Vec<f64> = q2_mat.row(0);

    // Find optimal warping via DP
    let gamma = dp_alignment_core(&q1, &q2, argvals, lambda);

    // Apply warping to f2
    let f_aligned = reparameterize_curve(f2, argvals, &gamma);

    // Compute elastic distance: L2 distance between q1 and aligned q2 SRSF
    let f_aligned_mat =
        FdMatrix::from_slice(&f_aligned, 1, m).expect("dimension invariant: data.len() == n * m");
    let q_aligned_mat = srsf_transform(&f_aligned_mat, argvals);
    let q_aligned: Vec<f64> = q_aligned_mat.row(0);

    let weights = simpsons_weights(argvals);
    let distance = l2_distance(&q1, &q_aligned, &weights);

    AlignmentResult {
        gamma,
        f_aligned,
        distance,
    }
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

// ─── Distance Matrices ──────────────────────────────────────────────────────

/// Compute the symmetric elastic distance matrix for a set of curves.
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

    let upper_vals: Vec<f64> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            let fi = data.row(i);
            ((i + 1)..n)
                .map(|j| {
                    let fj = data.row(j);
                    elastic_distance(&fi, &fj, argvals, lambda)
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

    let vals: Vec<f64> = iter_maybe_parallel!(0..n1)
        .flat_map(|i| {
            let fi = data1.row(i);
            (0..n2)
                .map(|j| {
                    let fj = data2.row(j);
                    elastic_distance(&fi, &fj, argvals, lambda)
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
