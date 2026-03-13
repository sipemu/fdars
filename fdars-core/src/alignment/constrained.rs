//! Landmark-constrained elastic alignment.

use super::pairwise::elastic_align_pair;
use super::srsf::{reparameterize_curve, srsf_transform};
use super::{dp_edge_weight, dp_grid_solve, dp_lambda_penalty};
use crate::helpers::{l2_distance, linear_interp, simpsons_weights};
use crate::matrix::FdMatrix;
use crate::warping::normalize_warp;

/// Result of landmark-constrained elastic alignment.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ConstrainedAlignmentResult {
    /// Optimal warping function (length m).
    pub gamma: Vec<f64>,
    /// Aligned curve f2∘γ (length m).
    pub f_aligned: Vec<f64>,
    /// Elastic distance after alignment.
    pub distance: f64,
    /// Enforced landmark pairs (snapped to grid): `(target_t, source_t)`.
    pub enforced_landmarks: Vec<(f64, f64)>,
}

/// Snap a time value to the nearest grid point index.
fn snap_to_grid(t_val: f64, argvals: &[f64]) -> usize {
    let mut best = 0;
    let mut best_dist = (t_val - argvals[0]).abs();
    for (i, &a) in argvals.iter().enumerate().skip(1) {
        let d = (t_val - a).abs();
        if d < best_dist {
            best = i;
            best_dist = d;
        }
    }
    best
}

/// Run DP on a rectangular sub-grid `[sc..=ec] × [sr..=er]`.
///
/// Uses global indices for `dp_edge_weight`. Returns the path segment
/// as a list of `(tc_idx, tr_idx)` pairs from start to end.
fn dp_segment(
    q1: &[f64],
    q2: &[f64],
    argvals: &[f64],
    sc: usize,
    ec: usize,
    sr: usize,
    er: usize,
    lambda: f64,
) -> Vec<(usize, usize)> {
    let nc = ec - sc + 1;
    let nr = er - sr + 1;

    if nc <= 1 || nr <= 1 {
        return vec![(sc, sr), (ec, er)];
    }

    let path = dp_grid_solve(nr, nc, |local_sr, local_sc, local_tr, local_tc| {
        let gsr = sr + local_sr;
        let gsc = sc + local_sc;
        let gtr = sr + local_tr;
        let gtc = sc + local_tc;
        dp_edge_weight(q1, q2, argvals, gsc, gtc, gsr, gtr)
            + dp_lambda_penalty(argvals, gsc, gtc, gsr, gtr, lambda)
    });

    // Convert local indices to global
    path.iter().map(|&(lr, lc)| (sc + lc, sr + lr)).collect()
}

/// Build DP waypoints from landmark pairs: snap to grid, deduplicate, add endpoints.
fn build_constrained_waypoints(
    landmark_pairs: &[(f64, f64)],
    argvals: &[f64],
    m: usize,
) -> Vec<(usize, usize)> {
    let mut waypoints: Vec<(usize, usize)> = Vec::with_capacity(landmark_pairs.len() + 2);
    waypoints.push((0, 0));
    for &(tt, st) in landmark_pairs {
        let tc = snap_to_grid(tt, argvals);
        let tr = snap_to_grid(st, argvals);
        if let Some(&(prev_c, prev_r)) = waypoints.last() {
            if tc > prev_c && tr > prev_r {
                waypoints.push((tc, tr));
            }
        }
    }
    let last = m - 1;
    if let Some(&(prev_c, prev_r)) = waypoints.last() {
        if prev_c != last || prev_r != last {
            waypoints.push((last, last));
        }
    }
    waypoints
}

/// Run DP segments between consecutive waypoints and assemble into a gamma warp.
fn segmented_dp_gamma(
    q1n: &[f64],
    q2n: &[f64],
    argvals: &[f64],
    waypoints: &[(usize, usize)],
    lambda: f64,
) -> Vec<f64> {
    let mut full_path_tc: Vec<f64> = Vec::new();
    let mut full_path_tr: Vec<f64> = Vec::new();

    for seg in 0..(waypoints.len() - 1) {
        let (sc, sr) = waypoints[seg];
        let (ec, er) = waypoints[seg + 1];
        let segment_path = dp_segment(q1n, q2n, argvals, sc, ec, sr, er, lambda);
        let start = usize::from(seg > 0);
        for &(tc, tr) in &segment_path[start..] {
            full_path_tc.push(argvals[tc]);
            full_path_tr.push(argvals[tr]);
        }
    }

    let mut gamma: Vec<f64> = argvals
        .iter()
        .map(|&t| linear_interp(&full_path_tc, &full_path_tr, t))
        .collect();
    normalize_warp(&mut gamma, argvals);
    gamma
}

/// Align f2 to f1 with landmark constraints.
///
/// Landmark pairs define waypoints on the DP grid. Between consecutive waypoints,
/// an independent smaller DP is run. The resulting warp passes through all landmarks.
///
/// # Arguments
/// * `f1` — Target curve (length m)
/// * `f2` — Curve to align (length m)
/// * `argvals` — Evaluation points (length m)
/// * `landmark_pairs` — `(target_t, source_t)` pairs in increasing order
/// * `lambda` — Penalty weight
///
/// # Returns
/// [`ConstrainedAlignmentResult`] with warp, aligned curve, and enforced landmarks.
pub fn elastic_align_pair_constrained(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    landmark_pairs: &[(f64, f64)],
    lambda: f64,
) -> ConstrainedAlignmentResult {
    let m = f1.len();

    if landmark_pairs.is_empty() {
        let r = elastic_align_pair(f1, f2, argvals, lambda);
        return ConstrainedAlignmentResult {
            gamma: r.gamma,
            f_aligned: r.f_aligned,
            distance: r.distance,
            enforced_landmarks: Vec::new(),
        };
    }

    // Compute & normalize SRSFs
    let f1_mat = FdMatrix::from_slice(f1, 1, m).expect("dimension invariant: data.len() == n * m");
    let f2_mat = FdMatrix::from_slice(f2, 1, m).expect("dimension invariant: data.len() == n * m");
    let q1_mat = srsf_transform(&f1_mat, argvals);
    let q2_mat = srsf_transform(&f2_mat, argvals);
    let q1: Vec<f64> = q1_mat.row(0);
    let q2: Vec<f64> = q2_mat.row(0);
    let norm1 = q1.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    let norm2 = q2.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    let q1n: Vec<f64> = q1.iter().map(|&v| v / norm1).collect();
    let q2n: Vec<f64> = q2.iter().map(|&v| v / norm2).collect();

    let waypoints = build_constrained_waypoints(landmark_pairs, argvals, m);
    let gamma = segmented_dp_gamma(&q1n, &q2n, argvals, &waypoints, lambda);

    let f_aligned = reparameterize_curve(f2, argvals, &gamma);
    let f_aligned_mat =
        FdMatrix::from_slice(&f_aligned, 1, m).expect("dimension invariant: data.len() == n * m");
    let q_aligned_mat = srsf_transform(&f_aligned_mat, argvals);
    let q_aligned: Vec<f64> = q_aligned_mat.row(0);
    let weights = simpsons_weights(argvals);
    let distance = l2_distance(&q1, &q_aligned, &weights);

    let enforced: Vec<(f64, f64)> = waypoints[1..waypoints.len() - 1]
        .iter()
        .map(|&(tc, tr)| (argvals[tc], argvals[tr]))
        .collect();

    ConstrainedAlignmentResult {
        gamma,
        f_aligned,
        distance,
        enforced_landmarks: enforced,
    }
}

/// Align f2 to f1 with automatic landmark detection and elastic constraints.
///
/// Detects landmarks in both curves, matches them, and uses the matches
/// as constraints for segmented DP alignment.
///
/// # Arguments
/// * `f1` — Target curve (length m)
/// * `f2` — Curve to align (length m)
/// * `argvals` — Evaluation points (length m)
/// * `kind` — Type of landmarks to detect
/// * `min_prominence` — Minimum prominence for landmark detection
/// * `expected_count` — Expected number of landmarks (0 = all detected)
/// * `lambda` — Penalty weight
pub fn elastic_align_pair_with_landmarks(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    kind: crate::landmark::LandmarkKind,
    min_prominence: f64,
    expected_count: usize,
    lambda: f64,
) -> ConstrainedAlignmentResult {
    let lm1 = crate::landmark::detect_landmarks(f1, argvals, kind, min_prominence);
    let lm2 = crate::landmark::detect_landmarks(f2, argvals, kind, min_prominence);

    // Match landmarks by order (take min count)
    let n_match = if expected_count > 0 {
        expected_count.min(lm1.len()).min(lm2.len())
    } else {
        lm1.len().min(lm2.len())
    };

    let pairs: Vec<(f64, f64)> = (0..n_match)
        .map(|i| (lm1[i].position, lm2[i].position))
        .collect();

    elastic_align_pair_constrained(f1, f2, argvals, &pairs, lambda)
}
