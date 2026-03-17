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
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::elastic_align_pair;
///
/// let argvals: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let f1: Vec<f64> = argvals.iter().map(|&t| (t * 6.0).sin()).collect();
/// let f2: Vec<f64> = argvals.iter().map(|&t| ((t + 0.1) * 6.0).sin()).collect();
/// let result = elastic_align_pair(&f1, &f2, &argvals, 0.0);
/// assert_eq!(result.f_aligned.len(), 20);
/// assert!(result.distance >= 0.0);
/// ```
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
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::elastic_distance;
///
/// let argvals: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let f1: Vec<f64> = argvals.iter().map(|&t| (t * 6.0).sin()).collect();
/// let f2: Vec<f64> = argvals.iter().map(|&t| ((t + 0.1) * 6.0).sin()).collect();
/// let d = elastic_distance(&f1, &f2, &argvals, 0.0);
/// assert!(d >= 0.0);
/// ```
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

// ─── Higher-Order Warp Penalties ─────────────────────────────────────────────

/// Penalty type for alignment regularization.
///
/// Controls how the warping function is penalized during alignment.
/// `FirstOrder` uses the standard DP penalty on slope deviation.
/// `SecondOrder` and `Combined` first run standard DP alignment, then
/// apply iterative Tikhonov smoothing to reduce warp curvature.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub enum WarpPenaltyType {
    /// Standard first-order penalty: lambda * (gamma' - 1)^2 * dt.
    #[default]
    FirstOrder,
    /// Second-order (curvature) penalty: standard DP + iterative curvature smoothing.
    SecondOrder,
    /// Combined first- and second-order: DP alignment + curvature smoothing
    /// weighted by `second_order_weight`.
    Combined {
        /// Relative weight of the curvature smoothing step (> 0).
        second_order_weight: f64,
    },
}

/// Number of Tikhonov smoothing iterations for second-order penalty.
const TIKHONOV_ITERS: usize = 8;

/// Apply Tikhonov curvature smoothing to a warping function.
///
/// Iteratively smooths toward the identity warp using Laplacian smoothing,
/// which reduces high-frequency curvature while preserving monotonicity
/// and boundary conditions. The smoothing weight `alpha` (clamped to [0,1])
/// controls how much each iteration pulls interior points toward the
/// midpoint of their neighbors.
fn tikhonov_smooth_gamma(gamma: &[f64], argvals: &[f64], alpha: f64, n_iter: usize) -> Vec<f64> {
    let m = gamma.len();
    if m < 3 || alpha <= 0.0 {
        return gamma.to_vec();
    }

    // Clamp effective weight to a stable range.
    let w = alpha.min(0.5);

    let mut gam = gamma.to_vec();

    for _ in 0..n_iter {
        let prev = gam.clone();

        // Laplacian smoothing: move each interior point toward the
        // midpoint of its neighbors, weighted by w.
        for j in 1..m - 1 {
            let mid = (prev[j - 1] + prev[j + 1]) / 2.0;
            gam[j] = prev[j] + w * (mid - prev[j]);
        }

        // Enforce boundary conditions.
        gam[0] = argvals[0];
        gam[m - 1] = argvals[m - 1];

        // Enforce monotonicity.
        crate::warping::normalize_warp(&mut gam, argvals);
    }

    gam
}

/// Align two curves with a configurable penalty type.
///
/// For [`WarpPenaltyType::FirstOrder`], this delegates directly to
/// [`elastic_align_pair`]. For [`WarpPenaltyType::SecondOrder`] and
/// [`WarpPenaltyType::Combined`], runs the standard DP alignment first,
/// then applies iterative Tikhonov smoothing to the warping function to
/// reduce curvature (gamma'') while preserving alignment quality.
///
/// # Arguments
/// * `f1` — Target curve (length m)
/// * `f2` — Curve to align (length m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — First-order penalty weight (passed to DP alignment)
/// * `penalty_type` — Which penalty type to apply
///
/// # Returns
/// [`AlignmentResult`] with warping function, aligned curve, and elastic distance.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{elastic_align_pair_penalized, WarpPenaltyType};
///
/// let argvals: Vec<f64> = (0..30).map(|i| i as f64 / 29.0).collect();
/// let f1: Vec<f64> = argvals.iter().map(|&t| (t * 6.0).sin()).collect();
/// let f2: Vec<f64> = argvals.iter().map(|&t| ((t + 0.1) * 6.0).sin()).collect();
///
/// // Standard first-order
/// let r1 = elastic_align_pair_penalized(&f1, &f2, &argvals, 0.0, WarpPenaltyType::FirstOrder);
/// assert!(r1.distance >= 0.0);
///
/// // Second-order smoothing
/// let r2 = elastic_align_pair_penalized(&f1, &f2, &argvals, 0.0, WarpPenaltyType::SecondOrder);
/// assert!(r2.distance >= 0.0);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_align_pair_penalized(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    lambda: f64,
    penalty_type: WarpPenaltyType,
) -> AlignmentResult {
    // Step 1: Run standard first-order DP alignment.
    let initial = elastic_align_pair(f1, f2, argvals, lambda);

    let smoothing_alpha = match penalty_type {
        WarpPenaltyType::FirstOrder => return initial,
        WarpPenaltyType::SecondOrder => lambda.max(0.01),
        WarpPenaltyType::Combined {
            second_order_weight,
        } => second_order_weight.max(1e-6),
    };

    // Step 2: Apply Tikhonov curvature smoothing to the warping function.
    let gamma_smooth =
        tikhonov_smooth_gamma(&initial.gamma, argvals, smoothing_alpha, TIKHONOV_ITERS);

    // Step 3: Recompute aligned curve and distance with smoothed gamma.
    let f_aligned = reparameterize_curve(f2, argvals, &gamma_smooth);
    let q1 = srsf_single(f1, argvals);
    let q_aligned = srsf_single(&f_aligned, argvals);
    let weights = simpsons_weights(argvals);
    let distance = l2_distance(&q1, &q_aligned, &weights);

    AlignmentResult {
        gamma: gamma_smooth,
        f_aligned,
        distance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    #[test]
    fn penalized_first_order_matches_standard() {
        let argvals = uniform_grid(30);
        let f1: Vec<f64> = argvals.iter().map(|&t| (t * 6.0).sin()).collect();
        let f2: Vec<f64> = argvals.iter().map(|&t| ((t + 0.1) * 6.0).sin()).collect();

        let standard = elastic_align_pair(&f1, &f2, &argvals, 0.0);
        let penalized =
            elastic_align_pair_penalized(&f1, &f2, &argvals, 0.0, WarpPenaltyType::FirstOrder);

        assert_eq!(standard.gamma, penalized.gamma);
        assert_eq!(standard.f_aligned, penalized.f_aligned);
        assert!((standard.distance - penalized.distance).abs() < 1e-12);
    }

    #[test]
    fn second_order_produces_valid_warp() {
        let argvals = uniform_grid(30);
        let f1: Vec<f64> = argvals.iter().map(|&t| (t * 6.0).sin()).collect();
        let f2: Vec<f64> = argvals.iter().map(|&t| ((t + 0.15) * 6.0).sin()).collect();

        let result =
            elastic_align_pair_penalized(&f1, &f2, &argvals, 0.1, WarpPenaltyType::SecondOrder);

        let m = argvals.len();
        assert_eq!(result.gamma.len(), m);
        assert_eq!(result.f_aligned.len(), m);
        assert!(result.distance >= 0.0);

        // Warp should be monotone non-decreasing.
        for j in 1..m {
            assert!(
                result.gamma[j] >= result.gamma[j - 1] - 1e-12,
                "gamma should be monotone at j={j}"
            );
        }

        // Boundary conditions.
        assert!((result.gamma[0] - argvals[0]).abs() < 1e-12);
        assert!((result.gamma[m - 1] - argvals[m - 1]).abs() < 1e-12);
    }

    #[test]
    fn combined_penalty_produces_valid_warp() {
        let argvals = uniform_grid(25);
        let f1: Vec<f64> = argvals.iter().map(|&t| (t * 4.0).sin()).collect();
        let f2: Vec<f64> = argvals.iter().map(|&t| ((t + 0.1) * 4.0).sin()).collect();

        let result = elastic_align_pair_penalized(
            &f1,
            &f2,
            &argvals,
            0.05,
            WarpPenaltyType::Combined {
                second_order_weight: 0.1,
            },
        );

        let m = argvals.len();
        assert_eq!(result.gamma.len(), m);
        assert!(result.distance >= 0.0);

        // Monotonicity.
        for j in 1..m {
            assert!(
                result.gamma[j] >= result.gamma[j - 1] - 1e-12,
                "gamma should be monotone at j={j}"
            );
        }
    }

    #[test]
    fn second_order_smoother_curvature() {
        let argvals = uniform_grid(40);
        let f1: Vec<f64> = argvals.iter().map(|&t| (t * 8.0).sin()).collect();
        let f2: Vec<f64> = argvals.iter().map(|&t| ((t + 0.2) * 8.0).sin()).collect();

        let first_order = elastic_align_pair(&f1, &f2, &argvals, 0.0);
        let second_order =
            elastic_align_pair_penalized(&f1, &f2, &argvals, 0.0, WarpPenaltyType::SecondOrder);

        // Compute bending energy (sum of squared second derivative).
        let bending = |g: &[f64]| -> f64 {
            let m = g.len();
            let mut energy = 0.0;
            for j in 1..m - 1 {
                let dt = argvals[j + 1] - argvals[j - 1];
                if dt > 0.0 {
                    let d2 = (g[j + 1] - 2.0 * g[j] + g[j - 1]) / (dt / 2.0).powi(2);
                    energy += d2 * d2 * dt / 2.0;
                }
            }
            energy
        };

        let be_first = bending(&first_order.gamma);
        let be_second = bending(&second_order.gamma);

        // Second-order penalty should reduce bending energy (or at least not
        // increase it much if the first-order warp is already smooth).
        assert!(
            be_second <= be_first + 1e-6,
            "second-order should reduce bending: first={be_first:.4}, second={be_second:.4}"
        );
    }

    #[test]
    fn warp_penalty_type_default_is_first_order() {
        let penalty: WarpPenaltyType = WarpPenaltyType::default();
        assert_eq!(penalty, WarpPenaltyType::FirstOrder);
    }
}
