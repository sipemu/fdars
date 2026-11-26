//! Alignment for periodic/closed curves where the starting point can shift.
//!
//! Closed curves have a natural ambiguity in the choice of starting point.
//! These functions optimize over circular rotations of the parameterization
//! in addition to the standard elastic warping.

use super::dp_alignment_core;
use super::optim::{check_grid, optimal_warp, WarpOptimConfig};
use super::pairwise::{elastic_align_pair, elastic_align_pair_with_config};
use super::srsf::{reparameterize_curve, srsf_single};
use crate::error::FdarError;
use crate::helpers::{l2_distance, simpsons_weights};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Result of aligning one closed curve to another.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ClosedAlignmentResult {
    /// Warping function on the domain.
    pub gamma: Vec<f64>,
    /// The aligned (reparameterized and rotated) curve.
    pub f_aligned: Vec<f64>,
    /// Elastic distance after alignment.
    pub distance: f64,
    /// Optimal circular shift index for the source curve.
    pub optimal_rotation: usize,
}

/// Result of computing the Karcher mean for closed curves.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ClosedKarcherMeanResult {
    /// Karcher mean curve.
    pub mean: Vec<f64>,
    /// SRSF of the Karcher mean.
    pub mean_srsf: Vec<f64>,
    /// Final warping functions (n x m).
    pub gammas: FdMatrix,
    /// Curves aligned to the mean (n x m).
    pub aligned_data: FdMatrix,
    /// Per-curve optimal rotations.
    pub rotations: Vec<usize>,
    /// Number of iterations used.
    pub n_iter: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Circularly shift a curve by `k` positions.
fn circular_shift(f: &[f64], k: usize) -> Vec<f64> {
    let m = f.len();
    if m == 0 || k == 0 {
        return f.to_vec();
    }
    let k = k % m;
    (0..m).map(|j| f[(j + k) % m]).collect()
}

/// Evaluate `dist(k)` for every candidate rotation, in parallel when the
/// `parallel` feature is on. Results come back in candidate order, and the
/// caller scans them sequentially with the same strict `<` rule as a plain
/// loop, so the selected rotation (first minimum) is identical either way.
fn candidate_distances<F>(cands: &[usize], dist: &F) -> Vec<f64>
where
    F: Fn(usize) -> f64 + Sync,
{
    iter_maybe_parallel!(cands).map(|&k| dist(k)).collect()
}

/// Find the best circular rotation of `f2` to match `f1`, using a coarse-then-fine strategy.
///
/// The SRSF of the (fixed) target `f1` and the Simpson integration weights are
/// loop invariants, so they are computed once and shared by every candidate
/// rotation (ALN-HYGIENE #56). Both are pure functions of `f1` / `argvals`, so
/// the result is bit-identical to recomputing them per candidate. Candidate
/// rotations are evaluated in parallel (see [`candidate_distances`]).
///
/// Returns `(best_rotation, best_distance)`.
fn find_best_rotation(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64) -> (usize, f64) {
    find_best_rotation_with(f1, f2, argvals, |q1, q2| {
        dp_alignment_core(q1, q2, argvals, lambda)
    })
}

/// [`find_best_rotation`] with an arbitrary warp optimiser
/// `warp(q1, q2) -> γ` (the classic DP closure reproduces it exactly).
fn find_best_rotation_with<W>(f1: &[f64], f2: &[f64], argvals: &[f64], warp: W) -> (usize, f64)
where
    W: Fn(&[f64], &[f64]) -> Vec<f64> + Sync,
{
    let m = f1.len();
    if m < 2 {
        return (0, 0.0);
    }

    let q1 = srsf_single(f1, argvals);
    let weights = simpsons_weights(argvals);
    let rotation_distance = |k: usize| -> f64 {
        let f2_rot = circular_shift(f2, k);
        let q2 = srsf_single(&f2_rot, argvals);
        let gamma = warp(&q1, &q2);
        let f_aligned = reparameterize_curve(&f2_rot, argvals, &gamma);
        let q_aligned = srsf_single(&f_aligned, argvals);
        l2_distance(&q1, &q_aligned, &weights)
    };

    // Coarse search: try m/step_size evenly spaced rotations
    let step_size = (m / 20).max(1);
    let mut best_k = 0;
    let mut best_dist = f64::INFINITY;

    let coarse: Vec<usize> = (0..m).step_by(step_size).collect();
    let coarse_d = candidate_distances(&coarse, &rotation_distance);
    for (&k, &dist) in coarse.iter().zip(&coarse_d) {
        if dist < best_dist {
            best_dist = dist;
            best_k = k;
        }
    }

    // Fine search: refine around best coarse rotation
    let search_start = best_k.saturating_sub(step_size);
    let search_end = (best_k + step_size).min(m);

    // Skip rotations already checked in the coarse pass.
    let fine: Vec<usize> = (search_start..search_end)
        .filter(|k| k % step_size != 0)
        .collect();
    let fine_d = candidate_distances(&fine, &rotation_distance);
    for (&k, &dist) in fine.iter().zip(&fine_d) {
        if dist < best_dist {
            best_dist = dist;
            best_k = k;
        }
    }

    (best_k, best_dist)
}

// ─── Karcher Mean Helpers ────────────────────────────────────────────────────

/// Align all curves to `mu` using closed alignment and accumulate the new SRSF mean.
///
/// Returns `(mu_q_new, align_results)` where `mu_q_new` is the averaged SRSF and
/// `align_results` carries per-curve `(ClosedAlignmentResult, q_aligned)` pairs for
/// the caller to store gammas/rotations.
fn karcher_iter_step(
    data: &FdMatrix,
    mu: &[f64],
    argvals: &[f64],
    lambda: f64,
    n: usize,
    m: usize,
) -> (Vec<f64>, Vec<(ClosedAlignmentResult, Vec<f64>)>) {
    let align_results: Vec<(ClosedAlignmentResult, Vec<f64>)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let fi = data.row(i);
            let res = elastic_align_pair_closed(mu, &fi, argvals, lambda)
                .expect("dimension invariant: all curves have length m");
            let q_warped = srsf_single(&res.f_aligned, argvals);
            (res, q_warped)
        })
        .collect();

    let mut mu_q_new = vec![0.0; m];
    for (_, q_aligned) in &align_results {
        for j in 0..m {
            mu_q_new[j] += q_aligned[j];
        }
    }
    for v in &mut mu_q_new {
        *v /= n as f64;
    }

    (mu_q_new, align_results)
}

/// Store per-curve gammas and rotations from a completed alignment step.
fn store_align_results(
    gammas: &mut FdMatrix,
    rotations: &mut [usize],
    align_results: &[(ClosedAlignmentResult, Vec<f64>)],
    m: usize,
) {
    for (i, (res, _)) in align_results.iter().enumerate() {
        for j in 0..m {
            gammas[(i, j)] = res.gamma[j];
        }
        rotations[i] = res.optimal_rotation;
    }
}

// ─── Public Functions ───────────────────────────────────────────────────────

/// Align closed curve `f2` to closed curve `f1` with rotation search.
///
/// For closed (periodic) curves, the starting point is arbitrary. This function
/// searches over circular shifts of `f2` to find the rotation that minimizes
/// the elastic distance, then performs full elastic alignment at that rotation.
///
/// # Arguments
/// * `f1` - Target curve (length m)
/// * `f2` - Curve to align (length m)
/// * `argvals` - Evaluation points (length m)
/// * `lambda` - Penalty weight on warp deviation from identity (0.0 = no penalty)
///
/// # Errors
/// Returns `FdarError::InvalidDimension` if lengths do not match or m < 2.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_align_pair_closed(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    lambda: f64,
) -> Result<ClosedAlignmentResult, FdarError> {
    let m = f1.len();
    if m != f2.len() || m != argvals.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "f1/f2/argvals",
            expected: format!("equal lengths, f1 has {m}"),
            actual: format!("f2 has {}, argvals has {}", f2.len(), argvals.len()),
        });
    }
    if m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "f1",
            expected: "length >= 2".to_string(),
            actual: format!("length {m}"),
        });
    }

    let (best_k, _) = find_best_rotation(f1, f2, argvals, lambda);

    // Full alignment at the best rotation
    let f2_rotated = circular_shift(f2, best_k);
    let result = elastic_align_pair(f1, &f2_rotated, argvals, lambda);

    Ok(ClosedAlignmentResult {
        gamma: result.gamma,
        f_aligned: result.f_aligned,
        distance: result.distance,
        optimal_rotation: best_k,
    })
}

/// Compute the elastic distance between two closed curves.
///
/// Optimizes over circular rotations of `f2` to find the minimum elastic distance.
///
/// # Arguments
/// * `f1` - First curve (length m)
/// * `f2` - Second curve (length m)
/// * `argvals` - Evaluation points (length m)
/// * `lambda` - Penalty weight on warp deviation from identity (0.0 = no penalty)
///
/// # Errors
/// Returns `FdarError::InvalidDimension` if lengths do not match or m < 2.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_distance_closed(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    lambda: f64,
) -> Result<f64, FdarError> {
    Ok(elastic_align_pair_closed(f1, f2, argvals, lambda)?.distance)
}

/// [`elastic_align_pair_closed`] with a configurable warp optimiser
/// ([`WarpOptimConfig`]): the circular-shift search and the final alignment
/// both use `config`'s optimiser and penalty.
///
/// The default DP leaves a grid-independent residual on smoothly warped
/// copies because its warp slopes are quantised to `{dr/dc : dr, dc ≤ 7}`
/// (GH #65); [`WarpOptimMethod::Rbfgs`](super::WarpOptimMethod::Rbfgs) or
/// `nbhd_dim = 10` + [`DpEdgeCost::PiecewiseLinear`](super::DpEdgeCost::PiecewiseLinear)
/// lower it. [`WarpOptimConfig::default`] with `lambda` set gives exactly
/// [`elastic_align_pair_closed`].
///
/// # Errors
/// [`FdarError::InvalidDimension`] if lengths differ or `m < 2`;
/// [`FdarError::InvalidParameter`] if `argvals` is not strictly increasing or
/// `config` is invalid ([`WarpOptimConfig::validate`]).
///
/// # Example
/// ```
/// use fdars_core::alignment::{
///     elastic_align_pair_closed_with_config, WarpOptimConfig, WarpOptimMethod,
/// };
/// let m = 60;
/// let t: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
/// let tau = 2.0 * std::f64::consts::PI;
/// let f1: Vec<f64> = t.iter().map(|&s| (tau * s).sin() + 0.5 * (2.0 * tau * s).cos()).collect();
/// // same periodic curve, started 9 samples later
/// let f2: Vec<f64> = (0..m).map(|j| f1[(j + 9) % (m - 1)]).collect();
/// let mut cfg = WarpOptimConfig::default();
/// cfg.method = WarpOptimMethod::Rbfgs;
/// let res = elastic_align_pair_closed_with_config(&f1, &f2, &t, &cfg).unwrap();
/// assert!(res.distance.is_finite());
/// assert!(res.gamma.windows(2).all(|w| w[1] >= w[0]));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_align_pair_closed_with_config(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    config: &WarpOptimConfig,
) -> Result<ClosedAlignmentResult, FdarError> {
    check_grid(argvals, f1.len(), "f1")?;
    check_grid(argvals, f2.len(), "f2")?;
    config.validate()?;
    let (best_k, _) = find_best_rotation_with(f1, f2, argvals, |q1, q2| {
        optimal_warp(q1, q2, argvals, config, None)
    });
    let f2_rotated = circular_shift(f2, best_k);
    let result = elastic_align_pair_with_config(f1, &f2_rotated, argvals, config)?;
    Ok(ClosedAlignmentResult {
        gamma: result.gamma,
        f_aligned: result.f_aligned,
        distance: result.distance,
        optimal_rotation: best_k,
    })
}

/// Elastic distance between two closed curves under a configurable warp
/// optimiser (see [`elastic_align_pair_closed_with_config`]).
///
/// # Errors
/// As [`elastic_align_pair_closed_with_config`].
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_distance_closed_with_config(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    config: &WarpOptimConfig,
) -> Result<f64, FdarError> {
    Ok(elastic_align_pair_closed_with_config(f1, f2, argvals, config)?.distance)
}

/// Compute the Karcher (Frechet) mean for closed curves.
///
/// Uses the standard Karcher mean iteration but with [`elastic_align_pair_closed`]
/// at each step, tracking per-curve optimal rotations.
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
/// * `argvals` - Evaluation points (length m)
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance for the SRSF mean
/// * `lambda` - Penalty weight on warp deviation from identity (0.0 = no penalty)
///
/// # Errors
/// Returns `FdarError::InvalidDimension` if dimensions are inconsistent or m < 2.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn karcher_mean_closed(
    data: &FdMatrix,
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
    lambda: f64,
) -> Result<ClosedKarcherMeanResult, FdarError> {
    let (n, m) = data.shape();
    if m != argvals.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {m}"),
            actual: format!("length {}", argvals.len()),
        });
    }
    if m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "ncols >= 2".to_string(),
            actual: format!("ncols = {m}"),
        });
    }
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "nrows > 0".to_string(),
            actual: "nrows = 0".to_string(),
        });
    }

    // Initialize mean as the first curve
    let mut mu: Vec<f64> = data.row(0);
    let mut mu_q = srsf_single(&mu, argvals);

    let mut gammas = FdMatrix::zeros(n, m);
    let mut rotations = vec![0usize; n];
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        // Align all curves to current mean and accumulate new SRSF mean
        let (mu_q_new, align_results) = karcher_iter_step(data, &mu, argvals, lambda, n, m);
        store_align_results(&mut gammas, &mut rotations, &align_results, m);

        // Check convergence
        let diff_norm: f64 = mu_q
            .iter()
            .zip(mu_q_new.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let old_norm: f64 = mu_q.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
        let rel = diff_norm / old_norm;

        mu_q = mu_q_new;

        if rel < tol {
            converged = true;
            break;
        }

        // Reconstruct mean curve from SRSF
        mu = crate::alignment::srsf::srsf_inverse(&mu_q, argvals, mu[0]);
    }

    // Compute final aligned data
    let mut aligned_data = FdMatrix::zeros(n, m);
    for i in 0..n {
        let fi = data.row(i);
        let f_rotated = circular_shift(&fi, rotations[i]);
        let gamma_i: Vec<f64> = (0..m).map(|j| gammas[(i, j)]).collect();
        let f_aligned = reparameterize_curve(&f_rotated, argvals, &gamma_i);
        for j in 0..m {
            aligned_data[(i, j)] = f_aligned[j];
        }
    }

    // Reconstruct final mean from SRSF
    mu = crate::alignment::srsf::srsf_inverse(&mu_q, argvals, mu[0]);

    Ok(ClosedKarcherMeanResult {
        mean: mu,
        mean_srsf: mu_q,
        gammas,
        aligned_data,
        rotations,
        n_iter,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    #[test]
    fn closed_align_identity() {
        let m = 30;
        let argvals = uniform_grid(m);
        let f: Vec<f64> = argvals
            .iter()
            .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
            .collect();

        let result = elastic_align_pair_closed(&f, &f, &argvals, 0.0).unwrap();
        assert!(
            result.distance < 0.1,
            "identical closed curves should have near-zero distance, got {}",
            result.distance
        );
        assert_eq!(
            result.optimal_rotation, 0,
            "identical curves should need no rotation"
        );
    }

    #[test]
    fn closed_align_shifted() {
        // Use a non-periodic curve shape so the shift is unambiguous
        let m = 40;
        let argvals = uniform_grid(m);
        let f1: Vec<f64> = argvals
            .iter()
            .map(|&t| (2.0 * std::f64::consts::PI * t).sin() + 0.5 * t)
            .collect();
        // Circularly shift f1 by 5 positions (small shift for reliable recovery)
        let shift = 5;
        let f2 = circular_shift(&f1, shift);

        let result = elastic_align_pair_closed(&f1, &f2, &argvals, 0.0).unwrap();
        // Distance should be small after alignment (shifted version of same curve)
        assert!(
            result.distance < 1.0,
            "distance after closed alignment should be small, got {}",
            result.distance
        );
    }

    #[test]
    fn closed_distance_symmetric() {
        let m = 25;
        let argvals = uniform_grid(m);
        let f1: Vec<f64> = argvals
            .iter()
            .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
            .collect();
        let f2: Vec<f64> = argvals
            .iter()
            .map(|&t| (2.0 * std::f64::consts::PI * t).cos())
            .collect();

        let d12 = elastic_distance_closed(&f1, &f2, &argvals, 0.0).unwrap();
        let d21 = elastic_distance_closed(&f2, &f1, &argvals, 0.0).unwrap();

        // Both distances should be finite and non-negative
        assert!(
            d12 >= 0.0 && d12.is_finite(),
            "d12 should be non-negative finite, got {d12}"
        );
        assert!(
            d21 >= 0.0 && d21.is_finite(),
            "d21 should be non-negative finite, got {d21}"
        );
        // Closed curve alignment is not perfectly symmetric due to the discrete
        // rotation search, but both distances should be in a reasonable range
        assert!(
            d12.max(d21) < 2.0 * d12.min(d21) + 0.5,
            "closed distances should be in comparable range: d12={d12:.4}, d21={d21:.4}"
        );
    }

    #[test]
    fn closed_karcher_mean_smoke() {
        let n = 5;
        let m = 25;
        let argvals = uniform_grid(m);

        // Create 5 shifted sine curves
        let mut data_flat = vec![0.0; n * m];
        for i in 0..n {
            let shift = i as f64 * 0.1;
            for j in 0..m {
                let t = argvals[j];
                data_flat[i + j * n] = (2.0 * std::f64::consts::PI * (t + shift)).sin();
            }
        }
        let data = FdMatrix::from_column_major(data_flat, n, m).unwrap();

        let result = karcher_mean_closed(&data, &argvals, 10, 1e-3, 0.0).unwrap();
        assert_eq!(result.mean.len(), m);
        assert_eq!(result.mean_srsf.len(), m);
        assert_eq!(result.gammas.shape(), (n, m));
        assert_eq!(result.aligned_data.shape(), (n, m));
        assert_eq!(result.rotations.len(), n);
        assert!(result.n_iter <= 10);
    }
}
