//! Karcher (Frechet) mean computation in the elastic metric.

use super::set::apply_stored_warps;
use super::srsf::{reparameterize_curve, srsf_inverse, srsf_transform};
use super::{dp_alignment_core, KarcherMeanResult};
use crate::fdata::mean_1d;
use crate::helpers::{gradient_uniform, linear_interp};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::warping::{
    exp_map_sphere, gam_to_psi, inv_exp_map_sphere, invert_gamma, l2_norm_l2, psi_to_gam,
};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// Re-export srsf_single from srsf module for internal use
use super::srsf::srsf_single;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// One Karcher iteration on the Hilbert sphere: compute mean shooting vector and update mu.
///
/// Returns `true` if converged (vbar norm ≤ threshold).
fn karcher_sphere_step(mu: &mut Vec<f64>, psis: &[Vec<f64>], time: &[f64], step_size: f64) -> bool {
    let m = mu.len();
    let n = psis.len();
    let mut vbar = vec![0.0; m];
    for psi in psis {
        let v = inv_exp_map_sphere(mu, psi, time);
        for j in 0..m {
            vbar[j] += v[j];
        }
    }
    for j in 0..m {
        vbar[j] /= n as f64;
    }
    if l2_norm_l2(&vbar, time) <= 1e-8 {
        return true;
    }
    let scaled: Vec<f64> = vbar.iter().map(|&v| v * step_size).collect();
    *mu = exp_map_sphere(mu, &scaled, time);
    false
}

/// Karcher mean of warping functions on the Hilbert sphere, then invert.
/// Port of fdasrvf's `SqrtMeanInverse`.
pub(crate) fn sqrt_mean_inverse(gammas: &FdMatrix, argvals: &[f64]) -> Vec<f64> {
    let (n, m) = gammas.shape();
    let t0 = argvals[0];
    let t1 = argvals[m - 1];
    let domain = t1 - t0;

    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let binsize = 1.0 / (m - 1) as f64;

    let psis: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let gam_01: Vec<f64> = (0..m).map(|j| (gammas[(i, j)] - t0) / domain).collect();
            gam_to_psi(&gam_01, binsize)
        })
        .collect();

    let mut mu = vec![0.0; m];
    for psi in &psis {
        for j in 0..m {
            mu[j] += psi[j];
        }
    }
    for j in 0..m {
        mu[j] /= n as f64;
    }

    for _ in 0..501 {
        if karcher_sphere_step(&mut mu, &psis, &time, 0.3) {
            break;
        }
    }

    let gam_mu = psi_to_gam(&mu, &time);
    let gam_inv = invert_gamma(&gam_mu, &time);
    gam_inv.iter().map(|&g| t0 + g * domain).collect()
}

/// Compute relative change between successive mean SRSFs.
///
/// Returns `‖q_new - q_old‖₂ / ‖q_old‖₂`, matching R's fdasrvf
/// `time_warping` convergence metric (unweighted discrete L2 norm).
fn relative_change(q_old: &[f64], q_new: &[f64]) -> f64 {
    let diff_norm: f64 = q_old
        .iter()
        .zip(q_new.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let old_norm: f64 = q_old.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    diff_norm / old_norm
}

/// Align a single SRSF q2 to q1 and return (gamma, aligned_q).
pub(super) fn align_srsf_pair(
    q1: &[f64],
    q2: &[f64],
    argvals: &[f64],
    lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let gamma = dp_alignment_core(q1, q2, argvals, lambda);

    // Warp q2 by gamma and adjust by sqrt(gamma')
    let q2_warped = reparameterize_curve(q2, argvals, &gamma);

    // Compute gamma' via finite differences
    let m = gamma.len();
    let mut gamma_dot = vec![0.0; m];
    gamma_dot[0] = (gamma[1] - gamma[0]) / (argvals[1] - argvals[0]);
    for j in 1..(m - 1) {
        gamma_dot[j] = (gamma[j + 1] - gamma[j - 1]) / (argvals[j + 1] - argvals[j - 1]);
    }
    gamma_dot[m - 1] = (gamma[m - 1] - gamma[m - 2]) / (argvals[m - 1] - argvals[m - 2]);

    // q2_aligned = (q2 ∘ γ) * sqrt(γ')
    let q2_aligned: Vec<f64> = q2_warped
        .iter()
        .zip(gamma_dot.iter())
        .map(|(&q, &gd)| q * gd.max(0.0).sqrt())
        .collect();

    (gamma, q2_aligned)
}

/// Accumulate alignment results: store gammas and return the mean of aligned SRSFs.
fn accumulate_alignments(
    results: &[(Vec<f64>, Vec<f64>)],
    gammas: &mut FdMatrix,
    m: usize,
    n: usize,
) -> Vec<f64> {
    let mut mu_q_new = vec![0.0; m];
    for (i, (gamma, q_aligned)) in results.iter().enumerate() {
        for j in 0..m {
            gammas[(i, j)] = gamma[j];
            mu_q_new[j] += q_aligned[j];
        }
    }
    for j in 0..m {
        mu_q_new[j] /= n as f64;
    }
    mu_q_new
}

/// Select the SRSF closest to the pointwise mean as template. Returns (mu_q, mu_f).
fn select_template(srsf_mat: &FdMatrix, data: &FdMatrix, argvals: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (n, m) = srsf_mat.shape();
    let mnq = mean_1d(srsf_mat);
    let mut min_dist = f64::INFINITY;
    let mut min_idx = 0;
    for i in 0..n {
        let dist_sq: f64 = (0..m).map(|j| (srsf_mat[(i, j)] - mnq[j]).powi(2)).sum();
        if dist_sq < min_dist {
            min_dist = dist_sq;
            min_idx = i;
        }
    }
    let _ = argvals; // kept for API consistency
    (srsf_mat.row(min_idx), data.row(min_idx))
}

/// Pre-centering: align all curves to template, compute inverse mean warp, re-center.
fn pre_center_template(
    data: &FdMatrix,
    mu_q: &[f64],
    mu: &[f64],
    argvals: &[f64],
    lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (n, m) = data.shape();
    let align_results: Vec<(Vec<f64>, Vec<f64>)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let fi = data.row(i);
            let qi = srsf_single(&fi, argvals);
            align_srsf_pair(mu_q, &qi, argvals, lambda)
        })
        .collect();

    let mut init_gammas = FdMatrix::zeros(n, m);
    for (i, (gamma, _)) in align_results.iter().enumerate() {
        for j in 0..m {
            init_gammas[(i, j)] = gamma[j];
        }
    }

    let gam_inv = sqrt_mean_inverse(&init_gammas, argvals);
    let mu_new = reparameterize_curve(mu, argvals, &gam_inv);
    let mu_q_new = srsf_single(&mu_new, argvals);
    (mu_q_new, mu_new)
}

/// Post-convergence centering: center mean SRSF and warps via SqrtMeanInverse.
fn post_center_results(
    data: &FdMatrix,
    mu_q: &[f64],
    final_gammas: &mut FdMatrix,
    argvals: &[f64],
) -> (Vec<f64>, Vec<f64>, FdMatrix) {
    let (n, m) = data.shape();
    let gam_inv = sqrt_mean_inverse(final_gammas, argvals);
    let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let gam_inv_dev = gradient_uniform(&gam_inv, h);

    let mu_q_warped = reparameterize_curve(mu_q, argvals, &gam_inv);
    let mu_q_centered: Vec<f64> = mu_q_warped
        .iter()
        .zip(gam_inv_dev.iter())
        .map(|(&q, &gd)| q * gd.max(0.0).sqrt())
        .collect();

    for i in 0..n {
        let gam_i: Vec<f64> = (0..m).map(|j| final_gammas[(i, j)]).collect();
        let gam_centered = reparameterize_curve(&gam_i, argvals, &gam_inv);
        for j in 0..m {
            final_gammas[(i, j)] = gam_centered[j];
        }
    }

    let initial_mean = mean_1d(data);
    let mu = srsf_inverse(&mu_q_centered, argvals, initial_mean[0]);
    let final_aligned = apply_stored_warps(data, final_gammas, argvals);
    (mu, mu_q_centered, final_aligned)
}

/// Downsample argvals and signal by `factor`, keeping first and last points.
fn downsample_uniform(signal: &[f64], argvals: &[f64], factor: usize) -> (Vec<f64>, Vec<f64>) {
    let m = signal.len();
    if factor <= 1 || m <= 2 {
        return (signal.to_vec(), argvals.to_vec());
    }
    let mut sig = Vec::new();
    let mut arg = Vec::new();
    for i in (0..m).step_by(factor) {
        sig.push(signal[i]);
        arg.push(argvals[i]);
    }
    // Ensure last point is included
    if (m - 1) % factor != 0 {
        sig.push(signal[m - 1]);
        arg.push(argvals[m - 1]);
    }
    (sig, arg)
}

/// Upsample signal from coarse grid to fine grid via linear interpolation.
fn upsample_to_fine(coarse: &[f64], argvals_coarse: &[f64], argvals_fine: &[f64]) -> Vec<f64> {
    argvals_fine
        .iter()
        .map(|&t| linear_interp(argvals_coarse, coarse, t))
        .collect()
}

// ─── Karcher Mean ───────────────────────────────────────────────────────────

/// Compute the Karcher (Frechet) mean in the elastic metric.
///
/// Iteratively aligns all curves to the current mean estimate in SRSF space,
/// computes the pointwise mean of aligned SRSFs, and reconstructs the mean curve.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `max_iter` — Maximum number of iterations
/// * `tol` — Convergence tolerance for the SRSF mean
///
/// # Returns
/// [`KarcherMeanResult`] with mean curve, warping functions, aligned data, and convergence info.
///
/// # Examples
///
/// ```
/// use fdars_core::simulation::{sim_fundata, EFunType, EValType};
/// use fdars_core::alignment::karcher_mean;
///
/// let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
/// let data = sim_fundata(20, &t, 3, EFunType::Fourier, EValType::Exponential, Some(42));
///
/// let result = karcher_mean(&data, &t, 20, 1e-4, 0.0);
/// assert_eq!(result.mean.len(), 50);
/// assert!(result.n_iter <= 20);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn karcher_mean(
    data: &FdMatrix,
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
    lambda: f64,
) -> KarcherMeanResult {
    let (n, m) = data.shape();

    let srsf_mat = srsf_transform(data, argvals);
    let (mut mu_q, mu) = select_template(&srsf_mat, data, argvals);
    let (mu_q_c, mu_c) = pre_center_template(data, &mu_q, &mu, argvals, lambda);
    mu_q = mu_q_c;
    let mut mu = mu_c;

    let mut converged = false;
    let mut n_iter = 0;
    let mut final_gammas = FdMatrix::zeros(n, m);

    // Coarse-to-fine strategy: run initial iterations on downsampled grid
    // Only worthwhile for large grids with enough iterations to split
    let coarse_factor = if m > 50 && max_iter >= 10 { 4 } else { 1 };
    let coarse_iters = if coarse_factor > 1 { max_iter / 2 } else { 0 };
    let fine_iters = max_iter - coarse_iters;

    // Phase 1: coarse iterations
    if coarse_iters > 0 {
        let (mu_q_coarse, argvals_coarse) = downsample_uniform(&mu_q, argvals, coarse_factor);
        let m_c = argvals_coarse.len();
        let mut mu_q_c = mu_q_coarse;

        // Downsample all curves to coarse grid
        let data_coarse: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let row = data.row(i);
                downsample_uniform(&row, argvals, coarse_factor).0
            })
            .collect();

        let mut coarse_gammas = FdMatrix::zeros(n, m_c);

        for iter in 0..coarse_iters {
            n_iter = iter + 1;

            let align_results: Vec<(Vec<f64>, Vec<f64>)> = iter_maybe_parallel!(0..n)
                .map(|i| {
                    let qi = srsf_single(&data_coarse[i], &argvals_coarse);
                    align_srsf_pair(&mu_q_c, &qi, &argvals_coarse, lambda)
                })
                .collect();

            let mu_q_new = accumulate_alignments(&align_results, &mut coarse_gammas, m_c, n);

            let rel = relative_change(&mu_q_c, &mu_q_new);
            if rel < tol {
                converged = true;
                mu_q_c = mu_q_new;
                break;
            }

            mu_q_c = mu_q_new;
        }

        // Upsample coarse mu_q to fine grid
        mu_q = upsample_to_fine(&mu_q_c, &argvals_coarse, argvals);
        mu = srsf_inverse(&mu_q, argvals, mu[0]);
    }

    // Phase 2: fine iterations (or all iterations if m <= 50)
    if fine_iters > 0 {
        converged = false; // Fine phase must independently converge
    }
    let fine_start = n_iter;
    for iter in 0..fine_iters {
        n_iter = fine_start + iter + 1;

        let align_results: Vec<(Vec<f64>, Vec<f64>)> = iter_maybe_parallel!(0..n)
            .map(|i| {
                let fi = data.row(i);
                let qi = srsf_single(&fi, argvals);
                align_srsf_pair(&mu_q, &qi, argvals, lambda)
            })
            .collect();

        let mu_q_new = accumulate_alignments(&align_results, &mut final_gammas, m, n);

        let rel = relative_change(&mu_q, &mu_q_new);
        if rel < tol {
            converged = true;
            mu_q = mu_q_new;
            break;
        }

        mu_q = mu_q_new;
        mu = srsf_inverse(&mu_q, argvals, mu[0]);
    }

    // If coarse converged but no fine iterations ran, do one fine pass for final_gammas
    if converged && fine_start > 0 {
        let align_results: Vec<(Vec<f64>, Vec<f64>)> = iter_maybe_parallel!(0..n)
            .map(|i| {
                let fi = data.row(i);
                let qi = srsf_single(&fi, argvals);
                align_srsf_pair(&mu_q, &qi, argvals, lambda)
            })
            .collect();
        let mu_q_new = accumulate_alignments(&align_results, &mut final_gammas, m, n);
        mu_q = mu_q_new;
    }

    let (mu_final, mu_q_final, final_aligned) =
        post_center_results(data, &mu_q, &mut final_gammas, argvals);

    KarcherMeanResult {
        mean: mu_final,
        mean_srsf: mu_q_final,
        gammas: final_gammas,
        aligned_data: final_aligned,
        n_iter,
        converged,
        aligned_srsfs: None,
    }
}
