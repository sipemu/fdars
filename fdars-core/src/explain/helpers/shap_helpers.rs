//! SHAP and Friedman H-statistic helpers.

use crate::matrix::FdMatrix;
use crate::scalar_on_function::cholesky_factor;
use crate::scalar_on_function::cholesky_forward_back;
use rand::prelude::*;

/// Binomial coefficient C(n, k).
fn binom_coeff(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result: usize = 1;
    for i in 0..k {
        result = result.saturating_mul(n - i) / (i + 1);
    }
    result
}

/// Compute Shapley kernel weight for a coalition of given size.
pub(crate) fn shapley_kernel_weight(ncomp: usize, s_size: usize) -> f64 {
    if s_size == 0 || s_size == ncomp {
        1e6
    } else {
        let binom = binom_coeff(ncomp, s_size) as f64;
        if binom > 0.0 {
            (ncomp - 1) as f64 / (binom * s_size as f64 * (ncomp - s_size) as f64)
        } else {
            1.0
        }
    }
}

/// Sample a random coalition of FPC components via Fisher-Yates partial shuffle.
pub(crate) fn sample_random_coalition(rng: &mut StdRng, ncomp: usize) -> (Vec<bool>, usize) {
    let s_size = if ncomp <= 1 {
        rng.gen_range(0..=1usize)
    } else {
        rng.gen_range(1..ncomp)
    };
    let mut coalition = vec![false; ncomp];
    let mut indices: Vec<usize> = (0..ncomp).collect();
    for j in 0..s_size.min(ncomp) {
        let swap = rng.gen_range(j..ncomp);
        indices.swap(j, swap);
    }
    for j in 0..s_size {
        coalition[indices[j]] = true;
    }
    (coalition, s_size)
}

/// Build coalition scores: use observation value if in coalition, mean otherwise.
pub(crate) fn build_coalition_scores(
    coalition: &[bool],
    obs_scores: &[f64],
    mean_scores: &[f64],
) -> Vec<f64> {
    coalition
        .iter()
        .enumerate()
        .map(|(k, &in_coal)| {
            if in_coal {
                obs_scores[k]
            } else {
                mean_scores[k]
            }
        })
        .collect()
}

/// Get observation's scalar covariates, or use mean if unavailable.
pub(crate) fn get_obs_scalar(
    scalar_covariates: Option<&FdMatrix>,
    i: usize,
    p_scalar: usize,
    mean_z: &[f64],
) -> Vec<f64> {
    if p_scalar == 0 {
        return vec![];
    }
    if let Some(sc) = scalar_covariates {
        (0..p_scalar).map(|j| sc[(i, j)]).collect()
    } else {
        mean_z.to_vec()
    }
}

/// Accumulate one WLS sample for Kernel SHAP: A'A += w z z', A'b += w z y.
pub(crate) fn accumulate_kernel_shap_sample(
    ata: &mut [f64],
    atb: &mut [f64],
    coalition: &[bool],
    weight: f64,
    y_val: f64,
    ncomp: usize,
) {
    for k1 in 0..ncomp {
        let z1 = if coalition[k1] { 1.0 } else { 0.0 };
        for k2 in 0..ncomp {
            let z2 = if coalition[k2] { 1.0 } else { 0.0 };
            ata[k1 * ncomp + k2] += weight * z1 * z2;
        }
        atb[k1] += weight * z1 * y_val;
    }
}

/// Solve Kernel SHAP for one observation: regularize ATA, Cholesky solve, store in values matrix.
pub(crate) fn solve_kernel_shap_obs(
    ata: &mut [f64],
    atb: &[f64],
    ncomp: usize,
    values: &mut FdMatrix,
    i: usize,
) {
    for k in 0..ncomp {
        ata[k * ncomp + k] += 1e-10;
    }
    if let Some(l) = cholesky_factor(ata, ncomp) {
        let phi = cholesky_forward_back(&l, atb, ncomp);
        for k in 0..ncomp {
            values[(i, k)] = phi[k];
        }
    }
}

/// Compute H^2 statistic from 1D and 2D PDPs.
pub(crate) fn compute_h_squared(
    pdp_2d: &FdMatrix,
    pdp_j: &[f64],
    pdp_k: &[f64],
    f_bar: f64,
    n_grid: usize,
) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for gj in 0..n_grid {
        for gk in 0..n_grid {
            let f2 = pdp_2d[(gj, gk)];
            let interaction = f2 - pdp_j[gj] - pdp_k[gk] + f_bar;
            num += interaction * interaction;
            let centered = f2 - f_bar;
            den += centered * centered;
        }
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}
