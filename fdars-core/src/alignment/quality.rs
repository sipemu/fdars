//! Alignment quality metrics: warp complexity, smoothness, variance decomposition,
//! and pairwise consistency.

use super::pairwise::elastic_align_pair;
use super::srsf::compose_warps;
use super::KarcherMeanResult;
use crate::helpers::{gradient_uniform, l2_distance, simpsons_weights};
use crate::matrix::FdMatrix;

/// Comprehensive alignment quality assessment.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignmentQuality {
    /// Per-curve geodesic distance from warp to identity.
    pub warp_complexity: Vec<f64>,
    /// Mean warp complexity.
    pub mean_warp_complexity: f64,
    /// Per-curve bending energy ∫(γ'')² dt.
    pub warp_smoothness: Vec<f64>,
    /// Mean warp smoothness (bending energy).
    pub mean_warp_smoothness: f64,
    /// Total variance: (1/n) Σ ∫(f_i - mean_orig)² dt.
    pub total_variance: f64,
    /// Amplitude variance: (1/n) Σ ∫(f_i^aligned - mean_aligned)² dt.
    pub amplitude_variance: f64,
    /// Phase variance: total - amplitude (clamped ≥ 0).
    pub phase_variance: f64,
    /// Phase-to-total variance ratio.
    pub phase_amplitude_ratio: f64,
    /// Pointwise ratio: aligned_var / orig_var per time point.
    pub pointwise_variance_ratio: Vec<f64>,
    /// Mean variance reduction.
    pub mean_variance_reduction: f64,
}

/// Compute warp complexity: geodesic distance from a warp to the identity.
///
/// This is `arccos(⟨ψ, ψ_id⟩)` on the Hilbert sphere.
pub fn warp_complexity(gamma: &[f64], argvals: &[f64]) -> f64 {
    crate::warping::phase_distance(gamma, argvals)
}

/// Compute warp smoothness (bending energy): ∫(γ'')² dt.
pub fn warp_smoothness(gamma: &[f64], argvals: &[f64]) -> f64 {
    let m = gamma.len();
    if m < 3 {
        return 0.0;
    }

    let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let gam_prime = gradient_uniform(gamma, h);
    let gam_pprime = gradient_uniform(&gam_prime, h);

    let integrand: Vec<f64> = gam_pprime.iter().map(|&g| g * g).collect();
    crate::helpers::trapz(&integrand, argvals)
}

/// Compute comprehensive alignment quality metrics.
///
/// # Arguments
/// * `data` — Original functional data (n × m)
/// * `karcher` — Pre-computed Karcher mean result
/// * `argvals` — Evaluation points (length m)
pub fn alignment_quality(
    data: &FdMatrix,
    karcher: &KarcherMeanResult,
    argvals: &[f64],
) -> AlignmentQuality {
    let (n, m) = data.shape();
    let weights = simpsons_weights(argvals);

    // Per-curve warp complexity and smoothness
    let wc: Vec<f64> = (0..n)
        .map(|i| {
            let gamma: Vec<f64> = (0..m).map(|j| karcher.gammas[(i, j)]).collect();
            warp_complexity(&gamma, argvals)
        })
        .collect();
    let ws: Vec<f64> = (0..n)
        .map(|i| {
            let gamma: Vec<f64> = (0..m).map(|j| karcher.gammas[(i, j)]).collect();
            warp_smoothness(&gamma, argvals)
        })
        .collect();

    let mean_wc = wc.iter().sum::<f64>() / n as f64;
    let mean_ws = ws.iter().sum::<f64>() / n as f64;

    // Compute original mean
    let orig_mean = crate::fdata::mean_1d(data);

    // Total variance
    let total_var: f64 = (0..n)
        .map(|i| {
            let fi = data.row(i);
            let d = l2_distance(&fi, &orig_mean, &weights);
            d * d
        })
        .sum::<f64>()
        / n as f64;

    // Aligned mean
    let aligned_mean = crate::fdata::mean_1d(&karcher.aligned_data);

    // Amplitude variance
    let amp_var: f64 = (0..n)
        .map(|i| {
            let fi = karcher.aligned_data.row(i);
            let d = l2_distance(&fi, &aligned_mean, &weights);
            d * d
        })
        .sum::<f64>()
        / n as f64;

    let phase_var = (total_var - amp_var).max(0.0);
    let ratio = if total_var > 1e-10 {
        phase_var / total_var
    } else {
        0.0
    };

    // Pointwise variance ratio
    let mut pw_ratio = vec![0.0; m];
    for j in 0..m {
        let col_orig = data.column(j);
        let mean_orig_j = col_orig.iter().sum::<f64>() / n as f64;
        let var_orig: f64 = col_orig
            .iter()
            .map(|&v| (v - mean_orig_j).powi(2))
            .sum::<f64>()
            / n as f64;

        let col_aligned = karcher.aligned_data.column(j);
        let mean_aligned_j = col_aligned.iter().sum::<f64>() / n as f64;
        let var_aligned: f64 = col_aligned
            .iter()
            .map(|&v| (v - mean_aligned_j).powi(2))
            .sum::<f64>()
            / n as f64;

        pw_ratio[j] = if var_orig > 1e-15 {
            var_aligned / var_orig
        } else {
            1.0
        };
    }

    let mean_vr = pw_ratio.iter().sum::<f64>() / m as f64;

    AlignmentQuality {
        warp_complexity: wc,
        mean_warp_complexity: mean_wc,
        warp_smoothness: ws,
        mean_warp_smoothness: mean_ws,
        total_variance: total_var,
        amplitude_variance: amp_var,
        phase_variance: phase_var,
        phase_amplitude_ratio: ratio,
        pointwise_variance_ratio: pw_ratio,
        mean_variance_reduction: mean_vr,
    }
}

/// Generate triplet indices (i,j,k) with i<j<k, capped at `max_triplets` (0 = all).
fn triplet_indices(n: usize, max_triplets: usize) -> Vec<(usize, usize, usize)> {
    let total = n * (n - 1) * (n - 2) / 6;
    let cap = if max_triplets > 0 {
        max_triplets.min(total)
    } else {
        total
    };
    (0..n)
        .flat_map(|i| ((i + 1)..n).flat_map(move |j| ((j + 1)..n).map(move |k| (i, j, k))))
        .take(cap)
        .collect()
}

/// Compute the warp deviation for one triplet: ‖γ_ij∘γ_jk − γ_ik‖_L2.
fn triplet_warp_deviation(
    data: &FdMatrix,
    argvals: &[f64],
    weights: &[f64],
    i: usize,
    j: usize,
    k: usize,
    lambda: f64,
) -> f64 {
    let fi = data.row(i);
    let fj = data.row(j);
    let fk = data.row(k);
    let rij = elastic_align_pair(&fi, &fj, argvals, lambda);
    let rjk = elastic_align_pair(&fj, &fk, argvals, lambda);
    let rik = elastic_align_pair(&fi, &fk, argvals, lambda);
    let composed = compose_warps(&rij.gamma, &rjk.gamma, argvals);
    l2_distance(&composed, &rik.gamma, weights)
}

/// Measure pairwise alignment consistency via triplet checks.
///
/// For triplets (i,j,k), checks `γ_ij ∘ γ_jk ≈ γ_ik` by measuring the L2
/// deviation of the composed warp from the direct warp.
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight
/// * `max_triplets` — Maximum number of triplets to check (0 = all)
pub fn pairwise_consistency(
    data: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
    max_triplets: usize,
) -> f64 {
    let n = data.nrows();
    if n < 3 {
        return 0.0;
    }

    let weights = simpsons_weights(argvals);
    let triplets = triplet_indices(n, max_triplets);
    if triplets.is_empty() {
        return 0.0;
    }

    let total_dev: f64 = triplets
        .iter()
        .map(|&(i, j, k)| triplet_warp_deviation(data, argvals, &weights, i, j, k, lambda))
        .sum();
    total_dev / triplets.len() as f64
}
