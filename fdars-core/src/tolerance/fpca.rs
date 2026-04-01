use super::helpers::{build_band, normal_quantile, percentile_sorted, pointwise_mean_std};
use super::BandType;
use super::ToleranceBand;
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;
use rand::prelude::*;
use rand_distr::StandardNormal;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─── FPCA + Bootstrap Tolerance Band ────────────────────────────────────────

/// Per-component score statistics needed for bootstrap sampling.
pub(super) struct ScoreStats {
    pub(super) means: Vec<f64>,
    pub(super) stds: Vec<f64>,
}

/// Compute per-component score mean and std from FPCA results.
pub(super) fn compute_score_stats(scores: &FdMatrix, n: usize) -> ScoreStats {
    let ncomp = scores.ncols();
    let mut means = vec![0.0; ncomp];
    let mut stds = vec![0.0; ncomp];
    for k in 0..ncomp {
        let col = scores.column(k);
        let mean = col.iter().sum::<f64>() / n as f64;
        means[k] = mean;
        let var = col.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (n as f64 - 1.0);
        stds[k] = var.sqrt().max(1e-15);
    }
    ScoreStats { means, stds }
}

/// Sample a single bootstrap curve from FPCA model.
fn sample_bootstrap_curve(
    rng: &mut StdRng,
    mean: &[f64],
    rotation: &FdMatrix,
    stats: &ScoreStats,
    curve: &mut [f64],
) {
    let m = mean.len();
    let ncomp = stats.means.len();
    curve[..m].copy_from_slice(&mean[..m]);
    for k in 0..ncomp {
        let z: f64 = rng.sample(StandardNormal);
        let score = stats.means[k] + stats.stds[k] * z;
        for j in 0..m {
            curve[j] += score * rotation[(j, k)];
        }
    }
}

/// Pointwise bootstrap: collect per-point std across bootstrap replicates.
fn fpca_pointwise_boot(
    fpca: &crate::regression::FpcaResult,
    stats: &ScoreStats,
    n: usize,
    m: usize,
    nb: usize,
    coverage: f64,
    seed: u64,
) -> ToleranceBand {
    let boot_stds: Vec<Vec<f64>> = iter_maybe_parallel!(0..nb)
        .map(|b| {
            let mut rng = StdRng::seed_from_u64(seed + b as u64);
            let mut curve = vec![0.0; m];
            let mut sum = vec![0.0; m];
            let mut sum_sq = vec![0.0; m];
            for _ in 0..n {
                sample_bootstrap_curve(&mut rng, &fpca.mean, &fpca.rotation, stats, &mut curve);
                for j in 0..m {
                    sum[j] += curve[j];
                    sum_sq[j] += curve[j] * curve[j];
                }
            }
            let nf = n as f64;
            (0..m)
                .map(|j| (sum_sq[j] / nf - (sum[j] / nf).powi(2)).max(0.0).sqrt())
                .collect::<Vec<f64>>()
        })
        .collect();

    let z = normal_quantile((1.0 + coverage) / 2.0);
    let center = fpca.mean.clone();
    let mut half_width = vec![0.0; m];
    for j in 0..m {
        let mut stds_j: Vec<f64> = boot_stds.iter().map(|s| s[j]).collect();
        let pct = percentile_sorted(&mut stds_j, coverage);
        half_width[j] = z * pct;
    }
    build_band(center, half_width)
}

/// Simultaneous bootstrap: compute sup-norm scaling factor.
fn fpca_simultaneous_boot(
    data: &FdMatrix,
    fpca: &crate::regression::FpcaResult,
    stats: &ScoreStats,
    n: usize,
    m: usize,
    nb: usize,
    coverage: f64,
    seed: u64,
) -> ToleranceBand {
    let (center, data_std) = pointwise_mean_std(data);

    let mut sup_norms: Vec<f64> = iter_maybe_parallel!(0..nb)
        .map(|b| {
            let mut rng = StdRng::seed_from_u64(seed + b as u64);
            let mut max_dev = 0.0_f64;
            let mut curve = vec![0.0; m];
            for _ in 0..n {
                sample_bootstrap_curve(&mut rng, &fpca.mean, &fpca.rotation, stats, &mut curve);
                let dev = (0..m)
                    .map(|j| (curve[j] - center[j]).abs() / data_std[j].max(1e-15))
                    .fold(0.0_f64, f64::max);
                max_dev = max_dev.max(dev);
            }
            max_dev
        })
        .collect();

    let k_factor = percentile_sorted(&mut sup_norms, coverage);
    let half_width: Vec<f64> = data_std.iter().map(|&s| k_factor * s).collect();
    build_band(center, half_width)
}

/// Compute a tolerance band via FPCA and bootstrap.
///
/// Decomposes the data using functional PCA, then uses a parametric bootstrap
/// on PC scores to estimate the band width.
///
/// # Arguments
/// * `data` — Functional data matrix (n observations x m evaluation points)
/// * `ncomp` — Number of principal components to retain
/// * `nb` — Number of bootstrap replicates
/// * `coverage` — Target coverage probability (e.g., 0.95)
/// * `band_type` — [`BandType::Pointwise`] or [`BandType::Simultaneous`]
/// * `seed` — Random seed for reproducibility
///
/// # Returns
/// `Ok(ToleranceBand)` on success, or `Err(FdarError)` if inputs are invalid or FPCA fails.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 3 rows
/// or zero columns.
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero, `nb` is zero,
/// or `coverage` is not in the open interval (0, 1).
/// Returns [`FdarError::ComputationFailed`] if the underlying FPCA fails.
///
/// # Examples
///
/// ```
/// use fdars_core::simulation::{sim_fundata, EFunType, EValType};
/// use fdars_core::tolerance::{fpca_tolerance_band, BandType};
///
/// let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
/// let data = sim_fundata(30, &t, 5, EFunType::Fourier, EValType::Exponential, Some(42));
///
/// let band = fpca_tolerance_band(&data, 3, 100, 0.95, BandType::Pointwise, 42).unwrap();
/// assert_eq!(band.lower.len(), 50);
/// assert!(band.lower.iter().zip(band.upper.iter()).all(|(l, u)| l < u));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fpca_tolerance_band(
    data: &FdMatrix,
    ncomp: usize,
    nb: usize,
    coverage: f64,
    band_type: BandType,
    seed: u64,
) -> Result<ToleranceBand, FdarError> {
    let (n, m) = data.shape();
    if n < 3 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 rows and 1 column".to_string(),
            actual: format!("{n} x {m}"),
        });
    }
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be >= 1".to_string(),
        });
    }
    if nb == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "nb",
            message: "must be >= 1".to_string(),
        });
    }
    if coverage <= 0.0 || coverage >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "coverage",
            message: format!("must be in (0, 1), got {coverage}"),
        });
    }

    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1).max(1) as f64).collect();
    let fpca = fdata_to_pc_1d(data, ncomp, &argvals)?;
    let stats = compute_score_stats(&fpca.scores, n);

    Ok(match band_type {
        BandType::Pointwise => fpca_pointwise_boot(&fpca, &stats, n, m, nb, coverage, seed),
        BandType::Simultaneous => {
            fpca_simultaneous_boot(data, &fpca, &stats, n, m, nb, coverage, seed)
        }
    })
}
