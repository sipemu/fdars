//! Tolerance bands for functional data.
//!
//! This module provides methods for constructing regions expected to contain
//! a given fraction of individual curves in a population — the functional
//! analogue of classical tolerance intervals.
//!
//! # Methods
//!
//! - [`fpca_tolerance_band`] — FPCA + bootstrap tolerance band (pointwise or simultaneous)
//! - [`conformal_prediction_band`] — Distribution-free conformal prediction band
//! - [`scb_mean_degras`] — Simultaneous confidence band for the mean (Degras method)
//! - [`exponential_family_tolerance_band`] — Tolerance band for exponential family data

use crate::fdata::mean_1d;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;
use crate::smoothing::local_polynomial;
use rand::prelude::*;
use rand_distr::StandardNormal;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Result of a tolerance band computation.
#[derive(Debug, Clone)]
pub struct ToleranceBand {
    /// Lower bound at each evaluation point
    pub lower: Vec<f64>,
    /// Upper bound at each evaluation point
    pub upper: Vec<f64>,
    /// Center function (typically the mean)
    pub center: Vec<f64>,
    /// Half-width at each evaluation point
    pub half_width: Vec<f64>,
}

/// Type of tolerance band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandType {
    /// Independent interval at each evaluation point
    Pointwise,
    /// Single scaling factor across all points (wider, controls family-wise error)
    Simultaneous,
}

/// Non-conformity score for conformal prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonConformityScore {
    /// Supremum norm: max_t |y(t) - center(t)|
    SupNorm,
    /// L2 norm: sqrt(sum (y(t) - center(t))^2)
    L2,
}

/// Multiplier distribution for Degras SCB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiplierDistribution {
    /// Standard normal multipliers
    Gaussian,
    /// Rademacher multipliers (+1/-1 with equal probability)
    Rademacher,
}

/// Exponential family for generalized FPCA tolerance bands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExponentialFamily {
    /// Gaussian (identity link)
    Gaussian,
    /// Binomial (logit link)
    Binomial,
    /// Poisson (log link)
    Poisson,
}

// ─── Private helpers ────────────────────────────────────────────────────────

/// Column-wise mean and standard deviation.
fn pointwise_mean_std(data: &FdMatrix) -> (Vec<f64>, Vec<f64>) {
    let (n, m) = data.shape();
    let nf = n as f64;
    let mut means = vec![0.0; m];
    let mut stds = vec![0.0; m];

    for j in 0..m {
        let col = data.column(j);
        let mean = col.iter().sum::<f64>() / nf;
        means[j] = mean;
        let var = col.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (nf - 1.0);
        stds[j] = var.sqrt();
    }
    (means, stds)
}

/// Inverse normal CDF (probit) via rational approximation (Abramowitz & Stegun 26.2.23).
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // Use symmetry: for p < 0.5, compute for 1-p and negate
    let (sign, q) = if p < 0.5 { (-1.0, 1.0 - p) } else { (1.0, p) };

    let t = (-2.0 * (1.0 - q).ln()).sqrt();

    // Rational approximation coefficients
    const C0: f64 = 2.515517;
    const C1: f64 = 0.802853;
    const C2: f64 = 0.010328;
    const D1: f64 = 1.432788;
    const D2: f64 = 0.189269;
    const D3: f64 = 0.001308;

    let numerator = C0 + C1 * t + C2 * t * t;
    let denominator = 1.0 + D1 * t + D2 * t * t + D3 * t * t * t;

    sign * (t - numerator / denominator)
}

/// Construct a tolerance band from center and half-width vectors.
fn build_band(center: Vec<f64>, half_width: Vec<f64>) -> ToleranceBand {
    let lower: Vec<f64> = center
        .iter()
        .zip(half_width.iter())
        .map(|(&c, &h)| c - h)
        .collect();
    let upper: Vec<f64> = center
        .iter()
        .zip(half_width.iter())
        .map(|(&c, &h)| c + h)
        .collect();
    ToleranceBand {
        lower,
        upper,
        center,
        half_width,
    }
}

/// Extract the percentile value from a sorted slice.
fn percentile_sorted(sorted: &mut [f64], p: f64) -> f64 {
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() as f64 * p).ceil() as usize).min(sorted.len()) - 1;
    sorted[idx]
}

/// Validate common tolerance band parameters. Returns false if any are invalid.
fn valid_band_params(n: usize, m: usize, ncomp: usize, nb: usize, coverage: f64) -> bool {
    n >= 3 && m > 0 && ncomp > 0 && nb > 0 && coverage > 0.0 && coverage < 1.0
}

// ─── FPCA + Bootstrap Tolerance Band ────────────────────────────────────────

/// Per-component score statistics needed for bootstrap sampling.
struct ScoreStats {
    means: Vec<f64>,
    stds: Vec<f64>,
}

/// Compute per-component score mean and std from FPCA results.
fn compute_score_stats(scores: &FdMatrix, n: usize) -> ScoreStats {
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
/// * `data` — Functional data matrix (n observations × m evaluation points)
/// * `ncomp` — Number of principal components to retain
/// * `nb` — Number of bootstrap replicates
/// * `coverage` — Target coverage probability (e.g., 0.95)
/// * `band_type` — [`BandType::Pointwise`] or [`BandType::Simultaneous`]
/// * `seed` — Random seed for reproducibility
///
/// # Returns
/// `Some(ToleranceBand)` on success, `None` if inputs are invalid or FPCA fails.
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
pub fn fpca_tolerance_band(
    data: &FdMatrix,
    ncomp: usize,
    nb: usize,
    coverage: f64,
    band_type: BandType,
    seed: u64,
) -> Option<ToleranceBand> {
    let (n, m) = data.shape();
    if !valid_band_params(n, m, ncomp, nb, coverage) {
        return None;
    }

    let fpca = fdata_to_pc_1d(data, ncomp)?;
    let stats = compute_score_stats(&fpca.scores, n);

    Some(match band_type {
        BandType::Pointwise => fpca_pointwise_boot(&fpca, &stats, n, m, nb, coverage, seed),
        BandType::Simultaneous => {
            fpca_simultaneous_boot(data, &fpca, &stats, n, m, nb, coverage, seed)
        }
    })
}

// ─── Conformal Prediction Band ──────────────────────────────────────────────

/// Compute a non-conformity score for a single curve against a center function.
fn nonconformity_score(
    data: &FdMatrix,
    i: usize,
    center: &[f64],
    m: usize,
    score_type: NonConformityScore,
) -> f64 {
    match score_type {
        NonConformityScore::SupNorm => (0..m)
            .map(|j| (data[(i, j)] - center[j]).abs())
            .fold(0.0_f64, f64::max),
        NonConformityScore::L2 => {
            let ss: f64 = (0..m).map(|j| (data[(i, j)] - center[j]).powi(2)).sum();
            ss.sqrt()
        }
    }
}

/// Build a subset matrix from selected row indices.
fn subset_rows(data: &FdMatrix, indices: &[usize], m: usize) -> FdMatrix {
    let n_sub = indices.len();
    let mut sub = FdMatrix::zeros(n_sub, m);
    for (new_i, &old_i) in indices.iter().enumerate() {
        for j in 0..m {
            sub[(new_i, j)] = data[(old_i, j)];
        }
    }
    sub
}

/// Compute a distribution-free conformal prediction band.
///
/// Splits data into training and calibration sets, computes a non-conformity
/// score on calibration curves, and uses the conformal quantile to build
/// a prediction band.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `cal_fraction` — Fraction of data used for calibration (e.g., 0.2)
/// * `coverage` — Target coverage probability (e.g., 0.95)
/// * `score_type` — [`NonConformityScore::SupNorm`] or [`NonConformityScore::L2`]
/// * `seed` — Random seed for the train/calibration split
///
/// # Returns
/// `Some(ToleranceBand)` on success, `None` if inputs are invalid.
///
/// # Examples
///
/// ```
/// use fdars_core::simulation::{sim_fundata, EFunType, EValType};
/// use fdars_core::tolerance::{conformal_prediction_band, NonConformityScore};
///
/// let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
/// let data = sim_fundata(40, &t, 5, EFunType::Fourier, EValType::Exponential, Some(42));
///
/// let band = conformal_prediction_band(&data, 0.2, 0.95, NonConformityScore::SupNorm, 42).unwrap();
/// // SupNorm gives constant half-width across all evaluation points
/// let hw = band.half_width[0];
/// assert!(band.half_width.iter().all(|&h| (h - hw).abs() < 1e-12));
/// ```
pub fn conformal_prediction_band(
    data: &FdMatrix,
    cal_fraction: f64,
    coverage: f64,
    score_type: NonConformityScore,
    seed: u64,
) -> Option<ToleranceBand> {
    let (n, m) = data.shape();
    if n < 4
        || m == 0
        || cal_fraction <= 0.0
        || cal_fraction >= 1.0
        || coverage <= 0.0
        || coverage >= 1.0
    {
        return None;
    }

    let n_cal = ((n as f64) * cal_fraction).max(1.0).min((n - 2) as f64) as usize;
    let n_train = n - n_cal;

    // Random permutation for split
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);

    let train_data = subset_rows(data, &indices[..n_train], m);
    let center = mean_1d(&train_data);

    // Compute non-conformity scores on calibration set
    let cal_idx = &indices[n_train..];
    let mut scores: Vec<f64> = cal_idx
        .iter()
        .map(|&i| nonconformity_score(data, i, &center, m, score_type))
        .collect();

    // Conformal quantile: ceil((n_cal + 1) * coverage) / n_cal
    let level = (((n_cal + 1) as f64 * coverage).ceil() / n_cal as f64).min(1.0);
    let q = percentile_sorted(&mut scores, level);

    // Build band depending on score type
    let half_width = match score_type {
        NonConformityScore::SupNorm => vec![q; m],
        NonConformityScore::L2 => vec![q / (m as f64).sqrt(); m],
    };

    Some(build_band(center, half_width))
}

// ─── SCB for the Mean (Degras) ──────────────────────────────────────────────

/// Compute pointwise residual standard deviation.
fn residual_sigma(data: &FdMatrix, center: &[f64], n: usize, m: usize) -> Vec<f64> {
    let nf = n as f64;
    (0..m)
        .map(|j| {
            let var: f64 = (0..n).map(|i| (data[(i, j)] - center[j]).powi(2)).sum();
            (var / nf).sqrt().max(1e-15)
        })
        .collect()
}

/// Generate n multiplier weights for Degras bootstrap.
fn generate_multiplier_weights(
    rng: &mut StdRng,
    n: usize,
    multiplier: MultiplierDistribution,
) -> Vec<f64> {
    (0..n)
        .map(|_| match multiplier {
            MultiplierDistribution::Gaussian => rng.sample::<f64, _>(StandardNormal),
            MultiplierDistribution::Rademacher => {
                if rng.gen_bool(0.5) {
                    1.0
                } else {
                    -1.0
                }
            }
        })
        .collect()
}

/// Compute a simultaneous confidence band for the mean function (Degras method).
///
/// Uses a multiplier bootstrap to estimate the critical value for a
/// simultaneous confidence band around the mean.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `bandwidth` — Kernel bandwidth for local polynomial smoothing
/// * `nb` — Number of bootstrap replicates
/// * `confidence` — Confidence level (e.g., 0.95)
/// * `multiplier` — [`MultiplierDistribution::Gaussian`] or [`MultiplierDistribution::Rademacher`]
///
/// # Returns
/// `Some(ToleranceBand)` on success, `None` if inputs are invalid.
///
/// # Examples
///
/// ```
/// use fdars_core::simulation::{sim_fundata, EFunType, EValType};
/// use fdars_core::tolerance::{scb_mean_degras, MultiplierDistribution};
///
/// let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
/// let data = sim_fundata(50, &t, 5, EFunType::Fourier, EValType::Exponential, Some(42));
///
/// let band = scb_mean_degras(&data, &t, 0.15, 200, 0.95, MultiplierDistribution::Gaussian).unwrap();
/// assert_eq!(band.center.len(), 50);
/// assert!(band.lower.iter().zip(band.upper.iter()).all(|(l, u)| l < u));
/// ```
pub fn scb_mean_degras(
    data: &FdMatrix,
    argvals: &[f64],
    bandwidth: f64,
    nb: usize,
    confidence: f64,
    multiplier: MultiplierDistribution,
) -> Option<ToleranceBand> {
    let (n, m) = data.shape();
    if n < 3
        || m == 0
        || argvals.len() != m
        || bandwidth <= 0.0
        || nb == 0
        || confidence <= 0.0
        || confidence >= 1.0
    {
        return None;
    }

    let raw_mean = mean_1d(data);
    let center = local_polynomial(argvals, &raw_mean, argvals, bandwidth, 1, "epanechnikov");
    let sigma_hat = residual_sigma(data, &center, n, m);

    let sqrt_n = (n as f64).sqrt();
    let mut sup_stats: Vec<f64> = iter_maybe_parallel!(0..nb)
        .map(|b| {
            let mut rng = StdRng::seed_from_u64(42 + b as u64);
            let weights = generate_multiplier_weights(&mut rng, n, multiplier);
            (0..m)
                .map(|j| {
                    let z: f64 = (0..n)
                        .map(|i| weights[i] * (data[(i, j)] - center[j]))
                        .sum::<f64>()
                        / (sqrt_n * sigma_hat[j]);
                    z.abs()
                })
                .fold(0.0_f64, f64::max)
        })
        .collect();

    let c = percentile_sorted(&mut sup_stats, confidence);
    let half_width: Vec<f64> = sigma_hat.iter().map(|&s| c * s / sqrt_n).collect();

    Some(build_band(center, half_width))
}

// ─── Exponential Family Tolerance Band ──────────────────────────────────────

/// Apply the link function for an exponential family.
fn apply_link(value: f64, family: ExponentialFamily) -> f64 {
    match family {
        ExponentialFamily::Gaussian => value,
        ExponentialFamily::Binomial => {
            // logit: log(p / (1-p)), clamp to avoid infinities
            let p = value.clamp(1e-10, 1.0 - 1e-10);
            (p / (1.0 - p)).ln()
        }
        ExponentialFamily::Poisson => {
            // log link, clamp to avoid log(0)
            value.max(1e-10).ln()
        }
    }
}

/// Apply the inverse link function for an exponential family.
fn apply_inverse_link(value: f64, family: ExponentialFamily) -> f64 {
    match family {
        ExponentialFamily::Gaussian => value,
        ExponentialFamily::Binomial => {
            // inverse logit: 1 / (1 + exp(-x))
            1.0 / (1.0 + (-value).exp())
        }
        ExponentialFamily::Poisson => {
            // exp
            value.exp()
        }
    }
}

/// Apply a link function element-wise to all data entries.
fn transform_data(data: &FdMatrix, family: ExponentialFamily) -> FdMatrix {
    let (n, m) = data.shape();
    let mut out = FdMatrix::zeros(n, m);
    for j in 0..m {
        for i in 0..n {
            out[(i, j)] = apply_link(data[(i, j)], family);
        }
    }
    out
}

/// Apply the inverse link to a band, recomputing half-widths on the response scale.
fn inverse_link_band(band: ToleranceBand, family: ExponentialFamily) -> ToleranceBand {
    let lower: Vec<f64> = band
        .lower
        .iter()
        .map(|&v| apply_inverse_link(v, family))
        .collect();
    let upper: Vec<f64> = band
        .upper
        .iter()
        .map(|&v| apply_inverse_link(v, family))
        .collect();
    let center: Vec<f64> = band
        .center
        .iter()
        .map(|&v| apply_inverse_link(v, family))
        .collect();
    let half_width: Vec<f64> = upper
        .iter()
        .zip(lower.iter())
        .map(|(&u, &l)| (u - l) / 2.0)
        .collect();
    ToleranceBand {
        lower,
        upper,
        center,
        half_width,
    }
}

/// Compute a tolerance band for exponential family functional data.
///
/// Transforms data via the canonical link function, applies FPCA + bootstrap
/// on the transformed scale, then maps the band back via the inverse link.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m), values in natural parameter space
/// * `family` — [`ExponentialFamily`] specifying the distribution
/// * `ncomp` — Number of principal components to retain
/// * `nb` — Number of bootstrap replicates
/// * `coverage` — Target coverage probability (e.g., 0.95)
/// * `seed` — Random seed for reproducibility
///
/// # Returns
/// `Some(ToleranceBand)` on success, `None` if inputs are invalid.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::simulation::{sim_fundata, EFunType, EValType};
/// use fdars_core::tolerance::{exponential_family_tolerance_band, ExponentialFamily};
///
/// // Create positive data suitable for Poisson family
/// let t: Vec<f64> = (0..30).map(|i| i as f64 / 29.0).collect();
/// let raw = sim_fundata(30, &t, 3, EFunType::Fourier, EValType::Exponential, Some(42));
/// let mut data = FdMatrix::zeros(30, 30);
/// for j in 0..30 {
///     for i in 0..30 {
///         data[(i, j)] = (raw[(i, j)] + 5.0).max(0.1);
///     }
/// }
///
/// let band = exponential_family_tolerance_band(
///     &data, ExponentialFamily::Poisson, 3, 50, 0.95, 42,
/// ).unwrap();
/// // Poisson inverse link (exp) ensures all bounds are positive
/// assert!(band.lower.iter().all(|&v| v > 0.0));
/// ```
pub fn exponential_family_tolerance_band(
    data: &FdMatrix,
    family: ExponentialFamily,
    ncomp: usize,
    nb: usize,
    coverage: f64,
    seed: u64,
) -> Option<ToleranceBand> {
    let (n, m) = data.shape();
    if !valid_band_params(n, m, ncomp, nb, coverage) {
        return None;
    }

    let transformed = transform_data(data, family);
    let band = fpca_tolerance_band(
        &transformed,
        ncomp,
        nb,
        coverage,
        BandType::Simultaneous,
        seed,
    )?;
    Some(inverse_link_band(band, family))
}

// ─── Elastic Tolerance Band ─────────────────────────────────────────────────

/// Compute a tolerance band in the elastic (aligned) space.
///
/// First computes the Karcher mean to align all curves, then applies the
/// FPCA tolerance band on the aligned data. This separates amplitude
/// variability from phase variability, giving bands that reflect shape
/// variation without contamination from timing differences.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `ncomp` — Number of principal components to retain
/// * `nb` — Number of bootstrap replicates
/// * `coverage` — Target coverage probability (e.g., 0.95)
/// * `band_type` — [`BandType::Pointwise`] or [`BandType::Simultaneous`]
/// * `max_iter` — Maximum iterations for Karcher mean convergence
/// * `seed` — Random seed for reproducibility
///
/// # Returns
/// `Some(ToleranceBand)` in the aligned space, `None` if inputs are invalid.
///
/// # Examples
///
/// ```
/// use fdars_core::simulation::{sim_fundata, EFunType, EValType};
/// use fdars_core::tolerance::{elastic_tolerance_band, BandType};
///
/// let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
/// let data = sim_fundata(30, &t, 5, EFunType::Fourier, EValType::Exponential, Some(42));
///
/// let band = elastic_tolerance_band(&data, &t, 3, 100, 0.95, BandType::Pointwise, 10, 42);
/// assert!(band.is_some());
/// ```
pub fn elastic_tolerance_band(
    data: &FdMatrix,
    argvals: &[f64],
    ncomp: usize,
    nb: usize,
    coverage: f64,
    band_type: BandType,
    max_iter: usize,
    seed: u64,
) -> Option<ToleranceBand> {
    let (n, m) = data.shape();
    if !valid_band_params(n, m, ncomp, nb, coverage) || argvals.len() != m || max_iter == 0 {
        return None;
    }

    // Step 1: Karcher mean → aligned data
    let karcher = crate::alignment::karcher_mean(data, argvals, max_iter, 1e-4);

    // Step 2: FPCA tolerance band on aligned data
    fpca_tolerance_band(&karcher.aligned_data, ncomp, nb, coverage, band_type, seed)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{sim_fundata, EFunType, EValType};

    fn uniform_grid(m: usize) -> Vec<f64> {
        (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
    }

    fn make_test_data() -> FdMatrix {
        let m = 50;
        let t = uniform_grid(m);
        sim_fundata(
            50,
            &t,
            5,
            EFunType::Fourier,
            EValType::Exponential,
            Some(42),
        )
    }

    // ── normal_quantile tests ──

    #[test]
    fn test_normal_quantile_symmetry() {
        for &p in &[0.1, 0.2, 0.3, 0.4] {
            let q_low = normal_quantile(p);
            let q_high = normal_quantile(1.0 - p);
            assert!(
                (q_low + q_high).abs() < 1e-6,
                "q({p}) + q({}) = {} (expected ~0)",
                1.0 - p,
                q_low + q_high
            );
        }
    }

    #[test]
    fn test_normal_quantile_known_values() {
        let q975 = normal_quantile(0.975);
        assert!(
            (q975 - 1.96).abs() < 0.01,
            "q(0.975) = {q975}, expected ~1.96"
        );

        let q50 = normal_quantile(0.5);
        assert!(q50.abs() < 1e-10, "q(0.5) = {q50}, expected 0.0");

        let q_invalid = normal_quantile(0.0);
        assert!(q_invalid.is_nan());
        let q_invalid2 = normal_quantile(1.0);
        assert!(q_invalid2.is_nan());
    }

    // ── FPCA tolerance band tests ──

    #[test]
    fn test_fpca_band_valid_output() {
        let data = make_test_data();
        let m = data.ncols();

        let band = fpca_tolerance_band(&data, 3, 100, 0.95, BandType::Pointwise, 42);
        let band = band.expect("FPCA band should succeed");

        assert_eq!(band.lower.len(), m);
        assert_eq!(band.upper.len(), m);
        assert_eq!(band.center.len(), m);
        assert_eq!(band.half_width.len(), m);
    }

    #[test]
    fn test_fpca_band_lower_less_than_upper() {
        let data = make_test_data();
        let band = fpca_tolerance_band(&data, 3, 100, 0.95, BandType::Pointwise, 42).unwrap();

        for j in 0..band.lower.len() {
            assert!(
                band.lower[j] < band.upper[j],
                "lower[{j}] = {} >= upper[{j}] = {}",
                band.lower[j],
                band.upper[j]
            );
        }
    }

    #[test]
    fn test_fpca_band_deterministic() {
        let data = make_test_data();
        let b1 = fpca_tolerance_band(&data, 3, 50, 0.95, BandType::Pointwise, 123).unwrap();
        let b2 = fpca_tolerance_band(&data, 3, 50, 0.95, BandType::Pointwise, 123).unwrap();

        for j in 0..b1.lower.len() {
            assert_eq!(b1.lower[j], b2.lower[j]);
            assert_eq!(b1.upper[j], b2.upper[j]);
        }
    }

    #[test]
    fn test_fpca_simultaneous_wider_than_pointwise() {
        let data = make_test_data();
        let pw = fpca_tolerance_band(&data, 3, 200, 0.95, BandType::Pointwise, 42).unwrap();
        let sim = fpca_tolerance_band(&data, 3, 200, 0.95, BandType::Simultaneous, 42).unwrap();

        let pw_mean_hw: f64 = pw.half_width.iter().sum::<f64>() / pw.half_width.len() as f64;
        let sim_mean_hw: f64 = sim.half_width.iter().sum::<f64>() / sim.half_width.len() as f64;

        assert!(
            sim_mean_hw > pw_mean_hw,
            "Simultaneous mean half-width ({sim_mean_hw}) should exceed pointwise ({pw_mean_hw})"
        );
    }

    #[test]
    fn test_fpca_higher_coverage_wider() {
        let data = make_test_data();
        let b90 = fpca_tolerance_band(&data, 3, 200, 0.90, BandType::Pointwise, 42).unwrap();
        let b99 = fpca_tolerance_band(&data, 3, 200, 0.99, BandType::Pointwise, 42).unwrap();

        let hw90: f64 = b90.half_width.iter().sum::<f64>();
        let hw99: f64 = b99.half_width.iter().sum::<f64>();

        assert!(
            hw99 > hw90,
            "99% band total half-width ({hw99}) should exceed 90% ({hw90})"
        );
    }

    #[test]
    fn test_fpca_band_invalid_input() {
        let data = make_test_data();
        assert!(fpca_tolerance_band(&data, 0, 100, 0.95, BandType::Pointwise, 42).is_none());
        assert!(fpca_tolerance_band(&data, 3, 0, 0.95, BandType::Pointwise, 42).is_none());
        assert!(fpca_tolerance_band(&data, 3, 100, 0.0, BandType::Pointwise, 42).is_none());
        assert!(fpca_tolerance_band(&data, 3, 100, 1.0, BandType::Pointwise, 42).is_none());

        let tiny = FdMatrix::zeros(2, 5);
        assert!(fpca_tolerance_band(&tiny, 1, 10, 0.95, BandType::Pointwise, 42).is_none());
    }

    // ── Conformal prediction band tests ──

    #[test]
    fn test_conformal_band_valid_output() {
        let data = make_test_data();
        let m = data.ncols();

        let band = conformal_prediction_band(&data, 0.2, 0.95, NonConformityScore::SupNorm, 42);
        let band = band.expect("Conformal band should succeed");

        assert_eq!(band.lower.len(), m);
        assert_eq!(band.upper.len(), m);
    }

    #[test]
    fn test_conformal_supnorm_constant_width() {
        let data = make_test_data();
        let band =
            conformal_prediction_band(&data, 0.3, 0.95, NonConformityScore::SupNorm, 42).unwrap();

        let first = band.half_width[0];
        for &hw in &band.half_width {
            assert!(
                (hw - first).abs() < 1e-12,
                "SupNorm band should have constant width"
            );
        }
    }

    #[test]
    fn test_conformal_l2_constant_width() {
        let data = make_test_data();
        let band = conformal_prediction_band(&data, 0.3, 0.95, NonConformityScore::L2, 42).unwrap();

        let first = band.half_width[0];
        for &hw in &band.half_width {
            assert!(
                (hw - first).abs() < 1e-12,
                "L2 band should have constant width"
            );
        }
    }

    #[test]
    fn test_conformal_coverage_monotonicity() {
        let data = make_test_data();
        let b80 =
            conformal_prediction_band(&data, 0.3, 0.80, NonConformityScore::SupNorm, 42).unwrap();
        let b95 =
            conformal_prediction_band(&data, 0.3, 0.95, NonConformityScore::SupNorm, 42).unwrap();

        assert!(
            b95.half_width[0] >= b80.half_width[0],
            "Higher coverage should give wider band"
        );
    }

    #[test]
    fn test_conformal_invalid_input() {
        let data = make_test_data();
        assert!(
            conformal_prediction_band(&data, 0.0, 0.95, NonConformityScore::SupNorm, 42).is_none()
        );
        assert!(
            conformal_prediction_band(&data, 1.0, 0.95, NonConformityScore::SupNorm, 42).is_none()
        );
        assert!(
            conformal_prediction_band(&data, 0.2, 0.0, NonConformityScore::SupNorm, 42).is_none()
        );

        let tiny = FdMatrix::zeros(3, 5);
        assert!(
            conformal_prediction_band(&tiny, 0.2, 0.95, NonConformityScore::SupNorm, 42).is_none()
        );
    }

    // ── SCB mean Degras tests ──

    #[test]
    fn test_scb_mean_valid_output() {
        let data = make_test_data();
        let m = data.ncols();
        let t = uniform_grid(m);

        let band = scb_mean_degras(&data, &t, 0.2, 100, 0.95, MultiplierDistribution::Gaussian);
        let band = band.expect("SCB mean should succeed");

        assert_eq!(band.lower.len(), m);
        assert_eq!(band.upper.len(), m);
        for j in 0..m {
            assert!(band.lower[j] < band.upper[j]);
        }
    }

    #[test]
    fn test_scb_gaussian_vs_rademacher() {
        let data = make_test_data();
        let m = data.ncols();
        let t = uniform_grid(m);

        let gauss =
            scb_mean_degras(&data, &t, 0.2, 200, 0.95, MultiplierDistribution::Gaussian).unwrap();
        let radem = scb_mean_degras(
            &data,
            &t,
            0.2,
            200,
            0.95,
            MultiplierDistribution::Rademacher,
        )
        .unwrap();

        // Both should produce valid bands; widths may differ but both should be positive
        let gauss_mean_hw: f64 = gauss.half_width.iter().sum::<f64>() / m as f64;
        let radem_mean_hw: f64 = radem.half_width.iter().sum::<f64>() / m as f64;
        assert!(gauss_mean_hw > 0.0);
        assert!(radem_mean_hw > 0.0);
    }

    #[test]
    fn test_scb_narrows_with_more_data() {
        let m = 50;
        let t = uniform_grid(m);

        let data_small = sim_fundata(
            20,
            &t,
            5,
            EFunType::Fourier,
            EValType::Exponential,
            Some(42),
        );
        let data_large = sim_fundata(
            200,
            &t,
            5,
            EFunType::Fourier,
            EValType::Exponential,
            Some(42),
        );

        let band_small = scb_mean_degras(
            &data_small,
            &t,
            0.2,
            100,
            0.95,
            MultiplierDistribution::Gaussian,
        )
        .unwrap();
        let band_large = scb_mean_degras(
            &data_large,
            &t,
            0.2,
            100,
            0.95,
            MultiplierDistribution::Gaussian,
        )
        .unwrap();

        let hw_small: f64 = band_small.half_width.iter().sum::<f64>() / m as f64;
        let hw_large: f64 = band_large.half_width.iter().sum::<f64>() / m as f64;

        assert!(
            hw_large < hw_small,
            "SCB should narrow with more data: hw_small={hw_small}, hw_large={hw_large}"
        );
    }

    #[test]
    fn test_scb_invalid_input() {
        let data = make_test_data();
        let t = uniform_grid(data.ncols());

        assert!(
            scb_mean_degras(&data, &t, 0.0, 100, 0.95, MultiplierDistribution::Gaussian).is_none()
        );
        assert!(
            scb_mean_degras(&data, &t, 0.2, 0, 0.95, MultiplierDistribution::Gaussian).is_none()
        );
        assert!(
            scb_mean_degras(&data, &t, 0.2, 100, 0.0, MultiplierDistribution::Gaussian).is_none()
        );
        // Wrong argvals length
        let wrong_t = uniform_grid(data.ncols() + 1);
        assert!(scb_mean_degras(
            &data,
            &wrong_t,
            0.2,
            100,
            0.95,
            MultiplierDistribution::Gaussian
        )
        .is_none());
    }

    // ── Exponential family tests ──

    #[test]
    fn test_exp_family_gaussian_matches_fpca() {
        let data = make_test_data();

        let exp_band =
            exponential_family_tolerance_band(&data, ExponentialFamily::Gaussian, 3, 100, 0.95, 42)
                .unwrap();

        let fpca_band =
            fpca_tolerance_band(&data, 3, 100, 0.95, BandType::Simultaneous, 42).unwrap();

        // Gaussian family with identity link should produce the same band
        for j in 0..data.ncols() {
            assert!(
                (exp_band.lower[j] - fpca_band.lower[j]).abs() < 1e-10,
                "Gaussian family should match FPCA at point {j}"
            );
            assert!(
                (exp_band.upper[j] - fpca_band.upper[j]).abs() < 1e-10,
                "Gaussian family should match FPCA at point {j}"
            );
        }
    }

    #[test]
    fn test_exp_family_poisson() {
        // Create data that looks like Poisson counts (positive)
        let m = 30;
        let t = uniform_grid(m);
        let raw = sim_fundata(
            40,
            &t,
            3,
            EFunType::Fourier,
            EValType::Exponential,
            Some(99),
        );

        // Shift to positive range and add offset
        let mut data = FdMatrix::zeros(40, m);
        for j in 0..m {
            for i in 0..40 {
                data[(i, j)] = (raw[(i, j)] + 5.0).max(0.1); // ensure positive
            }
        }

        let band =
            exponential_family_tolerance_band(&data, ExponentialFamily::Poisson, 3, 50, 0.95, 42);
        let band = band.expect("Poisson band should succeed");

        // All bounds should be positive (exp of anything is positive)
        for j in 0..m {
            assert!(
                band.lower[j] > 0.0,
                "Poisson lower bound should be positive"
            );
            assert!(
                band.upper[j] > 0.0,
                "Poisson upper bound should be positive"
            );
        }
    }

    #[test]
    fn test_exp_family_binomial() {
        // Create data in (0, 1) range
        let m = 30;
        let t = uniform_grid(m);
        let raw = sim_fundata(
            40,
            &t,
            3,
            EFunType::Fourier,
            EValType::Exponential,
            Some(77),
        );

        let mut data = FdMatrix::zeros(40, m);
        for j in 0..m {
            for i in 0..40 {
                // Map to (0, 1) via sigmoid
                data[(i, j)] = 1.0 / (1.0 + (-raw[(i, j)]).exp());
            }
        }

        let band =
            exponential_family_tolerance_band(&data, ExponentialFamily::Binomial, 3, 50, 0.95, 42);
        let band = band.expect("Binomial band should succeed");

        // All bounds should be in (0, 1) (inverse logit maps to (0, 1))
        for j in 0..m {
            assert!(
                band.lower[j] > 0.0 && band.lower[j] < 1.0,
                "Binomial lower bound at {j} = {} should be in (0,1)",
                band.lower[j]
            );
            assert!(
                band.upper[j] > 0.0 && band.upper[j] < 1.0,
                "Binomial upper bound at {j} = {} should be in (0,1)",
                band.upper[j]
            );
        }
    }

    #[test]
    fn test_exp_family_invalid_input() {
        let data = make_test_data();
        assert!(exponential_family_tolerance_band(
            &data,
            ExponentialFamily::Gaussian,
            0,
            100,
            0.95,
            42
        )
        .is_none());
        assert!(exponential_family_tolerance_band(
            &data,
            ExponentialFamily::Gaussian,
            3,
            0,
            0.95,
            42
        )
        .is_none());
    }

    // ── Elastic tolerance band tests ──

    fn make_elastic_test_data() -> (FdMatrix, Vec<f64>) {
        let m = 50;
        let t = uniform_grid(m);
        let data = sim_fundata(
            30,
            &t,
            5,
            EFunType::Fourier,
            EValType::Exponential,
            Some(42),
        );
        (data, t)
    }

    #[test]
    fn test_elastic_band_valid_output() {
        let (data, t) = make_elastic_test_data();
        let m = t.len();

        let band = elastic_tolerance_band(&data, &t, 3, 50, 0.95, BandType::Pointwise, 5, 42);
        let band = band.expect("Elastic band should succeed");

        assert_eq!(band.lower.len(), m);
        assert_eq!(band.upper.len(), m);
        assert_eq!(band.center.len(), m);
        assert_eq!(band.half_width.len(), m);
    }

    #[test]
    fn test_elastic_band_lower_less_than_upper() {
        let (data, t) = make_elastic_test_data();
        let m = t.len();

        let band =
            elastic_tolerance_band(&data, &t, 3, 50, 0.95, BandType::Pointwise, 5, 42).unwrap();

        for j in 0..m {
            assert!(
                band.lower[j] < band.upper[j],
                "lower[{j}] = {} >= upper[{j}] = {}",
                band.lower[j],
                band.upper[j]
            );
        }
    }

    #[test]
    fn test_elastic_band_center_within_bounds() {
        let (data, t) = make_elastic_test_data();
        let m = t.len();

        let band =
            elastic_tolerance_band(&data, &t, 3, 50, 0.95, BandType::Pointwise, 5, 42).unwrap();

        for j in 0..m {
            assert!(
                band.center[j] >= band.lower[j] && band.center[j] <= band.upper[j],
                "center[{j}]={} should be in [{}, {}]",
                band.center[j],
                band.lower[j],
                band.upper[j]
            );
        }
    }

    #[test]
    fn test_elastic_band_half_width_positive() {
        let (data, t) = make_elastic_test_data();

        let band =
            elastic_tolerance_band(&data, &t, 3, 50, 0.95, BandType::Pointwise, 5, 42).unwrap();

        for (j, &hw) in band.half_width.iter().enumerate() {
            assert!(hw > 0.0, "half_width[{j}] should be positive, got {hw}");
        }
    }

    #[test]
    fn test_elastic_band_simultaneous() {
        let (data, t) = make_elastic_test_data();
        let m = t.len();

        let band = elastic_tolerance_band(&data, &t, 3, 50, 0.95, BandType::Simultaneous, 5, 42);
        let band = band.expect("Elastic simultaneous band should succeed");

        assert_eq!(band.lower.len(), m);
        for j in 0..m {
            assert!(band.lower[j] < band.upper[j]);
        }
    }

    #[test]
    fn test_elastic_band_deterministic() {
        let (data, t) = make_elastic_test_data();

        let b1 =
            elastic_tolerance_band(&data, &t, 3, 50, 0.95, BandType::Pointwise, 5, 123).unwrap();
        let b2 =
            elastic_tolerance_band(&data, &t, 3, 50, 0.95, BandType::Pointwise, 5, 123).unwrap();

        for j in 0..t.len() {
            assert_eq!(
                b1.lower[j], b2.lower[j],
                "lower[{j}] should be deterministic"
            );
            assert_eq!(
                b1.upper[j], b2.upper[j],
                "upper[{j}] should be deterministic"
            );
        }
    }

    #[test]
    fn test_elastic_band_higher_coverage_wider() {
        let (data, t) = make_elastic_test_data();

        let b90 =
            elastic_tolerance_band(&data, &t, 3, 100, 0.90, BandType::Pointwise, 5, 42).unwrap();
        let b99 =
            elastic_tolerance_band(&data, &t, 3, 100, 0.99, BandType::Pointwise, 5, 42).unwrap();

        let hw90: f64 = b90.half_width.iter().sum();
        let hw99: f64 = b99.half_width.iter().sum();

        assert!(
            hw99 > hw90,
            "99% coverage band should be wider than 90%: hw99={hw99:.4}, hw90={hw90:.4}"
        );
    }

    #[test]
    fn test_elastic_band_invalid_input() {
        let (data, t) = make_elastic_test_data();

        // Wrong argvals length
        let wrong_t = uniform_grid(t.len() + 1);
        assert!(
            elastic_tolerance_band(&data, &wrong_t, 3, 50, 0.95, BandType::Pointwise, 5, 42)
                .is_none()
        );

        // max_iter = 0
        assert!(
            elastic_tolerance_band(&data, &t, 3, 50, 0.95, BandType::Pointwise, 0, 42).is_none()
        );

        // ncomp = 0
        assert!(
            elastic_tolerance_band(&data, &t, 0, 50, 0.95, BandType::Pointwise, 5, 42).is_none()
        );

        // nb = 0
        assert!(
            elastic_tolerance_band(&data, &t, 3, 0, 0.95, BandType::Pointwise, 5, 42).is_none()
        );

        // coverage out of range
        assert!(
            elastic_tolerance_band(&data, &t, 3, 50, 0.0, BandType::Pointwise, 5, 42).is_none()
        );
        assert!(
            elastic_tolerance_band(&data, &t, 3, 50, 1.0, BandType::Pointwise, 5, 42).is_none()
        );

        // Too few observations
        let tiny = FdMatrix::zeros(2, t.len());
        assert!(
            elastic_tolerance_band(&tiny, &t, 1, 10, 0.95, BandType::Pointwise, 5, 42).is_none()
        );
    }
}
