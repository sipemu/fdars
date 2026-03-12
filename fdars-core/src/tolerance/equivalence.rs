use super::degras::{generate_multiplier_weights, residual_sigma};
use super::helpers::{build_band, percentile_sorted};
use super::{EquivalenceBootstrap, EquivalenceTestResult, MultiplierDistribution};
use crate::fdata::mean_1d;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use rand::prelude::*;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─── Equivalence Test (TOST) ────────────────────────────────────────────────

/// Bootstrap sup-norm statistics for two-sample multiplier bootstrap.
fn equivalence_multiplier_sup_stats(
    centered1: &FdMatrix,
    centered2: &FdMatrix,
    pooled_se: &[f64],
    n1: usize,
    n2: usize,
    m: usize,
    nb: usize,
    multiplier: MultiplierDistribution,
    seed: u64,
) -> Vec<f64> {
    let n1f = n1 as f64;
    let n2f = n2 as f64;
    iter_maybe_parallel!(0..nb)
        .map(|b| {
            let mut rng = StdRng::seed_from_u64(seed + b as u64);
            let g1 = generate_multiplier_weights(&mut rng, n1, multiplier);
            let g2 = generate_multiplier_weights(&mut rng, n2, multiplier);
            (0..m)
                .map(|j| {
                    let s1: f64 = (0..n1).map(|i| g1[i] * centered1[(i, j)]).sum::<f64>() / n1f;
                    let s2: f64 = (0..n2).map(|i| g2[i] * centered2[(i, j)]).sum::<f64>() / n2f;
                    ((s1 - s2) / pooled_se[j]).abs()
                })
                .fold(0.0_f64, f64::max)
        })
        .collect()
}

/// Bootstrap sup-norm statistics for two-sample percentile bootstrap.
fn equivalence_percentile_sup_stats(
    data1: &FdMatrix,
    data2: &FdMatrix,
    d_hat: &[f64],
    n1: usize,
    n2: usize,
    m: usize,
    nb: usize,
    seed: u64,
) -> Vec<f64> {
    iter_maybe_parallel!(0..nb)
        .map(|b| {
            let mut rng = StdRng::seed_from_u64(seed + b as u64);
            let idx1: Vec<usize> = (0..n1).map(|_| rng.gen_range(0..n1)).collect();
            let idx2: Vec<usize> = (0..n2).map(|_| rng.gen_range(0..n2)).collect();
            (0..m)
                .map(|j| {
                    let m1: f64 = idx1.iter().map(|&i| data1[(i, j)]).sum::<f64>() / n1 as f64;
                    let m2: f64 = idx2.iter().map(|&i| data2[(i, j)]).sum::<f64>() / n2 as f64;
                    ((m1 - m2) - d_hat[j]).abs()
                })
                .fold(0.0_f64, f64::max)
        })
        .collect()
}

/// Bootstrap sup-norm statistics for one-sample multiplier bootstrap.
fn equivalence_one_sample_multiplier_stats(
    centered: &FdMatrix,
    sigma: &[f64],
    n: usize,
    m: usize,
    nb: usize,
    multiplier: MultiplierDistribution,
    seed: u64,
) -> Vec<f64> {
    let sqrt_n = (n as f64).sqrt();
    iter_maybe_parallel!(0..nb)
        .map(|b| {
            let mut rng = StdRng::seed_from_u64(seed + b as u64);
            let g = generate_multiplier_weights(&mut rng, n, multiplier);
            (0..m)
                .map(|j| {
                    let z: f64 =
                        (0..n).map(|i| g[i] * centered[(i, j)]).sum::<f64>() / (sqrt_n * sigma[j]);
                    z.abs()
                })
                .fold(0.0_f64, f64::max)
        })
        .collect()
}

/// Bootstrap sup-norm statistics for one-sample percentile bootstrap.
fn equivalence_one_sample_percentile_stats(
    data: &FdMatrix,
    mu0: &[f64],
    d_hat: &[f64],
    n: usize,
    m: usize,
    nb: usize,
    seed: u64,
) -> Vec<f64> {
    iter_maybe_parallel!(0..nb)
        .map(|b| {
            let mut rng = StdRng::seed_from_u64(seed + b as u64);
            let idx: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
            (0..m)
                .map(|j| {
                    let boot_mean: f64 = idx.iter().map(|&i| data[(i, j)]).sum::<f64>() / n as f64;
                    ((boot_mean - mu0[j]) - d_hat[j]).abs()
                })
                .fold(0.0_f64, f64::max)
        })
        .collect()
}

/// Validate common equivalence test parameters.
fn valid_equivalence_params(delta: f64, alpha: f64, nb: usize) -> bool {
    delta > 0.0 && alpha > 0.0 && alpha < 0.5 && nb > 0
}

/// Build an [`EquivalenceTestResult`] from bootstrap sup-norm statistics.
///
/// When `se` is non-empty the band is scaled per-point (multiplier bootstrap);
/// otherwise a constant half-width is used (percentile bootstrap).
fn build_equivalence_result(
    mut sup_stats: Vec<f64>,
    d_hat: Vec<f64>,
    se: &[f64],
    delta: f64,
    alpha: f64,
    nb: usize,
) -> EquivalenceTestResult {
    let m = d_hat.len();
    let test_statistic = d_hat.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let use_se = !se.is_empty();
    let c_alpha = percentile_sorted(&mut sup_stats, 1.0 - 2.0 * alpha);

    let half_width: Vec<f64> = if use_se {
        se.iter().map(|&s| c_alpha * s).collect()
    } else {
        vec![c_alpha; m]
    };
    let scb = build_band(d_hat.clone(), half_width);

    let equivalent = scb.upper.iter().all(|&u| u < delta) && scb.lower.iter().all(|&l| l > -delta);

    let c_threshold = if use_se {
        (0..m)
            .map(|j| (delta - d_hat[j].abs()) / se[j])
            .fold(f64::INFINITY, f64::min)
    } else {
        delta - test_statistic
    };
    let p_value = if c_threshold <= 0.0 {
        1.0
    } else {
        let count = sup_stats.iter().filter(|&&t| t >= c_threshold).count();
        (count as f64 / nb as f64).min(1.0)
    };

    EquivalenceTestResult {
        test_statistic,
        p_value,
        critical_value: c_alpha,
        scb,
        equivalent,
        delta,
        alpha,
    }
}

/// Test functional equivalence between two groups using TOST.
///
/// Determines whether the mean difference between two groups of functional
/// observations lies entirely within the margin \[-delta, delta\] using the sup-norm.
///
/// # Arguments
/// * `data1` — Functional data matrix for group 1 (n1 x m)
/// * `data2` — Functional data matrix for group 2 (n2 x m)
/// * `delta` — Equivalence margin (delta > 0)
/// * `alpha` — Significance level (0 < alpha < 0.5)
/// * `nb` — Number of bootstrap replicates
/// * `bootstrap` — Bootstrap method ([`EquivalenceBootstrap`])
/// * `seed` — Random seed for reproducibility
///
/// # Returns
/// `Some(EquivalenceTestResult)` on success, `None` if inputs are invalid.
pub fn equivalence_test(
    data1: &FdMatrix,
    data2: &FdMatrix,
    delta: f64,
    alpha: f64,
    nb: usize,
    bootstrap: EquivalenceBootstrap,
    seed: u64,
) -> Option<EquivalenceTestResult> {
    let (n1, m1) = data1.shape();
    let (n2, m2) = data2.shape();
    if n1 < 3 || n2 < 3 || m1 != m2 || m1 == 0 || !valid_equivalence_params(delta, alpha, nb) {
        return None;
    }
    let m = m1;

    let mean1 = mean_1d(data1);
    let mean2 = mean_1d(data2);
    let d_hat: Vec<f64> = (0..m).map(|j| mean1[j] - mean2[j]).collect();

    let (sup_stats, se) = match bootstrap {
        EquivalenceBootstrap::Multiplier(mult) => {
            let c1 = crate::fdata::center_1d(data1);
            let c2 = crate::fdata::center_1d(data2);
            let sig1 = residual_sigma(data1, &mean1, n1, m);
            let sig2 = residual_sigma(data2, &mean2, n2, m);
            let pse: Vec<f64> = (0..m)
                .map(|j| {
                    (sig1[j] * sig1[j] / n1 as f64 + sig2[j] * sig2[j] / n2 as f64)
                        .sqrt()
                        .max(1e-15)
                })
                .collect();
            let stats = equivalence_multiplier_sup_stats(&c1, &c2, &pse, n1, n2, m, nb, mult, seed);
            (stats, pse)
        }
        EquivalenceBootstrap::Percentile => {
            let stats = equivalence_percentile_sup_stats(data1, data2, &d_hat, n1, n2, m, nb, seed);
            (stats, vec![])
        }
    };

    Some(build_equivalence_result(
        sup_stats, d_hat, &se, delta, alpha, nb,
    ))
}

/// Test functional equivalence of one group's mean to a reference function.
///
/// Determines whether the mean of functional observations differs from
/// a reference function mu_0 by at most delta in sup-norm.
///
/// # Arguments
/// * `data` — Functional data matrix (n x m)
/// * `mu0` — Reference function values (length m)
/// * `delta` — Equivalence margin (delta > 0)
/// * `alpha` — Significance level (0 < alpha < 0.5)
/// * `nb` — Number of bootstrap replicates
/// * `bootstrap` — Bootstrap method ([`EquivalenceBootstrap`])
/// * `seed` — Random seed for reproducibility
///
/// # Returns
/// `Some(EquivalenceTestResult)` on success, `None` if inputs are invalid.
pub fn equivalence_test_one_sample(
    data: &FdMatrix,
    mu0: &[f64],
    delta: f64,
    alpha: f64,
    nb: usize,
    bootstrap: EquivalenceBootstrap,
    seed: u64,
) -> Option<EquivalenceTestResult> {
    let (n, m) = data.shape();
    if n < 3 || m == 0 || mu0.len() != m || !valid_equivalence_params(delta, alpha, nb) {
        return None;
    }

    let mean = mean_1d(data);
    let d_hat: Vec<f64> = (0..m).map(|j| mean[j] - mu0[j]).collect();

    let (sup_stats, se) = match bootstrap {
        EquivalenceBootstrap::Multiplier(mult) => {
            let centered = crate::fdata::center_1d(data);
            let sigma = residual_sigma(data, &mean, n, m);
            let se: Vec<f64> = sigma
                .iter()
                .map(|&s| (s / (n as f64).sqrt()).max(1e-15))
                .collect();
            let stats =
                equivalence_one_sample_multiplier_stats(&centered, &sigma, n, m, nb, mult, seed);
            (stats, se)
        }
        EquivalenceBootstrap::Percentile => {
            let stats = equivalence_one_sample_percentile_stats(data, mu0, &d_hat, n, m, nb, seed);
            (stats, vec![])
        }
    };

    Some(build_equivalence_result(
        sup_stats, d_hat, &se, delta, alpha, nb,
    ))
}
