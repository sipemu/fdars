//! Average Run Length (ARL) computation for SPM control charts.
//!
//! Provides Monte Carlo simulation of in-control (ARL0) and
//! out-of-control (ARL1) average run lengths for T-squared, SPE,
//! and EWMA-T-squared charts.

use crate::error::FdarError;
use crate::iter_maybe_parallel;

#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

/// Configuration for ARL simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct ArlConfig {
    /// Number of simulation replicates (default 10_000).
    pub n_simulations: usize,
    /// Maximum run length before truncation (default 5_000).
    pub max_run_length: usize,
    /// Random seed (default 42).
    pub seed: u64,
}

impl Default for ArlConfig {
    fn default() -> Self {
        Self {
            n_simulations: 10_000,
            max_run_length: 5_000,
            seed: 42,
        }
    }
}

/// Result of ARL computation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ArlResult {
    /// Average run length.
    pub arl: f64,
    /// Standard deviation of run lengths.
    pub std_dev: f64,
    /// Median run length.
    pub median_rl: f64,
    /// Individual run lengths from each simulation.
    pub run_lengths: Vec<usize>,
}

/// Compute in-control ARL for T-squared chart (ARL0).
///
/// Simulates observations from N(0, diag(eigenvalues)) and computes
/// T² = Σ score²/eigenvalue. Counts steps until T² > ucl.
///
/// # Arguments
/// * `eigenvalues` - Eigenvalues (one per PC)
/// * `ucl` - Upper control limit for T-squared
/// * `config` - ARL simulation configuration
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `eigenvalues` is empty.
/// Returns [`FdarError::InvalidParameter`] if `ucl` is not positive.
///
/// # Example
/// ```
/// use fdars_core::spm::arl::{arl0_t2, ArlConfig};
/// let eigenvalues = vec![2.0, 1.0];
/// let ucl = 5.991; // chi2(0.95, 2)
/// let config = ArlConfig { n_simulations: 1000, max_run_length: 500, seed: 42 };
/// let result = arl0_t2(&eigenvalues, ucl, &config).unwrap();
/// assert!(result.arl > 1.0);
/// assert!(result.std_dev > 0.0);
/// ```
#[must_use = "ARL result should not be discarded"]
pub fn arl0_t2(eigenvalues: &[f64], ucl: f64, config: &ArlConfig) -> Result<ArlResult, FdarError> {
    arl_t2_impl(eigenvalues, ucl, &vec![0.0; eigenvalues.len()], config)
}

/// Compute out-of-control ARL for T-squared chart (ARL1).
///
/// Simulates observations from N(shift, diag(eigenvalues)) and computes
/// T² = Σ score²/eigenvalue. Counts steps until T² > ucl.
///
/// # Arguments
/// * `eigenvalues` - Eigenvalues (one per PC)
/// * `ucl` - Upper control limit
/// * `shift` - Mean shift vector (one per PC)
/// * `config` - ARL simulation configuration
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if dimensions don't match.
/// Returns [`FdarError::InvalidParameter`] if `ucl` is not positive.
#[must_use = "ARL result should not be discarded"]
pub fn arl1_t2(
    eigenvalues: &[f64],
    ucl: f64,
    shift: &[f64],
    config: &ArlConfig,
) -> Result<ArlResult, FdarError> {
    if shift.len() != eigenvalues.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "shift",
            expected: format!("{}", eigenvalues.len()),
            actual: format!("{}", shift.len()),
        });
    }
    arl_t2_impl(eigenvalues, ucl, shift, config)
}

/// Compute in-control ARL for EWMA-T-squared chart.
///
/// Simulates observations from N(0, diag(eigenvalues)), applies EWMA
/// smoothing with parameter lambda, and computes T² on the smoothed
/// scores with adjusted eigenvalues lambda/(2-lambda) * eigenvalue.
///
/// # Arguments
/// * `eigenvalues` - Eigenvalues (one per PC)
/// * `ucl` - Upper control limit for EWMA T-squared
/// * `lambda` - EWMA smoothing parameter in (0, 1]
/// * `config` - ARL simulation configuration
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `lambda` is not in (0, 1].
#[must_use = "ARL result should not be discarded"]
pub fn arl0_ewma_t2(
    eigenvalues: &[f64],
    ucl: f64,
    lambda: f64,
    config: &ArlConfig,
) -> Result<ArlResult, FdarError> {
    validate_common(eigenvalues, ucl)?;
    if lambda <= 0.0 || lambda > 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "lambda",
            message: format!("lambda must be in (0, 1], got {lambda}"),
        });
    }

    let ncomp = eigenvalues.len();
    let adj_eigenvalues: Vec<f64> = eigenvalues
        .iter()
        .map(|&ev| ev * lambda / (2.0 - lambda))
        .collect();

    let run_lengths: Vec<usize> =
        iter_maybe_parallel!((0..config.n_simulations).collect::<Vec<_>>())
            .map(|rep| {
                let mut rng = StdRng::seed_from_u64(config.seed + rep as u64);
                let mut z = vec![0.0_f64; ncomp];
                for step in 1..=config.max_run_length {
                    // Generate N(0, eigenvalue) scores and apply EWMA
                    for l in 0..ncomp {
                        let xi: f64 = rng.sample::<f64, _>(StandardNormal) * eigenvalues[l].sqrt();
                        z[l] = lambda * xi + (1.0 - lambda) * z[l];
                    }
                    // T² on smoothed scores with adjusted eigenvalues
                    let t2: f64 = z
                        .iter()
                        .zip(adj_eigenvalues.iter())
                        .map(|(&zl, &ev)| zl * zl / ev)
                        .sum();
                    if t2 > ucl {
                        return step;
                    }
                }
                config.max_run_length
            })
            .collect();

    Ok(build_result(run_lengths))
}

/// Compute in-control ARL for SPE chart.
///
/// Simulates SPE values as `scale * chi²(df)` using a Gamma distribution.
///
/// # Arguments
/// * `spe_df` - Estimated degrees of freedom for SPE distribution
/// * `spe_scale` - Estimated scale for SPE distribution
/// * `ucl` - Upper control limit for SPE
/// * `config` - ARL simulation configuration
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if parameters are non-positive.
#[must_use = "ARL result should not be discarded"]
pub fn arl0_spe(
    spe_df: f64,
    spe_scale: f64,
    ucl: f64,
    config: &ArlConfig,
) -> Result<ArlResult, FdarError> {
    if spe_df <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "spe_df",
            message: format!("spe_df must be positive, got {spe_df}"),
        });
    }
    if spe_scale <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "spe_scale",
            message: format!("spe_scale must be positive, got {spe_scale}"),
        });
    }
    if ucl <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ucl",
            message: format!("ucl must be positive, got {ucl}"),
        });
    }

    // chi²(df) = Gamma(df/2, 2), so scale*chi²(df) = Gamma(df/2, 2*scale)
    let shape = spe_df / 2.0;
    let gamma_scale = 2.0 * spe_scale;

    let run_lengths: Vec<usize> =
        iter_maybe_parallel!((0..config.n_simulations).collect::<Vec<_>>())
            .map(|rep| {
                let mut rng = StdRng::seed_from_u64(config.seed + rep as u64);
                for step in 1..=config.max_run_length {
                    let spe = sample_gamma(&mut rng, shape) * gamma_scale;
                    if spe > ucl {
                        return step;
                    }
                }
                config.max_run_length
            })
            .collect();

    Ok(build_result(run_lengths))
}

// ── Internal helpers ─────────────────────────────────────────────────────

fn validate_common(eigenvalues: &[f64], ucl: f64) -> Result<(), FdarError> {
    if eigenvalues.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "eigenvalues",
            expected: "at least 1 eigenvalue".to_string(),
            actual: "0 eigenvalues".to_string(),
        });
    }
    if ucl <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ucl",
            message: format!("ucl must be positive, got {ucl}"),
        });
    }
    for (l, &ev) in eigenvalues.iter().enumerate() {
        if ev <= 0.0 {
            return Err(FdarError::InvalidParameter {
                parameter: "eigenvalues",
                message: format!("eigenvalue[{l}] = {ev} must be positive"),
            });
        }
    }
    Ok(())
}

fn arl_t2_impl(
    eigenvalues: &[f64],
    ucl: f64,
    shift: &[f64],
    config: &ArlConfig,
) -> Result<ArlResult, FdarError> {
    validate_common(eigenvalues, ucl)?;
    let ncomp = eigenvalues.len();

    let run_lengths: Vec<usize> =
        iter_maybe_parallel!((0..config.n_simulations).collect::<Vec<_>>())
            .map(|rep| {
                let mut rng = StdRng::seed_from_u64(config.seed + rep as u64);
                for step in 1..=config.max_run_length {
                    let mut t2 = 0.0_f64;
                    for l in 0..ncomp {
                        let score: f64 =
                            shift[l] + rng.sample::<f64, _>(StandardNormal) * eigenvalues[l].sqrt();
                        t2 += score * score / eigenvalues[l];
                    }
                    if t2 > ucl {
                        return step;
                    }
                }
                config.max_run_length
            })
            .collect();

    Ok(build_result(run_lengths))
}

fn build_result(run_lengths: Vec<usize>) -> ArlResult {
    let n = run_lengths.len() as f64;
    let arl = run_lengths.iter().map(|&r| r as f64).sum::<f64>() / n;
    let var = run_lengths
        .iter()
        .map(|&r| {
            let d = r as f64 - arl;
            d * d
        })
        .sum::<f64>()
        / (n - 1.0);
    let std_dev = var.sqrt();

    let mut sorted: Vec<usize> = run_lengths.clone();
    sorted.sort_unstable();
    let median_rl = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) as f64 / 2.0
    } else {
        sorted[sorted.len() / 2] as f64
    };

    ArlResult {
        arl,
        std_dev,
        median_rl,
        run_lengths,
    }
}

/// Sample from Gamma(shape, 1) using Marsaglia and Tsang's method.
///
/// Reference: Marsaglia, G. & Tsang, W.W. (2000). A simple method for
/// generating gamma variables. *ACM Transactions on Mathematical Software*,
/// 26(3), 363-372.
///
/// The squeeze test constant 0.0331 = (1/3) * (1/(3*(shape-1/3)))^2
/// (Eq. 6 of Marsaglia & Tsang 2000) provides a fast rejection path.
fn sample_gamma(rng: &mut StdRng, shape: f64) -> f64 {
    if shape < 1.0 {
        // Gamma(a, 1) = Gamma(a+1, 1) * U^(1/a)
        let g = sample_gamma(rng, shape + 1.0);
        let u: f64 = rng.gen();
        return g * u.powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x: f64 = rng.sample(StandardNormal);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u: f64 = rng.gen();
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}
