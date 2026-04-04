//! Bootstrap confidence intervals for curve shapes in the elastic metric.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use super::karcher::karcher_mean;
use super::pairwise::elastic_align_pair;
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Configuration for shape bootstrap confidence intervals.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeCiConfig {
    /// Number of bootstrap resamples.
    pub n_bootstrap: usize,
    /// Confidence level (e.g., 0.95 for 95% CI).
    pub confidence_level: f64,
    /// Roughness penalty for elastic alignment.
    pub lambda: f64,
    /// Maximum Karcher mean iterations.
    pub max_iter: usize,
    /// Convergence tolerance for the Karcher mean.
    pub tol: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for ShapeCiConfig {
    fn default() -> Self {
        Self {
            n_bootstrap: 200,
            confidence_level: 0.95,
            lambda: 0.0,
            max_iter: 15,
            tol: 1e-3,
            seed: 42,
        }
    }
}

/// Result of shape bootstrap confidence interval computation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ShapeCiResult {
    /// Karcher mean of the full sample.
    pub mean: Vec<f64>,
    /// Pointwise lower confidence band (length m).
    pub lower_band: Vec<f64>,
    /// Pointwise upper confidence band (length m).
    pub upper_band: Vec<f64>,
    /// Bootstrap Karcher means (n_bootstrap x m).
    pub bootstrap_means: FdMatrix,
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Compute bootstrap confidence intervals for the elastic Karcher mean.
///
/// Resamples the input curves with replacement, computes the Karcher mean
/// of each bootstrap sample, aligns each bootstrap mean to the full-sample
/// mean, and derives pointwise confidence bands from the empirical quantiles.
///
/// # Arguments
/// * `data`    - Functional data matrix (n x m).
/// * `argvals` - Evaluation points (length m).
/// * `config`  - Bootstrap configuration.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if `n < 3` or `argvals` length
/// does not match `m`.
/// Returns [`FdarError::InvalidParameter`] if `confidence_level` is not in
/// `(0, 1)` or `n_bootstrap < 1`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn shape_confidence_interval(
    data: &FdMatrix,
    argvals: &[f64],
    config: &ShapeCiConfig,
) -> Result<ShapeCiResult, FdarError> {
    let (n, m) = data.shape();

    // ── Validation ──
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 rows".to_string(),
            actual: format!("{n} rows"),
        });
    }
    if config.confidence_level <= 0.0 || config.confidence_level >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "confidence_level",
            message: format!("must be in (0, 1), got {}", config.confidence_level),
        });
    }
    if config.n_bootstrap < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_bootstrap",
            message: format!("must be >= 1, got {}", config.n_bootstrap),
        });
    }

    // ── Full-sample Karcher mean ──
    let full_karcher = karcher_mean(data, argvals, config.max_iter, config.tol, config.lambda);

    // ── Bootstrap loop ──
    let boot_means: Vec<Vec<f64>> = iter_maybe_parallel!(0..config.n_bootstrap)
        .map(|b| {
            let mut rng = StdRng::seed_from_u64(config.seed + b as u64);

            // Resample n indices with replacement
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();

            // Build bootstrap matrix
            let mut boot_data = FdMatrix::zeros(n, m);
            for (row, &idx) in indices.iter().enumerate() {
                for j in 0..m {
                    boot_data[(row, j)] = data[(idx, j)];
                }
            }

            // Compute bootstrap Karcher mean
            let boot_karcher = karcher_mean(
                &boot_data,
                argvals,
                config.max_iter,
                config.tol,
                config.lambda,
            );

            // Align bootstrap mean to full-sample mean
            let aligned = elastic_align_pair(
                &full_karcher.mean,
                &boot_karcher.mean,
                argvals,
                config.lambda,
            );

            aligned.f_aligned
        })
        .collect();

    // ── Build bootstrap_means matrix ──
    let mut bootstrap_means = FdMatrix::zeros(config.n_bootstrap, m);
    for (b, bm) in boot_means.iter().enumerate() {
        for j in 0..m {
            bootstrap_means[(b, j)] = bm[j];
        }
    }

    // ── Pointwise confidence bands ──
    let alpha = 1.0 - config.confidence_level;
    let mut lower_band = vec![0.0; m];
    let mut upper_band = vec![0.0; m];

    for j in 0..m {
        let mut col_vals: Vec<f64> = (0..config.n_bootstrap)
            .map(|b| bootstrap_means[(b, j)])
            .collect();
        col_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        lower_band[j] = quantile_sorted(&col_vals, alpha / 2.0);
        upper_band[j] = quantile_sorted(&col_vals, 1.0 - alpha / 2.0);
    }

    Ok(ShapeCiResult {
        mean: full_karcher.mean,
        lower_band,
        upper_band,
        bootstrap_means,
    })
}

use crate::helpers::quantile_sorted;

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{sim_fundata, EFunType, EValType};
    use crate::test_helpers::uniform_grid;

    fn make_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let t = uniform_grid(m);
        let data = sim_fundata(n, &t, 3, EFunType::Fourier, EValType::Exponential, Some(99));
        (data, t)
    }

    #[test]
    fn shape_ci_band_contains_mean() {
        let (data, t) = make_data(8, 20);
        let config = ShapeCiConfig {
            n_bootstrap: 30,
            confidence_level: 0.95,
            max_iter: 5,
            tol: 1e-2,
            ..Default::default()
        };
        let result = shape_confidence_interval(&data, &t, &config).unwrap();
        let m = t.len();
        for j in 0..m {
            assert!(
                result.lower_band[j] <= result.mean[j] + 1e-6
                    && result.mean[j] <= result.upper_band[j] + 1e-6,
                "mean[{j}]={} not in [{}, {}]",
                result.mean[j],
                result.lower_band[j],
                result.upper_band[j],
            );
        }
    }

    #[test]
    fn shape_ci_band_width_positive() {
        let (data, t) = make_data(8, 20);
        let config = ShapeCiConfig {
            n_bootstrap: 30,
            confidence_level: 0.95,
            max_iter: 5,
            tol: 1e-2,
            ..Default::default()
        };
        let result = shape_confidence_interval(&data, &t, &config).unwrap();
        let m = t.len();
        let n_positive = (0..m)
            .filter(|&j| result.upper_band[j] > result.lower_band[j] + 1e-12)
            .count();
        assert!(
            n_positive > m / 2,
            "upper > lower for only {n_positive}/{m} points, expected > {}/{}",
            m / 2,
            m
        );
    }

    #[test]
    fn shape_ci_bootstrap_means_shape() {
        let (data, t) = make_data(6, 20);
        let n_boot = 15;
        let config = ShapeCiConfig {
            n_bootstrap: n_boot,
            confidence_level: 0.90,
            max_iter: 3,
            tol: 1e-2,
            ..Default::default()
        };
        let result = shape_confidence_interval(&data, &t, &config).unwrap();
        assert_eq!(result.bootstrap_means.shape(), (n_boot, t.len()));
    }

    #[test]
    fn shape_ci_rejects_too_few_curves() {
        let t = uniform_grid(20);
        let data = FdMatrix::zeros(2, 20);
        let config = ShapeCiConfig::default();
        assert!(shape_confidence_interval(&data, &t, &config).is_err());
    }
}
