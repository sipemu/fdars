//! Partial-domain monitoring for functional data.
//!
//! Monitors processes where only a partial observation of the functional
//! domain is available (e.g., real-time monitoring of an ongoing process).
//! Supports three strategies for handling the unobserved domain:
//! - Conditional expectation (BLUP)
//! - Partial projection (scaled inner products)
//! - Zero padding
//!
//! # References
//!
//! - Yao, F., Müller, H.G. & Wang, J.L. (2005). Functional data analysis
//!   for sparse longitudinal data. *Journal of the American Statistical
//!   Association*, 100(470), 577-590.

use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;

use super::phase::SpmChart;
use super::stats::hotelling_t2;

/// Strategy for handling unobserved domain.
///
/// # Strategy comparison
///
/// - **ConditionalExpectation** (recommended): Best accuracy when the FPCA model
///   is well-specified. Uses BLUP to predict scores from partial observations.
///   Degrades gracefully as domain fraction decreases.
/// - **PartialProjection**: Computationally cheapest. Scales inner products by
///   domain fraction. Acceptable when domain fraction > 0.7.
/// - **ZeroPad**: Simplest baseline. Fills unobserved region with the mean
///   function. Biases scores toward zero for small domain fractions.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DomainCompletion {
    /// Partial inner products scaled by domain fraction.
    PartialProjection,
    /// Best Linear Unbiased Predictor (Yao et al., 2005).
    ConditionalExpectation,
    /// Pad unobserved region with the mean function.
    ZeroPad,
}

/// Configuration for partial-domain monitoring.
///
/// For domain fractions below 0.3, all strategies produce increasingly
/// uncertain estimates. The conditional expectation (BLUP) degrades most
/// gracefully due to its optimal shrinkage properties.
#[derive(Debug, Clone, PartialEq)]
pub struct PartialDomainConfig {
    /// Number of principal components (default 5).
    pub ncomp: usize,
    /// Significance level (default 0.05).
    pub alpha: f64,
    /// Domain completion strategy (default: ConditionalExpectation).
    pub completion: DomainCompletion,
    /// Tikhonov regularization strength for ill-conditioned systems (default 1e-10).
    ///
    /// Applied as Tikhonov regularization (diagonal loading) when the M matrix
    /// condition number proxy exceeds 1e12. Increase to 1e-6 if you observe
    /// numerical warnings or NaN in scores.
    pub regularization_eps: f64,
}

impl Default for PartialDomainConfig {
    fn default() -> Self {
        Self {
            ncomp: 5,
            alpha: 0.05,
            completion: DomainCompletion::ConditionalExpectation,
            regularization_eps: 1e-10,
        }
    }
}

/// Result of partial-domain monitoring for a single observation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PartialMonitorResult {
    /// Estimated FPC scores.
    pub scores: Vec<f64>,
    /// T-squared statistic.
    pub t2: f64,
    /// Whether T-squared exceeds the control limit.
    pub t2_alarm: bool,
    /// Fraction of domain observed.
    pub domain_fraction: f64,
    /// Completed curve (if using ConditionalExpectation or ZeroPad).
    pub completed_curve: Option<Vec<f64>>,
}

/// Monitor a single partially-observed curve.
///
/// # Arguments
/// * `chart` - Phase I SPM chart
/// * `partial_values` - Observed values (length m, but only first `n_observed` are used)
/// * `argvals` - Full grid points (length m)
/// * `n_observed` - Number of observed grid points (from the start of the domain)
/// * `config` - Partial domain configuration
///
/// For domain fractions below 0.3, conditional expectation accuracy degrades
/// significantly. Consider collecting more of the domain or using a dedicated
/// early-detection method.
///
/// # Example
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::phase::{spm_phase1, SpmConfig};
/// use fdars_core::spm::partial::{spm_monitor_partial, PartialDomainConfig, DomainCompletion};
/// let data = FdMatrix::from_column_major(
///     (0..200).map(|i| (i as f64 * 0.1).sin()).collect(), 20, 10
/// ).unwrap();
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let chart = spm_phase1(&data, &argvals, &SpmConfig { ncomp: 2, ..SpmConfig::default() }).unwrap();
/// let partial_values = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let config = PartialDomainConfig { ncomp: 2, completion: DomainCompletion::ZeroPad, ..PartialDomainConfig::default() };
/// let result = spm_monitor_partial(&chart, &partial_values, &argvals, 5, &config).unwrap();
/// assert!(result.domain_fraction > 0.0);
/// ```
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if dimensions are inconsistent.
/// Returns [`FdarError::InvalidParameter`] if `n_observed` is 0, if `argvals`
/// is not sorted, or if the computed domain fraction is out of range.
#[must_use = "monitoring result should not be discarded"]
pub fn spm_monitor_partial(
    chart: &SpmChart,
    partial_values: &[f64],
    argvals: &[f64],
    n_observed: usize,
    config: &PartialDomainConfig,
) -> Result<PartialMonitorResult, FdarError> {
    let m = chart.fpca.mean.len();
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if partial_values.len() < n_observed {
        return Err(FdarError::InvalidDimension {
            parameter: "partial_values",
            expected: format!("at least {n_observed} values"),
            actual: format!("{} values", partial_values.len()),
        });
    }
    if n_observed == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_observed",
            message: "n_observed must be at least 1".to_string(),
        });
    }
    // Ensure argvals is sorted (strictly increasing endpoints)
    if m > 1 && argvals[0] >= argvals[m - 1] {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "argvals must be sorted (first element must be less than last)".to_string(),
        });
    }

    let ncomp = config.ncomp.min(chart.eigenvalues.len());
    // Domain fraction: ratio of observed domain range to full range.
    // For n_observed=1, the range is zero so we fall back to point-count fraction.
    let domain_fraction = if m <= 1 {
        1.0
    } else {
        let range_fraction =
            (argvals[n_observed.min(m) - 1] - argvals[0]) / (argvals[m - 1] - argvals[0]);
        if range_fraction > 0.0 {
            range_fraction
        } else {
            // Single observed point: use point-count fraction as fallback.
            // The point-count fraction n_obs/m is used when the range-based
            // fraction is zero (single observed point). This is consistent
            // with the PACE framework (Yao et al., 2005) where prediction
            // from a single observation reduces to the marginal BLUP.
            n_observed as f64 / m as f64
        }
    };
    if !(0.0..=1.0).contains(&domain_fraction) {
        return Err(FdarError::InvalidParameter {
            parameter: "domain_fraction",
            message: format!("computed domain_fraction must be in [0, 1], got {domain_fraction}"),
        });
    }

    let (scores, completed_curve) = match &config.completion {
        DomainCompletion::ConditionalExpectation => conditional_expectation(
            chart,
            partial_values,
            argvals,
            n_observed,
            ncomp,
            config.regularization_eps,
        )?,
        DomainCompletion::PartialProjection => {
            let scores = partial_projection(chart, partial_values, argvals, n_observed, ncomp)?;
            (scores, None)
        }
        DomainCompletion::ZeroPad => zero_pad(chart, partial_values, argvals, n_observed, ncomp)?,
    };

    // Compute T-squared
    let eigenvalues = &chart.eigenvalues[..ncomp];
    let score_mat = FdMatrix::from_column_major(scores.clone(), 1, ncomp)?;
    let t2_vec = hotelling_t2(&score_mat, eigenvalues)?;
    let t2 = t2_vec[0];
    let t2_alarm = t2 > chart.t2_limit.ucl;

    Ok(PartialMonitorResult {
        scores,
        t2,
        t2_alarm,
        domain_fraction,
        completed_curve,
    })
}

/// Monitor a batch of partially-observed curves.
///
/// # Arguments
/// * `chart` - Phase I SPM chart
/// * `partial_data` - Slice of (values, n_observed) pairs
/// * `argvals` - Full grid points (length m)
/// * `config` - Partial domain configuration
///
/// # Errors
///
/// Returns errors from individual monitoring calls.
#[must_use = "monitoring results should not be discarded"]
pub fn spm_monitor_partial_batch(
    chart: &SpmChart,
    partial_data: &[(&[f64], usize)],
    argvals: &[f64],
    config: &PartialDomainConfig,
) -> Result<Vec<PartialMonitorResult>, FdarError> {
    partial_data
        .iter()
        .map(|(values, n_obs)| spm_monitor_partial(chart, values, argvals, *n_obs, config))
        .collect()
}

// -- Completion strategies ------------------------------------------------

/// Conditional Expectation (BLUP):
/// scores = (Lambda^{-1} + sigma^{-2} Phi_obs^T Phi_obs)^{-1} sigma^{-2} Phi_obs^T y_obs
///
/// Where:
/// - Lambda = diag(eigenvalues) (ncomp x ncomp)
/// - Phi_obs = rotation[0..n_observed, :] (n_observed x ncomp)
/// - y_obs = partial_values[0..n_observed] - mean[0..n_observed]
/// - sigma^2 estimated from SPE
fn conditional_expectation(
    chart: &SpmChart,
    partial_values: &[f64],
    _argvals: &[f64],
    n_observed: usize,
    ncomp: usize,
    reg_eps: f64,
) -> Result<(Vec<f64>, Option<Vec<f64>>), FdarError> {
    let m = chart.fpca.mean.len();
    let n_obs = n_observed.min(m);

    // Centered observed values
    let y_obs: Vec<f64> = (0..n_obs)
        .map(|j| partial_values[j] - chart.fpca.mean[j])
        .collect();

    // Estimate sigma^2 from median SPE (robust to outliers, Yao et al. 2005)
    let sigma2 = if !chart.spe_phase1.is_empty() {
        let mut sorted_spe = chart.spe_phase1.clone();
        sorted_spe.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted_spe.len() / 2;
        let median_spe = if sorted_spe.len() % 2 == 0 {
            (sorted_spe[mid - 1] + sorted_spe[mid]) / 2.0
        } else {
            sorted_spe[mid]
        };
        // Normalize by grid size to get per-point variance
        (median_spe / m as f64).max(1e-10)
    } else {
        1e-6
    };

    // Compute Phi_obs^T y_obs (ncomp vector)
    let phi_t_y: Vec<f64> = (0..ncomp)
        .map(|l| {
            (0..n_obs)
                .map(|j| chart.fpca.rotation[(j, l)] * y_obs[j])
                .sum::<f64>()
        })
        .collect();

    // Use the Woodbury identity to work with the small ncomp x ncomp system:
    // M = Lambda^{-1} + sigma^{-2} Phi_obs^T Phi_obs
    // scores = M^{-1} sigma^{-2} Phi_obs^T y_obs
    let mut m_matrix = vec![0.0_f64; ncomp * ncomp];
    for l in 0..ncomp {
        // Diagonal: Lambda^{-1}
        m_matrix[l * ncomp + l] += 1.0 / chart.eigenvalues[l];
        // Add sigma^{-2} Phi^T Phi
        for k in 0..ncomp {
            let phi_t_phi: f64 = (0..n_obs)
                .map(|j| chart.fpca.rotation[(j, l)] * chart.fpca.rotation[(j, k)])
                .sum();
            m_matrix[l * ncomp + k] += phi_t_phi / sigma2;
        }
    }

    // Check condition number via diagonal ratio (cheap proxy)
    let mut diag_min = f64::INFINITY;
    let mut diag_max = 0.0_f64;
    for l in 0..ncomp {
        let d = m_matrix[l * ncomp + l];
        if d > 0.0 {
            diag_min = diag_min.min(d);
            diag_max = diag_max.max(d);
        }
    }
    if diag_max > 0.0 && diag_max / diag_min > 1e12 {
        // Ill-conditioned: add Tikhonov regularization
        let reg = diag_max * reg_eps;
        for l in 0..ncomp {
            m_matrix[l * ncomp + l] += reg;
        }
    }

    // Right-hand side: sigma^{-2} Phi^T y
    let rhs: Vec<f64> = phi_t_y.iter().map(|&v| v / sigma2).collect();

    // Solve M * scores = rhs using Cholesky (M is SPD).
    // If Cholesky fails (near-indefinite), retry with progressively stronger regularization.
    let scores = match solve_spd(&m_matrix, &rhs, ncomp) {
        Ok(s) => s,
        Err(_) => {
            // Retry with stronger Tikhonov regularization
            let mut m_reg = m_matrix.clone();
            let mut reg_strength = diag_max * 1e-8;
            let mut result = None;
            for _ in 0..5 {
                for l in 0..ncomp {
                    m_reg[l * ncomp + l] = m_matrix[l * ncomp + l] + reg_strength;
                }
                if let Ok(s) = solve_spd(&m_reg, &rhs, ncomp) {
                    result = Some(s);
                    break;
                }
                reg_strength *= 10.0;
            }
            result.ok_or(FdarError::ComputationFailed {
                operation: "conditional_expectation",
                detail: "Cholesky failed even with strong regularization".to_string(),
            })?
        }
    };

    // Reconstruct the full curve: mean + Phi * scores
    let mut completed = chart.fpca.mean.clone();
    for j in 0..m {
        for l in 0..ncomp {
            completed[j] += chart.fpca.rotation[(j, l)] * scores[l];
        }
    }

    Ok((scores, Some(completed)))
}

/// Partial projection: scale inner products by domain fraction.
///
/// The scaling factor (full_total / partial_total) compensates for the reduced
/// integration domain, assuming the eigenfunction structure is approximately
/// uniform across the domain. This is a first-order correction that degrades
/// when eigenfunctions have localized support outside the observed region.
fn partial_projection(
    chart: &SpmChart,
    partial_values: &[f64],
    argvals: &[f64],
    n_observed: usize,
    ncomp: usize,
) -> Result<Vec<f64>, FdarError> {
    let m = chart.fpca.mean.len();
    let n_obs = n_observed.min(m);

    // Centered observed values
    let y_obs: Vec<f64> = (0..n_obs)
        .map(|j| partial_values[j] - chart.fpca.mean[j])
        .collect();

    // Integration weights for partial domain
    let partialargvals = &argvals[..n_obs];
    let weights = simpsons_weights(partialargvals);

    // Full domain weights (for normalization)
    let full_weights = simpsons_weights(argvals);
    let full_total: f64 = full_weights.iter().sum();
    let partial_total: f64 = weights.iter().sum();

    // Scale factor: full_domain_width / partial_domain_width
    let scale = if partial_total > 0.0 {
        full_total / partial_total
    } else {
        1.0
    };

    // Compute scores as scaled partial inner products
    let scores: Vec<f64> = (0..ncomp)
        .map(|l| {
            let ip: f64 = (0..n_obs)
                .map(|j| y_obs[j] * chart.fpca.rotation[(j, l)] * weights[j])
                .sum();
            ip * scale
        })
        .collect();

    Ok(scores)
}

/// Zero-pad: fill unobserved with mean, project normally.
fn zero_pad(
    chart: &SpmChart,
    partial_values: &[f64],
    _argvals: &[f64],
    n_observed: usize,
    ncomp: usize,
) -> Result<(Vec<f64>, Option<Vec<f64>>), FdarError> {
    let m = chart.fpca.mean.len();
    let n_obs = n_observed.min(m);

    // Build padded curve: observed values + mean for unobserved
    let mut padded = chart.fpca.mean.clone();
    padded[..n_obs].copy_from_slice(&partial_values[..n_obs]);

    // Center and project
    let mut centered = vec![0.0; m];
    for j in 0..m {
        centered[j] = padded[j] - chart.fpca.mean[j];
    }

    let scores: Vec<f64> = (0..ncomp)
        .map(|l| {
            (0..m)
                .map(|j| centered[j] * chart.fpca.rotation[(j, l)])
                .sum()
        })
        .collect();

    Ok((scores, Some(padded)))
}

// -- Small linear algebra helpers -----------------------------------------

/// Solve A*x = b where A is symmetric positive definite, via Cholesky.
fn solve_spd(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, FdarError> {
    // Cholesky factorization: A = L L^T
    let mut l = vec![0.0_f64; n * n];

    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j * n + k] * l[j * n + k];
        }
        let diag = a[j * n + j] - sum;
        if diag <= 0.0 || diag.is_nan() {
            return Err(FdarError::ComputationFailed {
                operation: "cholesky",
                detail: "matrix is not positive definite".to_string(),
            });
        }
        l[j * n + j] = diag.sqrt();

        for i in (j + 1)..n {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..i {
            sum += l[i * n + k] * y[k];
        }
        y[i] = (b[i] - sum) / l[i * n + i];
    }

    // Back substitution: L^T x = y
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for k in (i + 1)..n {
            sum += l[k * n + i] * x[k];
        }
        x[i] = (y[i] - sum) / l[i * n + i];
    }

    Ok(x)
}
