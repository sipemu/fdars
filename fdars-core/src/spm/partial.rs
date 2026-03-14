//! Partial-domain monitoring for functional data.
//!
//! Monitors processes where only a partial observation of the functional
//! domain is available (e.g., real-time monitoring of an ongoing process).
//! Supports three strategies for handling the unobserved domain:
//! - Conditional expectation (BLUP)
//! - Partial projection (scaled inner products)
//! - Zero padding

use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;

use super::phase::SpmChart;
use super::stats::hotelling_t2;

/// Strategy for handling unobserved domain.
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
#[derive(Debug, Clone, PartialEq)]
pub struct PartialDomainConfig {
    /// Number of principal components (default 5).
    pub ncomp: usize,
    /// Significance level (default 0.05).
    pub alpha: f64,
    /// Domain completion strategy (default: ConditionalExpectation).
    pub completion: DomainCompletion,
}

impl Default for PartialDomainConfig {
    fn default() -> Self {
        Self {
            ncomp: 5,
            alpha: 0.05,
            completion: DomainCompletion::ConditionalExpectation,
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
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if dimensions are inconsistent.
/// Returns [`FdarError::InvalidParameter`] if `n_observed` is 0.
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

    let ncomp = config.ncomp.min(chart.eigenvalues.len());
    let domain_fraction = if m > 1 {
        (argvals[n_observed.min(m) - 1] - argvals[0]) / (argvals[m - 1] - argvals[0])
    } else {
        1.0
    };

    let (scores, completed_curve) = match &config.completion {
        DomainCompletion::ConditionalExpectation => {
            conditional_expectation(chart, partial_values, argvals, n_observed, ncomp)?
        }
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
) -> Result<(Vec<f64>, Option<Vec<f64>>), FdarError> {
    let m = chart.fpca.mean.len();
    let n_obs = n_observed.min(m);

    // Centered observed values
    let y_obs: Vec<f64> = (0..n_obs)
        .map(|j| partial_values[j] - chart.fpca.mean[j])
        .collect();

    // Estimate sigma^2 from mean SPE (use a small floor)
    let sigma2 = if !chart.spe_phase1.is_empty() {
        let mean_spe: f64 = chart.spe_phase1.iter().sum::<f64>() / chart.spe_phase1.len() as f64;
        // Normalize by grid size to get per-point variance
        (mean_spe / m as f64).max(1e-10)
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

    // Right-hand side: sigma^{-2} Phi^T y
    let rhs: Vec<f64> = phi_t_y.iter().map(|&v| v / sigma2).collect();

    // Solve M * scores = rhs using Cholesky (M is SPD)
    let scores = solve_spd(&m_matrix, &rhs, ncomp)?;

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
    let partial_argvals = &argvals[..n_obs];
    let weights = simpsons_weights(partial_argvals);

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
        if diag <= 0.0 {
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
