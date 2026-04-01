//! Profile monitoring for functional data.
//!
//! Monitors the relationship between scalar predictors and functional
//! responses over time using Function-on-Scalar Regression (FOSR).
//! Detects changes in the coefficient functions beta(t) via FPCA and T-squared.
//!
//! # Mathematical framework
//!
//! The functional response model is y_i(t) = x_i^T beta(t) + epsilon_i(t),
//! where beta(t) = [beta_1(t), ..., beta_p(t)]^T are coefficient functions.
//! Rolling FOSR estimates beta(t) within each window, producing a sequence
//! of vectorized coefficient functions. FPCA extracts the dominant modes of
//! variation in the beta(t) sequence, and T-squared monitors for shifts
//! in the score distribution.
//!
//! Beta coefficients from rolling windows are vectorized column-major:
//! [beta_1(t_1), ..., beta_1(t_m), beta_2(t_1), ..., beta_2(t_m), ...]
//! to form the FPCA input matrix. This preserves the functional structure
//! within each predictor.
//!
//! **Note on overlapping windows:** When `step_size < window_size`, consecutive
//! windows share observations, inducing serial correlation in the beta(t) estimates.
//! The `effective_n_windows` field in [`ProfileChart`] provides a Bartlett-style
//! correction for the effective degrees of freedom. Specifically, for overlapping
//! windows with step_size < window_size, the effective number of independent
//! windows is reduced. The Bartlett correction n_eff = n_windows / (1 + 2|rho_1|)
//! accounts for AR(1) autocorrelation rho_1 in the windowed statistics, where
//! rho_1 is estimated from the lag-1 autocorrelation of the T-squared sequence
//! (Bartlett, 1946, section 3, pp. 31--33).
//!
//! # References
//!
//! - Bartlett, M.S. (1946). On the theoretical specification of sampling
//!   properties of autocorrelated time series. *Journal of the Royal
//!   Statistical Society B*, 8(1), 27--41, section 3, pp. 31--33.
//! - Ledolter, J. & Swersey, A.J. (2007). *Testing 1-2-3: Experimental
//!   design with applications in marketing and service operations*.
//!   Stanford University Press, Ch. 6 (profile monitoring).

use crate::error::FdarError;
use crate::function_on_scalar::{fosr, FosrResult};
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};
use crate::spm::control::{t2_control_limit, ControlLimit};
use crate::spm::stats::hotelling_t2;

/// Configuration for profile monitoring.
///
/// # Parameter guidance
///
/// - `window_size`: Must be >= p + 2 where p is the number of predictors.
///   Larger windows give more stable beta(t) estimates but reduce temporal resolution.
///   Typical: 20--50 for slowly varying processes. Recommended `window_size` is
///   5p to 10p where p is the number of predictors, to ensure stable FOSR
///   estimation within each window (the design matrix X^T X has condition number
///   roughly proportional to window_size / p).
/// - `step_size`: Controls window overlap. step_size = window_size gives no overlap
///   (independent windows); step_size = 1 gives maximum overlap (smoothest tracking
///   but highest autocorrelation). Typical: window_size/2 or window_size/4.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileMonitorConfig {
    /// FOSR smoothing parameter (default 1e-4).
    pub fosr_lambda: f64,
    /// Number of principal components for beta-function FPCA (default 3).
    /// Typically 2--5 suffices since coefficient functions are smoother than
    /// raw data. Use `select_ncomp()` on the beta eigenvalues for data-driven
    /// selection.
    pub ncomp: usize,
    /// Significance level (default 0.05).
    pub alpha: f64,
    /// Window size for rolling FOSR (default 20).
    pub window_size: usize,
    /// Step size between windows (default 1).
    pub step_size: usize,
}

impl Default for ProfileMonitorConfig {
    fn default() -> Self {
        Self {
            fosr_lambda: 1e-4,
            ncomp: 3,
            alpha: 0.05,
            window_size: 20,
            step_size: 1,
        }
    }
}

/// Phase I profile monitoring chart.
///
/// When `effective_n_windows` is much smaller than the actual number of
/// windows, the chi-squared UCL may need adjustment. A practical approach:
/// multiply the UCL by `effective_n_windows / n_windows` to approximate a
/// Bonferroni-like correction.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ProfileChart {
    /// FOSR result from the full reference data.
    pub reference_fosr: FosrResult,
    /// FPCA of the rolling beta coefficient functions.
    pub beta_fpca: FpcaResult,
    /// Eigenvalues: sv² / (n_windows - 1).
    pub eigenvalues: Vec<f64>,
    /// T-squared control limit for beta monitoring.
    pub t2_limit: ControlLimit,
    /// Lag-1 autocorrelation of the Phase I T-squared statistics from rolling windows.
    /// High values (> 0.5) indicate serial correlation from window overlap.
    ///
    /// Computed from the Phase I T-squared statistic sequence. Values |rho_1| > 0.3
    /// indicate substantial serial correlation; `effective_n_windows` will be
    /// notably reduced. The estimator uses the unbiased sample variance (n-1
    /// denominator) for both variance and covariance terms.
    pub lag1_autocorrelation: f64,
    /// Effective number of independent windows (Bartlett correction for overlap).
    /// When step_size < window_size, consecutive windows overlap and are
    /// correlated, reducing the effective degrees of freedom.
    ///
    /// Computed as n_eff = n_windows / (1 + 2|rho_1|), which is the Bartlett
    /// (1946, section 3) formula for AR(1) processes. For overlap fraction
    /// f = 1 - step_size/window_size, the lag-1 autocorrelation is approximately
    /// f, so n_eff ~ n_windows / (1 + 2f). This is conservative (underestimates
    /// n_eff) for non-AR(1) dependence structures.
    ///
    /// Use this to assess whether the chi-squared UCL is reliable. When
    /// `effective_n_windows` < 20, consider widening the control limit or
    /// using bootstrap limits instead.
    pub effective_n_windows: f64,
    /// Configuration used.
    pub config: ProfileMonitorConfig,
}

/// Result of Phase II profile monitoring.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ProfileMonitorResult {
    /// Per-window beta coefficient matrices (vectorized).
    pub betas: FdMatrix,
    /// T-squared values for each window.
    pub t2: Vec<f64>,
    /// T-squared alarm flags.
    pub t2_alarm: Vec<bool>,
    /// FPC scores for the beta functions.
    pub beta_scores: FdMatrix,
}

/// Build a Phase I profile monitoring chart.
///
/// 1. Fits FOSR on the full training data to get reference β(t)
/// 2. Rolls windows over the data, fitting FOSR per window
/// 3. Vectorizes the β(t) from each window
/// 4. Runs FPCA on the vectorized betas
/// 5. Computes T-squared control limits
///
/// When `step_size > window_size`, windows do not overlap and some observations
/// fall between windows (not monitored).
///
/// # Arguments
/// * `y_curves` - Response functional data (n × m)
/// * `predictors` - Scalar predictors (n × p)
/// * `argvals` - Grid points (length m)
/// * `config` - Profile monitoring configuration
///
/// # Errors
///
/// Returns errors from FOSR or FPCA computation.
///
/// # Example
/// ```no_run
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::profile::{profile_phase1, ProfileMonitorConfig};
/// let n = 50;
/// let m = 10;
/// let y = FdMatrix::from_column_major(
///     (0..n*m).map(|i| (i as f64 * 0.1).sin()).collect(), n, m
/// ).unwrap();
/// let pred = FdMatrix::from_column_major(
///     (0..n).map(|i| i as f64 / n as f64).collect(), n, 1
/// ).unwrap();
/// let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m-1) as f64).collect();
/// let config = ProfileMonitorConfig { window_size: 10, step_size: 5, ncomp: 2, ..ProfileMonitorConfig::default() };
/// let chart = profile_phase1(&y, &pred, &argvals, &config).unwrap();
/// assert!(chart.eigenvalues.len() >= 1);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn profile_phase1(
    y_curves: &FdMatrix,
    predictors: &FdMatrix,
    argvals: &[f64],
    config: &ProfileMonitorConfig,
) -> Result<ProfileChart, FdarError> {
    let (n, m) = y_curves.shape();
    if predictors.nrows() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "predictors",
            expected: format!("{n} rows"),
            actual: format!("{} rows", predictors.nrows()),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if config.step_size == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "step_size",
            message: "step_size must be at least 1".to_string(),
        });
    }
    if config.window_size < 3 {
        return Err(FdarError::InvalidParameter {
            parameter: "window_size",
            message: format!("window_size must be >= 3, got {}", config.window_size),
        });
    }
    if config.window_size > n {
        return Err(FdarError::InvalidParameter {
            parameter: "window_size",
            message: format!(
                "window_size ({}) exceeds data size ({n})",
                config.window_size
            ),
        });
    }

    // Early check: ensure enough windows before expensive FOSR fitting.
    let expected_n_windows = if n >= config.window_size {
        (n - config.window_size) / config.step_size + 1
    } else {
        0
    };
    if expected_n_windows < 4 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "enough data for at least 4 windows".to_string(),
            actual: format!(
                "{expected_n_windows} windows (n={n}, window_size={}, step_size={})",
                config.window_size, config.step_size
            ),
        });
    }

    // Fit reference FOSR on full data
    let reference_fosr = fosr(y_curves, predictors, config.fosr_lambda)?;

    // Rolling windows: extract per-window FOSR betas
    let beta_vecs = rolling_betas(y_curves, predictors, config)?;
    let n_windows = beta_vecs.nrows();

    if n_windows < 4 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "enough data for at least 4 windows".to_string(),
            actual: format!("{n_windows} windows"),
        });
    }

    // FPCA on vectorized betas
    let ncomp = config.ncomp.min(n_windows - 1).min(beta_vecs.ncols());
    let beta_m = beta_vecs.ncols();
    let beta_argvals: Vec<f64> = (0..beta_m)
        .map(|j| j as f64 / (beta_m - 1).max(1) as f64)
        .collect();
    let beta_fpca = fdata_to_pc_1d(&beta_vecs, ncomp, &beta_argvals)?;
    let actual_ncomp = beta_fpca.scores.ncols();

    // Eigenvalues
    let eigenvalues: Vec<f64> = beta_fpca
        .singular_values
        .iter()
        .take(actual_ncomp)
        .map(|&sv| sv * sv / (n_windows as f64 - 1.0))
        .collect();

    // Compute Phase I T² for autocorrelation diagnostic
    let phase1_t2 = hotelling_t2(&beta_fpca.scores, &eigenvalues)?;
    // Lag-1 autocorrelation using unbiased sample variance (n-1 denominator).
    // Both variance and covariance use the (n-1) denominator. The ratio
    // ρ₁ = cov/var is invariant to this choice, but (n-1) gives the unbiased
    // sample variance.
    let lag1_autocorrelation = if phase1_t2.len() > 2 {
        let n_t2 = phase1_t2.len();
        let mean_t2 = phase1_t2.iter().sum::<f64>() / n_t2 as f64;
        let var_t2: f64 = phase1_t2
            .iter()
            .map(|&v| (v - mean_t2).powi(2))
            .sum::<f64>()
            / (n_t2 - 1) as f64;
        if var_t2 > 0.0 {
            let cov1: f64 = (0..n_t2 - 1)
                .map(|i| (phase1_t2[i] - mean_t2) * (phase1_t2[i + 1] - mean_t2))
                .sum::<f64>()
                / (n_t2 - 1) as f64;
            (cov1 / var_t2).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Bartlett (1946) effective sample size: n_eff = n / (1 + 2|ρ₁|).
    // See Bartlett, M.S. (1946), Section 3.
    // The Bartlett (1946) formula n_eff = n/(1 + 2ρ₁) is derived for
    // AR(1) processes and provides a first-order correction for general
    // stationary processes. For rolling-window statistics with overlap
    // fraction f = 1 - step_size/window_size, the lag-1 autocorrelation
    // is approximately f, so n_eff ≈ n/(1 + 2f). This is conservative
    // (underestimates n_eff) for non-AR(1) dependence structures.
    let effective_n_windows = if lag1_autocorrelation.abs() > 0.01 {
        // Use absolute value: both positive (overlapping windows) and negative
        // (anti-correlated) autocorrelation reduce the effective degrees of freedom.
        let bartlett_factor = (1.0 + 2.0 * lag1_autocorrelation.abs()).max(1.0);
        (n_windows as f64 / bartlett_factor).max(2.0)
    } else {
        n_windows as f64
    };

    // Control limit
    let t2_limit = t2_control_limit(actual_ncomp, config.alpha)?;

    Ok(ProfileChart {
        reference_fosr,
        beta_fpca,
        eigenvalues,
        t2_limit,
        lag1_autocorrelation,
        effective_n_windows,
        config: config.clone(),
    })
}

/// Monitor new data against a Phase I profile chart.
///
/// 1. Fits FOSR per rolling window on new data
/// 2. Vectorizes β(t) and projects onto Phase I beta-FPCA
/// 3. Computes T-squared
///
/// # Arguments
/// * `chart` - Phase I profile chart
/// * `new_y` - New response functional data (n × m)
/// * `new_predictors` - New scalar predictors (n × p)
/// * `argvals` - Grid points (length m)
/// * `config` - Profile monitoring configuration
///
/// # Errors
///
/// Returns errors from FOSR or projection.
#[must_use = "monitoring result should not be discarded"]
pub fn profile_monitor(
    chart: &ProfileChart,
    new_y: &FdMatrix,
    new_predictors: &FdMatrix,
    _argvals: &[f64],
    config: &ProfileMonitorConfig,
) -> Result<ProfileMonitorResult, FdarError> {
    if config.step_size == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "step_size",
            message: "step_size must be at least 1".to_string(),
        });
    }
    let n = new_y.nrows();
    if new_predictors.nrows() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "new_predictors",
            expected: format!("{n} rows"),
            actual: format!("{} rows", new_predictors.nrows()),
        });
    }
    // Validate that new_y grid size matches the reference FOSR grid (m)
    let expected_m = chart.reference_fosr.beta.ncols();
    if new_y.ncols() != expected_m {
        return Err(FdarError::InvalidDimension {
            parameter: "new_y",
            expected: format!("{expected_m} columns (grid points)"),
            actual: format!("{} columns", new_y.ncols()),
        });
    }

    // Rolling windows on new data
    let beta_vecs = rolling_betas(new_y, new_predictors, config)?;

    // Project onto Phase I FPCA
    let beta_scores = chart.beta_fpca.project(&beta_vecs)?;

    // T-squared
    let t2 = hotelling_t2(&beta_scores, &chart.eigenvalues)?;

    // Alarms
    let t2_alarm: Vec<bool> = t2.iter().map(|&v| v > chart.t2_limit.ucl).collect();

    Ok(ProfileMonitorResult {
        betas: beta_vecs,
        t2,
        t2_alarm,
        beta_scores,
    })
}

/// Extract vectorized FOSR betas from rolling windows.
///
/// Each window must have sufficient rank for FOSR fitting. If a window fails
/// (e.g., due to collinear predictors), the function returns an error.
/// Ensure `window_size` is large enough relative to `p` (number of predictors).
fn rolling_betas(
    y_curves: &FdMatrix,
    predictors: &FdMatrix,
    config: &ProfileMonitorConfig,
) -> Result<FdMatrix, FdarError> {
    let n = y_curves.nrows();
    let m = y_curves.ncols();
    let p = predictors.ncols();

    if config.window_size < p + 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "window_size",
            message: format!(
                "window_size ({}) must be >= p + 2 = {} for FOSR fitting",
                config.window_size,
                p + 2
            ),
        });
    }

    let mut windows = Vec::new();
    let mut start = 0;
    while start + config.window_size <= n {
        windows.push(start);
        start += config.step_size;
    }

    if windows.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!(
                "at least {} observations for one window",
                config.window_size
            ),
            actual: format!("{n} observations"),
        });
    }

    // Beta coefficients are vectorized in row-major order:
    // β₁(t₁),...,β₁(t_m),β₂(t₁),...,β₂(t_m).
    // Each row of beta_mat represents one window's complete coefficient function.
    let beta_len = p * m;
    let n_windows = windows.len();
    let mut beta_mat = FdMatrix::zeros(n_windows, beta_len);

    for (w_idx, &win_start) in windows.iter().enumerate() {
        // Extract window data
        let mut y_window = FdMatrix::zeros(config.window_size, m);
        let mut pred_window = FdMatrix::zeros(config.window_size, p);
        for i in 0..config.window_size {
            for j in 0..m {
                y_window[(i, j)] = y_curves[(win_start + i, j)];
            }
            for j in 0..p {
                pred_window[(i, j)] = predictors[(win_start + i, j)];
            }
        }

        // Fit FOSR
        let fosr_result = fosr(&y_window, &pred_window, config.fosr_lambda)?;

        // Vectorize beta (p × m) into row of beta_mat
        // fosr_result.beta is p × m where row j = βⱼ(t)
        for j in 0..p {
            for t in 0..m {
                beta_mat[(w_idx, j * m + t)] = fosr_result.beta[(j, t)];
            }
        }
    }

    Ok(beta_mat)
}
