//! Phase I/II framework for statistical process monitoring.
//!
//! Phase I: Builds a monitoring chart from historical in-control data by
//! splitting into tuning (for FPCA) and calibration (for control limits) sets.
//!
//! Phase II: Monitors new observations against the established chart.
//!
//! Both univariate and multivariate variants are provided.
//!
//! # References
//!
//! - Horváth, L. & Kokoszka, P. (2012). *Inference for Functional Data
//!   with Applications*, Chapter 13, pp. 323--352. Springer.
//! - Flores, M., Naya, S., Fernández-Casal, R. & Zaragoza, S. (2022).
//!   Constructing a control chart using functional data, §2, Algorithm 1.
//!   *Mathematics*, 8(1), 58.

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};

use super::control::{spe_control_limit, t2_control_limit, ControlLimit};
use super::mfpca::{mfpca, MfpcaConfig, MfpcaResult};
use super::stats::{hotelling_t2, spe_multivariate, spe_univariate};

/// Configuration for SPM chart construction.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpmConfig {
    /// Number of principal components to retain (default 5).
    ///
    /// Typical range: 3--10. Use [`select_ncomp()`](super::ncomp::select_ncomp)
    /// for data-driven selection. More components capture finer structure but
    /// increase dimensionality of the monitoring statistic.
    pub ncomp: usize,
    /// Significance level for control limits (default 0.05).
    pub alpha: f64,
    /// Fraction of data used for tuning/FPCA (default 0.5).
    ///
    /// The remainder forms the calibration set for control limits. Default 0.5
    /// balances FPCA estimation quality against control limit precision. With
    /// small datasets (n < 50), consider 0.6--0.7 to ensure adequate FPCA
    /// estimation.
    ///
    /// The tuning/calibration split induces a bias-variance trade-off: a larger
    /// tuning fraction yields better FPCA eigenfunction estimates but less
    /// precise control limits. The optimal split depends on the eigenvalue
    /// decay rate — fast decay (smooth processes) favors allocating more data
    /// to calibration, while slow decay (rough processes) favors more tuning
    /// data for stable FPCA estimation.
    pub tuning_fraction: f64,
    /// Random seed for data splitting (default 42).
    pub seed: u64,
}

impl Default for SpmConfig {
    fn default() -> Self {
        Self {
            ncomp: 5,
            alpha: 0.05,
            tuning_fraction: 0.5,
            seed: 42,
        }
    }
}

/// Univariate SPM chart from Phase I.
///
/// The chart assumes approximate multivariate normality in the score space.
/// For non-Gaussian functional data, the chi-squared control limits are
/// approximate. Use `t2_limit_robust()` with bootstrap method for
/// distribution-free limits.
///
/// For finite calibration samples, the exact Hotelling T^2 distribution is
/// `(n_cal * ncomp / (n_cal - ncomp)) * F(ncomp, n_cal - ncomp)`. The
/// chi-squared limit `chi2(ncomp)` is asymptotically exact as `n_cal -> inf`,
/// but can be anti-conservative for small calibration sets (n_cal < 10 *
/// ncomp). When the calibration set is small, prefer bootstrap-based limits.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct SpmChart {
    /// FPCA result from the tuning set.
    pub fpca: FpcaResult,
    /// Eigenvalues lambda_l = s_l^2 / (n_tune - 1), where s_l are singular
    /// values from the SVD of the centered data matrix X = U Sigma V^T. This
    /// gives the sample covariance eigenvalues since Cov = X^T X / (n-1) has
    /// eigenvalues s_l^2 / (n-1).
    ///
    /// The actual number of components may be fewer than `config.ncomp` if
    /// limited by sample size or grid resolution.
    pub eigenvalues: Vec<f64>,
    /// T-squared values for the calibration set.
    pub t2_phase1: Vec<f64>,
    /// SPE values for the calibration set.
    pub spe_phase1: Vec<f64>,
    /// T-squared control limit.
    pub t2_limit: ControlLimit,
    /// SPE control limit.
    pub spe_limit: ControlLimit,
    /// Configuration used to build the chart.
    pub config: SpmConfig,
    /// Whether the sample size meets the recommended minimum (10 × ncomp).
    /// False indicates results may be unstable.
    pub sample_size_adequate: bool,
}

/// Multivariate SPM chart from Phase I.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct MfSpmChart {
    /// MFPCA result from the tuning set.
    pub mfpca: MfpcaResult,
    /// T-squared values for the calibration set.
    pub t2_phase1: Vec<f64>,
    /// SPE values for the calibration set.
    pub spe_phase1: Vec<f64>,
    /// T-squared control limit.
    pub t2_limit: ControlLimit,
    /// SPE control limit.
    pub spe_limit: ControlLimit,
    /// Configuration used to build the chart.
    pub config: SpmConfig,
}

/// Result of Phase II monitoring.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct SpmMonitorResult {
    /// T-squared values for new observations.
    pub t2: Vec<f64>,
    /// SPE values for new observations.
    pub spe: Vec<f64>,
    /// T-squared alarm flags.
    pub t2_alarm: Vec<bool>,
    /// SPE alarm flags.
    pub spe_alarm: Vec<bool>,
    /// Score matrix for new observations.
    pub scores: FdMatrix,
}

/// Split indices into tuning and calibration sets.
///
/// Uses a deterministic Fisher-Yates shuffle with PCG-XSH-RR output function
/// (O'Neill, 2014, §4.1, p. 14) for high-quality uniform sampling. PCG-XSH-RR
/// has period 2^64 and passes the full TestU01 BigCrush battery. The same seed
/// always produces the same split, ensuring reproducibility.
pub(super) fn split_indices(n: usize, tuning_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let n_tune = ((n as f64 * tuning_fraction).round() as usize)
        .max(2)
        .min(n - 1);

    // Generate a deterministic permutation using PCG-XSH-RR
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng_state: u64 = seed;
    for i in (1..n).rev() {
        let j = pcg_next(&mut rng_state) as usize % (i + 1);
        indices.swap(i, j);
    }

    let tune_indices: Vec<usize> = indices[..n_tune].to_vec();
    let cal_indices: Vec<usize> = indices[n_tune..].to_vec();
    (tune_indices, cal_indices)
}

/// PCG-XSH-RR 64→32 output function (O'Neill, 2014).
///
/// PCG-XSH-RR (O'Neill, 2014) produces uniformly distributed 32-bit outputs
/// from a 64-bit LCG state. The XSH-RR output function (xor-shift, random
/// rotation) provides excellent statistical quality (passes TestU01 BigCrush).
/// Used here for Fisher-Yates shuffle in `split_indices`.
fn pcg_next(state: &mut u64) -> u32 {
    let old = *state;
    *state = old.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
    let rot = (old >> 59) as u32;
    xorshifted.rotate_right(rot)
}

/// Compute centered reconstruction (without adding back mean) for SPE.
///
/// Returns the centered reconstruction: scores * rotation^T (no mean added).
pub(super) fn centered_reconstruct(fpca: &FpcaResult, scores: &FdMatrix, ncomp: usize) -> FdMatrix {
    let n = scores.nrows();
    let m = fpca.mean.len();
    let ncomp = ncomp.min(fpca.rotation.ncols()).min(scores.ncols());

    let mut recon = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            let mut val = 0.0;
            for k in 0..ncomp {
                val += scores[(i, k)] * fpca.rotation[(j, k)];
            }
            recon[(i, j)] = val;
        }
    }
    recon
}

/// Compute centered data for new observations (data - mean).
pub(super) fn center_data(data: &FdMatrix, mean: &[f64]) -> FdMatrix {
    let (n, m) = data.shape();
    let mut centered = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            centered[(i, j)] = data[(i, j)] - mean[j];
        }
    }
    centered
}

/// Build a univariate SPM chart from Phase I data.
///
/// # Arguments
/// * `data` - In-control functional data (n x m)
/// * `argvals` - Grid points (length m)
/// * `config` - SPM configuration
///
/// # Errors
///
/// Returns errors from FPCA, T-squared computation, or control limit estimation.
///
/// # Assumptions
///
/// - The in-control data should be approximately normally distributed in the
///   score space. Departures from normality may inflate false alarm rates.
/// - Recommended minimum sample size: at least 10 × ncomp observations to
///   ensure stable FPCA estimation (Horváth & Kokoszka, 2012).
/// - The tuning/calibration split means the effective sample for each step
///   is smaller than `n`. Ensure each subset has at least 2 × ncomp rows.
///
/// # Example
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::phase::{spm_phase1, SpmConfig};
/// let data = FdMatrix::from_column_major(
///     (0..200).map(|i| (i as f64 * 0.1).sin()).collect(), 20, 10
/// ).unwrap();
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let config = SpmConfig { ncomp: 2, ..SpmConfig::default() };
/// let chart = spm_phase1(&data, &argvals, &config).unwrap();
/// assert!(chart.eigenvalues.len() <= 2);
/// assert!(chart.t2_limit.ucl > 0.0);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn spm_phase1(
    data: &FdMatrix,
    argvals: &[f64],
    config: &SpmConfig,
) -> Result<SpmChart, FdarError> {
    let (n, m) = data.shape();
    if n < 4 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 4 observations for tuning/calibration split".to_string(),
            actual: format!("{n} observations"),
        });
    }
    let sample_size_adequate = n >= 10 * config.ncomp;
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }

    // Split into tuning and calibration
    let (tune_idx, cal_idx) = split_indices(n, config.tuning_fraction, config.seed);

    let tune_data = crate::cv::subset_rows(data, &tune_idx);
    let cal_data = crate::cv::subset_rows(data, &cal_idx);
    let n_tune = tune_data.nrows();
    if n_tune < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "tuning set with at least 3 observations".to_string(),
            actual: format!(
                "{n_tune} observations in tuning set (increase data size or tuning_fraction)"
            ),
        });
    }

    // FPCA on tuning set.
    // Clamp ncomp to at most (n_tune - 1) and m to avoid rank-deficient SVD.
    // The actual number of retained components may therefore be fewer than
    // config.ncomp; this is reflected in the chart's eigenvalues length.
    let ncomp = config.ncomp.min(n_tune - 1).min(m);
    let fpca = fdata_to_pc_1d(&tune_data, ncomp, argvals)?;
    let actual_ncomp = fpca.scores.ncols();

    // Eigenvalues are computed as λ_l = s_l² / (n-1) where s_l is the l-th
    // singular value from SVD of the centered data. This gives the sample
    // variance explained by each PC, consistent with the covariance-based PCA
    // formulation (cov = X'X / (n-1), whose eigenvalues are s_l² / (n-1)).
    let eigenvalues: Vec<f64> = fpca
        .singular_values
        .iter()
        .take(actual_ncomp)
        .map(|&sv| sv * sv / (n_tune as f64 - 1.0))
        .collect();

    // Project calibration set
    let cal_scores = fpca.project(&cal_data)?;

    // T-squared on calibration
    let t2_phase1 = hotelling_t2(&cal_scores, &eigenvalues)?;

    // SPE on calibration: need centered and reconstructed
    let cal_centered = center_data(&cal_data, &fpca.mean);
    let cal_recon_centered = centered_reconstruct(&fpca, &cal_scores, actual_ncomp);
    let spe_phase1 = spe_univariate(&cal_centered, &cal_recon_centered, argvals)?;

    // Control limits
    let t2_limit = t2_control_limit(actual_ncomp, config.alpha)?;
    let spe_limit = spe_control_limit(&spe_phase1, config.alpha)?;

    Ok(SpmChart {
        fpca,
        eigenvalues,
        t2_phase1,
        spe_phase1,
        t2_limit,
        spe_limit,
        config: config.clone(),
        sample_size_adequate,
    })
}

/// Monitor new univariate functional data against an established SPM chart.
///
/// # Arguments
/// * `chart` - Phase I SPM chart
/// * `new_data` - New functional observations (n_new x m)
/// * `argvals` - Grid points (length m)
///
/// # Errors
///
/// Returns errors from projection or statistic computation.
///
/// # Example
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::phase::{spm_phase1, spm_monitor, SpmConfig};
/// let data = FdMatrix::from_column_major(
///     (0..200).map(|i| (i as f64 * 0.1).sin()).collect(), 20, 10
/// ).unwrap();
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let config = SpmConfig { ncomp: 2, ..SpmConfig::default() };
/// let chart = spm_phase1(&data, &argvals, &config).unwrap();
/// let new_data = FdMatrix::from_column_major(
///     (0..50).map(|i| (i as f64 * 0.1).sin()).collect(), 5, 10
/// ).unwrap();
/// let result = spm_monitor(&chart, &new_data, &argvals).unwrap();
/// assert_eq!(result.t2.len(), 5);
/// ```
#[must_use = "monitoring result should not be discarded"]
pub fn spm_monitor(
    chart: &SpmChart,
    new_data: &FdMatrix,
    argvals: &[f64],
) -> Result<SpmMonitorResult, FdarError> {
    let m = chart.fpca.mean.len();
    if new_data.ncols() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "new_data",
            expected: format!("{m} columns"),
            actual: format!("{} columns", new_data.ncols()),
        });
    }

    let ncomp = chart.eigenvalues.len();

    // Project new data
    let scores = chart.fpca.project(new_data)?;

    // T-squared
    let t2 = hotelling_t2(&scores, &chart.eigenvalues)?;

    // SPE
    let centered = center_data(new_data, &chart.fpca.mean);
    let recon_centered = centered_reconstruct(&chart.fpca, &scores, ncomp);
    let spe = spe_univariate(&centered, &recon_centered, argvals)?;

    // Alarms
    let t2_alarm: Vec<bool> = t2.iter().map(|&v| v > chart.t2_limit.ucl).collect();
    let spe_alarm: Vec<bool> = spe.iter().map(|&v| v > chart.spe_limit.ucl).collect();

    Ok(SpmMonitorResult {
        t2,
        spe,
        t2_alarm,
        spe_alarm,
        scores,
    })
}

/// Phase II monitoring from decomposed fields (no [`SpmChart`] struct needed).
///
/// This enables monitoring from a serialized/restored `SpmChartLayer`
/// without reconstructing the full `SpmChart` struct.  The caller provides
/// the raw FPCA components and control limits directly.
///
/// # Arguments
/// * `fpca_mean` -- Mean function from FPCA (length m)
/// * `fpca_rotation` -- Eigenfunctions / rotation matrix (m x ncomp)
/// * `fpca_weights` -- Integration weights (length m)
/// * `eigenvalues` -- Eigenvalues (length ncomp)
/// * `t2_ucl` -- Upper control limit for T²
/// * `spe_ucl` -- Upper control limit for SPE
/// * `new_data` -- New functional observations (n_new x m)
/// * `argvals` -- Grid points (length m)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if any dimension is inconsistent.
/// Returns [`FdarError::InvalidParameter`] if any eigenvalue is non-positive.
///
/// # Example
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::phase::{spm_phase1, spm_monitor, spm_monitor_from_fields, SpmConfig};
///
/// let data = FdMatrix::from_column_major(
///     (0..200).map(|i| (i as f64 * 0.1).sin()).collect(), 20, 10
/// ).unwrap();
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let config = SpmConfig { ncomp: 2, ..SpmConfig::default() };
/// let chart = spm_phase1(&data, &argvals, &config).unwrap();
///
/// let new_data = FdMatrix::from_column_major(
///     (0..50).map(|i| (i as f64 * 0.1).sin()).collect(), 5, 10
/// ).unwrap();
///
/// // Equivalent to spm_monitor(&chart, &new_data, &argvals)
/// let result = spm_monitor_from_fields(
///     &chart.fpca.mean,
///     &chart.fpca.rotation,
///     &chart.fpca.weights,
///     &chart.eigenvalues,
///     chart.t2_limit.ucl,
///     chart.spe_limit.ucl,
///     &new_data,
///     &argvals,
/// ).unwrap();
/// assert_eq!(result.t2.len(), 5);
/// ```
#[must_use = "monitoring result should not be discarded"]
pub fn spm_monitor_from_fields(
    fpca_mean: &[f64],
    fpca_rotation: &FdMatrix,
    fpca_weights: &[f64],
    eigenvalues: &[f64],
    t2_ucl: f64,
    spe_ucl: f64,
    new_data: &FdMatrix,
    argvals: &[f64],
) -> Result<SpmMonitorResult, FdarError> {
    let m = fpca_mean.len();
    let ncomp = eigenvalues.len();

    if new_data.ncols() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "new_data",
            expected: format!("{m} columns"),
            actual: format!("{} columns", new_data.ncols()),
        });
    }
    if fpca_rotation.nrows() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "fpca_rotation",
            expected: format!("{m} rows (matching fpca_mean)"),
            actual: format!("{} rows", fpca_rotation.nrows()),
        });
    }
    if fpca_rotation.ncols() != ncomp {
        return Err(FdarError::InvalidDimension {
            parameter: "fpca_rotation",
            expected: format!("{ncomp} columns (matching eigenvalues)"),
            actual: format!("{} columns", fpca_rotation.ncols()),
        });
    }
    if fpca_weights.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "fpca_weights",
            expected: format!("{m} (matching fpca_mean)"),
            actual: format!("{}", fpca_weights.len()),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} (matching fpca_mean)"),
            actual: format!("{}", argvals.len()),
        });
    }

    let n_new = new_data.nrows();

    // 1. Project new data onto FPC scores (center + weight + rotate)
    let mut scores = FdMatrix::zeros(n_new, ncomp);
    for i in 0..n_new {
        for k in 0..ncomp {
            let mut sum = 0.0;
            for j in 0..m {
                sum += (new_data[(i, j)] - fpca_mean[j]) * fpca_rotation[(j, k)] * fpca_weights[j];
            }
            scores[(i, k)] = sum;
        }
    }

    // 2. Compute T² from scores / eigenvalues
    let t2 = hotelling_t2(&scores, eigenvalues)?;

    // 3. Reconstruct (centered) and compute SPE
    //    centered = new_data - mean
    let centered = center_data(new_data, fpca_mean);
    //    centered_reconstruction = scores * rotation^T
    let mut recon_centered = FdMatrix::zeros(n_new, m);
    for i in 0..n_new {
        for j in 0..m {
            let mut val = 0.0;
            for k in 0..ncomp {
                val += scores[(i, k)] * fpca_rotation[(j, k)];
            }
            recon_centered[(i, j)] = val;
        }
    }
    let spe = spe_univariate(&centered, &recon_centered, argvals)?;

    // 4. Flag alarms
    let t2_alarm: Vec<bool> = t2.iter().map(|&v| v > t2_ucl).collect();
    let spe_alarm: Vec<bool> = spe.iter().map(|&v| v > spe_ucl).collect();

    Ok(SpmMonitorResult {
        t2,
        spe,
        t2_alarm,
        spe_alarm,
        scores,
    })
}

/// Build a multivariate SPM chart from Phase I data.
///
/// # Arguments
/// * `variables` - Slice of in-control functional matrices (each n x m_p)
/// * `argvals_list` - Per-variable grid points
/// * `config` - SPM configuration
///
/// # Errors
///
/// Returns errors from MFPCA, T-squared computation, or control limit estimation.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn mf_spm_phase1(
    variables: &[&FdMatrix],
    argvals_list: &[&[f64]],
    config: &SpmConfig,
) -> Result<MfSpmChart, FdarError> {
    if variables.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "variables",
            expected: "at least 1 variable".to_string(),
            actual: "0 variables".to_string(),
        });
    }
    if variables.len() != argvals_list.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals_list",
            expected: format!("{} (matching variables)", variables.len()),
            actual: format!("{}", argvals_list.len()),
        });
    }

    let n = variables[0].nrows();
    if n < 4 {
        return Err(FdarError::InvalidDimension {
            parameter: "variables",
            expected: "at least 4 observations".to_string(),
            actual: format!("{n} observations"),
        });
    }

    // Validate argvals lengths
    for (p, (var, argvals)) in variables.iter().zip(argvals_list.iter()).enumerate() {
        if var.ncols() != argvals.len() {
            return Err(FdarError::InvalidDimension {
                parameter: "argvals_list",
                expected: format!("{} for variable {p}", var.ncols()),
                actual: format!("{}", argvals.len()),
            });
        }
    }

    // Split
    let (tune_idx, cal_idx) = split_indices(n, config.tuning_fraction, config.seed);

    let tune_vars: Vec<FdMatrix> = variables
        .iter()
        .map(|v| crate::cv::subset_rows(v, &tune_idx))
        .collect();
    let cal_vars: Vec<FdMatrix> = variables
        .iter()
        .map(|v| crate::cv::subset_rows(v, &cal_idx))
        .collect();

    let tune_refs: Vec<&FdMatrix> = tune_vars.iter().collect();

    // MFPCA on tuning set
    let mfpca_config = MfpcaConfig {
        ncomp: config.ncomp,
        weighted: true,
    };
    let mfpca_result = mfpca(&tune_refs, &mfpca_config)?;
    let actual_ncomp = mfpca_result.eigenvalues.len();

    // Project calibration set
    let cal_refs: Vec<&FdMatrix> = cal_vars.iter().collect();
    let cal_scores = mfpca_result.project(&cal_refs)?;

    // T-squared
    let t2_phase1 = hotelling_t2(&cal_scores, &mfpca_result.eigenvalues)?;

    // SPE on calibration: need standardized centered and reconstructed
    let cal_recon = mfpca_result.reconstruct(&cal_scores, actual_ncomp)?;

    // For SPE, we need standardized centered data and standardized reconstruction
    let n_cal = cal_vars[0].nrows();
    let mut std_vars: Vec<FdMatrix> = Vec::with_capacity(variables.len());
    let mut std_recon: Vec<FdMatrix> = Vec::with_capacity(variables.len());

    for (p, cal_var) in cal_vars.iter().enumerate() {
        let m_p = cal_var.ncols();
        let scale = if mfpca_result.scales[p] > 1e-15 {
            mfpca_result.scales[p]
        } else {
            1.0
        };

        let mut std_mat = FdMatrix::zeros(n_cal, m_p);
        let mut recon_mat = FdMatrix::zeros(n_cal, m_p);
        for i in 0..n_cal {
            for j in 0..m_p {
                std_mat[(i, j)] = (cal_var[(i, j)] - mfpca_result.means[p][j]) / scale;
                recon_mat[(i, j)] = (cal_recon[p][(i, j)] - mfpca_result.means[p][j]) / scale;
            }
        }
        std_vars.push(std_mat);
        std_recon.push(recon_mat);
    }

    let std_refs: Vec<&FdMatrix> = std_vars.iter().collect();
    let recon_refs: Vec<&FdMatrix> = std_recon.iter().collect();
    let spe_phase1 = spe_multivariate(&std_refs, &recon_refs, argvals_list)?;

    // Control limits
    let t2_limit = t2_control_limit(actual_ncomp, config.alpha)?;
    let spe_limit = spe_control_limit(&spe_phase1, config.alpha)?;

    Ok(MfSpmChart {
        mfpca: mfpca_result,
        t2_phase1,
        spe_phase1,
        t2_limit,
        spe_limit,
        config: config.clone(),
    })
}

/// Monitor new multivariate functional data against an established chart.
///
/// # Arguments
/// * `chart` - Phase I multivariate SPM chart
/// * `new_variables` - Per-variable new data (each n_new x m_p)
/// * `argvals_list` - Per-variable grid points
///
/// # Errors
///
/// Returns errors from projection or statistic computation.
#[must_use = "monitoring result should not be discarded"]
pub fn mf_spm_monitor(
    chart: &MfSpmChart,
    new_variables: &[&FdMatrix],
    argvals_list: &[&[f64]],
) -> Result<SpmMonitorResult, FdarError> {
    let n_vars = chart.mfpca.means.len();
    if new_variables.len() != n_vars {
        return Err(FdarError::InvalidDimension {
            parameter: "new_variables",
            expected: format!("{n_vars} variables"),
            actual: format!("{} variables", new_variables.len()),
        });
    }

    let actual_ncomp = chart.mfpca.eigenvalues.len();

    // Project
    let scores = chart.mfpca.project(new_variables)?;

    // T-squared
    let t2 = hotelling_t2(&scores, &chart.mfpca.eigenvalues)?;

    // SPE
    let recon = chart.mfpca.reconstruct(&scores, actual_ncomp)?;

    let n_new = new_variables[0].nrows();
    let mut std_vars: Vec<FdMatrix> = Vec::with_capacity(n_vars);
    let mut std_recon: Vec<FdMatrix> = Vec::with_capacity(n_vars);

    for (p, new_var) in new_variables.iter().enumerate() {
        let m_p = new_var.ncols();
        let scale = if chart.mfpca.scales[p] > 1e-15 {
            chart.mfpca.scales[p]
        } else {
            1.0
        };

        let mut std_mat = FdMatrix::zeros(n_new, m_p);
        let mut recon_mat = FdMatrix::zeros(n_new, m_p);
        for i in 0..n_new {
            for j in 0..m_p {
                std_mat[(i, j)] = (new_var[(i, j)] - chart.mfpca.means[p][j]) / scale;
                recon_mat[(i, j)] = (recon[p][(i, j)] - chart.mfpca.means[p][j]) / scale;
            }
        }
        std_vars.push(std_mat);
        std_recon.push(recon_mat);
    }

    let std_refs: Vec<&FdMatrix> = std_vars.iter().collect();
    let recon_refs: Vec<&FdMatrix> = std_recon.iter().collect();
    let spe = spe_multivariate(&std_refs, &recon_refs, argvals_list)?;

    // Alarms
    let t2_alarm: Vec<bool> = t2.iter().map(|&v| v > chart.t2_limit.ucl).collect();
    let spe_alarm: Vec<bool> = spe.iter().map(|&v| v > chart.spe_limit.ucl).collect();

    Ok(SpmMonitorResult {
        t2,
        spe,
        t2_alarm,
        spe_alarm,
        scores,
    })
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;
    use crate::simulation::{sim_fundata, EFunType, EValType};

    #[test]
    fn spm_chart_roundtrip_serde() {
        let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
        let data = sim_fundata(
            40,
            &t,
            5,
            EFunType::Fourier,
            EValType::Exponential,
            Some(42),
        );
        let config = SpmConfig {
            ncomp: 3,
            alpha: 0.05,
            ..Default::default()
        };
        let chart = spm_phase1(&data, &t, &config).unwrap();

        let json = serde_json::to_string(&chart).unwrap();
        let restored: SpmChart = serde_json::from_str(&json).unwrap();

        // Compare with tolerance for JSON floating-point roundtrip
        for (a, b) in chart.t2_phase1.iter().zip(&restored.t2_phase1) {
            assert!((a - b).abs() < 1e-12, "t2_phase1 mismatch: {a} vs {b}");
        }
        assert_eq!(chart.t2_limit.ucl, restored.t2_limit.ucl);
        assert_eq!(chart.spe_limit.ucl, restored.spe_limit.ucl);
        assert_eq!(chart.config, restored.config);
        assert_eq!(chart.eigenvalues.len(), restored.eigenvalues.len());

        // Monitor with the restored chart — should produce nearly identical results
        // (tiny floating-point rounding from JSON roundtrip is expected)
        let new_data = sim_fundata(
            10,
            &t,
            5,
            EFunType::Fourier,
            EValType::Exponential,
            Some(99),
        );
        let r1 = spm_monitor(&chart, &new_data, &t).unwrap();
        let r2 = spm_monitor(&restored, &new_data, &t).unwrap();
        for (a, b) in r1.t2.iter().zip(&r2.t2) {
            assert!((a - b).abs() < 1e-10, "t2 mismatch: {a} vs {b}");
        }
        assert_eq!(r1.t2_alarm, r2.t2_alarm);
        assert_eq!(r1.spe_alarm, r2.spe_alarm);
    }
}
