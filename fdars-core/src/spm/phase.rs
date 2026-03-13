//! Phase I/II framework for statistical process monitoring.
//!
//! Phase I: Builds a monitoring chart from historical in-control data by
//! splitting into tuning (for FPCA) and calibration (for control limits) sets.
//!
//! Phase II: Monitors new observations against the established chart.
//!
//! Both univariate and multivariate variants are provided.

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};

use super::control::{spe_control_limit, t2_control_limit, ControlLimit};
use super::mfpca::{mfpca, MfpcaConfig, MfpcaResult};
use super::stats::{hotelling_t2, spe_multivariate, spe_univariate};

/// Configuration for SPM chart construction.
#[derive(Debug, Clone, PartialEq)]
pub struct SpmConfig {
    /// Number of principal components (default 5).
    pub ncomp: usize,
    /// Significance level for control limits (default 0.05).
    pub alpha: f64,
    /// Fraction of data used for tuning/FPCA (default 0.5).
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
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SpmChart {
    /// FPCA result from the tuning set.
    pub fpca: FpcaResult,
    /// Eigenvalues: sv^2 / (n_tune - 1).
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
}

/// Multivariate SPM chart from Phase I.
#[derive(Debug, Clone, PartialEq)]
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
fn split_indices(n: usize, tuning_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let n_tune = ((n as f64 * tuning_fraction).round() as usize)
        .max(2)
        .min(n - 1);

    // Generate a deterministic permutation
    let mut indices: Vec<usize> = (0..n).collect();
    // Simple LCG-based shuffle
    let mut rng_state: u64 = seed;
    for i in (1..n).rev() {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let j = (rng_state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }

    let tune_indices: Vec<usize> = indices[..n_tune].to_vec();
    let cal_indices: Vec<usize> = indices[n_tune..].to_vec();
    (tune_indices, cal_indices)
}

/// Compute centered reconstruction (without adding back mean) for SPE.
///
/// Returns the centered reconstruction: scores * rotation^T (no mean added).
fn centered_reconstruct(fpca: &FpcaResult, scores: &FdMatrix, ncomp: usize) -> FdMatrix {
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
fn center_data(data: &FdMatrix, mean: &[f64]) -> FdMatrix {
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

    // FPCA on tuning set
    let ncomp = config.ncomp.min(n_tune - 1).min(m);
    let fpca = fdata_to_pc_1d(&tune_data, ncomp)?;
    let actual_ncomp = fpca.scores.ncols();

    // Eigenvalues
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
