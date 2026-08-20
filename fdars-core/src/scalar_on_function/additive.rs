//! Nonparametric additive scalar-on-function regression.
//!
//! Implements three additive estimators for the model
//! `E[Y | X] = μ + Σ_k f_k(ξ_k)` and its functional-distance variant:
//!
//! - [`fam`] — Functional Additive Model (Müller & Yao 2008): one-pass NW over
//!   FPC scores (no backfitting loop needed because FPC scores are uncorrelated).
//! - [`fregre_gkam`] — Generalized Kernel Additive Model: iterative backfitting
//!   over Nadaraya-Watson smoothers on functional L2 distances.
//! - [`fregre_gsam`] — Generalized Spectral Additive Model: FPC-score basis
//!   with additive NW smoothing; numerically equivalent to FAM under the
//!   Gaussian identity link.
//!
//! # R Baseline Divergences
//!
//! - **FAM:** R's `fdapace::FAM` uses PACE for FPC estimation; fdars uses
//!   `fdata_to_pc_1d` (nalgebra SVD with Simpson's weights). R selects
//!   per-component bandwidths by GCV; fdars does the same via `optim_bandwidth`.
//!   No backfitting loop is used in either implementation because FPC
//!   uncorrelatedness (Müller & Yao 2008) makes one pass equivalent to
//!   infinite-iteration backfitting.
//! - **GKAM:** R's `fregre.gkam` constructs explicit n×n hat matrices H_k and
//!   solves the composite H_Q = H_1 + … + H_q system. fdars implements the
//!   equivalent iterative update by applying NW weights directly (O(n) per
//!   prediction point, O(n²) per covariate per iteration), avoiding the full
//!   n×n hat-matrix materialisation. Only the Gaussian identity link is
//!   supported; logit/log links require IRLS wrapping (documented gap).
//! - **GSAM:** R's `fregre.gsam` delegates to `mgcv::gam` penalised splines.
//!   fdars uses Nadaraya-Watson smoothing on FPC score columns (same model
//!   class, different smoother). For the Gaussian identity case the two
//!   implementations are numerically equivalent in the limit of small bandwidth
//!   / large n. Non-Gaussian links are a documented known gap.

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};
use crate::smoothing::{nadaraya_watson, optim_bandwidth, CvCriterion};
use super::nonparametric::{compute_pairwise_distances, gaussian_kernel, select_bandwidth_loo};

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

/// Configuration for the Functional Additive Model ([`fam`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FamConfig {
    /// Number of FPC components to use. 0 = auto-select via GCV (default: 0).
    pub ncomp: usize,
    /// Per-component NW bandwidth. 0.0 = auto-select per component via GCV (default: 0.0).
    pub bandwidth: f64,
    /// Kernel type: "gaussian" | "epanechnikov" | "tricube" (default: "gaussian").
    pub kernel: String,
    /// Number of bandwidth-grid points for `optim_bandwidth` (default: 20).
    pub n_grid_bandwidth: usize,
}

impl Default for FamConfig {
    fn default() -> Self {
        Self {
            ncomp: 0,
            bandwidth: 0.0,
            kernel: "gaussian".to_string(),
            n_grid_bandwidth: 20,
        }
    }
}

/// Configuration for the Generalized Kernel Additive Model ([`fregre_gkam`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GkamConfig {
    /// Per-covariate bandwidth. 0.0 = auto via LOO-CV (default: 0.0).
    pub bandwidth: f64,
    /// Kernel type (default: "gaussian").
    pub kernel: String,
    /// Maximum backfitting iterations (default: 50).
    pub max_iter: usize,
    /// Convergence threshold on max component-delta (default: 1e-6).
    pub epsilon: f64,
}

impl Default for GkamConfig {
    fn default() -> Self {
        Self {
            bandwidth: 0.0,
            kernel: "gaussian".to_string(),
            max_iter: 50,
            epsilon: 1e-6,
        }
    }
}

/// Configuration for the Generalized Spectral Additive Model ([`fregre_gsam`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GsamConfig {
    /// Number of FPC components. 0 = auto-select via GCV (default: 0).
    pub ncomp: usize,
    /// Per-component bandwidth. 0.0 = auto per component (default: 0.0).
    pub bandwidth: f64,
    /// Kernel type (default: "gaussian").
    pub kernel: String,
    /// Bandwidth-grid size for `optim_bandwidth` (default: 20).
    pub n_grid_bandwidth: usize,
}

impl Default for GsamConfig {
    fn default() -> Self {
        Self {
            ncomp: 0,
            bandwidth: 0.0,
            kernel: "gaussian".to_string(),
            n_grid_bandwidth: 20,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of [`fam`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FamResult {
    /// Fitted values ŷ (length n).
    pub fitted_values: Vec<f64>,
    /// Residuals y − ŷ (length n).
    pub residuals: Vec<f64>,
    /// Component fits f_k(ξ_k) for each observation, outer index = component (ncomp × n).
    pub component_fits: Vec<Vec<f64>>,
    /// Mean response μ_y (intercept of the additive model).
    pub intercept: f64,
    /// Per-component optimal bandwidth (length ncomp).
    pub bandwidths: Vec<f64>,
    /// Number of FPC components used.
    pub ncomp: usize,
    /// R² statistic.
    pub r_squared: f64,
    /// Embedded FPCA result for projecting new data.
    pub fpca: FpcaResult,
}

/// Result of [`fregre_gkam`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GkamResult {
    /// Fitted values ŷ (length n).
    pub fitted_values: Vec<f64>,
    /// Residuals y − ŷ (length n).
    pub residuals: Vec<f64>,
    /// Component fits f_k per predictor (q × n), outer index = predictor.
    pub component_fits: Vec<Vec<f64>>,
    /// Mean response intercept.
    pub intercept: f64,
    /// Per-predictor bandwidth (length q).
    pub bandwidths: Vec<f64>,
    /// Number of backfitting iterations performed.
    pub iterations: usize,
    /// Whether the backfitting loop converged within `max_iter`.
    pub converged: bool,
    /// R² statistic.
    pub r_squared: f64,
}

/// Result of [`fregre_gsam`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GsamResult {
    /// Fitted values ŷ (length n).
    pub fitted_values: Vec<f64>,
    /// Residuals y − ŷ (length n).
    pub residuals: Vec<f64>,
    /// Component fits f_j(ξ_j) per FPC component (ncomp × n).
    pub component_fits: Vec<Vec<f64>>,
    /// Mean response intercept.
    pub intercept: f64,
    /// Per-component bandwidth (length ncomp).
    pub bandwidths: Vec<f64>,
    /// Number of FPC components used.
    pub ncomp: usize,
    /// R² statistic.
    pub r_squared: f64,
    /// Embedded FPCA result for projecting new data.
    pub fpca: FpcaResult,
}

// ---------------------------------------------------------------------------
// Private shared helpers
// ---------------------------------------------------------------------------

/// Resolve ncomp: auto-select by single-component GCV if 0, else clamp to min(n,m).
/// Returns `Err(InvalidParameter)` if the explicitly-requested ncomp exceeds min(n,m).
fn resolve_ncomp_additive(
    ncomp: usize,
    n: usize,
    m: usize,
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    kernel: &str,
    n_grid: usize,
) -> Result<usize, FdarError> {
    let max_ncomp = n.min(m);
    if ncomp == 0 {
        // Auto-select: fit 1-component FAM for each k and pick the k giving lowest GCV.
        // Simple heuristic: start from k=1, find the largest k where adding a component
        // improves GCV. Cap at min(n, m, 10) for speed.
        let cap = max_ncomp.min(10).max(1);
        // Use cross-validation over ncomp: try k = 1..cap and pick best via GCV proxy.
        // We evaluate FPCA scores for cap components and pick k using GCV on a 1D smooth.
        let fpca_full = fdata_to_pc_1d(data, cap, argvals)?;
        let mut best_ncomp = 1usize;
        let mut best_gcv = f64::INFINITY;
        for k in 1..=cap {
            let xi_k: Vec<f64> = (0..n).map(|i| fpca_full.scores[(i, k - 1)]).collect();
            let gcv = optim_bandwidth(&xi_k, y, None, CvCriterion::Gcv, kernel, n_grid).value;
            if gcv < best_gcv {
                best_gcv = gcv;
                best_ncomp = k;
            }
        }
        Ok(best_ncomp)
    } else if ncomp > max_ncomp {
        Err(FdarError::InvalidParameter {
            parameter: "config.ncomp",
            message: format!(
                "ncomp ({ncomp}) exceeds min(n, m) = {max_ncomp}; reduce ncomp or provide more data"
            ),
        })
    } else {
        Ok(ncomp)
    }
}

/// Core additive-smooth forward pass over FPC scores (shared by fam and fregre_gsam).
///
/// Fits `f_k(ξ_k)` for k = 0..ncomp via one sequential pass of NW smoothers on partial
/// residuals. Because FPC scores are uncorrelated (Müller & Yao 2008), this single pass
/// achieves the same result as infinite-iteration backfitting.
///
/// Returns `(component_fits, bandwidths, intercept, fitted_values, residuals, r_squared)`.
#[allow(clippy::too_many_arguments)]
fn fpc_additive_smooth(
    fpca: &FpcaResult,
    y: &[f64],
    n: usize,
    ncomp: usize,
    bandwidth: f64,
    kernel: &str,
    n_grid: usize,
    scalar_covariates: Option<&FdMatrix>,
) -> Result<(Vec<Vec<f64>>, Vec<f64>, f64, Vec<f64>, Vec<f64>, f64), FdarError> {
    let mu_y = y.iter().sum::<f64>() / n as f64;

    // Count total components including scalar covariates
    let p_scalar = scalar_covariates.map_or(0, FdMatrix::ncols);
    let total_comp = ncomp + p_scalar;

    // Collect all score columns: FPC scores first, then scalar covariates
    let mut all_scores: Vec<Vec<f64>> = Vec::with_capacity(total_comp);
    for k in 0..ncomp {
        all_scores.push((0..n).map(|i| fpca.scores[(i, k)]).collect());
    }
    if let Some(sc) = scalar_covariates {
        for j in 0..p_scalar {
            all_scores.push((0..n).map(|i| sc[(i, j)]).collect());
        }
    }

    // One forward pass: for each component, build partial residual and fit NW
    let mut component_fits: Vec<Vec<f64>> = vec![vec![0.0; n]; total_comp];
    let mut bandwidths = vec![0.0_f64; total_comp];

    for k in 0..total_comp {
        // Partial residual = y - mu_y - sum_{j != k} f_j
        let partial: Vec<f64> = (0..n)
            .map(|i| {
                let others: f64 = (0..total_comp)
                    .filter(|&j| j != k)
                    .map(|j| component_fits[j][i])
                    .sum();
                y[i] - mu_y - others
            })
            .collect();

        let xi_k = &all_scores[k];
        let h = if bandwidth > 0.0 {
            bandwidth
        } else {
            optim_bandwidth(xi_k, &partial, None, CvCriterion::Gcv, kernel, n_grid).h_opt
        };
        bandwidths[k] = h;

        // nadaraya_watson returns Err only if bandwidth <= 0 or slices are empty; h > 0 always here.
        component_fits[k] = nadaraya_watson(xi_k, &partial, xi_k, h, kernel)?;
    }

    // Assemble fitted values and residuals
    let fitted_values: Vec<f64> = (0..n)
        .map(|i| mu_y + (0..total_comp).map(|k| component_fits[k][i]).sum::<f64>())
        .collect();
    let residuals: Vec<f64> = y.iter().zip(&fitted_values).map(|(&yi, &yh)| yi - yh).collect();

    // R² via shared helper (p = total_comp for df counting)
    let (r_squared, _) = super::compute_r_squared(y, &residuals, total_comp);

    // Only return the ncomp FPC-score components (not scalar covariate components)
    // plus the bandwidths split accordingly.
    // But the contract says component_fits has length ncomp+p_scalar; callers can slice.
    Ok((component_fits, bandwidths, mu_y, fitted_values, residuals, r_squared))
}

// ---------------------------------------------------------------------------
// Public estimators
// ---------------------------------------------------------------------------

/// Functional Additive Model (FAM) — Müller & Yao (2008).
///
/// Fits `E[Y | X] = μ_Y + Σ_{k=1}^{K} f_k(ξ_k)` where `ξ_k` are the k-th
/// functional principal component scores of `X`. Because FPC scores are
/// uncorrelated (orthogonal in L²), fitting each component reduces to an
/// independent 1-D Nadaraya-Watson regression on the partial residual — a
/// single sequential forward pass achieves the same result as infinite-iteration
/// backfitting.
///
/// # Arguments
/// * `data` — Functional predictor matrix (n × m, column-major).
/// * `y` — Scalar response vector (length n).
/// * `argvals` — Evaluation grid (length m).
/// * `scalar_covariates` — Optional scalar covariates (n × p); treated as
///   additional additive components in the same forward pass.
/// * `config` — Tuning parameters; see [`FamConfig`].
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if:
/// - `data` has 0 rows or 0 columns,
/// - `y.len() != n`,
/// - `argvals.len() != m`, or
/// - `scalar_covariates.nrows() != n`.
///
/// Returns [`FdarError::InvalidParameter`] if an explicitly-provided
/// `config.ncomp` exceeds `min(n, m)`.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::fam;
/// use fdars_core::scalar_on_function::FamConfig;
///
/// let n = 30;
/// let m = 20;
/// let data = FdMatrix::from_column_major(
///     (0..n*m).map(|i| (i as f64 * 0.1).sin()).collect(),
///     n, m,
/// ).unwrap();
/// let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).cos()).collect();
/// let result = fam(&data, &y, &argvals, None, &FamConfig::default()).unwrap();
/// assert_eq!(result.fitted_values.len(), n);
/// assert!(result.r_squared >= 0.0);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fam(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    config: &FamConfig,
) -> Result<FamResult, FdarError> {
    let (n, m) = data.shape();

    // Validate inputs
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 row".to_string(),
            actual: "0".to_string(),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 column".to_string(),
            actual: "0".to_string(),
        });
    }
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n}"),
            actual: format!("{}", y.len()),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if let Some(sc) = scalar_covariates {
        if sc.nrows() != n {
            return Err(FdarError::InvalidDimension {
                parameter: "scalar_covariates",
                expected: format!("{n} rows"),
                actual: format!("{} rows", sc.nrows()),
            });
        }
    }

    // Resolve ncomp
    let ncomp = resolve_ncomp_additive(
        config.ncomp,
        n,
        m,
        data,
        y,
        argvals,
        &config.kernel,
        config.n_grid_bandwidth,
    )?;

    // Compute FPC scores
    let fpca = fdata_to_pc_1d(data, ncomp, argvals)?;

    // One-pass additive smooth
    let p_scalar = scalar_covariates.map_or(0, FdMatrix::ncols);
    let total_comp = ncomp + p_scalar;
    let (component_fits_all, bandwidths_all, intercept, fitted_values, residuals, r_squared) =
        fpc_additive_smooth(
            &fpca,
            y,
            n,
            ncomp,
            config.bandwidth,
            &config.kernel,
            config.n_grid_bandwidth,
            scalar_covariates,
        )?;

    // Separate FPC component fits from scalar covariate fits
    let component_fits: Vec<Vec<f64>> = component_fits_all.into_iter().take(total_comp).collect();
    let bandwidths: Vec<f64> = bandwidths_all.into_iter().take(total_comp).collect();

    Ok(FamResult {
        fitted_values,
        residuals,
        component_fits,
        intercept,
        bandwidths,
        ncomp,
        r_squared,
        fpca,
    })
}

/// Generalized Kernel Additive Model (GKAM).
///
/// Fits `ŷ = μ + Σ_k f_k(X^k)` by iterative backfitting where each `f_k` is a
/// Nadaraya-Watson smoother on the L2 distance kernel between functional curves.
/// Unlike FAM, the predictor distances are not orthogonal, so true iterative
/// backfitting is required for convergence.
///
/// # Arguments
/// * `predictors` — Slice of functional predictor matrices (each n × m_k).
/// * `y` — Scalar response (length n).
/// * `argvals_list` — Evaluation grids; `argvals_list[k]` has length `predictors[k].ncols()`.
/// * `scalar_covariates` — Optional scalar covariates (n × p); appended as extra additive terms.
/// * `config` — Tuning parameters; see [`GkamConfig`].
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if:
/// - `predictors` is empty,
/// - `predictors.len() != argvals_list.len()`,
/// - any `predictors[k].nrows() != y.len()`, or
/// - any `argvals_list[k].len() != predictors[k].ncols()`.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::fregre_gkam;
/// use fdars_core::scalar_on_function::GkamConfig;
///
/// let n = 20;
/// let m = 15;
/// let data = FdMatrix::from_column_major(
///     (0..n*m).map(|i| (i as f64 * 0.15).sin()).collect(),
///     n, m,
/// ).unwrap();
/// let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let y: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
/// let result = fregre_gkam(&[&data], &y, &[argvals.as_slice()], None, &GkamConfig::default()).unwrap();
/// assert_eq!(result.fitted_values.len(), n);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fregre_gkam(
    predictors: &[&FdMatrix],
    y: &[f64],
    argvals_list: &[&[f64]],
    scalar_covariates: Option<&FdMatrix>,
    config: &GkamConfig,
) -> Result<GkamResult, FdarError> {
    let n = y.len();

    // Validate inputs
    if predictors.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "predictors",
            expected: "at least 1 functional predictor".to_string(),
            actual: "0".to_string(),
        });
    }
    if predictors.len() != argvals_list.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals_list",
            expected: format!("{} (matching predictors.len())", predictors.len()),
            actual: format!("{}", argvals_list.len()),
        });
    }
    for (k, pred) in predictors.iter().enumerate() {
        if pred.nrows() != n {
            return Err(FdarError::InvalidDimension {
                parameter: "predictors[k].nrows()",
                expected: format!("{n} (y.len())"),
                actual: format!("{} for predictor {k}", pred.nrows()),
            });
        }
        if argvals_list[k].len() != pred.ncols() {
            return Err(FdarError::InvalidDimension {
                parameter: "argvals_list[k]",
                expected: format!("{} (predictors[k].ncols())", pred.ncols()),
                actual: format!("{} for predictor {k}", argvals_list[k].len()),
            });
        }
    }
    if let Some(sc) = scalar_covariates {
        if sc.nrows() != n {
            return Err(FdarError::InvalidDimension {
                parameter: "scalar_covariates",
                expected: format!("{n} rows"),
                actual: format!("{} rows", sc.nrows()),
            });
        }
    }

    let q = predictors.len();
    let p_scalar = scalar_covariates.map_or(0, FdMatrix::ncols);
    let total_comp = q + p_scalar;

    let mu_y = y.iter().sum::<f64>() / n as f64;

    // Precompute pairwise L2 distance matrices (once per predictor)
    let dist_matrices: Vec<Vec<f64>> = predictors
        .iter()
        .zip(argvals_list.iter())
        .map(|(pred, argvals)| compute_pairwise_distances(pred, argvals))
        .collect();

    // Select per-covariate bandwidths
    let bandwidths_func: Vec<f64> = if config.bandwidth > 0.0 {
        vec![config.bandwidth; q]
    } else {
        dist_matrices
            .iter()
            .map(|dists| select_bandwidth_loo(dists, y, n, None))
            .collect()
    };

    // For scalar covariates, compute Euclidean distances and bandwidths
    let scalar_dists: Vec<Vec<f64>> = if let Some(sc) = scalar_covariates {
        (0..p_scalar)
            .map(|j| {
                let mut d = vec![0.0_f64; n * n];
                for i in 0..n {
                    for jj in (i + 1)..n {
                        let diff = sc[(i, j)] - sc[(jj, j)];
                        let dist = diff.abs();
                        d[i * n + jj] = dist;
                        d[jj * n + i] = dist;
                    }
                }
                d
            })
            .collect()
    } else {
        Vec::new()
    };

    let scalar_bandwidths: Vec<f64> = if p_scalar > 0 {
        if config.bandwidth > 0.0 {
            vec![config.bandwidth; p_scalar]
        } else {
            scalar_dists
                .iter()
                .map(|dists| select_bandwidth_loo(dists, y, n, None))
                .collect()
        }
    } else {
        Vec::new()
    };

    // Merge bandwidths: functional first, then scalar
    let mut all_bandwidths = bandwidths_func.clone();
    all_bandwidths.extend_from_slice(&scalar_bandwidths);

    // Initialize component fits to zero
    let mut component_fits = vec![vec![0.0_f64; n]; total_comp];
    let mut converged = false;
    let mut iterations = 0;

    // Iterative backfitting loop (bounded by max_iter)
    for iter in 0..config.max_iter {
        let mut max_delta = 0.0_f64;

        // Update functional predictor components
        for k in 0..q {
            let h_k = all_bandwidths[k];
            let dists_k = &dist_matrices[k];

            // Compute adjusted response: y - mu - sum_{j != k} f_j
            let adjusted: Vec<f64> = (0..n)
                .map(|i| {
                    let others: f64 = (0..total_comp)
                        .filter(|&j| j != k)
                        .map(|j| component_fits[j][i])
                        .sum();
                    y[i] - mu_y - others
                })
                .collect();

            // Apply NW smoother on L2 distance kernel (O(n) per point)
            let new_fk: Vec<f64> = (0..n)
                .map(|i| {
                    let mut num = 0.0_f64;
                    let mut den = 0.0_f64;
                    for j in 0..n {
                        let w = gaussian_kernel(dists_k[i * n + j], h_k);
                        num += w * adjusted[j];
                        den += w;
                    }
                    if den > 1e-15 {
                        num / den
                    } else {
                        adjusted[i]
                    }
                })
                .collect();

            // Track max change across all observations
            let delta = component_fits[k]
                .iter()
                .zip(&new_fk)
                .map(|(old, &new)| (old - new).abs())
                .fold(0.0_f64, f64::max);
            max_delta = max_delta.max(delta);
            component_fits[k] = new_fk;
        }

        // Update scalar covariate components
        for s_idx in 0..p_scalar {
            let k = q + s_idx;
            let h_k = all_bandwidths[k];
            let dists_k = &scalar_dists[s_idx];

            let adjusted: Vec<f64> = (0..n)
                .map(|i| {
                    let others: f64 = (0..total_comp)
                        .filter(|&j| j != k)
                        .map(|j| component_fits[j][i])
                        .sum();
                    y[i] - mu_y - others
                })
                .collect();

            let new_fk: Vec<f64> = (0..n)
                .map(|i| {
                    let mut num = 0.0_f64;
                    let mut den = 0.0_f64;
                    for j in 0..n {
                        let w = gaussian_kernel(dists_k[i * n + j], h_k);
                        num += w * adjusted[j];
                        den += w;
                    }
                    if den > 1e-15 {
                        num / den
                    } else {
                        adjusted[i]
                    }
                })
                .collect();

            let delta = component_fits[k]
                .iter()
                .zip(&new_fk)
                .map(|(old, &new)| (old - new).abs())
                .fold(0.0_f64, f64::max);
            max_delta = max_delta.max(delta);
            component_fits[k] = new_fk;
        }

        iterations = iter + 1;
        if max_delta < config.epsilon {
            converged = true;
            break;
        }
    }

    // Assemble result
    let fitted_values: Vec<f64> = (0..n)
        .map(|i| mu_y + (0..total_comp).map(|k| component_fits[k][i]).sum::<f64>())
        .collect();
    let residuals: Vec<f64> =
        y.iter().zip(&fitted_values).map(|(&yi, &yh)| yi - yh).collect();
    let (r_squared, _) = super::compute_r_squared(y, &residuals, total_comp);

    Ok(GkamResult {
        fitted_values,
        residuals,
        component_fits,
        intercept: mu_y,
        bandwidths: all_bandwidths,
        iterations,
        converged,
        r_squared,
    })
}

/// Generalized Spectral Additive Model (GSAM).
///
/// Fits the same FPC-score additive model as [`fam`] but is framed as a
/// generalised additive model in the FPC score space. Under the Gaussian
/// identity link the implementation is numerically equivalent to FAM.
///
/// # Arguments
/// * `data` — Functional predictor matrix (n × m, column-major).
/// * `y` — Scalar response (length n).
/// * `argvals` — Evaluation grid (length m).
/// * `scalar_covariates` — Optional scalar covariates (n × p).
/// * `config` — Tuning parameters; see [`GsamConfig`].
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] or [`FdarError::InvalidParameter`]
/// (with `ncomp > min(n, m)`) under the same conditions as [`fam`].
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::fregre_gsam;
/// use fdars_core::scalar_on_function::GsamConfig;
///
/// let n = 30;
/// let m = 20;
/// let data = FdMatrix::from_column_major(
///     (0..n*m).map(|i| (i as f64 * 0.1).cos()).collect(),
///     n, m,
/// ).unwrap();
/// let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
/// let result = fregre_gsam(&data, &y, &argvals, None, &GsamConfig::default()).unwrap();
/// assert_eq!(result.fitted_values.len(), n);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fregre_gsam(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    config: &GsamConfig,
) -> Result<GsamResult, FdarError> {
    let (n, m) = data.shape();

    // Validate inputs — identical to fam
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 row".to_string(),
            actual: "0".to_string(),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 column".to_string(),
            actual: "0".to_string(),
        });
    }
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n}"),
            actual: format!("{}", y.len()),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if let Some(sc) = scalar_covariates {
        if sc.nrows() != n {
            return Err(FdarError::InvalidDimension {
                parameter: "scalar_covariates",
                expected: format!("{n} rows"),
                actual: format!("{} rows", sc.nrows()),
            });
        }
    }

    // Resolve ncomp (same logic as fam, including InvalidParameter for ncomp > min(n,m))
    let ncomp = resolve_ncomp_additive(
        config.ncomp,
        n,
        m,
        data,
        y,
        argvals,
        &config.kernel,
        config.n_grid_bandwidth,
    )?;

    // Compute FPC scores
    let fpca = fdata_to_pc_1d(data, ncomp, argvals)?;

    // One-pass additive smooth (identical path to fam — GSAM = FAM under Gaussian identity link)
    let p_scalar = scalar_covariates.map_or(0, FdMatrix::ncols);
    let total_comp = ncomp + p_scalar;
    let (component_fits_all, bandwidths_all, intercept, fitted_values, residuals, r_squared) =
        fpc_additive_smooth(
            &fpca,
            y,
            n,
            ncomp,
            config.bandwidth,
            &config.kernel,
            config.n_grid_bandwidth,
            scalar_covariates,
        )?;

    let component_fits: Vec<Vec<f64>> = component_fits_all.into_iter().take(total_comp).collect();
    let bandwidths: Vec<f64> = bandwidths_all.into_iter().take(total_comp).collect();

    Ok(GsamResult {
        fitted_values,
        residuals,
        component_fits,
        intercept,
        bandwidths,
        ncomp,
        r_squared,
        fpca,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    /// Build a synthetic FdMatrix from sinusoidal curves.
    fn make_sine_data(n: usize, m: usize, freq_scale: f64) -> FdMatrix {
        let data: Vec<f64> = (0..n)
            .flat_map(|i| {
                (0..m).map(move |j| {
                    let t = j as f64 / (m - 1) as f64;
                    (freq_scale * (i as f64 + 1.0) * t).sin()
                })
            })
            .collect();
        // column-major: column j contains all n observations at time-point j
        let mut cm = vec![0.0_f64; n * m];
        for i in 0..n {
            for j in 0..m {
                cm[j * n + i] = data[i * m + j];
            }
        }
        FdMatrix::from_column_major(cm, n, m).unwrap()
    }

    // -----------------------------------------------------------------------
    // FAM tests
    // -----------------------------------------------------------------------

    #[test]
    fn fam_synthetic_recovery() {
        // y_i = sin(xi_1) + xi_2^2 + noise — FAM with 2 FPC components should recover.
        let n = 50;
        let m = 20;
        let argvals = uniform_grid(m);

        // Generate curves as sine waves with random phase proxy (deterministic)
        let data = make_sine_data(n, m, 1.0);
        // Extract scores by running FPCA; build y from known structure
        let fpca = fdata_to_pc_1d(&data, 2, &argvals).unwrap();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let xi1 = fpca.scores[(i, 0)];
                let xi2 = fpca.scores[(i, 1)];
                // Small noise proportional to score range to keep SNR high
                let noise = (i as f64 * 0.31).sin() * 0.05;
                xi1.sin() + xi2 * xi2 + noise
            })
            .collect();

        let config = FamConfig {
            ncomp: 2,
            bandwidth: 0.0,
            ..Default::default()
        };
        let result = fam(&data, &y, &argvals, None, &config).unwrap();

        // R² should be substantially above a mean-only baseline
        assert!(
            result.r_squared > 0.75,
            "expected R² > 0.75, got {}",
            result.r_squared
        );

        // Relative fitted error < 30%
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let ss_y: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>();
        let ss_res: f64 = result.residuals.iter().map(|r| r * r).sum();
        let rel_err = (ss_res / ss_y).sqrt();
        assert!(
            rel_err < 0.30,
            "expected relative fitted error < 0.30, got {rel_err:.4}"
        );
    }

    #[test]
    fn fam_decomposition_identity() {
        // fitted_values + residuals == y elementwise (within 1e-9)
        let n = 30;
        let m = 15;
        let argvals = uniform_grid(m);
        let data = make_sine_data(n, m, 1.5);
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).cos()).collect();
        let config = FamConfig { ncomp: 2, ..Default::default() };
        let result = fam(&data, &y, &argvals, None, &config).unwrap();

        for i in 0..n {
            let reconstructed = result.fitted_values[i] + result.residuals[i];
            assert!(
                (reconstructed - y[i]).abs() < 1e-9,
                "decomposition failed at i={i}: fitted={} residual={} sum={} y={}",
                result.fitted_values[i],
                result.residuals[i],
                reconstructed,
                y[i]
            );
        }
    }

    #[test]
    fn fam_output_shapes() {
        let n = 25;
        let m = 12;
        let argvals = uniform_grid(m);
        let data = make_sine_data(n, m, 1.0);
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let config = FamConfig { ncomp: 3, ..Default::default() };
        let result = fam(&data, &y, &argvals, None, &config).unwrap();

        assert_eq!(result.ncomp, 3, "ncomp field should be 3");
        assert_eq!(result.component_fits.len(), 3, "component_fits.len() should equal ncomp");
        for (k, cf) in result.component_fits.iter().enumerate() {
            assert_eq!(cf.len(), n, "component_fits[{k}] should have length n={n}");
        }
        assert_eq!(result.bandwidths.len(), 3, "bandwidths.len() should equal ncomp");
        assert_eq!(result.fitted_values.len(), n);
        assert_eq!(result.residuals.len(), n);
    }

    #[test]
    fn fam_invalid_dimension() {
        let n = 20;
        let m = 10;
        let argvals = uniform_grid(m);
        let data = make_sine_data(n, m, 1.0);
        let y_ok: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let config = FamConfig { ncomp: 2, ..Default::default() };

        // Empty FdMatrix (0 rows)
        let empty_data = FdMatrix::zeros(0, m);
        let err = fam(&empty_data, &y_ok, &argvals, None, &config);
        assert!(err.is_err(), "empty data should return Err");
        match err.unwrap_err() {
            FdarError::InvalidDimension { parameter, .. } => {
                assert_eq!(parameter, "data");
            }
            e => panic!("expected InvalidDimension, got {e:?}"),
        }

        // y of wrong length
        let y_wrong: Vec<f64> = vec![1.0; n + 5];
        let err = fam(&data, &y_wrong, &argvals, None, &config);
        assert!(err.is_err(), "mismatched y length should return Err");
        match err.unwrap_err() {
            FdarError::InvalidDimension { parameter, .. } => {
                assert_eq!(parameter, "y");
            }
            e => panic!("expected InvalidDimension, got {e:?}"),
        }

        // argvals of wrong length
        let argvals_wrong: Vec<f64> = uniform_grid(m + 3);
        let err = fam(&data, &y_ok, &argvals_wrong, None, &config);
        assert!(err.is_err(), "mismatched argvals should return Err");
        match err.unwrap_err() {
            FdarError::InvalidDimension { parameter, .. } => {
                assert_eq!(parameter, "argvals");
            }
            e => panic!("expected InvalidDimension, got {e:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // GKAM tests
    // -----------------------------------------------------------------------

    #[test]
    fn gkam_r2_synthetic() {
        // One functional covariate; y is a pure function of the L2 norm of X (+ tiny noise).
        // The L2 distance kernel in GKAM should recover this functional dependence well.
        let n = 40;
        let m = 15;
        let argvals = uniform_grid(m);

        // Curves with varying amplitude: curve i has amplitude proportional to i
        let mut cm = vec![0.0_f64; n * m];
        for i in 0..n {
            let amp = (i as f64 + 1.0) / n as f64; // amplitude 1/n … 1
            for j in 0..m {
                let t = j as f64 / (m - 1) as f64;
                // column-major: index = j*n + i
                cm[j * n + i] = amp * (std::f64::consts::PI * 2.0 * t).sin();
            }
        }
        let data = FdMatrix::from_column_major(cm, n, m).unwrap();

        // y is a monotone function of the amplitude (== L2 norm up to constant factor)
        // So GKAM on L2 distances should recover this very well.
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let amp = (i as f64 + 1.0) / n as f64;
                // y = amp^2 (nonlinear in amp but determined by it — R² should be high)
                let noise = (i as f64 * 0.23).sin() * 0.002;
                amp * amp + noise
            })
            .collect();

        let config = GkamConfig {
            max_iter: 20,
            epsilon: 1e-4,
            ..Default::default()
        };
        let result = fregre_gkam(&[&data], &y, &[&argvals], None, &config).unwrap();

        assert!(
            result.r_squared > 0.70,
            "expected R² > 0.70, got {}",
            result.r_squared
        );
    }

    #[test]
    fn gkam_convergence() {
        // On smooth data GKAM should converge within max_iter iterations
        let n = 25;
        let m = 10;
        let argvals = uniform_grid(m);
        let data = make_sine_data(n, m, 1.0);
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

        let config = GkamConfig {
            max_iter: 50,
            epsilon: 1e-4,
            ..Default::default()
        };
        let result = fregre_gkam(&[&data], &y, &[&argvals], None, &config).unwrap();

        assert!(result.converged, "expected convergence, got iterations={}", result.iterations);
        assert!(
            result.iterations <= config.max_iter,
            "iterations {} > max_iter {}",
            result.iterations,
            config.max_iter
        );
    }

    #[test]
    fn gkam_invalid_inputs() {
        let n = 20;
        let m = 10;
        let argvals = uniform_grid(m);
        let data = make_sine_data(n, m, 1.0);
        let y_ok: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let config = GkamConfig::default();

        // Empty predictors list
        let err = fregre_gkam(&[], &y_ok, &[], None, &config);
        assert!(err.is_err(), "empty predictors should return Err");

        // Mismatched predictor/y lengths
        let data_wrong = make_sine_data(n + 5, m, 1.0);
        let err = fregre_gkam(&[&data_wrong], &y_ok, &[&argvals], None, &config);
        assert!(err.is_err(), "mismatched n should return Err");
        match err.unwrap_err() {
            FdarError::InvalidDimension { .. } => {}
            e => panic!("expected InvalidDimension, got {e:?}"),
        }

        // argvals_list length mismatch
        let err = fregre_gkam(&[&data], &y_ok, &[], None, &config);
        assert!(err.is_err(), "argvals_list length mismatch should return Err");
    }

    // -----------------------------------------------------------------------
    // GSAM tests
    // -----------------------------------------------------------------------

    #[test]
    fn gsam_matches_fam_identity() {
        // With identical config, gsam and fam should produce the same fitted values (within 1e-6).
        let n = 40;
        let m = 16;
        let argvals = uniform_grid(m);
        let data = make_sine_data(n, m, 1.0);
        let fpca_ref = fdata_to_pc_1d(&data, 2, &argvals).unwrap();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let xi1 = fpca_ref.scores[(i, 0)];
                let xi2 = fpca_ref.scores[(i, 1)];
                xi1 + xi2 * xi2 + (i as f64 * 0.23).sin() * 0.02
            })
            .collect();

        let fam_config = FamConfig {
            ncomp: 2,
            bandwidth: 0.5, // fixed bandwidth for deterministic comparison
            kernel: "gaussian".to_string(),
            n_grid_bandwidth: 20,
        };
        let gsam_config = GsamConfig {
            ncomp: 2,
            bandwidth: 0.5,
            kernel: "gaussian".to_string(),
            n_grid_bandwidth: 20,
        };

        let fam_res = fam(&data, &y, &argvals, None, &fam_config).unwrap();
        let gsam_res = fregre_gsam(&data, &y, &argvals, None, &gsam_config).unwrap();

        for i in 0..n {
            let diff = (fam_res.fitted_values[i] - gsam_res.fitted_values[i]).abs();
            assert!(
                diff < 1e-6,
                "fam vs gsam mismatch at i={i}: fam={} gsam={} diff={diff:.2e}",
                fam_res.fitted_values[i],
                gsam_res.fitted_values[i]
            );
        }
    }

    #[test]
    fn gsam_ncomp_too_large() {
        let n = 15;
        let m = 8;
        let argvals = uniform_grid(m);
        let data = make_sine_data(n, m, 1.0);
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();

        // ncomp > min(n, m) = 8
        let config = GsamConfig { ncomp: 100, ..Default::default() };
        let err = fregre_gsam(&data, &y, &argvals, None, &config);
        assert!(err.is_err(), "ncomp > min(n,m) should return Err");
        match err.unwrap_err() {
            FdarError::InvalidParameter { parameter, .. } => {
                assert_eq!(parameter, "config.ncomp");
            }
            e => panic!("expected InvalidParameter, got {e:?}"),
        }
    }

    #[test]
    fn gsam_output_shapes() {
        let n = 30;
        let m = 10;
        let argvals = uniform_grid(m);
        let data = make_sine_data(n, m, 1.0);
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

        let config = GsamConfig { ncomp: 3, ..Default::default() };
        let result = fregre_gsam(&data, &y, &argvals, None, &config).unwrap();

        assert_eq!(result.ncomp, 3);
        assert_eq!(result.component_fits.len(), 3, "component_fits.len() should equal ncomp");
        assert_eq!(result.fitted_values.len(), n);
    }
}
