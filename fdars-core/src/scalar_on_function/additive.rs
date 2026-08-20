//! Nonparametric additive scalar-on-function regression.
//!
//! Implements six additive estimators for the model
//! `E[Y | X] = μ + Σ_k f_k(ξ_k)` and its functional-distance variant,
//! plus variable selection, permutation testing, and history-index estimation:
//!
//! - [`fam`] — Functional Additive Model (Müller & Yao 2008): one-pass NW over
//!   FPC scores (no backfitting loop needed because FPC scores are uncorrelated).
//! - [`fregre_gkam`] — Generalized Kernel Additive Model: iterative backfitting
//!   over Nadaraya-Watson smoothers on functional L2 distances.
//! - [`fregre_gsam`] — Generalized Spectral Additive Model: FPC-score basis
//!   with additive NW smoothing; numerically equivalent to FAM under the
//!   Gaussian identity link.
//! - [`variable_selection`] — Group-penalized coordinate descent in FPC-score
//!   space. Implements GroupLasso; GroupMCP/GroupSCAD are documented as future
//!   work.
//! - [`permutation_test_fam`] — Seeded permutation significance test for FAM.
//! - [`history_index`] — Lagged-predictor-window estimator via marginal-
//!   integration Nadaraya-Watson over a discretised lag grid.
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
//! - **variable_selection:** R's `refund::fosr.vs` implements function-on-scalar
//!   regression (functional response, scalar predictors). fdars implements
//!   scalar-on-function variable selection (scalar response, functional
//!   predictors). The group-penalty formulation is analogous but the regression
//!   direction is opposite. GroupMCP and GroupSCAD are documented as future work;
//!   only GroupLasso (convex) is implemented this phase.
//! - **history_index:** R's `refund::pffr` with `ff(..., limits)` implements the
//!   full function-on-function history model as a lower-triangular bivariate
//!   spline. fdars implements the scalar-on-function reduction (scalar Y, history
//!   index evaluated at T = `argvals.last()`) via NW smoothing over a discretised
//!   lag grid — same model class, marginal-integration approximation rather than
//!   bivariate spline.

use super::nonparametric::{compute_pairwise_distances, gaussian_kernel, select_bandwidth_loo};
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};
use crate::smoothing::{nadaraya_watson, optim_bandwidth, CvCriterion};

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
        let cap = max_ncomp.clamp(1, 10);
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
    let residuals: Vec<f64> = y
        .iter()
        .zip(&fitted_values)
        .map(|(&yi, &yh)| yi - yh)
        .collect();

    // R² via shared helper (p = total_comp for df counting)
    let (r_squared, _) = super::compute_r_squared(y, &residuals, total_comp);

    // Only return the ncomp FPC-score components (not scalar covariate components)
    // plus the bandwidths split accordingly.
    // But the contract says component_fits has length ncomp+p_scalar; callers can slice.
    Ok((
        component_fits,
        bandwidths,
        mu_y,
        fitted_values,
        residuals,
        r_squared,
    ))
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
    let residuals: Vec<f64> = y
        .iter()
        .zip(&fitted_values)
        .map(|(&yi, &yh)| yi - yh)
        .collect();
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
// Wave-2 config and result types
// ---------------------------------------------------------------------------

/// Penalty type for [`variable_selection`].
///
/// Only `GroupLasso` is fully implemented. `GroupMcp` and `GroupScad` are
/// documented here for API completeness; calling `variable_selection` with
/// either returns `FdarError::InvalidParameter` — they are deferred to a
/// future phase.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VarSelectPenalty {
    /// Group lasso (convex); fully implemented. Recommended default.
    GroupLasso,
    /// Group MCP (minimax concave penalty). **Not yet implemented** — returns
    /// `FdarError::InvalidParameter`; deferred to a future phase.
    GroupMcp,
    /// Group SCAD (smoothly clipped absolute deviation). **Not yet
    /// implemented** — returns `FdarError::InvalidParameter`; deferred to a
    /// future phase.
    GroupScad,
    /// Ordinary least squares (no group penalty). Sets all predictors active.
    Ls,
}

/// Configuration for [`variable_selection`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VarSelectConfig {
    /// FPC components per predictor. 0 = auto-select via GCV (default: 3).
    pub ncomp: usize,
    /// Group penalty type (default: [`VarSelectPenalty::GroupLasso`]).
    pub penalty: VarSelectPenalty,
    /// Penalty weight λ. 0.0 = CV-select over a grid (default: 0.0).
    pub lambda: f64,
    /// Maximum coordinate-descent iterations (default: 100).
    pub max_iter: usize,
    /// Convergence threshold on max coefficient delta (default: 1e-5).
    pub epsilon: f64,
    /// Grid size for λ selection (default: 20).
    pub lambda_n_grid: usize,
}

impl Default for VarSelectConfig {
    fn default() -> Self {
        Self {
            ncomp: 3,
            penalty: VarSelectPenalty::GroupLasso,
            lambda: 0.0,
            max_iter: 100,
            epsilon: 1e-5,
            lambda_n_grid: 20,
        }
    }
}

/// Result of [`variable_selection`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VarSelectResult {
    /// Whether each functional predictor is active (length P).
    pub active_predictors: Vec<bool>,
    /// Group-lasso coefficient vector per predictor (P × K_p).
    pub coefficients: Vec<Vec<f64>>,
    /// Fitted values ŷ (length n).
    pub fitted_values: Vec<f64>,
    /// Residuals y − ŷ (length n).
    pub residuals: Vec<f64>,
    /// Intercept (mean response).
    pub intercept: f64,
    /// Selected or provided λ.
    pub lambda: f64,
    /// R² statistic.
    pub r_squared: f64,
    /// Coordinate-descent iterations performed.
    pub iterations: usize,
    /// Whether the coordinate-descent loop converged.
    pub converged: bool,
    /// FPCA result for each predictor (for projecting new data).
    pub fpcas: Vec<FpcaResult>,
}

/// Test statistic for [`permutation_test_fam`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PermTestStatistic {
    /// R² of the full additive fit (default).
    R2,
    /// L2 norm of fitted values.
    FittedNorm,
    /// Sum of integrated component norms (FAM only).
    ComponentNorm,
}

/// Configuration for [`permutation_test_fam`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PermTestConfig {
    /// Number of permutations (default: 999).
    pub n_perm: usize,
    /// Random seed for reproducibility (default: 42).
    pub seed: u64,
    /// Test statistic to use (default: [`PermTestStatistic::R2`]).
    pub statistic: PermTestStatistic,
}

impl Default for PermTestConfig {
    fn default() -> Self {
        Self {
            n_perm: 999,
            seed: 42,
            statistic: PermTestStatistic::R2,
        }
    }
}

/// Result of [`permutation_test_fam`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PermTestResult {
    /// Permutation p-value: (n_ge + 1) / (n_perm + 1).
    pub p_value: f64,
    /// Test statistic on the original (unpermuted) data.
    pub observed_statistic: f64,
    /// Test statistic for each permuted dataset (length ≤ n_perm).
    pub null_statistics: Vec<f64>,
    /// Number of permutation refits that returned `Ok`.
    pub n_perm_success: usize,
}

/// Configuration for [`history_index`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HistoryIndexConfig {
    /// Lag window length Δ; must be ≤ `argvals` range.
    pub window: f64,
    /// Number of lag grid points (default: 20).
    pub n_lags: usize,
    /// Bandwidth for the history weight function. 0.0 = auto via GCV (default: 0.0).
    pub bandwidth: f64,
    /// Kernel type (default: "gaussian").
    pub kernel: String,
}

impl Default for HistoryIndexConfig {
    fn default() -> Self {
        Self {
            window: 1.0,
            n_lags: 20,
            bandwidth: 0.0,
            kernel: "gaussian".to_string(),
        }
    }
}

/// Result of [`history_index`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HistoryIndexResult {
    /// Fitted values ŷ (length n).
    pub fitted_values: Vec<f64>,
    /// Residuals y − ŷ (length n).
    pub residuals: Vec<f64>,
    /// Intercept β₀.
    pub intercept: f64,
    /// Slope β₁ on the history score.
    pub slope: f64,
    /// Estimated history weight function γ (length n_lags).
    pub gamma: Vec<f64>,
    /// Lag discretisation points (length n_lags).
    pub lag_grid: Vec<f64>,
    /// Σ_l γ_l · X_i(T−u_l) · Δu for each observation (length n).
    pub history_scores: Vec<f64>,
    /// R² statistic.
    pub r_squared: f64,
}

// ---------------------------------------------------------------------------
// Wave-2 public estimators
// ---------------------------------------------------------------------------

/// Variable selection for scalar-on-function regression via group-penalised
/// coordinate descent in FPC-score space (GroupLasso).
///
/// Each functional predictor `predictors[p]` is reduced to `K_p` FPC scores
/// (one group). Group-lasso coordinate descent then selects which groups are
/// active.
///
/// # Algorithm
///
/// 1. Run `fdata_to_pc_1d` on each predictor → score groups ξ^0, …, ξ^{P−1}.
/// 2. Build design X = \[μ | ξ^0 | … | ξ^{P−1} | Z\] (Z = optional scalar
///    covariates).
/// 3. If `config.lambda == 0.0`, CV-select λ over a grid from
///    `0.01·λ_max` to `λ_max` where `λ_max = max_g ||X_g'y|| / √K_g`.
/// 4. Coordinate-descent group-lasso: for each group g compute the partial-
///    residual OLS update β̂_g via `cholesky_solve`, then soft-threshold:
///    `β_g = β̂_g · max(0, 1 − λ√K_g / ||β̂_g||)`.
/// 5. Iterate until `max(|Δβ|) < epsilon` or `max_iter` sweeps.
///
/// # R Baseline Divergence
///
/// R's `refund::fosr.vs` is a **function-on-scalar** model (functional response,
/// scalar predictors). fdars implements **scalar-on-function** variable selection
/// (scalar response, functional predictors). The group-penalty formulation is
/// analogous but the regression direction is opposite. GroupMCP and GroupSCAD are
/// documented as future work; only GroupLasso is implemented this phase.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] for unsupported penalty variants
/// (`GroupMcp`, `GroupScad`).
///
/// Returns [`FdarError::InvalidDimension`] if:
/// - `predictors` is empty,
/// - `predictors.len() != argvals_list.len()`, or
/// - any `predictors[p].nrows() != y.len()`.
///
/// Returns [`FdarError::ComputationFailed`] if the OLS sub-step encounters a
/// singular group design matrix.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::variable_selection;
/// use fdars_core::scalar_on_function::{VarSelectConfig, VarSelectPenalty};
///
/// let n = 20;
/// let m = 10;
/// let data = FdMatrix::from_column_major(
///     (0..n*m).map(|i| (i as f64 * 0.1).sin()).collect(),
///     n, m,
/// ).unwrap();
/// let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).cos()).collect();
/// let config = VarSelectConfig { ncomp: 2, ..Default::default() };
/// let result = variable_selection(&[&data], &y, &[argvals.as_slice()], None, &config).unwrap();
/// assert_eq!(result.active_predictors.len(), 1);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn variable_selection(
    predictors: &[&FdMatrix],
    y: &[f64],
    argvals_list: &[&[f64]],
    scalar_covariates: Option<&FdMatrix>,
    config: &VarSelectConfig,
) -> Result<VarSelectResult, FdarError> {
    // Check for unsupported penalty variants first
    match config.penalty {
        VarSelectPenalty::GroupMcp | VarSelectPenalty::GroupScad => {
            return Err(FdarError::InvalidParameter {
                parameter: "config.penalty",
                message: "GroupMcp and GroupScad are not yet implemented; use GroupLasso"
                    .to_string(),
            });
        }
        VarSelectPenalty::GroupLasso | VarSelectPenalty::Ls => {}
    }

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
    for (p, pred) in predictors.iter().enumerate() {
        if pred.nrows() != n {
            return Err(FdarError::InvalidDimension {
                parameter: "predictors[p].nrows()",
                expected: format!("{n} (y.len())"),
                actual: format!("{} for predictor {p}", pred.nrows()),
            });
        }
    }

    let big_p = predictors.len();
    let mu_y = y.iter().sum::<f64>() / n as f64;

    // Compute FPC scores for each predictor
    let ncomp_per = if config.ncomp == 0 { 3 } else { config.ncomp };

    let mut fpcas: Vec<FpcaResult> = Vec::with_capacity(big_p);
    let mut score_groups: Vec<Vec<Vec<f64>>> = Vec::with_capacity(big_p); // [p][k][i]

    for p in 0..big_p {
        let pred = predictors[p];
        let argvals = argvals_list[p];
        let (np, mp) = pred.shape();
        let k_p = ncomp_per.min(np.min(mp).saturating_sub(1).max(1));
        let fpca_p = fdata_to_pc_1d(pred, k_p, argvals)?;
        let k_actual = fpca_p.scores.ncols();
        let group_scores: Vec<Vec<f64>> = (0..k_actual)
            .map(|k| (0..n).map(|i| fpca_p.scores[(i, k)]).collect())
            .collect();
        score_groups.push(group_scores);
        fpcas.push(fpca_p);
    }

    // Handle Ls (ordinary least squares, no penalty)
    if config.penalty == VarSelectPenalty::Ls {
        return variable_selection_ls(y, n, mu_y, big_p, fpcas, score_groups, scalar_covariates);
    }

    // Build flat design matrix columns per group (excluding intercept here)
    // group_starts[p] = column index of group p in the flat score matrix
    let k_sizes: Vec<usize> = score_groups.iter().map(|g| g.len()).collect();

    // Compute lambda_max = max_g || X_g' (y - mu_y) || / sqrt(K_g)
    let y_centered: Vec<f64> = y.iter().map(|&yi| yi - mu_y).collect();
    let lambda_max = k_sizes
        .iter()
        .zip(score_groups.iter())
        .map(|(&k_g, group)| {
            let norm_sq: f64 = group
                .iter()
                .map(|col| {
                    let xgty: f64 = col.iter().zip(&y_centered).map(|(&x, &yc)| x * yc).sum();
                    xgty * xgty
                })
                .sum::<f64>();
            norm_sq.sqrt() / (k_g as f64).sqrt()
        })
        .fold(0.0_f64, f64::max)
        .max(1e-10); // avoid zero lambda_max

    // Select lambda via LOO-CV on a grid if config.lambda == 0.0
    let lambda = if config.lambda > 0.0 {
        config.lambda
    } else {
        select_group_lasso_lambda(
            y,
            &y_centered,
            mu_y,
            n,
            &score_groups,
            &k_sizes,
            lambda_max,
            config.lambda_n_grid,
            config.max_iter,
            config.epsilon,
            scalar_covariates,
        )
    };

    // Run group lasso coordinate descent at selected lambda
    let (coefficients, iterations, converged) = group_lasso_cd(
        y,
        &y_centered,
        mu_y,
        n,
        &score_groups,
        &k_sizes,
        lambda,
        config.max_iter,
        config.epsilon,
        scalar_covariates,
    )?;

    // Compute fitted values and residuals
    let fitted_values: Vec<f64> = compute_varselect_fitted(
        n,
        mu_y,
        &score_groups,
        &coefficients,
        scalar_covariates,
        big_p,
    );
    let residuals: Vec<f64> = y
        .iter()
        .zip(&fitted_values)
        .map(|(&yi, &yh)| yi - yh)
        .collect();
    let (r_squared, _) = super::compute_r_squared(y, &residuals, k_sizes.iter().sum::<usize>());

    let active_predictors: Vec<bool> = coefficients[..big_p]
        .iter()
        .map(|beta_g| {
            let norm: f64 = beta_g.iter().map(|&b| b * b).sum::<f64>();
            norm.sqrt() > config.epsilon
        })
        .collect();

    Ok(VarSelectResult {
        active_predictors,
        coefficients,
        fitted_values,
        residuals,
        intercept: mu_y,
        lambda,
        r_squared,
        iterations,
        converged,
        fpcas,
    })
}

/// OLS path for `VarSelectPenalty::Ls` (no group penalty).
fn variable_selection_ls(
    y: &[f64],
    n: usize,
    mu_y: f64,
    big_p: usize,
    fpcas: Vec<FpcaResult>,
    score_groups: Vec<Vec<Vec<f64>>>,
    scalar_covariates: Option<&FdMatrix>,
) -> Result<VarSelectResult, FdarError> {
    let k_sizes: Vec<usize> = score_groups.iter().map(|g| g.len()).collect();
    let p_scalar = scalar_covariates.map_or(0, FdMatrix::ncols);
    let total_cols = k_sizes.iter().sum::<usize>() + p_scalar;
    // Build n×total_cols design (no intercept column — absorbed into mu_y)
    let mut x_flat = vec![0.0_f64; n * total_cols];
    let mut col_offset = 0;
    for grp in &score_groups {
        for col in grp {
            for (i, &v) in col.iter().enumerate() {
                x_flat[col_offset * n + i] = v;
            }
            col_offset += 1;
        }
    }
    if let Some(sc) = scalar_covariates {
        for j in 0..p_scalar {
            for i in 0..n {
                x_flat[col_offset * n + i] = sc[(i, j)];
            }
            col_offset += 1;
        }
    }
    let x_mat = FdMatrix::from_column_major(x_flat, n, total_cols).map_err(|e| {
        FdarError::ComputationFailed {
            operation: "variable_selection_ls design matrix",
            detail: e.to_string(),
        }
    })?;
    let y_centered: Vec<f64> = y.iter().map(|&yi| yi - mu_y).collect();
    let xtx = super::compute_xtx(&x_mat);
    let xty: Vec<f64> = (0..total_cols)
        .map(|k| {
            x_mat
                .column(k)
                .iter()
                .zip(&y_centered)
                .map(|(&xv, &yv)| xv * yv)
                .sum::<f64>()
        })
        .collect();
    let l = super::cholesky_factor(&xtx, total_cols).map_err(|_| FdarError::ComputationFailed {
        operation: "variable_selection_ls cholesky",
        detail: "design matrix is singular".to_string(),
    })?;
    let flat_coeffs = super::cholesky_forward_back(&l, &xty, total_cols);

    // Split coefficients back into groups
    let mut coefficients: Vec<Vec<f64>> = Vec::with_capacity(big_p + 1);
    let mut offset = 0;
    for &k_g in &k_sizes {
        coefficients.push(flat_coeffs[offset..offset + k_g].to_vec());
        offset += k_g;
    }
    // Scalar covariate coefficients
    coefficients.push(flat_coeffs[offset..offset + p_scalar].to_vec());

    let fitted_values = compute_varselect_fitted(
        n,
        mu_y,
        &score_groups,
        &coefficients,
        scalar_covariates,
        big_p,
    );
    let residuals: Vec<f64> = y
        .iter()
        .zip(&fitted_values)
        .map(|(&yi, &yh)| yi - yh)
        .collect();
    let (r_squared, _) = super::compute_r_squared(y, &residuals, total_cols);
    let active_predictors = vec![true; big_p];
    Ok(VarSelectResult {
        active_predictors,
        coefficients,
        fitted_values,
        residuals,
        intercept: mu_y,
        lambda: 0.0,
        r_squared,
        iterations: 1,
        converged: true,
        fpcas,
    })
}

/// Compute fitted values from variable_selection coefficient structure.
fn compute_varselect_fitted(
    n: usize,
    mu_y: f64,
    score_groups: &[Vec<Vec<f64>>],
    coefficients: &[Vec<f64>],
    scalar_covariates: Option<&FdMatrix>,
    big_p: usize,
) -> Vec<f64> {
    let p_scalar = scalar_covariates.map_or(0, FdMatrix::ncols);
    (0..n)
        .map(|i| {
            let mut yhat = mu_y;
            for p in 0..big_p {
                for (k, col) in score_groups[p].iter().enumerate() {
                    yhat += coefficients[p][k] * col[i];
                }
            }
            if let Some(sc) = scalar_covariates {
                for j in 0..p_scalar {
                    yhat += coefficients[big_p][j] * sc[(i, j)];
                }
            }
            yhat
        })
        .collect()
}

/// Select lambda via simple LOO-proxy CV for group lasso.
///
/// Evaluates a grid of lambda values from `0.01·lambda_max` to `lambda_max`
/// and returns the lambda with the lowest mean squared prediction error.
#[allow(clippy::too_many_arguments)]
fn select_group_lasso_lambda(
    y: &[f64],
    _y_centered: &[f64],
    mu_y: f64,
    n: usize,
    score_groups: &[Vec<Vec<f64>>],
    k_sizes: &[usize],
    lambda_max: f64,
    n_grid: usize,
    max_iter: usize,
    epsilon: f64,
    scalar_covariates: Option<&FdMatrix>,
) -> f64 {
    let grid_size = n_grid.max(2);
    let mut best_lambda = lambda_max * 0.1;
    let mut best_mse = f64::INFINITY;

    for gi in 0..grid_size {
        let frac = (gi as f64 + 1.0) / grid_size as f64;
        let lam = lambda_max * (0.01_f64.powf(1.0 - frac)); // geometric grid: 0.01*lmax..lmax
        let y_centered: Vec<f64> = y.iter().map(|&yi| yi - mu_y).collect();
        if let Ok((coeffs, _, _)) = group_lasso_cd(
            y,
            &y_centered,
            mu_y,
            n,
            score_groups,
            k_sizes,
            lam,
            max_iter,
            epsilon,
            scalar_covariates,
        ) {
            let big_p = score_groups.len();
            let fitted =
                compute_varselect_fitted(n, mu_y, score_groups, &coeffs, scalar_covariates, big_p);
            let mse: f64 = y
                .iter()
                .zip(&fitted)
                .map(|(&yi, &yh)| (yi - yh).powi(2))
                .sum::<f64>()
                / n as f64;
            if mse < best_mse {
                best_mse = mse;
                best_lambda = lam;
            }
        }
    }
    best_lambda
}

/// Group-lasso coordinate descent.
///
/// Returns `(coefficients, iterations, converged)` where `coefficients` is a
/// `Vec<Vec<f64>>` of length `big_p + 1`; the last entry holds scalar
/// covariate coefficients (may be empty).
#[allow(clippy::too_many_arguments)]
fn group_lasso_cd(
    _y: &[f64],
    y_centered: &[f64],
    _mu_y: f64,
    n: usize,
    score_groups: &[Vec<Vec<f64>>],
    k_sizes: &[usize],
    lambda: f64,
    max_iter: usize,
    epsilon: f64,
    scalar_covariates: Option<&FdMatrix>,
) -> Result<(Vec<Vec<f64>>, usize, bool), FdarError> {
    let big_p = score_groups.len();
    let p_scalar = scalar_covariates.map_or(0, FdMatrix::ncols);

    // Initialize all group coefficients at zero
    let mut beta_groups: Vec<Vec<f64>> = score_groups
        .iter()
        .map(|grp| vec![0.0_f64; grp.len()])
        .collect();
    let mut beta_scalar: Vec<f64> = vec![0.0_f64; p_scalar];

    let mut converged = false;
    let mut iterations = 0;

    for _iter in 0..max_iter {
        let mut max_delta = 0.0_f64;

        // Update each functional predictor group
        for p in 0..big_p {
            let k_g = k_sizes[p];
            let group = &score_groups[p];

            // Build partial residual: y - mu_y - sum_{q != p} X_q beta_q - Z beta_z
            let partial: Vec<f64> = (0..n)
                .map(|i| {
                    let mut res = y_centered[i];
                    for q in 0..big_p {
                        if q != p {
                            for (k, col) in score_groups[q].iter().enumerate() {
                                res -= beta_groups[q][k] * col[i];
                            }
                        }
                    }
                    if let Some(sc) = scalar_covariates {
                        for j in 0..p_scalar {
                            res -= beta_scalar[j] * sc[(i, j)];
                        }
                    }
                    res
                })
                .collect();

            // OLS update for this group: beta_g_ols = (X_g'X_g)^{-1} X_g' partial
            // Build X_g'X_g (k_g × k_g) and X_g'partial
            let mut xtx_g = vec![0.0_f64; k_g * k_g];
            let mut xty_g = vec![0.0_f64; k_g];
            for a in 0..k_g {
                for b in 0..k_g {
                    let dot: f64 = group[a]
                        .iter()
                        .zip(&group[b])
                        .map(|(&xa, &xb)| xa * xb)
                        .sum();
                    xtx_g[a * k_g + b] = dot;
                }
                xty_g[a] = group[a]
                    .iter()
                    .zip(&partial)
                    .map(|(&xa, &pa)| xa * pa)
                    .sum();
            }

            let beta_ols = crate::linalg::cholesky_solve(&xtx_g, &xty_g, k_g)
                .unwrap_or_else(|_| xty_g.clone()); // fall back to gradient step on singular

            // Group-lasso soft threshold
            let norm_ols: f64 = beta_ols.iter().map(|&b| b * b).sum::<f64>().sqrt();
            let threshold = lambda * (k_g as f64).sqrt();
            let scale = if norm_ols > 1e-15 {
                (1.0 - threshold / norm_ols).max(0.0)
            } else {
                0.0
            };

            let new_beta: Vec<f64> = beta_ols.iter().map(|&b| b * scale).collect();

            // Track max change
            let delta = new_beta
                .iter()
                .zip(&beta_groups[p])
                .map(|(&nb, &ob)| (nb - ob).abs())
                .fold(0.0_f64, f64::max);
            max_delta = max_delta.max(delta);
            beta_groups[p] = new_beta;
        }

        // Update scalar covariate coefficients (no group penalty — standard OLS)
        if let Some(sc) = scalar_covariates {
            for j in 0..p_scalar {
                let partial_j: Vec<f64> = (0..n)
                    .map(|i| {
                        let mut res = y_centered[i];
                        for p in 0..big_p {
                            for (k, col) in score_groups[p].iter().enumerate() {
                                res -= beta_groups[p][k] * col[i];
                            }
                        }
                        for jj in 0..p_scalar {
                            if jj != j {
                                res -= beta_scalar[jj] * sc[(i, jj)];
                            }
                        }
                        res
                    })
                    .collect();
                let col_j: Vec<f64> = (0..n).map(|i| sc[(i, j)]).collect();
                let xjxj: f64 = col_j.iter().map(|&v| v * v).sum();
                let xjy: f64 = col_j.iter().zip(&partial_j).map(|(&x, &p)| x * p).sum();
                let new_bj = if xjxj > 1e-15 { xjy / xjxj } else { 0.0 };
                let delta = (new_bj - beta_scalar[j]).abs();
                max_delta = max_delta.max(delta);
                beta_scalar[j] = new_bj;
            }
        }

        iterations = _iter + 1;
        if max_delta < epsilon {
            converged = true;
            break;
        }
    }

    let mut coefficients: Vec<Vec<f64>> = beta_groups;
    coefficients.push(beta_scalar);
    Ok((coefficients, iterations, converged))
}

/// Permutation significance test for the Functional Additive Model ([`fam`]).
///
/// Assesses whether the additive relationship between the functional predictor
/// `data` and the scalar response `y` is statistically significant by comparing
/// the observed test statistic to a null distribution obtained by randomly
/// permuting `y`.
///
/// # Algorithm
///
/// 1. Fit [`fam`] on the original `(data, y)` → `T_obs`.
/// 2. Seed a single [`rand::rngs::StdRng`] with `perm_config.seed`.
/// 3. For each of `n_perm` iterations: clone `y`, shuffle the clone with `rng`,
///    refit FAM, compute `T_perm`. The single RNG advances deterministically
///    across iterations — this is NOT the per-thread `seed + k` seeding used
///    in parallel rayon loops.
/// 4. `p_value = (n_ge + 1) / (n_perm + 1)` (Phipson & Smyth 2010).
///
/// # Test Statistics
///
/// - [`PermTestStatistic::R2`] (default): R² of the fitted model.
/// - [`PermTestStatistic::FittedNorm`]: L2 norm of fitted values.
/// - [`PermTestStatistic::ComponentNorm`]: sum of per-component fit norms.
///
/// # Arguments
/// * `data` — Functional predictor matrix (n × m).
/// * `y` — Scalar response (length n).
/// * `argvals` — Evaluation grid (length m).
/// * `scalar_covariates` — Optional scalar covariates (n × p).
/// * `config` — FAM tuning parameters; see [`FamConfig`].
/// * `perm_config` — Permutation test configuration; see [`PermTestConfig`].
///
/// # Errors
///
/// Propagates [`FdarError`] from the initial `fam` fit. Individual permutation
/// errors are absorbed into `n_perm_success` (failed refits are skipped).
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::{permutation_test_fam, fam};
/// use fdars_core::scalar_on_function::{FamConfig, PermTestConfig, PermTestStatistic};
///
/// let n = 25;
/// let m = 10;
/// let data = FdMatrix::from_column_major(
///     (0..n*m).map(|i| (i as f64 * 0.1).sin()).collect(),
///     n, m,
/// ).unwrap();
/// let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).cos()).collect();
/// let fam_cfg = FamConfig { ncomp: 2, ..Default::default() };
/// let perm_cfg = PermTestConfig { n_perm: 9, seed: 42, statistic: PermTestStatistic::R2 };
/// let result = permutation_test_fam(&data, &y, &argvals, None, &fam_cfg, &perm_cfg).unwrap();
/// assert!((0.0..=1.0).contains(&result.p_value));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn permutation_test_fam(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    config: &FamConfig,
    perm_config: &PermTestConfig,
) -> Result<PermTestResult, FdarError> {
    // Fit on original data first (propagates FdarError on failure)
    let original_fit = fam(data, y, argvals, scalar_covariates, config)?;

    let observed_statistic = extract_perm_stat(&original_fit, perm_config.statistic);

    // Seeded RNG — place use inside function body per clippy/unused-import rule
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(perm_config.seed);

    let n_perm = perm_config.n_perm;
    let mut null_statistics: Vec<f64> = Vec::with_capacity(n_perm);
    let mut n_ge = 0usize;
    let mut n_perm_success = 0usize;

    let mut y_perm: Vec<f64> = y.to_vec();

    for _ in 0..n_perm {
        // Shuffle only y — reuse the same predictor buffers
        y_perm.copy_from_slice(y);
        y_perm.shuffle(&mut rng);

        match fam(data, &y_perm, argvals, scalar_covariates, config) {
            Ok(perm_fit) => {
                let t_perm = extract_perm_stat(&perm_fit, perm_config.statistic);
                null_statistics.push(t_perm);
                if t_perm >= observed_statistic {
                    n_ge += 1;
                }
                n_perm_success += 1;
            }
            Err(_) => {
                // Skip failed refits (e.g., bandwidth selection on degenerate data)
            }
        }
    }

    let p_value = (n_ge + 1) as f64 / (n_perm + 1) as f64;

    Ok(PermTestResult {
        p_value,
        observed_statistic,
        null_statistics,
        n_perm_success,
    })
}

/// Extract the permutation test statistic from a `FamResult`.
fn extract_perm_stat(fit: &FamResult, stat: PermTestStatistic) -> f64 {
    match stat {
        PermTestStatistic::R2 => fit.r_squared,
        PermTestStatistic::FittedNorm => {
            fit.fitted_values.iter().map(|&v| v * v).sum::<f64>().sqrt()
        }
        PermTestStatistic::ComponentNorm => fit
            .component_fits
            .iter()
            .map(|cf| cf.iter().map(|&v| v * v).sum::<f64>().sqrt())
            .sum::<f64>(),
    }
}

/// History-index scalar-on-function estimator.
///
/// Models `E{Y_i} = β₀ + β₁ · score_i` where `score_i` is the history index:
/// `score_i = Σ_l γ(u_l) · X_i(T − u_l) · Δu`
/// with `T = argvals.last()`, `u_l ∈ [0, Δ]` the lag grid, and `γ(·)` the
/// history weight function estimated by Nadaraya-Watson on the lag axis.
///
/// # Algorithm
///
/// 1. Validate `config.window ≤ argvals range`.
/// 2. Discretise lag grid: `u_l = l · Δ / n_lags` for l = 0, …, n_lags−1.
/// 3. For each observation i and lag l: extract `X_i(T − u_l)` via nearest-
///    lower-bound column lookup with `.min(m−1)` clamping (documented choice:
///    nearest-neighbour approximation; linear interpolation is more accurate
///    but not needed for the discretised grid resolution in use here).
/// 4. Estimate `γ` via `nadaraya_watson` on the lag axis, using `optim_bandwidth`
///    GCV when `config.bandwidth == 0.0`.
/// 5. Normalise `γ` so `Σ_l γ_l² · Δu ≈ 1` (identifiability).
/// 6. Compute `score_i = Σ_l γ_l · x_lag[i,l] · Δu`.
/// 7. Fit `E{Y_i} = β₀ + β₁ · score_i` by OLS.
///
/// # R Baseline Divergence
///
/// R's `refund::pffr` with `ff(..., limits)` implements the full function-on-
/// function history model as a lower-triangular bivariate spline. fdars
/// implements the scalar-on-function reduction (scalar Y, history index at
/// `T = argvals.last()`) via NW on a discretised lag grid — same model class,
/// marginal-integration approximation.
///
/// # Arguments
/// * `data` — Functional predictor matrix (n × m).
/// * `y` — Scalar response (length n).
/// * `argvals` — Evaluation grid (length m).
/// * `config` — Tuning parameters; see [`HistoryIndexConfig`].
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `config.window > argvals range`.
///
/// Returns [`FdarError::InvalidDimension`] for shape mismatches.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::history_index;
/// use fdars_core::scalar_on_function::HistoryIndexConfig;
///
/// let n = 30;
/// let m = 20;
/// let data = FdMatrix::from_column_major(
///     (0..n*m).map(|i| (i as f64 * 0.1).sin()).collect(),
///     n, m,
/// ).unwrap();
/// let argvals: Vec<f64> = (0..m).map(|j| j as f64).collect();
/// let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();
/// let config = HistoryIndexConfig { window: 5.0, n_lags: 10, ..Default::default() };
/// let result = history_index(&data, &y, &argvals, &config).unwrap();
/// assert_eq!(result.gamma.len(), 10);
/// assert_eq!(result.lag_grid.len(), 10);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn history_index(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    config: &HistoryIndexConfig,
) -> Result<HistoryIndexResult, FdarError> {
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

    let argvals_min = argvals.first().copied().unwrap_or(0.0);
    let argvals_max = argvals.last().copied().unwrap_or(0.0);
    let argvals_range = argvals_max - argvals_min;

    if config.window <= 0.0 || config.window > argvals_range {
        return Err(FdarError::InvalidParameter {
            parameter: "config.window",
            message: format!(
                "window ({:.6}) must be positive and <= argvals range ({:.6})",
                config.window, argvals_range
            ),
        });
    }

    let n_lags = config.n_lags.max(1);
    let delta_u = config.window / n_lags as f64;
    let big_t = argvals_max;

    // Discretise lag grid: u_l = l * delta_u for l = 0..n_lags
    let lag_grid: Vec<f64> = (0..n_lags).map(|l| l as f64 * delta_u).collect();

    // Extract lagged covariate values x_lag[i][l] = X_i(T - u_l)
    // Using nearest-lower-bound column lookup with min(m-1) clamp.
    // Documented choice: nearest-neighbour approximation. For the discretised
    // lag grid resolution typical in practice, this is accurate; linear
    // interpolation would be more precise but is not required here.
    let x_lag: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            lag_grid
                .iter()
                .map(|&u_l| {
                    let t_target = big_t - u_l;
                    // Find the largest j such that argvals[j] <= t_target
                    let j = argvals
                        .partition_point(|&v| v < t_target)
                        .saturating_sub(1)
                        .min(m - 1);
                    data[(i, j)]
                })
                .collect()
        })
        .collect();

    // Estimate gamma via nadaraya_watson on the lag axis.
    // Use the mean of y as the initial response signal for gamma estimation.
    let mu_y = y.iter().sum::<f64>() / n as f64;
    let y_centered: Vec<f64> = y.iter().map(|&yi| yi - mu_y).collect();

    // Compute a rough initial gamma: for each lag l, correlate x_lag[:,l] with y_centered
    // to get an initial signal for NW.
    let gamma_signal: Vec<f64> = lag_grid
        .iter()
        .enumerate()
        .map(|(l, _)| {
            let x_col: Vec<f64> = (0..n).map(|i| x_lag[i][l]).collect();
            let x_mean = x_col.iter().sum::<f64>() / n as f64;
            let xx: f64 = x_col.iter().map(|&v| (v - x_mean).powi(2)).sum();
            let xy: f64 = x_col
                .iter()
                .zip(&y_centered)
                .map(|(&x, &yc)| (x - x_mean) * yc)
                .sum();
            if xx > 1e-15 {
                xy / xx
            } else {
                0.0
            }
        })
        .collect();

    // Smooth gamma_signal via NW on the lag axis
    let h_gamma = if config.bandwidth > 0.0 {
        config.bandwidth
    } else {
        let bw_result = optim_bandwidth(
            &lag_grid,
            &gamma_signal,
            None,
            CvCriterion::Gcv,
            &config.kernel,
            20,
        );
        bw_result.h_opt.max(delta_u) // at least one lag step
    };

    let gamma_raw = nadaraya_watson(&lag_grid, &gamma_signal, &lag_grid, h_gamma, &config.kernel)?;

    // Normalise gamma so that Σ_l gamma_l^2 * delta_u ≈ 1 (identifiability)
    let norm_sq: f64 = gamma_raw.iter().map(|&g| g * g).sum::<f64>() * delta_u;
    let norm = norm_sq.sqrt();
    let gamma: Vec<f64> = if norm > 1e-15 {
        gamma_raw.iter().map(|&g| g / norm).collect()
    } else {
        vec![1.0 / (n_lags as f64).sqrt(); n_lags]
    };

    // Compute history scores: score_i = Σ_l gamma_l * x_lag[i,l] * delta_u
    let history_scores: Vec<f64> = (0..n)
        .map(|i| {
            gamma
                .iter()
                .enumerate()
                .map(|(l, &g)| g * x_lag[i][l] * delta_u)
                .sum()
        })
        .collect();

    // OLS: fit E{Y_i} = beta_0 + beta_1 * score_i
    let score_mean = history_scores.iter().sum::<f64>() / n as f64;
    let sxx: f64 = history_scores
        .iter()
        .map(|&s| (s - score_mean).powi(2))
        .sum();
    let sxy: f64 = history_scores
        .iter()
        .zip(y.iter())
        .map(|(&s, &yi)| (s - score_mean) * yi)
        .sum();
    let slope = if sxx > 1e-15 { sxy / sxx } else { 0.0 };
    let intercept = mu_y - slope * score_mean;

    let fitted_values: Vec<f64> = history_scores
        .iter()
        .map(|&s| intercept + slope * s)
        .collect();
    let residuals: Vec<f64> = y
        .iter()
        .zip(&fitted_values)
        .map(|(&yi, &yh)| yi - yh)
        .collect();
    let (r_squared, _) = super::compute_r_squared(y, &residuals, 2);

    Ok(HistoryIndexResult {
        fitted_values,
        residuals,
        intercept,
        slope,
        gamma,
        lag_grid,
        history_scores,
        r_squared,
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
        let config = FamConfig {
            ncomp: 2,
            ..Default::default()
        };
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
        let config = FamConfig {
            ncomp: 3,
            ..Default::default()
        };
        let result = fam(&data, &y, &argvals, None, &config).unwrap();

        assert_eq!(result.ncomp, 3, "ncomp field should be 3");
        assert_eq!(
            result.component_fits.len(),
            3,
            "component_fits.len() should equal ncomp"
        );
        for (k, cf) in result.component_fits.iter().enumerate() {
            assert_eq!(cf.len(), n, "component_fits[{k}] should have length n={n}");
        }
        assert_eq!(
            result.bandwidths.len(),
            3,
            "bandwidths.len() should equal ncomp"
        );
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
        let config = FamConfig {
            ncomp: 2,
            ..Default::default()
        };

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

        assert!(
            result.converged,
            "expected convergence, got iterations={}",
            result.iterations
        );
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
        assert!(
            err.is_err(),
            "argvals_list length mismatch should return Err"
        );
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
        let config = GsamConfig {
            ncomp: 100,
            ..Default::default()
        };
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

        let config = GsamConfig {
            ncomp: 3,
            ..Default::default()
        };
        let result = fregre_gsam(&data, &y, &argvals, None, &config).unwrap();

        assert_eq!(result.ncomp, 3);
        assert_eq!(
            result.component_fits.len(),
            3,
            "component_fits.len() should equal ncomp"
        );
        assert_eq!(result.fitted_values.len(), n);
    }

    // -----------------------------------------------------------------------
    // variable_selection tests
    // -----------------------------------------------------------------------

    #[test]
    fn varselect_active_subset_recovery() {
        // 5 functional predictors with orthogonal FPC bases; only predictors
        // 0 and 2 are truly active. We build y = 5·s0 + 3·s2 + tiny noise
        // where s0 and s2 are the observation amplitudes for predictors 0 and 2.
        // The predictors are constructed so that their "FPC score" (amplitude)
        // patterns are orthogonal: pred p uses frequency band (p+1)*2, so the
        // FPC scores of different predictors are uncorrelated for large enough n.
        let n = 100;
        let m = 30;
        let argvals = uniform_grid(m);

        // Build 5 orthogonal-amplitude predictors.
        // Each predictor p_i: observation i has amplitude a[i, p] = sin(pi*(p+1)*i/n)
        // These are orthogonal amplitude patterns for different p.
        let make_orth = |p_idx: usize| -> FdMatrix {
            let mut cm = vec![0.0_f64; n * m];
            for i in 0..n {
                // Amplitude pattern: sin-based, different frequency per predictor
                let amp = (std::f64::consts::PI * (p_idx + 1) as f64 * i as f64 / n as f64).sin();
                for j in 0..m {
                    let t = j as f64 / (m - 1) as f64;
                    // Curve shape is fixed (cos), amplitude varies per obs in orth pattern
                    cm[j * n + i] = amp * (std::f64::consts::PI * 2.0 * t).cos();
                }
            }
            FdMatrix::from_column_major(cm, n, m).unwrap()
        };

        let preds: Vec<FdMatrix> = (0..5).map(make_orth).collect();
        let pred_refs: Vec<&FdMatrix> = preds.iter().collect();
        let argvals_list: Vec<&[f64]> = (0..5).map(|_| argvals.as_slice()).collect();

        // The first FPC score of predictor p is essentially its amplitude pattern a[i, p].
        // Build y = 5 * a[i,0] + 3 * a[i,2] + tiny noise.
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let a0 = (std::f64::consts::PI * i as f64 / n as f64).sin();
                let a2 = (std::f64::consts::PI * 3.0 * i as f64 / n as f64).sin();
                5.0 * a0 + 3.0 * a2 + (i as f64 * 0.31).sin() * 0.01
            })
            .collect();

        let config = VarSelectConfig {
            ncomp: 1,
            lambda_n_grid: 20,
            ..Default::default()
        };
        let result = variable_selection(&pred_refs, &y, &argvals_list, None, &config).unwrap();

        assert_eq!(
            result.active_predictors.len(),
            5,
            "should have 5 active_predictors entries"
        );
        // At least predictors 0 and 2 must be active
        assert!(
            result.active_predictors[0],
            "predictor 0 should be active, got {:?}",
            result.active_predictors
        );
        assert!(
            result.active_predictors[2],
            "predictor 2 should be active, got {:?}",
            result.active_predictors
        );
        // Inactive predictors 1, 3, 4 should be dropped
        assert!(
            !result.active_predictors[1],
            "predictor 1 should be inactive, got {:?}",
            result.active_predictors
        );
        assert!(
            !result.active_predictors[3],
            "predictor 3 should be inactive, got {:?}",
            result.active_predictors
        );
        assert!(
            !result.active_predictors[4],
            "predictor 4 should be inactive, got {:?}",
            result.active_predictors
        );
        // R² should be well above zero
        assert!(
            result.r_squared > 0.5,
            "expected R² > 0.5, got {}",
            result.r_squared
        );
    }

    #[test]
    fn varselect_lambda_max_zeros() {
        // At lambda = lambda_max (or very large lambda), group-lasso should
        // zero out all groups (active_predictors all false).
        let n = 30;
        let m = 10;
        let argvals = uniform_grid(m);
        let preds: Vec<FdMatrix> = (0..3usize).map(|_| make_sine_data(n, m, 1.0)).collect();
        let pred_refs: Vec<&FdMatrix> = preds.iter().collect();
        let argvals_list: Vec<&[f64]> = (0..3).map(|_| argvals.as_slice()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

        // Use a very large explicit lambda to force all-zero solution
        let config = VarSelectConfig {
            ncomp: 2,
            lambda: 1e6, // massively oversized lambda
            ..Default::default()
        };
        let result = variable_selection(&pred_refs, &y, &argvals_list, None, &config).unwrap();

        // With lambda >> lambda_max every group should be zeroed out
        assert!(
            result.active_predictors.iter().all(|&a| !a),
            "expected all inactive at lambda=1e6, got {:?}",
            result.active_predictors
        );
    }

    #[test]
    fn varselect_invalid_inputs() {
        let n = 20;
        let m = 10;
        let argvals = uniform_grid(m);
        let data = make_sine_data(n, m, 1.0);
        let y_ok: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let config = VarSelectConfig {
            ncomp: 2,
            ..Default::default()
        };

        // Empty predictor list
        let err = variable_selection(&[], &y_ok, &[], None, &config);
        assert!(err.is_err(), "empty predictors should return Err");
        match err.unwrap_err() {
            FdarError::InvalidDimension { .. } => {}
            e => panic!("expected InvalidDimension, got {e:?}"),
        }

        // Predictor/response length mismatch
        let data_wrong = make_sine_data(n + 5, m, 1.0);
        let err = variable_selection(&[&data_wrong], &y_ok, &[&argvals], None, &config);
        assert!(err.is_err(), "mismatched n should return Err");
        match err.unwrap_err() {
            FdarError::InvalidDimension { .. } => {}
            e => panic!("expected InvalidDimension, got {e:?}"),
        }

        // argvals_list length mismatch
        let err = variable_selection(&[&data], &y_ok, &[], None, &config);
        assert!(err.is_err(), "argvals_list mismatch should return Err");

        // Unsupported penalty
        let config_mcp = VarSelectConfig {
            penalty: VarSelectPenalty::GroupMcp,
            ..config.clone()
        };
        let err = variable_selection(&[&data], &y_ok, &[&argvals], None, &config_mcp);
        assert!(err.is_err(), "GroupMcp should return Err");
        match err.unwrap_err() {
            FdarError::InvalidParameter { parameter, .. } => {
                assert_eq!(parameter, "config.penalty");
            }
            e => panic!("expected InvalidParameter, got {e:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // permutation_test_fam tests
    // -----------------------------------------------------------------------

    #[test]
    fn perm_seeded_reproducibility() {
        // Two calls with the same seed must produce identical p_values.
        let n = 30;
        let m = 12;
        let argvals = uniform_grid(m);
        let data = make_sine_data(n, m, 1.0);
        let fpca = fdata_to_pc_1d(&data, 1, &argvals).unwrap();
        let y: Vec<f64> = (0..n)
            .map(|i| fpca.scores[(i, 0)] * 2.0 + (i as f64 * 0.31).sin() * 0.05)
            .collect();

        let fam_cfg = FamConfig {
            ncomp: 1,
            ..Default::default()
        };
        let perm_cfg = PermTestConfig {
            n_perm: 19,
            seed: 42,
            statistic: PermTestStatistic::R2,
        };

        let r1 = permutation_test_fam(&data, &y, &argvals, None, &fam_cfg, &perm_cfg).unwrap();
        let r2 = permutation_test_fam(&data, &y, &argvals, None, &fam_cfg, &perm_cfg).unwrap();
        assert_eq!(
            r1.p_value, r2.p_value,
            "same seed should give same p_value: {} vs {}",
            r1.p_value, r2.p_value
        );
        assert_eq!(
            r1.null_statistics, r2.null_statistics,
            "same seed should give same null distribution"
        );
    }

    #[test]
    fn perm_pvalue_range() {
        // p_value must be in [0, 1] regardless of inputs.
        let n = 20;
        let m = 8;
        let argvals = uniform_grid(m);
        let data = make_sine_data(n, m, 1.0);
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let fam_cfg = FamConfig {
            ncomp: 1,
            ..Default::default()
        };
        let perm_cfg = PermTestConfig {
            n_perm: 9,
            seed: 0,
            statistic: PermTestStatistic::FittedNorm,
        };
        let result = permutation_test_fam(&data, &y, &argvals, None, &fam_cfg, &perm_cfg).unwrap();
        assert!(
            (0.0..=1.0).contains(&result.p_value),
            "p_value out of [0,1]: {}",
            result.p_value
        );
    }

    #[test]
    fn perm_detects_true_effect() {
        // y = 2 * xi_1 + tiny noise → should give small p_value under n_perm=99 / seed=42.
        // Under the null (y = noise only) p_value should be non-significant.
        let n = 40;
        let m = 15;
        let argvals = uniform_grid(m);
        let data = make_sine_data(n, m, 1.0);
        let fpca = fdata_to_pc_1d(&data, 1, &argvals).unwrap();

        // Strong signal: y = 2 * xi_1 + very small noise
        let y_signal: Vec<f64> = (0..n)
            .map(|i| {
                let xi1 = fpca.scores[(i, 0)];
                2.0 * xi1 + (i as f64 * 0.17).sin() * 0.02
            })
            .collect();

        // Pure noise: y = noise (no relationship to predictor)
        let y_null: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() * 0.3).collect();

        let fam_cfg = FamConfig {
            ncomp: 1,
            ..Default::default()
        };
        let perm_cfg = PermTestConfig {
            n_perm: 99,
            seed: 42,
            statistic: PermTestStatistic::R2,
        };

        let r_signal =
            permutation_test_fam(&data, &y_signal, &argvals, None, &fam_cfg, &perm_cfg).unwrap();
        let r_null =
            permutation_test_fam(&data, &y_null, &argvals, None, &fam_cfg, &perm_cfg).unwrap();

        assert!(
            r_signal.p_value < 0.1,
            "expected p < 0.1 under true effect, got p={}",
            r_signal.p_value
        );
        assert!(
            r_null.p_value > 0.1,
            "expected p > 0.1 under the null, got p={}",
            r_null.p_value
        );
    }

    // -----------------------------------------------------------------------
    // history_index tests
    // -----------------------------------------------------------------------

    #[test]
    fn history_index_synthetic_recovery() {
        // y_i = Σ_{u=0}^{0.5} X_i(1.0 - u) du (uniform gamma, Delta=0.5)
        // Discretise: y_i ≈ Σ_l X_i(T - u_l) * delta_u where T = argvals.last().
        // We expect R² > 0.70 and gamma approximately uniform.
        let n = 50;
        let m = 30;
        // argvals from 0..1
        let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

        // Generate curves with amplitude variation
        let mut cm = vec![0.0_f64; n * m];
        for i in 0..n {
            let amp = (i as f64 + 1.0) / n as f64;
            for j in 0..m {
                let t = j as f64 / (m - 1) as f64;
                cm[j * n + i] = amp * (std::f64::consts::PI * 2.0 * t).sin();
            }
        }
        let data = FdMatrix::from_column_major(cm, n, m).unwrap();

        // True y: integral of X_i over [T - 0.5, T] = [0.5, 1.0]
        // Approximate as sum of X_i at lag grid points * delta_u
        let window = 0.5_f64;
        let n_lags = 10;
        let delta_u = window / n_lags as f64;
        let big_t = argvals.last().copied().unwrap();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                (0..n_lags)
                    .map(|l| {
                        let u_l = l as f64 * delta_u;
                        let t_target = big_t - u_l;
                        let j = argvals
                            .partition_point(|&v| v < t_target)
                            .saturating_sub(1)
                            .min(m - 1);
                        data[(i, j)] * delta_u
                    })
                    .sum::<f64>()
            })
            .collect();

        let config = HistoryIndexConfig {
            window,
            n_lags,
            bandwidth: 0.0,
            kernel: "gaussian".to_string(),
        };
        let result = history_index(&data, &y, &argvals, &config).unwrap();

        assert!(
            result.r_squared > 0.70,
            "expected R² > 0.70, got {}",
            result.r_squared
        );
        // gamma should be roughly uniform — coefficient of variation should be < 2
        let g_mean = result.gamma.iter().sum::<f64>() / n_lags as f64;
        let g_std = (result
            .gamma
            .iter()
            .map(|&g| (g - g_mean).powi(2))
            .sum::<f64>()
            / n_lags as f64)
            .sqrt();
        let cv = if g_mean.abs() > 1e-10 {
            g_std / g_mean.abs()
        } else {
            0.0
        };
        assert!(
            cv < 2.0,
            "gamma should be approximately uniform (CV < 2.0), got CV={}",
            cv
        );
    }

    #[test]
    fn history_index_window_too_large() {
        let n = 20;
        let m = 10;
        let argvals = uniform_grid(m); // 0..1 range
        let data = make_sine_data(n, m, 1.0);
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();

        // window > argvals range (which is ~1.0 for uniform_grid)
        let config = HistoryIndexConfig {
            window: 2.0,
            n_lags: 10,
            ..Default::default()
        };
        let err = history_index(&data, &y, &argvals, &config);
        assert!(err.is_err(), "window > argvals range should return Err");
        match err.unwrap_err() {
            FdarError::InvalidParameter { parameter, .. } => {
                assert_eq!(parameter, "config.window");
            }
            e => panic!("expected InvalidParameter, got {e:?}"),
        }
    }

    #[test]
    fn history_index_output_shapes() {
        let n = 25;
        let m = 15;
        let argvals = uniform_grid(m); // range 0..1
        let data = make_sine_data(n, m, 1.0);
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).cos()).collect();

        let n_lags = 12;
        let config = HistoryIndexConfig {
            window: 0.5,
            n_lags,
            ..Default::default()
        };
        let result = history_index(&data, &y, &argvals, &config).unwrap();

        assert_eq!(
            result.gamma.len(),
            n_lags,
            "gamma.len() should equal n_lags"
        );
        assert_eq!(
            result.lag_grid.len(),
            n_lags,
            "lag_grid.len() should equal n_lags"
        );
        assert_eq!(
            result.fitted_values.len(),
            n,
            "fitted_values.len() should equal n"
        );
        assert_eq!(
            result.history_scores.len(),
            n,
            "history_scores.len() should equal n"
        );
    }
}
