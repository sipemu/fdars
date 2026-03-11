//! Explainability toolkit for FPC-based scalar-on-function models.
//!
//! - [`functional_pdp`] / [`functional_pdp_logistic`] — PDP/ICE
//! - [`beta_decomposition`] / [`beta_decomposition_logistic`] — per-FPC β(t) decomposition
//! - [`significant_regions`] / [`significant_regions_from_se`] — CI-based significant intervals
//! - [`fpc_permutation_importance`] / [`fpc_permutation_importance_logistic`] — permutation importance
//! - [`influence_diagnostics`] — Cook's distance and leverage
//! - [`friedman_h_statistic`] / [`friedman_h_statistic_logistic`] — FPC interaction detection
//! - [`pointwise_importance`] / [`pointwise_importance_logistic`] — pointwise variable importance
//! - [`fpc_vif`] / [`fpc_vif_logistic`] — variance inflation factors
//! - [`fpc_shap_values`] / [`fpc_shap_values_logistic`] — SHAP values
//! - [`dfbetas_dffits`] — DFBETAS and DFFITS influence diagnostics
//! - [`prediction_intervals`] — prediction intervals for new observations
//! - [`fpc_ale`] / [`fpc_ale_logistic`] — accumulated local effects
//! - [`loo_cv_press`] — LOO-CV / PRESS diagnostics
//! - [`sobol_indices`] / [`sobol_indices_logistic`] — Sobol sensitivity indices
//! - [`calibration_diagnostics`] — calibration diagnostics (logistic)
//! - [`functional_saliency`] / [`functional_saliency_logistic`] — functional saliency maps
//! - [`domain_selection`] / [`domain_selection_logistic`] — domain/interval importance
//! - [`conditional_permutation_importance`] / [`conditional_permutation_importance_logistic`]
//! - [`counterfactual_regression`] / [`counterfactual_logistic`] — counterfactual explanations
//! - [`prototype_criticism`] — MMD-based prototype/criticism selection
//! - [`lime_explanation`] / [`lime_explanation_logistic`] — LIME local surrogates
//! - [`expected_calibration_error`] — ECE, MCE, ACE calibration metrics
//! - [`conformal_prediction_residuals`] — split-conformal prediction intervals
//! - [`regression_depth`] / [`regression_depth_logistic`] — depth-based regression diagnostics
//! - [`explanation_stability`] / [`explanation_stability_logistic`] — bootstrap stability analysis
//! - [`anchor_explanation`] / [`anchor_explanation_logistic`] — beam-search anchor rules

use crate::depth;
use crate::matrix::FdMatrix;
use crate::regression::FpcaResult;
use crate::scalar_on_function::{
    build_design_matrix, cholesky_factor, cholesky_forward_back, compute_hat_diagonal, compute_xtx,
    fregre_lm, functional_logistic, sigmoid, FregreLmResult, FunctionalLogisticResult,
};
use rand::prelude::*;
use rand_distr::Normal;

/// Result of a functional partial dependence plot.
pub struct FunctionalPdpResult {
    /// FPC score grid values (length n_grid).
    pub grid_values: Vec<f64>,
    /// Average prediction across observations at each grid point (length n_grid).
    pub pdp_curve: Vec<f64>,
    /// Individual conditional expectation curves (n × n_grid).
    pub ice_curves: FdMatrix,
    /// Which FPC component was varied.
    pub component: usize,
}

/// Functional PDP/ICE for a linear functional regression model.
///
/// Varies the FPC score for `component` across a grid while keeping other scores
/// fixed, producing ICE curves and their average (PDP).
///
/// For a linear model, ICE curves are parallel lines (same slope, different intercepts).
///
/// # Arguments
/// * `fit` — A fitted [`FregreLmResult`]
/// * `data` — Original functional predictor matrix (n × m)
/// * `scalar_covariates` — Optional scalar covariates (n × p)
/// * `component` — Which FPC component to vary (0-indexed, must be < fit.ncomp)
/// * `n_grid` — Number of grid points (must be ≥ 2)
pub fn functional_pdp(
    fit: &FregreLmResult,
    data: &FdMatrix,
    _scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_grid: usize,
) -> Option<FunctionalPdpResult> {
    let (n, m) = data.shape();
    if component >= fit.ncomp
        || n_grid < 2
        || n == 0
        || m != fit.fpca.mean.len()
        || n != fit.fitted_values.len()
    {
        return None;
    }

    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let grid_values = make_grid(&scores, component, n_grid);

    let coef_c = fit.coefficients[1 + component];
    let mut ice_curves = FdMatrix::zeros(n, n_grid);
    for i in 0..n {
        let base = fit.fitted_values[i] - coef_c * scores[(i, component)];
        for g in 0..n_grid {
            ice_curves[(i, g)] = base + coef_c * grid_values[g];
        }
    }

    let pdp_curve = ice_to_pdp(&ice_curves, n, n_grid);

    Some(FunctionalPdpResult {
        grid_values,
        pdp_curve,
        ice_curves,
        component,
    })
}

/// Functional PDP/ICE for a functional logistic regression model.
///
/// Predictions pass through sigmoid, so ICE curves are non-parallel.
///
/// # Arguments
/// * `fit` — A fitted [`FunctionalLogisticResult`]
/// * `data` — Original functional predictor matrix (n × m)
/// * `scalar_covariates` — Optional scalar covariates (n × p)
/// * `component` — Which FPC component to vary (0-indexed, must be < fit.ncomp)
/// * `n_grid` — Number of grid points (must be ≥ 2)
pub fn functional_pdp_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_grid: usize,
) -> Option<FunctionalPdpResult> {
    let (n, m) = data.shape();
    if component >= fit.ncomp || n_grid < 2 || n == 0 || m != fit.fpca.mean.len() {
        return None;
    }

    let ncomp = fit.ncomp;
    let p_scalar = fit.gamma.len();
    if p_scalar > 0 && scalar_covariates.is_none() {
        return None;
    }

    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let grid_values = make_grid(&scores, component, n_grid);

    let mut ice_curves = FdMatrix::zeros(n, n_grid);
    let coef_c = fit.coefficients[1 + component];
    for i in 0..n {
        let eta_base = logistic_eta_base(
            fit.intercept,
            &fit.coefficients,
            &fit.gamma,
            &scores,
            scalar_covariates,
            i,
            ncomp,
            component,
        );
        for g in 0..n_grid {
            ice_curves[(i, g)] = sigmoid(eta_base + coef_c * grid_values[g]);
        }
    }

    let pdp_curve = ice_to_pdp(&ice_curves, n, n_grid);

    Some(FunctionalPdpResult {
        grid_values,
        pdp_curve,
        ice_curves,
        component,
    })
}

// ---------------------------------------------------------------------------
// Helper: project data → FPC scores
// ---------------------------------------------------------------------------

pub(crate) fn project_scores(
    data: &FdMatrix,
    mean: &[f64],
    rotation: &FdMatrix,
    ncomp: usize,
) -> FdMatrix {
    let (n, m) = data.shape();
    let mut scores = FdMatrix::zeros(n, ncomp);
    for i in 0..n {
        for k in 0..ncomp {
            let mut s = 0.0;
            for j in 0..m {
                s += (data[(i, j)] - mean[j]) * rotation[(j, k)];
            }
            scores[(i, k)] = s;
        }
    }
    scores
}

/// Subsample rows from an FdMatrix.
pub(crate) fn subsample_rows(data: &FdMatrix, indices: &[usize]) -> FdMatrix {
    let ncols = data.ncols();
    let mut out = FdMatrix::zeros(indices.len(), ncols);
    for (new_i, &orig_i) in indices.iter().enumerate() {
        for j in 0..ncols {
            out[(new_i, j)] = data[(orig_i, j)];
        }
    }
    out
}

/// Quantile of a pre-sorted slice using linear interpolation.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = q * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let lo = lo.min(n - 1);
    let hi = hi.min(n - 1);
    if lo == hi {
        sorted[lo]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Compute average ranks of a slice (1-based, average ranks for ties).
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (values[idx[j]] - values[idx[i]]).abs() < 1e-15 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based average
        for k in i..j {
            ranks[idx[k]] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Spearman rank correlation between two equal-length slices.
fn spearman_rank_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ra = compute_ranks(a);
    let rb = compute_ranks(b);
    let mean_a: f64 = ra.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = rb.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut da2 = 0.0;
    let mut db2 = 0.0;
    for i in 0..n {
        let da = ra[i] - mean_a;
        let db = rb[i] - mean_b;
        num += da * db;
        da2 += da * da;
        db2 += db * db;
    }
    let denom = (da2 * db2).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        num / denom
    }
}

/// Predict from FPC scores + scalar covariates using linear model coefficients.
fn predict_from_scores(
    scores: &FdMatrix,
    coefficients: &[f64],
    gamma: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Vec<f64> {
    let n = scores.nrows();
    let mut preds = vec![0.0; n];
    for i in 0..n {
        let mut yhat = coefficients[0];
        for k in 0..ncomp {
            yhat += coefficients[1 + k] * scores[(i, k)];
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..gamma.len() {
                yhat += gamma[j] * sc[(i, j)];
            }
        }
        preds[i] = yhat;
    }
    preds
}

/// Sample standard deviation of a slice.
fn sample_std(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let var = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

/// Mean pairwise Spearman rank correlation across a set of vectors.
fn mean_pairwise_spearman(vectors: &[Vec<f64>]) -> f64 {
    let n = vectors.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += spearman_rank_correlation(&vectors[i], &vectors[j]);
            count += 1;
        }
    }
    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

/// Compute pointwise mean, std, and coefficient of variation from bootstrap samples.
fn pointwise_mean_std_cv(samples: &[Vec<f64>], length: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = samples.len();
    let mut mean = vec![0.0; length];
    let mut std = vec![0.0; length];
    for j in 0..length {
        let vals: Vec<f64> = samples.iter().map(|s| s[j]).collect();
        mean[j] = vals.iter().sum::<f64>() / n as f64;
        let var = vals.iter().map(|&v| (v - mean[j]).powi(2)).sum::<f64>() / (n - 1) as f64;
        std[j] = var.sqrt();
    }
    let eps = 1e-15;
    let cv: Vec<f64> = (0..length)
        .map(|j| {
            if mean[j].abs() > eps {
                std[j] / mean[j].abs()
            } else {
                0.0
            }
        })
        .collect();
    (mean, std, cv)
}

/// Compute per-component std from bootstrap coefficient vectors.
fn coefficient_std_from_bootstrap(all_coefs: &[Vec<f64>], ncomp: usize) -> Vec<f64> {
    (0..ncomp)
        .map(|k| {
            let vals: Vec<f64> = all_coefs.iter().map(|c| c[k]).collect();
            sample_std(&vals)
        })
        .collect()
}

/// Compute depth of scores using the specified depth type.
pub(crate) fn compute_score_depths(scores: &FdMatrix, depth_type: DepthType) -> Vec<f64> {
    match depth_type {
        DepthType::FraimanMuniz => depth::fraiman_muniz_1d(scores, scores, false),
        DepthType::ModifiedBand => depth::modified_band_1d(scores, scores),
        DepthType::FunctionalSpatial => depth::functional_spatial_1d(scores, scores, None),
    }
}

/// Compute beta depth from bootstrap coefficient vectors.
pub(crate) fn beta_depth_from_bootstrap(
    boot_coefs: &[Vec<f64>],
    orig_coefs: &[f64],
    ncomp: usize,
    depth_type: DepthType,
) -> f64 {
    if boot_coefs.len() < 2 {
        return 0.0;
    }
    let mut boot_mat = FdMatrix::zeros(boot_coefs.len(), ncomp);
    for (i, coefs) in boot_coefs.iter().enumerate() {
        for k in 0..ncomp {
            boot_mat[(i, k)] = coefs[k];
        }
    }
    let mut orig_mat = FdMatrix::zeros(1, ncomp);
    for k in 0..ncomp {
        orig_mat[(0, k)] = orig_coefs[k];
    }
    compute_single_depth(&orig_mat, &boot_mat, depth_type)
}

/// Build stability result from collected bootstrap data.
pub(crate) fn build_stability_result(
    all_beta_t: &[Vec<f64>],
    all_coefs: &[Vec<f64>],
    all_abs_coefs: &[Vec<f64>],
    all_metrics: &[f64],
    m: usize,
    ncomp: usize,
) -> Option<StabilityAnalysisResult> {
    let n_success = all_beta_t.len();
    if n_success < 2 {
        return None;
    }
    let (_mean, beta_t_std, beta_t_cv) = pointwise_mean_std_cv(all_beta_t, m);
    let coefficient_std = coefficient_std_from_bootstrap(all_coefs, ncomp);
    let metric_std = sample_std(all_metrics);
    let importance_stability = mean_pairwise_spearman(all_abs_coefs);

    Some(StabilityAnalysisResult {
        beta_t_std,
        coefficient_std,
        metric_std,
        beta_t_cv,
        importance_stability,
        n_boot_success: n_success,
    })
}

/// Compute quantile bin edges for a column of scores.
fn compute_bin_edges(scores: &FdMatrix, component: usize, n: usize, n_bins: usize) -> Vec<f64> {
    let mut vals: Vec<f64> = (0..n).map(|i| scores[(i, component)]).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut edges = Vec::with_capacity(n_bins + 1);
    edges.push(f64::NEG_INFINITY);
    for b in 1..n_bins {
        edges.push(quantile_sorted(&vals, b as f64 / n_bins as f64));
    }
    edges.push(f64::INFINITY);
    edges
}

/// Find which bin a value falls into given bin edges.
fn find_bin(value: f64, edges: &[f64], n_bins: usize) -> usize {
    for bi in 0..n_bins {
        if value >= edges[bi] && value < edges[bi + 1] {
            return bi;
        }
    }
    n_bins - 1
}

/// Compute which observations match a bin constraint on a component.
fn apply_bin_filter(
    current_matching: &[bool],
    scores: &FdMatrix,
    component: usize,
    bin: usize,
    edges: &[f64],
    n_bins: usize,
) -> Vec<bool> {
    let lo = edges[bin];
    let hi = edges[bin + 1];
    let is_last = bin == n_bins - 1;
    (0..current_matching.len())
        .map(|i| {
            current_matching[i]
                && scores[(i, component)] >= lo
                && (is_last || scores[(i, component)] < hi)
        })
        .collect()
}

/// Compute weighted calibration gap for a group of sorted indices.
fn calibration_gap_weighted(
    indices: &[usize],
    y: &[f64],
    probabilities: &[f64],
    total_n: usize,
) -> f64 {
    let cnt = indices.len();
    if cnt == 0 {
        return 0.0;
    }
    let sum_y: f64 = indices.iter().map(|&i| y[i]).sum();
    let sum_p: f64 = indices.iter().map(|&i| probabilities[i]).sum();
    let gap = (sum_y / cnt as f64 - sum_p / cnt as f64).abs();
    cnt as f64 / total_n as f64 * gap
}

/// Validate inputs for conformal prediction. Returns (n_cal, n_proper) on success.
fn validate_conformal_inputs(
    n: usize,
    m: usize,
    n_test: usize,
    m_test: usize,
    train_y_len: usize,
    ncomp: usize,
    cal_fraction: f64,
    alpha: f64,
) -> Option<(usize, usize)> {
    let shapes_ok = n >= 4 && n == train_y_len && m > 0 && n_test > 0 && m_test == m;
    let params_ok = cal_fraction > 0.0 && cal_fraction < 1.0 && alpha > 0.0 && alpha < 1.0;
    if !(shapes_ok && params_ok) {
        return None;
    }
    let n_cal = ((n as f64 * cal_fraction).round() as usize).max(2);
    let n_proper = n - n_cal;
    (n_proper >= ncomp + 2).then_some((n_cal, n_proper))
}

// ---------------------------------------------------------------------------
// Feature 3: β(t) Effect Decomposition
// ---------------------------------------------------------------------------

/// Per-FPC decomposition of the functional coefficient β(t).
pub struct BetaDecomposition {
    /// `components[k]` = coef_k × φ_k(t), each of length m.
    pub components: Vec<Vec<f64>>,
    /// FPC regression coefficients (length ncomp).
    pub coefficients: Vec<f64>,
    /// Proportion of ||β(t)||² explained by each component.
    pub variance_proportion: Vec<f64>,
}

/// Decompose β(t) = Σ_k coef_k × φ_k(t) for a linear functional regression.
pub fn beta_decomposition(fit: &FregreLmResult) -> Option<BetaDecomposition> {
    let ncomp = fit.ncomp;
    let m = fit.fpca.mean.len();
    if ncomp == 0 || m == 0 {
        return None;
    }
    decompose_beta(&fit.coefficients, &fit.fpca.rotation, ncomp, m)
}

/// Decompose β(t) for a functional logistic regression.
pub fn beta_decomposition_logistic(fit: &FunctionalLogisticResult) -> Option<BetaDecomposition> {
    let ncomp = fit.ncomp;
    let m = fit.fpca.mean.len();
    if ncomp == 0 || m == 0 {
        return None;
    }
    decompose_beta(&fit.coefficients, &fit.fpca.rotation, ncomp, m)
}

fn decompose_beta(
    coefficients: &[f64],
    rotation: &FdMatrix,
    ncomp: usize,
    m: usize,
) -> Option<BetaDecomposition> {
    let mut components = Vec::with_capacity(ncomp);
    let mut coefs = Vec::with_capacity(ncomp);
    let mut norms_sq = Vec::with_capacity(ncomp);

    for k in 0..ncomp {
        let ck = coefficients[1 + k];
        coefs.push(ck);
        let comp: Vec<f64> = (0..m).map(|j| ck * rotation[(j, k)]).collect();
        let nsq: f64 = comp.iter().map(|v| v * v).sum();
        norms_sq.push(nsq);
        components.push(comp);
    }

    let total_sq: f64 = norms_sq.iter().sum();
    let variance_proportion = if total_sq > 0.0 {
        norms_sq.iter().map(|&s| s / total_sq).collect()
    } else {
        vec![0.0; ncomp]
    };

    Some(BetaDecomposition {
        components,
        coefficients: coefs,
        variance_proportion,
    })
}

// ---------------------------------------------------------------------------
// Feature 2: Significant Regions
// ---------------------------------------------------------------------------

/// Direction of a significant region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignificanceDirection {
    Positive,
    Negative,
}

/// A contiguous interval where the confidence band excludes zero.
#[derive(Debug, Clone)]
pub struct SignificantRegion {
    /// Start index (inclusive).
    pub start_idx: usize,
    /// End index (inclusive).
    pub end_idx: usize,
    /// Direction of the effect.
    pub direction: SignificanceDirection,
}

/// Identify contiguous regions where the CI `[lower, upper]` excludes zero.
pub fn significant_regions(lower: &[f64], upper: &[f64]) -> Option<Vec<SignificantRegion>> {
    if lower.len() != upper.len() || lower.is_empty() {
        return None;
    }
    let n = lower.len();
    let mut regions = Vec::new();
    let mut i = 0;
    while i < n {
        if let Some(d) = detect_direction(lower[i], upper[i]) {
            let start = i;
            i += 1;
            while i < n && detect_direction(lower[i], upper[i]) == Some(d) {
                i += 1;
            }
            regions.push(SignificantRegion {
                start_idx: start,
                end_idx: i - 1,
                direction: d,
            });
        } else {
            i += 1;
        }
    }
    Some(regions)
}

/// Build CI from β(t) ± z × SE, then find significant regions.
pub fn significant_regions_from_se(
    beta_t: &[f64],
    beta_se: &[f64],
    z_alpha: f64,
) -> Option<Vec<SignificantRegion>> {
    if beta_t.len() != beta_se.len() || beta_t.is_empty() {
        return None;
    }
    let lower: Vec<f64> = beta_t
        .iter()
        .zip(beta_se)
        .map(|(b, s)| b - z_alpha * s)
        .collect();
    let upper: Vec<f64> = beta_t
        .iter()
        .zip(beta_se)
        .map(|(b, s)| b + z_alpha * s)
        .collect();
    significant_regions(&lower, &upper)
}

// ---------------------------------------------------------------------------
// Feature 1: FPC Permutation Importance
// ---------------------------------------------------------------------------

/// Result of FPC permutation importance.
pub struct FpcPermutationImportance {
    /// R² (or accuracy) drop per component (length ncomp).
    pub importance: Vec<f64>,
    /// Baseline metric (R² or accuracy).
    pub baseline_metric: f64,
    /// Mean metric after permuting each component.
    pub permuted_metric: Vec<f64>,
}

/// Permutation importance for a linear functional regression (metric = R²).
pub fn fpc_permutation_importance(
    fit: &FregreLmResult,
    data: &FdMatrix,
    y: &[f64],
    n_perm: usize,
    seed: u64,
) -> Option<FpcPermutationImportance> {
    let (n, m) = data.shape();
    if n == 0 || n != y.len() || m != fit.fpca.mean.len() || n_perm == 0 {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    // Baseline R² — compute from same FPC-only prediction used in permuted path
    // to ensure consistent comparison (gamma terms are constant across permutations)
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    if ss_tot == 0.0 {
        return None;
    }
    let identity_idx: Vec<usize> = (0..n).collect();
    let ss_res_base = permuted_ss_res_linear(
        &scores,
        &fit.coefficients,
        y,
        n,
        ncomp,
        ncomp,
        &identity_idx,
    );
    let baseline = 1.0 - ss_res_base / ss_tot;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut importance = vec![0.0; ncomp];
    let mut permuted_metric = vec![0.0; ncomp];

    for k in 0..ncomp {
        let mut sum_r2 = 0.0;
        for _ in 0..n_perm {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.shuffle(&mut rng);
            let ss_res_perm =
                permuted_ss_res_linear(&scores, &fit.coefficients, y, n, ncomp, k, &idx);
            sum_r2 += 1.0 - ss_res_perm / ss_tot;
        }
        let mean_perm = sum_r2 / n_perm as f64;
        permuted_metric[k] = mean_perm;
        importance[k] = baseline - mean_perm;
    }

    Some(FpcPermutationImportance {
        importance,
        baseline_metric: baseline,
        permuted_metric,
    })
}

/// Permutation importance for functional logistic regression (metric = accuracy).
pub fn fpc_permutation_importance_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    y: &[f64],
    n_perm: usize,
    seed: u64,
) -> Option<FpcPermutationImportance> {
    let (n, m) = data.shape();
    if n == 0 || n != y.len() || m != fit.fpca.mean.len() || n_perm == 0 {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let baseline: f64 = (0..n)
        .filter(|&i| {
            let pred = if fit.probabilities[i] >= 0.5 {
                1.0
            } else {
                0.0
            };
            (pred - y[i]).abs() < 1e-10
        })
        .count() as f64
        / n as f64;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut importance = vec![0.0; ncomp];
    let mut permuted_metric = vec![0.0; ncomp];

    for k in 0..ncomp {
        let mut sum_acc = 0.0;
        for _ in 0..n_perm {
            let mut perm_scores = clone_scores_matrix(&scores, n, ncomp);
            shuffle_global(&mut perm_scores, &scores, k, n, &mut rng);
            sum_acc += logistic_accuracy_from_scores(
                &perm_scores,
                fit.intercept,
                &fit.coefficients,
                y,
                n,
                ncomp,
            );
        }
        let mean_acc = sum_acc / n_perm as f64;
        permuted_metric[k] = mean_acc;
        importance[k] = baseline - mean_acc;
    }

    Some(FpcPermutationImportance {
        importance,
        baseline_metric: baseline,
        permuted_metric,
    })
}

// ---------------------------------------------------------------------------
// Feature 4: Influence Diagnostics
// ---------------------------------------------------------------------------

/// Cook's distance and leverage diagnostics for the FPC regression.
pub struct InfluenceDiagnostics {
    /// Hat matrix diagonal h_ii (length n).
    pub leverage: Vec<f64>,
    /// Cook's distance D_i (length n).
    pub cooks_distance: Vec<f64>,
    /// Number of model parameters (intercept + ncomp + p_scalar).
    pub p: usize,
    /// Residual mean squared error.
    pub mse: f64,
}

/// Compute leverage and Cook's distance for a linear functional regression.
pub fn influence_diagnostics(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
) -> Option<InfluenceDiagnostics> {
    let (n, m) = data.shape();
    if n == 0 || m != fit.fpca.mean.len() {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let design = build_design_matrix(&scores, ncomp, scalar_covariates, n);
    let p = design.ncols();

    if n <= p {
        return None;
    }

    let xtx = compute_xtx(&design);
    let l = cholesky_factor(&xtx, p)?;
    let leverage = compute_hat_diagonal(&design, &l);

    let ss_res: f64 = fit.residuals.iter().map(|r| r * r).sum();
    let mse = ss_res / (n - p) as f64;

    let mut cooks_distance = vec![0.0; n];
    for i in 0..n {
        let e = fit.residuals[i];
        let h = leverage[i];
        let denom = p as f64 * mse * (1.0 - h).powi(2);
        cooks_distance[i] = if denom > 0.0 { e * e * h / denom } else { 0.0 };
    }

    Some(InfluenceDiagnostics {
        leverage,
        cooks_distance,
        p,
        mse,
    })
}

// ---------------------------------------------------------------------------
// Feature 5: Friedman H-statistic
// ---------------------------------------------------------------------------

/// Result of the Friedman H-statistic for interaction between two FPC components.
pub struct FriedmanHResult {
    /// First component index.
    pub component_j: usize,
    /// Second component index.
    pub component_k: usize,
    /// Interaction strength H².
    pub h_squared: f64,
    /// Grid values for component j.
    pub grid_j: Vec<f64>,
    /// Grid values for component k.
    pub grid_k: Vec<f64>,
    /// 2D partial dependence surface (n_grid × n_grid).
    pub pdp_2d: FdMatrix,
}

/// Compute the grid for a single FPC score column.
pub(crate) fn make_grid(scores: &FdMatrix, component: usize, n_grid: usize) -> Vec<f64> {
    let n = scores.nrows();
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for i in 0..n {
        let v = scores[(i, component)];
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    if (mx - mn).abs() < 1e-15 {
        mx = mn + 1.0;
    }
    (0..n_grid)
        .map(|g| mn + (mx - mn) * g as f64 / (n_grid - 1) as f64)
        .collect()
}

/// Friedman H-statistic for interaction between two FPC components (linear model).
pub fn friedman_h_statistic(
    fit: &FregreLmResult,
    data: &FdMatrix,
    component_j: usize,
    component_k: usize,
    n_grid: usize,
) -> Option<FriedmanHResult> {
    if component_j == component_k {
        return None;
    }
    let (n, m) = data.shape();
    if n == 0 || m != fit.fpca.mean.len() || n_grid < 2 {
        return None;
    }
    if component_j >= fit.ncomp || component_k >= fit.ncomp {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let grid_j = make_grid(&scores, component_j, n_grid);
    let grid_k = make_grid(&scores, component_k, n_grid);
    let coefs = &fit.coefficients;

    let pdp_j = pdp_1d_linear(&scores, coefs, ncomp, component_j, &grid_j, n);
    let pdp_k = pdp_1d_linear(&scores, coefs, ncomp, component_k, &grid_k, n);
    let pdp_2d = pdp_2d_linear(
        &scores,
        coefs,
        ncomp,
        component_j,
        component_k,
        &grid_j,
        &grid_k,
        n,
        n_grid,
    );

    let f_bar: f64 = fit.fitted_values.iter().sum::<f64>() / n as f64;
    let h_squared = compute_h_squared(&pdp_2d, &pdp_j, &pdp_k, f_bar, n_grid);

    Some(FriedmanHResult {
        component_j,
        component_k,
        h_squared,
        grid_j,
        grid_k,
        pdp_2d,
    })
}

/// Friedman H-statistic for interaction between two FPC components (logistic model).
pub fn friedman_h_statistic_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component_j: usize,
    component_k: usize,
    n_grid: usize,
) -> Option<FriedmanHResult> {
    let (n, m) = data.shape();
    let ncomp = fit.ncomp;
    let p_scalar = fit.gamma.len();
    if component_j == component_k
        || n == 0
        || m != fit.fpca.mean.len()
        || n_grid < 2
        || component_j >= ncomp
        || component_k >= ncomp
        || (p_scalar > 0 && scalar_covariates.is_none())
    {
        return None;
    }
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let grid_j = make_grid(&scores, component_j, n_grid);
    let grid_k = make_grid(&scores, component_k, n_grid);

    let pm = |replacements: &[(usize, f64)]| {
        logistic_pdp_mean(
            &scores,
            fit.intercept,
            &fit.coefficients,
            &fit.gamma,
            scalar_covariates,
            n,
            ncomp,
            replacements,
        )
    };

    let pdp_j: Vec<f64> = grid_j.iter().map(|&gj| pm(&[(component_j, gj)])).collect();
    let pdp_k: Vec<f64> = grid_k.iter().map(|&gk| pm(&[(component_k, gk)])).collect();

    let pdp_2d = logistic_pdp_2d(
        &scores,
        fit.intercept,
        &fit.coefficients,
        &fit.gamma,
        scalar_covariates,
        n,
        ncomp,
        component_j,
        component_k,
        &grid_j,
        &grid_k,
        n_grid,
    );

    let f_bar: f64 = fit.probabilities.iter().sum::<f64>() / n as f64;
    let h_squared = compute_h_squared(&pdp_2d, &pdp_j, &pdp_k, f_bar, n_grid);

    Some(FriedmanHResult {
        component_j,
        component_k,
        h_squared,
        grid_j,
        grid_k,
        pdp_2d,
    })
}

// ===========================================================================
// Feature 1: Pointwise Variable Importance
// ===========================================================================

/// Result of pointwise variable importance analysis.
pub struct PointwiseImportanceResult {
    /// Importance at each grid point (length m).
    pub importance: Vec<f64>,
    /// Normalized importance summing to 1 (length m).
    pub importance_normalized: Vec<f64>,
    /// Per-component importance (ncomp × m).
    pub component_importance: FdMatrix,
    /// Variance of each FPC score (length ncomp).
    pub score_variance: Vec<f64>,
}

/// Pointwise variable importance for a linear functional regression model.
///
/// Measures how much X(t_j) contributes to prediction variance via the FPC decomposition.
pub fn pointwise_importance(fit: &FregreLmResult) -> Option<PointwiseImportanceResult> {
    let ncomp = fit.ncomp;
    let m = fit.fpca.rotation.nrows();
    let n = fit.fpca.scores.nrows();
    if ncomp == 0 || m == 0 || n < 2 {
        return None;
    }

    let score_variance = compute_score_variance(&fit.fpca.scores, n, ncomp);
    let (component_importance, importance, importance_normalized) =
        compute_pointwise_importance_core(
            &fit.coefficients,
            &fit.fpca.rotation,
            &score_variance,
            ncomp,
            m,
        );

    Some(PointwiseImportanceResult {
        importance,
        importance_normalized,
        component_importance,
        score_variance,
    })
}

/// Pointwise variable importance for a functional logistic regression model.
pub fn pointwise_importance_logistic(
    fit: &FunctionalLogisticResult,
) -> Option<PointwiseImportanceResult> {
    let ncomp = fit.ncomp;
    let m = fit.fpca.rotation.nrows();
    let n = fit.fpca.scores.nrows();
    if ncomp == 0 || m == 0 || n < 2 {
        return None;
    }

    let score_variance = compute_score_variance(&fit.fpca.scores, n, ncomp);
    let (component_importance, importance, importance_normalized) =
        compute_pointwise_importance_core(
            &fit.coefficients,
            &fit.fpca.rotation,
            &score_variance,
            ncomp,
            m,
        );

    Some(PointwiseImportanceResult {
        importance,
        importance_normalized,
        component_importance,
        score_variance,
    })
}

// ===========================================================================
// Feature 2: VIF (Variance Inflation Factors)
// ===========================================================================

/// Result of VIF analysis for FPC-based regression.
pub struct VifResult {
    /// VIF values (length ncomp + p_scalar, excludes intercept).
    pub vif: Vec<f64>,
    /// Labels for each predictor.
    pub labels: Vec<String>,
    /// Mean VIF.
    pub mean_vif: f64,
    /// Number of predictors with VIF > 5.
    pub n_moderate: usize,
    /// Number of predictors with VIF > 10.
    pub n_severe: usize,
}

/// Variance inflation factors for FPC scores (and optional scalar covariates).
///
/// For orthogonal FPC scores without scalar covariates, VIF should be approximately 1.
pub fn fpc_vif(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
) -> Option<VifResult> {
    let (n, m) = data.shape();
    if n == 0 || m != fit.fpca.mean.len() {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    compute_vif_from_scores(&scores, ncomp, scalar_covariates, n)
}

/// VIF for a functional logistic regression model.
pub fn fpc_vif_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
) -> Option<VifResult> {
    let (n, m) = data.shape();
    if n == 0 || m != fit.fpca.mean.len() {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    compute_vif_from_scores(&scores, ncomp, scalar_covariates, n)
}

pub(crate) fn compute_vif_from_scores(
    scores: &FdMatrix,
    ncomp: usize,
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
) -> Option<VifResult> {
    let p_scalar = scalar_covariates.map_or(0, |sc| sc.ncols());
    let p = ncomp + p_scalar;
    if p == 0 || n <= p {
        return None;
    }

    let x_noi = build_no_intercept_matrix(scores, ncomp, scalar_covariates, n);
    let xtx = compute_xtx(&x_noi);
    let l = cholesky_factor(&xtx, p)?;

    let mut vif = vec![0.0; p];
    for k in 0..p {
        let mut ek = vec![0.0; p];
        ek[k] = 1.0;
        let v = cholesky_forward_back(&l, &ek, p);
        vif[k] = v[k] * xtx[k * p + k];
    }

    let mut labels = Vec::with_capacity(p);
    for k in 0..ncomp {
        labels.push(format!("FPC_{}", k));
    }
    for j in 0..p_scalar {
        labels.push(format!("scalar_{}", j));
    }

    let mean_vif = vif.iter().sum::<f64>() / p as f64;
    let n_moderate = vif.iter().filter(|&&v| v > 5.0).count();
    let n_severe = vif.iter().filter(|&&v| v > 10.0).count();

    Some(VifResult {
        vif,
        labels,
        mean_vif,
        n_moderate,
        n_severe,
    })
}

// ===========================================================================
// Feature 3: SHAP Values (FPC-level)
// ===========================================================================

/// FPC-level SHAP values for model interpretability.
pub struct FpcShapValues {
    /// SHAP values (n × ncomp).
    pub values: FdMatrix,
    /// Base value (mean prediction).
    pub base_value: f64,
    /// Mean FPC scores (length ncomp).
    pub mean_scores: Vec<f64>,
}

/// Exact SHAP values for a linear functional regression model.
///
/// For linear models, SHAP values are exact: `values[(i,k)] = coef[1+k] × (score_i_k - mean_k)`.
/// The efficiency property holds: `base_value + Σ_k values[(i,k)] ≈ fitted_values[i]`
/// (with scalar covariate effects absorbed into the base value).
pub fn fpc_shap_values(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
) -> Option<FpcShapValues> {
    let (n, m) = data.shape();
    if n == 0 || m != fit.fpca.mean.len() {
        return None;
    }
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return None;
    }
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let mean_scores = compute_column_means(&scores, ncomp);

    let mut base_value = fit.intercept;
    for k in 0..ncomp {
        base_value += fit.coefficients[1 + k] * mean_scores[k];
    }
    let p_scalar = fit.gamma.len();
    let mean_z = compute_mean_scalar(scalar_covariates, p_scalar, n);
    for j in 0..p_scalar {
        base_value += fit.gamma[j] * mean_z[j];
    }

    let mut values = FdMatrix::zeros(n, ncomp);
    for i in 0..n {
        for k in 0..ncomp {
            values[(i, k)] = fit.coefficients[1 + k] * (scores[(i, k)] - mean_scores[k]);
        }
    }

    Some(FpcShapValues {
        values,
        base_value,
        mean_scores,
    })
}

/// Kernel SHAP values for a functional logistic regression model.
///
/// Uses sampling-based Kernel SHAP approximation since the logistic link is nonlinear.
pub fn fpc_shap_values_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    n_samples: usize,
    seed: u64,
) -> Option<FpcShapValues> {
    let (n, m) = data.shape();
    if n == 0 || m != fit.fpca.mean.len() || n_samples == 0 {
        return None;
    }
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return None;
    }
    let p_scalar = fit.gamma.len();
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let mean_scores = compute_column_means(&scores, ncomp);
    let mean_z = compute_mean_scalar(scalar_covariates, p_scalar, n);

    let predict_proba = |obs_scores: &[f64], obs_z: &[f64]| -> f64 {
        let mut eta = fit.intercept;
        for k in 0..ncomp {
            eta += fit.coefficients[1 + k] * obs_scores[k];
        }
        for j in 0..p_scalar {
            eta += fit.gamma[j] * obs_z[j];
        }
        sigmoid(eta)
    };

    let base_value = predict_proba(&mean_scores, &mean_z);
    let mut values = FdMatrix::zeros(n, ncomp);
    let mut rng = StdRng::seed_from_u64(seed);

    for i in 0..n {
        let obs_scores: Vec<f64> = (0..ncomp).map(|k| scores[(i, k)]).collect();
        let obs_z = get_obs_scalar(scalar_covariates, i, p_scalar, &mean_z);

        let mut ata = vec![0.0; ncomp * ncomp];
        let mut atb = vec![0.0; ncomp];

        for _ in 0..n_samples {
            let (coalition, s_size) = sample_random_coalition(&mut rng, ncomp);
            let weight = shapley_kernel_weight(ncomp, s_size);
            let coal_scores = build_coalition_scores(&coalition, &obs_scores, &mean_scores);

            let f_coal = predict_proba(&coal_scores, &obs_z);
            let f_base = predict_proba(&mean_scores, &obs_z);
            let y_val = f_coal - f_base;

            accumulate_kernel_shap_sample(&mut ata, &mut atb, &coalition, weight, y_val, ncomp);
        }

        solve_kernel_shap_obs(&mut ata, &atb, ncomp, &mut values, i);
    }

    Some(FpcShapValues {
        values,
        base_value,
        mean_scores,
    })
}

/// Binomial coefficient C(n, k).
fn binom_coeff(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result: usize = 1;
    for i in 0..k {
        result = result.saturating_mul(n - i) / (i + 1);
    }
    result
}

// ===========================================================================
// Feature 4: DFBETAS / DFFITS
// ===========================================================================

/// Result of DFBETAS/DFFITS influence diagnostics.
pub struct DfbetasDffitsResult {
    /// DFBETAS values (n × p).
    pub dfbetas: FdMatrix,
    /// DFFITS values (length n).
    pub dffits: Vec<f64>,
    /// Studentized residuals (length n).
    pub studentized_residuals: Vec<f64>,
    /// Number of parameters p (including intercept).
    pub p: usize,
    /// DFBETAS cutoff: 2/√n.
    pub dfbetas_cutoff: f64,
    /// DFFITS cutoff: 2√(p/n).
    pub dffits_cutoff: f64,
}

/// DFBETAS and DFFITS for a linear functional regression model.
///
/// DFBETAS measures how much each coefficient changes when observation i is deleted.
/// DFFITS measures how much the fitted value changes when observation i is deleted.
pub fn dfbetas_dffits(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
) -> Option<DfbetasDffitsResult> {
    let (n, m) = data.shape();
    if n == 0 || m != fit.fpca.mean.len() {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let design = build_design_matrix(&scores, ncomp, scalar_covariates, n);
    let p = design.ncols();

    if n <= p {
        return None;
    }

    let xtx = compute_xtx(&design);
    let l = cholesky_factor(&xtx, p)?;
    let hat_diag = compute_hat_diagonal(&design, &l);

    let ss_res: f64 = fit.residuals.iter().map(|r| r * r).sum();
    let mse = ss_res / (n - p) as f64;
    let s = mse.sqrt();

    if s < 1e-15 {
        return None;
    }

    let se = compute_coefficient_se(&l, mse, p);

    let mut studentized_residuals = vec![0.0; n];
    let mut dffits = vec![0.0; n];
    let mut dfbetas = FdMatrix::zeros(n, p);

    for i in 0..n {
        let (t_i, dffits_i, dfb) =
            compute_obs_influence(&design, &l, fit.residuals[i], hat_diag[i], s, &se, p, i);
        studentized_residuals[i] = t_i;
        dffits[i] = dffits_i;
        for j in 0..p {
            dfbetas[(i, j)] = dfb[j];
        }
    }

    let dfbetas_cutoff = 2.0 / (n as f64).sqrt();
    let dffits_cutoff = 2.0 * (p as f64 / n as f64).sqrt();

    Some(DfbetasDffitsResult {
        dfbetas,
        dffits,
        studentized_residuals,
        p,
        dfbetas_cutoff,
        dffits_cutoff,
    })
}

// ===========================================================================
// Feature 5: Prediction Intervals
// ===========================================================================

/// Result of prediction interval computation.
pub struct PredictionIntervalResult {
    /// Point predictions ŷ_new (length n_new).
    pub predictions: Vec<f64>,
    /// Lower bounds (length n_new).
    pub lower: Vec<f64>,
    /// Upper bounds (length n_new).
    pub upper: Vec<f64>,
    /// Prediction standard errors: s × √(1 + h_new) (length n_new).
    pub prediction_se: Vec<f64>,
    /// Confidence level used.
    pub confidence_level: f64,
    /// Critical value used.
    pub t_critical: f64,
    /// Residual standard error from the training model.
    pub residual_se: f64,
}

/// Prediction intervals for new observations from a linear functional regression model.
///
/// Computes prediction intervals accounting for both estimation uncertainty
/// (through the hat matrix) and residual variance.
pub fn prediction_intervals(
    fit: &FregreLmResult,
    train_data: &FdMatrix,
    train_scalar: Option<&FdMatrix>,
    new_data: &FdMatrix,
    new_scalar: Option<&FdMatrix>,
    confidence_level: f64,
) -> Option<PredictionIntervalResult> {
    let (n_train, m) = train_data.shape();
    let (n_new, m_new) = new_data.shape();
    if confidence_level <= 0.0
        || confidence_level >= 1.0
        || n_train == 0
        || m != fit.fpca.mean.len()
        || n_new == 0
        || m_new != m
    {
        return None;
    }
    let ncomp = fit.ncomp;

    let train_scores = project_scores(train_data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let train_design = build_design_matrix(&train_scores, ncomp, train_scalar, n_train);
    let p = train_design.ncols();
    if n_train <= p {
        return None;
    }

    let xtx = compute_xtx(&train_design);
    let l = cholesky_factor(&xtx, p)?;

    let residual_se = fit.residual_se;
    let df = n_train - p;
    let t_crit = t_critical_value(confidence_level, df);

    // Project new data
    let new_scores = project_scores(new_data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let mut predictions = vec![0.0; n_new];
    let mut lower = vec![0.0; n_new];
    let mut upper = vec![0.0; n_new];
    let mut prediction_se = vec![0.0; n_new];

    let p_scalar = fit.gamma.len();

    for i in 0..n_new {
        let x_new = build_design_vector(&new_scores, new_scalar, i, ncomp, p_scalar, p);
        let (yhat, lo, up, pse) =
            compute_prediction_interval_obs(&l, &fit.coefficients, &x_new, p, residual_se, t_crit);
        predictions[i] = yhat;
        lower[i] = lo;
        upper[i] = up;
        prediction_se[i] = pse;
    }

    Some(PredictionIntervalResult {
        predictions,
        lower,
        upper,
        prediction_se,
        confidence_level,
        t_critical: t_crit,
        residual_se,
    })
}

/// Normal quantile approximation (Abramowitz & Stegun 26.2.23).
fn normal_quantile(p: f64) -> f64 {
    // Rational approximation for the inverse normal CDF
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };
    // Coefficients from Abramowitz & Stegun
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    let val = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if p < 0.5 {
        -val
    } else {
        val
    }
}

/// t-distribution critical value with Cornish-Fisher correction for small df.
fn t_critical_value(conf: f64, df: usize) -> f64 {
    let alpha = 1.0 - conf;
    let z = normal_quantile(1.0 - alpha / 2.0);
    if df == 0 {
        return z;
    }
    // Cornish-Fisher expansion for t-distribution
    let df_f = df as f64;
    let g1 = (z.powi(3) + z) / (4.0 * df_f);
    let g2 = (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / (96.0 * df_f * df_f);
    let g3 = (3.0 * z.powi(7) + 19.0 * z.powi(5) + 17.0 * z.powi(3) - 15.0 * z)
        / (384.0 * df_f * df_f * df_f);
    z + g1 + g2 + g3
}

// ===========================================================================
// Feature 6: ALE (Accumulated Local Effects)
// ===========================================================================

/// Result of Accumulated Local Effects analysis.
pub struct AleResult {
    /// Bin midpoints (length n_bins_actual).
    pub bin_midpoints: Vec<f64>,
    /// ALE values centered to mean zero (length n_bins_actual).
    pub ale_values: Vec<f64>,
    /// Bin edges (length n_bins_actual + 1).
    pub bin_edges: Vec<f64>,
    /// Number of observations in each bin (length n_bins_actual).
    pub bin_counts: Vec<usize>,
    /// Which FPC component was analyzed.
    pub component: usize,
}

/// ALE plot for an FPC component in a linear functional regression model.
///
/// ALE measures the average local effect of varying one FPC score on predictions,
/// avoiding the extrapolation issues of PDP.
pub fn fpc_ale(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_bins: usize,
) -> Option<AleResult> {
    let (n, m) = data.shape();
    if n < 2 || m != fit.fpca.mean.len() || n_bins == 0 || component >= fit.ncomp {
        return None;
    }
    let ncomp = fit.ncomp;
    let p_scalar = fit.gamma.len();
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    // Prediction function for linear model
    let predict = |obs_scores: &[f64], obs_scalar: Option<&[f64]>| -> f64 {
        let mut eta = fit.intercept;
        for k in 0..ncomp {
            eta += fit.coefficients[1 + k] * obs_scores[k];
        }
        if let Some(z) = obs_scalar {
            for j in 0..p_scalar {
                eta += fit.gamma[j] * z[j];
            }
        }
        eta
    };

    compute_ale(
        &scores,
        scalar_covariates,
        n,
        ncomp,
        p_scalar,
        component,
        n_bins,
        &predict,
    )
}

/// ALE plot for an FPC component in a functional logistic regression model.
pub fn fpc_ale_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_bins: usize,
) -> Option<AleResult> {
    let (n, m) = data.shape();
    if n < 2 || m != fit.fpca.mean.len() || n_bins == 0 || component >= fit.ncomp {
        return None;
    }
    let ncomp = fit.ncomp;
    let p_scalar = fit.gamma.len();
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    // Prediction function for logistic model
    let predict = |obs_scores: &[f64], obs_scalar: Option<&[f64]>| -> f64 {
        let mut eta = fit.intercept;
        for k in 0..ncomp {
            eta += fit.coefficients[1 + k] * obs_scores[k];
        }
        if let Some(z) = obs_scalar {
            for j in 0..p_scalar {
                eta += fit.gamma[j] * z[j];
            }
        }
        sigmoid(eta)
    };

    compute_ale(
        &scores,
        scalar_covariates,
        n,
        ncomp,
        p_scalar,
        component,
        n_bins,
        &predict,
    )
}

pub(crate) fn compute_ale(
    scores: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
    ncomp: usize,
    p_scalar: usize,
    component: usize,
    n_bins: usize,
    predict: &dyn Fn(&[f64], Option<&[f64]>) -> f64,
) -> Option<AleResult> {
    let mut col: Vec<(f64, usize)> = (0..n).map(|i| (scores[(i, component)], i)).collect();
    col.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let bin_edges = compute_ale_bin_edges(&col, n, n_bins);
    let n_bins_actual = bin_edges.len() - 1;
    let bin_assignments = assign_ale_bins(&col, &bin_edges, n, n_bins_actual);

    let mut deltas = vec![0.0; n_bins_actual];
    let mut bin_counts = vec![0usize; n_bins_actual];

    for i in 0..n {
        let b = bin_assignments[i];
        bin_counts[b] += 1;

        let mut obs_scores: Vec<f64> = (0..ncomp).map(|k| scores[(i, k)]).collect();
        let obs_z: Option<Vec<f64>> = if p_scalar > 0 {
            scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i, j)]).collect())
        } else {
            None
        };
        let z_ref = obs_z.as_deref();

        obs_scores[component] = bin_edges[b + 1];
        let f_upper = predict(&obs_scores, z_ref);
        obs_scores[component] = bin_edges[b];
        let f_lower = predict(&obs_scores, z_ref);

        deltas[b] += f_upper - f_lower;
    }

    for b in 0..n_bins_actual {
        if bin_counts[b] > 0 {
            deltas[b] /= bin_counts[b] as f64;
        }
    }

    let mut ale_values = vec![0.0; n_bins_actual];
    ale_values[0] = deltas[0];
    for b in 1..n_bins_actual {
        ale_values[b] = ale_values[b - 1] + deltas[b];
    }

    let total_n: usize = bin_counts.iter().sum();
    if total_n > 0 {
        let weighted_mean: f64 = ale_values
            .iter()
            .zip(&bin_counts)
            .map(|(&a, &c)| a * c as f64)
            .sum::<f64>()
            / total_n as f64;
        for v in &mut ale_values {
            *v -= weighted_mean;
        }
    }

    let bin_midpoints: Vec<f64> = (0..n_bins_actual)
        .map(|b| (bin_edges[b] + bin_edges[b + 1]) / 2.0)
        .collect();

    Some(AleResult {
        bin_midpoints,
        ale_values,
        bin_edges,
        bin_counts,
        component,
    })
}

// ===========================================================================
// Feature 7: LOO-CV / PRESS
// ===========================================================================

/// Result of leave-one-out cross-validation diagnostics.
pub struct LooCvResult {
    /// LOO residuals: e_i / (1 - h_ii), length n.
    pub loo_residuals: Vec<f64>,
    /// PRESS statistic: Σ loo_residuals².
    pub press: f64,
    /// LOO R²: 1 - PRESS / TSS.
    pub loo_r_squared: f64,
    /// Hat diagonal h_ii, length n.
    pub leverage: Vec<f64>,
    /// Total sum of squares: Σ (y_i - ȳ)².
    pub tss: f64,
}

/// LOO-CV / PRESS diagnostics for a linear functional regression model.
///
/// Uses the hat-matrix shortcut: LOO residual = e_i / (1 - h_ii).
pub fn loo_cv_press(
    fit: &FregreLmResult,
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
) -> Option<LooCvResult> {
    let (n, m) = data.shape();
    if n == 0 || n != y.len() || m != fit.fpca.mean.len() {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let design = build_design_matrix(&scores, ncomp, scalar_covariates, n);
    let p = design.ncols();
    if n <= p {
        return None;
    }

    let xtx = compute_xtx(&design);
    let l = cholesky_factor(&xtx, p)?;
    let leverage = compute_hat_diagonal(&design, &l);

    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    if tss == 0.0 {
        return None;
    }

    let mut loo_residuals = vec![0.0; n];
    let mut press = 0.0;
    for i in 0..n {
        let denom = (1.0 - leverage[i]).max(1e-15);
        loo_residuals[i] = fit.residuals[i] / denom;
        press += loo_residuals[i] * loo_residuals[i];
    }

    let loo_r_squared = 1.0 - press / tss;

    Some(LooCvResult {
        loo_residuals,
        press,
        loo_r_squared,
        leverage,
        tss,
    })
}

// ===========================================================================
// Feature 8: Sobol Sensitivity Indices
// ===========================================================================

/// Sobol first-order and total-order sensitivity indices.
pub struct SobolIndicesResult {
    /// First-order indices S_k, length ncomp.
    pub first_order: Vec<f64>,
    /// Total-order indices ST_k, length ncomp.
    pub total_order: Vec<f64>,
    /// Total variance of Y.
    pub var_y: f64,
    /// Per-component variance contribution, length ncomp.
    pub component_variance: Vec<f64>,
}

/// Exact Sobol sensitivity indices for a linear functional regression model.
///
/// For an additive model with orthogonal FPC predictors, first-order = total-order.
pub fn sobol_indices(
    fit: &FregreLmResult,
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
) -> Option<SobolIndicesResult> {
    let (n, m) = data.shape();
    if n < 2 || n != y.len() || m != fit.fpca.mean.len() {
        return None;
    }
    let _ = scalar_covariates; // not needed for variance decomposition
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return None;
    }

    let score_var = compute_score_variance(&fit.fpca.scores, n, ncomp);

    let component_variance: Vec<f64> = (0..ncomp)
        .map(|k| fit.coefficients[1 + k].powi(2) * score_var[k])
        .collect();

    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let var_y: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    if var_y == 0.0 {
        return None;
    }

    let first_order: Vec<f64> = component_variance.iter().map(|&cv| cv / var_y).collect();
    let total_order = first_order.clone(); // additive + orthogonal → S_k = ST_k

    Some(SobolIndicesResult {
        first_order,
        total_order,
        var_y,
        component_variance,
    })
}

/// Sobol sensitivity indices for a functional logistic regression model (Saltelli MC).
pub fn sobol_indices_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    n_samples: usize,
    seed: u64,
) -> Option<SobolIndicesResult> {
    let (n, m) = data.shape();
    if n < 2 || m != fit.fpca.mean.len() || n_samples == 0 {
        return None;
    }
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return None;
    }
    let p_scalar = fit.gamma.len();
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let mean_z = compute_mean_scalar(scalar_covariates, p_scalar, n);

    let eval_model = |s: &[f64]| -> f64 {
        let mut eta = fit.intercept;
        for k in 0..ncomp {
            eta += fit.coefficients[1 + k] * s[k];
        }
        for j in 0..p_scalar {
            eta += fit.gamma[j] * mean_z[j];
        }
        sigmoid(eta)
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let (mat_a, mat_b) = generate_sobol_matrices(&scores, n, ncomp, n_samples, &mut rng);

    let f_a: Vec<f64> = mat_a.iter().map(|s| eval_model(s)).collect();
    let f_b: Vec<f64> = mat_b.iter().map(|s| eval_model(s)).collect();

    let mean_fa = f_a.iter().sum::<f64>() / n_samples as f64;
    let var_fa = f_a.iter().map(|&v| (v - mean_fa).powi(2)).sum::<f64>() / n_samples as f64;

    if var_fa < 1e-15 {
        return None;
    }

    let mut first_order = vec![0.0; ncomp];
    let mut total_order = vec![0.0; ncomp];
    let mut component_variance = vec![0.0; ncomp];

    for k in 0..ncomp {
        let (s_k, st_k) = compute_sobol_component(
            &mat_a,
            &mat_b,
            &f_a,
            &f_b,
            var_fa,
            k,
            n_samples,
            &eval_model,
        );
        first_order[k] = s_k;
        total_order[k] = st_k;
        component_variance[k] = s_k * var_fa;
    }

    Some(SobolIndicesResult {
        first_order,
        total_order,
        var_y: var_fa,
        component_variance,
    })
}

// ===========================================================================
// Feature 9: Calibration Diagnostics (logistic only)
// ===========================================================================

/// Calibration diagnostics for a functional logistic regression model.
pub struct CalibrationDiagnosticsResult {
    /// Brier score: (1/n) Σ (p_i - y_i)².
    pub brier_score: f64,
    /// Log loss: -(1/n) Σ [y log p + (1-y) log(1-p)].
    pub log_loss: f64,
    /// Hosmer-Lemeshow chi² statistic.
    pub hosmer_lemeshow_chi2: f64,
    /// Degrees of freedom: n_groups - 2.
    pub hosmer_lemeshow_df: usize,
    /// Number of calibration groups.
    pub n_groups: usize,
    /// Reliability bins: (mean_predicted, mean_observed) per group.
    pub reliability_bins: Vec<(f64, f64)>,
    /// Number of observations in each group.
    pub bin_counts: Vec<usize>,
}

/// Calibration diagnostics for a functional logistic regression model.
pub fn calibration_diagnostics(
    fit: &FunctionalLogisticResult,
    y: &[f64],
    n_groups: usize,
) -> Option<CalibrationDiagnosticsResult> {
    let n = fit.probabilities.len();
    if n == 0 || n != y.len() || n_groups < 2 {
        return None;
    }

    // Brier score
    let brier_score: f64 = fit
        .probabilities
        .iter()
        .zip(y)
        .map(|(&p, &yi)| (p - yi).powi(2))
        .sum::<f64>()
        / n as f64;

    // Log loss
    let log_loss: f64 = -fit
        .probabilities
        .iter()
        .zip(y)
        .map(|(&p, &yi)| {
            let p_clip = p.clamp(1e-15, 1.0 - 1e-15);
            yi * p_clip.ln() + (1.0 - yi) * (1.0 - p_clip).ln()
        })
        .sum::<f64>()
        / n as f64;

    let (hosmer_lemeshow_chi2, reliability_bins, bin_counts) =
        hosmer_lemeshow_computation(&fit.probabilities, y, n, n_groups);

    let actual_groups = bin_counts.len();
    let hosmer_lemeshow_df = if actual_groups > 2 {
        actual_groups - 2
    } else {
        1
    };

    Some(CalibrationDiagnosticsResult {
        brier_score,
        log_loss,
        hosmer_lemeshow_chi2,
        hosmer_lemeshow_df,
        n_groups: actual_groups,
        reliability_bins,
        bin_counts,
    })
}

// ===========================================================================
// Feature 10: Functional Saliency Maps
// ===========================================================================

/// Functional saliency map result.
pub struct FunctionalSaliencyResult {
    /// Saliency map (n × m).
    pub saliency_map: FdMatrix,
    /// Mean absolute saliency at each grid point (length m).
    pub mean_absolute_saliency: Vec<f64>,
}

/// Functional saliency maps for a linear functional regression model.
///
/// Lifts FPC-level SHAP attributions to the function domain via the rotation matrix.
pub fn functional_saliency(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
) -> Option<FunctionalSaliencyResult> {
    let (n, m) = data.shape();
    if n == 0 || m != fit.fpca.mean.len() {
        return None;
    }
    let _ = scalar_covariates;
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return None;
    }
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let mean_scores = compute_column_means(&scores, ncomp);

    let weights: Vec<f64> = (0..ncomp).map(|k| fit.coefficients[1 + k]).collect();
    let saliency_map = compute_saliency_map(
        &scores,
        &mean_scores,
        &weights,
        &fit.fpca.rotation,
        n,
        m,
        ncomp,
    );
    let mean_absolute_saliency = mean_absolute_column(&saliency_map, n, m);

    Some(FunctionalSaliencyResult {
        saliency_map,
        mean_absolute_saliency,
    })
}

/// Functional saliency maps for a functional logistic regression model (gradient-based).
pub fn functional_saliency_logistic(
    fit: &FunctionalLogisticResult,
) -> Option<FunctionalSaliencyResult> {
    let m = fit.beta_t.len();
    let n = fit.probabilities.len();
    if n == 0 || m == 0 {
        return None;
    }

    // saliency[(i,j)] = p_i × (1 - p_i) × beta_t[j]
    let mut saliency_map = FdMatrix::zeros(n, m);
    for i in 0..n {
        let pi = fit.probabilities[i];
        let w = pi * (1.0 - pi);
        for j in 0..m {
            saliency_map[(i, j)] = w * fit.beta_t[j];
        }
    }

    let mut mean_absolute_saliency = vec![0.0; m];
    for j in 0..m {
        for i in 0..n {
            mean_absolute_saliency[j] += saliency_map[(i, j)].abs();
        }
        mean_absolute_saliency[j] /= n as f64;
    }

    Some(FunctionalSaliencyResult {
        saliency_map,
        mean_absolute_saliency,
    })
}

// ===========================================================================
// Feature 11: Domain Selection / Interval Importance
// ===========================================================================

/// An important interval in the function domain.
pub struct ImportantInterval {
    /// Start index (inclusive).
    pub start_idx: usize,
    /// End index (inclusive).
    pub end_idx: usize,
    /// Summed importance of the interval.
    pub importance: f64,
}

/// Result of domain selection analysis.
pub struct DomainSelectionResult {
    /// Pointwise importance: |β(t)|², length m.
    pub pointwise_importance: Vec<f64>,
    /// Important intervals sorted by importance descending.
    pub intervals: Vec<ImportantInterval>,
    /// Sliding window width used.
    pub window_width: usize,
    /// Threshold used.
    pub threshold: f64,
}

/// Domain selection for a linear functional regression model.
pub fn domain_selection(
    fit: &FregreLmResult,
    window_width: usize,
    threshold: f64,
) -> Option<DomainSelectionResult> {
    compute_domain_selection(&fit.beta_t, window_width, threshold)
}

/// Domain selection for a functional logistic regression model.
pub fn domain_selection_logistic(
    fit: &FunctionalLogisticResult,
    window_width: usize,
    threshold: f64,
) -> Option<DomainSelectionResult> {
    compute_domain_selection(&fit.beta_t, window_width, threshold)
}

pub(crate) fn compute_domain_selection(
    beta_t: &[f64],
    window_width: usize,
    threshold: f64,
) -> Option<DomainSelectionResult> {
    let m = beta_t.len();
    if m == 0 || window_width == 0 || window_width > m || threshold <= 0.0 {
        return None;
    }

    let pointwise_importance: Vec<f64> = beta_t.iter().map(|&b| b * b).collect();
    let total_imp: f64 = pointwise_importance.iter().sum();
    if total_imp == 0.0 {
        return Some(DomainSelectionResult {
            pointwise_importance,
            intervals: vec![],
            window_width,
            threshold,
        });
    }

    // Sliding window with running sum
    let mut window_sum: f64 = pointwise_importance[..window_width].iter().sum();
    let mut raw_intervals: Vec<(usize, usize, f64)> = Vec::new();
    if window_sum / total_imp >= threshold {
        raw_intervals.push((0, window_width - 1, window_sum));
    }
    for start in 1..=(m - window_width) {
        window_sum -= pointwise_importance[start - 1];
        window_sum += pointwise_importance[start + window_width - 1];
        if window_sum / total_imp >= threshold {
            raw_intervals.push((start, start + window_width - 1, window_sum));
        }
    }

    let mut intervals = merge_overlapping_intervals(raw_intervals);
    intervals.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());

    Some(DomainSelectionResult {
        pointwise_importance,
        intervals,
        window_width,
        threshold,
    })
}

// ===========================================================================
// Feature 12: Conditional Permutation Importance
// ===========================================================================

/// Result of conditional permutation importance.
pub struct ConditionalPermutationImportanceResult {
    /// Conditional importance per FPC component, length ncomp.
    pub importance: Vec<f64>,
    /// Baseline metric (R² or accuracy).
    pub baseline_metric: f64,
    /// Mean metric after conditional permutation, length ncomp.
    pub permuted_metric: Vec<f64>,
    /// Unconditional (standard) permutation importance for comparison, length ncomp.
    pub unconditional_importance: Vec<f64>,
}

/// Conditional permutation importance for a linear functional regression model.
pub fn conditional_permutation_importance(
    fit: &FregreLmResult,
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n_bins: usize,
    n_perm: usize,
    seed: u64,
) -> Option<ConditionalPermutationImportanceResult> {
    let (n, m) = data.shape();
    if n == 0 || n != y.len() || m != fit.fpca.mean.len() || n_perm == 0 || n_bins == 0 {
        return None;
    }
    let _ = scalar_covariates;
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    if ss_tot == 0.0 {
        return None;
    }
    let ss_res_base: f64 = fit.residuals.iter().map(|r| r * r).sum();
    let baseline = 1.0 - ss_res_base / ss_tot;

    let predict_r2 = |score_mat: &FdMatrix| -> f64 {
        let ss_res: f64 = (0..n)
            .map(|i| {
                let mut yhat = fit.coefficients[0];
                for c in 0..ncomp {
                    yhat += fit.coefficients[1 + c] * score_mat[(i, c)];
                }
                (y[i] - yhat).powi(2)
            })
            .sum();
        1.0 - ss_res / ss_tot
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut importance = vec![0.0; ncomp];
    let mut permuted_metric = vec![0.0; ncomp];
    let mut unconditional_importance = vec![0.0; ncomp];

    for k in 0..ncomp {
        let bins = compute_conditioning_bins(&scores, ncomp, k, n, n_bins);
        let (mean_cond, mean_uncond) =
            permute_component(&scores, &bins, k, n, ncomp, n_perm, &mut rng, &predict_r2);
        permuted_metric[k] = mean_cond;
        importance[k] = baseline - mean_cond;
        unconditional_importance[k] = baseline - mean_uncond;
    }

    Some(ConditionalPermutationImportanceResult {
        importance,
        baseline_metric: baseline,
        permuted_metric,
        unconditional_importance,
    })
}

/// Conditional permutation importance for a functional logistic regression model.
pub fn conditional_permutation_importance_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n_bins: usize,
    n_perm: usize,
    seed: u64,
) -> Option<ConditionalPermutationImportanceResult> {
    let (n, m) = data.shape();
    if n == 0 || n != y.len() || m != fit.fpca.mean.len() || n_perm == 0 || n_bins == 0 {
        return None;
    }
    let _ = scalar_covariates;
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let baseline: f64 = (0..n)
        .filter(|&i| {
            let pred = if fit.probabilities[i] >= 0.5 {
                1.0
            } else {
                0.0
            };
            (pred - y[i]).abs() < 1e-10
        })
        .count() as f64
        / n as f64;

    let predict_acc = |score_mat: &FdMatrix| -> f64 {
        let correct: usize = (0..n)
            .filter(|&i| {
                let mut eta = fit.intercept;
                for c in 0..ncomp {
                    eta += fit.coefficients[1 + c] * score_mat[(i, c)];
                }
                let pred = if sigmoid(eta) >= 0.5 { 1.0 } else { 0.0 };
                (pred - y[i]).abs() < 1e-10
            })
            .count();
        correct as f64 / n as f64
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut importance = vec![0.0; ncomp];
    let mut permuted_metric = vec![0.0; ncomp];
    let mut unconditional_importance = vec![0.0; ncomp];

    for k in 0..ncomp {
        let bins = compute_conditioning_bins(&scores, ncomp, k, n, n_bins);
        let (mean_cond, mean_uncond) =
            permute_component(&scores, &bins, k, n, ncomp, n_perm, &mut rng, &predict_acc);
        permuted_metric[k] = mean_cond;
        importance[k] = baseline - mean_cond;
        unconditional_importance[k] = baseline - mean_uncond;
    }

    Some(ConditionalPermutationImportanceResult {
        importance,
        baseline_metric: baseline,
        permuted_metric,
        unconditional_importance,
    })
}

// ===========================================================================
// Feature 13: Counterfactual Explanations
// ===========================================================================

/// Result of a counterfactual explanation.
pub struct CounterfactualResult {
    /// Index of the observation.
    pub observation: usize,
    /// Original FPC scores.
    pub original_scores: Vec<f64>,
    /// Counterfactual FPC scores.
    pub counterfactual_scores: Vec<f64>,
    /// Score deltas: counterfactual - original.
    pub delta_scores: Vec<f64>,
    /// Counterfactual perturbation in function domain: Σ_k Δξ_k φ_k(t), length m.
    pub delta_function: Vec<f64>,
    /// L2 distance in score space: ||Δξ||.
    pub distance: f64,
    /// Original model prediction.
    pub original_prediction: f64,
    /// Counterfactual prediction.
    pub counterfactual_prediction: f64,
    /// Whether a valid counterfactual was found.
    pub found: bool,
}

/// Counterfactual explanation for a linear functional regression model (analytical).
pub fn counterfactual_regression(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    target_value: f64,
) -> Option<CounterfactualResult> {
    let (n, m) = data.shape();
    if observation >= n || m != fit.fpca.mean.len() {
        return None;
    }
    let _ = scalar_covariates;
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return None;
    }
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let original_prediction = fit.fitted_values[observation];
    let gap = target_value - original_prediction;

    // γ = [coef[1], ..., coef[ncomp]]
    let gamma: Vec<f64> = (0..ncomp).map(|k| fit.coefficients[1 + k]).collect();
    let gamma_norm_sq: f64 = gamma.iter().map(|g| g * g).sum();

    if gamma_norm_sq < 1e-30 {
        return None;
    }

    let original_scores: Vec<f64> = (0..ncomp).map(|k| scores[(observation, k)]).collect();
    let delta_scores: Vec<f64> = gamma.iter().map(|&gk| gap * gk / gamma_norm_sq).collect();
    let counterfactual_scores: Vec<f64> = original_scores
        .iter()
        .zip(&delta_scores)
        .map(|(&o, &d)| o + d)
        .collect();

    let delta_function = reconstruct_delta_function(&delta_scores, &fit.fpca.rotation, ncomp, m);
    let distance: f64 = delta_scores.iter().map(|d| d * d).sum::<f64>().sqrt();
    let counterfactual_prediction = original_prediction + gap;

    Some(CounterfactualResult {
        observation,
        original_scores,
        counterfactual_scores,
        delta_scores,
        delta_function,
        distance,
        original_prediction,
        counterfactual_prediction,
        found: true,
    })
}

/// Counterfactual explanation for a functional logistic regression model (gradient descent).
pub fn counterfactual_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    max_iter: usize,
    step_size: f64,
) -> Option<CounterfactualResult> {
    let (n, m) = data.shape();
    if observation >= n || m != fit.fpca.mean.len() {
        return None;
    }
    let _ = scalar_covariates;
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return None;
    }
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let original_scores: Vec<f64> = (0..ncomp).map(|k| scores[(observation, k)]).collect();
    let original_prediction = fit.probabilities[observation];
    let original_class = if original_prediction >= 0.5 { 1 } else { 0 };
    let target_class = 1 - original_class;

    let (current_scores, current_pred, found) = logistic_counterfactual_descent(
        fit.intercept,
        &fit.coefficients,
        &original_scores,
        target_class,
        ncomp,
        max_iter,
        step_size,
    );

    let delta_scores: Vec<f64> = current_scores
        .iter()
        .zip(&original_scores)
        .map(|(&c, &o)| c - o)
        .collect();

    let delta_function = reconstruct_delta_function(&delta_scores, &fit.fpca.rotation, ncomp, m);
    let distance: f64 = delta_scores.iter().map(|d| d * d).sum::<f64>().sqrt();

    Some(CounterfactualResult {
        observation,
        original_scores,
        counterfactual_scores: current_scores,
        delta_scores,
        delta_function,
        distance,
        original_prediction,
        counterfactual_prediction: current_pred,
        found,
    })
}

// ===========================================================================
// Feature 14: Prototype / Criticism Selection (MMD-based)
// ===========================================================================

/// Result of prototype/criticism selection.
pub struct PrototypeCriticismResult {
    /// Indices of selected prototypes.
    pub prototype_indices: Vec<usize>,
    /// Witness function values for prototypes.
    pub prototype_witness: Vec<f64>,
    /// Indices of selected criticisms.
    pub criticism_indices: Vec<usize>,
    /// Witness function values for criticisms.
    pub criticism_witness: Vec<f64>,
    /// Bandwidth used for the Gaussian kernel.
    pub bandwidth: f64,
}

/// Compute pairwise Gaussian kernel matrix from FPC scores.
pub(crate) fn gaussian_kernel_matrix(scores: &FdMatrix, ncomp: usize, bandwidth: f64) -> Vec<f64> {
    let n = scores.nrows();
    let mut k = vec![0.0; n * n];
    let bw2 = 2.0 * bandwidth * bandwidth;
    for i in 0..n {
        k[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let mut dist_sq = 0.0;
            for c in 0..ncomp {
                let d = scores[(i, c)] - scores[(j, c)];
                dist_sq += d * d;
            }
            let val = (-dist_sq / bw2).exp();
            k[i * n + j] = val;
            k[j * n + i] = val;
        }
    }
    k
}

/// Select prototypes and criticisms from FPCA scores using MMD-based greedy selection.
///
/// Takes an `FpcaResult` directly — works with both linear and logistic models
/// (caller passes `&fit.fpca`).
pub fn prototype_criticism(
    fpca: &FpcaResult,
    ncomp: usize,
    n_prototypes: usize,
    n_criticisms: usize,
) -> Option<PrototypeCriticismResult> {
    let n = fpca.scores.nrows();
    let actual_ncomp = ncomp.min(fpca.scores.ncols());
    if n == 0 || actual_ncomp == 0 || n_prototypes == 0 || n_prototypes > n {
        return None;
    }
    let n_crit = n_criticisms.min(n.saturating_sub(n_prototypes));

    let bandwidth = median_bandwidth(&fpca.scores, n, actual_ncomp);
    let kernel = gaussian_kernel_matrix(&fpca.scores, actual_ncomp, bandwidth);
    let mu_data = compute_kernel_mean(&kernel, n);

    let (selected, is_selected) = greedy_prototype_selection(&mu_data, &kernel, n, n_prototypes);
    let witness = compute_witness(&kernel, &mu_data, &selected, n);
    let prototype_witness: Vec<f64> = selected.iter().map(|&i| witness[i]).collect();

    let mut criticism_candidates: Vec<(usize, f64)> = (0..n)
        .filter(|i| !is_selected[*i])
        .map(|i| (i, witness[i].abs()))
        .collect();
    criticism_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let criticism_indices: Vec<usize> = criticism_candidates
        .iter()
        .take(n_crit)
        .map(|&(i, _)| i)
        .collect();
    let criticism_witness: Vec<f64> = criticism_indices.iter().map(|&i| witness[i]).collect();

    Some(PrototypeCriticismResult {
        prototype_indices: selected,
        prototype_witness,
        criticism_indices,
        criticism_witness,
        bandwidth,
    })
}

// ===========================================================================
// Feature 15: LIME (Local Surrogate)
// ===========================================================================

/// Result of a LIME local surrogate explanation.
pub struct LimeResult {
    /// Index of the observation being explained.
    pub observation: usize,
    /// Local FPC-level attributions, length ncomp.
    pub attributions: Vec<f64>,
    /// Local intercept.
    pub local_intercept: f64,
    /// Local R² (weighted).
    pub local_r_squared: f64,
    /// Kernel width used.
    pub kernel_width: f64,
}

/// LIME explanation for a linear functional regression model.
pub fn lime_explanation(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    n_samples: usize,
    kernel_width: f64,
    seed: u64,
) -> Option<LimeResult> {
    let (n, m) = data.shape();
    if observation >= n || m != fit.fpca.mean.len() || n_samples == 0 || kernel_width <= 0.0 {
        return None;
    }
    let _ = scalar_covariates;
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return None;
    }
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let obs_scores: Vec<f64> = (0..ncomp).map(|k| scores[(observation, k)]).collect();

    // Score standard deviations
    let mut score_sd = vec![0.0; ncomp];
    for k in 0..ncomp {
        let mut ss = 0.0;
        for i in 0..n {
            let s = scores[(i, k)];
            ss += s * s;
        }
        score_sd[k] = (ss / (n - 1).max(1) as f64).sqrt().max(1e-10);
    }

    // Predict for linear model
    let predict = |s: &[f64]| -> f64 {
        let mut yhat = fit.coefficients[0];
        for k in 0..ncomp {
            yhat += fit.coefficients[1 + k] * s[k];
        }
        yhat
    };

    compute_lime(
        &obs_scores,
        &score_sd,
        ncomp,
        n_samples,
        kernel_width,
        seed,
        observation,
        &predict,
    )
}

/// LIME explanation for a functional logistic regression model.
pub fn lime_explanation_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    n_samples: usize,
    kernel_width: f64,
    seed: u64,
) -> Option<LimeResult> {
    let (n, m) = data.shape();
    if observation >= n || m != fit.fpca.mean.len() || n_samples == 0 || kernel_width <= 0.0 {
        return None;
    }
    let _ = scalar_covariates;
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return None;
    }
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let obs_scores: Vec<f64> = (0..ncomp).map(|k| scores[(observation, k)]).collect();

    let mut score_sd = vec![0.0; ncomp];
    for k in 0..ncomp {
        let mut ss = 0.0;
        for i in 0..n {
            let s = scores[(i, k)];
            ss += s * s;
        }
        score_sd[k] = (ss / (n - 1).max(1) as f64).sqrt().max(1e-10);
    }

    let predict = |s: &[f64]| -> f64 {
        let mut eta = fit.intercept;
        for k in 0..ncomp {
            eta += fit.coefficients[1 + k] * s[k];
        }
        sigmoid(eta)
    };

    compute_lime(
        &obs_scores,
        &score_sd,
        ncomp,
        n_samples,
        kernel_width,
        seed,
        observation,
        &predict,
    )
}

pub(crate) fn compute_lime(
    obs_scores: &[f64],
    score_sd: &[f64],
    ncomp: usize,
    n_samples: usize,
    kernel_width: f64,
    seed: u64,
    observation: usize,
    predict: &dyn Fn(&[f64]) -> f64,
) -> Option<LimeResult> {
    let mut rng = StdRng::seed_from_u64(seed);

    let (perturbed, predictions, weights) = sample_lime_perturbations(
        obs_scores,
        score_sd,
        ncomp,
        n_samples,
        kernel_width,
        &mut rng,
        predict,
    )?;

    // Weighted OLS: fit y = intercept + Σ β_k (z_k - obs_k)
    let p = ncomp + 1;
    let mut ata = vec![0.0; p * p];
    let mut atb = vec![0.0; p];

    for i in 0..n_samples {
        let w = weights[i];
        let mut x = vec![0.0; p];
        x[0] = 1.0;
        for k in 0..ncomp {
            x[1 + k] = perturbed[i][k] - obs_scores[k];
        }
        for j1 in 0..p {
            for j2 in 0..p {
                ata[j1 * p + j2] += w * x[j1] * x[j2];
            }
            atb[j1] += w * x[j1] * predictions[i];
        }
    }

    for j in 0..p {
        ata[j * p + j] += 1e-10;
    }

    let l = cholesky_factor(&ata, p)?;
    let beta = cholesky_forward_back(&l, &atb, p);

    let local_intercept = beta[0];
    let attributions: Vec<f64> = beta[1..].to_vec();
    let local_r_squared = weighted_r_squared(
        &predictions,
        &beta,
        &perturbed,
        obs_scores,
        &weights,
        ncomp,
        n_samples,
    );

    Some(LimeResult {
        observation,
        attributions,
        local_intercept,
        local_r_squared,
        kernel_width,
    })
}

// ===========================================================================
// Feature 24: Expected Calibration Error (ECE)
// ===========================================================================

/// Result of expected calibration error analysis.
pub struct EceResult {
    /// Expected calibration error: Σ (n_b/n) |acc_b - conf_b|.
    pub ece: f64,
    /// Maximum calibration error: max |acc_b - conf_b|.
    pub mce: f64,
    /// Adaptive calibration error (equal-mass bins).
    pub ace: f64,
    /// Number of bins used.
    pub n_bins: usize,
    /// Per-bin ECE contributions (length n_bins).
    pub bin_ece_contributions: Vec<f64>,
}

/// Compute expected, maximum, and adaptive calibration errors for a logistic model.
///
/// # Arguments
/// * `fit` — A fitted [`FunctionalLogisticResult`]
/// * `y` — Binary labels (0/1), length n
/// * `n_bins` — Number of bins for equal-width binning
pub fn expected_calibration_error(
    fit: &FunctionalLogisticResult,
    y: &[f64],
    n_bins: usize,
) -> Option<EceResult> {
    let n = fit.probabilities.len();
    if n == 0 || n != y.len() || n_bins == 0 {
        return None;
    }

    let (ece, mce, bin_ece_contributions) =
        compute_equal_width_ece(&fit.probabilities, y, n, n_bins);

    // ACE: equal-mass (quantile) bins
    let mut sorted_idx: Vec<usize> = (0..n).collect();
    sorted_idx.sort_by(|&a, &b| {
        fit.probabilities[a]
            .partial_cmp(&fit.probabilities[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let group_size = n / n_bins.max(1);
    let mut ace = 0.0;
    let mut start = 0;
    for g in 0..n_bins {
        if start >= n {
            break;
        }
        let end = if g < n_bins - 1 {
            (start + group_size).min(n)
        } else {
            n
        };
        ace += calibration_gap_weighted(&sorted_idx[start..end], y, &fit.probabilities, n);
        start = end;
    }

    Some(EceResult {
        ece,
        mce,
        ace,
        n_bins,
        bin_ece_contributions,
    })
}

// ===========================================================================
// Feature 25: Conformal Prediction Residuals
// ===========================================================================

/// Result of split-conformal prediction.
pub struct ConformalPredictionResult {
    /// Predictions on test data (length n_test).
    pub predictions: Vec<f64>,
    /// Lower bounds of prediction intervals (length n_test).
    pub lower: Vec<f64>,
    /// Upper bounds of prediction intervals (length n_test).
    pub upper: Vec<f64>,
    /// Quantile of calibration residuals.
    pub residual_quantile: f64,
    /// Empirical coverage on the calibration set.
    pub coverage: f64,
    /// Absolute residuals on calibration set.
    pub calibration_scores: Vec<f64>,
}

/// Split-conformal prediction intervals for a linear functional regression.
///
/// Randomly splits training data into proper-train and calibration subsets,
/// refits the model, and constructs distribution-free prediction intervals.
///
/// # Arguments
/// * `fit` — Original [`FregreLmResult`] (used for ncomp)
/// * `train_data` — Training functional data (n × m)
/// * `train_y` — Training response (length n)
/// * `test_data` — Test functional data (n_test × m)
/// * `scalar_covariates_train` — Optional scalar covariates for training
/// * `scalar_covariates_test` — Optional scalar covariates for test
/// * `cal_fraction` — Fraction of training data for calibration (0, 1)
/// * `alpha` — Miscoverage level (e.g. 0.1 for 90% intervals)
/// * `seed` — Random seed
pub fn conformal_prediction_residuals(
    fit: &FregreLmResult,
    train_data: &FdMatrix,
    train_y: &[f64],
    test_data: &FdMatrix,
    scalar_covariates_train: Option<&FdMatrix>,
    scalar_covariates_test: Option<&FdMatrix>,
    cal_fraction: f64,
    alpha: f64,
    seed: u64,
) -> Option<ConformalPredictionResult> {
    let (n, m) = train_data.shape();
    let (n_test, m_test) = test_data.shape();
    let ncomp = fit.ncomp;
    let (_n_cal, n_proper) = validate_conformal_inputs(
        n,
        m,
        n_test,
        m_test,
        train_y.len(),
        ncomp,
        cal_fraction,
        alpha,
    )?;

    // Random split
    let mut rng = StdRng::seed_from_u64(seed);
    let mut all_idx: Vec<usize> = (0..n).collect();
    all_idx.shuffle(&mut rng);
    let proper_idx = &all_idx[..n_proper];
    let cal_idx = &all_idx[n_proper..];

    // Subsample data
    let proper_data = subsample_rows(train_data, proper_idx);
    let proper_y: Vec<f64> = proper_idx.iter().map(|&i| train_y[i]).collect();
    let proper_sc = scalar_covariates_train.map(|sc| subsample_rows(sc, proper_idx));

    // Refit on proper-train
    let refit = fregre_lm(&proper_data, &proper_y, proper_sc.as_ref(), ncomp)?;

    // Predict on calibration set
    let cal_data = subsample_rows(train_data, cal_idx);
    let cal_sc = scalar_covariates_train.map(|sc| subsample_rows(sc, cal_idx));
    let cal_scores = project_scores(&cal_data, &refit.fpca.mean, &refit.fpca.rotation, ncomp);
    let cal_preds = predict_from_scores(
        &cal_scores,
        &refit.coefficients,
        &refit.gamma,
        cal_sc.as_ref(),
        ncomp,
    );
    let cal_n = cal_idx.len();

    let calibration_scores: Vec<f64> = cal_idx
        .iter()
        .enumerate()
        .map(|(i, &orig)| (train_y[orig] - cal_preds[i]).abs())
        .collect();

    let (residual_quantile, coverage) =
        conformal_quantile_and_coverage(&calibration_scores, cal_n, alpha);

    // Predict on test data
    let test_scores = project_scores(test_data, &refit.fpca.mean, &refit.fpca.rotation, ncomp);
    let predictions = predict_from_scores(
        &test_scores,
        &refit.coefficients,
        &refit.gamma,
        scalar_covariates_test,
        ncomp,
    );

    let lower: Vec<f64> = predictions.iter().map(|&p| p - residual_quantile).collect();
    let upper: Vec<f64> = predictions.iter().map(|&p| p + residual_quantile).collect();

    Some(ConformalPredictionResult {
        predictions,
        lower,
        upper,
        residual_quantile,
        coverage,
        calibration_scores,
    })
}

// ===========================================================================
// Feature 26: Regression Depth
// ===========================================================================

/// Type of functional depth measure for regression diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthType {
    FraimanMuniz,
    ModifiedBand,
    FunctionalSpatial,
}

/// Result of regression depth analysis.
pub struct RegressionDepthResult {
    /// Depth of β̂ in bootstrap distribution.
    pub beta_depth: f64,
    /// Depth of each observation's FPC scores (length n).
    pub score_depths: Vec<f64>,
    /// Mean of score_depths.
    pub mean_score_depth: f64,
    /// Depth method used.
    pub depth_type: DepthType,
    /// Number of successful bootstrap refits.
    pub n_boot_success: usize,
}

/// Compute depth of a single row among a reference matrix using the specified depth type.
fn compute_single_depth(row: &FdMatrix, reference: &FdMatrix, depth_type: DepthType) -> f64 {
    let depths = match depth_type {
        DepthType::FraimanMuniz => depth::fraiman_muniz_1d(row, reference, false),
        DepthType::ModifiedBand => depth::modified_band_1d(row, reference),
        DepthType::FunctionalSpatial => depth::functional_spatial_1d(row, reference, None),
    };
    if depths.is_empty() {
        0.0
    } else {
        depths[0]
    }
}

/// Regression depth diagnostics for a linear functional regression.
///
/// Computes depth of each observation's FPC scores and depth of the
/// regression coefficients in a bootstrap distribution.
///
/// # Arguments
/// * `fit` — Fitted [`FregreLmResult`]
/// * `data` — Functional data (n × m)
/// * `y` — Response (length n)
/// * `scalar_covariates` — Optional scalar covariates
/// * `n_boot` — Number of bootstrap iterations
/// * `depth_type` — Which depth measure to use
/// * `seed` — Random seed
pub fn regression_depth(
    fit: &FregreLmResult,
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n_boot: usize,
    depth_type: DepthType,
    seed: u64,
) -> Option<RegressionDepthResult> {
    let (n, m) = data.shape();
    if n < 4 || m == 0 || n != y.len() || n_boot == 0 {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let score_depths = compute_score_depths(&scores, depth_type);
    if score_depths.is_empty() {
        return None;
    }
    let mean_score_depth = score_depths.iter().sum::<f64>() / score_depths.len() as f64;

    let orig_coefs: Vec<f64> = (0..ncomp).map(|k| fit.coefficients[1 + k]).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut boot_coefs = Vec::new();
    for _ in 0..n_boot {
        let idx: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
        let boot_data = subsample_rows(data, &idx);
        let boot_y: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
        let boot_sc = scalar_covariates.map(|sc| subsample_rows(sc, &idx));
        if let Some(refit) = fregre_lm(&boot_data, &boot_y, boot_sc.as_ref(), ncomp) {
            boot_coefs.push((0..ncomp).map(|k| refit.coefficients[1 + k]).collect());
        }
    }

    let beta_depth = beta_depth_from_bootstrap(&boot_coefs, &orig_coefs, ncomp, depth_type);

    Some(RegressionDepthResult {
        beta_depth,
        score_depths,
        mean_score_depth,
        depth_type,
        n_boot_success: boot_coefs.len(),
    })
}

/// Regression depth diagnostics for a functional logistic regression.
pub fn regression_depth_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n_boot: usize,
    depth_type: DepthType,
    seed: u64,
) -> Option<RegressionDepthResult> {
    let (n, m) = data.shape();
    if n < 4 || m == 0 || n != y.len() || n_boot == 0 {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let score_depths = compute_score_depths(&scores, depth_type);
    if score_depths.is_empty() {
        return None;
    }
    let mean_score_depth = score_depths.iter().sum::<f64>() / score_depths.len() as f64;

    let orig_coefs: Vec<f64> = (0..ncomp).map(|k| fit.coefficients[1 + k]).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    let boot_coefs =
        bootstrap_logistic_coefs(data, y, scalar_covariates, n, ncomp, n_boot, &mut rng);

    let beta_depth = beta_depth_from_bootstrap(&boot_coefs, &orig_coefs, ncomp, depth_type);

    Some(RegressionDepthResult {
        beta_depth,
        score_depths,
        mean_score_depth,
        depth_type,
        n_boot_success: boot_coefs.len(),
    })
}

// ===========================================================================
// Feature 27: Stability / Robustness Analysis
// ===========================================================================

/// Result of bootstrap stability analysis.
pub struct StabilityAnalysisResult {
    /// Pointwise std of β(t) across bootstraps (length m).
    pub beta_t_std: Vec<f64>,
    /// Std of FPC coefficients γ_k across bootstraps (length ncomp).
    pub coefficient_std: Vec<f64>,
    /// Std of R² or accuracy across bootstraps.
    pub metric_std: f64,
    /// Coefficient of variation of β(t): std / |mean| (length m).
    pub beta_t_cv: Vec<f64>,
    /// Mean Spearman rank correlation of FPC importance rankings.
    pub importance_stability: f64,
    /// Number of successful bootstrap refits.
    pub n_boot_success: usize,
}

/// Bootstrap stability analysis of a linear functional regression.
///
/// Refits the model on `n_boot` bootstrap samples and reports variability
/// of β(t), FPC coefficients, R², and importance rankings.
pub fn explanation_stability(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    n_boot: usize,
    seed: u64,
) -> Option<StabilityAnalysisResult> {
    let (n, m) = data.shape();
    if n < 4 || m == 0 || n != y.len() || n_boot < 2 || ncomp == 0 {
        return None;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut all_beta_t: Vec<Vec<f64>> = Vec::new();
    let mut all_coefs: Vec<Vec<f64>> = Vec::new();
    let mut all_metrics: Vec<f64> = Vec::new();
    let mut all_abs_coefs: Vec<Vec<f64>> = Vec::new();

    for _ in 0..n_boot {
        let idx: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
        let boot_data = subsample_rows(data, &idx);
        let boot_y: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
        let boot_sc = scalar_covariates.map(|sc| subsample_rows(sc, &idx));
        if let Some(refit) = fregre_lm(&boot_data, &boot_y, boot_sc.as_ref(), ncomp) {
            all_beta_t.push(refit.beta_t.clone());
            let coefs: Vec<f64> = (0..ncomp).map(|k| refit.coefficients[1 + k]).collect();
            all_abs_coefs.push(coefs.iter().map(|c| c.abs()).collect());
            all_coefs.push(coefs);
            all_metrics.push(refit.r_squared);
        }
    }

    build_stability_result(
        &all_beta_t,
        &all_coefs,
        &all_abs_coefs,
        &all_metrics,
        m,
        ncomp,
    )
}

/// Bootstrap stability analysis of a functional logistic regression.
pub fn explanation_stability_logistic(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    n_boot: usize,
    seed: u64,
) -> Option<StabilityAnalysisResult> {
    let (n, m) = data.shape();
    if n < 4 || m == 0 || n != y.len() || n_boot < 2 || ncomp == 0 {
        return None;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let (all_beta_t, all_coefs, all_abs_coefs, all_metrics) =
        bootstrap_logistic_stability(data, y, scalar_covariates, n, ncomp, n_boot, &mut rng);

    build_stability_result(
        &all_beta_t,
        &all_coefs,
        &all_abs_coefs,
        &all_metrics,
        m,
        ncomp,
    )
}

// ===========================================================================
// Feature 28: Anchors / Rule Extraction
// ===========================================================================

/// A single condition in an anchor rule.
pub struct AnchorCondition {
    /// FPC component index.
    pub component: usize,
    /// Lower bound on FPC score.
    pub lower_bound: f64,
    /// Upper bound on FPC score.
    pub upper_bound: f64,
}

/// An anchor rule consisting of FPC score conditions.
pub struct AnchorRule {
    /// Conditions forming the rule (conjunction).
    pub conditions: Vec<AnchorCondition>,
    /// Precision: fraction of matching observations with same prediction.
    pub precision: f64,
    /// Coverage: fraction of all observations matching the rule.
    pub coverage: f64,
    /// Number of observations matching the rule.
    pub n_matching: usize,
}

/// Result of anchor explanation for one observation.
pub struct AnchorResult {
    /// The anchor rule.
    pub rule: AnchorRule,
    /// Which observation was explained.
    pub observation: usize,
    /// Predicted value for the observation.
    pub predicted_value: f64,
}

/// Anchor explanation for a linear functional regression.
///
/// Uses beam search in FPC score space to find a minimal set of conditions
/// (score bin memberships) that locally "anchor" the prediction.
///
/// # Arguments
/// * `fit` — Fitted [`FregreLmResult`]
/// * `data` — Functional data (n × m)
/// * `scalar_covariates` — Optional scalar covariates
/// * `observation` — Index of observation to explain
/// * `precision_threshold` — Minimum precision (e.g. 0.95)
/// * `n_bins` — Number of quantile bins per FPC dimension
pub fn anchor_explanation(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    precision_threshold: f64,
    n_bins: usize,
) -> Option<AnchorResult> {
    let (n, m) = data.shape();
    if n == 0 || m != fit.fpca.mean.len() || observation >= n || n_bins < 2 {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let obs_pred = fit.fitted_values[observation];
    let tol = fit.residual_se;

    // "Same prediction" for regression: within ±1 residual_se
    let same_pred = |i: usize| -> bool {
        let mut yhat = fit.coefficients[0];
        for k in 0..ncomp {
            yhat += fit.coefficients[1 + k] * scores[(i, k)];
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..fit.gamma.len() {
                yhat += fit.gamma[j] * sc[(i, j)];
            }
        }
        (yhat - obs_pred).abs() <= tol
    };

    let (rule, _) = anchor_beam_search(
        &scores,
        ncomp,
        n,
        observation,
        precision_threshold,
        n_bins,
        &same_pred,
    );

    Some(AnchorResult {
        rule,
        observation,
        predicted_value: obs_pred,
    })
}

/// Anchor explanation for a functional logistic regression.
///
/// "Same prediction" = same predicted class.
pub fn anchor_explanation_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    precision_threshold: f64,
    n_bins: usize,
) -> Option<AnchorResult> {
    let (n, m) = data.shape();
    if n == 0 || m != fit.fpca.mean.len() || observation >= n || n_bins < 2 {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let obs_class = fit.predicted_classes[observation];
    let obs_prob = fit.probabilities[observation];
    let p_scalar = fit.gamma.len();

    // "Same prediction" = same class
    let same_pred = |i: usize| -> bool {
        let mut eta = fit.intercept;
        for k in 0..ncomp {
            eta += fit.coefficients[1 + k] * scores[(i, k)];
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..p_scalar {
                eta += fit.gamma[j] * sc[(i, j)];
            }
        }
        let pred_class = if sigmoid(eta) >= 0.5 { 1u8 } else { 0u8 };
        pred_class == obs_class
    };

    let (rule, _) = anchor_beam_search(
        &scores,
        ncomp,
        n,
        observation,
        precision_threshold,
        n_bins,
        &same_pred,
    );

    Some(AnchorResult {
        rule,
        observation,
        predicted_value: obs_prob,
    })
}

/// Evaluate a candidate condition: add component to current matching and compute precision.
fn evaluate_anchor_candidate(
    current_matching: &[bool],
    scores: &FdMatrix,
    component: usize,
    bin: usize,
    edges: &[f64],
    n_bins: usize,
    same_pred: &dyn Fn(usize) -> bool,
) -> Option<(f64, Vec<bool>)> {
    let new_matching = apply_bin_filter(current_matching, scores, component, bin, edges, n_bins);
    let n_match = new_matching.iter().filter(|&&v| v).count();
    if n_match == 0 {
        return None;
    }
    let n_same = (0..new_matching.len())
        .filter(|&i| new_matching[i] && same_pred(i))
        .count();
    Some((n_same as f64 / n_match as f64, new_matching))
}

/// Build an AnchorRule from selected components, bin edges, and observation bins.
fn build_anchor_rule(
    components: &[usize],
    bin_edges: &[Vec<f64>],
    obs_bins: &[usize],
    precision: f64,
    matching: &[bool],
    n: usize,
) -> AnchorRule {
    let conditions: Vec<AnchorCondition> = components
        .iter()
        .map(|&k| AnchorCondition {
            component: k,
            lower_bound: bin_edges[k][obs_bins[k]],
            upper_bound: bin_edges[k][obs_bins[k] + 1],
        })
        .collect();
    let n_match = matching.iter().filter(|&&v| v).count();
    AnchorRule {
        conditions,
        precision,
        coverage: n_match as f64 / n as f64,
        n_matching: n_match,
    }
}

/// Compute column means of an FdMatrix.
pub(crate) fn compute_column_means(mat: &FdMatrix, ncols: usize) -> Vec<f64> {
    let n = mat.nrows();
    let mut means = vec![0.0; ncols];
    for k in 0..ncols {
        for i in 0..n {
            means[k] += mat[(i, k)];
        }
        means[k] /= n as f64;
    }
    means
}

/// Compute mean scalar covariates from an optional FdMatrix.
pub(crate) fn compute_mean_scalar(
    scalar_covariates: Option<&FdMatrix>,
    p_scalar: usize,
    n: usize,
) -> Vec<f64> {
    if p_scalar == 0 {
        return vec![];
    }
    if let Some(sc) = scalar_covariates {
        (0..p_scalar)
            .map(|j| {
                let mut s = 0.0;
                for i in 0..n {
                    s += sc[(i, j)];
                }
                s / n as f64
            })
            .collect()
    } else {
        vec![0.0; p_scalar]
    }
}

/// Compute Shapley kernel weight for a coalition of given size.
pub(crate) fn shapley_kernel_weight(ncomp: usize, s_size: usize) -> f64 {
    if s_size == 0 || s_size == ncomp {
        1e6
    } else {
        let binom = binom_coeff(ncomp, s_size) as f64;
        if binom > 0.0 {
            (ncomp - 1) as f64 / (binom * s_size as f64 * (ncomp - s_size) as f64)
        } else {
            1.0
        }
    }
}

/// Sample a random coalition of FPC components via Fisher-Yates partial shuffle.
pub(crate) fn sample_random_coalition(rng: &mut StdRng, ncomp: usize) -> (Vec<bool>, usize) {
    let s_size = if ncomp <= 1 {
        rng.gen_range(0..=1usize)
    } else {
        rng.gen_range(1..ncomp)
    };
    let mut coalition = vec![false; ncomp];
    let mut indices: Vec<usize> = (0..ncomp).collect();
    for j in 0..s_size.min(ncomp) {
        let swap = rng.gen_range(j..ncomp);
        indices.swap(j, swap);
    }
    for j in 0..s_size {
        coalition[indices[j]] = true;
    }
    (coalition, s_size)
}

/// Build coalition scores: use observation value if in coalition, mean otherwise.
pub(crate) fn build_coalition_scores(
    coalition: &[bool],
    obs_scores: &[f64],
    mean_scores: &[f64],
) -> Vec<f64> {
    coalition
        .iter()
        .enumerate()
        .map(|(k, &in_coal)| {
            if in_coal {
                obs_scores[k]
            } else {
                mean_scores[k]
            }
        })
        .collect()
}

/// Get observation's scalar covariates, or use mean if unavailable.
pub(crate) fn get_obs_scalar(
    scalar_covariates: Option<&FdMatrix>,
    i: usize,
    p_scalar: usize,
    mean_z: &[f64],
) -> Vec<f64> {
    if p_scalar == 0 {
        return vec![];
    }
    if let Some(sc) = scalar_covariates {
        (0..p_scalar).map(|j| sc[(i, j)]).collect()
    } else {
        mean_z.to_vec()
    }
}

/// Accumulate one WLS sample for Kernel SHAP: A'A += w z z', A'b += w z y.
pub(crate) fn accumulate_kernel_shap_sample(
    ata: &mut [f64],
    atb: &mut [f64],
    coalition: &[bool],
    weight: f64,
    y_val: f64,
    ncomp: usize,
) {
    for k1 in 0..ncomp {
        let z1 = if coalition[k1] { 1.0 } else { 0.0 };
        for k2 in 0..ncomp {
            let z2 = if coalition[k2] { 1.0 } else { 0.0 };
            ata[k1 * ncomp + k2] += weight * z1 * z2;
        }
        atb[k1] += weight * z1 * y_val;
    }
}

/// Compute 1D PDP for a linear model along one component.
fn pdp_1d_linear(
    scores: &FdMatrix,
    coefs: &[f64],
    ncomp: usize,
    component: usize,
    grid: &[f64],
    n: usize,
) -> Vec<f64> {
    grid.iter()
        .map(|&gval| {
            let mut sum = 0.0;
            for i in 0..n {
                let mut yhat = coefs[0];
                for c in 0..ncomp {
                    let s = if c == component { gval } else { scores[(i, c)] };
                    yhat += coefs[1 + c] * s;
                }
                sum += yhat;
            }
            sum / n as f64
        })
        .collect()
}

/// Compute 2D PDP for a linear model along two components.
fn pdp_2d_linear(
    scores: &FdMatrix,
    coefs: &[f64],
    ncomp: usize,
    comp_j: usize,
    comp_k: usize,
    grid_j: &[f64],
    grid_k: &[f64],
    n: usize,
    n_grid: usize,
) -> FdMatrix {
    let mut pdp_2d = FdMatrix::zeros(n_grid, n_grid);
    for (gj_idx, &gj) in grid_j.iter().enumerate() {
        for (gk_idx, &gk) in grid_k.iter().enumerate() {
            let replacements = [(comp_j, gj), (comp_k, gk)];
            let mut sum = 0.0;
            for i in 0..n {
                sum += linear_predict_replaced(scores, coefs, ncomp, i, &replacements);
            }
            pdp_2d[(gj_idx, gk_idx)] = sum / n as f64;
        }
    }
    pdp_2d
}

/// Compute H² statistic from 1D and 2D PDPs.
pub(crate) fn compute_h_squared(
    pdp_2d: &FdMatrix,
    pdp_j: &[f64],
    pdp_k: &[f64],
    f_bar: f64,
    n_grid: usize,
) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for gj in 0..n_grid {
        for gk in 0..n_grid {
            let f2 = pdp_2d[(gj, gk)];
            let interaction = f2 - pdp_j[gj] - pdp_k[gk] + f_bar;
            num += interaction * interaction;
            let centered = f2 - f_bar;
            den += centered * centered;
        }
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// Compute conditioning bins for conditional permutation importance.
pub(crate) fn compute_conditioning_bins(
    scores: &FdMatrix,
    ncomp: usize,
    target_k: usize,
    n: usize,
    n_bins: usize,
) -> Vec<Vec<usize>> {
    let mut cond_var: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        for c in 0..ncomp {
            if c != target_k {
                cond_var[i] += scores[(i, c)].abs();
            }
        }
    }

    let mut sorted_cond: Vec<(f64, usize)> =
        cond_var.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    sorted_cond.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let actual_bins = n_bins.min(n);
    let mut bin_assignment = vec![0usize; n];
    for (rank, &(_, idx)) in sorted_cond.iter().enumerate() {
        bin_assignment[idx] = (rank * actual_bins / n).min(actual_bins - 1);
    }

    let mut bins: Vec<Vec<usize>> = vec![vec![]; actual_bins];
    for i in 0..n {
        bins[bin_assignment[i]].push(i);
    }
    bins
}

/// Clone an FdMatrix of scores.
pub(crate) fn clone_scores_matrix(scores: &FdMatrix, n: usize, ncomp: usize) -> FdMatrix {
    let mut perm = FdMatrix::zeros(n, ncomp);
    for i in 0..n {
        for c in 0..ncomp {
            perm[(i, c)] = scores[(i, c)];
        }
    }
    perm
}

/// Shuffle component k within conditional bins.
pub(crate) fn shuffle_within_bins(
    perm_scores: &mut FdMatrix,
    scores: &FdMatrix,
    bins: &[Vec<usize>],
    k: usize,
    rng: &mut StdRng,
) {
    for bin in bins {
        if bin.len() <= 1 {
            continue;
        }
        let mut bin_indices = bin.clone();
        bin_indices.shuffle(rng);
        for (rank, &orig_idx) in bin.iter().enumerate() {
            perm_scores[(orig_idx, k)] = scores[(bin_indices[rank], k)];
        }
    }
}

/// Shuffle component k globally (unconditional).
pub(crate) fn shuffle_global(
    perm_scores: &mut FdMatrix,
    scores: &FdMatrix,
    k: usize,
    n: usize,
    rng: &mut StdRng,
) {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(rng);
    for i in 0..n {
        perm_scores[(i, k)] = scores[(idx[i], k)];
    }
}

/// Run conditional + unconditional permutations for one component and return mean metrics.
pub(crate) fn permute_component<F: Fn(&FdMatrix) -> f64>(
    scores: &FdMatrix,
    bins: &[Vec<usize>],
    k: usize,
    n: usize,
    ncomp: usize,
    n_perm: usize,
    rng: &mut StdRng,
    metric_fn: &F,
) -> (f64, f64) {
    let mut sum_cond = 0.0;
    let mut sum_uncond = 0.0;
    for _ in 0..n_perm {
        let mut perm_cond = clone_scores_matrix(scores, n, ncomp);
        let mut perm_uncond = clone_scores_matrix(scores, n, ncomp);
        shuffle_within_bins(&mut perm_cond, scores, bins, k, rng);
        shuffle_global(&mut perm_uncond, scores, k, n, rng);
        sum_cond += metric_fn(&perm_cond);
        sum_uncond += metric_fn(&perm_uncond);
    }
    (sum_cond / n_perm as f64, sum_uncond / n_perm as f64)
}

/// Greedy MMD-based prototype selection.
pub(crate) fn greedy_prototype_selection(
    mu_data: &[f64],
    kernel: &[f64],
    n: usize,
    n_prototypes: usize,
) -> (Vec<usize>, Vec<bool>) {
    let mut selected: Vec<usize> = Vec::with_capacity(n_prototypes);
    let mut is_selected = vec![false; n];

    for _ in 0..n_prototypes {
        let best_idx = find_best_prototype(mu_data, kernel, n, &is_selected, &selected);
        selected.push(best_idx);
        is_selected[best_idx] = true;
    }
    (selected, is_selected)
}

/// Compute witness function values.
pub(crate) fn compute_witness(
    kernel: &[f64],
    mu_data: &[f64],
    selected: &[usize],
    n: usize,
) -> Vec<f64> {
    let mut witness = vec![0.0; n];
    for i in 0..n {
        let mean_k_selected: f64 =
            selected.iter().map(|&j| kernel[i * n + j]).sum::<f64>() / selected.len() as f64;
        witness[i] = mu_data[i] - mean_k_selected;
    }
    witness
}

/// Compute mean logistic prediction with optional component replacements.
fn logistic_pdp_mean(
    scores: &FdMatrix,
    fit_intercept: f64,
    coefficients: &[f64],
    gamma: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
    ncomp: usize,
    replacements: &[(usize, f64)],
) -> f64 {
    let p_scalar = gamma.len();
    let mut sum = 0.0;
    for i in 0..n {
        let mut eta = fit_intercept;
        for c in 0..ncomp {
            let s = replacements
                .iter()
                .find(|&&(comp, _)| comp == c)
                .map(|&(_, val)| val)
                .unwrap_or(scores[(i, c)]);
            eta += coefficients[1 + c] * s;
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..p_scalar {
                eta += gamma[j] * sc[(i, j)];
            }
        }
        sum += sigmoid(eta);
    }
    sum / n as f64
}

/// Detect significance direction at a single point from CI bounds.
fn detect_direction(lower: f64, upper: f64) -> Option<SignificanceDirection> {
    if lower > 0.0 {
        Some(SignificanceDirection::Positive)
    } else if upper < 0.0 {
        Some(SignificanceDirection::Negative)
    } else {
        None
    }
}

/// Compute base logistic eta for one observation, excluding a given component.
fn logistic_eta_base(
    fit_intercept: f64,
    coefficients: &[f64],
    gamma: &[f64],
    scores: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    i: usize,
    ncomp: usize,
    exclude_component: usize,
) -> f64 {
    let mut eta = fit_intercept;
    for k in 0..ncomp {
        if k != exclude_component {
            eta += coefficients[1 + k] * scores[(i, k)];
        }
    }
    if let Some(sc) = scalar_covariates {
        for j in 0..gamma.len() {
            eta += gamma[j] * sc[(i, j)];
        }
    }
    eta
}

/// Compute column means of ICE curves → PDP.
pub(crate) fn ice_to_pdp(ice_curves: &FdMatrix, n: usize, n_grid: usize) -> Vec<f64> {
    let mut pdp = vec![0.0; n_grid];
    for g in 0..n_grid {
        for i in 0..n {
            pdp[g] += ice_curves[(i, g)];
        }
        pdp[g] /= n as f64;
    }
    pdp
}

/// Compute logistic accuracy from a score matrix.
fn logistic_accuracy_from_scores(
    score_mat: &FdMatrix,
    fit_intercept: f64,
    coefficients: &[f64],
    y: &[f64],
    n: usize,
    ncomp: usize,
) -> f64 {
    let correct: usize = (0..n)
        .filter(|&i| {
            let mut eta = fit_intercept;
            for c in 0..ncomp {
                eta += coefficients[1 + c] * score_mat[(i, c)];
            }
            let pred = if sigmoid(eta) >= 0.5 { 1.0 } else { 0.0 };
            (pred - y[i]).abs() < 1e-10
        })
        .count();
    correct as f64 / n as f64
}

/// Merge overlapping intervals, accumulating importance.
fn merge_overlapping_intervals(raw: Vec<(usize, usize, f64)>) -> Vec<ImportantInterval> {
    let mut intervals: Vec<ImportantInterval> = Vec::new();
    for (s, e, imp) in raw {
        if let Some(last) = intervals.last_mut() {
            if s <= last.end_idx + 1 {
                last.end_idx = e;
                last.importance += imp;
                continue;
            }
        }
        intervals.push(ImportantInterval {
            start_idx: s,
            end_idx: e,
            importance: imp,
        });
    }
    intervals
}

/// Reconstruct delta function from delta scores and rotation matrix.
pub(crate) fn reconstruct_delta_function(
    delta_scores: &[f64],
    rotation: &FdMatrix,
    ncomp: usize,
    m: usize,
) -> Vec<f64> {
    let mut delta_function = vec![0.0; m];
    for j in 0..m {
        for k in 0..ncomp {
            delta_function[j] += delta_scores[k] * rotation[(j, k)];
        }
    }
    delta_function
}

/// Equal-width binning: compute ECE, MCE, and per-bin contributions.
fn compute_equal_width_ece(
    probabilities: &[f64],
    y: &[f64],
    n: usize,
    n_bins: usize,
) -> (f64, f64, Vec<f64>) {
    let mut bin_sum_y = vec![0.0; n_bins];
    let mut bin_sum_p = vec![0.0; n_bins];
    let mut bin_count = vec![0usize; n_bins];

    for i in 0..n {
        let b = ((probabilities[i] * n_bins as f64).floor() as usize).min(n_bins - 1);
        bin_sum_y[b] += y[i];
        bin_sum_p[b] += probabilities[i];
        bin_count[b] += 1;
    }

    let mut ece = 0.0;
    let mut mce: f64 = 0.0;
    let mut bin_ece_contributions = vec![0.0; n_bins];

    for b in 0..n_bins {
        if bin_count[b] == 0 {
            continue;
        }
        let gap = (bin_sum_y[b] / bin_count[b] as f64 - bin_sum_p[b] / bin_count[b] as f64).abs();
        let contrib = bin_count[b] as f64 / n as f64 * gap;
        bin_ece_contributions[b] = contrib;
        ece += contrib;
        if gap > mce {
            mce = gap;
        }
    }

    (ece, mce, bin_ece_contributions)
}

/// Compute coefficient standard errors from Cholesky factor and MSE.
fn compute_coefficient_se(l: &[f64], mse: f64, p: usize) -> Vec<f64> {
    let mut se = vec![0.0; p];
    for j in 0..p {
        let mut ej = vec![0.0; p];
        ej[j] = 1.0;
        let v = cholesky_forward_back(l, &ej, p);
        se[j] = (mse * v[j].max(0.0)).sqrt();
    }
    se
}

/// Compute DFBETAS row, DFFITS, and studentized residual for a single observation.
fn compute_obs_influence(
    design: &FdMatrix,
    l: &[f64],
    residual: f64,
    h_ii: f64,
    s: f64,
    se: &[f64],
    p: usize,
    i: usize,
) -> (f64, f64, Vec<f64>) {
    let one_minus_h = (1.0 - h_ii).max(1e-15);
    let t_i = residual / (s * one_minus_h.sqrt());
    let dffits_i = t_i * (h_ii / one_minus_h).sqrt();

    let mut xi = vec![0.0; p];
    for j in 0..p {
        xi[j] = design[(i, j)];
    }
    let inv_xtx_xi = cholesky_forward_back(l, &xi, p);
    let mut dfb = vec![0.0; p];
    for j in 0..p {
        if se[j] > 1e-15 {
            dfb[j] = inv_xtx_xi[j] * residual / (one_minus_h * se[j]);
        }
    }

    (t_i, dffits_i, dfb)
}

/// Weighted R² from predictions, fitted values, and weights.
fn weighted_r_squared(
    predictions: &[f64],
    beta: &[f64],
    perturbed: &[Vec<f64>],
    obs_scores: &[f64],
    weights: &[f64],
    ncomp: usize,
    n_samples: usize,
) -> f64 {
    let w_sum: f64 = weights.iter().sum();
    let w_mean_y: f64 = weights
        .iter()
        .zip(predictions)
        .map(|(&w, &y)| w * y)
        .sum::<f64>()
        / w_sum;

    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    for i in 0..n_samples {
        let mut yhat = beta[0];
        for k in 0..ncomp {
            yhat += beta[1 + k] * (perturbed[i][k] - obs_scores[k]);
        }
        ss_tot += weights[i] * (predictions[i] - w_mean_y).powi(2);
        ss_res += weights[i] * (predictions[i] - yhat).powi(2);
    }

    if ss_tot > 0.0 {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Compute linear prediction with optional component replacements.
fn linear_predict_replaced(
    scores: &FdMatrix,
    coefs: &[f64],
    ncomp: usize,
    i: usize,
    replacements: &[(usize, f64)],
) -> f64 {
    let mut yhat = coefs[0];
    for c in 0..ncomp {
        let s = replacements
            .iter()
            .find(|&&(comp, _)| comp == c)
            .map_or(scores[(i, c)], |&(_, val)| val);
        yhat += coefs[1 + c] * s;
    }
    yhat
}

/// Logistic predict from FPC scores: eta = intercept + Σ coef[1+k] * scores[k], return sigmoid(eta).
fn logistic_predict_from_scores(
    intercept: f64,
    coefficients: &[f64],
    scores: &[f64],
    ncomp: usize,
) -> f64 {
    let mut eta = intercept;
    for k in 0..ncomp {
        eta += coefficients[1 + k] * scores[k];
    }
    sigmoid(eta)
}

/// Evaluate all unused components in beam search and return sorted candidates.
fn beam_search_candidates(
    scores: &FdMatrix,
    ncomp: usize,
    used: &[bool],
    obs_bins: &[usize],
    bin_edges: &[Vec<f64>],
    n_bins: usize,
    best_conditions: &[usize],
    best_matching: &[bool],
    same_pred: &dyn Fn(usize) -> bool,
    beam_width: usize,
) -> Vec<(Vec<usize>, f64, Vec<bool>)> {
    let mut candidates: Vec<(Vec<usize>, f64, Vec<bool>)> = Vec::new();

    for k in 0..ncomp {
        if used[k] {
            continue;
        }
        if let Some((precision, matching)) = evaluate_anchor_candidate(
            best_matching,
            scores,
            k,
            obs_bins[k],
            &bin_edges[k],
            n_bins,
            same_pred,
        ) {
            let mut conds = best_conditions.to_vec();
            conds.push(k);
            candidates.push((conds, precision, matching));
        }
    }

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(beam_width);
    candidates
}

/// Compute saliency map: saliency[(i,j)] = Σ_k weight_k × (scores[(i,k)] - mean_k) × rotation[(j,k)].
pub(crate) fn compute_saliency_map(
    scores: &FdMatrix,
    mean_scores: &[f64],
    weights: &[f64],
    rotation: &FdMatrix,
    n: usize,
    m: usize,
    ncomp: usize,
) -> FdMatrix {
    let mut saliency_map = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            let mut val = 0.0;
            for k in 0..ncomp {
                val += weights[k] * (scores[(i, k)] - mean_scores[k]) * rotation[(j, k)];
            }
            saliency_map[(i, j)] = val;
        }
    }
    saliency_map
}

/// Mean absolute value per column of an n×m matrix.
pub(crate) fn mean_absolute_column(mat: &FdMatrix, n: usize, m: usize) -> Vec<f64> {
    let mut result = vec![0.0; m];
    for j in 0..m {
        for i in 0..n {
            result[j] += mat[(i, j)].abs();
        }
        result[j] /= n as f64;
    }
    result
}

/// Compute SS_res with component k shuffled by given index permutation.
fn permuted_ss_res_linear(
    scores: &FdMatrix,
    coefficients: &[f64],
    y: &[f64],
    n: usize,
    ncomp: usize,
    k: usize,
    perm_idx: &[usize],
) -> f64 {
    (0..n)
        .map(|i| {
            let mut yhat = coefficients[0];
            for c in 0..ncomp {
                let s = if c == k {
                    scores[(perm_idx[i], c)]
                } else {
                    scores[(i, c)]
                };
                yhat += coefficients[1 + c] * s;
            }
            (y[i] - yhat).powi(2)
        })
        .sum()
}

/// Compute 2D logistic PDP on a grid using logistic_pdp_mean.
fn logistic_pdp_2d(
    scores: &FdMatrix,
    intercept: f64,
    coefficients: &[f64],
    gamma: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
    ncomp: usize,
    comp_j: usize,
    comp_k: usize,
    grid_j: &[f64],
    grid_k: &[f64],
    n_grid: usize,
) -> FdMatrix {
    let mut pdp_2d = FdMatrix::zeros(n_grid, n_grid);
    for (gj_idx, &gj) in grid_j.iter().enumerate() {
        for (gk_idx, &gk) in grid_k.iter().enumerate() {
            pdp_2d[(gj_idx, gk_idx)] = logistic_pdp_mean(
                scores,
                intercept,
                coefficients,
                gamma,
                scalar_covariates,
                n,
                ncomp,
                &[(comp_j, gj), (comp_k, gk)],
            );
        }
    }
    pdp_2d
}

/// Run logistic counterfactual gradient descent: returns (scores, prediction, found).
fn logistic_counterfactual_descent(
    intercept: f64,
    coefficients: &[f64],
    initial_scores: &[f64],
    target_class: i32,
    ncomp: usize,
    max_iter: usize,
    step_size: f64,
) -> (Vec<f64>, f64, bool) {
    let mut current_scores = initial_scores.to_vec();
    let mut current_pred =
        logistic_predict_from_scores(intercept, coefficients, &current_scores, ncomp);

    for _ in 0..max_iter {
        current_pred =
            logistic_predict_from_scores(intercept, coefficients, &current_scores, ncomp);
        let current_class = if current_pred >= 0.5 { 1 } else { 0 };
        if current_class == target_class {
            return (current_scores, current_pred, true);
        }
        for k in 0..ncomp {
            // Cross-entropy gradient: dL/ds_k = (p - target) * coef_k
            // The sigmoid derivative p*(1-p) cancels with the cross-entropy denominator.
            let grad = (current_pred - target_class as f64) * coefficients[1 + k];
            current_scores[k] -= step_size * grad;
        }
    }
    (current_scores, current_pred, false)
}

/// Generate Sobol A and B matrices by resampling from FPC scores.
pub(crate) fn generate_sobol_matrices(
    scores: &FdMatrix,
    n: usize,
    ncomp: usize,
    n_samples: usize,
    rng: &mut StdRng,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut mat_a = vec![vec![0.0; ncomp]; n_samples];
    let mut mat_b = vec![vec![0.0; ncomp]; n_samples];
    for i in 0..n_samples {
        let ia = rng.gen_range(0..n);
        let ib = rng.gen_range(0..n);
        for k in 0..ncomp {
            mat_a[i][k] = scores[(ia, k)];
            mat_b[i][k] = scores[(ib, k)];
        }
    }
    (mat_a, mat_b)
}

/// Compute first-order and total-order Sobol indices for one component.
pub(crate) fn compute_sobol_component(
    mat_a: &[Vec<f64>],
    mat_b: &[Vec<f64>],
    f_a: &[f64],
    f_b: &[f64],
    var_fa: f64,
    k: usize,
    n_samples: usize,
    eval_model: &dyn Fn(&[f64]) -> f64,
) -> (f64, f64) {
    let f_ab_k: Vec<f64> = (0..n_samples)
        .map(|i| {
            let mut s = mat_a[i].clone();
            s[k] = mat_b[i][k];
            eval_model(&s)
        })
        .collect();

    let s_k: f64 = (0..n_samples)
        .map(|i| f_b[i] * (f_ab_k[i] - f_a[i]))
        .sum::<f64>()
        / n_samples as f64
        / var_fa;

    let st_k: f64 = (0..n_samples)
        .map(|i| (f_a[i] - f_ab_k[i]).powi(2))
        .sum::<f64>()
        / (2.0 * n_samples as f64 * var_fa);

    (s_k, st_k)
}

/// Construct Hosmer-Lemeshow groups and compute chi², reliability bins, and counts.
fn hosmer_lemeshow_computation(
    probabilities: &[f64],
    y: &[f64],
    n: usize,
    n_groups: usize,
) -> (f64, Vec<(f64, f64)>, Vec<usize>) {
    let mut sorted_idx: Vec<usize> = (0..n).collect();
    sorted_idx.sort_by(|&a, &b| {
        probabilities[a]
            .partial_cmp(&probabilities[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let group_size = n / n_groups;
    let remainder = n % n_groups;
    let mut start = 0;

    let mut chi2 = 0.0;
    let mut reliability_bins = Vec::with_capacity(n_groups);
    let mut bin_counts = Vec::with_capacity(n_groups);

    for g in 0..n_groups {
        let sz = group_size + if g < remainder { 1 } else { 0 };
        let group = &sorted_idx[start..start + sz];
        start += sz;

        let ng = group.len();
        if ng == 0 {
            continue;
        }
        let o_g: f64 = group.iter().map(|&i| y[i]).sum();
        let e_g: f64 = group.iter().map(|&i| probabilities[i]).sum();
        let p_bar = e_g / ng as f64;
        let mean_obs = o_g / ng as f64;

        reliability_bins.push((p_bar, mean_obs));
        bin_counts.push(ng);

        let denom = ng as f64 * p_bar * (1.0 - p_bar);
        if denom > 1e-15 {
            chi2 += (o_g - e_g).powi(2) / denom;
        }
    }

    (chi2, reliability_bins, bin_counts)
}

/// Bootstrap logistic stability: collect beta_t, coefs, abs_coefs, and metrics.
fn bootstrap_logistic_stability(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
    ncomp: usize,
    n_boot: usize,
    rng: &mut StdRng,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    let mut all_beta_t: Vec<Vec<f64>> = Vec::new();
    let mut all_coefs: Vec<Vec<f64>> = Vec::new();
    let mut all_abs_coefs: Vec<Vec<f64>> = Vec::new();
    let mut all_metrics: Vec<f64> = Vec::new();

    for _ in 0..n_boot {
        let idx: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
        let boot_data = subsample_rows(data, &idx);
        let boot_y: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
        let boot_sc = scalar_covariates.map(|sc| subsample_rows(sc, &idx));
        let has_both = boot_y.iter().any(|&v| v < 0.5) && boot_y.iter().any(|&v| v >= 0.5);
        if !has_both {
            continue;
        }
        if let Some(refit) =
            functional_logistic(&boot_data, &boot_y, boot_sc.as_ref(), ncomp, 25, 1e-6)
        {
            all_beta_t.push(refit.beta_t.clone());
            let coefs: Vec<f64> = (0..ncomp).map(|k| refit.coefficients[1 + k]).collect();
            all_abs_coefs.push(coefs.iter().map(|c| c.abs()).collect());
            all_coefs.push(coefs);
            all_metrics.push(refit.accuracy);
        }
    }

    (all_beta_t, all_coefs, all_abs_coefs, all_metrics)
}

/// Compute median pairwise distance from FPC scores (bandwidth heuristic).
pub(crate) fn median_bandwidth(scores: &FdMatrix, n: usize, ncomp: usize) -> f64 {
    let mut dists: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d2 = 0.0;
            for c in 0..ncomp {
                let d = scores[(i, c)] - scores[(j, c)];
                d2 += d * d;
            }
            dists.push(d2.sqrt());
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if dists.is_empty() {
        1.0
    } else {
        dists[dists.len() / 2].max(1e-10)
    }
}

/// Compute kernel mean: mu_data[i] = (1/n) Σ_j K(i,j).
pub(crate) fn compute_kernel_mean(kernel: &[f64], n: usize) -> Vec<f64> {
    let mut mu_data = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            mu_data[i] += kernel[i * n + j];
        }
        mu_data[i] /= n as f64;
    }
    mu_data
}

/// Find the best unselected prototype candidate.
fn find_best_prototype(
    mu_data: &[f64],
    kernel: &[f64],
    n: usize,
    is_selected: &[bool],
    selected: &[usize],
) -> usize {
    let mut best_idx = 0;
    let mut best_val = f64::NEG_INFINITY;
    for i in 0..n {
        if is_selected[i] {
            continue;
        }
        let mut score = 2.0 * mu_data[i];
        if !selected.is_empty() {
            let mean_k: f64 =
                selected.iter().map(|&j| kernel[i * n + j]).sum::<f64>() / selected.len() as f64;
            score -= mean_k;
        }
        if score > best_val {
            best_val = score;
            best_idx = i;
        }
    }
    best_idx
}

/// Sample LIME perturbations, compute predictions and kernel weights.
/// Returns None if Normal distribution creation fails.
fn sample_lime_perturbations(
    obs_scores: &[f64],
    score_sd: &[f64],
    ncomp: usize,
    n_samples: usize,
    kernel_width: f64,
    rng: &mut StdRng,
    predict: &dyn Fn(&[f64]) -> f64,
) -> Option<(Vec<Vec<f64>>, Vec<f64>, Vec<f64>)> {
    let mut perturbed = vec![vec![0.0; ncomp]; n_samples];
    let mut predictions = vec![0.0; n_samples];
    let mut weights = vec![0.0; n_samples];

    for i in 0..n_samples {
        let mut dist_sq = 0.0;
        for k in 0..ncomp {
            let normal = Normal::new(obs_scores[k], score_sd[k]).ok()?;
            perturbed[i][k] = rng.sample(normal);
            let d = perturbed[i][k] - obs_scores[k];
            dist_sq += d * d;
        }
        predictions[i] = predict(&perturbed[i]);
        weights[i] = (-dist_sq / (2.0 * kernel_width * kernel_width)).exp();
    }
    Some((perturbed, predictions, weights))
}

/// Compute conformal calibration quantile and coverage from absolute residuals.
fn conformal_quantile_and_coverage(
    calibration_scores: &[f64],
    cal_n: usize,
    alpha: f64,
) -> (f64, f64) {
    let q_level = (((cal_n + 1) as f64 * (1.0 - alpha)).ceil() / cal_n as f64).min(1.0);
    let mut sorted_scores = calibration_scores.to_vec();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let residual_quantile = quantile_sorted(&sorted_scores, q_level);

    let coverage = calibration_scores
        .iter()
        .filter(|&&s| s <= residual_quantile)
        .count() as f64
        / cal_n as f64;

    (residual_quantile, coverage)
}

/// Compute score variance for each component (mean-zero scores from FPCA).
pub(crate) fn compute_score_variance(scores: &FdMatrix, n: usize, ncomp: usize) -> Vec<f64> {
    let mut score_variance = vec![0.0; ncomp];
    for k in 0..ncomp {
        let mut ss = 0.0;
        for i in 0..n {
            let s = scores[(i, k)];
            ss += s * s;
        }
        score_variance[k] = ss / (n - 1) as f64;
    }
    score_variance
}

/// Compute component importance matrix and aggregated importance.
fn compute_pointwise_importance_core(
    coefficients: &[f64],
    rotation: &FdMatrix,
    score_variance: &[f64],
    ncomp: usize,
    m: usize,
) -> (FdMatrix, Vec<f64>, Vec<f64>) {
    let mut component_importance = FdMatrix::zeros(ncomp, m);
    for k in 0..ncomp {
        let ck = coefficients[1 + k];
        for j in 0..m {
            component_importance[(k, j)] = (ck * rotation[(j, k)]).powi(2) * score_variance[k];
        }
    }

    let mut importance = vec![0.0; m];
    for j in 0..m {
        for k in 0..ncomp {
            importance[j] += component_importance[(k, j)];
        }
    }

    let total: f64 = importance.iter().sum();
    let importance_normalized = if total > 0.0 {
        importance.iter().map(|&v| v / total).collect()
    } else {
        vec![0.0; m]
    };

    (component_importance, importance, importance_normalized)
}

/// Compute prediction interval for a single observation.
fn compute_prediction_interval_obs(
    l: &[f64],
    coefficients: &[f64],
    x_new: &[f64],
    p: usize,
    residual_se: f64,
    t_crit: f64,
) -> (f64, f64, f64, f64) {
    let yhat: f64 = x_new.iter().zip(coefficients).map(|(a, b)| a * b).sum();
    let v = cholesky_forward_back(l, x_new, p);
    let h_new: f64 = x_new.iter().zip(&v).map(|(a, b)| a * b).sum();
    let pred_se = residual_se * (1.0 + h_new).sqrt();
    (
        yhat,
        yhat - t_crit * pred_se,
        yhat + t_crit * pred_se,
        pred_se,
    )
}

/// Build design matrix without intercept: scores + optional scalars.
fn build_no_intercept_matrix(
    scores: &FdMatrix,
    ncomp: usize,
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
) -> FdMatrix {
    let p_scalar = scalar_covariates.map_or(0, |sc| sc.ncols());
    let p = ncomp + p_scalar;
    let mut x = FdMatrix::zeros(n, p);
    for i in 0..n {
        for k in 0..ncomp {
            x[(i, k)] = scores[(i, k)];
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..p_scalar {
                x[(i, ncomp + j)] = sc[(i, j)];
            }
        }
    }
    x
}

/// Bootstrap logistic model coefficients by resampling with replacement.
fn bootstrap_logistic_coefs(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
    ncomp: usize,
    n_boot: usize,
    rng: &mut StdRng,
) -> Vec<Vec<f64>> {
    let mut boot_coefs = Vec::new();
    for _ in 0..n_boot {
        let idx: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
        let boot_data = subsample_rows(data, &idx);
        let boot_y: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
        let boot_sc = scalar_covariates.map(|sc| subsample_rows(sc, &idx));
        let has_both = boot_y.iter().any(|&v| v < 0.5) && boot_y.iter().any(|&v| v >= 0.5);
        if !has_both {
            continue;
        }
        if let Some(refit) =
            functional_logistic(&boot_data, &boot_y, boot_sc.as_ref(), ncomp, 25, 1e-6)
        {
            boot_coefs.push((0..ncomp).map(|k| refit.coefficients[1 + k]).collect());
        }
    }
    boot_coefs
}

/// Solve Kernel SHAP for one observation: regularize ATA, Cholesky solve, store in values matrix.
pub(crate) fn solve_kernel_shap_obs(
    ata: &mut [f64],
    atb: &[f64],
    ncomp: usize,
    values: &mut FdMatrix,
    i: usize,
) {
    for k in 0..ncomp {
        ata[k * ncomp + k] += 1e-10;
    }
    if let Some(l) = cholesky_factor(ata, ncomp) {
        let phi = cholesky_forward_back(&l, atb, ncomp);
        for k in 0..ncomp {
            values[(i, k)] = phi[k];
        }
    }
}

/// Build a design vector [1, scores, scalars] for one observation.
fn build_design_vector(
    new_scores: &FdMatrix,
    new_scalar: Option<&FdMatrix>,
    i: usize,
    ncomp: usize,
    p_scalar: usize,
    p: usize,
) -> Vec<f64> {
    let mut x = vec![0.0; p];
    x[0] = 1.0;
    for k in 0..ncomp {
        x[1 + k] = new_scores[(i, k)];
    }
    if let Some(ns) = new_scalar {
        for j in 0..p_scalar {
            x[1 + ncomp + j] = ns[(i, j)];
        }
    }
    x
}

/// Compute quantile-based ALE bin edges from sorted component values.
fn compute_ale_bin_edges(sorted_col: &[(f64, usize)], n: usize, n_bins: usize) -> Vec<f64> {
    let actual_bins = n_bins.min(n);
    let mut bin_edges = Vec::with_capacity(actual_bins + 1);
    bin_edges.push(sorted_col[0].0);
    for b in 1..actual_bins {
        let idx = (b as f64 / actual_bins as f64 * n as f64) as usize;
        let idx = idx.min(n - 1);
        let val = sorted_col[idx].0;
        if (val - *bin_edges.last().unwrap()).abs() > 1e-15 {
            bin_edges.push(val);
        }
    }
    let last_val = sorted_col[n - 1].0;
    if (last_val - *bin_edges.last().unwrap()).abs() > 1e-15 {
        bin_edges.push(last_val);
    }
    if bin_edges.len() < 2 {
        bin_edges.push(bin_edges[0] + 1.0);
    }
    bin_edges
}

/// Assign observations to ALE bins.
fn assign_ale_bins(
    sorted_col: &[(f64, usize)],
    bin_edges: &[f64],
    n: usize,
    n_bins_actual: usize,
) -> Vec<usize> {
    let mut bin_assignments = vec![0usize; n];
    for &(val, orig_idx) in sorted_col {
        let mut b = n_bins_actual - 1;
        for bb in 0..n_bins_actual - 1 {
            if val < bin_edges[bb + 1] {
                b = bb;
                break;
            }
        }
        bin_assignments[orig_idx] = b;
    }
    bin_assignments
}

/// Beam search for anchor rules in FPC score space.
pub(crate) fn anchor_beam_search(
    scores: &FdMatrix,
    ncomp: usize,
    n: usize,
    observation: usize,
    precision_threshold: f64,
    n_bins: usize,
    same_pred: &dyn Fn(usize) -> bool,
) -> (AnchorRule, Vec<bool>) {
    let bin_edges: Vec<Vec<f64>> = (0..ncomp)
        .map(|k| compute_bin_edges(scores, k, n, n_bins))
        .collect();

    let obs_bins: Vec<usize> = (0..ncomp)
        .map(|k| find_bin(scores[(observation, k)], &bin_edges[k], n_bins))
        .collect();

    let beam_width = 3;
    let mut best_conditions: Vec<usize> = Vec::new();
    let mut best_precision = 0.0;
    let mut best_matching = vec![true; n];
    let mut used = vec![false; ncomp];

    for _iter in 0..ncomp {
        let mut candidates = beam_search_candidates(
            scores,
            ncomp,
            &used,
            &obs_bins,
            &bin_edges,
            n_bins,
            &best_conditions,
            &best_matching,
            same_pred,
            beam_width,
        );

        if candidates.is_empty() {
            break;
        }

        let (new_conds, prec, matching) = candidates.remove(0);
        used[*new_conds.last().unwrap()] = true;
        best_conditions = new_conds;
        best_precision = prec;
        best_matching = matching;

        if best_precision >= precision_threshold {
            break;
        }
    }

    let rule = build_anchor_rule(
        &best_conditions,
        &bin_edges,
        &obs_bins,
        best_precision,
        &best_matching,
        n,
    );
    (rule, best_matching)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar_on_function::{fregre_lm, functional_logistic};
    use std::f64::consts::PI;

    fn generate_test_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);
        let mut y = vec![0.0; n];
        for i in 0..n {
            let phase =
                (seed.wrapping_mul(17).wrapping_add(i as u64 * 31) % 1000) as f64 / 1000.0 * PI;
            let amplitude =
                ((seed.wrapping_mul(13).wrapping_add(i as u64 * 7) % 100) as f64 / 100.0) - 0.5;
            for j in 0..m {
                data[(i, j)] =
                    (2.0 * PI * t[j] + phase).sin() + amplitude * (4.0 * PI * t[j]).cos();
            }
            y[i] = 2.0 * phase + 3.0 * amplitude;
        }
        (data, y)
    }

    #[test]
    fn test_functional_pdp_shape() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let pdp = functional_pdp(&fit, &data, None, 0, 20).unwrap();
        assert_eq!(pdp.grid_values.len(), 20);
        assert_eq!(pdp.pdp_curve.len(), 20);
        assert_eq!(pdp.ice_curves.shape(), (30, 20));
        assert_eq!(pdp.component, 0);
    }

    #[test]
    fn test_functional_pdp_linear_ice_parallel() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let pdp = functional_pdp(&fit, &data, None, 1, 10).unwrap();

        // For linear model, all ICE curves should have the same slope
        // slope = (ice[i, last] - ice[i, 0]) / (grid[last] - grid[0])
        let grid_range = pdp.grid_values[9] - pdp.grid_values[0];
        let slope_0 = (pdp.ice_curves[(0, 9)] - pdp.ice_curves[(0, 0)]) / grid_range;
        for i in 1..30 {
            let slope_i = (pdp.ice_curves[(i, 9)] - pdp.ice_curves[(i, 0)]) / grid_range;
            assert!(
                (slope_i - slope_0).abs() < 1e-10,
                "ICE curves should be parallel for linear model: slope_0={}, slope_{}={}",
                slope_0,
                i,
                slope_i
            );
        }
    }

    #[test]
    fn test_functional_pdp_logistic_probabilities() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();

        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let pdp = functional_pdp_logistic(&fit, &data, None, 0, 15).unwrap();

        assert_eq!(pdp.grid_values.len(), 15);
        assert_eq!(pdp.pdp_curve.len(), 15);
        assert_eq!(pdp.ice_curves.shape(), (30, 15));

        // All ICE values and PDP values should be valid probabilities in [0, 1]
        for g in 0..15 {
            assert!(
                pdp.pdp_curve[g] >= 0.0 && pdp.pdp_curve[g] <= 1.0,
                "PDP should be in [0,1], got {}",
                pdp.pdp_curve[g]
            );
            for i in 0..30 {
                assert!(
                    pdp.ice_curves[(i, g)] >= 0.0 && pdp.ice_curves[(i, g)] <= 1.0,
                    "ICE should be in [0,1], got {}",
                    pdp.ice_curves[(i, g)]
                );
            }
        }
    }

    #[test]
    fn test_functional_pdp_invalid_component() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        // component 3 is out of range (0..3)
        assert!(functional_pdp(&fit, &data, None, 3, 10).is_none());
        // n_grid < 2
        assert!(functional_pdp(&fit, &data, None, 0, 1).is_none());
    }

    #[test]
    fn test_functional_pdp_column_mismatch() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        // Wrong number of columns
        let wrong_data = FdMatrix::zeros(30, 40);
        assert!(functional_pdp(&fit, &wrong_data, None, 0, 10).is_none());
    }

    #[test]
    fn test_functional_pdp_zero_rows() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let empty_data = FdMatrix::zeros(0, 50);
        assert!(functional_pdp(&fit, &empty_data, None, 0, 10).is_none());
    }

    #[test]
    fn test_functional_pdp_logistic_column_mismatch() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let wrong_data = FdMatrix::zeros(30, 40);
        assert!(functional_pdp_logistic(&fit, &wrong_data, None, 0, 10).is_none());
    }

    #[test]
    fn test_functional_pdp_logistic_zero_rows() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let empty_data = FdMatrix::zeros(0, 50);
        assert!(functional_pdp_logistic(&fit, &empty_data, None, 0, 10).is_none());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Beta decomposition tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_beta_decomposition_sums_to_beta_t() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let dec = beta_decomposition(&fit).unwrap();
        for j in 0..50 {
            let sum: f64 = dec.components.iter().map(|c| c[j]).sum();
            assert!(
                (sum - fit.beta_t[j]).abs() < 1e-10,
                "Decomposition should sum to beta_t at j={}: {} vs {}",
                j,
                sum,
                fit.beta_t[j]
            );
        }
    }

    #[test]
    fn test_beta_decomposition_proportions_sum_to_one() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let dec = beta_decomposition(&fit).unwrap();
        let total: f64 = dec.variance_proportion.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Proportions should sum to 1: {}",
            total
        );
    }

    #[test]
    fn test_beta_decomposition_coefficients_match() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let dec = beta_decomposition(&fit).unwrap();
        for k in 0..3 {
            assert!(
                (dec.coefficients[k] - fit.coefficients[1 + k]).abs() < 1e-12,
                "Coefficient mismatch at k={}",
                k
            );
        }
    }

    #[test]
    fn test_beta_decomposition_logistic_sums() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let dec = beta_decomposition_logistic(&fit).unwrap();
        for j in 0..50 {
            let sum: f64 = dec.components.iter().map(|c| c[j]).sum();
            assert!(
                (sum - fit.beta_t[j]).abs() < 1e-10,
                "Logistic decomposition should sum to beta_t at j={}",
                j
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Significant regions tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_significant_regions_all_positive() {
        let lower = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let upper = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let regions = significant_regions(&lower, &upper).unwrap();
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].start_idx, 0);
        assert_eq!(regions[0].end_idx, 4);
        assert_eq!(regions[0].direction, SignificanceDirection::Positive);
    }

    #[test]
    fn test_significant_regions_none() {
        let lower = vec![-0.5, -0.3, -0.1, -0.5];
        let upper = vec![0.5, 0.3, 0.1, 0.5];
        let regions = significant_regions(&lower, &upper).unwrap();
        assert!(regions.is_empty());
    }

    #[test]
    fn test_significant_regions_mixed() {
        // Positive [0..1], gap [2], negative [3..4]
        let lower = vec![0.1, 0.2, -0.5, -1.0, -0.8];
        let upper = vec![0.9, 0.8, 0.5, -0.1, -0.2];
        let regions = significant_regions(&lower, &upper).unwrap();
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].direction, SignificanceDirection::Positive);
        assert_eq!(regions[0].start_idx, 0);
        assert_eq!(regions[0].end_idx, 1);
        assert_eq!(regions[1].direction, SignificanceDirection::Negative);
        assert_eq!(regions[1].start_idx, 3);
        assert_eq!(regions[1].end_idx, 4);
    }

    #[test]
    fn test_significant_regions_from_se() {
        let beta_t = vec![2.0, 2.0, 0.0, -2.0, -2.0];
        let beta_se = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let z = 1.96;
        let regions = significant_regions_from_se(&beta_t, &beta_se, z).unwrap();
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].direction, SignificanceDirection::Positive);
        assert_eq!(regions[1].direction, SignificanceDirection::Negative);
    }

    #[test]
    fn test_significant_regions_single_point() {
        let lower = vec![-1.0, 0.5, -1.0];
        let upper = vec![1.0, 1.0, 1.0];
        let regions = significant_regions(&lower, &upper).unwrap();
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].start_idx, 1);
        assert_eq!(regions[0].end_idx, 1);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FPC permutation importance tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_fpc_importance_shape() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let imp = fpc_permutation_importance(&fit, &data, &y, 10, 42).unwrap();
        assert_eq!(imp.importance.len(), 3);
        assert_eq!(imp.permuted_metric.len(), 3);
    }

    #[test]
    fn test_fpc_importance_nonnegative() {
        let (data, y) = generate_test_data(40, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let imp = fpc_permutation_importance(&fit, &data, &y, 50, 42).unwrap();
        for k in 0..3 {
            assert!(
                imp.importance[k] >= -0.05,
                "Importance should be approximately nonneg: k={}, val={}",
                k,
                imp.importance[k]
            );
        }
    }

    #[test]
    fn test_fpc_importance_dominant_largest() {
        let (data, y) = generate_test_data(50, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let imp = fpc_permutation_importance(&fit, &data, &y, 100, 42).unwrap();
        // The most important component should have the largest drop
        let max_imp = imp
            .importance
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_imp > 0.0,
            "At least one component should be important: {:?}",
            imp.importance
        );
    }

    #[test]
    fn test_fpc_importance_reproducible() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let imp1 = fpc_permutation_importance(&fit, &data, &y, 20, 999).unwrap();
        let imp2 = fpc_permutation_importance(&fit, &data, &y, 20, 999).unwrap();
        for k in 0..3 {
            assert!(
                (imp1.importance[k] - imp2.importance[k]).abs() < 1e-12,
                "Same seed should produce same result at k={}",
                k
            );
        }
    }

    #[test]
    fn test_fpc_importance_logistic_shape() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let imp = fpc_permutation_importance_logistic(&fit, &data, &y_bin, 10, 42).unwrap();
        assert_eq!(imp.importance.len(), 3);
        assert!(imp.baseline_metric >= 0.0 && imp.baseline_metric <= 1.0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Influence diagnostics tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_influence_leverage_sum_equals_p() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let diag = influence_diagnostics(&fit, &data, None).unwrap();
        let h_sum: f64 = diag.leverage.iter().sum();
        assert!(
            (h_sum - diag.p as f64).abs() < 1e-6,
            "Leverage sum should equal p={}: got {}",
            diag.p,
            h_sum
        );
    }

    #[test]
    fn test_influence_leverage_range() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let diag = influence_diagnostics(&fit, &data, None).unwrap();
        for (i, &h) in diag.leverage.iter().enumerate() {
            assert!(
                (-1e-10..=1.0 + 1e-10).contains(&h),
                "Leverage out of range at i={}: {}",
                i,
                h
            );
        }
    }

    #[test]
    fn test_influence_cooks_nonnegative() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let diag = influence_diagnostics(&fit, &data, None).unwrap();
        for (i, &d) in diag.cooks_distance.iter().enumerate() {
            assert!(d >= 0.0, "Cook's D should be nonneg at i={}: {}", i, d);
        }
    }

    #[test]
    fn test_influence_high_leverage_outlier() {
        let (mut data, mut y) = generate_test_data(30, 50, 42);
        // Make obs 0 an extreme outlier
        for j in 0..50 {
            data[(0, j)] *= 10.0;
        }
        y[0] = 100.0;
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let diag = influence_diagnostics(&fit, &data, None).unwrap();
        let max_cd = diag
            .cooks_distance
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (diag.cooks_distance[0] - max_cd).abs() < 1e-10,
            "Outlier should have max Cook's D"
        );
    }

    #[test]
    fn test_influence_shape() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let diag = influence_diagnostics(&fit, &data, None).unwrap();
        assert_eq!(diag.leverage.len(), 30);
        assert_eq!(diag.cooks_distance.len(), 30);
        assert_eq!(diag.p, 4); // 1 + 3 components
    }

    #[test]
    fn test_influence_column_mismatch_returns_none() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let wrong_data = FdMatrix::zeros(30, 40);
        assert!(influence_diagnostics(&fit, &wrong_data, None).is_none());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Friedman H-statistic tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_h_statistic_linear_zero() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let h = friedman_h_statistic(&fit, &data, 0, 1, 10).unwrap();
        assert!(
            h.h_squared.abs() < 1e-6,
            "H² should be ~0 for linear model: {}",
            h.h_squared
        );
    }

    #[test]
    fn test_h_statistic_logistic_positive() {
        let (data, y_cont) = generate_test_data(40, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let h = friedman_h_statistic_logistic(&fit, &data, None, 0, 1, 10).unwrap();
        // Sigmoid creates apparent interaction; H² may be small but should be >= 0
        assert!(
            h.h_squared >= 0.0,
            "H² should be nonneg for logistic: {}",
            h.h_squared
        );
    }

    #[test]
    fn test_h_statistic_symmetry() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let h01 = friedman_h_statistic(&fit, &data, 0, 1, 10).unwrap();
        let h10 = friedman_h_statistic(&fit, &data, 1, 0, 10).unwrap();
        assert!(
            (h01.h_squared - h10.h_squared).abs() < 1e-10,
            "H(0,1) should equal H(1,0): {} vs {}",
            h01.h_squared,
            h10.h_squared
        );
    }

    #[test]
    fn test_h_statistic_grid_shape() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let h = friedman_h_statistic(&fit, &data, 0, 2, 8).unwrap();
        assert_eq!(h.grid_j.len(), 8);
        assert_eq!(h.grid_k.len(), 8);
        assert_eq!(h.pdp_2d.shape(), (8, 8));
    }

    #[test]
    fn test_h_statistic_bounded() {
        let (data, y_cont) = generate_test_data(40, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let h = friedman_h_statistic_logistic(&fit, &data, None, 0, 1, 10).unwrap();
        assert!(
            h.h_squared >= 0.0 && h.h_squared <= 1.0 + 1e-6,
            "H² should be in [0,1]: {}",
            h.h_squared
        );
    }

    #[test]
    fn test_h_statistic_same_component_none() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        assert!(friedman_h_statistic(&fit, &data, 1, 1, 10).is_none());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Pointwise importance tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_pointwise_importance_shape() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let pi = pointwise_importance(&fit).unwrap();
        assert_eq!(pi.importance.len(), 50);
        assert_eq!(pi.importance_normalized.len(), 50);
        assert_eq!(pi.component_importance.shape(), (3, 50));
        assert_eq!(pi.score_variance.len(), 3);
    }

    #[test]
    fn test_pointwise_importance_normalized_sums_to_one() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let pi = pointwise_importance(&fit).unwrap();
        let total: f64 = pi.importance_normalized.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Normalized importance should sum to 1: {}",
            total
        );
    }

    #[test]
    fn test_pointwise_importance_all_nonneg() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let pi = pointwise_importance(&fit).unwrap();
        for (j, &v) in pi.importance.iter().enumerate() {
            assert!(v >= -1e-15, "Importance should be nonneg at j={}: {}", j, v);
        }
    }

    #[test]
    fn test_pointwise_importance_component_sum_equals_total() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let pi = pointwise_importance(&fit).unwrap();
        for j in 0..50 {
            let sum: f64 = (0..3).map(|k| pi.component_importance[(k, j)]).sum();
            assert!(
                (sum - pi.importance[j]).abs() < 1e-10,
                "Component sum should equal total at j={}: {} vs {}",
                j,
                sum,
                pi.importance[j]
            );
        }
    }

    #[test]
    fn test_pointwise_importance_logistic_shape() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let pi = pointwise_importance_logistic(&fit).unwrap();
        assert_eq!(pi.importance.len(), 50);
        assert_eq!(pi.score_variance.len(), 3);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VIF tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_vif_orthogonal_fpcs_near_one() {
        let (data, y) = generate_test_data(50, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let vif = fpc_vif(&fit, &data, None).unwrap();
        for (k, &v) in vif.vif.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 0.5,
                "Orthogonal FPC VIF should be ≈1 at k={}: {}",
                k,
                v
            );
        }
    }

    #[test]
    fn test_vif_all_positive() {
        let (data, y) = generate_test_data(50, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let vif = fpc_vif(&fit, &data, None).unwrap();
        for (k, &v) in vif.vif.iter().enumerate() {
            assert!(v >= 1.0 - 1e-6, "VIF should be ≥ 1 at k={}: {}", k, v);
        }
    }

    #[test]
    fn test_vif_shape() {
        let (data, y) = generate_test_data(50, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let vif = fpc_vif(&fit, &data, None).unwrap();
        assert_eq!(vif.vif.len(), 3);
        assert_eq!(vif.labels.len(), 3);
    }

    #[test]
    fn test_vif_labels_correct() {
        let (data, y) = generate_test_data(50, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let vif = fpc_vif(&fit, &data, None).unwrap();
        assert_eq!(vif.labels[0], "FPC_0");
        assert_eq!(vif.labels[1], "FPC_1");
        assert_eq!(vif.labels[2], "FPC_2");
    }

    #[test]
    fn test_vif_logistic_agrees_with_linear() {
        let (data, y_cont) = generate_test_data(50, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();
        let fit_lm = fregre_lm(&data, &y_cont, None, 3).unwrap();
        let fit_log = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let vif_lm = fpc_vif(&fit_lm, &data, None).unwrap();
        let vif_log = fpc_vif_logistic(&fit_log, &data, None).unwrap();
        // Same data → same VIF (VIF depends only on X, not y)
        for k in 0..3 {
            assert!(
                (vif_lm.vif[k] - vif_log.vif[k]).abs() < 1e-6,
                "VIF should agree: lm={}, log={}",
                vif_lm.vif[k],
                vif_log.vif[k]
            );
        }
    }

    #[test]
    fn test_vif_single_predictor() {
        let (data, y) = generate_test_data(50, 50, 42);
        let fit = fregre_lm(&data, &y, None, 1).unwrap();
        let vif = fpc_vif(&fit, &data, None).unwrap();
        assert_eq!(vif.vif.len(), 1);
        assert!(
            (vif.vif[0] - 1.0).abs() < 1e-6,
            "Single predictor VIF should be 1: {}",
            vif.vif[0]
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SHAP tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_shap_linear_sum_to_fitted() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let shap = fpc_shap_values(&fit, &data, None).unwrap();
        for i in 0..30 {
            let sum: f64 = (0..3).map(|k| shap.values[(i, k)]).sum::<f64>() + shap.base_value;
            assert!(
                (sum - fit.fitted_values[i]).abs() < 1e-8,
                "SHAP sum should equal fitted at i={}: {} vs {}",
                i,
                sum,
                fit.fitted_values[i]
            );
        }
    }

    #[test]
    fn test_shap_linear_shape() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let shap = fpc_shap_values(&fit, &data, None).unwrap();
        assert_eq!(shap.values.shape(), (30, 3));
        assert_eq!(shap.mean_scores.len(), 3);
    }

    #[test]
    fn test_shap_linear_sign_matches_coefficient() {
        let (data, y) = generate_test_data(50, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let shap = fpc_shap_values(&fit, &data, None).unwrap();
        // For each obs, if score > mean and coef > 0, SHAP should be > 0
        for k in 0..3 {
            let coef_k = fit.coefficients[1 + k];
            if coef_k.abs() < 1e-10 {
                continue;
            }
            for i in 0..50 {
                let score_centered = fit.fpca.scores[(i, k)] - shap.mean_scores[k];
                let expected_sign = (coef_k * score_centered).signum();
                if shap.values[(i, k)].abs() > 1e-10 {
                    assert_eq!(
                        shap.values[(i, k)].signum(),
                        expected_sign,
                        "SHAP sign mismatch at i={}, k={}",
                        i,
                        k
                    );
                }
            }
        }
    }

    #[test]
    fn test_shap_logistic_sum_approximates_prediction() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let shap = fpc_shap_values_logistic(&fit, &data, None, 500, 42).unwrap();
        // For logistic, SHAP sum + base should approximate prediction direction
        // Kernel SHAP is approximate, so we check correlation rather than exact match
        let mut shap_sums = Vec::new();
        for i in 0..30 {
            let sum: f64 = (0..3).map(|k| shap.values[(i, k)]).sum::<f64>() + shap.base_value;
            shap_sums.push(sum);
        }
        // SHAP sums should be correlated with probabilities
        let mean_shap: f64 = shap_sums.iter().sum::<f64>() / 30.0;
        let mean_prob: f64 = fit.probabilities.iter().sum::<f64>() / 30.0;
        let mut cov = 0.0;
        let mut var_s = 0.0;
        let mut var_p = 0.0;
        for i in 0..30 {
            let ds = shap_sums[i] - mean_shap;
            let dp = fit.probabilities[i] - mean_prob;
            cov += ds * dp;
            var_s += ds * ds;
            var_p += dp * dp;
        }
        let corr = if var_s > 0.0 && var_p > 0.0 {
            cov / (var_s.sqrt() * var_p.sqrt())
        } else {
            0.0
        };
        assert!(
            corr > 0.5,
            "Logistic SHAP sums should correlate with probabilities: r={}",
            corr
        );
    }

    #[test]
    fn test_shap_logistic_reproducible() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let s1 = fpc_shap_values_logistic(&fit, &data, None, 100, 999).unwrap();
        let s2 = fpc_shap_values_logistic(&fit, &data, None, 100, 999).unwrap();
        for i in 0..30 {
            for k in 0..3 {
                assert!(
                    (s1.values[(i, k)] - s2.values[(i, k)]).abs() < 1e-12,
                    "Same seed should give same SHAP at i={}, k={}",
                    i,
                    k
                );
            }
        }
    }

    #[test]
    fn test_shap_invalid_returns_none() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let empty = FdMatrix::zeros(0, 50);
        assert!(fpc_shap_values(&fit, &empty, None).is_none());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // DFBETAS / DFFITS tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dfbetas_shape() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let db = dfbetas_dffits(&fit, &data, None).unwrap();
        assert_eq!(db.dfbetas.shape(), (30, 4)); // n × p (intercept + 3 FPCs)
        assert_eq!(db.dffits.len(), 30);
        assert_eq!(db.studentized_residuals.len(), 30);
        assert_eq!(db.p, 4);
    }

    #[test]
    fn test_dffits_sign_matches_residual() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let db = dfbetas_dffits(&fit, &data, None).unwrap();
        for i in 0..30 {
            if fit.residuals[i].abs() > 1e-10 && db.dffits[i].abs() > 1e-10 {
                assert_eq!(
                    db.dffits[i].signum(),
                    fit.residuals[i].signum(),
                    "DFFITS sign should match residual at i={}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_dfbetas_outlier_flagged() {
        let (mut data, mut y) = generate_test_data(30, 50, 42);
        for j in 0..50 {
            data[(0, j)] *= 10.0;
        }
        y[0] = 100.0;
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let db = dfbetas_dffits(&fit, &data, None).unwrap();
        // Outlier should have large DFFITS
        let max_dffits = db
            .dffits
            .iter()
            .map(|v| v.abs())
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            db.dffits[0].abs() >= max_dffits - 1e-10,
            "Outlier should have max |DFFITS|"
        );
    }

    #[test]
    fn test_dfbetas_cutoff_value() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let db = dfbetas_dffits(&fit, &data, None).unwrap();
        assert!(
            (db.dfbetas_cutoff - 2.0 / (30.0_f64).sqrt()).abs() < 1e-10,
            "DFBETAS cutoff should be 2/√n"
        );
        assert!(
            (db.dffits_cutoff - 2.0 * (4.0 / 30.0_f64).sqrt()).abs() < 1e-10,
            "DFFITS cutoff should be 2√(p/n)"
        );
    }

    #[test]
    fn test_dfbetas_underdetermined_returns_none() {
        let (data, y) = generate_test_data(3, 50, 42);
        let fit = fregre_lm(&data, &y, None, 2).unwrap();
        // n=3, p=3 (intercept + 2 FPCs) → n <= p, should return None
        assert!(dfbetas_dffits(&fit, &data, None).is_none());
    }

    #[test]
    fn test_dffits_consistency_with_cooks() {
        let (data, y) = generate_test_data(40, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let db = dfbetas_dffits(&fit, &data, None).unwrap();
        let infl = influence_diagnostics(&fit, &data, None).unwrap();
        // DFFITS² ≈ p × Cook's D × (studentized_residuals² adjustment)
        // Both should rank observations similarly
        let mut dffits_order: Vec<usize> = (0..40).collect();
        dffits_order.sort_by(|&a, &b| db.dffits[b].abs().partial_cmp(&db.dffits[a].abs()).unwrap());
        let mut cooks_order: Vec<usize> = (0..40).collect();
        cooks_order.sort_by(|&a, &b| {
            infl.cooks_distance[b]
                .partial_cmp(&infl.cooks_distance[a])
                .unwrap()
        });
        // Top influential observation should be the same
        assert_eq!(
            dffits_order[0], cooks_order[0],
            "Top influential obs should agree between DFFITS and Cook's D"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Prediction interval tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_prediction_interval_training_data_matches_fitted() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let pi = prediction_intervals(&fit, &data, None, &data, None, 0.95).unwrap();
        for i in 0..30 {
            assert!(
                (pi.predictions[i] - fit.fitted_values[i]).abs() < 1e-6,
                "Prediction should match fitted at i={}: {} vs {}",
                i,
                pi.predictions[i],
                fit.fitted_values[i]
            );
        }
    }

    #[test]
    fn test_prediction_interval_covers_training_y() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let pi = prediction_intervals(&fit, &data, None, &data, None, 0.95).unwrap();
        let mut covered = 0;
        for i in 0..30 {
            if y[i] >= pi.lower[i] && y[i] <= pi.upper[i] {
                covered += 1;
            }
        }
        // At 95% confidence, most training points should be covered
        assert!(
            covered >= 20,
            "At least ~67% of training y should be covered: {}/30",
            covered
        );
    }

    #[test]
    fn test_prediction_interval_symmetry() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let pi = prediction_intervals(&fit, &data, None, &data, None, 0.95).unwrap();
        for i in 0..30 {
            let above = pi.upper[i] - pi.predictions[i];
            let below = pi.predictions[i] - pi.lower[i];
            assert!(
                (above - below).abs() < 1e-10,
                "Interval should be symmetric at i={}: above={}, below={}",
                i,
                above,
                below
            );
        }
    }

    #[test]
    fn test_prediction_interval_wider_at_99_than_95() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let pi95 = prediction_intervals(&fit, &data, None, &data, None, 0.95).unwrap();
        let pi99 = prediction_intervals(&fit, &data, None, &data, None, 0.99).unwrap();
        for i in 0..30 {
            let width95 = pi95.upper[i] - pi95.lower[i];
            let width99 = pi99.upper[i] - pi99.lower[i];
            assert!(
                width99 >= width95 - 1e-10,
                "99% interval should be wider at i={}: {} vs {}",
                i,
                width99,
                width95
            );
        }
    }

    #[test]
    fn test_prediction_interval_shape() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let pi = prediction_intervals(&fit, &data, None, &data, None, 0.95).unwrap();
        assert_eq!(pi.predictions.len(), 30);
        assert_eq!(pi.lower.len(), 30);
        assert_eq!(pi.upper.len(), 30);
        assert_eq!(pi.prediction_se.len(), 30);
        assert!((pi.confidence_level - 0.95).abs() < 1e-15);
    }

    #[test]
    fn test_prediction_interval_invalid_confidence_returns_none() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        assert!(prediction_intervals(&fit, &data, None, &data, None, 0.0).is_none());
        assert!(prediction_intervals(&fit, &data, None, &data, None, 1.0).is_none());
        assert!(prediction_intervals(&fit, &data, None, &data, None, -0.5).is_none());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ALE tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_ale_linear_is_linear() {
        let (data, y) = generate_test_data(50, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let ale = fpc_ale(&fit, &data, None, 0, 10).unwrap();
        // For a linear model, ALE should be approximately linear
        if ale.bin_midpoints.len() >= 3 {
            let slopes: Vec<f64> = ale
                .ale_values
                .windows(2)
                .zip(ale.bin_midpoints.windows(2))
                .map(|(v, m)| {
                    let dx = m[1] - m[0];
                    if dx.abs() > 1e-15 {
                        (v[1] - v[0]) / dx
                    } else {
                        0.0
                    }
                })
                .collect();
            // All slopes should be approximately equal
            let mean_slope = slopes.iter().sum::<f64>() / slopes.len() as f64;
            for (b, &s) in slopes.iter().enumerate() {
                assert!(
                    (s - mean_slope).abs() < mean_slope.abs() * 0.5 + 0.5,
                    "ALE slope should be constant for linear model at bin {}: {} vs mean {}",
                    b,
                    s,
                    mean_slope
                );
            }
        }
    }

    #[test]
    fn test_ale_centered_mean_zero() {
        let (data, y) = generate_test_data(50, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let ale = fpc_ale(&fit, &data, None, 0, 10).unwrap();
        let total_n: usize = ale.bin_counts.iter().sum();
        let weighted_mean: f64 = ale
            .ale_values
            .iter()
            .zip(&ale.bin_counts)
            .map(|(&a, &c)| a * c as f64)
            .sum::<f64>()
            / total_n as f64;
        assert!(
            weighted_mean.abs() < 1e-10,
            "ALE should be centered at zero: {}",
            weighted_mean
        );
    }

    #[test]
    fn test_ale_bin_counts_sum_to_n() {
        let (data, y) = generate_test_data(50, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let ale = fpc_ale(&fit, &data, None, 0, 10).unwrap();
        let total: usize = ale.bin_counts.iter().sum();
        assert_eq!(total, 50, "Bin counts should sum to n");
    }

    #[test]
    fn test_ale_shape() {
        let (data, y) = generate_test_data(50, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let ale = fpc_ale(&fit, &data, None, 0, 8).unwrap();
        let nb = ale.ale_values.len();
        assert_eq!(ale.bin_midpoints.len(), nb);
        assert_eq!(ale.bin_edges.len(), nb + 1);
        assert_eq!(ale.bin_counts.len(), nb);
        assert_eq!(ale.component, 0);
    }

    #[test]
    fn test_ale_logistic_bounded() {
        let (data, y_cont) = generate_test_data(50, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let ale = fpc_ale_logistic(&fit, &data, None, 0, 10).unwrap();
        // ALE values are centered diffs, they can be outside [0,1] but shouldn't be extreme
        for &v in &ale.ale_values {
            assert!(v.abs() < 2.0, "Logistic ALE should be bounded: {}", v);
        }
    }

    #[test]
    fn test_ale_invalid_returns_none() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        // Invalid component
        assert!(fpc_ale(&fit, &data, None, 5, 10).is_none());
        // Zero bins
        assert!(fpc_ale(&fit, &data, None, 0, 0).is_none());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // LOO-CV / PRESS tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_loo_cv_shape() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let loo = loo_cv_press(&fit, &data, &y, None).unwrap();
        assert_eq!(loo.loo_residuals.len(), 30);
        assert_eq!(loo.leverage.len(), 30);
    }

    #[test]
    fn test_loo_r_squared_leq_r_squared() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let loo = loo_cv_press(&fit, &data, &y, None).unwrap();
        assert!(
            loo.loo_r_squared <= fit.r_squared + 1e-10,
            "LOO R² ({}) should be ≤ training R² ({})",
            loo.loo_r_squared,
            fit.r_squared
        );
    }

    #[test]
    fn test_loo_press_equals_sum_squares() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let loo = loo_cv_press(&fit, &data, &y, None).unwrap();
        let manual_press: f64 = loo.loo_residuals.iter().map(|r| r * r).sum();
        assert!(
            (loo.press - manual_press).abs() < 1e-10,
            "PRESS mismatch: {} vs {}",
            loo.press,
            manual_press
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Sobol tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_sobol_linear_nonnegative() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let sobol = sobol_indices(&fit, &data, &y, None).unwrap();
        for (k, &s) in sobol.first_order.iter().enumerate() {
            assert!(s >= -1e-10, "S_{} should be ≥ 0: {}", k, s);
        }
    }

    #[test]
    fn test_sobol_linear_sum_approx_r2() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let sobol = sobol_indices(&fit, &data, &y, None).unwrap();
        let sum_s: f64 = sobol.first_order.iter().sum();
        // Σ S_k ≈ R² (model explains that fraction of variance)
        assert!(
            (sum_s - fit.r_squared).abs() < 0.2,
            "Σ S_k ({}) should be close to R² ({})",
            sum_s,
            fit.r_squared
        );
    }

    #[test]
    fn test_sobol_logistic_bounded() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_bin = {
            let mut s = y_cont.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = s[s.len() / 2];
            y_cont
                .iter()
                .map(|&v| if v >= med { 1.0 } else { 0.0 })
                .collect::<Vec<_>>()
        };
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let sobol = sobol_indices_logistic(&fit, &data, None, 500, 42).unwrap();
        for &s in &sobol.first_order {
            assert!(s > -0.5 && s < 1.5, "Logistic S_k should be bounded: {}", s);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Calibration tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_calibration_brier_range() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_bin = {
            let mut s = y_cont.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = s[s.len() / 2];
            y_cont
                .iter()
                .map(|&v| if v >= med { 1.0 } else { 0.0 })
                .collect::<Vec<_>>()
        };
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let cal = calibration_diagnostics(&fit, &y_bin, 5).unwrap();
        assert!(
            cal.brier_score >= 0.0 && cal.brier_score <= 1.0,
            "Brier score should be in [0,1]: {}",
            cal.brier_score
        );
    }

    #[test]
    fn test_calibration_bin_counts_sum_to_n() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_bin = {
            let mut s = y_cont.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = s[s.len() / 2];
            y_cont
                .iter()
                .map(|&v| if v >= med { 1.0 } else { 0.0 })
                .collect::<Vec<_>>()
        };
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let cal = calibration_diagnostics(&fit, &y_bin, 5).unwrap();
        let total: usize = cal.bin_counts.iter().sum();
        assert_eq!(total, 30, "Bin counts should sum to n");
    }

    #[test]
    fn test_calibration_n_groups_match() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_bin = {
            let mut s = y_cont.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = s[s.len() / 2];
            y_cont
                .iter()
                .map(|&v| if v >= med { 1.0 } else { 0.0 })
                .collect::<Vec<_>>()
        };
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let cal = calibration_diagnostics(&fit, &y_bin, 5).unwrap();
        assert_eq!(cal.n_groups, cal.reliability_bins.len());
        assert_eq!(cal.n_groups, cal.bin_counts.len());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Saliency tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_saliency_linear_shape() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let sal = functional_saliency(&fit, &data, None).unwrap();
        assert_eq!(sal.saliency_map.shape(), (30, 50));
        assert_eq!(sal.mean_absolute_saliency.len(), 50);
    }

    #[test]
    fn test_saliency_logistic_bounded_by_quarter_beta() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_bin = {
            let mut s = y_cont.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = s[s.len() / 2];
            y_cont
                .iter()
                .map(|&v| if v >= med { 1.0 } else { 0.0 })
                .collect::<Vec<_>>()
        };
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let sal = functional_saliency_logistic(&fit).unwrap();
        for i in 0..30 {
            for j in 0..50 {
                assert!(
                    sal.saliency_map[(i, j)].abs() <= 0.25 * fit.beta_t[j].abs() + 1e-10,
                    "|s| should be ≤ 0.25 × |β(t)| at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_saliency_mean_abs_nonneg() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let sal = functional_saliency(&fit, &data, None).unwrap();
        for &v in &sal.mean_absolute_saliency {
            assert!(v >= 0.0, "Mean absolute saliency should be ≥ 0: {}", v);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Domain selection tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_domain_selection_valid_indices() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let ds = domain_selection(&fit, 5, 0.01).unwrap();
        for iv in &ds.intervals {
            assert!(iv.start_idx <= iv.end_idx);
            assert!(iv.end_idx < 50);
        }
    }

    #[test]
    fn test_domain_selection_full_window_one_interval() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        // window_width = m should give at most one interval
        let ds = domain_selection(&fit, 50, 0.01).unwrap();
        assert!(
            ds.intervals.len() <= 1,
            "Full window should give ≤ 1 interval: {}",
            ds.intervals.len()
        );
    }

    #[test]
    fn test_domain_selection_high_threshold_fewer() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let ds_low = domain_selection(&fit, 5, 0.01).unwrap();
        let ds_high = domain_selection(&fit, 5, 0.5).unwrap();
        assert!(
            ds_high.intervals.len() <= ds_low.intervals.len(),
            "Higher threshold should give ≤ intervals: {} vs {}",
            ds_high.intervals.len(),
            ds_low.intervals.len()
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Conditional permutation importance tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_cond_perm_shape() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let cp = conditional_permutation_importance(&fit, &data, &y, None, 3, 5, 42).unwrap();
        assert_eq!(cp.importance.len(), 3);
        assert_eq!(cp.permuted_metric.len(), 3);
        assert_eq!(cp.unconditional_importance.len(), 3);
    }

    #[test]
    fn test_cond_perm_vs_unconditional_close() {
        let (data, y) = generate_test_data(40, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let cp = conditional_permutation_importance(&fit, &data, &y, None, 3, 20, 42).unwrap();
        // For orthogonal FPCs, conditional ≈ unconditional
        for k in 0..3 {
            let diff = (cp.importance[k] - cp.unconditional_importance[k]).abs();
            assert!(
                diff < 0.5,
                "Conditional vs unconditional should be similar for FPC {}: {} vs {}",
                k,
                cp.importance[k],
                cp.unconditional_importance[k]
            );
        }
    }

    #[test]
    fn test_cond_perm_importance_nonneg() {
        let (data, y) = generate_test_data(40, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let cp = conditional_permutation_importance(&fit, &data, &y, None, 3, 20, 42).unwrap();
        for k in 0..3 {
            assert!(
                cp.importance[k] >= -0.15,
                "Importance should be approx ≥ 0 for FPC {}: {}",
                k,
                cp.importance[k]
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Counterfactual tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_counterfactual_regression_exact() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let target = fit.fitted_values[0] + 1.0;
        let cf = counterfactual_regression(&fit, &data, None, 0, target).unwrap();
        assert!(cf.found);
        assert!(
            (cf.counterfactual_prediction - target).abs() < 1e-10,
            "Counterfactual prediction should match target: {} vs {}",
            cf.counterfactual_prediction,
            target
        );
    }

    #[test]
    fn test_counterfactual_regression_minimal() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let gap = 1.0;
        let target = fit.fitted_values[0] + gap;
        let cf = counterfactual_regression(&fit, &data, None, 0, target).unwrap();
        let gamma: Vec<f64> = (0..3).map(|k| fit.coefficients[1 + k]).collect();
        let gamma_norm: f64 = gamma.iter().map(|g| g * g).sum::<f64>().sqrt();
        let expected_dist = gap.abs() / gamma_norm;
        assert!(
            (cf.distance - expected_dist).abs() < 1e-6,
            "Distance should be |gap|/||γ||: {} vs {}",
            cf.distance,
            expected_dist
        );
    }

    #[test]
    fn test_counterfactual_logistic_flips_class() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_bin = {
            let mut s = y_cont.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = s[s.len() / 2];
            y_cont
                .iter()
                .map(|&v| if v >= med { 1.0 } else { 0.0 })
                .collect::<Vec<_>>()
        };
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let cf = counterfactual_logistic(&fit, &data, None, 0, 1000, 0.5).unwrap();
        if cf.found {
            let orig_class = if cf.original_prediction >= 0.5 { 1 } else { 0 };
            let new_class = if cf.counterfactual_prediction >= 0.5 {
                1
            } else {
                0
            };
            assert_ne!(
                orig_class, new_class,
                "Class should flip: orig={}, new={}",
                orig_class, new_class
            );
        }
    }

    #[test]
    fn test_counterfactual_invalid_obs_none() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        assert!(counterfactual_regression(&fit, &data, None, 100, 0.0).is_none());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Prototype/criticism tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_prototype_criticism_shape() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let pc = prototype_criticism(&fit.fpca, 3, 5, 3).unwrap();
        assert_eq!(pc.prototype_indices.len(), 5);
        assert_eq!(pc.prototype_witness.len(), 5);
        assert_eq!(pc.criticism_indices.len(), 3);
        assert_eq!(pc.criticism_witness.len(), 3);
    }

    #[test]
    fn test_prototype_criticism_no_overlap() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let pc = prototype_criticism(&fit.fpca, 3, 5, 3).unwrap();
        for &ci in &pc.criticism_indices {
            assert!(
                !pc.prototype_indices.contains(&ci),
                "Criticism {} should not be a prototype",
                ci
            );
        }
    }

    #[test]
    fn test_prototype_criticism_bandwidth_positive() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let pc = prototype_criticism(&fit.fpca, 3, 5, 3).unwrap();
        assert!(
            pc.bandwidth > 0.0,
            "Bandwidth should be > 0: {}",
            pc.bandwidth
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // LIME tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_lime_linear_matches_global() {
        let (data, y) = generate_test_data(40, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let lime = lime_explanation(&fit, &data, None, 0, 5000, 1.0, 42).unwrap();
        // For linear model, LIME attributions should approximate global coefficients
        for k in 0..3 {
            let global = fit.coefficients[1 + k];
            let local = lime.attributions[k];
            let rel_err = if global.abs() > 1e-6 {
                (local - global).abs() / global.abs()
            } else {
                local.abs()
            };
            assert!(
                rel_err < 0.5,
                "LIME should approximate global coef for FPC {}: local={}, global={}",
                k,
                local,
                global
            );
        }
    }

    #[test]
    fn test_lime_logistic_shape() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_bin = {
            let mut s = y_cont.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = s[s.len() / 2];
            y_cont
                .iter()
                .map(|&v| if v >= med { 1.0 } else { 0.0 })
                .collect::<Vec<_>>()
        };
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let lime = lime_explanation_logistic(&fit, &data, None, 0, 500, 1.0, 42).unwrap();
        assert_eq!(lime.attributions.len(), 3);
        assert!(
            lime.local_r_squared >= 0.0 && lime.local_r_squared <= 1.0,
            "R² should be in [0,1]: {}",
            lime.local_r_squared
        );
    }

    #[test]
    fn test_lime_invalid_none() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        assert!(lime_explanation(&fit, &data, None, 100, 100, 1.0, 42).is_none());
        assert!(lime_explanation(&fit, &data, None, 0, 0, 1.0, 42).is_none());
        assert!(lime_explanation(&fit, &data, None, 0, 100, 0.0, 42).is_none());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ECE tests
    // ═══════════════════════════════════════════════════════════════════════

    fn make_logistic_fit() -> (FdMatrix, Vec<f64>, FunctionalLogisticResult) {
        let (data, y_cont) = generate_test_data(40, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        (data, y_bin, fit)
    }

    #[test]
    fn test_ece_range() {
        let (_data, y_bin, fit) = make_logistic_fit();
        let ece = expected_calibration_error(&fit, &y_bin, 10).unwrap();
        assert!(
            ece.ece >= 0.0 && ece.ece <= 1.0,
            "ECE out of range: {}",
            ece.ece
        );
        assert!(
            ece.mce >= 0.0 && ece.mce <= 1.0,
            "MCE out of range: {}",
            ece.mce
        );
    }

    #[test]
    fn test_ece_leq_mce() {
        let (_data, y_bin, fit) = make_logistic_fit();
        let ece = expected_calibration_error(&fit, &y_bin, 10).unwrap();
        assert!(
            ece.ece <= ece.mce + 1e-10,
            "ECE should ≤ MCE: {} vs {}",
            ece.ece,
            ece.mce
        );
    }

    #[test]
    fn test_ece_bin_contributions_sum() {
        let (_data, y_bin, fit) = make_logistic_fit();
        let ece = expected_calibration_error(&fit, &y_bin, 10).unwrap();
        let sum: f64 = ece.bin_ece_contributions.iter().sum();
        assert!(
            (sum - ece.ece).abs() < 1e-10,
            "Contributions should sum to ECE: {} vs {}",
            sum,
            ece.ece
        );
    }

    #[test]
    fn test_ece_n_bins_match() {
        let (_data, y_bin, fit) = make_logistic_fit();
        let ece = expected_calibration_error(&fit, &y_bin, 10).unwrap();
        assert_eq!(ece.n_bins, 10);
        assert_eq!(ece.bin_ece_contributions.len(), 10);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Conformal prediction tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_conformal_coverage_near_target() {
        let (data, y) = generate_test_data(60, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let cp = conformal_prediction_residuals(&fit, &data, &y, &data, None, None, 0.3, 0.1, 42)
            .unwrap();
        // Coverage should be ≥ 1 - α approximately
        assert!(
            cp.coverage >= 0.8,
            "Coverage {} should be ≥ 0.8 for α=0.1",
            cp.coverage
        );
    }

    #[test]
    fn test_conformal_interval_width_positive() {
        let (data, y) = generate_test_data(60, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let cp = conformal_prediction_residuals(&fit, &data, &y, &data, None, None, 0.3, 0.1, 42)
            .unwrap();
        for i in 0..cp.predictions.len() {
            assert!(
                cp.upper[i] > cp.lower[i],
                "Upper should > lower at {}: {} vs {}",
                i,
                cp.upper[i],
                cp.lower[i]
            );
        }
    }

    #[test]
    fn test_conformal_quantile_positive() {
        let (data, y) = generate_test_data(60, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let cp = conformal_prediction_residuals(&fit, &data, &y, &data, None, None, 0.3, 0.1, 42)
            .unwrap();
        assert!(
            cp.residual_quantile >= 0.0,
            "Quantile should be ≥ 0: {}",
            cp.residual_quantile
        );
    }

    #[test]
    fn test_conformal_lengths_match() {
        let (data, y) = generate_test_data(60, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let test_data = FdMatrix::zeros(10, 50);
        let cp =
            conformal_prediction_residuals(&fit, &data, &y, &test_data, None, None, 0.3, 0.1, 42)
                .unwrap();
        assert_eq!(cp.predictions.len(), 10);
        assert_eq!(cp.lower.len(), 10);
        assert_eq!(cp.upper.len(), 10);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Regression depth tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_regression_depth_range() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let rd = regression_depth(&fit, &data, &y, None, 20, DepthType::FraimanMuniz, 42).unwrap();
        for (i, &d) in rd.score_depths.iter().enumerate() {
            assert!(
                (-1e-10..=1.0 + 1e-10).contains(&d),
                "Depth out of range at {}: {}",
                i,
                d
            );
        }
    }

    #[test]
    fn test_regression_depth_beta_nonneg() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let rd = regression_depth(&fit, &data, &y, None, 20, DepthType::FraimanMuniz, 42).unwrap();
        assert!(
            rd.beta_depth >= -1e-10,
            "Beta depth should be ≥ 0: {}",
            rd.beta_depth
        );
    }

    #[test]
    fn test_regression_depth_score_lengths() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let rd = regression_depth(&fit, &data, &y, None, 20, DepthType::ModifiedBand, 42).unwrap();
        assert_eq!(rd.score_depths.len(), 30);
    }

    #[test]
    fn test_regression_depth_types_all_work() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        for dt in [
            DepthType::FraimanMuniz,
            DepthType::ModifiedBand,
            DepthType::FunctionalSpatial,
        ] {
            let rd = regression_depth(&fit, &data, &y, None, 10, dt, 42);
            assert!(rd.is_some(), "Depth type {:?} should work", dt);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Stability tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_stability_beta_std_nonneg() {
        let (data, y) = generate_test_data(30, 50, 42);
        let sa = explanation_stability(&data, &y, None, 3, 20, 42).unwrap();
        for (j, &s) in sa.beta_t_std.iter().enumerate() {
            assert!(s >= 0.0, "Std should be ≥ 0 at {}: {}", j, s);
        }
    }

    #[test]
    fn test_stability_coefficient_std_length() {
        let (data, y) = generate_test_data(30, 50, 42);
        let sa = explanation_stability(&data, &y, None, 3, 20, 42).unwrap();
        assert_eq!(sa.coefficient_std.len(), 3);
    }

    #[test]
    fn test_stability_importance_bounded() {
        let (data, y) = generate_test_data(30, 50, 42);
        let sa = explanation_stability(&data, &y, None, 3, 20, 42).unwrap();
        assert!(
            sa.importance_stability >= -1.0 - 1e-10 && sa.importance_stability <= 1.0 + 1e-10,
            "Importance stability out of range: {}",
            sa.importance_stability
        );
    }

    #[test]
    fn test_stability_more_boots_more_stable() {
        let (data, y) = generate_test_data(40, 50, 42);
        let sa1 = explanation_stability(&data, &y, None, 3, 5, 42).unwrap();
        let sa2 = explanation_stability(&data, &y, None, 3, 50, 42).unwrap();
        // More bootstrap runs should give ≥ n_boot_success
        assert!(sa2.n_boot_success >= sa1.n_boot_success);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Anchor tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_anchor_precision_meets_threshold() {
        let (data, y) = generate_test_data(40, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let ar = anchor_explanation(&fit, &data, None, 0, 0.8, 5).unwrap();
        assert!(
            ar.rule.precision >= 0.8 - 1e-10,
            "Precision {} should meet 0.8",
            ar.rule.precision
        );
    }

    #[test]
    fn test_anchor_coverage_range() {
        let (data, y) = generate_test_data(40, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let ar = anchor_explanation(&fit, &data, None, 0, 0.8, 5).unwrap();
        assert!(
            ar.rule.coverage > 0.0 && ar.rule.coverage <= 1.0,
            "Coverage out of range: {}",
            ar.rule.coverage
        );
    }

    #[test]
    fn test_anchor_observation_matches() {
        let (data, y) = generate_test_data(40, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let ar = anchor_explanation(&fit, &data, None, 5, 0.8, 5).unwrap();
        assert_eq!(ar.observation, 5);
    }

    #[test]
    fn test_anchor_invalid_obs_none() {
        let (data, y) = generate_test_data(40, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        assert!(anchor_explanation(&fit, &data, None, 100, 0.8, 5).is_none());
    }
}
