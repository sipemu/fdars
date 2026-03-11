//! Generic explainability for any FPC-based model.
//!
//! Provides the [`FpcPredictor`] trait and generic functions that work with
//! any model that implements it — including linear regression, logistic regression,
//! and classification models (LDA, QDA, kNN).
//!
//! The generic functions delegate to internal helpers from [`crate::explain`].

use crate::explain::{
    accumulate_kernel_shap_sample, anchor_beam_search, build_coalition_scores,
    build_stability_result, clone_scores_matrix, compute_ale, compute_column_means,
    compute_conditioning_bins, compute_domain_selection, compute_kernel_mean, compute_lime,
    compute_mean_scalar, compute_saliency_map, compute_sobol_component, compute_vif_from_scores,
    compute_witness, gaussian_kernel_matrix, generate_sobol_matrices, get_obs_scalar,
    greedy_prototype_selection, ice_to_pdp, make_grid, mean_absolute_column, median_bandwidth,
    permute_component, project_scores, reconstruct_delta_function, sample_random_coalition,
    shapley_kernel_weight, solve_kernel_shap_obs, subsample_rows, AleResult, AnchorResult,
    ConditionalPermutationImportanceResult, CounterfactualResult, DomainSelectionResult,
    FpcPermutationImportance, FpcShapValues, FriedmanHResult, FunctionalPdpResult,
    FunctionalSaliencyResult, LimeResult, PrototypeCriticismResult, SobolIndicesResult,
    StabilityAnalysisResult, VifResult,
};
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{
    fregre_lm, functional_logistic, sigmoid, FregreLmResult, FunctionalLogisticResult,
};
use rand::prelude::*;

// ---------------------------------------------------------------------------
// TaskType + FpcPredictor trait
// ---------------------------------------------------------------------------

/// The type of prediction task a model solves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    Regression,
    BinaryClassification,
    MulticlassClassification(usize),
}

/// Trait abstracting over any FPC-based model for generic explainability.
///
/// Implement this for a model that projects functional data onto FPC scores
/// and produces a scalar prediction (value, probability, or class label).
pub trait FpcPredictor {
    /// Mean function from FPCA (length m).
    fn fpca_mean(&self) -> &[f64];

    /// Rotation matrix from FPCA (m × ncomp).
    fn fpca_rotation(&self) -> &FdMatrix;

    /// Number of FPC components used.
    fn ncomp(&self) -> usize;

    /// Training FPC scores matrix (n × ncomp).
    fn training_scores(&self) -> &FdMatrix;

    /// What kind of prediction task this model solves.
    fn task_type(&self) -> TaskType;

    /// Predict from FPC scores + optional scalar covariates → single f64.
    ///
    /// - **Regression**: predicted value
    /// - **Binary classification**: P(Y=1)
    /// - **Multiclass**: predicted class label as f64
    fn predict_from_scores(&self, scores: &[f64], scalar_covariates: Option<&[f64]>) -> f64;

    /// Project functional data to FPC scores.
    fn project(&self, data: &FdMatrix) -> FdMatrix {
        project_scores(data, self.fpca_mean(), self.fpca_rotation(), self.ncomp())
    }
}

// ---------------------------------------------------------------------------
// Implement FpcPredictor for FregreLmResult
// ---------------------------------------------------------------------------

impl FpcPredictor for FregreLmResult {
    fn fpca_mean(&self) -> &[f64] {
        &self.fpca.mean
    }

    fn fpca_rotation(&self) -> &FdMatrix {
        &self.fpca.rotation
    }

    fn ncomp(&self) -> usize {
        self.ncomp
    }

    fn training_scores(&self) -> &FdMatrix {
        &self.fpca.scores
    }

    fn task_type(&self) -> TaskType {
        TaskType::Regression
    }

    fn predict_from_scores(&self, scores: &[f64], scalar_covariates: Option<&[f64]>) -> f64 {
        let ncomp = self.ncomp;
        let mut yhat = self.coefficients[0]; // intercept
        for k in 0..ncomp {
            yhat += self.coefficients[1 + k] * scores[k];
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..self.gamma.len() {
                yhat += self.gamma[j] * sc[j];
            }
        }
        yhat
    }
}

// ---------------------------------------------------------------------------
// Implement FpcPredictor for FunctionalLogisticResult
// ---------------------------------------------------------------------------

impl FpcPredictor for FunctionalLogisticResult {
    fn fpca_mean(&self) -> &[f64] {
        &self.fpca.mean
    }

    fn fpca_rotation(&self) -> &FdMatrix {
        &self.fpca.rotation
    }

    fn ncomp(&self) -> usize {
        self.ncomp
    }

    fn training_scores(&self) -> &FdMatrix {
        &self.fpca.scores
    }

    fn task_type(&self) -> TaskType {
        TaskType::BinaryClassification
    }

    fn predict_from_scores(&self, scores: &[f64], scalar_covariates: Option<&[f64]>) -> f64 {
        let ncomp = self.ncomp;
        let mut eta = self.intercept;
        for k in 0..ncomp {
            eta += self.coefficients[1 + k] * scores[k];
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..self.gamma.len() {
                eta += self.gamma[j] * sc[j];
            }
        }
        sigmoid(eta)
    }
}

// ---------------------------------------------------------------------------
// Generic helper: build a predict closure from an FpcPredictor
// ---------------------------------------------------------------------------

/// Compute the baseline metric for a model on training data.
fn compute_baseline_metric(
    model: &dyn FpcPredictor,
    scores: &FdMatrix,
    y: &[f64],
    n: usize,
) -> f64 {
    match model.task_type() {
        TaskType::Regression => {
            // R²
            let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
            let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
            if ss_tot == 0.0 {
                return 0.0;
            }
            let ss_res: f64 = (0..n)
                .map(|i| {
                    let s: Vec<f64> = (0..model.ncomp()).map(|k| scores[(i, k)]).collect();
                    let pred = model.predict_from_scores(&s, None);
                    (y[i] - pred).powi(2)
                })
                .sum();
            1.0 - ss_res / ss_tot
        }
        TaskType::BinaryClassification => {
            let correct: usize = (0..n)
                .filter(|&i| {
                    let s: Vec<f64> = (0..model.ncomp()).map(|k| scores[(i, k)]).collect();
                    let pred = model.predict_from_scores(&s, None);
                    let pred_class = if pred >= 0.5 { 1.0 } else { 0.0 };
                    (pred_class - y[i]).abs() < 1e-10
                })
                .count();
            correct as f64 / n as f64
        }
        TaskType::MulticlassClassification(_) => {
            let correct: usize = (0..n)
                .filter(|&i| {
                    let s: Vec<f64> = (0..model.ncomp()).map(|k| scores[(i, k)]).collect();
                    let pred = model.predict_from_scores(&s, None);
                    (pred.round() - y[i]).abs() < 1e-10
                })
                .count();
            correct as f64 / n as f64
        }
    }
}

/// Compute the metric for permuted scores.
fn compute_metric_from_score_matrix(
    model: &dyn FpcPredictor,
    score_mat: &FdMatrix,
    y: &[f64],
    n: usize,
) -> f64 {
    let ncomp = model.ncomp();
    match model.task_type() {
        TaskType::Regression => {
            let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
            let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
            if ss_tot == 0.0 {
                return 0.0;
            }
            let ss_res: f64 = (0..n)
                .map(|i| {
                    let s: Vec<f64> = (0..ncomp).map(|k| score_mat[(i, k)]).collect();
                    let pred = model.predict_from_scores(&s, None);
                    (y[i] - pred).powi(2)
                })
                .sum();
            1.0 - ss_res / ss_tot
        }
        TaskType::BinaryClassification => {
            let correct: usize = (0..n)
                .filter(|&i| {
                    let s: Vec<f64> = (0..ncomp).map(|k| score_mat[(i, k)]).collect();
                    let pred = model.predict_from_scores(&s, None);
                    let pred_class = if pred >= 0.5 { 1.0 } else { 0.0 };
                    (pred_class - y[i]).abs() < 1e-10
                })
                .count();
            correct as f64 / n as f64
        }
        TaskType::MulticlassClassification(_) => {
            let correct: usize = (0..n)
                .filter(|&i| {
                    let s: Vec<f64> = (0..ncomp).map(|k| score_mat[(i, k)]).collect();
                    let pred = model.predict_from_scores(&s, None);
                    (pred.round() - y[i]).abs() < 1e-10
                })
                .count();
            correct as f64 / n as f64
        }
    }
}

// ===========================================================================
// 1. Generic PDP
// ===========================================================================

/// Generic partial dependence plot / ICE curves for any FPC-based model.
pub fn generic_pdp(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_grid: usize,
) -> Option<FunctionalPdpResult> {
    let (n, m) = data.shape();
    if component >= model.ncomp() || n_grid < 2 || n == 0 || m != model.fpca_mean().len() {
        return None;
    }
    let ncomp = model.ncomp();
    let scores = model.project(data);
    let grid_values = make_grid(&scores, component, n_grid);

    let p_scalar = scalar_covariates.map_or(0, |sc| sc.ncols());
    let mut ice_curves = FdMatrix::zeros(n, n_grid);
    for i in 0..n {
        let mut obs_scores: Vec<f64> = (0..ncomp).map(|k| scores[(i, k)]).collect();
        let obs_z: Option<Vec<f64>> = if p_scalar > 0 {
            scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i, j)]).collect())
        } else {
            None
        };
        for g in 0..n_grid {
            obs_scores[component] = grid_values[g];
            ice_curves[(i, g)] = model.predict_from_scores(&obs_scores, obs_z.as_deref());
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

// ===========================================================================
// 2. Generic Permutation Importance
// ===========================================================================

/// Generic permutation importance for any FPC-based model.
///
/// Uses R² for regression, accuracy for classification.
pub fn generic_permutation_importance(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    y: &[f64],
    n_perm: usize,
    seed: u64,
) -> Option<FpcPermutationImportance> {
    let (n, m) = data.shape();
    if n == 0 || n != y.len() || m != model.fpca_mean().len() || n_perm == 0 {
        return None;
    }
    let ncomp = model.ncomp();
    let scores = model.project(data);
    let baseline = compute_baseline_metric(model, &scores, y, n);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut importance = vec![0.0; ncomp];
    let mut permuted_metric = vec![0.0; ncomp];

    for k in 0..ncomp {
        let mut sum_metric = 0.0;
        for _ in 0..n_perm {
            let mut perm_scores = clone_scores_matrix(&scores, n, ncomp);
            let mut idx: Vec<usize> = (0..n).collect();
            idx.shuffle(&mut rng);
            for i in 0..n {
                perm_scores[(i, k)] = scores[(idx[i], k)];
            }
            sum_metric += compute_metric_from_score_matrix(model, &perm_scores, y, n);
        }
        let mean_perm = sum_metric / n_perm as f64;
        permuted_metric[k] = mean_perm;
        importance[k] = baseline - mean_perm;
    }

    Some(FpcPermutationImportance {
        importance,
        baseline_metric: baseline,
        permuted_metric,
    })
}

// ===========================================================================
// 3. Generic Friedman H-statistic
// ===========================================================================

/// Generic Friedman H-statistic for interaction between two FPC components.
pub fn generic_friedman_h(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component_j: usize,
    component_k: usize,
    n_grid: usize,
) -> Option<FriedmanHResult> {
    if component_j == component_k {
        return None;
    }
    let (n, m) = data.shape();
    let ncomp = model.ncomp();
    if n == 0 || m != model.fpca_mean().len() || n_grid < 2 {
        return None;
    }
    if component_j >= ncomp || component_k >= ncomp {
        return None;
    }

    let scores = model.project(data);
    let grid_j = make_grid(&scores, component_j, n_grid);
    let grid_k = make_grid(&scores, component_k, n_grid);
    let p_scalar = scalar_covariates.map_or(0, |sc| sc.ncols());

    // Compute 1D PDPs via generic predict
    let pdp_j: Vec<f64> = grid_j
        .iter()
        .map(|&gval| {
            let mut sum = 0.0;
            for i in 0..n {
                let mut s: Vec<f64> = (0..ncomp).map(|c| scores[(i, c)]).collect();
                s[component_j] = gval;
                let obs_z: Option<Vec<f64>> = if p_scalar > 0 {
                    scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i, j)]).collect())
                } else {
                    None
                };
                sum += model.predict_from_scores(&s, obs_z.as_deref());
            }
            sum / n as f64
        })
        .collect();

    let pdp_k: Vec<f64> = grid_k
        .iter()
        .map(|&gval| {
            let mut sum = 0.0;
            for i in 0..n {
                let mut s: Vec<f64> = (0..ncomp).map(|c| scores[(i, c)]).collect();
                s[component_k] = gval;
                let obs_z: Option<Vec<f64>> = if p_scalar > 0 {
                    scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i, j)]).collect())
                } else {
                    None
                };
                sum += model.predict_from_scores(&s, obs_z.as_deref());
            }
            sum / n as f64
        })
        .collect();

    // Compute 2D PDP
    let mut pdp_2d = FdMatrix::zeros(n_grid, n_grid);
    for (gj_idx, &gj) in grid_j.iter().enumerate() {
        for (gk_idx, &gk) in grid_k.iter().enumerate() {
            let mut sum = 0.0;
            for i in 0..n {
                let mut s: Vec<f64> = (0..ncomp).map(|c| scores[(i, c)]).collect();
                s[component_j] = gj;
                s[component_k] = gk;
                let obs_z: Option<Vec<f64>> = if p_scalar > 0 {
                    scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i, j)]).collect())
                } else {
                    None
                };
                sum += model.predict_from_scores(&s, obs_z.as_deref());
            }
            pdp_2d[(gj_idx, gk_idx)] = sum / n as f64;
        }
    }

    // Mean prediction
    let f_bar: f64 = (0..n)
        .map(|i| {
            let s: Vec<f64> = (0..ncomp).map(|c| scores[(i, c)]).collect();
            let obs_z: Option<Vec<f64>> = if p_scalar > 0 {
                scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i, j)]).collect())
            } else {
                None
            };
            model.predict_from_scores(&s, obs_z.as_deref())
        })
        .sum::<f64>()
        / n as f64;

    let h_squared = crate::explain::compute_h_squared(&pdp_2d, &pdp_j, &pdp_k, f_bar, n_grid);

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
// 4. Generic SHAP Values
// ===========================================================================

/// Generic Kernel SHAP values for any FPC-based model.
///
/// For nonlinear models uses sampling-based Kernel SHAP; linear models get
/// the same approximation (which converges to exact with enough samples).
pub fn generic_shap_values(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    n_samples: usize,
    seed: u64,
) -> Option<FpcShapValues> {
    let (n, m) = data.shape();
    if n == 0 || m != model.fpca_mean().len() || n_samples == 0 {
        return None;
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return None;
    }
    let p_scalar = scalar_covariates.map_or(0, |sc| sc.ncols());
    let scores = model.project(data);
    let mean_scores = compute_column_means(&scores, ncomp);
    let mean_z = compute_mean_scalar(scalar_covariates, p_scalar, n);

    let base_value: f64 = (0..n)
        .map(|i| {
            let s: Vec<f64> = (0..ncomp).map(|k| scores[(i, k)]).collect();
            let obs_z: Option<Vec<f64>> = if p_scalar > 0 {
                scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i, j)]).collect())
            } else {
                None
            };
            model.predict_from_scores(&s, obs_z.as_deref())
        })
        .sum::<f64>()
        / n as f64;

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

            let f_coal = model.predict_from_scores(
                &coal_scores,
                if obs_z.is_empty() { None } else { Some(&obs_z) },
            );
            let f_base = model.predict_from_scores(
                &mean_scores,
                if obs_z.is_empty() { None } else { Some(&obs_z) },
            );
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

// ===========================================================================
// 5. Generic ALE
// ===========================================================================

/// Generic ALE plot for an FPC component in any FPC-based model.
pub fn generic_ale(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_bins: usize,
) -> Option<AleResult> {
    let (n, m) = data.shape();
    if n < 2 || m != model.fpca_mean().len() || n_bins == 0 || component >= model.ncomp() {
        return None;
    }
    let ncomp = model.ncomp();
    let p_scalar = scalar_covariates.map_or(0, |sc| sc.ncols());
    let scores = model.project(data);

    let predict = |obs_scores: &[f64], obs_scalar: Option<&[f64]>| -> f64 {
        model.predict_from_scores(obs_scores, obs_scalar)
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

// ===========================================================================
// 6. Generic Sobol Indices
// ===========================================================================

/// Generic Sobol sensitivity indices for any FPC-based model (Saltelli MC).
pub fn generic_sobol_indices(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    n_samples: usize,
    seed: u64,
) -> Option<SobolIndicesResult> {
    let (n, m) = data.shape();
    if n < 2 || m != model.fpca_mean().len() || n_samples == 0 {
        return None;
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return None;
    }
    let p_scalar = scalar_covariates.map_or(0, |sc| sc.ncols());
    let scores = model.project(data);
    let mean_z = compute_mean_scalar(scalar_covariates, p_scalar, n);

    let eval_model = |s: &[f64]| -> f64 {
        let sc = if mean_z.is_empty() {
            None
        } else {
            Some(mean_z.as_slice())
        };
        model.predict_from_scores(s, sc)
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
// 7. Generic Conditional Permutation Importance
// ===========================================================================

/// Generic conditional permutation importance for any FPC-based model.
pub fn generic_conditional_permutation_importance(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    y: &[f64],
    _scalar_covariates: Option<&FdMatrix>,
    n_bins: usize,
    n_perm: usize,
    seed: u64,
) -> Option<ConditionalPermutationImportanceResult> {
    let (n, m) = data.shape();
    if n == 0 || n != y.len() || m != model.fpca_mean().len() || n_perm == 0 || n_bins == 0 {
        return None;
    }
    let ncomp = model.ncomp();
    let scores = model.project(data);

    let baseline = compute_baseline_metric(model, &scores, y, n);

    let metric_fn =
        |score_mat: &FdMatrix| -> f64 { compute_metric_from_score_matrix(model, score_mat, y, n) };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut importance = vec![0.0; ncomp];
    let mut permuted_metric = vec![0.0; ncomp];
    let mut unconditional_importance = vec![0.0; ncomp];

    for k in 0..ncomp {
        let bins = compute_conditioning_bins(&scores, ncomp, k, n, n_bins);
        let (mean_cond, mean_uncond) =
            permute_component(&scores, &bins, k, n, ncomp, n_perm, &mut rng, &metric_fn);
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
// 8. Generic Counterfactual
// ===========================================================================

/// Generic counterfactual explanation for any FPC-based model.
///
/// For regression: uses analytical projection toward target_value.
/// For classification: uses gradient descent toward the opposite class.
pub fn generic_counterfactual(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    _scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    target_value: f64,
    max_iter: usize,
    step_size: f64,
) -> Option<CounterfactualResult> {
    let (n, m) = data.shape();
    if observation >= n || m != model.fpca_mean().len() {
        return None;
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return None;
    }
    let scores = model.project(data);
    let original_scores: Vec<f64> = (0..ncomp).map(|k| scores[(observation, k)]).collect();
    let original_prediction = model.predict_from_scores(&original_scores, None);

    match model.task_type() {
        TaskType::Regression => {
            // Analytical: find nearest score change along gradient direction
            // Gradient of predict w.r.t. scores estimated by finite differences
            let eps = 1e-5;
            let mut grad = vec![0.0; ncomp];
            for k in 0..ncomp {
                let mut s_plus = original_scores.clone();
                s_plus[k] += eps;
                let f_plus = model.predict_from_scores(&s_plus, None);
                grad[k] = (f_plus - original_prediction) / eps;
            }
            let grad_norm_sq: f64 = grad.iter().map(|g| g * g).sum();
            if grad_norm_sq < 1e-30 {
                return None;
            }

            let gap = target_value - original_prediction;
            let delta_scores: Vec<f64> = grad.iter().map(|&gk| gap * gk / grad_norm_sq).collect();
            let counterfactual_scores: Vec<f64> = original_scores
                .iter()
                .zip(&delta_scores)
                .map(|(&o, &d)| o + d)
                .collect();
            let delta_function =
                reconstruct_delta_function(&delta_scores, model.fpca_rotation(), ncomp, m);
            let distance: f64 = delta_scores.iter().map(|d| d * d).sum::<f64>().sqrt();
            let counterfactual_prediction = model.predict_from_scores(&counterfactual_scores, None);

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
        TaskType::BinaryClassification => {
            // Gradient descent toward opposite class (binary: P(Y=1) threshold 0.5)
            let mut current_scores = original_scores.clone();
            let mut current_pred = original_prediction;
            let original_class = if original_prediction >= 0.5 { 1.0 } else { 0.0 };
            let target_class = 1.0 - original_class;

            let mut found = false;
            let eps = 1e-5;
            for _ in 0..max_iter {
                current_pred = model.predict_from_scores(&current_scores, None);
                let pred_class: f64 = if current_pred >= 0.5 { 1.0 } else { 0.0 };
                if (pred_class - target_class).abs() < 1e-10 {
                    found = true;
                    break;
                }
                let mut grads = vec![0.0; ncomp];
                for k in 0..ncomp {
                    let mut s_plus = current_scores.clone();
                    s_plus[k] += eps;
                    let f_plus = model.predict_from_scores(&s_plus, None);
                    grads[k] = (f_plus - current_pred) / eps;
                }
                for k in 0..ncomp {
                    current_scores[k] -= step_size * (current_pred - target_class) * grads[k];
                }
            }
            if !found {
                current_pred = model.predict_from_scores(&current_scores, None);
                let pred_class = if current_pred >= 0.5 { 1.0 } else { 0.0 };
                found = (pred_class - target_class).abs() < 1e-10;
            }

            let delta_scores: Vec<f64> = current_scores
                .iter()
                .zip(&original_scores)
                .map(|(&c, &o)| c - o)
                .collect();
            let delta_function =
                reconstruct_delta_function(&delta_scores, model.fpca_rotation(), ncomp, m);
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
        TaskType::MulticlassClassification(_) => {
            // Gradient descent toward nearest different class (multiclass: pred is class label)
            let mut current_scores = original_scores.clone();
            let mut current_pred = original_prediction;
            let original_class = original_prediction.round();

            let mut found = false;
            let eps = 1e-5;
            for _ in 0..max_iter {
                current_pred = model.predict_from_scores(&current_scores, None);
                let pred_class = current_pred.round();
                if (pred_class - original_class).abs() > 0.5 {
                    found = true;
                    break;
                }
                let mut grads = vec![0.0; ncomp];
                for k in 0..ncomp {
                    let mut s_plus = current_scores.clone();
                    s_plus[k] += eps;
                    let f_plus = model.predict_from_scores(&s_plus, None);
                    grads[k] = (f_plus - current_pred) / eps;
                }
                let grad_norm: f64 = grads.iter().map(|g| g * g).sum::<f64>().sqrt().max(1e-12);
                for k in 0..ncomp {
                    current_scores[k] += step_size * grads[k] / grad_norm;
                }
            }
            if !found {
                current_pred = model.predict_from_scores(&current_scores, None);
                found = (current_pred.round() - original_class).abs() > 0.5;
            }

            let delta_scores: Vec<f64> = current_scores
                .iter()
                .zip(&original_scores)
                .map(|(&c, &o)| c - o)
                .collect();
            let delta_function =
                reconstruct_delta_function(&delta_scores, model.fpca_rotation(), ncomp, m);
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
    }
}

// ===========================================================================
// 9. Generic LIME
// ===========================================================================

/// Generic LIME explanation for any FPC-based model.
pub fn generic_lime(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    _scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    n_samples: usize,
    kernel_width: f64,
    seed: u64,
) -> Option<LimeResult> {
    let (n, m) = data.shape();
    if observation >= n || m != model.fpca_mean().len() || n_samples == 0 || kernel_width <= 0.0 {
        return None;
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return None;
    }
    let scores = model.project(data);
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

    let predict = |s: &[f64]| -> f64 { model.predict_from_scores(s, None) };

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

// ===========================================================================
// 10. Generic Saliency
// ===========================================================================

/// Generic functional saliency maps via SHAP-weighted rotation.
///
/// Lifts FPC-level attributions to the function domain.
pub fn generic_saliency(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    n_samples: usize,
    seed: u64,
) -> Option<FunctionalSaliencyResult> {
    let (n, m) = data.shape();
    if n == 0 || m != model.fpca_mean().len() {
        return None;
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return None;
    }

    // Get SHAP values first
    let shap = generic_shap_values(model, data, scalar_covariates, n_samples, seed)?;

    // Compute per-observation saliency: saliency[(i,j)] = Σ_k shap[(i,k)] × rotation[(j,k)]
    let scores = model.project(data);
    let mean_scores = compute_column_means(&scores, ncomp);

    // Weights = mean |SHAP_k| / mean |score_k - mean_k| ≈ effective coefficient magnitude
    let mut weights = vec![0.0; ncomp];
    for k in 0..ncomp {
        let mut sum_shap = 0.0;
        let mut sum_score_dev = 0.0;
        for i in 0..n {
            sum_shap += shap.values[(i, k)].abs();
            sum_score_dev += (scores[(i, k)] - mean_scores[k]).abs();
        }
        weights[k] = if sum_score_dev > 1e-15 {
            sum_shap / sum_score_dev
        } else {
            0.0
        };
    }

    let saliency_map = compute_saliency_map(
        &scores,
        &mean_scores,
        &weights,
        model.fpca_rotation(),
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

// ===========================================================================
// 11. Generic Domain Selection
// ===========================================================================

/// Generic domain selection using SHAP-based functional importance.
///
/// Computes pointwise importance from the model's effective β(t) reconstruction
/// via SHAP weights, then finds important intervals via sliding window.
pub fn generic_domain_selection(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    window_width: usize,
    threshold: f64,
    n_samples: usize,
    seed: u64,
) -> Option<DomainSelectionResult> {
    let (n, m) = data.shape();
    if n == 0 || m != model.fpca_mean().len() {
        return None;
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return None;
    }

    // Reconstruct effective β(t) = Σ_k w_k × φ_k(t) using SHAP-derived weights
    let shap = generic_shap_values(model, data, scalar_covariates, n_samples, seed)?;
    let scores = model.project(data);
    let mean_scores = compute_column_means(&scores, ncomp);

    let mut effective_weights = vec![0.0; ncomp];
    for k in 0..ncomp {
        let mut sum_shap = 0.0;
        let mut sum_score_dev = 0.0;
        for i in 0..n {
            sum_shap += shap.values[(i, k)].abs();
            sum_score_dev += (scores[(i, k)] - mean_scores[k]).abs();
        }
        effective_weights[k] = if sum_score_dev > 1e-15 {
            sum_shap / sum_score_dev
        } else {
            0.0
        };
    }

    // Reconstruct β(t) = Σ_k w_k × φ_k(t)
    let rotation = model.fpca_rotation();
    let mut beta_t = vec![0.0; m];
    for j in 0..m {
        for k in 0..ncomp {
            beta_t[j] += effective_weights[k] * rotation[(j, k)];
        }
    }

    compute_domain_selection(&beta_t, window_width, threshold)
}

// ===========================================================================
// 12. Generic Anchor
// ===========================================================================

/// Generic anchor explanation for any FPC-based model.
pub fn generic_anchor(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    precision_threshold: f64,
    n_bins: usize,
) -> Option<AnchorResult> {
    let (n, m) = data.shape();
    if n == 0 || m != model.fpca_mean().len() || observation >= n || n_bins < 2 {
        return None;
    }
    let ncomp = model.ncomp();
    let scores = model.project(data);
    let p_scalar = scalar_covariates.map_or(0, |sc| sc.ncols());

    let obs_scores: Vec<f64> = (0..ncomp).map(|k| scores[(observation, k)]).collect();
    let obs_z: Option<Vec<f64>> = if p_scalar > 0 {
        scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(observation, j)]).collect())
    } else {
        None
    };
    let obs_pred = model.predict_from_scores(&obs_scores, obs_z.as_deref());

    // Pre-compute all predictions for the same_pred closure
    let all_preds: Vec<f64> = (0..n)
        .map(|i| {
            let s: Vec<f64> = (0..ncomp).map(|k| scores[(i, k)]).collect();
            let iz: Option<Vec<f64>> = if p_scalar > 0 {
                scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i, j)]).collect())
            } else {
                None
            };
            model.predict_from_scores(&s, iz.as_deref())
        })
        .collect();

    let same_pred: Box<dyn Fn(usize) -> bool> = match model.task_type() {
        TaskType::Regression => {
            let pred_mean = all_preds.iter().sum::<f64>() / n as f64;
            let pred_std = (all_preds
                .iter()
                .map(|&p| (p - pred_mean).powi(2))
                .sum::<f64>()
                / (n - 1).max(1) as f64)
                .sqrt();
            let tol = pred_std.max(1e-10);
            Box::new(move |i: usize| (all_preds[i] - obs_pred).abs() <= tol)
        }
        TaskType::BinaryClassification => {
            let obs_class: f64 = if obs_pred >= 0.5 { 1.0 } else { 0.0 };
            Box::new(move |i: usize| {
                let class_i: f64 = if all_preds[i] >= 0.5 { 1.0 } else { 0.0 };
                (class_i - obs_class).abs() < 1e-10
            })
        }
        TaskType::MulticlassClassification(_) => {
            let obs_class = obs_pred.round();
            Box::new(move |i: usize| (all_preds[i].round() - obs_class).abs() < 1e-10)
        }
    };

    let (rule, _) = anchor_beam_search(
        &scores,
        ncomp,
        n,
        observation,
        precision_threshold,
        n_bins,
        &*same_pred,
    );

    Some(AnchorResult {
        rule,
        observation,
        predicted_value: obs_pred,
    })
}

// ===========================================================================
// 13. Generic Stability
// ===========================================================================

/// Generic explanation stability via bootstrap resampling.
///
/// Refits the model on bootstrap samples and measures variability of
/// coefficients, β(t), and metric (R² or accuracy).
///
/// Note: This only works for regression and logistic models since it requires
/// refitting. For classification models, bootstrap refitting is not yet supported.
pub fn generic_stability(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    n_boot: usize,
    seed: u64,
    task_type: TaskType,
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

    match task_type {
        TaskType::Regression => {
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
        }
        TaskType::BinaryClassification => {
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
        }
        TaskType::MulticlassClassification(_) => {
            return None; // not supported for multiclass yet
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

// ===========================================================================
// 14. Generic VIF
// ===========================================================================

/// Generic VIF for any FPC-based model (only depends on score matrix).
pub fn generic_vif(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
) -> Option<VifResult> {
    let (n, m) = data.shape();
    if n == 0 || m != model.fpca_mean().len() {
        return None;
    }
    let ncomp = model.ncomp();
    let scores = model.project(data);
    compute_vif_from_scores(&scores, ncomp, scalar_covariates, n)
}

// ===========================================================================
// 15. Generic Prototype / Criticism
// ===========================================================================

/// Generic prototype/criticism selection for any FPC-based model.
pub fn generic_prototype_criticism(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    n_prototypes: usize,
    n_criticisms: usize,
) -> Option<PrototypeCriticismResult> {
    let (n, m) = data.shape();
    if n == 0 || m != model.fpca_mean().len() {
        return None;
    }
    let ncomp = model.ncomp();
    if ncomp == 0 || n_prototypes == 0 || n_prototypes > n {
        return None;
    }
    let n_crit = n_criticisms.min(n.saturating_sub(n_prototypes));

    let scores = model.project(data);
    let bandwidth = median_bandwidth(&scores, n, ncomp);
    let kernel = gaussian_kernel_matrix(&scores, ncomp, bandwidth);
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
