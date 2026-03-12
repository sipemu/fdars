//! Generic explainability for any FPC-based model.
//!
//! Provides the [`FpcPredictor`] trait and generic functions that work with
//! any model that implements it — including linear regression, logistic regression,
//! and classification models (LDA, QDA, kNN).
//!
//! The generic functions delegate to internal helpers from [`crate::explain`].

use crate::error::FdarError;
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
use crate::iter_maybe_parallel;
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
pub trait FpcPredictor: Send + Sync {
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
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its
/// column count does not match the model's FPCA mean length.
/// Returns [`FdarError::InvalidParameter`] if `component >= ncomp` or
/// `n_grid < 2`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_pdp(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_grid: usize,
) -> Result<FunctionalPdpResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0".into(),
            actual: "0 rows".into(),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    if component >= model.ncomp() {
        return Err(FdarError::InvalidParameter {
            parameter: "component",
            message: format!("component {} >= ncomp {}", component, model.ncomp()),
        });
    }
    if n_grid < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_grid",
            message: format!("n_grid must be >= 2, got {n_grid}"),
        });
    }
    let ncomp = model.ncomp();
    let scores = model.project(data);
    let grid_values = make_grid(&scores, component, n_grid);

    let p_scalar = scalar_covariates.map_or(0, super::matrix::FdMatrix::ncols);
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

    Ok(FunctionalPdpResult {
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
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows, its
/// column count does not match the model, or `y.len() != n`.
/// Returns [`FdarError::InvalidParameter`] if `n_perm` is zero.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_permutation_importance(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    y: &[f64],
    n_perm: usize,
    seed: u64,
) -> Result<FpcPermutationImportance, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0".into(),
            actual: "0 rows".into(),
        });
    }
    if n != y.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: n.to_string(),
            actual: y.len().to_string(),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    if n_perm == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_perm",
            message: "n_perm must be > 0".into(),
        });
    }
    let ncomp = model.ncomp();
    let scores = model.project(data);
    let baseline = compute_baseline_metric(model, &scores, y, n);

    #[cfg(feature = "parallel")]
    use rayon::iter::ParallelIterator;

    let results: Vec<(f64, f64)> = iter_maybe_parallel!(0..ncomp)
        .map(|k| {
            let mut rng_k = StdRng::seed_from_u64(seed.wrapping_add(k as u64));
            let mut sum_metric = 0.0;
            for _ in 0..n_perm {
                let mut perm_scores = clone_scores_matrix(&scores, n, ncomp);
                let mut idx: Vec<usize> = (0..n).collect();
                idx.shuffle(&mut rng_k);
                for i in 0..n {
                    perm_scores[(i, k)] = scores[(idx[i], k)];
                }
                sum_metric += compute_metric_from_score_matrix(model, &perm_scores, y, n);
            }
            let mean_perm = sum_metric / n_perm as f64;
            (baseline - mean_perm, mean_perm)
        })
        .collect();

    let importance: Vec<f64> = results.iter().map(|&(imp, _)| imp).collect();
    let permuted_metric: Vec<f64> = results.iter().map(|&(_, pm)| pm).collect();

    Ok(FpcPermutationImportance {
        importance,
        baseline_metric: baseline,
        permuted_metric,
    })
}

// ===========================================================================
// 3. Generic Friedman H-statistic
// ===========================================================================

/// Compute the 2D partial dependence surface over a grid of two components.
///
/// For each pair `(grid_j[gj], grid_k[gk])`, fixes components `component_j`
/// and `component_k` to those grid values and averages the model prediction
/// over all `n` observations.
fn compute_pdp_grid_2d(
    model: &dyn FpcPredictor,
    scores: &FdMatrix,
    grid_j: &[f64],
    grid_k: &[f64],
    component_j: usize,
    component_k: usize,
    ncomp: usize,
    n: usize,
    scalar_covariates: Option<&FdMatrix>,
    p_scalar: usize,
    n_grid: usize,
) -> FdMatrix {
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
    pdp_2d
}

/// Compute the H-squared statistic from marginal and joint PDP surfaces.
///
/// First computes the mean prediction `f_bar` (averaging over all observations),
/// then delegates to [`crate::explain::compute_h_squared`] for the actual
/// interaction / total-variance ratio.
fn compute_h_squared_from_pdps(
    model: &dyn FpcPredictor,
    scores: &FdMatrix,
    pdp_2d: &FdMatrix,
    pdp_j: &[f64],
    pdp_k: &[f64],
    ncomp: usize,
    n: usize,
    scalar_covariates: Option<&FdMatrix>,
    p_scalar: usize,
    n_grid: usize,
) -> f64 {
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
    crate::explain::compute_h_squared(pdp_2d, pdp_j, pdp_k, f_bar, n_grid)
}

/// Generic Friedman H-statistic for interaction between two FPC components.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `component_j == component_k`,
/// `n_grid < 2`, or either component index is `>= ncomp`.
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its
/// column count does not match the model.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_friedman_h(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component_j: usize,
    component_k: usize,
    n_grid: usize,
) -> Result<FriedmanHResult, FdarError> {
    if component_j == component_k {
        return Err(FdarError::InvalidParameter {
            parameter: "component_j/component_k",
            message: "component_j and component_k must differ".into(),
        });
    }
    let (n, m) = data.shape();
    let ncomp = model.ncomp();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0".into(),
            actual: "0 rows".into(),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    if n_grid < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_grid",
            message: format!("n_grid must be >= 2, got {n_grid}"),
        });
    }
    if component_j >= ncomp || component_k >= ncomp {
        return Err(FdarError::InvalidParameter {
            parameter: "component",
            message: format!(
                "component_j={component_j} or component_k={component_k} >= ncomp={ncomp}"
            ),
        });
    }

    let scores = model.project(data);
    let grid_j = make_grid(&scores, component_j, n_grid);
    let grid_k = make_grid(&scores, component_k, n_grid);
    let p_scalar = scalar_covariates.map_or(0, super::matrix::FdMatrix::ncols);

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

    let pdp_2d = compute_pdp_grid_2d(
        model,
        &scores,
        &grid_j,
        &grid_k,
        component_j,
        component_k,
        ncomp,
        n,
        scalar_covariates,
        p_scalar,
        n_grid,
    );

    let h_squared = compute_h_squared_from_pdps(
        model,
        &scores,
        &pdp_2d,
        &pdp_j,
        &pdp_k,
        ncomp,
        n,
        scalar_covariates,
        p_scalar,
        n_grid,
    );

    Ok(FriedmanHResult {
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
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its
/// column count does not match the model.
/// Returns [`FdarError::InvalidParameter`] if `n_samples` is zero or the
/// model has zero components.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_shap_values(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    n_samples: usize,
    seed: u64,
) -> Result<FpcShapValues, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0".into(),
            actual: "0 rows".into(),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    if n_samples == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_samples",
            message: "n_samples must be > 0".into(),
        });
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "model has 0 components".into(),
        });
    }
    let p_scalar = scalar_covariates.map_or(0, super::matrix::FdMatrix::ncols);
    let scores = model.project(data);
    let mean_scores = compute_column_means(&scores, ncomp);
    let mean_z = compute_mean_scalar(scalar_covariates, p_scalar, n);

    let base_value = model.predict_from_scores(
        &mean_scores,
        if mean_z.is_empty() {
            None
        } else {
            Some(&mean_z)
        },
    );

    #[cfg(feature = "parallel")]
    use rayon::iter::ParallelIterator;

    let rows: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let mut rng_i = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
            let obs_scores: Vec<f64> = (0..ncomp).map(|k| scores[(i, k)]).collect();
            let obs_z = get_obs_scalar(scalar_covariates, i, p_scalar, &mean_z);

            let mut ata = vec![0.0; ncomp * ncomp];
            let mut atb = vec![0.0; ncomp];

            for _ in 0..n_samples {
                let (coalition, s_size) = sample_random_coalition(&mut rng_i, ncomp);
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

            // Solve locally and return row
            let mut local_values = FdMatrix::zeros(1, ncomp);
            solve_kernel_shap_obs(&mut ata, &atb, ncomp, &mut local_values, 0);
            (0..ncomp).map(|k| local_values[(0, k)]).collect()
        })
        .collect();

    let mut values = FdMatrix::zeros(n, ncomp);
    for (i, row) in rows.iter().enumerate() {
        for (k, &v) in row.iter().enumerate() {
            values[(i, k)] = v;
        }
    }

    Ok(FpcShapValues {
        values,
        base_value,
        mean_scores,
    })
}

// ===========================================================================
// 5. Generic ALE
// ===========================================================================

/// Generic ALE plot for an FPC component in any FPC-based model.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 2 rows
/// or its column count does not match the model.
/// Returns [`FdarError::InvalidParameter`] if `n_bins` is zero or
/// `component >= ncomp`.
/// Returns [`FdarError::ComputationFailed`] if the internal ALE computation
/// fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_ale(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_bins: usize,
) -> Result<AleResult, FdarError> {
    let (n, m) = data.shape();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n >= 2".into(),
            actual: format!("{n} rows"),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    if n_bins == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_bins",
            message: "n_bins must be > 0".into(),
        });
    }
    if component >= model.ncomp() {
        return Err(FdarError::InvalidParameter {
            parameter: "component",
            message: format!("component {} >= ncomp {}", component, model.ncomp()),
        });
    }
    let ncomp = model.ncomp();
    let p_scalar = scalar_covariates.map_or(0, super::matrix::FdMatrix::ncols);
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
    .ok_or_else(|| FdarError::ComputationFailed {
        operation: "generic_ale",
        detail: "compute_ale returned None".into(),
    })
}

// ===========================================================================
// 6. Generic Sobol Indices
// ===========================================================================

/// Generic Sobol sensitivity indices for any FPC-based model (Saltelli MC).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 2 rows
/// or its column count does not match the model.
/// Returns [`FdarError::InvalidParameter`] if `n_samples` is zero or the
/// model has zero components.
/// Returns [`FdarError::ComputationFailed`] if the variance of model output
/// is near zero.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_sobol_indices(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    n_samples: usize,
    seed: u64,
) -> Result<SobolIndicesResult, FdarError> {
    let (n, m) = data.shape();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n >= 2".into(),
            actual: format!("{n} rows"),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    if n_samples == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_samples",
            message: "n_samples must be > 0".into(),
        });
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "model has 0 components".into(),
        });
    }
    let p_scalar = scalar_covariates.map_or(0, super::matrix::FdMatrix::ncols);
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
    // Monte Carlo estimate, population variance
    let var_fa = f_a.iter().map(|&v| (v - mean_fa).powi(2)).sum::<f64>() / n_samples as f64;

    if var_fa < 1e-15 {
        return Err(FdarError::ComputationFailed {
            operation: "generic_sobol_indices",
            detail: "variance of model output is near zero".into(),
        });
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

    Ok(SobolIndicesResult {
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
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows, its
/// column count does not match the model, or `y.len() != n`.
/// Returns [`FdarError::InvalidParameter`] if `n_perm` or `n_bins` is zero.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_conditional_permutation_importance(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    y: &[f64],
    _scalar_covariates: Option<&FdMatrix>,
    n_bins: usize,
    n_perm: usize,
    seed: u64,
) -> Result<ConditionalPermutationImportanceResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0".into(),
            actual: "0 rows".into(),
        });
    }
    if n != y.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: n.to_string(),
            actual: y.len().to_string(),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    if n_perm == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_perm",
            message: "n_perm must be > 0".into(),
        });
    }
    if n_bins == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_bins",
            message: "n_bins must be > 0".into(),
        });
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

    Ok(ConditionalPermutationImportanceResult {
        importance,
        baseline_metric: baseline,
        permuted_metric,
        unconditional_importance,
    })
}

// ===========================================================================
// 8. Generic Counterfactual
// ===========================================================================

/// Compute gradient of model prediction w.r.t. FPC scores via finite differences.
fn compute_gradient_finite_diff(
    model: &dyn FpcPredictor,
    scores: &[f64],
    ncomp: usize,
) -> Vec<f64> {
    let eps = 1e-5;
    let base = model.predict_from_scores(scores, None);
    let mut grad = vec![0.0; ncomp];
    for k in 0..ncomp {
        let mut s_plus = scores.to_vec();
        s_plus[k] += eps;
        grad[k] = (model.predict_from_scores(&s_plus, None) - base) / eps;
    }
    grad
}

/// Build a CounterfactualResult from original/final scores.
fn build_counterfactual_result(
    model: &dyn FpcPredictor,
    observation: usize,
    original_scores: Vec<f64>,
    final_scores: Vec<f64>,
    original_prediction: f64,
    ncomp: usize,
    m: usize,
    found: bool,
) -> CounterfactualResult {
    let delta_scores: Vec<f64> = final_scores
        .iter()
        .zip(&original_scores)
        .map(|(&c, &o)| c - o)
        .collect();
    let delta_function = reconstruct_delta_function(&delta_scores, model.fpca_rotation(), ncomp, m);
    let distance: f64 = delta_scores.iter().map(|d| d * d).sum::<f64>().sqrt();
    let counterfactual_prediction = model.predict_from_scores(&final_scores, None);

    CounterfactualResult {
        observation,
        original_scores,
        counterfactual_scores: final_scores,
        delta_scores,
        delta_function,
        distance,
        original_prediction,
        counterfactual_prediction,
        found,
    }
}

/// Gradient descent search for a counterfactual in classification models.
fn counterfactual_gd_search(
    model: &dyn FpcPredictor,
    original_scores: &[f64],
    max_iter: usize,
    ncomp: usize,
    converged: impl Fn(f64) -> bool,
    update: impl Fn(&mut [f64], &[f64], f64),
) -> (Vec<f64>, f64, bool) {
    let mut current_scores = original_scores.to_vec();
    let mut current_pred = model.predict_from_scores(&current_scores, None);
    let mut found = false;
    for _ in 0..max_iter {
        current_pred = model.predict_from_scores(&current_scores, None);
        if converged(current_pred) {
            found = true;
            break;
        }
        let grads = compute_gradient_finite_diff(model, &current_scores, ncomp);
        update(&mut current_scores, &grads, current_pred);
    }
    if !found {
        current_pred = model.predict_from_scores(&current_scores, None);
        found = converged(current_pred);
    }
    (current_scores, current_pred, found)
}

/// Generic counterfactual explanation for any FPC-based model.
///
/// For regression: uses analytical projection toward target_value.
/// For classification: uses gradient descent toward the opposite class.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `observation >= n` or the
/// model has zero components.
/// Returns [`FdarError::InvalidDimension`] if `data` columns do not match
/// the model.
/// Returns [`FdarError::ComputationFailed`] if the gradient norm is near
/// zero (regression only).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_counterfactual(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    _scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    target_value: f64,
    max_iter: usize,
    step_size: f64,
) -> Result<CounterfactualResult, FdarError> {
    let (n, m) = data.shape();
    if observation >= n {
        return Err(FdarError::InvalidParameter {
            parameter: "observation",
            message: format!("observation {observation} >= n {n}"),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "model has 0 components".into(),
        });
    }
    let scores = model.project(data);
    let original_scores: Vec<f64> = (0..ncomp).map(|k| scores[(observation, k)]).collect();
    let original_prediction = model.predict_from_scores(&original_scores, None);

    match model.task_type() {
        TaskType::Regression => {
            let grad = compute_gradient_finite_diff(model, &original_scores, ncomp);
            let grad_norm_sq: f64 = grad.iter().map(|g| g * g).sum();
            if grad_norm_sq < 1e-30 {
                return Err(FdarError::ComputationFailed {
                    operation: "generic_counterfactual",
                    detail: "gradient norm is near zero".into(),
                });
            }
            let gap = target_value - original_prediction;
            let delta_scores: Vec<f64> = grad.iter().map(|&gk| gap * gk / grad_norm_sq).collect();
            let final_scores: Vec<f64> = original_scores
                .iter()
                .zip(&delta_scores)
                .map(|(&o, &d)| o + d)
                .collect();
            Ok(build_counterfactual_result(
                model,
                observation,
                original_scores,
                final_scores,
                original_prediction,
                ncomp,
                m,
                true,
            ))
        }
        TaskType::BinaryClassification => {
            let original_class = if original_prediction >= 0.5 { 1.0 } else { 0.0 };
            let target_class = 1.0 - original_class;
            let (final_scores, _pred, found) = counterfactual_gd_search(
                model,
                &original_scores,
                max_iter,
                ncomp,
                |pred: f64| {
                    let pc: f64 = if pred >= 0.5 { 1.0 } else { 0.0 };
                    (pc - target_class).abs() < 1e-10
                },
                |scores, grads, pred| {
                    for k in 0..ncomp {
                        scores[k] -= step_size * (pred - target_class) * grads[k];
                    }
                },
            );
            Ok(build_counterfactual_result(
                model,
                observation,
                original_scores,
                final_scores,
                original_prediction,
                ncomp,
                m,
                found,
            ))
        }
        TaskType::MulticlassClassification(_) => {
            let original_class = original_prediction.round();
            let (final_scores, _pred, found) = counterfactual_gd_search(
                model,
                &original_scores,
                max_iter,
                ncomp,
                |pred| (pred.round() - original_class).abs() > 0.5,
                |scores, grads, _pred| {
                    let grad_norm: f64 = grads.iter().map(|g| g * g).sum::<f64>().sqrt().max(1e-12);
                    for k in 0..ncomp {
                        scores[k] += step_size * grads[k] / grad_norm;
                    }
                },
            );
            Ok(build_counterfactual_result(
                model,
                observation,
                original_scores,
                final_scores,
                original_prediction,
                ncomp,
                m,
                found,
            ))
        }
    }
}

// ===========================================================================
// 9. Generic LIME
// ===========================================================================

/// Generic LIME explanation for any FPC-based model.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `observation >= n`,
/// `n_samples` is zero, `kernel_width <= 0`, or the model has zero
/// components.
/// Returns [`FdarError::InvalidDimension`] if `data` columns do not match
/// the model.
/// Returns [`FdarError::ComputationFailed`] if the internal LIME
/// computation fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_lime(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    _scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    n_samples: usize,
    kernel_width: f64,
    seed: u64,
) -> Result<LimeResult, FdarError> {
    let (n, m) = data.shape();
    if observation >= n {
        return Err(FdarError::InvalidParameter {
            parameter: "observation",
            message: format!("observation {observation} >= n {n}"),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    if n_samples == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_samples",
            message: "n_samples must be > 0".into(),
        });
    }
    if kernel_width <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "kernel_width",
            message: format!("kernel_width must be > 0, got {kernel_width}"),
        });
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "model has 0 components".into(),
        });
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
    .ok_or_else(|| FdarError::ComputationFailed {
        operation: "generic_lime",
        detail: "compute_lime returned None".into(),
    })
}

// ===========================================================================
// 10. Generic Saliency
// ===========================================================================

/// Generic functional saliency maps via SHAP-weighted rotation.
///
/// Lifts FPC-level attributions to the function domain.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its
/// column count does not match the model.
/// Returns [`FdarError::InvalidParameter`] if the model has zero components
/// or `n_samples` is zero (propagated from [`generic_shap_values`]).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_saliency(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    n_samples: usize,
    seed: u64,
) -> Result<FunctionalSaliencyResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0".into(),
            actual: "0 rows".into(),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "model has 0 components".into(),
        });
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

    Ok(FunctionalSaliencyResult {
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
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its
/// column count does not match the model.
/// Returns [`FdarError::InvalidParameter`] if the model has zero components
/// or `n_samples` is zero (propagated from [`generic_shap_values`]).
/// Returns [`FdarError::ComputationFailed`] if the domain selection
/// computation fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_domain_selection(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    window_width: usize,
    threshold: f64,
    n_samples: usize,
    seed: u64,
) -> Result<DomainSelectionResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0".into(),
            actual: "0 rows".into(),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "model has 0 components".into(),
        });
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

    compute_domain_selection(&beta_t, window_width, threshold).ok_or_else(|| {
        FdarError::ComputationFailed {
            operation: "generic_domain_selection",
            detail: "compute_domain_selection returned None".into(),
        }
    })
}

// ===========================================================================
// 12. Generic Anchor
// ===========================================================================

/// Generic anchor explanation for any FPC-based model.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its
/// column count does not match the model.
/// Returns [`FdarError::InvalidParameter`] if `observation >= n` or
/// `n_bins < 2`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_anchor(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    precision_threshold: f64,
    n_bins: usize,
) -> Result<AnchorResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0".into(),
            actual: "0 rows".into(),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    if observation >= n {
        return Err(FdarError::InvalidParameter {
            parameter: "observation",
            message: format!("observation {observation} >= n {n}"),
        });
    }
    if n_bins < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_bins",
            message: format!("n_bins must be >= 2, got {n_bins}"),
        });
    }
    let ncomp = model.ncomp();
    let scores = model.project(data);
    let p_scalar = scalar_covariates.map_or(0, super::matrix::FdMatrix::ncols);

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

    Ok(AnchorResult {
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
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 rows,
/// zero columns, or `y.len() != n`.
/// Returns [`FdarError::InvalidParameter`] if `n_boot < 2`, `ncomp` is
/// zero, or `task_type` is `MulticlassClassification`.
/// Returns [`FdarError::ComputationFailed`] if not enough bootstrap refits
/// succeed.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_stability(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    n_boot: usize,
    seed: u64,
    task_type: TaskType,
) -> Result<StabilityAnalysisResult, FdarError> {
    let (n, m) = data.shape();
    if n < 4 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n >= 4".into(),
            actual: format!("{n} rows"),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "m > 0".into(),
            actual: "0 columns".into(),
        });
    }
    if n != y.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: n.to_string(),
            actual: y.len().to_string(),
        });
    }
    if n_boot < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_boot",
            message: format!("n_boot must be >= 2, got {n_boot}"),
        });
    }
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "ncomp must be > 0".into(),
        });
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
                if let Ok(refit) = fregre_lm(&boot_data, &boot_y, boot_sc.as_ref(), ncomp) {
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
                if let Ok(refit) =
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
            return Err(FdarError::InvalidParameter {
                parameter: "task_type",
                message: "stability analysis not supported for multiclass".into(),
            });
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
    .ok_or_else(|| FdarError::ComputationFailed {
        operation: "generic_stability",
        detail: "not enough successful bootstrap refits".into(),
    })
}

// ===========================================================================
// 14. Generic VIF
// ===========================================================================

/// Generic VIF for any FPC-based model (only depends on score matrix).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its
/// column count does not match the model.
/// Returns [`FdarError::ComputationFailed`] if the internal VIF computation
/// fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_vif(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
) -> Result<VifResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0".into(),
            actual: "0 rows".into(),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    let ncomp = model.ncomp();
    let scores = model.project(data);
    compute_vif_from_scores(&scores, ncomp, scalar_covariates, n).ok_or_else(|| {
        FdarError::ComputationFailed {
            operation: "generic_vif",
            detail: "compute_vif_from_scores returned None".into(),
        }
    })
}

// ===========================================================================
// 15. Generic Prototype / Criticism
// ===========================================================================

/// Generic prototype/criticism selection for any FPC-based model.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its
/// column count does not match the model.
/// Returns [`FdarError::InvalidParameter`] if the model has zero components,
/// `n_prototypes` is zero, or `n_prototypes > n`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_prototype_criticism(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    n_prototypes: usize,
    n_criticisms: usize,
) -> Result<PrototypeCriticismResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0".into(),
            actual: "0 rows".into(),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "model has 0 components".into(),
        });
    }
    if n_prototypes == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_prototypes",
            message: "n_prototypes must be > 0".into(),
        });
    }
    if n_prototypes > n {
        return Err(FdarError::InvalidParameter {
            parameter: "n_prototypes",
            message: format!("n_prototypes {n_prototypes} > n {n}"),
        });
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
    criticism_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let criticism_indices: Vec<usize> = criticism_candidates
        .iter()
        .take(n_crit)
        .map(|&(i, _)| i)
        .collect();
    let criticism_witness: Vec<f64> = criticism_indices.iter().map(|&i| witness[i]).collect();

    Ok(PrototypeCriticismResult {
        prototype_indices: selected,
        prototype_witness,
        criticism_indices,
        criticism_witness,
        bandwidth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate deterministic synthetic curves and a continuous response.
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

    /// Median-split continuous y into binary {0, 1}.
    fn make_binary_y(y: &[f64]) -> Vec<f64> {
        let mut sorted = y.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        y.iter()
            .map(|&v| if v >= median { 1.0 } else { 0.0 })
            .collect()
    }

    const N: usize = 30;
    const M: usize = 50;
    const NCOMP: usize = 3;
    const SEED: u64 = 42;

    #[test]
    fn test_generic_pdp() {
        let (data, y) = generate_test_data(N, M, SEED);
        let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
        let n_grid = 20;
        let res = generic_pdp(&fit, &data, None, 0, n_grid).unwrap();
        assert_eq!(res.grid_values.len(), n_grid);
        assert_eq!(res.pdp_curve.len(), n_grid);
        let (nr, nc) = res.ice_curves.shape();
        assert_eq!(nr, N);
        assert_eq!(nc, n_grid);
        // Out-of-range component should return Err
        assert!(generic_pdp(&fit, &data, None, NCOMP, n_grid).is_err());
    }

    #[test]
    fn test_generic_permutation_importance() {
        let (data, y) = generate_test_data(N, M, SEED);
        let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
        let res = generic_permutation_importance(&fit, &data, &y, 20, SEED).unwrap();
        assert_eq!(res.importance.len(), NCOMP);
        assert_eq!(res.permuted_metric.len(), NCOMP);
    }

    #[test]
    fn test_generic_friedman_h() {
        let (data, y) = generate_test_data(N, M, SEED);
        let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
        let res = generic_friedman_h(&fit, &data, None, 0, 1, 10).unwrap();
        assert!(res.h_squared >= 0.0);
        assert!(!res.h_squared.is_nan());
    }

    #[test]
    fn test_generic_shap_values() {
        let (data, y) = generate_test_data(N, M, SEED);
        let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
        let res = generic_shap_values(&fit, &data, None, 100, SEED).unwrap();
        let (nr, nc) = res.values.shape();
        assert_eq!(nr, N);
        assert_eq!(nc, NCOMP);
        assert!(!res.base_value.is_nan());
    }

    #[test]
    fn test_generic_ale() {
        let (data, y) = generate_test_data(N, M, SEED);
        let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
        let res = generic_ale(&fit, &data, None, 0, 10).unwrap();
        assert!(!res.bin_midpoints.is_empty());
        assert_eq!(res.ale_values.len(), res.bin_midpoints.len());
    }

    #[test]
    fn test_generic_sobol_indices() {
        let (data, y) = generate_test_data(N, M, SEED);
        let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
        let res = generic_sobol_indices(&fit, &data, None, 100, SEED).unwrap();
        assert_eq!(res.first_order.len(), NCOMP);
        assert_eq!(res.total_order.len(), NCOMP);
    }

    #[test]
    fn test_generic_conditional_perm() {
        let (data, y) = generate_test_data(N, M, SEED);
        let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
        let res =
            generic_conditional_permutation_importance(&fit, &data, &y, None, 5, 20, SEED).unwrap();
        assert_eq!(res.importance.len(), NCOMP);
    }

    #[test]
    fn test_generic_counterfactual_reg() {
        let (data, y) = generate_test_data(N, M, SEED);
        let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
        // Target = mean of y (should be reachable)
        let target = y.iter().sum::<f64>() / y.len() as f64;
        let res = generic_counterfactual(&fit, &data, None, 0, target, 200, 0.01).unwrap();
        assert!(res.found);
        assert!((res.counterfactual_prediction - target).abs() < 1.0);
    }

    #[test]
    fn test_generic_counterfactual_logistic() {
        let (data, y) = generate_test_data(N, M, SEED);
        let y_bin = make_binary_y(&y);
        let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();
        // Pick an observation predicted as class 0, try flipping to 1
        let scores = fit.project(&data);
        let obs = (0..N)
            .find(|&i| {
                let s: Vec<f64> = (0..NCOMP).map(|k| scores[(i, k)]).collect();
                fit.predict_from_scores(&s, None) < 0.5
            })
            .unwrap_or(0);
        let res = generic_counterfactual(&fit, &data, None, obs, 1.0, 500, 0.05).unwrap();
        assert_eq!(res.observation, obs);
        // Should either find a counterfactual or at least produce a result
        assert_eq!(res.delta_function.len(), M);
    }

    #[test]
    fn test_generic_lime() {
        let (data, y) = generate_test_data(N, M, SEED);
        let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
        let res = generic_lime(&fit, &data, None, 0, 100, 1.0, SEED).unwrap();
        assert_eq!(res.attributions.len(), NCOMP);
        assert_eq!(res.observation, 0);
    }

    #[test]
    fn test_generic_saliency() {
        let (data, y) = generate_test_data(N, M, SEED);
        let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
        let res = generic_saliency(&fit, &data, None, 100, SEED).unwrap();
        let (nr, nc) = res.saliency_map.shape();
        assert_eq!(nr, N);
        assert_eq!(nc, M);
        assert_eq!(res.mean_absolute_saliency.len(), M);
    }

    #[test]
    fn test_generic_domain_selection() {
        let (data, y) = generate_test_data(N, M, SEED);
        let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
        let res = generic_domain_selection(&fit, &data, None, 5, 0.5, 100, SEED).unwrap();
        assert_eq!(res.pointwise_importance.len(), M);
    }

    #[test]
    fn test_generic_anchor() {
        let (data, y) = generate_test_data(N, M, SEED);
        let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
        let res = generic_anchor(&fit, &data, None, 0, 0.9, 5).unwrap();
        assert_eq!(res.observation, 0);
    }

    #[test]
    fn test_generic_stability() {
        let (data, y) = generate_test_data(N, M, SEED);
        let res =
            generic_stability(&data, &y, None, NCOMP, 10, SEED, TaskType::Regression).unwrap();
        assert_eq!(res.beta_t_std.len(), M);
        assert_eq!(res.coefficient_std.len(), NCOMP);
        assert!(res.n_boot_success > 0);
    }

    #[test]
    fn test_generic_vif() {
        let (data, y) = generate_test_data(N, M, SEED);
        let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
        let res = generic_vif(&fit, &data, None).unwrap();
        assert_eq!(res.vif.len(), NCOMP);
        // VIF should be >= 1.0 for all components (allow small FP tolerance)
        for &v in &res.vif {
            assert!(v >= 1.0 - 1e-10, "VIF should be >= 1.0, got {}", v);
        }
    }

    #[test]
    fn test_generic_prototype_criticism() {
        let (data, y) = generate_test_data(N, M, SEED);
        let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
        let n_proto = 3;
        let n_crit = 3;
        let res = generic_prototype_criticism(&fit, &data, n_proto, n_crit).unwrap();
        assert_eq!(res.prototype_indices.len(), n_proto);
        assert_eq!(res.prototype_witness.len(), n_proto);
        assert!(res.criticism_indices.len() <= n_crit);
        assert!(res.bandwidth > 0.0);
    }
}
