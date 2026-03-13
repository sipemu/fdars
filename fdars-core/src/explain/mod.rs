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

// Submodules
pub(crate) mod helpers;

mod advanced;
mod ale_lime;
mod counterfactual;
mod diagnostics;
mod importance;
mod pdp;
mod sensitivity;
mod shap;

// ===========================================================================
// Public re-exports (backward compatible)
// ===========================================================================

// --- pdp.rs ---
pub use pdp::{
    beta_decomposition, beta_decomposition_logistic, functional_pdp, functional_pdp_logistic,
    significant_regions, significant_regions_from_se, BetaDecomposition, FunctionalPdpResult,
    SignificanceDirection, SignificantRegion,
};

// --- importance.rs ---
pub use importance::{
    conditional_permutation_importance, conditional_permutation_importance_logistic,
    fpc_permutation_importance, fpc_permutation_importance_logistic, pointwise_importance,
    pointwise_importance_logistic, ConditionalPermutationImportanceResult,
    FpcPermutationImportance, PointwiseImportanceResult,
};

// --- diagnostics.rs ---
pub use diagnostics::{
    dfbetas_dffits, fpc_vif, fpc_vif_logistic, influence_diagnostics, loo_cv_press,
    prediction_intervals, DfbetasDffitsResult, InfluenceDiagnostics, LooCvResult,
    PredictionIntervalResult, VifResult,
};

// --- shap.rs ---
pub use shap::{
    fpc_shap_values, fpc_shap_values_logistic, friedman_h_statistic, friedman_h_statistic_logistic,
    FpcShapValues, FriedmanHResult,
};

// --- ale_lime.rs ---
pub use ale_lime::{
    fpc_ale, fpc_ale_logistic, lime_explanation, lime_explanation_logistic, AleResult, LimeResult,
};

// --- sensitivity.rs ---
pub use sensitivity::{
    domain_selection, domain_selection_logistic, functional_saliency, functional_saliency_logistic,
    sobol_indices, sobol_indices_logistic, DomainSelectionResult, FunctionalSaliencyResult,
    ImportantInterval, SobolIndicesResult,
};

// --- counterfactual.rs ---
pub use counterfactual::{
    counterfactual_logistic, counterfactual_regression, prototype_criticism, CounterfactualResult,
    PrototypeCriticismResult,
};

// --- advanced.rs ---
pub use advanced::{
    anchor_explanation, anchor_explanation_logistic, calibration_diagnostics,
    conformal_prediction_residuals, expected_calibration_error, explanation_stability,
    explanation_stability_logistic, regression_depth, regression_depth_logistic, AnchorCondition,
    AnchorResult, AnchorRule, CalibrationDiagnosticsResult, ConformalPredictionResult, DepthType,
    EceResult, RegressionDepthResult, StabilityAnalysisResult,
};

// ===========================================================================
// pub(crate) re-exports from helpers (for explain_generic.rs, conformal.rs)
// ===========================================================================

pub(crate) use helpers::{
    accumulate_kernel_shap_sample, anchor_beam_search, build_stability_result, clone_scores_matrix,
    compute_ale, compute_column_means, compute_conditioning_bins, compute_domain_selection,
    compute_h_squared, compute_kernel_mean, compute_lime, compute_mean_scalar,
    compute_saliency_map, compute_sobol_component, compute_witness, gaussian_kernel_matrix,
    generate_sobol_matrices, get_obs_scalar, greedy_prototype_selection, ice_to_pdp, make_grid,
    mean_absolute_column, median_bandwidth, permute_component, project_scores,
    reconstruct_delta_function, sample_random_coalition, shapley_kernel_weight,
    solve_kernel_shap_obs, subsample_rows,
};

// pub(crate) re-export from diagnostics (used by explain_generic.rs)
pub(crate) use diagnostics::compute_vif_from_scores;

#[cfg(test)]
mod tests;
