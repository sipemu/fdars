//! Internal helper functions shared across explainability submodules.

mod ale_lime;
mod anchor;
mod kernel;
mod logistic;
mod permutation;
mod projection;
mod saliency;
mod shap_helpers;
pub(super) mod stability;

// ===========================================================================
// Re-exports (preserve existing pub(crate) API)
// ===========================================================================

// --- projection.rs ---
pub(crate) use projection::{
    clone_scores_matrix, compute_column_means, compute_mean_scalar, ice_to_pdp, make_grid,
    project_scores, subsample_rows,
};

// Used by sibling explain submodules (importance.rs, sensitivity.rs)
pub(crate) use projection::compute_score_variance;

// --- shap_helpers.rs ---
pub(crate) use shap_helpers::{
    accumulate_kernel_shap_sample, build_coalition_scores, compute_h_squared, get_obs_scalar,
    sample_random_coalition, shapley_kernel_weight, solve_kernel_shap_obs,
};

// --- permutation.rs ---
pub(crate) use permutation::{
    compute_conditioning_bins, compute_sobol_component, generate_sobol_matrices, permute_component,
    shuffle_global,
};

// --- kernel.rs ---
pub(crate) use kernel::{
    compute_kernel_mean, compute_witness, gaussian_kernel_matrix, greedy_prototype_selection,
    median_bandwidth,
};

// Used by sibling explain submodules (advanced.rs)
pub(crate) use kernel::{beta_depth_from_bootstrap, compute_score_depths};

// --- saliency.rs ---
pub(crate) use saliency::{
    compute_domain_selection, compute_saliency_map, mean_absolute_column,
    reconstruct_delta_function,
};

// --- ale_lime.rs ---
pub(crate) use ale_lime::{compute_ale, compute_lime};

// --- anchor.rs ---
pub(crate) use anchor::anchor_beam_search;

// --- stability.rs ---
pub(crate) use stability::build_stability_result;

// --- logistic.rs ---
pub(crate) use logistic::{
    calibration_gap_weighted, conformal_quantile_and_coverage, logistic_accuracy_from_scores,
    logistic_eta_base, logistic_pdp_mean, predict_from_scores, validate_conformal_inputs,
};
