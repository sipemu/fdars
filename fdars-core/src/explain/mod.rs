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
    accumulate_kernel_shap_sample, anchor_beam_search, build_coalition_scores,
    build_stability_result, clone_scores_matrix, compute_ale, compute_column_means,
    compute_conditioning_bins, compute_domain_selection, compute_h_squared, compute_kernel_mean,
    compute_lime, compute_mean_scalar, compute_saliency_map, compute_sobol_component,
    compute_witness, gaussian_kernel_matrix, generate_sobol_matrices, get_obs_scalar,
    greedy_prototype_selection, ice_to_pdp, make_grid, mean_absolute_column, median_bandwidth,
    permute_component, project_scores, reconstruct_delta_function, sample_random_coalition,
    shapley_kernel_weight, solve_kernel_shap_obs, subsample_rows,
};

// pub(crate) re-export from diagnostics (used by explain_generic.rs)
pub(crate) use diagnostics::compute_vif_from_scores;

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar_on_function::{fregre_lm, functional_logistic};
    use std::f64::consts::PI;

    fn generate_test_data(n: usize, m: usize, seed: u64) -> (crate::matrix::FdMatrix, Vec<f64>) {
        use crate::matrix::FdMatrix;
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
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
        assert!(functional_pdp(&fit, &data, None, 3, 10).is_none());
        assert!(functional_pdp(&fit, &data, None, 0, 1).is_none());
    }

    #[test]
    fn test_functional_pdp_column_mismatch() {
        use crate::matrix::FdMatrix;
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let wrong_data = FdMatrix::zeros(30, 40);
        assert!(functional_pdp(&fit, &wrong_data, None, 0, 10).is_none());
    }

    #[test]
    fn test_functional_pdp_zero_rows() {
        use crate::matrix::FdMatrix;
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let empty_data = FdMatrix::zeros(0, 50);
        assert!(functional_pdp(&fit, &empty_data, None, 0, 10).is_none());
    }

    #[test]
    fn test_functional_pdp_logistic_column_mismatch() {
        use crate::matrix::FdMatrix;
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
        use crate::matrix::FdMatrix;
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

    // Beta decomposition tests

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
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

    // Significant regions tests

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

    // FPC permutation importance tests

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
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

    // Influence diagnostics tests

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
        assert_eq!(diag.p, 4);
    }

    #[test]
    fn test_influence_column_mismatch_returns_none() {
        use crate::matrix::FdMatrix;
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let wrong_data = FdMatrix::zeros(30, 40);
        assert!(influence_diagnostics(&fit, &wrong_data, None).is_none());
    }

    // Friedman H-statistic tests

    #[test]
    fn test_h_statistic_linear_zero() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let h = friedman_h_statistic(&fit, &data, 0, 1, 10).unwrap();
        assert!(
            h.h_squared.abs() < 1e-6,
            "H^2 should be ~0 for linear model: {}",
            h.h_squared
        );
    }

    #[test]
    fn test_h_statistic_logistic_positive() {
        let (data, y_cont) = generate_test_data(40, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let h = friedman_h_statistic_logistic(&fit, &data, None, 0, 1, 10).unwrap();
        assert!(
            h.h_squared >= 0.0,
            "H^2 should be nonneg for logistic: {}",
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
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
            "H^2 should be in [0,1]: {}",
            h.h_squared
        );
    }

    #[test]
    fn test_h_statistic_same_component_none() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        assert!(friedman_h_statistic(&fit, &data, 1, 1, 10).is_none());
    }

    // Pointwise importance tests

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
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

    // VIF tests

    #[test]
    fn test_vif_orthogonal_fpcs_near_one() {
        let (data, y) = generate_test_data(50, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let vif = fpc_vif(&fit, &data, None).unwrap();
        for (k, &v) in vif.vif.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 0.5,
                "Orthogonal FPC VIF should be ~1 at k={}: {}",
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
            assert!(v >= 1.0 - 1e-6, "VIF should be >= 1 at k={}: {}", k, v);
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
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

    // SHAP tests

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
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let shap = fpc_shap_values_logistic(&fit, &data, None, 500, 42).unwrap();
        let mut shap_sums = Vec::new();
        for i in 0..30 {
            let sum: f64 = (0..3).map(|k| shap.values[(i, k)]).sum::<f64>() + shap.base_value;
            shap_sums.push(sum);
        }
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
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
        use crate::matrix::FdMatrix;
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let empty = FdMatrix::zeros(0, 50);
        assert!(fpc_shap_values(&fit, &empty, None).is_none());
    }

    // DFBETAS / DFFITS tests

    #[test]
    fn test_dfbetas_shape() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let db = dfbetas_dffits(&fit, &data, None).unwrap();
        assert_eq!(db.dfbetas.shape(), (30, 4));
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
            "DFBETAS cutoff should be 2/sqrt(n)"
        );
        assert!(
            (db.dffits_cutoff - 2.0 * (4.0 / 30.0_f64).sqrt()).abs() < 1e-10,
            "DFFITS cutoff should be 2*sqrt(p/n)"
        );
    }

    #[test]
    fn test_dfbetas_underdetermined_returns_none() {
        let (data, y) = generate_test_data(3, 50, 42);
        let fit = fregre_lm(&data, &y, None, 2).unwrap();
        assert!(dfbetas_dffits(&fit, &data, None).is_none());
    }

    #[test]
    fn test_dffits_consistency_with_cooks() {
        let (data, y) = generate_test_data(40, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let db = dfbetas_dffits(&fit, &data, None).unwrap();
        let infl = influence_diagnostics(&fit, &data, None).unwrap();
        let mut dffits_order: Vec<usize> = (0..40).collect();
        dffits_order.sort_by(|&a, &b| {
            db.dffits[b]
                .abs()
                .partial_cmp(&db.dffits[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cooks_order: Vec<usize> = (0..40).collect();
        cooks_order.sort_by(|&a, &b| {
            infl.cooks_distance[b]
                .partial_cmp(&infl.cooks_distance[a])
                .unwrap()
        });
        assert_eq!(
            dffits_order[0], cooks_order[0],
            "Top influential obs should agree between DFFITS and Cook's D"
        );
    }

    // Prediction interval tests

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

    // ALE tests

    #[test]
    fn test_ale_linear_is_linear() {
        let (data, y) = generate_test_data(50, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let ale = fpc_ale(&fit, &data, None, 0, 10).unwrap();
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
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let ale = fpc_ale_logistic(&fit, &data, None, 0, 10).unwrap();
        for &v in &ale.ale_values {
            assert!(v.abs() < 2.0, "Logistic ALE should be bounded: {}", v);
        }
    }

    #[test]
    fn test_ale_invalid_returns_none() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        assert!(fpc_ale(&fit, &data, None, 5, 10).is_none());
        assert!(fpc_ale(&fit, &data, None, 0, 0).is_none());
    }

    // LOO-CV / PRESS tests

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
            "LOO R^2 ({}) should be <= training R^2 ({})",
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

    // Sobol tests

    #[test]
    fn test_sobol_linear_nonnegative() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let sobol = sobol_indices(&fit, &data, &y, None).unwrap();
        for (k, &s) in sobol.first_order.iter().enumerate() {
            assert!(s >= -1e-10, "S_{} should be >= 0: {}", k, s);
        }
    }

    #[test]
    fn test_sobol_linear_sum_approx_r2() {
        let (data, y) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let sobol = sobol_indices(&fit, &data, &y, None).unwrap();
        let sum_s: f64 = sobol.first_order.iter().sum();
        assert!(
            (sum_s - fit.r_squared).abs() < 0.2,
            "Sum S_k ({}) should be close to R^2 ({})",
            sum_s,
            fit.r_squared
        );
    }

    #[test]
    fn test_sobol_logistic_bounded() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_bin = {
            let mut s = y_cont.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

    // Calibration tests

    #[test]
    fn test_calibration_brier_range() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_bin = {
            let mut s = y_cont.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

    // Saliency tests

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
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
                    "|s| should be <= 0.25 * |beta(t)| at ({},{})",
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
            assert!(v >= 0.0, "Mean absolute saliency should be >= 0: {}", v);
        }
    }

    // Domain selection tests

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
        let ds = domain_selection(&fit, 50, 0.01).unwrap();
        assert!(
            ds.intervals.len() <= 1,
            "Full window should give <= 1 interval: {}",
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
            "Higher threshold should give <= intervals: {} vs {}",
            ds_high.intervals.len(),
            ds_low.intervals.len()
        );
    }

    // Conditional permutation importance tests

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
                "Importance should be approx >= 0 for FPC {}: {}",
                k,
                cp.importance[k]
            );
        }
    }

    // Counterfactual tests

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
            "Distance should be |gap|/||gamma||: {} vs {}",
            cf.distance,
            expected_dist
        );
    }

    #[test]
    fn test_counterfactual_logistic_flips_class() {
        let (data, y_cont) = generate_test_data(30, 50, 42);
        let y_bin = {
            let mut s = y_cont.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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

    // Prototype/criticism tests

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

    // LIME tests

    #[test]
    fn test_lime_linear_matches_global() {
        let (data, y) = generate_test_data(40, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let lime = lime_explanation(&fit, &data, None, 0, 5000, 1.0, 42).unwrap();
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
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
            "R^2 should be in [0,1]: {}",
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

    // ECE tests

    fn make_logistic_fit() -> (
        crate::matrix::FdMatrix,
        Vec<f64>,
        crate::scalar_on_function::FunctionalLogisticResult,
    ) {
        let (data, y_cont) = generate_test_data(40, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
            "ECE should <= MCE: {} vs {}",
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

    // Conformal prediction tests

    #[test]
    fn test_conformal_coverage_near_target() {
        let (data, y) = generate_test_data(60, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let cp = conformal_prediction_residuals(&fit, &data, &y, &data, None, None, 0.3, 0.1, 42)
            .unwrap();
        assert!(
            cp.coverage >= 0.8,
            "Coverage {} should be >= 0.8 for alpha=0.1",
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
            "Quantile should be >= 0: {}",
            cp.residual_quantile
        );
    }

    #[test]
    fn test_conformal_lengths_match() {
        use crate::matrix::FdMatrix;
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

    // Regression depth tests

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
            "Beta depth should be >= 0: {}",
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

    // Stability tests

    #[test]
    fn test_stability_beta_std_nonneg() {
        let (data, y) = generate_test_data(30, 50, 42);
        let sa = explanation_stability(&data, &y, None, 3, 20, 42).unwrap();
        for (j, &s) in sa.beta_t_std.iter().enumerate() {
            assert!(s >= 0.0, "Std should be >= 0 at {}: {}", j, s);
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
        assert!(sa2.n_boot_success >= sa1.n_boot_success);
    }

    // Anchor tests

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
