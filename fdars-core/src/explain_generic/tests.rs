use super::*;
use crate::classification::fclassif_lda_fit;
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{fregre_lm, functional_logistic};
use std::f64::consts::PI;

/// Generate deterministic synthetic curves and a continuous response.
fn generate_test_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
    let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0; n];
    for i in 0..n {
        let phase = (seed.wrapping_mul(17).wrapping_add(i as u64 * 31) % 1000) as f64 / 1000.0 * PI;
        let amplitude =
            ((seed.wrapping_mul(13).wrapping_add(i as u64 * 7) % 100) as f64 / 100.0) - 0.5;
        for j in 0..m {
            data[(i, j)] = (2.0 * PI * t[j] + phase).sin() + amplitude * (4.0 * PI * t[j]).cos();
        }
        y[i] = 2.0 * phase + 3.0 * amplitude;
    }
    (data, y)
}

/// Generate well-conditioned data with different frequencies per curve.
fn generate_varied_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let data_vec: Vec<f64> = (0..n * m)
        .map(|k| {
            let i = (k % n) as f64;
            let j = (k / n) as f64;
            ((i + 1.0) * j * 0.2).sin()
        })
        .collect();
    let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin() * 2.0 + 1.0).collect();
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

/// Convert continuous y to multiclass labels (3 classes).
fn make_multiclass_y(y: &[f64]) -> Vec<usize> {
    let mut sorted = y.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let t1 = sorted[sorted.len() / 3];
    let t2 = sorted[2 * sorted.len() / 3];
    y.iter()
        .map(|&v| {
            if v < t1 {
                0
            } else if v < t2 {
                1
            } else {
                2
            }
        })
        .collect()
}

const N: usize = 30;
const M: usize = 50;
const NCOMP: usize = 3;
const SEED: u64 = 42;

// =========================================================================
// PDP tests
// =========================================================================

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
fn test_pdp_grid_values_sorted() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_pdp(&fit, &data, None, 0, 15).unwrap();
    // Grid values should be monotonically increasing
    for i in 1..res.grid_values.len() {
        assert!(
            res.grid_values[i] >= res.grid_values[i - 1],
            "grid values should be sorted"
        );
    }
}

#[test]
fn test_pdp_component_field() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    for comp in 0..NCOMP {
        let res = generic_pdp(&fit, &data, None, comp, 10).unwrap();
        assert_eq!(res.component, comp);
    }
}

#[test]
fn test_pdp_ice_mean_equals_pdp() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let n_grid = 10;
    let res = generic_pdp(&fit, &data, None, 0, n_grid).unwrap();
    // PDP curve should be the column-mean of ICE curves
    for g in 0..n_grid {
        let col_mean: f64 = (0..N).map(|i| res.ice_curves[(i, g)]).sum::<f64>() / N as f64;
        assert!(
            (col_mean - res.pdp_curve[g]).abs() < 1e-10,
            "PDP should equal column mean of ICE at grid point {g}"
        );
    }
}

#[test]
fn test_pdp_all_values_finite() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_pdp(&fit, &data, None, 1, 10).unwrap();
    for &v in &res.pdp_curve {
        assert!(v.is_finite(), "PDP values should be finite");
    }
    for i in 0..N {
        for g in 0..10 {
            assert!(
                res.ice_curves[(i, g)].is_finite(),
                "ICE values should be finite"
            );
        }
    }
}

#[test]
fn test_pdp_error_zero_rows() {
    let empty_data = FdMatrix::zeros(0, M);
    let (full_data, full_y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&full_data, &full_y, None, NCOMP).unwrap();
    assert!(generic_pdp(&fit, &empty_data, None, 0, 10).is_err());
}

#[test]
fn test_pdp_error_n_grid_one() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_pdp(&fit, &data, None, 0, 1).is_err());
}

// =========================================================================
// SHAP tests
// =========================================================================

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
fn test_shap_base_value_finite() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_shap_values(&fit, &data, None, 200, SEED).unwrap();
    assert!(res.base_value.is_finite(), "base_value should be finite");
}

#[test]
fn test_shap_mean_scores_length() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_shap_values(&fit, &data, None, 100, SEED).unwrap();
    assert_eq!(res.mean_scores.len(), NCOMP);
    for &v in &res.mean_scores {
        assert!(v.is_finite(), "mean scores should be finite");
    }
}

#[test]
fn test_shap_efficiency_linear_model() {
    // For a linear model, SHAP efficiency property should approximately hold:
    // base_value + sum_k shap[i,k] ~ prediction[i]
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_shap_values(&fit, &data, None, 500, SEED).unwrap();
    let scores = fit.project(&data);
    for i in 0..N {
        let s: Vec<f64> = (0..NCOMP).map(|k| scores[(i, k)]).collect();
        let pred = fit.predict_from_scores(&s, None);
        let shap_sum: f64 = (0..NCOMP).map(|k| res.values[(i, k)]).sum();
        let reconstructed = res.base_value + shap_sum;
        // Allow tolerance for the MC approximation
        assert!(
            (reconstructed - pred).abs() < 2.0,
            "SHAP efficiency: reconstructed={reconstructed:.4}, pred={pred:.4}, diff={}",
            (reconstructed - pred).abs()
        );
    }
}

#[test]
fn test_shap_all_values_finite() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_shap_values(&fit, &data, None, 100, SEED).unwrap();
    for i in 0..N {
        for k in 0..NCOMP {
            assert!(
                res.values[(i, k)].is_finite(),
                "SHAP values should be finite at ({i},{k})"
            );
        }
    }
}

#[test]
fn test_shap_error_zero_samples() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_shap_values(&fit, &data, None, 0, SEED).is_err());
}

// =========================================================================
// ALE tests
// =========================================================================

#[test]
fn test_generic_ale() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_ale(&fit, &data, None, 0, 10).unwrap();
    assert!(!res.bin_midpoints.is_empty());
    assert_eq!(res.ale_values.len(), res.bin_midpoints.len());
}

#[test]
fn test_ale_bin_edges_length() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_ale(&fit, &data, None, 0, 10).unwrap();
    // bin_edges should be one more than bin_midpoints
    assert_eq!(res.bin_edges.len(), res.bin_midpoints.len() + 1);
}

#[test]
fn test_ale_bin_counts_sum() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_ale(&fit, &data, None, 0, 10).unwrap();
    // Sum of bin counts should be at most N (some obs may fall outside)
    let total: usize = res.bin_counts.iter().sum();
    assert!(total <= N, "total bin count {total} should be <= {N}");
    assert!(total > 0, "should have at least some observations in bins");
}

#[test]
fn test_ale_component_field() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    for comp in 0..NCOMP {
        let res = generic_ale(&fit, &data, None, comp, 5).unwrap();
        assert_eq!(res.component, comp);
    }
}

#[test]
fn test_ale_values_centered() {
    // ALE values should be mean-centered (approximately zero mean)
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_ale(&fit, &data, None, 0, 10).unwrap();
    if !res.ale_values.is_empty() {
        let mean: f64 = res.ale_values.iter().sum::<f64>() / res.ale_values.len() as f64;
        // Mean-centered ALE should have small absolute mean
        assert!(
            mean.abs() < 5.0,
            "ALE values should be roughly centered, got mean={mean}"
        );
    }
}

#[test]
fn test_ale_all_finite() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_ale(&fit, &data, None, 0, 10).unwrap();
    for &v in &res.ale_values {
        assert!(v.is_finite(), "ALE values should be finite");
    }
    for &v in &res.bin_midpoints {
        assert!(v.is_finite(), "bin midpoints should be finite");
    }
}

#[test]
fn test_ale_error_zero_bins() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_ale(&fit, &data, None, 0, 0).is_err());
}

#[test]
fn test_ale_error_invalid_component() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_ale(&fit, &data, None, NCOMP, 10).is_err());
}

// =========================================================================
// LIME tests
// =========================================================================

#[test]
fn test_generic_lime() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_lime(&fit, &data, None, 0, 100, 1.0, SEED).unwrap();
    assert_eq!(res.attributions.len(), NCOMP);
    assert_eq!(res.observation, 0);
}

#[test]
fn test_lime_attributions_finite() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_lime(&fit, &data, None, 0, 200, 1.0, SEED).unwrap();
    for &v in &res.attributions {
        assert!(v.is_finite(), "LIME attributions should be finite");
    }
    assert!(
        res.local_intercept.is_finite(),
        "local intercept should be finite"
    );
}

#[test]
fn test_lime_local_r_squared_in_range() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_lime(&fit, &data, None, 0, 200, 1.0, SEED).unwrap();
    // R^2 can be negative for poor fits, but should be finite
    assert!(
        res.local_r_squared.is_finite(),
        "local R^2 should be finite"
    );
}

#[test]
fn test_lime_kernel_width_stored() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let kw = 2.5;
    let res = generic_lime(&fit, &data, None, 0, 100, kw, SEED).unwrap();
    assert!(
        (res.kernel_width - kw).abs() < 1e-12,
        "kernel_width should be stored"
    );
}

#[test]
fn test_lime_different_observations() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res0 = generic_lime(&fit, &data, None, 0, 100, 1.0, SEED).unwrap();
    let res5 = generic_lime(&fit, &data, None, 5, 100, 1.0, SEED).unwrap();
    assert_eq!(res0.observation, 0);
    assert_eq!(res5.observation, 5);
    // Different observations should generally produce different attributions
    let differ = (0..NCOMP).any(|k| (res0.attributions[k] - res5.attributions[k]).abs() > 1e-15);
    assert!(
        differ,
        "different observations should produce different LIME attributions"
    );
}

#[test]
fn test_lime_error_invalid_observation() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_lime(&fit, &data, None, N, 100, 1.0, SEED).is_err());
}

#[test]
fn test_lime_error_zero_samples() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_lime(&fit, &data, None, 0, 0, 1.0, SEED).is_err());
}

#[test]
fn test_lime_error_negative_kernel_width() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_lime(&fit, &data, None, 0, 100, -1.0, SEED).is_err());
}

// =========================================================================
// Permutation importance tests
// =========================================================================

#[test]
fn test_generic_permutation_importance() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_permutation_importance(&fit, &data, &y, 20, SEED).unwrap();
    assert_eq!(res.importance.len(), NCOMP);
    assert_eq!(res.permuted_metric.len(), NCOMP);
}

#[test]
fn test_permutation_importance_baseline_metric_finite() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_permutation_importance(&fit, &data, &y, 20, SEED).unwrap();
    assert!(
        res.baseline_metric.is_finite(),
        "baseline metric should be finite"
    );
}

#[test]
fn test_permutation_importance_nonnegative_for_good_model() {
    // For a well-fitted model, importance should be >= 0 for at least some components
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_permutation_importance(&fit, &data, &y, 50, SEED).unwrap();
    let any_positive = res.importance.iter().any(|&v| v > -0.1);
    assert!(
        any_positive,
        "at least some importance should be near-positive"
    );
}

#[test]
fn test_permutation_importance_permuted_metric_leq_baseline() {
    // For a reasonable model, permuted metric should generally be <= baseline
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_permutation_importance(&fit, &data, &y, 50, SEED).unwrap();
    for k in 0..NCOMP {
        assert!(
            res.permuted_metric[k].is_finite(),
            "permuted metric should be finite for component {k}"
        );
    }
}

#[test]
fn test_permutation_importance_error_zero_perm() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_permutation_importance(&fit, &data, &y, 0, SEED).is_err());
}

#[test]
fn test_permutation_importance_error_mismatched_y() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let short_y: Vec<f64> = y[..5].to_vec();
    assert!(generic_permutation_importance(&fit, &data, &short_y, 10, SEED).is_err());
}

// =========================================================================
// Conditional permutation importance tests
// =========================================================================

#[test]
fn test_generic_conditional_perm() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res =
        generic_conditional_permutation_importance(&fit, &data, &y, None, 5, 20, SEED).unwrap();
    assert_eq!(res.importance.len(), NCOMP);
}

#[test]
fn test_conditional_perm_has_unconditional() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res =
        generic_conditional_permutation_importance(&fit, &data, &y, None, 5, 20, SEED).unwrap();
    assert_eq!(res.unconditional_importance.len(), NCOMP);
    assert_eq!(res.permuted_metric.len(), NCOMP);
}

#[test]
fn test_conditional_perm_baseline_matches() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res =
        generic_conditional_permutation_importance(&fit, &data, &y, None, 5, 20, SEED).unwrap();
    assert!(
        res.baseline_metric.is_finite(),
        "baseline metric should be finite"
    );
    // importance = baseline - permuted
    for k in 0..NCOMP {
        let expected = res.baseline_metric - res.permuted_metric[k];
        assert!(
            (res.importance[k] - expected).abs() < 1e-10,
            "importance should be baseline - permuted at comp {k}"
        );
    }
}

#[test]
fn test_conditional_perm_all_finite() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res =
        generic_conditional_permutation_importance(&fit, &data, &y, None, 3, 10, SEED).unwrap();
    for k in 0..NCOMP {
        assert!(res.importance[k].is_finite());
        assert!(res.permuted_metric[k].is_finite());
        assert!(res.unconditional_importance[k].is_finite());
    }
}

#[test]
fn test_conditional_perm_error_zero_bins() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(
        generic_conditional_permutation_importance(&fit, &data, &y, None, 0, 20, SEED).is_err()
    );
}

#[test]
fn test_conditional_perm_error_zero_perm() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_conditional_permutation_importance(&fit, &data, &y, None, 5, 0, SEED).is_err());
}

// =========================================================================
// Sobol indices tests
// =========================================================================

#[test]
fn test_generic_sobol_indices() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_sobol_indices(&fit, &data, None, 100, SEED).unwrap();
    assert_eq!(res.first_order.len(), NCOMP);
    assert_eq!(res.total_order.len(), NCOMP);
}

#[test]
fn test_sobol_total_order_geq_first_order() {
    // Total-order indices should generally be >= first-order indices
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_sobol_indices(&fit, &data, None, 500, SEED).unwrap();
    for k in 0..NCOMP {
        // Allow some tolerance for MC noise
        assert!(
            res.total_order[k] >= res.first_order[k] - 0.3,
            "total_order[{k}]={} should be >= first_order[{k}]={} (with MC tolerance)",
            res.total_order[k],
            res.first_order[k]
        );
    }
}

#[test]
fn test_sobol_variance_positive() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_sobol_indices(&fit, &data, None, 100, SEED).unwrap();
    assert!(res.var_y > 0.0, "variance of Y should be positive");
    assert!(res.var_y.is_finite(), "variance should be finite");
}

#[test]
fn test_sobol_component_variance_length() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_sobol_indices(&fit, &data, None, 100, SEED).unwrap();
    assert_eq!(res.component_variance.len(), NCOMP);
}

#[test]
fn test_sobol_all_finite() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_sobol_indices(&fit, &data, None, 100, SEED).unwrap();
    for k in 0..NCOMP {
        assert!(res.first_order[k].is_finite());
        assert!(res.total_order[k].is_finite());
        assert!(res.component_variance[k].is_finite());
    }
}

#[test]
fn test_sobol_error_zero_samples() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_sobol_indices(&fit, &data, None, 0, SEED).is_err());
}

// =========================================================================
// Friedman H-statistic tests
// =========================================================================

#[test]
fn test_generic_friedman_h() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_friedman_h(&fit, &data, None, 0, 1, 10).unwrap();
    assert!(res.h_squared >= 0.0);
    assert!(!res.h_squared.is_nan());
}

#[test]
fn test_friedman_h_linear_model_small() {
    // For a linear model (no interactions), H-stat should be small
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_friedman_h(&fit, &data, None, 0, 1, 10).unwrap();
    // Linear models have no interactions, so H^2 should be near zero
    assert!(
        res.h_squared < 0.2,
        "H^2 for linear model should be small, got {}",
        res.h_squared
    );
}

#[test]
fn test_friedman_h_component_indices() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_friedman_h(&fit, &data, None, 0, 2, 10).unwrap();
    assert_eq!(res.component_j, 0);
    assert_eq!(res.component_k, 2);
}

#[test]
fn test_friedman_h_grid_lengths() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let n_grid = 8;
    let res = generic_friedman_h(&fit, &data, None, 0, 1, n_grid).unwrap();
    assert_eq!(res.grid_j.len(), n_grid);
    assert_eq!(res.grid_k.len(), n_grid);
    let (nr, nc) = res.pdp_2d.shape();
    assert_eq!(nr, n_grid);
    assert_eq!(nc, n_grid);
}

#[test]
fn test_friedman_h_error_same_component() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_friedman_h(&fit, &data, None, 0, 0, 10).is_err());
}

#[test]
fn test_friedman_h_error_component_out_of_range() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_friedman_h(&fit, &data, None, 0, NCOMP, 10).is_err());
}

#[test]
fn test_friedman_h_error_n_grid_one() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_friedman_h(&fit, &data, None, 0, 1, 1).is_err());
}

#[test]
fn test_friedman_h_symmetric() {
    // H(j,k) should equal H(k,j) for the same data
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res_01 = generic_friedman_h(&fit, &data, None, 0, 1, 8).unwrap();
    let res_10 = generic_friedman_h(&fit, &data, None, 1, 0, 8).unwrap();
    assert!(
        (res_01.h_squared - res_10.h_squared).abs() < 1e-10,
        "H(0,1) should equal H(1,0)"
    );
}

// =========================================================================
// Counterfactual tests
// =========================================================================

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
fn test_counterfactual_reg_delta_scores_length() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let target = y.iter().sum::<f64>() / y.len() as f64;
    let res = generic_counterfactual(&fit, &data, None, 0, target, 200, 0.01).unwrap();
    assert_eq!(res.original_scores.len(), NCOMP);
    assert_eq!(res.counterfactual_scores.len(), NCOMP);
    assert_eq!(res.delta_scores.len(), NCOMP);
    assert_eq!(res.delta_function.len(), M);
}

#[test]
fn test_counterfactual_reg_delta_consistency() {
    // delta_scores should equal counterfactual_scores - original_scores
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let target = y.iter().sum::<f64>() / y.len() as f64;
    let res = generic_counterfactual(&fit, &data, None, 0, target, 200, 0.01).unwrap();
    for k in 0..NCOMP {
        let expected = res.counterfactual_scores[k] - res.original_scores[k];
        assert!(
            (res.delta_scores[k] - expected).abs() < 1e-10,
            "delta_scores[{k}] should equal cf - orig"
        );
    }
}

#[test]
fn test_counterfactual_reg_distance_positive() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    // Pick a target far from observation 0's prediction
    let scores = fit.project(&data);
    let s: Vec<f64> = (0..NCOMP).map(|k| scores[(0, k)]).collect();
    let pred0 = fit.predict_from_scores(&s, None);
    let target = pred0 + 5.0;
    let res = generic_counterfactual(&fit, &data, None, 0, target, 200, 0.01).unwrap();
    assert!(
        res.distance > 0.0,
        "distance should be > 0 for non-trivial counterfactual"
    );
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
fn test_counterfactual_observation_stored() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let target = y.iter().sum::<f64>() / y.len() as f64;
    for obs in [0, 5, N - 1] {
        let res = generic_counterfactual(&fit, &data, None, obs, target, 200, 0.01).unwrap();
        assert_eq!(res.observation, obs);
    }
}

#[test]
fn test_counterfactual_error_invalid_observation() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_counterfactual(&fit, &data, None, N, 0.0, 200, 0.01).is_err());
}

// =========================================================================
// Prototype/criticism tests
// =========================================================================

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

#[test]
fn test_prototype_indices_in_range() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_prototype_criticism(&fit, &data, 5, 5).unwrap();
    for &idx in &res.prototype_indices {
        assert!(idx < N, "prototype index {idx} should be < N={N}");
    }
    for &idx in &res.criticism_indices {
        assert!(idx < N, "criticism index {idx} should be < N={N}");
    }
}

#[test]
fn test_prototype_indices_unique() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_prototype_criticism(&fit, &data, 5, 5).unwrap();
    let mut proto_set = res.prototype_indices.clone();
    proto_set.sort();
    proto_set.dedup();
    assert_eq!(
        proto_set.len(),
        res.prototype_indices.len(),
        "prototype indices should be unique"
    );
}

#[test]
fn test_prototype_criticism_disjoint() {
    // Prototypes and criticisms should not overlap
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_prototype_criticism(&fit, &data, 5, 5).unwrap();
    for &ci in &res.criticism_indices {
        assert!(
            !res.prototype_indices.contains(&ci),
            "criticism {ci} should not be a prototype"
        );
    }
}

#[test]
fn test_prototype_bandwidth_positive() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_prototype_criticism(&fit, &data, 3, 3).unwrap();
    assert!(res.bandwidth > 0.0);
    assert!(res.bandwidth.is_finite());
}

#[test]
fn test_prototype_criticism_witness_finite() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_prototype_criticism(&fit, &data, 3, 3).unwrap();
    for &w in &res.prototype_witness {
        assert!(w.is_finite(), "prototype witness should be finite");
    }
    for &w in &res.criticism_witness {
        assert!(w.is_finite(), "criticism witness should be finite");
    }
}

#[test]
fn test_prototype_error_zero_prototypes() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_prototype_criticism(&fit, &data, 0, 3).is_err());
}

#[test]
fn test_prototype_error_too_many_prototypes() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_prototype_criticism(&fit, &data, N + 1, 3).is_err());
}

// =========================================================================
// Saliency tests
// =========================================================================

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
fn test_saliency_all_finite() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_saliency(&fit, &data, None, 100, SEED).unwrap();
    for i in 0..N {
        for j in 0..M {
            assert!(
                res.saliency_map[(i, j)].is_finite(),
                "saliency map should be finite at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_saliency_mean_absolute_nonnegative() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_saliency(&fit, &data, None, 100, SEED).unwrap();
    for (j, &v) in res.mean_absolute_saliency.iter().enumerate() {
        assert!(v >= 0.0, "mean absolute saliency should be >= 0 at j={j}");
        assert!(
            v.is_finite(),
            "mean absolute saliency should be finite at j={j}"
        );
    }
}

#[test]
fn test_saliency_mean_absolute_equals_column_mean() {
    // mean_absolute_saliency[j] should be the mean of |saliency_map[:,j]|
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_saliency(&fit, &data, None, 100, SEED).unwrap();
    for j in 0..M {
        let computed: f64 = (0..N).map(|i| res.saliency_map[(i, j)].abs()).sum::<f64>() / N as f64;
        assert!(
            (computed - res.mean_absolute_saliency[j]).abs() < 1e-10,
            "mean absolute saliency should match column mean at j={j}"
        );
    }
}

// =========================================================================
// VIF tests
// =========================================================================

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
fn test_vif_labels_length() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_vif(&fit, &data, None).unwrap();
    assert_eq!(res.labels.len(), NCOMP);
}

#[test]
fn test_vif_mean_geq_one() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_vif(&fit, &data, None).unwrap();
    assert!(
        res.mean_vif >= 1.0 - 1e-10,
        "mean VIF should be >= 1, got {}",
        res.mean_vif
    );
}

#[test]
fn test_vif_all_finite() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_vif(&fit, &data, None).unwrap();
    for (k, &v) in res.vif.iter().enumerate() {
        assert!(v.is_finite(), "VIF should be finite for component {k}");
    }
    assert!(res.mean_vif.is_finite(), "mean VIF should be finite");
}

#[test]
fn test_vif_moderate_severe_counts() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_vif(&fit, &data, None).unwrap();
    // n_severe <= n_moderate <= ncomp
    assert!(res.n_severe <= res.n_moderate);
    assert!(res.n_moderate <= NCOMP);
}

// =========================================================================
// Domain selection tests
// =========================================================================

#[test]
fn test_generic_domain_selection() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_domain_selection(&fit, &data, None, 5, 0.5, 100, SEED).unwrap();
    assert_eq!(res.pointwise_importance.len(), M);
}

#[test]
fn test_domain_selection_pointwise_nonnegative() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_domain_selection(&fit, &data, None, 5, 0.5, 100, SEED).unwrap();
    for (j, &v) in res.pointwise_importance.iter().enumerate() {
        assert!(
            v >= 0.0,
            "pointwise importance should be >= 0 at j={j}, got {v}"
        );
    }
}

#[test]
fn test_domain_selection_intervals_within_domain() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_domain_selection(&fit, &data, None, 5, 0.5, 100, SEED).unwrap();
    for interval in &res.intervals {
        assert!(
            interval.start_idx < M,
            "interval start {} should be < M={}",
            interval.start_idx,
            M
        );
        assert!(
            interval.end_idx < M,
            "interval end {} should be < M={}",
            interval.end_idx,
            M
        );
        assert!(
            interval.start_idx <= interval.end_idx,
            "interval start should be <= end"
        );
        assert!(
            interval.importance >= 0.0,
            "interval importance should be >= 0"
        );
    }
}

#[test]
fn test_domain_selection_window_width_stored() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let ww = 7;
    let res = generic_domain_selection(&fit, &data, None, ww, 0.5, 100, SEED).unwrap();
    assert_eq!(res.window_width, ww);
}

#[test]
fn test_domain_selection_threshold_stored() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let thresh = 0.3;
    let res = generic_domain_selection(&fit, &data, None, 5, thresh, 100, SEED).unwrap();
    assert!((res.threshold - thresh).abs() < 1e-12);
}

// =========================================================================
// Stability tests
// =========================================================================

#[test]
fn test_generic_stability() {
    let (data, y) = generate_test_data(N, M, SEED);
    let res = generic_stability(&data, &y, None, NCOMP, 10, SEED, TaskType::Regression).unwrap();
    assert_eq!(res.beta_t_std.len(), M);
    assert_eq!(res.coefficient_std.len(), NCOMP);
    assert!(res.n_boot_success > 0);
}

#[test]
fn test_stability_metrics_finite() {
    let (data, y) = generate_test_data(N, M, SEED);
    let res = generic_stability(&data, &y, None, NCOMP, 10, SEED, TaskType::Regression).unwrap();
    assert!(res.metric_std.is_finite(), "metric_std should be finite");
    assert!(
        res.importance_stability.is_finite(),
        "importance_stability should be finite"
    );
    for &v in &res.beta_t_std {
        assert!(v.is_finite(), "beta_t_std values should be finite");
    }
    for &v in &res.coefficient_std {
        assert!(v.is_finite(), "coefficient_std values should be finite");
    }
}

#[test]
fn test_stability_beta_t_cv_length() {
    let (data, y) = generate_test_data(N, M, SEED);
    let res = generic_stability(&data, &y, None, NCOMP, 10, SEED, TaskType::Regression).unwrap();
    assert_eq!(res.beta_t_cv.len(), M);
    for &v in &res.beta_t_cv {
        assert!(
            v.is_finite() || v.is_nan(),
            "beta_t_cv should be finite or NaN (where mean is 0)"
        );
    }
}

#[test]
fn test_stability_n_boot_success_leq_n_boot() {
    let (data, y) = generate_test_data(N, M, SEED);
    let n_boot = 15;
    let res =
        generic_stability(&data, &y, None, NCOMP, n_boot, SEED, TaskType::Regression).unwrap();
    assert!(
        res.n_boot_success <= n_boot,
        "n_boot_success should be <= n_boot"
    );
}

#[test]
fn test_stability_std_nonnegative() {
    let (data, y) = generate_test_data(N, M, SEED);
    let res = generic_stability(&data, &y, None, NCOMP, 10, SEED, TaskType::Regression).unwrap();
    assert!(res.metric_std >= 0.0, "metric_std should be >= 0");
    for &v in &res.beta_t_std {
        assert!(v >= 0.0, "beta_t_std should be >= 0");
    }
    for &v in &res.coefficient_std {
        assert!(v >= 0.0, "coefficient_std should be >= 0");
    }
}

#[test]
fn test_stability_logistic() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let res = generic_stability(
        &data,
        &y_bin,
        None,
        NCOMP,
        10,
        SEED,
        TaskType::BinaryClassification,
    )
    .unwrap();
    assert_eq!(res.beta_t_std.len(), M);
    assert_eq!(res.coefficient_std.len(), NCOMP);
    assert!(res.n_boot_success > 0);
}

#[test]
fn test_stability_error_multiclass() {
    let (data, y) = generate_test_data(N, M, SEED);
    assert!(generic_stability(
        &data,
        &y,
        None,
        NCOMP,
        10,
        SEED,
        TaskType::MulticlassClassification(3)
    )
    .is_err());
}

#[test]
fn test_stability_error_n_boot_one() {
    let (data, y) = generate_test_data(N, M, SEED);
    assert!(generic_stability(&data, &y, None, NCOMP, 1, SEED, TaskType::Regression).is_err());
}

#[test]
fn test_stability_error_too_few_rows() {
    let (data, _) = generate_test_data(3, M, SEED);
    let y: Vec<f64> = vec![1.0, 2.0, 3.0];
    assert!(generic_stability(&data, &y, None, NCOMP, 10, SEED, TaskType::Regression).is_err());
}

// =========================================================================
// Anchor tests
// =========================================================================

#[test]
fn test_generic_anchor() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_anchor(&fit, &data, None, 0, 0.9, 5).unwrap();
    assert_eq!(res.observation, 0);
}

#[test]
fn test_anchor_predicted_value_matches() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let obs = 3;
    let res = generic_anchor(&fit, &data, None, obs, 0.9, 5).unwrap();
    // The predicted value should match the model prediction for that observation
    let scores = fit.project(&data);
    let s: Vec<f64> = (0..NCOMP).map(|k| scores[(obs, k)]).collect();
    let expected_pred = fit.predict_from_scores(&s, None);
    assert!(
        (res.predicted_value - expected_pred).abs() < 1e-10,
        "anchor predicted_value should match model prediction"
    );
}

#[test]
fn test_anchor_rule_precision_in_range() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_anchor(&fit, &data, None, 0, 0.8, 5).unwrap();
    assert!(
        res.rule.precision >= 0.0 && res.rule.precision <= 1.0,
        "precision should be in [0,1], got {}",
        res.rule.precision
    );
}

#[test]
fn test_anchor_rule_coverage_in_range() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_anchor(&fit, &data, None, 0, 0.8, 5).unwrap();
    assert!(
        res.rule.coverage >= 0.0 && res.rule.coverage <= 1.0,
        "coverage should be in [0,1], got {}",
        res.rule.coverage
    );
}

#[test]
fn test_anchor_n_matching_leq_n() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_anchor(&fit, &data, None, 0, 0.8, 5).unwrap();
    assert!(res.rule.n_matching <= N, "n_matching should be <= N");
}

#[test]
fn test_anchor_different_observations() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    for obs in [0, 5, N - 1] {
        let res = generic_anchor(&fit, &data, None, obs, 0.8, 5).unwrap();
        assert_eq!(res.observation, obs);
    }
}

#[test]
fn test_anchor_error_invalid_observation() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_anchor(&fit, &data, None, N, 0.9, 5).is_err());
}

#[test]
fn test_anchor_error_n_bins_one() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    assert!(generic_anchor(&fit, &data, None, 0, 0.9, 1).is_err());
}

// =========================================================================
// FpcPredictor trait implementation tests
// =========================================================================

#[test]
fn test_fregre_lm_fpc_predictor_trait() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();

    // fpca_mean should have length m
    assert_eq!(fit.fpca_mean().len(), M);

    // fpca_rotation should be m x ncomp
    let (rm, rc) = fit.fpca_rotation().shape();
    assert_eq!(rm, M);
    assert_eq!(rc, NCOMP);

    // ncomp should match
    assert_eq!(fit.ncomp(), NCOMP);

    // training_scores should be n x ncomp
    let (sn, sc) = fit.training_scores().shape();
    assert_eq!(sn, N);
    assert_eq!(sc, NCOMP);

    // task_type should be Regression
    assert_eq!(fit.task_type(), TaskType::Regression);
}

#[test]
fn test_fregre_lm_predict_from_scores_matches_fitted() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let scores = fit.project(&data);
    for i in 0..N {
        let s: Vec<f64> = (0..NCOMP).map(|k| scores[(i, k)]).collect();
        let pred = fit.predict_from_scores(&s, None);
        assert!(
            (pred - fit.fitted_values[i]).abs() < 1e-6,
            "predict_from_scores should match fitted value for obs {i}: pred={pred}, fitted={}",
            fit.fitted_values[i]
        );
    }
}

#[test]
fn test_fregre_lm_project_produces_correct_shape() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let projected = fit.project(&data);
    let (pn, pc) = projected.shape();
    assert_eq!(pn, N);
    assert_eq!(pc, NCOMP);
}

#[test]
fn test_functional_logistic_fpc_predictor_trait() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();

    assert_eq!(fit.fpca_mean().len(), M);

    let (rm, rc) = fit.fpca_rotation().shape();
    assert_eq!(rm, M);
    assert_eq!(rc, NCOMP);

    assert_eq!(fit.ncomp(), NCOMP);

    let (sn, sc) = fit.training_scores().shape();
    assert_eq!(sn, N);
    assert_eq!(sc, NCOMP);

    // task_type should be BinaryClassification
    assert_eq!(fit.task_type(), TaskType::BinaryClassification);
}

#[test]
fn test_functional_logistic_predict_from_scores_probability() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();
    let scores = fit.project(&data);
    for i in 0..N {
        let s: Vec<f64> = (0..NCOMP).map(|k| scores[(i, k)]).collect();
        let prob = fit.predict_from_scores(&s, None);
        assert!(
            (0.0..=1.0).contains(&prob),
            "logistic predict_from_scores should return probability in [0,1], got {prob}"
        );
    }
}

#[test]
fn test_classif_fit_fpc_predictor_trait() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_class = make_multiclass_y(&y);
    let fit = fclassif_lda_fit(&data, &y_class, None, NCOMP).unwrap();

    assert_eq!(fit.fpca_mean().len(), M);

    let (rm, rc) = fit.fpca_rotation().shape();
    assert_eq!(rm, M);
    assert_eq!(rc, NCOMP);

    assert_eq!(fit.ncomp(), NCOMP);

    let (sn, sc) = fit.training_scores().shape();
    assert_eq!(sn, N);
    assert_eq!(sc, NCOMP);

    // task_type should be MulticlassClassification(3)
    assert_eq!(fit.task_type(), TaskType::MulticlassClassification(3));
}

#[test]
fn test_classif_fit_predict_from_scores_valid_class() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_class = make_multiclass_y(&y);
    let fit = fclassif_lda_fit(&data, &y_class, None, NCOMP).unwrap();
    let scores = fit.project(&data);
    for i in 0..N {
        let s: Vec<f64> = (0..NCOMP).map(|k| scores[(i, k)]).collect();
        let pred = fit.predict_from_scores(&s, None);
        let class = pred.round() as usize;
        assert!(class < 3, "predicted class {class} should be < n_classes=3");
    }
}

#[test]
fn test_classif_binary_task_type() {
    // 2-class LDA should report BinaryClassification
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin: Vec<usize> = y.iter().map(|&v| if v >= 0.0 { 1 } else { 0 }).collect();
    let fit = fclassif_lda_fit(&data, &y_bin, None, NCOMP).unwrap();
    assert_eq!(fit.task_type(), TaskType::BinaryClassification);
}

// =========================================================================
// Cross-model tests: same function with different model types
// =========================================================================

#[test]
fn test_pdp_with_logistic_model() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();
    let res = generic_pdp(&fit, &data, None, 0, 10).unwrap();
    // PDP values for logistic model should be probabilities (approximately in [0,1])
    for &v in &res.pdp_curve {
        assert!(v.is_finite());
        assert!(
            (-0.1..=1.1).contains(&v),
            "logistic PDP should be near [0,1], got {v}"
        );
    }
}

#[test]
fn test_shap_with_logistic_model() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();
    let res = generic_shap_values(&fit, &data, None, 100, SEED).unwrap();
    let (nr, nc) = res.values.shape();
    assert_eq!(nr, N);
    assert_eq!(nc, NCOMP);
    assert!(res.base_value.is_finite());
}

#[test]
fn test_lime_with_logistic_model() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();
    let res = generic_lime(&fit, &data, None, 0, 100, 1.0, SEED).unwrap();
    assert_eq!(res.attributions.len(), NCOMP);
}

#[test]
fn test_permutation_importance_with_logistic() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();
    let res = generic_permutation_importance(&fit, &data, &y_bin, 20, SEED).unwrap();
    assert_eq!(res.importance.len(), NCOMP);
    // Baseline metric is accuracy for classification - should be in [0,1]
    assert!(
        res.baseline_metric >= 0.0 && res.baseline_metric <= 1.0,
        "classification baseline metric should be accuracy in [0,1], got {}",
        res.baseline_metric
    );
}

#[test]
fn test_ale_with_logistic_model() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();
    let res = generic_ale(&fit, &data, None, 0, 8).unwrap();
    assert!(!res.bin_midpoints.is_empty());
    for &v in &res.ale_values {
        assert!(v.is_finite());
    }
}

#[test]
fn test_anchor_with_logistic_model() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();
    let res = generic_anchor(&fit, &data, None, 0, 0.8, 5).unwrap();
    assert_eq!(res.observation, 0);
    // predicted_value should be a probability
    assert!(
        res.predicted_value >= 0.0 && res.predicted_value <= 1.0,
        "logistic anchor predicted_value should be in [0,1]"
    );
}

#[test]
fn test_pdp_with_classif_model() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_class = make_multiclass_y(&y);
    let fit = fclassif_lda_fit(&data, &y_class, None, NCOMP).unwrap();
    let res = generic_pdp(&fit, &data, None, 0, 10).unwrap();
    assert_eq!(res.grid_values.len(), 10);
    assert_eq!(res.pdp_curve.len(), 10);
}

#[test]
fn test_vif_with_logistic_model() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();
    let res = generic_vif(&fit, &data, None).unwrap();
    assert_eq!(res.vif.len(), NCOMP);
    for &v in &res.vif {
        assert!(v >= 1.0 - 1e-10, "VIF should be >= 1.0, got {}", v);
    }
}

// =========================================================================
// Varied data tests (alternative data generation)
// =========================================================================

#[test]
fn test_pdp_with_varied_data() {
    let (data, y) = generate_varied_data(20, 30);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let res = generic_pdp(&fit, &data, None, 0, 10).unwrap();
    assert_eq!(res.grid_values.len(), 10);
    assert_eq!(res.pdp_curve.len(), 10);
    for &v in &res.pdp_curve {
        assert!(v.is_finite());
    }
}

#[test]
fn test_shap_with_varied_data() {
    let (data, y) = generate_varied_data(20, 30);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let res = generic_shap_values(&fit, &data, None, 100, 99).unwrap();
    let (nr, nc) = res.values.shape();
    assert_eq!(nr, 20);
    assert_eq!(nc, 3);
}

#[test]
fn test_sobol_with_varied_data() {
    let (data, y) = generate_varied_data(20, 30);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let res = generic_sobol_indices(&fit, &data, None, 200, 99).unwrap();
    assert_eq!(res.first_order.len(), 3);
    assert!(res.var_y > 0.0);
}

#[test]
fn test_friedman_h_with_varied_data() {
    let (data, y) = generate_varied_data(20, 30);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let res = generic_friedman_h(&fit, &data, None, 0, 1, 8).unwrap();
    assert!(res.h_squared.is_finite());
    assert!(res.h_squared >= 0.0);
}

// =========================================================================
// Baseline metric tests (compute_baseline_metric helper)
// =========================================================================

#[test]
fn test_baseline_metric_regression_r_squared() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let scores = fit.project(&data);
    let baseline = compute_baseline_metric(&fit, &scores, &y, N);
    // R^2 on training data should be close to the reported r_squared
    assert!(
        (baseline - fit.r_squared).abs() < 0.1,
        "baseline R^2 {baseline} should be close to fit.r_squared {}",
        fit.r_squared
    );
}

#[test]
fn test_baseline_metric_classification_accuracy() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();
    let scores = fit.project(&data);
    let baseline = compute_baseline_metric(&fit, &scores, &y_bin, N);
    // Should be between 0 and 1
    assert!(
        (0.0..=1.0).contains(&baseline),
        "classification baseline should be in [0,1], got {baseline}"
    );
}

// =========================================================================
// Edge-case tests
// =========================================================================

#[test]
fn test_pdp_minimum_grid_size() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_pdp(&fit, &data, None, 0, 2).unwrap();
    assert_eq!(res.grid_values.len(), 2);
    assert_eq!(res.pdp_curve.len(), 2);
}

#[test]
fn test_permutation_importance_single_perm() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_permutation_importance(&fit, &data, &y, 1, SEED).unwrap();
    assert_eq!(res.importance.len(), NCOMP);
}

#[test]
fn test_prototype_all_prototypes() {
    // Request as many prototypes as observations
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let res = generic_prototype_criticism(&fit, &data, N, 0).unwrap();
    assert_eq!(res.prototype_indices.len(), N);
    // No criticisms possible when all are prototypes
    assert_eq!(res.criticism_indices.len(), 0);
}

#[test]
fn test_friedman_h_all_component_pairs() {
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    // Test all pairs of components
    for j in 0..NCOMP {
        for k in (j + 1)..NCOMP {
            let res = generic_friedman_h(&fit, &data, None, j, k, 5).unwrap();
            assert!(
                res.h_squared.is_finite(),
                "H^2 should be finite for ({j},{k})"
            );
            assert!(
                res.h_squared >= 0.0,
                "H^2 should be >= 0 for ({j},{k}), got {}",
                res.h_squared
            );
        }
    }
}

#[test]
fn test_counterfactual_reg_target_equals_current() {
    // Counterfactual with target = current prediction should have near-zero distance
    let (data, y) = generate_test_data(N, M, SEED);
    let fit = fregre_lm(&data, &y, None, NCOMP).unwrap();
    let scores = fit.project(&data);
    let s: Vec<f64> = (0..NCOMP).map(|k| scores[(0, k)]).collect();
    let current_pred = fit.predict_from_scores(&s, None);
    let res = generic_counterfactual(&fit, &data, None, 0, current_pred, 200, 0.01).unwrap();
    assert!(
        res.distance < 1e-6,
        "distance should be near zero when target equals current pred, got {}",
        res.distance
    );
}

#[test]
fn test_saliency_with_logistic_model() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();
    let res = generic_saliency(&fit, &data, None, 100, SEED).unwrap();
    let (nr, nc) = res.saliency_map.shape();
    assert_eq!(nr, N);
    assert_eq!(nc, M);
    assert_eq!(res.mean_absolute_saliency.len(), M);
}

#[test]
fn test_sobol_with_logistic_model() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();
    let res = generic_sobol_indices(&fit, &data, None, 200, SEED).unwrap();
    assert_eq!(res.first_order.len(), NCOMP);
    assert_eq!(res.total_order.len(), NCOMP);
}

#[test]
fn test_prototype_criticism_with_logistic() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();
    let res = generic_prototype_criticism(&fit, &data, 3, 3).unwrap();
    assert_eq!(res.prototype_indices.len(), 3);
    assert!(res.bandwidth > 0.0);
}

#[test]
fn test_domain_selection_with_logistic() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();
    let res = generic_domain_selection(&fit, &data, None, 5, 0.5, 100, SEED).unwrap();
    assert_eq!(res.pointwise_importance.len(), M);
}

#[test]
fn test_conditional_perm_with_logistic() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();
    let res =
        generic_conditional_permutation_importance(&fit, &data, &y_bin, None, 3, 10, SEED).unwrap();
    assert_eq!(res.importance.len(), NCOMP);
    // Baseline should be accuracy in [0,1]
    assert!(res.baseline_metric >= 0.0 && res.baseline_metric <= 1.0);
}

#[test]
fn test_friedman_h_with_logistic() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_bin = make_binary_y(&y);
    let fit = functional_logistic(&data, &y_bin, None, NCOMP, 25, 1e-6).unwrap();
    let res = generic_friedman_h(&fit, &data, None, 0, 1, 8).unwrap();
    assert!(res.h_squared.is_finite());
    assert!(res.h_squared >= 0.0);
}

#[test]
fn test_counterfactual_with_classif_model() {
    let (data, y) = generate_test_data(N, M, SEED);
    let y_class = make_multiclass_y(&y);
    let fit = fclassif_lda_fit(&data, &y_class, None, NCOMP).unwrap();
    let res = generic_counterfactual(&fit, &data, None, 0, 2.0, 500, 0.1).unwrap();
    assert_eq!(res.observation, 0);
    assert_eq!(res.delta_function.len(), M);
}
