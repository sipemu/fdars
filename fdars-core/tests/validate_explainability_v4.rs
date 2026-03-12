//! Integration tests for Explainability Round 4:
//! LOO-CV/PRESS, Sobol, Calibration, Saliency, Domain Selection,
//! Conditional Permutation, Counterfactual, Prototype/Criticism, LIME.

use fdars_core::matrix::FdMatrix;
use fdars_core::{
    calibration_diagnostics, conditional_permutation_importance,
    conditional_permutation_importance_logistic, counterfactual_logistic,
    counterfactual_regression, domain_selection, domain_selection_logistic, fregre_lm,
    functional_logistic, functional_saliency, functional_saliency_logistic, lime_explanation,
    lime_explanation_logistic, loo_cv_press, prototype_criticism, sobol_indices,
    sobol_indices_logistic,
};
use std::f64::consts::PI;

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

fn make_binary(y: &[f64]) -> Vec<f64> {
    let mut sorted = y.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    y.iter()
        .map(|&v| if v >= median { 1.0 } else { 0.0 })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// LOO-CV / PRESS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_loo_cv_shape() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let loo = loo_cv_press(&fit, &data, &y, None).unwrap();
    assert_eq!(loo.loo_residuals.len(), 40);
    assert_eq!(loo.leverage.len(), 40);
    assert!(loo.tss > 0.0);
}

#[test]
fn test_loo_r_squared_leq_r_squared() {
    let (data, y) = generate_test_data(40, 50, 42);
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
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let loo = loo_cv_press(&fit, &data, &y, None).unwrap();
    let manual_press: f64 = loo.loo_residuals.iter().map(|r| r * r).sum();
    assert!(
        (loo.press - manual_press).abs() < 1e-10,
        "PRESS ({}) should equal Σ loo_residuals² ({})",
        loo.press,
        manual_press
    );
}

#[test]
fn test_loo_leverage_bounded() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let loo = loo_cv_press(&fit, &data, &y, None).unwrap();
    for (i, &h) in loo.leverage.iter().enumerate() {
        assert!(
            (0.0..=1.0 + 1e-10).contains(&h),
            "Leverage h_ii should be in [0,1] at i={}: {}",
            i,
            h
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sobol Indices
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_sobol_linear_nonnegative() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sobol = sobol_indices(&fit, &data, &y, None).unwrap();
    for (k, &s) in sobol.first_order.iter().enumerate() {
        assert!(s >= -1e-10, "S_{} should be ≥ 0: {}", k, s);
    }
    for (k, &st) in sobol.total_order.iter().enumerate() {
        assert!(st >= -1e-10, "ST_{} should be ≥ 0: {}", k, st);
    }
}

#[test]
fn test_sobol_linear_sum_approx_r2() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sobol = sobol_indices(&fit, &data, &y, None).unwrap();
    let sum_s: f64 = sobol.first_order.iter().sum();
    assert!(
        (sum_s - fit.r_squared).abs() < 0.25,
        "Σ S_k ({}) should approximate R² ({})",
        sum_s,
        fit.r_squared
    );
}

#[test]
fn test_sobol_linear_first_equals_total() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sobol = sobol_indices(&fit, &data, &y, None).unwrap();
    for k in 0..3 {
        assert!(
            (sobol.first_order[k] - sobol.total_order[k]).abs() < 1e-15,
            "For additive+orthogonal model, S_k should equal ST_k"
        );
    }
}

#[test]
fn test_sobol_logistic_bounded() {
    let (data, y_cont) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let sobol = sobol_indices_logistic(&fit, &data, None, 1000, 42).unwrap();
    for &s in &sobol.first_order {
        assert!(s > -1.0 && s < 2.0, "Logistic S_k should be bounded: {}", s);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Calibration Diagnostics
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_calibration_brier_range() {
    let (data, y_cont) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let cal = calibration_diagnostics(&fit, &y_bin, 5).unwrap();
    assert!(
        cal.brier_score >= 0.0 && cal.brier_score <= 1.0,
        "Brier score should be in [0,1]: {}",
        cal.brier_score
    );
}

#[test]
fn test_calibration_log_loss_positive() {
    let (data, y_cont) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let cal = calibration_diagnostics(&fit, &y_bin, 5).unwrap();
    assert!(
        cal.log_loss > 0.0,
        "Log loss should be > 0: {}",
        cal.log_loss
    );
}

#[test]
fn test_calibration_bin_counts_sum_to_n() {
    let (data, y_cont) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let cal = calibration_diagnostics(&fit, &y_bin, 5).unwrap();
    let total: usize = cal.bin_counts.iter().sum();
    assert_eq!(total, 40, "Bin counts should sum to n");
}

#[test]
fn test_calibration_n_groups_match() {
    let (data, y_cont) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let cal = calibration_diagnostics(&fit, &y_bin, 5).unwrap();
    assert_eq!(cal.n_groups, cal.reliability_bins.len());
    assert_eq!(cal.n_groups, cal.bin_counts.len());
}

#[test]
fn test_calibration_hl_df() {
    let (data, y_cont) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let cal = calibration_diagnostics(&fit, &y_bin, 10).unwrap();
    assert!(cal.hosmer_lemeshow_chi2 >= 0.0);
    assert!(cal.hosmer_lemeshow_df > 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Functional Saliency Maps
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_saliency_linear_shape() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sal = functional_saliency(&fit, &data, None).unwrap();
    assert_eq!(sal.saliency_map.shape(), (40, 50));
    assert_eq!(sal.mean_absolute_saliency.len(), 50);
}

#[test]
fn test_saliency_logistic_bounded_by_quarter_beta() {
    let (data, y_cont) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let sal = functional_saliency_logistic(&fit).unwrap();
    for i in 0..40 {
        for j in 0..50 {
            assert!(
                sal.saliency_map[(i, j)].abs() <= 0.25 * fit.beta_t[j].abs() + 1e-10,
                "|saliency| should be ≤ 0.25|β(t)| at ({},{}): {} vs {}",
                i,
                j,
                sal.saliency_map[(i, j)].abs(),
                0.25 * fit.beta_t[j].abs()
            );
        }
    }
}

#[test]
fn test_saliency_mean_abs_nonneg() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sal = functional_saliency(&fit, &data, None).unwrap();
    for &v in &sal.mean_absolute_saliency {
        assert!(v >= 0.0, "Mean absolute saliency should be ≥ 0: {}", v);
    }
}

#[test]
fn test_saliency_logistic_shape() {
    let (data, y_cont) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let sal = functional_saliency_logistic(&fit).unwrap();
    assert_eq!(sal.saliency_map.shape(), (40, 50));
    assert_eq!(sal.mean_absolute_saliency.len(), 50);
}

// ═══════════════════════════════════════════════════════════════════════════
// Domain Selection
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_domain_selection_valid_indices() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let ds = domain_selection(&fit, 5, 0.01).unwrap();
    assert_eq!(ds.pointwise_importance.len(), 50);
    for iv in &ds.intervals {
        assert!(iv.start_idx <= iv.end_idx, "start should be ≤ end");
        assert!(iv.end_idx < 50, "end should be < m");
    }
}

#[test]
fn test_domain_selection_full_window_one_interval() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let ds = domain_selection(&fit, 50, 0.01).unwrap();
    assert!(
        ds.intervals.len() <= 1,
        "Full window should give at most 1 interval: {}",
        ds.intervals.len()
    );
}

#[test]
fn test_domain_selection_high_threshold_fewer() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let ds_low = domain_selection(&fit, 5, 0.01).unwrap();
    let ds_high = domain_selection(&fit, 5, 0.5).unwrap();
    assert!(
        ds_high.intervals.len() <= ds_low.intervals.len(),
        "Higher threshold → fewer intervals: {} vs {}",
        ds_high.intervals.len(),
        ds_low.intervals.len()
    );
}

#[test]
fn test_domain_selection_logistic() {
    let (data, y_cont) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let ds = domain_selection_logistic(&fit, 5, 0.01).unwrap();
    assert_eq!(ds.pointwise_importance.len(), 50);
}

// ═══════════════════════════════════════════════════════════════════════════
// Conditional Permutation Importance
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_cond_perm_shape() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let cp = conditional_permutation_importance(&fit, &data, &y, None, 3, 10, 42).unwrap();
    assert_eq!(cp.importance.len(), 3);
    assert_eq!(cp.permuted_metric.len(), 3);
    assert_eq!(cp.unconditional_importance.len(), 3);
}

#[test]
fn test_cond_perm_vs_unconditional_close() {
    let (data, y) = generate_test_data(50, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let cp = conditional_permutation_importance(&fit, &data, &y, None, 3, 30, 42).unwrap();
    for k in 0..3 {
        let diff = (cp.importance[k] - cp.unconditional_importance[k]).abs();
        assert!(
            diff < 0.5,
            "FPC {} cond vs uncond should be similar: {} vs {}",
            k,
            cp.importance[k],
            cp.unconditional_importance[k]
        );
    }
}

#[test]
fn test_cond_perm_logistic_shape() {
    let (data, y_cont) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let cp =
        conditional_permutation_importance_logistic(&fit, &data, &y_bin, None, 3, 10, 42).unwrap();
    assert_eq!(cp.importance.len(), 3);
    assert_eq!(cp.unconditional_importance.len(), 3);
}

// ═══════════════════════════════════════════════════════════════════════════
// Counterfactual Explanations
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_counterfactual_regression_exact() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let target = fit.fitted_values[5] + 2.0;
    let cf = counterfactual_regression(&fit, &data, None, 5, target).unwrap();
    assert!(cf.found);
    assert!(
        (cf.counterfactual_prediction - target).abs() < 1e-8,
        "CF prediction ({}) should match target ({})",
        cf.counterfactual_prediction,
        target
    );
}

#[test]
fn test_counterfactual_regression_minimal_distance() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let gap = 1.0;
    let target = fit.fitted_values[0] + gap;
    let cf = counterfactual_regression(&fit, &data, None, 0, target).unwrap();
    let gamma: Vec<f64> = (0..3).map(|k| fit.coefficients[1 + k]).collect();
    let gamma_norm: f64 = gamma.iter().map(|g| g * g).sum::<f64>().sqrt();
    let expected_dist = gap.abs() / gamma_norm;
    assert!(
        (cf.distance - expected_dist).abs() < 1e-6,
        "Distance ({}) should be |gap|/||γ|| ({})",
        cf.distance,
        expected_dist
    );
}

#[test]
fn test_counterfactual_regression_delta_function_length() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let cf = counterfactual_regression(&fit, &data, None, 0, 0.0).unwrap();
    assert_eq!(cf.delta_function.len(), 50);
    assert_eq!(cf.delta_scores.len(), 3);
}

#[test]
fn test_counterfactual_logistic_flips_class() {
    let (data, y_cont) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let cf = counterfactual_logistic(&fit, &data, None, 0, 1000, 0.5).unwrap();
    if cf.found {
        let orig_class = if cf.original_prediction >= 0.5 { 1 } else { 0 };
        let new_class = if cf.counterfactual_prediction >= 0.5 {
            1
        } else {
            0
        };
        assert_ne!(orig_class, new_class, "Class should flip");
    }
}

#[test]
fn test_counterfactual_invalid_obs_none() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    assert!(counterfactual_regression(&fit, &data, None, 100, 0.0).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Prototype / Criticism
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_prototype_criticism_shape() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pc = prototype_criticism(&fit.fpca, 3, 5, 3).unwrap();
    assert_eq!(pc.prototype_indices.len(), 5);
    assert_eq!(pc.prototype_witness.len(), 5);
    assert_eq!(pc.criticism_indices.len(), 3);
    assert_eq!(pc.criticism_witness.len(), 3);
}

#[test]
fn test_prototype_criticism_no_overlap() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pc = prototype_criticism(&fit.fpca, 3, 5, 3).unwrap();
    for &ci in &pc.criticism_indices {
        assert!(
            !pc.prototype_indices.contains(&ci),
            "Criticism {} overlaps with prototypes",
            ci
        );
    }
}

#[test]
fn test_prototype_criticism_bandwidth_positive() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pc = prototype_criticism(&fit.fpca, 3, 5, 3).unwrap();
    assert!(
        pc.bandwidth > 0.0,
        "Bandwidth should be > 0: {}",
        pc.bandwidth
    );
}

#[test]
fn test_prototype_criticism_logistic() {
    let (data, y_cont) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let pc = prototype_criticism(&fit.fpca, 3, 5, 3).unwrap();
    assert_eq!(pc.prototype_indices.len(), 5);
}

#[test]
fn test_prototype_indices_valid() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pc = prototype_criticism(&fit.fpca, 3, 5, 5).unwrap();
    for &pi in &pc.prototype_indices {
        assert!(pi < 40, "Prototype index should be < n: {}", pi);
    }
    for &ci in &pc.criticism_indices {
        assert!(ci < 40, "Criticism index should be < n: {}", ci);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LIME
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_lime_linear_matches_global() {
    let (data, y) = generate_test_data(50, 50, 42);
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
            "LIME FPC {} local ({}) should approximate global ({})",
            k,
            local,
            global
        );
    }
}

#[test]
fn test_lime_logistic_shape() {
    let (data, y_cont) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let lime = lime_explanation_logistic(&fit, &data, None, 0, 500, 1.0, 42).unwrap();
    assert_eq!(lime.attributions.len(), 3);
    assert!(
        lime.local_r_squared >= 0.0 && lime.local_r_squared <= 1.0,
        "Local R² should be in [0,1]: {}",
        lime.local_r_squared
    );
    assert_eq!(lime.observation, 0);
}

#[test]
fn test_lime_invalid_none() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    assert!(lime_explanation(&fit, &data, None, 100, 100, 1.0, 42).is_err());
    assert!(lime_explanation(&fit, &data, None, 0, 0, 1.0, 42).is_err());
    assert!(lime_explanation(&fit, &data, None, 0, 100, 0.0, 42).is_err());
    assert!(lime_explanation(&fit, &data, None, 0, 100, -1.0, 42).is_err());
}

#[test]
fn test_lime_different_observations() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let lime0 = lime_explanation(&fit, &data, None, 0, 1000, 1.0, 42).unwrap();
    let lime1 = lime_explanation(&fit, &data, None, 1, 1000, 1.0, 42).unwrap();
    // For a linear model, attributions should be similar across observations
    // (they approximate the same global coefficients)
    assert_eq!(lime0.observation, 0);
    assert_eq!(lime1.observation, 1);
}
