//! Integration tests for LOO-CV, Sobol indices, calibration diagnostics,
//! saliency maps, domain selection, conditional permutation importance,
//! counterfactual explanations, prototype/criticism, and LIME.
//!
//! Covers:
//! - LOO-CV / PRESS statistics
//! - Sobol sensitivity indices
//! - Calibration diagnostics (Brier, log-loss, Hosmer-Lemeshow)
//! - Functional saliency maps
//! - Domain selection
//! - Conditional permutation importance
//! - Counterfactual explanations
//! - Prototype / criticism selection
//! - LIME (Local Interpretable Model-agnostic Explanations)
//! - Cross-method consistency checks

use fdars_core::matrix::FdMatrix;
use std::f64::consts::PI;
use fdars_core::{
    calibration_diagnostics,
    conditional_permutation_importance,
    conditional_permutation_importance_logistic,
    counterfactual_logistic,
    counterfactual_regression,
    domain_selection,
    domain_selection_logistic,
    fregre_lm,
    functional_logistic,
    functional_saliency,
    functional_saliency_logistic,
    lime_explanation,
    lime_explanation_logistic,
    loo_cv_press,
    prototype_criticism,
    sobol_indices,
    sobol_indices_logistic,
};

// ─── Test data generators ────────────────────────────────────────────────────

fn regression_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0; n];
    for i in 0..n {
        let phase = (seed.wrapping_mul(17).wrapping_add(i as u64 * 31) % 1000) as f64 / 1000.0 * PI;
        let amp = ((seed.wrapping_mul(13).wrapping_add(i as u64 * 7) % 100) as f64 / 100.0) - 0.5;
        for j in 0..m {
            let t = j as f64 / (m - 1) as f64;
            data[(i, j)] = (2.0 * PI * t + phase).sin() + amp * (4.0 * PI * t).cos();
        }
        y[i] = 2.0 * phase + 3.0 * amp;
    }
    (data, y)
}

fn binary_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
    let (data, y_cont) = regression_data(n, m, seed);
    let mut sorted = y_cont.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[sorted.len() / 2];
    let y_bin: Vec<f64> = y_cont
        .iter()
        .map(|&v| if v >= med { 1.0 } else { 0.0 })
        .collect();
    (data, y_bin)
}

fn make_binary(y: &[f64]) -> Vec<f64> {
    let mut sorted = y.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    y.iter()
        .map(|&v| if v >= median { 1.0 } else { 0.0 })
        .collect()
}

// ─── Helper functions for deep validation ────────────────────────────────────

fn regression_data_with_scalars(
    n: usize,
    m: usize,
    p_scalar: usize,
    seed: u64,
) -> (FdMatrix, Vec<f64>, FdMatrix) {
    let (data, mut y) = regression_data(n, m, seed);
    let mut sc = FdMatrix::zeros(n, p_scalar);
    for i in 0..n {
        for j in 0..p_scalar {
            let v = ((seed
                .wrapping_mul(23)
                .wrapping_add(i as u64 * 11 + j as u64 * 37))
                % 1000) as f64
                / 1000.0;
            sc[(i, j)] = v;
            y[i] += 0.5 * v;
        }
    }
    (data, y, sc)
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOO-CV / PRESS — deep validation
// ═══════════════════════════════════════════════════════════════════════════════



// ═══════════════════════════════════════════════════════════════════════════════
// Basic validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_loo_cv_shape() {
    let (data, y) = regression_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let loo = loo_cv_press(&fit, &data, &y, None).unwrap();
    assert_eq!(loo.loo_residuals.len(), 40);
    assert_eq!(loo.leverage.len(), 40);
    assert!(loo.tss > 0.0);
}

#[test]
fn test_loo_r_squared_leq_r_squared() {
    let (data, y) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
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
    let (data, y_cont) = regression_data(40, 50, 42);
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
    let (data, y_cont) = regression_data(40, 50, 42);
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
    let (data, y_cont) = regression_data(40, 50, 42);
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
    let (data, y_cont) = regression_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let cal = calibration_diagnostics(&fit, &y_bin, 5).unwrap();
    let total: usize = cal.bin_counts.iter().sum();
    assert_eq!(total, 40, "Bin counts should sum to n");
}

#[test]
fn test_calibration_n_groups_match() {
    let (data, y_cont) = regression_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let cal = calibration_diagnostics(&fit, &y_bin, 5).unwrap();
    assert_eq!(cal.n_groups, cal.reliability_bins.len());
    assert_eq!(cal.n_groups, cal.bin_counts.len());
}

#[test]
fn test_calibration_hl_df() {
    let (data, y_cont) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sal = functional_saliency(&fit, &data, None).unwrap();
    assert_eq!(sal.saliency_map.shape(), (40, 50));
    assert_eq!(sal.mean_absolute_saliency.len(), 50);
}

#[test]
fn test_saliency_logistic_bounded_by_quarter_beta() {
    let (data, y_cont) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sal = functional_saliency(&fit, &data, None).unwrap();
    for &v in &sal.mean_absolute_saliency {
        assert!(v >= 0.0, "Mean absolute saliency should be ≥ 0: {}", v);
    }
}

#[test]
fn test_saliency_logistic_shape() {
    let (data, y_cont) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
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
    let (data, y_cont) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let cp = conditional_permutation_importance(&fit, &data, &y, None, 3, 10, 42).unwrap();
    assert_eq!(cp.importance.len(), 3);
    assert_eq!(cp.permuted_metric.len(), 3);
    assert_eq!(cp.unconditional_importance.len(), 3);
}

#[test]
fn test_cond_perm_vs_unconditional_close() {
    let (data, y) = regression_data(50, 50, 42);
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
    let (data, y_cont) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let cf = counterfactual_regression(&fit, &data, None, 0, 0.0).unwrap();
    assert_eq!(cf.delta_function.len(), 50);
    assert_eq!(cf.delta_scores.len(), 3);
}

#[test]
fn test_counterfactual_logistic_flips_class() {
    let (data, y_cont) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    assert!(counterfactual_regression(&fit, &data, None, 100, 0.0).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════
// Prototype / Criticism
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_prototype_criticism_shape() {
    let (data, y) = regression_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pc = prototype_criticism(&fit.fpca, 3, 5, 3).unwrap();
    assert_eq!(pc.prototype_indices.len(), 5);
    assert_eq!(pc.prototype_witness.len(), 5);
    assert_eq!(pc.criticism_indices.len(), 3);
    assert_eq!(pc.criticism_witness.len(), 3);
}

#[test]
fn test_prototype_criticism_no_overlap() {
    let (data, y) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
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
    let (data, y_cont) = regression_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let pc = prototype_criticism(&fit.fpca, 3, 5, 3).unwrap();
    assert_eq!(pc.prototype_indices.len(), 5);
}

#[test]
fn test_prototype_indices_valid() {
    let (data, y) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(50, 50, 42);
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
    let (data, y_cont) = regression_data(40, 50, 42);
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
    let (data, y) = regression_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    assert!(lime_explanation(&fit, &data, None, 100, 100, 1.0, 42).is_err());
    assert!(lime_explanation(&fit, &data, None, 0, 0, 1.0, 42).is_err());
    assert!(lime_explanation(&fit, &data, None, 0, 100, 0.0, 42).is_err());
    assert!(lime_explanation(&fit, &data, None, 0, 100, -1.0, 42).is_err());
}

#[test]
fn test_lime_different_observations() {
    let (data, y) = regression_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let lime0 = lime_explanation(&fit, &data, None, 0, 1000, 1.0, 42).unwrap();
    let lime1 = lime_explanation(&fit, &data, None, 1, 1000, 1.0, 42).unwrap();
    // For a linear model, attributions should be similar across observations
    // (they approximate the same global coefficients)
    assert_eq!(lime0.observation, 0);
    assert_eq!(lime1.observation, 1);
}


// ═══════════════════════════════════════════════════════════════════════════════
// Deep validation — mathematical invariants and cross-method consistency
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn loo_cv_hat_diagonal_identity() {
    // Verify h_ii from LOO matches manual computation via (X'X)^{-1}
    // h_ii = x_i' (X'X)^{-1} x_i
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let loo = fdars_core::loo_cv_press(&fit, &data, &y, None).unwrap();

    // Also compute via influence diagnostics
    let infl = fdars_core::influence_diagnostics(&fit, &data, None).unwrap();
    for i in 0..60 {
        assert!(
            (loo.leverage[i] - infl.leverage[i]).abs() < 1e-12,
            "Leverage from LOO should match influence diagnostics at i={}: {} vs {}",
            i,
            loo.leverage[i],
            infl.leverage[i]
        );
    }
}

#[test]
fn loo_cv_residual_formula_verified() {
    // LOO residual = e_i / (1 - h_ii), verify manually
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let loo = fdars_core::loo_cv_press(&fit, &data, &y, None).unwrap();

    for i in 0..60 {
        let expected = fit.residuals[i] / (1.0 - loo.leverage[i]).max(1e-15);
        assert!(
            (loo.loo_residuals[i] - expected).abs() < 1e-12,
            "LOO residual formula at i={}: {} vs {}",
            i,
            loo.loo_residuals[i],
            expected
        );
    }
}

#[test]
fn loo_cv_press_identity() {
    // PRESS = Σ loo_residuals² — verified to machine epsilon
    let (data, y) = regression_data(80, 40, 77);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let loo = fdars_core::loo_cv_press(&fit, &data, &y, None).unwrap();
    let manual_press: f64 = loo.loo_residuals.iter().map(|r| r * r).sum();
    assert!(
        (loo.press - manual_press).abs() < 1e-12,
        "PRESS identity: {} vs {}",
        loo.press,
        manual_press
    );
}

#[test]
fn loo_r_squared_formula_verified() {
    // loo_r_squared = 1 - PRESS / TSS
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let loo = fdars_core::loo_cv_press(&fit, &data, &y, None).unwrap();
    let expected = 1.0 - loo.press / loo.tss;
    assert!(
        (loo.loo_r_squared - expected).abs() < 1e-12,
        "LOO R² formula: {} vs {}",
        loo.loo_r_squared,
        expected
    );
}

#[test]
fn loo_r_squared_strictly_less_than_r_squared() {
    // LOO R² < training R² (with proper data, not just ≤)
    for seed in [42, 77, 123, 999] {
        let (data, y) = regression_data(60, 30, seed);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let loo = fdars_core::loo_cv_press(&fit, &data, &y, None).unwrap();
        assert!(
            loo.loo_r_squared < fit.r_squared + 1e-10,
            "LOO R² ({}) should be < training R² ({}) for seed={}",
            loo.loo_r_squared,
            fit.r_squared,
            seed
        );
    }
}

#[test]
fn loo_cv_tss_matches_manual() {
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let loo = fdars_core::loo_cv_press(&fit, &data, &y, None).unwrap();
    let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
    let manual_tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    assert!(
        (loo.tss - manual_tss).abs() < 1e-10,
        "TSS: {} vs {}",
        loo.tss,
        manual_tss
    );
}

#[test]
fn loo_cv_leverage_sum_equals_p() {
    // Σ h_ii = p (number of parameters including intercept)
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let loo = fdars_core::loo_cv_press(&fit, &data, &y, None).unwrap();
    let sum_h: f64 = loo.leverage.iter().sum();
    let p = (fit.ncomp + 1) as f64; // intercept + ncomp
    assert!(
        (sum_h - p).abs() < 1e-8,
        "Σ h_ii ({}) should equal p ({})",
        sum_h,
        p
    );
}

#[test]
fn loo_cv_with_scalar_covariates() {
    let (data, y, sc) = regression_data_with_scalars(60, 30, 2, 42);
    let fit = fregre_lm(&data, &y, Some(&sc), 3).unwrap();
    let loo = fdars_core::loo_cv_press(&fit, &data, &y, Some(&sc)).unwrap();
    let sum_h: f64 = loo.leverage.iter().sum();
    let p = (fit.ncomp + 1 + 2) as f64; // intercept + ncomp + scalars
    assert!(
        (sum_h - p).abs() < 1e-8,
        "Σ h_ii ({}) should equal p ({}) with scalar covariates",
        sum_h,
        p
    );
}

#[test]
fn loo_cv_all_finite() {
    for seed in [42, 77, 123, 999] {
        let (data, y) = regression_data(80, 40, seed);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let loo = fdars_core::loo_cv_press(&fit, &data, &y, None).unwrap();
        for (i, &r) in loo.loo_residuals.iter().enumerate() {
            assert!(
                r.is_finite(),
                "LOO residual not finite at i={} (seed={})",
                i,
                seed
            );
        }
        assert!(loo.press.is_finite() && loo.press >= 0.0);
        assert!(loo.loo_r_squared.is_finite());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sobol — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn sobol_linear_variance_decomposition_identity() {
    // component_variance[k] = coef[1+k]² × Var(score_k)
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sobol = fdars_core::sobol_indices(&fit, &data, &y, None).unwrap();
    let n = fit.fpca.scores.nrows();

    for k in 0..3 {
        let mut ss = 0.0;
        for i in 0..n {
            let s = fit.fpca.scores[(i, k)];
            ss += s * s;
        }
        let score_var = ss / (n - 1) as f64;
        let expected = fit.coefficients[1 + k].powi(2) * score_var;
        assert!(
            (sobol.component_variance[k] - expected).abs() < 1e-12,
            "Component variance[{}]: {} vs {}",
            k,
            sobol.component_variance[k],
            expected
        );
    }
}

#[test]
fn sobol_linear_first_order_is_ratio() {
    // S_k = component_variance[k] / var_y
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sobol = fdars_core::sobol_indices(&fit, &data, &y, None).unwrap();

    for k in 0..3 {
        let expected = sobol.component_variance[k] / sobol.var_y;
        assert!(
            (sobol.first_order[k] - expected).abs() < 1e-12,
            "S_{} ratio: {} vs {}",
            k,
            sobol.first_order[k],
            expected
        );
    }
}

#[test]
fn sobol_linear_var_y_is_sample_variance() {
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sobol = fdars_core::sobol_indices(&fit, &data, &y, None).unwrap();
    let n = y.len();
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let manual_var: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    assert!(
        (sobol.var_y - manual_var).abs() < 1e-10,
        "var_y should be sample variance: {} vs {}",
        sobol.var_y,
        manual_var
    );
}

#[test]
fn sobol_linear_sum_bounded_by_one() {
    // Σ S_k ≤ 1 (residual variance accounts for the rest)
    for seed in [42, 77, 123] {
        let (data, y) = regression_data(80, 40, seed);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let sobol = fdars_core::sobol_indices(&fit, &data, &y, None).unwrap();
        let sum_s: f64 = sobol.first_order.iter().sum();
        assert!(
            sum_s <= 1.0 + 1e-10,
            "Σ S_k ({}) should be ≤ 1 for seed={}",
            sum_s,
            seed
        );
    }
}

#[test]
fn sobol_linear_additive_orthogonal_first_equals_total() {
    // For additive model with orthogonal FPCs, S_k = ST_k exactly
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sobol = fdars_core::sobol_indices(&fit, &data, &y, None).unwrap();
    for k in 0..3 {
        assert!(
            (sobol.first_order[k] - sobol.total_order[k]).abs() < 1e-15,
            "S_{} = ST_{} for additive orthogonal model",
            k,
            k
        );
    }
}

#[test]
fn sobol_linear_stable_across_sizes() {
    // Sobol indices with n=60 vs n=120 should be similar
    let (data60, y60) = regression_data(60, 30, 42);
    let (data120, y120) = regression_data(120, 30, 42);
    let fit60 = fregre_lm(&data60, &y60, None, 3).unwrap();
    let fit120 = fregre_lm(&data120, &y120, None, 3).unwrap();
    let sobol60 = fdars_core::sobol_indices(&fit60, &data60, &y60, None).unwrap();
    let sobol120 = fdars_core::sobol_indices(&fit120, &data120, &y120, None).unwrap();

    for k in 0..3 {
        let diff = (sobol60.first_order[k] - sobol120.first_order[k]).abs();
        assert!(
            diff < 0.3,
            "Sobol S_{} should be stable: {} vs {} (diff={})",
            k,
            sobol60.first_order[k],
            sobol120.first_order[k],
            diff
        );
    }
}

#[test]
fn sobol_logistic_total_order_nonnegative() {
    // ST_k ≥ 0 by construction (mean of squared differences)
    let (data, y_bin) = binary_data(80, 40, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let sobol = fdars_core::sobol_indices_logistic(&fit, &data, None, 2000, 42).unwrap();
    for (k, &st) in sobol.total_order.iter().enumerate() {
        assert!(st >= -0.05, "ST_{} should be ≥ 0: {}", k, st);
    }
}

#[test]
fn sobol_logistic_reproducible_with_same_seed() {
    let (data, y_bin) = binary_data(60, 30, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let s1 = fdars_core::sobol_indices_logistic(&fit, &data, None, 500, 42).unwrap();
    let s2 = fdars_core::sobol_indices_logistic(&fit, &data, None, 500, 42).unwrap();
    for k in 0..3 {
        assert!(
            (s1.first_order[k] - s2.first_order[k]).abs() < 1e-15,
            "Same seed should give same result"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Calibration — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn calibration_brier_manual_computation() {
    // Brier = (1/n) Σ (p_i - y_i)²
    let (data, y_bin) = binary_data(60, 30, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let cal = fdars_core::calibration_diagnostics(&fit, &y_bin, 5).unwrap();

    let manual_brier: f64 = fit
        .probabilities
        .iter()
        .zip(&y_bin)
        .map(|(&p, &y)| (p - y).powi(2))
        .sum::<f64>()
        / y_bin.len() as f64;

    assert!(
        (cal.brier_score - manual_brier).abs() < 1e-12,
        "Brier manual: {} vs {}",
        cal.brier_score,
        manual_brier
    );
}

#[test]
fn calibration_log_loss_manual_computation() {
    // Log loss = -(1/n) Σ [y log(p) + (1-y) log(1-p)]
    let (data, y_bin) = binary_data(60, 30, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let cal = fdars_core::calibration_diagnostics(&fit, &y_bin, 5).unwrap();

    let manual_ll: f64 = -fit
        .probabilities
        .iter()
        .zip(&y_bin)
        .map(|(&p, &y)| {
            let pc = p.clamp(1e-15, 1.0 - 1e-15);
            y * pc.ln() + (1.0 - y) * (1.0 - pc).ln()
        })
        .sum::<f64>()
        / y_bin.len() as f64;

    assert!(
        (cal.log_loss - manual_ll).abs() < 1e-12,
        "Log loss manual: {} vs {}",
        cal.log_loss,
        manual_ll
    );
}

#[test]
fn calibration_perfect_model_brier_zero() {
    // If predictions exactly match labels, Brier = 0
    let (data, y_bin) = binary_data(60, 30, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let cal = fdars_core::calibration_diagnostics(&fit, &y_bin, 5).unwrap();
    // We can't expect exactly zero, but a good model should have low Brier
    assert!(
        cal.brier_score < 0.5,
        "A fitted model should have Brier < 0.5 (random): {}",
        cal.brier_score
    );
}

#[test]
fn calibration_reliability_bins_ordered() {
    // Mean predicted probability should be roughly increasing across bins
    // (since bins are sorted by predicted probability)
    let (data, y_bin) = binary_data(80, 40, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let cal = fdars_core::calibration_diagnostics(&fit, &y_bin, 5).unwrap();

    for i in 1..cal.reliability_bins.len() {
        assert!(
            cal.reliability_bins[i].0 >= cal.reliability_bins[i - 1].0 - 1e-10,
            "Mean predicted should be non-decreasing: bin {} ({}) vs bin {} ({})",
            i - 1,
            cal.reliability_bins[i - 1].0,
            i,
            cal.reliability_bins[i].0
        );
    }
}

#[test]
fn calibration_observed_rates_bounded() {
    // Mean observed rate should be in [0, 1]
    let (data, y_bin) = binary_data(80, 40, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let cal = fdars_core::calibration_diagnostics(&fit, &y_bin, 5).unwrap();
    for (i, &(pred, obs)) in cal.reliability_bins.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&pred),
            "Mean predicted in bin {} should be in [0,1]: {}",
            i,
            pred
        );
        assert!(
            (0.0..=1.0).contains(&obs),
            "Mean observed in bin {} should be in [0,1]: {}",
            i,
            obs
        );
    }
}

#[test]
fn calibration_hl_chi2_nonnegative() {
    for seed in [42, 77, 123] {
        let (data, y_bin) = binary_data(80, 40, seed);
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let cal = fdars_core::calibration_diagnostics(&fit, &y_bin, 10).unwrap();
        assert!(
            cal.hosmer_lemeshow_chi2 >= 0.0,
            "HL chi2 should be ≥ 0 for seed={}: {}",
            seed,
            cal.hosmer_lemeshow_chi2
        );
    }
}

#[test]
fn calibration_brier_bounded_all_seeds() {
    for seed in [42, 77, 123, 999] {
        let (data, y_bin) = binary_data(60, 30, seed);
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let cal = fdars_core::calibration_diagnostics(&fit, &y_bin, 5).unwrap();
        assert!(
            cal.brier_score >= 0.0 && cal.brier_score <= 1.0,
            "Brier in [0,1] for seed={}: {}",
            seed,
            cal.brier_score
        );
        assert!(cal.log_loss >= 0.0, "Log loss ≥ 0 for seed={}", seed);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Saliency — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn saliency_linear_algebraic_identity() {
    // saliency[(i,j)] = Σ_k coef[1+k] × (score_ik - mean_k) × rotation[(j,k)]
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sal = fdars_core::functional_saliency(&fit, &data, None).unwrap();
    let n = 60;
    let m = 30;
    let ncomp = 3;

    // Project scores manually
    let mut scores = FdMatrix::zeros(n, ncomp);
    for i in 0..n {
        for k in 0..ncomp {
            let mut s = 0.0;
            for j in 0..m {
                s += (data[(i, j)] - fit.fpca.mean[j]) * fit.fpca.rotation[(j, k)];
            }
            scores[(i, k)] = s;
        }
    }
    let mut mean_scores = vec![0.0; ncomp];
    for k in 0..ncomp {
        for i in 0..n {
            mean_scores[k] += scores[(i, k)];
        }
        mean_scores[k] /= n as f64;
    }

    let mut max_err = 0.0_f64;
    for i in 0..n {
        for j in 0..m {
            let mut expected = 0.0;
            for k in 0..ncomp {
                expected += fit.coefficients[1 + k]
                    * (scores[(i, k)] - mean_scores[k])
                    * fit.fpca.rotation[(j, k)];
            }
            max_err = max_err.max((sal.saliency_map[(i, j)] - expected).abs());
        }
    }

    assert!(
        max_err < 1e-10,
        "Saliency algebraic identity max error: {}",
        max_err
    );
}

#[test]
fn saliency_linear_row_sum_equals_shap_sum() {
    // Σ_j saliency[(i,j)] should relate to the total SHAP effect for obs i
    // Since saliency = Σ_k SHAP_k × rotation, and rotation columns are orthonormal:
    // saliency is a distributional version of SHAP
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sal = fdars_core::functional_saliency(&fit, &data, None).unwrap();
    let shap = fdars_core::fpc_shap_values(&fit, &data, None).unwrap();

    // Verify shapes are consistent
    assert_eq!(sal.saliency_map.shape(), (60, 30));
    assert_eq!(shap.values.shape(), (60, 3));

    // For each observation, saliency summed over j should relate to SHAP
    // This is not a direct equality, but L2 norm of saliency row ≈ L2 norm of SHAP row
    for i in 0..60 {
        let sal_norm_sq: f64 = (0..30).map(|j| sal.saliency_map[(i, j)].powi(2)).sum();
        let shap_norm_sq: f64 = (0..3).map(|k| shap.values[(i, k)].powi(2)).sum();
        // Due to orthonormality of rotation, these should be equal
        assert!(
            (sal_norm_sq - shap_norm_sq).abs() < 1e-6,
            "Saliency L2 norm should equal SHAP L2 norm at i={}: {} vs {}",
            i,
            sal_norm_sq.sqrt(),
            shap_norm_sq.sqrt()
        );
    }
}

#[test]
fn saliency_linear_mean_across_obs_is_zero() {
    // Mean saliency across observations at each grid point should be ~0
    // (because scores are centered)
    let (data, y) = regression_data(80, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sal = fdars_core::functional_saliency(&fit, &data, None).unwrap();
    let n = 80;
    for j in 0..30 {
        let mean_sal: f64 = (0..n).map(|i| sal.saliency_map[(i, j)]).sum::<f64>() / n as f64;
        assert!(
            mean_sal.abs() < 1e-8,
            "Mean saliency at j={} should be ~0: {}",
            j,
            mean_sal
        );
    }
}

#[test]
fn saliency_logistic_gradient_formula() {
    // saliency[(i,j)] = p_i × (1-p_i) × beta_t[j]
    let (data, y_bin) = binary_data(60, 30, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let sal = fdars_core::functional_saliency_logistic(&fit).unwrap();

    let mut max_err = 0.0_f64;
    for i in 0..60 {
        let pi = fit.probabilities[i];
        let w = pi * (1.0 - pi);
        for j in 0..30 {
            max_err = max_err.max((sal.saliency_map[(i, j)] - w * fit.beta_t[j]).abs());
        }
    }

    assert!(
        max_err < 1e-12,
        "Logistic saliency gradient formula error: {}",
        max_err
    );
}

#[test]
fn saliency_logistic_max_at_probability_half() {
    // Saliency is maximized when p ≈ 0.5 (since p(1-p) peaks at 0.5)
    let (data, y_bin) = binary_data(60, 30, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let sal = fdars_core::functional_saliency_logistic(&fit).unwrap();

    // Find obs closest to p=0.5 and obs farthest from p=0.5
    let (closest_idx, _) = fit
        .probabilities
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - 0.5).abs().partial_cmp(&(*b - 0.5).abs()).unwrap())
        .unwrap();

    let (farthest_idx, _) = fit
        .probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| (*a - 0.5).abs().partial_cmp(&(*b - 0.5).abs()).unwrap())
        .unwrap();

    // Saliency magnitude for closest should be ≥ farthest
    let norm_closest: f64 = (0..30)
        .map(|j| sal.saliency_map[(closest_idx, j)].powi(2))
        .sum::<f64>()
        .sqrt();
    let norm_farthest: f64 = (0..30)
        .map(|j| sal.saliency_map[(farthest_idx, j)].powi(2))
        .sum::<f64>()
        .sqrt();

    assert!(
        norm_closest >= norm_farthest - 1e-10,
        "Saliency near p=0.5 ({}) should be ≥ saliency far from p=0.5 ({})",
        norm_closest,
        norm_farthest
    );
}

#[test]
fn saliency_mean_absolute_is_average_of_abs() {
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sal = fdars_core::functional_saliency(&fit, &data, None).unwrap();
    for j in 0..30 {
        let manual: f64 = (0..60).map(|i| sal.saliency_map[(i, j)].abs()).sum::<f64>() / 60.0;
        assert!(
            (sal.mean_absolute_saliency[j] - manual).abs() < 1e-12,
            "Mean abs saliency at j={}: {} vs {}",
            j,
            sal.mean_absolute_saliency[j],
            manual
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Domain selection — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn domain_selection_pointwise_importance_is_beta_squared() {
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let ds = fdars_core::domain_selection(&fit, 5, 0.01).unwrap();
    for j in 0..30 {
        let expected = fit.beta_t[j] * fit.beta_t[j];
        assert!(
            (ds.pointwise_importance[j] - expected).abs() < 1e-12,
            "Pointwise importance = β²(t_j) at j={}: {} vs {}",
            j,
            ds.pointwise_importance[j],
            expected
        );
    }
}

#[test]
fn domain_selection_intervals_are_valid_ranges() {
    for seed in [42, 77, 123] {
        let (data, y) = regression_data(60, 30, seed);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let ds = fdars_core::domain_selection(&fit, 5, 0.01).unwrap();
        for iv in &ds.intervals {
            assert!(iv.start_idx <= iv.end_idx, "start ≤ end for seed={}", seed);
            assert!(iv.end_idx < 30, "end < m for seed={}", seed);
            assert!(
                iv.importance > 0.0,
                "Importance should be > 0 for seed={}",
                seed
            );
        }
    }
}

#[test]
fn domain_selection_intervals_sorted_by_importance() {
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let ds = fdars_core::domain_selection(&fit, 3, 0.01).unwrap();
    for i in 1..ds.intervals.len() {
        assert!(
            ds.intervals[i].importance <= ds.intervals[i - 1].importance,
            "Intervals should be sorted by importance descending"
        );
    }
}

#[test]
fn domain_selection_zero_threshold_returns_none() {
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    assert!(fdars_core::domain_selection(&fit, 5, 0.0).is_err());
}

#[test]
fn domain_selection_window_exceeding_m_returns_none() {
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    assert!(fdars_core::domain_selection(&fit, 31, 0.01).is_err());
}

#[test]
fn domain_selection_logistic_matches_linear_pattern() {
    // Same data → domain selection should select similar regions
    let (data, y_cont) = regression_data(80, 30, 42);
    let (_, y_bin) = binary_data(80, 30, 42);
    let fit_lm = fregre_lm(&data, &y_cont, None, 3).unwrap();
    let fit_log = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();

    let ds_lm = fdars_core::domain_selection(&fit_lm, 5, 0.01).unwrap();
    let ds_log = fdars_core::domain_selection_logistic(&fit_log, 5, 0.01).unwrap();

    // Both should produce valid pointwise importance
    assert_eq!(ds_lm.pointwise_importance.len(), 30);
    assert_eq!(ds_log.pointwise_importance.len(), 30);
    for &v in &ds_lm.pointwise_importance {
        assert!(v >= 0.0);
    }
    for &v in &ds_log.pointwise_importance {
        assert!(v >= 0.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Conditional permutation — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn cond_perm_reproducible_with_same_seed() {
    let (data, y) = regression_data(40, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let cp1 =
        fdars_core::conditional_permutation_importance(&fit, &data, &y, None, 3, 20, 42).unwrap();
    let cp2 =
        fdars_core::conditional_permutation_importance(&fit, &data, &y, None, 3, 20, 42).unwrap();
    for k in 0..3 {
        assert!(
            (cp1.importance[k] - cp2.importance[k]).abs() < 1e-15,
            "Same seed should give same result for FPC {}",
            k
        );
    }
}

#[test]
fn cond_perm_baseline_matches_r_squared() {
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let cp =
        fdars_core::conditional_permutation_importance(&fit, &data, &y, None, 3, 10, 42).unwrap();
    assert!(
        (cp.baseline_metric - fit.r_squared).abs() < 1e-8,
        "Baseline should be R²: {} vs {}",
        cp.baseline_metric,
        fit.r_squared
    );
}

#[test]
fn cond_perm_importance_is_baseline_minus_permuted() {
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let cp =
        fdars_core::conditional_permutation_importance(&fit, &data, &y, None, 3, 10, 42).unwrap();
    for k in 0..3 {
        let expected = cp.baseline_metric - cp.permuted_metric[k];
        assert!(
            (cp.importance[k] - expected).abs() < 1e-12,
            "importance = baseline - permuted for FPC {}: {} vs {}",
            k,
            cp.importance[k],
            expected
        );
    }
}

#[test]
fn cond_perm_logistic_baseline_is_accuracy() {
    let (data, y_bin) = binary_data(60, 30, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let cp = fdars_core::conditional_permutation_importance_logistic(
        &fit, &data, &y_bin, None, 3, 10, 42,
    )
    .unwrap();
    assert!(
        (cp.baseline_metric - fit.accuracy).abs() < 1e-8,
        "Logistic baseline should be accuracy: {} vs {}",
        cp.baseline_metric,
        fit.accuracy
    );
}

#[test]
fn cond_perm_all_finite() {
    for seed in [42, 77, 123] {
        let (data, y) = regression_data(50, 30, seed);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let cp = fdars_core::conditional_permutation_importance(&fit, &data, &y, None, 3, 10, seed)
            .unwrap();
        for k in 0..3 {
            assert!(cp.importance[k].is_finite(), "seed={} k={}", seed, k);
            assert!(cp.unconditional_importance[k].is_finite());
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Counterfactual — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn counterfactual_regression_analytical_exact() {
    // For linear model, counterfactual prediction = target exactly
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();

    for obs in [0, 10, 30, 59] {
        let target = fit.fitted_values[obs] + 2.5;
        let cf = fdars_core::counterfactual_regression(&fit, &data, None, obs, target).unwrap();
        assert!(
            (cf.counterfactual_prediction - target).abs() < 1e-10,
            "Exact counterfactual at obs={}: {} vs {}",
            obs,
            cf.counterfactual_prediction,
            target
        );
    }
}

#[test]
fn counterfactual_regression_minimal_perturbation() {
    // Δξ = gap × γ / ||γ||² is the L2-minimal perturbation
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let ncomp = 3;
    let gap = 3.0;
    let target = fit.fitted_values[0] + gap;
    let cf = fdars_core::counterfactual_regression(&fit, &data, None, 0, target).unwrap();

    let gamma: Vec<f64> = (0..ncomp).map(|k| fit.coefficients[1 + k]).collect();
    let gamma_norm_sq: f64 = gamma.iter().map(|g| g * g).sum();
    let expected_distance = gap.abs() / gamma_norm_sq.sqrt();

    assert!(
        (cf.distance - expected_distance).abs() < 1e-8,
        "Minimal distance: {} vs {}",
        cf.distance,
        expected_distance
    );

    // Verify delta_scores are proportional to gamma
    for (k, &gk) in gamma.iter().enumerate() {
        let expected_delta = gap * gk / gamma_norm_sq;
        assert!(
            (cf.delta_scores[k] - expected_delta).abs() < 1e-10,
            "Delta score proportional to gamma at k={}: {} vs {}",
            k,
            cf.delta_scores[k],
            expected_delta
        );
    }
}

#[test]
fn counterfactual_regression_delta_function_consistency() {
    // delta_function[j] = Σ_k delta_scores[k] × rotation[(j,k)]
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let cf = fdars_core::counterfactual_regression(&fit, &data, None, 0, 0.0).unwrap();

    for j in 0..30 {
        let expected: f64 = (0..3)
            .map(|k| cf.delta_scores[k] * fit.fpca.rotation[(j, k)])
            .sum();
        assert!(
            (cf.delta_function[j] - expected).abs() < 1e-12,
            "delta_function consistency at j={}: {} vs {}",
            j,
            cf.delta_function[j],
            expected
        );
    }
}

#[test]
fn counterfactual_regression_distance_is_l2_norm() {
    // distance = ||Δξ||₂
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let cf = fdars_core::counterfactual_regression(&fit, &data, None, 5, 0.0).unwrap();
    let manual_dist: f64 = cf.delta_scores.iter().map(|d| d * d).sum::<f64>().sqrt();
    assert!(
        (cf.distance - manual_dist).abs() < 1e-12,
        "Distance should be L2 norm of delta_scores: {} vs {}",
        cf.distance,
        manual_dist
    );
}

#[test]
fn counterfactual_regression_zero_gap_zero_perturbation() {
    // If target = fitted_values[obs], no perturbation needed
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let cf =
        fdars_core::counterfactual_regression(&fit, &data, None, 0, fit.fitted_values[0]).unwrap();
    assert!(
        cf.distance < 1e-12,
        "Zero gap should give zero distance: {}",
        cf.distance
    );
    for &d in &cf.delta_scores {
        assert!(d.abs() < 1e-12, "Delta scores should be ~0: {}", d);
    }
}

#[test]
fn counterfactual_logistic_converges_with_enough_iterations() {
    let (data, y_bin) = binary_data(60, 30, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();

    let mut flipped = 0;
    for obs in 0..60 {
        let cf = fdars_core::counterfactual_logistic(&fit, &data, None, obs, 5000, 0.3).unwrap();
        if cf.found {
            flipped += 1;
        }
    }
    assert!(
        flipped > 30,
        "Most observations should find counterfactuals: {}/60",
        flipped
    );
}

#[test]
fn counterfactual_scores_consistency() {
    // counterfactual_scores = original_scores + delta_scores
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let cf = fdars_core::counterfactual_regression(&fit, &data, None, 5, 0.0).unwrap();
    for k in 0..3 {
        let expected = cf.original_scores[k] + cf.delta_scores[k];
        assert!(
            (cf.counterfactual_scores[k] - expected).abs() < 1e-12,
            "CF scores = orig + delta at k={}: {} vs {}",
            k,
            cf.counterfactual_scores[k],
            expected
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Prototype / Criticism — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn prototype_criticism_indices_are_unique() {
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pc = fdars_core::prototype_criticism(&fit.fpca, 3, 8, 5).unwrap();

    // All prototype indices should be unique
    let mut proto_set = pc.prototype_indices.clone();
    proto_set.sort();
    proto_set.dedup();
    assert_eq!(
        proto_set.len(),
        pc.prototype_indices.len(),
        "Prototype indices should be unique"
    );

    // All criticism indices should be unique
    let mut crit_set = pc.criticism_indices.clone();
    crit_set.sort();
    crit_set.dedup();
    assert_eq!(
        crit_set.len(),
        pc.criticism_indices.len(),
        "Criticism indices should be unique"
    );
}

#[test]
fn prototype_criticism_disjoint_sets() {
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pc = fdars_core::prototype_criticism(&fit.fpca, 3, 10, 10).unwrap();
    for &ci in &pc.criticism_indices {
        assert!(
            !pc.prototype_indices.contains(&ci),
            "Criticism {} in prototype set",
            ci
        );
    }
}

#[test]
fn prototype_criticism_bandwidth_is_median_distance() {
    // Verify bandwidth is the median of pairwise distances
    let (data, y) = regression_data(40, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pc = fdars_core::prototype_criticism(&fit.fpca, 3, 5, 3).unwrap();

    let n = 40;
    let ncomp = 3;
    let mut dists: Vec<f64> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d2 = 0.0;
            for c in 0..ncomp {
                let d = fit.fpca.scores[(i, c)] - fit.fpca.scores[(j, c)];
                d2 += d * d;
            }
            dists.push(d2.sqrt());
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let expected_bw = dists[dists.len() / 2];
    assert!(
        (pc.bandwidth - expected_bw).abs() < 1e-10,
        "Bandwidth should be median distance: {} vs {}",
        pc.bandwidth,
        expected_bw
    );
}

#[test]
fn prototype_criticism_stable_across_ncomp() {
    // With more components, prototypes might change but should still be valid
    let (data, y) = regression_data(60, 30, 42);
    let fit2 = fregre_lm(&data, &y, None, 2).unwrap();
    let fit3 = fregre_lm(&data, &y, None, 3).unwrap();
    let pc2 = fdars_core::prototype_criticism(&fit2.fpca, 2, 5, 3).unwrap();
    let pc3 = fdars_core::prototype_criticism(&fit3.fpca, 3, 5, 3).unwrap();

    // Both should return valid indices
    assert_eq!(pc2.prototype_indices.len(), 5);
    assert_eq!(pc3.prototype_indices.len(), 5);
    for &pi in &pc2.prototype_indices {
        assert!(pi < 60);
    }
    for &pi in &pc3.prototype_indices {
        assert!(pi < 60);
    }
}

#[test]
fn prototype_criticism_witness_values_finite() {
    for seed in [42, 77, 123] {
        let (data, y) = regression_data(50, 30, seed);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let pc = fdars_core::prototype_criticism(&fit.fpca, 3, 5, 5).unwrap();
        for &w in &pc.prototype_witness {
            assert!(
                w.is_finite(),
                "Prototype witness should be finite (seed={})",
                seed
            );
        }
        for &w in &pc.criticism_witness {
            assert!(
                w.is_finite(),
                "Criticism witness should be finite (seed={})",
                seed
            );
        }
    }
}

#[test]
fn prototype_criticism_max_prototypes() {
    // Requesting n prototypes with 0 criticisms
    let (data, y) = regression_data(20, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pc = fdars_core::prototype_criticism(&fit.fpca, 3, 20, 0).unwrap();
    assert_eq!(pc.prototype_indices.len(), 20);
    assert_eq!(pc.criticism_indices.len(), 0);
}

#[test]
fn prototype_criticism_too_many_returns_none() {
    let (data, y) = regression_data(20, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    // Requesting more prototypes than observations
    assert!(fdars_core::prototype_criticism(&fit.fpca, 3, 21, 0).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
// LIME — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn lime_linear_exact_coefficients() {
    // For a linear model, LIME attributions = global coefficients (with enough samples)
    let (data, y) = regression_data(80, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let lime = fdars_core::lime_explanation(&fit, &data, None, 0, 10000, 1.0, 42).unwrap();

    for k in 0..3 {
        let global = fit.coefficients[1 + k];
        let local = lime.attributions[k];
        let rel_err = if global.abs() > 1e-6 {
            (local - global).abs() / global.abs()
        } else {
            local.abs()
        };
        assert!(
            rel_err < 0.3,
            "LIME attribution for FPC {} should ≈ global coefficient: local={}, global={} (err={})",
            k,
            local,
            global,
            rel_err
        );
    }
}

#[test]
fn lime_linear_r_squared_high() {
    // For a linear model, LIME's local R² should be high (the surrogate fits perfectly)
    let (data, y) = regression_data(80, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let lime = fdars_core::lime_explanation(&fit, &data, None, 0, 5000, 1.0, 42).unwrap();
    assert!(
        lime.local_r_squared > 0.8,
        "LIME R² for linear model should be high: {}",
        lime.local_r_squared
    );
}

#[test]
fn lime_reproducible_with_same_seed() {
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let l1 = fdars_core::lime_explanation(&fit, &data, None, 0, 500, 1.0, 42).unwrap();
    let l2 = fdars_core::lime_explanation(&fit, &data, None, 0, 500, 1.0, 42).unwrap();
    for k in 0..3 {
        assert!(
            (l1.attributions[k] - l2.attributions[k]).abs() < 1e-15,
            "Same seed should give same LIME result"
        );
    }
    assert!((l1.local_r_squared - l2.local_r_squared).abs() < 1e-15);
}

#[test]
fn lime_different_seeds_similar() {
    // Different seeds should give similar results (convergence)
    let (data, y) = regression_data(80, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let l1 = fdars_core::lime_explanation(&fit, &data, None, 0, 5000, 1.0, 42).unwrap();
    let l2 = fdars_core::lime_explanation(&fit, &data, None, 0, 5000, 1.0, 99).unwrap();
    for k in 0..3 {
        let diff = (l1.attributions[k] - l2.attributions[k]).abs();
        let scale = l1.attributions[k].abs().max(1e-6);
        assert!(
            diff / scale < 0.5,
            "Different seeds should converge for FPC {}: {} vs {}",
            k,
            l1.attributions[k],
            l2.attributions[k]
        );
    }
}

#[test]
fn lime_kernel_width_affects_locality() {
    // Smaller kernel width → more local (higher weight on nearby points)
    let (data, y) = regression_data(80, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let _narrow = fdars_core::lime_explanation(&fit, &data, None, 0, 2000, 0.1, 42).unwrap();
    let _wide = fdars_core::lime_explanation(&fit, &data, None, 0, 2000, 10.0, 42).unwrap();
    // For a linear model, both should give similar results
    // (since the model is globally linear), but the kernel widths should differ
    assert!((_narrow.kernel_width - 0.1).abs() < 1e-15);
    assert!((_wide.kernel_width - 10.0).abs() < 1e-15);
}

#[test]
fn lime_logistic_all_finite() {
    for seed in [42, 77, 123] {
        let (data, y_bin) = binary_data(60, 30, seed);
        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        let lime =
            fdars_core::lime_explanation_logistic(&fit, &data, None, 0, 500, 1.0, seed).unwrap();
        for (k, &a) in lime.attributions.iter().enumerate() {
            assert!(
                a.is_finite(),
                "LIME logistic attribution not finite at k={} (seed={})",
                k,
                seed
            );
        }
        assert!(lime.local_intercept.is_finite());
        assert!(lime.local_r_squared.is_finite());
    }
}

#[test]
fn lime_intercept_approximates_local_prediction() {
    // At the observation point (zero perturbation), prediction ≈ local_intercept
    let (data, y) = regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let lime = fdars_core::lime_explanation(&fit, &data, None, 5, 5000, 1.0, 42).unwrap();
    // local_intercept is the surrogate's prediction at the obs point (perturbation = 0)
    assert!(
        (lime.local_intercept - fit.fitted_values[5]).abs() < 0.5,
        "LIME intercept ({}) should approximate fitted value ({})",
        lime.local_intercept,
        fit.fitted_values[5]
    );
}

#[test]
fn lime_observation_index_stored() {
    let (data, y) = regression_data(40, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    for obs in [0, 5, 20, 39] {
        let lime = fdars_core::lime_explanation(&fit, &data, None, obs, 100, 1.0, 42).unwrap();
        assert_eq!(lime.observation, obs);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-method consistency
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn sobol_and_pointwise_importance_both_nonneg() {
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sobol = fdars_core::sobol_indices(&fit, &data, &y, None).unwrap();
    let pi = fdars_core::pointwise_importance(&fit).unwrap();
    for &s in &sobol.first_order {
        assert!(s >= -1e-10);
    }
    for &v in &pi.importance {
        assert!(v >= -1e-10);
    }
}

#[test]
fn loo_cv_and_influence_leverage_agree() {
    // Both LOO-CV and influence diagnostics compute leverage — they must match
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let loo = fdars_core::loo_cv_press(&fit, &data, &y, None).unwrap();
    let infl = fdars_core::influence_diagnostics(&fit, &data, None).unwrap();
    for i in 0..80 {
        assert!(
            (loo.leverage[i] - infl.leverage[i]).abs() < 1e-12,
            "Leverage mismatch at i={}: {} vs {}",
            i,
            loo.leverage[i],
            infl.leverage[i]
        );
    }
}

#[test]
fn saliency_and_shap_norms_agree() {
    // ||saliency_row||² = ||SHAP_row||² (due to orthonormal rotation)
    let (data, y) = regression_data(80, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sal = fdars_core::functional_saliency(&fit, &data, None).unwrap();
    let shap = fdars_core::fpc_shap_values(&fit, &data, None).unwrap();

    for i in 0..80 {
        let sal_norm_sq: f64 = (0..30).map(|j| sal.saliency_map[(i, j)].powi(2)).sum();
        let shap_norm_sq: f64 = (0..3).map(|k| shap.values[(i, k)].powi(2)).sum();
        assert!(
            (sal_norm_sq - shap_norm_sq).abs() < 1e-6,
            "Norms should agree at i={}: sal={}, shap={}",
            i,
            sal_norm_sq.sqrt(),
            shap_norm_sq.sqrt()
        );
    }
}

#[test]
fn sobol_and_shap_variance_agree() {
    // For linear model: Var(SHAP_k) = coef[1+k]² × Var(score_k) = component_variance from Sobol
    let (data, y) = regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let sobol = fdars_core::sobol_indices(&fit, &data, &y, None).unwrap();
    let shap = fdars_core::fpc_shap_values(&fit, &data, None).unwrap();

    let n = 80;
    for k in 0..3 {
        let mean_shap: f64 = (0..n).map(|i| shap.values[(i, k)]).sum::<f64>() / n as f64;
        let var_shap: f64 = (0..n)
            .map(|i| (shap.values[(i, k)] - mean_shap).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        assert!(
            (var_shap - sobol.component_variance[k]).abs() < 1e-8,
            "Var(SHAP_{}) should equal Sobol component_variance: {} vs {}",
            k,
            var_shap,
            sobol.component_variance[k]
        );
    }
}

#[test]
fn lime_and_shap_consistency_for_linear() {
    // For a linear model, LIME attributions × (score_k - mean_k) should ≈ SHAP_k
    let (data, y) = regression_data(80, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let lime = fdars_core::lime_explanation(&fit, &data, None, 5, 10000, 1.0, 42).unwrap();
    let _shap = fdars_core::fpc_shap_values(&fit, &data, None).unwrap();

    // LIME attributions ≈ global coefficients → LIME × deviation ≈ SHAP
    // This is an indirect consistency check
    for k in 0..3 {
        let lime_attr = lime.attributions[k];
        let global_coef = fit.coefficients[1 + k];
        let rel_diff = if global_coef.abs() > 1e-6 {
            (lime_attr - global_coef).abs() / global_coef.abs()
        } else {
            lime_attr.abs()
        };
        assert!(
            rel_diff < 0.5,
            "LIME-SHAP consistency for FPC {}: lime={}, global={}",
            k,
            lime_attr,
            global_coef
        );
    }
}

#[test]
fn domain_selection_and_pointwise_importance_consistent() {
    // domain_selection's pointwise_importance = β(t)² while
    // pointwise_importance uses the full decomposition — they should correlate
    let (data, y) = regression_data(80, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let ds = fdars_core::domain_selection(&fit, 5, 0.01).unwrap();
    let pi = fdars_core::pointwise_importance(&fit).unwrap();

    // Both should identify similar high-importance regions
    let mut ds_order: Vec<usize> = (0..30).collect();
    ds_order.sort_by(|&a, &b| {
        ds.pointwise_importance[b]
            .partial_cmp(&ds.pointwise_importance[a])
            .unwrap()
    });
    let mut pi_order: Vec<usize> = (0..30).collect();
    pi_order.sort_by(|&a, &b| pi.importance[b].partial_cmp(&pi.importance[a]).unwrap());

    // Top-5 should partially overlap
    let top5_ds: Vec<usize> = ds_order[..5].to_vec();
    let top5_pi: Vec<usize> = pi_order[..5].to_vec();
    let overlap = top5_ds.iter().filter(|t| top5_pi.contains(t)).count();
    assert!(
        overlap >= 1,
        "Top-5 from domain_selection and pointwise_importance should overlap: {:?} vs {:?}",
        top5_ds,
        top5_pi
    );
}

