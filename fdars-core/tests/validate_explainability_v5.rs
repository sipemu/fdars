//! Integration tests for Explainability Round 5:
//! ECE, Conformal Prediction, Regression Depth, Stability, Anchors

use fdars_core::matrix::FdMatrix;
use fdars_core::scalar_on_function::{fregre_lm, functional_logistic};
use fdars_core::{
    anchor_explanation, anchor_explanation_logistic, conformal_prediction_residuals,
    expected_calibration_error, explanation_stability, explanation_stability_logistic,
    regression_depth, regression_depth_logistic, DepthType,
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
    let med = sorted[sorted.len() / 2];
    y.iter()
        .map(|&v| if v >= med { 1.0 } else { 0.0 })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// ECE
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_ece_ace_range() {
    let (data, y) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let ece = expected_calibration_error(&fit, &y_bin, 10).unwrap();
    assert!(
        ece.ace >= 0.0 && ece.ace <= 1.0,
        "ACE out of range: {}",
        ece.ace
    );
}

#[test]
fn test_ece_different_bins() {
    let (data, y) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let ece5 = expected_calibration_error(&fit, &y_bin, 5).unwrap();
    let ece20 = expected_calibration_error(&fit, &y_bin, 20).unwrap();
    assert_eq!(ece5.n_bins, 5);
    assert_eq!(ece20.n_bins, 20);
    assert_eq!(ece5.bin_ece_contributions.len(), 5);
    assert_eq!(ece20.bin_ece_contributions.len(), 20);
}

// ═══════════════════════════════════════════════════════════════════════════
// Conformal Prediction
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_conformal_different_alpha() {
    let (data, y) = generate_test_data(60, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let cp_wide =
        conformal_prediction_residuals(&fit, &data, &y, &data, None, None, 0.3, 0.05, 42).unwrap();
    let cp_narrow =
        conformal_prediction_residuals(&fit, &data, &y, &data, None, None, 0.3, 0.2, 42).unwrap();
    // Wider intervals (smaller alpha) should have larger quantile
    assert!(
        cp_wide.residual_quantile >= cp_narrow.residual_quantile,
        "α=0.05 quantile {} should be ≥ α=0.2 quantile {}",
        cp_wide.residual_quantile,
        cp_narrow.residual_quantile
    );
}

#[test]
fn test_conformal_invalid_params() {
    let (data, y) = generate_test_data(60, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    // cal_fraction out of range
    assert!(
        conformal_prediction_residuals(&fit, &data, &y, &data, None, None, 0.0, 0.1, 42).is_err()
    );
    assert!(
        conformal_prediction_residuals(&fit, &data, &y, &data, None, None, 1.0, 0.1, 42).is_err()
    );
    // alpha out of range
    assert!(
        conformal_prediction_residuals(&fit, &data, &y, &data, None, None, 0.3, 0.0, 42).is_err()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Regression Depth
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_regression_depth_logistic_works() {
    let (data, y) = generate_test_data(30, 50, 42);
    let y_bin = make_binary(&y);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let rd = regression_depth_logistic(&fit, &data, &y_bin, None, 15, DepthType::FraimanMuniz, 42)
        .unwrap();
    assert_eq!(rd.score_depths.len(), 30);
    assert!(rd.beta_depth >= -1e-10);
    assert!(rd.n_boot_success > 0);
}

#[test]
fn test_regression_depth_mean_in_range() {
    let (data, y) = generate_test_data(30, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let rd = regression_depth(&fit, &data, &y, None, 20, DepthType::FraimanMuniz, 42).unwrap();
    assert!(
        rd.mean_score_depth >= 0.0 && rd.mean_score_depth <= 1.0,
        "Mean depth out of range: {}",
        rd.mean_score_depth
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Stability
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_stability_logistic_works() {
    let (data, y) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y);
    let sa = explanation_stability_logistic(&data, &y_bin, None, 3, 15, 42).unwrap();
    assert_eq!(sa.beta_t_std.len(), 50);
    assert_eq!(sa.coefficient_std.len(), 3);
    assert!(sa.n_boot_success > 0);
}

#[test]
fn test_stability_metric_std_nonneg() {
    let (data, y) = generate_test_data(30, 50, 42);
    let sa = explanation_stability(&data, &y, None, 3, 20, 42).unwrap();
    assert!(
        sa.metric_std >= 0.0,
        "Metric std should be ≥ 0: {}",
        sa.metric_std
    );
}

#[test]
fn test_stability_cv_length() {
    let (data, y) = generate_test_data(30, 50, 42);
    let sa = explanation_stability(&data, &y, None, 3, 20, 42).unwrap();
    assert_eq!(sa.beta_t_cv.len(), 50);
}

// ═══════════════════════════════════════════════════════════════════════════
// Anchors
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_anchor_logistic_works() {
    let (data, y) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let ar = anchor_explanation_logistic(&fit, &data, None, 0, 0.8, 5).unwrap();
    assert_eq!(ar.observation, 0);
    assert!(ar.rule.coverage > 0.0 && ar.rule.coverage <= 1.0);
    assert!(ar.rule.n_matching > 0);
}

#[test]
fn test_anchor_conditions_valid() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let ar = anchor_explanation(&fit, &data, None, 0, 0.8, 5).unwrap();
    for cond in &ar.rule.conditions {
        assert!(
            cond.component < 3,
            "Component {} out of range",
            cond.component
        );
        assert!(
            cond.lower_bound <= cond.upper_bound,
            "Lower > upper: {} > {}",
            cond.lower_bound,
            cond.upper_bound
        );
    }
}

#[test]
fn test_anchor_high_precision_threshold() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let ar = anchor_explanation(&fit, &data, None, 0, 0.99, 10).unwrap();
    // With enough bins, should still produce a valid rule
    assert!(!ar.rule.conditions.is_empty());
}
