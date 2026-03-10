//! Integration tests for Explainability Round 3:
//! Pointwise Importance, VIF, SHAP, DFBETAS/DFFITS, Prediction Intervals, ALE.

use fdars_core::matrix::FdMatrix;
use fdars_core::{
    dfbetas_dffits, fpc_ale, fpc_ale_logistic, fpc_shap_values, fpc_shap_values_logistic, fpc_vif,
    fpc_vif_logistic, fregre_lm, functional_logistic, pointwise_importance,
    pointwise_importance_logistic, prediction_intervals,
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
// Pointwise Importance
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_pointwise_importance_shape() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pi = pointwise_importance(&fit).unwrap();
    assert_eq!(pi.importance.len(), 50);
    assert_eq!(pi.importance_normalized.len(), 50);
    assert_eq!(pi.component_importance.shape(), (3, 50));
    assert_eq!(pi.score_variance.len(), 3);
}

#[test]
fn test_pointwise_importance_normalized_sums_to_one() {
    let (data, y) = generate_test_data(40, 50, 42);
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
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pi = pointwise_importance(&fit).unwrap();
    for (j, &v) in pi.importance.iter().enumerate() {
        assert!(v >= -1e-15, "Importance should be nonneg at j={}: {}", j, v);
    }
}

#[test]
fn test_pointwise_importance_component_sum_equals_total() {
    let (data, y) = generate_test_data(40, 50, 42);
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
    let (data, y_cont) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let pi = pointwise_importance_logistic(&fit).unwrap();
    assert_eq!(pi.importance.len(), 50);
    assert_eq!(pi.score_variance.len(), 3);
}

// ═══════════════════════════════════════════════════════════════════════════
// VIF
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_vif_orthogonal_fpcs_near_one() {
    let (data, y) = generate_test_data(60, 50, 42);
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
    let (data, y) = generate_test_data(60, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let vif = fpc_vif(&fit, &data, None).unwrap();
    for (k, &v) in vif.vif.iter().enumerate() {
        assert!(v >= 1.0 - 1e-6, "VIF should be ≥ 1 at k={}: {}", k, v);
    }
}

#[test]
fn test_vif_shape() {
    let (data, y) = generate_test_data(60, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let vif = fpc_vif(&fit, &data, None).unwrap();
    assert_eq!(vif.vif.len(), 3);
    assert_eq!(vif.labels.len(), 3);
}

#[test]
fn test_vif_labels_correct() {
    let (data, y) = generate_test_data(60, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let vif = fpc_vif(&fit, &data, None).unwrap();
    assert_eq!(vif.labels[0], "FPC_0");
    assert_eq!(vif.labels[1], "FPC_1");
    assert_eq!(vif.labels[2], "FPC_2");
}

#[test]
fn test_vif_logistic_agrees_with_linear() {
    let (data, y_cont) = generate_test_data(60, 50, 42);
    let y_bin = make_binary(&y_cont);
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
    let (data, y) = generate_test_data(60, 50, 42);
    let fit = fregre_lm(&data, &y, None, 1).unwrap();
    let vif = fpc_vif(&fit, &data, None).unwrap();
    assert_eq!(vif.vif.len(), 1);
    assert!(
        (vif.vif[0] - 1.0).abs() < 1e-6,
        "Single predictor VIF should be 1: {}",
        vif.vif[0]
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// SHAP
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_shap_linear_sum_to_fitted() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let shap = fpc_shap_values(&fit, &data, None).unwrap();
    for i in 0..40 {
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
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let shap = fpc_shap_values(&fit, &data, None).unwrap();
    assert_eq!(shap.values.shape(), (40, 3));
    assert_eq!(shap.mean_scores.len(), 3);
}

#[test]
fn test_shap_linear_sign_matches_coefficient() {
    let (data, y) = generate_test_data(60, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let shap = fpc_shap_values(&fit, &data, None).unwrap();
    for k in 0..3 {
        let coef_k = fit.coefficients[1 + k];
        if coef_k.abs() < 1e-10 {
            continue;
        }
        for i in 0..60 {
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
    let (data, y_cont) = generate_test_data(40, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let shap = fpc_shap_values_logistic(&fit, &data, None, 500, 42).unwrap();
    // Kernel SHAP is approximate — check correlation with predictions
    let mut shap_sums = Vec::new();
    for i in 0..40 {
        let sum: f64 = (0..3).map(|k| shap.values[(i, k)]).sum::<f64>() + shap.base_value;
        shap_sums.push(sum);
    }
    let mean_s: f64 = shap_sums.iter().sum::<f64>() / 40.0;
    let mean_p: f64 = fit.probabilities.iter().sum::<f64>() / 40.0;
    let mut cov = 0.0;
    let mut var_s = 0.0;
    let mut var_p = 0.0;
    for (i, &ss) in shap_sums.iter().enumerate() {
        let ds = ss - mean_s;
        let dp = fit.probabilities[i] - mean_p;
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
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let s1 = fpc_shap_values_logistic(&fit, &data, None, 100, 999).unwrap();
    let s2 = fpc_shap_values_logistic(&fit, &data, None, 100, 999).unwrap();
    for i in 0..30 {
        for k in 0..3 {
            assert!(
                (s1.values[(i, k)] - s2.values[(i, k)]).abs() < 1e-12,
                "Same seed should give same SHAP"
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

// ═══════════════════════════════════════════════════════════════════════════
// DFBETAS / DFFITS
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_dfbetas_shape() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let db = dfbetas_dffits(&fit, &data, None).unwrap();
    assert_eq!(db.dfbetas.shape(), (40, 4));
    assert_eq!(db.dffits.len(), 40);
    assert_eq!(db.studentized_residuals.len(), 40);
    assert_eq!(db.p, 4);
}

#[test]
fn test_dffits_sign_matches_residual() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let db = dfbetas_dffits(&fit, &data, None).unwrap();
    for i in 0..40 {
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
    let (mut data, mut y) = generate_test_data(40, 50, 42);
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
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let db = dfbetas_dffits(&fit, &data, None).unwrap();
    assert!(
        (db.dfbetas_cutoff - 2.0 / (40.0_f64).sqrt()).abs() < 1e-10,
        "DFBETAS cutoff should be 2/√n"
    );
    assert!(
        (db.dffits_cutoff - 2.0 * (4.0 / 40.0_f64).sqrt()).abs() < 1e-10,
        "DFFITS cutoff should be 2√(p/n)"
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
    let (data, y) = generate_test_data(50, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let db = dfbetas_dffits(&fit, &data, None).unwrap();
    let infl = fdars_core::influence_diagnostics(&fit, &data, None).unwrap();
    // Top influential observation should agree
    let mut dffits_order: Vec<usize> = (0..50).collect();
    dffits_order.sort_by(|&a, &b| db.dffits[b].abs().partial_cmp(&db.dffits[a].abs()).unwrap());
    let mut cooks_order: Vec<usize> = (0..50).collect();
    cooks_order.sort_by(|&a, &b| {
        infl.cooks_distance[b]
            .partial_cmp(&infl.cooks_distance[a])
            .unwrap()
    });
    assert_eq!(dffits_order[0], cooks_order[0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Prediction Intervals
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_prediction_interval_training_data_matches_fitted() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pi = prediction_intervals(&fit, &data, None, &data, None, 0.95).unwrap();
    for i in 0..40 {
        assert!(
            (pi.predictions[i] - fit.fitted_values[i]).abs() < 1e-6,
            "Prediction should match fitted at i={}",
            i
        );
    }
}

#[test]
fn test_prediction_interval_covers_training_y() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pi = prediction_intervals(&fit, &data, None, &data, None, 0.95).unwrap();
    let covered: usize = (0..40)
        .filter(|&i| y[i] >= pi.lower[i] && y[i] <= pi.upper[i])
        .count();
    assert!(
        covered >= 25,
        "Most training y should be covered: {}/40",
        covered
    );
}

#[test]
fn test_prediction_interval_symmetry() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pi = prediction_intervals(&fit, &data, None, &data, None, 0.95).unwrap();
    for i in 0..40 {
        let above = pi.upper[i] - pi.predictions[i];
        let below = pi.predictions[i] - pi.lower[i];
        assert!(
            (above - below).abs() < 1e-10,
            "Interval should be symmetric at i={}",
            i
        );
    }
}

#[test]
fn test_prediction_interval_wider_at_99_than_95() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pi95 = prediction_intervals(&fit, &data, None, &data, None, 0.95).unwrap();
    let pi99 = prediction_intervals(&fit, &data, None, &data, None, 0.99).unwrap();
    for i in 0..40 {
        let w95 = pi95.upper[i] - pi95.lower[i];
        let w99 = pi99.upper[i] - pi99.lower[i];
        assert!(
            w99 >= w95 - 1e-10,
            "99% should be wider at i={}: {} vs {}",
            i,
            w99,
            w95
        );
    }
}

#[test]
fn test_prediction_interval_shape() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pi = prediction_intervals(&fit, &data, None, &data, None, 0.95).unwrap();
    assert_eq!(pi.predictions.len(), 40);
    assert_eq!(pi.lower.len(), 40);
    assert_eq!(pi.upper.len(), 40);
    assert_eq!(pi.prediction_se.len(), 40);
    assert!((pi.confidence_level - 0.95).abs() < 1e-15);
}

#[test]
fn test_prediction_interval_invalid_confidence_returns_none() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    assert!(prediction_intervals(&fit, &data, None, &data, None, 0.0).is_none());
    assert!(prediction_intervals(&fit, &data, None, &data, None, 1.0).is_none());
}

// ═══════════════════════════════════════════════════════════════════════════
// ALE
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_ale_linear_is_linear() {
    let (data, y) = generate_test_data(60, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let ale = fpc_ale(&fit, &data, None, 0, 10).unwrap();
    // For linear model, ALE slope should be roughly constant
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
    let (data, y) = generate_test_data(60, 50, 42);
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
    let (data, y) = generate_test_data(60, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let ale = fpc_ale(&fit, &data, None, 0, 10).unwrap();
    let total: usize = ale.bin_counts.iter().sum();
    assert_eq!(total, 60, "Bin counts should sum to n");
}

#[test]
fn test_ale_shape() {
    let (data, y) = generate_test_data(60, 50, 42);
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
    let (data, y_cont) = generate_test_data(60, 50, 42);
    let y_bin = make_binary(&y_cont);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let ale = fpc_ale_logistic(&fit, &data, None, 0, 10).unwrap();
    for &v in &ale.ale_values {
        assert!(v.abs() < 2.0, "Logistic ALE should be bounded: {}", v);
    }
}

#[test]
fn test_ale_invalid_returns_none() {
    let (data, y) = generate_test_data(40, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    assert!(fpc_ale(&fit, &data, None, 5, 10).is_none());
    assert!(fpc_ale(&fit, &data, None, 0, 0).is_none());
}
