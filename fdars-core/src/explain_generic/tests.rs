use super::*;
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
    let res = generic_stability(&data, &y, None, NCOMP, 10, SEED, TaskType::Regression).unwrap();
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
