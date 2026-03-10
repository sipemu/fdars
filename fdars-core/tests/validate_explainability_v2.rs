//! Integration tests for explainability round 2:
//! - FPC permutation importance
//! - Significant regions
//! - β(t) effect decomposition
//! - Influence diagnostics (Cook's D / leverage)
//! - Friedman H-statistic (FPC interaction)

use fdars_core::matrix::FdMatrix;
use std::f64::consts::PI;

// ─── Test data generators ────────────────────────────────────────────────────

fn generate_regression_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
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

fn generate_binary_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
    let (data, y_cont) = generate_regression_data(n, m, seed);
    let y_median = {
        let mut sorted = y_cont.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };
    let y_bin: Vec<f64> = y_cont
        .iter()
        .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
        .collect();
    (data, y_bin)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Feature 1: Beta decomposition
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn beta_decomposition_reconstructs_beta_t() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();
    let dec = fdars_core::beta_decomposition(&fit).unwrap();

    assert_eq!(dec.components.len(), 3);
    for j in 0..30 {
        let sum: f64 = dec.components.iter().map(|c| c[j]).sum();
        assert!(
            (sum - fit.beta_t[j]).abs() < 1e-10,
            "Sum of components should equal beta_t at j={}",
            j
        );
    }
}

#[test]
fn beta_decomposition_variance_proportions_valid() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();
    let dec = fdars_core::beta_decomposition(&fit).unwrap();

    let total: f64 = dec.variance_proportion.iter().sum();
    assert!((total - 1.0).abs() < 1e-10, "Proportions should sum to 1");
    for &p in &dec.variance_proportion {
        assert!(p >= 0.0, "Each proportion should be non-negative");
    }
}

#[test]
fn beta_decomposition_logistic_reconstructs() {
    let (data, y_bin) = generate_binary_data(50, 30, 42);
    let fit = fdars_core::functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let dec = fdars_core::beta_decomposition_logistic(&fit).unwrap();

    for j in 0..30 {
        let sum: f64 = dec.components.iter().map(|c| c[j]).sum();
        assert!(
            (sum - fit.beta_t[j]).abs() < 1e-10,
            "Logistic decomposition should reconstruct beta_t at j={}",
            j
        );
    }
}

#[test]
fn beta_decomposition_with_different_ncomp() {
    let (data, y) = generate_regression_data(60, 30, 42);
    for ncomp in [2, 3, 4] {
        if let Some(fit) = fdars_core::fregre_lm(&data, &y, None, ncomp) {
            let dec = fdars_core::beta_decomposition(&fit).unwrap();
            assert_eq!(dec.components.len(), ncomp);
            assert_eq!(dec.coefficients.len(), ncomp);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Feature 2: Significant regions
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn significant_regions_from_model_fit() {
    let (data, y) = generate_regression_data(50, 30, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();

    let regions = fdars_core::significant_regions_from_se(&fit.beta_t, &fit.beta_se, 1.96).unwrap();

    // Regions should be well-formed
    for r in &regions {
        assert!(r.start_idx <= r.end_idx);
        assert!(r.end_idx < 30);
    }
}

#[test]
fn significant_regions_consistency_with_ci() {
    // Build explicit CI and check consistency
    let (data, y) = generate_regression_data(50, 30, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();

    let z = 1.96;
    let lower: Vec<f64> = fit
        .beta_t
        .iter()
        .zip(&fit.beta_se)
        .map(|(b, s)| b - z * s)
        .collect();
    let upper: Vec<f64> = fit
        .beta_t
        .iter()
        .zip(&fit.beta_se)
        .map(|(b, s)| b + z * s)
        .collect();

    let regions_direct = fdars_core::significant_regions(&lower, &upper).unwrap();
    let regions_se = fdars_core::significant_regions_from_se(&fit.beta_t, &fit.beta_se, z).unwrap();

    assert_eq!(regions_direct.len(), regions_se.len());
    for (a, b) in regions_direct.iter().zip(regions_se.iter()) {
        assert_eq!(a.start_idx, b.start_idx);
        assert_eq!(a.end_idx, b.end_idx);
        assert_eq!(a.direction, b.direction);
    }
}

#[test]
fn significant_regions_wider_ci_finds_fewer_regions() {
    let (data, y) = generate_regression_data(50, 30, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();

    let regions_95 =
        fdars_core::significant_regions_from_se(&fit.beta_t, &fit.beta_se, 1.96).unwrap();
    let regions_99 =
        fdars_core::significant_regions_from_se(&fit.beta_t, &fit.beta_se, 2.576).unwrap();

    // Wider CI (99%) should find no more significant indices than narrower (95%)
    let sig_count_95: usize = regions_95.iter().map(|r| r.end_idx - r.start_idx + 1).sum();
    let sig_count_99: usize = regions_99.iter().map(|r| r.end_idx - r.start_idx + 1).sum();
    assert!(
        sig_count_99 <= sig_count_95,
        "99% CI should have ≤ significant indices than 95%: {} vs {}",
        sig_count_99,
        sig_count_95
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Feature 3: FPC permutation importance
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn permutation_importance_baseline_matches_r2() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();
    let imp = fdars_core::fpc_permutation_importance(&fit, &data, &y, 20, 42).unwrap();

    assert!(
        (imp.baseline_metric - fit.r_squared).abs() < 1e-6,
        "Baseline should match model R²: {} vs {}",
        imp.baseline_metric,
        fit.r_squared
    );
}

#[test]
fn permutation_importance_sum_bounded() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();
    let imp = fdars_core::fpc_permutation_importance(&fit, &data, &y, 50, 42).unwrap();

    let total_imp: f64 = imp.importance.iter().sum();
    // Total importance can exceed R² due to correlated components, but should be reasonable
    assert!(
        total_imp <= fit.r_squared * 3.0 + 0.5,
        "Total importance ({}) should not greatly exceed R² ({})",
        total_imp,
        fit.r_squared
    );
}

#[test]
fn permutation_importance_logistic_valid() {
    let (data, y_bin) = generate_binary_data(50, 30, 42);
    let fit = fdars_core::functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let imp = fdars_core::fpc_permutation_importance_logistic(&fit, &data, &y_bin, 20, 42).unwrap();

    assert_eq!(imp.importance.len(), 3);
    assert!(imp.baseline_metric >= 0.0 && imp.baseline_metric <= 1.0);
    for &pm in &imp.permuted_metric {
        assert!((0.0..=1.0).contains(&pm));
    }
}

#[test]
fn permutation_importance_different_seeds_differ() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();
    let imp1 = fdars_core::fpc_permutation_importance(&fit, &data, &y, 20, 42).unwrap();
    let imp2 = fdars_core::fpc_permutation_importance(&fit, &data, &y, 20, 99).unwrap();

    // Different seeds should (almost certainly) produce different results
    let any_differ = imp1
        .importance
        .iter()
        .zip(&imp2.importance)
        .any(|(a, b)| (a - b).abs() > 1e-15);
    assert!(
        any_differ,
        "Different seeds should produce different results"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Feature 4: Influence diagnostics
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn influence_diagnostics_basic_properties() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();
    let diag = fdars_core::influence_diagnostics(&fit, &data, None).unwrap();

    assert_eq!(diag.leverage.len(), 40);
    assert_eq!(diag.cooks_distance.len(), 40);
    assert_eq!(diag.p, 4); // 1 intercept + 3 FPCs
    assert!(diag.mse > 0.0);
}

#[test]
fn influence_leverage_hat_matrix_trace() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();
    let diag = fdars_core::influence_diagnostics(&fit, &data, None).unwrap();

    let h_sum: f64 = diag.leverage.iter().sum();
    assert!(
        (h_sum - diag.p as f64).abs() < 1e-6,
        "tr(H) should equal p={}: got {}",
        diag.p,
        h_sum
    );
}

#[test]
fn influence_outlier_detection() {
    let (mut data, mut y) = generate_regression_data(40, 30, 42);
    // Inject extreme outlier at index 5
    for j in 0..30 {
        data[(5, j)] = 50.0;
    }
    y[5] = 200.0;

    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();
    let diag = fdars_core::influence_diagnostics(&fit, &data, None).unwrap();

    // The outlier should have the highest Cook's distance
    let max_cd = diag
        .cooks_distance
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (diag.cooks_distance[5] - max_cd).abs() < 1e-10,
        "Outlier at index 5 should have max Cook's D"
    );
}

#[test]
fn influence_with_scalar_covariates() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let mut sc = FdMatrix::zeros(40, 2);
    for i in 0..40 {
        sc[(i, 0)] = i as f64 / 40.0;
        sc[(i, 1)] = (i as f64 * 0.3).sin();
    }
    let fit = fdars_core::fregre_lm(&data, &y, Some(&sc), 3).unwrap();
    let diag = fdars_core::influence_diagnostics(&fit, &data, Some(&sc)).unwrap();

    assert_eq!(diag.p, 6); // 1 + 3 + 2
    let h_sum: f64 = diag.leverage.iter().sum();
    assert!(
        (h_sum - 6.0).abs() < 1e-5,
        "tr(H) should equal 6: got {}",
        h_sum
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Feature 5: Friedman H-statistic
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn h_statistic_linear_no_interaction() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();

    for j in 0..3 {
        for k in (j + 1)..3 {
            let h = fdars_core::friedman_h_statistic(&fit, &data, j, k, 10).unwrap();
            assert!(
                h.h_squared.abs() < 1e-6,
                "Linear model H²({},{}) should be ~0: {}",
                j,
                k,
                h.h_squared
            );
        }
    }
}

#[test]
fn h_statistic_logistic_produces_result() {
    let (data, y_bin) = generate_binary_data(50, 30, 42);
    let fit = fdars_core::functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let h = fdars_core::friedman_h_statistic_logistic(&fit, &data, None, 0, 1, 10).unwrap();

    assert_eq!(h.component_j, 0);
    assert_eq!(h.component_k, 1);
    assert!(h.h_squared >= 0.0);
    assert_eq!(h.pdp_2d.shape(), (10, 10));
}

#[test]
fn h_statistic_symmetry_property() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();

    let h01 = fdars_core::friedman_h_statistic(&fit, &data, 0, 1, 10).unwrap();
    let h10 = fdars_core::friedman_h_statistic(&fit, &data, 1, 0, 10).unwrap();
    assert!(
        (h01.h_squared - h10.h_squared).abs() < 1e-10,
        "H(0,1) should equal H(1,0)"
    );
}

#[test]
fn h_statistic_rejects_same_component() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();
    assert!(fdars_core::friedman_h_statistic(&fit, &data, 0, 0, 10).is_none());
}

#[test]
fn h_statistic_rejects_invalid_component() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();
    assert!(fdars_core::friedman_h_statistic(&fit, &data, 0, 5, 10).is_none());
}
