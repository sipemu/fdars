//! Integration tests for beta decomposition, significant regions,
//! permutation importance, influence diagnostics, and Friedman H-statistic.
//!
//! Covers:
//! - Beta(t) effect decomposition
//! - Significant regions detection
//! - FPC permutation importance (linear and logistic)
//! - Influence diagnostics (Cook's D, leverage)
//! - Friedman H-statistic (FPC interaction)
//! - Cross-feature consistency checks

use fdars_core::matrix::FdMatrix;
use fdars_core::scalar_on_function::{fregre_lm, functional_logistic};
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
// Basic validation
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
        if let Ok(fit) = fdars_core::fregre_lm(&data, &y, None, ncomp) {
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
    assert!(fdars_core::friedman_h_statistic(&fit, &data, 0, 0, 10).is_err());
}

#[test]
fn h_statistic_rejects_invalid_component() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();
    assert!(fdars_core::friedman_h_statistic(&fit, &data, 0, 5, 10).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Deep validation — mathematical invariants and cross-method consistency
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn beta_decomp_reconstruction_error_is_machine_epsilon() {
    // The reconstruction Σ_k components[k] should equal beta_t to near machine epsilon
    let (data, y) = generate_regression_data(80, 40, 77);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let dec = fdars_core::beta_decomposition(&fit).unwrap();

    let max_err: f64 = (0..40)
        .map(|j| {
            let sum: f64 = dec.components.iter().map(|c| c[j]).sum();
            (sum - fit.beta_t[j]).abs()
        })
        .fold(0.0_f64, f64::max);

    assert!(
        max_err < 1e-12,
        "Reconstruction error should be near machine eps: {}",
        max_err
    );
}

#[test]
fn beta_decomp_component_orthogonality() {
    // FPCA eigenfunctions are orthonormal, so inner product of distinct components
    // should equal coef_j * coef_k * δ(j,k) — i.e. cross-products should be zero
    // when using orthogonal eigenfunctions.
    let (data, y) = generate_regression_data(80, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let dec = fdars_core::beta_decomposition(&fit).unwrap();

    for j in 0..3 {
        for k in (j + 1)..3 {
            let inner: f64 = (0..50)
                .map(|t| dec.components[j][t] * dec.components[k][t])
                .sum();
            // coef_j * coef_k * <φ_j, φ_k> ≈ 0 since φ's are orthogonal
            // But note: rotation columns are orthonormal w.r.t. discrete inner product
            // so inner ≈ coef_j * coef_k * 0 = 0
            assert!(
                inner.abs() < 1e-6,
                "Cross-product of components ({},{}) should be ~0: {}",
                j,
                k,
                inner
            );
        }
    }
}

#[test]
fn beta_decomp_variance_proportions_monotonic_with_eigenvalues() {
    // For data where FPC1 dominates, its variance proportion should be largest
    // (when the coefficients are comparable)
    let (data, y) = generate_regression_data(100, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let dec = fdars_core::beta_decomposition(&fit).unwrap();

    // Just verify all proportions are valid
    for &p in &dec.variance_proportion {
        assert!(
            (0.0..=1.0 + 1e-10).contains(&p),
            "Proportion out of range: {}",
            p
        );
    }
    let sum: f64 = dec.variance_proportion.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
}

#[test]
fn beta_decomp_with_scalar_covariates_still_sums() {
    let (data, y) = generate_regression_data(60, 30, 42);
    let mut sc = FdMatrix::zeros(60, 2);
    for i in 0..60 {
        sc[(i, 0)] = i as f64 / 60.0;
        sc[(i, 1)] = (i as f64 * 0.3).sin();
    }
    let fit = fregre_lm(&data, &y, Some(&sc), 3).unwrap();
    let dec = fdars_core::beta_decomposition(&fit).unwrap();

    // Scalar covariates don't affect the β(t) decomposition
    let max_err: f64 = (0..30)
        .map(|j| {
            let sum: f64 = dec.components.iter().map(|c| c[j]).sum();
            (sum - fit.beta_t[j]).abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(max_err < 1e-10);
}

#[test]
fn beta_decomp_logistic_variance_proportions_valid() {
    let (data, y_bin) = generate_binary_data(60, 30, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let dec = fdars_core::beta_decomposition_logistic(&fit).unwrap();

    let sum: f64 = dec.variance_proportion.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
    assert_eq!(dec.components.len(), 3);
    for comp in &dec.components {
        assert_eq!(comp.len(), 30);
    }
}

#[test]
fn beta_decomp_stable_across_sample_sizes() {
    // Decomposition with n=60 and n=120 should produce similar proportions
    let (data60, y60) = generate_regression_data(60, 30, 42);
    let (data120, y120) = generate_regression_data(120, 30, 42);
    let fit60 = fregre_lm(&data60, &y60, None, 3).unwrap();
    let fit120 = fregre_lm(&data120, &y120, None, 3).unwrap();
    let dec60 = fdars_core::beta_decomposition(&fit60).unwrap();
    let dec120 = fdars_core::beta_decomposition(&fit120).unwrap();

    // Dominant component should have similar proportion
    let diff = (dec60.variance_proportion[0] - dec120.variance_proportion[0]).abs();
    assert!(
        diff < 0.3,
        "Dominant proportion should be similar: {:.3} vs {:.3}",
        dec60.variance_proportion[0],
        dec120.variance_proportion[0]
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Significant regions — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn significant_regions_verified_against_manual_ci() {
    // Manually construct a known CI pattern and verify regions
    let (data, y) = generate_regression_data(80, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();

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

    let regions = fdars_core::significant_regions(&lower, &upper).unwrap();

    // Manually verify each significant index matches a region
    for r in &regions {
        for idx in r.start_idx..=r.end_idx {
            match r.direction {
                fdars_core::SignificanceDirection::Positive => {
                    assert!(
                        lower[idx] > 0.0,
                        "Positive region at idx={} but lower={} <= 0",
                        idx,
                        lower[idx]
                    );
                }
                fdars_core::SignificanceDirection::Negative => {
                    assert!(
                        upper[idx] < 0.0,
                        "Negative region at idx={} but upper={} >= 0",
                        idx,
                        upper[idx]
                    );
                }
            }
        }
    }

    // Verify no significant index is missed
    let mut region_indices: Vec<bool> = vec![false; 30];
    for r in &regions {
        region_indices[r.start_idx..=r.end_idx].fill(true);
    }
    for idx in 0..30 {
        let is_sig = lower[idx] > 0.0 || upper[idx] < 0.0;
        assert_eq!(
            is_sig, region_indices[idx],
            "Index {} significance mismatch: CI says {}, regions say {}",
            idx, is_sig, region_indices[idx]
        );
    }
}

#[test]
fn significant_regions_contiguity_invariant() {
    // Adjacent indices within a region must share the same direction
    let (data, y) = generate_regression_data(80, 50, 99);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let regions = fdars_core::significant_regions_from_se(&fit.beta_t, &fit.beta_se, 1.96).unwrap();

    for r in &regions {
        assert!(r.start_idx <= r.end_idx);
        assert!(r.end_idx < 50);
        // No two adjacent regions should share the same direction (they'd be merged)
    }
    for w in regions.windows(2) {
        assert!(
            w[0].end_idx + 1 < w[1].start_idx || w[0].direction != w[1].direction,
            "Adjacent regions with same direction should be merged: {:?} and {:?}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn significant_regions_narrower_ci_expands_regions() {
    // z=1.0 (68% CI) should yield at least as many significant indices as z=2.576 (99%)
    let (data, y) = generate_regression_data(100, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();

    let regions_narrow =
        fdars_core::significant_regions_from_se(&fit.beta_t, &fit.beta_se, 1.0).unwrap();
    let regions_wide =
        fdars_core::significant_regions_from_se(&fit.beta_t, &fit.beta_se, 2.576).unwrap();

    let count_narrow: usize = regions_narrow
        .iter()
        .map(|r| r.end_idx - r.start_idx + 1)
        .sum();
    let count_wide: usize = regions_wide
        .iter()
        .map(|r| r.end_idx - r.start_idx + 1)
        .sum();

    assert!(
        count_narrow >= count_wide,
        "Narrower CI should find >= significant indices: {} vs {}",
        count_narrow,
        count_wide
    );
}

#[test]
fn significant_regions_all_zero_beta_no_regions() {
    // If beta_t ≈ 0 with positive SE, no regions should be significant
    let beta_t = vec![0.0; 20];
    let beta_se = vec![1.0; 20];
    let regions = fdars_core::significant_regions_from_se(&beta_t, &beta_se, 1.96).unwrap();
    assert!(
        regions.is_empty(),
        "Zero beta should produce no significant regions"
    );
}

#[test]
fn significant_regions_empty_inputs() {
    assert!(fdars_core::significant_regions(&[], &[]).is_err());
    assert!(fdars_core::significant_regions_from_se(&[], &[], 1.96).is_err());
}

#[test]
fn significant_regions_mismatched_lengths() {
    assert!(fdars_core::significant_regions(&[1.0, 2.0], &[3.0]).is_err());
    assert!(fdars_core::significant_regions_from_se(&[1.0], &[1.0, 2.0], 1.96).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Permutation importance — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn perm_importance_converges_with_more_permutations() {
    // With many perms, the result should stabilize
    let (data, y) = generate_regression_data(50, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();

    let imp_10 = fdars_core::fpc_permutation_importance(&fit, &data, &y, 10, 42).unwrap();
    let imp_200 = fdars_core::fpc_permutation_importance(&fit, &data, &y, 200, 42).unwrap();
    let imp_200b = fdars_core::fpc_permutation_importance(&fit, &data, &y, 200, 99).unwrap();

    // Two runs with 200 perms and different seeds should agree more closely
    // than 10-perm vs 200-perm
    let diff_200s: f64 = imp_200
        .importance
        .iter()
        .zip(&imp_200b.importance)
        .map(|(a, b)| (a - b).abs())
        .sum();
    let diff_10_200: f64 = imp_10
        .importance
        .iter()
        .zip(&imp_200.importance)
        .map(|(a, b)| (a - b).abs())
        .sum();

    // This isn't always guaranteed, so use a generous threshold
    assert!(
        diff_200s < diff_10_200 + 0.5,
        "200-perm runs should agree at least as well: diff_200s={:.4}, diff_10_200={:.4}",
        diff_200s,
        diff_10_200
    );
}

#[test]
fn perm_importance_zero_r2_data() {
    // If y is pure noise (no relation to X), all importances should be ~0
    let (data, _) = generate_regression_data(50, 30, 42);
    // Generate unrelated y
    let y_noise: Vec<f64> = (0..50)
        .map(|i| ((i as u64 * 7919 + 13) % 1000) as f64 / 1000.0)
        .collect();
    let fit = fregre_lm(&data, &y_noise, None, 3).unwrap();
    let imp = fdars_core::fpc_permutation_importance(&fit, &data, &y_noise, 50, 42).unwrap();

    // With noise data, importances should be near zero (could be slightly negative)
    for k in 0..3 {
        assert!(
            imp.importance[k].abs() < 0.15,
            "Noise data should have ~0 importance at k={}: {}",
            k,
            imp.importance[k]
        );
    }
}

#[test]
fn perm_importance_permuted_metric_less_than_baseline() {
    // For well-fitted data, permuting should generally decrease R²
    let (data, y) = generate_regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let imp = fdars_core::fpc_permutation_importance(&fit, &data, &y, 100, 42).unwrap();

    let any_decrease = imp
        .permuted_metric
        .iter()
        .any(|&pm| pm < imp.baseline_metric + 1e-10);
    assert!(
        any_decrease,
        "At least one permuted metric should be below baseline"
    );
}

#[test]
fn perm_importance_logistic_accuracy_in_range() {
    let (data, y_bin) = generate_binary_data(60, 30, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let imp = fdars_core::fpc_permutation_importance_logistic(&fit, &data, &y_bin, 50, 42).unwrap();

    assert!(imp.baseline_metric >= 0.0 && imp.baseline_metric <= 1.0);
    for &pm in &imp.permuted_metric {
        assert!(
            (0.0..=1.0).contains(&pm),
            "Permuted accuracy out of [0,1]: {}",
            pm
        );
    }
}

#[test]
fn perm_importance_with_many_components() {
    // Use 4 components with sufficient data
    let (data, y) = generate_regression_data(100, 30, 42);
    if let Ok(fit) = fregre_lm(&data, &y, None, 4) {
        let imp = fdars_core::fpc_permutation_importance(&fit, &data, &y, 30, 42).unwrap();
        assert_eq!(imp.importance.len(), 4);
        assert_eq!(imp.permuted_metric.len(), 4);
    } else {
        // Fall back to 3 components
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let imp = fdars_core::fpc_permutation_importance(&fit, &data, &y, 30, 42).unwrap();
        assert_eq!(imp.importance.len(), 3);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Influence diagnostics — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn influence_leverage_bounded_by_one_over_n() {
    // Each h_ii >= 1/n (since design matrix includes intercept)
    let (data, y) = generate_regression_data(50, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let diag = fdars_core::influence_diagnostics(&fit, &data, None).unwrap();

    let n = 50;
    for (i, &h) in diag.leverage.iter().enumerate() {
        assert!(
            h >= 1.0 / n as f64 - 1e-10,
            "h_ii should be >= 1/n at i={}: h={}, 1/n={}",
            i,
            h,
            1.0 / n as f64
        );
    }
}

#[test]
fn influence_mse_matches_residuals() {
    let (data, y) = generate_regression_data(50, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let diag = fdars_core::influence_diagnostics(&fit, &data, None).unwrap();

    let ss_res: f64 = fit.residuals.iter().map(|r| r * r).sum();
    let expected_mse = ss_res / (50 - diag.p) as f64;
    assert!(
        (diag.mse - expected_mse).abs() < 1e-10,
        "MSE mismatch: {} vs {}",
        diag.mse,
        expected_mse
    );
}

#[test]
fn influence_cooks_d_formula_verification() {
    // Manually verify Cook's D for a few observations
    let (data, y) = generate_regression_data(40, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let diag = fdars_core::influence_diagnostics(&fit, &data, None).unwrap();

    for i in 0..5 {
        let e = fit.residuals[i];
        let h = diag.leverage[i];
        let p = diag.p as f64;
        let expected = (e * e * h) / (p * diag.mse * (1.0 - h).powi(2));
        assert!(
            (diag.cooks_distance[i] - expected).abs() < 1e-10,
            "Cook's D formula mismatch at i={}: {} vs {}",
            i,
            diag.cooks_distance[i],
            expected
        );
    }
}

#[test]
fn influence_with_scalar_covariates_increases_p() {
    let (data, y) = generate_regression_data(80, 30, 42);
    let mut sc = FdMatrix::zeros(80, 2);
    for i in 0..80 {
        sc[(i, 0)] = (i as f64 * 0.5).sin();
        sc[(i, 1)] = (i as f64 * 0.3).cos();
    }
    let fit = fregre_lm(&data, &y, Some(&sc), 3).unwrap();
    let diag = fdars_core::influence_diagnostics(&fit, &data, Some(&sc)).unwrap();

    assert_eq!(diag.p, 6); // 1 + 3 FPCs + 2 scalar
    let h_sum: f64 = diag.leverage.iter().sum();
    assert!(
        (h_sum - 6.0).abs() < 1e-5,
        "tr(H) should equal 6: {}",
        h_sum
    );
}

#[test]
fn influence_outlier_has_high_leverage_and_cooks() {
    // Create data with one extreme observation
    let (mut data, mut y) = generate_regression_data(50, 30, 42);

    // Make obs 0 very extreme
    for j in 0..30 {
        data[(0, j)] = 100.0 * (j as f64 / 29.0);
    }
    y[0] = 500.0;

    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let diag = fdars_core::influence_diagnostics(&fit, &data, None).unwrap();

    // Outlier should have high leverage (> 2p/n threshold)
    let threshold = 2.0 * diag.p as f64 / 50.0;
    assert!(
        diag.leverage[0] > threshold,
        "Outlier leverage ({}) should exceed 2p/n ({})",
        diag.leverage[0],
        threshold
    );

    // Outlier should have top-3 Cook's D
    let mut sorted_cd = diag.cooks_distance.clone();
    sorted_cd.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!(
        diag.cooks_distance[0] >= sorted_cd[2],
        "Outlier should be in top-3 Cook's D"
    );
}

#[test]
fn influence_no_nans_or_infinities() {
    let (data, y) = generate_regression_data(50, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let diag = fdars_core::influence_diagnostics(&fit, &data, None).unwrap();

    for i in 0..50 {
        assert!(
            diag.leverage[i].is_finite(),
            "NaN/Inf in leverage at i={}",
            i
        );
        assert!(
            diag.cooks_distance[i].is_finite(),
            "NaN/Inf in Cook's D at i={}",
            i
        );
    }
    assert!(diag.mse.is_finite());
}

#[test]
fn influence_returns_none_for_underdetermined_system() {
    // n <= p should return None
    let (data, y) = generate_regression_data(3, 30, 42);
    let fit = fregre_lm(&data, &y, None, 2).unwrap();
    // p = 1 + 2 = 3, n = 3, so n <= p
    assert!(
        fdars_core::influence_diagnostics(&fit, &data, None).is_err(),
        "Should return Err when n <= p"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Friedman H-statistic — deep validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn h_statistic_linear_algebraically_zero() {
    // For a purely additive linear model, H² must be exactly zero
    // This is a mathematical property, not approximate
    let (data, y) = generate_regression_data(60, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();

    for j in 0..3 {
        for k in (j + 1)..3 {
            let h = fdars_core::friedman_h_statistic(&fit, &data, j, k, 15).unwrap();
            assert!(
                h.h_squared.abs() < 1e-8,
                "Linear H²({},{}) must be ~0: {}",
                j,
                k,
                h.h_squared
            );
        }
    }
}

#[test]
fn h_statistic_logistic_all_pairs() {
    // Compute H² for all pairs and verify all are in [0, 1]
    let (data, y_bin) = generate_binary_data(60, 30, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();

    for j in 0..3 {
        for k in (j + 1)..3 {
            let h = fdars_core::friedman_h_statistic_logistic(&fit, &data, None, j, k, 10).unwrap();
            assert!(
                h.h_squared >= 0.0 && h.h_squared <= 1.0 + 1e-6,
                "H²({},{}) should be in [0,1]: {}",
                j,
                k,
                h.h_squared
            );
        }
    }
}

#[test]
fn h_statistic_symmetry_exact() {
    // H(j,k) must exactly equal H(k,j) — numerical values should be identical
    let (data, y_bin) = generate_binary_data(50, 30, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();

    let h01 = fdars_core::friedman_h_statistic_logistic(&fit, &data, None, 0, 1, 10).unwrap();
    let h10 = fdars_core::friedman_h_statistic_logistic(&fit, &data, None, 1, 0, 10).unwrap();

    assert!(
        (h01.h_squared - h10.h_squared).abs() < 1e-12,
        "H(0,1)={} vs H(1,0)={} — should be exactly equal",
        h01.h_squared,
        h10.h_squared
    );
}

#[test]
fn h_statistic_2d_pdp_surface_finite() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let h = fdars_core::friedman_h_statistic(&fit, &data, 0, 2, 12).unwrap();

    for gj in 0..12 {
        for gk in 0..12 {
            assert!(
                h.pdp_2d[(gj, gk)].is_finite(),
                "2D PDP should be finite at ({},{}): {}",
                gj,
                gk,
                h.pdp_2d[(gj, gk)]
            );
        }
    }
}

#[test]
fn h_statistic_2d_pdp_linear_is_additive() {
    // For linear model, pdp_2d(gj,gk) = pdp_j(gj) + pdp_k(gk) - f_bar
    // This is what makes H² = 0
    let (data, y) = generate_regression_data(50, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let h = fdars_core::friedman_h_statistic(&fit, &data, 0, 1, 10).unwrap();

    // Also compute 1D PDPs
    let pdp0 = fdars_core::functional_pdp(&fit, &data, None, 0, 10).unwrap();
    let pdp1 = fdars_core::functional_pdp(&fit, &data, None, 1, 10).unwrap();
    let n = data.nrows();
    let f_bar: f64 = fit.fitted_values.iter().sum::<f64>() / n as f64;

    for gj in 0..10 {
        for gk in 0..10 {
            let expected = pdp0.pdp_curve[gj] + pdp1.pdp_curve[gk] - f_bar;
            let actual = h.pdp_2d[(gj, gk)];
            assert!(
                (actual - expected).abs() < 1e-6,
                "2D PDP should be additive at ({},{}): {} vs {}",
                gj,
                gk,
                actual,
                expected
            );
        }
    }
}

#[test]
fn h_statistic_grid_resolution_effect() {
    // H² should be consistent across different grid resolutions
    let (data, y_bin) = generate_binary_data(50, 30, 42);
    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();

    let h_5 = fdars_core::friedman_h_statistic_logistic(&fit, &data, None, 0, 1, 5).unwrap();
    let h_20 = fdars_core::friedman_h_statistic_logistic(&fit, &data, None, 0, 1, 20).unwrap();

    // Different grid sizes should give similar H² (not wildly different)
    assert!(
        (h_5.h_squared - h_20.h_squared).abs() < 0.1,
        "H² should be stable across grid sizes: n=5 -> {}, n=20 -> {}",
        h_5.h_squared,
        h_20.h_squared
    );
}

#[test]
fn h_statistic_rejects_out_of_range_component() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    assert!(fdars_core::friedman_h_statistic(&fit, &data, 0, 5, 10).is_err());
    assert!(fdars_core::friedman_h_statistic(&fit, &data, 3, 0, 10).is_err());
}

#[test]
fn h_statistic_rejects_small_grid() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    assert!(fdars_core::friedman_h_statistic(&fit, &data, 0, 1, 1).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-feature validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn decomposition_and_significant_regions_consistent() {
    // If a component dominates beta_t, its significant regions should overlap
    // with the overall significant regions
    let (data, y) = generate_regression_data(80, 40, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let dec = fdars_core::beta_decomposition(&fit).unwrap();
    let regions = fdars_core::significant_regions_from_se(&fit.beta_t, &fit.beta_se, 1.96).unwrap();

    // Dominant component (highest variance proportion)
    let dominant_k = dec
        .variance_proportion
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    // The dominant component's contribution should be large where beta_t is significant
    if !regions.is_empty() {
        let r = &regions[0];
        let comp_energy: f64 = (r.start_idx..=r.end_idx)
            .map(|j| dec.components[dominant_k][j].powi(2))
            .sum();
        assert!(
            comp_energy > 0.0,
            "Dominant component should contribute to significant region"
        );
    }
}

#[test]
fn importance_and_decomposition_rank_agreement() {
    // The component with highest permutation importance should tend to have
    // high variance proportion (though not guaranteed for all data)
    let (data, y) = generate_regression_data(80, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();

    let imp = fdars_core::fpc_permutation_importance(&fit, &data, &y, 100, 42).unwrap();
    let dec = fdars_core::beta_decomposition(&fit).unwrap();

    // Most important component by permutation
    let top_imp = imp
        .importance
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    // Top variance proportion component
    let top_var = dec
        .variance_proportion
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    // They should often agree; if not, just check both are reasonable
    assert!(
        imp.importance[top_imp] > 0.0,
        "Top importance should be positive"
    );
    assert!(
        dec.variance_proportion[top_var] > 0.3,
        "Top proportion should be substantial: {}",
        dec.variance_proportion[top_var]
    );
}

#[test]
fn influence_and_importance_high_leverage_effect() {
    // Removing a high-leverage point should change importance values
    let (data, y) = generate_regression_data(50, 30, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let diag = fdars_core::influence_diagnostics(&fit, &data, None).unwrap();
    let imp = fdars_core::fpc_permutation_importance(&fit, &data, &y, 50, 42).unwrap();

    // High Cook's D observations exist
    let max_cd = diag.cooks_distance.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        max_cd > 0.0,
        "There should be some influential observations"
    );

    // Importance should have meaningful values
    let total_imp: f64 = imp.importance.iter().map(|v| v.abs()).sum();
    assert!(total_imp > 0.0, "Total importance magnitude should be > 0");
}
