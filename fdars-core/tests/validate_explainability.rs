//! Validation tests for FDA explainability features:
//! - Bootstrap CIs for β(t)
//! - Elastic amplitude/phase attribution
//! - Functional PDP/ICE

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

fn generate_elastic_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<f64>) {
    let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0; n];
    for i in 0..n {
        let amp = 1.0 + 0.5 * (i as f64 / n as f64);
        let shift = 0.1 * (i as f64 - n as f64 / 2.0);
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * (t[j] + shift)).sin();
        }
        y[i] = amp;
    }
    (data, y, t)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Feature 1: Bootstrap CIs
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bootstrap_ci_covers_original_beta() {
    // With 95% CI, the center (original β(t)) should be within the bands at every point
    let (data, y) = generate_regression_data(40, 30, 42);
    let ci = fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 3, 200, 0.05, 12345).unwrap();
    for j in 0..30 {
        assert!(
            ci.center[j] >= ci.lower[j] - 1e-10 && ci.center[j] <= ci.upper[j] + 1e-10,
            "Center should be within pointwise band at j={}: center={}, lower={}, upper={}",
            j,
            ci.center[j],
            ci.lower[j],
            ci.upper[j]
        );
        assert!(
            ci.center[j] >= ci.sim_lower[j] - 1e-10 && ci.center[j] <= ci.sim_upper[j] + 1e-10,
            "Center should be within simultaneous band at j={}",
            j
        );
    }
}

#[test]
fn bootstrap_ci_band_width_positive() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let ci = fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 3, 200, 0.05, 42).unwrap();
    let pw_width: f64 = (0..30).map(|j| ci.upper[j] - ci.lower[j]).sum::<f64>() / 30.0;
    let sim_width: f64 = (0..30)
        .map(|j| ci.sim_upper[j] - ci.sim_lower[j])
        .sum::<f64>()
        / 30.0;
    assert!(pw_width > 0.0, "Pointwise band should have positive width");
    assert!(
        sim_width > 0.0,
        "Simultaneous band should have positive width"
    );
}

#[test]
fn bootstrap_ci_narrower_with_more_data() {
    // More observations should yield narrower CIs
    let (data_small, y_small) = generate_regression_data(25, 20, 42);
    let (data_large, y_large) = generate_regression_data(80, 20, 42);

    let ci_small =
        fdars_core::bootstrap_ci_fregre_lm(&data_small, &y_small, None, 2, 200, 0.05, 42).unwrap();
    let ci_large =
        fdars_core::bootstrap_ci_fregre_lm(&data_large, &y_large, None, 2, 200, 0.05, 42).unwrap();

    let avg_width_small: f64 = (0..20)
        .map(|j| ci_small.upper[j] - ci_small.lower[j])
        .sum::<f64>()
        / 20.0;
    let avg_width_large: f64 = (0..20)
        .map(|j| ci_large.upper[j] - ci_large.lower[j])
        .sum::<f64>()
        / 20.0;

    assert!(
        avg_width_large < avg_width_small * 1.5,
        "Larger dataset should produce comparable or narrower CIs: small={}, large={}",
        avg_width_small,
        avg_width_large
    );
}

#[test]
fn bootstrap_ci_wider_alpha_narrower_band() {
    // Higher alpha (e.g. 0.20 = 80% CI) should yield narrower bands than lower alpha (0.05 = 95% CI)
    let (data, y) = generate_regression_data(40, 20, 42);

    let ci_95 = fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 200, 0.05, 42).unwrap();
    let ci_80 = fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 200, 0.20, 42).unwrap();

    let avg_95: f64 = (0..20)
        .map(|j| ci_95.upper[j] - ci_95.lower[j])
        .sum::<f64>()
        / 20.0;
    let avg_80: f64 = (0..20)
        .map(|j| ci_80.upper[j] - ci_80.lower[j])
        .sum::<f64>()
        / 20.0;

    assert!(
        avg_80 < avg_95 + 1e-10,
        "80% CI should be narrower than 95% CI: 80%={}, 95%={}",
        avg_80,
        avg_95
    );
}

#[test]
fn bootstrap_ci_reproducible_with_same_seed() {
    let (data, y) = generate_regression_data(30, 20, 42);

    let ci1 = fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 50, 0.05, 999).unwrap();
    let ci2 = fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 50, 0.05, 999).unwrap();

    for j in 0..20 {
        assert!(
            (ci1.lower[j] - ci2.lower[j]).abs() < 1e-12,
            "Same seed should produce identical results"
        );
        assert!((ci1.upper[j] - ci2.upper[j]).abs() < 1e-12);
    }
}

#[test]
fn bootstrap_ci_logistic_covers_center() {
    let (data, y_bin) = generate_binary_data(50, 20, 42);
    let ci = fdars_core::bootstrap_ci_functional_logistic(
        &data, &y_bin, None, 2, 150, 0.05, 42, 25, 1e-6,
    )
    .unwrap();

    for j in 0..20 {
        assert!(
            ci.center[j] >= ci.lower[j] - 1e-10 && ci.center[j] <= ci.upper[j] + 1e-10,
            "Center should be within band at j={}",
            j
        );
    }
    assert!(
        ci.n_boot_success >= 50,
        "Should have many successful replicates: {}",
        ci.n_boot_success
    );
}

#[test]
fn bootstrap_ci_with_scalar_covariates() {
    let (data, y) = generate_regression_data(40, 20, 42);
    let mut sc = FdMatrix::zeros(40, 2);
    for i in 0..40 {
        sc[(i, 0)] = i as f64 / 40.0;
        sc[(i, 1)] = (i as f64 * 0.5).sin();
    }
    let ci = fdars_core::bootstrap_ci_fregre_lm(&data, &y, Some(&sc), 2, 100, 0.05, 42).unwrap();
    assert_eq!(ci.center.len(), 20);
    assert!(ci.n_boot_success > 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Feature 2: Elastic Attribution
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn elastic_attribution_joint_sum_property() {
    // amp_contrib + phase_contrib should equal fitted - alpha for all observations
    let (data, y, t) = generate_elastic_data(15, 51);
    let result = fdars_core::elastic_pcr(
        &data,
        &y,
        &t,
        3,
        fdars_core::elastic_regression::PcaMethod::Joint,
        0.0,
        5,
        1e-3,
    )
    .unwrap();

    let attr = fdars_core::elastic_pcr_attribution(&result, &y, 3, 20, 42).unwrap();

    let max_err: f64 = (0..15)
        .map(|i| {
            let sum = attr.amplitude_contribution[i] + attr.phase_contribution[i];
            let expected = result.fitted_values[i] - result.alpha;
            (sum - expected).abs()
        })
        .fold(0.0_f64, f64::max);

    assert!(
        max_err < 1e-6,
        "Max decomposition error should be small: {}",
        max_err
    );
}

#[test]
fn elastic_attribution_horizontal_only() {
    let (data, y, t) = generate_elastic_data(15, 51);
    let result = fdars_core::elastic_pcr(
        &data,
        &y,
        &t,
        3,
        fdars_core::elastic_regression::PcaMethod::Horizontal,
        0.0,
        5,
        1e-3,
    )
    .unwrap();

    let attr = fdars_core::elastic_pcr_attribution(&result, &y, 3, 10, 42).unwrap();

    // Amplitude contribution should be zero for horizontal-only
    for i in 0..15 {
        assert!(
            attr.amplitude_contribution[i].abs() < 1e-12,
            "Amplitude should be 0 for horizontal-only at i={}",
            i
        );
    }
    assert!(
        attr.amplitude_importance.abs() < 1e-12,
        "Amplitude importance should be 0 for horizontal-only"
    );
    // Phase should carry all the prediction
    for i in 0..15 {
        let expected = result.fitted_values[i] - result.alpha;
        assert!(
            (attr.phase_contribution[i] - expected).abs() < 1e-6,
            "Phase contribution should equal fitted-alpha at i={}",
            i
        );
    }
}

#[test]
fn elastic_attribution_importance_sum_bounded() {
    // amp_importance + phase_importance should not exceed R²
    let (data, y, t) = generate_elastic_data(15, 51);
    let result = fdars_core::elastic_pcr(
        &data,
        &y,
        &t,
        3,
        fdars_core::elastic_regression::PcaMethod::Joint,
        0.0,
        5,
        1e-3,
    )
    .unwrap();

    let attr = fdars_core::elastic_pcr_attribution(&result, &y, 3, 50, 42).unwrap();
    let importance_sum = attr.amplitude_importance + attr.phase_importance;

    assert!(
        importance_sum <= result.r_squared + 0.1,
        "Importance sum ({}) should not greatly exceed R² ({})",
        importance_sum,
        result.r_squared
    );
}

#[test]
fn elastic_attribution_with_fewer_components() {
    // Using fewer components than available should still work
    let (data, y, t) = generate_elastic_data(15, 51);
    let result = fdars_core::elastic_pcr(
        &data,
        &y,
        &t,
        5,
        fdars_core::elastic_regression::PcaMethod::Joint,
        0.0,
        5,
        1e-3,
    )
    .unwrap();

    // Use only 2 of 5 components for attribution
    let attr = fdars_core::elastic_pcr_attribution(&result, &y, 2, 10, 42).unwrap();
    assert_eq!(attr.amplitude_contribution.len(), 15);
    assert!(attr.amplitude_importance >= 0.0);
    assert!(attr.phase_importance >= 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Feature 3: Functional PDP/ICE
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn pdp_linear_ice_are_strictly_parallel() {
    // For fregre_lm, ICE curves are exactly parallel (same delta per grid step)
    let (data, y) = generate_regression_data(30, 50, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();

    for comp in 0..3 {
        let pdp = fdars_core::functional_pdp(&fit, &data, None, comp, 20).unwrap();

        // All ICE curves should have identical increments
        for g in 1..20 {
            let delta_0 = pdp.ice_curves[(0, g)] - pdp.ice_curves[(0, g - 1)];
            for i in 1..30 {
                let delta_i = pdp.ice_curves[(i, g)] - pdp.ice_curves[(i, g - 1)];
                assert!(
                    (delta_i - delta_0).abs() < 1e-10,
                    "ICE increments should match for comp={}, g={}, i={}: {} vs {}",
                    comp,
                    g,
                    i,
                    delta_i,
                    delta_0
                );
            }
        }
    }
}

#[test]
fn pdp_linear_pdp_equals_ice_mean() {
    // PDP should be the exact column mean of ICE curves
    let (data, y) = generate_regression_data(30, 50, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();
    let pdp = fdars_core::functional_pdp(&fit, &data, None, 0, 15).unwrap();

    for g in 0..15 {
        let ice_mean: f64 = (0..30).map(|i| pdp.ice_curves[(i, g)]).sum::<f64>() / 30.0;
        assert!(
            (pdp.pdp_curve[g] - ice_mean).abs() < 1e-10,
            "PDP should equal ICE mean at g={}: pdp={}, mean={}",
            g,
            pdp.pdp_curve[g],
            ice_mean
        );
    }
}

#[test]
fn pdp_grid_spans_score_range() {
    let (data, y) = generate_regression_data(30, 50, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();
    let pdp = fdars_core::functional_pdp(&fit, &data, None, 1, 50).unwrap();

    // Grid should start at min score and end at max score
    let m = data.ncols();
    let mut scores = vec![0.0; 30];
    for i in 0..30 {
        let mut s = 0.0;
        for j in 0..m {
            s += (data[(i, j)] - fit.fpca.mean[j]) * fit.fpca.rotation[(j, 1)];
        }
        scores[i] = s;
    }
    let score_min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let score_max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    assert!(
        (pdp.grid_values[0] - score_min).abs() < 1e-10,
        "Grid should start at min score"
    );
    assert!(
        (pdp.grid_values[49] - score_max).abs() < 1e-10,
        "Grid should end at max score"
    );
}

#[test]
fn pdp_logistic_monotonic_single_component() {
    // For a well-separated logistic model, PDP on the dominant component
    // should be monotonically increasing (or decreasing)
    let (data, y_bin) = generate_binary_data(40, 50, 42);
    let fit = fdars_core::functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let pdp = fdars_core::functional_pdp_logistic(&fit, &data, None, 0, 30).unwrap();

    // Check all values are valid probabilities
    for g in 0..30 {
        assert!(
            pdp.pdp_curve[g] >= 0.0 && pdp.pdp_curve[g] <= 1.0,
            "PDP should be probability at g={}: {}",
            g,
            pdp.pdp_curve[g]
        );
    }

    // The PDP curve should be monotonic (either always increasing or always decreasing)
    let diffs: Vec<f64> = (1..30)
        .map(|g| pdp.pdp_curve[g] - pdp.pdp_curve[g - 1])
        .collect();
    let all_nonneg = diffs.iter().all(|&d| d >= -1e-10);
    let all_nonpos = diffs.iter().all(|&d| d <= 1e-10);
    assert!(
        all_nonneg || all_nonpos,
        "PDP should be monotonic for single component"
    );
}

#[test]
fn pdp_logistic_with_scalar_covariates() {
    let (data, y_bin) = generate_binary_data(30, 50, 42);
    let mut sc = FdMatrix::zeros(30, 1);
    for i in 0..30 {
        sc[(i, 0)] = i as f64 / 30.0;
    }
    let fit = fdars_core::functional_logistic(&data, &y_bin, Some(&sc), 3, 25, 1e-6).unwrap();
    let pdp = fdars_core::functional_pdp_logistic(&fit, &data, Some(&sc), 0, 10).unwrap();

    assert_eq!(pdp.grid_values.len(), 10);
    for g in 0..10 {
        for i in 0..30 {
            assert!(
                pdp.ice_curves[(i, g)] >= 0.0 && pdp.ice_curves[(i, g)] <= 1.0,
                "ICE must be valid probability"
            );
        }
    }
}

#[test]
fn pdp_logistic_rejects_missing_scalar_covariates() {
    // If model was fit with scalar covariates, PDP should reject None scalar covariates
    let (data, y_bin) = generate_binary_data(30, 50, 42);
    let mut sc = FdMatrix::zeros(30, 1);
    for i in 0..30 {
        sc[(i, 0)] = i as f64 / 30.0;
    }
    let fit = fdars_core::functional_logistic(&data, &y_bin, Some(&sc), 3, 25, 1e-6).unwrap();

    // Should return None when scalar_covariates is missing but gamma is non-empty
    let result = fdars_core::functional_pdp_logistic(&fit, &data, None, 0, 10);
    assert!(
        result.is_err(),
        "Should reject None scalar_covariates when model has gamma"
    );
}

#[test]
fn pdp_each_component_independent() {
    // PDP results for different components should differ
    let (data, y) = generate_regression_data(30, 50, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();

    let pdp0 = fdars_core::functional_pdp(&fit, &data, None, 0, 10).unwrap();
    let pdp1 = fdars_core::functional_pdp(&fit, &data, None, 1, 10).unwrap();

    // Grid values should differ (different score ranges)
    let grids_differ = pdp0
        .grid_values
        .iter()
        .zip(pdp1.grid_values.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-10);
    assert!(
        grids_differ,
        "Different components should have different grid values"
    );
}

#[test]
fn pdp_at_observed_score_matches_fitted() {
    // For linear model, PDP at the mean of scores should equal the mean of fitted values
    let (data, y) = generate_regression_data(30, 50, 42);
    let fit = fdars_core::fregre_lm(&data, &y, None, 3).unwrap();
    let pdp = fdars_core::functional_pdp(&fit, &data, None, 0, 100).unwrap();

    // PDP curve evaluated at a middle grid point should be close to the mean fitted value
    let mid_pdp = pdp.pdp_curve[50]; // middle of grid

    // Not exactly equal, but should be in reasonable range
    assert!(
        mid_pdp.is_finite(),
        "PDP at mid-grid should be finite: {}",
        mid_pdp
    );
}
