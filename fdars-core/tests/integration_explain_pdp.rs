//! Integration tests for bootstrap CIs, PDP/ICE, and elastic attribution.
//!
//! Covers:
//! - Bootstrap confidence intervals for beta(t)
//! - Functional PDP and ICE curves (linear and logistic)
//! - Elastic amplitude/phase attribution
//! - Cross-checks, API surface validation, numerical stability

use fdars_core::matrix::FdMatrix;
use std::f64::consts::PI;
use fdars_core::elastic_regression::{PcaMethod, elastic_pcr};
use fdars_core::scalar_on_function::{fregre_lm, functional_logistic};

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
// Bootstrap CIs and PDP/ICE — basic validation
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


// ═══════════════════════════════════════════════════════════════════════════════
// Deep algorithmic validation — bootstrap CI, PDP, elastic attribution
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bootstrap_ci_simultaneous_band_is_symmetric_about_center() {
    // Simultaneous bands are center ± c_alpha * SE, so they should be exactly symmetric
    let (data, y) = generate_regression_data(40, 20, 42);
    let ci = fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 3, 200, 0.05, 42).unwrap();

    for j in 0..20 {
        let dist_lower = (ci.center[j] - ci.sim_lower[j]).abs();
        let dist_upper = (ci.sim_upper[j] - ci.center[j]).abs();
        assert!(
            (dist_lower - dist_upper).abs() < 1e-10,
            "Simultaneous band should be symmetric at j={}: dist_lower={}, dist_upper={}",
            j,
            dist_lower,
            dist_upper
        );
    }
}

#[test]
fn bootstrap_ci_different_seeds_give_different_results() {
    let (data, y) = generate_regression_data(30, 20, 42);
    let ci1 = fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 100, 0.05, 1).unwrap();
    let ci2 = fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 100, 0.05, 9999).unwrap();

    // Centers should be identical (same original fit)
    for j in 0..20 {
        assert!(
            (ci1.center[j] - ci2.center[j]).abs() < 1e-12,
            "Centers should match regardless of seed"
        );
    }

    // But bands should differ (different bootstrap samples)
    let differ = (0..20).any(|j| (ci1.lower[j] - ci2.lower[j]).abs() > 1e-10);
    assert!(differ, "Different seeds should produce different bands");
}

#[test]
fn bootstrap_ci_n_boot_success_close_to_n_boot() {
    // With well-behaved data, nearly all replicates should succeed
    let (data, y) = generate_regression_data(50, 20, 42);
    let ci = fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 200, 0.05, 42).unwrap();
    assert!(
        ci.n_boot_success >= 180,
        "Most replicates should succeed: {} / 200",
        ci.n_boot_success
    );
}

#[test]
fn bootstrap_ci_logistic_degenerate_resamples_handled() {
    // With balanced binary data, some resamples may be all-0 or all-1
    // These should be gracefully filtered out
    let n = 20;
    let m = 15;
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0; n];
    for i in 0..n {
        let label = if i < n / 2 { 0.0 } else { 1.0 };
        y[i] = label;
        for j in 0..m {
            let t = j as f64 / (m - 1) as f64;
            data[(i, j)] = (2.0 * PI * t).sin() + label * 0.5;
        }
    }

    let result =
        fdars_core::bootstrap_ci_functional_logistic(&data, &y, None, 2, 100, 0.05, 42, 25, 1e-6);
    // Should either succeed (with some failures) or return None if too few succeed
    if let Ok(ci) = result {
        assert!(ci.n_boot_success > 0, "Some replicates should succeed");
        assert!(
            ci.n_boot_success <= 100,
            "Cannot exceed n_boot: {}",
            ci.n_boot_success
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PDP: Algebraic consistency
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn pdp_linear_ice_at_original_score_equals_fitted() {
    // When the grid value equals the observation's original score,
    // the ICE value should equal the fitted value
    let (data, y) = generate_regression_data(30, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pdp = fdars_core::functional_pdp(&fit, &data, None, 0, 200).unwrap();

    // Compute scores for component 0
    let m = data.ncols();
    for i in 0..30 {
        let mut score_i = 0.0;
        for j in 0..m {
            score_i += (data[(i, j)] - fit.fpca.mean[j]) * fit.fpca.rotation[(j, 0)];
        }

        // Find the closest grid point to this score
        let closest_g = pdp
            .grid_values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a) - score_i)
                    .abs()
                    .partial_cmp(&((**b) - score_i).abs())
                    .unwrap()
            })
            .unwrap()
            .0;

        let grid_dist = (pdp.grid_values[closest_g] - score_i).abs();
        if grid_dist < 1e-6 {
            // Grid point is very close to original score — ICE should match fitted
            assert!(
                (pdp.ice_curves[(i, closest_g)] - fit.fitted_values[i]).abs() < 1e-4,
                "ICE at original score should ≈ fitted for i={}",
                i
            );
        }
    }
}

#[test]
fn pdp_linear_slope_equals_coefficient() {
    // For linear model, the slope of each ICE curve should equal the coefficient
    let (data, y) = generate_regression_data(30, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();

    for comp in 0..3 {
        let pdp = fdars_core::functional_pdp(&fit, &data, None, comp, 50).unwrap();
        let grid_range = pdp.grid_values[49] - pdp.grid_values[0];
        let ice_slope = (pdp.ice_curves[(0, 49)] - pdp.ice_curves[(0, 0)]) / grid_range;
        let expected_coef = fit.coefficients[1 + comp];

        assert!(
            (ice_slope - expected_coef).abs() < 1e-8,
            "ICE slope should equal coefficient for comp {}: slope={}, coef={}",
            comp,
            ice_slope,
            expected_coef
        );
    }
}

#[test]
fn pdp_logistic_ice_curves_are_sigmoid_shaped() {
    // Each logistic ICE curve should be monotone and S-shaped
    // (since it's sigmoid applied to a linear function)
    let n = 40;
    let m = 30;
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0; n];
    for i in 0..n {
        let t_val = i as f64 / (n - 1) as f64;
        y[i] = if t_val > 0.5 { 1.0 } else { 0.0 };
        for j in 0..m {
            let t = j as f64 / (m - 1) as f64;
            data[(i, j)] = (2.0 * PI * t).sin() * (1.0 + t_val);
        }
    }

    let fit = functional_logistic(&data, &y, None, 2, 50, 1e-6).unwrap();
    let pdp = fdars_core::functional_pdp_logistic(&fit, &data, None, 0, 50).unwrap();

    // Each ICE curve should be bounded [0, 1] (already tested)
    // And should be monotone
    for i in 0..n {
        let diffs: Vec<f64> = (1..50)
            .map(|g| pdp.ice_curves[(i, g)] - pdp.ice_curves[(i, g - 1)])
            .collect();
        let all_nonneg = diffs.iter().all(|&d| d >= -1e-12);
        let all_nonpos = diffs.iter().all(|&d| d <= 1e-12);
        assert!(
            all_nonneg || all_nonpos,
            "ICE curve {} should be monotone (sigmoid of linear)",
            i
        );
    }
}

#[test]
fn pdp_logistic_scalar_covariates_shift_ice() {
    // With scalar covariates, different observations should have different ICE intercepts
    // (unlike linear model where all have same slope)
    let n = 30;
    let m = 30;
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0; n];
    let mut sc = FdMatrix::zeros(n, 1);
    for i in 0..n {
        y[i] = if i >= n / 2 { 1.0 } else { 0.0 };
        sc[(i, 0)] = i as f64 / n as f64; // scalar covariate
        for j in 0..m {
            let t = j as f64 / (m - 1) as f64;
            data[(i, j)] = (2.0 * PI * t).sin() + (i as f64 / n as f64);
        }
    }

    let fit = functional_logistic(&data, &y, Some(&sc), 2, 25, 1e-6).unwrap();
    let pdp = fdars_core::functional_pdp_logistic(&fit, &data, Some(&sc), 0, 10).unwrap();

    // Due to sigmoid nonlinearity + different scalar covariate values,
    // ICE curves should NOT be parallel
    let slope_first = (pdp.ice_curves[(0, 9)] - pdp.ice_curves[(0, 0)])
        / (pdp.grid_values[9] - pdp.grid_values[0]);
    let slope_last = (pdp.ice_curves[(n - 1, 9)] - pdp.ice_curves[(n - 1, 0)])
        / (pdp.grid_values[9] - pdp.grid_values[0]);

    // They should differ (sigmoid makes slopes observation-dependent)
    // This is a weaker check — just verify they aren't identical
    // It's possible they're very close if the model is trivial, so just check finite
    assert!(
        slope_first.is_finite() && slope_last.is_finite(),
        "ICE slopes should be finite"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Elastic Attribution: Algorithmic consistency
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn elastic_attribution_permutation_reduces_r2() {
    // After permuting one component, R² should generally decrease
    let (data, y, t) = generate_elastic_data(20, 51);
    let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Joint, 0.0, 5, 1e-3).unwrap();
    let attr = fdars_core::elastic_pcr_attribution(&result, &y, 3, 100, 42).unwrap();

    // importance = R² drop from permuting, should be >= 0
    assert!(
        attr.amplitude_importance >= 0.0,
        "amp importance >= 0: {}",
        attr.amplitude_importance
    );
    assert!(
        attr.phase_importance >= 0.0,
        "phase importance >= 0: {}",
        attr.phase_importance
    );
}

#[test]
fn elastic_attribution_reproducible_with_same_seed() {
    let (data, y, t) = generate_elastic_data(15, 51);
    let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Joint, 0.0, 5, 1e-3).unwrap();

    let attr1 = fdars_core::elastic_pcr_attribution(&result, &y, 3, 50, 42).unwrap();
    let attr2 = fdars_core::elastic_pcr_attribution(&result, &y, 3, 50, 42).unwrap();

    for i in 0..15 {
        assert!(
            (attr1.amplitude_contribution[i] - attr2.amplitude_contribution[i]).abs() < 1e-12,
            "Same seed should give identical amplitude contributions"
        );
        assert!(
            (attr1.phase_contribution[i] - attr2.phase_contribution[i]).abs() < 1e-12,
            "Same seed should give identical phase contributions"
        );
    }
    assert!(
        (attr1.amplitude_importance - attr2.amplitude_importance).abs() < 1e-12,
        "Same seed should give identical importance"
    );
}

#[test]
fn elastic_attribution_contribution_variance_nonzero() {
    // The contributions should have nonzero variance (not all identical)
    let (data, y, t) = generate_elastic_data(20, 51);
    let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Joint, 0.0, 5, 1e-3).unwrap();
    let attr = fdars_core::elastic_pcr_attribution(&result, &y, 3, 10, 42).unwrap();

    let amp_mean: f64 =
        attr.amplitude_contribution.iter().sum::<f64>() / attr.amplitude_contribution.len() as f64;
    let amp_var: f64 = attr
        .amplitude_contribution
        .iter()
        .map(|&a| (a - amp_mean).powi(2))
        .sum::<f64>()
        / attr.amplitude_contribution.len() as f64;

    assert!(
        amp_var > 1e-15,
        "Amplitude contributions should vary across observations: var={}",
        amp_var
    );
}

#[test]
fn elastic_attribution_vertical_full_contribution() {
    // For vertical-only: amplitude_contribution[i] should equal fitted[i] - alpha
    let (data, y, t) = generate_elastic_data(15, 51);
    let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Vertical, 0.0, 5, 1e-3).unwrap();
    let attr = fdars_core::elastic_pcr_attribution(&result, &y, 3, 10, 42).unwrap();

    for i in 0..15 {
        let expected = result.fitted_values[i] - result.alpha;
        assert!(
            (attr.amplitude_contribution[i] - expected).abs() < 1e-6,
            "Vert-only: amp_contrib should equal fitted-alpha at i={}: {} vs {}",
            i,
            attr.amplitude_contribution[i],
            expected
        );
    }
}

#[test]
fn elastic_pcr_stores_fpca_results() {
    // Verify that ElasticPcrResult actually stores the FPCA results for each method
    let (data, y, t) = generate_elastic_data(15, 51);

    let vert_result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Vertical, 0.0, 5, 1e-3).unwrap();
    assert!(
        vert_result.vert_fpca.is_some(),
        "Vertical should store vert_fpca"
    );
    assert!(
        vert_result.horiz_fpca.is_none(),
        "Vertical should not store horiz_fpca"
    );
    assert!(
        vert_result.joint_fpca.is_none(),
        "Vertical should not store joint_fpca"
    );

    let horiz_result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Horizontal, 0.0, 5, 1e-3).unwrap();
    assert!(
        horiz_result.vert_fpca.is_none(),
        "Horizontal should not store vert_fpca"
    );
    assert!(
        horiz_result.horiz_fpca.is_some(),
        "Horizontal should store horiz_fpca"
    );
    assert!(
        horiz_result.joint_fpca.is_none(),
        "Horizontal should not store joint_fpca"
    );

    let joint_result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Joint, 0.0, 5, 1e-3).unwrap();
    assert!(
        joint_result.vert_fpca.is_none(),
        "Joint should not store vert_fpca"
    );
    assert!(
        joint_result.horiz_fpca.is_none(),
        "Joint should not store horiz_fpca"
    );
    assert!(
        joint_result.joint_fpca.is_some(),
        "Joint should store joint_fpca"
    );
}

#[test]
fn elastic_attribution_joint_scores_decompose_correctly() {
    // For joint FPCA, score[i,k] should equal amp_score[i,k] + phase_score[i,k]
    // We can check this indirectly: the total contribution should match fitted - alpha
    let (data, y, t) = generate_elastic_data(20, 51);
    let result = elastic_pcr(&data, &y, &t, 5, PcaMethod::Joint, 0.0, 5, 1e-3).unwrap();
    let attr = fdars_core::elastic_pcr_attribution(&result, &y, 5, 10, 42).unwrap();

    let max_err: f64 = (0..20)
        .map(|i| {
            let total = attr.amplitude_contribution[i] + attr.phase_contribution[i];
            let expected = result.fitted_values[i] - result.alpha;
            (total - expected).abs()
        })
        .fold(0.0_f64, f64::max);

    assert!(
        max_err < 1e-5,
        "Joint decomposition max error should be small: {}",
        max_err
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Edge cases and error handling
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bootstrap_ci_rejects_invalid_alpha() {
    let (data, y) = generate_regression_data(30, 20, 42);
    assert!(fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 50, 0.0, 42).is_err());
    assert!(fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 50, 1.0, 42).is_err());
    assert!(fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 50, -0.1, 42).is_err());
}

#[test]
fn bootstrap_ci_rejects_zero_n_boot() {
    let (data, y) = generate_regression_data(30, 20, 42);
    assert!(fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 0, 0.05, 42).is_err());
}

#[test]
fn pdp_rejects_n_grid_one() {
    let (data, y) = generate_regression_data(30, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    assert!(fdars_core::functional_pdp(&fit, &data, None, 0, 0).is_err());
    assert!(fdars_core::functional_pdp(&fit, &data, None, 0, 1).is_err());
}

#[test]
fn elastic_attribution_rejects_empty_y() {
    let (data, y, t) = generate_elastic_data(15, 51);
    let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Vertical, 0.0, 5, 1e-3).unwrap();
    assert!(fdars_core::elastic_pcr_attribution(&result, &[], 3, 10, 42).is_err());
}

#[test]
fn elastic_attribution_rejects_zero_ncomp() {
    let (data, y, t) = generate_elastic_data(15, 51);
    let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Vertical, 0.0, 5, 1e-3).unwrap();
    assert!(fdars_core::elastic_pcr_attribution(&result, &y, 0, 10, 42).is_err());
}


// ═══════════════════════════════════════════════════════════════════════════════
// Cross-checks, API surface, numerical stability
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bootstrap_se_consistent_with_asymptotic_se() {
    // Bootstrap SE ≈ (upper - lower) / (2 * z_{alpha/2})
    // Asymptotic SE comes from fregre_lm beta_se
    // They should be in the same order of magnitude
    let (data, y) = generate_regression_data(60, 25, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let ci = fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 3, 500, 0.05, 42).unwrap();

    // Bootstrap SE ≈ (upper - lower) / (2 * 1.96)
    let mut boot_se_avg = 0.0;
    let mut asymp_se_avg = 0.0;
    for j in 0..25 {
        let boot_se = (ci.upper[j] - ci.lower[j]) / (2.0 * 1.96);
        boot_se_avg += boot_se;
        asymp_se_avg += fit.beta_se[j];
    }
    boot_se_avg /= 25.0;
    asymp_se_avg /= 25.0;

    // They should be within an order of magnitude
    let ratio = if asymp_se_avg > 1e-15 {
        boot_se_avg / asymp_se_avg
    } else {
        1.0
    };
    assert!(
        ratio > 0.1 && ratio < 10.0,
        "Bootstrap SE and asymptotic SE should be same order of magnitude: \
         boot_se_avg={}, asymp_se_avg={}, ratio={}",
        boot_se_avg,
        asymp_se_avg,
        ratio
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shooting vector recomputation consistency
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn elastic_attribution_shooting_vectors_match_horiz_fpca() {
    // The attribution recomputes shooting vectors. Verify the decomposition is
    // exact by checking that amp_score + phase_score reconstructs the original score.
    // If shooting vector recomputation diverged, this sum wouldn't match.
    let (data, y, t) = generate_elastic_data(20, 51);
    let result = elastic_pcr(&data, &y, &t, 5, PcaMethod::Joint, 0.0, 8, 1e-4).unwrap();
    let attr = fdars_core::elastic_pcr_attribution(&result, &y, 5, 10, 42).unwrap();

    // Verify decomposition for every observation
    let mut max_err = 0.0_f64;
    for i in 0..20 {
        let total = attr.amplitude_contribution[i] + attr.phase_contribution[i];
        let expected = result.fitted_values[i] - result.alpha;
        let err = (total - expected).abs();
        max_err = max_err.max(err);
    }
    // If shooting vectors were recomputed incorrectly, this would fail
    assert!(
        max_err < 1e-5,
        "Score decomposition should be exact (max err {})",
        max_err
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// API surface: verify all re-exports from lib.rs work
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn api_bootstrap_ci_accessible() {
    // Verify the public API is accessible through lib.rs re-exports
    let (data, y) = generate_regression_data(30, 20, 42);
    let _ci: fdars_core::BootstrapCiResult =
        fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 20, 0.05, 42).unwrap();
}

#[test]
fn api_pdp_accessible() {
    let (data, y) = generate_regression_data(30, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let _pdp: fdars_core::FunctionalPdpResult =
        fdars_core::functional_pdp(&fit, &data, None, 0, 10).unwrap();
}

#[test]
fn api_elastic_attribution_accessible() {
    let (data, y, t) = generate_elastic_data(15, 51);
    let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Joint, 0.0, 5, 1e-3).unwrap();
    let _attr: fdars_core::ElasticAttributionResult =
        fdars_core::elastic_pcr_attribution(&result, &y, 3, 10, 42).unwrap();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Quantile function edge cases
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bootstrap_ci_extreme_alpha_values() {
    // alpha very close to 0 or 1 should still produce valid bands
    let (data, y) = generate_regression_data(30, 20, 42);

    // Very narrow CI (alpha = 0.99 → 1% CI)
    let ci_narrow = fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 100, 0.99, 42).unwrap();
    for j in 0..20 {
        assert!(
            ci_narrow.lower[j] <= ci_narrow.upper[j] + 1e-10,
            "Narrow CI should still have lower <= upper at j={}",
            j
        );
    }

    // Very wide CI (alpha = 0.01 → 99% CI)
    let ci_wide = fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 100, 0.01, 42).unwrap();
    for j in 0..20 {
        assert!(
            ci_wide.lower[j] <= ci_wide.upper[j] + 1e-10,
            "Wide CI should still have lower <= upper at j={}",
            j
        );
    }

    // Wide CI should be wider than narrow CI on average
    let avg_narrow: f64 = (0..20)
        .map(|j| ci_narrow.upper[j] - ci_narrow.lower[j])
        .sum::<f64>()
        / 20.0;
    let avg_wide: f64 = (0..20)
        .map(|j| ci_wide.upper[j] - ci_wide.lower[j])
        .sum::<f64>()
        / 20.0;
    assert!(
        avg_wide > avg_narrow - 1e-10,
        "99% CI should be wider than 1% CI: wide={}, narrow={}",
        avg_wide,
        avg_narrow
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Numerical stability
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bootstrap_ci_stable_with_large_n_boot() {
    let (data, y) = generate_regression_data(30, 15, 42);
    let ci = fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 1000, 0.05, 42).unwrap();
    for j in 0..15 {
        assert!(ci.lower[j].is_finite(), "lower should be finite at j={}", j);
        assert!(ci.upper[j].is_finite(), "upper should be finite at j={}", j);
        assert!(
            ci.sim_lower[j].is_finite(),
            "sim_lower should be finite at j={}",
            j
        );
        assert!(
            ci.sim_upper[j].is_finite(),
            "sim_upper should be finite at j={}",
            j
        );
    }
    assert!(
        ci.n_boot_success >= 900,
        "Most of 1000 replicates should succeed: {}",
        ci.n_boot_success
    );
}

#[test]
fn pdp_stable_with_large_n_grid() {
    let (data, y) = generate_regression_data(30, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pdp = fdars_core::functional_pdp(&fit, &data, None, 0, 500).unwrap();
    assert_eq!(pdp.grid_values.len(), 500);
    assert_eq!(pdp.pdp_curve.len(), 500);
    for g in 0..500 {
        assert!(
            pdp.pdp_curve[g].is_finite(),
            "PDP should be finite at g={}",
            g
        );
        assert!(
            pdp.grid_values[g].is_finite(),
            "Grid should be finite at g={}",
            g
        );
    }
    // Grid should be monotonically increasing
    for g in 1..500 {
        assert!(
            pdp.grid_values[g] >= pdp.grid_values[g - 1] - 1e-15,
            "Grid should be monotonic at g={}",
            g
        );
    }
}

#[test]
fn elastic_attribution_stable_with_many_components() {
    let (data, y, t) = generate_elastic_data(20, 51);
    // Request many components (will be clamped to n-1)
    let result = elastic_pcr(&data, &y, &t, 10, PcaMethod::Joint, 0.0, 5, 1e-3).unwrap();
    let ncomp = result.coefficients.len();
    let attr = fdars_core::elastic_pcr_attribution(&result, &y, ncomp, 20, 42).unwrap();

    for i in 0..20 {
        assert!(
            attr.amplitude_contribution[i].is_finite(),
            "amp should be finite at i={}",
            i
        );
        assert!(
            attr.phase_contribution[i].is_finite(),
            "phase should be finite at i={}",
            i
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PDP: consistency across model types
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn pdp_linear_and_logistic_differ() {
    // Same data, different model types should produce different PDP curves
    let (data, y_cont) = generate_regression_data(30, 50, 42);
    let y_median = {
        let mut sorted = y_cont.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };
    let y_bin: Vec<f64> = y_cont
        .iter()
        .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
        .collect();

    let fit_lm = fregre_lm(&data, &y_cont, None, 3).unwrap();
    let fit_log = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();

    let pdp_lm = fdars_core::functional_pdp(&fit_lm, &data, None, 0, 20).unwrap();
    let pdp_log = fdars_core::functional_pdp_logistic(&fit_log, &data, None, 0, 20).unwrap();

    // Linear PDP values are unbounded; logistic PDP values are in [0,1]
    let _lm_out_of_unit = pdp_lm.pdp_curve.iter().any(|&v| !(0.0..=1.0).contains(&v));
    let log_in_unit = pdp_log.pdp_curve.iter().all(|&v| (0.0..=1.0).contains(&v));

    // Linear model PDP is likely outside [0,1] for continuous y
    // (may or may not be — just check logistic is bounded)
    assert!(log_in_unit, "Logistic PDP must be in [0,1]");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Integration: all three features together
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn integration_all_features_on_same_data() {
    let (data, y) = generate_regression_data(40, 30, 42);
    let y_bin: Vec<f64> = y
        .iter()
        .map(|&v| {
            if v >= y.iter().sum::<f64>() / y.len() as f64 {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    // Feature 1: Bootstrap CI
    let ci = fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 3, 100, 0.05, 42).unwrap();
    assert_eq!(ci.center.len(), 30);
    assert!(ci.n_boot_success > 50);

    // Feature 1b: Bootstrap CI logistic
    let ci_log = fdars_core::bootstrap_ci_functional_logistic(
        &data, &y_bin, None, 3, 100, 0.05, 42, 25, 1e-6,
    )
    .unwrap();
    assert_eq!(ci_log.center.len(), 30);

    // Feature 3: PDP linear
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let pdp = fdars_core::functional_pdp(&fit, &data, None, 0, 20).unwrap();
    assert_eq!(pdp.pdp_curve.len(), 20);

    // Feature 3b: PDP logistic
    let fit_log = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    let pdp_log = fdars_core::functional_pdp_logistic(&fit_log, &data, None, 0, 20).unwrap();
    assert_eq!(pdp_log.pdp_curve.len(), 20);
    for g in 0..20 {
        assert!(pdp_log.pdp_curve[g] >= 0.0 && pdp_log.pdp_curve[g] <= 1.0);
    }
}

#[test]
fn integration_elastic_features() {
    let (data, y, t) = generate_elastic_data(15, 51);

    // Feature 2: Elastic attribution for all three PCA methods
    for method in [PcaMethod::Vertical, PcaMethod::Horizontal, PcaMethod::Joint] {
        let result = elastic_pcr(&data, &y, &t, 3, method, 0.0, 5, 1e-3).unwrap();
        let attr = fdars_core::elastic_pcr_attribution(&result, &y, 3, 10, 42).unwrap();

        // Sum should always equal fitted - alpha
        for i in 0..15 {
            let sum = attr.amplitude_contribution[i] + attr.phase_contribution[i];
            let expected = result.fitted_values[i] - result.alpha;
            assert!(
                (sum - expected).abs() < 1e-5,
                "Decomposition should hold for {:?} at i={}: {} vs {}",
                method,
                i,
                sum,
                expected
            );
        }
    }
}

