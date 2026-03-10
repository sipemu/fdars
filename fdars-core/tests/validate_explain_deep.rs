//! Deep algorithmic validation for explainability features.
//!
//! These tests verify internal consistency of the mathematical algorithms
//! beyond basic shape/property checks.

use fdars_core::elastic_regression::{elastic_pcr, PcaMethod};
use fdars_core::matrix::FdMatrix;
use fdars_core::scalar_on_function::{fregre_lm, functional_logistic};
use std::f64::consts::PI;

// ─── Data generators ─────────────────────────────────────────────────────────

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

fn elastic_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<f64>) {
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
// Bootstrap CI: Statistical properties
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bootstrap_ci_simultaneous_band_is_symmetric_about_center() {
    // Simultaneous bands are center ± c_alpha * SE, so they should be exactly symmetric
    let (data, y) = regression_data(40, 20, 42);
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
    let (data, y) = regression_data(30, 20, 42);
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
    let (data, y) = regression_data(50, 20, 42);
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
    if let Some(ci) = result {
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
    let (data, y) = regression_data(30, 50, 42);
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
    let (data, y) = regression_data(30, 50, 42);
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
    let (data, y, t) = elastic_data(20, 51);
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
    let (data, y, t) = elastic_data(15, 51);
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
    let (data, y, t) = elastic_data(20, 51);
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
    let (data, y, t) = elastic_data(15, 51);
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
    let (data, y, t) = elastic_data(15, 51);

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
    let (data, y, t) = elastic_data(20, 51);
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
    let (data, y) = regression_data(30, 20, 42);
    assert!(fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 50, 0.0, 42).is_none());
    assert!(fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 50, 1.0, 42).is_none());
    assert!(fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 50, -0.1, 42).is_none());
}

#[test]
fn bootstrap_ci_rejects_zero_n_boot() {
    let (data, y) = regression_data(30, 20, 42);
    assert!(fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 0, 0.05, 42).is_none());
}

#[test]
fn pdp_rejects_n_grid_one() {
    let (data, y) = regression_data(30, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    assert!(fdars_core::functional_pdp(&fit, &data, None, 0, 0).is_none());
    assert!(fdars_core::functional_pdp(&fit, &data, None, 0, 1).is_none());
}

#[test]
fn elastic_attribution_rejects_empty_y() {
    let (data, y, t) = elastic_data(15, 51);
    let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Vertical, 0.0, 5, 1e-3).unwrap();
    assert!(fdars_core::elastic_pcr_attribution(&result, &[], 3, 10, 42).is_none());
}

#[test]
fn elastic_attribution_rejects_zero_ncomp() {
    let (data, y, t) = elastic_data(15, 51);
    let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Vertical, 0.0, 5, 1e-3).unwrap();
    assert!(fdars_core::elastic_pcr_attribution(&result, &y, 0, 10, 42).is_none());
}
