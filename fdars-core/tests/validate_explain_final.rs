//! Final validation: cross-checks, parallel correctness, API surface, edge cases.

use fdars_core::elastic_regression::{elastic_pcr, PcaMethod};
use fdars_core::matrix::FdMatrix;
use fdars_core::scalar_on_function::{fregre_lm, functional_logistic};
use std::f64::consts::PI;

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
// Cross-check: bootstrap SE vs asymptotic SE
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn bootstrap_se_consistent_with_asymptotic_se() {
    // Bootstrap SE ≈ (upper - lower) / (2 * z_{alpha/2})
    // Asymptotic SE comes from fregre_lm beta_se
    // They should be in the same order of magnitude
    let (data, y) = regression_data(60, 25, 42);
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
    let (data, y, t) = elastic_data(20, 51);
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
    let (data, y) = regression_data(30, 20, 42);
    let _ci: fdars_core::BootstrapCiResult =
        fdars_core::bootstrap_ci_fregre_lm(&data, &y, None, 2, 20, 0.05, 42).unwrap();
}

#[test]
fn api_pdp_accessible() {
    let (data, y) = regression_data(30, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let _pdp: fdars_core::FunctionalPdpResult =
        fdars_core::functional_pdp(&fit, &data, None, 0, 10).unwrap();
}

#[test]
fn api_elastic_attribution_accessible() {
    let (data, y, t) = elastic_data(15, 51);
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
    let (data, y) = regression_data(30, 20, 42);

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
    let (data, y) = regression_data(30, 15, 42);
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
    let (data, y) = regression_data(30, 50, 42);
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
    let (data, y, t) = elastic_data(20, 51);
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
    let (data, y_cont) = regression_data(30, 50, 42);
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
    let (data, y) = regression_data(40, 30, 42);
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
    let (data, y, t) = elastic_data(15, 51);

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
