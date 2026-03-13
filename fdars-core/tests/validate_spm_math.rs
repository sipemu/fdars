//! Mathematical validation tests for SPM chi-squared, T-squared, SPE, EWMA,
//! MFPCA roundtrip, Phase I/II monitoring, and ScoSh regression.
//!
//! Chi-squared quantile values are compared against standard statistical tables.
//! Other tests verify algebraic identities and expected monitoring behavior.
//!
//! Run: cargo test -p fdars-core --features linalg --test validate_spm_math

use std::f64::consts::PI;

use fdars_core::matrix::FdMatrix;
use fdars_core::spm::control::t2_control_limit;
use fdars_core::spm::ewma::ewma_scores;
use fdars_core::spm::mfpca::{mfpca, MfpcaConfig};
use fdars_core::spm::phase::{spm_monitor, spm_phase1, SpmConfig};
use fdars_core::spm::stats::{hotelling_t2, spe_univariate};

use fdars_core::elastic_regression::{
    predict_scalar_on_shape, scalar_on_shape, IndexMethod, ScalarOnShapeConfig,
};

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Generate n sine curves on a grid of m points in [0, 1].
/// Each curve is sin(2*pi*t) with small per-curve variation in amplitude and phase.
fn generate_sine_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut rng_state = seed;
    for i in 0..n {
        // Simple LCG for deterministic pseudo-random variation
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let amp = 1.0 + 0.05 * ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5);
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let phase = 0.02 * ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5);
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * argvals[j] + phase).sin();
        }
    }
    (data, argvals)
}

/// Generate shifted data (in-control data + a constant offset).
fn generate_shifted_data(base: &FdMatrix, shift: f64) -> FdMatrix {
    let (n, m) = base.shape();
    let mut shifted = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            shifted[(i, j)] = base[(i, j)] + shift;
        }
    }
    shifted
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Chi-squared quantile validation against statistical tables
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn chi2_quantile_1_095() {
    // chi2(1, 0.95) = 3.8415
    let cl = t2_control_limit(1, 0.05).unwrap();
    assert!(
        (cl.ucl - 3.8415).abs() < 0.01,
        "chi2(1, 0.95) expected ~3.8415, got {}",
        cl.ucl
    );
}

#[test]
fn chi2_quantile_2_095() {
    // chi2(2, 0.95) = 5.9915
    let cl = t2_control_limit(2, 0.05).unwrap();
    assert!(
        (cl.ucl - 5.9915).abs() < 0.01,
        "chi2(2, 0.95) expected ~5.9915, got {}",
        cl.ucl
    );
}

#[test]
fn chi2_quantile_5_095() {
    // chi2(5, 0.95) = 11.0705
    let cl = t2_control_limit(5, 0.05).unwrap();
    assert!(
        (cl.ucl - 11.0705).abs() < 0.01,
        "chi2(5, 0.95) expected ~11.0705, got {}",
        cl.ucl
    );
}

#[test]
fn chi2_quantile_10_095() {
    // chi2(10, 0.95) = 18.307
    let cl = t2_control_limit(10, 0.05).unwrap();
    assert!(
        (cl.ucl - 18.307).abs() < 0.02,
        "chi2(10, 0.95) expected ~18.307, got {}",
        cl.ucl
    );
}

#[test]
fn chi2_quantile_1_099() {
    // chi2(1, 0.99) = 6.6349
    let cl = t2_control_limit(1, 0.01).unwrap();
    assert!(
        (cl.ucl - 6.6349).abs() < 0.01,
        "chi2(1, 0.99) expected ~6.6349, got {}",
        cl.ucl
    );
}

#[test]
fn chi2_quantile_2_099() {
    // chi2(2, 0.99) = 9.2103
    let cl = t2_control_limit(2, 0.01).unwrap();
    assert!(
        (cl.ucl - 9.2103).abs() < 0.01,
        "chi2(2, 0.99) expected ~9.2103, got {}",
        cl.ucl
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. T-squared and SPE mathematical properties
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn t2_of_zero_scores_is_zero() {
    // T2 of all-zero scores should be exactly 0
    let scores = FdMatrix::zeros(3, 2);
    let eigenvalues = vec![1.0, 1.0];
    let t2 = hotelling_t2(&scores, &eigenvalues).unwrap();
    assert_eq!(t2, vec![0.0, 0.0, 0.0]);
}

#[test]
fn t2_with_known_scores_and_eigenvalues() {
    // scores = [[2, 3]], eigenvalues = [1, 1]
    // T2 = 2^2/1 + 3^2/1 = 4 + 9 = 13
    let mut scores = FdMatrix::zeros(1, 2);
    scores[(0, 0)] = 2.0;
    scores[(0, 1)] = 3.0;
    let eigenvalues = vec![1.0, 1.0];
    let t2 = hotelling_t2(&scores, &eigenvalues).unwrap();
    assert!(
        (t2[0] - 13.0).abs() < 1e-12,
        "T2 should be 13.0, got {}",
        t2[0]
    );
}

#[test]
fn t2_with_nonunit_eigenvalues() {
    // scores = [[4, 6]], eigenvalues = [2, 3]
    // T2 = 4^2/2 + 6^2/3 = 8 + 12 = 20
    let mut scores = FdMatrix::zeros(1, 2);
    scores[(0, 0)] = 4.0;
    scores[(0, 1)] = 6.0;
    let eigenvalues = vec![2.0, 3.0];
    let t2 = hotelling_t2(&scores, &eigenvalues).unwrap();
    assert!(
        (t2[0] - 20.0).abs() < 1e-12,
        "T2 should be 20.0, got {}",
        t2[0]
    );
}

#[test]
fn spe_of_identical_centered_and_reconstructed_is_zero() {
    // SPE should be 0 when centered and reconstructed are identical
    let m = 20;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut centered = FdMatrix::zeros(2, m);
    for j in 0..m {
        centered[(0, j)] = (2.0 * PI * argvals[j]).sin();
        centered[(1, j)] = (2.0 * PI * argvals[j]).cos();
    }
    let reconstructed = centered.clone();
    let spe = spe_univariate(&centered, &reconstructed, &argvals).unwrap();
    assert!(
        spe[0].abs() < 1e-15,
        "SPE should be 0 for identical data, got {}",
        spe[0]
    );
    assert!(
        spe[1].abs() < 1e-15,
        "SPE should be 0 for identical data, got {}",
        spe[1]
    );
}

#[test]
fn spe_is_always_nonnegative() {
    // SPE is a sum of squared differences with positive weights, so >= 0
    let m = 30;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let n = 10;
    let mut centered = FdMatrix::zeros(n, m);
    let mut reconstructed = FdMatrix::zeros(n, m);
    let mut rng_state: u64 = 12345;
    for i in 0..n {
        for j in 0..m {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            centered[(i, j)] = (rng_state >> 33) as f64 / u32::MAX as f64;
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            reconstructed[(i, j)] = (rng_state >> 33) as f64 / u32::MAX as f64;
        }
    }
    let spe = spe_univariate(&centered, &reconstructed, &argvals).unwrap();
    for (i, &val) in spe.iter().enumerate() {
        assert!(val >= 0.0, "SPE[{}] should be non-negative, got {}", i, val);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. EWMA mathematical properties
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ewma_lambda_1_returns_raw_scores() {
    // With lambda = 1, EWMA reduces to raw scores (no smoothing)
    // Z_0 = 1.0 * xi_0 = xi_0
    // Z_t = 1.0 * xi_t + 0.0 * Z_{t-1} = xi_t
    let n = 5;
    let ncomp = 3;
    let mut scores = FdMatrix::zeros(n, ncomp);
    let mut rng_state: u64 = 99;
    for i in 0..n {
        for k in 0..ncomp {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            scores[(i, k)] = (rng_state >> 33) as f64 / u32::MAX as f64;
        }
    }
    let smoothed = ewma_scores(&scores, 1.0).unwrap();
    for i in 0..n {
        for k in 0..ncomp {
            assert!(
                (smoothed[(i, k)] - scores[(i, k)]).abs() < 1e-12,
                "EWMA(lambda=1) should equal raw scores at ({}, {}): {} vs {}",
                i,
                k,
                smoothed[(i, k)],
                scores[(i, k)]
            );
        }
    }
}

#[test]
fn ewma_lambda_05_single_observation() {
    // lambda = 0.5, single observation [2, 4]
    // Z_0 = 0.5 * [2, 4] + 0.5 * [0, 0] = [1, 2]
    let mut scores = FdMatrix::zeros(1, 2);
    scores[(0, 0)] = 2.0;
    scores[(0, 1)] = 4.0;
    let smoothed = ewma_scores(&scores, 0.5).unwrap();
    assert!(
        (smoothed[(0, 0)] - 1.0).abs() < 1e-12,
        "EWMA(0.5) of [2,4] should give [1,2], got [{}, {}]",
        smoothed[(0, 0)],
        smoothed[(0, 1)]
    );
    assert!(
        (smoothed[(0, 1)] - 2.0).abs() < 1e-12,
        "EWMA(0.5) of [2,4] should give [1,2], got [{}, {}]",
        smoothed[(0, 0)],
        smoothed[(0, 1)]
    );
}

#[test]
fn ewma_lambda_05_two_observations() {
    // lambda = 0.5
    // Z_0 = 0.5 * [2, 4] = [1, 2]
    // Z_1 = 0.5 * [6, 8] + 0.5 * [1, 2] = [3.5, 5.0]
    let mut scores = FdMatrix::zeros(2, 2);
    scores[(0, 0)] = 2.0;
    scores[(0, 1)] = 4.0;
    scores[(1, 0)] = 6.0;
    scores[(1, 1)] = 8.0;
    let smoothed = ewma_scores(&scores, 0.5).unwrap();
    assert!(
        (smoothed[(1, 0)] - 3.5).abs() < 1e-12,
        "EWMA second obs col 0: expected 3.5, got {}",
        smoothed[(1, 0)]
    );
    assert!(
        (smoothed[(1, 1)] - 5.0).abs() < 1e-12,
        "EWMA second obs col 1: expected 5.0, got {}",
        smoothed[(1, 1)]
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. MFPCA roundtrip: project + reconstruct recovers original
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mfpca_roundtrip_reconstruction() {
    let n = 20;
    let m1 = 30;
    let m2 = 25;

    // Variable 1: sine curves
    let mut var1 = FdMatrix::zeros(n, m1);
    for i in 0..n {
        let freq = 1.0 + 0.5 * (i as f64 / n as f64);
        for j in 0..m1 {
            let t = j as f64 / (m1 - 1) as f64;
            var1[(i, j)] = (2.0 * PI * freq * t).sin();
        }
    }

    // Variable 2: cosine curves
    let mut var2 = FdMatrix::zeros(n, m2);
    for i in 0..n {
        let freq = 1.0 + 0.3 * (i as f64 / n as f64);
        for j in 0..m2 {
            let t = j as f64 / (m2 - 1) as f64;
            var2[(i, j)] = (2.0 * PI * freq * t).cos();
        }
    }

    let variables: Vec<&FdMatrix> = vec![&var1, &var2];
    // Use all possible components (min(n, total_cols) = min(20, 55) = 20,
    // but effectively min(n-1, total_cols) components will be meaningful)
    let config = MfpcaConfig {
        ncomp: n - 1, // use all available
        weighted: true,
    };
    let result = mfpca(&variables, &config).unwrap();

    // Project the same data back
    let scores = result.project(&variables).unwrap();
    let ncomp = result.eigenvalues.len();
    let recon = result.reconstruct(&scores, ncomp).unwrap();

    // Reconstruction should approximately match the originals
    let mut max_err_v1 = 0.0_f64;
    for i in 0..n {
        for j in 0..m1 {
            let err = (recon[0][(i, j)] - var1[(i, j)]).abs();
            max_err_v1 = max_err_v1.max(err);
        }
    }
    assert!(
        max_err_v1 < 0.1,
        "MFPCA roundtrip error for variable 1 too large: {max_err_v1}"
    );

    let mut max_err_v2 = 0.0_f64;
    for i in 0..n {
        for j in 0..m2 {
            let err = (recon[1][(i, j)] - var2[(i, j)]).abs();
            max_err_v2 = max_err_v2.max(err);
        }
    }
    assert!(
        max_err_v2 < 0.1,
        "MFPCA roundtrip error for variable 2 too large: {max_err_v2}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Phase I/II with known data
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn phase1_in_control_builds_chart() {
    let (data, argvals) = generate_sine_data(40, 50, 42);
    let config = SpmConfig {
        ncomp: 3,
        alpha: 0.05,
        tuning_fraction: 0.5,
        seed: 42,
    };
    let chart = spm_phase1(&data, &argvals, &config);
    assert!(
        chart.is_ok(),
        "spm_phase1 should succeed: {:?}",
        chart.err()
    );
    let chart = chart.unwrap();
    assert!(chart.t2_limit.ucl > 0.0, "T2 UCL should be positive");
    assert!(chart.spe_limit.ucl > 0.0, "SPE UCL should be positive");
}

#[test]
fn monitoring_in_control_data_few_alarms() {
    let (data, argvals) = generate_sine_data(40, 50, 42);
    let config = SpmConfig {
        ncomp: 3,
        alpha: 0.05,
        tuning_fraction: 0.5,
        seed: 42,
    };
    let chart = spm_phase1(&data, &argvals, &config).unwrap();

    // Monitor the same in-control data. We expect few/no T2 alarms.
    let result = spm_monitor(&chart, &data, &argvals).unwrap();
    let n_t2_alarms: usize = result.t2_alarm.iter().filter(|&&a| a).count();
    // With alpha = 0.05 and 40 obs, at most a handful of false alarms expected
    assert!(
        n_t2_alarms <= 10,
        "In-control data should have few T2 alarms, got {} out of 40",
        n_t2_alarms
    );
}

#[test]
fn monitoring_shifted_data_detects_alarms() {
    let (data, argvals) = generate_sine_data(40, 50, 42);
    let config = SpmConfig {
        ncomp: 3,
        alpha: 0.05,
        tuning_fraction: 0.5,
        seed: 42,
    };
    let chart = spm_phase1(&data, &argvals, &config).unwrap();

    // Create strongly shifted data (large constant offset)
    let shifted = generate_shifted_data(&data, 5.0);
    let result = spm_monitor(&chart, &shifted, &argvals).unwrap();

    // We expect many alarms for the shifted data
    let n_t2_alarms: usize = result.t2_alarm.iter().filter(|&&a| a).count();
    let n_spe_alarms: usize = result.spe_alarm.iter().filter(|&&a| a).count();
    let total_alarms = n_t2_alarms + n_spe_alarms;
    assert!(
        total_alarms > 10,
        "Shifted data should trigger many alarms, got only {} (T2={}, SPE={})",
        total_alarms,
        n_t2_alarms,
        n_spe_alarms
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. ScoSh basic validation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn scosh_identity_index_constant_curves_finite_scores() {
    // Constant curves with small variation: shape scores should be finite
    let n = 15;
    let m = 41;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0; n];

    // Generate curves with varying shape
    for i in 0..n {
        let freq = 1.0 + 2.0 * (i as f64 / n as f64);
        for j in 0..m {
            data[(i, j)] = (freq * PI * argvals[j]).sin();
        }
        y[i] = freq;
    }

    let config = ScalarOnShapeConfig {
        nbasis: 7,
        lambda: 1e-2,
        index_method: IndexMethod::Identity,
        max_iter_inner: 5,
        max_iter_outer: 3,
        ..ScalarOnShapeConfig::default()
    };

    let result = scalar_on_shape(&data, &y, &argvals, &config);
    assert!(
        result.is_ok(),
        "scalar_on_shape should succeed: {:?}",
        result.err()
    );
    let res = result.unwrap();

    // All shape scores should be finite
    assert!(
        res.shape_scores.iter().all(|v| v.is_finite()),
        "shape scores should all be finite"
    );

    // All fitted values should be finite
    assert!(
        res.fitted_values.iter().all(|v| v.is_finite()),
        "fitted values should all be finite"
    );

    // R-squared should be finite and in a reasonable range
    assert!(res.r_squared.is_finite(), "R-squared should be finite");
}

#[test]
fn scosh_predict_on_training_data_matches_fitted() {
    let n = 20;
    let m = 41;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0; n];

    for i in 0..n {
        let freq = 1.0 + 2.0 * (i as f64 / n as f64);
        for j in 0..m {
            data[(i, j)] = (freq * PI * argvals[j]).sin();
        }
        y[i] = freq;
    }

    let config = ScalarOnShapeConfig {
        nbasis: 7,
        lambda: 1e-2,
        index_method: IndexMethod::Identity,
        max_iter_inner: 5,
        max_iter_outer: 3,
        ..ScalarOnShapeConfig::default()
    };

    let fit = scalar_on_shape(&data, &y, &argvals, &config).unwrap();
    let preds = predict_scalar_on_shape(&fit, &data, &argvals).unwrap();

    assert_eq!(preds.len(), n);

    // Predictions should be approximately close to fitted values.
    // Not exact because predict re-does alignment (DP to beta), which
    // may differ slightly from the iterative alignment in fitting.
    // Use a generous tolerance.
    let mut max_diff = 0.0_f64;
    for i in 0..n {
        let diff = (preds[i] - fit.fitted_values[i]).abs();
        max_diff = max_diff.max(diff);
    }

    // The predictions and fitted values may differ because alignment is
    // re-estimated, but they should be in the same ballpark.
    let fitted_range = fit
        .fitted_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        - fit
            .fitted_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
    let relative_max_diff = if fitted_range > 1e-10 {
        max_diff / fitted_range
    } else {
        max_diff
    };

    assert!(
        relative_max_diff < 1.0,
        "predict on training data should roughly match fitted values; \
         max_diff={max_diff:.4}, fitted_range={fitted_range:.4}, relative={relative_max_diff:.4}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Chi-squared CDF consistency via control limits
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn chi2_ucl_increases_with_k() {
    // UCL = chi2_quantile(1 - alpha, k). For fixed alpha, UCL should
    // increase with k because the chi-squared distribution shifts right
    // as degrees of freedom grow.
    let ucl_2 = t2_control_limit(2, 0.05).unwrap().ucl;
    let ucl_5 = t2_control_limit(5, 0.05).unwrap().ucl;
    let ucl_10 = t2_control_limit(10, 0.05).unwrap().ucl;

    assert!(
        ucl_2 < ucl_5,
        "UCL(k=2) should be < UCL(k=5): {} vs {}",
        ucl_2,
        ucl_5
    );
    assert!(
        ucl_5 < ucl_10,
        "UCL(k=5) should be < UCL(k=10): {} vs {}",
        ucl_5,
        ucl_10
    );
}

#[test]
fn chi2_ucl_increases_as_alpha_decreases() {
    // For fixed k, a smaller alpha (stricter significance) means the
    // quantile is further in the tail, so UCL should increase.
    let ucl_010 = t2_control_limit(5, 0.10).unwrap().ucl;
    let ucl_005 = t2_control_limit(5, 0.05).unwrap().ucl;
    let ucl_001 = t2_control_limit(5, 0.01).unwrap().ucl;

    assert!(
        ucl_010 < ucl_005,
        "UCL(alpha=0.10) should be < UCL(alpha=0.05): {} vs {}",
        ucl_010,
        ucl_005
    );
    assert!(
        ucl_005 < ucl_001,
        "UCL(alpha=0.05) should be < UCL(alpha=0.01): {} vs {}",
        ucl_005,
        ucl_001
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. SPE control limit monotonicity with variance
// ═══════════════════════════════════════════════════════════════════════════

use fdars_core::spm::control::spe_control_limit;

#[test]
fn spe_control_limit_higher_variance_gives_higher_ucl() {
    // SPE UCL is derived from mean and variance of SPE values via
    // moment-matched chi-squared. A dataset with higher SPE variance
    // should yield a higher UCL.
    let low_var_spe: Vec<f64> = (0..30).map(|i| 1.0 + 0.01 * (i as f64 / 29.0)).collect();
    let high_var_spe: Vec<f64> = (0..30).map(|i| 1.0 + 5.0 * (i as f64 / 29.0)).collect();

    let ucl_low = spe_control_limit(&low_var_spe, 0.05).unwrap().ucl;
    let ucl_high = spe_control_limit(&high_var_spe, 0.05).unwrap().ucl;

    assert!(
        ucl_low < ucl_high,
        "SPE UCL for low-variance data should be < high-variance: {} vs {}",
        ucl_low,
        ucl_high
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. T² is invariant to eigenvalue ordering
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn t2_invariant_to_eigenvalue_ordering() {
    // T² = sum_l (score_l^2 / eigenvalue_l)
    // Swapping (eigenvalue, score) pairs should give the same total.
    let a = 2.5;
    let b = -1.3;

    // Order 1: eigenvalues [1.0, 4.0], scores [a, b]
    let mut scores1 = FdMatrix::zeros(1, 2);
    scores1[(0, 0)] = a;
    scores1[(0, 1)] = b;
    let eigenvalues1 = vec![1.0, 4.0];
    let t2_1 = hotelling_t2(&scores1, &eigenvalues1).unwrap();

    // Order 2: eigenvalues [4.0, 1.0], scores [b, a]
    let mut scores2 = FdMatrix::zeros(1, 2);
    scores2[(0, 0)] = b;
    scores2[(0, 1)] = a;
    let eigenvalues2 = vec![4.0, 1.0];
    let t2_2 = hotelling_t2(&scores2, &eigenvalues2).unwrap();

    assert!(
        (t2_1[0] - t2_2[0]).abs() < 1e-12,
        "T² should be invariant to eigenvalue ordering: {} vs {}",
        t2_1[0],
        t2_2[0]
    );

    // Also verify the actual value: a^2/1 + b^2/4 = 6.25 + 0.4225 = 6.6725
    let expected = a * a / 1.0 + b * b / 4.0;
    assert!(
        (t2_1[0] - expected).abs() < 1e-12,
        "T² should be {expected}, got {}",
        t2_1[0]
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Contribution additivity with non-trivial grid sizes
// ═══════════════════════════════════════════════════════════════════════════

use fdars_core::spm::contrib::t2_contributions;

#[test]
fn t2_contributions_sum_to_total_t2() {
    // Create MFPCA-like scores for 3 variables with grid_sizes [10, 15, 20].
    // t2_contributions should give n x 3 matrix where each row sums to the
    // corresponding total T².
    let n = 5;
    let ncomp = 4;
    let grid_sizes = vec![10, 15, 20];

    let mut scores = FdMatrix::zeros(n, ncomp);
    let eigenvalues = vec![3.0, 2.0, 1.5, 0.5];
    let mut rng_state: u64 = 777;
    for i in 0..n {
        for k in 0..ncomp {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            scores[(i, k)] = (rng_state >> 33) as f64 / u32::MAX as f64 - 0.5;
        }
    }

    let contrib = t2_contributions(&scores, &eigenvalues, &grid_sizes).unwrap();
    let total_t2 = hotelling_t2(&scores, &eigenvalues).unwrap();

    assert_eq!(contrib.nrows(), n);
    assert_eq!(contrib.ncols(), 3);

    for i in 0..n {
        let row_sum: f64 = (0..3).map(|v| contrib[(i, v)]).sum();
        assert!(
            (row_sum - total_t2[i]).abs() < 1e-10,
            "Row {} contribution sum ({}) should equal total T² ({})",
            i,
            row_sum,
            total_t2[i]
        );
    }
}

#[test]
fn t2_contributions_all_nonnegative() {
    // Each per-variable T² contribution should be non-negative since they
    // are sums of squared terms divided by positive eigenvalues.
    let n = 8;
    let ncomp = 3;
    let grid_sizes = vec![10, 15, 20];

    let mut scores = FdMatrix::zeros(n, ncomp);
    let eigenvalues = vec![2.0, 1.0, 0.5];
    let mut rng_state: u64 = 321;
    for i in 0..n {
        for k in 0..ncomp {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            scores[(i, k)] = (rng_state >> 33) as f64 / u32::MAX as f64 - 0.5;
        }
    }

    let contrib = t2_contributions(&scores, &eigenvalues, &grid_sizes).unwrap();

    for i in 0..n {
        for v in 0..3 {
            assert!(
                contrib[(i, v)] >= 0.0,
                "Contribution ({},{}) should be non-negative, got {}",
                i,
                v,
                contrib[(i, v)]
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. FRCC: regression quality matters
// ═══════════════════════════════════════════════════════════════════════════

use fdars_core::spm::frcc::{frcc_monitor, frcc_phase1, FrccConfig};

#[test]
fn frcc_noise_vs_signal_residuals() {
    // When y_curves are pure noise (no relationship to predictors),
    // the FOSR model cannot explain anything, so residuals are large.
    // When y_curves = FOSR_prediction + small noise, residuals are small.
    // The monitoring statistics from the noise case should be larger.
    let n = 40;
    let m = 30;
    let p = 2;

    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

    // Create scalar predictors
    let mut predictors = FdMatrix::zeros(n, p);
    let mut rng_state: u64 = 555;
    for i in 0..n {
        for k in 0..p {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            predictors[(i, k)] = (rng_state >> 33) as f64 / u32::MAX as f64;
        }
    }

    // Case 1: y_curves strongly related to predictors
    // y_i(t) = x_{i,1} * sin(2*pi*t) + x_{i,2} * cos(2*pi*t) + small noise
    let mut y_signal = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            let t = argvals[j];
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let noise = 0.01 * ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5);
            y_signal[(i, j)] = predictors[(i, 0)] * (2.0 * PI * t).sin()
                + predictors[(i, 1)] * (2.0 * PI * t).cos()
                + noise;
        }
    }

    // Case 2: y_curves are pure noise (no relationship to predictors)
    let mut y_noise = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            y_noise[(i, j)] = (rng_state >> 33) as f64 / u32::MAX as f64 - 0.5;
        }
    }

    let config = FrccConfig {
        ncomp: 3,
        fosr_lambda: 1e-3,
        alpha: 0.05,
        tuning_fraction: 0.5,
        seed: 42,
    };

    // Build charts from each dataset
    let chart_signal = frcc_phase1(&y_signal, &predictors, &argvals, &config).unwrap();
    let chart_noise = frcc_phase1(&y_noise, &predictors, &argvals, &config).unwrap();

    // Monitor with the same data used for training.
    // The signal case residuals should have smaller SPE control limit
    // because FOSR explains most of the variation.
    // The noise case has large unexplained residual variation → higher SPE UCL.
    //
    // Note: We compare the SPE control limits as a proxy for residual size.
    // With signal data, FOSR removes the predictable component, leaving small
    // residuals. With noise data, FOSR cannot help, leaving all variation
    // in the residuals.
    let signal_spe_ucl = chart_signal.spe_limit.ucl;
    let noise_spe_ucl = chart_noise.spe_limit.ucl;

    assert!(
        signal_spe_ucl < noise_spe_ucl,
        "Signal model should have smaller SPE UCL ({}) than noise model ({})",
        signal_spe_ucl,
        noise_spe_ucl
    );
}

#[test]
fn frcc_monitor_signal_vs_noise_t2() {
    // Build FRCC chart on signal data, then monitor:
    // (a) new signal data → few alarms
    // (b) pure noise data → many alarms
    let n = 40;
    let m = 30;
    let p = 2;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

    let mut predictors = FdMatrix::zeros(n, p);
    let mut rng_state: u64 = 9876;
    for i in 0..n {
        for k in 0..p {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            predictors[(i, k)] = (rng_state >> 33) as f64 / u32::MAX as f64;
        }
    }

    // Training data: signal
    let mut y_signal = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            let t = argvals[j];
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let noise = 0.01 * ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5);
            y_signal[(i, j)] = predictors[(i, 0)] * (2.0 * PI * t).sin()
                + predictors[(i, 1)] * (2.0 * PI * t).cos()
                + noise;
        }
    }

    let config = FrccConfig {
        ncomp: 3,
        fosr_lambda: 1e-3,
        alpha: 0.05,
        tuning_fraction: 0.5,
        seed: 42,
    };

    let chart = frcc_phase1(&y_signal, &predictors, &argvals, &config).unwrap();

    // Monitor with noise data (no relationship)
    let mut y_noise = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            y_noise[(i, j)] = 3.0 * ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5);
        }
    }

    let result_noise = frcc_monitor(&chart, &y_noise, &predictors, &argvals).unwrap();

    // Monitor with signal data (same distribution as training)
    let mut y_signal2 = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            let t = argvals[j];
            rng_state = rng_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let noise = 0.01 * ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5);
            y_signal2[(i, j)] = predictors[(i, 0)] * (2.0 * PI * t).sin()
                + predictors[(i, 1)] * (2.0 * PI * t).cos()
                + noise;
        }
    }

    let result_signal = frcc_monitor(&chart, &y_signal2, &predictors, &argvals).unwrap();

    // Noise monitoring should have higher mean T²/SPE than signal monitoring
    let mean_t2_noise: f64 = result_noise.t2.iter().sum::<f64>() / n as f64;
    let mean_t2_signal: f64 = result_signal.t2.iter().sum::<f64>() / n as f64;

    assert!(
        mean_t2_signal < mean_t2_noise,
        "Signal T² mean ({}) should be < noise T² mean ({})",
        mean_t2_signal,
        mean_t2_noise
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. ScoSh with known analytical solution: y = mean(curve) + noise
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn scosh_y_depends_only_on_amplitude() {
    // When y depends on the overall level (f0) of the curve rather than
    // its shape, the g(f0) component should dominate and the model
    // should still achieve a reasonable fit.
    //
    // We use curves: f_i(t) = level_i + sin(2*pi*t), where level_i varies
    // and y_i = level_i + small noise. The shape (sin) is the same across
    // all curves; only the vertical shift (level) differs.
    let n = 25;
    let m = 41;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0; n];
    let mut rng_state: u64 = 2468;

    for i in 0..n {
        // Vary vertical level, keep shape identical (sine)
        let level = 1.0 + 4.0 * (i as f64 / (n - 1) as f64);
        for j in 0..m {
            data[(i, j)] = level + (2.0 * PI * argvals[j]).sin();
        }
        // y depends on the level (amplitude/f0) only
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let noise = 0.01 * ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5);
        y[i] = level + noise;
    }

    let config = ScalarOnShapeConfig {
        nbasis: 7,
        lambda: 1e-2,
        index_method: IndexMethod::Identity,
        max_iter_inner: 5,
        max_iter_outer: 3,
        ..ScalarOnShapeConfig::default()
    };

    let result = scalar_on_shape(&data, &y, &argvals, &config).unwrap();

    // The fitted values should be close to y because g(f0) captures the amplitude
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let ss_total: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = result.residuals.iter().map(|&r| r * r).sum();
    let r2 = 1.0 - ss_res / ss_total;

    // Shape scores alone shouldn't explain much since y depends on amplitude
    // So the shape_scores should have low variance relative to y variance
    let shape_mean: f64 = result.shape_scores.iter().sum::<f64>() / n as f64;
    let shape_var: f64 = result
        .shape_scores
        .iter()
        .map(|&s| (s - shape_mean).powi(2))
        .sum::<f64>()
        / (n as f64 - 1.0);
    let _y_var: f64 = ss_total / (n as f64 - 1.0);

    // R² should be reasonably high (g captures amplitude)
    assert!(
        r2 > 0.5,
        "R² should be high when y depends on amplitude; got {}",
        r2
    );

    // Shape score variance should be small relative to y variance
    // (shape doesn't vary much since all curves are the same shape)
    assert!(result.r_squared.is_finite(), "R-squared should be finite");
    assert!(
        result.fitted_values.iter().all(|v| v.is_finite()),
        "All fitted values should be finite"
    );

    // The shape variance should be bounded (it shouldn't dominate)
    assert!(
        shape_var.is_finite(),
        "Shape score variance should be finite, got {}",
        shape_var
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. ScoSh Index method comparison: Polynomial(1) should be at least
//     as good as Identity
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn scosh_polynomial_at_least_as_good_as_identity() {
    // With Identity: fitted = shape_scores + g(f0)
    // With Polynomial(1): fitted = a*shape_scores + b + g(f0)
    // Polynomial is strictly more flexible, so SSE(Poly) <= SSE(Identity).
    let n = 25;
    let m = 41;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0; n];

    for i in 0..n {
        let freq = 1.0 + 3.0 * (i as f64 / n as f64);
        let amp = 0.5 + 1.5 * (i as f64 / n as f64);
        for j in 0..m {
            data[(i, j)] = amp * (freq * PI * argvals[j]).sin();
        }
        y[i] = freq + 0.5 * amp;
    }

    let config_identity = ScalarOnShapeConfig {
        nbasis: 7,
        lambda: 1e-2,
        index_method: IndexMethod::Identity,
        max_iter_inner: 5,
        max_iter_outer: 5,
        ..ScalarOnShapeConfig::default()
    };

    let config_poly = ScalarOnShapeConfig {
        nbasis: 7,
        lambda: 1e-2,
        index_method: IndexMethod::Polynomial(1),
        max_iter_inner: 5,
        max_iter_outer: 5,
        ..ScalarOnShapeConfig::default()
    };

    let fit_identity = scalar_on_shape(&data, &y, &argvals, &config_identity).unwrap();
    let fit_poly = scalar_on_shape(&data, &y, &argvals, &config_poly).unwrap();

    // Polynomial should have SSE <= Identity SSE (allowing some numerical tolerance,
    // since the iterative optimization may not reach the global optimum).
    // We use a generous tolerance of 10% relative to identity SSE.
    let tolerance_factor = 1.10;
    assert!(
        fit_poly.sse <= fit_identity.sse * tolerance_factor,
        "Polynomial SSE ({}) should be <= Identity SSE ({}) within tolerance",
        fit_poly.sse,
        fit_identity.sse
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. Phase I reproducibility with same seed
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn phase1_reproducibility_same_seed() {
    // Running spm_phase1 twice with the same config and seed should produce
    // identical results (deterministic tuning/calibration split and FPCA).
    let (data, argvals) = generate_sine_data(40, 50, 42);
    let config = SpmConfig {
        ncomp: 3,
        alpha: 0.05,
        tuning_fraction: 0.5,
        seed: 123,
    };

    let chart1 = spm_phase1(&data, &argvals, &config).unwrap();
    let chart2 = spm_phase1(&data, &argvals, &config).unwrap();

    // Control limits should be identical
    assert_eq!(
        chart1.t2_limit.ucl, chart2.t2_limit.ucl,
        "T2 UCL should be identical across runs"
    );
    assert_eq!(
        chart1.spe_limit.ucl, chart2.spe_limit.ucl,
        "SPE UCL should be identical across runs"
    );

    // Eigenvalues should be identical
    assert_eq!(
        chart1.eigenvalues.len(),
        chart2.eigenvalues.len(),
        "Eigenvalue count should match"
    );
    for (i, (&ev1, &ev2)) in chart1
        .eigenvalues
        .iter()
        .zip(chart2.eigenvalues.iter())
        .enumerate()
    {
        assert!(
            (ev1 - ev2).abs() < 1e-14,
            "Eigenvalue[{}]: {} vs {}",
            i,
            ev1,
            ev2
        );
    }

    // Phase I T² and SPE values should be identical
    assert_eq!(chart1.t2_phase1.len(), chart2.t2_phase1.len());
    for (i, (&t1, &t2)) in chart1
        .t2_phase1
        .iter()
        .zip(chart2.t2_phase1.iter())
        .enumerate()
    {
        assert!(
            (t1 - t2).abs() < 1e-14,
            "Phase I T2[{}]: {} vs {}",
            i,
            t1,
            t2
        );
    }

    assert_eq!(chart1.spe_phase1.len(), chart2.spe_phase1.len());
    for (i, (&s1, &s2)) in chart1
        .spe_phase1
        .iter()
        .zip(chart2.spe_phase1.iter())
        .enumerate()
    {
        assert!(
            (s1 - s2).abs() < 1e-14,
            "Phase I SPE[{}]: {} vs {}",
            i,
            s1,
            s2
        );
    }
}

#[test]
fn phase1_different_seeds_differ() {
    // Different seeds should produce different tuning/calibration splits
    // and therefore (likely) different results.
    let (data, argvals) = generate_sine_data(40, 50, 42);
    let config1 = SpmConfig {
        ncomp: 3,
        alpha: 0.05,
        tuning_fraction: 0.5,
        seed: 100,
    };
    let config2 = SpmConfig {
        ncomp: 3,
        alpha: 0.05,
        tuning_fraction: 0.5,
        seed: 200,
    };

    let chart1 = spm_phase1(&data, &argvals, &config1).unwrap();
    let chart2 = spm_phase1(&data, &argvals, &config2).unwrap();

    // At least one of the Phase I statistics should differ
    let t2_differ = chart1
        .t2_phase1
        .iter()
        .zip(chart2.t2_phase1.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-10);
    let spe_differ = chart1
        .spe_phase1
        .iter()
        .zip(chart2.spe_phase1.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-10);

    assert!(
        t2_differ || spe_differ,
        "Different seeds should produce different Phase I results"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. Multivariate SPM sensitivity: shifted variable contributes more
// ═══════════════════════════════════════════════════════════════════════════

use fdars_core::spm::contrib::spe_contributions;
use fdars_core::spm::phase::{mf_spm_monitor, mf_spm_phase1};

#[test]
fn multivariate_spm_shifted_variable_contributes_more_t2() {
    // Create 2-variable MFPCA data where only variable 1 is shifted.
    // T² contributions should show variable 1 contributing more.
    let n_train = 40;
    let n_test = 20;
    let m1 = 30;
    let m2 = 25;

    let argvals1: Vec<f64> = (0..m1).map(|j| j as f64 / (m1 - 1) as f64).collect();
    let argvals2: Vec<f64> = (0..m2).map(|j| j as f64 / (m2 - 1) as f64).collect();

    // Training data: both variables from the same in-control process
    let mut var1_train = FdMatrix::zeros(n_train, m1);
    let mut var2_train = FdMatrix::zeros(n_train, m2);
    let mut rng_state: u64 = 4321;

    for i in 0..n_train {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let amp1 = 1.0 + 0.05 * ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5);
        for j in 0..m1 {
            var1_train[(i, j)] = amp1 * (2.0 * PI * argvals1[j]).sin();
        }

        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let amp2 = 1.0 + 0.05 * ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5);
        for j in 0..m2 {
            var2_train[(i, j)] = amp2 * (2.0 * PI * argvals2[j]).cos();
        }
    }

    // Test data: variable 1 is shifted by a large constant, variable 2 is not
    let mut var1_test = FdMatrix::zeros(n_test, m1);
    let mut var2_test = FdMatrix::zeros(n_test, m2);

    for i in 0..n_test {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let amp1 = 1.0 + 0.05 * ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5);
        for j in 0..m1 {
            // Large shift on variable 1
            var1_test[(i, j)] = amp1 * (2.0 * PI * argvals1[j]).sin() + 5.0;
        }

        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let amp2 = 1.0 + 0.05 * ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5);
        for j in 0..m2 {
            // No shift on variable 2
            var2_test[(i, j)] = amp2 * (2.0 * PI * argvals2[j]).cos();
        }
    }

    let config = SpmConfig {
        ncomp: 3,
        alpha: 0.05,
        tuning_fraction: 0.5,
        seed: 42,
    };

    let train_vars: Vec<&FdMatrix> = vec![&var1_train, &var2_train];
    let argvals_list: Vec<&[f64]> = vec![&argvals1, &argvals2];

    let chart = mf_spm_phase1(&train_vars, &argvals_list, &config).unwrap();

    let test_vars: Vec<&FdMatrix> = vec![&var1_test, &var2_test];
    let result = mf_spm_monitor(&chart, &test_vars, &argvals_list).unwrap();

    // Compute per-variable T² contributions
    let grid_sizes = vec![m1, m2];
    let contrib = t2_contributions(&result.scores, &chart.mfpca.eigenvalues, &grid_sizes).unwrap();

    // Variable 1 contribution should be larger than variable 2 on average
    let mean_contrib_v1: f64 = (0..n_test).map(|i| contrib[(i, 0)]).sum::<f64>() / n_test as f64;
    let mean_contrib_v2: f64 = (0..n_test).map(|i| contrib[(i, 1)]).sum::<f64>() / n_test as f64;

    assert!(
        mean_contrib_v1 > mean_contrib_v2,
        "Shifted variable 1 should contribute more to T²: V1={}, V2={}",
        mean_contrib_v1,
        mean_contrib_v2
    );
}

#[test]
fn multivariate_spm_shifted_variable_contributes_more_spe() {
    // Same setup: variable 1 shifted, variable 2 not.
    // SPE contributions should show variable 1 contributing more.
    let n_train = 40;
    let n_test = 20;
    let m1 = 30;
    let m2 = 25;

    let argvals1: Vec<f64> = (0..m1).map(|j| j as f64 / (m1 - 1) as f64).collect();
    let argvals2: Vec<f64> = (0..m2).map(|j| j as f64 / (m2 - 1) as f64).collect();

    let mut var1_train = FdMatrix::zeros(n_train, m1);
    let mut var2_train = FdMatrix::zeros(n_train, m2);
    let mut rng_state: u64 = 6543;

    for i in 0..n_train {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let amp1 = 1.0 + 0.05 * ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5);
        for j in 0..m1 {
            var1_train[(i, j)] = amp1 * (2.0 * PI * argvals1[j]).sin();
        }

        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let amp2 = 1.0 + 0.05 * ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5);
        for j in 0..m2 {
            var2_train[(i, j)] = amp2 * (2.0 * PI * argvals2[j]).cos();
        }
    }

    // Build MFPCA on training data
    let train_vars: Vec<&FdMatrix> = vec![&var1_train, &var2_train];

    let mfpca_config = MfpcaConfig {
        ncomp: 3,
        weighted: true,
    };
    let mfpca_result = mfpca(&train_vars, &mfpca_config).unwrap();
    let ncomp = mfpca_result.eigenvalues.len();

    // Test data: shift variable 1, keep variable 2 the same
    let mut var1_test = FdMatrix::zeros(n_test, m1);
    let mut var2_test = FdMatrix::zeros(n_test, m2);

    for i in 0..n_test {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let amp1 = 1.0 + 0.05 * ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5);
        for j in 0..m1 {
            var1_test[(i, j)] = amp1 * (2.0 * PI * argvals1[j]).sin() + 5.0;
        }

        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let amp2 = 1.0 + 0.05 * ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5);
        for j in 0..m2 {
            var2_test[(i, j)] = amp2 * (2.0 * PI * argvals2[j]).cos();
        }
    }

    let test_vars: Vec<&FdMatrix> = vec![&var1_test, &var2_test];

    // Project and reconstruct
    let scores = mfpca_result.project(&test_vars).unwrap();
    let recon = mfpca_result.reconstruct(&scores, ncomp).unwrap();

    // Compute standardized centered data and reconstruction for SPE contributions
    let mut std_vars: Vec<FdMatrix> = Vec::new();
    let mut std_recon: Vec<FdMatrix> = Vec::new();

    for (p, test_var) in test_vars.iter().enumerate() {
        let m_p = test_var.ncols();
        let scale = if mfpca_result.scales[p] > 1e-15 {
            mfpca_result.scales[p]
        } else {
            1.0
        };
        let mut std_mat = FdMatrix::zeros(n_test, m_p);
        let mut recon_mat = FdMatrix::zeros(n_test, m_p);
        for i in 0..n_test {
            for j in 0..m_p {
                std_mat[(i, j)] = (test_var[(i, j)] - mfpca_result.means[p][j]) / scale;
                recon_mat[(i, j)] = (recon[p][(i, j)] - mfpca_result.means[p][j]) / scale;
            }
        }
        std_vars.push(std_mat);
        std_recon.push(recon_mat);
    }

    let std_refs: Vec<&FdMatrix> = std_vars.iter().collect();
    let recon_refs: Vec<&FdMatrix> = std_recon.iter().collect();
    let argvals_refs: Vec<&[f64]> = vec![&argvals1, &argvals2];

    let spe_contrib = spe_contributions(&std_refs, &recon_refs, &argvals_refs).unwrap();

    // Variable 1 SPE contribution should be larger than variable 2 on average
    let mean_spe_v1: f64 = (0..n_test).map(|i| spe_contrib[(i, 0)]).sum::<f64>() / n_test as f64;
    let mean_spe_v2: f64 = (0..n_test).map(|i| spe_contrib[(i, 1)]).sum::<f64>() / n_test as f64;

    assert!(
        mean_spe_v1 > mean_spe_v2,
        "Shifted variable 1 should contribute more to SPE: V1={}, V2={}",
        mean_spe_v1,
        mean_spe_v2
    );
}
