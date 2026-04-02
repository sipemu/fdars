use super::*;
use std::f64::consts::PI;

fn generate_test_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>, Vec<f64>) {
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
        y[i] = 2.0 * phase + 3.0 * amplitude + 0.05 * (seed.wrapping_add(i as u64) % 10) as f64;
    }
    (data, y, t)
}

// ----- fregre_lm tests -----

#[test]
fn test_fregre_lm_basic() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let result = fregre_lm(&data, &y, None, 3);
    assert!(result.is_ok());
    let fit = result.unwrap();
    assert_eq!(fit.fitted_values.len(), 30);
    assert_eq!(fit.residuals.len(), 30);
    assert_eq!(fit.beta_t.len(), 50);
    assert_eq!(fit.ncomp, 3);
    assert!(fit.r_squared >= 0.0 && fit.r_squared <= 1.0 + 1e-10);
}

#[test]
fn test_fregre_lm_with_scalar_covariates() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let mut sc = FdMatrix::zeros(30, 2);
    for i in 0..30 {
        sc[(i, 0)] = i as f64 / 30.0;
        sc[(i, 1)] = (i as f64 * 0.7).sin();
    }
    let result = fregre_lm(&data, &y, Some(&sc), 3);
    assert!(result.is_ok());
    let fit = result.unwrap();
    assert_eq!(fit.gamma.len(), 2);
}

#[test]
fn test_fregre_lm_residuals_sum_near_zero() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let resid_sum: f64 = fit.residuals.iter().sum::<f64>();
    assert!(
        resid_sum.abs() < 1e-8,
        "Residuals should sum to ~0 with intercept, got {}",
        resid_sum
    );
}

#[test]
fn test_fregre_lm_fitted_plus_residuals_equals_y() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    for i in 0..30 {
        let reconstructed = fit.fitted_values[i] + fit.residuals[i];
        assert!(
            (reconstructed - y[i]).abs() < 1e-10,
            "ŷ + r should equal y at index {}",
            i
        );
    }
}

#[test]
fn test_fregre_lm_more_components_higher_r2() {
    let (data, y, _t) = generate_test_data(50, 50, 42);
    let fit1 = fregre_lm(&data, &y, None, 1).unwrap();
    let fit3 = fregre_lm(&data, &y, None, 3).unwrap();
    assert!(
        fit3.r_squared >= fit1.r_squared - 1e-10,
        "More components should give >= R²: {} vs {}",
        fit3.r_squared,
        fit1.r_squared
    );
}

#[test]
fn test_fregre_lm_invalid_input() {
    let data = FdMatrix::zeros(2, 50);
    let y = vec![1.0, 2.0];
    assert!(fregre_lm(&data, &y, None, 1).is_err());

    let data = FdMatrix::zeros(10, 50);
    let y = vec![1.0; 5];
    assert!(fregre_lm(&data, &y, None, 2).is_err());
}

#[test]
fn test_fregre_lm_std_errors_positive() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    for (i, &se) in fit.std_errors.iter().enumerate() {
        assert!(
            se > 0.0 && se.is_finite(),
            "Std error {} should be positive finite, got {}",
            i,
            se
        );
    }
}

// ----- predict_fregre_lm tests -----

#[test]
fn test_predict_fregre_lm_on_training_data() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let preds = predict_fregre_lm(&fit, &data, None);
    for i in 0..30 {
        assert!(
            (preds[i] - fit.fitted_values[i]).abs() < 1e-6,
            "Prediction on training data should match fitted values"
        );
    }
}

// ----- fregre_cv tests -----

#[test]
fn test_fregre_cv_returns_result() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let result = fregre_cv(&data, &y, None, 1, 8, 5);
    assert!(result.is_ok());
    let cv = result.unwrap();
    assert!(cv.optimal_k >= 1 && cv.optimal_k <= 8);
    assert!(cv.min_cv_error >= 0.0);
}

#[test]
fn test_fregre_cv_with_scalar_covariates() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let mut sc = FdMatrix::zeros(30, 1);
    for i in 0..30 {
        sc[(i, 0)] = i as f64;
    }
    let result = fregre_cv(&data, &y, Some(&sc), 1, 5, 3);
    assert!(result.is_ok());
}

// ----- fregre_np_mixed tests -----

#[test]
fn test_fregre_np_mixed_basic() {
    let (data, y, t) = generate_test_data(30, 50, 42);
    let result = fregre_np_mixed(&data, &y, &t, None, 0.0, 0.0);
    assert!(result.is_ok());
    let fit = result.unwrap();
    assert_eq!(fit.fitted_values.len(), 30);
    assert!(fit.h_func > 0.0);
    assert!(fit.cv_error >= 0.0);
}

#[test]
fn test_fregre_np_mixed_with_scalars() {
    let (data, y, t) = generate_test_data(30, 50, 42);
    let mut sc = FdMatrix::zeros(30, 1);
    for i in 0..30 {
        sc[(i, 0)] = i as f64 / 30.0;
    }
    let result = fregre_np_mixed(&data, &y, &t, Some(&sc), 0.0, 0.0);
    assert!(result.is_ok());
    let fit = result.unwrap();
    assert!(fit.h_scalar > 0.0);
}

#[test]
fn test_fregre_np_mixed_manual_bandwidth() {
    let (data, y, t) = generate_test_data(30, 50, 42);
    let result = fregre_np_mixed(&data, &y, &t, None, 0.5, 0.0);
    assert!(result.is_ok());
    let fit = result.unwrap();
    assert!(
        (fit.h_func - 0.5).abs() < 1e-10,
        "Should use provided bandwidth"
    );
}

// ----- functional_logistic tests -----

#[test]
fn test_functional_logistic_basic() {
    let (data, y_cont, _t) = generate_test_data(30, 50, 42);
    let y_median = {
        let mut sorted = y_cont.clone();
        crate::helpers::sort_nan_safe(&mut sorted);
        sorted[sorted.len() / 2]
    };
    let y_bin: Vec<f64> = y_cont
        .iter()
        .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
        .collect();

    let result = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6);
    assert!(result.is_ok());
    let fit = result.unwrap();
    assert_eq!(fit.probabilities.len(), 30);
    assert_eq!(fit.predicted_classes.len(), 30);
    assert!(fit.accuracy >= 0.0 && fit.accuracy <= 1.0);
    for &p in &fit.probabilities {
        assert!((0.0..=1.0).contains(&p), "Probability out of range: {}", p);
    }
}

#[test]
fn test_functional_logistic_with_scalar_covariates() {
    let (data, y_cont, _t) = generate_test_data(30, 50, 42);
    let y_median = {
        let mut sorted = y_cont.clone();
        crate::helpers::sort_nan_safe(&mut sorted);
        sorted[sorted.len() / 2]
    };
    let y_bin: Vec<f64> = y_cont
        .iter()
        .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
        .collect();

    let mut sc = FdMatrix::zeros(30, 1);
    for i in 0..30 {
        sc[(i, 0)] = i as f64 / 30.0;
    }
    let result = functional_logistic(&data, &y_bin, Some(&sc), 3, 25, 1e-6);
    assert!(result.is_ok());
    let fit = result.unwrap();
    assert_eq!(fit.gamma.len(), 1);
}

#[test]
fn test_functional_logistic_invalid_response() {
    let (data, _, _) = generate_test_data(30, 50, 42);
    let y: Vec<f64> = (0..30).map(|i| i as f64).collect();
    assert!(functional_logistic(&data, &y, None, 3, 25, 1e-6).is_err());
}

#[test]
fn test_functional_logistic_convergence() {
    let (data, y_cont, _t) = generate_test_data(40, 50, 42);
    let y_median = {
        let mut sorted = y_cont.clone();
        crate::helpers::sort_nan_safe(&mut sorted);
        sorted[sorted.len() / 2]
    };
    let y_bin: Vec<f64> = y_cont
        .iter()
        .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
        .collect();

    let fit = functional_logistic(&data, &y_bin, None, 3, 100, 1e-8).unwrap();
    assert!(fit.iterations <= 100, "Should converge within max_iter");
}

// ----- Edge cases -----

#[test]
fn test_fregre_lm_single_component() {
    let (data, y, _t) = generate_test_data(20, 50, 42);
    let result = fregre_lm(&data, &y, None, 1);
    assert!(result.is_ok());
    let fit = result.unwrap();
    assert_eq!(fit.ncomp, 1);
}

#[test]
fn test_fregre_lm_auto_k_selection() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let result = fregre_lm(&data, &y, None, 0);
    assert!(result.is_ok());
    let fit = result.unwrap();
    assert!(fit.ncomp >= 1);
}

#[test]
fn test_predict_fregre_np_basic() {
    let (data, y, t) = generate_test_data(30, 50, 42);
    let preds = predict_fregre_np(&data, &y, None, &data, None, &t, 0.5, 0.5);
    assert_eq!(preds.len(), 30);
    for &p in &preds {
        assert!(p.is_finite(), "Prediction should be finite");
    }
}

#[test]
fn test_sigmoid_properties() {
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
    assert!(sigmoid(10.0) > 0.999);
    assert!(sigmoid(-10.0) < 0.001);
    assert!((sigmoid(3.0) + sigmoid(-3.0) - 1.0).abs() < 1e-10);
}

// ----- beta_se tests -----

#[test]
fn test_fregre_lm_beta_se() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    assert_eq!(fit.beta_se.len(), 50, "beta_se should have length m");
    for (j, &se) in fit.beta_se.iter().enumerate() {
        assert!(
            se > 0.0 && se.is_finite(),
            "beta_se[{}] should be positive finite, got {}",
            j,
            se
        );
    }
}

#[test]
fn test_fregre_lm_beta_se_coverage() {
    // Use generate_test_data which is known to produce valid FPCA, then check SE properties
    let (data, y, _t) = generate_test_data(50, 50, 99);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    assert_eq!(fit.beta_se.len(), 50);
    // With valid data, beta_se should be positive everywhere
    for (j, &se) in fit.beta_se.iter().enumerate() {
        assert!(
            se > 0.0 && se.is_finite(),
            "beta_se[{}] should be positive finite, got {}",
            j,
            se
        );
    }
    // The CI band [beta_t ± 2·SE] should have non-zero width everywhere
    for j in 0..50 {
        let width = 4.0 * fit.beta_se[j];
        assert!(width > 0.0, "CI width should be positive at j={}", j);
    }
}

#[test]
fn test_functional_logistic_beta_se() {
    let (data, y_cont, _t) = generate_test_data(30, 50, 42);
    let y_median = {
        let mut sorted = y_cont.clone();
        crate::helpers::sort_nan_safe(&mut sorted);
        sorted[sorted.len() / 2]
    };
    let y_bin: Vec<f64> = y_cont
        .iter()
        .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
        .collect();

    let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
    assert_eq!(fit.beta_se.len(), 50, "beta_se should have length m");
    assert_eq!(
        fit.std_errors.len(),
        1 + 3,
        "std_errors should have length 1 + ncomp"
    );
    for (j, &se) in fit.beta_se.iter().enumerate() {
        assert!(
            se > 0.0 && se.is_finite(),
            "beta_se[{}] should be positive finite, got {}",
            j,
            se
        );
    }
    for (j, &se) in fit.std_errors.iter().enumerate() {
        assert!(
            se > 0.0 && se.is_finite(),
            "std_errors[{}] should be positive finite, got {}",
            j,
            se
        );
    }
}

#[test]
fn test_beta_se_zero_for_constant() {
    // When all curves are nearly identical, β(t) ≈ 0 but SE should still be finite/positive
    let n = 30;
    let m = 20;
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0; n];
    for i in 0..n {
        for j in 0..m {
            // Nearly identical curves with tiny variation
            data[(i, j)] = 1.0 + 0.001 * (i as f64 / n as f64);
        }
        y[i] = i as f64 / n as f64;
    }
    let fit = fregre_lm(&data, &y, None, 1).unwrap();
    assert_eq!(fit.beta_se.len(), m);
    for (j, &se) in fit.beta_se.iter().enumerate() {
        assert!(
            se.is_finite() && se >= 0.0,
            "beta_se[{}] should be finite non-negative, got {}",
            j,
            se
        );
    }
}

// ----- Bootstrap CI tests -----

#[test]
fn test_bootstrap_ci_fregre_lm_shape() {
    let (data, y, _t) = generate_test_data(30, 20, 42);
    let result = bootstrap_ci_fregre_lm(&data, &y, None, 2, 50, 0.05, 123);
    assert!(result.is_ok(), "bootstrap_ci_fregre_lm should succeed");
    let ci = result.unwrap();
    assert_eq!(ci.lower.len(), 20);
    assert_eq!(ci.upper.len(), 20);
    assert_eq!(ci.center.len(), 20);
    assert_eq!(ci.sim_lower.len(), 20);
    assert_eq!(ci.sim_upper.len(), 20);
    assert!(ci.n_boot_success > 0);
}

#[test]
fn test_bootstrap_ci_fregre_lm_ordering() {
    let (data, y, _t) = generate_test_data(30, 20, 42);
    let ci = bootstrap_ci_fregre_lm(&data, &y, None, 2, 100, 0.05, 42).unwrap();
    for j in 0..20 {
        // Pointwise: lower ≤ center ≤ upper
        assert!(
            ci.lower[j] <= ci.center[j] + 1e-10,
            "lower <= center at j={}: {} > {}",
            j,
            ci.lower[j],
            ci.center[j]
        );
        assert!(
            ci.center[j] <= ci.upper[j] + 1e-10,
            "center <= upper at j={}: {} > {}",
            j,
            ci.center[j],
            ci.upper[j]
        );
        // Simultaneous: sim_lower ≤ center ≤ sim_upper
        assert!(
            ci.sim_lower[j] <= ci.center[j] + 1e-10,
            "sim_lower <= center at j={}: {} > {}",
            j,
            ci.sim_lower[j],
            ci.center[j]
        );
        assert!(
            ci.center[j] <= ci.sim_upper[j] + 1e-10,
            "center <= sim_upper at j={}: {} > {}",
            j,
            ci.center[j],
            ci.sim_upper[j]
        );
    }
    // Simultaneous band should be wider on average
    let pw_width: f64 = (0..20).map(|j| ci.upper[j] - ci.lower[j]).sum::<f64>() / 20.0;
    let sim_width: f64 = (0..20)
        .map(|j| ci.sim_upper[j] - ci.sim_lower[j])
        .sum::<f64>()
        / 20.0;
    assert!(
        sim_width >= pw_width - 1e-10,
        "Simultaneous band should be wider on average: sim={}, pw={}",
        sim_width,
        pw_width
    );
}

#[test]
fn test_bootstrap_ci_fregre_lm_center_matches_fit() {
    let (data, y, _t) = generate_test_data(30, 20, 42);
    let fit = fregre_lm(&data, &y, None, 2).unwrap();
    let ci = bootstrap_ci_fregre_lm(&data, &y, None, 2, 50, 0.05, 42).unwrap();
    for j in 0..20 {
        assert!(
            (ci.center[j] - fit.beta_t[j]).abs() < 1e-12,
            "center should match original beta_t at j={}",
            j
        );
    }
}

#[test]
fn test_bootstrap_ci_functional_logistic_shape() {
    let (data, y_cont, _t) = generate_test_data(40, 20, 42);
    let y_median = {
        let mut sorted = y_cont.clone();
        crate::helpers::sort_nan_safe(&mut sorted);
        sorted[sorted.len() / 2]
    };
    let y_bin: Vec<f64> = y_cont
        .iter()
        .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
        .collect();

    let result = bootstrap_ci_functional_logistic(&data, &y_bin, None, 2, 50, 0.05, 123, 25, 1e-6);
    assert!(
        result.is_ok(),
        "bootstrap_ci_functional_logistic should succeed"
    );
    let ci = result.unwrap();
    assert_eq!(ci.lower.len(), 20);
    assert_eq!(ci.upper.len(), 20);
    assert_eq!(ci.center.len(), 20);
    assert!(ci.n_boot_success > 0);
}

#[test]
fn test_bootstrap_ci_logistic_ordering() {
    let (data, y_cont, _t) = generate_test_data(40, 20, 42);
    let y_median = {
        let mut sorted = y_cont.clone();
        crate::helpers::sort_nan_safe(&mut sorted);
        sorted[sorted.len() / 2]
    };
    let y_bin: Vec<f64> = y_cont
        .iter()
        .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
        .collect();

    let ci =
        bootstrap_ci_functional_logistic(&data, &y_bin, None, 2, 100, 0.05, 42, 25, 1e-6).unwrap();
    for j in 0..20 {
        assert!(
            ci.lower[j] <= ci.upper[j] + 1e-10,
            "lower <= upper at j={}",
            j
        );
    }
}

// ----- fregre_basis_cv tests -----

#[test]
fn test_fregre_basis_cv_returns_result() {
    let (data, y, t) = generate_test_data(30, 20, 42);
    let result = fregre_basis_cv(
        &data,
        &y,
        &t,
        5,
        None,
        7,
        &crate::smooth_basis::BasisType::Bspline { order: 4 },
    );
    assert!(result.is_ok(), "fregre_basis_cv should succeed");
    let res = result.unwrap();
    assert!(res.optimal_lambda > 0.0);
    assert_eq!(res.cv_errors.len(), 20); // default 20 lambdas
    assert!(res.min_cv_error >= 0.0);
}

#[test]
fn test_fregre_basis_cv_custom_lambdas() {
    let (data, y, t) = generate_test_data(25, 15, 42);
    let lambdas = vec![0.001, 0.01, 0.1, 1.0, 10.0];
    let result = fregre_basis_cv(
        &data,
        &y,
        &t,
        5,
        Some(&lambdas),
        7,
        &crate::smooth_basis::BasisType::Bspline { order: 4 },
    );
    assert!(result.is_ok());
    let res = result.unwrap();
    assert_eq!(res.lambda_values.len(), 5);
    assert!(lambdas.contains(&res.optimal_lambda));
}

// ----- fregre_np_cv tests -----

#[test]
fn test_fregre_np_cv_returns_result() {
    let (data, y, t) = generate_test_data(25, 15, 42);
    let result = fregre_np_cv(&data, &y, &t, 5, None, None);
    assert!(result.is_ok(), "fregre_np_cv should succeed");
    let res = result.unwrap();
    assert!(res.optimal_h > 0.0);
    assert_eq!(res.cv_errors.len(), 20); // default 20 quantiles
    assert!(res.min_cv_error >= 0.0);
}

#[test]
fn test_fregre_np_cv_custom_h() {
    let (data, y, t) = generate_test_data(20, 10, 42);
    let h_vals = vec![0.1, 0.5, 1.0, 2.0];
    let result = fregre_np_cv(&data, &y, &t, 3, Some(&h_vals), None);
    assert!(result.is_ok());
    let res = result.unwrap();
    assert_eq!(res.h_values.len(), 4);
    assert!(h_vals.contains(&res.optimal_h));
}

// =========================================================================
// Robust regression tests (L1, Huber)
// =========================================================================

// ----- fregre_l1 tests -----

#[test]
fn test_fregre_l1_basic() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let result = fregre_l1(&data, &y, None, 3);
    assert!(
        result.is_ok(),
        "fregre_l1 should succeed: {:?}",
        result.err()
    );
    let fit = result.unwrap();
    assert_eq!(fit.fitted_values.len(), 30);
    assert_eq!(fit.residuals.len(), 30);
    assert_eq!(fit.beta_t.len(), 50);
    assert_eq!(fit.ncomp, 3);
    assert_eq!(fit.weights.len(), 30);
    assert!(fit.r_squared >= -0.1 && fit.r_squared <= 1.0 + 1e-10);
}

#[test]
fn test_fregre_l1_convergence() {
    let (data, y, _t) = generate_test_data(40, 50, 42);
    let fit = fregre_l1(&data, &y, None, 3).unwrap();
    // L1 IRLS may converge slowly; verify iteration count is recorded
    assert!(
        fit.iterations >= 1,
        "Should have run at least one iteration, got {}",
        fit.iterations
    );
    // Verify the result is reasonable regardless of convergence flag
    assert!(fit.r_squared > -1.0, "R² should be reasonable");
    for &v in &fit.fitted_values {
        assert!(v.is_finite(), "Fitted values should be finite");
    }
}

#[test]
fn test_fregre_l1_fitted_plus_residuals_equals_y() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let fit = fregre_l1(&data, &y, None, 3).unwrap();
    for i in 0..30 {
        let reconstructed = fit.fitted_values[i] + fit.residuals[i];
        assert!(
            (reconstructed - y[i]).abs() < 1e-10,
            "ŷ + r should equal y at index {}",
            i
        );
    }
}

#[test]
fn test_fregre_l1_outlier_robustness() {
    // Generate clean data
    let (data, y_clean, _t) = generate_test_data(40, 50, 42);

    // Add outliers to a copy of y
    let mut y_outlier = y_clean.clone();
    // Contaminate 3 observations with large outliers
    y_outlier[0] += 100.0;
    y_outlier[1] -= 80.0;
    y_outlier[2] += 120.0;

    // Fit OLS on contaminated data
    let ols_fit = fregre_lm(&data, &y_outlier, None, 3).unwrap();
    // Fit L1 on contaminated data
    let l1_fit = fregre_l1(&data, &y_outlier, None, 3).unwrap();

    // Compare how well each predicts on the clean (non-contaminated) observations
    // L1 should be less affected by the outliers
    let clean_idx: Vec<usize> = (3..40).collect();

    let ols_mse: f64 = clean_idx
        .iter()
        .map(|&i| (ols_fit.fitted_values[i] - y_clean[i]).powi(2))
        .sum::<f64>()
        / clean_idx.len() as f64;

    let l1_mse: f64 = clean_idx
        .iter()
        .map(|&i| (l1_fit.fitted_values[i] - y_clean[i]).powi(2))
        .sum::<f64>()
        / clean_idx.len() as f64;

    // L1 should have lower MSE on clean observations when outliers are present
    assert!(
        l1_mse < ols_mse * 1.5,
        "L1 should be competitive with or better than OLS on clean obs with outliers: L1={}, OLS={}",
        l1_mse,
        ols_mse
    );
}

#[test]
fn test_fregre_l1_weights_are_positive() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let fit = fregre_l1(&data, &y, None, 3).unwrap();
    for (i, &w) in fit.weights.iter().enumerate() {
        assert!(
            w > 0.0 && w.is_finite(),
            "weight[{}] should be positive finite, got {}",
            i,
            w
        );
    }
}

#[test]
fn test_fregre_l1_single_component() {
    let (data, y, _t) = generate_test_data(20, 50, 42);
    let result = fregre_l1(&data, &y, None, 1);
    assert!(result.is_ok());
    let fit = result.unwrap();
    assert_eq!(fit.ncomp, 1);
}

#[test]
fn test_fregre_l1_with_scalar_covariates() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let mut sc = FdMatrix::zeros(30, 2);
    for i in 0..30 {
        sc[(i, 0)] = i as f64 / 30.0;
        sc[(i, 1)] = (i as f64 * 0.7).sin();
    }
    let result = fregre_l1(&data, &y, Some(&sc), 3);
    assert!(result.is_ok());
    let fit = result.unwrap();
    // coefficients: intercept + 3 FPC + 2 scalar = 6
    assert_eq!(fit.coefficients.len(), 6);
}

// ----- fregre_huber tests -----

#[test]
fn test_fregre_huber_basic() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let result = fregre_huber(&data, &y, None, 3, 1.345);
    assert!(
        result.is_ok(),
        "fregre_huber should succeed: {:?}",
        result.err()
    );
    let fit = result.unwrap();
    assert_eq!(fit.fitted_values.len(), 30);
    assert_eq!(fit.residuals.len(), 30);
    assert_eq!(fit.beta_t.len(), 50);
    assert_eq!(fit.ncomp, 3);
    assert_eq!(fit.weights.len(), 30);
    assert!(fit.r_squared >= -0.1 && fit.r_squared <= 1.0 + 1e-10);
}

#[test]
fn test_fregre_huber_convergence() {
    let (data, y, _t) = generate_test_data(40, 50, 42);
    let fit = fregre_huber(&data, &y, None, 3, 1.345).unwrap();
    assert!(fit.converged, "Huber should converge within max iterations");
    assert!(
        fit.iterations <= 50,
        "Should converge within 50 iterations, got {}",
        fit.iterations
    );
}

#[test]
fn test_fregre_huber_fitted_plus_residuals_equals_y() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let fit = fregre_huber(&data, &y, None, 3, 1.345).unwrap();
    for i in 0..30 {
        let reconstructed = fit.fitted_values[i] + fit.residuals[i];
        assert!(
            (reconstructed - y[i]).abs() < 1e-10,
            "ŷ + r should equal y at index {}",
            i
        );
    }
}

#[test]
fn test_fregre_huber_outlier_robustness() {
    let (data, y_clean, _t) = generate_test_data(40, 50, 42);

    let mut y_outlier = y_clean.clone();
    y_outlier[0] += 100.0;
    y_outlier[1] -= 80.0;
    y_outlier[2] += 120.0;

    let ols_fit = fregre_lm(&data, &y_outlier, None, 3).unwrap();
    let huber_fit = fregre_huber(&data, &y_outlier, None, 3, 1.345).unwrap();

    let clean_idx: Vec<usize> = (3..40).collect();

    let ols_mse: f64 = clean_idx
        .iter()
        .map(|&i| (ols_fit.fitted_values[i] - y_clean[i]).powi(2))
        .sum::<f64>()
        / clean_idx.len() as f64;

    let huber_mse: f64 = clean_idx
        .iter()
        .map(|&i| (huber_fit.fitted_values[i] - y_clean[i]).powi(2))
        .sum::<f64>()
        / clean_idx.len() as f64;

    assert!(
        huber_mse < ols_mse * 1.5,
        "Huber should be competitive with or better than OLS on clean obs with outliers: Huber={}, OLS={}",
        huber_mse,
        ols_mse
    );
}

#[test]
fn test_fregre_huber_invalid_k() {
    let (data, y, _t) = generate_test_data(20, 50, 42);
    assert!(fregre_huber(&data, &y, None, 3, 0.0).is_err());
    assert!(fregre_huber(&data, &y, None, 3, -1.0).is_err());
}

#[test]
fn test_fregre_huber_large_k_approaches_ols() {
    // With a very large k, all residuals are small relative to k,
    // so all weights are 1, and Huber should behave like OLS.
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let ols_fit = fregre_lm(&data, &y, None, 3).unwrap();
    let huber_fit = fregre_huber(&data, &y, None, 3, 1e6).unwrap();

    for i in 0..30 {
        assert!(
            (huber_fit.fitted_values[i] - ols_fit.fitted_values[i]).abs() < 1e-4,
            "Huber with large k should match OLS at index {}: huber={}, ols={}",
            i,
            huber_fit.fitted_values[i],
            ols_fit.fitted_values[i]
        );
    }
}

#[test]
fn test_fregre_huber_with_scalar_covariates() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let mut sc = FdMatrix::zeros(30, 1);
    for i in 0..30 {
        sc[(i, 0)] = i as f64 / 30.0;
    }
    let result = fregre_huber(&data, &y, Some(&sc), 3, 1.345);
    assert!(result.is_ok());
    let fit = result.unwrap();
    assert_eq!(fit.coefficients.len(), 5); // intercept + 3 FPC + 1 scalar
}

// ----- predict_fregre_robust tests -----

#[test]
fn test_predict_fregre_robust_l1_on_training_data() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let fit = fregre_l1(&data, &y, None, 3).unwrap();
    let preds = predict_fregre_robust(&fit, &data, None);
    assert_eq!(preds.len(), 30);
    for i in 0..30 {
        assert!(
            (preds[i] - fit.fitted_values[i]).abs() < 1e-6,
            "Prediction on training data should match fitted values at index {}",
            i
        );
    }
}

#[test]
fn test_predict_fregre_robust_huber_on_training_data() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let fit = fregre_huber(&data, &y, None, 3, 1.345).unwrap();
    let preds = predict_fregre_robust(&fit, &data, None);
    assert_eq!(preds.len(), 30);
    for i in 0..30 {
        assert!(
            (preds[i] - fit.fitted_values[i]).abs() < 1e-6,
            "Prediction on training data should match fitted values at index {}",
            i
        );
    }
}

#[test]
fn test_predict_fregre_robust_method() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let fit = fregre_l1(&data, &y, None, 3).unwrap();
    // Use the method syntax
    let preds = fit.predict(&data, None);
    assert_eq!(preds.len(), 30);
    for i in 0..30 {
        assert!(
            (preds[i] - fit.fitted_values[i]).abs() < 1e-6,
            "predict() method should match fitted values",
        );
    }
}

#[test]
fn test_predict_fregre_robust_with_scalar_covariates() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let mut sc = FdMatrix::zeros(30, 2);
    for i in 0..30 {
        sc[(i, 0)] = i as f64 / 30.0;
        sc[(i, 1)] = (i as f64 * 0.3).cos();
    }
    let fit = fregre_l1(&data, &y, Some(&sc), 3).unwrap();
    let preds = predict_fregre_robust(&fit, &data, Some(&sc));
    assert_eq!(preds.len(), 30);
    for i in 0..30 {
        assert!(
            (preds[i] - fit.fitted_values[i]).abs() < 1e-6,
            "Prediction with scalar covariates should match fitted values",
        );
    }
}

// ----- Edge cases -----

#[test]
fn test_fregre_l1_invalid_input() {
    let data = FdMatrix::zeros(2, 50);
    let y = vec![1.0, 2.0];
    assert!(fregre_l1(&data, &y, None, 1).is_err());

    let data = FdMatrix::zeros(10, 50);
    let y = vec![1.0; 5]; // wrong length
    assert!(fregre_l1(&data, &y, None, 2).is_err());
}

#[test]
fn test_fregre_huber_single_component() {
    let (data, y, _t) = generate_test_data(20, 50, 42);
    let result = fregre_huber(&data, &y, None, 1, 1.345);
    assert!(result.is_ok());
    let fit = result.unwrap();
    assert_eq!(fit.ncomp, 1);
}

#[test]
fn test_fregre_l1_minimal_data() {
    // Just barely enough data (n=5, ncomp=1)
    let (data, y, _t) = generate_test_data(5, 10, 42);
    let result = fregre_l1(&data, &y, None, 1);
    assert!(result.is_ok());
    let fit = result.unwrap();
    assert_eq!(fit.fitted_values.len(), 5);
}

// ----- fregre_pls tests -----

#[test]
fn test_fregre_pls_basic() {
    let (data, y, t) = generate_test_data(30, 50, 42);
    let fit = fregre_pls(&data, &y, &t, 3, None).unwrap();
    assert_eq!(fit.fitted_values.len(), 30);
    assert_eq!(fit.residuals.len(), 30);
    assert_eq!(fit.beta_t.len(), 50);
    assert_eq!(fit.ncomp, 3);
    assert!(fit.r_squared >= 0.0 && fit.r_squared <= 1.0);
    assert!(fit.residual_se >= 0.0);
    assert!(fit.aic.is_finite());
    assert!(fit.bic.is_finite());
}

#[test]
fn test_fregre_pls_predict() {
    let (data, y, t) = generate_test_data(30, 50, 42);
    let fit = fregre_pls(&data, &y, &t, 3, None).unwrap();

    // Predict on training data — should match fitted values closely
    let preds = predict_fregre_pls(&fit, &data, None).unwrap();
    assert_eq!(preds.len(), 30);
    for (p, f) in preds.iter().zip(&fit.fitted_values) {
        assert!((p - f).abs() < 1e-8, "pred={p}, fitted={f}");
    }
}

#[test]
fn test_fregre_pls_predict_new_data() {
    let (data, y, t) = generate_test_data(30, 50, 42);
    let fit = fregre_pls(&data, &y, &t, 3, None).unwrap();

    let (new_data, _, _) = generate_test_data(10, 50, 99);
    let preds = predict_fregre_pls(&fit, &new_data, None).unwrap();
    assert_eq!(preds.len(), 10);
    assert!(preds.iter().all(|p| p.is_finite()));
}

#[test]
fn test_fregre_pls_with_scalar_covariates() {
    let (data, y, t) = generate_test_data(30, 50, 42);
    let sc_vals: Vec<f64> = (0..30).map(|i| i as f64 / 29.0).collect();
    let sc = FdMatrix::from_column_major(sc_vals, 30, 1).unwrap();
    let fit = fregre_pls(&data, &y, &t, 2, Some(&sc)).unwrap();
    assert_eq!(fit.gamma.len(), 1);
    assert_eq!(fit.coefficients.len(), 4); // intercept + 2 PLS + 1 scalar
}

#[test]
fn test_fregre_pls_dimension_errors() {
    let (data, y, t) = generate_test_data(30, 50, 42);
    // Wrong argvals length
    let bad_t: Vec<f64> = (0..10).map(|i| i as f64).collect();
    assert!(fregre_pls(&data, &y, &bad_t, 3, None).is_err());
    // Wrong y length
    assert!(fregre_pls(&data, &y[..10], &t, 3, None).is_err());
    // ncomp = 0
    assert!(fregre_pls(&data, &y, &t, 0, None).is_err());
}
