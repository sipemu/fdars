use super::*;
use std::f64::consts::PI;

fn generate_test_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<f64>) {
    let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0; n];

    for i in 0..n {
        let amp = 1.0 + 0.5 * (i as f64 / n as f64);
        let shift = 0.1 * (i as f64 - n as f64 / 2.0);
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * (t[j] + shift)).sin();
        }
        y[i] = amp; // Response related to amplitude
    }
    (data, y, t)
}

#[test]
fn test_elastic_regression_basic() {
    let (data, y, t) = generate_test_data(15, 51);
    let result = elastic_regression(&data, &y, &t, 10, 1e-3, 5, 1e-3);
    assert!(result.is_ok(), "elastic_regression should succeed");

    let res = result.unwrap();
    assert_eq!(res.fitted_values.len(), 15);
    assert_eq!(res.beta.len(), 51);
    assert_eq!(res.gammas.shape(), (15, 51));
    assert!(res.n_iter > 0);
}

#[test]
fn test_elastic_logistic_basic() {
    let n = 20;
    let m = 51;
    let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0_i8; n];

    for i in 0..n {
        let label = if i < n / 2 { -1_i8 } else { 1_i8 };
        y[i] = label;
        let amp = if label == 1 { 2.0 } else { 1.0 };
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * t[j]).sin();
        }
    }

    let result = elastic_logistic(&data, &y, &t, 10, 1e-2, 5, 1e-3);
    assert!(result.is_ok(), "elastic_logistic should succeed");

    let res = result.unwrap();
    assert_eq!(res.probabilities.len(), n);
    assert_eq!(res.predicted_classes.len(), n);
    assert!(res.accuracy >= 0.0 && res.accuracy <= 1.0);
}

#[test]
fn test_elastic_pcr_vertical() {
    let (data, y, t) = generate_test_data(15, 51);
    let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Vertical, 0.0, 5, 1e-3);
    assert!(result.is_ok(), "elastic_pcr (vertical) should succeed");

    let res = result.unwrap();
    assert_eq!(res.fitted_values.len(), 15);
    assert_eq!(res.coefficients.len(), 3);
}

#[test]
fn test_elastic_pcr_horizontal() {
    let (data, y, t) = generate_test_data(15, 51);
    let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Horizontal, 0.0, 5, 1e-3);
    assert!(result.is_ok(), "elastic_pcr (horizontal) should succeed");
}

#[test]
fn test_elastic_regression_invalid() {
    let data = FdMatrix::zeros(1, 10);
    let y = vec![1.0];
    let t: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
    assert!(elastic_regression(&data, &y, &t, 5, 1e-3, 5, 1e-3).is_err());
}

#[test]
fn test_predict_elastic_regression() {
    let (data, y, t) = generate_test_data(15, 51);
    let fit = elastic_regression(&data, &y, &t, 10, 1e-3, 5, 1e-3)
        .expect("elastic_regression should succeed");

    // Standalone function
    let preds = predict_elastic_regression(&fit, &data, &t);
    assert_eq!(preds.len(), 15);
    assert!(preds.iter().all(|v| v.is_finite()));

    // Method syntax
    let preds2 = fit.predict(&data, &t);
    assert_eq!(preds, preds2);
}

#[test]
fn test_predict_elastic_logistic() {
    let n = 20;
    let m = 51;
    let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0_i8; n];

    for i in 0..n {
        let label = if i < n / 2 { -1_i8 } else { 1_i8 };
        y[i] = label;
        let amp = if label == 1 { 2.0 } else { 1.0 };
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * t[j]).sin();
        }
    }

    let fit = elastic_logistic(&data, &y, &t, 10, 1e-2, 5, 1e-3)
        .expect("elastic_logistic should succeed");

    // Standalone function
    let probs = predict_elastic_logistic(&fit, &data, &t);
    assert_eq!(probs.len(), n);
    assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));

    // Method syntax
    let probs2 = fit.predict(&data, &t);
    assert_eq!(probs, probs2);
}

#[test]
fn test_elastic_pcr_joint() {
    let (data, y, t) = generate_test_data(15, 51);
    let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Joint, 0.0, 5, 1e-3);
    assert!(result.is_ok(), "elastic_pcr (joint) should succeed");

    let res = result.unwrap();
    assert_eq!(res.fitted_values.len(), 15);
    assert_eq!(res.coefficients.len(), 3);
    assert!(res.joint_fpca.is_some());
}

#[test]
fn test_elastic_logistic_probability_bounds() {
    let n = 20;
    let m = 51;
    let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0_i8; n];

    for i in 0..n {
        let label = if i < n / 2 { -1_i8 } else { 1_i8 };
        y[i] = label;
        let amp = if label == 1 { 2.0 } else { 1.0 };
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * t[j]).sin();
        }
    }

    let res = elastic_logistic(&data, &y, &t, 10, 1e-2, 5, 1e-3)
        .expect("elastic_logistic should succeed");

    // All probabilities in [0, 1]
    assert!(res.probabilities.iter().all(|&p| (0.0..=1.0).contains(&p)));

    // All predicted classes in {0, 1}
    assert!(res.predicted_classes.iter().all(|&c| c == 0 || c == 1));

    // Accuracy in [0, 1]
    assert!((0.0..=1.0).contains(&res.accuracy));
}

// -- Config variant tests ──────────────────────────────────────────────

#[test]
fn test_elastic_regression_with_default_config() {
    let (data, y, t) = generate_test_data(15, 51);
    let config = ElasticConfig::default();
    let result = elastic_regression_with_config(&data, &y, &t, &config);
    assert!(
        result.is_ok(),
        "elastic_regression_with_config (default) should succeed"
    );

    let res = result.unwrap();
    assert_eq!(res.fitted_values.len(), 15);
    assert_eq!(res.beta.len(), 51);
    assert!(res.n_iter > 0);
}

#[test]
fn test_elastic_regression_with_custom_config() {
    let (data, y, t) = generate_test_data(15, 51);
    let config = ElasticConfig {
        ncomp_beta: 8,
        lambda: 1e-2,
        max_iter: 10,
        tol: 1e-3,
    };
    let result = elastic_regression_with_config(&data, &y, &t, &config);
    assert!(
        result.is_ok(),
        "elastic_regression_with_config (custom) should succeed"
    );

    let res = result.unwrap();
    assert_eq!(res.fitted_values.len(), 15);
    assert_eq!(res.beta.len(), 51);
    assert!(res.fitted_values.iter().all(|v| v.is_finite()));
}

#[test]
fn test_elastic_regression_config_matches_direct() {
    let (data, y, t) = generate_test_data(15, 51);
    let config = ElasticConfig {
        ncomp_beta: 10,
        lambda: 1e-3,
        max_iter: 5,
        tol: 1e-3,
    };
    let res_config = elastic_regression_with_config(&data, &y, &t, &config).unwrap();
    let res_direct = elastic_regression(&data, &y, &t, 10, 1e-3, 5, 1e-3).unwrap();

    // Both should produce identical results
    assert_eq!(
        res_config.fitted_values.len(),
        res_direct.fitted_values.len()
    );
    for (a, b) in res_config
        .fitted_values
        .iter()
        .zip(&res_direct.fitted_values)
    {
        assert!(
            (a - b).abs() < 1e-10,
            "Config and direct calls should produce identical results"
        );
    }
    assert!((res_config.alpha - res_direct.alpha).abs() < 1e-10);
}

#[test]
fn test_elastic_logistic_with_default_config() {
    let n = 20;
    let m = 51;
    let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0_i8; n];
    for i in 0..n {
        let label = if i < n / 2 { -1_i8 } else { 1_i8 };
        y[i] = label;
        let amp = if label == 1 { 2.0 } else { 1.0 };
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * t[j]).sin();
        }
    }

    let config = ElasticConfig::default();
    let result = elastic_logistic_with_config(&data, &y, &t, &config);
    assert!(
        result.is_ok(),
        "elastic_logistic_with_config (default) should succeed"
    );

    let res = result.unwrap();
    assert_eq!(res.probabilities.len(), n);
    assert!(res.probabilities.iter().all(|&p| (0.0..=1.0).contains(&p)));
    assert!(res.predicted_classes.iter().all(|&c| c == 0 || c == 1));
}

#[test]
fn test_elastic_logistic_with_custom_config() {
    let n = 20;
    let m = 51;
    let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0_i8; n];
    for i in 0..n {
        let label = if i < n / 2 { -1_i8 } else { 1_i8 };
        y[i] = label;
        let amp = if label == 1 { 2.0 } else { 1.0 };
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * t[j]).sin();
        }
    }

    let config = ElasticConfig {
        ncomp_beta: 8,
        lambda: 0.05,
        max_iter: 10,
        tol: 1e-3,
    };
    let result = elastic_logistic_with_config(&data, &y, &t, &config);
    assert!(
        result.is_ok(),
        "elastic_logistic_with_config (custom) should succeed"
    );

    let res = result.unwrap();
    assert_eq!(res.probabilities.len(), n);
    assert!((0.0..=1.0).contains(&res.accuracy));
}

#[test]
fn test_elastic_logistic_config_matches_direct() {
    let n = 20;
    let m = 51;
    let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0_i8; n];
    for i in 0..n {
        let label = if i < n / 2 { -1_i8 } else { 1_i8 };
        y[i] = label;
        let amp = if label == 1 { 2.0 } else { 1.0 };
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * t[j]).sin();
        }
    }

    let config = ElasticConfig {
        ncomp_beta: 10,
        lambda: 1e-2,
        max_iter: 5,
        tol: 1e-3,
    };
    let res_config = elastic_logistic_with_config(&data, &y, &t, &config).unwrap();
    let res_direct = elastic_logistic(&data, &y, &t, 10, 1e-2, 5, 1e-3).unwrap();

    assert_eq!(
        res_config.probabilities.len(),
        res_direct.probabilities.len()
    );
    for (a, b) in res_config
        .probabilities
        .iter()
        .zip(&res_direct.probabilities)
    {
        assert!(
            (a - b).abs() < 1e-10,
            "Config and direct calls should produce identical results"
        );
    }
}

#[test]
fn test_elastic_pcr_with_default_config() {
    let (data, y, t) = generate_test_data(15, 51);
    let config = ElasticPcrConfig::default();
    let result = elastic_pcr_with_config(&data, &y, &t, &config);
    assert!(
        result.is_ok(),
        "elastic_pcr_with_config (default) should succeed"
    );

    let res = result.unwrap();
    assert_eq!(res.fitted_values.len(), 15);
    assert_eq!(res.coefficients.len(), 3); // default ncomp
    assert!(res.fitted_values.iter().all(|v| v.is_finite()));
}

#[test]
fn test_elastic_pcr_with_custom_config() {
    let (data, y, t) = generate_test_data(15, 51);
    let config = ElasticPcrConfig {
        ncomp: 2,
        pca_method: PcaMethod::Horizontal,
        lambda: 0.01,
        max_iter: 10,
        tol: 1e-3,
    };
    let result = elastic_pcr_with_config(&data, &y, &t, &config);
    assert!(
        result.is_ok(),
        "elastic_pcr_with_config (custom horizontal) should succeed"
    );

    let res = result.unwrap();
    assert_eq!(res.fitted_values.len(), 15);
    assert!(res.horiz_fpca.is_some());
}

#[test]
fn test_elastic_pcr_config_matches_direct() {
    let (data, y, t) = generate_test_data(15, 51);
    let config = ElasticPcrConfig {
        ncomp: 3,
        pca_method: PcaMethod::Vertical,
        lambda: 0.0,
        max_iter: 5,
        tol: 1e-3,
    };
    let res_config = elastic_pcr_with_config(&data, &y, &t, &config).unwrap();
    let res_direct = elastic_pcr(&data, &y, &t, 3, PcaMethod::Vertical, 0.0, 5, 1e-3).unwrap();

    assert_eq!(
        res_config.fitted_values.len(),
        res_direct.fitted_values.len()
    );
    for (a, b) in res_config
        .fitted_values
        .iter()
        .zip(&res_direct.fitted_values)
    {
        assert!(
            (a - b).abs() < 1e-10,
            "Config and direct calls should produce identical results"
        );
    }
    assert!((res_config.alpha - res_direct.alpha).abs() < 1e-10);
}

#[test]
fn test_elastic_pcr_with_joint_config() {
    let (data, y, t) = generate_test_data(15, 51);
    let config = ElasticPcrConfig {
        ncomp: 2,
        pca_method: PcaMethod::Joint,
        lambda: 0.0,
        max_iter: 5,
        tol: 1e-3,
    };
    let result = elastic_pcr_with_config(&data, &y, &t, &config);
    assert!(
        result.is_ok(),
        "elastic_pcr_with_config (joint) should succeed"
    );

    let res = result.unwrap();
    assert!(res.joint_fpca.is_some());
    assert_eq!(res.pca_method, PcaMethod::Joint);
}
