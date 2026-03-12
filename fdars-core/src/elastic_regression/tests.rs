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
