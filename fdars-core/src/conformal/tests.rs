use super::*;
use crate::classification::fclassif_lda_fit;
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{fregre_lm, predict_fregre_lm};
use rand::prelude::*;
use std::f64::consts::PI;

fn make_test_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>, FdMatrix) {
    let mut rng = StdRng::seed_from_u64(seed);
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0; n];
    for i in 0..n {
        let a = rng.gen::<f64>() * 2.0 - 1.0;
        let b = rng.gen::<f64>() * 2.0 - 1.0;
        for j in 0..m {
            data[(i, j)] = a * (2.0 * PI * argvals[j]).sin()
                + b * (4.0 * PI * argvals[j]).cos()
                + 0.1 * rng.gen::<f64>();
        }
        y[i] = 2.0 * a + 3.0 * b + 0.5 * rng.gen::<f64>();
    }
    let n_test = 5;
    let mut test_data = FdMatrix::zeros(n_test, m);
    for i in 0..n_test {
        let a = rng.gen::<f64>() * 2.0 - 1.0;
        let b = rng.gen::<f64>() * 2.0 - 1.0;
        for j in 0..m {
            test_data[(i, j)] = a * (2.0 * PI * argvals[j]).sin()
                + b * (4.0 * PI * argvals[j]).cos()
                + 0.1 * rng.gen::<f64>();
        }
    }
    (data, y, test_data)
}

fn make_classif_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<usize>, FdMatrix) {
    let mut rng = StdRng::seed_from_u64(seed);
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0usize; n];
    for i in 0..n {
        let class = if i < n / 2 { 0 } else { 1 };
        y[i] = class;
        let offset = if class == 0 { -1.0 } else { 1.0 };
        for j in 0..m {
            data[(i, j)] = offset * (2.0 * PI * argvals[j]).sin() + 0.3 * rng.gen::<f64>();
        }
    }
    let n_test = 4;
    let mut test_data = FdMatrix::zeros(n_test, m);
    for i in 0..n_test {
        let offset = if i < 2 { -1.0 } else { 1.0 };
        for j in 0..m {
            test_data[(i, j)] = offset * (2.0 * PI * argvals[j]).sin() + 0.3 * rng.gen::<f64>();
        }
    }
    (data, y, test_data)
}

// -- Core helper tests ────────────────────────────────────────────────

#[test]
fn test_conformal_split_sizes() {
    let (proper, cal) = conformal_split(100, 0.2, 42);
    assert_eq!(proper.len() + cal.len(), 100);
    assert!((cal.len() as f64 - 20.0).abs() <= 2.0);
}

#[test]
fn test_conformal_quantile_monotonic() {
    let mut scores: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
    let q1 = conformal_quantile(&mut scores.clone(), 0.1);
    let q2 = conformal_quantile(&mut scores, 0.2);
    assert!(
        q1 >= q2,
        "Lower alpha should give wider intervals (higher quantile)"
    );
}

#[test]
fn test_lac_and_aps_scores() {
    let probs = vec![0.7, 0.2, 0.1];
    assert!((lac_score(&probs, 0) - 0.3).abs() < 1e-10);
    assert!((lac_score(&probs, 1) - 0.8).abs() < 1e-10);

    // APS: for true class 0, sorted order is [0, 1, 2], cumulative at class 0 = 0.7
    let aps0 = aps_score(&probs, 0);
    assert!((aps0 - 0.7).abs() < 1e-10);

    // APS: for true class 2, sorted order is [0, 1, 2], cumulative at class 2 = 1.0
    let aps2 = aps_score(&probs, 2);
    assert!((aps2 - 1.0).abs() < 1e-10);
}

#[test]
fn test_prediction_sets_lac() {
    let probs = vec![0.7, 0.2, 0.1];
    // quantile = 0.5: include class k if 1 - P(k) <= 0.5 -> P(k) >= 0.5 -> only class 0
    let set = lac_prediction_set(&probs, 0.5);
    assert_eq!(set, vec![0]);

    // quantile = 0.9: include class k if 1 - P(k) <= 0.9 -> P(k) >= 0.1 -> all classes
    let set = lac_prediction_set(&probs, 0.9);
    assert_eq!(set, vec![0, 1, 2]);
}

#[test]
fn test_prediction_sets_aps() {
    let probs = vec![0.7, 0.2, 0.1];
    // quantile = 0.5: include classes until cumulative >= 0.5
    // Sorted: [0(0.7), 1(0.2), 2(0.1)]. Cumulative: 0.7 >= 0.5 -> {0}
    let set = aps_prediction_set(&probs, 0.5);
    assert_eq!(set, vec![0]);

    // quantile = 0.85: include classes until cumulative >= 0.85
    // Sorted: [0(0.7), 1(0.2), 2(0.1)]. 0.7 < 0.85, add 1: 0.9 >= 0.85 -> {0, 1}
    let set = aps_prediction_set(&probs, 0.85);
    assert_eq!(set, vec![0, 1]);

    // quantile = 0.95: include classes until cumulative >= 0.95
    // 0.7 < 0.95, 0.9 < 0.95, 1.0 >= 0.95 -> {0, 1, 2}
    let set = aps_prediction_set(&probs, 0.95);
    assert_eq!(set, vec![0, 1, 2]);
}

// -- Regression integration tests ─────────────────────────────────────

#[test]
fn test_conformal_fregre_lm_basic() {
    let (data, y, test_data) = make_test_data(40, 20, 42);
    let result = conformal_fregre_lm(&data, &y, &test_data, None, None, 3, 0.3, 0.1, 42);
    let r = result.unwrap();
    assert_eq!(r.predictions.len(), 5);
    assert_eq!(r.lower.len(), 5);
    assert_eq!(r.upper.len(), 5);
    // Intervals should have positive width
    for i in 0..5 {
        assert!(r.upper[i] > r.lower[i]);
    }
    // Coverage on calibration set should be reasonable
    assert!(r.coverage >= 0.5);
}

#[test]
fn test_conformal_fregre_np_basic() {
    let (data, y, test_data) = make_test_data(30, 15, 123);
    let argvals: Vec<f64> = (0..15).map(|j| j as f64 / 14.0).collect();
    let result = conformal_fregre_np(
        &data, &y, &test_data, &argvals, None, None, 1.0, 1.0, 0.3, 0.1, 123,
    );
    let r = result.unwrap();
    assert_eq!(r.predictions.len(), 5);
    for i in 0..5 {
        assert!(r.upper[i] > r.lower[i]);
    }
}

// -- Classification integration tests ─────────────────────────────────

#[test]
fn test_conformal_classif_lda() {
    let (data, y, test_data) = make_classif_data(40, 20, 42);
    let result = conformal_classif(
        &data,
        &y,
        &test_data,
        None,
        None,
        3,
        "lda",
        5,
        ClassificationScore::Lac,
        0.3,
        0.1,
        42,
    );
    let r = result.unwrap();
    assert_eq!(r.prediction_sets.len(), 4);
    // All prediction sets should be non-empty
    for set in &r.prediction_sets {
        assert!(!set.is_empty());
    }
    assert!(r.average_set_size >= 1.0);
}

#[test]
fn test_conformal_classif_aps() {
    let (data, y, test_data) = make_classif_data(40, 20, 42);
    let result = conformal_classif(
        &data,
        &y,
        &test_data,
        None,
        None,
        3,
        "lda",
        5,
        ClassificationScore::Aps,
        0.3,
        0.1,
        42,
    );
    let r = result.unwrap();
    assert_eq!(r.prediction_sets.len(), 4);
    for set in &r.prediction_sets {
        assert!(!set.is_empty());
    }
}

#[test]
fn test_conformal_logistic_basic() {
    let (data, y_usize, test_data) = make_classif_data(40, 20, 42);
    let y: Vec<f64> = y_usize.iter().map(|&c| c as f64).collect();
    let result = conformal_logistic(
        &data,
        &y,
        &test_data,
        None,
        None,
        3,
        100,
        1e-4,
        ClassificationScore::Lac,
        0.3,
        0.1,
        42,
    );
    let r = result.unwrap();
    assert_eq!(r.prediction_sets.len(), 4);
    for set in &r.prediction_sets {
        assert!(!set.is_empty());
        // Binary: set size should be 1 or 2
        assert!(set.len() <= 2);
    }
}

// -- Generic conformal tests ──────────────────────────────────────────

#[test]
fn test_conformal_generic_regression() {
    let (data, y, test_data) = make_test_data(40, 20, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let result =
        conformal_generic_regression(&fit, &data, &y, &test_data, None, None, 0.3, 0.1, 42);
    let r = result.unwrap();
    assert_eq!(r.predictions.len(), 5);
    for i in 0..5 {
        assert!(r.upper[i] > r.lower[i]);
    }
}

#[test]
fn test_conformal_generic_classification() {
    let (data, y, test_data) = make_classif_data(40, 20, 42);
    let fit = fclassif_lda_fit(&data, &y, None, 3).unwrap();
    let result = conformal_generic_classification(
        &fit,
        &data,
        &y,
        &test_data,
        None,
        None,
        ClassificationScore::Lac,
        0.3,
        0.1,
        42,
    );
    let r = result.unwrap();
    assert_eq!(r.prediction_sets.len(), 4);
    for set in &r.prediction_sets {
        assert!(!set.is_empty());
    }
}

// -- CV+ conformal tests ──────────────────────────────────────────────

#[test]
fn test_cv_conformal_regression() {
    let (data, y, test_data) = make_test_data(40, 20, 42);
    let result = cv_conformal_regression(
        &data,
        &y,
        &test_data,
        None,
        None,
        |train_d, train_y, _train_sc, pred_d, _pred_sc| {
            let fit = fregre_lm(train_d, train_y, None, 3).ok()?;
            let cal = predict_fregre_lm(&fit, pred_d, None);
            let test = predict_fregre_lm(&fit, pred_d, None);
            Some((cal, test))
        },
        5,
        0.1,
        42,
    );
    let r = result.unwrap();
    assert_eq!(r.predictions.len(), test_data.nrows());
    for i in 0..r.predictions.len() {
        assert!(r.upper[i] > r.lower[i]);
    }
}

// -- Validation tests ─────────────────────────────────────────────────

#[test]
fn test_invalid_inputs() {
    let data = FdMatrix::zeros(2, 5);
    let y = vec![1.0, 2.0];
    let test = FdMatrix::zeros(1, 5);
    // Too few observations
    assert!(conformal_fregre_lm(&data, &y, &test, None, None, 1, 0.3, 0.1, 42).is_err());

    // Invalid alpha
    let (data, y, test) = make_test_data(20, 10, 42);
    assert!(conformal_fregre_lm(&data, &y, &test, None, None, 2, 0.3, 0.0, 42).is_err());
    assert!(conformal_fregre_lm(&data, &y, &test, None, None, 2, 0.3, 1.0, 42).is_err());
}

#[test]
fn test_alpha_affects_interval_width() {
    let (data, y, test_data) = make_test_data(40, 20, 42);
    let r1 = conformal_fregre_lm(&data, &y, &test_data, None, None, 3, 0.3, 0.1, 42).unwrap();
    let r2 = conformal_fregre_lm(&data, &y, &test_data, None, None, 3, 0.3, 0.3, 42).unwrap();
    // Wider alpha -> narrower intervals (lower quantile)
    assert!(r1.residual_quantile >= r2.residual_quantile);
}

// -- Elastic conformal regression tests ────────────────────────────────

fn make_elastic_test_data(
    n: usize,
    m: usize,
    seed: u64,
) -> (FdMatrix, Vec<f64>, FdMatrix, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0; n];
    for i in 0..n {
        let amp = 1.0 + 0.5 * (i as f64 / n as f64);
        let shift = 0.1 * rng.gen::<f64>();
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * (argvals[j] + shift)).sin() + 0.05 * rng.gen::<f64>();
        }
        y[i] = amp + 0.1 * rng.gen::<f64>();
    }
    let n_test = 5;
    let mut test_data = FdMatrix::zeros(n_test, m);
    for i in 0..n_test {
        let amp = 1.0 + 0.3 * rng.gen::<f64>();
        let shift = 0.1 * rng.gen::<f64>();
        for j in 0..m {
            test_data[(i, j)] =
                amp * (2.0 * PI * (argvals[j] + shift)).sin() + 0.05 * rng.gen::<f64>();
        }
    }
    (data, y, test_data, argvals)
}

fn make_elastic_classif_data(
    n: usize,
    m: usize,
    seed: u64,
) -> (FdMatrix, Vec<i8>, FdMatrix, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0_i8; n];
    for i in 0..n {
        let label = if i < n / 2 { -1_i8 } else { 1_i8 };
        y[i] = label;
        let amp = if label == 1 { 2.0 } else { 1.0 };
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * argvals[j]).sin() + 0.1 * rng.gen::<f64>();
        }
    }
    let n_test = 4;
    let mut test_data = FdMatrix::zeros(n_test, m);
    for i in 0..n_test {
        let amp = if i < 2 { 1.0 } else { 2.0 };
        for j in 0..m {
            test_data[(i, j)] = amp * (2.0 * PI * argvals[j]).sin() + 0.1 * rng.gen::<f64>();
        }
    }
    (data, y, test_data, argvals)
}

#[test]
fn test_conformal_elastic_regression_basic() {
    let (data, y, test_data, argvals) = make_elastic_test_data(20, 51, 42);
    let r = conformal_elastic_regression(&data, &y, &test_data, &argvals, 10, 1e-3, 0.3, 0.1, 42)
        .unwrap();
    assert_eq!(r.predictions.len(), 5);
    assert_eq!(r.lower.len(), 5);
    assert_eq!(r.upper.len(), 5);
    for i in 0..5 {
        assert!(
            r.upper[i] > r.lower[i],
            "Interval must have positive width at index {i}"
        );
        assert!(r.predictions[i].is_finite());
    }
    assert!(r.coverage >= 0.0 && r.coverage <= 1.0);
    assert!(!r.calibration_scores.is_empty());
}

#[test]
fn test_conformal_elastic_regression_dimension_mismatch_y() {
    let (data, _y, test_data, argvals) = make_elastic_test_data(20, 51, 42);
    let wrong_y = vec![1.0; 10]; // wrong length
    let result = conformal_elastic_regression(
        &data, &wrong_y, &test_data, &argvals, 10, 1e-3, 0.3, 0.1, 42,
    );
    assert!(result.is_err());
}

#[test]
fn test_conformal_elastic_regression_dimension_mismatch_cols() {
    let (data, y, _test_data, argvals) = make_elastic_test_data(20, 51, 42);
    let bad_test = FdMatrix::zeros(3, 30); // different ncols
    let result =
        conformal_elastic_regression(&data, &y, &bad_test, &argvals, 10, 1e-3, 0.3, 0.1, 42);
    assert!(result.is_err());
}

#[test]
fn test_conformal_elastic_regression_too_few_obs() {
    let data = FdMatrix::zeros(3, 10);
    let y = vec![1.0; 3];
    let test = FdMatrix::zeros(2, 10);
    let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
    let result = conformal_elastic_regression(&data, &y, &test, &argvals, 5, 1e-3, 0.3, 0.1, 42);
    assert!(result.is_err());
}

#[test]
fn test_conformal_elastic_regression_invalid_alpha() {
    let (data, y, test_data, argvals) = make_elastic_test_data(20, 51, 42);
    assert!(
        conformal_elastic_regression(&data, &y, &test_data, &argvals, 10, 1e-3, 0.3, 0.0, 42)
            .is_err()
    );
    assert!(
        conformal_elastic_regression(&data, &y, &test_data, &argvals, 10, 1e-3, 0.3, 1.0, 42)
            .is_err()
    );
}

#[test]
fn test_conformal_elastic_regression_invalid_cal_fraction() {
    let (data, y, test_data, argvals) = make_elastic_test_data(20, 51, 42);
    assert!(
        conformal_elastic_regression(&data, &y, &test_data, &argvals, 10, 1e-3, 0.0, 0.1, 42)
            .is_err()
    );
    assert!(
        conformal_elastic_regression(&data, &y, &test_data, &argvals, 10, 1e-3, 1.0, 0.1, 42)
            .is_err()
    );
}

#[test]
fn test_conformal_elastic_pcr_basic() {
    use crate::elastic_regression::PcaMethod;
    let (data, y, test_data, argvals) = make_elastic_test_data(20, 51, 42);
    let r = conformal_elastic_pcr(
        &data,
        &y,
        &test_data,
        &argvals,
        2,
        PcaMethod::Vertical,
        0.0,
        0.3,
        0.1,
        42,
    )
    .unwrap();
    assert_eq!(r.predictions.len(), 5);
    assert_eq!(r.lower.len(), 5);
    assert_eq!(r.upper.len(), 5);
    for i in 0..5 {
        assert!(r.upper[i] > r.lower[i]);
        assert!(r.predictions[i].is_finite());
    }
    assert!(r.coverage >= 0.0 && r.coverage <= 1.0);
}

#[test]
fn test_conformal_elastic_pcr_dimension_mismatch() {
    use crate::elastic_regression::PcaMethod;
    let (data, y, _test, argvals) = make_elastic_test_data(20, 51, 42);
    let bad_test = FdMatrix::zeros(3, 30);
    let result = conformal_elastic_pcr(
        &data,
        &y,
        &bad_test,
        &argvals,
        2,
        PcaMethod::Vertical,
        0.0,
        0.3,
        0.1,
        42,
    );
    assert!(result.is_err());
}

#[test]
fn test_conformal_elastic_pcr_y_length_mismatch() {
    use crate::elastic_regression::PcaMethod;
    let (data, _y, test_data, argvals) = make_elastic_test_data(20, 51, 42);
    let wrong_y = vec![1.0; 5];
    let result = conformal_elastic_pcr(
        &data,
        &wrong_y,
        &test_data,
        &argvals,
        2,
        PcaMethod::Vertical,
        0.0,
        0.3,
        0.1,
        42,
    );
    assert!(result.is_err());
}

#[test]
fn test_conformal_elastic_logistic_basic() {
    let (data, y, test_data, argvals) = make_elastic_classif_data(20, 51, 42);
    let r = conformal_elastic_logistic(
        &data,
        &y,
        &test_data,
        &argvals,
        1e-2,
        ClassificationScore::Lac,
        0.3,
        0.1,
        42,
    )
    .unwrap();
    assert_eq!(r.prediction_sets.len(), 4);
    for set in &r.prediction_sets {
        assert!(!set.is_empty());
        assert!(set.len() <= 2, "Binary classification: set size at most 2");
    }
    assert!(r.average_set_size >= 1.0);
    assert!(r.coverage >= 0.0 && r.coverage <= 1.0);
}

#[test]
fn test_conformal_elastic_logistic_aps() {
    let (data, y, test_data, argvals) = make_elastic_classif_data(20, 51, 42);
    let r = conformal_elastic_logistic(
        &data,
        &y,
        &test_data,
        &argvals,
        1e-2,
        ClassificationScore::Aps,
        0.3,
        0.1,
        42,
    )
    .unwrap();
    assert_eq!(r.prediction_sets.len(), 4);
    for set in &r.prediction_sets {
        assert!(!set.is_empty());
    }
}

#[test]
fn test_conformal_elastic_logistic_dimension_mismatch() {
    let (data, y, _test, argvals) = make_elastic_classif_data(20, 51, 42);
    let bad_test = FdMatrix::zeros(2, 30);
    let result = conformal_elastic_logistic(
        &data,
        &y,
        &bad_test,
        &argvals,
        1e-2,
        ClassificationScore::Lac,
        0.3,
        0.1,
        42,
    );
    assert!(result.is_err());
}

#[test]
fn test_conformal_elastic_logistic_y_length_mismatch() {
    let (data, _y, test_data, argvals) = make_elastic_classif_data(20, 51, 42);
    let wrong_y = vec![1_i8; 5];
    let result = conformal_elastic_logistic(
        &data,
        &wrong_y,
        &test_data,
        &argvals,
        1e-2,
        ClassificationScore::Lac,
        0.3,
        0.1,
        42,
    );
    assert!(result.is_err());
}

#[test]
fn test_conformal_elastic_logistic_too_few_obs() {
    let data = FdMatrix::zeros(3, 10);
    let y = vec![1_i8, -1, 1];
    let test = FdMatrix::zeros(2, 10);
    let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
    let result = conformal_elastic_logistic(
        &data,
        &y,
        &test,
        &argvals,
        1e-2,
        ClassificationScore::Lac,
        0.3,
        0.1,
        42,
    );
    assert!(result.is_err());
}
