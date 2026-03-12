use super::*;
use std::f64::consts::PI;

#[test]
fn test_detrend_linear_removes_linear_trend() {
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();
    let data_vec: Vec<f64> = argvals
        .iter()
        .map(|&t| 2.0 + 0.5 * t + (2.0 * PI * t / 2.0).sin())
        .collect();
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result = detrend_linear(&data, &argvals);
    let expected: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * PI * t / 2.0).sin())
        .collect();
    let mut max_diff = 0.0f64;
    for j in 0..m {
        let diff = (result.detrended[(0, j)] - expected[j]).abs();
        max_diff = max_diff.max(diff);
    }
    assert!(max_diff < 0.2, "Max difference: {}", max_diff);
}

#[test]
fn test_detrend_polynomial_removes_quadratic_trend() {
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();
    let data_vec: Vec<f64> = argvals
        .iter()
        .map(|&t| 1.0 + 0.5 * t - 0.1 * t * t + (2.0 * PI * t / 2.0).sin())
        .collect();
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result = detrend_polynomial(&data, &argvals, 2);
    let expected: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * PI * t / 2.0).sin())
        .collect();
    let detrended_vec: Vec<f64> = (0..m).map(|j| result.detrended[(0, j)]).collect();
    let mean_det: f64 = detrended_vec.iter().sum::<f64>() / m as f64;
    let mean_exp: f64 = expected.iter().sum::<f64>() / m as f64;
    let mut num = 0.0;
    let mut den_det = 0.0;
    let mut den_exp = 0.0;
    for j in 0..m {
        num += (detrended_vec[j] - mean_det) * (expected[j] - mean_exp);
        den_det += (detrended_vec[j] - mean_det).powi(2);
        den_exp += (expected[j] - mean_exp).powi(2);
    }
    let corr = num / (den_det.sqrt() * den_exp.sqrt());
    assert!(corr > 0.95, "Correlation: {}", corr);
}

#[test]
fn test_detrend_diff1() {
    let m = 100;
    let data_vec: Vec<f64> = {
        let mut v = vec![0.0; m];
        v[0] = 1.0;
        for i in 1..m {
            v[i] = v[i - 1] + 0.1 * (i as f64).sin();
        }
        v
    };
    let data = FdMatrix::from_column_major(data_vec.clone(), 1, m).unwrap();
    let result = detrend_diff(&data, 1);
    for j in 0..m - 1 {
        let expected = data_vec[j + 1] - data_vec[j];
        assert!(
            (result.detrended[(0, j)] - expected).abs() < 1e-10,
            "Mismatch at {}: {} vs {}",
            j,
            result.detrended[(0, j)],
            expected
        );
    }
}

#[test]
fn test_auto_detrend_selects_linear_for_linear_data() {
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64).collect();
    let data_vec: Vec<f64> = argvals.iter().map(|&t| 2.0 + 0.5 * t).collect();
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result = auto_detrend(&data, &argvals);
    assert!(
        result.method.contains("linear") || result.method.contains("polynomial"),
        "Method: {}",
        result.method
    );
}

#[test]
fn test_detrend_loess_removes_linear_trend() {
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();
    let data_vec: Vec<f64> = argvals
        .iter()
        .map(|&t| 2.0 + 0.5 * t + (2.0 * PI * t / 2.0).sin())
        .collect();
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result = detrend_loess(&data, &argvals, 0.3, 1);
    let expected: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * PI * t / 2.0).sin())
        .collect();
    let detrended_vec: Vec<f64> = (0..m).map(|j| result.detrended[(0, j)]).collect();
    let mean_det: f64 = detrended_vec.iter().sum::<f64>() / m as f64;
    let mean_exp: f64 = expected.iter().sum::<f64>() / m as f64;
    let mut num = 0.0;
    let mut den_det = 0.0;
    let mut den_exp = 0.0;
    for j in 0..m {
        num += (detrended_vec[j] - mean_det) * (expected[j] - mean_exp);
        den_det += (detrended_vec[j] - mean_det).powi(2);
        den_exp += (expected[j] - mean_exp).powi(2);
    }
    let corr = num / (den_det.sqrt() * den_exp.sqrt());
    assert!(corr > 0.9, "Correlation: {}", corr);
    assert_eq!(result.method, "loess");
}

#[test]
fn test_detrend_loess_removes_quadratic_trend() {
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();
    let data_vec: Vec<f64> = argvals
        .iter()
        .map(|&t| 1.0 + 0.3 * t - 0.05 * t * t + (2.0 * PI * t / 2.0).sin())
        .collect();
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result = detrend_loess(&data, &argvals, 0.3, 2);
    assert_eq!(result.trend.ncols(), m);
    assert_eq!(result.detrended.ncols(), m);
    assert!(result.rss[0] > 0.0);
}

#[test]
fn test_detrend_loess_different_bandwidths() {
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();
    let data_vec: Vec<f64> = argvals
        .iter()
        .enumerate()
        .map(|(i, &t)| (2.0 * PI * t / 2.0).sin() + 0.1 * ((i * 17) % 100) as f64 / 100.0)
        .collect();
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result_small = detrend_loess(&data, &argvals, 0.1, 1);
    let result_large = detrend_loess(&data, &argvals, 0.5, 1);
    assert_eq!(result_small.trend.ncols(), m);
    assert_eq!(result_large.trend.ncols(), m);
    assert!(result_large.n_params >= result_small.n_params);
}

#[test]
fn test_detrend_loess_short_series() {
    let m = 10;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64).collect();
    let data_vec: Vec<f64> = argvals.iter().map(|&t| t * 2.0).collect();
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result = detrend_loess(&data, &argvals, 0.3, 1);
    assert_eq!(result.trend.ncols(), m);
    assert_eq!(result.detrended.ncols(), m);
}

#[test]
fn test_decompose_additive_separates_components() {
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();
    let data_vec: Vec<f64> = argvals
        .iter()
        .map(|&t| 2.0 + 0.5 * t + (2.0 * PI * t / period).sin())
        .collect();
    let data = FdMatrix::from_column_major(data_vec.clone(), 1, m).unwrap();
    let result = decompose_additive(&data, &argvals, period, "loess", 0.3, 3);
    assert_eq!(result.trend.ncols(), m);
    assert_eq!(result.seasonal.ncols(), m);
    assert_eq!(result.remainder.ncols(), m);
    assert_eq!(result.method, "additive");
    assert_eq!(result.period, period);
    for j in 0..m {
        let reconstructed =
            result.trend[(0, j)] + result.seasonal[(0, j)] + result.remainder[(0, j)];
        assert!(
            (reconstructed - data_vec[j]).abs() < 0.5,
            "Reconstruction error at {}: {} vs {}",
            j,
            reconstructed,
            data_vec[j]
        );
    }
}

#[test]
fn test_decompose_additive_different_harmonics() {
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();
    let data_vec: Vec<f64> = argvals
        .iter()
        .map(|&t| 1.0 + (2.0 * PI * t / period).sin())
        .collect();
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result1 = decompose_additive(&data, &argvals, period, "loess", 0.3, 1);
    let result5 = decompose_additive(&data, &argvals, period, "loess", 0.3, 5);
    assert_eq!(result1.seasonal.ncols(), m);
    assert_eq!(result5.seasonal.ncols(), m);
}

#[test]
fn test_decompose_additive_residual_properties() {
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();
    let data_vec: Vec<f64> = argvals
        .iter()
        .map(|&t| 2.0 + 0.3 * t + (2.0 * PI * t / period).sin())
        .collect();
    let data = FdMatrix::from_column_major(data_vec.clone(), 1, m).unwrap();
    let result = decompose_additive(&data, &argvals, period, "loess", 0.3, 3);
    let remainder_vec: Vec<f64> = (0..m).map(|j| result.remainder[(0, j)]).collect();
    let mean_rem: f64 = remainder_vec.iter().sum::<f64>() / m as f64;
    assert!(mean_rem.abs() < 0.5, "Remainder mean: {}", mean_rem);
    let data_mean: f64 = data_vec.iter().sum::<f64>() / m as f64;
    let var_data: f64 = data_vec
        .iter()
        .map(|&x| (x - data_mean).powi(2))
        .sum::<f64>()
        / m as f64;
    let var_rem: f64 = remainder_vec
        .iter()
        .map(|&x| (x - mean_rem).powi(2))
        .sum::<f64>()
        / m as f64;
    assert!(
        var_rem < var_data,
        "Remainder variance {} should be < data variance {}",
        var_rem,
        var_data
    );
}

#[test]
fn test_decompose_additive_multi_sample() {
    let n = 3;
    let m = 100;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();
    let mut data = FdMatrix::zeros(n, m);
    for i in 0..n {
        let amp = (i + 1) as f64;
        for j in 0..m {
            data[(i, j)] = 1.0 + 0.1 * argvals[j] + amp * (2.0 * PI * argvals[j] / period).sin();
        }
    }
    let result = decompose_additive(&data, &argvals, period, "loess", 0.3, 2);
    assert_eq!(result.trend.shape(), (n, m));
    assert_eq!(result.seasonal.shape(), (n, m));
    assert_eq!(result.remainder.shape(), (n, m));
}

#[test]
fn test_decompose_multiplicative_basic() {
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();
    let data_vec: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 + 0.1 * t) * (1.0 + 0.3 * (2.0 * PI * t / period).sin()))
        .collect();
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result = decompose_multiplicative(&data, &argvals, period, "loess", 0.3, 3);
    assert_eq!(result.trend.ncols(), m);
    assert_eq!(result.seasonal.ncols(), m);
    assert_eq!(result.remainder.ncols(), m);
    assert_eq!(result.method, "multiplicative");
    let seasonal_vec: Vec<f64> = (0..m).map(|j| result.seasonal[(0, j)]).collect();
    let mean_seasonal: f64 = seasonal_vec.iter().sum::<f64>() / m as f64;
    assert!(
        (mean_seasonal - 1.0).abs() < 0.5,
        "Mean seasonal factor: {}",
        mean_seasonal
    );
}

#[test]
fn test_decompose_multiplicative_non_positive_data() {
    let m = 100;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();
    let data_vec: Vec<f64> = argvals
        .iter()
        .map(|&t| -1.0 + (2.0 * PI * t / period).sin())
        .collect();
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result = decompose_multiplicative(&data, &argvals, period, "loess", 0.3, 2);
    assert_eq!(result.trend.ncols(), m);
    assert_eq!(result.seasonal.ncols(), m);
    for j in 0..m {
        let s = result.seasonal[(0, j)];
        assert!(s.is_finite(), "Seasonal should be finite");
    }
}

#[test]
fn test_decompose_multiplicative_vs_additive() {
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();
    let data_vec: Vec<f64> = argvals
        .iter()
        .map(|&t| 5.0 + (2.0 * PI * t / period).sin())
        .collect();
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let add_result = decompose_additive(&data, &argvals, period, "loess", 0.3, 3);
    let mult_result = decompose_multiplicative(&data, &argvals, period, "loess", 0.3, 3);
    assert_eq!(add_result.seasonal.ncols(), m);
    assert_eq!(mult_result.seasonal.ncols(), m);
    let add_seasonal_vec: Vec<f64> = (0..m).map(|j| add_result.seasonal[(0, j)]).collect();
    let add_mean: f64 = add_seasonal_vec.iter().sum::<f64>() / m as f64;
    let mult_seasonal_vec: Vec<f64> = (0..m).map(|j| mult_result.seasonal[(0, j)]).collect();
    let mult_mean: f64 = mult_seasonal_vec.iter().sum::<f64>() / m as f64;
    assert!(
        add_mean.abs() < mult_mean,
        "Additive mean {} vs mult mean {}",
        add_mean,
        mult_mean
    );
}

#[test]
fn test_decompose_multiplicative_edge_cases() {
    let empty = FdMatrix::zeros(0, 0);
    let result = decompose_multiplicative(&empty, &[], 2.0, "loess", 0.3, 2);
    assert_eq!(result.trend.len(), 0);
    let m = 5;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64).collect();
    let data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 2.0, 1.0], 1, m).unwrap();
    let result = decompose_multiplicative(&data, &argvals, 2.0, "loess", 0.3, 1);
    assert_eq!(result.remainder.ncols(), m);
}

#[test]
fn test_stl_decompose_basic() {
    let period = 12;
    let n_cycles = 10;
    let m = period * n_cycles;
    let data_vec: Vec<f64> = (0..m)
        .map(|i| {
            let t = i as f64;
            0.01 * t + (2.0 * PI * t / period as f64).sin()
        })
        .collect();
    let data = FdMatrix::from_column_major(data_vec.clone(), 1, m).unwrap();
    let result = stl_decompose(&data, period, None, None, None, false, None, None);
    assert_eq!(result.trend.ncols(), m);
    assert_eq!(result.seasonal.ncols(), m);
    assert_eq!(result.remainder.ncols(), m);
    assert_eq!(result.period, period);
    for j in 0..m {
        let reconstructed =
            result.trend[(0, j)] + result.seasonal[(0, j)] + result.remainder[(0, j)];
        assert!(
            (reconstructed - data_vec[j]).abs() < 1e-8,
            "Reconstruction error at {}: {} vs {}",
            j,
            reconstructed,
            data_vec[j]
        );
    }
}

#[test]
fn test_stl_decompose_robust() {
    let period = 12;
    let n_cycles = 10;
    let m = period * n_cycles;
    let mut data_vec: Vec<f64> = (0..m)
        .map(|i| {
            let t = i as f64;
            0.01 * t + (2.0 * PI * t / period as f64).sin()
        })
        .collect();
    data_vec[30] += 10.0;
    data_vec[60] += 10.0;
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result = stl_decompose(&data, period, None, None, None, true, None, Some(5));
    assert!(
        result.weights[(0, 30)] < 1.0,
        "Weight at outlier should be < 1.0: {}",
        result.weights[(0, 30)]
    );
    assert!(
        result.weights[(0, 60)] < 1.0,
        "Weight at outlier should be < 1.0: {}",
        result.weights[(0, 60)]
    );
    let non_outlier_weight = result.weights[(0, 15)];
    assert!(
        non_outlier_weight > result.weights[(0, 30)],
        "Non-outlier weight {} should be > outlier weight {}",
        non_outlier_weight,
        result.weights[(0, 30)]
    );
}

#[test]
fn test_stl_decompose_default_params() {
    let period = 10;
    let m = period * 8;
    let data_vec: Vec<f64> = (0..m)
        .map(|i| (2.0 * PI * i as f64 / period as f64).sin())
        .collect();
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result = stl_decompose(&data, period, None, None, None, false, None, None);
    assert_eq!(result.trend.ncols(), m);
    assert_eq!(result.seasonal.ncols(), m);
    assert!(result.s_window >= 3);
    assert!(result.t_window >= 3);
    assert_eq!(result.inner_iterations, 2);
    assert_eq!(result.outer_iterations, 1);
}

#[test]
fn test_stl_decompose_invalid() {
    let data = FdMatrix::from_column_major(vec![1.0, 2.0], 1, 2).unwrap();
    let result = stl_decompose(&data, 1, None, None, None, false, None, None);
    assert_eq!(result.s_window, 0);
    let data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
    let result = stl_decompose(&data, 5, None, None, None, false, None, None);
    assert_eq!(result.s_window, 0);
    let data = FdMatrix::zeros(0, 0);
    let result = stl_decompose(&data, 10, None, None, None, false, None, None);
    assert_eq!(result.trend.len(), 0);
}

#[test]
fn test_stl_fdata() {
    let n = 3;
    let period = 10;
    let m = period * 5;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    for i in 0..n {
        let amp = (i + 1) as f64;
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * argvals[j] / period as f64).sin();
        }
    }
    let result = stl_fdata(&data, &argvals, period, None, None, false);
    assert_eq!(result.trend.shape(), (n, m));
    assert_eq!(result.seasonal.shape(), (n, m));
    assert_eq!(result.remainder.shape(), (n, m));
    for i in 0..n {
        for j in 0..m {
            let reconstructed =
                result.trend[(i, j)] + result.seasonal[(i, j)] + result.remainder[(i, j)];
            assert!(
                (reconstructed - data[(i, j)]).abs() < 1e-8,
                "Reconstruction error for sample {} at {}: {} vs {}",
                i,
                j,
                reconstructed,
                data[(i, j)]
            );
        }
    }
}

#[test]
fn test_stl_decompose_multi_sample() {
    let n = 5;
    let period = 10;
    let m = period * 6;
    let mut data = FdMatrix::zeros(n, m);
    for i in 0..n {
        let offset = i as f64 * 0.5;
        for j in 0..m {
            data[(i, j)] = offset + 0.01 * j as f64 + (2.0 * PI * j as f64 / period as f64).sin();
        }
    }
    let result = stl_decompose(&data, period, None, None, None, false, None, None);
    assert_eq!(result.trend.shape(), (n, m));
    assert_eq!(result.seasonal.shape(), (n, m));
    assert_eq!(result.remainder.shape(), (n, m));
    assert_eq!(result.weights.shape(), (n, m));
}

#[test]
fn test_detrend_diff_order2() {
    let m = 50;
    let data_vec: Vec<f64> = (0..m).map(|i| (i as f64).powi(2)).collect();
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result = detrend_diff(&data, 2);
    for j in 0..m - 2 {
        assert!(
            (result.detrended[(0, j)] - 2.0).abs() < 1e-10,
            "Second diff at {}: expected 2.0, got {}",
            j,
            result.detrended[(0, j)]
        );
    }
}

#[test]
fn test_detrend_polynomial_degree3() {
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 5.0).collect();
    let data_vec: Vec<f64> = argvals
        .iter()
        .map(|&t| 1.0 + 2.0 * t - 0.5 * t * t + 0.1 * t * t * t)
        .collect();
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result = detrend_polynomial(&data, &argvals, 3);
    assert_eq!(result.method, "polynomial(3)");
    assert!(result.coefficients.is_some());
    let max_detrend: f64 = (0..m)
        .map(|j| result.detrended[(0, j)].abs())
        .fold(0.0, f64::max);
    assert!(
        max_detrend < 0.1,
        "Pure cubic should be nearly zero after degree-3 detrend: {}",
        max_detrend
    );
}

#[test]
fn test_detrend_loess_invalid() {
    let data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 5).unwrap();
    let argvals = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let result = detrend_loess(&data, &argvals, -0.1, 1);
    assert_eq!(result.detrended.as_slice(), data.as_slice());
    let data2 = FdMatrix::from_column_major(vec![1.0, 2.0], 1, 2).unwrap();
    let result = detrend_loess(&data2, &[0.0, 1.0], 0.3, 1);
    assert_eq!(result.detrended.as_slice(), &[1.0, 2.0]);
}

#[test]
fn test_nan_linear_detrend() {
    let m = 20;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let mut data_vec = vec![1.0; m];
    data_vec[5] = f64::NAN;
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result = detrend_linear(&data, &argvals);
    // Should not panic
    assert_eq!(result.detrended.nrows(), 1);
}

#[test]
fn test_n1_detrend() {
    let m = 50;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let data_vec: Vec<f64> = argvals.iter().map(|&t| 2.0 * t + 1.0).collect();
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result = detrend_linear(&data, &argvals);
    assert_eq!(result.detrended.ncols(), m);
    // Detrended linear should be near zero
    for j in 0..m {
        assert!(result.detrended[(0, j)].abs() < 0.1);
    }
}

#[test]
fn test_constant_signal_detrend() {
    let m = 30;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let data = FdMatrix::from_column_major(vec![5.0; m], 1, m).unwrap();
    let result = detrend_linear(&data, &argvals);
    // A constant has zero slope, so detrended = data - mean
    assert_eq!(result.detrended.ncols(), m);
}

#[test]
fn test_m2_minimal_detrend() {
    // Minimal grid: 2 points
    let argvals = vec![0.0, 1.0];
    let data = FdMatrix::from_column_major(vec![0.0, 1.0], 1, 2).unwrap();
    let result = detrend_linear(&data, &argvals);
    assert_eq!(result.detrended.ncols(), 2);
}
