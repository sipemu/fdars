//! Tests for basis functions.

use super::*;
use crate::matrix::FdMatrix;
use nalgebra::DVector;
use std::f64::consts::PI;

/// Generate a uniform grid of points
fn uniform_grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

/// Generate sine wave data
fn sine_wave(t: &[f64], freq: f64) -> Vec<f64> {
    t.iter().map(|&ti| (2.0 * PI * freq * ti).sin()).collect()
}

// ============== B-spline basis tests ==============

#[test]
fn test_bspline_basis_dimensions() {
    let t = uniform_grid(50);
    let nknots = 10;
    let order = 4;
    let basis = bspline_basis(&t, nknots, order);

    let expected_nbasis = nknots + order;
    assert_eq!(basis.len(), t.len() * expected_nbasis);
}

#[test]
fn test_bspline_basis_partition_of_unity() {
    // B-splines should sum to 1 at each point (partition of unity)
    let t = uniform_grid(50);
    let nknots = 8;
    let order = 4;
    let basis = bspline_basis(&t, nknots, order);

    let nbasis = nknots + order;
    for i in 0..t.len() {
        let sum: f64 = (0..nbasis).map(|j| basis[i + j * t.len()]).sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "B-spline partition of unity failed at point {}: sum = {}",
            i,
            sum
        );
    }
}

#[test]
fn test_bspline_basis_non_negative() {
    let t = uniform_grid(50);
    let basis = bspline_basis(&t, 8, 4);

    for &val in &basis {
        assert!(val >= -1e-10, "B-spline values should be non-negative");
    }
}

#[test]
fn test_bspline_basis_boundary() {
    // Test that basis functions work at boundary points
    let t = vec![0.0, 0.5, 1.0];
    let basis = bspline_basis(&t, 5, 4);

    // Should have valid output (no NaN or Inf)
    for &val in &basis {
        assert!(val.is_finite(), "B-spline should produce finite values");
    }
}

// ============== Fourier basis tests ==============

#[test]
fn test_fourier_basis_dimensions() {
    let t = uniform_grid(50);
    let nbasis = 7;
    let basis = fourier_basis(&t, nbasis);

    assert_eq!(basis.len(), t.len() * nbasis);
}

#[test]
fn test_fourier_basis_constant_first_column() {
    let t = uniform_grid(50);
    let nbasis = 7;
    let basis = fourier_basis(&t, nbasis);

    // First column should be constant (DC component = 1)
    let first_val = basis[0];
    for i in 0..t.len() {
        assert!(
            (basis[i] - first_val).abs() < 1e-10,
            "First Fourier column should be constant"
        );
    }
}

#[test]
fn test_fourier_basis_sin_cos_range() {
    let t = uniform_grid(100);
    let nbasis = 11;
    let basis = fourier_basis(&t, nbasis);

    // Sin and cos values should be in [-1, 1]
    for &val in &basis {
        assert!((-1.0 - 1e-10..=1.0 + 1e-10).contains(&val));
    }
}

#[test]
fn test_fourier_basis_with_period() {
    let t = uniform_grid(100);
    let nbasis = 5;
    let period = 0.5;
    let basis = fourier_basis_with_period(&t, nbasis, period);

    assert_eq!(basis.len(), t.len() * nbasis);
    // Verify first column is constant
    let first_val = basis[0];
    for i in 0..t.len() {
        assert!((basis[i] - first_val).abs() < 1e-10);
    }
}

#[test]
fn test_fourier_basis_period_affects_frequency() {
    let t = uniform_grid(100);
    let nbasis = 5;

    let basis1 = fourier_basis_with_period(&t, nbasis, 1.0);
    let basis2 = fourier_basis_with_period(&t, nbasis, 0.5);

    // Different periods should give different basis matrices
    let n = t.len();
    let mut any_different = false;
    for i in 0..n {
        // Compare second column (first sin term)
        if (basis1[i + n] - basis2[i + n]).abs() > 1e-10 {
            any_different = true;
            break;
        }
    }
    assert!(
        any_different,
        "Different periods should produce different bases"
    );
}

// ============== Difference matrix tests ==============

#[test]
fn test_difference_matrix_order_zero() {
    let d = difference_matrix(5, 0);
    assert_eq!(d.nrows(), 5);
    assert_eq!(d.ncols(), 5);

    // Should be identity matrix
    for i in 0..5 {
        for j in 0..5 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((d[(i, j)] - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn test_difference_matrix_first_order() {
    let d = difference_matrix(5, 1);
    assert_eq!(d.nrows(), 4);
    assert_eq!(d.ncols(), 5);

    // First order differences: D * [1,2,3,4,5] = [1,1,1,1]
    let x = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let dx = &d * x;
    for i in 0..4 {
        assert!((dx[i] - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_difference_matrix_second_order() {
    let d = difference_matrix(5, 2);
    assert_eq!(d.nrows(), 3);
    assert_eq!(d.ncols(), 5);

    // Second order differences of linear: D^2 * [1,2,3,4,5] = [0,0,0]
    let x = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let dx = &d * x;
    for i in 0..3 {
        assert!(dx[i].abs() < 1e-10, "Second diff of linear should be zero");
    }
}

#[test]
fn test_difference_matrix_quadratic() {
    let d = difference_matrix(5, 2);

    // Second order differences of quadratic: D^2 * [1,4,9,16,25] = [2,2,2]
    let x = DVector::from_vec(vec![1.0, 4.0, 9.0, 16.0, 25.0]);
    let dx = &d * x;
    for i in 0..3 {
        assert!(
            (dx[i] - 2.0).abs() < 1e-10,
            "Second diff of squares should be 2"
        );
    }
}

// ============== Basis projection tests ==============

/// Create an FdMatrix from per-curve data (each curve is one row).
/// The input flat data is in "row of rows" order: all values for curve 0, then curve 1, etc.
/// We need to convert to column-major layout.
fn make_matrix(flat_row_major: &[f64], n: usize, m: usize) -> FdMatrix {
    let mut col_major = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            col_major[i + j * n] = flat_row_major[i * m + j];
        }
    }
    FdMatrix::from_column_major(col_major, n, m).unwrap()
}

#[test]
fn test_fdata_to_basis_1d_bspline() {
    let t = uniform_grid(50);
    let n = 5;
    let m = t.len();

    // Create simple data (n curves, each shifted linear)
    let flat: Vec<f64> = (0..n)
        .flat_map(|i| t.iter().map(move |&ti| ti + i as f64 * 0.1))
        .collect();
    let data = make_matrix(&flat, n, m);

    let result = fdata_to_basis_1d(&data, &t, 10, 0);
    assert!(result.is_some());

    let res = result.unwrap();
    assert!(res.n_basis > 0);
    assert_eq!(res.coefficients.nrows(), n);
    assert_eq!(res.coefficients.ncols(), res.n_basis);
}

#[test]
fn test_fdata_to_basis_1d_fourier() {
    let t = uniform_grid(50);
    let n = 5;
    let m = t.len();

    // Create sine wave data
    let flat: Vec<f64> = (0..n).flat_map(|_| sine_wave(&t, 2.0)).collect();
    let data = make_matrix(&flat, n, m);

    let result = fdata_to_basis_1d(&data, &t, 7, 1);
    assert!(result.is_some());

    let res = result.unwrap();
    assert_eq!(res.n_basis, 7);
}

#[test]
fn test_fdata_to_basis_1d_invalid_input() {
    let t = uniform_grid(50);

    // Empty data
    let empty = FdMatrix::zeros(0, 50);
    let result = fdata_to_basis_1d(&empty, &t, 10, 0);
    assert!(result.is_none());

    // nbasis too small
    let data = FdMatrix::zeros(1, 50);
    let result = fdata_to_basis_1d(&data, &t, 1, 0);
    assert!(result.is_none());
}

#[test]
fn test_basis_roundtrip() {
    let t = uniform_grid(100);
    let n = 1;
    let m = t.len();

    // Create smooth sine wave data (Fourier basis should represent exactly)
    let raw = sine_wave(&t, 1.0);
    let data = FdMatrix::from_column_major(raw.clone(), n, m).unwrap();

    // Project to Fourier basis with enough terms
    let proj = fdata_to_basis_1d(&data, &t, 5, 1).unwrap();

    // Reconstruct
    let reconstructed = basis_to_fdata_1d(&proj.coefficients, &t, proj.n_basis, 1);

    // Should approximately match original for a simple sine wave
    let mut max_error = 0.0;
    for j in 0..m {
        let err = (raw[j] - reconstructed[(0, j)]).abs();
        if err > max_error {
            max_error = err;
        }
    }
    assert!(max_error < 0.5, "Roundtrip error too large: {}", max_error);
}

#[test]
fn test_basis_to_fdata_empty_input() {
    let empty = FdMatrix::zeros(0, 0);
    let result = basis_to_fdata_1d(&empty, &[], 5, 0);
    assert!(result.is_empty());
}

// ============== P-spline fitting tests ==============

#[test]
fn test_pspline_fit_1d_basic() {
    let t = uniform_grid(50);
    let n = 3;
    let m = t.len();

    // Create noisy data
    let flat: Vec<f64> = (0..n)
        .flat_map(|i| {
            t.iter()
                .enumerate()
                .map(move |(j, &ti)| (2.0 * PI * ti).sin() + 0.1 * (i * j) as f64 % 1.0)
        })
        .collect();
    let data = make_matrix(&flat, n, m);

    let result = pspline_fit_1d(&data, &t, 15, 1.0, 2);
    assert!(result.is_some());

    let res = result.unwrap();
    assert!(res.n_basis > 0);
    assert_eq!(res.fitted.nrows(), n);
    assert_eq!(res.fitted.ncols(), m);
    assert!(res.rss >= 0.0);
    assert!(res.edf > 0.0);
    assert!(res.gcv.is_finite());
}

#[test]
fn test_pspline_fit_1d_smoothness() {
    let t = uniform_grid(50);
    let n = 1;
    let m = t.len();

    // Create noisy sine wave
    let raw: Vec<f64> = t
        .iter()
        .enumerate()
        .map(|(i, &ti)| (2.0 * PI * ti).sin() + 0.3 * ((i * 17) % 100) as f64 / 100.0)
        .collect();
    let data = FdMatrix::from_column_major(raw, n, m).unwrap();

    let low_lambda = pspline_fit_1d(&data, &t, 15, 0.01, 2).unwrap();
    let high_lambda = pspline_fit_1d(&data, &t, 15, 100.0, 2).unwrap();

    // Higher lambda should give lower edf (more smoothing)
    assert!(high_lambda.edf < low_lambda.edf);
}

#[test]
fn test_pspline_fit_1d_invalid_input() {
    let t = uniform_grid(50);
    let empty = FdMatrix::zeros(0, 50);
    let result = pspline_fit_1d(&empty, &t, 15, 1.0, 2);
    assert!(result.is_none());
}

// ============== Fourier fitting tests ==============

#[test]
fn test_fourier_fit_1d_sine_wave() {
    let t = uniform_grid(100);
    let n = 1;
    let m = t.len();

    // Create pure sine wave
    let raw = sine_wave(&t, 2.0);
    let data = FdMatrix::from_column_major(raw, n, m).unwrap();

    let result = fourier_fit_1d(&data, &t, 11);
    assert!(result.is_ok());

    let res = result.unwrap();
    assert!(res.rss < 1e-6, "Pure sine should have near-zero RSS");
}

#[test]
fn test_fourier_fit_1d_makes_nbasis_odd() {
    let t = uniform_grid(50);
    let raw = sine_wave(&t, 1.0);
    let data = FdMatrix::from_column_major(raw, 1, t.len()).unwrap();

    // Pass even nbasis
    let result = fourier_fit_1d(&data, &t, 6);
    assert!(result.is_ok());

    // Should have been adjusted to odd
    let res = result.unwrap();
    assert!(res.n_basis % 2 == 1);
}

#[test]
fn test_fourier_fit_1d_criteria() {
    let t = uniform_grid(50);
    let raw = sine_wave(&t, 2.0);
    let data = FdMatrix::from_column_major(raw, 1, t.len()).unwrap();

    let result = fourier_fit_1d(&data, &t, 9).unwrap();

    // All criteria should be finite
    assert!(result.gcv.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());
}

#[test]
fn test_fourier_fit_1d_invalid_nbasis() {
    let t = uniform_grid(50);
    let raw = sine_wave(&t, 1.0);
    let data = FdMatrix::from_column_major(raw, 1, t.len()).unwrap();

    // nbasis < 3 should return None
    let result = fourier_fit_1d(&data, &t, 2);
    assert!(result.is_err());
}

// ============== GCV selection tests ==============

#[test]
fn test_select_fourier_nbasis_gcv_range() {
    let t = uniform_grid(100);
    let raw = sine_wave(&t, 3.0);
    let data = FdMatrix::from_column_major(raw, 1, t.len()).unwrap();

    let best = select_fourier_nbasis_gcv(&data, &t, 3, 15);

    assert!((3..=15).contains(&best));
    assert!(best % 2 == 1, "Selected nbasis should be odd");
}

#[test]
fn test_select_fourier_nbasis_gcv_respects_min() {
    let t = uniform_grid(50);
    let raw = sine_wave(&t, 1.0);
    let data = FdMatrix::from_column_major(raw, 1, t.len()).unwrap();

    let best = select_fourier_nbasis_gcv(&data, &t, 7, 15);
    assert!(best >= 7);
}

// ============== Auto selection tests ==============

#[test]
fn test_select_basis_auto_1d_returns_results() {
    let t = uniform_grid(50);
    let n = 3;
    let m = t.len();

    let flat: Vec<f64> = (0..n).flat_map(|i| sine_wave(&t, 1.0 + i as f64)).collect();
    let data = make_matrix(&flat, n, m);

    let result = select_basis_auto_1d(&data, &t, 0, 5, 15, 1.0, false);

    assert_eq!(result.selections.len(), n);
    for sel in &result.selections {
        assert!(sel.nbasis >= 3);
        assert!(!sel.coefficients.is_empty());
        assert_eq!(sel.fitted.len(), m);
    }
}

#[test]
fn test_select_basis_auto_1d_seasonal_hint() {
    let t = uniform_grid(100);
    let n = 1;
    let m = t.len();

    // Strong seasonal pattern
    let raw = sine_wave(&t, 5.0);
    let data = FdMatrix::from_column_major(raw, n, m).unwrap();

    let result = select_basis_auto_1d(&data, &t, 0, 0, 0, -1.0, true);

    assert_eq!(result.selections.len(), 1);
    assert!(result.selections[0].seasonal_detected);
}

#[test]
fn test_select_basis_auto_1d_non_seasonal() {
    let t = uniform_grid(50);
    let n = 1;
    let m = t.len();

    // Constant data (definitely not seasonal)
    let raw: Vec<f64> = vec![1.0; m];
    let data = FdMatrix::from_column_major(raw, n, m).unwrap();

    let result = select_basis_auto_1d(&data, &t, 0, 0, 0, -1.0, true);

    // Constant data shouldn't be detected as seasonal
    assert!(!result.selections[0].seasonal_detected);
}

#[test]
fn test_select_basis_auto_1d_criterion_options() {
    let t = uniform_grid(50);
    let raw = sine_wave(&t, 2.0);
    let data = FdMatrix::from_column_major(raw, 1, t.len()).unwrap();

    // Test all three criteria
    let gcv_result = select_basis_auto_1d(&data, &t, 0, 0, 0, 1.0, false);
    let aic_result = select_basis_auto_1d(&data, &t, 1, 0, 0, 1.0, false);
    let bic_result = select_basis_auto_1d(&data, &t, 2, 0, 0, 1.0, false);

    assert_eq!(gcv_result.criterion, 0);
    assert_eq!(aic_result.criterion, 1);
    assert_eq!(bic_result.criterion, 2);
}

#[test]
fn test_nan_pspline_no_panic() {
    let t = uniform_grid(50);
    let mut y = sine_wave(&t, 2.0);
    y[10] = f64::NAN;
    let data = FdMatrix::from_column_major(y, 1, t.len()).unwrap();
    let result = pspline_fit_1d(&data, &t, 10, 1.0, 2);
    // Should not panic; result may contain NaN
    assert!(result.is_some() || result.is_none());
}

#[test]
fn test_n1_fit() {
    // Single observation
    let t = uniform_grid(50);
    let y = sine_wave(&t, 1.0);
    let data = FdMatrix::from_column_major(y, 1, t.len()).unwrap();
    let result = fdata_to_basis_1d(&data, &t, 7, 1);
    assert!(result.is_some());
    let res = result.unwrap();
    assert_eq!(res.coefficients.nrows(), 1);
}

#[test]
fn test_single_point_basis() {
    // Single evaluation point: period = 0 so sin/cos terms produce NaN,
    // but the constant basis function (index 0) should be 1.0.
    let t = vec![0.5];
    let basis = fourier_basis(&t, 3);
    // fourier_basis returns column-major Vec<f64> of size n_points x nbasis
    assert_eq!(basis.len(), 3);
    assert!(
        (basis[0] - 1.0).abs() < 1e-12,
        "constant basis should be 1.0"
    );
}
