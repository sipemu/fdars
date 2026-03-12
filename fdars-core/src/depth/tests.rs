use super::*;
use crate::matrix::FdMatrix;
use std::f64::consts::PI;

fn uniform_grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

fn generate_centered_data(n: usize, m: usize) -> FdMatrix {
    let argvals = uniform_grid(m);
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        let offset = (i as f64 - n as f64 / 2.0) / (n as f64);
        for j in 0..m {
            data[i + j * n] = (2.0 * PI * argvals[j]).sin() + offset;
        }
    }
    FdMatrix::from_column_major(data, n, m).unwrap()
}

// ============== Fraiman-Muniz tests ==============

#[test]
fn test_fraiman_muniz() {
    // Simple test: identical data should give maximum depth
    let data = FdMatrix::from_column_major(vec![1.0, 1.0, 2.0, 2.0], 2, 2).unwrap(); // 2 identical curves, 2 points each
    let depths = fraiman_muniz_1d(&data, &data, true);
    assert_eq!(depths.len(), 2);
}

#[test]
fn test_fraiman_muniz_central_deeper() {
    let n = 20;
    let m = 30;
    let data = generate_centered_data(n, m);
    let depths = fraiman_muniz_1d(&data, &data, true);

    // Central curve (index n/2) should have higher depth than extreme curves
    let central_depth = depths[n / 2];
    let edge_depth = depths[0];
    assert!(
        central_depth > edge_depth,
        "Central curve should be deeper: {} > {}",
        central_depth,
        edge_depth
    );
}

#[test]
fn test_fraiman_muniz_range() {
    let n = 15;
    let m = 20;
    let data = generate_centered_data(n, m);
    let depths = fraiman_muniz_1d(&data, &data, true);

    for d in &depths {
        assert!(*d >= 0.0 && *d <= 1.0, "Depth should be in [0, 1]");
    }
}

#[test]
fn test_fraiman_muniz_invalid() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(fraiman_muniz_1d(&empty, &empty, true).is_empty());
}

// ============== Modal depth tests ==============

#[test]
fn test_modal_central_deeper() {
    let n = 20;
    let m = 30;
    let data = generate_centered_data(n, m);
    let depths = modal_1d(&data, &data, 0.5);

    let central_depth = depths[n / 2];
    let edge_depth = depths[0];
    assert!(central_depth > edge_depth, "Central curve should be deeper");
}

#[test]
fn test_modal_positive() {
    let n = 10;
    let m = 20;
    let data = generate_centered_data(n, m);
    let depths = modal_1d(&data, &data, 0.5);

    for d in &depths {
        assert!(*d > 0.0, "Modal depth should be positive");
    }
}

#[test]
fn test_modal_invalid() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(modal_1d(&empty, &empty, 0.5).is_empty());
}

// ============== Random projection depth tests ==============

#[test]
fn test_rp_depth_range() {
    let n = 15;
    let m = 20;
    let data = generate_centered_data(n, m);
    let depths = random_projection_1d(&data, &data, 50);

    for d in &depths {
        assert!(*d >= 0.0 && *d <= 1.0, "RP depth should be in [0, 1]");
    }
}

#[test]
fn test_rp_depth_invalid() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(random_projection_1d(&empty, &empty, 10).is_empty());
}

// ============== Random Tukey depth tests ==============

#[test]
fn test_random_tukey_range() {
    let n = 15;
    let m = 20;
    let data = generate_centered_data(n, m);
    let depths = random_tukey_1d(&data, &data, 50);

    for d in &depths {
        assert!(*d >= 0.0 && *d <= 1.0, "Tukey depth should be in [0, 1]");
    }
}

#[test]
fn test_random_tukey_invalid() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(random_tukey_1d(&empty, &empty, 10).is_empty());
}

// ============== Functional spatial depth tests ==============

#[test]
fn test_functional_spatial_range() {
    let n = 15;
    let m = 20;
    let data = generate_centered_data(n, m);
    let depths = functional_spatial_1d(&data, &data, None);

    for d in &depths {
        assert!(*d >= 0.0 && *d <= 1.0, "Spatial depth should be in [0, 1]");
    }
}

#[test]
fn test_functional_spatial_invalid() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(functional_spatial_1d(&empty, &empty, None).is_empty());
}

// ============== Band depth tests ==============

#[test]
fn test_band_depth_central_deeper() {
    let n = 10;
    let m = 20;
    let data = generate_centered_data(n, m);
    let depths = band_1d(&data, &data);

    // Central curve should be in more bands
    let central_depth = depths[n / 2];
    let edge_depth = depths[0];
    assert!(
        central_depth >= edge_depth,
        "Central curve should have higher band depth"
    );
}

#[test]
fn test_band_depth_range() {
    let n = 10;
    let m = 20;
    let data = generate_centered_data(n, m);
    let depths = band_1d(&data, &data);

    for d in &depths {
        assert!(*d >= 0.0 && *d <= 1.0, "Band depth should be in [0, 1]");
    }
}

#[test]
fn test_band_depth_invalid() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(band_1d(&empty, &empty).is_empty());
    let single = FdMatrix::from_column_major(vec![1.0, 2.0], 1, 2).unwrap();
    assert!(band_1d(&single, &single).is_empty()); // need at least 2 ref curves
}

// ============== Modified band depth tests ==============

#[test]
fn test_modified_band_depth_range() {
    let n = 10;
    let m = 20;
    let data = generate_centered_data(n, m);
    let depths = modified_band_1d(&data, &data);

    for d in &depths {
        assert!(*d >= 0.0 && *d <= 1.0, "MBD should be in [0, 1]");
    }
}

#[test]
fn test_modified_band_depth_invalid() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(modified_band_1d(&empty, &empty).is_empty());
}

// ============== Modified epigraph index tests ==============

#[test]
fn test_mei_range() {
    let n = 15;
    let m = 20;
    let data = generate_centered_data(n, m);
    let mei = modified_epigraph_index_1d(&data, &data);

    for d in &mei {
        assert!(*d >= 0.0 && *d <= 1.0, "MEI should be in [0, 1]");
    }
}

#[test]
fn test_mei_invalid() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(modified_epigraph_index_1d(&empty, &empty).is_empty());
}

// ============== KFSD 1D tests ==============

#[test]
fn test_kfsd_1d_range() {
    let n = 10;
    let m = 20;
    let argvals = uniform_grid(m);
    let data = generate_centered_data(n, m);
    let depths = kernel_functional_spatial_1d(&data, &data, &argvals, 0.5);

    assert_eq!(depths.len(), n);
    for d in &depths {
        assert!(
            *d >= -0.01 && *d <= 1.01,
            "KFSD depth should be near [0, 1], got {}",
            d
        );
        assert!(d.is_finite(), "KFSD depth should be finite");
    }

    // Central curve should have higher depth
    let central_depth = depths[n / 2];
    let edge_depth = depths[0];
    assert!(
        central_depth > edge_depth,
        "Central KFSD depth {} should be > edge depth {}",
        central_depth,
        edge_depth
    );
}

#[test]
fn test_kfsd_1d_identical() {
    // All identical curves should exercise the denom_j_sq < 1e-20 path
    let n = 5;
    let m = 10;
    let argvals = uniform_grid(m);
    let data_vec: Vec<f64> = (0..n * m)
        .map(|i| (2.0 * PI * (i % m) as f64 / (m - 1) as f64).sin())
        .collect();
    let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();

    // When all curves are identical, kernel distances are all 1.0
    // and denom_j_sq = K(x,x) + K(y,y) - 2*K(x,y) = 1 + 1 - 2*1 = 0
    let depths = kernel_functional_spatial_1d(&data, &data, &argvals, 0.5);

    assert_eq!(depths.len(), n);
    for d in &depths {
        assert!(
            d.is_finite(),
            "KFSD depth should be finite for identical curves"
        );
    }
}

#[test]
fn test_kfsd_1d_invalid() {
    let argvals = uniform_grid(10);
    let empty = FdMatrix::zeros(0, 0);
    assert!(kernel_functional_spatial_1d(&empty, &empty, &argvals, 0.5).is_empty());
    let empty_obj = FdMatrix::zeros(0, 0);
    assert!(kernel_functional_spatial_1d(&empty_obj, &empty_obj, &argvals, 0.5).is_empty());
}

// ============== KFSD 2D tests ==============

#[test]
fn test_kfsd_2d_range() {
    let n = 8;
    let m = 15;
    let data = generate_centered_data(n, m);
    let depths = kernel_functional_spatial_2d(&data, &data, 0.5);

    assert_eq!(depths.len(), n);
    for d in &depths {
        assert!(
            *d >= -0.01 && *d <= 1.01,
            "KFSD 2D depth should be near [0, 1], got {}",
            d
        );
        assert!(d.is_finite(), "KFSD 2D depth should be finite");
    }
}

// ============== 2D delegation tests ==============

#[test]
fn test_fraiman_muniz_2d_delegates() {
    let n = 10;
    let m = 15;
    let data = generate_centered_data(n, m);
    let depths_1d = fraiman_muniz_1d(&data, &data, true);
    let depths_2d = fraiman_muniz_2d(&data, &data, true);
    assert_eq!(depths_1d, depths_2d);
}

#[test]
fn test_modal_2d_delegates() {
    let n = 10;
    let m = 15;
    let data = generate_centered_data(n, m);
    let depths_1d = modal_1d(&data, &data, 0.5);
    let depths_2d = modal_2d(&data, &data, 0.5);
    assert_eq!(depths_1d, depths_2d);
}

#[test]
fn test_functional_spatial_2d_delegates() {
    let n = 10;
    let m = 15;
    let data = generate_centered_data(n, m);
    let depths_1d = functional_spatial_1d(&data, &data, None);
    let depths_2d = functional_spatial_2d(&data, &data);
    assert_eq!(depths_1d, depths_2d);
}

#[test]
fn test_random_projection_2d_returns_valid() {
    let n = 10;
    let m = 15;
    let data = generate_centered_data(n, m);
    let depths = random_projection_2d(&data, &data, 20);
    assert_eq!(depths.len(), n);
    for d in &depths {
        assert!(*d >= 0.0 && *d <= 1.0, "RP 2D depth should be in [0, 1]");
    }
}

// ============== Golden-value regression tests ==============

/// Fixed small dataset: 5 curves, 10 time points (deterministic)
fn golden_data() -> FdMatrix {
    let n = 5;
    let m = 10;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        let offset = (i as f64 - n as f64 / 2.0) / (n as f64);
        for j in 0..m {
            data[i + j * n] = (2.0 * PI * argvals[j]).sin() + offset;
        }
    }
    FdMatrix::from_column_major(data, n, m).unwrap()
}

#[test]
fn test_fm_golden_values_scaled() {
    let data = golden_data();
    let depths = fraiman_muniz_1d(&data, &data, true);
    let expected = [0.4, 0.8, 0.8, 0.4, 0.0];
    assert_eq!(depths.len(), expected.len());
    for (d, e) in depths.iter().zip(expected.iter()) {
        assert!(
            (d - e).abs() < 1e-10,
            "FM scaled golden mismatch: got {}, expected {}",
            d,
            e
        );
    }
}

#[test]
fn test_fm_golden_values_unscaled() {
    let data = golden_data();
    let depths = fraiman_muniz_1d(&data, &data, false);
    let expected = [0.2, 0.4, 0.4, 0.2, 0.0];
    assert_eq!(depths.len(), expected.len());
    for (d, e) in depths.iter().zip(expected.iter()) {
        assert!(
            (d - e).abs() < 1e-10,
            "FM unscaled golden mismatch: got {}, expected {}",
            d,
            e
        );
    }
}

#[test]
fn test_mbd_golden_values() {
    let data = golden_data();
    let depths = modified_band_1d(&data, &data);
    let expected = [0.4, 0.7, 0.8, 0.7, 0.4];
    assert_eq!(depths.len(), expected.len());
    for (d, e) in depths.iter().zip(expected.iter()) {
        assert!(
            (d - e).abs() < 1e-10,
            "MBD golden mismatch: got {}, expected {}",
            d,
            e
        );
    }
}

#[test]
fn test_nan_fm_no_panic() {
    let n = 5;
    let m = 10;
    let mut data_vec = vec![0.0; n * m];
    data_vec[3] = f64::NAN;
    let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();
    let depths = fraiman_muniz_1d(&data, &data, true);
    assert_eq!(depths.len(), n);
}

#[test]
fn test_nan_mbd_no_panic() {
    let n = 5;
    let m = 10;
    let mut data_vec = vec![0.0; n * m];
    data_vec[3] = f64::NAN;
    let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();
    let depths = modified_band_1d(&data, &data);
    assert_eq!(depths.len(), n);
}

#[test]
fn test_n1_depths() {
    let data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
    let fm = fraiman_muniz_1d(&data, &data, true);
    assert_eq!(fm.len(), 1);
    let modal = modal_1d(&data, &data, 0.5);
    assert_eq!(modal.len(), 1);
    let spatial = functional_spatial_1d(&data, &data, None);
    assert_eq!(spatial.len(), 1);
}

#[test]
fn test_n2_band_depth() {
    // Band depth needs at least 2 reference curves
    let data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let depths = band_1d(&data, &data);
    assert_eq!(depths.len(), 2);
    for &d in &depths {
        assert!((0.0..=1.0).contains(&d));
    }
}

#[test]
fn test_inf_spatial_depth() {
    let n = 3;
    let m = 5;
    let mut data_vec = vec![1.0; n * m];
    data_vec[0] = f64::INFINITY;
    let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();
    let depths = functional_spatial_1d(&data, &data, None);
    assert_eq!(depths.len(), n);
    // Should not panic
}
